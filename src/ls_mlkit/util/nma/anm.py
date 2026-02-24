from typing import Callable

import biotite.structure as struc
import einops
import torch
from torch import Tensor

from .force_fields import ForceField

K_B = 1.380649e-23  # Boltzmann constant, J/K
N_A = 6.02214076e23  # Avogadro constant, mol^-1


r"""
Boltzmann distribution
$$
P(x) \propto \exp \left(-\frac{1}{2k_BT}(x-x_0)^T H(x-x_0)\right)
$$
Multi-dimensional Gaussian distribution
$$
P(x) \propto \exp \left(-\frac{1}{2}(x-\mu)^T \Sigma^{-1}(x-\mu)\right)
$$
"""


class ANM:
    """
    Anisotropic Network Model.

    Args:
        hessian : tensor, shape=``(..., n*3, n*3)``, dtype=float
            The *Hessian* matrix for this model.
            Each dimension is partitioned in the form
            ``[x1, y1, z1, ... xn, yn, zn]``.
            This is not a copy: Create a copy before modifying this matrix.
        masses : tensor, shape=(..., n), dtype=float
            The mass for each atom, `None` if no mass weighting is applied.
    """

    def __init__(
        self,
        atoms: Tensor,
        force_field: ForceField,
        masses=None,
        device="cuda",
        node_mask: Tensor = None,
    ):
        """
        Args

            atoms : tensor, shape=(..., n, 3), dtype=float
                Atom coordinates that are part of the model. It usually contains only CA atoms.
            force_field : ForceField, natoms=(..., n)
                The :class:`ForceField` that defines the force constants between the given `atoms`.
            masses : ndarray, shape=(..., n), dtype=float, optional
                If an array is given, the Hessian is weighted with the inverse square root of the given masses.
                By default no mass-weighting is applied.
                0 for invalid nodes.
            device : str, optional
            node_mask : tensor, shape=(..., n), dtype=long, optional, 1 for valid nodes, 0 for invalid nodes
        """
        self._coord = atoms
        self._ff = force_field
        self.device = device
        self._node_mask = node_mask

        if masses is None:
            self._masses = None
        else:
            assert (
                masses.shape == atoms.shape[:-1]
            ), f"shape(masses) = {masses.shape} != shape(atoms[:-1]) = {atoms.shape[:-1]}"
            if torch.any(masses < 0):
                raise ValueError("Masses must not be negative")
            self._masses = masses * node_mask

        if self._masses is not None:
            mass_weights = torch.where(
                node_mask == 1, 1 / torch.sqrt(self._masses), torch.zeros_like(self._masses)
            )  # (..., n)
            mass_weights = mass_weights.repeat_interleave(repeats=3, dim=-1)  # (..., 3n)
            self._mass_weight_matrix = torch.einsum("...i,...j->...ij", mass_weights, mass_weights)
            """
            Shape of mass_weights: (..., n)
            Shape of self._mass_weight_matrix: (..., 3n, 3n)
            """
        else:
            self._mass_weight_matrix = None

        self._hessian = None

    @property
    def masses(self):
        return self._masses

    @property
    def hessian(self):
        if self._hessian is None:
            self._hessian, _ = self.compute_hessian(
                self._coord, self._ff, device=self.device, use_cell_list=False, node_mask=self._node_mask
            )
            if self._mass_weight_matrix is not None:
                self._hessian *= self._mass_weight_matrix
            self._hessian *= torch.einsum(
                "...i,...j->...ij",
                self._node_mask.repeat_interleave(repeats=3, dim=-1),
                self._node_mask.repeat_interleave(repeats=3, dim=-1),
            )
        return self._hessian

    @hessian.setter
    def hessian(self, value):
        self._hessian = value

    def eigen(self, epsilon=1e-7):
        """
        Compute the Eigenvalues and Eigenvectors of the *Hessian* matrix.
        The first six Eigenvalues/Eigenvectors correspond to trivial modes (translations/rotations) and are usually omitted in normal mode analysis.

        Returns:

            eig_values : tensor, shape=(..., n*3), dtype=float
                Eigenvalues of the *Hessian* matrix in ascending order.
            eig_vectors : tensor, shape=(..., n*3, n*3), dtype=float
                Eigenvectors of the *Hessian* matrix.
                Eigenvectors will have the same dtype as the *Hessian* matrix and will contain the eigenvectors as its columns.
        """
        torch.linalg.eigh: Callable
        eig_values, eig_vectors = torch.linalg.eigh(self.hessian + epsilon * torch.randn_like(self.hessian))
        return eig_values, eig_vectors

    def get_displacements_from_normal_modes(self, indexes: list[int]):
        """
        Get the displacement vectors for the given normal modes.

        Args:
            indexes: list of integers, the indexes of the normal modes.
        Returns:
            displacement_vectors: tensor of shape (..., 3n, len(indexes)), where n is the number of atoms.
        """
        k = len(indexes)
        node_mask = self._node_mask  # (..., n)
        n = node_mask.shape[-1]

        skips = (node_mask == 0).sum(dim=-1, keepdim=True)  # (..., 1)
        skips = skips.expand(*skips.shape[:-1], k)  # (..., k)
        skips *= 3

        indexes: Tensor = torch.tensor(indexes, device=self.device, dtype=torch.long)  # (k)
        indexes = indexes.expand_as(skips)  # (..., k)

        indexes = indexes + skips  # (..., k)

        _, eig_vectors = self.eigen()  # (..., 3n),(..., 3n, 3n)

        macro_shape = indexes.shape[:-1]  # (...), k = indexes.shape[-1]
        mesh = torch.meshgrid([torch.arange(s, device=indexes.device) for s in macro_shape], indexing="ij")
        mesh = [m.unsqueeze(-1).expand(*macro_shape, k) for m in mesh]
        mode_vectors = eig_vectors[(*mesh, slice(None), indexes)]  # shape: (..., k, 3n)

        mode_vectors = mode_vectors.reshape(*macro_shape, k, n, 3)
        if k == 1:
            return mode_vectors[..., 0, :, :]
        else:
            return mode_vectors

    def compute_hessian(
        self, coord: Tensor, force_field: ForceField, device, use_cell_list=False, node_mask: Tensor = None
    ):
        """
        Compute the *Hessian* matrix for atoms with given coordinates and
        the chosen force field.

        Args:
            coord : tensor, shape=(..., n, 3), dtype=float
                The coordinates.
            force_field : ForceField, natoms=(..., n)
                The :class:`ForceField` that defines the force constants.
            use_cell_list : bool, optional
                If true, a *cell list* is used to find atoms within cutoff
                distance instead of checking all pairwise atom distances.
                This significantly increases the performance for large number of
                atoms, but is slower for very small systems.
                If the `force_field` does not provide a cutoff, no cell list is
                used regardless.
            node_mask : tensor, shape=(..., n), dtype=long, optional, 1 for valid nodes, 0 for invalid nodes

        Returns:

            hessian : tensor, shape=(..., n*3, n*3), dtype=float
                The computed *Hessian* matrix.
                Each dimension is partitioned in the form
                ``[x1, y1, z1, ... xn, yn, zn]``.
            pairs : tensor, shape=(len(...) + 2, m), dtype=int
                Indices for interacting atoms, i.e. atoms within
                `cutoff_distance`.
        """
        # Convert into higher precision to avert numerical issues in
        # pseudoinverse calculation
        coord = coord.to(torch.float64)
        pairs, disp, sq_dist = self._prepare_values_for_interaction_matrix(
            coord, force_field, device, use_cell_list, node_mask=node_mask
        )
        """
        pair: tensor, shape=(len(...) + 2, m), len(...)=macro_shape, dtype=int,
            Indices for interacting atoms, i.e. atoms within
            `cutoff_distance`.
        disp: tensor, shape=(m, 3), dtype=float
            The displacement vector for the atom `pair`.
        sq_dist: tensor, shape=(m), dtype=float
            The squared distance for the atom `pair`.
        """

        macro_shape = coord.shape[:-2]
        n = coord.shape[-2]
        hessian_shape = list(macro_shape) + [n, n, 3, 3]
        hessian = torch.zeros(hessian_shape, dtype=torch.float64, device=device)

        atom_i = pairs[:-1]
        atom_j = torch.concat([pairs[:-2], pairs[-1].unsqueeze(0)], dim=0)

        force_constants = force_field.force_constant(atom_i, atom_j, sq_dist)  # (m)

        hessian[*pairs] = -(force_constants / sq_dist).view(-1, 1, 1) * disp.view(-1, 3, 1) * disp.view(-1, 1, 3)
        # Set values for main diagonal
        """
        hessian.shape = (macro_shape, n, n, 3, 3)
        torch.sum(hessian, dim=-4).shape = (macro_shape, n, 3, 3)
        """
        indices = torch.arange(n, device=device)
        hessian[..., indices, indices, :, :] = -torch.sum(hessian, dim=-4)

        hessian = einops.rearrange(hessian, "... a b c d -> ... (a c) (b d)")
        return hessian, pairs

    def _prepare_values_for_interaction_matrix(self, coord, force_field, device, use_cell_list, node_mask):
        """
        Check input values and calculate common intermediate values for
        :func:`compute_kirchhoff()` and :func:`compute_hessian()`.

        Args:

            coord : ndarray, shape=(..., n,3), dtype=float
                The coordinates.
            force_field : ForceField
                The :class:`ForceField` that defines the force constants.
            node_mask : tensor, shape=(..., n), dtype=long, optional, 1 for valid nodes, 0 for invalid nodes

        Returns:
            pair_indices : ndarray, shape=(len(...) + 2, m), len(...)=macro_shape, dtype=int,
                Indices for interacting atoms, i.e. atoms within
                `cutoff_distance`.
            disp : ndarray, shape=(m, 3), dtype=float
                The displacement vector for the atom `pair_indices`.
            sq_dist : ndarray, shape=(m), dtype=float
                The squared distance for the atom `pair_indices`.
        """
        if coord.shape[-1] != 3:
            raise ValueError(f"Expected coordinates with shape (..., n, 3), got {coord.shape}")

        # Find interacting atoms within cutoff distance
        cutoff_distance = force_field.cutoff_distance
        macro_shape = coord.shape[:-2]
        n = coord.shape[-2]
        adj_matrix_shape = list(macro_shape) + [n, n]
        if cutoff_distance is None:
            # Include all possible interactions
            adj_matrix = torch.ones(adj_matrix_shape, dtype=bool, device=device)
        else:
            dist_matrix = torch.cdist(coord, coord, p=2).reshape(adj_matrix_shape)
            sq_dist_matrix = dist_matrix**2
            adj_matrix = sq_dist_matrix <= cutoff_distance**2

        # Remove interactions of atoms with themselves
        adj_matrix = adj_matrix.squeeze(-1)
        adj_matrix = adj_matrix & (~torch.eye(n, dtype=bool, device=device).view([1 for _ in macro_shape] + [n, n]))
        # (..., n, n)

        node_mask_matrix = torch.einsum("...i,...j->...ij", node_mask, node_mask)
        adj_matrix = adj_matrix * node_mask_matrix

        # self._patch_adjacency_matrix(
        #     adj_matrix,
        #     force_field.contact_shutdown,
        #     force_field.contact_pair_off,
        #     force_field.contact_pair_on,
        # )

        # Convert matrix to indices where interaction exists
        pair_indices = torch.where(adj_matrix)  # ((len(marcro_shape) + 2), m)
        pair_indices = torch.stack(pair_indices, dim=0)  # ((len(marcro_shape) + 2), m)

        atom_i = pair_indices[:-1]  # ((len(marcro_shape) + 1), m)
        atom_j = torch.concat([pair_indices[:-2], pair_indices[-1].unsqueeze(0)], dim=0)  # ((len(marcro_shape) + 1), m)

        disp = coord[*atom_i] - coord[*atom_j]

        # Get displacement vector for ANMs
        # and squared distances for distance-dependent force fields
        if cutoff_distance is None:
            disp = struc.index_displacement(coord, pair_indices)
            sq_dist = torch.sum(disp * disp, axis=-1)
        else:
            sq_dist = sq_dist_matrix[*pair_indices]

        return pair_indices, disp, sq_dist

    def _patch_adjacency_matrix(self, matrix, contact_shutdown, contact_pair_off, contact_pair_on):
        """
        NOT USED

        Apply contacts that are artificially switched off/on to an
        adjacency matrix.
        The matrix is modified in-place.

        Args

            matrix: tensor of shape (..., n, n), dtype=bool
            contact_shutdown: tensor of shape (..., n), dtype=int
            contact_pair_off: tensor of shape (..., m, 2), dtype=int
            contact_pair_on: tensor of shape (..., m, 2), dtype=int
        """

        if contact_shutdown is not None:
            matrix[:, contact_shutdown] = False
            matrix[contact_shutdown, :] = False
        if contact_pair_off is not None:
            atom_i, atom_j = contact_pair_off.T
            matrix[atom_i, atom_j] = False
            matrix[atom_j, atom_i] = False
        if contact_pair_on is not None:
            atom_i, atom_j = contact_pair_on.T
            if (atom_i == atom_j).any():
                raise ValueError("Cannot turn on interaction of an atom with itself")
            matrix[atom_i, atom_j] = True
            matrix[atom_j, atom_i] = True
