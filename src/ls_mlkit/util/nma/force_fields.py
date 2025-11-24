import abc

import torch
from torch import Tensor


class ForceField(metaclass=abc.ABCMeta):
    r"""
    Subclasses of this abstract base class define the force constants of
    the modeled springs between atoms in a *Elastic network model*.
    ``...`` is arbitrary number of dimensions, for example, batch size.


    Args:

        n: int, the number of atoms
        m: int, the number of edges
        cutoff_distance : float or None
            The interaction of two atoms is only considered, if the distance
            between them is smaller or equal to this value.
            If ``None``, the interaction between all atoms is considered.
        natoms : [...] or None
            The number of atoms in the model.
            If a :class:`ForceField` does not depend on the respective
            atoms, i.e. `atom_i` and `atom_j` is unused in
            :meth:`force_constant()`, this attribute is ``None`` instead.
        contact_shutdown : Tensor, shape=(..., n), dtype=float, optional
            Indices that point to atoms, whose contacts to all other atoms
            are artificially switched off.
            If ``None``, no contacts are switched off.
        contact_pair_off : Tensor, shape=(..., m, 2), dtype=int, optional
            Indices that point to pairs of atoms, whose contacts
            are artificially switched off.
            If ``None``, no contacts are switched off.
        contact_pair_on : Tensor, shape=(..., m, 2), dtype=int, optional
            Indices that point to pairs of atoms, whose contacts
            are are established in any case.
            If ``None``, no contacts are artificially switched on.
    """

    @abc.abstractmethod
    def force_constant(self, atom_i: Tensor, atom_j: Tensor, sq_distance: Tensor):
        """
        Get the force constant for the interaction of the given atoms.

        ABSTRACT:
            Override when inheriting.

        Parameters:

            atom_i, atom_j : Tensor, shape=(len(...) + 2, m), len(...)=macro_shape, dtype=int
                The indices to the first and second atoms in each
                interacting atom pair.
            sq_distance : Tensor, shape=(m), dtype=float
                The distance between the atoms indicated by `atom_i` and
                `atom_j`.

        Notes:
            Implementations of this method do not need
            to check whether two atoms are within the cutoff distance of the
            :class:`ForceField`:
            The given pairs of atoms are limited to pairs within cutoff
            distance of each other.
            However, if `cutoff_distance` is ``None``, the atom indices
            contain the Cartesian product of all atom indices, i.e. each
            possible combination.
        """

    @property
    def cutoff_distance(self):
        return None

    @property
    def contact_shutdown(self):
        return None

    @property
    def contact_pair_off(self):
        return None

    @property
    def contact_pair_on(self):
        return None

    @property
    def natoms(self):
        return None


class InvariantForceField(ForceField):
    """
    This force field treats every interaction with the same force
    constant.

    Parameters:

        cutoff_distance : float
            The interaction of two atoms is only considered, if the distance
            between them is smaller or equal to this value.
    """

    def __init__(self, cutoff_distance: float):
        if cutoff_distance is None:
            # A value of 'None' would give a fully connected network
            # with equal force constants for each connection,
            # which is unreasonable
            raise ValueError("Cutoff distance must be a float")
        self._cutoff_distance = cutoff_distance

    def force_constant(self, atom_i: Tensor, atom_j: Tensor = None, sq_distance: Tensor = None):
        """
        Calculate force constants for atom interactions.

        Args:
            atom_i: Tensor, shape=(len(...) + 2, m), len(...)=macro_shape, dtype=int
            atom_j: Tensor, shape=(len(...) + 2, m), len(...)=macro_shape, dtype=int
            sq_distance: Tensor, shape=(m), dtype=float
        """
        n_edges = atom_i.shape[-1]  # (m)
        force_constants = torch.ones(n_edges)
        return force_constants

    @property
    def cutoff_distance(self):
        return self._cutoff_distance


class HinsenForceField(ForceField):
    """
    The Hinsen force field was parametrized using the *Amber94* force
    field for a local energy minimum, with crambin as template.
    In a strict distance-dependent manner, contacts are subdivided
    into nearest-neighbour pairs along the backbone (r < 4 Å) and
    mid-/far-range pair interactions (r >= 4 Å).
    Force constants for these interactions are computed with two
    distinct formulas.
    2.9 Å is the lowest accepted distance between ``CA`` atoms.
    Values below that threshold are set to 2.9 Å.

    Parameters:

        cutoff_distance : float, optional
            The interaction of two atoms is only considered, if the distance
            between them is smaller or equal to this value.
            By default all interactions are included.
    """

    def __init__(self, cutoff_distance: float = None):
        self._cutoff_distance = cutoff_distance

    def force_constant(self, atom_i: Tensor, atom_j: Tensor, sq_distance: Tensor):
        """
        Calculate force constants using the Hinsen force field parameters.

        Args:
            atom_i: Tensor, indices of first atoms
            atom_j: Tensor, indices of second atoms
            sq_distance: Tensor, squared distances between atom pairs

        Returns:
            Tensor: Force constants for each atom pair
        """
        distance = torch.sqrt(sq_distance)
        distance = torch.clip(distance, min=2.9, max=None)
        return torch.where(distance < 4.0, distance * 8.6e2 - 2.39e3, distance ** (-6) * 128e4)

    @property
    def cutoff_distance(self):
        return self._cutoff_distance
