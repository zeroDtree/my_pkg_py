import torch
from torch import Tensor

from .anm import ANM
from .force_fields import HinsenForceField


def get_nma_displacement_from_node_coordinates(
    node_coordinates: Tensor,
    cutoff_distance: float = 10.0,
    indexes: list[int] = [6],
    node_mask: Tensor = None,
) -> Tensor:
    """
    node_coordinates: shape = (..., n, 3)
    node_mask: shape = (..., n)
    """
    force_field = HinsenForceField(cutoff_distance=cutoff_distance)
    anm = ANM(
        atoms=node_coordinates,
        force_field=force_field,
        masses=None,
        device=node_coordinates.device,
        node_mask=node_mask,
    )
    return anm.get_displacements_from_normal_modes(indexes=indexes)


def get_nma_displacement_from_protein_ligand_complex(
    protein_ca_coordinates: Tensor,
    ligand_center_of_mass: Tensor,
    cutoff_distance: float = 10.0,
    indexes: list[int] = [6],
    protein_mask: Tensor = None,
) -> Tensor:
    """
    protein_ca_coordinates: shape = (..., n, 3),
    ligand_center_of_mass: shape = (..., 3)
    protein_mask: shape = (..., n)

    Returns:
        displacements: shape = (..., k, n, 3) or (..., n, 3) if k == 1,
        where k is the number of normal modes, n is the number of atoms, and 3 is the number of coordinates (x, y, z)
    """
    ligand_center_of_mass = ligand_center_of_mass.unsqueeze(-2)
    node_coordinates = torch.cat([protein_ca_coordinates, ligand_center_of_mass], dim=-2)
    # (..., (n+1), 3)

    ligand_mask_shape = list(ligand_center_of_mass.shape[:-1]) + [1]
    ligand_mask = torch.ones_like(ligand_mask_shape)
    # (..., 1)

    node_mask = torch.cat([protein_mask, ligand_mask], dim=-1)  # (..., n+1)

    return get_nma_displacement_from_node_coordinates(
        node_coordinates=node_coordinates,
        cutoff_distance=cutoff_distance,
        indexes=indexes,
        node_mask=node_mask,
    )
