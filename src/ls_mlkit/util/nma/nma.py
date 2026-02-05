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
