from typing import Any

import torch
from torch import Tensor

from .conditioner import Conditioner


def get_accumulated_guidance(
    conditioner_list: list[Conditioner],
    x_t: Tensor,
    t: Tensor | None,
    padding_mask: Tensor | None,
    *args: Any,
    **kwargs: Any,
) -> Tensor:
    r"""Get the accumulated guidance vector

    Args:
        x_t (Tensor): $x_t$
        t (Tensor): $t$
        padding_mask (Tensor): the padding mask

    Returns:
        Tensor: the accumulated guidance vector
    """
    if t is None or padding_mask is None:
        raise ValueError("t and padding_mask are required for guidance accumulation")
    accumulated_guidance = torch.zeros_like(x_t)
    for conditioner in conditioner_list:
        if not conditioner.is_enabled():
            continue
        accumulated_guidance += conditioner.get_guidance(x_t, t, padding_mask, *args, **kwargs)
    return accumulated_guidance
