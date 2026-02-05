from typing import Any

import torch
from torch import Tensor

from .conditioner import Conditioner


def get_accumulated_conditional_score(
    conditioner_list: list[Conditioner], x_t: Tensor, t: Tensor, padding_mask: Tensor, *args: Any, **kwargs: Any
) -> Tensor:
    r"""Get the accumulated conditional score

    Args:
        x_t (``Tensor``): :math:`x_t`
        t (``Tensor``): :math:`t`
        padding_mask (``Tensor``): the padding mask

    Returns:
        ``Tensor``: the accumulated conditional score
    """
    accumulated_conditional_score = torch.zeros_like(x_t)
    for conditioner in conditioner_list:
        if not conditioner.is_enabled():
            continue
        accumulated_conditional_score += conditioner.get_conditional_score(x_t, t, padding_mask, *args, **kwargs)
    return accumulated_conditional_score
