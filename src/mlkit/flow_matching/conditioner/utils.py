from typing import Any, cast

from torch import Tensor

from ...util.base_class.gm_conditioner import Conditioner as _GMConditioner
from ...util.base_class.gm_conditioner.utils import get_accumulated_guidance as _get_accumulated_guidance
from .conditioner import Conditioner


def get_accumulated_guidance(
    conditioner_list: list[Conditioner],
    x_t: Tensor,
    t: Tensor | None,
    padding_mask: Tensor | None,
    *args: Any,
    **kwargs: Any,
) -> Tensor:
    r"""Get the accumulated guidance vector for flow matching.

    Args:
        x_t (Tensor): $x_t$
        t (Tensor): $t$
        padding_mask (Tensor): the padding mask

    Returns:
        Tensor: the accumulated guidance vector
    """
    return _get_accumulated_guidance(
        cast(list[_GMConditioner], conditioner_list),
        x_t,
        t,
        padding_mask,
        *args,
        **kwargs,
    )
