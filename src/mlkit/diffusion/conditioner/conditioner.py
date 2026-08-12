import abc
from typing import Any

from torch import Tensor

from ...util.base_class.gm_conditioner import (
    Conditioner as _GMConditioner,
)
from ...util.base_class.gm_conditioner import (
    LossGuidanceConditioner as _LossGuidanceConditioner,
)
from ...util.decorators import inherit_docstrings


@inherit_docstrings
class Conditioner(_GMConditioner):
    @abc.abstractmethod
    def get_conditional_score(
        self,
        x_t: Tensor,
        t: Tensor,
        padding_mask: Tensor,
        *args: Any,
        **kwargs: Any,
    ) -> Tensor:
        r"""Get conditional score

        Args:
            x_t (Tensor): the input tensor
            t (Tensor): the time tensor
            padding_mask (Tensor): the padding mask

        Returns:
            Tensor: the conditional score
        """

    def get_guidance(
        self,
        x_t: Tensor,
        t: Tensor,
        padding_mask: Tensor,
        *args: Any,
        **kwargs: Any,
    ) -> Tensor:
        return self.get_conditional_score(x_t, t, padding_mask, *args, **kwargs)


@inherit_docstrings
class LGDConditioner(_LossGuidanceConditioner, Conditioner):
    r"""Loss Guidance Diffusion Conditioner"""

    def get_conditional_score(
        self,
        x_t: Tensor,
        t: Tensor,
        padding_mask: Tensor,
        *args: Any,
        **kwargs: Any,
    ) -> Tensor:
        return self.get_guidance(x_t, t, padding_mask, *args, **kwargs)
