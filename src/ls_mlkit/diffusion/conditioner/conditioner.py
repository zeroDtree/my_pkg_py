import abc
from typing import Any

import torch
from torch import Tensor

from ...util.decorators import inherit_docstrings


@inherit_docstrings
class Conditioner(abc.ABC):
    def __init__(self, guidance_scale: float = 1.0):
        self._enabled: bool = True
        self.ready: bool = False
        self._guidance_scale: float = guidance_scale

    @abc.abstractmethod
    def prepare_condition_dict(self, train: bool = True, *args: list[Any], **kwargs: dict[Any, Any]) -> dict[str, Any]:
        r"""Prepare the condition dictionary

        Args:
            train (``bool``, *optional*): whether the conditioner is used in training. Defaults to True.

        Returns:
            ``dict[str, Any]``: the condition dictionary
        """

    @abc.abstractmethod
    def set_condition(self, *args: list[Any], **kwargs: dict[Any, Any]) -> None:
        r"""Set the condition

        Args:
            *args: additional arguments
            **kwargs: additional keyword arguments
        """

    @abc.abstractmethod
    def get_conditional_score(self, x_t: Tensor, t: Tensor, padding_mask: Tensor, *args: Any, **kwargs: Any) -> Tensor:
        r"""Get conditional score

        Args:
            x_t (``Tensor``): the input tensor
            t (``Tensor``): the time tensor
            padding_mask (``Tensor``): the padding mask

        Returns:
            ``Tensor``: the conditional score
        """

    @property
    def guidance_scale(self):
        return self._guidance_scale

    def get_guidance_scale(self):
        return self._guidance_scale

    def set_guidance_scale(self, guidance_scale: float):
        self._guidance_scale = guidance_scale

    def enable(self):
        self._enabled = True

    def disable(self):
        self._enabled = False

    def is_enabled(self) -> bool:
        return self._enabled


@inherit_docstrings
class LGDConditioner(Conditioner):
    r"""Loss Guidance Diffusion Conditioner"""

    def __init__(
        self,
        guidance_scale: float = 1.0,
    ):
        super().__init__(guidance_scale)
        self.posterior_mean_fn = None

    @abc.abstractmethod
    def compute_conditional_loss(self, p_gt_data: Tensor, padding_mask: Tensor) -> Tensor:
        r"""Compute the conditional loss

        Args:
            p_gt_data (``Tensor``): predicted clean data.
            padding_mask (``Tensor``): the padding mask

        Returns:
            ``Tensor``: the conditional loss
        """

    def get_conditional_score(self, x_t: Tensor, t: Tensor, padding_mask: Tensor, *args: Any, **kwargs: Any) -> Tensor:
        r"""Get conditional score

        Args:
            x_t (``Tensor``): the input tensor
            t (``Tensor``): the time tensor
            padding_mask (``Tensor``): the padding mask

        Returns:
            ``Tensor``: the conditional score
        """
        if not self._enabled:
            return torch.zeros_like(x_t, device=x_t.device)
        assert self.ready == True, "Conditioner is not ready, please call set_condition first"
        with torch.autograd.set_detect_anomaly(True, check_nan=True):
            with torch.enable_grad():
                x_t = x_t.detach().clone().requires_grad_(True)
                p_gt_data = self.posterior_mean_fn(x_t, t, padding_mask, *args, **kwargs)
                conditional_loss = self.compute_conditional_loss(p_gt_data, padding_mask)
                grad = torch.autograd.grad(conditional_loss, x_t)[0]
        score = -grad
        return score * self.guidance_scale
