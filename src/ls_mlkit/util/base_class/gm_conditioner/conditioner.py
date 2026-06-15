import abc
from typing import Any, Callable, cast

import torch
from torch import Tensor

from ...decorators import inherit_docstrings


@inherit_docstrings
class Conditioner(abc.ABC):
    def __init__(self, guidance_scale: float = 1.0):
        self._enabled: bool = True
        self.ready: bool = False
        self._guidance_scale: float = guidance_scale

    @abc.abstractmethod
    def prepare_condition_dict(self, train: bool = True, *args: Any, **kwargs: Any) -> dict[str, Any]:
        r"""Prepare the condition dictionary

        Args:
            train (`bool`, *optional*): whether the conditioner is used in training. Defaults to True.

        Returns:
            dict[str, Any]: the condition dictionary
        """

    @abc.abstractmethod
    def set_condition(self, *args: Any, **kwargs: Any) -> None:
        r"""Set the condition

        Args:
            *args: additional arguments
            **kwargs: additional keyword arguments
        """

    @abc.abstractmethod
    def get_guidance(
        self,
        x_t: Tensor,
        t: Tensor,
        padding_mask: Tensor,
        *args: Any,
        **kwargs: Any,
    ) -> Tensor:
        r"""Get guidance vector g(x_t)

        Args:
            x_t (Tensor): the input tensor
            t (Tensor): the time tensor
            padding_mask (Tensor): the padding mask

        Returns:
            Tensor: the guidance vector g(x_t)
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
class LossGuidanceConditioner(Conditioner):
    r"""Loss-gradient guidance conditioner.

    Computes g(x_t) = -∇_{x_t} l(posterior_mean_fn(x_t), y).
    """

    def __init__(
        self,
        guidance_scale: float = 1.0,
    ):
        super().__init__(guidance_scale)
        self.posterior_mean_fn = None
        self.last_step_metrics: dict[str, float] = {}

    @abc.abstractmethod
    def compute_conditional_loss(self, p_gt_data: Tensor, padding_mask: Tensor) -> Tensor:
        r"""Compute the conditional loss

        Args:
            p_gt_data (Tensor): predicted clean data.
            padding_mask (Tensor): the padding mask

        Returns:
            Tensor: the conditional loss
        """

    def get_guidance(
        self,
        x_t: Tensor,
        t: Tensor,
        padding_mask: Tensor,
        *args: Any,
        **kwargs: Any,
    ) -> Tensor:
        r"""Get guidance vector g(x_t) = -∇_{x_t} l(posterior_mean_fn(x_t), y)

        Args:
            x_t (Tensor): the input tensor
            t (Tensor): the time tensor
            padding_mask (Tensor): the padding mask

        Returns:
            Tensor: the guidance vector
        """
        if not self._enabled:
            return torch.zeros_like(x_t, device=x_t.device)
        assert self.ready, "Conditioner is not ready, please call set_condition first"
        with torch.autograd.set_detect_anomaly(True, check_nan=True):
            with torch.enable_grad():
                x_t = x_t.detach().clone().requires_grad_(True)
                self.posterior_mean_fn = cast(Callable, self.posterior_mean_fn)
                p_gt_data = self.posterior_mean_fn(x_t, t, padding_mask, *args, **kwargs)
                conditional_loss = self.compute_conditional_loss(p_gt_data, padding_mask)
                grad = torch.autograd.grad(conditional_loss, x_t)[0]
        guidance = -grad * self.guidance_scale
        self.last_step_metrics = {
            "conditional_loss": float(conditional_loss.detach()),
            "guidance_norm": float(guidance.detach().norm()),
            "guidance_scale": float(self.guidance_scale),
        }
        return guidance
