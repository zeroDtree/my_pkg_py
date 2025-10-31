import abc
from typing import Any

import torch
from torch import Tensor

from ls_mlkit.my_utils.decorators import inherit_docstrings


class Conditioner(abc.ABC):
    def __init__(self, guidance_scale: float = 1.0):
        """Initialize the Conditioner

        Args:
            guidance_scale (float, optional): the guidance scale of the conditioner. Defaults to 1.0.
        """
        self.enabled: bool = True
        self.ready: bool = False
        self._guidance_scale: float = guidance_scale

    @abc.abstractmethod
    def prepare_condition_dict(self, train: bool = True, *args: list[Any], **kwargs: dict[Any, Any]) -> dict[str, Any]:
        """Prepare the condition dictionary

        Args:
            train (bool, optional): whether the conditioner is used in training. Defaults to True.

        Returns:
            dict[str, Any]: the condition dictionary
        """

    @abc.abstractmethod
    def set_condition(self, *args: list[Any], **kwargs: dict[Any, Any]) -> None:
        """Set the condition

        Args:
            *args: additional arguments
            **kwargs: additional keyword arguments
        """

    @abc.abstractmethod
    def get_conditional_score(self, x_t: Tensor, t: Tensor, padding_mask: Tensor, *args: Any, **kwargs: Any) -> Tensor:
        """Get conditional score

        Args:
            x_t (Tensor): the input tensor
            t (Tensor): the time tensor
            padding_mask (Tensor): the padding mask

        Returns:
            Tensor: the conditional score
        """

    @property
    def guidance_scale(self):
        return self._guidance_scale

    @guidance_scale.setter
    def set_guidance_scale(self, guidance_scale: float):
        self._guidance_scale = guidance_scale

    def enable(self):
        self.enabled = True

    def disable(self):
        self.enabled = False


@inherit_docstrings
class LGDConditioner(Conditioner):
    """Loss Guidance Conditioner"""

    def __init__(
        self,
        guidance_scale: float = 1.0,
    ):
        """Initialize the LGDConditioner

        Args:
            guidance_scale (float, optional): the guidance scale of the conditioner. Defaults to 1.0.
        """
        super().__init__(guidance_scale)

    @abc.abstractmethod
    def compute_conditional_loss(self, x_t: Tensor, t: Tensor, padding_mask: Tensor) -> Tensor:
        """Compute the conditional loss

        Args:
            x_t (Tensor): the input tensor
            t (Tensor): the time tensor
            padding_mask (Tensor): the padding mask

        Returns:
            Tensor: the conditional loss
        """

    def get_conditional_score(self, x_t: Tensor, t: Tensor, padding_mask: Tensor, *args: Any, **kwargs: Any) -> Tensor:
        """Get conditional score

        Args:
            x_t (Tensor): the input tensor
            t (Tensor): the time tensor
            padding_mask (Tensor): the padding mask

        Returns:
            Tensor: the conditional score
        """
        assert self.ready == True, "Conditioner is not ready, please call set_condition first"
        with torch.autograd.set_detect_anomaly(True, check_nan=True):
            x = x_t.detach().clone().requires_grad_(True)
            conditional_loss = self.compute_conditional_loss(x, t, padding_mask)
            grad = torch.autograd.grad(conditional_loss, x, create_graph=True)[0]
            score = -grad
        return score * self.guidance_scale
