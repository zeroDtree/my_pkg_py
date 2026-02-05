r"""
Base Diffuser Config and Base Diffuser.
"""

import abc
from typing import Any

from torch import Tensor
from torch.nn import Module

from ..decorators import inherit_docstrings
from ..shape_class import Shape, ShapeConfig
from .base_config_class import DeviceConfig


@inherit_docstrings
class BaseLossConfig(DeviceConfig):
    def __init__(self, ndim_micro_shape: int, *args: list[Any], **kwargs: dict[Any, Any]):
        super().__init__(*args, **kwargs)
        self.ndim_micro_shape: int = ndim_micro_shape


@inherit_docstrings
class BaseLoss(Module, abc.ABC):
    r"""
    abstract method: compute_loss
    """

    def __init__(
        self,
        config: BaseLossConfig,
    ):
        Module.__init__(self)
        abc.ABC.__init__(self)
        self.config: BaseLossConfig = config
        self.shape_util = Shape(
            config=ShapeConfig(ndim_micro_shape=config.ndim_micro_shape),
        )

    @abc.abstractmethod
    def compute_loss(self, **batch) -> dict | Tensor:
        r"""Compute loss

        Args:
            batch (``dict[str, Any]``): the batch of data

        Returns:
            ``dict``|``Tensor``: a dictionary that must contain the key "loss" or a tensor of loss
        """

    def get_macro_shape(self, x: Tensor) -> tuple[int, ...]:
        return self.shape_util.get_macro_shape(x)

    def complete_micro_shape(self, x: Tensor) -> Tensor:
        return self.shape_util.complete_micro_shape(x)
