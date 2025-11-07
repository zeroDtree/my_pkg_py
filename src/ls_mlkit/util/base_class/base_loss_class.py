r"""
Base Diffuser Config and Base Diffuser.
"""

import abc
from typing import Any

from torch import Tensor
from torch.nn import Module

from ..decorators import inherit_docstrings
from .base_shape_class import BaseShapeClass, BaseShapeConfig


@inherit_docstrings
class BaseLossConfig(BaseShapeConfig):
    def __init__(self, ndim_micro_shape: int, *args: list[Any], **kwargs: dict[Any, Any]):
        super().__init__(ndim_micro_shape, *args, **kwargs)


@inherit_docstrings
class BaseLossClass(Module, BaseShapeClass, abc.ABC):
    r"""
    abstract method: compute_loss
    """

    def __init__(
        self,
        config: BaseLossConfig,
    ):
        r"""Initialize the BaseLossClass

        Args:
            config (``BaseLossConfig``): the config
        """
        Module.__init__(self)
        BaseShapeClass.__init__(self, config)
        abc.ABC.__init__(self)
        self.config: BaseLossConfig = config

    @abc.abstractmethod
    def compute_loss(self, batch: dict[str, Any], *args: list[Any], **kwargs: dict[Any, Any]) -> dict | Tensor:
        r"""Compute loss

        Args:
            batch (``dict[str, Any]``): the batch of data

        Returns:
            ``dict``|``Tensor``: a dictionary that must contain the key "loss" or a tensor of loss
        """

    def forward(self, batch: dict[str, Any], *args: list[Any], **kwargs: dict[Any, Any]) -> dict | Tensor:
        r"""Forward function, input batch of data and return the dictionary containing the loss

        Args:
            batch (``dict[str, Any]``): the batch of data

        Returns:
            ``dict`` | ``Tensor``: a dictionary that must contain the key "loss" or a tensor of loss
        """
        return self.compute_loss(batch, *args, **kwargs)
