r"""
Base Diffuser Config and Base Diffuser.
"""

import abc
from typing import Any, Callable

from numpy import isin
from torch import Tensor
from torch.nn import Module

from ..decorators import inherit_docstrings
from .base_shape_class import BaseShapeClass, BaseShapeConfig
from .base_hook import HookStage, Hook, HookManager


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
        self.hook_manager: HookManager = HookManager()

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
        result = self.compute_loss(batch, *args, **kwargs)
        hook_result = self.hook_manager.run_hooks(stage=HookStage.POST_LOSS_COMPUTE, **result)
        if hook_result is not None:
            assert isinstance(hook_result, (dict, Tensor))
            result = hook_result
        return result

    def register_after_compute_loss_hook(
        self, name: str, fn: Callable[..., Any], priority: int = 0, enabled: bool = True
    ) -> None:
        r"""Register a hook to be called after loss computation

        Args:
            name (``str``): the name of the hook
            fn (``Callable[..., Any]``): the function to be called
            priority (``int``, optional): the priority of the hook. Defaults to 0.
            enabled (``bool``, optional): whether the hook is enabled. Defaults to True.
        """
        hook = Hook(name=name, stage=HookStage.POST_LOSS_COMPUTE, fn=fn, priority=priority, enabled=enabled)
        self.hook_manager.register_hook(hook)
