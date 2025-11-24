from typing import Any

from torch import Tensor

from ..decorators import inherit_docstrings
from .base_config_class import BaseConfig


@inherit_docstrings
class BaseShapeConfig(BaseConfig):
    def __init__(
        self,
        ndim_micro_shape: int,
        *args: list[Any],
        **kwargs: dict[Any, Any],
    ):
        r"""
        Args:
            ndim_micro_shape (``int``): number of dimensions of a sample
        """
        super().__init__(*args, **kwargs)
        self.ndim_micro_shape: int = ndim_micro_shape


@inherit_docstrings
class BaseShapeClass(object):
    def __init__(
        self,
        config: BaseShapeConfig,
    ):
        r"""Initialize the BaseClass4Pipeline

        Args:
            config (``BaseShapeConfig``): the config of the shape
        """
        super().__init__()
        self.config: BaseShapeConfig = config

    def get_macro_and_micro_shape(self, x: Tensor) -> tuple[tuple[int, ...], tuple[int, ...]]:
        r"""Get the macro and micro shape of :math:`x`

        Args:
            x (``Tensor``): :math:`x`

        Returns:
            ``tuple[tuple[int, ...], tuple[int, ...]]``: the macro and micro shape of :math:`x`
        """
        ndim_micro_shape = self.config.ndim_micro_shape
        return x.shape[:-ndim_micro_shape], x.shape[-ndim_micro_shape:]

    def get_macro_shape(self, x: Tensor) -> tuple[int, ...]:
        r"""Get the macro shape of :math:`x`

        Args:
            x (``Tensor``): :math:`x`

        Returns:
            ``tuple[int, ...]``: the shape of the macro part of :math:`x`
        """
        return x.shape[: -self.config.ndim_micro_shape]

    def complete_micro_shape(self, x: Tensor) -> Tensor:
        """Complete the micro shape of :math:`x`, assuming the macro shape is already known

        Args:
            x (``Tensor``): :math:`x`

        Returns:
            ``Tensor``: :math:`x` with the micro shape completed
        """
        return x.view(*x.shape, *([1] * self.config.ndim_micro_shape))
