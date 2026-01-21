from torch import Tensor

from .decorators import inherit_docstrings


@inherit_docstrings
class ShapeConfig:
    def __init__(
        self,
        ndim_micro_shape: int,
    ):
        self.ndim_micro_shape: int = ndim_micro_shape


@inherit_docstrings
class Shape(object):
    def __init__(
        self,
        config: ShapeConfig,
    ):
        super().__init__()
        self.config: ShapeConfig = config

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
