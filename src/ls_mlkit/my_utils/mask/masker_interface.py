import abc
from torch import Tensor
from typing import Any


class MaskerInterface(abc.ABC):
    def __init__(self, ndim_mini_micro_shape: int = 1, **kwargs: dict[Any, Any]):
        self.kwargs: dict[Any, Any] = kwargs
        self.ndim_mini_micro_shape: int = ndim_mini_micro_shape

    @abc.abstractmethod
    def apply_mask(self, x: Tensor, mask: Tensor) -> Tensor:
        pass

    @abc.abstractmethod
    def check_mask_shape(self, x: Tensor, mask: Tensor):
        """
        check whether the shape of mask is as expected
        """
        pass

    @abc.abstractmethod
    def count_bright_area(self, mask: Tensor) -> Tensor:
        """
        Bright area can be seen
        Dark area cannot be seen
        """
        pass

    @abc.abstractmethod
    def get_full_bright_mask(self, x: Tensor) -> Tensor:
        """
        Return a mask that is all bright
        """
        pass

    @abc.abstractmethod
    def apply_inpainting_mask(self, x_0: Tensor, x_t: Tensor, inpainting_mask: Tensor) -> Tensor:
        """
        1 represents the region that can be seen
        """
        pass
