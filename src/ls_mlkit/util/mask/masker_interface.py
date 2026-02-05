import abc
from typing import Any

from torch import Tensor


class MaskerInterface(abc.ABC):
    def __init__(self, *args, **kwargs: dict[Any, Any]):
        self.args = args
        self.kwargs: dict[Any, Any] = kwargs

    @abc.abstractmethod
    def apply_mask(self, x: Tensor, mask: Tensor) -> Tensor:
        pass

    @abc.abstractmethod
    def check_mask_shape(self, x: Tensor, mask: Tensor):
        """
        check whether the shape of mask is as expected
        """

    @abc.abstractmethod
    def count_bright_area(self, mask: Tensor) -> Tensor:
        """
        Bright area can be seen
        Dark area cannot be seen
        """

    @abc.abstractmethod
    def get_full_bright_mask(self, x: Tensor) -> Tensor:
        """
        Return a mask that is all bright
        """

    @abc.abstractmethod
    def apply_inpainting_mask(self, x_0: Tensor, x_t: Tensor, inpainting_mask: Tensor) -> Tensor:
        """
        1 represents the region that can be seen
        """
