from typing import Any

import torch
from torch import Tensor

from .masker_interface import MaskerInterface


class Masker(MaskerInterface):

    def __init__(self, ndim_mini_micro_shape: int = 0, **kwargs: dict[Any, Any]):
        super().__init__(**kwargs)
        self.ndim_mini_micro_shape: int = ndim_mini_micro_shape

    def apply_mask(self, x: Tensor, mask: Tensor) -> Tensor:
        self.check_mask_shape(x, mask)
        if self.ndim_mini_micro_shape == 0:
            return x * mask
        else:
            return x * mask.view(*mask.shape, *[1 for _ in range(self.ndim_mini_micro_shape)])

    def check_mask_shape(self, x: Tensor, mask: Tensor):
        if self.ndim_mini_micro_shape == 0:
            assert x.shape == mask.shape
        else:
            assert x.shape[: -self.ndim_mini_micro_shape] == mask.shape

    def count_bright_area(self, mask: Tensor) -> Tensor:
        r"""
        Bright area can be seen
        Dark area cannot be seen
        """
        return torch.sum(mask)

    def get_full_bright_mask(self, x: Tensor) -> Tensor:
        if self.ndim_mini_micro_shape == 0:
            shape = x.shape
        else:
            shape = x.shape[: -self.ndim_mini_micro_shape]
        device = x.device

        return torch.ones(shape, device=device)

    def apply_inpainting_mask(self, x_0: Tensor, x_t: Tensor, inpainting_mask: Tensor) -> Tensor:
        r"""
        1 represents the region that can be seen
        """
        self.check_mask_shape(x_0, inpainting_mask)
        inpainting_mask = inpainting_mask.view(*inpainting_mask.shape, *[1 for _ in range(self.ndim_mini_micro_shape)])
        return x_t * (1 - inpainting_mask) + x_0 * inpainting_mask
