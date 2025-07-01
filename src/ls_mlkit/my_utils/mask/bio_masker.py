import torch
from torch import Tensor
from .masker_interface import MaskerInterface
from typing import Any


class BioCAOnlyMasker(MaskerInterface):

    def __init__(self, ndim_mini_micro_shape: int = 1, **kwargs: dict[Any, Any]):
        super().__init__(ndim_mini_micro_shape, **kwargs)

    def apply_mask(self, x: Tensor, mask: Tensor) -> Tensor:
        self.check_mask_shape(x, mask)
        return x * mask.view(*mask.shape, *[1 for _ in range(self.ndim_mini_micro_shape)])

    def check_mask_shape(self, x: Tensor, mask: Tensor):
        assert x.shape[: -self.ndim_mini_micro_shape] == mask.shape

    def count_bright_area(self, mask: Tensor) -> Tensor:
        """
        Bright area can be seen
        Dark area cannot be seen
        """
        return torch.sum(mask)

    def get_full_bright_mask(self, x: Tensor) -> Tensor:
        """
        b, n, 3 -> b, n
        """
        shape = x.shape[: -self.ndim_mini_micro_shape]
        device = x.device

        return torch.ones(shape, device=device)

    def apply_inpainting_mask(self, x_0: Tensor, x_t: Tensor, inpainting_mask: Tensor) -> Tensor:
        """
        1 represents the region that can be seen
        """
        self.check_mask_shape(x_0, inpainting_mask)
        inpainting_mask = inpainting_mask.view(*inpainting_mask.shape, *[1 for _ in range(self.ndim_mini_micro_shape)])
        return x_t * (1 - inpainting_mask) + x_0 * inpainting_mask
