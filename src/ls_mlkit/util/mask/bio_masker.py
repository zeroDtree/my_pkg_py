from typing import Any

import torch
from torch import Tensor

from ...util.se3 import T
from .masker_interface import MaskerInterface


class BioCAOnlyMasker(MaskerInterface):

    def __init__(self, ndim_mini_micro_shape: int = 1, **kwargs: dict[Any, Any]):
        super().__init__(**kwargs)
        self.ndim_mini_micro_shape: int = ndim_mini_micro_shape

    def apply_mask(self, x: Tensor, mask: Tensor) -> Tensor:
        self.check_mask_shape(x, mask)
        return x * mask.view(*mask.shape, *[1 for _ in range(self.ndim_mini_micro_shape)])

    def check_mask_shape(self, x: Tensor, mask: Tensor):
        if self.ndim_mini_micro_shape == 0:
            assert x.shape == mask.shape
        else:
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
        shape = x.shape[: -self.ndim_mini_micro_shape] if self.ndim_mini_micro_shape != 0 else x.shape
        device = x.device

        return torch.ones(shape, device=device)

    def apply_inpainting_mask(self, x_0: Tensor, x_t: Tensor, inpainting_mask: Tensor) -> Tensor:
        """
        1 represents the region that can be seen
        """
        self.check_mask_shape(x_0, inpainting_mask)
        inpainting_mask = inpainting_mask.view(*inpainting_mask.shape, *[1 for _ in range(self.ndim_mini_micro_shape)])
        return x_t * (1 - inpainting_mask) + x_0 * inpainting_mask


class BioBackboneFrameMasker(MaskerInterface):
    def __init__(self, **kwargs: dict[Any, Any]):
        super().__init__(**kwargs)
        self.trans_ndim_mini_micro_shape: int = 1
        self.rots_ndim_mini_micro_shape: int = 2

    def apply_mask(self, x: T, mask: Tensor) -> Tensor:
        r"""
        Args:
            x: T, (b,n,3,3) rotation matrix and (b,n,3) translation vector
            mask: b, n
        """
        frames: T = x
        self.check_mask_shape(frames, mask)
        frames.trans *= mask.view(*mask.shape, *[1 for _ in range(self.trans_ndim_mini_micro_shape)])
        identity = torch.eye(3, device=frames.rots.device).view(1, 1, 3, 3)
        identity = identity.expand(frames.rots.shape[0], frames.rots.shape[1], 3, 3)
        rot_mask = mask.view(mask.shape[0], mask.shape[1], 1, 1)
        frames.rots = frames.rots * rot_mask + identity * (1 - rot_mask)
        return frames

    def check_mask_shape(self, x: T, mask: Tensor):
        assert x.trans[: -self.trans_ndim_mini_micro_shape] == mask.shape

    def count_bright_area(self, mask: Tensor) -> Tensor:
        """
        Bright area can be seen
        Dark area cannot be seen
        """
        return torch.sum(mask)

    def get_full_bright_mask(self, x: T) -> Tensor:
        """
        b, n, 3 -> b, n
        """
        x: Tensor = x.trans
        shape = x.shape[: -self.trans_ndim_mini_micro_shape]
        device = x.device

        return torch.ones(shape, device=device)

    def apply_inpainting_mask(self, x_0: T, x_t: T, inpainting_mask: Tensor) -> Tensor:
        """
        1 represents the region that can be seen
        """
        self.check_mask_shape(x_0, inpainting_mask)
        b, n = x_0.rots.shape[:2]
        inpainting_mask = inpainting_mask.view(
            *inpainting_mask.shape, *[1 for _ in range(self.trans_ndim_mini_micro_shape)]
        )
        x_t.trans = x_t.trans * (1 - inpainting_mask) + x_0.trans * inpainting_mask
        rot_inpainting_mask = inpainting_mask.view(b, n, 1, 1)
        x_t.rots = x_t.rots * (1 - rot_inpainting_mask) + x_0.rots * rot_inpainting_mask


class BioSO3Masker(MaskerInterface):
    def __init__(self, **kwargs: dict[Any, Any]):
        super().__init__(**kwargs)
        self.ndim_mini_micro_shape: int = 2

    def apply_mask(self, x: Tensor, mask: Tensor) -> Tensor:
        r"""
        Args:
            x: (b,n,3,3) rotation matrix
            mask: b, n
        """
        b, n = x.shape[:2]
        device = x.device
        self.check_mask_shape(x, mask)
        identity = torch.eye(3, device=device).view(1, 1, 3, 3)
        identity = identity.expand(b, n, -1, -1)
        rot_mask = mask.view(b, n, 1, 1)
        x = x * rot_mask + identity * (1 - rot_mask)
        return x

    def check_mask_shape(self, x: Tensor, mask: Tensor):
        # print(f"x.shape: {x.shape}, mask.shape: {mask.shape}")
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
        b, n = x_0.shape[:2]
        rot_inpainting_mask = inpainting_mask.view(b, n, 1, 1)
        x_t.rots = x_t * (1 - rot_inpainting_mask) + x_0 * rot_inpainting_mask
