from torch import Tensor

from .masker import Masker


class ImageMasker(Masker):

    def check_mask_shape(self, x: Tensor, mask: Tensor):
        if self.ndim_mini_micro_shape == 0:
            if mask.shape[-3] == 1:
                mask = mask.expand(-1, x.shape[-3], -1, -1)
            assert x.shape == mask.shape
        else:
            assert x.shape[: -self.ndim_mini_micro_shape] == mask.shape
