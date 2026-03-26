from typing import List

import torch
from diffusers.utils import numpy_to_pil
from PIL import Image


def pt_to_pil(images: torch.Tensor) -> List[Image.Image]:
    """
    Convert a torch image to a PIL image. Range of input tensor is assumed to be [0, 1].
    """
    images = (images / 2 + 0.5).clamp(0, 1)
    images = images.cpu().permute(0, 2, 3, 1).float().numpy()
    images = numpy_to_pil(images)
    return images
