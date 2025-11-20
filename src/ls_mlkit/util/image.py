from typing import List

import numpy as np
import torch
from PIL import Image


def pt_to_pil(images: torch.Tensor) -> List[Image.Image]:
    """
    Convert a torch image to a PIL image. Range of input tensor is assumed to be [0, 1].
    """
    images = (images / 2 + 0.5).clamp(0, 1)
    images = images.cpu().permute(0, 2, 3, 1).float().numpy()
    images = numpy_to_pil(images)
    return images


def numpy_to_pil(images: np.ndarray) -> List[Image.Image]:
    """
    Convert a numpy image or a batch of images to a PIL image. Range of input tensor is assumed to be [0, 1].
    """
    if images.ndim == 3:
        images = images[None, ...]
    images = (images * 255).round().astype("uint8")
    if images.shape[-1] == 1:
        # special case for grayscale (single channel) images
        pil_images = [Image.fromarray(image.squeeze(), mode="L") for image in images]
    else:
        pil_images = [Image.fromarray(image) for image in images]

    return pil_images
