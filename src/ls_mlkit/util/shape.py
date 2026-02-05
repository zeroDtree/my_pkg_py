from typing import Tuple, Union

from torch import Tensor


def get_macroscopic_shape(obj: Union[Tensor, tuple], ndim_microscopic: int) -> Tuple[int]:
    """
    Get the macroscopic shape of an object.
    """
    if isinstance(obj, tuple):
        if len(obj) == ndim_microscopic:
            result = (1,)
        else:
            result = obj[:-ndim_microscopic]
    elif isinstance(obj, Tensor):
        if obj.ndim == ndim_microscopic:
            result = (1,)
        else:
            result = obj.shape[:-ndim_microscopic]
    else:
        raise ValueError(f"Invalid type: {type(obj)}")

    return result


def show_shape(x, prefix=""):
    import numpy as np
    import torch

    if torch.is_tensor(x):
        print(prefix, "Tensor", tuple(x.shape), x.dtype, x.device)
    elif isinstance(x, dict):
        for k, v in x.items():
            show_shape(v, prefix + f"{k}: ")
    elif isinstance(x, (list, tuple)):
        print(prefix, type(x), len(x))
        for i, v in enumerate(x[:3]):
            show_shape(v, prefix + f"[{i}] ")
    elif isinstance(x, np.ndarray):
        print(prefix, "ndarray", x.shape, x.dtype)
    else:
        print(prefix, type(x))
