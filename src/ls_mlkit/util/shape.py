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
