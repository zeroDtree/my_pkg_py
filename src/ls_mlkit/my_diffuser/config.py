"""Official package"""

from typing import Any
import torch
from torch import Tensor
from copy import deepcopy

"""my package"""


class DiffusionConfig(object):
    def __init__(
        self,
        n_discretization_steps: int,
        ndim_micro_shape: int,
        custom_config: dict[str, Any] = {},
        denoise_at_final: bool = True,
        *args: list[Any],
        **kwargs: dict[Any, Any],
    ):
        super().__init__()  # type: ignore
        self.n_discretization_steps: int = n_discretization_steps
        self.custom_config: dict[str, Any] = custom_config
        self.ndim_micro_shape: int = ndim_micro_shape
        self.denoise_at_final: bool = denoise_at_final
        self.args = args
        self.kwargs = kwargs

    def to(self, device: torch.device | str | Tensor, inplace: bool = True) -> "DiffusionConfig":
        obj = self if inplace else deepcopy(self)
        if isinstance(device, Tensor):
            device = device.device
        for k, v in obj.__dict__.items():
            if isinstance(v, Tensor):
                setattr(obj, k, v.to(device))
        return obj
