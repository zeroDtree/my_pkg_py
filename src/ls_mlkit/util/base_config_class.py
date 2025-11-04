from copy import deepcopy
from typing import Any

import torch
from torch import Tensor


class BaseConfigClass:
    def __init__(self, *args: list[Any], **kwargs: dict[Any, Any]):
        pass

    def to(self, device: torch.device | str | Tensor, inplace: bool = True) -> "BaseConfigClass":
        """Move the config to the given device

        Args:
            device (torch.device | str | Tensor): the device to move the config to
            inplace (bool, optional): whether to move the config in place. Defaults to True.

        Returns:
            BaseConfigClass: the config moved to the given device
        """
        obj = self if inplace else deepcopy(self)
        if isinstance(device, Tensor):
            device = device.device
        for k, v in obj.__dict__.items():
            if isinstance(v, Tensor):
                setattr(obj, k, v.to(device))
        return obj
