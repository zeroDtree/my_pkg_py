"""Exponential moving average of module parameters."""

from __future__ import annotations

from copy import deepcopy

import torch
from torch.nn import Module


class EMA:
    """Keep a decayed shadow copy of a module's parameters for inference."""

    def __init__(self, module: Module, decay: float = 0.999) -> None:
        self.decay = decay
        self.shadow = deepcopy(module).eval().requires_grad_(False)

    def _sync_device(self, module: Module) -> None:
        device = next(module.parameters()).device
        if next(self.shadow.parameters()).device != device:
            self.shadow.to(device)

    @torch.no_grad()
    def update(self, module: Module) -> None:
        """Lerp shadow parameters toward the current module parameters."""
        self._sync_device(module)
        for s_param, param in zip(self.shadow.parameters(), module.parameters(), strict=True):
            s_param.data.lerp_(param.data, 1.0 - self.decay)

    @torch.no_grad()
    def copy_to(self, module: Module) -> None:
        """Write shadow parameters into ``module`` in-place."""
        self._sync_device(module)
        for s_param, param in zip(self.shadow.parameters(), module.parameters(), strict=True):
            param.data.copy_(s_param.data)
