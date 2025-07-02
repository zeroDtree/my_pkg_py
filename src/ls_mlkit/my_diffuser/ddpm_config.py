"""Official package"""

import torch
from torch import Tensor
from typing import Any

"""my package"""
from .config import DiffusionConfig


class DDPMConfig(DiffusionConfig):
    def __init__(
        self,
        n_discretization_steps: int = 1000,
        ndim_micro_shape: int = 2,
        denoise_at_final: bool = True,
        custom_config: dict[str, Any] = {},
    ):
        super().__init__(
            n_discretization_steps=n_discretization_steps,
            ndim_micro_shape=ndim_micro_shape,
            denoise_at_final=denoise_at_final,
            custom_config=custom_config,
        )
        self.betas: Tensor

        betas: Tensor | None = custom_config.get("betas", None)
        if betas is None:
            self.betas = torch.linspace(0.0001, 0.02, steps=self.n_discretization_steps)
        else:
            self.betas = betas

        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)  # expectation
        self.sqrt_1m_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)  # std
