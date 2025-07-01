"""Official package"""

import torch
from torch import Tensor
from typing import Literal, Any

"""my package"""
from .sde import VPSDE, VESDE, SubVPSDE


class DiffusionConfig(object):
    def __init__(
        self,
        diffusion_type: Literal["DDPM", "VPSDE", "VESDE", "SubVPSDE"] = "DDPM",
        continuous: bool = False,
        n_discretization_steps: int = 1000,
        custom_config: dict[str, Any] = {},
        ndim_micro_shape: int = 2,
        denoise_at_final: bool = True,
    ):
        super().__init__()  # type: ignore
        self.diffusion_type: Literal["DDPM", "VPSDE", "VESDE", "SubVPSDE"] = diffusion_type
        self.continuous: bool = continuous
        self.n_discretization_steps: int = n_discretization_steps
        self.custom_config: dict[str, Any] = custom_config
        self.ndim_micro_shape: int = ndim_micro_shape
        self.denoise_at_final: bool = denoise_at_final

        if self.diffusion_type == "DDPM":
            betas: Tensor | None = custom_config.get("betas", None)
            if betas is None:
                self.betas: Tensor = torch.linspace(0.0001, 0.02, steps=self.n_discretization_steps)
            else:
                self.betas: Tensor = betas

            self.alphas = 1.0 - self.betas
            self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
            self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)  # expectation
            self.sqrt_1m_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)  # std
        elif self.diffusion_type == "VPSDE":
            self.beta_min: float = custom_config.get("beta_min", 0.1)
            self.beta_max: float = custom_config.get("beta_max", 20)
            self.sde = VPSDE(
                beta_min=self.beta_min,
                beta_max=self.beta_max,
                n_discretization_steps=self.n_discretization_steps,
                ndim_micro_shape=self.ndim_micro_shape,
            )
        elif self.diffusion_type == "VESDE":
            sigma_min = custom_config.get("sigma_min", 0.01)
            sigma_max = custom_config.get("sigma_max", 50)
            self.sde = VESDE(
                sigma_min=sigma_min,
                sigma_max=sigma_max,
                n_discretization_steps=self.n_discretization_steps,
                ndim_micro_shape=self.ndim_micro_shape,
            )
        elif self.diffusion_type == "SubVPSDE":
            self.beta_min = custom_config.get("beta_min", 0.1)
            self.beta_max = custom_config.get("beta_max", 20)
            self.sde = SubVPSDE(
                beta_min=self.beta_min,
                beta_max=self.beta_max,
                n_discretization_steps=self.n_discretization_steps,
                ndim_micro_shape=self.ndim_micro_shape,
            )
        else:
            raise ValueError(f"Invalid diffusion type: {self.diffusion_type}")
