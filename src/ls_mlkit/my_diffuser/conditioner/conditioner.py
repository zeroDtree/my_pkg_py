"""Official package"""

import torch
from torch import Tensor
import abc
from typing import Any

"""my package"""
from ls_mlkit.my_diffuser.config import DiffusionConfig


class Conditioner(abc.ABC):
    def __init__(self, diffusion_config: DiffusionConfig, guidance_scale: float = 1.0):
        self.diffusion_config: DiffusionConfig = diffusion_config
        self.enabled: bool = True
        self.ready: bool = False
        self._guidance_scale: float = guidance_scale

    @abc.abstractmethod
    def prepare_condition_dict(self, train: bool = True, *args: list[Any], **kwargs: dict[Any, Any]) -> dict[str, Any]:
        pass

    @abc.abstractmethod
    def set_condition(self, *args: list[Any], **kwargs: dict[Any, Any]): ...

    @abc.abstractmethod
    def get_conditional_score(self, x_t: Tensor, t: Tensor, mask: Tensor) -> Tensor:
        pass

    @property
    def guidance_scale(self):
        return self._guidance_scale

    @guidance_scale.setter
    def set_guidance_scale(self, guidance_scale: float):
        self._guidance_scale = guidance_scale

    def enable(self):
        self.enabled = True

    def disable(self):
        self.enabled = False


class LGDConditioner(Conditioner):
    def __init__(
        self,
        diffusion_config: DiffusionConfig,
        guidance_scale: float = 1.0,
    ):
        super().__init__(diffusion_config, guidance_scale)

    @abc.abstractmethod
    def compute_conditional_loss(self, x_t: Tensor, t: Tensor, mask: Tensor) -> Tensor:
        pass

    def get_conditional_score(self, x_t: Tensor, t: Tensor, mask: Tensor) -> Tensor:
        assert self.ready == True, "Conditioner is not ready, please call set_condition first"
        with torch.autograd.set_detect_anomaly(True, check_nan=True):
            x = x_t.detach().clone().requires_grad_(True)
            conditional_loss = self.compute_conditional_loss(x, t, mask)
            grad = torch.autograd.grad(conditional_loss, x, create_graph=True)[0]
            score = -grad
        return score * self.guidance_scale
