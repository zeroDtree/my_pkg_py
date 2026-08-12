from typing import Any

import torch
from torch import Tensor

from ..util.decorators import inherit_docstrings
from .base_fm import BaseFlow, BaseFlowConfig
from .time_scheduler import FlowMatchingTimeScheduler


@inherit_docstrings
class IndependentCFMFlowConfig(BaseFlowConfig):
    def __init__(
        self,
        n_discretization_steps: int,
        ndim_micro_shape: int = 2,
        n_inference_steps: int | None = None,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            ndim_micro_shape=ndim_micro_shape,
            n_discretization_steps=n_discretization_steps,
            n_inference_steps=n_inference_steps,
            **kwargs,
        )


@inherit_docstrings
class IndependentCFMFlow(BaseFlow):
    def __init__(
        self,
        config: IndependentCFMFlowConfig,
        time_scheduler: FlowMatchingTimeScheduler,
    ) -> None:
        super().__init__(config=config, time_scheduler=time_scheduler)
        self.config: IndependentCFMFlowConfig = config

    def prior_sampling(self, shape) -> torch.Tensor:
        return torch.randn(shape)

    def sample_x_0(self, x_1: Tensor) -> Tensor:
        return torch.randn_like(x_1, device=x_1.device)
