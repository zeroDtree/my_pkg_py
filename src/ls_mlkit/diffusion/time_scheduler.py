r"""
Time Scheduler for Diffusion
"""

import abc
from typing import Tuple

import torch
from torch import Tensor

from ..util.decorators import inherit_docstrings
from ..util.base_class.base_time_class import BaseTimeScheduler


@inherit_docstrings
class DiffusionTimeScheduler(BaseTimeScheduler):
    def initialize_timesteps_schedule(self) -> None:
        if self.num_inference_timesteps == self.num_train_timesteps:
            self._discrete_timesteps = torch.arange(self.num_train_timesteps - 1, -1, -1, dtype=torch.int64)
        else:
            self._discrete_timesteps = (
                torch.linspace(0, self.num_train_timesteps - 1, self.num_inference_timesteps)
                .round()
                .flip(0)
                .to(torch.int64)
            )
        self._continuous_timesteps = (
            torch.linspace(self.continuous_time_start, self.continuous_time_end, self.num_inference_timesteps + 1)
            .flip(0)
            .to(torch.float32)
        )
