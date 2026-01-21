import torch

from ..util.base_class.base_time_class import BaseTimeScheduler
from ..util.decorators import inherit_docstrings


@inherit_docstrings
class FlowMatchingTimeScheduler(BaseTimeScheduler):
    def initialize_timesteps_schedule(self) -> None:
        if self.num_inference_timesteps == self.num_train_timesteps:
            self._discrete_timesteps = torch.arange(0, self.num_train_timesteps, dtype=torch.long)
        else:
            self._discrete_timesteps = (
                torch.linspace(0, self.num_train_timesteps - 1, self.num_inference_timesteps).round().to(torch.long)
            )
        self._continuous_timesteps = torch.linspace(
            self.continuous_time_start, self.continuous_time_end, self.num_inference_timesteps + 1, dtype=torch.float32
        )
