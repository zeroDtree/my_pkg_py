import torch

from ..util.base_class.base_time_class import BaseTimeScheduler
from ..util.decorators import inherit_docstrings


@inherit_docstrings
class FlowMatchingTimeScheduler(BaseTimeScheduler):
    def initialize_timesteps_schedule(self) -> None:
        # Timestep indices: [idx_start, ..., idx_start + N - 1] (ascending for flow matching)
        idx_min = self.idx_start
        idx_max = self.idx_start + self.num_train_timesteps - 1

        if self.num_inference_timesteps == self.num_train_timesteps:
            self._timesteps_idx = torch.arange(idx_min, idx_max + 1, dtype=torch.int64)
        else:
            self._timesteps_idx = torch.linspace(idx_min, idx_max, self.num_inference_timesteps).round().to(torch.int64)

        # ODE boundaries: [t_0, ..., t_N] (ascending)
        self._continuous_boundaries = torch.linspace(
            self.continuous_time_start,
            self.continuous_time_end,
            self.num_inference_timesteps + 1,
            dtype=torch.float32,
        )
        # Interior knot convention [t_1, ..., t_N], derived from boundaries
        self._continuous_timesteps = self._continuous_boundaries[1:]
