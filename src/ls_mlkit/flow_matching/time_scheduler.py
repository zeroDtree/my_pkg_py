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

        # Continuous times: [t_1, ..., t_N] (ascending), excluding t_0
        # t_1 = t_0 + T/N, t_N = t_0 + T
        t_min = self.continuous_time_start + self.T / self.num_train_timesteps  # t_1
        t_max = self.continuous_time_end  # t_N

        self._continuous_timesteps = torch.linspace(t_min, t_max, self.num_inference_timesteps, dtype=torch.float32)
