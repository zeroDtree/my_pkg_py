r"""
Time Scheduler for Diffusion
"""

import torch

from ..util.base_class.base_time_class import BaseTimeScheduler
from ..util.decorators import inherit_docstrings


@inherit_docstrings
class DiffusionTimeScheduler(BaseTimeScheduler):
    def initialize_timesteps_schedule(self) -> None:
        # Timestep indices: [idx_start + N - 1, ..., idx_start] (descending for reverse diffusion)
        idx_min = self.idx_start
        idx_max = self.idx_start + self.num_train_timesteps - 1

        if self.num_inference_timesteps == self.num_train_timesteps:
            self._timesteps_idx = torch.arange(idx_max, idx_min - 1, -1, dtype=torch.int64)
        else:
            self._timesteps_idx = (
                torch.linspace(idx_min, idx_max, self.num_inference_timesteps).round().flip(0).to(torch.int64)
            )

        # Continuous times: [t_N, ..., t_1] (descending), excluding t_0
        # t_1 = t_0 + T/N, t_N = t_0 + T
        t_min = self.continuous_time_start + self.T / self.num_train_timesteps  # t_1
        t_max = self.continuous_time_end  # t_N

        self._continuous_timesteps = torch.linspace(t_max, t_min, self.num_inference_timesteps).to(torch.float32)


if __name__ == "__main__":
    """
    uv run python -m ls_mlkit.diffusion.time_scheduler
    """
    scheduler = DiffusionTimeScheduler(
        continuous_time_start=0.0,
        continuous_time_end=1.0,
        num_train_timesteps=1000,
        idx_start=0,
    )
    print(scheduler.get_timestep_indices_schedule())
    print(scheduler.get_continuous_timesteps_schedule())
    print(scheduler.timestep_index_to_continuous_time(scheduler.get_timestep_indices_schedule()))
    print(scheduler.continuous_time_to_timestep_index(scheduler.get_continuous_timesteps_schedule()))
