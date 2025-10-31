r"""
Time Scheduler for Diffusion
"""

import abc
from typing import Tuple

import torch
from torch import Tensor


class TimeScheduler(abc.ABC):
    """
    Time Scheduler for Diffusion
    """

    def __init__(
        self,
        continuous_time_start: float = 0.0,
        continuous_time_end: float = 1.0,
        num_train_timesteps: int = 1000,
        num_inference_steps: int = None,
    ):
        self.continuous_time_start = continuous_time_start
        self.continuous_time_end = continuous_time_end
        self.num_train_timesteps = num_train_timesteps
        self.num_inference_timesteps = num_inference_steps if num_inference_steps is not None else num_train_timesteps
        self._discrete_timesteps: list[int] = None
        self._continuous_timesteps: list[float] = None
        self.T = continuous_time_end - continuous_time_start
        self.initialize_timesteps_schedule()

    def initialize_timesteps_schedule(self):
        r"""Initialize timesteps schedule for DDPM sampling.
        For DDPM, we should use ALL timesteps in reverse order for best quality.
        For DDIM-style sampling, use linear spacing.
        """
        if self.num_inference_timesteps == self.num_train_timesteps:
            # Standard DDPM: use all timesteps in reverse order
            self._discrete_timesteps = torch.arange(self.num_train_timesteps - 1, -1, -1, dtype=torch.int64)
        else:
            # DDIM-style: linear spacing for fewer steps
            self._discrete_timesteps = (
                torch.linspace(0, self.num_train_timesteps - 1, self.num_inference_timesteps)
                .round()
                .flip(0)
                .to(torch.int64)
            )

        # Continuous timesteps (for compatibility)
        self._continuous_timesteps = (
            torch.linspace(self.continuous_time_start, self.continuous_time_end, self.num_inference_timesteps)
            .flip(0)
            .to(torch.float32)
        )

    def continuous_time_to_discrete_time(self, continuous_time: Tensor) -> Tensor:
        """Convert a continuous time to a discrete time.

        Args:
            continuous_time (Tensor): the continuous time

        Returns:
            Tensor: the discrete time
        """
        return torch.round(continuous_time / self.T * (self.num_train_timesteps - 1))

    def discrete_time_to_continuous_time(self, discrete_time: Tensor) -> Tensor:
        """Convert a discrete time to a continuous time.

        Args:
            discrete_time (Tensor): the discrete time

        Returns:
            Tensor: the continuous time
        """
        return discrete_time / (self.num_train_timesteps - 1) * self.T

    def get_discrete_timesteps_schedule(self) -> Tensor:
        """Get the discrete one-dimensional timesteps for sampling/inference.

        Returns:
            Tensor: the discrete one-dimensional timesteps for sampling/inference
        """
        assert self._discrete_timesteps is not None, "discrete_timesteps is not set"
        assert isinstance(self._discrete_timesteps, Tensor), "discrete_timesteps must be a Tensor"
        assert self._discrete_timesteps.ndim == 1, "discrete_timesteps must be a one-dimensional Tensor"
        return self._discrete_timesteps

    def get_continuous_timesteps_schedule(self) -> Tensor:
        """Get the continuous one-dimensional timesteps for sampling/inference.

        Returns:
            Tensor: the continuous one-dimensional timesteps for sampling/inference
        """
        assert self._continuous_timesteps is not None, "_continuous_timesteps is not set"
        assert isinstance(self._continuous_timesteps, Tensor), "continuous_timesteps must be a Tensor"
        assert self._continuous_timesteps.ndim == 1, "continuous_timesteps must be a one-dimensional Tensor"
        return self._continuous_timesteps

    def set_discrete_timesteps_schedule(self, discrete_timesteps: list[int]):
        """Set the discrete one-dimensional timesteps for sampling/inference.

        Args:
            discrete_timesteps (list[int]): the discrete one-dimensional timesteps for sampling/inference
        """
        self._discrete_timesteps = discrete_timesteps

    def set_continuous_timesteps_schedule(self, continuous_timesteps: list[float]):
        """Set the continuous one-dimensional timesteps for sampling/inference.

        Args:
            continuous_timesteps (list[float]): the continuous one-dimensional timesteps for sampling/inference
        """
        self._continuous_timesteps = continuous_timesteps

    def sample_a_discrete_time_step_uniformly(
        self, macro_shape: Tuple[int, ...], same_timesteps_for_all_samples: bool = False
    ) -> Tensor:
        """Sample a discrete timestep in the range of [0, num_train_timesteps - 1] uniformly.

        Args:
            macro_shape (Tuple[int, ...]): the macro shape of the samples
            same_timesteps_for_all_samples (bool, optional): whether to use the same timestep for all samples. Defaults to False.

        Returns:
            Tensor: the discrete timestep
        """
        if same_timesteps_for_all_samples:
            return torch.ones(macro_shape, dtype=torch.int64) * torch.randint(
                0, self.num_train_timesteps, (1,), dtype=torch.int64
            )
        else:
            return torch.randint(0, self.num_train_timesteps, macro_shape, dtype=torch.int64)

    def sample_a_continuous_time_step_uniformly(
        self, macro_shape: Tuple[int, ...], same_timesteps_for_all_samples: bool = False
    ) -> Tensor:
        """Sample a continuous timestep in the range of [continuous_time_start, continuous_time_end] uniformly.

        Args:
            macro_shape (Tuple[int, ...]): the macro shape of the samples
            same_timesteps_for_all_samples (bool, optional): whether to use the same timestep for all samples. Defaults to False.

        Returns:
            Tensor: the continuous timestep
        """
        if same_timesteps_for_all_samples:
            return torch.ones(macro_shape) * torch.rand(1) * self.T + self.continuous_time_start
        else:
            return torch.rand(macro_shape) * self.T + self.continuous_time_start
