from abc import ABC, abstractmethod
from typing import Tuple

import torch
from torch import Tensor

from ..decorators import inherit_docstrings


@inherit_docstrings
class BaseTimeScheduler(ABC):
    r"""Base class for time schedulers in diffusion models.

    Notation Convention
    -------------------
    Let the total diffusion time be :math:`T`, discretized into :math:`N` diffusion steps,
    corresponding to :math:`N+1` continuous time points:

    .. math::
        0 = t_0 < t_1 < \cdots < t_N = T

    where :math:`\{t_i\}_{i=0}^N` represents continuous time. For uniform discretization:

    .. math::
        t_i = \frac{i}{N} \cdot T

    The corresponding discrete time steps are defined as:

    .. math::
        i \in \{0, 1, \ldots, N\}

    In diffusion models, :math:`t_0` corresponds to the clean data distribution :math:`q(x_0)`,
    so training and sampling typically only consider:

    .. math::
        i \in \{1, \ldots, N\}

    For engineering convenience (0-based array indexing), we use:

    .. math::
        \text{idx} = i - 1

    Therefore:

    - ``idx = 0`` corresponds to discrete step :math:`i=1`, i.e., continuous time :math:`t_1`
    - ``idx = N-1`` corresponds to discrete step :math:`i=N`, i.e., continuous time :math:`t_N = T`

    In this implementation:

    - ``num_train_timesteps`` = :math:`N` (number of diffusion steps)
    - ``timestep_index`` (or ``idx``) :math:`\in \{\text{idx\_start}, \ldots, \text{idx\_start} + N - 1\}`
    - ``continuous_time`` :math:`\in [t_1, t_N] = [\frac{T}{N}, T]` for training/sampling

    The ``idx_start`` parameter controls the starting value of timestep indices:

    - When ``idx_start = 0`` (default): :math:`\text{idx} = i - 1`, so :math:`\text{idx} \in \{0, \ldots, N-1\}`
    - When ``idx_start = 1``: :math:`\text{idx} = i`, so :math:`\text{idx} \in \{1, \ldots, N\}`

    Args:
        continuous_time_start (``float``, *optional*): The start of continuous time range (typically 0). Defaults to 0.0.
        continuous_time_end (``float``, *optional*): The end of continuous time range (i.e., :math:`T`). Defaults to 1.0.
        num_train_timesteps (``int``, *optional*): Number of diffusion steps :math:`N`. Defaults to 1000.
        num_inference_steps (``int``, *optional*): Number of inference steps. If None, uses ``num_train_timesteps``. Defaults to None.
        idx_start (``int``, *optional*): The starting value for timestep indices.
            Set to 1 if you prefer 1-based indexing where idx directly equals the discrete step i. Defaults to 0.
    """

    def __init__(
        self,
        continuous_time_start: float = 0.0,
        continuous_time_end: float = 1.0,
        num_train_timesteps: int = 1000,
        num_inference_steps: int = None,
        idx_start: int = 0,
    ) -> None:
        self.continuous_time_start: float = continuous_time_start
        self.continuous_time_end: float = continuous_time_end
        self.num_train_timesteps: int = num_train_timesteps  # This is N
        self.num_inference_timesteps: int = (
            num_inference_steps if num_inference_steps is not None else num_train_timesteps
        )
        self.idx_start: int = idx_start  # Starting value for timestep indices
        self._timesteps_idx: Tensor = None  # Stores timestep indices
        self._continuous_timesteps: Tensor = None  # Stores continuous times
        self.T: float = continuous_time_end - continuous_time_start
        self.initialize_timesteps_schedule()

    @abstractmethod
    def initialize_timesteps_schedule(self) -> None:
        """Initialize timesteps schedule for sampling/inference."""

    def continuous_time_to_timestep_index(self, continuous_time: Tensor) -> Tensor:
        r"""Convert continuous time to timestep index.

        Given continuous time :math:`t`, compute the timestep index:

        .. math::
            \text{idx} = \text{round}\left(\frac{t - t_0}{T} \cdot N\right) - 1 + \text{idx\_start}

        where :math:`t_0` is ``continuous_time_start``, :math:`T` is the total time span,
        :math:`N` is ``num_train_timesteps``, and :math:`\text{idx\_start}` is the starting index.

        The result is clamped to :math:`[\text{idx\_start}, \text{idx\_start} + N - 1]`.

        Args:
            continuous_time (``Tensor``): Continuous time values :math:`t \in [t_0, t_0 + T]`.

        Returns:
            ``Tensor``: Timestep indices :math:`\text{idx} \in \{\text{idx\_start}, \ldots, \text{idx\_start} + N - 1\}`.
        """
        # Normalize time to [0, 1] range, then scale to [0, N]
        normalized = (continuous_time - self.continuous_time_start) / self.T
        # idx = round(normalized * N) - 1 + idx_start
        return torch.clamp(
            torch.round(normalized * self.num_train_timesteps) - 1 + self.idx_start,
            min=self.idx_start,
            max=self.idx_start + self.num_train_timesteps - 1,
        ).to(torch.int64)

    def timestep_index_to_continuous_time(self, timestep_index: Tensor) -> Tensor:
        r"""Convert timestep index to continuous time.

        Given timestep index :math:`\text{idx}`, compute the continuous time:

        .. math::
            t = t_0 + \frac{\text{idx} + 1 - \text{idx\_start}}{N} \cdot T

        where :math:`t_0` is ``continuous_time_start``, :math:`T` is the total time span,
        :math:`N` is ``num_train_timesteps``, and :math:`\text{idx\_start}` is the starting index.

        Args:
            timestep_index (``Tensor``): Timestep indices :math:`\text{idx} \in \{\text{idx\_start}, \ldots, \text{idx\_start} + N - 1\}`.

        Returns:
            ``Tensor``: Continuous time values :math:`t \in [t_1, t_N]`.
        """
        # t = t_0 + (idx + 1 - idx_start) / N * T
        return (
            self.continuous_time_start
            + (timestep_index + 1 - self.idx_start).float() / self.num_train_timesteps * self.T
        )

    def get_timestep_indices_schedule(self) -> Tensor:
        r"""Get the timestep indices schedule for sampling/inference.

        Returns:
            ``Tensor``: 1D tensor of timestep indices :math:`\text{idx} \in \{\text{idx\_start}, \ldots, \text{idx\_start} + N - 1\}`.
        """
        assert self._timesteps_idx is not None, "timestep indices schedule is not set"
        assert isinstance(self._timesteps_idx, Tensor), "timestep indices must be a Tensor"
        assert self._timesteps_idx.ndim == 1, "timestep indices must be a 1D Tensor"
        return self._timesteps_idx

    def get_continuous_timesteps_schedule(self) -> Tensor:
        r"""Get the continuous timesteps schedule for sampling/inference.

        Returns:
            ``Tensor``: 1D tensor of continuous time values :math:`t \in [t_1, t_N]`.
        """
        assert self._continuous_timesteps is not None, "continuous timesteps schedule is not set"
        assert isinstance(self._continuous_timesteps, Tensor), "continuous timesteps must be a Tensor"
        assert self._continuous_timesteps.ndim == 1, "continuous timesteps must be a 1D Tensor"
        return self._continuous_timesteps

    def set_timestep_indices_schedule(self, timestep_indices: Tensor) -> None:
        r"""Set the timestep indices schedule for sampling/inference.

        Args:
            timestep_indices (``Tensor``): 1D tensor of timestep indices :math:`\text{idx} \in \{\text{idx\_start}, \ldots, \text{idx\_start} + N - 1\}`.
        """
        self._timesteps_idx = timestep_indices

    def set_continuous_timesteps_schedule(self, continuous_timesteps: Tensor) -> None:
        r"""Set the continuous timesteps schedule for sampling/inference.

        Args:
            continuous_timesteps (``Tensor``): 1D tensor of continuous time values.
        """
        self._continuous_timesteps = continuous_timesteps

    def sample_timestep_index_uniformly(
        self, macro_shape: Tuple[int, ...], same_for_all_samples: bool = False
    ) -> Tensor:
        r"""Sample timestep indices uniformly from :math:`\{\text{idx\_start}, \ldots, \text{idx\_start} + N - 1\}`.

        This corresponds to sampling discrete steps :math:`i` uniformly from :math:`\{1, \ldots, N\}`
        and converting to index via :math:`\text{idx} = i - 1 + \text{idx\_start}`.

        Args:
            macro_shape (``Tuple[int, ...]``): Shape of the output tensor.
            same_for_all_samples (``bool``, *optional*): If True, use the same timestep index for all samples. Defaults to False.

        Returns:
            ``Tensor``: Timestep indices with shape ``macro_shape``.
        """
        idx_min = self.idx_start
        idx_max = self.idx_start + self.num_train_timesteps  # exclusive upper bound
        if same_for_all_samples:
            return torch.ones(macro_shape, dtype=torch.int64) * torch.randint(idx_min, idx_max, (1,), dtype=torch.int64)
        else:
            return torch.randint(idx_min, idx_max, macro_shape, dtype=torch.int64)

    def sample_continuous_time_uniformly(
        self, macro_shape: Tuple[int, ...], same_for_all_samples: bool = False
    ) -> Tensor:
        r"""Sample continuous time uniformly from :math:`[t_1, t_N]`.

        Note: This samples from :math:`[t_0 + \frac{T}{N}, t_0 + T]` to exclude :math:`t_0`
        (the clean data point).

        Args:
            macro_shape (``Tuple[int, ...]``): Shape of the output tensor.
            same_for_all_samples (``bool``, *optional*): If True, use the same time for all samples. Defaults to False.

        Returns:
            ``Tensor``: Continuous time values with shape ``macro_shape``.
        """
        # Sample from [t_1, t_N] = [t_0 + T/N, t_0 + T]
        t_min = self.continuous_time_start + self.T / self.num_train_timesteps  # t_1
        t_max = self.continuous_time_end  # t_N = T
        t_range = t_max - t_min

        if same_for_all_samples:
            return torch.ones(macro_shape) * (torch.rand(1) * t_range + t_min)
        else:
            return torch.rand(macro_shape) * t_range + t_min


if __name__ == "__main__":
    """
    uv run python -m ls_mlkit.util.base_class.base_time_class
    """
    from pathlib import Path

    import matplotlib.pyplot as plt

    # Create a concrete implementation for testing
    class TestTimeScheduler(BaseTimeScheduler):
        def initialize_timesteps_schedule(self) -> None:
            self._timesteps_idx = torch.arange(
                self.idx_start + self.num_train_timesteps - 1,
                self.idx_start - 1,
                -1,
                dtype=torch.int64,
            )
            self._continuous_timesteps = torch.linspace(
                self.continuous_time_end,
                self.continuous_time_start + self.T / self.num_train_timesteps,
                self.num_inference_timesteps,
            )

    # Test parameters
    num_samples = 100000
    num_train_timesteps = 1000
    macro_shape = (num_samples,)

    # Create scheduler instance
    scheduler = TestTimeScheduler(
        continuous_time_start=0.0,
        continuous_time_end=1.0,
        num_train_timesteps=num_train_timesteps,
        idx_start=0,
    )

    # Sample timestep indices and continuous times
    timestep_indices = scheduler.sample_timestep_index_uniformly(macro_shape)
    continuous_times = scheduler.sample_continuous_time_uniformly(macro_shape)

    # Convert to numpy for plotting
    timestep_indices_np = timestep_indices.numpy()
    continuous_times_np = continuous_times.numpy()

    # Create figure with 2 subplots
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Histogram of timestep indices
    ax1 = axes[0]
    counts, bins, _ = ax1.hist(
        timestep_indices_np,
        bins=min(50, num_train_timesteps),
        density=True,
        alpha=0.7,
        color="steelblue",
        edgecolor="white",
    )
    # Add theoretical uniform distribution line
    expected_density = 1.0 / num_train_timesteps
    ax1.axhline(y=expected_density, color="red", linestyle="--", linewidth=2, label=f"Expected: {expected_density:.6f}")
    ax1.set_xlabel("Timestep Index", fontsize=12)
    ax1.set_ylabel("Density", fontsize=12)
    ax1.set_title(f"Distribution of Timestep Indices\n(N={num_train_timesteps}, samples={num_samples})", fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Add statistics text
    stats_text1 = f"Mean: {timestep_indices_np.mean():.2f}\nStd: {timestep_indices_np.std():.2f}\nMin: {timestep_indices_np.min()}\nMax: {timestep_indices_np.max()}"
    ax1.text(
        0.02,
        0.98,
        stats_text1,
        transform=ax1.transAxes,
        fontsize=10,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    # Plot 2: Histogram of continuous times
    ax2 = axes[1]
    t_min = scheduler.continuous_time_start + scheduler.T / num_train_timesteps
    t_max = scheduler.continuous_time_end
    t_range = t_max - t_min

    counts2, bins2, _ = ax2.hist(
        continuous_times_np,
        bins=50,
        density=True,
        alpha=0.7,
        color="darkorange",
        edgecolor="white",
    )
    # Add theoretical uniform distribution line
    expected_density2 = 1.0 / t_range
    ax2.axhline(
        y=expected_density2, color="red", linestyle="--", linewidth=2, label=f"Expected: {expected_density2:.4f}"
    )
    ax2.set_xlabel("Continuous Time", fontsize=12)
    ax2.set_ylabel("Density", fontsize=12)
    ax2.set_title(
        f"Distribution of Continuous Time\n(range=[{t_min:.4f}, {t_max:.4f}], samples={num_samples})", fontsize=14
    )
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Add statistics text
    stats_text2 = f"Mean: {continuous_times_np.mean():.4f}\nStd: {continuous_times_np.std():.4f}\nMin: {continuous_times_np.min():.4f}\nMax: {continuous_times_np.max():.4f}"
    ax2.text(
        0.02,
        0.98,
        stats_text2,
        transform=ax2.transAxes,
        fontsize=10,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    plt.tight_layout()

    # Save the figure
    output_dir = Path(__file__).parent.parent.parent.parent.parent / "test"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "time_distribution_test.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Figure saved to: {output_path}")

    plt.show()
