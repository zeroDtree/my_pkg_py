"""Training step counts aligned with DistributedSampler + DataLoader iteration."""

from __future__ import annotations

import math
from typing import Literal


def steps_per_epoch(dataset_len: int, num_processes: int, batch_size: int) -> int:
    """Batches per rank per epoch: ceil( ceil(N / P) / B )."""
    samples_per_rank = math.ceil(dataset_len / num_processes)
    return math.ceil(samples_per_rank / batch_size)


def get_total_training_steps(
    *,
    train_strategy: Literal["epochs", "steps"],
    dataset_len: int,
    n_epochs: int,
    n_steps: int,
    num_processes: int,
    batch_size: int,
) -> int:
    """Total training steps (batch iterations) for LR / conditioner schedulers."""
    if train_strategy == "epochs":
        return n_epochs * steps_per_epoch(dataset_len, num_processes, batch_size)
    if train_strategy == "steps":
        return n_steps
    raise ValueError(f"Unsupported train_strategy: {train_strategy!r}")
