"""A mid-epoch halt must save the interrupted counters and resume there.

Run with:
    uv run pytest tests/pipeline/test_halt_keeps_epoch_progress.py
"""

from __future__ import annotations

import logging
import os
import sys
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from mlkit.pipeline.callback import BaseCallback, CallbackEvent  # noqa: E402
from mlkit.pipeline.distributed_pipeline import (  # noqa: E402
    DistributedPipeline,
    DistributedTrainingConfig,
)
from mlkit.pipeline.pipeline import LogConfig, TrainingState, latest_checkpoint_dir  # noqa: E402
from mlkit.util.report import clear_deferred_error, defer_error, use_reporter  # noqa: E402


class CountingDataset(torch.utils.data.Dataset):
    def __init__(self, n: int) -> None:
        self.n = n
        self.access_log: list[int] = []

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, index: int):
        self.access_log.append(index)
        return {"x": torch.tensor([float(index)])}


class TinyModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


class TinyDistributedPipeline(DistributedPipeline):
    def compute_loss(self, model, batch):
        return model(batch["x"]).sum()


class DeferOnFirstStep(BaseCallback):
    """Remember a failure on the first step, which is not a logging step."""

    def on_event(self, event: CallbackEvent, *args, **kwargs) -> None:
        del args
        if event != CallbackEvent.STEP_END:
            return
        state: TrainingState = kwargs["training_state"]
        if state.current_global_step == 0:
            defer_error(RuntimeError("step log failed"))


class FailingReporter:
    def report(self, data: Mapping[str, Any], *, step: int | None = None) -> None:
        del data, step
        raise RuntimeError("epoch log failed")


def _make_accelerator() -> MagicMock:
    accelerator = MagicMock()
    accelerator.is_main_process = True
    accelerator.is_local_main_process = True
    accelerator.num_processes = 1
    accelerator.device = torch.device("cpu")
    accelerator.prepare.side_effect = lambda *args: args
    accelerator.save_state = MagicMock()
    accelerator.load_state = MagicMock()
    return accelerator


def _bind_accelerator(monkeypatch: pytest.MonkeyPatch, accelerator: MagicMock) -> None:
    monkeypatch.setattr(
        "mlkit.pipeline.distributed_pipeline.Accelerator",
        lambda *args, **kwargs: accelerator,
    )
    monkeypatch.setattr(
        "mlkit.pipeline.distributed_pipeline.accelerate.utils.set_seed",
        lambda *args, **kwargs: None,
    )


def _pipeline(
    dataset: CountingDataset,
    *,
    save_dir: str,
    log_config: LogConfig,
    callbacks: list[BaseCallback] | None = None,
) -> TinyDistributedPipeline:
    model = TinyModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda step: 1.0)
    config = DistributedTrainingConfig(
        n_epochs=1,
        batch_size=1,
        device="cpu",
        save_strategy="epochs",
        save_dir=save_dir,
        save_epochs=1,
        num_workers=0,
        train_shuffle=False,
        mixed_precision="no",
        grad_clip_strategy=None,
        eval_strategy=None,
    )
    return TinyDistributedPipeline(
        model=model,
        dataset=dataset,
        optimizers=(optimizer, scheduler),
        training_config=config,
        log_config=log_config,
        logger=logging.getLogger("halt-epoch-progress"),
        callbacks=callbacks,
    )


def _saved_state(save_dir: str) -> TrainingState:
    checkpoint_dir = latest_checkpoint_dir(save_dir)
    assert checkpoint_dir is not None
    state = torch.load(os.path.join(checkpoint_dir, "training_state.pth"), weights_only=False)
    assert isinstance(state, TrainingState)
    return state


def test_mid_epoch_halt_resumes_the_interrupted_epoch(monkeypatch: pytest.MonkeyPatch):
    accelerator = _make_accelerator()
    _bind_accelerator(monkeypatch, accelerator)
    dataset = CountingDataset(4)
    log_config = LogConfig(log_dir="logs", log_steps=2, log_strategy="steps")
    clear_deferred_error()
    try:
        with tempfile.TemporaryDirectory() as save_dir:
            pipeline = _pipeline(
                dataset,
                save_dir=save_dir,
                log_config=log_config,
                callbacks=[DeferOnFirstStep()],
            )
            pipeline.train()
            state = _saved_state(save_dir)
            assert state.current_epoch == 0
            assert state.current_step_in_epoch == 2
            assert state.current_global_step == 2

            dataset.access_log.clear()
            clear_deferred_error()
            resumed = _pipeline(dataset, save_dir=save_dir, log_config=log_config)
            resumed.train()
            assert dataset.access_log == [2, 3]
    finally:
        clear_deferred_error()


def test_halt_after_a_finished_epoch_saves_the_next_epoch(monkeypatch: pytest.MonkeyPatch):
    accelerator = _make_accelerator()
    _bind_accelerator(monkeypatch, accelerator)
    dataset = CountingDataset(4)
    log_config = LogConfig(log_dir="logs", log_epochs=1, log_strategy="epochs")
    clear_deferred_error()
    try:
        with tempfile.TemporaryDirectory() as save_dir:
            pipeline = _pipeline(dataset, save_dir=save_dir, log_config=log_config)
            with use_reporter(FailingReporter()):
                pipeline.train()
            state = _saved_state(save_dir)
            assert state.current_epoch == 1
            assert state.current_step_in_epoch == 0
            assert state.current_global_step == 4
            assert dataset.access_log == [0, 1, 2, 3]
    finally:
        clear_deferred_error()
