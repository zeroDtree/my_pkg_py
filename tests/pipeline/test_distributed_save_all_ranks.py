"""Regression test: DistributedPipeline.save must call save_state on all ranks.

FSDP/DeepSpeed use collectives inside accelerator.save_state, so non-main ranks
must not early-return before that call.

Run with:
    uv run pytest tests/pipeline/test_distributed_save_all_ranks.py
"""

from __future__ import annotations

import logging
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from mlkit.pipeline.distributed_pipeline import (  # noqa: E402
    DistributedPipeline,
    DistributedTrainingConfig,
)
from mlkit.pipeline.pipeline import LogConfig, TrainingState  # noqa: E402
from mlkit.util.report import (  # noqa: E402
    clear_deferred_error,
    defer_error,
    has_deferred_error,
    raise_deferred_error,
)


class TinyDataset(torch.utils.data.Dataset):
    def __len__(self) -> int:
        return 4

    def __getitem__(self, index: int):
        return {"x": torch.tensor([float(index)])}


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(1, 1)

    def forward(self, x):
        return self.linear(x)


class TinyDistributedPipeline(DistributedPipeline):
    def compute_loss(self, model, batch):
        pred = model(batch["x"])
        return pred.sum()


def _make_stub_accelerator(*, is_main_process: bool) -> MagicMock:
    accelerator = MagicMock()
    accelerator.is_main_process = is_main_process
    accelerator.is_local_main_process = is_main_process
    accelerator.num_processes = 2
    accelerator.device = torch.device("cpu")
    accelerator.prepare.side_effect = lambda *args: args
    accelerator.wait_for_everyone = MagicMock()
    accelerator.save_state = MagicMock()
    accelerator.check_trigger.return_value = False
    return accelerator


def test_save_calls_save_state_on_non_main_process(monkeypatch):
    accelerator = _make_stub_accelerator(is_main_process=False)

    # Bypass real Accelerator construction in DistributedPipeline.__init__.
    monkeypatch.setattr(
        "mlkit.pipeline.distributed_pipeline.Accelerator",
        lambda *args, **kwargs: accelerator,
    )
    monkeypatch.setattr(
        "mlkit.pipeline.distributed_pipeline.accelerate.utils.set_seed",
        lambda *args, **kwargs: None,
    )

    with tempfile.TemporaryDirectory() as tmp:
        model = TinyModel()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda step: 1.0)
        config = DistributedTrainingConfig(
            n_epochs=1,
            batch_size=2,
            device="cpu",
            save_strategy="steps",
            save_dir=tmp,
            save_steps=1,
            num_workers=0,
            train_shuffle=False,
            mixed_precision="no",
        )
        pipeline = TinyDistributedPipeline(
            model=model,
            dataset=TinyDataset(),
            optimizers=(optimizer, scheduler),
            training_config=config,
            log_config=LogConfig(log_dir=""),
            logger=None,
        )
        # Re-attach the stub after __init__ (prepare may replace references).
        pipeline.accelerator = accelerator
        pipeline.training_state = TrainingState(current_epoch=0, current_step_in_epoch=1, current_global_step=1)

        pipeline.save()

        accelerator.save_state.assert_called_once()
        accelerator.wait_for_everyone.assert_called()
        # Non-main process must not rename into a final checkpoint directory.
        checkpoint_dirs = list(Path(tmp).glob("checkpoint_*"))
        assert checkpoint_dirs == []


def test_main_process_checkpoint_write_failure_is_deferred(monkeypatch):
    accelerator = _make_stub_accelerator(is_main_process=True)
    monkeypatch.setattr(
        "mlkit.pipeline.distributed_pipeline.Accelerator",
        lambda *args, **kwargs: accelerator,
    )
    monkeypatch.setattr(
        "mlkit.pipeline.distributed_pipeline.accelerate.utils.set_seed",
        lambda *args, **kwargs: None,
    )

    with tempfile.TemporaryDirectory() as tmp:
        model = TinyModel()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda step: 1.0)
        config = DistributedTrainingConfig(
            n_epochs=1,
            batch_size=2,
            device="cpu",
            save_strategy="steps",
            save_dir=tmp,
            save_steps=1,
            num_workers=0,
            train_shuffle=False,
            mixed_precision="no",
        )
        pipeline = TinyDistributedPipeline(
            model=model,
            dataset=TinyDataset(),
            optimizers=(optimizer, scheduler),
            training_config=config,
            log_config=LogConfig(log_dir=""),
            logger=logging.getLogger("mlkit.test.distributed_save"),
        )
        pipeline.accelerator = accelerator
        pipeline.training_state = TrainingState(current_epoch=0, current_step_in_epoch=1, current_global_step=1)

        def fail_persist(directory: str) -> None:
            del directory
            raise OSError("disk full")

        monkeypatch.setattr("mlkit.pipeline.pipeline.persist_report_run", fail_persist)
        try:
            pipeline.save()
            accelerator.save_state.assert_called_once()
            assert accelerator.wait_for_everyone.call_count == 2
            assert has_deferred_error()
            with pytest.raises(OSError, match="disk full"):
                raise_deferred_error()
        finally:
            clear_deferred_error()


def test_multi_process_halt_shares_the_trigger():
    pipeline = object.__new__(TinyDistributedPipeline)
    accelerator = MagicMock()
    accelerator.num_processes = 2
    accelerator.is_main_process = True
    accelerator.check_trigger.return_value = True
    pipeline.accelerator = accelerator
    pipeline._halt = False
    try:
        defer_error(RuntimeError("log failed"))
        assert pipeline._halt_if_any_rank_failed() is True
        accelerator.set_trigger.assert_called_once()
        accelerator.check_trigger.assert_called_once()
    finally:
        clear_deferred_error()


def test_non_main_halt_checks_trigger_without_setting_it():
    pipeline = object.__new__(TinyDistributedPipeline)
    accelerator = MagicMock()
    accelerator.num_processes = 2
    accelerator.is_main_process = False
    accelerator.check_trigger.return_value = True
    pipeline.accelerator = accelerator
    assert pipeline._halt_if_any_rank_failed() is True
    accelerator.set_trigger.assert_not_called()
    accelerator.check_trigger.assert_called_once()


def test_single_process_halt_reads_the_local_error():
    pipeline = object.__new__(TinyDistributedPipeline)
    accelerator = MagicMock()
    accelerator.num_processes = 1
    pipeline.accelerator = accelerator
    try:
        defer_error(RuntimeError("log failed"))
        assert pipeline._halt_if_any_rank_failed() is True
        accelerator.set_trigger.assert_not_called()
        accelerator.check_trigger.assert_not_called()
    finally:
        clear_deferred_error()
