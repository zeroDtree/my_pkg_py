"""Regression test: DistributedPipeline.save must call save_state on all ranks.

FSDP/DeepSpeed use collectives inside accelerator.save_state, so non-main ranks
must not early-return before that call.

Run with:
    uv run pytest tests/pipeline/test_distributed_save_all_ranks.py
"""

from __future__ import annotations

import logging
import os
import sys
import tempfile
from pathlib import Path
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


class _EventLog(BaseCallback):
    def __init__(self) -> None:
        self.events: list[CallbackEvent] = []

    def on_event(self, event: CallbackEvent, *args, **kwargs) -> None:
        del args, kwargs
        self.events.append(event)


def _install_trigger(accelerator: MagicMock) -> None:
    """set_trigger makes the next check_trigger true, then the flag clears."""
    flag = {"set": False}

    def set_trigger() -> None:
        flag["set"] = True

    def check_trigger() -> bool:
        value = flag["set"]
        flag["set"] = False
        return value

    accelerator.set_trigger.side_effect = set_trigger
    accelerator.check_trigger.side_effect = check_trigger


def _make_stub_accelerator(*, is_main_process: bool) -> MagicMock:
    accelerator = MagicMock()
    accelerator.is_main_process = is_main_process
    accelerator.is_local_main_process = is_main_process
    accelerator.num_processes = 2
    accelerator.device = torch.device("cpu")
    accelerator.prepare.side_effect = lambda *args: args
    accelerator.wait_for_everyone = MagicMock()
    accelerator.save_state = MagicMock()
    _install_trigger(accelerator)
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


def _training_pipeline(
    accelerator: MagicMock,
    *,
    save_dir: str | None,
    logger: logging.Logger | None,
    log_config: LogConfig | None = None,
) -> TinyDistributedPipeline:
    model = TinyModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda step: 1.0)
    config = DistributedTrainingConfig(
        n_epochs=1,
        batch_size=2,
        device="cpu",
        save_strategy="steps",
        save_dir=save_dir,
        save_steps=1,
        num_workers=0,
        train_shuffle=False,
        mixed_precision="no",
        grad_clip_strategy=None,
    )
    pipeline = TinyDistributedPipeline(
        model=model,
        dataset=TinyDataset(),
        optimizers=(optimizer, scheduler),
        training_config=config,
        log_config=log_config or LogConfig(log_dir=""),
        logger=logger,
    )
    pipeline.accelerator = accelerator
    pipeline.training_state = TrainingState(current_epoch=0, current_step_in_epoch=1, current_global_step=1)
    return pipeline


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


def test_halted_save_failure_skips_post_save(monkeypatch: pytest.MonkeyPatch):
    accelerator = _make_stub_accelerator(is_main_process=True)
    _bind_accelerator(monkeypatch, accelerator)

    def fail_persist(directory: str) -> None:
        del directory
        raise OSError("disk full")

    monkeypatch.setattr("mlkit.pipeline.pipeline.persist_report_run", fail_persist)
    with tempfile.TemporaryDirectory() as tmp:
        pipeline = _training_pipeline(
            accelerator,
            save_dir=tmp,
            logger=logging.getLogger("mlkit.test.save_fail_post"),
        )
        events = _EventLog()
        pipeline.add_callbacks([events])
        pipeline._halt = True
        try:
            pipeline.save()
            assert CallbackEvent.POST_SAVE not in events.events
            assert pipeline._halt is True
            assert has_deferred_error()
            assert list(Path(tmp).glob("checkpoint_*")) == []
            assert list(Path(tmp).glob("tmp_*")) == []
        finally:
            clear_deferred_error()


def test_halted_successful_save_still_runs_post_save(monkeypatch: pytest.MonkeyPatch):
    accelerator = _make_stub_accelerator(is_main_process=True)
    _bind_accelerator(monkeypatch, accelerator)
    with tempfile.TemporaryDirectory() as tmp:
        pipeline = _training_pipeline(
            accelerator,
            save_dir=tmp,
            logger=logging.getLogger("mlkit.test.save_ok_post"),
        )
        events = _EventLog()
        pipeline.add_callbacks([events])
        pipeline._halt = True
        pipeline.save()
        assert CallbackEvent.POST_SAVE in events.events
        assert pipeline._halt is True
        assert not has_deferred_error()
        assert list(Path(tmp).glob("checkpoint_*")) != []


def test_save_state_error_is_deferred_and_temp_dir_removed(monkeypatch: pytest.MonkeyPatch):
    accelerator = _make_stub_accelerator(is_main_process=True)
    accelerator.save_state.side_effect = OSError("disk full")
    _bind_accelerator(monkeypatch, accelerator)
    with tempfile.TemporaryDirectory() as tmp:
        pipeline = _training_pipeline(
            accelerator,
            save_dir=tmp,
            logger=logging.getLogger("mlkit.test.save_state_fail"),
        )
        events = _EventLog()
        pipeline.add_callbacks([events])
        try:
            pipeline.save()
            accelerator.save_state.assert_called_once()
            assert CallbackEvent.POST_SAVE not in events.events
            assert pipeline._halt is True
            assert has_deferred_error()
            with pytest.raises(OSError, match="disk full"):
                raise_deferred_error()
            assert list(Path(tmp).glob("tmp_*")) == []
            assert list(Path(tmp).glob("checkpoint_*")) == []
        finally:
            clear_deferred_error()


def test_non_main_save_state_error_stays_in_save(monkeypatch: pytest.MonkeyPatch):
    accelerator = _make_stub_accelerator(is_main_process=False)
    accelerator.save_state.side_effect = OSError("disk full")
    _bind_accelerator(monkeypatch, accelerator)
    with tempfile.TemporaryDirectory() as tmp:
        pipeline = _training_pipeline(accelerator, save_dir=tmp, logger=None)
        events = _EventLog()
        pipeline.add_callbacks([events])
        try:
            pipeline.save()
            accelerator.set_trigger.assert_called()
            assert CallbackEvent.POST_SAVE not in events.events
            assert pipeline._halt is True
            assert has_deferred_error()
        finally:
            clear_deferred_error()


def test_non_log_step_does_not_sync_halt(monkeypatch: pytest.MonkeyPatch):
    accelerator = _make_stub_accelerator(is_main_process=True)
    _bind_accelerator(monkeypatch, accelerator)
    pipeline = _training_pipeline(
        accelerator,
        save_dir=None,
        logger=logging.getLogger("mlkit.test.halt_quiet"),
        log_config=LogConfig(log_dir="logs", log_steps=100, log_strategy="steps"),
    )
    pipeline.training_state = TrainingState()
    accelerator.check_trigger.reset_mock()
    accelerator.set_trigger.reset_mock()
    pipeline.train_a_step({"x": torch.tensor([[1.0]])})
    accelerator.check_trigger.assert_not_called()
    accelerator.set_trigger.assert_not_called()


def test_log_step_syncs_halt(monkeypatch: pytest.MonkeyPatch):
    accelerator = _make_stub_accelerator(is_main_process=True)
    _bind_accelerator(monkeypatch, accelerator)
    pipeline = _training_pipeline(
        accelerator,
        save_dir=None,
        logger=logging.getLogger("mlkit.test.halt_log"),
        log_config=LogConfig(log_dir="logs", log_steps=1, log_strategy="steps"),
    )
    pipeline.training_state = TrainingState()
    accelerator.check_trigger.reset_mock()
    pipeline.train_a_step({"x": torch.tensor([[1.0]])})
    accelerator.check_trigger.assert_called()


def test_peer_save_state_failure_does_not_publish(monkeypatch: pytest.MonkeyPatch):
    accelerator = _make_stub_accelerator(is_main_process=True)
    _bind_accelerator(monkeypatch, accelerator)
    renamed: list[tuple[str, str]] = []
    real_rename = os.rename

    def rename(src: str, dst: str) -> None:
        renamed.append((src, dst))
        real_rename(src, dst)

    monkeypatch.setattr("mlkit.pipeline.distributed_pipeline.os.rename", rename)
    with tempfile.TemporaryDirectory() as tmp:
        pipeline = _training_pipeline(
            accelerator,
            save_dir=tmp,
            logger=logging.getLogger("mlkit.test.peer_save_state"),
        )
        events = _EventLog()
        pipeline.add_callbacks([events])
        calls = {"n": 0}

        def share(local: bool) -> bool:
            del local
            calls["n"] += 1
            # exists, prepare, then the save_state result from another rank.
            return calls["n"] == 3

        pipeline._share_flag = share  # type: ignore[method-assign]
        try:
            pipeline.save()
            accelerator.save_state.assert_called_once()
            assert renamed == []
            assert CallbackEvent.POST_SAVE not in events.events
            assert pipeline._halt is True
            assert list(Path(tmp).glob("checkpoint_*")) == []
            assert list(Path(tmp).glob("tmp_*")) == []
        finally:
            clear_deferred_error()


def test_makedirs_failure_skips_save_state(monkeypatch: pytest.MonkeyPatch):
    accelerator = _make_stub_accelerator(is_main_process=True)
    _bind_accelerator(monkeypatch, accelerator)

    def fail_makedirs(path: str, mode: int = 0o777, exist_ok: bool = False) -> None:
        del path, mode, exist_ok
        raise OSError("permission denied")

    monkeypatch.setattr("mlkit.pipeline.distributed_pipeline.os.makedirs", fail_makedirs)
    with tempfile.TemporaryDirectory() as tmp:
        pipeline = _training_pipeline(
            accelerator,
            save_dir=tmp,
            logger=logging.getLogger("mlkit.test.makedirs_fail"),
        )
        try:
            pipeline.save()
            accelerator.save_state.assert_not_called()
            assert pipeline._halt is True
            assert has_deferred_error()
            with pytest.raises(OSError, match="permission denied"):
                raise_deferred_error()
        finally:
            clear_deferred_error()


def test_non_main_skips_save_when_main_sees_final_checkpoint(monkeypatch: pytest.MonkeyPatch):
    accelerator = _make_stub_accelerator(is_main_process=False)
    _bind_accelerator(monkeypatch, accelerator)
    with tempfile.TemporaryDirectory() as tmp:
        pipeline = _training_pipeline(accelerator, save_dir=tmp, logger=None)
        pipeline._share_flag = lambda local: True  # type: ignore[method-assign]
        pipeline.save()
        accelerator.save_state.assert_not_called()
        assert pipeline._halt is False


def test_cleanup_failure_keeps_committed_checkpoint(monkeypatch: pytest.MonkeyPatch):
    accelerator = _make_stub_accelerator(is_main_process=True)
    _bind_accelerator(monkeypatch, accelerator)
    with tempfile.TemporaryDirectory() as tmp:
        pipeline = _training_pipeline(
            accelerator,
            save_dir=tmp,
            logger=logging.getLogger("mlkit.test.cleanup_fail"),
        )
        events = _EventLog()
        pipeline.add_callbacks([events])

        def fail_cleanup(save_dir: str) -> None:
            del save_dir
            raise OSError("busy")

        monkeypatch.setattr(pipeline, "_cleanup_old_checkpoints", fail_cleanup)
        pipeline.save()
        assert CallbackEvent.POST_SAVE in events.events
        assert not has_deferred_error()
        assert pipeline._halt is False
        assert list(Path(tmp).glob("checkpoint_*")) != []
