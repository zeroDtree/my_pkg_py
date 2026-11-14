"""run_experiment destroys the process group before post-training work.

Run with:
    uv run pytest tests/pipeline/test_run_experiment_shutdown.py
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from contextlib import contextmanager
from typing import Any
from unittest.mock import MagicMock

import pytest
import torch
from omegaconf import DictConfig, OmegaConf

from example.shared.runner import run_experiment
from mlkit.util.report import NullReporter, defer_error, has_deferred_error


def _cfg() -> DictConfig:
    return OmegaConf.create(
        {
            "train": {
                "seed": 0,
                "mixed_precision": "no",
                "train_strategy": "steps",
                "n_epochs": 1,
                "n_steps": 1,
                "batch_size": 2,
                "device": "cpu",
                "save_strategy": None,
                "save_dir": "checkpoints",
                "save_steps": 10,
                "save_epochs": 1,
                "save_total_limit": 1,
                "num_workers": 0,
                "train_shuffle": False,
                "eval_strategy": None,
                "eval_steps": 1,
                "eval_epochs": 1,
                "grad_clip_strategy": None,
                "max_grad_norm": 1.0,
                "max_grad_value": 1.0,
                "gradient_accumulation_steps": 1,
                "n_warmup_steps": 0,
                "lr_scheduler_type": "constant",
            },
            "log": {
                "log_dir": "logs",
                "log_steps": 1,
                "log_epochs": 1,
                "log_strategy": "steps",
            },
            "reporter": "none",
            "mlflow": {},
            "wandb": {},
            "dataset": {"name": "toy"},
            "model": {"name": "toy"},
            "optimizer": {"name": "SGD", "lr": 0.01},
        }
    )


def _install(
    monkeypatch: pytest.MonkeyPatch,
    events: list[str],
    *,
    is_main: bool,
    num_processes: int = 1,
    train_error: BaseException | None = None,
    on_train: Callable[[], None] | None = None,
    configs: list[Any] | None = None,
    halt: bool = False,
) -> MagicMock:
    accelerator = MagicMock()
    accelerator.is_local_main_process = is_main
    accelerator.is_main_process = is_main
    accelerator.device = torch.device("cpu")
    accelerator.num_processes = num_processes
    accelerator.check_trigger.return_value = False
    accelerator.wait_for_everyone.side_effect = lambda: events.append("wait")
    accelerator.end_training.side_effect = lambda: events.append("end")
    monkeypatch.setattr("accelerate.Accelerator", lambda *args, **kwargs: accelerator)

    class Pipeline:
        def __init__(self, **kwargs: Any) -> None:
            del kwargs
            self._halt = False

        def train(self) -> None:
            events.append("train")
            if on_train is not None:
                on_train()
            if train_error is not None:
                raise train_error
            # The real pipeline sets this after sharing a deferred failure.
            if halt or has_deferred_error():
                self._halt = True

    class TrainingConfig:
        def __init__(self, **kwargs: Any) -> None:
            self.__dict__.update(kwargs)
            if configs is not None:
                configs.append(self)

    monkeypatch.setattr("mlkit.util.utils_for_main.get_train_class", lambda: (Pipeline, TrainingConfig))
    monkeypatch.setattr("mlkit.util.utils_for_main.get_learing_rate_scheduler", lambda *args, **kwargs: MagicMock())
    monkeypatch.setattr(
        "mlkit.util.log.get_logger",
        lambda *args, **kwargs: logging.getLogger("mlkit.test.run_experiment_shutdown"),
    )
    monkeypatch.setattr("mlkit.util.log.get_and_create_new_log_dir", lambda *args, **kwargs: "logs")
    return accelerator


def _run(events: list[str], *, post_train_fn, after_train_fn=None):
    model = torch.nn.Linear(1, 1)

    def get_model(cfg: DictConfig):
        del cfg
        return model

    def get_dataset(cfg: DictConfig):
        del cfg
        return [0, 1], None, None

    def get_collate(cfg: DictConfig):
        del cfg
        return None

    run_experiment(
        _cfg(),
        get_model,
        get_dataset,
        get_collate,
        post_train_fn=post_train_fn,
        after_train_fn=after_train_fn,
    )


def test_every_rank_uses_the_same_checkpoint_dir(monkeypatch: pytest.MonkeyPatch):
    expected = "checkpoints/toy/toy/-lr:0.01-bs:2"

    def post_train(*args, **kwargs) -> None:
        del args, kwargs

    for is_main in (True, False):
        events: list[str] = []
        configs: list[Any] = []
        _install(monkeypatch, events, is_main=is_main, configs=configs)
        _run(events, post_train_fn=post_train)
        assert [config.save_dir for config in configs] == [expected]


def test_main_process_destroys_group_before_post_train(monkeypatch: pytest.MonkeyPatch):
    events: list[str] = []
    accelerator = _install(monkeypatch, events, is_main=True)

    def post_train(cfg, model_result, pipeline, received_accelerator, train_set) -> None:
        del cfg, model_result, pipeline, train_set
        assert received_accelerator is accelerator
        events.append("post")

    _run(events, post_train_fn=post_train)
    assert events == ["train", "wait", "end", "post"]


def test_non_main_process_destroys_group_without_post_train(monkeypatch: pytest.MonkeyPatch):
    events: list[str] = []
    _install(monkeypatch, events, is_main=False)

    def post_train(*args, **kwargs) -> None:
        del args, kwargs
        events.append("post")

    _run(events, post_train_fn=post_train)
    assert events == ["train", "wait", "end"]


def test_train_error_destroys_group_without_barrier(monkeypatch: pytest.MonkeyPatch):
    events: list[str] = []
    accelerator = _install(monkeypatch, events, is_main=True, train_error=RuntimeError("boom"))

    def post_train(*args, **kwargs) -> None:
        del args, kwargs
        events.append("post")

    with pytest.raises(RuntimeError, match="boom"):
        _run(events, post_train_fn=post_train)

    assert events == ["train", "end"]
    accelerator.wait_for_everyone.assert_not_called()


def test_end_training_error_keeps_the_train_error(monkeypatch: pytest.MonkeyPatch):
    events: list[str] = []
    accelerator = _install(monkeypatch, events, is_main=True, train_error=RuntimeError("boom"))

    def end_training() -> None:
        events.append("end")
        raise RuntimeError("end failed")

    accelerator.end_training.side_effect = end_training

    def post_train(*args, **kwargs) -> None:
        del args, kwargs
        events.append("post")

    with pytest.raises(RuntimeError, match="boom"):
        _run(events, post_train_fn=post_train)

    assert events == ["train", "end"]
    accelerator.wait_for_everyone.assert_not_called()


def test_post_train_error_stays_inside_reporter(monkeypatch: pytest.MonkeyPatch):
    events: list[str] = []
    seen: list[BaseException] = []
    _install(monkeypatch, events, is_main=True)

    @contextmanager
    def fake_open(*args, **kwargs):
        del args, kwargs
        try:
            yield NullReporter()
        except BaseException as exc:
            seen.append(exc)
            raise

    monkeypatch.setattr("mlkit.util.report.open_reporter", fake_open)

    def post_train(*args, **kwargs) -> None:
        del args, kwargs
        events.append("post")
        raise RuntimeError("post failed")

    with pytest.raises(RuntimeError, match="post failed"):
        _run(events, post_train_fn=post_train)

    assert events == ["train", "wait", "end", "post"]
    assert [type(exc) for exc in seen] == [RuntimeError]


def test_deferred_error_is_raised_after_teardown(monkeypatch: pytest.MonkeyPatch):
    events: list[str] = []
    _install(
        monkeypatch,
        events,
        is_main=True,
        on_train=lambda: defer_error(RuntimeError("log failed")),
    )

    def post_train(*args, **kwargs) -> None:
        del args, kwargs
        events.append("post")

    with pytest.raises(RuntimeError, match="log failed"):
        _run(events, post_train_fn=post_train)

    assert events == ["train", "wait", "end"]


def test_deferred_error_skips_after_train_on_main(monkeypatch: pytest.MonkeyPatch):
    events: list[str] = []
    _install(
        monkeypatch,
        events,
        is_main=True,
        on_train=lambda: defer_error(RuntimeError("log failed")),
    )

    def after_train(*args, **kwargs) -> None:
        del args, kwargs
        events.append("after")

    def post_train(*args, **kwargs) -> None:
        del args, kwargs
        events.append("post")

    with pytest.raises(RuntimeError, match="log failed"):
        _run(events, post_train_fn=post_train, after_train_fn=after_train)

    assert events == ["train", "wait", "end"]


def test_shared_halt_skips_after_train_on_non_main(monkeypatch: pytest.MonkeyPatch):
    events: list[str] = []
    accelerator = _install(monkeypatch, events, is_main=False, halt=True)

    def after_train(*args, **kwargs) -> None:
        del args, kwargs
        events.append("after")

    def post_train(*args, **kwargs) -> None:
        del args, kwargs
        events.append("post")

    _run(events, post_train_fn=post_train, after_train_fn=after_train)

    assert events == ["train", "wait", "end"]
    accelerator.set_trigger.assert_not_called()


def test_after_train_runs_on_main_before_teardown(monkeypatch: pytest.MonkeyPatch):
    events: list[str] = []
    _install(monkeypatch, events, is_main=True)

    def after_train(*args, **kwargs) -> None:
        del args, kwargs
        events.append("after")

    def post_train(*args, **kwargs) -> None:
        del args, kwargs
        events.append("post")

    _run(events, post_train_fn=post_train, after_train_fn=after_train)
    assert events == ["train", "after", "wait", "end", "post"]


def test_after_train_runs_on_non_main_without_post_train(monkeypatch: pytest.MonkeyPatch):
    events: list[str] = []
    _install(monkeypatch, events, is_main=False)

    def after_train(*args, **kwargs) -> None:
        del args, kwargs
        events.append("after")

    def post_train(*args, **kwargs) -> None:
        del args, kwargs
        events.append("post")

    _run(events, post_train_fn=post_train, after_train_fn=after_train)
    assert events == ["train", "after", "wait", "end"]


def test_after_train_error_destroys_group_without_barrier(monkeypatch: pytest.MonkeyPatch):
    events: list[str] = []
    accelerator = _install(monkeypatch, events, is_main=True)

    def after_train(*args, **kwargs) -> None:
        del args, kwargs
        events.append("after")
        raise RuntimeError("after failed")

    def post_train(*args, **kwargs) -> None:
        del args, kwargs
        events.append("post")

    with pytest.raises(RuntimeError, match="after failed"):
        _run(events, post_train_fn=post_train, after_train_fn=after_train)

    assert events == ["train", "after", "end"]
    accelerator.wait_for_everyone.assert_not_called()


def test_log_dir_error_destroys_group_without_training(monkeypatch: pytest.MonkeyPatch):
    events: list[str] = []
    accelerator = _install(monkeypatch, events, is_main=True)

    def create_log_dir(*args, **kwargs):
        del args, kwargs
        raise OSError("disk full")

    monkeypatch.setattr("mlkit.util.log.get_and_create_new_log_dir", create_log_dir)

    def post_train(*args, **kwargs) -> None:
        del args, kwargs
        events.append("post")

    with pytest.raises(OSError, match="disk full"):
        _run(events, post_train_fn=post_train)

    assert events == ["end"]
    accelerator.wait_for_everyone.assert_not_called()


def test_reporter_open_error_destroys_group_without_training(monkeypatch: pytest.MonkeyPatch):
    events: list[str] = []
    accelerator = _install(monkeypatch, events, is_main=True)

    def open_reporter(*args, **kwargs):
        del args, kwargs
        raise RuntimeError("mlflow down")

    monkeypatch.setattr("mlkit.util.report.open_reporter", open_reporter)

    def post_train(*args, **kwargs) -> None:
        del args, kwargs
        events.append("post")

    with pytest.raises(RuntimeError, match="mlflow down"):
        _run(events, post_train_fn=post_train)

    assert events == ["end"]
    accelerator.wait_for_everyone.assert_not_called()
    accelerator.set_trigger.assert_not_called()


def test_end_training_error_keeps_the_reporter_open_error(monkeypatch: pytest.MonkeyPatch):
    events: list[str] = []
    accelerator = _install(monkeypatch, events, is_main=True)

    def end_training() -> None:
        events.append("end")
        raise RuntimeError("end failed")

    accelerator.end_training.side_effect = end_training

    def open_reporter(*args, **kwargs):
        del args, kwargs
        raise RuntimeError("mlflow down")

    monkeypatch.setattr("mlkit.util.report.open_reporter", open_reporter)

    def post_train(*args, **kwargs) -> None:
        del args, kwargs
        events.append("post")

    with pytest.raises(RuntimeError, match="mlflow down"):
        _run(events, post_train_fn=post_train)

    assert events == ["end"]
    accelerator.wait_for_everyone.assert_not_called()


def test_reporter_open_is_shared_before_training(monkeypatch: pytest.MonkeyPatch):
    events: list[str] = []
    accelerator = _install(monkeypatch, events, is_main=True, num_processes=2)
    accelerator.check_trigger.side_effect = lambda: events.append("sync") or False

    def post_train(*args, **kwargs) -> None:
        del args, kwargs
        events.append("post")

    _run(events, post_train_fn=post_train)
    assert events == ["sync", "train", "wait", "end", "post"]
    accelerator.set_trigger.assert_not_called()


def test_non_main_waits_for_reporter_open_before_training(monkeypatch: pytest.MonkeyPatch):
    events: list[str] = []
    accelerator = _install(monkeypatch, events, is_main=False, num_processes=2)
    accelerator.check_trigger.side_effect = lambda: events.append("sync") or False

    def post_train(*args, **kwargs) -> None:
        del args, kwargs
        events.append("post")

    _run(events, post_train_fn=post_train)
    assert events == ["sync", "train", "wait", "end"]
    accelerator.set_trigger.assert_not_called()


def test_reporter_open_error_sets_the_trigger(monkeypatch: pytest.MonkeyPatch):
    events: list[str] = []
    accelerator = _install(monkeypatch, events, is_main=True, num_processes=2)
    accelerator.set_trigger.side_effect = lambda: events.append("trigger")
    accelerator.check_trigger.side_effect = lambda: events.append("sync") or True

    def open_reporter(*args, **kwargs):
        del args, kwargs
        raise RuntimeError("mlflow down")

    monkeypatch.setattr("mlkit.util.report.open_reporter", open_reporter)

    def post_train(*args, **kwargs) -> None:
        del args, kwargs
        events.append("post")

    with pytest.raises(RuntimeError, match="mlflow down"):
        _run(events, post_train_fn=post_train)

    assert events == ["trigger", "sync", "end"]
    accelerator.wait_for_everyone.assert_not_called()


def test_other_rank_stops_when_reporter_open_fails(monkeypatch: pytest.MonkeyPatch):
    events: list[str] = []
    accelerator = _install(monkeypatch, events, is_main=False, num_processes=2)
    accelerator.check_trigger.side_effect = lambda: events.append("sync") or True

    def open_reporter(*args, **kwargs):
        del args, kwargs
        events.append("open")
        raise AssertionError("non-main rank opened the reporter")

    monkeypatch.setattr("mlkit.util.report.open_reporter", open_reporter)

    def post_train(*args, **kwargs) -> None:
        del args, kwargs
        events.append("post")

    with pytest.raises(RuntimeError, match="Reporter failed to open on another rank"):
        _run(events, post_train_fn=post_train)

    assert events == ["sync", "end"]
    accelerator.set_trigger.assert_not_called()
    accelerator.wait_for_everyone.assert_not_called()


def test_local_main_that_is_not_global_main_skips_the_reporter(monkeypatch: pytest.MonkeyPatch):
    events: list[str] = []
    accelerator = _install(monkeypatch, events, is_main=False, num_processes=2)
    accelerator.is_local_main_process = True
    accelerator.is_main_process = False
    accelerator.check_trigger.side_effect = lambda: events.append("sync") or False

    def open_reporter(*args, **kwargs):
        del args, kwargs
        events.append("open")
        raise AssertionError("local main opened the reporter")

    monkeypatch.setattr("mlkit.util.report.open_reporter", open_reporter)

    def post_train(*args, **kwargs) -> None:
        del args, kwargs
        events.append("post")

    _run(events, post_train_fn=post_train)
    assert events == ["sync", "train", "wait", "end"]
    accelerator.set_trigger.assert_not_called()
