"""Reporter routing, the optional text logger, and the MLflow and wandb adapters."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any, cast

import pytest

from mlkit.util.report import (
    MLflowReporter,
    NullReporter,
    WandbReporter,
    open_reporter,
    report,
    use_reporter,
)


class RecordingReporter:
    def __init__(self):
        self.calls: list[tuple[dict[str, Any], int | None]] = []

    def report(self, data: Mapping[str, Any], *, step: int | None = None) -> None:
        self.calls.append((dict(data), step))


def test_report_with_logger_writes_backend_and_text(caplog):
    recorder = RecordingReporter()
    logger = logging.getLogger("mlkit.test.reporter.echo")
    logger.setLevel(logging.INFO)
    with use_reporter(recorder):
        with caplog.at_level(logging.INFO, logger=logger.name):
            report({"loss": 1.5, "note": "x"}, step=3, logger=logger)

    assert recorder.calls == [({"loss": 1.5, "note": "x"}, 3)]
    messages = [record.message for record in caplog.records if record.name == logger.name]
    assert messages == ["step=3 loss=1.5 note=x"]


def test_report_without_logger_writes_backend_only(caplog):
    recorder = RecordingReporter()
    with use_reporter(recorder):
        with caplog.at_level(logging.INFO):
            report({"loss": 1.0}, step=1)

    assert recorder.calls == [({"loss": 1.0}, 1)]
    assert caplog.records == []


def test_use_reporter_resets_after_exception():
    outer = RecordingReporter()
    inner = RecordingReporter()
    with use_reporter(outer):
        with pytest.raises(RuntimeError, match="boom"):
            with use_reporter(inner):
                raise RuntimeError("boom")
        report({"loss": 1.0}, step=0)
    report({"loss": 2.0}, step=1)

    assert inner.calls == []
    assert outer.calls == [({"loss": 1.0}, 0)]


def test_null_reporter_still_writes_text(caplog):
    logger = logging.getLogger("mlkit.test.reporter.null")
    logger.setLevel(logging.INFO)
    with use_reporter(NullReporter()):
        with caplog.at_level(logging.INFO, logger=logger.name):
            report({"loss": 0.25}, step=2, logger=logger)

    messages = [record.message for record in caplog.records if record.name == logger.name]
    assert messages == ["step=2 loss=0.25"]


def test_open_reporter_rejects_unknown_kind():
    with pytest.raises(ValueError, match="Unknown reporter"):
        with open_reporter("nope", run_name="run", config={}):
            pass


class _FakeMlflow:
    def __init__(self) -> None:
        self.calls: list[tuple[str, object]] = []

    def set_tracking_uri(self, uri: object) -> None:
        self.calls.append(("uri", uri))

    def set_experiment(self, name: object) -> None:
        self.calls.append(("experiment", name))

    def start_run(self, run_name: object = None) -> None:
        self.calls.append(("start", run_name))

    def log_dict(self, data: object, artifact_file: object) -> None:
        self.calls.append(("dict", artifact_file))

    def end_run(self, status: object = None) -> None:
        self.calls.append(("end", status))


def test_mlflow_reporter_defaults_to_sqlite_without_env(monkeypatch):
    import sys

    fake = _FakeMlflow()
    monkeypatch.setitem(sys.modules, "mlflow", cast(Any, fake))
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)

    with MLflowReporter(tracking_uri=None, experiment="local", run_name="run", config={"a": 1}):
        pass

    assert ("uri", "sqlite:///mlflow.db") in fake.calls
    assert ("experiment", "local") in fake.calls
    assert ("end", "FINISHED") in fake.calls


def test_mlflow_reporter_keeps_tracking_uri_env(monkeypatch):
    import sys

    fake = _FakeMlflow()
    monkeypatch.setitem(sys.modules, "mlflow", cast(Any, fake))
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")

    with MLflowReporter(tracking_uri=None, experiment="remote", run_name="run", config={}):
        pass

    assert not any(name == "uri" for name, _ in fake.calls)
    assert ("experiment", "remote") in fake.calls


def test_mlflow_reporter_sqlite_store(tmp_path):
    import mlflow
    from mlflow.tracking import MlflowClient

    tracking_uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    with (
        MLflowReporter(
            tracking_uri=tracking_uri,
            experiment="unit",
            run_name="run-a",
            config={"lr": 0.1, "nested": {"a": 1}},
        ) as mlflow_reporter,
        use_reporter(mlflow_reporter),
    ):
        report({"loss": 1.0, "bad": float("nan"), "name": "skip"}, step=0)
        report({"loss": 0.5}, step=1)

    mlflow.set_tracking_uri(tracking_uri)
    experiment = mlflow.get_experiment_by_name("unit")
    assert experiment is not None
    frame = mlflow.search_runs(experiment_ids=[experiment.experiment_id])
    assert len(frame) == 1
    assert frame.iloc[0]["metrics.loss"] == 0.5
    assert "metrics.bad" not in frame.columns
    assert "metrics.name" not in frame.columns
    assert frame.iloc[0]["tags.mlflow.runName"] == "run-a"

    client = MlflowClient(tracking_uri=tracking_uri)
    run_id = frame.iloc[0]["run_id"]
    history = client.get_metric_history(run_id, "loss")
    assert [(metric.step, metric.value) for metric in history] == [(0, 1.0), (1, 0.5)]
    artifacts = client.list_artifacts(run_id)
    assert any(artifact.path == "config.yaml" for artifact in artifacts)


class _FakeWandb:
    def __init__(self) -> None:
        self.events: list[tuple] = []

    def init(self, **kwargs: object) -> None:
        self.events.append(("init", kwargs))

    def log(self, data: Mapping[str, object], step: object = None) -> None:
        self.events.append(("log", dict(data), step))

    def finish(self) -> None:
        self.events.append(("finish",))


def test_wandb_reporter_calls_init_log_finish(monkeypatch):
    import sys

    fake_wandb = _FakeWandb()
    monkeypatch.setitem(sys.modules, "wandb", cast(Any, fake_wandb))

    with (
        WandbReporter(
            project="p",
            name="n",
            group="g",
            entity="e",
            mode="offline",
            reinit=True,
            config={"a": 1},
        ) as wandb_reporter,
        use_reporter(wandb_reporter),
    ):
        report({"loss": 0.2}, step=4)

    events = fake_wandb.events
    assert events[0][0] == "init"
    assert events[0][1]["project"] == "p"
    assert events[0][1]["name"] == "n"
    assert events[0][1]["group"] == "g"
    assert events[0][1]["entity"] == "e"
    assert events[0][1]["mode"] == "offline"
    assert events[0][1]["config"] == {"a": 1}
    assert events[1] == ("log", {"loss": 0.2}, 4)
    assert events[2] == ("finish",)
