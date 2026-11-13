"""Reporter routing, the optional text logger, and the MLflow and wandb adapters."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any

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


def test_mlflow_reporter_defaults_to_sqlite_without_env(monkeypatch):
    import sys
    import types

    calls: list[tuple[str, object]] = []
    fake = types.ModuleType("mlflow")
    fake.set_tracking_uri = lambda uri: calls.append(("uri", uri))
    fake.set_experiment = lambda name: calls.append(("experiment", name))
    fake.start_run = lambda run_name=None: calls.append(("start", run_name))
    fake.log_dict = lambda data, artifact_file: calls.append(("dict", artifact_file))
    fake.end_run = lambda status=None: calls.append(("end", status))
    monkeypatch.setitem(sys.modules, "mlflow", fake)
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)

    with MLflowReporter(tracking_uri=None, experiment="local", run_name="run", config={"a": 1}):
        pass

    assert ("uri", "sqlite:///mlflow.db") in calls
    assert ("experiment", "local") in calls
    assert ("end", "FINISHED") in calls


def test_mlflow_reporter_keeps_tracking_uri_env(monkeypatch):
    import sys
    import types

    calls: list[tuple[str, object]] = []
    fake = types.ModuleType("mlflow")
    fake.set_tracking_uri = lambda uri: calls.append(("uri", uri))
    fake.set_experiment = lambda name: calls.append(("experiment", name))
    fake.start_run = lambda run_name=None: calls.append(("start", run_name))
    fake.log_dict = lambda data, artifact_file: None
    fake.end_run = lambda status=None: None
    monkeypatch.setitem(sys.modules, "mlflow", fake)
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")

    with MLflowReporter(tracking_uri=None, experiment="remote", run_name="run", config={}):
        pass

    assert not any(name == "uri" for name, _ in calls)
    assert ("experiment", "remote") in calls


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


def test_wandb_reporter_calls_init_log_finish(monkeypatch):
    import sys
    import types

    events: list[tuple] = []

    def init(**kwargs):
        events.append(("init", kwargs))

    def log(data, step=None):
        events.append(("log", dict(data), step))

    def finish():
        events.append(("finish",))

    fake_wandb = types.ModuleType("wandb")
    fake_wandb.init = init
    fake_wandb.log = log
    fake_wandb.finish = finish
    monkeypatch.setitem(sys.modules, "wandb", fake_wandb)

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

    assert events[0][0] == "init"
    assert events[0][1]["project"] == "p"
    assert events[0][1]["name"] == "n"
    assert events[0][1]["group"] == "g"
    assert events[0][1]["entity"] == "e"
    assert events[0][1]["mode"] == "offline"
    assert events[0][1]["config"] == {"a": 1}
    assert events[1] == ("log", {"loss": 0.2}, 4)
    assert events[2] == ("finish",)
