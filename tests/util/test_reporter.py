"""Reporter routing, the optional text logger, and the MLflow and wandb adapters."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

import pytest

from mlkit.pipeline.pipeline import latest_checkpoint_dir
from mlkit.util.report import (
    MLflowReporter,
    NullReporter,
    WandbReporter,
    clear_deferred_error,
    has_deferred_error,
    open_reporter,
    persist_report_run,
    raise_deferred_error,
    report,
    resume_run_id,
    use_reporter,
    write_report_run,
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


class _BoomReporter:
    def __init__(self, message: str) -> None:
        self.message = message

    def report(self, data: Mapping[str, Any], *, step: int | None = None) -> None:
        del data, step
        raise RuntimeError(self.message)


def test_report_defer_keeps_the_first_error_and_writes_text(caplog):
    logger = logging.getLogger("mlkit.test.reporter.defer")
    logger.setLevel(logging.INFO)
    try:
        with use_reporter(_BoomReporter("backend 1.0")):
            with caplog.at_level(logging.INFO, logger=logger.name):
                report({"loss": 1.0}, step=3, logger=logger, on_error="defer")
        with use_reporter(_BoomReporter("backend 2.0")):
            report({"loss": 2.0}, step=4, logger=logger, on_error="defer")

        assert has_deferred_error()
        messages = [record.message for record in caplog.records if record.name == logger.name]
        assert "step=3 loss=1.0" in messages
        assert any(message == "report failed: backend 1.0" for message in messages)
        with pytest.raises(RuntimeError, match="backend 1.0"):
            raise_deferred_error()
        assert not has_deferred_error()
    finally:
        clear_deferred_error()


def test_report_raise_is_the_default():
    with use_reporter(_BoomReporter("backend")):
        with pytest.raises(RuntimeError, match="backend"):
            report({"loss": 1.0}, step=1)
    assert not has_deferred_error()


def test_open_reporter_rejects_unknown_kind():
    with pytest.raises(ValueError, match="Unknown reporter"):
        with open_reporter("nope", run_name="run", config={}):
            pass


class _FakeMlflow:
    def __init__(self) -> None:
        self.calls: list[tuple[object, ...]] = []

    def set_tracking_uri(self, uri: object) -> None:
        self.calls.append(("uri", uri))

    def set_experiment(self, name: object) -> None:
        self.calls.append(("experiment", name))

    def start_run(self, run_id: object = None, run_name: object = None) -> Any:
        resolved = run_id if isinstance(run_id, str) and run_id != "" else "new-run"
        self.calls.append(("start", run_id, run_name))
        info = type("Info", (), {"run_id": resolved})()
        return type("ActiveRun", (), {"info": info})()

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

    assert not any(call[0] == "uri" for call in fake.calls)
    assert ("experiment", "remote") in fake.calls


def test_mlflow_reporter_sqlite_store(tmp_path):
    import mlflow
    from mlflow.tracking import MlflowClient

    tracking_uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    artifact_location = tmp_path / "mlruns"
    with (
        MLflowReporter(
            tracking_uri=tracking_uri,
            experiment="unit",
            run_name="run-a",
            config={"lr": 0.1, "nested": {"a": 1}},
            artifact_location=str(artifact_location),
        ) as mlflow_reporter,
        use_reporter(mlflow_reporter),
    ):
        report({"loss": 1.0, "bad": float("nan"), "name": "skip"}, step=0)
        report({"loss": 0.5}, step=1)

    mlflow.set_tracking_uri(tracking_uri)
    experiment = mlflow.get_experiment_by_name("unit")
    assert experiment is not None
    assert Path(experiment.artifact_location).resolve() == artifact_location.resolve()
    assert any(path.name == "config.yaml" for path in artifact_location.rglob("config.yaml"))
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


class _FakeWandbRun:
    def __init__(self, run_id: str) -> None:
        self.id = run_id


class _FakeWandb:
    def __init__(self) -> None:
        self.events: list[tuple] = []
        self.run: _FakeWandbRun | None = None

    def init(self, **kwargs: object) -> _FakeWandbRun:
        self.events.append(("init", kwargs))
        run_id = kwargs.get("id")
        if not isinstance(run_id, str) or run_id == "":
            run_id = "generated-id"
        self.run = _FakeWandbRun(run_id)
        return self.run

    def log(self, data: Mapping[str, object], step: object = None) -> None:
        self.events.append(("log", dict(data), step))

    def finish(self, exit_code: int | None = None) -> None:
        self.events.append(("finish", exit_code))


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
    assert "id" not in events[0][1]
    assert "resume" not in events[0][1]
    assert wandb_reporter.run_id == "generated-id"
    assert events[1] == ("log", {"loss": 0.2}, 4)
    assert events[2] == ("finish", 0)


def test_wandb_reporter_finish_marks_failure(monkeypatch):
    import sys

    fake_wandb = _FakeWandb()
    monkeypatch.setitem(sys.modules, "wandb", cast(Any, fake_wandb))

    with pytest.raises(RuntimeError, match="boom"):
        with WandbReporter(
            project="p",
            name="n",
            group=None,
            entity=None,
            mode="offline",
            reinit=True,
            config={},
        ):
            raise RuntimeError("boom")

    assert fake_wandb.events[-1] == ("finish", 1)


def test_wandb_reporter_resumes_with_allow(monkeypatch):
    import sys

    fake_wandb = _FakeWandb()
    monkeypatch.setitem(sys.modules, "wandb", cast(Any, fake_wandb))

    with WandbReporter(
        project="p",
        name="n",
        group=None,
        entity=None,
        mode="offline",
        reinit=True,
        config={},
        run_id="abc123",
    ) as wandb_reporter:
        assert wandb_reporter.run_id == "abc123"

    init_kwargs = fake_wandb.events[0][1]
    assert init_kwargs["id"] == "abc123"
    assert init_kwargs["resume"] == "allow"


def test_mlflow_reporter_artifact_location(tmp_path):
    import mlflow

    tracking_uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    artifact_location = tmp_path / "mlruns"
    with MLflowReporter(
        tracking_uri=tracking_uri,
        experiment="placed",
        run_name="run-a",
        config={"lr": 0.1},
        artifact_location=str(artifact_location),
    ):
        pass

    mlflow.set_tracking_uri(tracking_uri)
    experiment = mlflow.get_experiment_by_name("placed")
    assert experiment is not None
    assert Path(experiment.artifact_location).resolve() == artifact_location.resolve()
    assert any(path.name == "config.yaml" for path in artifact_location.rglob("config.yaml"))

    with MLflowReporter(
        tracking_uri=tracking_uri,
        experiment="placed",
        run_name="run-b",
        config={"lr": 0.2},
        artifact_location=str(tmp_path / "elsewhere"),
    ):
        pass

    experiment = mlflow.get_experiment_by_name("placed")
    assert experiment is not None
    assert Path(experiment.artifact_location).resolve() == artifact_location.resolve()
    assert not (tmp_path / "elsewhere").exists()


def test_mlflow_reporter_resumes_same_run(tmp_path):
    import mlflow
    from mlflow.tracking import MlflowClient

    tracking_uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    artifact_location = tmp_path / "mlruns"
    with (
        MLflowReporter(
            tracking_uri=tracking_uri,
            experiment="unit",
            run_name="run-a",
            config={"lr": 0.1},
            artifact_location=str(artifact_location),
        ) as first,
        use_reporter(first),
    ):
        report({"loss": 1.0}, step=0)
        run_id = first.run_id

    assert isinstance(run_id, str)
    with (
        MLflowReporter(
            tracking_uri=tracking_uri,
            experiment="unit",
            run_name="run-a",
            config={"lr": 0.1},
            run_id=run_id,
            artifact_location=str(artifact_location),
        ) as second,
        use_reporter(second),
    ):
        assert second.run_id == run_id
        report({"loss": 0.5}, step=1)

    mlflow.set_tracking_uri(tracking_uri)
    experiment = mlflow.get_experiment_by_name("unit")
    assert experiment is not None
    assert Path(experiment.artifact_location).resolve() == artifact_location.resolve()
    assert any(path.name == "config.yaml" for path in artifact_location.rglob("config.yaml"))
    frame = mlflow.search_runs(experiment_ids=[experiment.experiment_id])
    assert len(frame) == 1
    assert frame.iloc[0]["run_id"] == run_id

    client = MlflowClient(tracking_uri=tracking_uri)
    history = client.get_metric_history(run_id, "loss")
    assert [(metric.step, metric.value) for metric in history] == [(0, 1.0), (1, 0.5)]


def test_report_run_file_follows_latest_checkpoint_and_backend(tmp_path):
    older = tmp_path / "checkpoint_epoch0_step10_global10"
    newer = tmp_path / "checkpoint_epoch0_step20_global20"
    older.mkdir()
    newer.mkdir()
    write_report_run(str(older), backend="mlflow", run_id="old-run")
    write_report_run(str(newer), backend="mlflow", run_id="new-run")

    checkpoint_dir = latest_checkpoint_dir(str(tmp_path))
    assert checkpoint_dir == str(newer)
    assert resume_run_id(checkpoint_dir, "mlflow") == "new-run"
    assert resume_run_id(checkpoint_dir, "wandb") is None
    assert resume_run_id(None, "mlflow") is None

    empty = tmp_path / "checkpoint_epoch0_step30_global30"
    empty.mkdir()
    with use_reporter(NullReporter()):
        persist_report_run(str(empty))
    assert resume_run_id(str(empty), "mlflow") is None
