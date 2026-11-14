import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .data import _finite_metrics, _plain_data


def _local_sqlite_path(uri: str) -> Path | None:
    """Return the database file behind a local sqlite URI.

    ``None`` means there is no file to prepare: the URI is not sqlite, the path
    is empty (``sqlite:///``), or it is an in-memory database
    (``sqlite:///:memory:``).
    """
    prefix = "sqlite:///"
    if not uri.startswith(prefix):
        return None
    raw = uri[len(prefix) :]
    if raw == "" or raw.startswith(":"):
        return None
    return Path(raw)


def _prepare_sqlite_dir(uri: str) -> None:
    """Create the parent directory of a local sqlite database file.

    Sqlite creates the database file but not its parent. Remote URIs, an empty
    path (``sqlite:///``), and in-memory databases are skipped.
    """
    db_path = _local_sqlite_path(uri)
    if db_path is not None:
        db_path.parent.mkdir(parents=True, exist_ok=True)


def _prepare_artifact_dir(artifact_location: str) -> None:
    """Create a local artifact directory and any missing parents.

    Remote URIs that contain ``://`` are left untouched.
    """
    if "://" not in artifact_location:
        Path(artifact_location).mkdir(parents=True, exist_ok=True)


class MLflowReporter:
    """Reporter that writes a run to MLflow Tracking.

    Finite numbers go to ``log_metrics``. The resolved config is stored as a
    ``config.yaml`` artifact. Other values are left for the text logger.

    When ``tracking_uri`` is omitted and ``MLFLOW_TRACKING_URI`` is unset, the
    run is stored in a local sqlite database, ``sqlite:///mlflow.db``. MLflow
    3 no longer accepts the ``./mlruns`` file store unless that store is
    explicitly opted into.

    When ``artifact_location`` is set and the experiment does not exist yet,
    the experiment is created with that directory. An existing experiment keeps
    its artifact location. Paths are relative to the current working directory.
    """

    def __init__(
        self,
        *,
        tracking_uri: str | None,
        experiment: str,
        run_name: str,
        config: Mapping[str, Any],
        run_id: str | None = None,
        artifact_location: str | None = None,
    ):
        """Initialize the reporter.

        Args:
            tracking_uri: MLflow tracking URI. ``None`` keeps
                ``MLFLOW_TRACKING_URI`` when that variable is set, and otherwise
                uses ``sqlite:///mlflow.db``.
            experiment: experiment name.
            run_name: run name.
            config: resolved config stored as an artifact.
            run_id: existing run to continue. ``None`` starts a new run.
            artifact_location: artifact directory for a new experiment.
                ``None`` leaves the directory to MLflow.
        """
        self.backend = "mlflow"
        self.tracking_uri = tracking_uri or None
        self.experiment = experiment
        self.run_name = run_name
        self.config = config
        self.run_id = run_id
        self.artifact_location = artifact_location or None
        self._started = False

    def __enter__(self) -> "MLflowReporter":
        import mlflow

        if self.tracking_uri:
            active_uri = self.tracking_uri
        elif env_uri := os.environ.get("MLFLOW_TRACKING_URI"):
            active_uri = env_uri
        else:
            active_uri = "sqlite:///mlflow.db"
        _prepare_sqlite_dir(active_uri)
        if self.tracking_uri or not os.environ.get("MLFLOW_TRACKING_URI"):
            mlflow.set_tracking_uri(active_uri)
        if self.artifact_location and mlflow.get_experiment_by_name(self.experiment) is None:
            from mlflow.exceptions import MlflowException

            _prepare_artifact_dir(self.artifact_location)
            try:
                mlflow.create_experiment(self.experiment, artifact_location=self.artifact_location)
            except MlflowException as exc:
                if getattr(exc, "error_code", None) != "RESOURCE_ALREADY_EXISTS":
                    raise
        mlflow.set_experiment(self.experiment)
        active = mlflow.start_run(run_id=self.run_id, run_name=self.run_name)
        try:
            run_id = active.info.run_id
            if not isinstance(run_id, str) or run_id == "":
                raise RuntimeError("MLflow did not start a run")
            self.run_id = run_id
            mlflow.log_dict(_plain_data(self.config), "config.yaml")
        except Exception:
            mlflow.end_run(status="FAILED")
            raise
        self._started = True
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if not self._started:
            return None
        import mlflow

        mlflow.end_run(status="FAILED" if exc_type is not None else "FINISHED")
        self._started = False
        return None

    def store_config(self, config: Mapping[str, Any]) -> None:
        """Replace the stored config with the one training actually used.

        Args:
            config: resolved config after the scheduler has rewritten it.
        """
        self.config = config
        if not self._started:
            return
        import mlflow

        mlflow.log_dict(_plain_data(self.config), "config.yaml")

    def report(self, data: Mapping[str, Any], *, step: int | None = None) -> None:
        """Write finite numeric values as MLflow metrics.

        Args:
            data: values to record.
            step: optional step index for this payload.
        """
        metrics = _finite_metrics(data)
        if not metrics:
            return
        import mlflow

        if step is None:
            mlflow.log_metrics(metrics)
        else:
            mlflow.log_metrics(metrics, step=step)
