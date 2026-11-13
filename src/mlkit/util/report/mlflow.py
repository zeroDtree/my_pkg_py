import os
from collections.abc import Mapping
from typing import Any

from .data import _finite_metrics, _plain_data


class MLflowReporter:
    """Reporter that writes a run to MLflow Tracking.

    Finite numbers go to ``log_metrics``. The resolved config is stored as a
    ``config.yaml`` artifact. Other values are left for the text logger.

    When ``tracking_uri`` is omitted and ``MLFLOW_TRACKING_URI`` is unset, the
    run is stored in a local sqlite database, ``sqlite:///mlflow.db``. MLflow
    3 no longer accepts the ``./mlruns`` file store unless that store is
    explicitly opted into.
    """

    def __init__(
        self,
        *,
        tracking_uri: str | None,
        experiment: str,
        run_name: str,
        config: Mapping[str, Any],
        run_id: str | None = None,
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
        """
        self.backend = "mlflow"
        self.tracking_uri = tracking_uri or None
        self.experiment = experiment
        self.run_name = run_name
        self.config = config
        self.run_id = run_id
        self._started = False

    def __enter__(self) -> "MLflowReporter":
        import mlflow

        if self.tracking_uri:
            mlflow.set_tracking_uri(self.tracking_uri)
        elif not os.environ.get("MLFLOW_TRACKING_URI"):
            mlflow.set_tracking_uri("sqlite:///mlflow.db")
        mlflow.set_experiment(self.experiment)
        active = mlflow.start_run(run_id=self.run_id, run_name=self.run_name)
        run_id = active.info.run_id
        if not isinstance(run_id, str) or run_id == "":
            raise RuntimeError("MLflow did not start a run")
        self.run_id = run_id
        self._started = True
        mlflow.log_dict(_plain_data(self.config), "config.yaml")
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if not self._started:
            return None
        import mlflow

        mlflow.end_run(status="FAILED" if exc_type is not None else "FINISHED")
        self._started = False
        return None

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
