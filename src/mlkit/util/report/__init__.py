from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from typing import Any

from .core import (
    NullReporter,
    Reporter,
    clear_deferred_error,
    defer_error,
    get_reporter,
    has_deferred_error,
    raise_deferred_error,
    report,
    store_run_config,
    use_reporter,
)
from .mlflow import MLflowReporter
from .run_file import persist_report_run, resume_run_id, write_report_run
from .wandb import WandbReporter

__all__ = [
    "MLflowReporter",
    "NullReporter",
    "Reporter",
    "WandbReporter",
    "clear_deferred_error",
    "defer_error",
    "get_reporter",
    "has_deferred_error",
    "open_reporter",
    "persist_report_run",
    "raise_deferred_error",
    "report",
    "resume_run_id",
    "store_run_config",
    "use_reporter",
    "write_report_run",
]


@contextmanager
def open_reporter(
    kind: str,
    *,
    run_name: str,
    config: Mapping[str, Any],
    mlflow: Mapping[str, Any] | None = None,
    wandb: Mapping[str, Any] | None = None,
    resume_run_id: str | None = None,
) -> Iterator[Reporter]:
    """Open the reporter selected by ``kind``.

    Args:
        kind: ``"mlflow"``, ``"wandb"``, or ``"none"``.
        run_name: run name passed to the selected backend.
        config: resolved config stored with the run.
        mlflow: MLflow settings with ``tracking_uri``, ``experiment``, and
            optional ``artifact_location``.
        wandb: wandb settings with ``project``, ``group``, ``entity``, ``mode``,
            and ``reinit``.
        resume_run_id: existing run to continue. ``None`` starts a new run.

    Yields:
        The active reporter. The run is closed when the context exits.

    Raises:
        ValueError: ``kind`` is not one of the known reporters.
    """
    kind_name = kind.strip().lower()
    if kind_name == "none":
        yield NullReporter()
        return
    if kind_name == "mlflow":
        settings = dict(mlflow or {})
        reporter = MLflowReporter(
            tracking_uri=settings.get("tracking_uri") or None,
            experiment=str(settings.get("experiment") or "default"),
            run_name=run_name,
            config=config,
            run_id=resume_run_id,
            artifact_location=settings.get("artifact_location") or None,
        )
        with reporter:
            yield reporter
        return
    if kind_name == "wandb":
        settings = dict(wandb or {})
        reporter = WandbReporter(
            project=settings.get("project"),
            name=run_name,
            group=settings.get("group"),
            entity=settings.get("entity"),
            mode=settings.get("mode"),
            reinit=bool(settings.get("reinit", True)),
            config=config,
            run_id=resume_run_id,
        )
        with reporter:
            yield reporter
        return
    raise ValueError(f"Unknown reporter: {kind}")
