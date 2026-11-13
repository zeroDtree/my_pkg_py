from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from typing import Any

from .core import NullReporter, Reporter, get_reporter, report, use_reporter
from .mlflow import MLflowReporter
from .wandb import WandbReporter

__all__ = [
    "MLflowReporter",
    "NullReporter",
    "Reporter",
    "WandbReporter",
    "get_reporter",
    "open_reporter",
    "report",
    "use_reporter",
]


@contextmanager
def open_reporter(
    kind: str,
    *,
    run_name: str,
    config: Mapping[str, Any],
    mlflow: Mapping[str, Any] | None = None,
    wandb: Mapping[str, Any] | None = None,
) -> Iterator[Reporter]:
    """Open the reporter selected by ``kind``.

    Args:
        kind: ``"mlflow"``, ``"wandb"``, or ``"none"``.
        run_name: run name passed to the selected backend.
        config: resolved config stored with the run.
        mlflow: MLflow settings with ``tracking_uri`` and ``experiment``.
        wandb: wandb settings with ``project``, ``group``, ``entity``, ``mode``,
            and ``reinit``.

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
        )
        with reporter:
            yield reporter
        return
    raise ValueError(f"Unknown reporter: {kind}")
