import json
import os

from .core import get_reporter

REPORT_RUN_FILE = "report_run.json"


def write_report_run(directory: str, *, backend: str, run_id: str) -> None:
    """Store the backend run id inside a checkpoint directory.

    Args:
        directory: checkpoint directory.
        backend: reporter name, ``"mlflow"`` or ``"wandb"``.
        run_id: id of the run that was open when the checkpoint was saved.
    """
    path = os.path.join(directory, REPORT_RUN_FILE)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump({"backend": backend, "run_id": run_id}, handle)


def read_report_run(directory: str) -> tuple[str, str] | None:
    """Read a backend name and run id from a checkpoint directory.

    Args:
        directory: checkpoint directory.

    Returns:
        ``(backend, run_id)`` when the file is present and well formed,
        otherwise ``None``.
    """
    path = os.path.join(directory, REPORT_RUN_FILE)
    if not os.path.isfile(path):
        return None
    try:
        with open(path, encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    backend = payload.get("backend")
    run_id = payload.get("run_id")
    if not isinstance(backend, str) or not isinstance(run_id, str) or backend == "" or run_id == "":
        return None
    return backend, run_id


def persist_report_run(directory: str) -> None:
    """Write the current reporter's run id into ``directory`` when one is open."""
    reporter = get_reporter()
    backend = getattr(reporter, "backend", None)
    run_id = getattr(reporter, "run_id", None)
    if not isinstance(backend, str) or not isinstance(run_id, str) or run_id == "":
        return
    write_report_run(directory, backend=backend, run_id=run_id)


def resume_run_id(checkpoint_dir: str | None, backend: str) -> str | None:
    """Return the run id to continue, when the checkpoint was written by ``backend``.

    Args:
        checkpoint_dir: latest checkpoint directory, or ``None`` when there is none.
        backend: reporter that is about to be opened.

    Returns:
        The stored run id when the file exists and names the same backend.
    """
    if checkpoint_dir is None:
        return None
    found = read_report_run(checkpoint_dir)
    if found is None or found[0] != backend:
        return None
    return found[1]
