from collections.abc import Mapping
from typing import Any, Literal

from .data import _plain_data

WandbMode = Literal["online", "offline", "disabled", "shared"]


def _wandb_mode(mode: object) -> WandbMode | None:
    if mode is None:
        return None
    if mode == "online":
        return "online"
    if mode == "offline":
        return "offline"
    if mode == "disabled":
        return "disabled"
    if mode == "shared":
        return "shared"
    raise ValueError(f"Unknown wandb mode: {mode}")


class WandbReporter:
    """Reporter that writes a run through ``wandb.log``."""

    def __init__(
        self,
        *,
        project: str | None,
        name: str,
        group: str | None,
        entity: str | None,
        mode: str | None,
        reinit: bool,
        config: Mapping[str, Any],
        run_id: str | None = None,
    ):
        """Initialize the reporter.

        Args:
            project: wandb project.
            name: run name.
            group: wandb group.
            entity: wandb entity.
            mode: wandb mode, for example ``"offline"``.
            reinit: passed through to ``wandb.init``.
            config: resolved config attached to the run.
            run_id: existing run to continue. ``None`` starts a new run.
        """
        self.backend = "wandb"
        self.project = project
        self.name = name
        self.group = group
        self.entity = entity
        self.mode = mode
        self.reinit = reinit
        self.config = config
        self.run_id = run_id
        self._started = False

    def __enter__(self) -> "WandbReporter":
        import wandb

        if self.run_id is None:
            run = wandb.init(
                reinit=self.reinit,
                mode=_wandb_mode(self.mode),
                project=self.project,
                name=self.name,
                group=self.group,
                entity=self.entity,
                config=_plain_data(self.config),
            )
        else:
            run = wandb.init(
                reinit=self.reinit,
                mode=_wandb_mode(self.mode),
                project=self.project,
                name=self.name,
                group=self.group,
                entity=self.entity,
                config=_plain_data(self.config),
                id=self.run_id,
                resume="allow",
            )
        try:
            run_id = None if run is None else run.id
            if not isinstance(run_id, str) or run_id == "":
                raise RuntimeError("wandb did not start a run")
            self.run_id = run_id
        except Exception:
            wandb.finish(exit_code=1)
            raise
        self._started = True
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if not self._started:
            return None
        import wandb

        wandb.finish(exit_code=1 if exc_type is not None else 0)
        self._started = False
        return None

    def store_config(self, config: Mapping[str, Any]) -> None:
        """Replace the run config with the one training actually used.

        Args:
            config: resolved config after the scheduler has rewritten it.
        """
        self.config = config
        if not self._started:
            return
        import wandb

        wandb.config.update(_plain_data(self.config), allow_val_change=True)

    def report(self, data: Mapping[str, Any], *, step: int | None = None) -> None:
        """Forward the payload to ``wandb.log``.

        Args:
            data: values to record.
            step: optional step index for this payload.
        """
        import wandb

        payload = dict(data)
        if step is None:
            wandb.log(payload)
        else:
            wandb.log(payload, step=step)
