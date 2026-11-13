from collections.abc import Mapping
from typing import Any

from .data import _plain_data


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
        """
        self.project = project
        self.name = name
        self.group = group
        self.entity = entity
        self.mode = mode
        self.reinit = reinit
        self.config = config
        self._started = False

    def __enter__(self) -> "WandbReporter":
        import wandb

        wandb.init(
            reinit=self.reinit,
            mode=self.mode,
            project=self.project,
            name=self.name,
            group=self.group,
            entity=self.entity,
            config=_plain_data(self.config),
        )
        self._started = True
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if not self._started:
            return None
        import wandb

        wandb.finish()
        self._started = False
        return None

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
