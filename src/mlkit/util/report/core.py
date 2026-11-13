import logging
from collections.abc import Mapping
from contextvars import ContextVar, Token
from typing import Any, Protocol


class Reporter(Protocol):
    """Backend that receives one reported payload.

    The payload is an ordinary mapping. A backend decides which values it
    stores and how.
    """

    def report(self, data: Mapping[str, Any], *, step: int | None = None) -> None:
        """Record one payload.

        Args:
            data: values to record.
            step: optional step index for this payload.
        """
        ...


class NullReporter:
    """Reporter that discards every payload."""

    def report(self, data: Mapping[str, Any], *, step: int | None = None) -> None:
        """Discard the payload.

        Args:
            data: values to record.
            step: optional step index for this payload.
        """


def _format_report(data: Mapping[str, Any], step: int | None) -> str:
    body = " ".join(f"{key}={value}" for key, value in data.items())
    if step is None:
        return body
    return f"step={step} {body}"


_current: ContextVar[Reporter] = ContextVar("reporter", default=NullReporter())


def get_reporter() -> Reporter:
    """Return the reporter installed for the current context."""
    return _current.get()


class use_reporter:
    """Install a reporter for the body of a ``with`` statement."""

    def __init__(self, reporter: Reporter):
        """Remember the reporter to install.

        Args:
            reporter: backend used by :func:`report` inside the block.
        """
        self._reporter = reporter
        self._token: Token[Reporter] | None = None

    def __enter__(self) -> Reporter:
        """Install the reporter and return it."""
        self._token = _current.set(self._reporter)
        return self._reporter

    def __exit__(self, exc_type, exc, tb) -> None:
        """Restore the reporter that was current before this block."""
        if self._token is not None:
            _current.reset(self._token)
            self._token = None


def report(
    data: Mapping[str, Any],
    *,
    step: int | None = None,
    logger: logging.Logger | None = None,
) -> None:
    """Report ``data`` to the current backend and optionally to a text logger.

    Args:
        data: values to record. The text line keeps every value. Each backend
            decides which of those values it stores.
        step: optional step index for this payload.
        logger: when given, also write one text line to this logger.
    """
    get_reporter().report(data, step=step)
    if logger is not None:
        logger.info(_format_report(data, step))
