import logging
from collections.abc import Mapping
from contextvars import ContextVar, Token
from typing import Any, Literal, Protocol


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

    def store_config(self, config: Mapping[str, Any]) -> None:
        """Discard a replacement config.

        Args:
            config: resolved config after the scheduler has rewritten it.
        """


def _format_report(data: Mapping[str, Any], step: int | None) -> str:
    body = " ".join(f"{key}={value}" for key, value in data.items())
    if step is None:
        return body
    return f"step={step} {body}"


_current: ContextVar[Reporter] = ContextVar("reporter", default=NullReporter())
_deferred: ContextVar[BaseException | None] = ContextVar("deferred_report_error", default=None)


def get_reporter() -> Reporter:
    """Return the reporter installed for the current context."""
    return _current.get()


def store_run_config(config: Mapping[str, Any]) -> None:
    """Replace the config stored by the current reporter.

    Reporters that do not implement ``store_config`` are left unchanged.

    Args:
        config: resolved config after the scheduler has rewritten it.
    """
    store = getattr(get_reporter(), "store_config", None)
    if store is None:
        return
    store(config)


def defer_error(exc: BaseException) -> None:
    """Remember the first rank-local failure.

    Later failures stay behind the first one. :func:`raise_deferred_error`
    raises it after every rank has left the training loop.

    Args:
        exc: failure that must not abort this rank while others are still
            inside a collective.
    """
    if _deferred.get() is None:
        _deferred.set(exc)


def has_deferred_error() -> bool:
    """Return whether this process has remembered a rank-local failure."""
    return _deferred.get() is not None


def clear_deferred_error() -> None:
    """Drop a remembered rank-local failure without raising it."""
    _deferred.set(None)


def raise_deferred_error() -> None:
    """Raise the remembered rank-local failure, if this process stored one."""
    exc = _deferred.get()
    _deferred.set(None)
    if exc is not None:
        raise exc


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
    on_error: Literal["raise", "defer"] = "raise",
) -> None:
    """Report ``data`` to the current backend and optionally to a text logger.

    Args:
        data: values to record. The text line keeps every value. Each backend
            decides which of those values it stores.
        step: optional step index for this payload.
        logger: when given, also write one text line to this logger.
        on_error: ``"raise"`` propagates a backend failure immediately.
            ``"defer"`` remembers the first failure and still writes the text
            line, so the other ranks can leave the training loop together.

    Raises:
        ValueError: ``on_error`` is not ``"raise"`` or ``"defer"``.
    """
    if on_error not in ("raise", "defer"):
        raise ValueError(f"Unknown on_error: {on_error}")
    try:
        get_reporter().report(data, step=step)
    except Exception as exc:
        if on_error == "raise":
            raise
        defer_error(exc)
        if logger is not None:
            logger.error("report failed: %s", exc)
    if logger is not None:
        logger.info(_format_report(data, step))
