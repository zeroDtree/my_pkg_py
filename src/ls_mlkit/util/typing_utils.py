from typing import TypeVar

T = TypeVar("T")


def require(value: T | None, name: str) -> T:
    """Return `value` or raise if it is `None`."""
    if value is None:
        raise TypeError(f"{name} is required")
    return value
