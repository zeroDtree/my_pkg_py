from contextlib import contextmanager
from typing import Any, Iterable, Union


class TemporaryKeyRemover:
    """Context manager for temporarily removing keys from a mutable mapping."""

    def __init__(self, mapping: Any, keys: Union[str, Iterable[str]], default: Any = None):
        self.mapping = mapping
        self.keys = [keys] if isinstance(keys, str) else list(keys)
        self.default = default
        self.removed = {}

    def __enter__(self):
        for key in self.keys:
            try:
                self.removed[key] = self.mapping[key]
                del self.mapping[key]
            except (KeyError, TypeError):
                self.removed[key] = self.default
        return self

    def __exit__(self, *exc):
        for key, value in self.removed.items():
            if value is not None:
                self.mapping[key] = value


@contextmanager
def without_keys(mapping: Any, *keys: str):
    """Temporarily remove specified keys from a mapping."""
    removed = {}

    for key in keys:
        try:
            removed[key] = mapping[key]
            del mapping[key]
        except (KeyError, TypeError):
            removed[key] = None

    try:
        yield mapping
    finally:
        for key, value in removed.items():
            if value is not None:
                mapping[key] = value


if __name__ == "__main__":
    # Usage
    data = {"a": 1, "b": 2, "c": 3}

    # Class-based approach
    with TemporaryKeyRemover(data, ["a", "b"]):
        print(f"Without keys: {data}")  # {'c': 3}

    print(f"Restored: {data}")  # {'a': 1, 'b': 2, 'c': 3}
