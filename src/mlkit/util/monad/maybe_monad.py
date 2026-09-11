"""Maybe monad: optional values. `None` is Nothing.

Implements `Monad`: unit, fmap, and join. See `Monad` for the category-theory correspondence.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass

from .base_monad import Monad


@dataclass(frozen=True, slots=True)
class Maybe[A](Monad[A]):
    """Optional value. `None` is Nothing."""

    value: A | None = None

    @classmethod
    def unit(cls, x: A) -> Maybe[A]:
        """η: Id → T. Lift a value into Maybe."""
        return cls(x)

    @classmethod
    def nothing(cls) -> Maybe[A]:
        return cls()

    def fmap[B](self, f: Callable[[A], B]) -> Maybe[B]:
        """T(f): map `f` over the contained value, or Nothing."""
        return Maybe.nothing() if self.value is None else Maybe.unit(f(self.value))

    def join[B](self: Maybe[Maybe[B]]) -> Maybe[B]:
        """μ: T² → T. Flatten `Maybe[Maybe[B]]` to `Maybe[B]`."""
        return self.value if isinstance(self.value, Maybe) else Maybe.nothing()

    def bind[B](self, f: Callable[[A], Maybe[B]]) -> Maybe[B]:  # ty: ignore[invalid-method-override]
        """bind(f) = join ∘ fmap(f)."""
        return self.fmap(f).join()

    def get_or(self, default: A) -> A:
        return default if self.value is None else self.value

    def get[V](self, key: object) -> Maybe[V]:
        """Look up `key` in a mapping. Missing key or `None` is Nothing."""
        if not isinstance(self.value, Mapping):
            return Maybe.nothing()
        inner = self.value.get(key)
        return Maybe.nothing() if inner is None else Maybe.unit(inner)


if __name__ == "__main__":

    def get_user_city(user_data: Mapping[str, object] | None, default: str = "Unknown") -> str:
        """Read nested `user.address.city`. Missing keys or `None` yield `default`."""
        return Maybe.unit(user_data).get("address").get("city").get_or(default)

    def get_user_city_without_monad(user_data: dict | None) -> str:
        """Each missing layer is another branch."""
        if user_data is None:
            return "Unknown"
        if "address" not in user_data:
            return "Unknown"
        if user_data["address"] is None:
            return "Unknown"
        if "city" not in user_data["address"]:
            return "Unknown"
        if user_data["address"]["city"] is None:
            return "Unknown"
        return user_data["address"]["city"]

    def get_user_city_with_monad(user_data: dict | None) -> str:
        """Nothing short-circuits the chain; no nested branches."""
        return (
            Maybe.unit(user_data)
            .bind(lambda data: Maybe.unit(data.get("address")))
            .bind(lambda address: Maybe.unit(address.get("city")))
            .get_or("Unknown")
        )

    cases: dict[str, dict | None] = {
        "complete": {"address": {"city": "Paris"}},
        "missing_city": {"address": {}},
        "none_city": {"address": {"city": None}},
        "none_address": {"address": None},
        "empty": {},
        "none_user": None,
    }

    print(f"{'case':<16} {'nested':<10} {'bind':<10} {'get'}")
    for name, data in cases.items():
        nested = get_user_city_without_monad(data)
        bound = get_user_city_with_monad(data)
        chained = get_user_city(data)
        print(f"{name:<16} {nested:<10} {bound:<10} {chained}")
        assert nested == bound == chained
