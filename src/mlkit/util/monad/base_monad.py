"""Abstract monad: functor T with unit, fmap, join, and bind.

Category-theory correspondence:

- Object `A`              → type `A`
- Morphism `f: A → B`     → function `f: A -> B`
- Functor `T`             → type constructor (subclass of Monad)
- `T(f): T(A) → T(B)`     → `fmap`
- `η: Id → T`             → `unit`
- `μ: T² → T`             → `join`
- `T(f)` then `μ`         → `bind` (`bind(f) = join ∘ fmap(f)`)
"""

from __future__ import annotations

import abc
from collections.abc import Callable
from typing import Self


class Monad[A](abc.ABC):
    """Abstract monad. Subclasses implement unit, fmap, and join; bind is derived."""

    @classmethod
    @abc.abstractmethod
    def unit(cls, x: A) -> Self:
        """η: Id → T. Lift a value into the monad."""

    @abc.abstractmethod
    def fmap[B](self, f: Callable[[A], B]) -> Monad[B]:
        """T(f): map `f` over the contained value."""

    @abc.abstractmethod
    def join[B](self: Monad[Monad[B]]) -> Monad[B]:
        """μ: T² → T. Flatten `T(T(B))` to `T(B)`."""

    def bind[B](self, f: Callable[[A], Monad[B]]) -> Monad[B]:
        """bind(f) = join ∘ fmap(f)."""
        return self.fmap(f).join()
