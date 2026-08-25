"""Event-driven, generic, Enum-typed synchronous finite state machine.

The machine stores topology, ``Context`` stores data, and guards/actions store
behavior. Extra runtime data belongs on the context object; events carry no
payload. Not thread-safe.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Generic, TypeVar, cast

S = TypeVar("S", bound=Enum)  # State
E = TypeVar("E", bound=Enum)  # Event
C = TypeVar("C")  # Context

type Guard[C, E] = Callable[[C, E], bool]  # Guard function that returns a boolean
type Action[C, E] = Callable[[C, E], None]  # Action function that takes a context and an event and returns None
type OnStateChange[S, E, C] = Callable[
    [S, S, E, C], None
]  # OnStateChange function that takes a source state, a target state, an event, and a context and returns None


class AnySource:
    """Sentinel matching any source state."""

    __slots__ = ()

    def __repr__(self) -> str:
        return "ANY"

    def __hash__(self) -> int:
        return hash("ANY")

    def __eq__(self, other: object) -> bool:
        return isinstance(other, AnySource)


ANY = AnySource()


class StateMachineError(Exception):
    """Base error for the state machine."""


class InvalidTransitionError(StateMachineError):
    """Raised when ``send`` cannot fire a transition."""


@dataclass(frozen=True, slots=True)
class Snapshot(Generic[S, C]):
    """Point-in-time machine state. Context is stored by reference."""

    state: S
    context: C


@dataclass(frozen=True, slots=True)
class Transition(Generic[S, E, C]):
    """One directed edge in the state graph."""

    source: S | AnySource
    event: E
    target: S
    guard: Guard[C, E] | None = None
    actions: tuple[Action[C, E], ...] = ()


class StateMachine(Generic[S, E, C]):
    """Synchronous event-driven state machine.

    Configure transitions with the fluent ``add_transition`` / ``on_enter`` /
    ``on_exit`` API, then drive the machine with ``send`` or ``try_send``.
    """

    def __init__(
        self,
        initial_state: S,
        context: C,
        *,
        on_state_change: OnStateChange[S, E, C] | None = None,
    ) -> None:
        self._current_state = initial_state
        self._context = context
        self._on_state_change = on_state_change
        self._transitions: dict[tuple[S | AnySource, E], list[Transition[S, E, C]]] = {}
        self._enter_actions: dict[S, list[Action[C, E]]] = {}
        self._exit_actions: dict[S, list[Action[C, E]]] = {}
        self._known_states: dict[S, None] = {initial_state: None}
        self._initial_state = initial_state

    @property
    def current_state(self) -> S:
        return self._current_state

    @property
    def context(self) -> C:
        return self._context

    def add_transition(
        self,
        source: S | Sequence[S] | AnySource,
        event: E,
        target: S,
        guard: Guard[C, E] | None = None,
        actions: Sequence[Action[C, E]] | None = None,
    ) -> StateMachine[S, E, C]:
        """Register a transition. Returns ``self`` for chaining."""
        action_tuple = tuple(actions) if actions is not None else ()
        sources = self._iter_sources(source)
        if not sources:
            raise ValueError("add_transition requires at least one source state")

        self._remember_state(target)
        for src in sources:
            if isinstance(src, AnySource):
                trans_source: S | AnySource = ANY
            else:
                self._remember_state(src)
                trans_source = src
            trans = Transition(
                source=trans_source,
                event=event,
                target=target,
                guard=guard,
                actions=action_tuple,
            )
            self._transitions.setdefault((trans_source, event), []).append(trans)
        return self

    def on_enter(self, state: S, action: Action[C, E]) -> StateMachine[S, E, C]:
        """Register a hook that runs after the machine enters ``state``."""
        self._remember_state(state)
        self._enter_actions.setdefault(state, []).append(action)
        return self

    def on_exit(self, state: S, action: Action[C, E]) -> StateMachine[S, E, C]:
        """Register a hook that runs before the machine leaves ``state``."""
        self._remember_state(state)
        self._exit_actions.setdefault(state, []).append(action)
        return self

    def can_handle(self, event: E) -> bool:
        """Return whether a matching transition would fire for ``event``."""
        return self._match(event) is not None

    def available_events(self) -> tuple[E, ...]:
        """Events that currently have at least one matching, passing transition."""
        seen: list[E] = []
        seen_set: set[E] = set()
        for (source, event), transitions in self._transitions.items():
            if event in seen_set:
                continue
            if source is not ANY and source != self._current_state:
                continue
            for trans in transitions:
                if self._guard_passes(trans, event):
                    seen.append(event)
                    seen_set.add(event)
                    break
        return tuple(seen)

    def send(self, event: E) -> S:
        """Fire the first matching transition, or raise ``InvalidTransitionError``."""
        trans = self._match(event)
        if trans is None:
            raise InvalidTransitionError(
                f"No valid transition from state '{self._current_state.name}' on event '{event.name}'"
            )
        return self._apply(trans, event)

    def try_send(self, event: E) -> bool:
        """Like ``send``, but return ``False`` instead of raising when nothing matches."""
        trans = self._match(event)
        if trans is None:
            return False
        self._apply(trans, event)
        return True

    def snapshot(self) -> Snapshot[S, C]:
        """Capture the current state and context reference."""
        return Snapshot(state=self._current_state, context=self._context)

    def restore(self, snapshot: Snapshot[S, C]) -> None:
        """Replace the current state and context from a snapshot."""
        self._current_state = snapshot.state
        self._context = snapshot.context

    def to_dot(self) -> str:
        """Return a Graphviz DOT representation of the registered topology."""
        lines = [
            "digraph StateMachine {",
            "  rankdir=LR;",
        ]
        for state in self._known_states:
            name = state.name
            shape = "doublecircle" if state is self._initial_state else "circle"
            lines.append(f'  "{name}" [shape={shape}];')

        if any(source is ANY for source, _event in self._transitions):
            lines.append('  "*" [shape=plaintext];')

        for (source, event), transitions in self._transitions.items():
            if isinstance(source, AnySource):
                src_label = "*"
            else:
                src_label = source.name
            for trans in transitions:
                lines.append(f'  "{src_label}" -> "{trans.target.name}" [label="{event.name}"];')

        lines.append("}")
        return "\n".join(lines) + "\n"

    def _remember_state(self, state: S) -> None:
        self._known_states[state] = None

    def _iter_sources(self, source: S | Sequence[S] | AnySource) -> list[S | AnySource]:
        if isinstance(source, AnySource):
            return [ANY]
        if isinstance(source, Enum):
            return [cast(S, source)]
        return list(source)

    def _candidates(self, event: E) -> list[Transition[S, E, C]]:
        specific = self._transitions.get((self._current_state, event), [])
        wildcard = self._transitions.get((ANY, event), [])
        return [*specific, *wildcard]

    def _guard_passes(self, trans: Transition[S, E, C], event: E) -> bool:
        return trans.guard is None or trans.guard(self._context, event)

    def _match(self, event: E) -> Transition[S, E, C] | None:
        for trans in self._candidates(event):
            if self._guard_passes(trans, event):
                return trans
        return None

    def _apply(self, trans: Transition[S, E, C], event: E) -> S:
        from_state = self._current_state
        to_state = trans.target

        for action in self._exit_actions.get(from_state, []):
            action(self._context, event)
        for action in trans.actions:
            action(self._context, event)

        self._current_state = to_state

        for action in self._enter_actions.get(to_state, []):
            action(self._context, event)
        if self._on_state_change is not None:
            self._on_state_change(from_state, to_state, event, self._context)

        return self._current_state


if __name__ == "__main__":

    class TaskState(Enum):
        PENDING = "pending"
        RUNNING = "running"
        COMPLETED = "completed"
        FAILED = "failed"

    class TaskEvent(Enum):
        START = "start"
        SUCCESS = "success"
        FAIL = "fail"
        RETRY = "retry"
        CANCEL = "cancel"

    @dataclass
    class TaskContext:
        task_id: str
        retry_count: int = 0
        max_retries: int = 2
        last_error: str | None = None

    def can_retry(ctx: TaskContext, event: TaskEvent) -> bool:
        return ctx.retry_count < ctx.max_retries

    def increment_retry(ctx: TaskContext, event: TaskEvent) -> None:
        ctx.retry_count += 1
        print(f"[action] retry {ctx.task_id} ({ctx.retry_count}/{ctx.max_retries})")

    def log_enter(ctx: TaskContext, event: TaskEvent) -> None:
        print(f"[enter] {event.name} -> running task {ctx.task_id}")

    def log_exit(ctx: TaskContext, event: TaskEvent) -> None:
        print(f"[exit] leaving failed after {event.name}")

    def log_state_change(
        from_state: TaskState,
        to_state: TaskState,
        event: TaskEvent,
        ctx: TaskContext,
    ) -> None:
        print(f"[change] {from_state.name} --({event.name})--> {to_state.name}")

    ctx = TaskContext(task_id="JOB-1001", max_retries=2)
    sm = StateMachine[TaskState, TaskEvent, TaskContext](
        initial_state=TaskState.PENDING,
        context=ctx,
        on_state_change=log_state_change,
    )
    (
        sm.add_transition(TaskState.PENDING, TaskEvent.START, TaskState.RUNNING)
        .add_transition(TaskState.RUNNING, TaskEvent.SUCCESS, TaskState.COMPLETED)
        .add_transition(TaskState.RUNNING, TaskEvent.FAIL, TaskState.FAILED)
        .add_transition(
            TaskState.FAILED,
            TaskEvent.RETRY,
            TaskState.RUNNING,
            guard=can_retry,
            actions=[increment_retry],
        )
        .add_transition(ANY, TaskEvent.CANCEL, TaskState.FAILED)
        .on_enter(TaskState.RUNNING, log_enter)
        .on_exit(TaskState.FAILED, log_exit)
    )

    print("DOT graph:")
    print(sm.to_dot())

    try:
        import os
        import shutil
        from pathlib import Path

        import graphviz

        conda_bin = Path.home() / "miniconda3" / "bin"
        if shutil.which("dot") is None and (conda_bin / "dot").is_file():
            os.environ["PATH"] = str(conda_bin) + os.pathsep + os.environ.get("PATH", "")

        image_path = graphviz.Source(sm.to_dot()).render(
            filename="/tmp/state_machine_example",
            format="png",
            cleanup=True,
        )
        print(f"graph image: {image_path}")
    except Exception as exc:
        print(f"could not render graph image: {exc}")

    sm.send(TaskEvent.START)
    sm.send(TaskEvent.FAIL)
    sm.send(TaskEvent.RETRY)
    sm.send(TaskEvent.FAIL)
    sm.send(TaskEvent.RETRY)
    sm.send(TaskEvent.SUCCESS)

    print(f"final state={sm.current_state.value}, retries={sm.context.retry_count}")
    print(f"available events={tuple(e.name for e in sm.available_events())}")
    print(f"try extra retry={sm.try_send(TaskEvent.RETRY)}")
