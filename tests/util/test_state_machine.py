"""Tests for the event-driven typed state machine."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import pytest

from mlkit.util.state_machine import (
    ANY,
    InvalidTransitionError,
    Snapshot,
    StateMachine,
)


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
    max_retries: int = 3
    last_error: str | None = None


def can_retry(ctx: TaskContext, event: TaskEvent) -> bool:
    return ctx.retry_count < ctx.max_retries


def increment_retry(ctx: TaskContext, event: TaskEvent) -> None:
    ctx.retry_count += 1


def build_task_machine(
    ctx: TaskContext,
    *,
    on_state_change=None,
) -> StateMachine[TaskState, TaskEvent, TaskContext]:
    sm = StateMachine[TaskState, TaskEvent, TaskContext](
        initial_state=TaskState.PENDING,
        context=ctx,
        on_state_change=on_state_change,
    )
    sm.add_transition(TaskState.PENDING, TaskEvent.START, TaskState.RUNNING)
    sm.add_transition(TaskState.RUNNING, TaskEvent.SUCCESS, TaskState.COMPLETED)
    sm.add_transition(TaskState.RUNNING, TaskEvent.FAIL, TaskState.FAILED)
    sm.add_transition(
        TaskState.FAILED,
        TaskEvent.RETRY,
        TaskState.RUNNING,
        guard=can_retry,
        actions=[increment_retry],
    )
    return sm


def test_task_retry_scenario_blocks_extra_retries() -> None:
    ctx = TaskContext(task_id="JOB-1001", max_retries=2)
    sm = build_task_machine(ctx)

    assert sm.send(TaskEvent.START) is TaskState.RUNNING
    assert sm.send(TaskEvent.FAIL) is TaskState.FAILED

    assert sm.send(TaskEvent.RETRY) is TaskState.RUNNING
    assert ctx.retry_count == 1
    assert sm.send(TaskEvent.FAIL) is TaskState.FAILED

    assert sm.send(TaskEvent.RETRY) is TaskState.RUNNING
    assert ctx.retry_count == 2
    assert sm.send(TaskEvent.FAIL) is TaskState.FAILED

    assert sm.can_handle(TaskEvent.RETRY) is False
    with pytest.raises(InvalidTransitionError, match="FAILED"):
        sm.send(TaskEvent.RETRY)
    assert sm.current_state is TaskState.FAILED
    assert sm.context.retry_count == 2


def test_guard_fallback_second_transition_fires() -> None:
    class Phase(Enum):
        A = "a"
        B = "b"
        C = "c"

    class Evt(Enum):
        GO = "go"

    @dataclass
    class Ctx:
        allow_b: bool = False

    sm = StateMachine[Phase, Evt, Ctx](Phase.A, Ctx())
    sm.add_transition(Phase.A, Evt.GO, Phase.B, guard=lambda ctx, _e: ctx.allow_b)
    sm.add_transition(Phase.A, Evt.GO, Phase.C, guard=lambda ctx, _e: not ctx.allow_b)

    assert sm.send(Evt.GO) is Phase.C
    sm.restore(Snapshot(state=Phase.A, context=Ctx(allow_b=True)))
    assert sm.send(Evt.GO) is Phase.B


def test_hook_order_and_enter_sees_new_state() -> None:
    log: list[str] = []

    def on_exit(ctx: TaskContext, event: TaskEvent) -> None:
        log.append(f"exit:{sm.current_state.name}:{event.name}")

    def on_trans(ctx: TaskContext, event: TaskEvent) -> None:
        log.append(f"action:{sm.current_state.name}:{event.name}")

    def on_enter(ctx: TaskContext, event: TaskEvent) -> None:
        log.append(f"enter:{sm.current_state.name}:{event.name}")

    def on_change(from_s: TaskState, to_s: TaskState, event: TaskEvent, ctx: TaskContext) -> None:
        log.append(f"change:{from_s.name}->{to_s.name}:{event.name}")

    ctx = TaskContext(task_id="order")
    sm = StateMachine[TaskState, TaskEvent, TaskContext](
        TaskState.PENDING,
        ctx,
        on_state_change=on_change,
    )
    sm.add_transition(
        TaskState.PENDING,
        TaskEvent.START,
        TaskState.RUNNING,
        actions=[on_trans],
    )
    sm.on_exit(TaskState.PENDING, on_exit)
    sm.on_enter(TaskState.RUNNING, on_enter)

    sm.send(TaskEvent.START)

    assert log == [
        "exit:PENDING:START",
        "action:PENDING:START",
        "enter:RUNNING:START",
        "change:PENDING->RUNNING:START",
    ]
    assert sm.current_state is TaskState.RUNNING


def test_any_source_and_multi_source() -> None:
    ctx = TaskContext(task_id="any")
    sm = build_task_machine(ctx)
    sm.add_transition(ANY, TaskEvent.CANCEL, TaskState.FAILED)
    sm.add_transition(
        (TaskState.PENDING, TaskState.RUNNING),
        TaskEvent.SUCCESS,
        TaskState.COMPLETED,
    )

    sm.send(TaskEvent.START)
    assert sm.send(TaskEvent.CANCEL) is TaskState.FAILED

    sm.restore(Snapshot(state=TaskState.PENDING, context=ctx))
    assert sm.send(TaskEvent.SUCCESS) is TaskState.COMPLETED


def test_can_handle_available_events_try_send_vs_send() -> None:
    ctx = TaskContext(task_id="query")
    sm = build_task_machine(ctx)

    assert sm.can_handle(TaskEvent.START) is True
    assert sm.can_handle(TaskEvent.FAIL) is False
    assert TaskEvent.START in sm.available_events()
    assert TaskEvent.FAIL not in sm.available_events()

    assert sm.try_send(TaskEvent.FAIL) is False
    assert sm.current_state is TaskState.PENDING

    assert sm.try_send(TaskEvent.START) is True
    assert sm.current_state is TaskState.RUNNING

    with pytest.raises(InvalidTransitionError):
        sm.send(TaskEvent.START)


def test_snapshot_restore_swaps_state_and_context() -> None:
    ctx = TaskContext(task_id="snap", retry_count=1)
    sm = build_task_machine(ctx)
    sm.send(TaskEvent.START)

    snap = sm.snapshot()
    assert snap.state is TaskState.RUNNING
    assert snap.context is ctx

    other = TaskContext(task_id="other")
    sm.restore(Snapshot(state=TaskState.FAILED, context=other))
    assert sm.current_state is TaskState.FAILED
    assert sm.context is other
    assert sm.context.task_id == "other"

    sm.restore(snap)
    assert sm.current_state is TaskState.RUNNING
    assert sm.context is ctx


def test_to_dot_contains_nodes_and_edges() -> None:
    ctx = TaskContext(task_id="dot")
    sm = build_task_machine(ctx)
    sm.add_transition(ANY, TaskEvent.CANCEL, TaskState.FAILED)
    dot = sm.to_dot()

    assert "digraph StateMachine" in dot
    assert '"PENDING" [shape=doublecircle];' in dot
    assert '"RUNNING" [shape=circle];' in dot
    assert '"FAILED" [shape=circle];' in dot
    assert '"COMPLETED" [shape=circle];' in dot
    assert '"PENDING" -> "RUNNING" [label="START"];' in dot
    assert '"RUNNING" -> "COMPLETED" [label="SUCCESS"];' in dot
    assert '"RUNNING" -> "FAILED" [label="FAIL"];' in dot
    assert '"FAILED" -> "RUNNING" [label="RETRY"];' in dot
    assert '"*" [shape=plaintext];' in dot
    assert '"*" -> "FAILED" [label="CANCEL"];' in dot


def test_unhandled_event_raises() -> None:
    sm = StateMachine[TaskState, TaskEvent, TaskContext](
        TaskState.PENDING,
        TaskContext(task_id="none"),
    )
    with pytest.raises(InvalidTransitionError, match="PENDING"):
        sm.send(TaskEvent.SUCCESS)


def test_self_loop_still_runs_exit_enter_and_on_state_change() -> None:
    log: list[str] = []

    sm = StateMachine[TaskState, TaskEvent, TaskContext](
        TaskState.RUNNING,
        TaskContext(task_id="loop"),
        on_state_change=lambda f, t, e, c: log.append(f"change:{f.name}->{t.name}"),
    )
    sm.add_transition(TaskState.RUNNING, TaskEvent.RETRY, TaskState.RUNNING)
    sm.on_exit(TaskState.RUNNING, lambda c, e: log.append("exit"))
    sm.on_enter(TaskState.RUNNING, lambda c, e: log.append("enter"))

    sm.send(TaskEvent.RETRY)
    assert log == ["exit", "enter", "change:RUNNING->RUNNING"]
    assert sm.current_state is TaskState.RUNNING


def test_fluent_api_returns_self() -> None:
    sm = StateMachine[TaskState, TaskEvent, TaskContext](
        TaskState.PENDING,
        TaskContext(task_id="fluent"),
    )
    result = (
        sm.add_transition(TaskState.PENDING, TaskEvent.START, TaskState.RUNNING)
        .on_enter(TaskState.RUNNING, lambda c, e: None)
        .on_exit(TaskState.PENDING, lambda c, e: None)
    )
    assert result is sm


def test_empty_source_raises() -> None:
    sm = StateMachine[TaskState, TaskEvent, TaskContext](
        TaskState.PENDING,
        TaskContext(task_id="empty"),
    )
    with pytest.raises(ValueError, match="at least one source"):
        sm.add_transition([], TaskEvent.START, TaskState.RUNNING)
