"""Tests for the state-machine-backed resumable saver."""

from __future__ import annotations

from pathlib import Path

import pytest

from mlkit.util.resumable_saver import (
    RecoveryReport,
    ResumableSaver,
    SampleContext,
    SaveEvent,
    SaveStatus,
    build_sample_id,
)
from mlkit.util.state_machine import InvalidTransitionError


def test_run_sample_pending_to_done_and_skips_on_rerun(tmp_path: Path) -> None:
    calls = {"n": 0}

    def build() -> dict[str, int]:
        calls["n"] += 1
        return {"value": calls["n"]}

    with ResumableSaver(tmp_path) as saver:
        first = saver.run_sample("sample-a", build, meta={"tag": "v1"})
        assert first == {"value": 1}
        assert saver.is_done("sample-a")
        record = saver.get_record("sample-a")
        assert record is not None
        assert record.status is SaveStatus.DONE
        assert record.meta == {"tag": "v1"}

        second = saver.run_sample("sample-a", build)
        assert second == {"value": 1}
        assert calls["n"] == 1
        assert saver.load("sample-a") == {"value": 1}
        assert saver.stats()["done"] == 1
        assert saver.stats()["running"] == 0


def test_run_sample_failed_then_explicit_retry(tmp_path: Path) -> None:
    with ResumableSaver(tmp_path) as saver:
        with pytest.raises(RuntimeError, match="boom"):
            saver.run_sample("sample-b", lambda: (_ for _ in ()).throw(RuntimeError("boom")))

        record = saver.get_record("sample-b")
        assert record is not None
        assert record.status is SaveStatus.FAILED
        assert record.error is not None
        assert "boom" in record.error
        assert list(saver.iter_todo()) == []

        result = saver.run_sample("sample-b", lambda: {"ok": True})
        assert result == {"ok": True}
        assert saver.is_done("sample-b")


def test_running_leftover_recovered_to_pending(tmp_path: Path) -> None:
    with ResumableSaver(tmp_path, auto_recover=False) as saver:
        saver.register_pending("sample-c")
        saver._ensure_running("sample-c")
        record = saver.get_record("sample-c")
        assert record is not None
        assert record.status is SaveStatus.RUNNING
        assert saver.stats()["running"] == 1
        assert list(saver.iter_todo()) == []

        report = saver.recover_stale_records()
        assert report.running_to_pending == 1
        recovered = saver.get_record("sample-c")
        assert recovered is not None
        assert recovered.status is SaveStatus.PENDING
        assert list(saver.iter_todo()) == ["sample-c"]


def test_recover_done_missing_file(tmp_path: Path) -> None:
    with ResumableSaver(tmp_path) as saver:
        saver.run_sample("sample-d", lambda: {"n": 1})
        record = saver.get_record("sample-d")
        assert record is not None and record.output_path is not None
        Path(record.output_path).unlink()

        report = saver.recover_stale_records()
        assert report.done_missing_file_to_pending == 1
        pending = saver.get_record("sample-d")
        assert pending is not None
        assert pending.status is SaveStatus.PENDING
        assert pending.output_path is None
        assert pending.checksum is None


def test_recover_done_checksum_mismatch_deletes_file(tmp_path: Path) -> None:
    with ResumableSaver(tmp_path) as saver:
        saver.run_sample("sample-e", lambda: {"n": 1})
        record = saver.get_record("sample-e")
        assert record is not None and record.output_path is not None
        output_path = Path(record.output_path)
        output_path.write_bytes(b"corrupt")

        report = saver.recover_stale_records()
        assert report.done_checksum_mismatch_to_pending == 1
        assert not output_path.is_file()
        pending = saver.get_record("sample-e")
        assert pending is not None
        assert pending.status is SaveStatus.PENDING


def test_retry_failed_policy_vs_iter_todo(tmp_path: Path) -> None:
    with ResumableSaver(tmp_path, retry_failed=False) as saver:
        with pytest.raises(RuntimeError):
            saver.run_sample("sample-f", lambda: (_ for _ in ()).throw(RuntimeError("x")))
        assert list(saver.iter_todo()) == []
        report = saver.recover_stale_records()
        assert report.failed_to_pending == 0
        failed = saver.get_record("sample-f")
        assert failed is not None
        assert failed.status is SaveStatus.FAILED

    with ResumableSaver(tmp_path, retry_failed=True) as saver:
        record = saver.get_record("sample-f")
        assert record is not None
        assert record.status is SaveStatus.PENDING
        assert list(saver.iter_todo()) == ["sample-f"]
        assert saver.stats()["failed"] == 0


def test_illegal_success_from_pending_raises(tmp_path: Path) -> None:
    with ResumableSaver(tmp_path) as saver:
        sm = saver._machine_for(SaveStatus.PENDING, SampleContext(sample_id="sample-g"))
        with pytest.raises(InvalidTransitionError, match="PENDING"):
            sm.send(SaveEvent.SUCCESS)
        assert sm.current_state is SaveStatus.PENDING
        assert saver.get_record("sample-g") is None

        saver.run_sample("sample-g", lambda: {"n": 1})
        done = saver.get_record("sample-g")
        assert done is not None
        sm_done = saver._machine_for(
            done.status,
            saver._context_from_record(done, "sample-g", None),
            persist=False,
        )
        with pytest.raises(InvalidTransitionError, match="DONE"):
            sm_done.send(SaveEvent.FAIL)
        assert saver.is_done("sample-g")


def test_register_pending_does_not_overwrite_existing_row(tmp_path: Path) -> None:
    with ResumableSaver(tmp_path) as saver:
        saver.register_pending("sample-h", meta={"version": 1})
        saver.register_pending("sample-h", meta={"version": 2})
        record = saver.get_record("sample-h")
        assert record is not None
        assert record.status is SaveStatus.PENDING
        assert record.meta == {"version": 1}


def test_stale_done_is_not_recovered_by_run_sample(tmp_path: Path) -> None:
    with ResumableSaver(tmp_path, auto_recover=False) as saver:
        saver.run_sample("sample-i", lambda: {"n": 1})
        record = saver.get_record("sample-i")
        assert record is not None and record.output_path is not None
        Path(record.output_path).unlink()

        with pytest.raises(InvalidTransitionError, match="DONE"):
            saver.run_sample("sample-i", lambda: {"n": 2})
        stuck = saver.get_record("sample-i")
        assert stuck is not None
        assert stuck.status is SaveStatus.DONE

        report = saver.recover_stale_records()
        assert report.done_missing_file_to_pending == 1
        assert saver.run_sample("sample-i", lambda: {"n": 2}) == {"n": 2}
        assert saver.load("sample-i") == {"n": 2}


def test_lifecycle_dot_contains_topology(tmp_path: Path) -> None:
    with ResumableSaver(tmp_path) as saver:
        dot = saver.lifecycle_dot()
        assert saver.get_record("_") is None
    assert "digraph StateMachine" in dot
    assert '"PENDING" -> "RUNNING" [label="START"];' in dot
    assert '"RUNNING" -> "DONE" [label="SUCCESS"];' in dot
    assert '"RUNNING" -> "FAILED" [label="FAIL"];' in dot
    assert '"DONE" -> "PENDING" [label="RECOVER"];' in dot
    assert '"RUNNING" -> "PENDING" [label="RECOVER"];' in dot
    assert '"FAILED" -> "PENDING" [label="RESET_FAILED"];' in dot


def test_recovery_report_defaults() -> None:
    report = RecoveryReport()
    assert report.running_to_pending == 0
    assert report.done_missing_file_to_pending == 0
    assert report.done_checksum_mismatch_to_pending == 0
    assert report.failed_to_pending == 0


def test_build_sample_id_separates_part_boundaries_and_config_hash() -> None:
    assert build_sample_id("a|b", "c") != build_sample_id("a", "b|c")
    assert build_sample_id("x", "y") != build_sample_id("x", config_hash="y")
    assert build_sample_id("pocket", 1, config_hash="v1") == build_sample_id(
        "pocket", 1, config_hash="v1"
    )
    assert build_sample_id("pocket", 1) != build_sample_id("pocket", 1, config_hash="")
