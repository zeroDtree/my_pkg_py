"""Resumable incremental saver for long-running single-threaded generation jobs.

Each sample is tracked in a SQLite manifest with states:
``pending -> running -> done`` or ``pending -> running -> failed``.
Crashed ``running`` rows, corrupted or missing output files, and (optionally)
``failed`` rows are reset to ``pending`` via recover events. Completed samples
are skipped on resume.
"""

from __future__ import annotations

import hashlib
import json
import os
import pickle
import sqlite3
import tempfile
import time
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass
from enum import Enum, StrEnum
from pathlib import Path
from typing import Any, Protocol, TypeVar

from mlkit.util.state_machine import InvalidTransitionError, StateMachine

PayloadT = TypeVar("PayloadT")


class SaveStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"


class SaveEvent(Enum):
    START = "start"
    SUCCESS = "success"
    FAIL = "fail"
    RECOVER = "recover"
    RESET_FAILED = "reset_failed"


@dataclass
class SampleContext:
    """Runtime data for one sample's lifecycle machine."""

    sample_id: str
    output_path: Path | None = None
    checksum: str | None = None
    error: str | None = None
    meta: dict[str, Any] | None = None


class Serializer(Protocol[PayloadT]):
    def dumps(self, payload: PayloadT) -> bytes: ...

    def loads(self, data: bytes) -> PayloadT: ...


class PickleSerializer(Serializer[Any]):
    """Default serializer using pickle."""

    def dumps(self, payload: Any) -> bytes:
        return pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL)

    def loads(self, data: bytes) -> Any:
        return pickle.loads(data)


@dataclass(frozen=True)
class SaveRecord:
    sample_id: str
    status: SaveStatus
    output_path: str | None
    checksum: str | None
    error: str | None
    meta: dict[str, Any] | None
    updated_at: float

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> SaveRecord:
        meta_raw = row["meta_json"]
        meta = json.loads(meta_raw) if meta_raw else None
        return cls(
            sample_id=row["sample_id"],
            status=SaveStatus(row["status"]),
            output_path=row["output_path"],
            checksum=row["checksum"],
            error=row["error"],
            meta=meta,
            updated_at=float(row["updated_at"]),
        )


@dataclass
class RecoveryReport:
    done_missing_file_to_pending: int = 0
    done_checksum_mismatch_to_pending: int = 0
    failed_to_pending: int = 0
    running_to_pending: int = 0


def _utc_now() -> float:
    return time.time()


def _atomic_write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_str = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    tmp_path = Path(tmp_str)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise
    os.replace(tmp_path, path)


def _unlink_output(ctx: SampleContext, event: SaveEvent) -> None:
    if ctx.output_path is not None:
        Path(ctx.output_path).unlink(missing_ok=True)


def _clear_runtime_fields(ctx: SampleContext, event: SaveEvent) -> None:
    ctx.output_path = None
    ctx.checksum = None
    ctx.error = None


def _clear_output_fields(ctx: SampleContext, event: SaveEvent) -> None:
    ctx.output_path = None
    ctx.checksum = None


def _can_succeed(ctx: SampleContext, event: SaveEvent) -> bool:
    return ctx.output_path is not None and ctx.checksum is not None


def _configure_lifecycle(
    sm: StateMachine[SaveStatus, SaveEvent, SampleContext],
) -> StateMachine[SaveStatus, SaveEvent, SampleContext]:
    return (
        sm.add_transition(SaveStatus.PENDING, SaveEvent.START, SaveStatus.RUNNING)
        .add_transition(
            SaveStatus.RUNNING,
            SaveEvent.SUCCESS,
            SaveStatus.DONE,
            guard=_can_succeed,
        )
        .add_transition(SaveStatus.RUNNING, SaveEvent.FAIL, SaveStatus.FAILED)
        .add_transition(
            SaveStatus.DONE,
            SaveEvent.RECOVER,
            SaveStatus.PENDING,
            actions=[_unlink_output],
        )
        .add_transition(
            SaveStatus.RUNNING,
            SaveEvent.RECOVER,
            SaveStatus.PENDING,
            actions=[_unlink_output],
        )
        .add_transition(SaveStatus.FAILED, SaveEvent.RESET_FAILED, SaveStatus.PENDING)
        .on_enter(SaveStatus.PENDING, _clear_runtime_fields)
        .on_enter(SaveStatus.FAILED, _clear_output_fields)
    )


class ResumableSaver:
    """Track and persist generated samples with resume/skip support.

    Not thread-safe. Designed for single-threaded generation loops where each
    sample is expensive to compute. Status changes are driven by a state
    machine; SQLite is only a persistence hook. Completed samples are skipped
    automatically on re-runs; failed samples can optionally be retried.
    """

    def __init__(
        self,
        root_dir: str | Path,
        *,
        serializer: Serializer[Any] | None = None,
        retry_failed: bool = False,
        auto_recover: bool = True,
    ) -> None:
        self.root_dir = Path(root_dir)
        self.output_dir = self.root_dir / "outputs"
        self.manifest_path = self.root_dir / "manifest.db"
        self.serializer = serializer or PickleSerializer()
        self.retry_failed = retry_failed

        self.root_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self._conn = sqlite3.connect(self.manifest_path)
        self._conn.row_factory = sqlite3.Row
        self._init_schema()

        if auto_recover:
            self.recover_stale_records()

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> ResumableSaver:
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self.close()

    def _init_schema(self) -> None:
        self._conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS records (
                sample_id   TEXT PRIMARY KEY,
                status      TEXT NOT NULL,
                output_path TEXT,
                checksum    TEXT,
                error       TEXT,
                meta_json   TEXT,
                updated_at  REAL NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_records_status ON records(status);
            """
        )
        self._conn.commit()

    def _output_path_for(self, sample_id: str) -> Path:
        shard = sample_id[:2]
        return self.output_dir / shard / f"{sample_id}.pkl"

    def _verify_output_file(self, output_path: Path, checksum: str | None) -> bool:
        if not output_path.is_file():
            return False
        if checksum is None:
            return True
        data = output_path.read_bytes()
        return hashlib.sha256(data).hexdigest() == checksum

    def _sync_to_db(
        self,
        from_s: SaveStatus,
        to_s: SaveStatus,
        evt: SaveEvent,
        ctx: SampleContext,
    ) -> None:
        meta_json = json.dumps(ctx.meta) if ctx.meta is not None else None
        out_path_str = str(ctx.output_path) if ctx.output_path is not None else None
        with self._conn:
            self._conn.execute(
                """
                INSERT OR REPLACE INTO records(
                    sample_id, status, output_path, checksum,
                    error, meta_json, updated_at
                ) VALUES(?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    ctx.sample_id,
                    to_s.value,
                    out_path_str,
                    ctx.checksum,
                    ctx.error,
                    meta_json,
                    _utc_now(),
                ),
            )

    def _machine_for(
        self,
        status: SaveStatus,
        ctx: SampleContext,
        *,
        persist: bool = True,
    ) -> StateMachine[SaveStatus, SaveEvent, SampleContext]:
        sm = StateMachine[SaveStatus, SaveEvent, SampleContext](
            initial_state=status,
            context=ctx,
            on_state_change=self._sync_to_db if persist else None,
        )
        return _configure_lifecycle(sm)

    def _context_from_record(
        self,
        record: SaveRecord | None,
        sample_id: str,
        meta: Mapping[str, Any] | None,
    ) -> SampleContext:
        if record is None:
            return SampleContext(
                sample_id=sample_id,
                meta=dict(meta) if meta is not None else None,
            )
        return SampleContext(
            sample_id=record.sample_id,
            output_path=Path(record.output_path) if record.output_path else None,
            checksum=record.checksum,
            error=record.error,
            meta=dict(meta) if meta is not None else record.meta,
        )

    def _ensure_running(
        self,
        sample_id: str,
        meta: Mapping[str, Any] | None = None,
    ) -> tuple[StateMachine[SaveStatus, SaveEvent, SampleContext], SampleContext]:
        record = self.get_record(sample_id)
        ctx = self._context_from_record(record, sample_id, meta)
        status = record.status if record is not None else SaveStatus.PENDING
        sm = self._machine_for(status, ctx)
        if sm.current_state is SaveStatus.DONE:
            raise InvalidTransitionError(
                f"Cannot start sample_id={sample_id!r} from state '{SaveStatus.DONE.name}'; "
                "reset stale DONE records with recover_stale_records()"
            )
        if sm.current_state is SaveStatus.FAILED:
            sm.send(SaveEvent.RESET_FAILED)
        if sm.current_state is SaveStatus.PENDING:
            sm.send(SaveEvent.START)
        return sm, ctx

    def _write_payload(self, sample_id: str, payload: Any) -> tuple[Path, str]:
        output_path = self._output_path_for(sample_id)
        data = self.serializer.dumps(payload)
        checksum = hashlib.sha256(data).hexdigest()
        _atomic_write_bytes(output_path, data)
        return output_path, checksum

    def is_done(self, sample_id: str) -> bool:
        record = self.get_record(sample_id)
        if record is None or record.status != SaveStatus.DONE or record.output_path is None:
            return False
        return self._verify_output_file(Path(record.output_path), record.checksum)

    def get_record(self, sample_id: str) -> SaveRecord | None:
        row = self._conn.execute(
            "SELECT * FROM records WHERE sample_id = ?",
            (sample_id,),
        ).fetchone()
        return SaveRecord.from_row(row) if row is not None else None

    def register_pending(self, sample_id: str, meta: Mapping[str, Any] | None = None) -> None:
        """Insert a sample as pending if it does not already exist."""
        meta_json = json.dumps(dict(meta)) if meta is not None else None
        self._conn.execute(
            """
            INSERT OR IGNORE INTO records(
                sample_id, status, output_path, checksum,
                error, meta_json, updated_at
            ) VALUES(?, ?, NULL, NULL, NULL, ?, ?)
            """,
            (sample_id, SaveStatus.PENDING.value, meta_json, _utc_now()),
        )
        self._conn.commit()

    def run_sample(
        self,
        sample_id: str,
        fn: Callable[[], PayloadT],
        *,
        meta: Mapping[str, Any] | None = None,
    ) -> PayloadT:
        """Skip if done; otherwise compute, persist, and return the payload.

        Marks the sample ``running`` before ``fn`` so a crash can be recovered.
        If ``fn`` raises, the failure is recorded and the exception re-raised.
        """
        if self.is_done(sample_id):
            return self.load(sample_id)  # type: ignore[return-value]
        sm, ctx = self._ensure_running(sample_id, meta)
        try:
            payload = fn()
            output_path, checksum = self._write_payload(sample_id, payload)
            ctx.output_path = output_path
            ctx.checksum = checksum
            ctx.error = None
            try:
                sm.send(SaveEvent.SUCCESS)
            except Exception:
                output_path.unlink(missing_ok=True)
                raise
            return payload  # type: ignore[return-value]
        except Exception as exc:
            if sm.current_state is SaveStatus.RUNNING:
                ctx.error = str(exc)
                sm.send(SaveEvent.FAIL)
            raise

    def load(self, sample_id: str) -> Any:
        """Load a previously saved payload for a done sample."""
        record = self.get_record(sample_id)
        if record is None or record.status != SaveStatus.DONE or record.output_path is None:
            raise FileNotFoundError(f"No completed output found for sample_id={sample_id!r}")
        output_path = Path(record.output_path)
        if not self._verify_output_file(output_path, record.checksum):
            raise FileNotFoundError(f"Output file missing or invalid for sample_id={sample_id!r}")
        return self.serializer.loads(output_path.read_bytes())

    def recover_stale_records(self) -> RecoveryReport:
        """Reset inconsistent records to pending so they are retried.

        - ``running`` records left by a crashed process are reset to ``pending``.
        - ``done`` records with a missing output file are reset to ``pending``.
        - ``done`` records whose checksum does not match are reset to ``pending``
          and the corrupted file is deleted.
        - ``failed`` records are reset to ``pending`` when ``retry_failed=True``.
        """
        report = RecoveryReport()
        for rec in self.list_records():
            ctx = self._context_from_record(rec, rec.sample_id, None)
            sm = self._machine_for(rec.status, ctx)

            if sm.current_state is SaveStatus.RUNNING:
                sm.send(SaveEvent.RECOVER)
                report.running_to_pending += 1
            elif sm.current_state is SaveStatus.DONE:
                out_path = Path(rec.output_path) if rec.output_path else None
                if out_path is None or not out_path.is_file():
                    sm.send(SaveEvent.RECOVER)
                    report.done_missing_file_to_pending += 1
                elif rec.checksum is not None:
                    actual = hashlib.sha256(out_path.read_bytes()).hexdigest()
                    if actual != rec.checksum:
                        sm.send(SaveEvent.RECOVER)
                        report.done_checksum_mismatch_to_pending += 1
            elif sm.current_state is SaveStatus.FAILED and self.retry_failed:
                sm.send(SaveEvent.RESET_FAILED)
                report.failed_to_pending += 1

        return report

    def list_records(self, status: SaveStatus | None = None) -> list[SaveRecord]:
        if status is None:
            rows = self._conn.execute("SELECT * FROM records ORDER BY updated_at ASC").fetchall()
        else:
            rows = self._conn.execute(
                "SELECT * FROM records WHERE status = ? ORDER BY updated_at ASC",
                (status.value,),
            ).fetchall()
        return [SaveRecord.from_row(row) for row in rows]

    def iter_todo(self) -> Iterator[str]:
        """Yield sample IDs that still need processing (pending, and optionally failed)."""
        statuses = [SaveStatus.PENDING.value]
        if self.retry_failed:
            statuses.append(SaveStatus.FAILED.value)
        placeholders = ",".join("?" * len(statuses))
        rows = self._conn.execute(
            f"SELECT sample_id FROM records WHERE status IN ({placeholders}) ORDER BY updated_at ASC",
            statuses,
        ).fetchall()
        for row in rows:
            yield row["sample_id"]

    def stats(self) -> dict[str, int]:
        rows = self._conn.execute("SELECT status, COUNT(*) AS count FROM records GROUP BY status").fetchall()
        counts = {status.value: 0 for status in SaveStatus}
        for row in rows:
            counts[row["status"]] = int(row["count"])
        counts["total"] = sum(counts.values())
        return counts

    def lifecycle_dot(self) -> str:
        """Return a Graphviz DOT graph of the sample lifecycle topology."""
        sm = self._machine_for(SaveStatus.PENDING, SampleContext(sample_id="_"), persist=False)
        return sm.to_dot()



if __name__ == "__main__":

    def build_sample_id(*parts: str | int, config_hash: str | None = None) -> str:
        """Build a stable filesystem-safe sample identifier from key parts."""
        safe_parts = [
            str(part).replace("/", "_").replace("\\", "_").replace("::", "__")
            for part in parts
        ]
        normalized = "::".join(safe_parts)
        if config_hash:
            normalized = f"{normalized}::{config_hash}"
        return normalized
    # Two usage patterns are demonstrated below:
    #
    # Example 1 (register_pending + iter_todo): preferred for production batch jobs.
    # Pre-registering all sample IDs gives upfront totals via stats()/SQLite,
    # pairs cleanly with auto_recover + iter_todo for stale/failed retries, and
    # matches a task-board pattern that scales to multi-worker setups later.
    #
    # Example 2 (direct run_sample loop): shorter code for demos and quick local
    # experiments when you already have the full ID list and do not need a
    # persisted job queue or progress visibility before processing starts.

    root = Path(".resumable_saver_demo")
    config_hash = "demo_v1"
    sample_ids = [build_sample_id("pocket_a", i, config_hash=config_hash) for i in range(5)]

    with ResumableSaver(root) as saver:
        for sid in sample_ids:
            saver.register_pending(sid, meta={"pocket": "pocket_a"})

        for sid in saver.iter_todo():
            result = saver.run_sample(
                sid,
                lambda s=sid: {"sample_id": s, "coords": [1.0, 2.0, 3.0]},
                meta={"tag": "demo"},
            )
            print(f"  saved {sid[:8]}... -> {result}")

        stats = saver.stats()
        print(f"\nStats after run: {stats}")

        first_done = next((sid for sid in sample_ids if saver.is_done(sid)), None)
        if first_done is not None:
            loaded = saver.load(first_done)
            print(f"Loaded result for {first_done[:8]}...: {loaded}")

    # Example 2: direct loop when you already have the full ID list and do not
    # need upfront manifest stats or iter_todo(). For a persisted job queue, use
    # register_pending + iter_todo() as in Example 1 above.
    print("\n=== Example 2: direct loop (no pre-registration) ===")
    root_direct = Path(".resumable_saver_demo_direct")
    sample_ids_direct = [build_sample_id("pocket_b", i, config_hash=config_hash) for i in range(5)]

    with ResumableSaver(root_direct) as saver:
        for sid in sample_ids_direct:
            result = saver.run_sample(
                sid,
                lambda s=sid: {"sample_id": s, "coords": [1.0, 2.0, 3.0]},
                meta={"tag": "demo_direct"},
            )
            print(f"  saved {sid[:8]}... -> {result}")

        stats = saver.stats()
        print(f"\nStats after run: {stats}")
