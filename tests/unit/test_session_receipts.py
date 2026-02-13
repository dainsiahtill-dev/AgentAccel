from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

from accel.storage.session_receipts import SessionReceiptError, SessionReceiptStore


def test_session_lifecycle_attach_heartbeat_close(tmp_path: Path) -> None:
    store = SessionReceiptStore(tmp_path / "session_receipts.db")
    opened = store.open_session(run_id="hp_1", owner="codex")
    session_id = str(opened["session_id"])
    lease_id = str(opened["lease_id"])

    readonly = store.attach_session(
        session_id=session_id,
        run_id="hp_1",
        actor="policy",
        readonly=True,
    )
    assert readonly["session_id"] == session_id
    assert readonly["run_id"] == "hp_1"
    assert readonly["status"] in {"open", "active"}

    try:
        store.attach_session(
            session_id=session_id,
            run_id="hp_1",
            actor="policy",
            readonly=False,
        )
    except SessionReceiptError as exc:
        assert exc.code == "E_SESSION_CONFLICT"
    else:
        raise AssertionError("expected lease conflict for writable attach")

    writable = store.attach_session(
        session_id=session_id,
        run_id="hp_1",
        actor="codex",
        readonly=False,
    )
    assert writable["session_id"] == session_id
    assert writable["status"] == "active"
    assert str(writable["lease_owner"]) == "codex"
    updated_lease_id = str(writable["lease_id"])
    assert updated_lease_id
    assert updated_lease_id != lease_id

    heartbeat = store.heartbeat_session(
        session_id=session_id,
        lease_id=updated_lease_id,
    )
    assert heartbeat["session_id"] == session_id
    assert heartbeat["status"] == "active"

    closed = store.close_session(session_id=session_id, final_status="succeeded")
    assert closed["status"] == "succeeded"


def test_receipt_upsert_list_and_args_hash_guard(tmp_path: Path) -> None:
    store = SessionReceiptStore(tmp_path / "session_receipts.db")
    opened = store.open_session(run_id="hp_2", owner="codex")
    session_id = str(opened["session_id"])

    first = store.upsert_receipt(
        job_id="verify_123",
        session_id=session_id,
        run_id="hp_2",
        tool="accel_verify",
        args_hash="a" * 64,
        status="queued",
        evidence_run=True,
        changed_files="a.py,b.py",
    )
    assert first["status"] == "queued"
    assert first["session_id"] == session_id

    second = store.upsert_receipt(
        job_id="verify_123",
        session_id=session_id,
        run_id="hp_2",
        tool="accel_verify",
        args_hash="a" * 64,
        status="running",
        evidence_run=True,
        changed_files="a.py,b.py",
    )
    assert second["status"] == "running"

    rows = store.list_receipts(session_id=session_id, tool="accel_verify", limit=10)
    assert len(rows) == 1
    assert rows[0]["job_id"] == "verify_123"
    assert rows[0]["status"] == "running"

    try:
        store.upsert_receipt(
            job_id="verify_123",
            session_id=session_id,
            run_id="hp_2",
            tool="accel_verify",
            args_hash="b" * 64,
            status="failed",
        )
    except SessionReceiptError as exc:
        assert exc.code == "E_ARGS_HASH_MISMATCH"
    else:
        raise AssertionError("expected args hash mismatch")


def test_recover_expired_running_receipts(tmp_path: Path) -> None:
    store = SessionReceiptStore(tmp_path / "session_receipts.db")
    opened = store.open_session(run_id="hp_3", owner="codex", ttl_seconds=60)
    session_id = str(opened["session_id"])

    store.upsert_receipt(
        job_id="context_123",
        session_id=session_id,
        run_id="hp_3",
        tool="accel_context",
        args_hash="c" * 64,
        status="running",
    )

    # Simulate lease expiry directly in DB.
    expired_ts = (datetime.now(timezone.utc) - timedelta(seconds=120)).isoformat()
    conn = store._connect()  # type: ignore[attr-defined]
    try:
        conn.execute("BEGIN IMMEDIATE")
        conn.execute(
            "UPDATE sessions SET lease_until = ?, updated_at = ? WHERE session_id = ?",
            (expired_ts, expired_ts, session_id),
        )
        conn.commit()
    finally:
        conn.close()

    repaired = store.recover_expired_running_receipts(terminal_status="failed")
    assert repaired == 1
    receipt = store.get_receipt(job_id="context_123")
    assert receipt is not None
    assert receipt["status"] == "failed"
    assert receipt["error_code"] == "E_SESSION_EXPIRED"
