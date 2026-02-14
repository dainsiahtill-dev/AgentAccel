from __future__ import annotations

import threading
from pathlib import Path
from typing import Any, cast

from accel.indexers import discovery
from accel.storage.cache import read_json
from accel.verify.job_manager import JobManager, JobState


def test_job_manager_singleton_initialized_in_new_thread_safe() -> None:
    original_instance = JobManager._instance
    original_instance_lock = JobManager._instance_lock
    JobManager._instance = None
    JobManager._instance_lock = threading.Lock()
    try:
        instances: list[JobManager] = []
        errors: list[Exception] = []
        barrier = threading.Barrier(16)

        def _build_instance() -> None:
            try:
                barrier.wait()
                manager = JobManager()
                _ = manager._jobs
                _ = manager._lock
                _ = manager._initialized
                instances.append(manager)
            except Exception as exc:  # pragma: no cover - asserted in test body
                errors.append(exc)

        threads = [threading.Thread(target=_build_instance) for _ in range(16)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        assert not errors
        assert len(instances) == 16
        first = instances[0]
        assert all(item is first for item in instances)
        assert getattr(first, "_initialized", False) is True
        first.create_job(prefix="smoke")
        assert JobManager() is first
        assert len(first.get_all_jobs()) == 1
    finally:
        JobManager._instance = original_instance
        JobManager._instance_lock = original_instance_lock


def test_collect_source_files_auto_retry_depth_guard(
    monkeypatch, tmp_path: Path
) -> None:
    logs: list[str] = []

    def _sticky_include_patterns(value: object, fallback: list[str]) -> list[str]:
        if fallback == ["**/*"]:
            # Keep include constrained so auto fallback would recurse forever
            # without depth guard.
            return ["src/**"]
        if fallback == []:
            return []
        return [str(item).strip() for item in list(value or []) if str(item).strip()]

    monkeypatch.setattr(discovery, "_normalize_patterns", _sticky_include_patterns)
    monkeypatch.setattr(discovery, "_log_deadlock_info", logs.append)

    files = discovery.collect_source_files(
        tmp_path,
        {
            "index": {
                "scope_mode": "auto",
                "include": ["src/**"],
                "exclude": [],
                "scan_timeout_seconds": 1,
                "max_files_to_scan": 10,
            }
        },
    )

    assert files == []
    assert any("retry depth exceeded" in message.lower() for message in logs)


def test_job_manager_cancel_job_uses_job_lock() -> None:
    class _ProbeLock:
        def __init__(self) -> None:
            self.enter_count = 0
            self._inner = threading.Lock()

        def __enter__(self) -> "_ProbeLock":
            self.enter_count += 1
            self._inner.acquire()
            return self

        def __exit__(self, exc_type, exc, tb) -> bool:
            self._inner.release()
            return False

    manager = JobManager()
    job = manager.create_job(prefix="cancel-lock")
    probe = _ProbeLock()
    cast(Any, job)._lock = probe

    assert manager.cancel_job(job.job_id) is True
    assert probe.enter_count == 1
    assert job.state == JobState.CANCELLING


def test_read_json_invalid_payload_returns_empty_dict(tmp_path: Path) -> None:
    bad_json = tmp_path / "bad.json"
    bad_json.write_text("{", encoding="utf-8")
    assert read_json(bad_json) == {}

    non_dict_json = tmp_path / "list.json"
    non_dict_json.write_text("[1,2,3]", encoding="utf-8")
    assert read_json(non_dict_json) == {}
