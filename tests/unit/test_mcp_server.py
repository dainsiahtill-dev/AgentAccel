from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path

import accel.mcp_server as mcp_server


def test_create_server_registers_core_tools_and_resources() -> None:
    server = mcp_server.create_server()
    tools = asyncio.run(server.get_tools())
    resources = asyncio.run(server.get_resources())
    templates = asyncio.run(server.get_resource_templates())

    tool_names = set(tools.keys())
    resource_uris = set(resources.keys())
    template_uris = set(templates.keys())

    assert "accel_index_build" in tool_names
    assert "accel_index_update" in tool_names
    assert "accel_context" in tool_names
    assert "accel_verify" in tool_names
    assert "agent-accel://status" in resource_uris
    assert "agent-accel://template/{kind}" in template_uris


def test_tool_context_requires_task() -> None:
    try:
        mcp_server._tool_context(task="   ")
    except ValueError as exc:
        assert str(exc) == "task is required"
    else:
        raise AssertionError("expected ValueError for empty task")


def test_tool_context_builds_pack_for_temp_project(tmp_path: Path) -> None:
    project_dir = tmp_path / "sample_project"
    (project_dir / "src").mkdir(parents=True)
    (project_dir / "src" / "sample.py").write_text(
        "def add(a: int, b: int) -> int:\n    return a + b\n",
        encoding="utf-8",
    )
    (project_dir / "accel.yaml").write_text(
        json.dumps(
            {
                "verify": {"python": ["python -c \"print('ok')\""], "node": []},
                "index": {"include": ["src/**"], "exclude": [], "max_file_mb": 2},
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    build = mcp_server._tool_index_build(project=str(project_dir), full=True)
    assert build["status"] == "ok"

    context = mcp_server._tool_context(
        project=str(project_dir),
        task="Summarize sample add implementation",
        changed_files=["src/sample.py"],
    )
    assert context["status"] == "ok"
    assert Path(context["out"]).is_file()


def test_tool_context_budget_and_list_string_compat(tmp_path: Path) -> None:
    project_dir = tmp_path / "context_compat_project"
    (project_dir / "src").mkdir(parents=True)
    (project_dir / "src" / "sample.py").write_text(
        "def add(a: int, b: int) -> int:\n    return a + b\n",
        encoding="utf-8",
    )
    (project_dir / "accel.yaml").write_text(
        json.dumps(
            {
                "verify": {"python": ["python -c \"print('ok')\""], "node": []},
                "index": {"include": ["src/**"], "exclude": [], "max_file_mb": 2},
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    mcp_server._tool_index_build(project=str(project_dir), full=True)
    context = mcp_server._tool_context(
        project=str(project_dir),
        task="Summarize sample add implementation",
        changed_files="src/sample.py,src/other.py",
        hints='["focus:add","risk:low"]',
        budget="small",
    )
    assert context["status"] == "ok"
    assert Path(context["out"]).is_file()


def test_tool_verify_runs_with_evidence_mode(tmp_path: Path) -> None:
    project_dir = tmp_path / "verify_project"
    (project_dir / "src").mkdir(parents=True)
    (project_dir / "src" / "sample.py").write_text("print('ok')\n", encoding="utf-8")
    (project_dir / "accel.yaml").write_text(
        json.dumps(
            {
                "verify": {"python": ["python -c \"print('ok')\""], "node": []},
                "index": {"include": ["src/**"], "exclude": [], "max_file_mb": 2},
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    mcp_server._tool_index_build(project=str(project_dir), full=True)
    result = mcp_server._tool_verify(
        project=str(project_dir),
        changed_files=["src/sample.py"],
        evidence_run=True,
        per_command_timeout_seconds=10,
    )

    assert result["status"] in {"success", "degraded"}
    assert int(result["exit_code"]) in {0, 2}


def test_verify_starts_async_by_default(tmp_path: Path, monkeypatch) -> None:
    project_dir = tmp_path / "sync_wait_project"
    project_dir.mkdir(parents=True)

    jm = mcp_server.JobManager()
    jm._jobs.clear()  # type: ignore[attr-defined]

    monkeypatch.setattr(mcp_server, "_sync_verify_wait_seconds", 1)
    monkeypatch.setattr(mcp_server, "_sync_verify_poll_seconds", 0.05)
    monkeypatch.setattr(
        mcp_server,
        "resolve_effective_config",
        lambda project_dir, cli_overrides=None: {
            "runtime": {"accel_home": str(tmp_path / ".accel-home")}
        },
    )

    def fake_run_verify_with_callback(project_dir, config, changed_files=None, callback=None):
        time.sleep(1.4)
        return {
            "status": "success",
            "exit_code": 0,
            "nonce": "fake_nonce",
            "log_path": str(tmp_path / "verify.log"),
            "jsonl_path": str(tmp_path / "verify.jsonl"),
            "commands": [],
            "results": [],
            "degraded": False,
            "fail_fast": False,
            "fail_fast_skipped_commands": [],
            "cache_enabled": False,
            "cache_hits": 0,
            "cache_misses": 0,
        }

    monkeypatch.setattr(mcp_server, "run_verify_with_callback", fake_run_verify_with_callback)

    server = mcp_server.create_server()
    tools = asyncio.run(server.get_tools())
    verify_fn = tools["accel_verify"].fn
    status_fn = tools["accel_verify_status"].fn
    events_fn = tools["accel_verify_events"].fn

    started = time.perf_counter()
    result = verify_fn(project=str(project_dir))
    elapsed = time.perf_counter() - started

    assert elapsed < 1.3
    assert result["status"] == "started"
    job_id = str(result["job_id"])

    seen_events = events_fn(job_id=job_id, since_seq=0)
    assert int(seen_events["count"]) > 0

    final_status = {}
    for _ in range(30):
        final_status = status_fn(job_id=job_id)
        if final_status.get("state") == mcp_server.JobState.COMPLETED:
            break
        time.sleep(0.1)

    assert final_status.get("state") == mcp_server.JobState.COMPLETED


def test_verify_cancel_converges_to_cancelled(tmp_path: Path, monkeypatch) -> None:
    project_dir = tmp_path / "cancel_project"
    project_dir.mkdir(parents=True)

    jm = mcp_server.JobManager()
    jm._jobs.clear()  # type: ignore[attr-defined]

    monkeypatch.setattr(
        mcp_server,
        "resolve_effective_config",
        lambda project_dir, cli_overrides=None: {
            "runtime": {"accel_home": str(tmp_path / ".accel-home")}
        },
    )

    def fake_run_verify_with_callback(project_dir, config, changed_files=None, callback=None):
        time.sleep(2.0)
        return {
            "status": "success",
            "exit_code": 0,
            "nonce": "cancel_nonce",
            "log_path": str(tmp_path / "verify_cancel.log"),
            "jsonl_path": str(tmp_path / "verify_cancel.jsonl"),
            "commands": [],
            "results": [],
            "degraded": False,
            "fail_fast": False,
            "fail_fast_skipped_commands": [],
            "cache_enabled": False,
            "cache_hits": 0,
            "cache_misses": 0,
        }

    monkeypatch.setattr(mcp_server, "run_verify_with_callback", fake_run_verify_with_callback)

    server = mcp_server.create_server()
    tools = asyncio.run(server.get_tools())
    start_fn = tools["accel_verify_start"].fn
    cancel_fn = tools["accel_verify_cancel"].fn
    status_fn = tools["accel_verify_status"].fn

    started = start_fn(project=str(project_dir))
    job_id = str(started["job_id"])
    cancel = cancel_fn(job_id=job_id)

    assert bool(cancel["cancelled"]) is True
    assert cancel["status"] == mcp_server.JobState.CANCELLED

    status = status_fn(job_id=job_id)
    assert status["state"] == mcp_server.JobState.CANCELLED


def test_sync_verify_returns_running_when_wait_window_exceeded(tmp_path: Path, monkeypatch) -> None:
    project_dir = tmp_path / "sync_fast_project"
    project_dir.mkdir(parents=True)

    jm = mcp_server.JobManager()
    jm._jobs.clear()  # type: ignore[attr-defined]

    monkeypatch.setattr(mcp_server, "_sync_verify_wait_seconds", 1)
    monkeypatch.setattr(mcp_server, "_sync_verify_poll_seconds", 0.05)
    monkeypatch.setattr(
        mcp_server,
        "resolve_effective_config",
        lambda project_dir, cli_overrides=None: {
            "runtime": {"accel_home": str(tmp_path / ".accel-home")}
        },
    )

    def fake_run_verify_with_callback(project_dir, config, changed_files=None, callback=None):
        time.sleep(1.4)
        return {
            "status": "success",
            "exit_code": 0,
            "nonce": "fake_nonce",
            "log_path": str(tmp_path / "verify.log"),
            "jsonl_path": str(tmp_path / "verify.jsonl"),
            "commands": [],
            "results": [],
            "degraded": False,
            "fail_fast": False,
            "fail_fast_skipped_commands": [],
            "cache_enabled": False,
            "cache_hits": 0,
            "cache_misses": 0,
        }

    monkeypatch.setattr(mcp_server, "run_verify_with_callback", fake_run_verify_with_callback)

    server = mcp_server.create_server()
    tools = asyncio.run(server.get_tools())
    verify_fn = tools["accel_verify"].fn
    status_fn = tools["accel_verify_status"].fn

    started = time.perf_counter()
    result = verify_fn(project=str(project_dir), wait_for_completion=True)
    elapsed = time.perf_counter() - started

    assert elapsed < 1.3
    assert result["status"] == "running"
    assert bool(result["timed_out"]) is True
    job_id = str(result["job_id"])

    final_status = {}
    for _ in range(30):
        final_status = status_fn(job_id=job_id)
        if final_status.get("state") == mcp_server.JobState.COMPLETED:
            break
        time.sleep(0.1)

    assert final_status.get("state") == mcp_server.JobState.COMPLETED


def test_sync_verify_returns_completed_result_for_fast_job(tmp_path: Path, monkeypatch) -> None:
    project_dir = tmp_path / "sync_fast_project"
    project_dir.mkdir(parents=True)

    jm = mcp_server.JobManager()
    jm._jobs.clear()  # type: ignore[attr-defined]

    monkeypatch.setattr(mcp_server, "_sync_verify_wait_seconds", 5)
    monkeypatch.setattr(mcp_server, "_sync_verify_poll_seconds", 0.05)
    monkeypatch.setattr(
        mcp_server,
        "resolve_effective_config",
        lambda project_dir, cli_overrides=None: {
            "runtime": {"accel_home": str(tmp_path / ".accel-home")}
        },
    )

    expected = {
        "status": "success",
        "exit_code": 0,
        "nonce": "fast_nonce",
        "log_path": str(tmp_path / "verify_fast.log"),
        "jsonl_path": str(tmp_path / "verify_fast.jsonl"),
        "commands": [],
        "results": [],
        "degraded": False,
        "fail_fast": False,
        "fail_fast_skipped_commands": [],
        "cache_enabled": False,
        "cache_hits": 0,
        "cache_misses": 0,
    }

    def fake_run_verify_with_callback(project_dir, config, changed_files=None, callback=None):
        return dict(expected)

    monkeypatch.setattr(mcp_server, "run_verify_with_callback", fake_run_verify_with_callback)

    server = mcp_server.create_server()
    tools = asyncio.run(server.get_tools())
    verify_fn = tools["accel_verify"].fn
    result = verify_fn(project=str(project_dir), wait_for_completion=True)

    assert result["status"] == "success"
    assert int(result["exit_code"]) == 0
    assert result["nonce"] == "fast_nonce"
    assert "job_id" in result
