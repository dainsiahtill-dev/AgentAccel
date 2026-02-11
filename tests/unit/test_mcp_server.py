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
    assert int(context["estimated_tokens"]) > 0
    assert float(context["compression_ratio"]) > 0.0
    assert context["budget_source"] == "user"
    assert context["budget_preset"] == "small"
    assert context["changed_files_source"] == "user"
    assert int(context["selected_tests_count"]) >= 0
    assert int(context["selected_checks_count"]) >= 1
    token_estimator = context.get("token_estimator", {})
    assert isinstance(token_estimator, dict)
    assert str(token_estimator.get("backend_used", "")) != ""
    assert float(token_estimator.get("calibration", 0.0)) > 0.0


def test_tool_context_auto_budget_and_git_changed_files(tmp_path: Path, monkeypatch) -> None:
    project_dir = tmp_path / "context_auto_project"
    project_dir.mkdir(parents=True)
    out_path = project_dir / "context_auto.json"

    monkeypatch.setattr(
        mcp_server,
        "resolve_effective_config",
        lambda project_dir: {"runtime": {"accel_home": str(tmp_path / ".accel-home")}},
    )
    monkeypatch.setattr(
        mcp_server,
        "_discover_changed_files_from_git",
        lambda project_dir, limit=200: ["src/auto_changed.py"],
    )

    captured: dict[str, object] = {}

    def fake_compile_context_pack(
        project_dir,
        config,
        task,
        changed_files=None,
        hints=None,
        previous_attempt_feedback=None,
        budget_override=None,
    ):
        captured["changed_files"] = list(changed_files or [])
        captured["budget_override"] = dict(budget_override or {})
        return {
            "version": 1,
            "task": task,
            "generated_at": "2026-02-12T00:00:00+00:00",
            "budget": {
                "max_chars": int((budget_override or {}).get("max_chars", 0)),
                "max_snippets": int((budget_override or {}).get("max_snippets", 0)),
                "top_n_files": int((budget_override or {}).get("top_n_files", 0)),
            },
            "top_files": [{"path": "src/auto_changed.py", "score": 1.0, "reasons": ["changed_file"], "signals": []}],
            "snippets": [{"path": "src/auto_changed.py", "start_line": 1, "end_line": 2, "symbol": "", "reason": "", "content": "print('ok')"}],
            "verify_plan": {"target_tests": [], "target_checks": ["pytest -q"]},
            "meta": {"source_chars_est": 50000},
        }

    monkeypatch.setattr(mcp_server, "compile_context_pack", fake_compile_context_pack)

    result = mcp_server._tool_context(
        project=str(project_dir),
        task="quick typo fix in one file",
        changed_files=None,
        hints=None,
        out=str(out_path),
        budget=None,
    )

    assert result["status"] == "ok"
    assert Path(result["out"]).is_file()
    assert result["changed_files_source"] == "git_auto"
    assert int(result["changed_files_count"]) == 1
    assert result["budget_source"] == "auto"
    assert result["budget_preset"] == "tiny"
    assert int(result["estimated_tokens"]) > 0
    assert float(result["compression_ratio"]) < 1.0
    assert captured["changed_files"] == ["src/auto_changed.py"]
    assert isinstance(captured["budget_override"], dict) and int(captured["budget_override"]["max_chars"]) > 0


def test_tool_context_accepts_non_string_budget_value(tmp_path: Path, monkeypatch) -> None:
    project_dir = tmp_path / "context_budget_compat"
    project_dir.mkdir(parents=True)

    monkeypatch.setattr(
        mcp_server,
        "resolve_effective_config",
        lambda project_dir: {"runtime": {"accel_home": str(tmp_path / ".accel-home")}},
    )
    monkeypatch.setattr(mcp_server, "_discover_changed_files_from_git", lambda project_dir, limit=200: [])
    monkeypatch.setattr(
        mcp_server,
        "compile_context_pack",
        lambda **kwargs: {
            "version": 1,
            "task": str(kwargs.get("task", "")),
            "generated_at": "2026-02-12T00:00:00+00:00",
            "budget": {"max_chars": 6000, "max_snippets": 16, "top_n_files": 6},
            "top_files": [],
            "snippets": [],
            "verify_plan": {"target_tests": [], "target_checks": []},
            "meta": {"source_chars_est": 6000},
        },
    )

    result = mcp_server._tool_context(
        project=str(project_dir),
        task="quick docs check",
        changed_files="src/notes.md",
        hints=None,
        budget=12345,
    )
    assert result["status"] == "ok"
    assert result["budget_source"] == "auto"
    assert result["budget_preset"] in {"tiny", "small", "medium"}


def test_tool_context_token_estimator_calibration_metadata(tmp_path: Path, monkeypatch) -> None:
    project_dir = tmp_path / "context_token_calibration_project"
    project_dir.mkdir(parents=True)
    out_path = project_dir / "context_token_calibration.json"

    monkeypatch.setattr(
        mcp_server,
        "resolve_effective_config",
        lambda project_dir: {
            "runtime": {
                "accel_home": str(tmp_path / ".accel-home"),
                "context_require_changed_files": True,
                "token_estimator_backend": "heuristic",
                "token_estimator_calibration": 1.5,
                "token_estimator_fallback_chars_per_token": 4.0,
            }
        },
    )
    monkeypatch.setattr(
        mcp_server,
        "compile_context_pack",
        lambda **kwargs: {
            "version": 1,
            "task": str(kwargs.get("task", "")),
            "generated_at": "2026-02-12T00:00:00+00:00",
            "budget": {"max_chars": 6000, "max_snippets": 16, "top_n_files": 6},
            "top_files": [],
            "snippets": [{"path": "src/sample.py", "content": "print('ok')", "start_line": 1, "end_line": 1}],
            "verify_plan": {"target_tests": [], "target_checks": ["pytest -q"]},
            "meta": {"source_chars_est": 10000},
        },
    )

    result = mcp_server._tool_context(
        project=str(project_dir),
        task="measure token estimator calibration",
        changed_files="src/sample.py",
        out=str(out_path),
    )
    assert result["status"] == "ok"
    token_estimator = result.get("token_estimator", {})
    assert isinstance(token_estimator, dict)
    assert token_estimator.get("backend_requested") == "heuristic"
    assert token_estimator.get("backend_used") == "heuristic"
    assert float(token_estimator.get("calibration", 0.0)) == 1.5
    assert int(token_estimator.get("raw_context_tokens", 0)) >= 1
    assert int(result.get("estimated_tokens", 0)) >= int(token_estimator.get("raw_context_tokens", 0))


def test_tool_context_requires_changed_files_when_git_delta_missing(tmp_path: Path, monkeypatch) -> None:
    project_dir = tmp_path / "context_require_changed_files_project"
    project_dir.mkdir(parents=True)

    monkeypatch.setattr(
        mcp_server,
        "resolve_effective_config",
        lambda project_dir: {
            "runtime": {
                "accel_home": str(tmp_path / ".accel-home"),
                "context_require_changed_files": True,
            }
        },
    )
    monkeypatch.setattr(mcp_server, "_discover_changed_files_from_git", lambda project_dir, limit=200: [])

    try:
        mcp_server._tool_context(project=str(project_dir), task="quick health check", changed_files=None)
    except ValueError as exc:
        assert "changed_files is required" in str(exc)
    else:
        raise AssertionError("expected ValueError when changed_files is missing and git delta is empty")


def test_tool_context_allows_wide_scope_when_require_changed_files_disabled(tmp_path: Path, monkeypatch) -> None:
    project_dir = tmp_path / "context_allow_wide_scope_project"
    project_dir.mkdir(parents=True)

    monkeypatch.setattr(
        mcp_server,
        "resolve_effective_config",
        lambda project_dir: {
            "runtime": {
                "accel_home": str(tmp_path / ".accel-home"),
                "context_require_changed_files": False,
            }
        },
    )
    monkeypatch.setattr(mcp_server, "_discover_changed_files_from_git", lambda project_dir, limit=200: [])
    monkeypatch.setattr(
        mcp_server,
        "compile_context_pack",
        lambda **kwargs: {
            "version": 1,
            "task": str(kwargs.get("task", "")),
            "generated_at": "2026-02-12T00:00:00+00:00",
            "budget": {"max_chars": 6000, "max_snippets": 16, "top_n_files": 6},
            "top_files": [],
            "snippets": [],
            "verify_plan": {"target_tests": [], "target_checks": []},
            "meta": {"source_chars_est": 6000},
        },
    )

    result = mcp_server._tool_context(
        project=str(project_dir),
        task="quick docs check",
        changed_files=None,
        hints=None,
        budget="tiny",
    )
    assert result["status"] == "ok"
    assert result["changed_files_source"] == "none"
    warnings = result.get("warnings", [])
    assert isinstance(warnings, list) and len(warnings) >= 1


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


def test_verify_events_compact_summary_and_tail() -> None:
    jm = mcp_server.JobManager()
    jm._jobs.clear()  # type: ignore[attr-defined]
    job = jm.create_job()
    job.mark_running("running")
    for idx in range(1, 8):
        job.add_event(
            "heartbeat" if idx % 2 == 0 else "progress",
            {"elapsed_sec": float(idx), "state": "running", "stage": "running"},
        )
    job.add_event("job_completed", {"status": "success", "exit_code": 0})
    job.mark_completed(status="success", exit_code=0, result={"status": "success", "exit_code": 0})

    server = mcp_server.create_server()
    tools = asyncio.run(server.get_tools())
    events_fn = tools["accel_verify_events"].fn
    payload = events_fn(job_id=job.job_id, since_seq=0, max_events=3, include_summary=True)

    assert int(payload["count"]) == 3
    assert int(payload["total_available"]) >= 8
    assert bool(payload["truncated"]) is True
    summary = payload.get("summary", {})
    assert isinstance(summary, dict)
    assert summary.get("latest_state") in {mcp_server.JobState.COMPLETED, "running"}
    assert bool(summary.get("terminal_event_seen")) is True
    event_type_counts = summary.get("event_type_counts", {})
    assert isinstance(event_type_counts, dict)
    assert int(event_type_counts.get("job_completed", 0)) >= 1


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


def test_verify_start_accepts_string_runtime_overrides(tmp_path: Path, monkeypatch) -> None:
    project_dir = tmp_path / "verify_string_overrides_project"
    project_dir.mkdir(parents=True)

    jm = mcp_server.JobManager()
    jm._jobs.clear()  # type: ignore[attr-defined]

    monkeypatch.setattr(
        mcp_server,
        "resolve_effective_config",
        lambda project_dir, cli_overrides=None: {
            "runtime": {"accel_home": str(tmp_path / ".accel-home")},
            "verify": {},
        },
    )

    def fake_run_verify_with_callback(project_dir, config, changed_files=None, callback=None):
        return {
            "status": "success",
            "exit_code": 0,
            "nonce": "string_override_nonce",
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
    start_fn = tools["accel_verify_start"].fn
    status_fn = tools["accel_verify_status"].fn

    started = start_fn(
        project=str(project_dir),
        changed_files="src/app.py,src/lib.py",
        evidence_run="true",
        fast_loop="false",
        verify_workers="2",
        per_command_timeout_seconds="10",
        verify_cache_ttl_seconds="300",
        verify_cache_max_entries="200",
    )
    assert started["status"] == "started"

    job_id = str(started["job_id"])
    final_status = {}
    for _ in range(30):
        final_status = status_fn(job_id=job_id)
        if final_status.get("state") == mcp_server.JobState.COMPLETED:
            break
        time.sleep(0.05)
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
