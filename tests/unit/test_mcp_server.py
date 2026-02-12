from __future__ import annotations

import asyncio
import json
import subprocess
import threading
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
    assert "accel_index_status" in tool_names
    assert "accel_index_events" in tool_names
    assert "accel_index_cancel" in tool_names
    assert "accel_context" in tool_names
    assert "accel_context_start" in tool_names
    assert "accel_context_status" in tool_names
    assert "accel_context_events" in tool_names
    assert "accel_context_cancel" in tool_names
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


def test_accel_context_uses_runtime_or_override_timeout(tmp_path: Path, monkeypatch) -> None:
    project_dir = tmp_path / "context_timeout_project"
    project_dir.mkdir(parents=True)

    captured: dict[str, float] = {}

    monkeypatch.setattr(
        mcp_server,
        "resolve_effective_config",
        lambda project_dir: {
            "runtime": {
                "accel_home": str(tmp_path / ".accel-home"),
                "context_rpc_timeout_seconds": 444,
                "semantic_cache_mode": "hybrid",
                "constraint_mode": "warn",
            }
        },
    )
    monkeypatch.setattr(
        mcp_server,
        "_tool_context",
        lambda **kwargs: {"status": "ok", "out": str(project_dir / "ctx.json"), "top_files": 0, "snippets": 0},
    )

    def fake_with_timeout(func, timeout_seconds: int = 300):
        captured["timeout_seconds"] = float(timeout_seconds)

        def _run(*args, **kwargs):
            return func(*args, **kwargs)

        return _run

    monkeypatch.setattr(mcp_server, "_with_timeout", fake_with_timeout)

    server = mcp_server.create_server()
    tools = asyncio.run(server.get_tools())
    context_fn = tools["accel_context"].fn

    result_default = context_fn(project=str(project_dir), task="timeout default")
    assert result_default["status"] == "ok"
    assert float(captured.get("timeout_seconds", 0.0)) == 444.0

    result_override = context_fn(project=str(project_dir), task="timeout override", rpc_timeout_seconds=77)
    assert result_override["status"] == "ok"
    assert float(captured.get("timeout_seconds", 0.0)) == 77.0


def test_accel_context_sync_timeout_falls_back_to_async(tmp_path: Path, monkeypatch) -> None:
    project_dir = tmp_path / "context_sync_timeout_project"
    project_dir.mkdir(parents=True)

    jm = mcp_server.JobManager()
    jm._jobs.clear()  # type: ignore[attr-defined]

    monkeypatch.setattr(
        mcp_server,
        "resolve_effective_config",
        lambda project_dir: {
            "runtime": {
                "accel_home": str(tmp_path / ".accel-home"),
                "semantic_cache_mode": "hybrid",
                "constraint_mode": "warn",
                "context_rpc_timeout_seconds": 300,
                "sync_context_wait_seconds": 1,
                "sync_context_timeout_action": "fallback_async",
            }
        },
    )
    monkeypatch.setattr(mcp_server, "_sync_context_wait_seconds", 1)
    monkeypatch.setattr(mcp_server, "_sync_context_poll_seconds", 0.05)

    def fake_tool_context(**kwargs):
        time.sleep(1.4)
        return {
            "status": "ok",
            "out": str(project_dir / "ctx.json"),
            "out_meta": str(project_dir / "ctx.meta.json"),
            "top_files": 1,
            "snippets": 1,
        }

    monkeypatch.setattr(mcp_server, "_tool_context", fake_tool_context)

    server = mcp_server.create_server()
    tools = asyncio.run(server.get_tools())
    context_fn = tools["accel_context"].fn
    status_fn = tools["accel_context_status"].fn

    started_at = time.perf_counter()
    payload = context_fn(project=str(project_dir), task="timeout fallback")
    elapsed = time.perf_counter() - started_at

    assert elapsed < 1.3
    assert payload["status"] == "running"
    assert bool(payload.get("timed_out")) is True
    assert payload.get("timeout_action") == "fallback_async"
    job_id = str(payload.get("job_id", ""))
    assert job_id.startswith("context_")

    final_status = {}
    for _ in range(40):
        final_status = status_fn(job_id=job_id)
        if final_status.get("state") == mcp_server.JobState.COMPLETED:
            break
        time.sleep(0.05)
    assert final_status.get("state") == mcp_server.JobState.COMPLETED


def test_accel_context_start_status_events_and_cancel(tmp_path: Path, monkeypatch) -> None:
    project_dir = tmp_path / "context_async_project"
    project_dir.mkdir(parents=True)

    jm = mcp_server.JobManager()
    jm._jobs.clear()  # type: ignore[attr-defined]

    call_count = {"n": 0}

    def fake_tool_context(**kwargs):
        call_count["n"] += 1
        if call_count["n"] == 1:
            time.sleep(0.08)
            return {
                "status": "ok",
                "out": str(project_dir / "context_1.json"),
                "out_meta": str(project_dir / "context_1.meta.json"),
                "top_files": 2,
                "snippets": 3,
                "selected_tests_count": 1,
                "selected_checks_count": 1,
            }
        time.sleep(0.5)
        return {
            "status": "ok",
            "out": str(project_dir / "context_2.json"),
            "out_meta": str(project_dir / "context_2.meta.json"),
            "top_files": 1,
            "snippets": 1,
        }

    monkeypatch.setattr(mcp_server, "_tool_context", fake_tool_context)
    monkeypatch.setattr(
        mcp_server,
        "resolve_effective_config",
        lambda project_dir: {
            "runtime": {
                "accel_home": str(tmp_path / ".accel-home"),
                "semantic_cache_mode": "hybrid",
                "constraint_mode": "warn",
            }
        },
    )

    server = mcp_server.create_server()
    tools = asyncio.run(server.get_tools())
    start_fn = tools["accel_context_start"].fn
    status_fn = tools["accel_context_status"].fn
    events_fn = tools["accel_context_events"].fn
    cancel_fn = tools["accel_context_cancel"].fn

    started = start_fn(project=str(project_dir), task="build async context")
    assert started["status"] == "started"
    job_id_1 = str(started["job_id"])
    assert job_id_1.startswith("context_")

    final_status = {}
    for _ in range(40):
        final_status = status_fn(job_id=job_id_1)
        if final_status.get("state") == mcp_server.JobState.COMPLETED:
            break
        time.sleep(0.03)

    assert final_status.get("state") == mcp_server.JobState.COMPLETED
    assert final_status.get("status") == "ok"
    assert int(final_status.get("top_files", 0)) == 2

    events_payload = events_fn(job_id=job_id_1, since_seq=0, max_events=100, include_summary=True)
    assert isinstance(events_payload.get("events"), list)
    summary = events_payload.get("summary", {})
    assert isinstance(summary, dict)
    assert "latest_state" in summary
    assert "event_type_counts" in summary

    started_cancel = start_fn(project=str(project_dir), task="cancel context")
    job_id_2 = str(started_cancel["job_id"])
    cancelled = cancel_fn(job_id=job_id_2)
    assert bool(cancelled.get("cancelled")) is True
    cancelled_status = status_fn(job_id=job_id_2)
    assert cancelled_status.get("state") == mcp_server.JobState.CANCELLED


def test_accel_context_events_summary_prefers_terminal_event_state() -> None:
    jm = mcp_server.JobManager()
    jm._jobs.clear()  # type: ignore[attr-defined]
    job = jm.create_job(prefix="context")
    job.mark_running("running")
    job.add_event("heartbeat", {"state": "running", "stage": "running", "elapsed_sec": 1.0})
    job.add_event("context_completed", {"status": "ok", "out": "context.json", "top_files": 1, "snippets": 1})
    # Simulate transient stale job status before mark_completed is observed.
    job.state = mcp_server.JobState.RUNNING

    server = mcp_server.create_server()
    tools = asyncio.run(server.get_tools())
    events_fn = tools["accel_context_events"].fn

    payload = events_fn(job_id=job.job_id, since_seq=0, max_events=100, include_summary=True)
    summary = payload.get("summary", {})
    assert isinstance(summary, dict)
    assert summary.get("latest_state") == mcp_server.JobState.COMPLETED
    assert summary.get("state_source") == "event_terminal"
    assert bool(summary.get("terminal_event_seen")) is True


def test_accel_context_start_rejects_invalid_enum_inputs_early(tmp_path: Path, monkeypatch) -> None:
    project_dir = tmp_path / "context_enum_validation_project"
    project_dir.mkdir(parents=True)

    monkeypatch.setattr(
        mcp_server,
        "resolve_effective_config",
        lambda project_dir: {
            "runtime": {
                "accel_home": str(tmp_path / ".accel-home"),
                "semantic_cache_mode": "hybrid",
                "constraint_mode": "warn",
            }
        },
    )

    server = mcp_server.create_server()
    tools = asyncio.run(server.get_tools())
    start_fn = tools["accel_context_start"].fn

    try:
        start_fn(project=str(project_dir), task="invalid enum", semantic_cache_mode="invalid")
    except RuntimeError as exc:
        assert "semantic_cache_mode must be one of" in str(exc)
    else:
        raise AssertionError("expected invalid semantic_cache_mode to fail before async job start")


def test_semantic_cache_mode_accepts_read_write_alias() -> None:
    assert mcp_server._resolve_semantic_cache_mode("read_write", default_mode="hybrid") == "hybrid"
    assert mcp_server._resolve_semantic_cache_mode("read-write", default_mode="hybrid") == "hybrid"


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
    assert "semantic_cache_hit" in context
    assert "constraint_repair_count" in context
    assert "compression_rules_applied" in context
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


def test_tool_context_semantic_cache_hit_on_repeat(tmp_path: Path, monkeypatch) -> None:
    project_dir = tmp_path / "context_semantic_cache_project"
    project_dir.mkdir(parents=True)
    out_path = project_dir / "context_semantic_cache.json"

    compile_calls = {"count": 0}

    monkeypatch.setattr(
        mcp_server,
        "resolve_effective_config",
        lambda project_dir: {
            "runtime": {
                "accel_home": str(tmp_path / ".accel-home"),
                "semantic_cache_enabled": True,
                "semantic_cache_mode": "hybrid",
                "semantic_cache_ttl_seconds": 7200,
                "semantic_cache_hybrid_threshold": 0.5,
                "semantic_cache_max_entries": 100,
                "rule_compression_enabled": True,
                "constraint_mode": "warn",
            }
        },
    )
    monkeypatch.setattr(mcp_server, "_discover_changed_files_from_git", lambda project_dir, limit=200: ["src/a.py"])

    def fake_compile_context_pack(**kwargs):
        compile_calls["count"] += 1
        return {
            "version": 1,
            "task": str(kwargs.get("task", "")),
            "generated_at": "2026-02-12T00:00:00+00:00",
            "budget": {"max_chars": 9000, "max_snippets": 16, "top_n_files": 6},
            "top_files": [{"path": "src/a.py", "score": 0.9, "reasons": ["changed_file"], "signals": []}],
            "snippets": [{"path": "src/a.py", "start_line": 1, "end_line": 2, "symbol": "", "reason": "", "content": "print('ok')"}],
            "verify_plan": {"target_tests": [], "target_checks": ["pytest -q"]},
            "meta": {"source_chars_est": 9000, "compression_rules_applied": {"trim_import_block": 1}, "compression_saved_chars": 10},
        }

    monkeypatch.setattr(mcp_server, "compile_context_pack", fake_compile_context_pack)

    first = mcp_server._tool_context(
        project=str(project_dir),
        task="cache semantic context",
        changed_files=["src/a.py"],
        out=str(out_path),
        semantic_cache=True,
    )
    second = mcp_server._tool_context(
        project=str(project_dir),
        task="cache semantic context",
        changed_files=["src/a.py"],
        out=str(out_path),
        semantic_cache=True,
    )

    assert compile_calls["count"] == 1
    assert bool(first.get("semantic_cache_hit")) is False
    assert bool(second.get("semantic_cache_hit")) is True
    assert str(second.get("semantic_cache_mode_used", "")) == "exact"


def test_tool_context_rejects_invalid_semantic_cache_mode(tmp_path: Path, monkeypatch) -> None:
    project_dir = tmp_path / "context_semantic_mode_validation"
    project_dir.mkdir(parents=True)

    monkeypatch.setattr(
        mcp_server,
        "resolve_effective_config",
        lambda project_dir: {"runtime": {"accel_home": str(tmp_path / ".accel-home"), "semantic_cache_mode": "hybrid"}},
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
            "snippets": [],
            "verify_plan": {"target_tests": [], "target_checks": []},
            "meta": {"source_chars_est": 6000},
        },
    )

    try:
        mcp_server._tool_context(
            project=str(project_dir),
            task="validate semantic mode",
            changed_files="src/a.py",
            semantic_cache_mode="invalid_mode",
        )
    except ValueError as exc:
        assert "semantic_cache_mode must be one of" in str(exc)
    else:
        raise AssertionError("expected ValueError for invalid semantic_cache_mode")


def test_tool_context_constraint_mode_enforce_alias_maps_to_strict(tmp_path: Path, monkeypatch) -> None:
    project_dir = tmp_path / "context_constraint_alias"
    project_dir.mkdir(parents=True)

    monkeypatch.setattr(
        mcp_server,
        "resolve_effective_config",
        lambda project_dir: {"runtime": {"accel_home": str(tmp_path / ".accel-home"), "constraint_mode": "warn"}},
    )
    monkeypatch.setattr(
        mcp_server,
        "compile_context_pack",
        lambda **kwargs: {
            "version": 1,
            "task": str(kwargs.get("task", "")),
            "generated_at": "2026-02-12T00:00:00+00:00",
            "budget": {"max_chars": 6000, "max_snippets": 16, "top_n_files": 6},
            "top_files": [{"path": "src/a.py", "score": 1.0, "reasons": ["changed_file"], "signals": []}],
            "snippets": [{"path": "src/a.py", "start_line": 1, "end_line": 1, "symbol": "", "reason": "", "content": "print('ok')"}],
            "verify_plan": {"target_tests": [], "target_checks": ["pytest -q"]},
            "meta": {"source_chars_est": 6000},
        },
    )

    result = mcp_server._tool_context(
        project=str(project_dir),
        task="validate constraint alias",
        changed_files="src/a.py",
        constraint_mode="enforce",
    )

    assert result["status"] == "ok"
    assert result["constraint_mode"] == "strict"
    assert isinstance(result.get("warnings", []), list)
    assert not any("warnings must be list" in str(item) for item in result.get("constraint_warnings", []))


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


def test_tool_context_default_allows_missing_changed_files(tmp_path: Path, monkeypatch) -> None:
    project_dir = tmp_path / "context_default_allow_missing_changed_files_project"
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
        task="default mode should not hard fail",
        changed_files=None,
        budget="tiny",
    )
    assert result["status"] == "ok"
    assert result["changed_files_source"] == "none"
    warnings = result.get("warnings", [])
    assert isinstance(warnings, list) and len(warnings) >= 1


def test_discover_changed_files_from_git_prefers_status_and_parses_untracked(monkeypatch, tmp_path: Path) -> None:
    project_dir = tmp_path / "git_changed_files_project"
    project_dir.mkdir(parents=True)

    calls: list[list[str]] = []

    def fake_run(cmd, capture_output, text, encoding, errors, timeout, check):
        calls.append(list(cmd))
        if "status" in cmd:
            return subprocess.CompletedProcess(
                args=cmd,
                returncode=0,
                stdout=" M src/a.py\n?? src/new.py\nR  old.py -> src/renamed.py\n",
                stderr="",
            )
        return subprocess.CompletedProcess(
            args=cmd,
            returncode=0,
            stdout="src/a.py\n",
            stderr="",
        )

    monkeypatch.setattr(mcp_server.subprocess, "run", fake_run)

    changed = mcp_server._discover_changed_files_from_git(project_dir, limit=10)
    assert changed[:3] == ["src/a.py", "src/new.py", "src/renamed.py"]
    assert len(changed) == len(set(changed))
    assert any("status" in cmd for cmd in calls)


def test_tool_context_uses_manifest_recent_changed_files_fallback(tmp_path: Path, monkeypatch) -> None:
    project_dir = tmp_path / "context_manifest_fallback_project"
    project_dir.mkdir(parents=True)
    accel_home = tmp_path / ".accel-home"

    monkeypatch.setattr(
        mcp_server,
        "resolve_effective_config",
        lambda project_dir: {
            "runtime": {
                "accel_home": str(accel_home),
                "context_require_changed_files": True,
            }
        },
    )
    monkeypatch.setattr(mcp_server, "_discover_changed_files_from_git", lambda project_dir, limit=200: [])

    index_dir = mcp_server.project_paths(accel_home, project_dir)["index"]
    index_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "indexed_files": ["src/a.py", "src/b.py"],
        "changed_files": ["src/b.py"],
    }
    (index_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False), encoding="utf-8")

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
        return {
            "version": 1,
            "task": str(task),
            "generated_at": "2026-02-12T00:00:00+00:00",
            "budget": {"max_chars": 6000, "max_snippets": 16, "top_n_files": 6},
            "top_files": [{"path": "src/b.py", "score": 1.0, "reasons": ["changed_file"], "signals": []}],
            "snippets": [{"path": "src/b.py", "content": "print('ok')", "start_line": 1, "end_line": 1}],
            "verify_plan": {"target_tests": [], "target_checks": ["pytest -q"]},
            "meta": {"source_chars_est": 8000},
        }

    monkeypatch.setattr(mcp_server, "compile_context_pack", fake_compile_context_pack)

    result = mcp_server._tool_context(
        project=str(project_dir),
        task="use fallback manifest changed files",
        changed_files=None,
    )
    assert result["status"] == "ok"
    assert result["changed_files_source"] == "manifest_recent"
    assert result["changed_files_used"] == ["src/b.py"]
    assert captured["changed_files"] == ["src/b.py"]
    warnings = result.get("warnings", [])
    assert isinstance(warnings, list)
    assert any("manifest_recent" in str(item) for item in warnings)


def test_tool_context_strict_changed_files_blocks_non_git_fallback(tmp_path: Path, monkeypatch) -> None:
    project_dir = tmp_path / "context_strict_project"
    project_dir.mkdir(parents=True)

    monkeypatch.setattr(
        mcp_server,
        "resolve_effective_config",
        lambda project_dir: {"runtime": {"accel_home": str(tmp_path / ".accel-home")}},
    )
    monkeypatch.setattr(mcp_server, "_discover_changed_files_from_git", lambda project_dir, limit=200: [])
    monkeypatch.setattr(
        mcp_server,
        "_discover_changed_files_from_index_fallback",
        lambda *args, **kwargs: (["src/fallback.py"], "planner_fallback", 0.42),
    )

    try:
        mcp_server._tool_context(
            project=str(project_dir),
            task="strict mode should reject inferred changed files",
            changed_files=None,
            strict_changed_files=True,
        )
    except ValueError as exc:
        assert "strict_changed_files=true" in str(exc)
    else:
        raise AssertionError("expected strict_changed_files to block non-git fallback")


def test_tool_context_strict_mode_accepts_missing_warnings_field(tmp_path: Path, monkeypatch) -> None:
    project_dir = tmp_path / "context_strict_contract_project"
    project_dir.mkdir(parents=True)

    monkeypatch.setattr(
        mcp_server,
        "resolve_effective_config",
        lambda project_dir: {"runtime": {"accel_home": str(tmp_path / ".accel-home"), "constraint_mode": "warn"}},
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
            "top_files": [{"path": "src/target.py", "score": 1.0, "reasons": ["changed_file"], "signals": []}],
            "snippets": [{"path": "src/target.py", "content": "print('ok')", "start_line": 1, "end_line": 1}],
            "verify_plan": {"target_tests": [], "target_checks": ["pytest -q"]},
            "meta": {"source_chars_est": 6000},
        },
    )

    result = mcp_server._tool_context(
        project=str(project_dir),
        task="strict context payload contract check",
        changed_files=["src/target.py"],
        strict_changed_files=True,
        constraint_mode="strict",
    )
    assert result["status"] == "ok"
    assert int(result.get("constraint_repair_count", 0)) == 0
    warnings = result.get("warnings", [])
    assert isinstance(warnings, list)
    assert not any("context payload warnings must be list" in str(item) for item in warnings)


def test_tool_context_strict_changed_files_prunes_non_changed_items(tmp_path: Path, monkeypatch) -> None:
    project_dir = tmp_path / "context_strict_scope_project"
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
            "top_files": [
                {"path": "src/target.py", "score": 1.0, "reasons": ["changed_file"], "signals": []},
                {"path": "tests/test_target.py", "score": 0.9, "reasons": ["test_relevance"], "signals": []},
            ],
            "snippets": [
                {"path": "src/target.py", "content": "print('ok')", "start_line": 1, "end_line": 1},
                {"path": "tests/test_target.py", "content": "assert True", "start_line": 1, "end_line": 1},
            ],
            "verify_plan": {"target_tests": ["tests/test_target.py"], "target_checks": ["pytest -q"]},
            "meta": {"source_chars_est": 8000},
        },
    )

    result = mcp_server._tool_context(
        project=str(project_dir),
        task="strict changed files must stay in scope",
        changed_files=["src/target.py"],
        strict_changed_files=True,
        include_pack=True,
        constraint_mode="strict",
    )

    assert result["status"] == "ok"
    assert int(result["top_files"]) == 1
    assert int(result["snippets"]) == 1
    assert int(result.get("strict_scope_filtered_top_files", 0)) >= 1
    assert int(result.get("strict_scope_filtered_snippets", 0)) >= 1
    packed = result.get("pack", {})
    assert isinstance(packed, dict)
    top_paths = [str(item.get("path", "")) for item in packed.get("top_files", [])]
    snippet_paths = [str(item.get("path", "")) for item in packed.get("snippets", [])]
    assert top_paths == ["src/target.py"]
    assert snippet_paths == ["src/target.py"]
    warnings = result.get("warnings", [])
    assert any("strict_changed_files pruned" in str(item) for item in warnings)


def test_tool_context_strict_changed_files_injects_changed_file_when_pack_drifted(tmp_path: Path, monkeypatch) -> None:
    project_dir = tmp_path / "context_strict_inject_project"
    (project_dir / "src").mkdir(parents=True)
    (project_dir / "src" / "target.py").write_text("def target() -> int:\n    return 1\n", encoding="utf-8")

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
            "budget": {"max_chars": 6000, "max_snippets": 16, "top_n_files": 6, "per_snippet_max_chars": 1200},
            "top_files": [{"path": "tests/test_target.py", "score": 0.9, "reasons": ["test_relevance"], "signals": []}],
            "snippets": [{"path": "tests/test_target.py", "content": "assert True", "start_line": 1, "end_line": 1}],
            "verify_plan": {"target_tests": ["tests/test_target.py"], "target_checks": ["pytest -q"]},
            "meta": {"source_chars_est": 8000},
        },
    )

    result = mcp_server._tool_context(
        project=str(project_dir),
        task="strict changed file fallback injection",
        changed_files=["src/target.py"],
        strict_changed_files=True,
        include_pack=True,
        constraint_mode="strict",
    )

    assert result["status"] == "ok"
    assert int(result["top_files"]) == 1
    assert int(result["snippets"]) == 1
    assert int(result.get("strict_scope_filtered_top_files", 0)) >= 1
    assert int(result.get("strict_scope_filtered_snippets", 0)) >= 1
    assert int(result.get("strict_scope_injected_top_files", 0)) >= 1
    assert int(result.get("strict_scope_injected_snippets", 0)) >= 1
    packed = result.get("pack", {})
    assert isinstance(packed, dict)
    top_paths = [str(item.get("path", "")) for item in packed.get("top_files", [])]
    snippet_paths = [str(item.get("path", "")) for item in packed.get("snippets", [])]
    assert top_paths == ["src/target.py"]
    assert snippet_paths == ["src/target.py"]
    warnings = result.get("warnings", [])
    assert any("strict_changed_files injected" in str(item) for item in warnings)


def test_tool_context_exposes_fallback_confidence_and_token_reduction_baselines(tmp_path: Path, monkeypatch) -> None:
    project_dir = tmp_path / "context_confidence_project"
    (project_dir / "src").mkdir(parents=True)
    (project_dir / "src" / "fallback.py").write_text("def foo() -> int:\n    return 1\n", encoding="utf-8")
    out_path = project_dir / "context_confidence.json"

    monkeypatch.setattr(
        mcp_server,
        "resolve_effective_config",
        lambda project_dir: {"runtime": {"accel_home": str(tmp_path / ".accel-home")}},
    )
    monkeypatch.setattr(mcp_server, "_discover_changed_files_from_git", lambda project_dir, limit=200: [])
    monkeypatch.setattr(
        mcp_server,
        "_discover_changed_files_from_index_fallback",
        lambda *args, **kwargs: (["src/fallback.py"], "planner_fallback", 0.42),
    )
    monkeypatch.setattr(
        mcp_server,
        "compile_context_pack",
        lambda **kwargs: {
            "version": 1,
            "task": str(kwargs.get("task", "")),
            "generated_at": "2026-02-12T00:00:00+00:00",
            "budget": {"max_chars": 6000, "max_snippets": 16, "top_n_files": 6},
            "top_files": [{"path": "src/fallback.py", "score": 0.9, "reasons": ["fallback"], "signals": []}],
            "snippets": [{"path": "src/fallback.py", "content": "def foo() -> int:\n    return 1", "start_line": 1, "end_line": 2}],
            "verify_plan": {"target_tests": [], "target_checks": ["pytest -q"]},
            "meta": {"source_chars_est": 25000},
        },
    )

    result = mcp_server._tool_context(
        project=str(project_dir),
        task="fallback confidence and token baseline metrics",
        changed_files=None,
        out=str(out_path),
    )

    assert result["status"] == "ok"
    assert result["changed_files_source"] == "planner_fallback"
    assert float(result["fallback_confidence"]) == 0.42
    token_reduction = result.get("token_reduction", {})
    assert isinstance(token_reduction, dict)
    assert "vs_full_index" in token_reduction
    assert "vs_changed_files" in token_reduction
    assert "vs_snippets_only" in token_reduction
    assert "token_reduction_ratio_vs_full_index" in result
    assert "token_reduction_ratio_vs_snippets_only" in result
    assert int(result.get("estimated_changed_files_tokens", 0)) > 0
    warnings = result.get("warnings", [])
    assert isinstance(warnings, list)
    assert any("low" in str(item).lower() for item in warnings)


def test_tool_context_snippets_only_and_include_metadata_controls_output(tmp_path: Path, monkeypatch) -> None:
    project_dir = tmp_path / "context_snippets_only_project"
    project_dir.mkdir(parents=True)
    out_path = project_dir / "context_snippets_only.json"

    monkeypatch.setattr(
        mcp_server,
        "resolve_effective_config",
        lambda project_dir: {"runtime": {"accel_home": str(tmp_path / ".accel-home")}},
    )
    monkeypatch.setattr(mcp_server, "_discover_changed_files_from_git", lambda project_dir, limit=200: ["src/a.py"])
    monkeypatch.setattr(
        mcp_server,
        "compile_context_pack",
        lambda **kwargs: {
            "version": 1,
            "task": str(kwargs.get("task", "")),
            "generated_at": "2026-02-12T00:00:00+00:00",
            "budget": {"max_chars": 6000, "max_snippets": 16, "top_n_files": 6},
            "top_files": [{"path": "src/a.py", "score": 1.0, "reasons": ["changed_file"], "signals": []}],
            "snippets": [{"path": "src/a.py", "content": "print('ok')", "start_line": 1, "end_line": 1}],
            "verify_plan": {"target_tests": [], "target_checks": ["pytest -q"]},
            "meta": {"source_chars_est": 7000, "snippet_chars": 11},
        },
    )

    result = mcp_server._tool_context(
        project=str(project_dir),
        task="return snippets only for quick ask",
        changed_files=None,
        snippets_only=True,
        include_metadata=False,
        include_pack=True,
        out=str(out_path),
    )

    assert result["status"] == "ok"
    assert result["output_mode"] == "snippets_only"
    assert bool(result["include_metadata"]) is False
    assert int(result["top_files"]) == 0
    packed = result.get("pack", {})
    assert isinstance(packed, dict)
    assert "snippets" in packed
    assert "top_files" not in packed
    assert "meta" not in packed

    persisted = json.loads(out_path.read_text(encoding="utf-8"))
    assert "snippets" in persisted
    assert "top_files" not in persisted
    assert "meta" not in persisted

    out_meta = result.get("out_meta")
    assert isinstance(out_meta, str) and out_meta.endswith(".meta.json")
    sidecar = Path(out_meta)
    assert sidecar.is_file()
    sidecar_payload = json.loads(sidecar.read_text(encoding="utf-8"))
    assert sidecar_payload.get("out") == str(out_path)
    estimates = sidecar_payload.get("estimates", {})
    assert isinstance(estimates, dict)
    assert int(estimates.get("estimated_tokens", 0)) > 0
    scope = sidecar_payload.get("scope", {})
    assert isinstance(scope, dict)
    assert scope.get("changed_files_source") == "git_auto"


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


def test_tool_verify_fast_loop_defaults_cache_failed_results(tmp_path: Path, monkeypatch) -> None:
    project_dir = tmp_path / "verify_fast_loop_defaults_project"
    project_dir.mkdir(parents=True)

    captured: dict[str, object] = {}

    def fake_resolve_effective_config(project_dir, cli_overrides=None):
        captured["cli_overrides"] = cli_overrides
        return {"runtime": {"accel_home": str(tmp_path / ".accel-home")}}

    monkeypatch.setattr(mcp_server, "resolve_effective_config", fake_resolve_effective_config)
    monkeypatch.setattr(
        mcp_server,
        "run_verify",
        lambda project_dir, config, changed_files=None: {"status": "success", "exit_code": 0, "commands": [], "results": []},
    )

    result = mcp_server._tool_verify(
        project=str(project_dir),
        changed_files=["src/a.py"],
        fast_loop=True,
        evidence_run=False,
    )

    assert result["status"] == "success"
    overrides = captured.get("cli_overrides")
    assert isinstance(overrides, dict)
    runtime = overrides.get("runtime", {})
    assert isinstance(runtime, dict)
    assert bool(runtime.get("verify_fail_fast")) is True
    assert bool(runtime.get("verify_cache_enabled")) is True
    assert bool(runtime.get("verify_cache_failed_results")) is True


def test_tool_verify_accepts_timebox_and_stall_overrides(tmp_path: Path, monkeypatch) -> None:
    project_dir = tmp_path / "verify_timebox_override_project"
    project_dir.mkdir(parents=True)

    captured: dict[str, object] = {}

    def fake_resolve_effective_config(project_dir, cli_overrides=None):
        captured["cli_overrides"] = cli_overrides
        return {"runtime": {"accel_home": str(tmp_path / ".accel-home")}}

    monkeypatch.setattr(mcp_server, "resolve_effective_config", fake_resolve_effective_config)
    monkeypatch.setattr(
        mcp_server,
        "run_verify",
        lambda project_dir, config, changed_files=None: {"status": "partial", "exit_code": 124, "commands": [], "results": []},
    )

    result = mcp_server._tool_verify(
        project=str(project_dir),
        changed_files=["src/a.py"],
        verify_stall_timeout_seconds=6.0,
        verify_max_wall_time_seconds=45.0,
        verify_auto_cancel_on_stall=True,
    )

    assert result["status"] == "partial"
    overrides = captured.get("cli_overrides")
    assert isinstance(overrides, dict)
    runtime = overrides.get("runtime", {})
    assert isinstance(runtime, dict)
    assert float(runtime.get("verify_stall_timeout_seconds", 0.0)) == 6.0
    assert float(runtime.get("verify_max_wall_time_seconds", 0.0)) == 45.0
    assert bool(runtime.get("verify_auto_cancel_on_stall")) is True


def test_tool_verify_applies_verify_preset_fast_and_full(tmp_path: Path, monkeypatch) -> None:
    project_dir = tmp_path / "verify_preset_project"
    project_dir.mkdir(parents=True)

    captured: dict[str, object] = {}

    def fake_resolve_effective_config(project_dir, cli_overrides=None):
        captured["cli_overrides"] = cli_overrides
        return {"runtime": {"accel_home": str(tmp_path / ".accel-home")}}

    monkeypatch.setattr(mcp_server, "resolve_effective_config", fake_resolve_effective_config)
    monkeypatch.setattr(
        mcp_server,
        "run_verify",
        lambda project_dir, config, changed_files=None: {"status": "success", "exit_code": 0, "commands": [], "results": []},
    )

    result_fast = mcp_server._tool_verify(
        project=str(project_dir),
        changed_files=["src/a.py"],
        verify_preset="fast",
    )
    assert result_fast["status"] == "success"
    overrides_fast = captured.get("cli_overrides")
    assert isinstance(overrides_fast, dict)
    runtime_fast = overrides_fast.get("runtime", {})
    assert isinstance(runtime_fast, dict)
    assert bool(runtime_fast.get("verify_fail_fast")) is True
    assert bool(runtime_fast.get("verify_cache_enabled")) is True
    assert bool(runtime_fast.get("verify_cache_failed_results")) is True

    result_full = mcp_server._tool_verify(
        project=str(project_dir),
        changed_files=["src/a.py"],
        verify_preset="full",
        verify_fail_fast=True,  # explicit override should win
    )
    assert result_full["status"] == "success"
    overrides_full = captured.get("cli_overrides")
    assert isinstance(overrides_full, dict)
    runtime_full = overrides_full.get("runtime", {})
    assert isinstance(runtime_full, dict)
    assert bool(runtime_full.get("verify_cache_enabled")) is False
    assert bool(runtime_full.get("verify_fail_fast")) is True


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
    assert summary.get("latest_state") == mcp_server.JobState.COMPLETED
    assert summary.get("state_source") in {"events", "event_terminal", "status_terminal"}
    assert int(summary.get("constraint_repair_count", 0)) >= 0
    assert bool(summary.get("terminal_event_seen")) is True
    event_type_counts = summary.get("event_type_counts", {})
    assert isinstance(event_type_counts, dict)
    assert int(event_type_counts.get("job_completed", 0)) >= 1


def test_verify_events_summary_prefers_terminal_state_over_late_heartbeat() -> None:
    jm = mcp_server.JobManager()
    jm._jobs.clear()  # type: ignore[attr-defined]
    job = jm.create_job()
    job.mark_running("running")
    job.add_event("heartbeat", {"elapsed_sec": 1.0, "state": "running", "stage": "running"})
    job.add_event("job_cancelled_finalized", {"reason": "user_request"})
    job.mark_cancelled()
    # Simulate late/out-of-order heartbeat carrying stale running state.
    job.add_event("heartbeat", {"elapsed_sec": 2.0, "state": "running", "stage": "running"})

    server = mcp_server.create_server()
    tools = asyncio.run(server.get_tools())
    events_fn = tools["accel_verify_events"].fn
    payload = events_fn(job_id=job.job_id, since_seq=0, max_events=50, include_summary=True)

    summary = payload.get("summary", {})
    assert isinstance(summary, dict)
    assert summary.get("latest_state") == mcp_server.JobState.CANCELLED
    assert summary.get("state_source") == "status_terminal"
    assert int(summary.get("constraint_repair_count", 0)) >= 0
    assert bool(summary.get("terminal_event_seen")) is True


def test_verify_events_include_command_progress_and_output_tail(tmp_path: Path, monkeypatch) -> None:
    project_dir = tmp_path / "events_progress_project"
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
        assert callback is not None
        cmd = "python -c \"print('ok')\""
        callback.on_start("verify_fake", 1)
        callback.on_stage_change("verify_fake", mcp_server.VerifyStage.RUNNING)
        callback.on_command_start("verify_fake", cmd, 0, 1)
        callback.on_heartbeat(
            "verify_fake",
            0.8,
            9.2,
            "running",
            current_command=cmd,
            command_elapsed_sec=0.8,
            command_timeout_sec=10.0,
            command_progress_pct=8.0,
        )
        callback.on_command_complete(
            "verify_fake",
            cmd,
            0,
            1.2,
            completed=1,
            total=1,
            stdout_tail="line1\nline2",
            stderr_tail="",
        )
        callback.on_progress("verify_fake", 1, 1, "")
        callback.on_complete("verify_fake", "success", 0)
        return {
            "status": "success",
            "exit_code": 0,
            "nonce": "events_progress_nonce",
            "log_path": str(tmp_path / "verify_events_progress.log"),
            "jsonl_path": str(tmp_path / "verify_events_progress.jsonl"),
            "commands": [cmd],
            "results": [
                {
                    "command": cmd,
                    "exit_code": 0,
                    "duration_seconds": 1.2,
                    "stdout": "line1\nline2",
                    "stderr": "",
                    "timed_out": False,
                    "cached": False,
                }
            ],
            "degraded": False,
            "fail_fast": False,
            "fail_fast_skipped_commands": [],
            "cache_enabled": False,
            "cache_hits": 0,
            "cache_misses": 1,
        }

    monkeypatch.setattr(mcp_server, "run_verify_with_callback", fake_run_verify_with_callback)

    server = mcp_server.create_server()
    tools = asyncio.run(server.get_tools())
    start_fn = tools["accel_verify_start"].fn
    status_fn = tools["accel_verify_status"].fn
    events_fn = tools["accel_verify_events"].fn

    started = start_fn(project=str(project_dir))
    job_id = str(started["job_id"])

    final_status = {}
    for _ in range(30):
        final_status = status_fn(job_id=job_id)
        if final_status.get("state") == mcp_server.JobState.COMPLETED:
            break
        time.sleep(0.05)
    assert final_status.get("state") == mcp_server.JobState.COMPLETED

    payload = events_fn(job_id=job_id, since_seq=0, max_events=200, include_summary=False)
    events = payload.get("events", [])
    assert isinstance(events, list) and len(events) > 0

    heartbeat_events = [event for event in events if event.get("event") == "heartbeat"]
    assert any("command_progress_pct" in event for event in heartbeat_events)

    command_complete_events = [event for event in events if event.get("event") == "command_complete"]
    assert len(command_complete_events) >= 1
    command_complete = command_complete_events[-1]
    assert command_complete.get("completed") == 1
    assert command_complete.get("total") == 1
    assert "line1" in str(command_complete.get("stdout_tail", ""))


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


def test_verify_cancel_blocks_late_running_heartbeat_events(tmp_path: Path, monkeypatch) -> None:
    project_dir = tmp_path / "cancel_consistency_project"
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

    started_evt = threading.Event()
    continue_evt = threading.Event()

    def fake_run_verify_with_callback(project_dir, config, changed_files=None, callback=None):
        assert callback is not None
        cmd = "python -c \"import time; time.sleep(1)\""
        callback.on_start("verify_fake", 1)
        callback.on_stage_change("verify_fake", mcp_server.VerifyStage.RUNNING)
        callback.on_command_start("verify_fake", cmd, 0, 1)
        started_evt.set()

        # Wait for cancel request, then emit a stale running heartbeat sequence
        # to verify callback-side terminal guards suppress it.
        continue_evt.wait(timeout=2.0)
        callback.on_heartbeat(
            "verify_fake",
            1.2,
            9.8,
            "running",
            current_command=cmd,
            command_elapsed_sec=1.2,
            command_timeout_sec=10.0,
            command_progress_pct=12.0,
        )
        callback.on_progress("verify_fake", 1, 1, "")
        callback.on_command_complete(
            "verify_fake",
            cmd,
            0,
            1.2,
            completed=1,
            total=1,
            stdout_tail="late output",
            stderr_tail="",
        )
        callback.on_complete("verify_fake", "success", 0)
        return {
            "status": "success",
            "exit_code": 0,
            "nonce": "cancel_consistency_nonce",
            "log_path": str(tmp_path / "verify_cancel_consistency.log"),
            "jsonl_path": str(tmp_path / "verify_cancel_consistency.jsonl"),
            "commands": [cmd],
            "results": [],
            "degraded": False,
            "fail_fast": False,
            "fail_fast_skipped_commands": [],
            "cache_enabled": False,
            "cache_hits": 0,
            "cache_misses": 1,
        }

    monkeypatch.setattr(mcp_server, "run_verify_with_callback", fake_run_verify_with_callback)

    server = mcp_server.create_server()
    tools = asyncio.run(server.get_tools())
    start_fn = tools["accel_verify_start"].fn
    cancel_fn = tools["accel_verify_cancel"].fn
    status_fn = tools["accel_verify_status"].fn
    events_fn = tools["accel_verify_events"].fn

    started = start_fn(project=str(project_dir))
    job_id = str(started["job_id"])
    assert started_evt.wait(timeout=1.0)

    cancel_payload = cancel_fn(job_id=job_id)
    assert cancel_payload["status"] == mcp_server.JobState.CANCELLED
    assert bool(cancel_payload["cancelled"]) is True

    # Release worker thread to emit post-cancel callback notifications.
    continue_evt.set()

    for _ in range(30):
        status = status_fn(job_id=job_id)
        if status.get("state") == mcp_server.JobState.CANCELLED:
            break
        time.sleep(0.05)
    assert status.get("state") == mcp_server.JobState.CANCELLED

    # Give the worker callback thread time to attempt stale events.
    time.sleep(0.2)

    payload = events_fn(job_id=job_id, since_seq=0, max_events=200, include_summary=True)
    events = payload.get("events", [])
    assert isinstance(events, list)

    cancel_seq = max(
        (int(event.get("seq", 0)) for event in events if event.get("event") == "job_cancelled_finalized"),
        default=0,
    )
    assert cancel_seq > 0

    late_running_heartbeat = [
        event
        for event in events
        if int(event.get("seq", 0)) > cancel_seq
        and event.get("event") == "heartbeat"
        and str(event.get("state", "")) == mcp_server.JobState.RUNNING
    ]
    assert late_running_heartbeat == []

    late_job_completed = [
        event
        for event in events
        if int(event.get("seq", 0)) > cancel_seq and event.get("event") == "job_completed"
    ]
    assert late_job_completed == []

    summary = payload.get("summary", {})
    assert isinstance(summary, dict)
    assert summary.get("latest_state") == mcp_server.JobState.CANCELLED


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
    assert result.get("timeout_action") == "poll"
    job_id = str(result["job_id"])

    final_status = {}
    for _ in range(30):
        final_status = status_fn(job_id=job_id)
        if final_status.get("state") == mcp_server.JobState.COMPLETED:
            break
        time.sleep(0.1)

    assert final_status.get("state") == mcp_server.JobState.COMPLETED


def test_sync_verify_wait_runtime_default_is_capped_for_rpc_safety(tmp_path: Path, monkeypatch) -> None:
    project_dir = tmp_path / "sync_verify_runtime_wait_project"
    project_dir.mkdir(parents=True)

    captured: dict[str, float] = {}

    monkeypatch.setattr(
        mcp_server,
        "resolve_effective_config",
        lambda *args, **kwargs: {
            "runtime": {
                "accel_home": str(tmp_path / ".accel-home"),
                "sync_verify_wait_seconds": 180.0,
                "sync_verify_timeout_action": "poll",
                "sync_verify_cancel_grace_seconds": 1.0,
            }
        },
    )

    monkeypatch.setattr(
        mcp_server,
        "run_verify_with_callback",
        lambda project_dir, config, changed_files=None, callback=None: {
            "status": "success",
            "exit_code": 0,
            "nonce": "runtime_wait_nonce",
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
        },
    )

    def fake_wait_for_verify(job_id: str, *, max_wait_seconds: float, poll_seconds: float):
        captured["wait_seconds"] = float(max_wait_seconds)
        return {"status": "success", "exit_code": 0, "job_id": job_id}

    monkeypatch.setattr(mcp_server, "_wait_for_verify_job_result", fake_wait_for_verify)

    server = mcp_server.create_server()
    tools = asyncio.run(server.get_tools())
    verify_fn = tools["accel_verify"].fn

    result = verify_fn(project=str(project_dir), wait_for_completion=True)
    assert result["status"] == "success"
    assert float(captured.get("wait_seconds", 0.0)) == 45.0


def test_sync_verify_wait_explicit_override_is_capped_for_rpc_safety(tmp_path: Path, monkeypatch) -> None:
    project_dir = tmp_path / "sync_verify_explicit_wait_project"
    project_dir.mkdir(parents=True)

    captured: dict[str, float] = {}

    monkeypatch.setattr(
        mcp_server,
        "resolve_effective_config",
        lambda *args, **kwargs: {
            "runtime": {
                "accel_home": str(tmp_path / ".accel-home"),
                "sync_verify_wait_seconds": 45.0,
                "sync_verify_timeout_action": "poll",
                "sync_verify_cancel_grace_seconds": 1.0,
            }
        },
    )

    monkeypatch.setattr(
        mcp_server,
        "run_verify_with_callback",
        lambda project_dir, config, changed_files=None, callback=None: {
            "status": "success",
            "exit_code": 0,
            "nonce": "runtime_wait_nonce_2",
            "log_path": str(tmp_path / "verify2.log"),
            "jsonl_path": str(tmp_path / "verify2.jsonl"),
            "commands": [],
            "results": [],
            "degraded": False,
            "fail_fast": False,
            "fail_fast_skipped_commands": [],
            "cache_enabled": False,
            "cache_hits": 0,
            "cache_misses": 0,
        },
    )

    def fake_wait_for_verify(job_id: str, *, max_wait_seconds: float, poll_seconds: float):
        captured["wait_seconds"] = float(max_wait_seconds)
        return {"status": "success", "exit_code": 0, "job_id": job_id}

    monkeypatch.setattr(mcp_server, "_wait_for_verify_job_result", fake_wait_for_verify)

    server = mcp_server.create_server()
    tools = asyncio.run(server.get_tools())
    verify_fn = tools["accel_verify"].fn

    result = verify_fn(project=str(project_dir), wait_for_completion=True, sync_wait_seconds=180)
    assert result["status"] == "success"
    assert float(captured.get("wait_seconds", 0.0)) == 45.0


def test_sync_verify_timeout_action_cancel_finalizes_job(tmp_path: Path, monkeypatch) -> None:
    project_dir = tmp_path / "sync_timeout_cancel_project"
    project_dir.mkdir(parents=True)

    jm = mcp_server.JobManager()
    jm._jobs.clear()  # type: ignore[attr-defined]

    monkeypatch.setattr(mcp_server, "_sync_verify_wait_seconds", 1)
    monkeypatch.setattr(mcp_server, "_sync_verify_poll_seconds", 0.05)
    monkeypatch.setattr(
        mcp_server,
        "resolve_effective_config",
        lambda project_dir, cli_overrides=None: {
            "runtime": {
                "accel_home": str(tmp_path / ".accel-home"),
                "sync_verify_timeout_action": "cancel",
                "sync_verify_cancel_grace_seconds": 0.5,
            }
        },
    )

    def fake_run_verify_with_callback(project_dir, config, changed_files=None, callback=None):
        time.sleep(1.4)
        return {
            "status": "success",
            "exit_code": 0,
            "nonce": "late_nonce",
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

    result = verify_fn(
        project=str(project_dir),
        wait_for_completion=True,
        sync_wait_seconds=1,
        sync_timeout_action="cancel",
        sync_cancel_grace_seconds=0.5,
    )
    assert result["status"] == "cancelled"
    assert int(result["exit_code"]) == 130
    assert bool(result["timed_out"]) is True
    assert result.get("timeout_action") == "cancel"
    assert bool(result.get("auto_cancel_requested")) is True

    job_id = str(result["job_id"])
    status = status_fn(job_id=job_id)
    assert status.get("state") == mcp_server.JobState.CANCELLED


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
        verify_cache_failed_results="true",
        verify_cache_ttl_seconds="300",
        verify_cache_failed_ttl_seconds="90",
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


def test_verify_start_accepts_timebox_runtime_overrides(tmp_path: Path, monkeypatch) -> None:
    project_dir = tmp_path / "verify_timebox_overrides_project"
    project_dir.mkdir(parents=True)

    jm = mcp_server.JobManager()
    jm._jobs.clear()  # type: ignore[attr-defined]

    captured: dict[str, object] = {}

    def fake_resolve_effective_config(project_dir, cli_overrides=None):
        captured["cli_overrides"] = cli_overrides
        return {
            "runtime": {"accel_home": str(tmp_path / ".accel-home")},
            "verify": {},
        }

    monkeypatch.setattr(mcp_server, "resolve_effective_config", fake_resolve_effective_config)
    monkeypatch.setattr(
        mcp_server,
        "run_verify_with_callback",
        lambda project_dir, config, changed_files=None, callback=None: {
            "status": "partial",
            "exit_code": 124,
            "nonce": "timebox_override_nonce",
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
            "partial": True,
            "partial_reason": "max_wall_time_exceeded",
            "unfinished_items": [],
            "unfinished_commands": [],
            "timed_out": True,
        },
    )

    server = mcp_server.create_server()
    tools = asyncio.run(server.get_tools())
    start_fn = tools["accel_verify_start"].fn
    status_fn = tools["accel_verify_status"].fn

    started = start_fn(
        project=str(project_dir),
        verify_stall_timeout_seconds="9",
        verify_max_wall_time_seconds="77",
        verify_auto_cancel_on_stall="true",
    )
    assert started["status"] == "started"

    job_id = str(started["job_id"])
    for _ in range(30):
        status = status_fn(job_id=job_id)
        if status.get("state") == mcp_server.JobState.COMPLETED:
            break
        time.sleep(0.05)

    overrides = captured.get("cli_overrides")
    assert isinstance(overrides, dict)
    runtime = overrides.get("runtime", {})
    assert isinstance(runtime, dict)
    assert float(runtime.get("verify_stall_timeout_seconds", 0.0)) == 9.0
    assert float(runtime.get("verify_max_wall_time_seconds", 0.0)) == 77.0
    assert bool(runtime.get("verify_auto_cancel_on_stall")) is True


def test_verify_start_fast_loop_defaults_cache_failed_results(tmp_path: Path, monkeypatch) -> None:
    project_dir = tmp_path / "verify_start_fastloop_defaults_project"
    project_dir.mkdir(parents=True)

    jm = mcp_server.JobManager()
    jm._jobs.clear()  # type: ignore[attr-defined]

    captured: dict[str, object] = {}

    def fake_resolve_effective_config(project_dir, cli_overrides=None):
        captured["cli_overrides"] = cli_overrides
        return {"runtime": {"accel_home": str(tmp_path / ".accel-home")}, "verify": {}}

    monkeypatch.setattr(mcp_server, "resolve_effective_config", fake_resolve_effective_config)
    monkeypatch.setattr(
        mcp_server,
        "run_verify_with_callback",
        lambda project_dir, config, changed_files=None, callback=None: {
            "status": "success",
            "exit_code": 0,
            "nonce": "start_fastloop_nonce",
            "log_path": str(tmp_path / "verify.log"),
            "jsonl_path": str(tmp_path / "verify.jsonl"),
            "commands": [],
            "results": [],
            "degraded": False,
            "fail_fast": True,
            "fail_fast_skipped_commands": [],
            "cache_enabled": True,
            "cache_hits": 0,
            "cache_misses": 0,
        },
    )

    server = mcp_server.create_server()
    tools = asyncio.run(server.get_tools())
    start_fn = tools["accel_verify_start"].fn
    started = start_fn(project=str(project_dir), fast_loop=True, evidence_run=False)
    assert started["status"] == "started"

    overrides = captured.get("cli_overrides")
    assert isinstance(overrides, dict)
    runtime = overrides.get("runtime", {})
    assert isinstance(runtime, dict)
    assert bool(runtime.get("verify_fail_fast")) is True
    assert bool(runtime.get("verify_cache_enabled")) is True
    assert bool(runtime.get("verify_cache_failed_results")) is True


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


def test_verify_status_exposes_partial_fields_from_completed_result() -> None:
    jm = mcp_server.JobManager()
    jm._jobs.clear()  # type: ignore[attr-defined]
    job = jm.create_job()
    job.mark_running("running")
    job.mark_completed(
        status="partial",
        exit_code=124,
        result={
            "status": "partial",
            "exit_code": 124,
            "partial": True,
            "partial_reason": "max_wall_time_exceeded",
            "unfinished_commands": ["pytest -q tests/test_a.py"],
            "timed_out": True,
        },
    )

    server = mcp_server.create_server()
    tools = asyncio.run(server.get_tools())
    status_fn = tools["accel_verify_status"].fn
    payload = status_fn(job_id=job.job_id)

    assert payload.get("state") == mcp_server.JobState.COMPLETED
    assert payload.get("status") == "partial"
    assert bool(payload.get("partial")) is True
    assert payload.get("partial_reason") == "max_wall_time_exceeded"
    assert isinstance(payload.get("unfinished_commands"), list)


def test_verify_status_exposes_failure_classification_from_completed_result() -> None:
    jm = mcp_server.JobManager()
    jm._jobs.clear()  # type: ignore[attr-defined]
    job = jm.create_job()
    job.mark_running("running")
    job.mark_completed(
        status="failed",
        exit_code=3,
        result={
            "status": "failed",
            "exit_code": 3,
            "failure_kind": "project_gate_failed",
            "failed_commands": ["pytest -q tests/test_auth.py"],
            "executor_failed_commands": [],
            "project_failed_commands": ["pytest -q tests/test_auth.py"],
            "failure_counts": {"failed_total": 1, "executor_failed": 0, "project_failed": 1},
        },
    )

    server = mcp_server.create_server()
    tools = asyncio.run(server.get_tools())
    status_fn = tools["accel_verify_status"].fn
    payload = status_fn(job_id=job.job_id)

    assert payload.get("state") == mcp_server.JobState.COMPLETED
    assert payload.get("status") == "failed"
    assert payload.get("failure_kind") == "project_gate_failed"
    assert isinstance(payload.get("project_failed_commands"), list)
    failure_counts = payload.get("failure_counts", {})
    assert isinstance(failure_counts, dict)
    assert int(failure_counts.get("project_failed", 0)) == 1


def test_verify_status_clamps_finalizing_progress_before_terminal_transition() -> None:
    jm = mcp_server.JobManager()
    jm._jobs.clear()  # type: ignore[attr-defined]
    job = jm.create_job()
    job.mark_running("running")
    job.update_progress(2, 2, "")

    server = mcp_server.create_server()
    tools = asyncio.run(server.get_tools())
    status_fn = tools["accel_verify_status"].fn
    payload = status_fn(job_id=job.job_id)

    assert payload.get("state") == mcp_server.JobState.RUNNING
    assert float(payload.get("progress", 0.0)) < 100.0
    assert payload.get("state_consistency") == "finalizing"


def test_sync_index_timeout_returns_running_then_completes(tmp_path: Path, monkeypatch) -> None:
    project_dir = tmp_path / "sync_index_timeout_project"
    project_dir.mkdir(parents=True)

    jm = mcp_server.JobManager()
    jm._jobs.clear()  # type: ignore[attr-defined]

    monkeypatch.setattr(mcp_server, "_sync_index_wait_seconds", 1)
    monkeypatch.setattr(mcp_server, "_sync_index_poll_seconds", 0.05)
    monkeypatch.setattr(
        mcp_server,
        "resolve_effective_config",
        lambda *args, **kwargs: {
            "runtime": {
                "accel_home": str(tmp_path / ".accel-home"),
                "sync_index_wait_seconds": 1.0,
            }
        },
    )

    def fake_tool_index_build(project: str = ".", full: bool = True):
        time.sleep(1.3)
        return {"status": "ok", "manifest": {"counts": {"files": 2}, "indexed_files": ["a.py", "b.py"]}}

    monkeypatch.setattr(mcp_server, "_tool_index_build", fake_tool_index_build)

    server = mcp_server.create_server()
    tools = asyncio.run(server.get_tools())
    build_fn = tools["accel_index_build"].fn
    status_fn = tools["accel_index_status"].fn

    started_at = time.perf_counter()
    result = build_fn(project=str(project_dir), full=True, wait_for_completion=True)
    elapsed = time.perf_counter() - started_at

    assert elapsed < 1.3
    assert result["status"] == "running"
    assert bool(result["timed_out"]) is True
    job_id = str(result["job_id"])
    assert job_id.startswith("index_")

    final_status = {}
    for _ in range(40):
        final_status = status_fn(job_id=job_id)
        if final_status.get("state") == mcp_server.JobState.COMPLETED:
            break
        time.sleep(0.1)
    assert final_status.get("state") == mcp_server.JobState.COMPLETED


def test_sync_index_wait_runtime_default_is_capped_for_rpc_safety(tmp_path: Path, monkeypatch) -> None:
    project_dir = tmp_path / "sync_index_runtime_wait_project"
    project_dir.mkdir(parents=True)

    captured: dict[str, float] = {}

    monkeypatch.setattr(
        mcp_server,
        "resolve_effective_config",
        lambda *args, **kwargs: {
            "runtime": {
                "accel_home": str(tmp_path / ".accel-home"),
                "sync_index_wait_seconds": 205.0,
            }
        },
    )

    class _DummyJob:
        job_id = "index_runtime_wait_01"

        @staticmethod
        def to_status():
            return {"state": "running", "stage": "indexing", "progress": 0.0, "elapsed_sec": 0.0}

    monkeypatch.setattr(mcp_server, "_start_index_job", lambda **kwargs: _DummyJob())

    def fake_wait_for_index(job_id: str, *, max_wait_seconds: float, poll_seconds: float):
        captured["wait_seconds"] = float(max_wait_seconds)
        return {"status": "ok", "exit_code": 0, "job_id": job_id, "timed_out": False}

    monkeypatch.setattr(mcp_server, "_wait_for_index_job_result", fake_wait_for_index)

    server = mcp_server.create_server()
    tools = asyncio.run(server.get_tools())
    build_fn = tools["accel_index_build"].fn

    result = build_fn(project=str(project_dir), full=True, wait_for_completion=True)
    assert result["status"] == "ok"
    assert float(captured.get("wait_seconds", 0.0)) == 45.0


def test_sync_index_wait_explicit_override_is_capped_for_rpc_safety(tmp_path: Path, monkeypatch) -> None:
    project_dir = tmp_path / "sync_index_explicit_wait_project"
    project_dir.mkdir(parents=True)

    captured: dict[str, float] = {}

    monkeypatch.setattr(
        mcp_server,
        "resolve_effective_config",
        lambda *args, **kwargs: {
            "runtime": {
                "accel_home": str(tmp_path / ".accel-home"),
                "sync_index_wait_seconds": 45.0,
            }
        },
    )

    class _DummyJob:
        job_id = "index_runtime_wait_02"

        @staticmethod
        def to_status():
            return {"state": "running", "stage": "indexing", "progress": 0.0, "elapsed_sec": 0.0}

    monkeypatch.setattr(mcp_server, "_start_index_job", lambda **kwargs: _DummyJob())

    def fake_wait_for_index(job_id: str, *, max_wait_seconds: float, poll_seconds: float):
        captured["wait_seconds"] = float(max_wait_seconds)
        return {"status": "ok", "exit_code": 0, "job_id": job_id, "timed_out": False}

    monkeypatch.setattr(mcp_server, "_wait_for_index_job_result", fake_wait_for_index)

    server = mcp_server.create_server()
    tools = asyncio.run(server.get_tools())
    build_fn = tools["accel_index_build"].fn

    result = build_fn(project=str(project_dir), full=True, wait_for_completion=True, sync_wait_seconds=205)
    assert result["status"] == "ok"
    assert float(captured.get("wait_seconds", 0.0)) == 45.0


def test_sync_index_returns_manifest_for_fast_job(tmp_path: Path, monkeypatch) -> None:
    project_dir = tmp_path / "sync_index_fast_project"
    project_dir.mkdir(parents=True)

    jm = mcp_server.JobManager()
    jm._jobs.clear()  # type: ignore[attr-defined]

    monkeypatch.setattr(mcp_server, "_sync_index_wait_seconds", 3)
    monkeypatch.setattr(mcp_server, "_sync_index_poll_seconds", 0.05)
    monkeypatch.setattr(
        mcp_server,
        "_tool_index_build",
        lambda project=".", full=True: {
            "status": "ok",
            "manifest": {"counts": {"files": 3}, "indexed_files": ["x.py", "y.py", "z.py"]},
        },
    )

    server = mcp_server.create_server()
    tools = asyncio.run(server.get_tools())
    build_fn = tools["accel_index_build"].fn

    result = build_fn(project=str(project_dir), full=True, wait_for_completion=True)
    assert result["status"] == "ok"
    assert bool(result["timed_out"]) is False
    assert int(result["manifest"]["counts"]["files"]) == 3
    assert str(result["job_id"]).startswith("index_")


def test_index_cancel_finalizes_running_job(tmp_path: Path, monkeypatch) -> None:
    project_dir = tmp_path / "index_cancel_project"
    project_dir.mkdir(parents=True)

    jm = mcp_server.JobManager()
    jm._jobs.clear()  # type: ignore[attr-defined]

    def fake_tool_index_update(project: str = "."):
        time.sleep(2.0)
        return {"status": "ok", "manifest": {"counts": {"files": 1}}}

    monkeypatch.setattr(mcp_server, "_tool_index_update", fake_tool_index_update)

    server = mcp_server.create_server()
    tools = asyncio.run(server.get_tools())
    update_fn = tools["accel_index_update"].fn
    cancel_fn = tools["accel_index_cancel"].fn
    status_fn = tools["accel_index_status"].fn

    started = update_fn(project=str(project_dir), wait_for_completion=False)
    assert started["status"] == "started"
    job_id = str(started["job_id"])
    assert job_id.startswith("index_")

    cancelled = cancel_fn(job_id=job_id)
    assert bool(cancelled.get("cancelled")) is True
    assert cancelled.get("status") == mcp_server.JobState.CANCELLED

    status = status_fn(job_id=job_id)
    assert status.get("state") == mcp_server.JobState.CANCELLED


def test_index_progress_events_expose_file_metrics(tmp_path: Path, monkeypatch) -> None:
    project_dir = tmp_path / "index_progress_metrics_project"
    project_dir.mkdir(parents=True)

    jm = mcp_server.JobManager()
    jm._jobs.clear()  # type: ignore[attr-defined]

    def fake_tool_index_build(project: str = ".", full: bool = True, progress_callback=None):
        if progress_callback is not None:
            progress_callback(
                {
                    "stage": "scan_files",
                    "processed_files": 2,
                    "total_files": 8,
                    "current_path": "src/a.py",
                    "changed_files": 1,
                }
            )
            progress_callback(
                {
                    "stage": "index_payloads",
                    "processed_files": 5,
                    "total_files": 8,
                    "current_path": "src/b.py",
                    "changed_files": 2,
                }
            )
        time.sleep(0.05)
        return {
            "status": "ok",
            "manifest": {"counts": {"files": 8}, "indexed_files": [f"src/{idx}.py" for idx in range(8)]},
        }

    monkeypatch.setattr(mcp_server, "_tool_index_build", fake_tool_index_build)

    server = mcp_server.create_server()
    tools = asyncio.run(server.get_tools())
    build_fn = tools["accel_index_build"].fn
    status_fn = tools["accel_index_status"].fn
    events_fn = tools["accel_index_events"].fn

    started = build_fn(project=str(project_dir), full=True, wait_for_completion=False)
    job_id = str(started["job_id"])

    final_status = {}
    for _ in range(40):
        final_status = status_fn(job_id=job_id)
        if final_status.get("state") == mcp_server.JobState.COMPLETED:
            break
        time.sleep(0.05)

    assert final_status.get("state") == mcp_server.JobState.COMPLETED
    assert int(final_status.get("processed_files", 0)) >= 0
    assert int(final_status.get("total_files", 0)) >= 0
    assert "current_path" in final_status

    events_payload = events_fn(job_id=job_id, since_seq=0, max_events=200, include_summary=True)
    events = events_payload.get("events", [])
    assert isinstance(events, list)
    progress_events = [event for event in events if event.get("event") == "index_progress"]
    assert len(progress_events) >= 1
    assert any("processed_files" in event for event in progress_events)
    summary = events_payload.get("summary", {})
    assert isinstance(summary, dict)
    progress_summary = summary.get("progress", {})
    assert isinstance(progress_summary, dict)
    assert "processed_files" in progress_summary
    assert "total_files" in progress_summary


def test_index_status_clamps_finalizing_progress_before_terminal_transition() -> None:
    jm = mcp_server.JobManager()
    jm._jobs.clear()  # type: ignore[attr-defined]
    job = jm.create_job(prefix="index")
    job.mark_running("indexing")
    job.update_progress(3, 3, "")

    server = mcp_server.create_server()
    tools = asyncio.run(server.get_tools())
    status_fn = tools["accel_index_status"].fn

    payload = status_fn(job_id=job.job_id)
    assert payload.get("state") == mcp_server.JobState.RUNNING
    assert float(payload.get("progress_pct", 0.0)) < 100.0
    assert payload.get("state_consistency") == "finalizing"


def test_verify_events_capture_stream_output_and_stall_signal(tmp_path: Path, monkeypatch) -> None:
    project_dir = tmp_path / "verify_stream_output_project"
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
        assert callback is not None
        cmd = "python -c \"print('ok')\""
        callback.on_start("verify_fake", 1)
        callback.on_stage_change("verify_fake", mcp_server.VerifyStage.RUNNING)
        callback.on_command_start("verify_fake", cmd, 0, 1)
        callback.on_command_output("verify_fake", cmd, "stdout", "line-a\n", truncated=False)
        callback.on_command_output("verify_fake", cmd, "stderr", "warn-a\n", truncated=False)
        callback.on_heartbeat(
            "verify_fake",
            5.0,
            1.0,
            "running",
            current_command=cmd,
            command_elapsed_sec=5.0,
            command_timeout_sec=8.0,
            command_progress_pct=62.5,
            stall_detected=True,
            stall_elapsed_sec=2.0,
        )
        callback.on_command_complete(
            "verify_fake",
            cmd,
            0,
            6.0,
            completed=1,
            total=1,
            stdout_tail="line-a",
            stderr_tail="warn-a",
        )
        callback.on_progress("verify_fake", 1, 1, "")
        callback.on_complete("verify_fake", "success", 0)
        return {
            "status": "success",
            "exit_code": 0,
            "nonce": "stream_output_nonce",
            "log_path": str(tmp_path / "verify_stream.log"),
            "jsonl_path": str(tmp_path / "verify_stream.jsonl"),
            "commands": [cmd],
            "results": [],
            "degraded": False,
            "fail_fast": False,
            "fail_fast_skipped_commands": [],
            "cache_enabled": False,
            "cache_hits": 0,
            "cache_misses": 1,
        }

    monkeypatch.setattr(mcp_server, "run_verify_with_callback", fake_run_verify_with_callback)

    server = mcp_server.create_server()
    tools = asyncio.run(server.get_tools())
    start_fn = tools["accel_verify_start"].fn
    status_fn = tools["accel_verify_status"].fn
    events_fn = tools["accel_verify_events"].fn

    started = start_fn(project=str(project_dir))
    job_id = str(started["job_id"])

    final_status = {}
    for _ in range(30):
        final_status = status_fn(job_id=job_id)
        if final_status.get("state") == mcp_server.JobState.COMPLETED:
            break
        time.sleep(0.05)

    assert final_status.get("state") == mcp_server.JobState.COMPLETED
    recent_output = final_status.get("recent_output", [])
    assert isinstance(recent_output, list)
    assert len(recent_output) >= 1
    assert bool(final_status.get("stall_detected")) is True

    payload = events_fn(job_id=job_id, since_seq=0, max_events=200, include_summary=True)
    summary = payload.get("summary", {})
    assert isinstance(summary, dict)
    stats = summary.get("command_stats", {})
    assert isinstance(stats, dict)
    assert int(stats.get("output_chunks", 0)) >= 2
    assert int(stats.get("stall_heartbeats", 0)) >= 1
    summary_output = summary.get("recent_output", [])
    assert isinstance(summary_output, list)
    assert len(summary_output) >= 1


def test_tool_context_exposes_changed_files_detection_details(tmp_path: Path, monkeypatch) -> None:
    project_dir = tmp_path / "context_changed_detection_project"
    project_dir.mkdir(parents=True)

    monkeypatch.setattr(
        mcp_server,
        "resolve_effective_config",
        lambda project_dir: {"runtime": {"accel_home": str(tmp_path / ".accel-home"), "semantic_cache_enabled": False}},
    )
    monkeypatch.setattr(mcp_server, "_discover_changed_files_from_git", lambda project_dir, limit=200: ["src/a.py"])
    monkeypatch.setattr(
        mcp_server,
        "_discover_changed_files_from_git_details",
        lambda project_dir, limit=200: (
            ["src/a.py"],
            {
                "provider": "git",
                "available": True,
                "reason": "ok",
                "sources": ["status_porcelain", "staged_diff"],
                "source_counts": {"status_porcelain": 1, "staged_diff": 1},
                "confidence": 0.97,
                "git_root": str(project_dir),
            },
        ),
    )
    monkeypatch.setattr(
        mcp_server,
        "compile_context_pack",
        lambda **kwargs: {
            "version": 1,
            "task": str(kwargs.get("task", "")),
            "generated_at": "2026-02-12T00:00:00+00:00",
            "budget": {"max_chars": 6000, "max_snippets": 16, "top_n_files": 6},
            "top_files": [{"path": "src/a.py", "score": 1.0, "reasons": ["changed_file"], "signals": []}],
            "snippets": [{"path": "src/a.py", "content": "print('ok')", "start_line": 1, "end_line": 1}],
            "verify_plan": {"target_tests": [], "target_checks": ["pytest -q"]},
            "meta": {"source_chars_est": 7000},
        },
    )

    result = mcp_server._tool_context(
        project=str(project_dir),
        task="inspect changed files detection details",
        changed_files=None,
    )

    assert result["status"] == "ok"
    assert result["changed_files_source"] == "git_auto"
    detection = result.get("changed_files_detection", {})
    assert isinstance(detection, dict)
    assert detection.get("provider") == "git"
    assert float(detection.get("confidence", 0.0)) >= 0.9
    assert "sources" in detection
