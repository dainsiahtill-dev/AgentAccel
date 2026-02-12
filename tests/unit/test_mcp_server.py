from __future__ import annotations

import asyncio
import json
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
    assert summary.get("latest_state") == mcp_server.JobState.COMPLETED
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
