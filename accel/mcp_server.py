from __future__ import annotations

import argparse
import json
import logging
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any
from uuid import uuid4

from fastmcp import FastMCP

from .config import resolve_effective_config
from .indexers import build_or_update_indexes
from .query.context_compiler import compile_context_pack, write_context_pack
from .query.planner import normalize_task_tokens
from .storage.cache import ensure_project_dirs, project_paths
from .verify.orchestrator import run_verify, run_verify_with_callback
from .verify.callbacks import VerifyProgressCallback, VerifyStage
from .verify.job_manager import JobManager, JobState, VerifyJob


JSONDict = dict[str, Any]
SERVER_NAME = "agent-accel-mcp"
SERVER_VERSION = "0.2.2"
TOOL_ERROR_EXECUTION_FAILED = "ACCEL_TOOL_EXECUTION_FAILED"

# Debug logging setup
_debug_enabled = os.environ.get("ACCEL_MCP_DEBUG", "").lower() in {"1", "true", "yes"}
_debug_logger: logging.Logger | None = None

# Global server timeout protection
_server_start_time = 0.0
_server_max_runtime = int(os.environ.get("ACCEL_MCP_MAX_RUNTIME", "3600"))  # 1 hour default
_shutdown_requested = False

# Keep sync verify bounded so a single call cannot monopolize MCP request handling.
_sync_verify_wait_seconds = int(os.environ.get("ACCEL_MCP_SYNC_VERIFY_WAIT_SECONDS", "45"))
_sync_verify_poll_seconds = float(os.environ.get("ACCEL_MCP_SYNC_VERIFY_POLL_SECONDS", "0.2"))
_FALSE_LITERALS = {"", "0", "false", "no", "off", "none", "null", "undefined", "pydanticundefined"}
_TRUE_LITERALS = {"1", "true", "yes", "on"}

def _signal_handler(signum: int, frame) -> None:
    """Handle shutdown signals gracefully."""
    global _shutdown_requested
    _shutdown_requested = True
    _debug_log(f"Received signal {signum}, initiating graceful shutdown")
    # Give the server a moment to clean up
    time.sleep(0.1)
    sys.exit(0)

def _check_server_runtime() -> None:
    """Check if server has exceeded maximum runtime."""
    global _server_start_time, _shutdown_requested
    
    if _shutdown_requested:
        return
        
    if _server_start_time > 0:
        elapsed = time.perf_counter() - _server_start_time
        if elapsed > _server_max_runtime:
            _debug_log(f"Server exceeded maximum runtime ({_server_max_runtime}s), shutting down")
            sys.exit(0)

def _setup_debug_log() -> logging.Logger:
    """Setup debug logging for MCP protocol tracing."""
    if _debug_enabled:
        logger = logging.getLogger("accel_mcp_debug")
        logger.setLevel(logging.DEBUG)
        
        # Create log directory and file
        log_dir = Path.home() / ".accel" / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / f"mcp_debug_{int(time.time())}.log"
        
        handler = logging.FileHandler(log_file, encoding="utf-8")
        formatter = logging.Formatter(
            "%(asctime)s.%(msecs)03d [%(levelname)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        
        logger.debug("MCP debug logging initialized")
        return logger
    else:
        # Return a no-op logger
        logger = logging.getLogger("accel_mcp_noop")
        logger.setLevel(logging.CRITICAL + 1)  # Disable all logging
        return logger

def _debug_log(message: str) -> None:
    """Log debug message if debugging is enabled."""
    if _debug_enabled:
        global _debug_logger
        if _debug_logger is None:
            _debug_logger = _setup_debug_log()
        _debug_logger.debug(message)


def _effective_sync_verify_wait_seconds() -> int:
    return max(1, min(50, int(_sync_verify_wait_seconds)))


def _effective_sync_verify_poll_seconds() -> float:
    return max(0.05, float(_sync_verify_poll_seconds))


def _coerce_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        token = value.strip().lower()
        if token in _FALSE_LITERALS:
            return False
        if token in _TRUE_LITERALS:
            return True
        return default
    return default


def _coerce_optional_bool(value: Any) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        token = value.strip().lower()
        if token in _FALSE_LITERALS:
            return False
        if token in _TRUE_LITERALS:
            return True
        return None
    return None


def _coerce_optional_int(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        token = value.strip()
        if not token:
            return None
        if token.lower() in _FALSE_LITERALS:
            return None
        try:
            return int(token)
        except ValueError:
            try:
                return int(float(token))
            except ValueError:
                return None
    return None


def _wait_for_verify_job_result(
    job_id: str,
    *,
    max_wait_seconds: float,
    poll_seconds: float,
) -> JSONDict | None:
    jm = JobManager()
    deadline = time.perf_counter() + max(0.0, max_wait_seconds)
    poll = max(0.05, poll_seconds)

    while True:
        job = jm.get_job(job_id)
        if job is None:
            return {
                "status": "failed",
                "exit_code": 1,
                "job_id": job_id,
                "error": "job_not_found",
            }

        if job.state == JobState.COMPLETED:
            if isinstance(job.result, dict) and job.result:
                payload = dict(job.result)
            else:
                payload = {"status": "success", "exit_code": 0}
            payload.setdefault("job_id", job_id)
            return payload

        if job.state == JobState.FAILED:
            return {
                "status": "failed",
                "exit_code": 1,
                "job_id": job_id,
                "error": str(job.error or "verify job failed"),
            }

        if job.state == JobState.CANCELLED:
            return {
                "status": "cancelled",
                "exit_code": 130,
                "job_id": job_id,
            }

        if time.perf_counter() >= deadline:
            return None
        time.sleep(poll)


def _sync_verify_timeout_payload(job_id: str, wait_seconds: float) -> JSONDict:
    jm = JobManager()
    job = jm.get_job(job_id)
    status = job.to_status() if job is not None else {}
    return {
        "status": "running",
        "exit_code": 124,
        "timed_out": True,
        "job_id": job_id,
        "message": (
            f"accel_verify exceeded synchronous wait window ({wait_seconds:.1f}s); "
            "use accel_verify_status/accel_verify_events to continue polling."
        ),
        "poll_interval_sec": 1.0,
        "state": status.get("state", "running"),
        "stage": status.get("stage", "running"),
        "progress": status.get("progress", 0.0),
        "elapsed_sec": status.get("elapsed_sec", 0.0),
    }

def _with_timeout(func, timeout_seconds: int = 300):
    """Wrapper to add timeout to MCP tool functions."""
    def wrapper(*args, **kwargs):
        # Check server runtime before each tool call
        _check_server_runtime()
        
        start_time = time.perf_counter()
        _debug_log(f"Starting {func.__name__} with timeout {timeout_seconds}s")
        
        try:
            # This is a simple timeout implementation
            # For more robust timeout handling, consider using async/await
            result = func(*args, **kwargs)
            elapsed = time.perf_counter() - start_time
            _debug_log(f"Completed {func.__name__} in {elapsed:.3f}s")
            return result
        except Exception as exc:
            elapsed = time.perf_counter() - start_time
            _debug_log(f"Failed {func.__name__} after {elapsed:.3f}s: {exc!r}")
            raise RuntimeError(f"{TOOL_ERROR_EXECUTION_FAILED}: {exc}") from exc
    
    return wrapper


def _normalize_project_dir(project_value: Any) -> Path:
    project_text = str(project_value or ".")
    return Path(os.path.abspath(project_text))


def _resolve_path(project_dir: Path, path_value: Any) -> Path | None:
    path_text = str(path_value or "").strip()
    if not path_text:
        return None
    path = Path(path_text)
    if not path.is_absolute():
        path = project_dir / path
    return Path(os.path.abspath(str(path)))


_BUDGET_PRESETS: dict[str, dict[str, int]] = {
    "tiny": {"max_chars": 6000, "max_snippets": 16, "top_n_files": 6, "snippet_radius": 20},
    "small": {"max_chars": 12000, "max_snippets": 30, "top_n_files": 8, "snippet_radius": 24},
    "medium": {"max_chars": 24000, "max_snippets": 60, "top_n_files": 12, "snippet_radius": 30},
    "large": {"max_chars": 36000, "max_snippets": 90, "top_n_files": 16, "snippet_radius": 40},
    "xlarge": {"max_chars": 50000, "max_snippets": 120, "top_n_files": 20, "snippet_radius": 50},
}
_BUDGET_PRESET_ALIASES: dict[str, str] = {
    "s": "small",
    "sm": "small",
    "m": "medium",
    "med": "medium",
    "balanced": "medium",
    "l": "large",
    "lg": "large",
    "xl": "xlarge",
    "default": "small",
}


def _to_string_list(value: list[str] | str | None) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]

    text = str(value).strip()
    if not text:
        return []

    if text.startswith("[") and text.endswith("]"):
        try:
            parsed = json.loads(text)
            if isinstance(parsed, list):
                return [str(item).strip() for item in parsed if str(item).strip()]
        except Exception:
            pass

    normalized = text.replace("\r\n", ",").replace("\n", ",").replace(";", ",")
    return [part.strip() for part in normalized.split(",") if part.strip()]


def _to_budget_override(value: dict[str, int] | str | None) -> dict[str, int]:
    if value is None:
        return {}

    if isinstance(value, str):
        token = value.strip().lower()
        if not token:
            return {}

        canonical = _BUDGET_PRESET_ALIASES.get(token, token)
        if canonical in _BUDGET_PRESETS:
            return dict(_BUDGET_PRESETS[canonical])

        if token.startswith("{") and token.endswith("}"):
            try:
                parsed = json.loads(token)
                if isinstance(parsed, dict):
                    return _to_budget_override(parsed)
            except Exception as exc:
                raise ValueError(f"invalid budget json: {exc}") from exc

        if "=" in token:
            parsed_pairs: dict[str, int] = {}
            for part in token.replace(";", ",").split(","):
                segment = part.strip()
                if not segment:
                    continue
                key, sep, raw = segment.partition("=")
                if not sep:
                    continue
                parsed_pairs[key.strip()] = int(raw.strip())
            if parsed_pairs:
                return _to_budget_override(parsed_pairs)

        preset_names = ", ".join(sorted(_BUDGET_PRESETS.keys()))
        raise ValueError(
            f"unsupported budget preset '{value}'. supported presets: {preset_names}, or pass a budget object."
        )

    if not isinstance(value, dict):
        return {}
    out: dict[str, int] = {}
    for key in ("max_chars", "max_snippets", "top_n_files", "snippet_radius"):
        if key in value:
            out[key] = int(value[key])
    return out


def _discover_changed_files_from_git(project_dir: Path, limit: int = 200) -> list[str]:
    discovered: list[str] = []
    lock = threading.Lock()

    def _collect() -> None:
        commands = [
            ["git", "-C", str(project_dir), "diff", "--name-only", "--relative", "HEAD"],
            ["git", "-C", str(project_dir), "diff", "--name-only", "--relative", "--cached"],
        ]
        seen: set[str] = set()
        output: list[str] = []
        for cmd in commands:
            try:
                proc = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    timeout=2,
                    check=False,
                )
            except Exception:
                continue
            if int(proc.returncode) != 0:
                continue
            for raw_line in proc.stdout.splitlines():
                rel = str(raw_line).strip().replace("\\", "/")
                if not rel or rel in seen:
                    continue
                seen.add(rel)
                output.append(rel)
                if len(output) >= limit:
                    break
            if len(output) >= limit:
                break
        with lock:
            discovered.extend(output)

    worker = threading.Thread(target=_collect, daemon=True)
    worker.start()
    worker.join(timeout=2.5)
    if worker.is_alive():
        _debug_log("_discover_changed_files_from_git timed out; fallback to empty changed_files")
        return []
    with lock:
        return list(discovered)


def _auto_context_budget_preset(task: str, changed_files: list[str], hints: list[str]) -> tuple[str, str]:
    task_text = str(task or "").strip()
    task_low = task_text.lower()
    task_tokens = normalize_task_tokens(task_text)
    changed_count = len(changed_files)
    hint_count = len(hints)

    quick_markers = {"explain", "summary", "summarize", "quick", "small", "typo", "doc", "readme"}
    complex_markers = {
        "architecture",
        "migration",
        "investigate",
        "root cause",
        "refactor",
        "performance",
        "security",
        "concurrency",
        "state machine",
        "deep",
    }

    score = 0
    score += min(6, len(task_tokens) // 4)
    score += min(5, changed_count // 2)
    score += 1 if hint_count >= 3 else 0
    if any(marker in task_low for marker in complex_markers):
        score += 2
    if any(marker in task_low for marker in quick_markers):
        score -= 1

    if changed_count <= 1 and len(task_tokens) <= 8 and len(task_text) <= 120 and score <= 2:
        return "tiny", "auto:tiny_for_quick_task"
    if changed_count >= 8 or len(task_tokens) >= 20 or len(task_text) >= 260 or score >= 8:
        return "medium", "auto:medium_for_complex_scope"
    return "small", "auto:small_default"


def _estimate_tokens(chars: int) -> int:
    return max(1, int((max(0, int(chars)) + 3) // 4))


def _resolve_context_budget(
    budget: dict[str, int] | str | None,
    *,
    task: str,
    changed_files: list[str],
    hints: list[str],
) -> tuple[dict[str, int], str, str, str]:
    override = _to_budget_override(budget)
    if override:
        preset = "custom"
        if isinstance(budget, str):
            token = budget.strip().lower()
            canonical = _BUDGET_PRESET_ALIASES.get(token, token)
            if canonical in _BUDGET_PRESETS:
                preset = canonical
        return override, "user", preset, "user_budget_input"

    auto_preset, auto_reason = _auto_context_budget_preset(task, changed_files, hints)
    return dict(_BUDGET_PRESETS[auto_preset]), "auto", auto_preset, auto_reason


def _tool_index_build(project: str = ".", full: bool = True) -> JSONDict:
    _debug_log(f"accel_index_build called: project={project}, full={full}")
    project_dir = _normalize_project_dir(project)
    config = resolve_effective_config(project_dir)
    manifest = build_or_update_indexes(
        project_dir=project_dir,
        config=config,
        mode="build",
        full=bool(full),
    )
    _debug_log(f"accel_index_build completed: {len(manifest.get('indexed_files', []))} files indexed")
    return {"status": "ok", "manifest": manifest}


def _tool_index_update(project: str = ".") -> JSONDict:
    _debug_log(f"accel_index_update called: project={project}")
    project_dir = _normalize_project_dir(project)
    config = resolve_effective_config(project_dir)
    manifest = build_or_update_indexes(
        project_dir=project_dir,
        config=config,
        mode="update",
        full=False,
    )
    _debug_log(f"accel_index_update completed: {len(manifest.get('changed_files', []))} files updated")
    return {"status": "ok", "manifest": manifest}


def _tool_context(
    *,
    project: str = ".",
    task: str,
    changed_files: Any = None,
    hints: Any = None,
    feedback_path: str | None = None,
    out: str | None = None,
    include_pack: Any = False,
    budget: Any = None,
) -> JSONDict:
    _debug_log("_tool_context: begin")
    project_dir = _normalize_project_dir(project)
    _debug_log(f"_tool_context: normalized project_dir={project_dir}")
    task_text = str(task).strip()
    if not task_text:
        raise ValueError("task is required")
    changed_files_list = _to_string_list(changed_files)
    hints_list = _to_string_list(hints)
    include_pack_flag = _coerce_bool(include_pack, False)
    _debug_log(
        "_tool_context: parsed inputs "
        f"changed_files={len(changed_files_list)} hints={len(hints_list)} include_pack={include_pack_flag}"
    )

    changed_files_source = "user" if changed_files_list else "none"
    if not changed_files_list:
        auto_changed = _discover_changed_files_from_git(project_dir)
        if auto_changed:
            changed_files_list = auto_changed
            changed_files_source = "git_auto"
    _debug_log(
        "_tool_context: changed_files resolution "
        f"source={changed_files_source} count={len(changed_files_list)}"
    )

    budget_override, budget_source, budget_preset, budget_reason = _resolve_context_budget(
        budget,
        task=task_text,
        changed_files=changed_files_list,
        hints=hints_list,
    )
    _debug_log(
        "_tool_context: budget resolution "
        f"source={budget_source} preset={budget_preset} max_chars={budget_override.get('max_chars', 0)}"
    )
    _debug_log(
        "accel_context called: "
        f"project={project}, task='{task[:50]}...', "
        f"changed_files={len(changed_files_list)}({changed_files_source}), "
        f"budget={budget_preset}({budget_source})"
    )

    config = resolve_effective_config(project_dir)
    _debug_log("_tool_context: config resolved")
    feedback_payload: JSONDict | None = None

    resolved_feedback_path = _resolve_path(project_dir, feedback_path)
    if resolved_feedback_path is not None and resolved_feedback_path.exists():
        feedback_payload = json.loads(resolved_feedback_path.read_text(encoding="utf-8"))
    _debug_log("_tool_context: feedback loaded")

    _debug_log("_tool_context: compile_context_pack start")
    pack = compile_context_pack(
        project_dir=project_dir,
        config=config,
        task=task_text,
        changed_files=changed_files_list,
        hints=hints_list,
        previous_attempt_feedback=feedback_payload,
        budget_override=budget_override,
    )
    _debug_log("_tool_context: compile_context_pack done")

    accel_home = Path(config["runtime"]["accel_home"]).resolve()
    paths = project_paths(accel_home, project_dir)
    ensure_project_dirs(paths)

    out_path = _resolve_path(project_dir, out)
    if out_path is None:
        out_path = paths["context"] / f"context_pack_{uuid4().hex[:10]}.json"

    _debug_log(f"_tool_context: write_context_pack start out={out_path}")
    write_context_pack(out_path, pack)
    _debug_log("_tool_context: write_context_pack done")
    pack_json_text = json.dumps(pack, ensure_ascii=False)
    context_chars = len(pack_json_text)
    meta = pack.get("meta", {}) if isinstance(pack.get("meta", {}), dict) else {}
    source_chars = int(meta.get("source_chars_est", 0) or 0)
    if source_chars <= 0:
        source_chars = context_chars
    compression_ratio = float(context_chars / source_chars) if source_chars > 0 else 1.0
    token_reduction_ratio = max(0.0, 1.0 - compression_ratio)
    warnings: list[str] = []
    if changed_files_source == "none":
        warnings.append("changed_files not provided and no git delta detected; context scope may be wider than necessary")

    payload: JSONDict = {
        "status": "ok",
        "out": str(out_path),
        "top_files": len(pack.get("top_files", [])),
        "snippets": len(pack.get("snippets", [])),
        "verify_plan": pack.get("verify_plan", {}),
        "context_chars": context_chars,
        "source_chars": source_chars,
        "estimated_tokens": _estimate_tokens(context_chars),
        "estimated_source_tokens": _estimate_tokens(source_chars),
        "compression_ratio": round(compression_ratio, 6),
        "token_reduction_ratio": round(token_reduction_ratio, 6),
        "budget_effective": pack.get("budget", {}),
        "budget_source": budget_source,
        "budget_preset": budget_preset,
        "budget_reason": budget_reason,
        "changed_files_used": changed_files_list,
        "changed_files_source": changed_files_source,
        "changed_files_count": len(changed_files_list),
    }
    if warnings:
        payload["warnings"] = warnings
    if include_pack_flag:
        payload["pack"] = pack

    _debug_log(f"accel_context completed: {len(pack.get('top_files', []))} files, {len(pack.get('snippets', []))} snippets")
    return payload


def _tool_verify(
    *,
    project: str = ".",
    changed_files: list[str] | str | None = None,
    evidence_run: bool = False,
    fast_loop: bool = False,
    verify_workers: int | None = None,
    per_command_timeout_seconds: int | None = None,
    verify_fail_fast: bool | str | None = None,
    verify_cache_enabled: bool | str | None = None,
    verify_cache_ttl_seconds: int | None = None,
    verify_cache_max_entries: int | None = None,
) -> JSONDict:
    # Normalize changed_files: handle comma-separated string
    if isinstance(changed_files, str):
        changed_files = [f.strip() for f in changed_files.split(",") if f.strip()]

    evidence_run = _coerce_bool(evidence_run, False)
    fast_loop = _coerce_bool(fast_loop, False)
    verify_fail_fast = _coerce_optional_bool(verify_fail_fast)
    verify_cache_enabled = _coerce_optional_bool(verify_cache_enabled)
    
    _debug_log(f"accel_verify called: project={project}, changed_files={len(changed_files or [])}, evidence_run={evidence_run}")
    project_dir = _normalize_project_dir(project)

    runtime_overrides: JSONDict = {}
    if verify_workers is not None:
        runtime_overrides["verify_workers"] = int(verify_workers)
    if per_command_timeout_seconds is not None:
        runtime_overrides["per_command_timeout_seconds"] = int(per_command_timeout_seconds)
    if verify_cache_ttl_seconds is not None:
        runtime_overrides["verify_cache_ttl_seconds"] = int(verify_cache_ttl_seconds)
    if verify_cache_max_entries is not None:
        runtime_overrides["verify_cache_max_entries"] = int(verify_cache_max_entries)

    if evidence_run and not fast_loop:
        runtime_overrides["verify_fail_fast"] = False
        runtime_overrides["verify_cache_enabled"] = False
    elif fast_loop and not evidence_run:
        runtime_overrides["verify_fail_fast"] = True
        runtime_overrides["verify_cache_enabled"] = True

    if verify_fail_fast is not None:
        runtime_overrides["verify_fail_fast"] = bool(verify_fail_fast)
    if verify_cache_enabled is not None:
        runtime_overrides["verify_cache_enabled"] = bool(verify_cache_enabled)

    cli_overrides = {"runtime": runtime_overrides} if runtime_overrides else None
    config = resolve_effective_config(project_dir, cli_overrides=cli_overrides)

    result = run_verify(
        project_dir=project_dir,
        config=config,
        changed_files=_to_string_list(changed_files),
    )
    
    _debug_log(f"accel_verify completed: status={result.get('status')}, exit_code={result.get('exit_code')}")
    return result


def create_server() -> FastMCP:
    _debug_log("Creating FastMCP server")
    server = FastMCP(
        name=SERVER_NAME,
        version=SERVER_VERSION,
        instructions=(
            "agent-accel MCP server for incremental index build, context pack generation, "
            "and scoped verification."
        ),
        strict_input_validation=True,
    )

    @server.tool(
        name="accel_index_build",
        description="Build indexes for the target project.",
    )
    def accel_index_build(project: str = ".", full: bool = True) -> JSONDict:
        try:
            return _with_timeout(_tool_index_build, timeout_seconds=600)(project=project, full=full)
        except Exception as exc:
            _debug_log(f"accel_index_build failed: {exc!r}")
            raise RuntimeError(f"{TOOL_ERROR_EXECUTION_FAILED}: {exc}") from exc

    @server.tool(
        name="accel_index_update",
        description="Incrementally update indexes for changed files.",
    )
    def accel_index_update(project: str = ".") -> JSONDict:
        try:
            return _with_timeout(_tool_index_update, timeout_seconds=300)(project=project)
        except Exception as exc:
            _debug_log(f"accel_index_update failed: {exc!r}")
            raise RuntimeError(f"{TOOL_ERROR_EXECUTION_FAILED}: {exc}") from exc

    @server.tool(
        name="accel_context",
        description="Generate a budgeted context pack for a task.",
    )
    def accel_context(
        task: str,
        project: str = ".",
        changed_files: Any = None,
        hints: Any = None,
        feedback_path: str | None = None,
        out: str | None = None,
        include_pack: Any = False,
        budget: Any = None,
    ) -> JSONDict:
        try:
            return _with_timeout(_tool_context, timeout_seconds=300)(
                project=project,
                task=task,
                changed_files=changed_files,
                hints=hints,
                feedback_path=feedback_path,
                out=out,
                include_pack=include_pack,
                budget=budget,
            )
        except Exception as exc:
            _debug_log(f"accel_context failed: {exc!r}")
            raise RuntimeError(f"{TOOL_ERROR_EXECUTION_FAILED}: {exc}") from exc

    def _start_verify_job(
        *,
        project: str = ".",
        changed_files: Any = None,
        evidence_run: Any = False,
        fast_loop: Any = False,
        verify_workers: Any = None,
        per_command_timeout_seconds: Any = None,
        verify_fail_fast: bool | str | None = None,
        verify_cache_enabled: bool | str | None = None,
        verify_cache_ttl_seconds: Any = None,
        verify_cache_max_entries: Any = None,
    ) -> VerifyJob:
        changed_files_list = _to_string_list(changed_files)

        evidence_run_normalized = _coerce_bool(evidence_run, False)
        fast_loop_normalized = _coerce_bool(fast_loop, False)
        verify_fail_fast_normalized = _coerce_optional_bool(verify_fail_fast)
        verify_cache_enabled_normalized = _coerce_optional_bool(verify_cache_enabled)
        verify_workers_value = _coerce_optional_int(verify_workers)
        per_command_timeout_value = _coerce_optional_int(per_command_timeout_seconds)
        verify_cache_ttl_value = _coerce_optional_int(verify_cache_ttl_seconds)
        verify_cache_max_entries_value = _coerce_optional_int(verify_cache_max_entries)

        _debug_log(f"_start_verify_job called: project={project}, changed_files={len(changed_files_list)}")
        project_dir = _normalize_project_dir(project)

        runtime_overrides: JSONDict = {}
        if verify_workers_value is not None:
            runtime_overrides["verify_workers"] = int(verify_workers_value)
        if per_command_timeout_value is not None:
            runtime_overrides["per_command_timeout_seconds"] = int(per_command_timeout_value)
        if verify_cache_ttl_value is not None:
            runtime_overrides["verify_cache_ttl_seconds"] = int(verify_cache_ttl_value)
        if verify_cache_max_entries_value is not None:
            runtime_overrides["verify_cache_max_entries"] = int(verify_cache_max_entries_value)

        if evidence_run_normalized and not fast_loop_normalized:
            runtime_overrides["verify_fail_fast"] = False
            runtime_overrides["verify_cache_enabled"] = False
        elif fast_loop_normalized and not evidence_run_normalized:
            runtime_overrides["verify_fail_fast"] = True
            runtime_overrides["verify_cache_enabled"] = True

        if verify_fail_fast_normalized is not None:
            runtime_overrides["verify_fail_fast"] = bool(verify_fail_fast_normalized)
        if verify_cache_enabled_normalized is not None:
            runtime_overrides["verify_cache_enabled"] = bool(verify_cache_enabled_normalized)

        cli_overrides = {"runtime": runtime_overrides} if runtime_overrides else None
        config = resolve_effective_config(project_dir, cli_overrides=cli_overrides)

        jm = JobManager()
        job = jm.create_job()

        def _run_verify_thread(job_id: str, project_dir: Path, config: dict[str, Any], changed_files: list[str] | None) -> None:
            j = jm.get_job(job_id)
            if j is None:
                return
            try:
                j.mark_running("running")
                callback = _JobCallback(j)
                result = run_verify_with_callback(
                    project_dir=project_dir,
                    config=config,
                    changed_files=changed_files,
                    callback=callback,
                )
                # Do not overwrite a user-cancelled terminal state with completion.
                if j.state in (JobState.CANCELLING, JobState.CANCELLED):
                    if j.state != JobState.CANCELLED:
                        j.mark_cancelled()
                    j.add_event("job_cancelled_finalized", {"reason": "worker_observed_cancel"})
                    return
                j.mark_completed(result.get("status", "unknown"), result.get("exit_code", 1), result)
            except Exception as exc:
                if j.state in (JobState.CANCELLING, JobState.CANCELLED):
                    if j.state != JobState.CANCELLED:
                        j.mark_cancelled()
                    j.add_event("job_cancelled_finalized", {"reason": "worker_exception_after_cancel"})
                    return
                j.mark_failed(str(exc))
                j.add_event("job_failed", {"error": str(exc)})

        import threading

        thread = threading.Thread(
            target=_run_verify_thread,
            args=(job.job_id, project_dir, config, changed_files_list),
            daemon=True,
        )
        thread.start()

        def _heartbeat_thread(job_id: str) -> None:
            while True:
                current = jm.get_job(job_id)
                if current is None:
                    return
                status = current.to_status()
                state = str(status.get("state", ""))
                if state in {JobState.COMPLETED, JobState.FAILED, JobState.CANCELLED}:
                    return
                current.add_event(
                    "heartbeat",
                    {
                        "elapsed_sec": float(status.get("elapsed_sec", 0.0)),
                        "eta_sec": status.get("eta_sec"),
                        "state": state,
                        "stage": status.get("stage", "running"),
                        "progress": float(status.get("progress", 0.0)),
                    },
                )
                time.sleep(1.0)

        hb = threading.Thread(target=_heartbeat_thread, args=(job.job_id,), daemon=True)
        hb.start()
        return job

    @server.tool(
        name="accel_verify",
        description="Run incremental verification with runtime override options.",
    )
    def accel_verify(
        project: str = ".",
        changed_files: Any = None,
        evidence_run: Any = False,
        fast_loop: Any = False,
        verify_workers: Any = None,
        per_command_timeout_seconds: Any = None,
        verify_fail_fast: bool | str | None = None,
        verify_cache_enabled: bool | str | None = None,
        verify_cache_ttl_seconds: Any = None,
        verify_cache_max_entries: Any = None,
        wait_for_completion: Any = False,
        sync_wait_seconds: Any = None,
    ) -> JSONDict:
        try:
            wait_flag = _coerce_bool(wait_for_completion, False)
            job = _start_verify_job(
                project=project,
                changed_files=changed_files,
                evidence_run=evidence_run,
                fast_loop=fast_loop,
                verify_workers=verify_workers,
                per_command_timeout_seconds=per_command_timeout_seconds,
                verify_fail_fast=verify_fail_fast,
                verify_cache_enabled=verify_cache_enabled,
                verify_cache_ttl_seconds=verify_cache_ttl_seconds,
                verify_cache_max_entries=verify_cache_max_entries,
            )
            job_id = str(job.job_id).strip()

            if not wait_flag:
                status_payload = job.to_status()
                return {
                    "status": "started",
                    "exit_code": 0,
                    "timed_out": False,
                    "job_id": job_id,
                    "message": "Verification started asynchronously. Use accel_verify_status/events for live progress.",
                    "poll_interval_sec": 1.0,
                    "state": status_payload.get("state", "pending"),
                    "stage": status_payload.get("stage", "init"),
                    "progress": status_payload.get("progress", 0.0),
                    "elapsed_sec": status_payload.get("elapsed_sec", 0.0),
                }

            sync_wait_seconds_value = _coerce_optional_int(sync_wait_seconds)
            wait_seconds = float(
                sync_wait_seconds_value if sync_wait_seconds_value is not None else _effective_sync_verify_wait_seconds()
            )
            wait_seconds = max(1.0, min(55.0, wait_seconds))
            result = _wait_for_verify_job_result(
                job_id,
                max_wait_seconds=wait_seconds,
                poll_seconds=_effective_sync_verify_poll_seconds(),
            )
            if result is None:
                _debug_log(f"accel_verify sync wait timeout for job_id={job_id}")
                return _sync_verify_timeout_payload(job_id, wait_seconds)
            return result
        except Exception as exc:
            _debug_log(f"accel_verify failed: {exc!r}")
            raise RuntimeError(f"{TOOL_ERROR_EXECUTION_FAILED}: {exc}") from exc

    class _JobCallback(VerifyProgressCallback):
        def __init__(self, job: VerifyJob) -> None:
            self._job = job

        def on_start(self, job_id: str, total_commands: int) -> None:
            self._job.update_progress(0, total_commands, "")
            self._job.add_event("job_started", {"total_commands": total_commands})

        def on_stage_change(self, job_id: str, stage: VerifyStage) -> None:
            self._job.stage = stage.name.lower()
            self._job.add_event("stage_change", {"stage": stage.name.lower()})

        def on_command_start(self, job_id: str, command: str, index: int, total: int) -> None:
            self._job.current_command = command
            self._job.add_event("command_start", {"command": command, "index": index, "total": total})

        def on_command_complete(self, job_id: str, command: str, exit_code: int, duration: float) -> None:
            self._job.add_event("command_complete", {"command": command, "exit_code": exit_code, "duration": round(duration, 3)})

        def on_progress(self, job_id: str, completed: int, total: int, current_command: str) -> None:
            self._job.update_progress(completed, total, current_command)
            self._job.add_event("progress", {"completed": completed, "total": total, "progress_pct": round(self._job.progress, 1)})

        def on_heartbeat(self, job_id: str, elapsed_sec: float, eta_sec: float | None, state: str) -> None:
            self._job.elapsed_sec = elapsed_sec
            self._job.eta_sec = eta_sec
            self._job.add_event("heartbeat", {"elapsed_sec": round(elapsed_sec, 1), "eta_sec": round(eta_sec, 1) if eta_sec else None, "state": state})

        def on_cache_hit(self, job_id: str, command: str) -> None:
            self._job.add_event("cache_hit", {"command": command})

        def on_skip(self, job_id: str, command: str, reason: str) -> None:
            self._job.add_event("command_skipped", {"command": command, "reason": reason})

        def on_error(self, job_id: str, command: str | None, error: str) -> None:
            self._job.add_event("error", {"command": command, "error": error})

        def on_complete(self, job_id: str, status: str, exit_code: int) -> None:
            self._job.add_event("job_completed", {"status": status, "exit_code": exit_code})

    @server.tool(
        name="accel_verify_start",
        description="Start an async verification job. Returns job_id immediately.",
    )
    def accel_verify_start(
        project: str = ".",
        changed_files: Any = None,
        evidence_run: Any = False,
        fast_loop: Any = False,
        verify_workers: Any = None,
        per_command_timeout_seconds: Any = None,
        verify_fail_fast: bool | str | None = None,
        verify_cache_enabled: bool | str | None = None,
        verify_cache_ttl_seconds: Any = None,
        verify_cache_max_entries: Any = None,
    ) -> JSONDict:
        try:
            job = _start_verify_job(
                project=project,
                changed_files=changed_files,
                evidence_run=evidence_run,
                fast_loop=fast_loop,
                verify_workers=verify_workers,
                per_command_timeout_seconds=per_command_timeout_seconds,
                verify_fail_fast=verify_fail_fast,
                verify_cache_enabled=verify_cache_enabled,
                verify_cache_ttl_seconds=verify_cache_ttl_seconds,
                verify_cache_max_entries=verify_cache_max_entries,
            )
            _debug_log(f"accel_verify_start created job: {job.job_id}")
            return {"job_id": job.job_id, "status": "started"}

        except Exception as exc:
            _debug_log(f"accel_verify_start failed: {exc!r}")
            raise RuntimeError(f"{TOOL_ERROR_EXECUTION_FAILED}: {exc}") from exc

    @server.tool(
        name="accel_verify_status",
        description="Get current status of a verification job.",
    )
    def accel_verify_status(job_id: str) -> JSONDict:
        try:
            _debug_log(f"accel_verify_status called: job_id={job_id}")
            jm = JobManager()
            job = jm.get_job(job_id)
            if job is None:
                return {"error": "job_not_found", "job_id": job_id}
            return job.to_status()
        except Exception as exc:
            _debug_log(f"accel_verify_status failed: {exc!r}")
            raise RuntimeError(f"{TOOL_ERROR_EXECUTION_FAILED}: {exc}") from exc

    @server.tool(
        name="accel_verify_events",
        description="Get events for a verification job since given sequence number.",
    )
    def accel_verify_events(job_id: str, since_seq: int = 0) -> JSONDict:
        try:
            _debug_log(f"accel_verify_events called: job_id={job_id}, since_seq={since_seq}")
            jm = JobManager()
            job = jm.get_job(job_id)
            if job is None:
                return {"error": "job_not_found", "job_id": job_id}
            events = job.get_events(since_seq)
            return {"job_id": job_id, "events": events, "count": len(events)}
        except Exception as exc:
            _debug_log(f"accel_verify_events failed: {exc!r}")
            raise RuntimeError(f"{TOOL_ERROR_EXECUTION_FAILED}: {exc}") from exc

    @server.tool(
        name="accel_verify_cancel",
        description="Cancel a running verification job.",
    )
    def accel_verify_cancel(job_id: str) -> JSONDict:
        try:
            _debug_log(f"accel_verify_cancel called: job_id={job_id}")
            jm = JobManager()
            job = jm.get_job(job_id)
            if job is None:
                return {"error": "job_not_found", "job_id": job_id}
            if job.state in (JobState.COMPLETED, JobState.FAILED, JobState.CANCELLED):
                return {"job_id": job_id, "status": job.state, "cancelled": False, "message": "Job already in terminal state"}
            jm.cancel_job(job_id)
            job.add_event("job_cancel_requested", {"reason": "user_request"})
            # Finalize state immediately so status polling converges deterministically.
            job.mark_cancelled()
            job.add_event("job_cancelled_finalized", {"reason": "user_request"})
            return {"job_id": job_id, "status": JobState.CANCELLED, "cancelled": True, "message": "Cancellation requested and finalized"}
        except Exception as exc:
            _debug_log(f"accel_verify_cancel failed: {exc!r}")
            raise RuntimeError(f"{TOOL_ERROR_EXECUTION_FAILED}: {exc}") from exc

    @server.resource(
        "agent-accel://status",
        name="agent-accel-status",
        description="Static service status payload for MCP compatibility probes.",
    )
    def status_resource() -> str:
        payload = {
            "name": SERVER_NAME,
            "version": SERVER_VERSION,
            "transport": "stdio",
            "framework": "fastmcp",
            "pid": os.getpid(),
            "module_path": str(Path(__file__).resolve()),
        }
        return json.dumps(payload, ensure_ascii=False)

    @server.resource(
        "agent-accel://template/{kind}",
        name="agent-accel-template",
        description="Return lightweight template payloads for supported task kinds.",
    )
    def template_resource(kind: str) -> str:
        key = str(kind or "").strip().lower()
        templates: dict[str, JSONDict] = {
            "verify": {
                "project": ".",
                "changed_files": [],
                "evidence_run": True,
            },
            "context": {
                "project": ".",
                "task": "Describe the requested code change",
                "changed_files": [],
            },
        }
        return json.dumps(templates.get(key, {"kind": key, "template": None}), ensure_ascii=False)

    return server


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="accel-mcp",
        description="agent-accel MCP server powered by FastMCP (stdio transport)",
    )
    parser.add_argument(
        "--transport",
        choices=["stdio"],
        default="stdio",
        help="MCP transport mode. Only stdio is supported in this deployment.",
    )
    parser.add_argument(
        "--show-banner",
        action="store_true",
        default=False,
        help="Display FastMCP startup banner.",
    )
    return parser


def main() -> None:
    global _server_start_time
    
    # Setup signal handlers for graceful shutdown
    signal.signal(signal.SIGTERM, _signal_handler)
    signal.signal(signal.SIGINT, _signal_handler)
    if hasattr(signal, 'SIGBREAK'):  # Windows-specific
        signal.signal(signal.SIGBREAK, _signal_handler)
    
    args = build_parser().parse_args()
    
    if _debug_enabled:
        _debug_log(f"Starting agent-accel MCP server with args: {vars(args)}")
        print("agent-accel MCP debug mode enabled. Logs will be written to ~/.accel/logs/", file=sys.stderr)
        print(f"Server max runtime: {_server_max_runtime}s", file=sys.stderr)
    
    server = create_server()
    _server_start_time = time.perf_counter()
    
    try:
        _debug_log(f"Running FastMCP server with transport: {args.transport}")
        
        # Run server with timeout monitoring
        # Note: FastMCP's server.run() is blocking, so we can't directly add timeout here
        # But we have runtime checks in tool wrappers
        server.run(transport=args.transport, show_banner=bool(args.show_banner))
        
    except KeyboardInterrupt:
        _debug_log("Server interrupted by user")
    except Exception as exc:
        _debug_log(f"Server crashed: {exc!r}")
        raise
    finally:
        _debug_log("Server shutdown complete")


if __name__ == "__main__":
    main()
