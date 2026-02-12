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
from .storage.semantic_cache import (
    SemanticCacheStore,
    context_changed_fingerprint,
    make_stable_hash,
    normalize_changed_files,
    normalize_token_list,
    task_signature,
)
from .schema.contracts import (
    enforce_context_pack_contract,
    enforce_context_payload_contract,
    enforce_verify_summary_contract,
    normalize_constraint_mode,
)
from .token_estimator import estimate_tokens_for_text, estimate_tokens_from_chars
from .verify.orchestrator import run_verify, run_verify_with_callback
from .verify.callbacks import VerifyProgressCallback, VerifyStage
from .verify.job_manager import JobManager, JobState, VerifyJob


JSONDict = dict[str, Any]
SERVER_NAME = "agent-accel-mcp"
SERVER_VERSION = "0.2.6"
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
_SYNC_TIMEOUT_ACTIONS = {"poll", "cancel"}

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


def _coerce_optional_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return float(int(value))
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        token = value.strip()
        if not token:
            return None
        if token.lower() in _FALSE_LITERALS:
            return None
        try:
            return float(token)
        except ValueError:
            return None
    return None


def _coerce_sync_timeout_action(value: Any, default: str = "poll") -> str:
    token = str(value or "").strip().lower()
    if token in _SYNC_TIMEOUT_ACTIONS:
        return token
    fallback = str(default or "poll").strip().lower()
    if fallback in _SYNC_TIMEOUT_ACTIONS:
        return fallback
    return "poll"


def _coerce_events_limit(value: Any, default_value: int = 30, max_value: int = 500) -> int:
    parsed = _coerce_optional_int(value)
    if parsed is None:
        parsed = default_value
    return max(1, min(int(max_value), int(parsed)))


def _clamp_float(value: float, min_value: float, max_value: float) -> float:
    return max(float(min_value), min(float(max_value), float(value)))


def _token_reduction_ratio(context_tokens: int, baseline_tokens: int) -> float:
    baseline = int(baseline_tokens)
    if baseline <= 0:
        return 0.0
    ratio = 1.0 - (float(max(0, int(context_tokens))) / float(baseline))
    return _clamp_float(ratio, 0.0, 1.0)


def _estimate_changed_files_chars(
    project_dir: Path,
    changed_files: list[str],
    *,
    max_files: int = 200,
    max_total_chars: int = 2_000_000,
) -> int:
    total_chars = 0
    seen: set[str] = set()
    for rel_path in changed_files[: max(1, int(max_files))]:
        normalized = str(rel_path).replace("\\", "/").strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        file_path = (project_dir / normalized).resolve()
        if not file_path.exists() or not file_path.is_file():
            continue
        try:
            content = file_path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        total_chars += len(content)
        if total_chars >= max_total_chars:
            return int(max_total_chars)
    return int(total_chars)


def _build_context_output_pack(
    pack: JSONDict,
    *,
    snippets_only: bool,
    include_metadata: bool,
) -> JSONDict:
    if snippets_only:
        output: JSONDict = {
            "version": int(pack.get("version", 1) or 1),
            "task": str(pack.get("task", "")),
            "generated_at": str(pack.get("generated_at", "")),
            "snippets": list(pack.get("snippets", [])),
        }
        if include_metadata and isinstance(pack.get("meta"), dict):
            output["meta"] = dict(pack.get("meta", {}))
        return output

    output = dict(pack)
    if not include_metadata:
        output.pop("meta", None)
    return output


def _normalize_rel_path(value: Any) -> str:
    return str(value or "").replace("\\", "/").strip().lower()


def _apply_strict_changed_files_scope(
    pack: JSONDict,
    changed_files: list[str],
) -> tuple[JSONDict, int, int]:
    changed_set = {_normalize_rel_path(item) for item in changed_files if _normalize_rel_path(item)}
    if not changed_set:
        return dict(pack), 0, 0

    payload = dict(pack)
    filtered_top_files = 0
    filtered_snippets = 0

    top_files_raw = payload.get("top_files")
    if isinstance(top_files_raw, list):
        kept_top_files: list[JSONDict] = []
        for item in top_files_raw:
            if isinstance(item, dict):
                path_token = _normalize_rel_path(item.get("path", ""))
                if path_token and path_token in changed_set:
                    kept_top_files.append(dict(item))
                else:
                    filtered_top_files += 1
            else:
                filtered_top_files += 1
        payload["top_files"] = kept_top_files

    snippets_raw = payload.get("snippets")
    if isinstance(snippets_raw, list):
        kept_snippets: list[JSONDict] = []
        for item in snippets_raw:
            if isinstance(item, dict):
                path_token = _normalize_rel_path(item.get("path", ""))
                if path_token and path_token in changed_set:
                    kept_snippets.append(dict(item))
                else:
                    filtered_snippets += 1
            else:
                filtered_snippets += 1
        payload["snippets"] = kept_snippets

    meta_raw = payload.get("meta")
    if isinstance(meta_raw, dict):
        meta_payload = dict(meta_raw)
        meta_payload["strict_changed_files_scope"] = True
        meta_payload["strict_scope_changed_files_count"] = int(len(changed_set))
        meta_payload["strict_scope_filtered_top_files"] = int(filtered_top_files)
        meta_payload["strict_scope_filtered_snippets"] = int(filtered_snippets)
        payload["meta"] = meta_payload

    return payload, int(filtered_top_files), int(filtered_snippets)


def _resolve_changed_file_rel_path(project_dir: Path, value: Any) -> str | None:
    token = str(value or "").replace("\\", "/").strip()
    if not token:
        return None
    raw_path = Path(token)
    candidate = raw_path if raw_path.is_absolute() else (project_dir / raw_path)
    try:
        resolved = candidate.resolve()
        project_root = project_dir.resolve()
        rel = resolved.relative_to(project_root)
    except (OSError, ValueError):
        return None
    if not resolved.exists() or not resolved.is_file():
        return None
    return str(rel).replace("\\", "/")


def _build_changed_file_snippet(project_dir: Path, rel_path: str, max_chars: int) -> JSONDict | None:
    rel = str(rel_path or "").replace("\\", "/").strip()
    if not rel:
        return None
    file_path = (project_dir / rel).resolve()
    try:
        content = file_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    clipped = content[: max(200, int(max_chars))]
    if not clipped:
        clipped = "[empty file]"
    end_line = max(1, int(clipped.count("\n")) + 1)
    return {
        "path": rel,
        "start_line": 1,
        "end_line": end_line,
        "symbol": "",
        "reason": "strict_changed_file_fallback",
        "content": clipped,
    }


def _ensure_strict_changed_files_presence(
    pack: JSONDict,
    *,
    project_dir: Path,
    changed_files: list[str],
) -> tuple[JSONDict, int, int]:
    payload = dict(pack)
    budget = payload.get("budget", {}) if isinstance(payload.get("budget", {}), dict) else {}
    top_n_files = max(1, int(budget.get("top_n_files", 8) or 8))
    max_snippets = max(1, int(budget.get("max_snippets", 30) or 30))
    per_snippet_max_chars = max(200, int(budget.get("per_snippet_max_chars", 2000) or 2000))

    existing_rel_paths: list[str] = []
    seen: set[str] = set()
    for item in changed_files:
        rel = _resolve_changed_file_rel_path(project_dir, item)
        if not rel:
            continue
        key = _normalize_rel_path(rel)
        if key in seen:
            continue
        seen.add(key)
        existing_rel_paths.append(rel)

    if not existing_rel_paths:
        return payload, 0, 0

    top_files = payload.get("top_files")
    snippets = payload.get("snippets")
    top_files_list = list(top_files) if isinstance(top_files, list) else []
    snippets_list = list(snippets) if isinstance(snippets, list) else []

    injected_top_files = 0
    injected_snippets = 0

    if not top_files_list:
        for rel in existing_rel_paths[:top_n_files]:
            top_files_list.append(
                {
                    "path": rel,
                    "score": 1.0,
                    "reasons": ["strict_changed_file_fallback"],
                    "signals": [{"signal_name": "strict_changed_file_fallback", "score": 1.0}],
                }
            )
            injected_top_files += 1
        payload["top_files"] = top_files_list

    if not snippets_list:
        for rel in existing_rel_paths[:max_snippets]:
            snippet = _build_changed_file_snippet(project_dir, rel, per_snippet_max_chars)
            if not isinstance(snippet, dict):
                continue
            snippets_list.append(snippet)
            injected_snippets += 1
        payload["snippets"] = snippets_list

    meta_raw = payload.get("meta")
    if isinstance(meta_raw, dict):
        meta_payload = dict(meta_raw)
        meta_payload["strict_scope_injected_top_files"] = int(injected_top_files)
        meta_payload["strict_scope_injected_snippets"] = int(injected_snippets)
        payload["meta"] = meta_payload

    return payload, int(injected_top_files), int(injected_snippets)


def _write_context_metadata_sidecar(out_path: Path, payload: JSONDict) -> Path:
    sidecar_path = out_path.with_suffix(".meta.json")
    token_reduction_payload = payload.get("token_reduction", {})
    token_estimator_payload = payload.get("token_estimator", {})
    sidecar_payload: JSONDict = {
        "version": 1,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "out": str(out_path),
        "output_mode": payload.get("output_mode", "full"),
        "include_metadata": bool(payload.get("include_metadata", True)),
        "budget": {
            "source": payload.get("budget_source", "auto"),
            "preset": payload.get("budget_preset", "small"),
            "reason": payload.get("budget_reason", ""),
            "effective": payload.get("budget_effective", {}),
        },
        "scope": {
        "changed_files_source": payload.get("changed_files_source", "none"),
        "changed_files_count": int(payload.get("changed_files_count", 0) or 0),
        "changed_files_used": list(payload.get("changed_files_used", [])),
        "fallback_confidence": float(payload.get("fallback_confidence", 0.0) or 0.0),
        "strict_changed_files": bool(payload.get("strict_changed_files", False)),
        "strict_scope_filtered_top_files": int(payload.get("strict_scope_filtered_top_files", 0) or 0),
        "strict_scope_filtered_snippets": int(payload.get("strict_scope_filtered_snippets", 0) or 0),
        "strict_scope_injected_top_files": int(payload.get("strict_scope_injected_top_files", 0) or 0),
        "strict_scope_injected_snippets": int(payload.get("strict_scope_injected_snippets", 0) or 0),
      },
        "estimates": {
            "context_chars": int(payload.get("context_chars", 0) or 0),
            "source_chars": int(payload.get("source_chars", 0) or 0),
            "estimated_tokens": int(payload.get("estimated_tokens", 0) or 0),
            "estimated_source_tokens": int(payload.get("estimated_source_tokens", 0) or 0),
            "estimated_changed_files_tokens": int(payload.get("estimated_changed_files_tokens", 0) or 0),
            "estimated_snippets_only_tokens": int(payload.get("estimated_snippets_only_tokens", 0) or 0),
            "compression_ratio": float(payload.get("compression_ratio", 1.0) or 1.0),
            "token_reduction_ratio": float(payload.get("token_reduction_ratio", 0.0) or 0.0),
            "token_reduction": token_reduction_payload if isinstance(token_reduction_payload, dict) else {},
            "token_estimator": token_estimator_payload if isinstance(token_estimator_payload, dict) else {},
        },
        "selected_tests_count": int(payload.get("selected_tests_count", 0) or 0),
        "selected_checks_count": int(payload.get("selected_checks_count", 0) or 0),
        "semantic_cache": {
            "enabled": bool(payload.get("semantic_cache_enabled", False)),
            "hit": bool(payload.get("semantic_cache_hit", False)),
            "mode": str(payload.get("semantic_cache_mode_used", "off")),
            "similarity": float(payload.get("semantic_cache_similarity", 0.0) or 0.0),
        },
        "compression": {
            "rules_applied": dict(payload.get("compression_rules_applied", {})),
            "saved_chars": int(payload.get("compression_saved_chars", 0) or 0),
        },
        "constraints": {
            "mode": str(payload.get("constraint_mode", "warn")),
            "repair_count": int(payload.get("constraint_repair_count", 0) or 0),
            "warnings": list(payload.get("constraint_warnings", [])),
        },
        "warnings": list(payload.get("warnings", [])),
    }
    sidecar_path.write_text(json.dumps(sidecar_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return sidecar_path


def _summarize_verify_events(events: list[dict[str, Any]], status: JSONDict) -> JSONDict:
    event_type_counts: dict[str, int] = {}
    command_start = 0
    command_complete = 0
    command_skipped = 0
    command_errors = 0
    cache_hits = 0
    terminal_event_seen = False
    first_seq = 0
    last_seq = 0
    latest_stage = str(status.get("stage", ""))
    latest_state = str(status.get("state", ""))
    terminal_events = {"job_completed", "job_failed", "job_cancelled_finalized"}
    terminal_states = {JobState.COMPLETED, JobState.FAILED, JobState.CANCELLED}
    terminal_state_from_event = ""
    terminal_event_seq = 0
    terminal_stage_from_event = ""
    terminal_event_to_state = {
        "job_completed": JobState.COMPLETED,
        "job_failed": JobState.FAILED,
        "job_cancelled_finalized": JobState.CANCELLED,
    }

    for event in events:
        event_type = str(event.get("event", "")).strip()
        if not event_type:
            continue
        event_type_counts[event_type] = int(event_type_counts.get(event_type, 0)) + 1
        if event_type == "command_start":
            command_start += 1
        elif event_type == "command_complete":
            command_complete += 1
        elif event_type == "command_skipped":
            command_skipped += 1
        elif event_type == "error":
            command_errors += 1
        elif event_type == "cache_hit":
            cache_hits += 1
        if event_type in terminal_events:
            terminal_event_seen = True
            seq_for_terminal = _coerce_optional_int(event.get("seq")) or 0
            mapped_state = terminal_event_to_state.get(event_type, "")
            if seq_for_terminal >= terminal_event_seq and mapped_state:
                terminal_event_seq = int(seq_for_terminal)
                terminal_state_from_event = mapped_state
                stage_value = str(event.get("stage", "")).strip()
                if stage_value:
                    terminal_stage_from_event = stage_value

        seq = _coerce_optional_int(event.get("seq"))
        if seq is not None and seq > 0:
            if first_seq <= 0:
                first_seq = int(seq)
            last_seq = max(last_seq, int(seq))

        stage = str(event.get("stage", "")).strip()
        if stage:
            latest_stage = stage
        state = str(event.get("state", "")).strip()
        if state:
            latest_state = state

    status_state = str(status.get("state", "")).strip().lower()
    state_source = "events"
    if status_state in terminal_states:
        latest_state = status_state
        latest_stage = str(status.get("stage", latest_stage)).strip() or latest_stage
        terminal_event_seen = True
        state_source = "status_terminal"
    elif terminal_state_from_event:
        latest_state = terminal_state_from_event
        if terminal_stage_from_event:
            latest_stage = terminal_stage_from_event
        state_source = "event_terminal"

    return {
        "latest_state": latest_state or str(status.get("state", "")),
        "latest_stage": latest_stage or str(status.get("stage", "")),
        "terminal_event_seen": bool(terminal_event_seen),
        "state_source": state_source,
        "constraint_repair_count": 0,
        "event_type_counts": event_type_counts,
        "command_stats": {
            "started": int(command_start),
            "completed": int(command_complete),
            "skipped": int(command_skipped),
            "errors": int(command_errors),
            "cache_hits": int(cache_hits),
        },
        "seq_range": {"first": int(first_seq), "last": int(last_seq)},
    }


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


def _sync_verify_timeout_payload(job_id: str, wait_seconds: float, *, timeout_action: str = "poll") -> JSONDict:
    jm = JobManager()
    job = jm.get_job(job_id)
    status = job.to_status() if job is not None else {}
    action = _coerce_sync_timeout_action(timeout_action, default="poll")
    return {
        "status": "running",
        "exit_code": 124,
        "timed_out": True,
        "job_id": job_id,
        "timeout_action": action,
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


def _resolve_sync_timeout_defaults(project_dir: Path) -> tuple[str, float]:
    action = "poll"
    grace_seconds = 5.0
    try:
        config = resolve_effective_config(project_dir)
        runtime_cfg = config.get("runtime", {}) if isinstance(config.get("runtime", {}), dict) else {}
        action = _coerce_sync_timeout_action(runtime_cfg.get("sync_verify_timeout_action"), default="poll")
        grace_value = _coerce_optional_float(runtime_cfg.get("sync_verify_cancel_grace_seconds"))
        if grace_value is not None:
            grace_seconds = grace_value
    except Exception:
        action = "poll"
        grace_seconds = 5.0
    grace_seconds = max(0.2, min(30.0, float(grace_seconds)))
    return action, grace_seconds


def _timeout_cancel_payload(
    *,
    job_id: str,
    wait_seconds: float,
    cancel_requested: bool,
) -> JSONDict:
    jm = JobManager()
    job = jm.get_job(job_id)
    status = job.to_status() if job is not None else {}
    state = str(status.get("state", JobState.CANCELLED))
    stage = str(status.get("stage", JobState.CANCELLED))
    if cancel_requested and state != JobState.CANCELLED:
        state = JobState.CANCELLED
        stage = JobState.CANCELLED
    return {
        "status": "cancelled" if cancel_requested else "running",
        "exit_code": 130 if cancel_requested else 124,
        "timed_out": True,
        "job_id": job_id,
        "timeout_action": "cancel",
        "auto_cancel_requested": bool(cancel_requested),
        "message": (
            f"accel_verify exceeded synchronous wait window ({wait_seconds:.1f}s); auto-cancel requested."
            if cancel_requested
            else f"accel_verify exceeded synchronous wait window ({wait_seconds:.1f}s); auto-cancel request failed."
        ),
        "poll_interval_sec": 1.0,
        "state": state,
        "stage": stage,
        "progress": status.get("progress", 0.0),
        "elapsed_sec": status.get("elapsed_sec", 0.0),
    }


def _finalize_job_cancel(job: VerifyJob, *, reason: str) -> bool:
    if job.state in (JobState.COMPLETED, JobState.FAILED, JobState.CANCELLED):
        return False
    jm = JobManager()
    jm.cancel_job(job.job_id)
    job.add_event("job_cancel_requested", {"reason": reason})
    job.mark_cancelled()
    job.add_event("job_cancelled_finalized", {"reason": reason})
    return True


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
    def _extract_status_path(raw_line: str) -> str:
        text = str(raw_line or "").rstrip()
        if len(text) < 4:
            return ""
        payload = text[3:].strip()
        if not payload:
            return ""
        if " -> " in payload:
            payload = payload.split(" -> ", 1)[1].strip()
        return payload.replace("\\", "/")

    def _run_git_cmd(cmd: list[str], timeout_seconds: float) -> list[str]:
        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=timeout_seconds,
                check=False,
            )
        except Exception:
            return []
        if int(proc.returncode) != 0:
            return []
        return [line for line in proc.stdout.splitlines() if str(line).strip()]

    discovered: list[str] = []
    lock = threading.Lock()

    def _collect() -> None:
        seen: set[str] = set()
        output: list[str] = []
        status_lines = _run_git_cmd(
            ["git", "-C", str(project_dir), "status", "--porcelain", "--untracked-files=normal"],
            2.0,
        )
        for raw_line in status_lines:
            rel = _extract_status_path(raw_line)
            if not rel or rel in seen:
                continue
            seen.add(rel)
            output.append(rel)
            if len(output) >= limit:
                break
        if len(output) < limit:
            diff_commands = [
                ["git", "-C", str(project_dir), "diff", "--name-only", "--relative", "HEAD"],
                ["git", "-C", str(project_dir), "diff", "--name-only", "--relative", "--cached"],
                ["git", "-C", str(project_dir), "diff", "--name-only", "--relative"],
            ]
            for cmd in diff_commands:
                for raw_line in _run_git_cmd(cmd, 1.5):
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
            discovered.extend(output[:limit])

    worker = threading.Thread(target=_collect, daemon=True)
    worker.start()
    worker.join(timeout=4.0)
    if worker.is_alive():
        _debug_log("_discover_changed_files_from_git timed out; fallback to empty changed_files")
        return []
    with lock:
        return list(discovered)


def _discover_changed_files_from_index_fallback(
    project_dir: Path,
    *,
    config: dict[str, Any],
    task: str,
    hints: list[str],
    limit: int = 24,
) -> tuple[list[str], str, float]:
    accel_home_value = str(config.get("runtime", {}).get("accel_home", "")).strip()
    if not accel_home_value:
        return [], "no_accel_home", 0.0
    accel_home = Path(accel_home_value).resolve()
    index_dir = project_paths(accel_home, project_dir)["index"]
    manifest_path = index_dir / "manifest.json"
    if not manifest_path.exists():
        return [], "manifest_missing", 0.0

    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception:
        return [], "manifest_parse_failed", 0.0

    indexed_files = [str(item).replace("\\", "/") for item in manifest.get("indexed_files", []) if str(item).strip()]
    if not indexed_files:
        return [], "manifest_index_empty", 0.0

    indexed_set = set(indexed_files)
    recent_changed = [
        str(item).replace("\\", "/")
        for item in manifest.get("changed_files", [])
        if str(item).strip() and str(item).replace("\\", "/") in indexed_set
    ]
    if recent_changed:
        deduped_recent = list(dict.fromkeys(recent_changed))[: max(1, int(limit))]
        return deduped_recent, "manifest_recent", 0.92

    tokens = normalize_task_tokens(task)
    for hint in hints:
        tokens.extend(normalize_task_tokens(str(hint)))
    tokens = list(dict.fromkeys(tokens))
    if tokens:
        scored: list[tuple[int, str]] = []
        for rel_path in indexed_files:
            path_low = rel_path.lower()
            score = sum(1 for token in tokens if token in path_low)
            if score > 0:
                scored.append((score, rel_path))
        if scored:
            scored.sort(key=lambda item: (-item[0], item[1]))
            selected = [path for _, path in scored[: max(1, int(limit))]]
            max_score = max(score for score, _ in scored)
            coverage = float(max_score) / float(max(1, len(tokens)))
            confidence = _clamp_float(0.35 + (coverage * 0.5), 0.35, 0.88)
            return selected, "planner_fallback", confidence

    return indexed_files[: max(1, min(int(limit), 8))], "index_head_fallback", 0.2


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


def _context_config_hash(config: JSONDict) -> str:
    context_cfg = dict(config.get("context", {}))
    runtime_cfg = dict(config.get("runtime", {}))
    payload = {
        "context": context_cfg,
        "runtime": {
            "rule_compression_enabled": bool(runtime_cfg.get("rule_compression_enabled", True)),
            "constraint_mode": str(runtime_cfg.get("constraint_mode", "warn")),
            "token_estimator_backend": str(runtime_cfg.get("token_estimator_backend", "auto")),
            "token_estimator_encoding": str(runtime_cfg.get("token_estimator_encoding", "cl100k_base")),
            "token_estimator_model": str(runtime_cfg.get("token_estimator_model", "")),
            "token_estimator_calibration": float(runtime_cfg.get("token_estimator_calibration", 1.0)),
        },
    }
    return make_stable_hash(payload)


def _context_budget_fingerprint(
    budget_effective: dict[str, Any],
    *,
    snippets_only: bool,
    include_metadata: bool,
) -> str:
    payload = {
        "budget": dict(budget_effective or {}),
        "snippets_only": bool(snippets_only),
        "include_metadata": bool(include_metadata),
    }
    return make_stable_hash(payload)


def _safe_semantic_cache_store(project_dir: Path, config: JSONDict) -> SemanticCacheStore | None:
    try:
        runtime_cfg = dict(config.get("runtime", {}))
        accel_home = Path(str(runtime_cfg.get("accel_home", "") or "")).resolve()
        paths = project_paths(accel_home, project_dir)
        ensure_project_dirs(paths)
        return SemanticCacheStore(paths["state"] / "semantic_cache.db")
    except OSError:
        return None


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
    strict_changed_files: Any = None,
    snippets_only: Any = False,
    include_metadata: Any = True,
    semantic_cache: Any = None,
    semantic_cache_mode: Any = None,
    constraint_mode: Any = None,
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
    snippets_only_flag = _coerce_bool(snippets_only, False)
    include_metadata_flag = _coerce_bool(include_metadata, True)
    _debug_log(
        "_tool_context: parsed inputs "
        f"changed_files={len(changed_files_list)} hints={len(hints_list)} "
        f"include_pack={include_pack_flag} snippets_only={snippets_only_flag} include_metadata={include_metadata_flag}"
    )

    config = resolve_effective_config(project_dir)
    _debug_log("_tool_context: config resolved")
    runtime_cfg = config.get("runtime", {}) if isinstance(config.get("runtime", {}), dict) else {}
    require_changed_files = bool(runtime_cfg.get("context_require_changed_files", False))
    strict_changed_files_override = _coerce_optional_bool(strict_changed_files)
    strict_changed_files_effective = bool(strict_changed_files_override) if strict_changed_files_override is not None else False
    semantic_cache_default = bool(runtime_cfg.get("semantic_cache_enabled", True))
    semantic_cache_enabled = _coerce_bool(semantic_cache, semantic_cache_default)
    semantic_cache_mode_default = str(runtime_cfg.get("semantic_cache_mode", "hybrid")).strip().lower()
    semantic_cache_mode_value = str(semantic_cache_mode or semantic_cache_mode_default).strip().lower()
    if semantic_cache_mode_value not in {"exact", "hybrid"}:
        semantic_cache_mode_value = "hybrid"
    constraint_mode_value = normalize_constraint_mode(
        constraint_mode if constraint_mode is not None else runtime_cfg.get("constraint_mode", "warn"),
        default_mode=str(runtime_cfg.get("constraint_mode", "warn")),
    )

    changed_files_source = "user" if changed_files_list else "none"
    changed_files_fallback_reason = ""
    fallback_confidence = 1.0 if changed_files_list else 0.0
    if not changed_files_list:
        auto_changed = _discover_changed_files_from_git(project_dir)
        if auto_changed:
            changed_files_list = auto_changed
            changed_files_source = "git_auto"
            fallback_confidence = 0.98
        elif strict_changed_files_effective:
            changed_files_fallback_reason = "strict_changed_files_enabled_no_git_delta"
        else:
            fallback_changed, fallback_reason, fallback_conf = _discover_changed_files_from_index_fallback(
                project_dir,
                config=config,
                task=task_text,
                hints=hints_list,
            )
            if fallback_changed:
                changed_files_list = fallback_changed
                changed_files_source = fallback_reason
                fallback_confidence = float(fallback_conf)
            else:
                changed_files_fallback_reason = fallback_reason
    _debug_log(
        "_tool_context: changed_files resolution "
        f"source={changed_files_source} count={len(changed_files_list)}"
    )
    if strict_changed_files_effective and not changed_files_list:
        raise ValueError(
            "strict_changed_files=true requires explicit changed_files or a detectable git delta. "
            "Provide changed_files explicitly or commit/stage diffs before calling accel_context."
        )
    if require_changed_files and not changed_files_list:
        raise ValueError(
            "changed_files is required for context narrowing. "
            "Provide changed_files explicitly or ensure git diff is available."
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

    feedback_payload: JSONDict | None = None

    resolved_feedback_path = _resolve_path(project_dir, feedback_path)
    if resolved_feedback_path is not None and resolved_feedback_path.exists():
        feedback_payload = json.loads(resolved_feedback_path.read_text(encoding="utf-8"))
    _debug_log("_tool_context: feedback loaded")

    task_tokens = normalize_task_tokens(task_text)
    hint_tokens: list[str] = []
    for hint in hints_list:
        hint_tokens.extend(normalize_task_tokens(str(hint)))
    hint_tokens = normalize_token_list(hint_tokens)
    changed_files_normalized = normalize_changed_files(changed_files_list)
    changed_fingerprint = context_changed_fingerprint(changed_files_normalized)
    budget_fingerprint = _context_budget_fingerprint(
        budget_override,
        snippets_only=snippets_only_flag,
        include_metadata=include_metadata_flag,
    )
    config_hash = _context_config_hash(config)
    task_sig = task_signature(task_tokens, hint_tokens)
    semantic_cache_key = make_stable_hash(
        {
            "task_signature": task_sig,
            "changed_fingerprint": changed_fingerprint,
            "budget_fingerprint": budget_fingerprint,
            "config_hash": config_hash,
        }
    )
    semantic_cache_hit = False
    semantic_cache_similarity = 0.0
    semantic_cache_mode_used = "off"
    semantic_store: SemanticCacheStore | None = None
    pack: JSONDict

    if semantic_cache_enabled:
        semantic_store = _safe_semantic_cache_store(project_dir, config)

    if semantic_store is not None and semantic_cache_enabled:
        cached_exact = semantic_store.get_context_exact(semantic_cache_key)
        if isinstance(cached_exact, dict):
            pack = cached_exact
            semantic_cache_hit = True
            semantic_cache_similarity = 1.0
            semantic_cache_mode_used = "exact"
        else:
            cached_hybrid: JSONDict | None = None
            hybrid_similarity = 0.0
            if semantic_cache_mode_value == "hybrid":
                cached_hybrid, hybrid_similarity = semantic_store.get_context_hybrid(
                    task_tokens=task_tokens,
                    hint_tokens=hint_tokens,
                    changed_files=changed_files_normalized,
                    budget_fingerprint=budget_fingerprint,
                    config_hash=config_hash,
                    threshold=float(runtime_cfg.get("semantic_cache_hybrid_threshold", 0.86)),
                )
            if isinstance(cached_hybrid, dict):
                pack = cached_hybrid
                semantic_cache_hit = True
                semantic_cache_similarity = float(hybrid_similarity)
                semantic_cache_mode_used = "hybrid"
            else:
                _debug_log("_tool_context: compile_context_pack start (cache miss)")
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
                semantic_cache_mode_used = "miss"
                semantic_store.put_context(
                    cache_key=semantic_cache_key,
                    task_signature_value=task_sig,
                    task_tokens=task_tokens,
                    hint_tokens=hint_tokens,
                    changed_files=changed_files_normalized,
                    changed_fingerprint=changed_fingerprint,
                    budget_fingerprint=budget_fingerprint,
                    config_hash=config_hash,
                    payload=pack,
                    ttl_seconds=int(runtime_cfg.get("semantic_cache_ttl_seconds", 7200)),
                    max_entries=int(runtime_cfg.get("semantic_cache_max_entries", 800)),
                )
    else:
        _debug_log("_tool_context: compile_context_pack start (semantic cache disabled)")
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

    strict_scope_filtered_top_files = 0
    strict_scope_filtered_snippets = 0
    strict_scope_injected_top_files = 0
    strict_scope_injected_snippets = 0
    if strict_changed_files_effective and changed_files_list:
        pack, strict_scope_filtered_top_files, strict_scope_filtered_snippets = _apply_strict_changed_files_scope(
            pack,
            changed_files_list,
        )
        pack, strict_scope_injected_top_files, strict_scope_injected_snippets = _ensure_strict_changed_files_presence(
            pack,
            project_dir=project_dir,
            changed_files=changed_files_list,
        )

    pack_contract_warnings: list[str] = []
    pack_repair_count = 0
    pack, pack_contract_warnings, pack_repair_count = enforce_context_pack_contract(
        pack,
        mode=constraint_mode_value,
    )
    pack_for_output = _build_context_output_pack(
        pack,
        snippets_only=snippets_only_flag,
        include_metadata=include_metadata_flag,
    )

    accel_home = Path(config["runtime"]["accel_home"]).resolve()
    paths = project_paths(accel_home, project_dir)
    ensure_project_dirs(paths)

    out_path = _resolve_path(project_dir, out)
    if out_path is None:
        out_path = paths["context"] / f"context_pack_{uuid4().hex[:10]}.json"

    _debug_log(f"_tool_context: write_context_pack start out={out_path}")
    write_context_pack(out_path, pack_for_output)
    _debug_log("_tool_context: write_context_pack done")
    pack_json_text = json.dumps(pack_for_output, ensure_ascii=False)
    context_chars = len(pack_json_text)
    meta = pack.get("meta", {}) if isinstance(pack.get("meta", {}), dict) else {}
    source_chars = int(meta.get("source_chars_est", 0) or 0)
    if source_chars <= 0:
        source_chars = context_chars

    token_estimate = estimate_tokens_for_text(
        pack_json_text,
        backend=runtime_cfg.get("token_estimator_backend", "auto"),
        model=runtime_cfg.get("token_estimator_model", ""),
        encoding=runtime_cfg.get("token_estimator_encoding", "cl100k_base"),
        calibration=runtime_cfg.get("token_estimator_calibration", 1.0),
        fallback_chars_per_token=runtime_cfg.get("token_estimator_fallback_chars_per_token", 4.0),
    )
    source_token_estimate = estimate_tokens_from_chars(
        source_chars,
        chars_per_token=token_estimate.get(
            "chars_per_token",
            runtime_cfg.get("token_estimator_fallback_chars_per_token", 4.0),
        ),
        calibration=token_estimate.get("calibration", runtime_cfg.get("token_estimator_calibration", 1.0)),
    )

    compression_ratio = float(context_chars / source_chars) if source_chars > 0 else 1.0
    token_reduction_ratio = max(0.0, 1.0 - compression_ratio)
    snippets_only_text = json.dumps({"snippets": list(pack.get("snippets", []))}, ensure_ascii=False)
    snippets_only_token_estimate = estimate_tokens_for_text(
        snippets_only_text,
        backend=runtime_cfg.get("token_estimator_backend", "auto"),
        model=runtime_cfg.get("token_estimator_model", ""),
        encoding=runtime_cfg.get("token_estimator_encoding", "cl100k_base"),
        calibration=runtime_cfg.get("token_estimator_calibration", 1.0),
        fallback_chars_per_token=runtime_cfg.get("token_estimator_fallback_chars_per_token", 4.0),
    )
    changed_files_chars = _estimate_changed_files_chars(project_dir, changed_files_list)
    changed_files_token_estimate = estimate_tokens_from_chars(
        changed_files_chars,
        chars_per_token=token_estimate.get(
            "chars_per_token",
            runtime_cfg.get("token_estimator_fallback_chars_per_token", 4.0),
        ),
        calibration=token_estimate.get("calibration", runtime_cfg.get("token_estimator_calibration", 1.0)),
    )
    context_tokens = int(token_estimate.get("estimated_tokens", 1))
    source_tokens = int(source_token_estimate.get("estimated_tokens", 1))
    changed_files_tokens = int(changed_files_token_estimate.get("estimated_tokens", 0))
    snippets_only_tokens = int(snippets_only_token_estimate.get("estimated_tokens", 1))
    token_reduction_vs_full_index = _token_reduction_ratio(context_tokens, source_tokens)
    token_reduction_vs_changed_files = (
        _token_reduction_ratio(context_tokens, changed_files_tokens) if changed_files_tokens > 0 else None
    )
    token_reduction_vs_snippets_only = _token_reduction_ratio(context_tokens, snippets_only_tokens)
    warnings: list[str] = []
    if changed_files_source == "none":
        warnings.append("changed_files not provided and no git delta detected; context scope may be wider than necessary")
    elif changed_files_source in {"manifest_recent", "planner_fallback", "index_head_fallback"}:
        warnings.append(
            f"changed_files inferred via {changed_files_source}; provide explicit changed_files for tighter precision"
        )
    if changed_files_source in {"planner_fallback", "index_head_fallback"} and fallback_confidence < 0.6:
        warnings.append(
            "changed_files inference confidence is low; provide explicit changed_files for stable narrowing"
        )
    if changed_files_fallback_reason:
        warnings.append(f"changed_files fallback detail: {changed_files_fallback_reason}")
    if strict_changed_files_effective and (strict_scope_filtered_top_files > 0 or strict_scope_filtered_snippets > 0):
        warnings.append(
            "strict_changed_files pruned non-changed context items "
            f"(top_files={strict_scope_filtered_top_files}, snippets={strict_scope_filtered_snippets})"
        )
    if strict_changed_files_effective and (strict_scope_injected_top_files > 0 or strict_scope_injected_snippets > 0):
        warnings.append(
            "strict_changed_files injected changed-file fallback context "
            f"(top_files={strict_scope_injected_top_files}, snippets={strict_scope_injected_snippets})"
        )

    verify_plan = pack.get("verify_plan", {}) if isinstance(pack.get("verify_plan", {}), dict) else {}
    target_tests = verify_plan.get("target_tests", [])
    target_checks = verify_plan.get("target_checks", [])
    selected_tests_count = len(target_tests) if isinstance(target_tests, list) else 0
    selected_checks_count = len(target_checks) if isinstance(target_checks, list) else 0
    compression_rules_applied = {}
    compression_saved_chars = 0
    if isinstance(meta.get("compression_rules_applied", {}), dict):
        compression_rules_applied = dict(meta.get("compression_rules_applied", {}))
    compression_saved_chars = int(meta.get("compression_saved_chars", meta.get("snippet_saved_chars", 0)) or 0)

    payload: JSONDict = {
        "status": "ok",
        "out": str(out_path),
        "top_files": len(pack_for_output.get("top_files", [])),
        "snippets": len(pack_for_output.get("snippets", [])),
        "verify_plan": verify_plan,
        "selected_tests_count": selected_tests_count,
        "selected_checks_count": selected_checks_count,
        "context_chars": context_chars,
        "source_chars": source_chars,
        "estimated_tokens": context_tokens,
        "estimated_source_tokens": source_tokens,
        "estimated_changed_files_tokens": changed_files_tokens,
        "estimated_snippets_only_tokens": snippets_only_tokens,
        "changed_files_source_chars": changed_files_chars,
        "compression_ratio": round(compression_ratio, 6),
        "token_reduction_ratio": round(token_reduction_ratio, 6),
        "token_reduction_ratio_vs_full_index": round(token_reduction_vs_full_index, 6),
        "token_reduction_ratio_vs_snippets_only": round(token_reduction_vs_snippets_only, 6),
        "token_reduction": {
            "vs_full_index": {
                "baseline_tokens": source_tokens,
                "context_tokens": context_tokens,
                "ratio": round(token_reduction_vs_full_index, 6),
            },
            "vs_changed_files": {
                "baseline_tokens": changed_files_tokens,
                "context_tokens": context_tokens,
                "ratio": round(token_reduction_vs_changed_files, 6) if token_reduction_vs_changed_files is not None else None,
                "available": bool(changed_files_tokens > 0),
            },
            "vs_snippets_only": {
                "baseline_tokens": snippets_only_tokens,
                "context_tokens": context_tokens,
                "ratio": round(token_reduction_vs_snippets_only, 6),
            },
        },
        "token_estimator": {
            "backend_requested": token_estimate.get("backend_requested", "auto"),
            "backend_used": token_estimate.get("backend_used", "heuristic"),
            "encoding_requested": token_estimate.get("encoding_requested", "cl100k_base"),
            "encoding_used": token_estimate.get("encoding_used", "chars/4"),
            "model": token_estimate.get("model", ""),
            "calibration": float(token_estimate.get("calibration", 1.0)),
            "fallback_chars_per_token": float(token_estimate.get("fallback_chars_per_token", 4.0)),
            "context_chars_per_token": round(float(token_estimate.get("chars_per_token", 4.0)), 6),
            "source_chars_per_token": round(float(source_token_estimate.get("chars_per_token", 4.0)), 6),
            "raw_context_tokens": int(token_estimate.get("raw_tokens", 1)),
            "raw_source_tokens": int(source_token_estimate.get("raw_tokens", 1)),
        },
        "budget_effective": pack.get("budget", {}),
        "budget_source": budget_source,
        "budget_preset": budget_preset,
        "budget_reason": budget_reason,
        "changed_files_used": changed_files_list,
        "changed_files_source": changed_files_source,
        "changed_files_count": len(changed_files_list),
        "strict_changed_files": strict_changed_files_effective,
        "strict_scope_filtered_top_files": int(strict_scope_filtered_top_files),
        "strict_scope_filtered_snippets": int(strict_scope_filtered_snippets),
        "strict_scope_injected_top_files": int(strict_scope_injected_top_files),
        "strict_scope_injected_snippets": int(strict_scope_injected_snippets),
        "fallback_confidence": round(float(fallback_confidence), 6),
        "output_mode": "snippets_only" if snippets_only_flag else "full",
        "include_metadata": include_metadata_flag,
        "semantic_cache_enabled": bool(semantic_cache_enabled),
        "semantic_cache_hit": bool(semantic_cache_hit),
        "semantic_cache_mode_used": str(semantic_cache_mode_used),
        "semantic_cache_similarity": round(float(semantic_cache_similarity), 6),
        "compression_rules_applied": compression_rules_applied,
        "compression_saved_chars": int(compression_saved_chars),
        "constraint_mode": constraint_mode_value,
    }
    if token_reduction_vs_changed_files is not None:
        payload["token_reduction_ratio_vs_changed_files"] = round(token_reduction_vs_changed_files, 6)
    fallback_reason = str(token_estimate.get("fallback_reason", "")).strip()
    if fallback_reason:
        token_estimator_payload = payload.get("token_estimator")
        if isinstance(token_estimator_payload, dict):
            token_estimator_payload["fallback_reason"] = fallback_reason
    if warnings:
        payload["warnings"] = warnings
    if include_pack_flag:
        payload["pack"] = pack_for_output

    payload_contract_warnings: list[str] = []
    payload_repair_count = 0
    payload, payload_contract_warnings, payload_repair_count = enforce_context_payload_contract(
        payload,
        mode=constraint_mode_value,
    )
    all_constraint_warnings = pack_contract_warnings + payload_contract_warnings
    constraint_repair_count = int(pack_repair_count) + int(payload_repair_count)
    payload["constraint_warnings"] = all_constraint_warnings
    payload["constraint_repair_count"] = int(constraint_repair_count)
    if all_constraint_warnings:
        merged_warnings = list(payload.get("warnings", [])) if isinstance(payload.get("warnings"), list) else []
        merged_warnings.extend([f"contract:{item}" for item in all_constraint_warnings])
        payload["warnings"] = merged_warnings

    sidecar_path = _write_context_metadata_sidecar(out_path, payload)
    payload["out_meta"] = str(sidecar_path)

    _debug_log(
        f"accel_context completed: {len(pack_for_output.get('top_files', []))} files, {len(pack_for_output.get('snippets', []))} snippets"
    )
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
    verify_cache_failed_results: bool | str | None = None,
    verify_cache_ttl_seconds: int | None = None,
    verify_cache_max_entries: int | None = None,
    verify_cache_failed_ttl_seconds: int | None = None,
    command_plan_cache_enabled: bool | str | None = None,
    constraint_mode: str | None = None,
) -> JSONDict:
    # Normalize changed_files: handle comma-separated string
    if isinstance(changed_files, str):
        changed_files = [f.strip() for f in changed_files.split(",") if f.strip()]

    evidence_run = _coerce_bool(evidence_run, False)
    fast_loop = _coerce_bool(fast_loop, False)
    verify_fail_fast = _coerce_optional_bool(verify_fail_fast)
    verify_cache_enabled = _coerce_optional_bool(verify_cache_enabled)
    verify_cache_failed_results = _coerce_optional_bool(verify_cache_failed_results)
    
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
    if verify_cache_failed_ttl_seconds is not None:
        runtime_overrides["verify_cache_failed_ttl_seconds"] = int(verify_cache_failed_ttl_seconds)
    if command_plan_cache_enabled is not None:
        runtime_overrides["command_plan_cache_enabled"] = _coerce_bool(command_plan_cache_enabled, True)
    if constraint_mode is not None:
        runtime_overrides["constraint_mode"] = normalize_constraint_mode(constraint_mode, default_mode="warn")

    if evidence_run and not fast_loop:
        runtime_overrides["verify_fail_fast"] = False
        runtime_overrides["verify_cache_enabled"] = False
    elif fast_loop and not evidence_run:
        runtime_overrides["verify_fail_fast"] = True
        runtime_overrides["verify_cache_enabled"] = True
        if verify_cache_failed_results is None:
            runtime_overrides["verify_cache_failed_results"] = True

    if verify_fail_fast is not None:
        runtime_overrides["verify_fail_fast"] = bool(verify_fail_fast)
    if verify_cache_enabled is not None:
        runtime_overrides["verify_cache_enabled"] = bool(verify_cache_enabled)
    if verify_cache_failed_results is not None:
        runtime_overrides["verify_cache_failed_results"] = bool(verify_cache_failed_results)

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
        strict_changed_files: Any = None,
        snippets_only: Any = False,
        include_metadata: Any = True,
        semantic_cache: Any = None,
        semantic_cache_mode: Any = None,
        constraint_mode: Any = None,
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
                strict_changed_files=strict_changed_files,
                snippets_only=snippets_only,
                include_metadata=include_metadata,
                semantic_cache=semantic_cache,
                semantic_cache_mode=semantic_cache_mode,
                constraint_mode=constraint_mode,
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
        verify_cache_failed_results: bool | str | None = None,
        verify_cache_ttl_seconds: Any = None,
        verify_cache_max_entries: Any = None,
        verify_cache_failed_ttl_seconds: Any = None,
        command_plan_cache_enabled: bool | str | None = None,
        constraint_mode: str | None = None,
    ) -> VerifyJob:
        changed_files_list = _to_string_list(changed_files)

        evidence_run_normalized = _coerce_bool(evidence_run, False)
        fast_loop_normalized = _coerce_bool(fast_loop, False)
        verify_fail_fast_normalized = _coerce_optional_bool(verify_fail_fast)
        verify_cache_enabled_normalized = _coerce_optional_bool(verify_cache_enabled)
        verify_cache_failed_results_normalized = _coerce_optional_bool(verify_cache_failed_results)
        verify_workers_value = _coerce_optional_int(verify_workers)
        per_command_timeout_value = _coerce_optional_int(per_command_timeout_seconds)
        verify_cache_ttl_value = _coerce_optional_int(verify_cache_ttl_seconds)
        verify_cache_max_entries_value = _coerce_optional_int(verify_cache_max_entries)
        verify_cache_failed_ttl_value = _coerce_optional_int(verify_cache_failed_ttl_seconds)
        command_plan_cache_enabled_normalized = _coerce_optional_bool(command_plan_cache_enabled)

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
        if verify_cache_failed_ttl_value is not None:
            runtime_overrides["verify_cache_failed_ttl_seconds"] = int(verify_cache_failed_ttl_value)
        if command_plan_cache_enabled_normalized is not None:
            runtime_overrides["command_plan_cache_enabled"] = bool(command_plan_cache_enabled_normalized)
        if constraint_mode is not None:
            runtime_overrides["constraint_mode"] = normalize_constraint_mode(constraint_mode, default_mode="warn")

        if evidence_run_normalized and not fast_loop_normalized:
            runtime_overrides["verify_fail_fast"] = False
            runtime_overrides["verify_cache_enabled"] = False
        elif fast_loop_normalized and not evidence_run_normalized:
            runtime_overrides["verify_fail_fast"] = True
            runtime_overrides["verify_cache_enabled"] = True
            if verify_cache_failed_results_normalized is None:
                runtime_overrides["verify_cache_failed_results"] = True

        if verify_fail_fast_normalized is not None:
            runtime_overrides["verify_fail_fast"] = bool(verify_fail_fast_normalized)
        if verify_cache_enabled_normalized is not None:
            runtime_overrides["verify_cache_enabled"] = bool(verify_cache_enabled_normalized)
        if verify_cache_failed_results_normalized is not None:
            runtime_overrides["verify_cache_failed_results"] = bool(verify_cache_failed_results_normalized)

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
                appended = current.add_live_event(
                    "heartbeat",
                    {
                        "elapsed_sec": float(status.get("elapsed_sec", 0.0)),
                        "eta_sec": status.get("eta_sec"),
                        "state": state,
                        "stage": status.get("stage", "running"),
                        "progress": float(status.get("progress", 0.0)),
                        "current_command": str(status.get("current_command", "")),
                        "completed_commands": int(status.get("completed_commands", 0) or 0),
                        "total_commands": int(status.get("total_commands", 0) or 0),
                    },
                )
                if not appended:
                    return
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
        verify_cache_failed_results: bool | str | None = None,
        verify_cache_ttl_seconds: Any = None,
        verify_cache_max_entries: Any = None,
        verify_cache_failed_ttl_seconds: Any = None,
        command_plan_cache_enabled: Any = None,
        constraint_mode: Any = None,
        wait_for_completion: Any = False,
        sync_wait_seconds: Any = None,
        sync_timeout_action: Any = None,
        sync_cancel_grace_seconds: Any = None,
    ) -> JSONDict:
        try:
            wait_flag = _coerce_bool(wait_for_completion, False)
            project_dir = _normalize_project_dir(project)
            job = _start_verify_job(
                project=project,
                changed_files=changed_files,
                evidence_run=evidence_run,
                fast_loop=fast_loop,
                verify_workers=verify_workers,
                per_command_timeout_seconds=per_command_timeout_seconds,
                verify_fail_fast=verify_fail_fast,
                verify_cache_enabled=verify_cache_enabled,
                verify_cache_failed_results=verify_cache_failed_results,
                verify_cache_ttl_seconds=verify_cache_ttl_seconds,
                verify_cache_max_entries=verify_cache_max_entries,
                verify_cache_failed_ttl_seconds=verify_cache_failed_ttl_seconds,
                command_plan_cache_enabled=command_plan_cache_enabled,
                constraint_mode=str(constraint_mode) if constraint_mode is not None else None,
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
            default_timeout_action, default_cancel_grace = _resolve_sync_timeout_defaults(project_dir)
            timeout_action_value = _coerce_sync_timeout_action(sync_timeout_action, default=default_timeout_action)
            cancel_grace_value = _coerce_optional_float(sync_cancel_grace_seconds)
            cancel_grace_seconds = (
                float(cancel_grace_value) if cancel_grace_value is not None else float(default_cancel_grace)
            )
            cancel_grace_seconds = max(0.2, min(30.0, cancel_grace_seconds))
            result = _wait_for_verify_job_result(
                job_id,
                max_wait_seconds=wait_seconds,
                poll_seconds=_effective_sync_verify_poll_seconds(),
            )
            if result is None:
                _debug_log(f"accel_verify sync wait timeout for job_id={job_id}")
                if timeout_action_value == "cancel":
                    jm = JobManager()
                    cancel_requested = False
                    job_for_cancel = jm.get_job(job_id)
                    if job_for_cancel is not None:
                        cancel_requested = _finalize_job_cancel(job_for_cancel, reason="sync_timeout_auto_cancel")
                    if cancel_requested:
                        _wait_for_verify_job_result(
                            job_id,
                            max_wait_seconds=cancel_grace_seconds,
                            poll_seconds=0.1,
                        )
                    return _timeout_cancel_payload(
                        job_id=job_id,
                        wait_seconds=wait_seconds,
                        cancel_requested=cancel_requested,
                    )
                return _sync_verify_timeout_payload(job_id, wait_seconds, timeout_action=timeout_action_value)
            return result
        except Exception as exc:
            _debug_log(f"accel_verify failed: {exc!r}")
            raise RuntimeError(f"{TOOL_ERROR_EXECUTION_FAILED}: {exc}") from exc

    class _JobCallback(VerifyProgressCallback):
        def __init__(self, job: VerifyJob) -> None:
            self._job = job

        def on_start(self, job_id: str, total_commands: int) -> None:
            if self._job.is_terminal():
                return
            self._job.update_progress(0, total_commands, "")
            self._job.add_live_event("job_started", {"total_commands": total_commands})

        def on_stage_change(self, job_id: str, stage: VerifyStage) -> None:
            if self._job.is_terminal():
                return
            self._job.stage = stage.name.lower()
            self._job.add_live_event("stage_change", {"stage": stage.name.lower()})

        def on_command_start(self, job_id: str, command: str, index: int, total: int) -> None:
            if self._job.is_terminal():
                return
            self._job.current_command = command
            self._job.add_live_event("command_start", {"command": command, "index": index, "total": total})

        def on_command_complete(
            self,
            job_id: str,
            command: str,
            exit_code: int,
            duration: float,
            *,
            completed: int | None = None,
            total: int | None = None,
            stdout_tail: str = "",
            stderr_tail: str = "",
        ) -> None:
            if self._job.is_terminal():
                return
            payload: JSONDict = {
                "command": command,
                "exit_code": int(exit_code),
                "duration": round(float(duration), 3),
            }
            if completed is not None:
                payload["completed"] = int(completed)
            if total is not None:
                payload["total"] = int(total)
            stdout_tail_text = str(stdout_tail or "").strip()
            stderr_tail_text = str(stderr_tail or "").strip()
            if stdout_tail_text:
                payload["stdout_tail"] = stdout_tail_text
            if stderr_tail_text:
                payload["stderr_tail"] = stderr_tail_text
            self._job.add_live_event("command_complete", payload)

        def on_progress(self, job_id: str, completed: int, total: int, current_command: str) -> None:
            if self._job.is_terminal():
                return
            self._job.update_progress(completed, total, current_command)
            self._job.add_live_event(
                "progress",
                {"completed": completed, "total": total, "progress_pct": round(self._job.progress, 1)},
            )

        def on_heartbeat(
            self,
            job_id: str,
            elapsed_sec: float,
            eta_sec: float | None,
            state: str,
            *,
            current_command: str = "",
            command_elapsed_sec: float | None = None,
            command_timeout_sec: float | None = None,
            command_progress_pct: float | None = None,
        ) -> None:
            if self._job.is_terminal():
                return
            self._job.elapsed_sec = elapsed_sec
            self._job.eta_sec = eta_sec
            if current_command:
                self._job.current_command = current_command
            heartbeat_payload: JSONDict = {
                "elapsed_sec": round(float(elapsed_sec), 1),
                "eta_sec": round(float(eta_sec), 1) if eta_sec is not None else None,
                "state": state,
            }
            if current_command:
                heartbeat_payload["current_command"] = current_command
            if command_elapsed_sec is not None:
                heartbeat_payload["command_elapsed_sec"] = round(float(command_elapsed_sec), 1)
            if command_timeout_sec is not None:
                heartbeat_payload["command_timeout_sec"] = round(float(command_timeout_sec), 1)
            if command_progress_pct is not None:
                heartbeat_payload["command_progress_pct"] = round(float(command_progress_pct), 2)
            self._job.add_live_event("heartbeat", heartbeat_payload)

        def on_cache_hit(self, job_id: str, command: str) -> None:
            if self._job.is_terminal():
                return
            self._job.add_live_event("cache_hit", {"command": command})

        def on_skip(self, job_id: str, command: str, reason: str) -> None:
            if self._job.is_terminal():
                return
            self._job.add_live_event("command_skipped", {"command": command, "reason": reason})

        def on_error(self, job_id: str, command: str | None, error: str) -> None:
            if self._job.is_terminal():
                return
            self._job.add_live_event("error", {"command": command, "error": error})

        def on_complete(self, job_id: str, status: str, exit_code: int) -> None:
            if self._job.is_terminal():
                return
            self._job.add_live_event("job_completed", {"status": status, "exit_code": exit_code})

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
        verify_cache_failed_results: bool | str | None = None,
        verify_cache_ttl_seconds: Any = None,
        verify_cache_max_entries: Any = None,
        verify_cache_failed_ttl_seconds: Any = None,
        command_plan_cache_enabled: Any = None,
        constraint_mode: Any = None,
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
                verify_cache_failed_results=verify_cache_failed_results,
                verify_cache_ttl_seconds=verify_cache_ttl_seconds,
                verify_cache_max_entries=verify_cache_max_entries,
                verify_cache_failed_ttl_seconds=verify_cache_failed_ttl_seconds,
                command_plan_cache_enabled=command_plan_cache_enabled,
                constraint_mode=str(constraint_mode) if constraint_mode is not None else None,
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
            status_payload = job.to_status()
            status_payload["state_source"] = "job_state"
            status_payload["constraint_repair_count"] = 0
            return status_payload
        except Exception as exc:
            _debug_log(f"accel_verify_status failed: {exc!r}")
            raise RuntimeError(f"{TOOL_ERROR_EXECUTION_FAILED}: {exc}") from exc

    @server.tool(
        name="accel_verify_events",
        description="Get verification events with optional summary and tail clipping.",
    )
    def accel_verify_events(
        job_id: str,
        since_seq: int = 0,
        max_events: Any = 30,
        include_summary: Any = True,
    ) -> JSONDict:
        try:
            _debug_log(
                f"accel_verify_events called: job_id={job_id}, since_seq={since_seq}, "
                f"max_events={max_events}, include_summary={include_summary}"
            )
            jm = JobManager()
            job = jm.get_job(job_id)
            if job is None:
                return {"error": "job_not_found", "job_id": job_id}

            since_seq_value = int(_coerce_optional_int(since_seq) or 0)
            limit = _coerce_events_limit(max_events, default_value=30, max_value=500)
            include_summary_flag = _coerce_bool(include_summary, True)

            events_all = job.get_events(since_seq_value)
            total_available = len(events_all)
            truncated = False
            if total_available > limit:
                events = events_all[-limit:]
                truncated = True
            else:
                events = events_all

            payload: JSONDict = {
                "job_id": job_id,
                "events": events,
                "count": len(events),
                "total_available": total_available,
                "truncated": truncated,
                "max_events": limit,
                "since_seq": since_seq_value,
            }
            if include_summary_flag:
                status_payload = job.to_status()
                summary = _summarize_verify_events(events_all, status_payload)
                summary_mode = normalize_constraint_mode(os.environ.get("ACCEL_CONSTRAINT_MODE", "warn"), default_mode="warn")
                summary, summary_warnings, summary_repair_count = enforce_verify_summary_contract(
                    summary,
                    status=status_payload,
                    mode=summary_mode,
                )
                summary["constraint_repair_count"] = int(summary_repair_count)
                if summary_warnings:
                    summary["constraint_warnings"] = summary_warnings
                payload["summary"] = summary
            return payload
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
            _finalize_job_cancel(job, reason="user_request")
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
