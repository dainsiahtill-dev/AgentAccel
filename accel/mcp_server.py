from __future__ import annotations

import argparse
import copy
import json
import logging
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Callable
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
    enforce_verify_events_payload_contract,
    enforce_verify_summary_contract,
    normalize_constraint_mode,
)
from .token_estimator import estimate_tokens_for_text, estimate_tokens_from_chars
from .verify.orchestrator import run_verify, run_verify_with_callback
from .verify.callbacks import VerifyProgressCallback, VerifyStage
from .verify.job_manager import JobManager, JobState, VerifyJob
from .mcp_context_utils import (
    _apply_strict_changed_files_scope,
    _build_context_output_pack,
    _clamp_float,
    _ensure_strict_changed_files_presence,
    _estimate_changed_files_chars,
    _token_reduction_ratio,
    _write_context_metadata_sidecar,
)


JSONDict = dict[str, Any]
SERVER_NAME = "agent-accel-mcp"
SERVER_VERSION = "0.2.6"
TOOL_ERROR_EXECUTION_FAILED = "ACCEL_TOOL_EXECUTION_FAILED"

# Debug logging setup
_debug_enabled = os.environ.get("ACCEL_MCP_DEBUG", "").lower() in {"1", "true", "yes"}
_debug_logger: logging.Logger | None = None

# Global server timeout protection
_server_start_time = 0.0
_server_max_runtime = int(
    os.environ.get("ACCEL_MCP_MAX_RUNTIME", "3600")
)  # 1 hour default
_shutdown_requested = False

# Keep sync verify bounded so a single call cannot monopolize MCP request handling.
_sync_verify_wait_seconds = int(
    os.environ.get("ACCEL_MCP_SYNC_VERIFY_WAIT_SECONDS", "45")
)
_sync_verify_poll_seconds = float(
    os.environ.get("ACCEL_MCP_SYNC_VERIFY_POLL_SECONDS", "0.2")
)
_sync_index_wait_seconds = int(
    os.environ.get("ACCEL_MCP_SYNC_INDEX_WAIT_SECONDS", "45")
)
_sync_index_poll_seconds = float(
    os.environ.get("ACCEL_MCP_SYNC_INDEX_POLL_SECONDS", "0.2")
)
_sync_context_wait_seconds = int(
    os.environ.get("ACCEL_MCP_SYNC_CONTEXT_WAIT_SECONDS", "45")
)
_sync_context_poll_seconds = float(
    os.environ.get("ACCEL_MCP_SYNC_CONTEXT_POLL_SECONDS", "0.2")
)
_FALSE_LITERALS = {
    "",
    "0",
    "false",
    "no",
    "off",
    "none",
    "null",
    "undefined",
    "pydanticundefined",
}
_TRUE_LITERALS = {"1", "true", "yes", "on"}
_SYNC_TIMEOUT_ACTIONS = {"poll", "cancel"}
_CONTEXT_SYNC_TIMEOUT_ACTIONS = {"fallback_async", "cancel"}
_SYNC_WAIT_SECONDS_MAX = 7200.0
_SYNC_WAIT_RPC_SAFE_SECONDS = 45.0
_SEMANTIC_CACHE_MODE_ALIASES = {
    "readwrite": "hybrid",
    "read_write": "hybrid",
    "read-write": "hybrid",
    "rw": "hybrid",
}
_VERIFY_PRESET_ALIASES = {
    "quick": "fast",
    "speed": "fast",
    "full_check": "full",
    "evidence": "full",
    "balanced": "full",
    "default": "full",
}
_VERIFY_OUTPUT_CHUNK_LIMIT = 600


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
            _debug_log(
                f"Server exceeded maximum runtime ({_server_max_runtime}s), shutting down"
            )
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
            datefmt="%Y-%m-%d %H:%M:%S",
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
    return max(1, int(min(_SYNC_WAIT_SECONDS_MAX, float(_sync_verify_wait_seconds))))


def _effective_sync_verify_poll_seconds() -> float:
    return max(0.05, float(_sync_verify_poll_seconds))


def _effective_sync_index_wait_seconds() -> int:
    return max(1, int(min(_SYNC_WAIT_SECONDS_MAX, float(_sync_index_wait_seconds))))


def _effective_sync_index_poll_seconds() -> float:
    return max(0.05, float(_sync_index_poll_seconds))


def _effective_sync_context_wait_seconds() -> int:
    return max(1, int(min(_SYNC_WAIT_SECONDS_MAX, float(_sync_context_wait_seconds))))


def _effective_sync_context_poll_seconds() -> float:
    return max(0.05, float(_sync_context_poll_seconds))


def _normalize_job_status_payload(status_payload: JSONDict) -> JSONDict:
    normalized = dict(status_payload)
    state = str(normalized.get("state", "")).strip().lower()
    completed = max(0, int(normalized.get("completed_commands", 0) or 0))
    total = max(0, int(normalized.get("total_commands", 0) or 0))
    progress = float(normalized.get("progress", 0.0) or 0.0)
    if total > 0:
        derived_progress = (float(completed) / float(total)) * 100.0
        if abs(progress - derived_progress) > 0.5:
            progress = derived_progress
    progress = max(0.0, min(100.0, progress))
    consistency = "normal"
    if (
        state in {JobState.PENDING, JobState.RUNNING, JobState.CANCELLING}
        and total > 0
        and completed >= total
    ):
        progress = min(progress, 99.9)
        consistency = "finalizing"
    elif (
        state in {JobState.COMPLETED, JobState.FAILED, JobState.CANCELLED}
        and total > 0
        and completed >= total
    ):
        progress = 100.0
    normalized["progress"] = round(progress, 2)
    normalized["state_consistency"] = consistency
    return normalized


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


def _resolve_semantic_cache_mode(value: Any, *, default_mode: str) -> str:
    if value is None:
        token = str(default_mode or "hybrid").strip().lower() or "hybrid"
    else:
        token = str(value).strip().lower()
    token = _SEMANTIC_CACHE_MODE_ALIASES.get(token, token)
    if token in {"exact", "hybrid"}:
        return token
    if value is None:
        return "hybrid"
    raise ValueError(
        "semantic_cache_mode must be one of exact|hybrid (aliases: readwrite|read_write|read-write|rw)"
    )


def _resolve_constraint_mode(value: Any, *, default_mode: str = "warn") -> str:
    normalized = normalize_constraint_mode(
        value if value is not None else default_mode, default_mode=default_mode
    )
    if value is None:
        return normalized
    raw = str(value).strip().lower()
    if raw in {"off", "warn", "strict", "enforce", "error", "errors", "on", "default"}:
        return normalized
    raise ValueError("constraint_mode must be one of off|warn|strict (alias: enforce)")


def _resolve_verify_preset(value: Any) -> str | None:
    if value is None:
        return None
    token = str(value).strip().lower()
    if token in {"", "auto", "none"}:
        return None
    token = _VERIFY_PRESET_ALIASES.get(token, token)
    if token in {"fast", "full"}:
        return token
    raise ValueError("verify_preset must be one of auto|fast|full")


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


def _coerce_context_sync_timeout_action(
    value: Any, default: str = "fallback_async"
) -> str:
    token = str(value or "").strip().lower()
    if token == "poll":
        token = "fallback_async"
    if token in _CONTEXT_SYNC_TIMEOUT_ACTIONS:
        return token
    fallback = str(default or "fallback_async").strip().lower()
    if fallback == "poll":
        fallback = "fallback_async"
    if fallback in _CONTEXT_SYNC_TIMEOUT_ACTIONS:
        return fallback
    return "fallback_async"


def _coerce_events_limit(
    value: Any, default_value: int = 30, max_value: int = 500
) -> int:
    parsed = _coerce_optional_int(value)
    if parsed is None:
        parsed = default_value
    return max(1, min(int(max_value), int(parsed)))


def _summarize_verify_events(
    events: list[dict[str, Any]], status: JSONDict
) -> JSONDict:
    event_type_counts: dict[str, int] = {}
    command_start = 0
    command_complete = 0
    command_skipped = 0
    command_errors = 0
    cache_hits = 0
    command_output_chunks = 0
    stall_heartbeats = 0
    terminal_event_seen = False
    first_seq = 0
    last_seq = 0
    latest_stage = str(status.get("stage", ""))
    latest_state = str(status.get("state", ""))
    latest_command_complete: JSONDict = {}
    recent_output: list[JSONDict] = []
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
        elif event_type == "command_output":
            command_output_chunks += 1
            chunk = str(event.get("chunk", "")).strip()
            if chunk:
                recent_output.append(
                    {
                        "seq": int(event.get("seq", 0) or 0),
                        "stream": str(event.get("stream", "stdout")),
                        "command": str(event.get("command", "")),
                        "chunk": chunk,
                    }
                )
                if len(recent_output) > 3:
                    recent_output = recent_output[-3:]
        elif event_type == "command_complete":
            latest_command_complete = {
                "command": str(event.get("command", "")),
                "exit_code": int(event.get("exit_code", 1) or 1),
                "duration": float(event.get("duration", 0.0) or 0.0),
                "stdout_tail": str(event.get("stdout_tail", "")),
                "stderr_tail": str(event.get("stderr_tail", "")),
                "completed": int(event.get("completed", 0) or 0),
                "total": int(event.get("total", 0) or 0),
            }
        elif event_type == "heartbeat" and _coerce_bool(
            event.get("stall_detected"), False
        ):
            stall_heartbeats += 1
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
            "output_chunks": int(command_output_chunks),
            "stall_heartbeats": int(stall_heartbeats),
        },
        "seq_range": {"first": int(first_seq), "last": int(last_seq)},
        "latest_command_complete": latest_command_complete,
        "recent_output": recent_output,
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


def _sync_verify_timeout_payload(
    job_id: str, wait_seconds: float, *, timeout_action: str = "poll"
) -> JSONDict:
    jm = JobManager()
    job = jm.get_job(job_id)
    status = _normalize_job_status_payload(job.to_status()) if job is not None else {}
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
        runtime_cfg = (
            config.get("runtime", {})
            if isinstance(config.get("runtime", {}), dict)
            else {}
        )
        action = _coerce_sync_timeout_action(
            runtime_cfg.get("sync_verify_timeout_action"), default="poll"
        )
        grace_value = _coerce_optional_float(
            runtime_cfg.get("sync_verify_cancel_grace_seconds")
        )
        if grace_value is not None:
            grace_seconds = grace_value
    except Exception:
        action = "poll"
        grace_seconds = 5.0
    grace_seconds = max(0.2, min(30.0, float(grace_seconds)))
    return action, grace_seconds


def _resolve_sync_wait_seconds(
    *,
    project_dir: Path,
    override_value: Any,
    runtime_key: str,
    fallback_seconds: float,
    rpc_safe_cap_seconds: float | None = _SYNC_WAIT_RPC_SAFE_SECONDS,
) -> float:
    parsed_override = _coerce_optional_float(override_value)
    wait_seconds = float(fallback_seconds)
    if parsed_override is not None:
        wait_seconds = parsed_override
    else:
        try:
            config = resolve_effective_config(project_dir)
            runtime_cfg = (
                config.get("runtime", {})
                if isinstance(config.get("runtime", {}), dict)
                else {}
            )
            configured_value = _coerce_optional_float(runtime_cfg.get(runtime_key))
            if configured_value is not None:
                wait_seconds = configured_value
        except Exception:
            wait_seconds = float(fallback_seconds)
    resolved = _clamp_float(wait_seconds, 1.0, _SYNC_WAIT_SECONDS_MAX)
    cap_value = _coerce_optional_float(rpc_safe_cap_seconds)
    if cap_value is not None and cap_value > 0:
        resolved = min(resolved, _clamp_float(cap_value, 1.0, _SYNC_WAIT_SECONDS_MAX))
    return resolved


def _resolve_context_sync_timeout_action(
    project_dir: Path, override_value: Any = None
) -> str:
    action_override = str(override_value or "").strip().lower()
    if action_override:
        return _coerce_context_sync_timeout_action(
            action_override, default="fallback_async"
        )
    action = "fallback_async"
    try:
        config = resolve_effective_config(project_dir)
        runtime_cfg = (
            config.get("runtime", {})
            if isinstance(config.get("runtime", {}), dict)
            else {}
        )
        action = _coerce_context_sync_timeout_action(
            runtime_cfg.get("sync_context_timeout_action"),
            default="fallback_async",
        )
    except Exception:
        action = "fallback_async"
    return action


def _sync_context_timeout_payload(
    job_id: str, wait_seconds: float, *, timeout_action: str
) -> JSONDict:
    jm = JobManager()
    job = jm.get_job(job_id)
    status = _normalize_job_status_payload(job.to_status()) if job is not None else {}
    action = _coerce_context_sync_timeout_action(
        timeout_action, default="fallback_async"
    )
    return {
        "status": "running",
        "exit_code": 124,
        "timed_out": True,
        "job_id": job_id,
        "timeout_action": action,
        "message": (
            f"accel_context exceeded synchronous wait window ({wait_seconds:.1f}s); "
            "switched to async polling. Use accel_context_status/accel_context_events."
        ),
        "poll_interval_sec": 1.0,
        "state": status.get("state", "running"),
        "stage": status.get("stage", "running"),
        "progress": status.get("progress", 0.0),
        "elapsed_sec": status.get("elapsed_sec", 0.0),
    }


def _timeout_cancel_payload_context(
    *,
    job_id: str,
    wait_seconds: float,
    cancel_requested: bool,
) -> JSONDict:
    jm = JobManager()
    job = jm.get_job(job_id)
    status = _normalize_job_status_payload(job.to_status()) if job is not None else {}
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
            f"accel_context exceeded synchronous wait window ({wait_seconds:.1f}s); auto-cancel requested."
            if cancel_requested
            else f"accel_context exceeded synchronous wait window ({wait_seconds:.1f}s); auto-cancel request failed."
        ),
        "poll_interval_sec": 1.0,
        "state": state,
        "stage": stage,
        "progress": status.get("progress", 0.0),
        "elapsed_sec": status.get("elapsed_sec", 0.0),
    }


def _timeout_cancel_payload(
    *,
    job_id: str,
    wait_seconds: float,
    cancel_requested: bool,
) -> JSONDict:
    jm = JobManager()
    job = jm.get_job(job_id)
    status = _normalize_job_status_payload(job.to_status()) if job is not None else {}
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


def _wait_for_index_job_result(
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
                payload = {"status": "ok", "exit_code": 0}
            payload.setdefault("job_id", job_id)
            payload.setdefault("timed_out", False)
            return payload

        if job.state == JobState.FAILED:
            return {
                "status": "failed",
                "exit_code": 1,
                "job_id": job_id,
                "error": str(job.error or "index job failed"),
                "timed_out": False,
            }

        if job.state == JobState.CANCELLED:
            return {
                "status": "cancelled",
                "exit_code": 130,
                "job_id": job_id,
                "timed_out": False,
            }

        if time.perf_counter() >= deadline:
            return None
        time.sleep(poll)


def _wait_for_context_job_result(
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
                payload = {"status": "ok", "exit_code": 0}
            payload.setdefault("job_id", job_id)
            payload.setdefault("timed_out", False)
            return payload

        if job.state == JobState.FAILED:
            return {
                "status": "failed",
                "exit_code": 1,
                "job_id": job_id,
                "error": str(job.error or "context job failed"),
                "timed_out": False,
            }

        if job.state == JobState.CANCELLED:
            return {
                "status": "cancelled",
                "exit_code": 130,
                "job_id": job_id,
                "timed_out": False,
            }

        if time.perf_counter() >= deadline:
            return None
        time.sleep(poll)


def _sync_index_timeout_payload(job_id: str, wait_seconds: float) -> JSONDict:
    jm = JobManager()
    job = jm.get_job(job_id)
    status = _normalize_job_status_payload(job.to_status()) if job is not None else {}
    return {
        "status": "running",
        "exit_code": 124,
        "timed_out": True,
        "job_id": job_id,
        "message": (
            f"accel_index_build exceeded synchronous wait window ({wait_seconds:.1f}s); "
            "use accel_index_status/accel_index_events to continue polling."
        ),
        "poll_interval_sec": 1.0,
        "state": status.get("state", "running"),
        "stage": status.get("stage", "running"),
        "progress": status.get("progress", 0.0),
        "elapsed_sec": status.get("elapsed_sec", 0.0),
        "processed_files": int(status.get("completed_commands", 0) or 0),
        "total_files": int(status.get("total_commands", 0) or 0),
        "current_path": str(status.get("current_command", "")),
        "eta_sec": status.get("eta_sec"),
        "state_consistency": status.get("state_consistency", "normal"),
    }


def _start_index_job(*, project: str, mode: str, full: bool) -> VerifyJob:
    project_dir = _normalize_project_dir(project)
    mode_value = str(mode).strip().lower()
    if mode_value not in {"build", "update"}:
        raise ValueError("index mode must be build|update")
    full_value = bool(full)

    jm = JobManager()
    job = jm.create_job(prefix="index")
    job.add_event(
        "index_queued",
        {
            "project": str(project_dir),
            "mode": mode_value,
            "full": bool(full_value),
        },
    )

    def _run_index_thread(job_id: str) -> None:
        current = jm.get_job(job_id)
        if current is None:
            return
        try:
            current.mark_running("indexing")
            current.add_event(
                "index_started",
                {
                    "project": str(project_dir),
                    "mode": mode_value,
                    "full": bool(full_value),
                },
            )

            def _on_index_progress(progress_payload: JSONDict) -> None:
                stage_name = (
                    str(progress_payload.get("stage", "indexing")).strip().lower()
                    or "indexing"
                )
                processed_files = max(
                    0, int(progress_payload.get("processed_files", 0) or 0)
                )
                total_files = max(0, int(progress_payload.get("total_files", 0) or 0))
                changed_files = max(
                    0, int(progress_payload.get("changed_files", 0) or 0)
                )
                current_path = str(progress_payload.get("current_path", "")).strip()
                message = str(progress_payload.get("message", "")).strip()
                if stage_name:
                    current.stage = stage_name
                if total_files > 0:
                    current.update_progress(
                        min(processed_files, total_files), total_files, current_path
                    )
                elif current_path:
                    current.current_command = current_path
                progress_pct_value = float(current.progress)
                if (
                    total_files > 0
                    and processed_files >= total_files
                    and current.state == JobState.RUNNING
                ):
                    progress_pct_value = min(progress_pct_value, 99.9)
                index_event_payload: JSONDict = {
                    "stage": stage_name,
                    "processed_files": int(processed_files),
                    "total_files": int(total_files),
                    "changed_files": int(changed_files),
                    "progress_pct": round(float(progress_pct_value), 2),
                    "eta_sec": round(float(current.eta_sec), 1)
                    if current.eta_sec is not None
                    else None,
                }
                if current_path:
                    index_event_payload["current_path"] = current_path
                if message:
                    index_event_payload["message"] = message
                current.add_live_event("index_progress", index_event_payload)

            if mode_value == "build":
                try:
                    result = _tool_index_build(
                        project=str(project_dir),
                        full=full_value,
                        progress_callback=_on_index_progress,
                    )
                except TypeError:
                    result = _tool_index_build(
                        project=str(project_dir),
                        full=full_value,
                    )
            else:
                try:
                    result = _tool_index_update(
                        project=str(project_dir),
                        progress_callback=_on_index_progress,
                    )
                except TypeError:
                    result = _tool_index_update(project=str(project_dir))
            if current.state in (JobState.CANCELLING, JobState.CANCELLED):
                if current.state != JobState.CANCELLED:
                    current.mark_cancelled()
                current.add_event(
                    "job_cancelled_finalized",
                    {"reason": "index_worker_observed_cancel"},
                )
                return
            payload: JSONDict = {
                "status": "ok",
                "exit_code": 0,
                "manifest": dict(result.get("manifest", {}))
                if isinstance(result.get("manifest"), dict)
                else {},
                "mode": mode_value,
                "full": bool(full_value),
                "timed_out": False,
            }
            current.mark_completed("ok", 0, payload)
            manifest_payload = payload.get("manifest", {})
            files = 0
            if isinstance(manifest_payload, dict):
                files = int(manifest_payload.get("counts", {}).get("files", 0) or 0)
            current.add_event("index_completed", {"files": files, "mode": mode_value})
        except Exception as exc:
            if current.state in (JobState.CANCELLING, JobState.CANCELLED):
                if current.state != JobState.CANCELLED:
                    current.mark_cancelled()
                current.add_event(
                    "job_cancelled_finalized",
                    {"reason": "index_worker_exception_after_cancel"},
                )
                return
            current.mark_failed(str(exc))
            current.add_event("index_failed", {"error": str(exc)})

    thread = threading.Thread(target=_run_index_thread, args=(job.job_id,), daemon=True)
    thread.start()

    def _heartbeat_thread(job_id: str) -> None:
        while True:
            current = jm.get_job(job_id)
            if current is None:
                return
            status = _normalize_job_status_payload(current.to_status())
            state = str(status.get("state", ""))
            if state in {JobState.COMPLETED, JobState.FAILED, JobState.CANCELLED}:
                return
            appended = current.add_live_event(
                "heartbeat",
                {
                    "elapsed_sec": float(status.get("elapsed_sec", 0.0)),
                    "state": state or JobState.RUNNING,
                    "stage": status.get("stage", "indexing"),
                    "progress": float(status.get("progress", 0.0)),
                    "processed_files": int(status.get("completed_commands", 0) or 0),
                    "total_files": int(status.get("total_commands", 0) or 0),
                    "current_path": str(status.get("current_command", "")),
                    "eta_sec": status.get("eta_sec"),
                },
            )
            if not appended:
                return
            time.sleep(1.0)

    hb = threading.Thread(target=_heartbeat_thread, args=(job.job_id,), daemon=True)
    hb.start()
    return job


def _with_timeout(func, timeout_seconds: int = 300):
    """Wrapper to add timeout to MCP tool functions."""

    def wrapper(*args, **kwargs):
        # Check server runtime before each tool call
        _check_server_runtime()

        start_time = time.perf_counter()
        _debug_log(f"Starting {func.__name__} with timeout {timeout_seconds}s")

        result_holder: dict[str, Any] = {}
        error_holder: dict[str, BaseException] = {}

        def _target() -> None:
            try:
                result_holder["result"] = func(*args, **kwargs)
            except BaseException as exc:  # noqa: BLE001
                error_holder["error"] = exc

        worker = threading.Thread(target=_target, daemon=True)
        worker.start()

        try:
            timeout_value = max(1.0, float(timeout_seconds))
            worker.join(timeout=timeout_value)
            if worker.is_alive():
                elapsed = time.perf_counter() - start_time
                _debug_log(f"Timed out {func.__name__} after {elapsed:.3f}s")
                raise TimeoutError(
                    f"{func.__name__} timed out after {timeout_value:.1f}s"
                )

            if "error" in error_holder:
                raise error_holder["error"]

            result = result_holder.get("result")
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
    "tiny": {
        "max_chars": 6000,
        "max_snippets": 16,
        "top_n_files": 6,
        "snippet_radius": 20,
    },
    "small": {
        "max_chars": 12000,
        "max_snippets": 30,
        "top_n_files": 8,
        "snippet_radius": 24,
    },
    "medium": {
        "max_chars": 24000,
        "max_snippets": 60,
        "top_n_files": 12,
        "snippet_radius": 30,
    },
    "large": {
        "max_chars": 36000,
        "max_snippets": 90,
        "top_n_files": 16,
        "snippet_radius": 40,
    },
    "xlarge": {
        "max_chars": 50000,
        "max_snippets": 120,
        "top_n_files": 20,
        "snippet_radius": 50,
    },
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


def _run_git_cmd_lines(cmd: list[str], timeout_seconds: float) -> tuple[list[str], int]:
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
        return [], -1
    if int(proc.returncode) != 0:
        return [], int(proc.returncode)
    lines = [line for line in str(proc.stdout).splitlines() if str(line).strip()]
    return lines, int(proc.returncode)


def _discover_changed_files_from_git_details(
    project_dir: Path, limit: int = 200
) -> tuple[list[str], JSONDict]:
    discovered: list[str] = []
    details: JSONDict = {}
    lock = threading.Lock()
    max_items = max(1, int(limit))

    def _collect() -> None:
        seen: set[str] = set()
        output: list[str] = []
        source_counts: dict[str, int] = {}
        source_order: list[str] = []

        def _add_source(
            name: str, lines: list[str], *, status_parser: bool = False
        ) -> None:
            if not lines or len(output) >= max_items:
                return
            source_order.append(name)
            accepted = 0
            for raw_line in lines:
                rel = (
                    _extract_status_path(raw_line)
                    if status_parser
                    else str(raw_line).strip().replace("\\", "/")
                )
                if not rel or rel in seen:
                    continue
                seen.add(rel)
                output.append(rel)
                accepted += 1
                if len(output) >= max_items:
                    break
            source_counts[name] = int(accepted)

        inside_lines, _inside_rc = _run_git_cmd_lines(
            ["git", "-C", str(project_dir), "rev-parse", "--is-inside-work-tree"],
            1.0,
        )
        inside_token = str(inside_lines[0]).strip().lower() if inside_lines else ""
        if _inside_rc != 0 or inside_token in {"", "false", "0", "no"}:
            payload = {
                "provider": "git",
                "available": False,
                "reason": "not_git_worktree",
                "sources": [],
                "source_counts": {},
                "confidence": 0.0,
            }
            with lock:
                details.clear()
                details.update(payload)
            return

        git_root_lines, _root_rc = _run_git_cmd_lines(
            ["git", "-C", str(project_dir), "rev-parse", "--show-toplevel"],
            1.0,
        )
        git_root = str(git_root_lines[0]).strip() if git_root_lines else ""

        status_lines, _status_rc = _run_git_cmd_lines(
            [
                "git",
                "-C",
                str(project_dir),
                "status",
                "--porcelain",
                "--untracked-files=normal",
            ],
            2.0,
        )
        _add_source("status_porcelain", status_lines, status_parser=True)

        diff_commands = [
            (
                "staged_diff",
                [
                    "git",
                    "-C",
                    str(project_dir),
                    "diff",
                    "--name-only",
                    "--relative",
                    "--cached",
                ],
            ),
            (
                "working_tree_diff",
                ["git", "-C", str(project_dir), "diff", "--name-only", "--relative"],
            ),
            (
                "head_diff",
                [
                    "git",
                    "-C",
                    str(project_dir),
                    "diff",
                    "--name-only",
                    "--relative",
                    "HEAD",
                ],
            ),
        ]
        for source_name, cmd in diff_commands:
            if len(output) >= max_items:
                break
            lines, _rc = _run_git_cmd_lines(cmd, 1.5)
            _add_source(source_name, lines, status_parser=False)

        upstream = ""
        merge_base = ""
        if len(output) < max_items:
            upstream_lines, _upstream_rc = _run_git_cmd_lines(
                [
                    "git",
                    "-C",
                    str(project_dir),
                    "rev-parse",
                    "--abbrev-ref",
                    "--symbolic-full-name",
                    "@{upstream}",
                ],
                1.0,
            )
            if upstream_lines:
                upstream = str(upstream_lines[0]).strip()
            if upstream:
                merge_base_lines, _mb_rc = _run_git_cmd_lines(
                    ["git", "-C", str(project_dir), "merge-base", "HEAD", upstream],
                    1.0,
                )
                if merge_base_lines:
                    merge_base = str(merge_base_lines[0]).strip()
            if merge_base:
                upstream_diff_lines, _upstream_diff_rc = _run_git_cmd_lines(
                    [
                        "git",
                        "-C",
                        str(project_dir),
                        "diff",
                        "--name-only",
                        "--relative",
                        f"{merge_base}..HEAD",
                    ],
                    2.0,
                )
                _add_source("upstream_diff", upstream_diff_lines, status_parser=False)

        confidence = 0.0
        if source_counts.get("upstream_diff", 0) > 0:
            confidence = 0.99
        elif source_counts.get("status_porcelain", 0) > 0:
            confidence = 0.98
        elif (
            source_counts.get("staged_diff", 0) > 0
            or source_counts.get("working_tree_diff", 0) > 0
        ):
            confidence = 0.96
        elif source_counts.get("head_diff", 0) > 0:
            confidence = 0.9

        payload = {
            "provider": "git",
            "available": True,
            "reason": "ok" if output else "git_no_changes_detected",
            "git_root": git_root,
            "upstream": upstream,
            "merge_base": merge_base,
            "sources": list(source_order),
            "source_counts": dict(source_counts),
            "confidence": round(float(confidence), 6),
        }

        with lock:
            discovered.clear()
            discovered.extend(output[:max_items])
            details.clear()
            details.update(payload)

    worker = threading.Thread(target=_collect, daemon=True)
    worker.start()
    worker.join(timeout=6.0)
    if worker.is_alive():
        _debug_log(
            "_discover_changed_files_from_git_details timed out; fallback to empty changed_files"
        )
        return [], {
            "provider": "git",
            "available": False,
            "reason": "git_detection_timeout",
            "sources": [],
            "source_counts": {},
            "confidence": 0.0,
        }
    with lock:
        return list(discovered), dict(details)


def _discover_changed_files_from_git(project_dir: Path, limit: int = 200) -> list[str]:
    files, _details = _discover_changed_files_from_git_details(project_dir, limit=limit)
    return files


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

    indexed_files = [
        str(item).replace("\\", "/")
        for item in manifest.get("indexed_files", [])
        if str(item).strip()
    ]
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


def _auto_context_budget_preset(
    task: str, changed_files: list[str], hints: list[str]
) -> tuple[str, str]:
    task_text = str(task or "").strip()
    task_low = task_text.lower()
    task_tokens = normalize_task_tokens(task_text)
    changed_count = len(changed_files)
    hint_count = len(hints)

    quick_markers = {
        "explain",
        "summary",
        "summarize",
        "quick",
        "small",
        "typo",
        "doc",
        "readme",
    }
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

    if (
        changed_count <= 1
        and len(task_tokens) <= 8
        and len(task_text) <= 120
        and score <= 2
    ):
        return "tiny", "auto:tiny_for_quick_task"
    if (
        changed_count >= 8
        or len(task_tokens) >= 20
        or len(task_text) >= 260
        or score >= 8
    ):
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
            "rule_compression_enabled": bool(
                runtime_cfg.get("rule_compression_enabled", True)
            ),
            "constraint_mode": str(runtime_cfg.get("constraint_mode", "warn")),
            "token_estimator_backend": str(
                runtime_cfg.get("token_estimator_backend", "auto")
            ),
            "token_estimator_encoding": str(
                runtime_cfg.get("token_estimator_encoding", "cl100k_base")
            ),
            "token_estimator_model": str(runtime_cfg.get("token_estimator_model", "")),
            "token_estimator_calibration": float(
                runtime_cfg.get("token_estimator_calibration", 1.0)
            ),
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


def _resolve_git_head_commit(project_dir: Path) -> str:
    lines, rc = _run_git_cmd_lines(
        ["git", "-C", str(project_dir), "rev-parse", "HEAD"],
        1.0,
    )
    if rc != 0 or not lines:
        return ""
    return str(lines[0]).strip()


def _collect_changed_files_state(
    project_dir: Path, changed_files: list[str]
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    project_root = project_dir.resolve()
    for item in normalize_changed_files(changed_files):
        path_token = str(item or "").strip()
        if not path_token:
            continue
        raw_path = Path(path_token)
        abs_path = raw_path if raw_path.is_absolute() else (project_root / raw_path)
        rel_path = path_token
        try:
            rel_path = abs_path.resolve().relative_to(project_root).as_posix()
        except Exception:
            rel_path = path_token.replace("\\", "/")
        row: dict[str, Any] = {"path": rel_path}
        if abs_path.exists():
            stat = abs_path.stat()
            row["exists"] = True
            row["size"] = int(stat.st_size)
            row["mtime_ns"] = int(stat.st_mtime_ns)
            row["is_dir"] = bool(abs_path.is_dir())
        else:
            row["exists"] = False
        rows.append(row)
    rows.sort(key=lambda entry: str(entry.get("path", "")))
    return rows


def _semantic_cache_safety_fingerprint(
    project_dir: Path,
    changed_files: list[str],
) -> tuple[str, JSONDict]:
    git_head = _resolve_git_head_commit(project_dir)
    changed_state = _collect_changed_files_state(project_dir, changed_files)
    payload = {
        "git_head": git_head,
        "changed_files_state": changed_state,
    }
    fingerprint = make_stable_hash(payload)
    metadata: JSONDict = {
        "git_head": git_head,
        "changed_files_state": changed_state,
        "fingerprint": fingerprint,
    }
    return fingerprint, metadata


def _safe_semantic_cache_store(
    project_dir: Path, config: JSONDict
) -> SemanticCacheStore | None:
    try:
        runtime_cfg = dict(config.get("runtime", {}))
        accel_home = Path(str(runtime_cfg.get("accel_home", "") or "")).resolve()
        paths = project_paths(accel_home, project_dir)
        ensure_project_dirs(paths)
        return SemanticCacheStore(paths["state"] / "semantic_cache.db")
    except OSError:
        return None


def _tool_index_build(
    project: str = ".",
    full: bool = True,
    progress_callback: Callable[[JSONDict], None] | None = None,
) -> JSONDict:
    _debug_log(f"accel_index_build called: project={project}, full={full}")
    project_dir = _normalize_project_dir(project)
    config = resolve_effective_config(project_dir)
    manifest = build_or_update_indexes(
        project_dir=project_dir,
        config=config,
        mode="build",
        full=bool(full),
        progress_callback=progress_callback,
    )
    file_count = int(manifest.get("counts", {}).get("files", 0) or 0)
    if file_count <= 0:
        retry_config = copy.deepcopy(config)
        retry_index_cfg = dict(retry_config.get("index", {}))
        retry_index_cfg["scope_mode"] = "all"
        retry_index_cfg["include"] = ["**/*"]
        retry_config["index"] = retry_index_cfg
        retry_manifest = build_or_update_indexes(
            project_dir=project_dir,
            config=retry_config,
            mode="build",
            full=bool(full),
            progress_callback=progress_callback,
        )
        retry_file_count = int(retry_manifest.get("counts", {}).get("files", 0) or 0)
        if retry_file_count > file_count:
            retry_manifest["scope_retry_used"] = True
            retry_manifest["scope_retry_reason"] = "initial_scope_selected_zero_files"
            manifest = retry_manifest
    _debug_log(
        f"accel_index_build completed: {len(manifest.get('indexed_files', []))} files indexed"
    )
    return {"status": "ok", "manifest": manifest}


def _tool_index_update(
    project: str = ".",
    progress_callback: Callable[[JSONDict], None] | None = None,
) -> JSONDict:
    _debug_log(f"accel_index_update called: project={project}")
    project_dir = _normalize_project_dir(project)
    config = resolve_effective_config(project_dir)
    manifest = build_or_update_indexes(
        project_dir=project_dir,
        config=config,
        mode="update",
        full=False,
        progress_callback=progress_callback,
    )
    file_count = int(manifest.get("counts", {}).get("files", 0) or 0)
    if file_count <= 0:
        retry_config = copy.deepcopy(config)
        retry_index_cfg = dict(retry_config.get("index", {}))
        retry_index_cfg["scope_mode"] = "all"
        retry_index_cfg["include"] = ["**/*"]
        retry_config["index"] = retry_index_cfg
        retry_manifest = build_or_update_indexes(
            project_dir=project_dir,
            config=retry_config,
            mode="update",
            full=False,
            progress_callback=progress_callback,
        )
        retry_file_count = int(retry_manifest.get("counts", {}).get("files", 0) or 0)
        if retry_file_count > file_count:
            retry_manifest["scope_retry_used"] = True
            retry_manifest["scope_retry_reason"] = "initial_scope_selected_zero_files"
            manifest = retry_manifest
    _debug_log(
        f"accel_index_update completed: {len(manifest.get('changed_files', []))} files updated"
    )
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
    runtime_cfg = (
        config.get("runtime", {}) if isinstance(config.get("runtime", {}), dict) else {}
    )
    require_changed_files = bool(
        runtime_cfg.get("context_require_changed_files", False)
    )
    strict_changed_files_override = _coerce_optional_bool(strict_changed_files)
    strict_changed_files_effective = (
        bool(strict_changed_files_override)
        if strict_changed_files_override is not None
        else False
    )
    semantic_cache_default = bool(runtime_cfg.get("semantic_cache_enabled", True))
    semantic_cache_enabled = _coerce_bool(semantic_cache, semantic_cache_default)
    semantic_cache_mode_default = (
        str(runtime_cfg.get("semantic_cache_mode", "hybrid")).strip().lower()
    )
    semantic_cache_mode_value = _resolve_semantic_cache_mode(
        semantic_cache_mode,
        default_mode=semantic_cache_mode_default,
    )
    constraint_mode_value = _resolve_constraint_mode(
        constraint_mode,
        default_mode=str(runtime_cfg.get("constraint_mode", "warn")),
    )

    changed_files_source = "user" if changed_files_list else "none"
    changed_files_fallback_reason = ""
    fallback_confidence = 1.0 if changed_files_list else 0.0
    changed_files_detection: JSONDict = {
        "provider": "user",
        "reason": "explicit_changed_files",
        "confidence": 1.0 if changed_files_list else 0.0,
        "sources": ["user_input"] if changed_files_list else [],
        "source_counts": {"user_input": len(changed_files_list)}
        if changed_files_list
        else {},
    }
    if not changed_files_list:
        auto_changed = _discover_changed_files_from_git(project_dir)
        _, git_detection = _discover_changed_files_from_git_details(project_dir)
        if auto_changed:
            changed_files_list = auto_changed
            changed_files_source = "git_auto"
            if not _coerce_bool(git_detection.get("available"), True):
                git_detection = {
                    "provider": "git",
                    "available": True,
                    "reason": "git_auto_legacy",
                    "sources": ["git_auto_legacy"],
                    "source_counts": {"git_auto_legacy": len(auto_changed)},
                    "confidence": 0.98,
                }
            fallback_confidence = float(git_detection.get("confidence", 0.98) or 0.98)
            changed_files_detection = {
                **dict(git_detection),
                "provider": "git",
                "reason": str(git_detection.get("reason", "ok")),
                "confidence": round(float(fallback_confidence), 6),
            }
        elif strict_changed_files_effective:
            changed_files_fallback_reason = "strict_changed_files_enabled_no_git_delta"
            changed_files_detection = {
                **dict(git_detection),
                "provider": "git",
                "reason": "strict_changed_files_enabled_no_git_delta",
                "confidence": float(git_detection.get("confidence", 0.0) or 0.0),
            }
        else:
            fallback_changed, fallback_reason, fallback_conf = (
                _discover_changed_files_from_index_fallback(
                    project_dir,
                    config=config,
                    task=task_text,
                    hints=hints_list,
                )
            )
            if fallback_changed:
                changed_files_list = fallback_changed
                changed_files_source = fallback_reason
                fallback_confidence = float(fallback_conf)
                changed_files_detection = {
                    "provider": "index_fallback",
                    "reason": fallback_reason,
                    "confidence": round(float(fallback_confidence), 6),
                    "sources": [fallback_reason],
                    "source_counts": {fallback_reason: len(fallback_changed)},
                }
            else:
                changed_files_fallback_reason = fallback_reason
                changed_files_detection = {
                    "provider": "index_fallback",
                    "reason": fallback_reason,
                    "confidence": 0.0,
                    "sources": [],
                    "source_counts": {},
                }
    changed_files_detection["selected_count"] = int(len(changed_files_list))
    changed_files_detection["selected_source"] = str(changed_files_source)
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

    budget_override, budget_source, budget_preset, budget_reason = (
        _resolve_context_budget(
            budget,
            task=task_text,
            changed_files=changed_files_list,
            hints=hints_list,
        )
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
        feedback_payload = json.loads(
            resolved_feedback_path.read_text(encoding="utf-8")
        )
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
    cache_safety_fingerprint, cache_safety_metadata = (
        _semantic_cache_safety_fingerprint(project_dir, changed_files_list)
    )
    task_sig = task_signature(task_tokens, hint_tokens)
    semantic_cache_key = make_stable_hash(
        {
            "task_signature": task_sig,
            "changed_fingerprint": changed_fingerprint,
            "budget_fingerprint": budget_fingerprint,
            "config_hash": config_hash,
            "safety_fingerprint": cache_safety_fingerprint,
        }
    )
    semantic_cache_hit = False
    semantic_cache_similarity = 0.0
    semantic_cache_mode_used = "off"
    semantic_cache_reason = "cache_disabled"
    semantic_cache_invalidation_reason = "none"
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
            semantic_cache_reason = "exact_key_match"
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
                    safety_fingerprint=cache_safety_fingerprint,
                    threshold=float(
                        runtime_cfg.get("semantic_cache_hybrid_threshold", 0.86)
                    ),
                )
            if isinstance(cached_hybrid, dict):
                pack = cached_hybrid
                semantic_cache_hit = True
                semantic_cache_similarity = float(hybrid_similarity)
                semantic_cache_mode_used = "hybrid"
                semantic_cache_reason = "hybrid_similarity_match"
            else:
                miss_explain = semantic_store.explain_context_miss(
                    task_signature_value=task_sig,
                    budget_fingerprint=budget_fingerprint,
                    config_hash=config_hash,
                    safety_fingerprint=cache_safety_fingerprint,
                    changed_fingerprint=changed_fingerprint,
                    git_head=str(cache_safety_metadata.get("git_head", "")),
                )
                semantic_cache_reason = "miss"
                semantic_cache_invalidation_reason = str(
                    miss_explain.get("reason", "no_prior_entry")
                )
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
                    safety_fingerprint=cache_safety_fingerprint,
                    git_head=str(cache_safety_metadata.get("git_head", "")),
                    changed_files_state=list(
                        cache_safety_metadata.get("changed_files_state", [])
                    ),
                    payload=pack,
                    ttl_seconds=int(
                        runtime_cfg.get("semantic_cache_ttl_seconds", 7200)
                    ),
                    max_entries=int(runtime_cfg.get("semantic_cache_max_entries", 800)),
                )
    else:
        if semantic_cache_enabled and semantic_store is None:
            semantic_cache_reason = "store_unavailable"
        _debug_log(
            "_tool_context: compile_context_pack start (semantic cache disabled)"
        )
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
        pack, strict_scope_filtered_top_files, strict_scope_filtered_snippets = (
            _apply_strict_changed_files_scope(
                pack,
                changed_files_list,
            )
        )
        pack, strict_scope_injected_top_files, strict_scope_injected_snippets = (
            _ensure_strict_changed_files_presence(
                pack,
                project_dir=project_dir,
                changed_files=changed_files_list,
            )
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
        fallback_chars_per_token=runtime_cfg.get(
            "token_estimator_fallback_chars_per_token", 4.0
        ),
    )
    source_token_estimate = estimate_tokens_from_chars(
        source_chars,
        chars_per_token=token_estimate.get(
            "chars_per_token",
            runtime_cfg.get("token_estimator_fallback_chars_per_token", 4.0),
        ),
        calibration=token_estimate.get(
            "calibration", runtime_cfg.get("token_estimator_calibration", 1.0)
        ),
    )

    compression_ratio = float(context_chars / source_chars) if source_chars > 0 else 1.0
    token_reduction_ratio = max(0.0, 1.0 - compression_ratio)
    snippets_only_text = json.dumps(
        {"snippets": list(pack.get("snippets", []))}, ensure_ascii=False
    )
    snippets_only_token_estimate = estimate_tokens_for_text(
        snippets_only_text,
        backend=runtime_cfg.get("token_estimator_backend", "auto"),
        model=runtime_cfg.get("token_estimator_model", ""),
        encoding=runtime_cfg.get("token_estimator_encoding", "cl100k_base"),
        calibration=runtime_cfg.get("token_estimator_calibration", 1.0),
        fallback_chars_per_token=runtime_cfg.get(
            "token_estimator_fallback_chars_per_token", 4.0
        ),
    )
    changed_files_chars = _estimate_changed_files_chars(project_dir, changed_files_list)
    changed_files_token_estimate = estimate_tokens_from_chars(
        changed_files_chars,
        chars_per_token=token_estimate.get(
            "chars_per_token",
            runtime_cfg.get("token_estimator_fallback_chars_per_token", 4.0),
        ),
        calibration=token_estimate.get(
            "calibration", runtime_cfg.get("token_estimator_calibration", 1.0)
        ),
    )
    context_tokens = int(token_estimate.get("estimated_tokens", 1))
    source_tokens = int(source_token_estimate.get("estimated_tokens", 1))
    changed_files_tokens = int(changed_files_token_estimate.get("estimated_tokens", 0))
    snippets_only_tokens = int(snippets_only_token_estimate.get("estimated_tokens", 1))
    token_reduction_vs_full_index = _token_reduction_ratio(
        context_tokens, source_tokens
    )
    token_reduction_vs_changed_files = (
        _token_reduction_ratio(context_tokens, changed_files_tokens)
        if changed_files_tokens > 0
        else None
    )
    token_reduction_vs_snippets_only = _token_reduction_ratio(
        context_tokens, snippets_only_tokens
    )
    warnings: list[str] = []
    if changed_files_source == "none":
        warnings.append(
            "changed_files not provided and no git delta detected; context scope may be wider than necessary"
        )
    elif changed_files_source in {
        "manifest_recent",
        "planner_fallback",
        "index_head_fallback",
    }:
        warnings.append(
            f"changed_files inferred via {changed_files_source}; provide explicit changed_files for tighter precision"
        )
    if (
        changed_files_source in {"planner_fallback", "index_head_fallback"}
        and fallback_confidence < 0.6
    ):
        warnings.append(
            "changed_files inference confidence is low; provide explicit changed_files for stable narrowing"
        )
    if changed_files_fallback_reason:
        warnings.append(
            f"changed_files fallback detail: {changed_files_fallback_reason}"
        )
    detection_confidence = float(
        changed_files_detection.get("confidence", fallback_confidence)
        or fallback_confidence
    )
    detection_provider = (
        str(changed_files_detection.get("provider", "")).strip().lower()
    )
    if detection_provider == "git" and detection_confidence < 0.95:
        warnings.append(
            "git changed_files detection confidence below expected deterministic threshold; "
            "consider passing changed_files explicitly"
        )
    if strict_changed_files_effective and (
        strict_scope_filtered_top_files > 0 or strict_scope_filtered_snippets > 0
    ):
        warnings.append(
            "strict_changed_files pruned non-changed context items "
            f"(top_files={strict_scope_filtered_top_files}, snippets={strict_scope_filtered_snippets})"
        )
    if strict_changed_files_effective and (
        strict_scope_injected_top_files > 0 or strict_scope_injected_snippets > 0
    ):
        warnings.append(
            "strict_changed_files injected changed-file fallback context "
            f"(top_files={strict_scope_injected_top_files}, snippets={strict_scope_injected_snippets})"
        )
    if semantic_cache_enabled and semantic_cache_reason == "store_unavailable":
        warnings.append(
            "semantic cache requested but storage backend is unavailable; fell back to fresh compilation"
        )

    verify_plan = (
        pack.get("verify_plan", {})
        if isinstance(pack.get("verify_plan", {}), dict)
        else {}
    )
    target_tests = verify_plan.get("target_tests", [])
    target_checks = verify_plan.get("target_checks", [])
    selected_tests_count = len(target_tests) if isinstance(target_tests, list) else 0
    selected_checks_count = len(target_checks) if isinstance(target_checks, list) else 0
    compression_rules_applied = {}
    compression_saved_chars = 0
    if isinstance(meta.get("compression_rules_applied", {}), dict):
        compression_rules_applied = dict(meta.get("compression_rules_applied", {}))
    compression_saved_chars = int(
        meta.get("compression_saved_chars", meta.get("snippet_saved_chars", 0)) or 0
    )

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
        "token_reduction_ratio_vs_snippets_only": round(
            token_reduction_vs_snippets_only, 6
        ),
        "token_reduction": {
            "vs_full_index": {
                "baseline_tokens": source_tokens,
                "context_tokens": context_tokens,
                "ratio": round(token_reduction_vs_full_index, 6),
            },
            "vs_changed_files": {
                "baseline_tokens": changed_files_tokens,
                "context_tokens": context_tokens,
                "ratio": round(token_reduction_vs_changed_files, 6)
                if token_reduction_vs_changed_files is not None
                else None,
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
            "encoding_requested": token_estimate.get(
                "encoding_requested", "cl100k_base"
            ),
            "encoding_used": token_estimate.get("encoding_used", "chars/4"),
            "model": token_estimate.get("model", ""),
            "calibration": float(token_estimate.get("calibration", 1.0)),
            "fallback_chars_per_token": float(
                token_estimate.get("fallback_chars_per_token", 4.0)
            ),
            "context_chars_per_token": round(
                float(token_estimate.get("chars_per_token", 4.0)), 6
            ),
            "source_chars_per_token": round(
                float(source_token_estimate.get("chars_per_token", 4.0)), 6
            ),
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
        "changed_files_detection": dict(changed_files_detection),
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
        "semantic_cache_reason": str(semantic_cache_reason),
        "semantic_cache_invalidation_reason": str(semantic_cache_invalidation_reason),
        "semantic_cache_safety_fingerprint": str(cache_safety_fingerprint),
        "semantic_cache_safety": {
            "git_head": str(cache_safety_metadata.get("git_head", "")),
            "changed_files_state_count": int(
                len(cache_safety_metadata.get("changed_files_state", []))
            ),
            "fingerprint": str(
                cache_safety_metadata.get("fingerprint", cache_safety_fingerprint)
            ),
        },
        "compression_rules_applied": compression_rules_applied,
        "compression_saved_chars": int(compression_saved_chars),
        "constraint_mode": constraint_mode_value,
    }
    if token_reduction_vs_changed_files is not None:
        payload["token_reduction_ratio_vs_changed_files"] = round(
            token_reduction_vs_changed_files, 6
        )
    fallback_reason = str(token_estimate.get("fallback_reason", "")).strip()
    if fallback_reason:
        token_estimator_payload = payload.get("token_estimator")
        if isinstance(token_estimator_payload, dict):
            token_estimator_payload["fallback_reason"] = fallback_reason
    payload["warnings"] = list(warnings)
    if include_pack_flag:
        payload["pack"] = pack_for_output

    payload_contract_warnings: list[str] = []
    payload_repair_count = 0
    payload, payload_contract_warnings, payload_repair_count = (
        enforce_context_payload_contract(
            payload,
            mode=constraint_mode_value,
        )
    )
    all_constraint_warnings = pack_contract_warnings + payload_contract_warnings
    constraint_repair_count = int(pack_repair_count) + int(payload_repair_count)
    payload["constraint_warnings"] = all_constraint_warnings
    payload["constraint_repair_count"] = int(constraint_repair_count)
    if all_constraint_warnings:
        merged_warnings = (
            list(payload.get("warnings", []))
            if isinstance(payload.get("warnings"), list)
            else []
        )
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
    verify_stall_timeout_seconds: float | None = None,
    verify_max_wall_time_seconds: float | None = None,
    verify_auto_cancel_on_stall: bool | str | None = None,
    verify_preset: str | None = None,
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
    verify_auto_cancel_on_stall = _coerce_optional_bool(verify_auto_cancel_on_stall)
    verify_preset_value = _resolve_verify_preset(verify_preset)
    verify_stall_timeout_value = _coerce_optional_float(verify_stall_timeout_seconds)
    verify_max_wall_time_value = _coerce_optional_float(verify_max_wall_time_seconds)

    _debug_log(
        f"accel_verify called: project={project}, changed_files={len(changed_files or [])}, evidence_run={evidence_run}"
    )
    project_dir = _normalize_project_dir(project)

    runtime_overrides: JSONDict = {}
    if verify_workers is not None:
        runtime_overrides["verify_workers"] = int(verify_workers)
    if per_command_timeout_seconds is not None:
        runtime_overrides["per_command_timeout_seconds"] = int(
            per_command_timeout_seconds
        )
    if verify_stall_timeout_value is not None:
        runtime_overrides["verify_stall_timeout_seconds"] = float(
            verify_stall_timeout_value
        )
    if verify_max_wall_time_value is not None:
        runtime_overrides["verify_max_wall_time_seconds"] = float(
            verify_max_wall_time_value
        )
    if verify_cache_ttl_seconds is not None:
        runtime_overrides["verify_cache_ttl_seconds"] = int(verify_cache_ttl_seconds)
    if verify_cache_max_entries is not None:
        runtime_overrides["verify_cache_max_entries"] = int(verify_cache_max_entries)
    if verify_cache_failed_ttl_seconds is not None:
        runtime_overrides["verify_cache_failed_ttl_seconds"] = int(
            verify_cache_failed_ttl_seconds
        )
    if command_plan_cache_enabled is not None:
        runtime_overrides["command_plan_cache_enabled"] = _coerce_bool(
            command_plan_cache_enabled, True
        )
    if constraint_mode is not None:
        runtime_overrides["constraint_mode"] = _resolve_constraint_mode(
            constraint_mode, default_mode="warn"
        )

    if verify_preset_value == "fast":
        runtime_overrides["verify_fail_fast"] = True
        runtime_overrides["verify_cache_enabled"] = True
        if verify_cache_failed_results is None:
            runtime_overrides["verify_cache_failed_results"] = True
    elif verify_preset_value == "full":
        runtime_overrides["verify_fail_fast"] = False
        runtime_overrides["verify_cache_enabled"] = False
    elif evidence_run and not fast_loop:
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
        runtime_overrides["verify_cache_failed_results"] = bool(
            verify_cache_failed_results
        )
    if verify_auto_cancel_on_stall is not None:
        runtime_overrides["verify_auto_cancel_on_stall"] = bool(
            verify_auto_cancel_on_stall
        )

    cli_overrides = {"runtime": runtime_overrides} if runtime_overrides else None
    config = resolve_effective_config(project_dir, cli_overrides=cli_overrides)

    result = run_verify(
        project_dir=project_dir,
        config=config,
        changed_files=_to_string_list(changed_files),
    )

    _debug_log(
        f"accel_verify completed: status={result.get('status')}, exit_code={result.get('exit_code')}"
    )
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
        description="Build indexes for the target project (bounded synchronous wait with async fallback).",
    )
    def accel_index_build(
        project: str = ".",
        full: Any = True,
        wait_for_completion: Any = True,
        sync_wait_seconds: Any = None,
    ) -> JSONDict:
        try:
            full_flag = _coerce_bool(full, True)
            wait_flag = _coerce_bool(wait_for_completion, True)
            job = _start_index_job(project=project, mode="build", full=full_flag)
            job_id = str(job.job_id).strip()

            if not wait_flag:
                status_payload = _normalize_job_status_payload(job.to_status())
                return {
                    "status": "started",
                    "exit_code": 0,
                    "timed_out": False,
                    "job_id": job_id,
                    "mode": "build",
                    "full": bool(full_flag),
                    "message": "Index build started asynchronously. Use accel_index_status/events for live progress.",
                    "poll_interval_sec": 1.0,
                    "state": status_payload.get("state", "pending"),
                    "stage": status_payload.get("stage", "indexing"),
                    "progress": status_payload.get("progress", 0.0),
                    "elapsed_sec": status_payload.get("elapsed_sec", 0.0),
                }

            wait_seconds = _resolve_sync_wait_seconds(
                project_dir=_normalize_project_dir(project),
                override_value=sync_wait_seconds,
                runtime_key="sync_index_wait_seconds",
                fallback_seconds=float(_effective_sync_index_wait_seconds()),
            )
            result = _wait_for_index_job_result(
                job_id,
                max_wait_seconds=wait_seconds,
                poll_seconds=_effective_sync_index_poll_seconds(),
            )
            if result is not None:
                return result
            return _sync_index_timeout_payload(job_id, wait_seconds)
        except Exception as exc:
            _debug_log(f"accel_index_build failed: {exc!r}")
            raise RuntimeError(f"{TOOL_ERROR_EXECUTION_FAILED}: {exc}") from exc

    @server.tool(
        name="accel_index_update",
        description="Incrementally update indexes for changed files (bounded synchronous wait with async fallback).",
    )
    def accel_index_update(
        project: str = ".",
        wait_for_completion: Any = True,
        sync_wait_seconds: Any = None,
    ) -> JSONDict:
        try:
            wait_flag = _coerce_bool(wait_for_completion, True)
            job = _start_index_job(project=project, mode="update", full=False)
            job_id = str(job.job_id).strip()

            if not wait_flag:
                status_payload = _normalize_job_status_payload(job.to_status())
                return {
                    "status": "started",
                    "exit_code": 0,
                    "timed_out": False,
                    "job_id": job_id,
                    "mode": "update",
                    "full": False,
                    "message": "Index update started asynchronously. Use accel_index_status/events for live progress.",
                    "poll_interval_sec": 1.0,
                    "state": status_payload.get("state", "pending"),
                    "stage": status_payload.get("stage", "indexing"),
                    "progress": status_payload.get("progress", 0.0),
                    "elapsed_sec": status_payload.get("elapsed_sec", 0.0),
                }

            wait_seconds = _resolve_sync_wait_seconds(
                project_dir=_normalize_project_dir(project),
                override_value=sync_wait_seconds,
                runtime_key="sync_index_wait_seconds",
                fallback_seconds=float(_effective_sync_index_wait_seconds()),
            )
            result = _wait_for_index_job_result(
                job_id,
                max_wait_seconds=wait_seconds,
                poll_seconds=_effective_sync_index_poll_seconds(),
            )
            if result is not None:
                return result
            return _sync_index_timeout_payload(job_id, wait_seconds)
        except Exception as exc:
            _debug_log(f"accel_index_update failed: {exc!r}")
            raise RuntimeError(f"{TOOL_ERROR_EXECUTION_FAILED}: {exc}") from exc

    @server.tool(
        name="accel_index_status",
        description="Get current status of an async index build/update job.",
    )
    def accel_index_status(job_id: str) -> JSONDict:
        try:
            jm = JobManager()
            job = jm.get_job(job_id)
            if job is None or not str(job_id).strip().lower().startswith("index_"):
                return {"error": "job_not_found", "job_id": job_id}
            status_payload = _normalize_job_status_payload(job.to_status())
            status_payload["processed_files"] = int(
                status_payload.get("completed_commands", 0) or 0
            )
            status_payload["total_files"] = int(
                status_payload.get("total_commands", 0) or 0
            )
            status_payload["current_path"] = str(
                status_payload.get("current_command", "")
            )
            status_payload["progress_pct"] = float(
                status_payload.get("progress", 0.0) or 0.0
            )
            status_payload["state_source"] = "job_state"
            return status_payload
        except Exception as exc:
            _debug_log(f"accel_index_status failed: {exc!r}")
            raise RuntimeError(f"{TOOL_ERROR_EXECUTION_FAILED}: {exc}") from exc

    @server.tool(
        name="accel_index_events",
        description="Get index job events with optional summary and tail clipping.",
    )
    def accel_index_events(
        job_id: str,
        since_seq: int = 0,
        max_events: Any = 30,
        include_summary: Any = True,
    ) -> JSONDict:
        try:
            jm = JobManager()
            job = jm.get_job(job_id)
            if job is None or not str(job_id).strip().lower().startswith("index_"):
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
                status_payload = _normalize_job_status_payload(job.to_status())
                event_type_counts: JSONDict = {}
                first_seq = 0
                last_seq = 0
                latest_progress_event: JSONDict = {}
                for item in events_all:
                    event_name = str(item.get("event", "")).strip().lower() or "unknown"
                    event_type_counts[event_name] = (
                        int(event_type_counts.get(event_name, 0)) + 1
                    )
                    seq = int(item.get("seq", 0) or 0)
                    if seq > 0:
                        if first_seq == 0:
                            first_seq = seq
                        last_seq = seq
                    if event_name == "index_progress":
                        latest_progress_event = dict(item)
                payload["summary"] = {
                    "latest_state": str(status_payload.get("state", "")),
                    "latest_stage": str(status_payload.get("stage", "")),
                    "state_source": "job_state",
                    "event_type_counts": event_type_counts,
                    "seq_range": {"first": int(first_seq), "last": int(last_seq)},
                    "constraint_repair_count": 0,
                    "progress": {
                        "processed_files": int(
                            status_payload.get("completed_commands", 0) or 0
                        ),
                        "total_files": int(
                            status_payload.get("total_commands", 0) or 0
                        ),
                        "current_path": str(status_payload.get("current_command", "")),
                        "eta_sec": status_payload.get("eta_sec"),
                        "progress_pct": float(
                            status_payload.get("progress", 0.0) or 0.0
                        ),
                        "latest_progress_event_seq": int(
                            latest_progress_event.get("seq", 0) or 0
                        ),
                    },
                }
            return payload
        except Exception as exc:
            _debug_log(f"accel_index_events failed: {exc!r}")
            raise RuntimeError(f"{TOOL_ERROR_EXECUTION_FAILED}: {exc}") from exc

    @server.tool(
        name="accel_index_cancel",
        description="Cancel a running async index job.",
    )
    def accel_index_cancel(job_id: str) -> JSONDict:
        try:
            jm = JobManager()
            job = jm.get_job(job_id)
            if job is None or not str(job_id).strip().lower().startswith("index_"):
                return {"error": "job_not_found", "job_id": job_id}
            if job.state in (JobState.COMPLETED, JobState.FAILED, JobState.CANCELLED):
                return {
                    "job_id": job_id,
                    "status": job.state,
                    "cancelled": False,
                    "message": "Job already in terminal state",
                }
            _finalize_job_cancel(job, reason="user_request")
            return {
                "job_id": job_id,
                "status": JobState.CANCELLED,
                "cancelled": True,
                "message": "Cancellation requested and finalized",
            }
        except Exception as exc:
            _debug_log(f"accel_index_cancel failed: {exc!r}")
            raise RuntimeError(f"{TOOL_ERROR_EXECUTION_FAILED}: {exc}") from exc

    def _resolve_context_rpc_timeout_seconds(
        *,
        project_dir: Path,
        rpc_timeout_seconds: Any = None,
        config: JSONDict | None = None,
    ) -> float:
        timeout_override = _coerce_optional_int(rpc_timeout_seconds)
        if timeout_override is not None:
            return max(30.0, min(3600.0, float(timeout_override)))
        effective_config = (
            config
            if isinstance(config, dict)
            else resolve_effective_config(project_dir)
        )
        runtime_cfg = (
            effective_config.get("runtime", {})
            if isinstance(effective_config.get("runtime", {}), dict)
            else {}
        )
        timeout_config = _coerce_optional_int(
            runtime_cfg.get("context_rpc_timeout_seconds")
        )
        if timeout_config is None:
            timeout_config = 300
        return max(30.0, min(3600.0, float(timeout_config)))

    def _summarize_context_events(
        events: list[dict[str, Any]], status: JSONDict
    ) -> JSONDict:
        event_type_counts: dict[str, int] = {}
        first_seq = 0
        last_seq = 0
        latest_state = str(status.get("state", ""))
        latest_stage = str(status.get("stage", ""))
        latest_error = ""
        terminal_event_seen = False
        state_source = "job_state"
        for event in events:
            event_name = str(event.get("event", "")).strip().lower()
            if not event_name:
                continue
            event_type_counts[event_name] = (
                int(event_type_counts.get(event_name, 0)) + 1
            )
            seq = int(event.get("seq", 0) or 0)
            if seq > 0:
                if first_seq <= 0:
                    first_seq = seq
                last_seq = max(last_seq, seq)
            state_value = str(event.get("state", "")).strip()
            stage_value = str(event.get("stage", "")).strip()
            if state_value:
                latest_state = state_value
                state_source = "events"
            if stage_value:
                latest_stage = stage_value
            if event_name == "context_failed":
                terminal_event_seen = True
                latest_state = JobState.FAILED
                latest_stage = "failed"
                state_source = "event_terminal"
                latest_error = str(event.get("error", "")).strip()
            if event_name == "context_completed":
                terminal_event_seen = True
                latest_state = JobState.COMPLETED
                latest_stage = "completed"
                state_source = "event_terminal"
            if event_name == "job_cancelled_finalized":
                terminal_event_seen = True
                latest_state = JobState.CANCELLED
                latest_stage = "cancelled"
                state_source = "event_terminal"
        return {
            "latest_state": latest_state or str(status.get("state", "")),
            "latest_stage": latest_stage or str(status.get("stage", "")),
            "state_source": state_source,
            "terminal_event_seen": bool(terminal_event_seen),
            "event_type_counts": event_type_counts,
            "seq_range": {"first": int(first_seq), "last": int(last_seq)},
            "latest_error": latest_error,
            "constraint_repair_count": 0,
        }

    def _start_context_job(
        *,
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
        rpc_timeout_seconds: Any = None,
    ) -> VerifyJob:
        task_text = str(task).strip()
        if not task_text:
            raise ValueError("task is required")
        project_dir = _normalize_project_dir(project)
        config_preview = resolve_effective_config(project_dir)
        runtime_preview = (
            config_preview.get("runtime", {})
            if isinstance(config_preview.get("runtime", {}), dict)
            else {}
        )
        _resolve_semantic_cache_mode(
            semantic_cache_mode,
            default_mode=str(runtime_preview.get("semantic_cache_mode", "hybrid")),
        )
        _resolve_constraint_mode(
            constraint_mode,
            default_mode=str(runtime_preview.get("constraint_mode", "warn")),
        )
        timeout_seconds = _resolve_context_rpc_timeout_seconds(
            project_dir=project_dir,
            rpc_timeout_seconds=rpc_timeout_seconds,
            config=config_preview,
        )

        jm = JobManager()
        job = jm.create_job(prefix="context")
        changed_files_list = _to_string_list(changed_files)
        hints_list = _to_string_list(hints)
        job.add_event(
            "context_queued",
            {
                "project": str(project_dir),
                "changed_files_count": int(len(changed_files_list)),
                "hints_count": int(len(hints_list)),
                "timeout_seconds": float(timeout_seconds),
            },
        )

        def _run_context_thread(job_id: str) -> None:
            j = jm.get_job(job_id)
            if j is None:
                return
            try:
                j.mark_running("running")
                j.update_progress(0, 1, "compile_context_pack")
                j.add_live_event(
                    "context_started",
                    {
                        "changed_files_count": int(len(changed_files_list)),
                        "hints_count": int(len(hints_list)),
                        "timeout_seconds": float(timeout_seconds),
                    },
                )
                result = _with_timeout(
                    _tool_context, timeout_seconds=int(timeout_seconds)
                )(
                    project=project,
                    task=task_text,
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
                if j.state in (JobState.CANCELLING, JobState.CANCELLED):
                    if j.state != JobState.CANCELLED:
                        j.mark_cancelled()
                    j.add_event(
                        "job_cancelled_finalized",
                        {"reason": "context_worker_observed_cancel"},
                    )
                    return
                j.update_progress(1, 1, "")
                j.add_event(
                    "context_completed",
                    {
                        "status": str(result.get("status", "ok")),
                        "out": str(result.get("out", "")),
                        "top_files": int(result.get("top_files", 0) or 0),
                        "snippets": int(result.get("snippets", 0) or 0),
                    },
                )
                j.mark_completed(
                    str(result.get("status", "ok")),
                    int(result.get("exit_code", 0) or 0),
                    result,
                )
            except Exception as exc:
                if j.state in (JobState.CANCELLING, JobState.CANCELLED):
                    if j.state != JobState.CANCELLED:
                        j.mark_cancelled()
                    j.add_event(
                        "job_cancelled_finalized",
                        {"reason": "context_worker_exception_after_cancel"},
                    )
                    return
                j.mark_failed(str(exc))
                j.add_event("context_failed", {"error": str(exc)})

        thread = threading.Thread(
            target=_run_context_thread, args=(job.job_id,), daemon=True
        )
        thread.start()

        def _heartbeat_thread(job_id: str) -> None:
            while True:
                current = jm.get_job(job_id)
                if current is None:
                    return
                status = _normalize_job_status_payload(current.to_status())
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
                        "state_consistency": str(
                            status.get("state_consistency", "normal")
                        ),
                        "current_command": str(status.get("current_command", "")),
                    },
                )
                if not appended:
                    return
                time.sleep(1.0)

        hb = threading.Thread(target=_heartbeat_thread, args=(job.job_id,), daemon=True)
        hb.start()
        return job

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
        rpc_timeout_seconds: Any = None,
        wait_for_completion: Any = True,
        sync_wait_seconds: Any = None,
        sync_timeout_action: Any = None,
    ) -> JSONDict:
        try:
            wait_flag = _coerce_bool(wait_for_completion, True)
            project_dir = _normalize_project_dir(project)
            timeout_action_value = _resolve_context_sync_timeout_action(
                project_dir=project_dir,
                override_value=sync_timeout_action,
            )
            job = _start_context_job(
                task=task,
                project=project,
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
                rpc_timeout_seconds=rpc_timeout_seconds,
            )
            job_id = str(job.job_id).strip()

            if not wait_flag:
                status_payload = _normalize_job_status_payload(job.to_status())
                return {
                    "status": "started",
                    "exit_code": 0,
                    "timed_out": False,
                    "job_id": job_id,
                    "message": "Context generation started asynchronously. Use accel_context_status/events for live progress.",
                    "poll_interval_sec": 1.0,
                    "state": status_payload.get("state", "pending"),
                    "stage": status_payload.get("stage", "running"),
                    "progress": status_payload.get("progress", 0.0),
                    "elapsed_sec": status_payload.get("elapsed_sec", 0.0),
                }

            wait_seconds = _resolve_sync_wait_seconds(
                project_dir=project_dir,
                override_value=sync_wait_seconds,
                runtime_key="sync_context_wait_seconds",
                fallback_seconds=float(_effective_sync_context_wait_seconds()),
            )
            result = _wait_for_context_job_result(
                job_id,
                max_wait_seconds=wait_seconds,
                poll_seconds=_effective_sync_context_poll_seconds(),
            )
            if result is not None:
                return result

            if timeout_action_value == "cancel":
                jm = JobManager()
                cancel_requested = False
                job_for_cancel = jm.get_job(job_id)
                if job_for_cancel is not None:
                    cancel_requested = _finalize_job_cancel(
                        job_for_cancel, reason="sync_timeout_auto_cancel"
                    )
                return _timeout_cancel_payload_context(
                    job_id=job_id,
                    wait_seconds=wait_seconds,
                    cancel_requested=cancel_requested,
                )
            return _sync_context_timeout_payload(
                job_id,
                wait_seconds,
                timeout_action=timeout_action_value,
            )
        except Exception as exc:
            _debug_log(f"accel_context failed: {exc!r}")
            raise RuntimeError(f"{TOOL_ERROR_EXECUTION_FAILED}: {exc}") from exc

    @server.tool(
        name="accel_context_start",
        description="Start an async context-pack generation job. Returns job_id immediately.",
    )
    def accel_context_start(
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
        rpc_timeout_seconds: Any = None,
    ) -> JSONDict:
        try:
            job = _start_context_job(
                task=task,
                project=project,
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
                rpc_timeout_seconds=rpc_timeout_seconds,
            )
            return {"job_id": job.job_id, "status": "started"}
        except Exception as exc:
            _debug_log(f"accel_context_start failed: {exc!r}")
            raise RuntimeError(f"{TOOL_ERROR_EXECUTION_FAILED}: {exc}") from exc

    @server.tool(
        name="accel_context_status",
        description="Get current status of an async context generation job.",
    )
    def accel_context_status(job_id: str) -> JSONDict:
        try:
            jm = JobManager()
            job = jm.get_job(job_id)
            if job is None or not str(job_id).strip().lower().startswith("context_"):
                return {"error": "job_not_found", "job_id": job_id}
            status_payload = _normalize_job_status_payload(job.to_status())
            if job.state == JobState.COMPLETED and isinstance(job.result, dict):
                for key in (
                    "status",
                    "out",
                    "out_meta",
                    "top_files",
                    "snippets",
                    "selected_tests_count",
                    "selected_checks_count",
                    "warnings",
                    "constraint_mode",
                    "semantic_cache_hit",
                ):
                    if key in job.result:
                        status_payload[key] = job.result.get(key)
            status_payload["state_source"] = "job_state"
            return status_payload
        except Exception as exc:
            _debug_log(f"accel_context_status failed: {exc!r}")
            raise RuntimeError(f"{TOOL_ERROR_EXECUTION_FAILED}: {exc}") from exc

    @server.tool(
        name="accel_context_events",
        description="Get context job events with optional summary and tail clipping.",
    )
    def accel_context_events(
        job_id: str,
        since_seq: int = 0,
        max_events: Any = 30,
        include_summary: Any = True,
    ) -> JSONDict:
        try:
            jm = JobManager()
            job = jm.get_job(job_id)
            if job is None or not str(job_id).strip().lower().startswith("context_"):
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
                status_payload = _normalize_job_status_payload(job.to_status())
                payload["summary"] = _summarize_context_events(
                    events_all, status_payload
                )
            return payload
        except Exception as exc:
            _debug_log(f"accel_context_events failed: {exc!r}")
            raise RuntimeError(f"{TOOL_ERROR_EXECUTION_FAILED}: {exc}") from exc

    @server.tool(
        name="accel_context_cancel",
        description="Cancel a running async context generation job.",
    )
    def accel_context_cancel(job_id: str) -> JSONDict:
        try:
            jm = JobManager()
            job = jm.get_job(job_id)
            if job is None or not str(job_id).strip().lower().startswith("context_"):
                return {"error": "job_not_found", "job_id": job_id}
            if job.state in (JobState.COMPLETED, JobState.FAILED, JobState.CANCELLED):
                return {
                    "job_id": job_id,
                    "status": job.state,
                    "cancelled": False,
                    "message": "Job already in terminal state",
                }
            _finalize_job_cancel(job, reason="user_request")
            return {
                "job_id": job_id,
                "status": JobState.CANCELLED,
                "cancelled": True,
                "message": "Cancellation requested and finalized",
            }
        except Exception as exc:
            _debug_log(f"accel_context_cancel failed: {exc!r}")
            raise RuntimeError(f"{TOOL_ERROR_EXECUTION_FAILED}: {exc}") from exc

    def _start_verify_job(
        *,
        project: str = ".",
        changed_files: Any = None,
        evidence_run: Any = False,
        fast_loop: Any = False,
        verify_workers: Any = None,
        per_command_timeout_seconds: Any = None,
        verify_stall_timeout_seconds: Any = None,
        verify_max_wall_time_seconds: Any = None,
        verify_auto_cancel_on_stall: bool | str | None = None,
        verify_preset: Any = None,
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
        verify_cache_failed_results_normalized = _coerce_optional_bool(
            verify_cache_failed_results
        )
        verify_auto_cancel_on_stall_normalized = _coerce_optional_bool(
            verify_auto_cancel_on_stall
        )
        verify_preset_value = _resolve_verify_preset(verify_preset)
        verify_workers_value = _coerce_optional_int(verify_workers)
        per_command_timeout_value = _coerce_optional_int(per_command_timeout_seconds)
        verify_stall_timeout_value = _coerce_optional_float(
            verify_stall_timeout_seconds
        )
        verify_max_wall_time_value = _coerce_optional_float(
            verify_max_wall_time_seconds
        )
        verify_cache_ttl_value = _coerce_optional_int(verify_cache_ttl_seconds)
        verify_cache_max_entries_value = _coerce_optional_int(verify_cache_max_entries)
        verify_cache_failed_ttl_value = _coerce_optional_int(
            verify_cache_failed_ttl_seconds
        )
        command_plan_cache_enabled_normalized = _coerce_optional_bool(
            command_plan_cache_enabled
        )

        _debug_log(
            f"_start_verify_job called: project={project}, changed_files={len(changed_files_list)}"
        )
        project_dir = _normalize_project_dir(project)

        runtime_overrides: JSONDict = {}
        if verify_workers_value is not None:
            runtime_overrides["verify_workers"] = int(verify_workers_value)
        if per_command_timeout_value is not None:
            runtime_overrides["per_command_timeout_seconds"] = int(
                per_command_timeout_value
            )
        if verify_stall_timeout_value is not None:
            runtime_overrides["verify_stall_timeout_seconds"] = float(
                verify_stall_timeout_value
            )
        if verify_max_wall_time_value is not None:
            runtime_overrides["verify_max_wall_time_seconds"] = float(
                verify_max_wall_time_value
            )
        if verify_cache_ttl_value is not None:
            runtime_overrides["verify_cache_ttl_seconds"] = int(verify_cache_ttl_value)
        if verify_cache_max_entries_value is not None:
            runtime_overrides["verify_cache_max_entries"] = int(
                verify_cache_max_entries_value
            )
        if verify_cache_failed_ttl_value is not None:
            runtime_overrides["verify_cache_failed_ttl_seconds"] = int(
                verify_cache_failed_ttl_value
            )
        if command_plan_cache_enabled_normalized is not None:
            runtime_overrides["command_plan_cache_enabled"] = bool(
                command_plan_cache_enabled_normalized
            )
        if constraint_mode is not None:
            runtime_overrides["constraint_mode"] = _resolve_constraint_mode(
                constraint_mode, default_mode="warn"
            )

        if verify_preset_value == "fast":
            runtime_overrides["verify_fail_fast"] = True
            runtime_overrides["verify_cache_enabled"] = True
            if verify_cache_failed_results_normalized is None:
                runtime_overrides["verify_cache_failed_results"] = True
        elif verify_preset_value == "full":
            runtime_overrides["verify_fail_fast"] = False
            runtime_overrides["verify_cache_enabled"] = False
        elif evidence_run_normalized and not fast_loop_normalized:
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
            runtime_overrides["verify_cache_enabled"] = bool(
                verify_cache_enabled_normalized
            )
        if verify_cache_failed_results_normalized is not None:
            runtime_overrides["verify_cache_failed_results"] = bool(
                verify_cache_failed_results_normalized
            )
        if verify_auto_cancel_on_stall_normalized is not None:
            runtime_overrides["verify_auto_cancel_on_stall"] = bool(
                verify_auto_cancel_on_stall_normalized
            )

        cli_overrides = {"runtime": runtime_overrides} if runtime_overrides else None
        config = resolve_effective_config(project_dir, cli_overrides=cli_overrides)

        jm = JobManager()
        job = jm.create_job()

        def _run_verify_thread(
            job_id: str,
            project_dir: Path,
            config: dict[str, Any],
            changed_files: list[str] | None,
        ) -> None:
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
                    j.add_event(
                        "job_cancelled_finalized", {"reason": "worker_observed_cancel"}
                    )
                    return
                j.mark_completed(
                    result.get("status", "unknown"), result.get("exit_code", 1), result
                )
            except Exception as exc:
                if j.state in (JobState.CANCELLING, JobState.CANCELLED):
                    if j.state != JobState.CANCELLED:
                        j.mark_cancelled()
                    j.add_event(
                        "job_cancelled_finalized",
                        {"reason": "worker_exception_after_cancel"},
                    )
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
                status = _normalize_job_status_payload(current.to_status())
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
                        "completed_commands": int(
                            status.get("completed_commands", 0) or 0
                        ),
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
        verify_stall_timeout_seconds: Any = None,
        verify_max_wall_time_seconds: Any = None,
        verify_auto_cancel_on_stall: Any = None,
        verify_preset: Any = None,
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
                verify_stall_timeout_seconds=verify_stall_timeout_seconds,
                verify_max_wall_time_seconds=verify_max_wall_time_seconds,
                verify_auto_cancel_on_stall=verify_auto_cancel_on_stall,
                verify_preset=verify_preset,
                verify_fail_fast=verify_fail_fast,
                verify_cache_enabled=verify_cache_enabled,
                verify_cache_failed_results=verify_cache_failed_results,
                verify_cache_ttl_seconds=verify_cache_ttl_seconds,
                verify_cache_max_entries=verify_cache_max_entries,
                verify_cache_failed_ttl_seconds=verify_cache_failed_ttl_seconds,
                command_plan_cache_enabled=command_plan_cache_enabled,
                constraint_mode=str(constraint_mode)
                if constraint_mode is not None
                else None,
            )
            job_id = str(job.job_id).strip()

            if not wait_flag:
                status_payload = _normalize_job_status_payload(job.to_status())
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

            wait_seconds = _resolve_sync_wait_seconds(
                project_dir=project_dir,
                override_value=sync_wait_seconds,
                runtime_key="sync_verify_wait_seconds",
                fallback_seconds=float(_effective_sync_verify_wait_seconds()),
            )
            default_timeout_action, default_cancel_grace = (
                _resolve_sync_timeout_defaults(project_dir)
            )
            timeout_action_value = _coerce_sync_timeout_action(
                sync_timeout_action, default=default_timeout_action
            )
            cancel_grace_value = _coerce_optional_float(sync_cancel_grace_seconds)
            cancel_grace_seconds = (
                float(cancel_grace_value)
                if cancel_grace_value is not None
                else float(default_cancel_grace)
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
                        cancel_requested = _finalize_job_cancel(
                            job_for_cancel, reason="sync_timeout_auto_cancel"
                        )
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
                return _sync_verify_timeout_payload(
                    job_id, wait_seconds, timeout_action=timeout_action_value
                )
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

        def on_command_start(
            self, job_id: str, command: str, index: int, total: int
        ) -> None:
            if self._job.is_terminal():
                return
            self._job.current_command = command
            self._job.add_live_event(
                "command_start", {"command": command, "index": index, "total": total}
            )

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

        def on_progress(
            self, job_id: str, completed: int, total: int, current_command: str
        ) -> None:
            if self._job.is_terminal():
                return
            self._job.update_progress(completed, total, current_command)
            progress_pct = float(self._job.progress)
            if (
                total > 0
                and completed >= total
                and self._job.state in {JobState.RUNNING, JobState.CANCELLING}
            ):
                progress_pct = min(progress_pct, 99.9)
            self._job.add_live_event(
                "progress",
                {
                    "completed": completed,
                    "total": total,
                    "progress_pct": round(progress_pct, 1),
                },
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
            stall_detected: bool | None = None,
            stall_elapsed_sec: float | None = None,
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
                heartbeat_payload["command_elapsed_sec"] = round(
                    float(command_elapsed_sec), 1
                )
            if command_timeout_sec is not None:
                heartbeat_payload["command_timeout_sec"] = round(
                    float(command_timeout_sec), 1
                )
            if command_progress_pct is not None:
                heartbeat_payload["command_progress_pct"] = round(
                    float(command_progress_pct), 2
                )
            if stall_detected is not None:
                heartbeat_payload["stall_detected"] = bool(stall_detected)
            if stall_elapsed_sec is not None:
                heartbeat_payload["stall_elapsed_sec"] = round(
                    float(stall_elapsed_sec), 1
                )
            self._job.add_live_event("heartbeat", heartbeat_payload)

        def on_command_output(
            self,
            job_id: str,
            command: str,
            stream: str,
            chunk: str,
            *,
            truncated: bool = False,
        ) -> None:
            if self._job.is_terminal():
                return
            text = str(chunk or "").strip()
            if not text:
                return
            if len(text) > _VERIFY_OUTPUT_CHUNK_LIMIT:
                text = text[-_VERIFY_OUTPUT_CHUNK_LIMIT:]
                truncated = True
            self._job.add_live_event(
                "command_output",
                {
                    "command": command,
                    "stream": str(stream or "stdout"),
                    "chunk": text,
                    "truncated": bool(truncated),
                },
            )

        def on_cache_hit(self, job_id: str, command: str) -> None:
            if self._job.is_terminal():
                return
            self._job.add_live_event("cache_hit", {"command": command})

        def on_skip(self, job_id: str, command: str, reason: str) -> None:
            if self._job.is_terminal():
                return
            self._job.add_live_event(
                "command_skipped", {"command": command, "reason": reason}
            )

        def on_error(self, job_id: str, command: str | None, error: str) -> None:
            if self._job.is_terminal():
                return
            self._job.add_live_event("error", {"command": command, "error": error})

        def on_complete(self, job_id: str, status: str, exit_code: int) -> None:
            if self._job.is_terminal():
                return
            self._job.add_live_event(
                "job_completed", {"status": status, "exit_code": exit_code}
            )

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
        verify_stall_timeout_seconds: Any = None,
        verify_max_wall_time_seconds: Any = None,
        verify_auto_cancel_on_stall: Any = None,
        verify_preset: Any = None,
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
                verify_stall_timeout_seconds=verify_stall_timeout_seconds,
                verify_max_wall_time_seconds=verify_max_wall_time_seconds,
                verify_auto_cancel_on_stall=verify_auto_cancel_on_stall,
                verify_preset=verify_preset,
                verify_fail_fast=verify_fail_fast,
                verify_cache_enabled=verify_cache_enabled,
                verify_cache_failed_results=verify_cache_failed_results,
                verify_cache_ttl_seconds=verify_cache_ttl_seconds,
                verify_cache_max_entries=verify_cache_max_entries,
                verify_cache_failed_ttl_seconds=verify_cache_failed_ttl_seconds,
                command_plan_cache_enabled=command_plan_cache_enabled,
                constraint_mode=str(constraint_mode)
                if constraint_mode is not None
                else None,
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
            status_payload = _normalize_job_status_payload(job.to_status())
            if job.state == JobState.COMPLETED and isinstance(job.result, dict):
                for key in (
                    "status",
                    "exit_code",
                    "partial",
                    "partial_reason",
                    "unfinished_commands",
                    "unfinished_items",
                    "failure_kind",
                    "failed_commands",
                    "executor_failed_commands",
                    "project_failed_commands",
                    "failure_counts",
                    "timed_out",
                    "max_wall_time_seconds",
                    "auto_cancel_on_stall",
                ):
                    if key in job.result:
                        status_payload[key] = job.result.get(key)
            tail_events = job.get_events(0)[-40:]
            recent_output: list[JSONDict] = []
            stall_detected = False
            stall_elapsed_sec = 0.0
            for event in tail_events:
                event_name = str(event.get("event", "")).strip().lower()
                if event_name == "command_output":
                    chunk = str(event.get("chunk", "")).strip()
                    if chunk:
                        recent_output.append(
                            {
                                "seq": int(event.get("seq", 0) or 0),
                                "stream": str(event.get("stream", "stdout")),
                                "command": str(event.get("command", "")),
                                "chunk": chunk,
                            }
                        )
                        if len(recent_output) > 5:
                            recent_output = recent_output[-5:]
                if event_name == "heartbeat" and _coerce_bool(
                    event.get("stall_detected"), False
                ):
                    stall_detected = True
                    stall_elapsed_sec = max(
                        stall_elapsed_sec,
                        float(event.get("stall_elapsed_sec", 0.0) or 0.0),
                    )
            status_payload["recent_output"] = recent_output
            status_payload["stall_detected"] = bool(stall_detected)
            if stall_detected:
                status_payload["stall_elapsed_sec"] = round(float(stall_elapsed_sec), 1)
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
                status_payload = _normalize_job_status_payload(job.to_status())
                summary = _summarize_verify_events(events_all, status_payload)
                summary_mode = normalize_constraint_mode(
                    os.environ.get("ACCEL_CONSTRAINT_MODE", "warn"), default_mode="warn"
                )
                summary, summary_warnings, summary_repair_count = (
                    enforce_verify_summary_contract(
                        summary,
                        status=status_payload,
                        mode=summary_mode,
                    )
                )
                summary["constraint_repair_count"] = int(summary_repair_count)
                if summary_warnings:
                    summary["constraint_warnings"] = summary_warnings
                payload["summary"] = summary
            payload_mode = normalize_constraint_mode(
                os.environ.get("ACCEL_CONSTRAINT_MODE", "warn"), default_mode="warn"
            )
            payload, payload_warnings, payload_repairs = (
                enforce_verify_events_payload_contract(payload, mode=payload_mode)
            )
            payload["constraint_repair_count"] = int(payload_repairs)
            if payload_warnings:
                payload["constraint_warnings"] = payload_warnings
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
                return {
                    "job_id": job_id,
                    "status": job.state,
                    "cancelled": False,
                    "message": "Job already in terminal state",
                }
            _finalize_job_cancel(job, reason="user_request")
            return {
                "job_id": job_id,
                "status": JobState.CANCELLED,
                "cancelled": True,
                "message": "Cancellation requested and finalized",
            }
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
        return json.dumps(
            templates.get(key, {"kind": key, "template": None}), ensure_ascii=False
        )

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
    if hasattr(signal, "SIGBREAK"):  # Windows-specific
        signal.signal(signal.SIGBREAK, _signal_handler)

    args = build_parser().parse_args()

    if _debug_enabled:
        _debug_log(f"Starting agent-accel MCP server with args: {vars(args)}")
        print(
            "agent-accel MCP debug mode enabled. Logs will be written to ~/.accel/logs/",
            file=sys.stderr,
        )
        print(f"Server max runtime: {_server_max_runtime}s", file=sys.stderr)

    server = create_server()
    _server_start_time = time.perf_counter()

    try:
        _debug_log(f"Running FastMCP server with transport: {args.transport}")
        if bool(args.show_banner):
            _debug_log(
                "Ignoring --show-banner for stdio transport to preserve MCP framing"
            )

        # Run server with timeout monitoring
        # Note: FastMCP's server.run() is blocking, so we can't directly add timeout here
        # But we have runtime checks in tool wrappers
        server.run(transport=args.transport, show_banner=False)

    except KeyboardInterrupt:
        _debug_log("Server interrupted by user")
    except Exception as exc:
        _debug_log(f"Server crashed: {exc!r}")
        raise
    finally:
        _debug_log("Server shutdown complete")


if __name__ == "__main__":
    main()
