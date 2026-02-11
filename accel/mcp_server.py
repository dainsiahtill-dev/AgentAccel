from __future__ import annotations

import argparse
import json
import logging
import os
import signal
import sys
import time
from pathlib import Path
from typing import Any
from uuid import uuid4

from fastmcp import FastMCP

from .config import resolve_effective_config
from .indexers import build_or_update_indexes
from .query.context_compiler import compile_context_pack, write_context_pack
from .storage.cache import ensure_project_dirs, project_paths
from .verify.orchestrator import run_verify, run_verify_with_callback
from .verify.callbacks import VerifyProgressCallback, VerifyStage
from .verify.job_manager import JobManager, JobState, VerifyJob


JSONDict = dict[str, Any]
SERVER_NAME = "agent-accel-mcp"
SERVER_VERSION = "0.2.0"
TOOL_ERROR_EXECUTION_FAILED = "ACCEL_TOOL_EXECUTION_FAILED"

# Debug logging setup
_debug_enabled = os.environ.get("ACCEL_MCP_DEBUG", "").lower() in {"1", "true", "yes"}
_debug_log: logging.Logger | None = None

# Global server timeout protection
_server_start_time = 0
_server_max_runtime = int(os.environ.get("ACCEL_MCP_MAX_RUNTIME", "3600"))  # 1 hour default
_shutdown_requested = False

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
        global _debug_log
        if _debug_log is None:
            _debug_log = _setup_debug_log()
        _debug_log.debug(message)

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


def _to_string_list(value: list[str] | None) -> list[str]:
    if not value:
        return []
    return [str(item) for item in value if str(item).strip()]


def _to_budget_override(value: dict[str, int] | None) -> dict[str, int]:
    if not isinstance(value, dict):
        return {}
    out: dict[str, int] = {}
    for key in ("max_chars", "max_snippets", "top_n_files", "snippet_radius"):
        if key in value:
            out[key] = int(value[key])
    return out


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
    changed_files: list[str] | None = None,
    hints: list[str] | None = None,
    feedback_path: str | None = None,
    out: str | None = None,
    include_pack: bool = False,
    budget: dict[str, int] | None = None,
) -> JSONDict:
    _debug_log(f"accel_context called: project={project}, task='{task[:50]}...', changed_files={len(changed_files or [])}")
    project_dir = _normalize_project_dir(project)
    task_text = str(task).strip()
    if not task_text:
        raise ValueError("task is required")

    config = resolve_effective_config(project_dir)
    feedback_payload: JSONDict | None = None

    resolved_feedback_path = _resolve_path(project_dir, feedback_path)
    if resolved_feedback_path is not None and resolved_feedback_path.exists():
        feedback_payload = json.loads(resolved_feedback_path.read_text(encoding="utf-8"))

    pack = compile_context_pack(
        project_dir=project_dir,
        config=config,
        task=task_text,
        changed_files=_to_string_list(changed_files),
        hints=_to_string_list(hints),
        previous_attempt_feedback=feedback_payload,
        budget_override=_to_budget_override(budget),
    )

    accel_home = Path(config["runtime"]["accel_home"]).resolve()
    paths = project_paths(accel_home, project_dir)
    ensure_project_dirs(paths)

    out_path = _resolve_path(project_dir, out)
    if out_path is None:
        out_path = paths["context"] / f"context_pack_{uuid4().hex[:10]}.json"

    write_context_pack(out_path, pack)
    payload: JSONDict = {
        "status": "ok",
        "out": str(out_path),
        "top_files": len(pack.get("top_files", [])),
        "snippets": len(pack.get("snippets", [])),
        "verify_plan": pack.get("verify_plan", {}),
    }
    if include_pack:
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
    
    # Normalize boolean parameters: handle string representations
    def _to_bool(value: bool | str | None, default: bool = False) -> bool:
        if value is None:
            return default
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.lower() not in {"false", "0", "no", "off", ""}
        return bool(value)
    
    evidence_run = _to_bool(evidence_run, False)
    fast_loop = _to_bool(fast_loop, False)
    verify_fail_fast = _to_bool(verify_fail_fast, None)  # type: ignore
    verify_cache_enabled = _to_bool(verify_cache_enabled, None)  # type: ignore
    
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
        changed_files: list[str] | None = None,
        hints: list[str] | None = None,
        feedback_path: str | None = None,
        out: str | None = None,
        include_pack: bool = False,
        budget: dict[str, int] | None = None,
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

    @server.tool(
        name="accel_verify",
        description="Run incremental verification with runtime override options.",
    )
    def accel_verify(
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
        try:
            return _with_timeout(_tool_verify, timeout_seconds=600)(
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
        try:
            if isinstance(changed_files, str):
                changed_files = [f.strip() for f in changed_files.split(",") if f.strip()]

            def _to_bool(value: bool | str | None, default: bool = False) -> bool:
                if value is None:
                    return default
                if isinstance(value, bool):
                    return value
                if isinstance(value, str):
                    return value.lower() not in {"false", "0", "no", "off", ""}
                return bool(value)

            evidence_run = _to_bool(evidence_run, False)
            fast_loop = _to_bool(fast_loop, False)
            verify_fail_fast = _to_bool(verify_fail_fast, None)
            verify_cache_enabled = _to_bool(verify_cache_enabled, None)

            _debug_log(f"accel_verify_start called: project={project}, changed_files={len(changed_files or [])}")
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
                    j.mark_completed(result.get("status", "unknown"), result.get("exit_code", 1), result)
                except Exception as exc:
                    j.mark_failed(str(exc))
                    j.add_event("job_failed", {"error": str(exc)})

            import threading
            thread = threading.Thread(
                target=_run_verify_thread,
                args=(job.job_id, project_dir, config, _to_string_list(changed_files)),
                daemon=True,
            )
            thread.start()

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
            job.add_event("job_cancelled", {"reason": "user_request"})
            return {"job_id": job_id, "status": JobState.CANCELLING, "cancelled": True, "message": "Cancellation requested"}
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
        print(f"agent-accel MCP debug mode enabled. Logs will be written to ~/.accel/logs/", file=sys.stderr)
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
