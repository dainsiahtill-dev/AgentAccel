from __future__ import annotations

import hashlib
import json
import os
import re
import shlex
import shutil
import subprocess
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable
from uuid import uuid4

from .runners import run_command
from .sharding import select_verify_commands
from ..storage.cache import ensure_project_dirs, project_paths
from .callbacks import VerifyProgressCallback, VerifyStage, NoOpCallback


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _command_binary(command: str) -> str:
    effective_command = _effective_shell_command(command)
    parts = shlex.split(effective_command, posix=os.name != "nt")
    return parts[0] if parts else ""


def _effective_shell_command(command: str) -> str:
    text = str(command or "").strip()
    if "&&" not in text:
        return text
    parts = [part.strip() for part in text.split("&&") if part.strip()]
    return parts[-1] if parts else text


def _command_workdir(project_dir: Path, command: str) -> Path:
    text = str(command or "").strip()
    if "&&" not in text:
        return project_dir
    first, _, _tail = text.partition("&&")
    prefix = first.strip()
    match = re.match(r'^cd\s+(?:/d\s+)?(?:"([^"]+)"|\'([^\']+)\'|([^"\']\S*))$', prefix, flags=re.IGNORECASE)
    if not match:
        return project_dir
    path_token = next((item for item in match.groups() if item), "")
    if not path_token:
        return project_dir
    candidate = Path(path_token)
    if candidate.is_absolute():
        return candidate
    return (project_dir / candidate).resolve()


def _extract_python_module(command: str) -> str:
    effective_command = _effective_shell_command(command)
    parts = shlex.split(effective_command, posix=os.name != "nt")
    if len(parts) < 3:
        return ""
    if str(parts[1]).strip() != "-m":
        return ""
    return str(parts[2]).strip()


def _preflight_warnings_for_command(
    *,
    project_dir: Path,
    command: str,
    timeout_seconds: int,
    import_probe_cache: dict[tuple[str, str], bool],
) -> list[str]:
    warnings: list[str] = []
    binary = _command_binary(command).strip().lower()
    if not binary:
        return warnings
    workdir = _command_workdir(project_dir, command)
    if binary in {"npm", "pnpm", "yarn"}:
        if not (workdir / "package.json").exists():
            warnings.append(f"node workspace missing package.json: {workdir}")
    module = _extract_python_module(command).strip().lower()
    if module in {"pytest", "ruff", "mypy"}:
        python_bin = shutil.which("python")
        if python_bin:
            cache_key = (str(workdir), module)
            cached_ok = import_probe_cache.get(cache_key)
            if cached_ok is None:
                try:
                    proc = subprocess.run(
                        [python_bin, "-c", f"import {module}"],
                        cwd=str(workdir),
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.PIPE,
                        text=True,
                        encoding="utf-8",
                        errors="replace",
                        timeout=max(1, int(timeout_seconds)),
                        check=False,
                    )
                    import_probe_cache[cache_key] = int(proc.returncode) == 0
                except (OSError, subprocess.SubprocessError):
                    import_probe_cache[cache_key] = False
            if not import_probe_cache.get(cache_key, False):
                warnings.append(f"python module unavailable for verify preflight: {module}")
    return warnings


def _detect_missing_python_deps(results: list[dict[str, Any]]) -> list[str]:
    missing: set[str] = set()
    pattern = re.compile(r"ModuleNotFoundError:\s+No module named ['\"]([^'\"]+)['\"]")
    for row in results:
        stderr = str(row.get("stderr", ""))
        if not stderr:
            continue
        for match in pattern.findall(stderr):
            token = str(match).strip()
            if token:
                missing.add(token)
    return sorted(missing)


def _default_backfill_deadline(hours: int = 24) -> str:
    return (datetime.now(timezone.utc) + timedelta(hours=max(1, int(hours)))).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _append_line(path: Path, line: str) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def _normalize_positive_int(value: Any, default_value: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return max(1, int(default_value))
    return max(1, parsed)


def _normalize_bool(value: Any, default_value: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default_value
    text = str(value).strip().lower()
    return text in {"1", "true", "yes", "on"}


def _resolve_verify_workers(runtime_cfg: dict[str, Any]) -> int:
    fallback = _normalize_positive_int(runtime_cfg.get("max_workers", 8), 8)
    return _normalize_positive_int(runtime_cfg.get("verify_workers", fallback), fallback)


def _run_with_timeout_detection(
    command: str, 
    project_dir: Path, 
    timeout_seconds: int,
    log_path: Path,
    jsonl_path: Path,
    output_callback: Callable[[str, str], None] | None = None,
) -> dict[str, Any]:
    """Run command with enhanced timeout detection and logging."""
    start_time = time.perf_counter()
    try:
        if output_callback is None:
            result = run_command(command, project_dir, timeout_seconds)
        else:
            try:
                result = run_command(
                    command,
                    project_dir,
                    timeout_seconds,
                    output_callback=output_callback,
                )
            except TypeError:
                # Backward-compatible path for monkeypatched runners that do not accept output_callback.
                result = run_command(command, project_dir, timeout_seconds)
        elapsed = time.perf_counter() - start_time
        _append_line(log_path, f"COMMAND_COMPLETE {command} DURATION={elapsed:.3f}s")
        return result
    except Exception as exc:
        elapsed = time.perf_counter() - start_time
        _append_line(log_path, f"COMMAND_ERROR {command} DURATION={elapsed:.3f}s ERROR={exc!r}")
        _append_jsonl(jsonl_path, {
            "event": "command_error",
            "command": command,
            "duration_seconds": elapsed,
            "error": str(exc),
            "ts": _utc_now(),
        })
        # Return a failure result instead of raising
        return {
            "command": command,
            "exit_code": 1,
            "duration_seconds": elapsed,
            "stdout": "",
            "stderr": f"agent-accel error: {exc}",
            "timed_out": False,
        }


def _normalize_changed_path(project_dir: Path, changed_file: str) -> tuple[str, Path]:
    raw = changed_file.replace("\\", "/").strip()
    candidate = Path(raw)
    abs_path = candidate if candidate.is_absolute() else (project_dir / candidate)
    abs_resolved = abs_path.resolve()
    project_resolved = project_dir.resolve()
    try:
        rel = abs_resolved.relative_to(project_resolved).as_posix()
        return rel, abs_resolved
    except ValueError:
        return raw, abs_resolved


def _build_changed_files_fingerprint(
    project_dir: Path,
    changed_files: list[str] | None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for changed_file in (changed_files or []):
        key, abs_path = _normalize_changed_path(project_dir, str(changed_file))
        row: dict[str, Any] = {"path": key}
        if abs_path.exists():
            stat = abs_path.stat()
            row["exists"] = True
            row["size"] = int(stat.st_size)
            row["mtime_ns"] = int(stat.st_mtime_ns)
            row["is_dir"] = bool(abs_path.is_dir())
        else:
            row["exists"] = False
        rows.append(row)
    rows.sort(key=lambda item: str(item.get("path", "")))
    return rows


def _cache_file_path(paths: dict[str, Path]) -> Path:
    return paths["verify"] / "command_cache.json"


def _cache_key(
    command: str,
    project_dir: Path,
    changed_fingerprint: list[dict[str, Any]],
) -> str:
    payload = {
        "version": 1,
        "command": command,
        "project_dir": str(project_dir.resolve()),
        "changed_files": changed_fingerprint,
    }
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _parse_utc(value: Any) -> datetime | None:
    if not isinstance(value, str):
        return None
    text = value.strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(text)
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _load_cache_entries(cache_path: Path) -> dict[str, dict[str, Any]]:
    if not cache_path.exists():
        return {}
    try:
        payload = json.loads(cache_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    entries_raw = payload.get("entries", {})
    if not isinstance(entries_raw, dict):
        return {}
    entries: dict[str, dict[str, Any]] = {}
    for key, value in entries_raw.items():
        if isinstance(key, str) and isinstance(value, dict):
            entries[key] = value
    return entries


def _prune_cache_entries(
    entries: dict[str, dict[str, Any]],
    ttl_seconds: int,
    max_entries: int,
) -> tuple[dict[str, dict[str, Any]], bool]:
    now = datetime.now(timezone.utc)
    max_count = max(1, max_entries)

    valid: list[tuple[str, datetime, dict[str, Any]]] = []
    for key, entry in entries.items():
        saved_at = _parse_utc(entry.get("saved_utc"))
        if saved_at is None:
            continue
        entry_ttl_seconds = _normalize_positive_int(entry.get("ttl_seconds", ttl_seconds), ttl_seconds)
        entry_ttl = timedelta(seconds=max(1, entry_ttl_seconds))
        if now - saved_at > entry_ttl:
            continue
        valid.append((key, saved_at, entry))

    valid.sort(key=lambda item: item[1], reverse=True)
    trimmed = valid[:max_count]
    pruned = {key: entry for key, _, entry in trimmed}
    was_pruned = len(pruned) != len(entries)
    return pruned, was_pruned


def _write_cache_entries_atomic(cache_path: Path, entries: dict[str, dict[str, Any]]) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": 1,
        "updated_utc": _utc_now(),
        "entries": entries,
    }
    tmp_file = tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        delete=False,
        dir=str(cache_path.parent),
        prefix=f".{cache_path.name}.",
        suffix=".tmp",
        newline="\n",
    )
    tmp_path = Path(tmp_file.name)
    try:
        with tmp_file:
            tmp_file.write(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
            tmp_file.flush()
            os.fsync(tmp_file.fileno())
        os.replace(tmp_path, cache_path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)


def _is_failure(result: dict[str, Any]) -> bool:
    return int(result.get("exit_code", 1)) != 0


def _normalize_live_result(result: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(result)
    normalized["command"] = str(result.get("command", ""))
    normalized["exit_code"] = int(result.get("exit_code", 1))
    normalized["duration_seconds"] = float(result.get("duration_seconds", 0.0))
    normalized["stdout"] = str(result.get("stdout", ""))
    normalized["stderr"] = str(result.get("stderr", ""))
    normalized["timed_out"] = bool(result.get("timed_out", False))
    normalized["cached"] = False
    return normalized


def _normalize_cached_result(command: str, entry: dict[str, Any]) -> dict[str, Any]:
    stored = entry.get("result", {})
    if not isinstance(stored, dict):
        stored = {}
    cache_kind = str(entry.get("cache_kind", "success") or "success")
    return {
        "command": command,
        "exit_code": int(stored.get("exit_code", 1)),
        "duration_seconds": float(stored.get("duration_seconds", 0.0)),
        "stdout": str(stored.get("stdout", "")),
        "stderr": str(stored.get("stderr", "")),
        "timed_out": bool(stored.get("timed_out", False)),
        "cached": True,
        "cache_kind": cache_kind,
    }


def _cache_entry_is_failure(entry: dict[str, Any]) -> bool:
    cache_kind = str(entry.get("cache_kind", "success") or "success").strip().lower()
    if cache_kind == "failure":
        return True
    stored = entry.get("result", {})
    if not isinstance(stored, dict):
        return False
    if bool(stored.get("timed_out", False)):
        return True
    return int(stored.get("exit_code", 0)) != 0


def _can_use_cached_entry(entry: dict[str, Any], *, allow_failed: bool) -> bool:
    if allow_failed:
        return True
    return not _cache_entry_is_failure(entry)


def _safe_callback_call(callback: VerifyProgressCallback, method_name: str, *args: Any, **kwargs: Any) -> None:
    method = getattr(callback, method_name, None)
    if method is None:
        return
    try:
        method(*args, **kwargs)
    except TypeError:
        # Backward compatibility for callbacks that do not accept newer keyword args.
        method(*args)


def _tail_output_text(value: Any, limit: int = 600) -> str:
    text = str(value or "")
    if len(text) <= limit:
        return text
    return text[-limit:]


def _emit_command_complete_event(
    callback: VerifyProgressCallback,
    job_id: str,
    command: str,
    result: dict[str, Any],
    *,
    completed: int,
    total: int,
) -> None:
    _safe_callback_call(
        callback,
        "on_command_complete",
        job_id,
        command,
        int(result.get("exit_code", 1)),
        float(result.get("duration_seconds", 0.0)),
        completed=completed,
        total=total,
        stdout_tail=_tail_output_text(result.get("stdout", "")),
        stderr_tail=_tail_output_text(result.get("stderr", "")),
    )


def _start_command_tick_thread(
    callback: VerifyProgressCallback,
    *,
    job_id: str,
    command: str,
    timeout_seconds: int,
    stall_timeout_seconds: float | None = None,
    activity_probe: Callable[[], float] | None = None,
) -> tuple[threading.Event, float, threading.Thread]:
    stop_event = threading.Event()
    started = time.perf_counter()
    timeout_sec = max(1.0, float(timeout_seconds))
    stall_timeout = None
    if stall_timeout_seconds is not None:
        stall_timeout = max(1.0, float(stall_timeout_seconds))

    def _ticker() -> None:
        while not stop_event.wait(1.0):
            elapsed = max(0.0, time.perf_counter() - started)
            progress_pct = min(99.0, (elapsed / timeout_sec) * 100.0)
            eta_sec = max(0.0, timeout_sec - elapsed)
            last_activity = started
            if activity_probe is not None:
                try:
                    last_activity = float(activity_probe())
                except Exception:
                    last_activity = started
            stall_detected = False
            stall_elapsed = 0.0
            if stall_timeout is not None:
                idle_sec = max(0.0, time.perf_counter() - last_activity)
                stall_detected = idle_sec >= stall_timeout
                stall_elapsed = max(0.0, idle_sec - stall_timeout)
            _safe_callback_call(
                callback,
                "on_heartbeat",
                job_id,
                elapsed,
                eta_sec,
                "running",
                current_command=command,
                command_elapsed_sec=elapsed,
                command_timeout_sec=timeout_sec,
                command_progress_pct=progress_pct,
                stall_detected=stall_detected if stall_timeout is not None else None,
                stall_elapsed_sec=stall_elapsed if stall_timeout is not None else None,
            )

    thread = threading.Thread(target=_ticker, daemon=True)
    thread.start()
    return stop_event, started, thread


def _store_cache_result(
    entries: dict[str, dict[str, Any]],
    key: str,
    command: str,
    result: dict[str, Any],
    *,
    cache_failed_results: bool,
    failed_ttl_seconds: int,
) -> bool:
    is_failure = _is_failure(result) or bool(result.get("timed_out", False))
    if is_failure and not cache_failed_results:
        return False
    cache_kind = "failure" if is_failure else "success"
    ttl_seconds = _normalize_positive_int(failed_ttl_seconds, 120) if is_failure else None
    entries[key] = {
        "saved_utc": _utc_now(),
        "command": command,
        "cache_kind": cache_kind,
        "ttl_seconds": ttl_seconds,
        "result": {
            "exit_code": int(result.get("exit_code", 0)),
            "duration_seconds": float(result.get("duration_seconds", 0.0)),
            "stdout": str(result.get("stdout", "")),
            "stderr": str(result.get("stderr", "")),
            "timed_out": bool(result.get("timed_out", False)),
        },
    }
    return True


def _log_command_result(
    log_path: Path,
    jsonl_path: Path,
    result: dict[str, Any],
    *,
    mode: str,
    fail_fast: bool,
    cache_hits: int,
    cache_misses: int,
    command_index: int | None = None,
    total_commands: int | None = None,
    fail_fast_skipped: bool = False,
) -> None:
    _append_line(log_path, f"CMD {result['command']}")
    _append_line(log_path, f"EXIT {result['exit_code']} DURATION {result['duration_seconds']}s")
    if result["stdout"]:
        _append_line(log_path, f"STDOUT {str(result['stdout']).strip()[:2000]}")
    if result["stderr"]:
        _append_line(log_path, f"STDERR {str(result['stderr']).strip()[:2000]}")
    _append_jsonl(
        jsonl_path,
        {
            "event": "command_result",
            "ts": _utc_now(),
            "command": result["command"],
            "exit_code": result["exit_code"],
            "duration_seconds": result["duration_seconds"],
            "timed_out": result["timed_out"],
            "cached": bool(result.get("cached", False)),
            "cache_kind": str(result.get("cache_kind", "success")),
            "mode": str(mode),
            "fail_fast": bool(fail_fast),
            "cache_hits": int(cache_hits),
            "cache_misses": int(cache_misses),
            "fail_fast_skipped": bool(fail_fast_skipped),
            "command_index": int(command_index) if command_index is not None else None,
            "total_commands": int(total_commands) if total_commands is not None else None,
        },
    )


def run_verify(
    project_dir: Path,
    config: dict[str, Any],
    changed_files: list[str] | None = None,
) -> dict[str, Any]:
    accel_home = Path(config["runtime"]["accel_home"]).resolve()
    paths = project_paths(accel_home, project_dir)
    ensure_project_dirs(paths)

    runtime_cfg = config.get("runtime", {})
    verify_workers = _resolve_verify_workers(runtime_cfg)
    per_command_timeout = int(runtime_cfg.get("per_command_timeout_seconds", 1200))
    verify_fail_fast = _normalize_bool(runtime_cfg.get("verify_fail_fast", False), False)
    verify_cache_enabled_cfg = _normalize_bool(runtime_cfg.get("verify_cache_enabled", True), True)
    verify_cache_failed_results = _normalize_bool(runtime_cfg.get("verify_cache_failed_results", False), False)
    verify_cache_ttl_seconds = _normalize_positive_int(runtime_cfg.get("verify_cache_ttl_seconds", 900), 900)
    verify_cache_max_entries = _normalize_positive_int(runtime_cfg.get("verify_cache_max_entries", 400), 400)
    verify_cache_failed_ttl_seconds = _normalize_positive_int(
        runtime_cfg.get("verify_cache_failed_ttl_seconds", 120),
        120,
    )
    verify_preflight_enabled = _normalize_bool(runtime_cfg.get("verify_preflight_enabled", True), True)
    verify_preflight_timeout_seconds = _normalize_positive_int(
        runtime_cfg.get("verify_preflight_timeout_seconds", 5),
        5,
    )
    try:
        stall_timeout_raw = float(runtime_cfg.get("verify_stall_timeout_seconds", 20.0))
    except (TypeError, ValueError):
        stall_timeout_raw = 20.0
    verify_stall_timeout_seconds: float | None
    if stall_timeout_raw <= 0:
        verify_stall_timeout_seconds = None
    else:
        verify_stall_timeout_seconds = min(float(per_command_timeout), max(1.0, stall_timeout_raw))
    workspace_routing_enabled = _normalize_bool(runtime_cfg.get("verify_workspace_routing_enabled", True), True)
    verify_mode = "fail_fast" if verify_fail_fast else ("parallel" if verify_workers > 1 else "sequential")

    nonce = uuid4().hex[:12]
    log_path = paths["verify"] / f"verify_{nonce}.log"
    jsonl_path = paths["verify"] / f"verify_{nonce}.jsonl"
    commands = select_verify_commands(config=config, changed_files=changed_files)

    _append_line(log_path, f"VERIFICATION_START {nonce} {_utc_now()}")
    _append_line(log_path, f"ENV cwd={project_dir} python={shutil.which('python') or ''}")
    _append_jsonl(
        jsonl_path,
        {
            "event": "verification_start",
            "nonce": nonce,
            "ts": _utc_now(),
            "mode": verify_mode,
            "fail_fast": bool(verify_fail_fast),
            "cache_failed_results": bool(verify_cache_failed_results),
            "cache_failed_ttl_seconds": int(verify_cache_failed_ttl_seconds),
            "preflight_enabled": bool(verify_preflight_enabled),
            "stall_timeout_seconds": float(verify_stall_timeout_seconds) if verify_stall_timeout_seconds is not None else None,
            "workspace_routing_enabled": bool(workspace_routing_enabled),
        },
    )

    changed_fingerprint = _build_changed_files_fingerprint(project_dir=project_dir, changed_files=changed_files)
    cache_enabled = bool(verify_cache_enabled_cfg and changed_fingerprint)
    cache_path = _cache_file_path(paths)
    cache_entries: dict[str, dict[str, Any]] = {}
    cache_dirty = False
    cache_hits = 0
    cache_misses = 0
    if cache_enabled:
        loaded_entries = _load_cache_entries(cache_path)
        cache_entries, was_pruned = _prune_cache_entries(
            loaded_entries,
            ttl_seconds=verify_cache_ttl_seconds,
            max_entries=verify_cache_max_entries,
        )
        cache_dirty = was_pruned
    elif verify_cache_enabled_cfg:
        _append_line(log_path, "CACHE_DISABLED no changed_files fingerprint")

    degraded = False
    degrade_reasons: list[str] = []
    import_probe_cache: dict[tuple[str, str], bool] = {}
    runnable_commands: list[str] = []
    for command in commands:
        if verify_preflight_enabled:
            warnings = _preflight_warnings_for_command(
                project_dir=project_dir,
                command=command,
                timeout_seconds=verify_preflight_timeout_seconds,
                import_probe_cache=import_probe_cache,
            )
            for warning in warnings:
                if warning not in degrade_reasons:
                    degrade_reasons.append(warning)
                degraded = True
                _append_line(log_path, f"PREFLIGHT_WARN {command} ({warning})")
                _append_jsonl(
                    jsonl_path,
                    {
                        "event": "command_preflight_warning",
                        "command": command,
                        "reason": warning,
                        "ts": _utc_now(),
                    },
                )
        binary = _command_binary(command)
        if binary and shutil.which(binary) is None:
            degraded = True
            reason = f"missing command binary: {binary}"
            degrade_reasons.append(reason)
            _append_line(log_path, f"SKIP {command} ({reason})")
            _append_jsonl(
                jsonl_path,
                {
                    "event": "command_skipped",
                    "command": command,
                    "reason": reason,
                    "ts": _utc_now(),
                    "mode": verify_mode,
                    "fail_fast": bool(verify_fail_fast),
                    "cache_hits": int(cache_hits),
                    "cache_misses": int(cache_misses),
                    "fail_fast_skipped": False,
                },
            )
            continue
        runnable_commands.append(command)

    results: list[dict[str, Any]] = []
    fail_fast_skipped_commands: list[str] = []

    if verify_fail_fast:
        for index, command in enumerate(runnable_commands):
            cache_key: str | None = None
            if cache_enabled:
                cache_key = _cache_key(
                    command=command,
                    project_dir=project_dir,
                    changed_fingerprint=changed_fingerprint,
                )
                cached_entry = cache_entries.get(cache_key)
                if cached_entry is not None and _can_use_cached_entry(
                    cached_entry,
                    allow_failed=verify_cache_failed_results,
                ):
                    cache_hits += 1
                    cached_result = _normalize_cached_result(command=command, entry=cached_entry)
                    results.append(cached_result)
                    _append_line(log_path, f"CACHE_HIT {command}")
                    _append_jsonl(
                        jsonl_path,
                        {"event": "command_cache_hit", "command": command, "ts": _utc_now()},
                    )
                    cache_failure = _is_failure(cached_result)
                    fail_fast_skip_flag = bool(cache_failure and (index < (len(runnable_commands) - 1)))
                    _log_command_result(
                        log_path,
                        jsonl_path,
                        cached_result,
                        mode=verify_mode,
                        fail_fast=verify_fail_fast,
                        cache_hits=cache_hits,
                        cache_misses=cache_misses,
                        command_index=index + 1,
                        total_commands=len(runnable_commands),
                        fail_fast_skipped=fail_fast_skip_flag,
                    )
                    if cache_failure:
                        fail_fast_skipped_commands = runnable_commands[index + 1 :]
                        break
                    continue
                cache_misses += 1

            live_result = _normalize_live_result(run_command(command, project_dir, per_command_timeout))
            results.append(live_result)
            fail_fast_skip_flag = bool(_is_failure(live_result) and (index < (len(runnable_commands) - 1)))
            _log_command_result(
                log_path,
                jsonl_path,
                live_result,
                mode=verify_mode,
                fail_fast=verify_fail_fast,
                cache_hits=cache_hits,
                cache_misses=cache_misses,
                command_index=index + 1,
                total_commands=len(runnable_commands),
                fail_fast_skipped=fail_fast_skip_flag,
            )
            if cache_enabled and cache_key:
                if _store_cache_result(
                    cache_entries,
                    cache_key,
                    command,
                    live_result,
                    cache_failed_results=verify_cache_failed_results,
                    failed_ttl_seconds=verify_cache_failed_ttl_seconds,
                ):
                    cache_dirty = True
            if _is_failure(live_result):
                fail_fast_skipped_commands = runnable_commands[index + 1 :]
                break
    else:
        runnable_with_keys: list[tuple[str, str | None, int]] = []
        for index, command in enumerate(runnable_commands):
            cache_key = None
            if cache_enabled:
                cache_key = _cache_key(
                    command=command,
                    project_dir=project_dir,
                    changed_fingerprint=changed_fingerprint,
                )
                cached_entry = cache_entries.get(cache_key)
                if cached_entry is not None and _can_use_cached_entry(
                    cached_entry,
                    allow_failed=verify_cache_failed_results,
                ):
                    cache_hits += 1
                    cached_result = _normalize_cached_result(command=command, entry=cached_entry)
                    results.append(cached_result)
                    _append_line(log_path, f"CACHE_HIT {command}")
                    _append_jsonl(
                        jsonl_path,
                        {"event": "command_cache_hit", "command": command, "ts": _utc_now()},
                    )
                    _log_command_result(
                        log_path,
                        jsonl_path,
                        cached_result,
                        mode=verify_mode,
                        fail_fast=verify_fail_fast,
                        cache_hits=cache_hits,
                        cache_misses=cache_misses,
                        command_index=index + 1,
                        total_commands=len(runnable_commands),
                        fail_fast_skipped=False,
                    )
                    continue
                cache_misses += 1
            runnable_with_keys.append((command, cache_key, index))

        if runnable_with_keys:
            # Add overall timeout detection for the entire parallel execution
            overall_start = time.perf_counter()
            max_overall_timeout = per_command_timeout * len(runnable_with_keys) * 2  # generous buffer
            
            try:
                with ThreadPoolExecutor(max_workers=min(verify_workers, len(runnable_with_keys))) as pool:
                    future_map = {
                        pool.submit(_run_with_timeout_detection, command, project_dir, per_command_timeout, log_path, jsonl_path): (command, cache_key, index)
                        for command, cache_key, index in runnable_with_keys
                    }
                    
                    for future in as_completed(future_map, timeout=max_overall_timeout):
                        command, cache_key, command_index = future_map[future]
                        try:
                            live_result = _normalize_live_result(future.result(timeout=per_command_timeout))
                            results.append(live_result)
                            _log_command_result(
                                log_path,
                                jsonl_path,
                                live_result,
                                mode=verify_mode,
                                fail_fast=verify_fail_fast,
                                cache_hits=cache_hits,
                                cache_misses=cache_misses,
                                command_index=command_index + 1,
                                total_commands=len(runnable_commands),
                                fail_fast_skipped=False,
                            )
                            if cache_enabled and cache_key:
                                if _store_cache_result(
                                    cache_entries,
                                    cache_key,
                                    command,
                                    live_result,
                                    cache_failed_results=verify_cache_failed_results,
                                    failed_ttl_seconds=verify_cache_failed_ttl_seconds,
                                ):
                                    cache_dirty = True
                        except TimeoutError:
                            # Handle individual future timeout
                            elapsed = time.perf_counter() - overall_start
                            timeout_result = {
                                "command": command,
                                "exit_code": 124,
                                "duration_seconds": elapsed,
                                "stdout": "",
                                "stderr": f"agent-accel: ThreadPool future timeout after {per_command_timeout}s",
                                "timed_out": True,
                            }
                            results.append(timeout_result)
                            _log_command_result(
                                log_path,
                                jsonl_path,
                                timeout_result,
                                mode=verify_mode,
                                fail_fast=verify_fail_fast,
                                cache_hits=cache_hits,
                                cache_misses=cache_misses,
                                command_index=command_index + 1,
                                total_commands=len(runnable_commands),
                                fail_fast_skipped=False,
                            )
                            _append_line(log_path, f"FUTURE_TIMEOUT {command}")
                        except Exception as exc:
                            # Handle other future errors
                            elapsed = time.perf_counter() - overall_start
                            error_result = {
                                "command": command,
                                "exit_code": 1,
                                "duration_seconds": elapsed,
                                "stdout": "",
                                "stderr": f"agent-accel: ThreadPool future error: {exc}",
                                "timed_out": False,
                            }
                            results.append(error_result)
                            _log_command_result(
                                log_path,
                                jsonl_path,
                                error_result,
                                mode=verify_mode,
                                fail_fast=verify_fail_fast,
                                cache_hits=cache_hits,
                                cache_misses=cache_misses,
                                command_index=command_index + 1,
                                total_commands=len(runnable_commands),
                                fail_fast_skipped=False,
                            )
                            _append_line(log_path, f"FUTURE_ERROR {command} ERROR={exc!r}")
                            
            except TimeoutError:
                # Handle overall ThreadPool timeout
                elapsed = time.perf_counter() - overall_start
                _append_line(log_path, f"THREADPOOL_TIMEOUT overall_timeout={max_overall_timeout}s elapsed={elapsed:.3f}s")
                # Add failure results for any remaining commands
                for command, cache_key, command_index in runnable_with_keys:
                    if not any(r["command"] == command for r in results):
                        timeout_result = {
                            "command": command,
                            "exit_code": 124,
                            "duration_seconds": elapsed,
                            "stdout": "",
                            "stderr": f"agent-accel: ThreadPool overall timeout after {max_overall_timeout}s",
                            "timed_out": True,
                        }
                        results.append(timeout_result)
                        _log_command_result(
                            log_path,
                            jsonl_path,
                            timeout_result,
                            mode=verify_mode,
                            fail_fast=verify_fail_fast,
                            cache_hits=cache_hits,
                            cache_misses=cache_misses,
                            command_index=command_index + 1,
                            total_commands=len(runnable_commands),
                            fail_fast_skipped=False,
                        )
            except Exception as exc:
                # Handle ThreadPool creation/execution errors
                elapsed = time.perf_counter() - overall_start
                _append_line(log_path, f"THREADPOOL_ERROR ERROR={exc!r}")
                # Fallback to sequential execution
                _append_line(log_path, "FALLBACK_SEQUENTIAL_EXECUTION")
                for command, cache_key, command_index in runnable_with_keys:
                    if not any(r["command"] == command for r in results):
                        live_result = _normalize_live_result(_run_with_timeout_detection(command, project_dir, per_command_timeout, log_path, jsonl_path))
                        results.append(live_result)
                        _log_command_result(
                            log_path,
                            jsonl_path,
                            live_result,
                            mode=verify_mode,
                            fail_fast=verify_fail_fast,
                            cache_hits=cache_hits,
                            cache_misses=cache_misses,
                            command_index=command_index + 1,
                            total_commands=len(runnable_commands),
                            fail_fast_skipped=False,
                        )
                        if cache_enabled and cache_key:
                            if _store_cache_result(
                                cache_entries,
                                cache_key,
                                command,
                                live_result,
                                cache_failed_results=verify_cache_failed_results,
                                failed_ttl_seconds=verify_cache_failed_ttl_seconds,
                            ):
                                cache_dirty = True

    if fail_fast_skipped_commands:
        for command in fail_fast_skipped_commands:
            _append_line(log_path, f"SKIP {command} (fail-fast)")
            _append_jsonl(
                jsonl_path,
                {
                    "event": "command_skipped",
                    "command": command,
                    "reason": "fail_fast",
                    "ts": _utc_now(),
                    "mode": verify_mode,
                    "fail_fast": bool(verify_fail_fast),
                    "cache_hits": int(cache_hits),
                    "cache_misses": int(cache_misses),
                    "fail_fast_skipped": True,
                },
            )

    if cache_enabled and cache_dirty:
        cache_entries, _ = _prune_cache_entries(
            cache_entries,
            ttl_seconds=verify_cache_ttl_seconds,
            max_entries=verify_cache_max_entries,
        )
        _write_cache_entries_atomic(cache_path, cache_entries)
        _append_line(log_path, f"CACHE_WRITE entries={len(cache_entries)}")

    results.sort(key=lambda row: row["command"])
    missing_python_deps = _detect_missing_python_deps(results)
    if missing_python_deps:
        degraded = True
        reason_text = "missing python dependencies: " + ", ".join(missing_python_deps)
        if reason_text not in degrade_reasons:
            degrade_reasons.append(reason_text)
    has_failure = any(_is_failure(item) for item in results)
    if has_failure:
        exit_code = 3
        status = "failed"
    elif degraded:
        exit_code = 2
        status = "degraded"
    else:
        exit_code = 0
        status = "success"

    if degraded:
        reason_text = "; ".join(degrade_reasons) if degrade_reasons else "degraded execution"
        _append_line(log_path, f"DEGRADE_REASON: {reason_text}")
        _append_line(log_path, "RISK: some checks were skipped because required tools are unavailable")
        deadline_utc = _default_backfill_deadline(24)
        _append_line(
            log_path,
            f"BACKFILL_PLAN: owner=user commands=\"install missing tools/dependencies and rerun accel verify\" deadline_utc={deadline_utc}",
        )

    _append_line(log_path, f"VERIFICATION_END {nonce} {_utc_now()}")
    _append_jsonl(
        jsonl_path,
        {
            "event": "verification_end",
            "nonce": nonce,
            "status": status,
            "exit_code": exit_code,
            "ts": _utc_now(),
            "mode": verify_mode,
            "fail_fast": bool(verify_fail_fast),
            "cache_hits": int(cache_hits),
            "cache_misses": int(cache_misses),
            "fail_fast_skipped": int(len(fail_fast_skipped_commands)),
            "cache_failed_results": bool(verify_cache_failed_results),
            "cache_failed_ttl_seconds": int(verify_cache_failed_ttl_seconds),
        },
    )

    return {
        "status": status,
        "exit_code": exit_code,
        "nonce": nonce,
        "log_path": str(log_path),
        "jsonl_path": str(jsonl_path),
        "commands": commands,
        "results": results,
        "degraded": degraded,
        "fail_fast": verify_fail_fast,
        "fail_fast_skipped_commands": fail_fast_skipped_commands,
        "cache_enabled": cache_enabled,
        "cache_hits": cache_hits,
        "cache_misses": cache_misses,
        "mode": verify_mode,
        "cache_failed_results": verify_cache_failed_results,
        "cache_failed_ttl_seconds": verify_cache_failed_ttl_seconds,
    }


def run_verify_with_callback(
    project_dir: Path,
    config: dict[str, Any],
    changed_files: list[str] | None = None,
    callback: VerifyProgressCallback | None = None,
) -> dict[str, Any]:
    if callback is None:
        callback = NoOpCallback()

    accel_home = Path(config["runtime"]["accel_home"]).resolve()
    paths = project_paths(accel_home, project_dir)
    ensure_project_dirs(paths)

    runtime_cfg = config.get("runtime", {})
    verify_workers = _resolve_verify_workers(runtime_cfg)
    per_command_timeout = int(runtime_cfg.get("per_command_timeout_seconds", 1200))
    verify_fail_fast = _normalize_bool(runtime_cfg.get("verify_fail_fast", False), False)
    verify_cache_enabled_cfg = _normalize_bool(runtime_cfg.get("verify_cache_enabled", True), True)
    verify_cache_failed_results = _normalize_bool(runtime_cfg.get("verify_cache_failed_results", False), False)
    verify_cache_ttl_seconds = _normalize_positive_int(runtime_cfg.get("verify_cache_ttl_seconds", 900), 900)
    verify_cache_max_entries = _normalize_positive_int(runtime_cfg.get("verify_cache_max_entries", 400), 400)
    verify_cache_failed_ttl_seconds = _normalize_positive_int(
        runtime_cfg.get("verify_cache_failed_ttl_seconds", 120),
        120,
    )
    verify_preflight_enabled = _normalize_bool(runtime_cfg.get("verify_preflight_enabled", True), True)
    verify_preflight_timeout_seconds = _normalize_positive_int(
        runtime_cfg.get("verify_preflight_timeout_seconds", 5),
        5,
    )
    try:
        stall_timeout_raw = float(runtime_cfg.get("verify_stall_timeout_seconds", 20.0))
    except (TypeError, ValueError):
        stall_timeout_raw = 20.0
    verify_stall_timeout_seconds: float | None
    if stall_timeout_raw <= 0:
        verify_stall_timeout_seconds = None
    else:
        verify_stall_timeout_seconds = min(float(per_command_timeout), max(1.0, stall_timeout_raw))
    workspace_routing_enabled = _normalize_bool(runtime_cfg.get("verify_workspace_routing_enabled", True), True)
    verify_mode = "fail_fast" if verify_fail_fast else ("parallel" if verify_workers > 1 else "sequential")

    nonce = uuid4().hex[:12]
    verify_job_id = "verify_" + nonce
    log_path = paths["verify"] / f"verify_{nonce}.log"
    jsonl_path = paths["verify"] / f"verify_{nonce}.jsonl"
    commands = select_verify_commands(config=config, changed_files=changed_files)

    callback.on_start(verify_job_id, len(commands))

    _append_line(log_path, f"VERIFICATION_START {nonce} {_utc_now()}")
    _append_line(log_path, f"ENV cwd={project_dir} python={shutil.which('python') or ''}")
    _append_jsonl(
        jsonl_path,
        {
            "event": "verification_start",
            "nonce": nonce,
            "ts": _utc_now(),
            "mode": verify_mode,
            "fail_fast": bool(verify_fail_fast),
            "cache_failed_results": bool(verify_cache_failed_results),
            "cache_failed_ttl_seconds": int(verify_cache_failed_ttl_seconds),
            "preflight_enabled": bool(verify_preflight_enabled),
            "stall_timeout_seconds": float(verify_stall_timeout_seconds) if verify_stall_timeout_seconds is not None else None,
            "workspace_routing_enabled": bool(workspace_routing_enabled),
        },
    )

    changed_fingerprint = _build_changed_files_fingerprint(project_dir=project_dir, changed_files=changed_files)
    cache_enabled = bool(verify_cache_enabled_cfg and changed_fingerprint)
    cache_path = _cache_file_path(paths)
    cache_entries: dict[str, dict[str, Any]] = {}
    cache_dirty = False
    cache_hits = 0
    cache_misses = 0

    callback.on_stage_change(verify_job_id, VerifyStage.LOAD_CACHE)

    if cache_enabled:
        loaded_entries = _load_cache_entries(cache_path)
        cache_entries, was_pruned = _prune_cache_entries(
            loaded_entries,
            ttl_seconds=verify_cache_ttl_seconds,
            max_entries=verify_cache_max_entries,
        )
        cache_dirty = was_pruned
    elif verify_cache_enabled_cfg:
        _append_line(log_path, "CACHE_DISABLED no changed_files fingerprint")

    callback.on_stage_change(verify_job_id, VerifyStage.SELECT_CMDS)

    degraded = False
    degrade_reasons: list[str] = []
    import_probe_cache: dict[tuple[str, str], bool] = {}
    runnable_commands: list[str] = []
    for command in commands:
        if verify_preflight_enabled:
            warnings = _preflight_warnings_for_command(
                project_dir=project_dir,
                command=command,
                timeout_seconds=verify_preflight_timeout_seconds,
                import_probe_cache=import_probe_cache,
            )
            for warning in warnings:
                if warning not in degrade_reasons:
                    degrade_reasons.append(warning)
                degraded = True
                _append_line(log_path, f"PREFLIGHT_WARN {command} ({warning})")
                _append_jsonl(
                    jsonl_path,
                    {
                        "event": "command_preflight_warning",
                        "command": command,
                        "reason": warning,
                        "ts": _utc_now(),
                    },
                )
        binary = _command_binary(command)
        if binary and shutil.which(binary) is None:
            degraded = True
            reason = f"missing command binary: {binary}"
            degrade_reasons.append(reason)
            _append_line(log_path, f"SKIP {command} ({reason})")
            _append_jsonl(
                jsonl_path,
                {
                    "event": "command_skipped",
                    "command": command,
                    "reason": reason,
                    "ts": _utc_now(),
                    "mode": verify_mode,
                    "fail_fast": bool(verify_fail_fast),
                    "cache_hits": int(cache_hits),
                    "cache_misses": int(cache_misses),
                    "fail_fast_skipped": False,
                },
            )
            callback.on_skip(verify_job_id, command, reason)
            continue
        runnable_commands.append(command)

    results: list[dict[str, Any]] = []
    fail_fast_skipped_commands: list[str] = []

    def _make_output_callback(command_name: str, activity_ref: dict[str, float]) -> Callable[[str, str], None]:
        def _emit(stream: str, chunk: str) -> None:
            activity_ref["ts"] = time.perf_counter()
            text = str(chunk or "")
            truncated = len(text) > 600
            if truncated:
                text = text[-600:]
            _safe_callback_call(
                callback,
                "on_command_output",
                verify_job_id,
                command_name,
                str(stream or "stdout"),
                text,
                truncated=truncated,
            )

        return _emit

    callback.on_stage_change(verify_job_id, VerifyStage.RUNNING)

    if verify_fail_fast:
        for index, command in enumerate(runnable_commands):
            callback.on_command_start(verify_job_id, command, index, len(runnable_commands))
            callback.on_progress(verify_job_id, index, len(runnable_commands), command)

            cache_key: str | None = None
            if cache_enabled:
                cache_key = _cache_key(
                    command=command,
                    project_dir=project_dir,
                    changed_fingerprint=changed_fingerprint,
                )
                cached_entry = cache_entries.get(cache_key)
                if cached_entry is not None and _can_use_cached_entry(
                    cached_entry,
                    allow_failed=verify_cache_failed_results,
                ):
                    cache_hits += 1
                    cached_result = _normalize_cached_result(command=command, entry=cached_entry)
                    results.append(cached_result)
                    _append_line(log_path, f"CACHE_HIT {command}")
                    _append_jsonl(
                        jsonl_path,
                        {"event": "command_cache_hit", "command": command, "ts": _utc_now()},
                    )
                    cache_failure = _is_failure(cached_result)
                    fail_fast_skip_flag = bool(cache_failure and (index < (len(runnable_commands) - 1)))
                    _log_command_result(
                        log_path,
                        jsonl_path,
                        cached_result,
                        mode=verify_mode,
                        fail_fast=verify_fail_fast,
                        cache_hits=cache_hits,
                        cache_misses=cache_misses,
                        command_index=index + 1,
                        total_commands=len(runnable_commands),
                        fail_fast_skipped=fail_fast_skip_flag,
                    )
                    callback.on_cache_hit(verify_job_id, command)
                    _emit_command_complete_event(
                        callback,
                        verify_job_id,
                        command,
                        cached_result,
                        completed=index + 1,
                        total=len(runnable_commands),
                    )
                    callback.on_progress(verify_job_id, index + 1, len(runnable_commands), "")
                    if cache_failure:
                        fail_fast_skipped_commands = runnable_commands[index + 1 :]
                        break
                    continue
                cache_misses += 1

            activity_ref = {"ts": time.perf_counter()}
            output_callback = _make_output_callback(command, activity_ref)
            def _activity_probe(ref: dict[str, float] = activity_ref) -> float:
                return float(ref["ts"])
            stop_tick, cmd_started, tick_thread = _start_command_tick_thread(
                callback,
                job_id=verify_job_id,
                command=command,
                timeout_seconds=per_command_timeout,
                stall_timeout_seconds=verify_stall_timeout_seconds,
                activity_probe=_activity_probe,
            )
            try:
                try:
                    live_result = _normalize_live_result(
                        run_command(
                            command,
                            project_dir,
                            per_command_timeout,
                            output_callback=output_callback,
                        )
                    )
                except TypeError:
                    live_result = _normalize_live_result(run_command(command, project_dir, per_command_timeout))
            finally:
                stop_tick.set()
                tick_thread.join(timeout=0.1)
            cmd_elapsed = max(0.0, time.perf_counter() - cmd_started)
            _safe_callback_call(
                callback,
                "on_heartbeat",
                verify_job_id,
                cmd_elapsed,
                max(0.0, float(per_command_timeout) - cmd_elapsed),
                "running",
                current_command=command,
                command_elapsed_sec=cmd_elapsed,
                command_timeout_sec=float(per_command_timeout),
                command_progress_pct=100.0,
                stall_detected=False if verify_stall_timeout_seconds is not None else None,
                stall_elapsed_sec=0.0 if verify_stall_timeout_seconds is not None else None,
            )
            _emit_command_complete_event(
                callback,
                verify_job_id,
                command,
                live_result,
                completed=index + 1,
                total=len(runnable_commands),
            )
            results.append(live_result)
            fail_fast_skip_flag = bool(_is_failure(live_result) and (index < (len(runnable_commands) - 1)))
            _log_command_result(
                log_path,
                jsonl_path,
                live_result,
                mode=verify_mode,
                fail_fast=verify_fail_fast,
                cache_hits=cache_hits,
                cache_misses=cache_misses,
                command_index=index + 1,
                total_commands=len(runnable_commands),
                fail_fast_skipped=fail_fast_skip_flag,
            )
            if cache_enabled and cache_key:
                if _store_cache_result(
                    cache_entries,
                    cache_key,
                    command,
                    live_result,
                    cache_failed_results=verify_cache_failed_results,
                    failed_ttl_seconds=verify_cache_failed_ttl_seconds,
                ):
                    cache_dirty = True
            callback.on_progress(verify_job_id, index + 1, len(runnable_commands), "")
            if _is_failure(live_result):
                fail_fast_skipped_commands = runnable_commands[index + 1 :]
                break
    else:
        runnable_with_keys: list[tuple[str, str | None, int]] = []
        for index, command in enumerate(runnable_commands):
            cache_key = None
            if cache_enabled:
                cache_key = _cache_key(
                    command=command,
                    project_dir=project_dir,
                    changed_fingerprint=changed_fingerprint,
                )
                cached_entry = cache_entries.get(cache_key)
                if cached_entry is not None and _can_use_cached_entry(
                    cached_entry,
                    allow_failed=verify_cache_failed_results,
                ):
                    cache_hits += 1
                    cached_result = _normalize_cached_result(command=command, entry=cached_entry)
                    results.append(cached_result)
                    _append_line(log_path, f"CACHE_HIT {command}")
                    _append_jsonl(
                        jsonl_path,
                        {"event": "command_cache_hit", "command": command, "ts": _utc_now()},
                    )
                    _log_command_result(
                        log_path,
                        jsonl_path,
                        cached_result,
                        mode=verify_mode,
                        fail_fast=verify_fail_fast,
                        cache_hits=cache_hits,
                        cache_misses=cache_misses,
                        command_index=index + 1,
                        total_commands=len(runnable_commands),
                        fail_fast_skipped=False,
                    )
                    callback.on_cache_hit(verify_job_id, command)
                    continue
                cache_misses += 1
            runnable_with_keys.append((command, cache_key, index))

        if runnable_with_keys:
            overall_start = time.perf_counter()
            max_overall_timeout = per_command_timeout * len(runnable_with_keys) * 2

            callback.on_stage_change(verify_job_id, VerifyStage.PARALLEL)

            try:
                with ThreadPoolExecutor(max_workers=min(verify_workers, len(runnable_with_keys))) as pool:
                    future_map: dict[Any, tuple[str, str | None, float, int, threading.Event, threading.Thread]] = {}
                    total_commands = len(runnable_with_keys)
                    completed_count = 0
                    for index, (command, cache_key, command_index) in enumerate(runnable_with_keys):
                        callback.on_command_start(verify_job_id, command, index, total_commands)
                        callback.on_progress(verify_job_id, completed_count, total_commands, command)
                        activity_ref = {"ts": time.perf_counter()}
                        output_callback = _make_output_callback(command, activity_ref)
                        def _activity_probe(ref: dict[str, float] = activity_ref) -> float:
                            return float(ref["ts"])
                        stop_tick, cmd_started, tick_thread = _start_command_tick_thread(
                            callback,
                            job_id=verify_job_id,
                            command=command,
                            timeout_seconds=per_command_timeout,
                            stall_timeout_seconds=verify_stall_timeout_seconds,
                            activity_probe=_activity_probe,
                        )
                        future = pool.submit(
                            _run_with_timeout_detection,
                            command,
                            project_dir,
                            per_command_timeout,
                            log_path,
                            jsonl_path,
                            output_callback,
                        )
                        future_map[future] = (command, cache_key, cmd_started, command_index, stop_tick, tick_thread)

                    for future in as_completed(future_map, timeout=max_overall_timeout):
                        command, cache_key, started_at, command_index, stop_tick, tick_thread = future_map[future]

                        try:
                            live_result = _normalize_live_result(future.result(timeout=per_command_timeout))
                            cmd_elapsed = max(0.0, time.perf_counter() - started_at)
                            _safe_callback_call(
                                callback,
                                "on_heartbeat",
                                verify_job_id,
                                cmd_elapsed,
                                max(0.0, float(per_command_timeout) - cmd_elapsed),
                                "running",
                                current_command=command,
                                command_elapsed_sec=cmd_elapsed,
                                command_timeout_sec=float(per_command_timeout),
                                command_progress_pct=100.0,
                                stall_detected=False if verify_stall_timeout_seconds is not None else None,
                                stall_elapsed_sec=0.0 if verify_stall_timeout_seconds is not None else None,
                            )
                            _emit_command_complete_event(
                                callback,
                                verify_job_id,
                                command,
                                live_result,
                                completed=completed_count + 1,
                                total=total_commands,
                            )
                            results.append(live_result)
                            _log_command_result(
                                log_path,
                                jsonl_path,
                                live_result,
                                mode=verify_mode,
                                fail_fast=verify_fail_fast,
                                cache_hits=cache_hits,
                                cache_misses=cache_misses,
                                command_index=command_index + 1,
                                total_commands=len(runnable_commands),
                                fail_fast_skipped=False,
                            )
                            if cache_enabled and cache_key:
                                if _store_cache_result(
                                    cache_entries,
                                    cache_key,
                                    command,
                                    live_result,
                                    cache_failed_results=verify_cache_failed_results,
                                    failed_ttl_seconds=verify_cache_failed_ttl_seconds,
                                ):
                                    cache_dirty = True
                        except TimeoutError:
                            elapsed = time.perf_counter() - overall_start
                            timeout_result = {
                                "command": command,
                                "exit_code": 124,
                                "duration_seconds": elapsed,
                                "stdout": "",
                                "stderr": f"agent-accel: ThreadPool future timeout after {per_command_timeout}s",
                                "timed_out": True,
                            }
                            callback.on_error(verify_job_id, command, "timeout")
                            _safe_callback_call(
                                callback,
                                "on_heartbeat",
                                verify_job_id,
                                elapsed,
                                0.0,
                                "running",
                                current_command=command,
                                command_elapsed_sec=elapsed,
                                command_timeout_sec=float(per_command_timeout),
                                command_progress_pct=100.0,
                                stall_detected=False if verify_stall_timeout_seconds is not None else None,
                                stall_elapsed_sec=0.0 if verify_stall_timeout_seconds is not None else None,
                            )
                            _emit_command_complete_event(
                                callback,
                                verify_job_id,
                                command,
                                timeout_result,
                                completed=completed_count + 1,
                                total=total_commands,
                            )
                            results.append(timeout_result)
                            _log_command_result(
                                log_path,
                                jsonl_path,
                                timeout_result,
                                mode=verify_mode,
                                fail_fast=verify_fail_fast,
                                cache_hits=cache_hits,
                                cache_misses=cache_misses,
                                command_index=command_index + 1,
                                total_commands=len(runnable_commands),
                                fail_fast_skipped=False,
                            )
                            _append_line(log_path, f"FUTURE_TIMEOUT {command}")
                        except Exception as exc:
                            elapsed = time.perf_counter() - overall_start
                            error_result = {
                                "command": command,
                                "exit_code": 1,
                                "duration_seconds": elapsed,
                                "stdout": "",
                                "stderr": f"agent-accel: ThreadPool future error: {exc}",
                                "timed_out": False,
                            }
                            callback.on_error(verify_job_id, command, str(exc))
                            _safe_callback_call(
                                callback,
                                "on_heartbeat",
                                verify_job_id,
                                elapsed,
                                None,
                                "running",
                                current_command=command,
                                command_elapsed_sec=elapsed,
                                command_timeout_sec=float(per_command_timeout),
                                command_progress_pct=100.0,
                                stall_detected=False if verify_stall_timeout_seconds is not None else None,
                                stall_elapsed_sec=0.0 if verify_stall_timeout_seconds is not None else None,
                            )
                            _emit_command_complete_event(
                                callback,
                                verify_job_id,
                                command,
                                error_result,
                                completed=completed_count + 1,
                                total=total_commands,
                            )
                            results.append(error_result)
                            _log_command_result(
                                log_path,
                                jsonl_path,
                                error_result,
                                mode=verify_mode,
                                fail_fast=verify_fail_fast,
                                cache_hits=cache_hits,
                                cache_misses=cache_misses,
                                command_index=command_index + 1,
                                total_commands=len(runnable_commands),
                                fail_fast_skipped=False,
                            )
                            _append_line(log_path, f"FUTURE_ERROR {command} ERROR={exc!r}")
                        finally:
                            stop_tick.set()
                            tick_thread.join(timeout=0.1)

                        completed_count += 1
                        callback.on_progress(verify_job_id, completed_count, len(runnable_with_keys), "")

                        elapsed = time.perf_counter() - overall_start
                        eta = None
                        if completed_count > 0:
                            avg_time = elapsed / completed_count
                            remaining = len(runnable_with_keys) - completed_count
                            eta = avg_time * remaining
                        _safe_callback_call(
                            callback,
                            "on_heartbeat",
                            verify_job_id,
                            elapsed,
                            eta,
                            "running",
                            current_command="",
                            command_elapsed_sec=None,
                            command_timeout_sec=None,
                            command_progress_pct=None,
                        )

            except TimeoutError:
                for _future, (_command, _cache_key_value, _started_at, _index, stop_tick, tick_thread) in future_map.items():
                    stop_tick.set()
                    tick_thread.join(timeout=0.1)
                elapsed = time.perf_counter() - overall_start
                _append_line(log_path, f"THREADPOOL_TIMEOUT overall_timeout={max_overall_timeout}s elapsed={elapsed:.3f}s")
                for command, cache_key, command_index in runnable_with_keys:
                    if not any(r["command"] == command for r in results):
                        timeout_result = {
                            "command": command,
                            "exit_code": 124,
                            "duration_seconds": elapsed,
                            "stdout": "",
                            "stderr": f"agent-accel: ThreadPool overall timeout after {max_overall_timeout}s",
                            "timed_out": True,
                        }
                        results.append(timeout_result)
                        _log_command_result(
                            log_path,
                            jsonl_path,
                            timeout_result,
                            mode=verify_mode,
                            fail_fast=verify_fail_fast,
                            cache_hits=cache_hits,
                            cache_misses=cache_misses,
                            command_index=command_index + 1,
                            total_commands=len(runnable_commands),
                            fail_fast_skipped=False,
                        )
            except Exception as exc:
                for _future, (_command, _cache_key_value, _started_at, _index, stop_tick, tick_thread) in future_map.items():
                    stop_tick.set()
                    tick_thread.join(timeout=0.1)
                elapsed = time.perf_counter() - overall_start
                _append_line(log_path, f"THREADPOOL_ERROR ERROR={exc!r}")
                _append_line(log_path, "FALLBACK_SEQUENTIAL_EXECUTION")
                callback.on_stage_change(verify_job_id, VerifyStage.SEQUENTIAL)

                for i, (command, cache_key, command_index) in enumerate(runnable_with_keys):
                    if not any(r["command"] == command for r in results):
                        callback.on_command_start(verify_job_id, command, i, len(runnable_with_keys))
                        activity_ref = {"ts": time.perf_counter()}
                        output_callback = _make_output_callback(command, activity_ref)
                        def _activity_probe(ref: dict[str, float] = activity_ref) -> float:
                            return float(ref["ts"])
                        stop_tick, cmd_started, tick_thread = _start_command_tick_thread(
                            callback,
                            job_id=verify_job_id,
                            command=command,
                            timeout_seconds=per_command_timeout,
                            stall_timeout_seconds=verify_stall_timeout_seconds,
                            activity_probe=_activity_probe,
                        )
                        try:
                            live_result = _normalize_live_result(
                                _run_with_timeout_detection(
                                    command,
                                    project_dir,
                                    per_command_timeout,
                                    log_path,
                                    jsonl_path,
                                    output_callback,
                                )
                            )
                        finally:
                            stop_tick.set()
                            tick_thread.join(timeout=0.1)
                        cmd_elapsed = max(0.0, time.perf_counter() - cmd_started)
                        _safe_callback_call(
                            callback,
                            "on_heartbeat",
                            verify_job_id,
                            cmd_elapsed,
                            max(0.0, float(per_command_timeout) - cmd_elapsed),
                            "running",
                            current_command=command,
                            command_elapsed_sec=cmd_elapsed,
                            command_timeout_sec=float(per_command_timeout),
                            command_progress_pct=100.0,
                            stall_detected=False if verify_stall_timeout_seconds is not None else None,
                            stall_elapsed_sec=0.0 if verify_stall_timeout_seconds is not None else None,
                        )
                        _emit_command_complete_event(
                            callback,
                            verify_job_id,
                            command,
                            live_result,
                            completed=i + 1,
                            total=len(runnable_with_keys),
                        )
                        results.append(live_result)
                        _log_command_result(
                            log_path,
                            jsonl_path,
                            live_result,
                            mode=verify_mode,
                            fail_fast=verify_fail_fast,
                            cache_hits=cache_hits,
                            cache_misses=cache_misses,
                            command_index=command_index + 1,
                            total_commands=len(runnable_commands),
                            fail_fast_skipped=False,
                        )
                        if cache_enabled and cache_key:
                            if _store_cache_result(
                                cache_entries,
                                cache_key,
                                command,
                                live_result,
                                cache_failed_results=verify_cache_failed_results,
                                failed_ttl_seconds=verify_cache_failed_ttl_seconds,
                            ):
                                cache_dirty = True
                        callback.on_progress(verify_job_id, i + 1, len(runnable_with_keys), "")

    if fail_fast_skipped_commands:
        for command in fail_fast_skipped_commands:
            _append_line(log_path, f"SKIP {command} (fail-fast)")
            _append_jsonl(
                jsonl_path,
                {
                    "event": "command_skipped",
                    "command": command,
                    "reason": "fail_fast",
                    "ts": _utc_now(),
                    "mode": verify_mode,
                    "fail_fast": bool(verify_fail_fast),
                    "cache_hits": int(cache_hits),
                    "cache_misses": int(cache_misses),
                    "fail_fast_skipped": True,
                },
            )
            callback.on_skip(verify_job_id, command, "fail_fast")

    callback.on_stage_change(verify_job_id, VerifyStage.CLEANUP)

    if cache_enabled and cache_dirty:
        cache_entries, _ = _prune_cache_entries(
            cache_entries,
            ttl_seconds=verify_cache_ttl_seconds,
            max_entries=verify_cache_max_entries,
        )
        _write_cache_entries_atomic(cache_path, cache_entries)
        _append_line(log_path, f"CACHE_WRITE entries={len(cache_entries)}")

    results.sort(key=lambda row: row["command"])
    missing_python_deps = _detect_missing_python_deps(results)
    if missing_python_deps:
        degraded = True
        reason_text = "missing python dependencies: " + ", ".join(missing_python_deps)
        if reason_text not in degrade_reasons:
            degrade_reasons.append(reason_text)
    has_failure = any(_is_failure(item) for item in results)
    if has_failure:
        exit_code = 3
        status = "failed"
    elif degraded:
        exit_code = 2
        status = "degraded"
    else:
        exit_code = 0
        status = "success"

    if degraded:
        reason_text = "; ".join(degrade_reasons) if degrade_reasons else "degraded execution"
        _append_line(log_path, f"DEGRADE_REASON: {reason_text}")
        _append_line(log_path, "RISK: some checks were skipped because required tools are unavailable")
        deadline_utc = _default_backfill_deadline(24)
        _append_line(
            log_path,
            f"BACKFILL_PLAN: owner=user commands=\"install missing tools/dependencies and rerun accel verify\" deadline_utc={deadline_utc}",
        )

    _append_line(log_path, f"VERIFICATION_END {nonce} {_utc_now()}")
    _append_jsonl(
        jsonl_path,
        {
            "event": "verification_end",
            "nonce": nonce,
            "status": status,
            "exit_code": exit_code,
            "ts": _utc_now(),
            "mode": verify_mode,
            "fail_fast": bool(verify_fail_fast),
            "cache_hits": int(cache_hits),
            "cache_misses": int(cache_misses),
            "fail_fast_skipped": int(len(fail_fast_skipped_commands)),
            "cache_failed_results": bool(verify_cache_failed_results),
            "cache_failed_ttl_seconds": int(verify_cache_failed_ttl_seconds),
        },
    )

    callback.on_complete(verify_job_id, status, exit_code)

    return {
        "status": status,
        "exit_code": exit_code,
        "nonce": nonce,
        "log_path": str(log_path),
        "jsonl_path": str(jsonl_path),
        "commands": commands,
        "results": results,
        "degraded": degraded,
        "fail_fast": verify_fail_fast,
        "fail_fast_skipped_commands": fail_fast_skipped_commands,
        "cache_enabled": cache_enabled,
        "cache_hits": cache_hits,
        "cache_misses": cache_misses,
        "mode": verify_mode,
        "cache_failed_results": verify_cache_failed_results,
        "cache_failed_ttl_seconds": verify_cache_failed_ttl_seconds,
    }
