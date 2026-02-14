from __future__ import annotations

import json
import os
import re
import shlex
import shutil
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable
from uuid import uuid4

from ..language_profiles import (
    resolve_enabled_verify_groups,
    resolve_extension_verify_group_map,
)
from .runners import run_command
from .sharding import select_verify_commands
from .orchestrator_helpers import (
    _append_unfinished_entries,
    _build_changed_files_fingerprint,
    _cache_file_path,
    _cache_key,
    _can_use_cached_entry,
    _classify_verify_failures,
    _emit_command_complete_event,
    _is_failure,
    _load_cache_entries,
    _normalize_cached_result,
    _normalize_live_result,
    _prune_cache_entries,
    _remaining_wall_time_seconds,
    _safe_callback_call,
    _start_command_tick_thread,
    _timeboxed_command_timeout,
    _write_cache_entries_atomic,
)
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
    match = re.match(
        r'^cd\s+(?:/d\s+)?(?:"([^"]+)"|\'([^\']+)\'|([^"\']\S*))$',
        prefix,
        flags=re.IGNORECASE,
    )
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


def _parse_command_tokens(command: str) -> list[str]:
    try:
        return shlex.split(command, posix=os.name != "nt")
    except ValueError:
        return [part for part in str(command or "").strip().split(" ") if part]


def _extract_node_script(command: str) -> str:
    tokens = _parse_command_tokens(_effective_shell_command(command))
    if not tokens:
        return ""
    binary = str(tokens[0]).strip().lower()
    if binary not in {"npm", "pnpm", "yarn"}:
        return ""
    if len(tokens) <= 1:
        return ""
    if binary == "yarn":
        script = str(tokens[1]).strip().lower()
        return script if script and not script.startswith("-") else ""
    action = str(tokens[1]).strip().lower()
    if action in {"test", "lint", "typecheck", "build"}:
        return action
    if action in {"run", "run-script"} and len(tokens) >= 3:
        script = str(tokens[2]).strip().lower()
        return script if script else ""
    return ""


def _split_pytest_target(token: str) -> str:
    value = str(token or "").strip().strip('"').strip("'")
    if "::" not in value:
        return value
    return str(value.split("::", 1)[0]).strip()


def _missing_or_root_only_pytest_targets(
    *,
    project_dir: Path,
    command: str,
) -> tuple[list[str], list[str]]:
    missing: list[str] = []
    root_only: list[str] = []
    workdir = _command_workdir(project_dir, command)
    for raw_target in _extract_pytest_targets(command):
        target = _split_pytest_target(raw_target)
        if not target:
            continue
        token_path = Path(target)
        if token_path.is_absolute():
            if not token_path.exists():
                missing.append(target)
            continue
        workdir_candidate = (workdir / token_path).resolve()
        if workdir_candidate.exists():
            continue
        project_candidate = (project_dir / token_path).resolve()
        if project_candidate.exists():
            root_only.append(target)
        else:
            missing.append(target)
    return missing, root_only


def _should_skip_for_preflight(warning: str) -> bool:
    token = str(warning or "").strip().lower()
    if not token:
        return False
    skip_prefixes = (
        "node workspace missing package.json:",
        "node workspace missing script:",
        "python module unavailable for verify preflight:",
        "pytest target missing:",
        "pytest target only exists at project root:",
    )
    return any(token.startswith(prefix) for prefix in skip_prefixes)


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
        package_json_path = workdir / "package.json"
        if not package_json_path.exists():
            warnings.append(f"node workspace missing package.json: {workdir}")
        else:
            script = _extract_node_script(command)
            if script:
                try:
                    payload = json.loads(package_json_path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    payload = {}
                scripts_payload = payload.get("scripts", {})
                scripts = (
                    {
                        str(key).strip().lower()
                        for key in scripts_payload.keys()
                        if str(key).strip()
                    }
                    if isinstance(scripts_payload, dict)
                    else set()
                )
                if script not in scripts:
                    warnings.append(
                        f"node workspace missing script: {script} ({workdir})"
                    )
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
                warnings.append(
                    f"python module unavailable for verify preflight: {module}"
                )
    if module == "pytest":
        missing_targets, root_only_targets = _missing_or_root_only_pytest_targets(
            project_dir=project_dir,
            command=command,
        )
        if missing_targets:
            warnings.append(f"pytest target missing: {missing_targets[0]}")
        if root_only_targets:
            warnings.append(
                f"pytest target only exists at project root: {root_only_targets[0]}"
            )
    return warnings


def _resolve_windows_compatible_command(
    project_dir: Path, command: str
) -> tuple[str, str]:
    text = str(command or "").strip()
    if os.name != "nt":
        return text, ""
    if not text:
        return text, ""
    binary = _command_binary(text).strip().lower()
    if binary != "make":
        return text, ""
    if shutil.which("make") is not None:
        return text, ""

    parts = shlex.split(text, posix=False)
    if len(parts) < 2:
        return text, ""
    target = str(parts[1]).strip().lower()
    if target != "install-pre-commit-hooks":
        return text, ""

    # Windows-compatible fallback for the common make target used in Python projects.
    fallback = "python -m pre_commit install"
    if not shutil.which("python"):
        fallback = "pre-commit install"
    return fallback, "windows_make_target_compat:install-pre-commit-hooks"


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
    return (
        (datetime.now(timezone.utc) + timedelta(hours=max(1, int(hours))))
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


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


def _normalize_optional_positive_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if parsed <= 0:
        return None
    return float(parsed)


def _resolve_verify_workers(runtime_cfg: dict[str, Any]) -> int:
    fallback = _normalize_positive_int(runtime_cfg.get("max_workers", 8), 8)
    return _normalize_positive_int(
        runtime_cfg.get("verify_workers", fallback), fallback
    )


def _extract_pytest_targets(command: str) -> list[str]:
    if "pytest" not in str(command).lower():
        return []
    effective = _effective_shell_command(command)
    parts = shlex.split(effective, posix=os.name != "nt")
    if not parts:
        return []
    targets: list[str] = []
    for token in parts[1:]:
        cleaned = str(token).strip().strip('"').strip("'")
        if not cleaned or cleaned.startswith("-"):
            continue
        if cleaned.endswith(".py"):
            targets.append(cleaned.replace("\\", "/"))
    return targets


def _build_verify_selection_evidence(
    *,
    config: dict[str, Any],
    changed_files: list[str] | None,
    commands: list[str],
) -> dict[str, Any]:
    verify_cfg = config.get("verify", {})
    verify_group_order = [
        str(group).strip().lower()
        for group, raw_commands in dict(verify_cfg).items()
        if isinstance(raw_commands, list)
    ]
    verify_group_commands: dict[str, list[str]] = {}
    for group in verify_group_order:
        raw_commands = verify_cfg.get(group, [])
        if isinstance(raw_commands, list):
            verify_group_commands[group] = [str(item) for item in raw_commands]

    enabled_groups = resolve_enabled_verify_groups(config)
    if not enabled_groups:
        enabled_groups = list(verify_group_order)
    extension_group_map = resolve_extension_verify_group_map(config)

    changed = [str(item).replace("\\", "/").lower() for item in (changed_files or [])]
    changed_groups: set[str] = set()
    for item in changed:
        suffix = Path(item).suffix
        if not suffix:
            continue
        group = str(extension_group_map.get(suffix, "")).strip().lower()
        if group:
            changed_groups.add(group)
    run_all = len(changed) == 0

    baseline_commands: list[str] = []
    for group in verify_group_order:
        if group not in enabled_groups:
            continue
        if not run_all and group not in changed_groups:
            continue
        baseline_commands.extend(verify_group_commands.get(group, []))

    accelerated_commands = [command for command in commands if command not in baseline_commands]
    targeted_tests: list[str] = []
    seen_targets: set[str] = set()
    for command in commands:
        for target in _extract_pytest_targets(command):
            if target in seen_targets:
                continue
            seen_targets.add(target)
            targeted_tests.append(target)

    return {
        "run_all": bool(run_all),
        "changed_files": list(changed_files or []),
        "enabled_verify_groups": list(enabled_groups),
        "changed_verify_groups": sorted(changed_groups),
        "layers": [
            {
                "layer": "safety_baseline",
                "reason": "run_all" if run_all else "changed_verify_groups_match",
                "commands": list(baseline_commands),
            },
            {
                "layer": "incremental_acceleration",
                "commands": list(accelerated_commands),
                "targeted_tests": list(targeted_tests),
            },
        ],
        "baseline_commands_count": int(len(baseline_commands)),
        "accelerated_commands_count": int(len(accelerated_commands)),
        "final_commands_count": int(len(commands)),
    }


def _invoke_run_command(
    command: str,
    project_dir: Path,
    timeout_seconds: int,
    *,
    output_callback: Callable[[str, str], None] | None = None,
    cancel_event: threading.Event | None = None,
    stall_timeout_seconds: float | None = None,
) -> dict[str, Any]:
    kwargs: dict[str, Any] = {}
    if output_callback is not None:
        kwargs["output_callback"] = output_callback
    if cancel_event is not None:
        kwargs["cancel_event"] = cancel_event
    if stall_timeout_seconds is not None:
        kwargs["stall_timeout_seconds"] = stall_timeout_seconds
    if kwargs:
        try:
            return run_command(command, project_dir, timeout_seconds, **kwargs)
        except TypeError:
            # Backward-compatible path for monkeypatched runners that do not accept newer kwargs.
            if output_callback is not None:
                try:
                    return run_command(
                        command,
                        project_dir,
                        timeout_seconds,
                        output_callback=output_callback,
                    )
                except TypeError:
                    pass
    return run_command(command, project_dir, timeout_seconds)


def _run_with_timeout_detection(
    command: str,
    project_dir: Path,
    timeout_seconds: int,
    log_path: Path,
    jsonl_path: Path,
    output_callback: Callable[[str, str], None] | None = None,
    cancel_event: threading.Event | None = None,
    stall_timeout_seconds: float | None = None,
) -> dict[str, Any]:
    """Run command with enhanced timeout detection and logging."""
    start_time = time.perf_counter()
    try:
        result = _invoke_run_command(
            command,
            project_dir,
            timeout_seconds,
            output_callback=output_callback,
            cancel_event=cancel_event,
            stall_timeout_seconds=stall_timeout_seconds,
        )
        elapsed = time.perf_counter() - start_time
        _append_line(log_path, f"COMMAND_COMPLETE {command} DURATION={elapsed:.3f}s")
        return result
    except Exception as exc:
        elapsed = time.perf_counter() - start_time
        _append_line(
            log_path, f"COMMAND_ERROR {command} DURATION={elapsed:.3f}s ERROR={exc!r}"
        )
        _append_jsonl(
            jsonl_path,
            {
                "event": "command_error",
                "command": command,
                "duration_seconds": elapsed,
                "error": str(exc),
                "ts": _utc_now(),
            },
        )
        # Return a failure result instead of raising
        return {
            "command": command,
            "exit_code": 1,
            "duration_seconds": elapsed,
            "stdout": "",
            "stderr": f"agent-accel error: {exc}",
            "timed_out": False,
            "cancelled": False,
            "stalled": False,
            "cancel_reason": "",
        }


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
    ttl_seconds = (
        _normalize_positive_int(failed_ttl_seconds, 120) if is_failure else None
    )
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
            "cancelled": bool(result.get("cancelled", False)),
            "stalled": bool(result.get("stalled", False)),
            "cancel_reason": str(result.get("cancel_reason", "")),
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
    _append_line(
        log_path, f"EXIT {result['exit_code']} DURATION {result['duration_seconds']}s"
    )
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
            "cancelled": bool(result.get("cancelled", False)),
            "stalled": bool(result.get("stalled", False)),
            "cancel_reason": str(result.get("cancel_reason", "")),
            "cached": bool(result.get("cached", False)),
            "cache_kind": str(result.get("cache_kind", "success")),
            "mode": str(mode),
            "fail_fast": bool(fail_fast),
            "cache_hits": int(cache_hits),
            "cache_misses": int(cache_misses),
            "fail_fast_skipped": bool(fail_fast_skipped),
            "command_index": int(command_index) if command_index is not None else None,
            "total_commands": int(total_commands)
            if total_commands is not None
            else None,
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
    verify_fail_fast = _normalize_bool(
        runtime_cfg.get("verify_fail_fast", False), False
    )
    verify_cache_enabled_cfg = _normalize_bool(
        runtime_cfg.get("verify_cache_enabled", True), True
    )
    verify_cache_failed_results = _normalize_bool(
        runtime_cfg.get("verify_cache_failed_results", False), False
    )
    verify_cache_ttl_seconds = _normalize_positive_int(
        runtime_cfg.get("verify_cache_ttl_seconds", 900), 900
    )
    verify_cache_max_entries = _normalize_positive_int(
        runtime_cfg.get("verify_cache_max_entries", 400), 400
    )
    verify_cache_failed_ttl_seconds = _normalize_positive_int(
        runtime_cfg.get("verify_cache_failed_ttl_seconds", 120),
        120,
    )
    verify_preflight_enabled = _normalize_bool(
        runtime_cfg.get("verify_preflight_enabled", True), True
    )
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
        verify_stall_timeout_seconds = min(
            float(per_command_timeout), max(1.0, stall_timeout_raw)
        )
    verify_max_wall_time_seconds = _normalize_optional_positive_float(
        runtime_cfg.get(
            "verify_max_wall_time_seconds",
            runtime_cfg.get("total_verify_timeout_seconds"),
        )
    )
    verify_auto_cancel_on_stall = _normalize_bool(
        runtime_cfg.get("verify_auto_cancel_on_stall", False), False
    )
    if verify_stall_timeout_seconds is None:
        verify_auto_cancel_on_stall = False
    workspace_routing_enabled = _normalize_bool(
        runtime_cfg.get("verify_workspace_routing_enabled", True), True
    )
    verify_mode = (
        "fail_fast"
        if verify_fail_fast
        else ("parallel" if verify_workers > 1 else "sequential")
    )

    nonce = uuid4().hex[:12]
    log_path = paths["verify"] / f"verify_{nonce}.log"
    jsonl_path = paths["verify"] / f"verify_{nonce}.jsonl"
    commands = select_verify_commands(config=config, changed_files=changed_files)
    selection_evidence = _build_verify_selection_evidence(
        config=config,
        changed_files=changed_files,
        commands=commands,
    )
    selected_commands_count = int(len(commands))

    _append_line(log_path, f"VERIFICATION_START {nonce} {_utc_now()}")
    _append_line(
        log_path, f"ENV cwd={project_dir} python={shutil.which('python') or ''}"
    )
    _append_line(
        log_path,
        "SELECTION_EVIDENCE "
        + json.dumps(selection_evidence, ensure_ascii=False),
    )
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
            "stall_timeout_seconds": float(verify_stall_timeout_seconds)
            if verify_stall_timeout_seconds is not None
            else None,
            "auto_cancel_on_stall": bool(verify_auto_cancel_on_stall),
            "max_wall_time_seconds": (
                float(verify_max_wall_time_seconds)
                if verify_max_wall_time_seconds is not None
                else None
            ),
            "workspace_routing_enabled": bool(workspace_routing_enabled),
            "selection_evidence": selection_evidence,
        },
    )

    changed_fingerprint = _build_changed_files_fingerprint(
        project_dir=project_dir, changed_files=changed_files
    )
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
    if selected_commands_count == 0:
        degraded = True
        degrade_reasons.append("no verify commands selected")
        _append_line(log_path, "NO_COMMANDS selected verify command list is empty")
        _append_jsonl(
            jsonl_path,
            {
                "event": "verify_no_commands",
                "reason": "no verify commands selected",
                "ts": _utc_now(),
            },
        )
    import_probe_cache: dict[tuple[str, str], bool] = {}
    runnable_commands: list[str] = []
    for command in commands:
        effective_command, compat_reason = _resolve_windows_compatible_command(
            project_dir, command
        )
        if compat_reason:
            _append_line(
                log_path,
                f"CMD_COMPAT {command} -> {effective_command} ({compat_reason})",
            )
            _append_jsonl(
                jsonl_path,
                {
                    "event": "command_compat_resolved",
                    "command": command,
                    "resolved_command": effective_command,
                    "reason": compat_reason,
                    "ts": _utc_now(),
                },
            )
        if verify_preflight_enabled:
            warnings = _preflight_warnings_for_command(
                project_dir=project_dir,
                command=effective_command,
                timeout_seconds=verify_preflight_timeout_seconds,
                import_probe_cache=import_probe_cache,
            )
            preflight_skip_reason = ""
            for warning in warnings:
                if warning not in degrade_reasons:
                    degrade_reasons.append(warning)
                degraded = True
                if (not preflight_skip_reason) and _should_skip_for_preflight(warning):
                    preflight_skip_reason = warning
                _append_line(
                    log_path, f"PREFLIGHT_WARN {effective_command} ({warning})"
                )
                _append_jsonl(
                    jsonl_path,
                    {
                        "event": "command_preflight_warning",
                        "command": effective_command,
                        "reason": warning,
                        "ts": _utc_now(),
                    },
                )
            if preflight_skip_reason:
                _append_line(
                    log_path,
                    f"SKIP {effective_command} ({preflight_skip_reason})",
                )
                _append_jsonl(
                    jsonl_path,
                    {
                        "event": "command_skipped",
                        "command": effective_command,
                        "reason": preflight_skip_reason,
                        "ts": _utc_now(),
                        "mode": verify_mode,
                        "fail_fast": bool(verify_fail_fast),
                        "cache_hits": int(cache_hits),
                        "cache_misses": int(cache_misses),
                        "fail_fast_skipped": False,
                    },
                )
                continue
        binary = _command_binary(effective_command)
        if binary and shutil.which(binary) is None:
            degraded = True
            reason = f"missing command binary: {binary}"
            degrade_reasons.append(reason)
            _append_line(log_path, f"SKIP {effective_command} ({reason})")
            _append_jsonl(
                jsonl_path,
                {
                    "event": "command_skipped",
                    "command": effective_command,
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
        runnable_commands.append(effective_command)
    runnable_commands_count = int(len(runnable_commands))

    results: list[dict[str, Any]] = []
    fail_fast_skipped_commands: list[str] = []
    unfinished_items: list[dict[str, Any]] = []
    termination_reason = ""
    verify_started_at = time.perf_counter()

    if verify_fail_fast:
        for index, command in enumerate(runnable_commands):
            remaining_wall_time = _remaining_wall_time_seconds(
                started_at=verify_started_at,
                max_wall_time_seconds=verify_max_wall_time_seconds,
            )
            command_timeout_budget = _timeboxed_command_timeout(
                per_command_timeout=per_command_timeout,
                remaining_wall_time=remaining_wall_time,
            )
            if command_timeout_budget <= 0:
                if not termination_reason:
                    termination_reason = "max_wall_time_exceeded"
                    _append_line(
                        log_path,
                        f"MAX_WALL_TIME_EXCEEDED elapsed={time.perf_counter() - verify_started_at:.3f}s "
                        f"limit={float(verify_max_wall_time_seconds or 0.0):.3f}s",
                    )
                _append_unfinished_entries(
                    unfinished_items=unfinished_items,
                    commands=runnable_commands[index:],
                    reason=termination_reason,
                )
                break
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
                    cached_result = _normalize_cached_result(
                        command=command, entry=cached_entry
                    )
                    results.append(cached_result)
                    _append_line(log_path, f"CACHE_HIT {command}")
                    _append_jsonl(
                        jsonl_path,
                        {
                            "event": "command_cache_hit",
                            "command": command,
                            "ts": _utc_now(),
                        },
                    )
                    cache_failure = _is_failure(cached_result)
                    fail_fast_skip_flag = bool(
                        cache_failure and (index < (len(runnable_commands) - 1))
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
                        fail_fast_skipped=fail_fast_skip_flag,
                    )
                    if cache_failure:
                        fail_fast_skipped_commands = runnable_commands[index + 1 :]
                        break
                    continue
                cache_misses += 1

            live_result = _normalize_live_result(
                _invoke_run_command(
                    command,
                    project_dir,
                    command_timeout_budget,
                    stall_timeout_seconds=(
                        verify_stall_timeout_seconds
                        if verify_auto_cancel_on_stall
                        else None
                    ),
                )
            )
            results.append(live_result)
            fail_fast_skip_flag = bool(
                _is_failure(live_result) and (index < (len(runnable_commands) - 1))
            )
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
            if bool(live_result.get("stalled", False)) and verify_auto_cancel_on_stall:
                if not termination_reason:
                    termination_reason = "stall_auto_cancel"
                    _append_line(
                        log_path, f"STALL_AUTO_CANCEL {command} idle_sec=unknown"
                    )
                _append_unfinished_entries(
                    unfinished_items=unfinished_items,
                    commands=runnable_commands[index + 1 :],
                    reason=termination_reason,
                )
                break
            if _is_failure(live_result):
                fail_fast_skipped_commands = runnable_commands[index + 1 :]
                break
    else:
        runnable_with_keys: list[tuple[str, str | None, int]] = []
        for index, command in enumerate(runnable_commands):
            remaining_wall_time = _remaining_wall_time_seconds(
                started_at=verify_started_at,
                max_wall_time_seconds=verify_max_wall_time_seconds,
            )
            command_timeout_budget = _timeboxed_command_timeout(
                per_command_timeout=per_command_timeout,
                remaining_wall_time=remaining_wall_time,
            )
            if command_timeout_budget <= 0:
                if not termination_reason:
                    termination_reason = "max_wall_time_exceeded"
                    _append_line(
                        log_path,
                        f"MAX_WALL_TIME_EXCEEDED elapsed={time.perf_counter() - verify_started_at:.3f}s "
                        f"limit={float(verify_max_wall_time_seconds or 0.0):.3f}s",
                    )
                _append_unfinished_entries(
                    unfinished_items=unfinished_items,
                    commands=runnable_commands[index:],
                    reason=termination_reason,
                )
                break
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
                    cached_result = _normalize_cached_result(
                        command=command, entry=cached_entry
                    )
                    results.append(cached_result)
                    _append_line(log_path, f"CACHE_HIT {command}")
                    _append_jsonl(
                        jsonl_path,
                        {
                            "event": "command_cache_hit",
                            "command": command,
                            "ts": _utc_now(),
                        },
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
            max_overall_timeout = (
                per_command_timeout * len(runnable_with_keys) * 2
            )  # generous buffer

            try:
                with ThreadPoolExecutor(
                    max_workers=min(verify_workers, len(runnable_with_keys))
                ) as pool:
                    future_map = {
                        pool.submit(
                            _run_with_timeout_detection,
                            command,
                            project_dir,
                            per_command_timeout,
                            log_path,
                            jsonl_path,
                        ): (command, cache_key, index)
                        for command, cache_key, index in runnable_with_keys
                    }

                    for future in as_completed(future_map, timeout=max_overall_timeout):
                        command, cache_key, command_index = future_map[future]
                        try:
                            live_result = _normalize_live_result(
                                future.result(timeout=per_command_timeout)
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
                            _append_line(
                                log_path, f"FUTURE_ERROR {command} ERROR={exc!r}"
                            )

            except TimeoutError:
                # Handle overall ThreadPool timeout
                elapsed = time.perf_counter() - overall_start
                _append_line(
                    log_path,
                    f"THREADPOOL_TIMEOUT overall_timeout={max_overall_timeout}s elapsed={elapsed:.3f}s",
                )
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
                        live_result = _normalize_live_result(
                            _run_with_timeout_detection(
                                command,
                                project_dir,
                                per_command_timeout,
                                log_path,
                                jsonl_path,
                            )
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

    if termination_reason:
        executed_commands = {str(item.get("command", "")) for item in results}
        deferred_commands = [
            command
            for command in runnable_commands
            if command not in executed_commands
            and command not in fail_fast_skipped_commands
        ]
        _append_unfinished_entries(
            unfinished_items=unfinished_items,
            commands=deferred_commands,
            reason=termination_reason,
        )
        for item in unfinished_items:
            command = str(item.get("command", "")).strip()
            reason = (
                str(item.get("reason", termination_reason)).strip()
                or termination_reason
            )
            if not command:
                continue
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
    failure_summary = _classify_verify_failures(results)
    failure_kind = str(failure_summary.get("failure_kind", "none"))
    has_failure = any(_is_failure(item) for item in results)
    partial = bool(termination_reason or unfinished_items)
    if partial:
        if termination_reason == "stall_auto_cancel":
            exit_code = 130
        else:
            exit_code = 124
        status = "partial"
    elif has_failure:
        exit_code = 3
        status = "failed"
    elif degraded:
        exit_code = 2
        status = "degraded"
    else:
        exit_code = 0
        status = "success"
    if has_failure:
        _append_line(log_path, f"FAILURE_KIND {failure_kind}")
        _append_line(
            log_path,
            "FAILURE_COUNTS "
            f"failed_total={int(failure_summary.get('failure_counts', {}).get('failed_total', 0))} "
            f"executor_failed={int(failure_summary.get('failure_counts', {}).get('executor_failed', 0))} "
            f"project_failed={int(failure_summary.get('failure_counts', {}).get('project_failed', 0))}",
        )

    if degraded:
        reason_text = (
            "; ".join(degrade_reasons) if degrade_reasons else "degraded execution"
        )
        _append_line(log_path, f"DEGRADE_REASON: {reason_text}")
        _append_line(
            log_path,
            "RISK: some checks were skipped because required tools are unavailable",
        )
        deadline_utc = _default_backfill_deadline(24)
        _append_line(
            log_path,
            f'BACKFILL_PLAN: owner=user commands="install missing tools/dependencies and rerun accel verify" deadline_utc={deadline_utc}',
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
            "partial": bool(partial),
            "partial_reason": str(termination_reason),
            "unfinished_count": int(len(unfinished_items)),
            "failure_kind": failure_kind,
            "failure_counts": failure_summary.get("failure_counts", {}),
            "selected_commands_count": int(selected_commands_count),
            "runnable_commands_count": int(runnable_commands_count),
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
        "partial": bool(partial),
        "partial_reason": str(termination_reason),
        "unfinished_items": unfinished_items,
        "unfinished_commands": [
            str(item.get("command", ""))
            for item in unfinished_items
            if str(item.get("command", "")).strip()
        ],
        "failure_kind": failure_kind,
        "failed_commands": list(failure_summary.get("failed_commands", [])),
        "executor_failed_commands": list(
            failure_summary.get("executor_failed_commands", [])
        ),
        "project_failed_commands": list(
            failure_summary.get("project_failed_commands", [])
        ),
        "failure_counts": dict(failure_summary.get("failure_counts", {})),
        "selected_commands_count": int(selected_commands_count),
        "runnable_commands_count": int(runnable_commands_count),
        "max_wall_time_seconds": float(verify_max_wall_time_seconds)
        if verify_max_wall_time_seconds is not None
        else None,
        "auto_cancel_on_stall": bool(verify_auto_cancel_on_stall),
        "timed_out": bool(termination_reason == "max_wall_time_exceeded"),
        "verify_selection_evidence": selection_evidence,
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
    verify_fail_fast = _normalize_bool(
        runtime_cfg.get("verify_fail_fast", False), False
    )
    verify_cache_enabled_cfg = _normalize_bool(
        runtime_cfg.get("verify_cache_enabled", True), True
    )
    verify_cache_failed_results = _normalize_bool(
        runtime_cfg.get("verify_cache_failed_results", False), False
    )
    verify_cache_ttl_seconds = _normalize_positive_int(
        runtime_cfg.get("verify_cache_ttl_seconds", 900), 900
    )
    verify_cache_max_entries = _normalize_positive_int(
        runtime_cfg.get("verify_cache_max_entries", 400), 400
    )
    verify_cache_failed_ttl_seconds = _normalize_positive_int(
        runtime_cfg.get("verify_cache_failed_ttl_seconds", 120),
        120,
    )
    verify_preflight_enabled = _normalize_bool(
        runtime_cfg.get("verify_preflight_enabled", True), True
    )
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
        verify_stall_timeout_seconds = min(
            float(per_command_timeout), max(1.0, stall_timeout_raw)
        )
    verify_max_wall_time_seconds = _normalize_optional_positive_float(
        runtime_cfg.get(
            "verify_max_wall_time_seconds",
            runtime_cfg.get("total_verify_timeout_seconds"),
        )
    )
    verify_auto_cancel_on_stall = _normalize_bool(
        runtime_cfg.get("verify_auto_cancel_on_stall", False), False
    )
    if verify_stall_timeout_seconds is None:
        verify_auto_cancel_on_stall = False
    workspace_routing_enabled = _normalize_bool(
        runtime_cfg.get("verify_workspace_routing_enabled", True), True
    )
    verify_mode = (
        "fail_fast"
        if verify_fail_fast
        else ("parallel" if verify_workers > 1 else "sequential")
    )

    nonce = uuid4().hex[:12]
    verify_job_id = "verify_" + nonce
    log_path = paths["verify"] / f"verify_{nonce}.log"
    jsonl_path = paths["verify"] / f"verify_{nonce}.jsonl"
    commands = select_verify_commands(config=config, changed_files=changed_files)
    selection_evidence = _build_verify_selection_evidence(
        config=config,
        changed_files=changed_files,
        commands=commands,
    )
    selected_commands_count = int(len(commands))
    verify_started_at = time.perf_counter()

    callback.on_start(verify_job_id, len(commands))

    _append_line(log_path, f"VERIFICATION_START {nonce} {_utc_now()}")
    _append_line(
        log_path, f"ENV cwd={project_dir} python={shutil.which('python') or ''}"
    )
    _append_line(
        log_path,
        "SELECTION_EVIDENCE "
        + json.dumps(selection_evidence, ensure_ascii=False),
    )
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
            "stall_timeout_seconds": float(verify_stall_timeout_seconds)
            if verify_stall_timeout_seconds is not None
            else None,
            "auto_cancel_on_stall": bool(verify_auto_cancel_on_stall),
            "max_wall_time_seconds": (
                float(verify_max_wall_time_seconds)
                if verify_max_wall_time_seconds is not None
                else None
            ),
            "workspace_routing_enabled": bool(workspace_routing_enabled),
            "selection_evidence": selection_evidence,
        },
    )

    changed_fingerprint = _build_changed_files_fingerprint(
        project_dir=project_dir, changed_files=changed_files
    )
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
    if selected_commands_count == 0:
        degraded = True
        degrade_reasons.append("no verify commands selected")
        _append_line(log_path, "NO_COMMANDS selected verify command list is empty")
        _append_jsonl(
            jsonl_path,
            {
                "event": "verify_no_commands",
                "reason": "no verify commands selected",
                "ts": _utc_now(),
            },
        )
    import_probe_cache: dict[tuple[str, str], bool] = {}
    runnable_commands: list[str] = []
    for command in commands:
        effective_command, compat_reason = _resolve_windows_compatible_command(
            project_dir, command
        )
        if compat_reason:
            _append_line(
                log_path,
                f"CMD_COMPAT {command} -> {effective_command} ({compat_reason})",
            )
            _append_jsonl(
                jsonl_path,
                {
                    "event": "command_compat_resolved",
                    "command": command,
                    "resolved_command": effective_command,
                    "reason": compat_reason,
                    "ts": _utc_now(),
                },
            )
        if verify_preflight_enabled:
            warnings = _preflight_warnings_for_command(
                project_dir=project_dir,
                command=effective_command,
                timeout_seconds=verify_preflight_timeout_seconds,
                import_probe_cache=import_probe_cache,
            )
            preflight_skip_reason = ""
            for warning in warnings:
                if warning not in degrade_reasons:
                    degrade_reasons.append(warning)
                degraded = True
                if (not preflight_skip_reason) and _should_skip_for_preflight(warning):
                    preflight_skip_reason = warning
                _append_line(
                    log_path, f"PREFLIGHT_WARN {effective_command} ({warning})"
                )
                _append_jsonl(
                    jsonl_path,
                    {
                        "event": "command_preflight_warning",
                        "command": effective_command,
                        "reason": warning,
                        "ts": _utc_now(),
                    },
                )
            if preflight_skip_reason:
                _append_line(
                    log_path,
                    f"SKIP {effective_command} ({preflight_skip_reason})",
                )
                _append_jsonl(
                    jsonl_path,
                    {
                        "event": "command_skipped",
                        "command": effective_command,
                        "reason": preflight_skip_reason,
                        "ts": _utc_now(),
                        "mode": verify_mode,
                        "fail_fast": bool(verify_fail_fast),
                        "cache_hits": int(cache_hits),
                        "cache_misses": int(cache_misses),
                        "fail_fast_skipped": False,
                    },
                )
                callback.on_skip(verify_job_id, effective_command, preflight_skip_reason)
                continue
        binary = _command_binary(effective_command)
        if binary and shutil.which(binary) is None:
            degraded = True
            reason = f"missing command binary: {binary}"
            degrade_reasons.append(reason)
            _append_line(log_path, f"SKIP {effective_command} ({reason})")
            _append_jsonl(
                jsonl_path,
                {
                    "event": "command_skipped",
                    "command": effective_command,
                    "reason": reason,
                    "ts": _utc_now(),
                    "mode": verify_mode,
                    "fail_fast": bool(verify_fail_fast),
                    "cache_hits": int(cache_hits),
                    "cache_misses": int(cache_misses),
                    "fail_fast_skipped": False,
                },
            )
            callback.on_skip(verify_job_id, effective_command, reason)
            continue
        runnable_commands.append(effective_command)
    runnable_commands_count = int(len(runnable_commands))

    results: list[dict[str, Any]] = []
    fail_fast_skipped_commands: list[str] = []
    unfinished_items: list[dict[str, Any]] = []
    termination_reason = ""

    def _make_output_callback(
        command_name: str, activity_ref: dict[str, float]
    ) -> Callable[[str, str], None]:
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
            remaining_wall_time = _remaining_wall_time_seconds(
                started_at=verify_started_at,
                max_wall_time_seconds=verify_max_wall_time_seconds,
            )
            command_timeout_budget = _timeboxed_command_timeout(
                per_command_timeout=per_command_timeout,
                remaining_wall_time=remaining_wall_time,
            )
            if command_timeout_budget <= 0:
                termination_reason = "max_wall_time_exceeded"
                _append_line(
                    log_path,
                    f"MAX_WALL_TIME_EXCEEDED elapsed={time.perf_counter() - verify_started_at:.3f}s "
                    f"limit={float(verify_max_wall_time_seconds or 0.0):.3f}s",
                )
                _append_unfinished_entries(
                    unfinished_items=unfinished_items,
                    commands=runnable_commands[index:],
                    reason=termination_reason,
                )
                break

            callback.on_command_start(
                verify_job_id, command, index, len(runnable_commands)
            )
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
                    cached_result = _normalize_cached_result(
                        command=command, entry=cached_entry
                    )
                    results.append(cached_result)
                    _append_line(log_path, f"CACHE_HIT {command}")
                    _append_jsonl(
                        jsonl_path,
                        {
                            "event": "command_cache_hit",
                            "command": command,
                            "ts": _utc_now(),
                        },
                    )
                    cache_failure = _is_failure(cached_result)
                    fail_fast_skip_flag = bool(
                        cache_failure and (index < (len(runnable_commands) - 1))
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
                    callback.on_progress(
                        verify_job_id, index + 1, len(runnable_commands), ""
                    )
                    if cache_failure:
                        fail_fast_skipped_commands = runnable_commands[index + 1 :]
                        break
                    continue
                cache_misses += 1

            activity_ref = {"ts": time.perf_counter()}
            output_callback = _make_output_callback(command, activity_ref)

            def _activity_probe(ref: dict[str, float] = activity_ref) -> float:
                return float(ref["ts"])

            cancel_event = threading.Event()

            def _on_stall_auto_cancel(
                idle_sec: float, command_name: str = command
            ) -> None:
                nonlocal termination_reason
                if not termination_reason:
                    termination_reason = "stall_auto_cancel"
                    _append_line(
                        log_path,
                        f"STALL_AUTO_CANCEL {command_name} idle_sec={idle_sec:.3f}",
                    )

            stop_tick, cmd_started, tick_thread = _start_command_tick_thread(
                callback,
                job_id=verify_job_id,
                command=command,
                timeout_seconds=command_timeout_budget,
                stall_timeout_seconds=verify_stall_timeout_seconds,
                activity_probe=_activity_probe,
                auto_cancel_on_stall=verify_auto_cancel_on_stall,
                cancel_event=cancel_event,
                on_stall_auto_cancel=_on_stall_auto_cancel,
            )
            try:
                live_result = _normalize_live_result(
                    _invoke_run_command(
                        command,
                        project_dir,
                        command_timeout_budget,
                        output_callback=output_callback,
                        cancel_event=cancel_event,
                        stall_timeout_seconds=(
                            verify_stall_timeout_seconds
                            if verify_auto_cancel_on_stall
                            else None
                        ),
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
                max(0.0, float(command_timeout_budget) - cmd_elapsed),
                "running",
                current_command=command,
                command_elapsed_sec=cmd_elapsed,
                command_timeout_sec=float(command_timeout_budget),
                command_progress_pct=100.0,
                stall_detected=False
                if verify_stall_timeout_seconds is not None
                else None,
                stall_elapsed_sec=0.0
                if verify_stall_timeout_seconds is not None
                else None,
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
            fail_fast_skip_flag = bool(
                _is_failure(live_result) and (index < (len(runnable_commands) - 1))
            )
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
            if bool(live_result.get("stalled", False)) and verify_auto_cancel_on_stall:
                if not termination_reason:
                    termination_reason = "stall_auto_cancel"
                    _append_line(
                        log_path, f"STALL_AUTO_CANCEL {command} idle_sec=unknown"
                    )
                _append_unfinished_entries(
                    unfinished_items=unfinished_items,
                    commands=runnable_commands[index + 1 :],
                    reason=termination_reason,
                )
                break
            if _is_failure(live_result):
                fail_fast_skipped_commands = runnable_commands[index + 1 :]
                break
    else:
        runnable_with_keys: list[tuple[str, str | None, int, int]] = []
        for index, command in enumerate(runnable_commands):
            remaining_wall_time = _remaining_wall_time_seconds(
                started_at=verify_started_at,
                max_wall_time_seconds=verify_max_wall_time_seconds,
            )
            command_timeout_budget = _timeboxed_command_timeout(
                per_command_timeout=per_command_timeout,
                remaining_wall_time=remaining_wall_time,
            )
            if command_timeout_budget <= 0:
                if not termination_reason:
                    termination_reason = "max_wall_time_exceeded"
                    _append_line(
                        log_path,
                        f"MAX_WALL_TIME_EXCEEDED elapsed={time.perf_counter() - verify_started_at:.3f}s "
                        f"limit={float(verify_max_wall_time_seconds or 0.0):.3f}s",
                    )
                _append_unfinished_entries(
                    unfinished_items=unfinished_items,
                    commands=runnable_commands[index:],
                    reason=termination_reason,
                )
                break
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
                    cached_result = _normalize_cached_result(
                        command=command, entry=cached_entry
                    )
                    results.append(cached_result)
                    _append_line(log_path, f"CACHE_HIT {command}")
                    _append_jsonl(
                        jsonl_path,
                        {
                            "event": "command_cache_hit",
                            "command": command,
                            "ts": _utc_now(),
                        },
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
            runnable_with_keys.append(
                (command, cache_key, index, command_timeout_budget)
            )

        if runnable_with_keys:
            overall_start = time.perf_counter()
            max_overall_timeout = float(
                per_command_timeout * len(runnable_with_keys) * 2
            )
            if verify_max_wall_time_seconds is not None:
                max_overall_timeout = min(
                    float(max_overall_timeout),
                    float(verify_max_wall_time_seconds)
                    + max(5.0, 2.0 * float(per_command_timeout)),
                )

            callback.on_stage_change(verify_job_id, VerifyStage.PARALLEL)

            try:
                with ThreadPoolExecutor(
                    max_workers=min(verify_workers, len(runnable_with_keys))
                ) as pool:
                    future_map: dict[
                        Any,
                        tuple[
                            str,
                            str | None,
                            float,
                            int,
                            int,
                            threading.Event,
                            threading.Thread,
                            threading.Event,
                        ],
                    ] = {}
                    total_commands = len(runnable_with_keys)
                    completed_count = 0
                    for index, (
                        command,
                        cache_key,
                        command_index,
                        command_timeout_budget,
                    ) in enumerate(runnable_with_keys):
                        callback.on_command_start(
                            verify_job_id, command, index, total_commands
                        )
                        callback.on_progress(
                            verify_job_id, completed_count, total_commands, command
                        )
                        activity_ref = {"ts": time.perf_counter()}
                        output_callback = _make_output_callback(command, activity_ref)

                        def _activity_probe(
                            ref: dict[str, float] = activity_ref,
                        ) -> float:
                            return float(ref["ts"])

                        cancel_event = threading.Event()

                        def _on_stall_auto_cancel(
                            idle_sec: float, command_name: str = command
                        ) -> None:
                            nonlocal termination_reason
                            if not termination_reason:
                                termination_reason = "stall_auto_cancel"
                                _append_line(
                                    log_path,
                                    f"STALL_AUTO_CANCEL {command_name} idle_sec={idle_sec:.3f}",
                                )

                        stop_tick, cmd_started, tick_thread = (
                            _start_command_tick_thread(
                                callback,
                                job_id=verify_job_id,
                                command=command,
                                timeout_seconds=command_timeout_budget,
                                stall_timeout_seconds=verify_stall_timeout_seconds,
                                activity_probe=_activity_probe,
                                auto_cancel_on_stall=verify_auto_cancel_on_stall,
                                cancel_event=cancel_event,
                                on_stall_auto_cancel=_on_stall_auto_cancel,
                            )
                        )
                        future = pool.submit(
                            _run_with_timeout_detection,
                            command,
                            project_dir,
                            command_timeout_budget,
                            log_path,
                            jsonl_path,
                            output_callback,
                            cancel_event,
                            verify_stall_timeout_seconds
                            if verify_auto_cancel_on_stall
                            else None,
                        )
                        future_map[future] = (
                            command,
                            cache_key,
                            cmd_started,
                            command_index,
                            command_timeout_budget,
                            stop_tick,
                            tick_thread,
                            cancel_event,
                        )

                    for future in as_completed(future_map, timeout=max_overall_timeout):
                        (
                            command,
                            cache_key,
                            started_at,
                            command_index,
                            command_timeout_budget,
                            stop_tick,
                            tick_thread,
                            cancel_event,
                        ) = future_map[future]

                        try:
                            live_result = _normalize_live_result(
                                future.result(timeout=command_timeout_budget)
                            )
                            cmd_elapsed = max(0.0, time.perf_counter() - started_at)
                            _safe_callback_call(
                                callback,
                                "on_heartbeat",
                                verify_job_id,
                                cmd_elapsed,
                                max(0.0, float(command_timeout_budget) - cmd_elapsed),
                                "running",
                                current_command=command,
                                command_elapsed_sec=cmd_elapsed,
                                command_timeout_sec=float(command_timeout_budget),
                                command_progress_pct=100.0,
                                stall_detected=False
                                if verify_stall_timeout_seconds is not None
                                else None,
                                stall_elapsed_sec=0.0
                                if verify_stall_timeout_seconds is not None
                                else None,
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
                                "stderr": f"agent-accel: ThreadPool future timeout after {command_timeout_budget}s",
                                "timed_out": True,
                                "cancelled": False,
                                "stalled": False,
                                "cancel_reason": "",
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
                                command_timeout_sec=float(command_timeout_budget),
                                command_progress_pct=100.0,
                                stall_detected=False
                                if verify_stall_timeout_seconds is not None
                                else None,
                                stall_elapsed_sec=0.0
                                if verify_stall_timeout_seconds is not None
                                else None,
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
                                "cancelled": False,
                                "stalled": False,
                                "cancel_reason": "",
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
                                stall_detected=False
                                if verify_stall_timeout_seconds is not None
                                else None,
                                stall_elapsed_sec=0.0
                                if verify_stall_timeout_seconds is not None
                                else None,
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
                            _append_line(
                                log_path, f"FUTURE_ERROR {command} ERROR={exc!r}"
                            )
                        finally:
                            stop_tick.set()
                            tick_thread.join(timeout=0.1)

                        completed_count += 1
                        callback.on_progress(
                            verify_job_id, completed_count, len(runnable_with_keys), ""
                        )

                        if (
                            bool(live_result.get("stalled", False))
                            and verify_auto_cancel_on_stall
                        ):
                            if not termination_reason:
                                termination_reason = "stall_auto_cancel"
                                _append_line(
                                    log_path,
                                    f"STALL_AUTO_CANCEL {command} idle_sec=unknown",
                                )
                            for pending_future, pending_meta in future_map.items():
                                if pending_future is future:
                                    continue
                                if pending_future.done():
                                    continue
                                pending_meta[-1].set()

                        remaining_wall_time = _remaining_wall_time_seconds(
                            started_at=verify_started_at,
                            max_wall_time_seconds=verify_max_wall_time_seconds,
                        )
                        if remaining_wall_time is not None and remaining_wall_time <= 0:
                            if not termination_reason:
                                termination_reason = "max_wall_time_exceeded"
                                _append_line(
                                    log_path,
                                    f"MAX_WALL_TIME_EXCEEDED elapsed={time.perf_counter() - verify_started_at:.3f}s "
                                    f"limit={float(verify_max_wall_time_seconds or 0.0):.3f}s",
                                )
                            for pending_future, pending_meta in future_map.items():
                                if pending_future is future:
                                    continue
                                if pending_future.done():
                                    continue
                                pending_meta[-1].set()

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
                for _future, (
                    _command,
                    _cache_key_value,
                    _started_at,
                    _index,
                    _timeout_budget,
                    stop_tick,
                    tick_thread,
                    cancel_event,
                ) in future_map.items():
                    cancel_event.set()
                    stop_tick.set()
                    tick_thread.join(timeout=0.1)
                elapsed = time.perf_counter() - overall_start
                _append_line(
                    log_path,
                    f"THREADPOOL_TIMEOUT overall_timeout={max_overall_timeout}s elapsed={elapsed:.3f}s",
                )
                if not termination_reason and verify_max_wall_time_seconds is not None:
                    termination_reason = "max_wall_time_exceeded"
                for (
                    command,
                    cache_key,
                    command_index,
                    _timeout_budget,
                ) in runnable_with_keys:
                    if not any(r["command"] == command for r in results):
                        _append_unfinished_entries(
                            unfinished_items=unfinished_items,
                            commands=[command],
                            reason=termination_reason or "parallel_timeout",
                        )
            except Exception as exc:
                for _future, (
                    _command,
                    _cache_key_value,
                    _started_at,
                    _index,
                    _timeout_budget,
                    stop_tick,
                    tick_thread,
                    cancel_event,
                ) in future_map.items():
                    cancel_event.set()
                    stop_tick.set()
                    tick_thread.join(timeout=0.1)
                elapsed = time.perf_counter() - overall_start
                _append_line(log_path, f"THREADPOOL_ERROR ERROR={exc!r}")
                _append_line(log_path, "FALLBACK_SEQUENTIAL_EXECUTION")
                callback.on_stage_change(verify_job_id, VerifyStage.SEQUENTIAL)

                for i, (
                    command,
                    cache_key,
                    command_index,
                    command_timeout_budget,
                ) in enumerate(runnable_with_keys):
                    if not any(r["command"] == command for r in results):
                        callback.on_command_start(
                            verify_job_id, command, i, len(runnable_with_keys)
                        )
                        activity_ref = {"ts": time.perf_counter()}
                        output_callback = _make_output_callback(command, activity_ref)

                        def _activity_probe(
                            ref: dict[str, float] = activity_ref,
                        ) -> float:
                            return float(ref["ts"])

                        cancel_event = threading.Event()

                        def _on_stall_auto_cancel(
                            idle_sec: float, command_name: str = command
                        ) -> None:
                            nonlocal termination_reason
                            if not termination_reason:
                                termination_reason = "stall_auto_cancel"
                                _append_line(
                                    log_path,
                                    f"STALL_AUTO_CANCEL {command_name} idle_sec={idle_sec:.3f}",
                                )

                        stop_tick, cmd_started, tick_thread = (
                            _start_command_tick_thread(
                                callback,
                                job_id=verify_job_id,
                                command=command,
                                timeout_seconds=command_timeout_budget,
                                stall_timeout_seconds=verify_stall_timeout_seconds,
                                activity_probe=_activity_probe,
                                auto_cancel_on_stall=verify_auto_cancel_on_stall,
                                cancel_event=cancel_event,
                                on_stall_auto_cancel=_on_stall_auto_cancel,
                            )
                        )
                        try:
                            live_result = _normalize_live_result(
                                _run_with_timeout_detection(
                                    command,
                                    project_dir,
                                    command_timeout_budget,
                                    log_path,
                                    jsonl_path,
                                    output_callback,
                                    cancel_event,
                                    verify_stall_timeout_seconds
                                    if verify_auto_cancel_on_stall
                                    else None,
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
                            max(0.0, float(command_timeout_budget) - cmd_elapsed),
                            "running",
                            current_command=command,
                            command_elapsed_sec=cmd_elapsed,
                            command_timeout_sec=float(command_timeout_budget),
                            command_progress_pct=100.0,
                            stall_detected=False
                            if verify_stall_timeout_seconds is not None
                            else None,
                            stall_elapsed_sec=0.0
                            if verify_stall_timeout_seconds is not None
                            else None,
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
                        callback.on_progress(
                            verify_job_id, i + 1, len(runnable_with_keys), ""
                        )

        if termination_reason:
            executed_commands = {str(row.get("command", "")) for row in results}
            pending_commands = [
                command
                for command, _cache_key, _command_index, _timeout_budget in runnable_with_keys
                if command not in executed_commands
            ]
            _append_unfinished_entries(
                unfinished_items=unfinished_items,
                commands=pending_commands,
                reason=termination_reason,
            )

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

    if termination_reason:
        executed_commands = {str(item.get("command", "")) for item in results}
        deferred_commands = [
            command
            for command in runnable_commands
            if command not in executed_commands
            and command not in fail_fast_skipped_commands
        ]
        _append_unfinished_entries(
            unfinished_items=unfinished_items,
            commands=deferred_commands,
            reason=termination_reason,
        )
        emitted_unfinished: set[str] = set()
        for item in unfinished_items:
            command = str(item.get("command", "")).strip()
            reason = (
                str(item.get("reason", termination_reason)).strip()
                or termination_reason
            )
            if not command or command in emitted_unfinished:
                continue
            emitted_unfinished.add(command)
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
    failure_summary = _classify_verify_failures(results)
    failure_kind = str(failure_summary.get("failure_kind", "none"))
    has_failure = any(_is_failure(item) for item in results)
    partial = bool(termination_reason or unfinished_items)
    if partial:
        if termination_reason == "stall_auto_cancel":
            exit_code = 130
        else:
            exit_code = 124
        status = "partial"
    elif has_failure:
        exit_code = 3
        status = "failed"
    elif degraded:
        exit_code = 2
        status = "degraded"
    else:
        exit_code = 0
        status = "success"
    if has_failure:
        _append_line(log_path, f"FAILURE_KIND {failure_kind}")
        _append_line(
            log_path,
            "FAILURE_COUNTS "
            f"failed_total={int(failure_summary.get('failure_counts', {}).get('failed_total', 0))} "
            f"executor_failed={int(failure_summary.get('failure_counts', {}).get('executor_failed', 0))} "
            f"project_failed={int(failure_summary.get('failure_counts', {}).get('project_failed', 0))}",
        )

    if degraded:
        reason_text = (
            "; ".join(degrade_reasons) if degrade_reasons else "degraded execution"
        )
        _append_line(log_path, f"DEGRADE_REASON: {reason_text}")
        _append_line(
            log_path,
            "RISK: some checks were skipped because required tools are unavailable",
        )
        deadline_utc = _default_backfill_deadline(24)
        _append_line(
            log_path,
            f'BACKFILL_PLAN: owner=user commands="install missing tools/dependencies and rerun accel verify" deadline_utc={deadline_utc}',
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
            "partial": bool(partial),
            "partial_reason": str(termination_reason),
            "unfinished_count": int(len(unfinished_items)),
            "failure_kind": failure_kind,
            "failure_counts": failure_summary.get("failure_counts", {}),
            "selected_commands_count": int(selected_commands_count),
            "runnable_commands_count": int(runnable_commands_count),
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
        "partial": bool(partial),
        "partial_reason": str(termination_reason),
        "unfinished_items": unfinished_items,
        "unfinished_commands": [
            str(item.get("command", ""))
            for item in unfinished_items
            if str(item.get("command", "")).strip()
        ],
        "failure_kind": failure_kind,
        "failed_commands": list(failure_summary.get("failed_commands", [])),
        "executor_failed_commands": list(
            failure_summary.get("executor_failed_commands", [])
        ),
        "project_failed_commands": list(
            failure_summary.get("project_failed_commands", [])
        ),
        "failure_counts": dict(failure_summary.get("failure_counts", {})),
        "selected_commands_count": int(selected_commands_count),
        "runnable_commands_count": int(runnable_commands_count),
        "max_wall_time_seconds": float(verify_max_wall_time_seconds)
        if verify_max_wall_time_seconds is not None
        else None,
        "auto_cancel_on_stall": bool(verify_auto_cancel_on_stall),
        "timed_out": bool(termination_reason == "max_wall_time_exceeded"),
        "verify_selection_evidence": selection_evidence,
    }
