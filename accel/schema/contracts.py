from __future__ import annotations

from typing import Any


_CONSTRAINT_MODES = {"off", "warn", "strict"}


def normalize_constraint_mode(value: Any, default_mode: str = "warn") -> str:
    token = str(value or default_mode).strip().lower()
    if token in _CONSTRAINT_MODES:
        return token
    fallback = str(default_mode or "warn").strip().lower()
    return fallback if fallback in _CONSTRAINT_MODES else "warn"


def _coerce_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def _ensure_list_of_dicts(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    out: list[dict[str, Any]] = []
    for item in value:
        if isinstance(item, dict):
            out.append(dict(item))
    return out


def _apply_or_raise(
    *,
    mode: str,
    warnings: list[str],
    repair_count: int,
    message: str,
) -> tuple[list[str], int]:
    if mode == "strict":
        raise ValueError(message)
    warnings.append(message)
    return warnings, int(repair_count) + 1


def enforce_context_pack_contract(pack: dict[str, Any], mode: str) -> tuple[dict[str, Any], list[str], int]:
    normalized_mode = normalize_constraint_mode(mode, default_mode="warn")
    if normalized_mode == "off":
        return dict(pack), [], 0

    payload = dict(pack)
    warnings: list[str] = []
    repair_count = 0

    if _coerce_int(payload.get("version", 0), 0) <= 0:
        warnings, repair_count = _apply_or_raise(
            mode=normalized_mode,
            warnings=warnings,
            repair_count=repair_count,
            message="context pack version missing/invalid",
        )
        payload["version"] = 1

    if not str(payload.get("task", "")).strip():
        warnings, repair_count = _apply_or_raise(
            mode=normalized_mode,
            warnings=warnings,
            repair_count=repair_count,
            message="context pack task missing/invalid",
        )
        payload["task"] = str(payload.get("task", "")).strip() or "unknown_task"

    if not isinstance(payload.get("top_files"), list):
        warnings, repair_count = _apply_or_raise(
            mode=normalized_mode,
            warnings=warnings,
            repair_count=repair_count,
            message="context pack top_files must be list",
        )
        payload["top_files"] = []

    if not isinstance(payload.get("snippets"), list):
        warnings, repair_count = _apply_or_raise(
            mode=normalized_mode,
            warnings=warnings,
            repair_count=repair_count,
            message="context pack snippets must be list",
        )
        payload["snippets"] = []

    verify_plan = payload.get("verify_plan")
    if not isinstance(verify_plan, dict):
        warnings, repair_count = _apply_or_raise(
            mode=normalized_mode,
            warnings=warnings,
            repair_count=repair_count,
            message="context pack verify_plan must be object",
        )
        verify_plan = {}
        payload["verify_plan"] = verify_plan

    target_tests = verify_plan.get("target_tests", [])
    if not isinstance(target_tests, list):
        warnings, repair_count = _apply_or_raise(
            mode=normalized_mode,
            warnings=warnings,
            repair_count=repair_count,
            message="context pack verify_plan.target_tests must be list",
        )
        verify_plan["target_tests"] = []
    else:
        verify_plan["target_tests"] = [str(item) for item in target_tests if str(item).strip()]

    target_checks = verify_plan.get("target_checks", [])
    if not isinstance(target_checks, list):
        warnings, repair_count = _apply_or_raise(
            mode=normalized_mode,
            warnings=warnings,
            repair_count=repair_count,
            message="context pack verify_plan.target_checks must be list",
        )
        verify_plan["target_checks"] = []
    else:
        verify_plan["target_checks"] = [str(item) for item in target_checks if str(item).strip()]

    snippets = _ensure_list_of_dicts(payload.get("snippets"))
    repaired_snippets: list[dict[str, Any]] = []
    for snippet in snippets:
        fixed = dict(snippet)
        for key in ("path", "content", "reason", "symbol"):
            fixed[key] = str(fixed.get(key, ""))
        fixed["start_line"] = max(1, _coerce_int(fixed.get("start_line", 1), 1))
        fixed["end_line"] = max(fixed["start_line"], _coerce_int(fixed.get("end_line", fixed["start_line"]), fixed["start_line"]))
        repaired_snippets.append(fixed)
    payload["snippets"] = repaired_snippets

    return payload, warnings, repair_count


def enforce_context_payload_contract(payload: dict[str, Any], mode: str) -> tuple[dict[str, Any], list[str], int]:
    normalized_mode = normalize_constraint_mode(mode, default_mode="warn")
    if normalized_mode == "off":
        return dict(payload), [], 0

    out = dict(payload)
    warnings: list[str] = []
    repair_count = 0

    int_fields = (
        "estimated_tokens",
        "estimated_source_tokens",
        "estimated_changed_files_tokens",
        "estimated_snippets_only_tokens",
        "selected_tests_count",
        "selected_checks_count",
    )
    for field in int_fields:
        value = _coerce_int(out.get(field, 0), 0)
        if value < 0:
            value = 0
        if out.get(field) != value:
            warnings, repair_count = _apply_or_raise(
                mode=normalized_mode,
                warnings=warnings,
                repair_count=repair_count,
                message=f"context payload field {field} repaired",
            )
        out[field] = int(value)

    warnings_value = out.get("warnings")
    if warnings_value is None:
        out["warnings"] = []
    elif not isinstance(warnings_value, list):
        warnings, repair_count = _apply_or_raise(
            mode=normalized_mode,
            warnings=warnings,
            repair_count=repair_count,
            message="context payload warnings must be list",
        )
        out["warnings"] = []
    else:
        out["warnings"] = [str(item) for item in warnings_value]

    return out, warnings, repair_count


def enforce_verify_summary_contract(
    summary: dict[str, Any],
    *,
    status: dict[str, Any],
    mode: str,
) -> tuple[dict[str, Any], list[str], int]:
    normalized_mode = normalize_constraint_mode(mode, default_mode="warn")
    if normalized_mode == "off":
        out = dict(summary)
        out.setdefault("state_source", "raw")
        out.setdefault("constraint_repair_count", 0)
        return out, [], 0

    payload = dict(summary)
    warnings: list[str] = []
    repair_count = 0

    latest_state = str(payload.get("latest_state", "")).strip().lower()
    status_state = str(status.get("state", "")).strip().lower()
    terminal_states = {"completed", "failed", "cancelled"}
    state_source = str(payload.get("state_source", "events")).strip().lower() or "events"

    if status_state in terminal_states:
        if latest_state != status_state:
            warnings, repair_count = _apply_or_raise(
                mode=normalized_mode,
                warnings=warnings,
                repair_count=repair_count,
                message=f"verify summary latest_state repaired from {latest_state or 'empty'} to {status_state}",
            )
            payload["latest_state"] = status_state
        else:
            payload["latest_state"] = status_state
        state_source = "status_terminal"
    elif latest_state:
        payload["latest_state"] = latest_state
    else:
        warnings, repair_count = _apply_or_raise(
            mode=normalized_mode,
            warnings=warnings,
            repair_count=repair_count,
            message="verify summary latest_state missing, using status state",
        )
        payload["latest_state"] = status_state or "unknown"
        state_source = "status_fallback"

    payload["state_source"] = state_source
    payload["constraint_repair_count"] = int(repair_count)
    return payload, warnings, repair_count
