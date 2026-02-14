"""Type coercion utilities for MCP server parameters.

This module provides functions for converting various input types
to their expected types with sensible defaults and error handling.
"""

from __future__ import annotations

import json
from typing import Any

# Literal values recognized as False
FALSE_LITERALS = {
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

# Literal values recognized as True
TRUE_LITERALS = {"1", "true", "yes", "on"}

# Valid sync timeout actions
SYNC_TIMEOUT_ACTIONS = {"poll", "cancel"}

# Valid context sync timeout actions
CONTEXT_SYNC_TIMEOUT_ACTIONS = {"fallback_async", "cancel"}

# Budget presets for context generation
BUDGET_PRESETS: dict[str, dict[str, int]] = {
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

BUDGET_PRESET_ALIASES: dict[str, str] = {
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


def coerce_bool(value: Any, default: bool = False) -> bool:
    """Coerce a value to boolean with default.

    Args:
        value: Value to coerce.
        default: Default value if coercion fails.

    Returns:
        Boolean value.
    """
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        token = value.strip().lower()
        if token in FALSE_LITERALS:
            return False
        if token in TRUE_LITERALS:
            return True
        return default
    return default


def coerce_optional_bool(value: Any) -> bool | None:
    """Coerce a value to optional boolean.

    Args:
        value: Value to coerce.

    Returns:
        Boolean value or None.
    """
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        token = value.strip().lower()
        if token in FALSE_LITERALS:
            return False
        if token in TRUE_LITERALS:
            return True
        return None
    return None


def coerce_optional_int(value: Any) -> int | None:
    """Coerce a value to optional integer.

    Args:
        value: Value to coerce.

    Returns:
        Integer value or None.
    """
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        token = value.strip()
        if not token:
            return None
        if token.lower() in FALSE_LITERALS:
            return None
        try:
            return int(token)
        except ValueError:
            try:
                return int(float(token))
            except ValueError:
                return None
    return None


def coerce_optional_float(value: Any) -> float | None:
    """Coerce a value to optional float.

    Args:
        value: Value to coerce.

    Returns:
        Float value or None.
    """
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
        if token.lower() in FALSE_LITERALS:
            return None
        try:
            return float(token)
        except ValueError:
            return None
    return None


def coerce_sync_timeout_action(value: Any, default: str = "poll") -> str:
    """Coerce a value to sync timeout action.

    Args:
        value: Value to coerce.
        default: Default action.

    Returns:
        Valid sync timeout action string.
    """
    token = str(value or "").strip().lower()
    if token in SYNC_TIMEOUT_ACTIONS:
        return token
    fallback = str(default or "poll").strip().lower()
    if fallback in SYNC_TIMEOUT_ACTIONS:
        return fallback
    return "poll"


def coerce_context_sync_timeout_action(
    value: Any, default: str = "fallback_async"
) -> str:
    """Coerce a value to context sync timeout action.

    Args:
        value: Value to coerce.
        default: Default action.

    Returns:
        Valid context sync timeout action string.
    """
    token = str(value or "").strip().lower()
    if token == "poll":
        token = "fallback_async"
    if token in CONTEXT_SYNC_TIMEOUT_ACTIONS:
        return token
    fallback = str(default or "fallback_async").strip().lower()
    if fallback == "poll":
        fallback = "fallback_async"
    if fallback in CONTEXT_SYNC_TIMEOUT_ACTIONS:
        return fallback
    return "fallback_async"


def coerce_events_limit(
    value: Any, default_value: int = 30, max_value: int = 500
) -> int:
    """Coerce a value to events limit.

    Args:
        value: Value to coerce.
        default_value: Default limit.
        max_value: Maximum allowed limit.

    Returns:
        Events limit within bounds.
    """
    parsed = coerce_optional_int(value)
    if parsed is None:
        parsed = default_value
    return max(1, min(int(max_value), int(parsed)))


def to_string_list(value: list[str] | str | None) -> list[str]:
    """Convert a value to a list of strings.

    Handles JSON arrays, comma/newline/semicolon separated strings.

    Args:
        value: Value to convert.

    Returns:
        List of non-empty strings.
    """
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
        except (json.JSONDecodeError, TypeError):
            pass

    normalized = text.replace("\r\n", ",").replace("\n", ",").replace(";", ",")
    return [part.strip() for part in normalized.split(",") if part.strip()]


def to_budget_override(value: dict[str, int] | str | None) -> dict[str, int]:
    """Convert a value to a budget override dict.

    Handles preset names, JSON objects, and key=value strings.

    Args:
        value: Value to convert.

    Returns:
        Budget override dictionary.

    Raises:
        ValueError: If the value cannot be parsed.
    """
    if value is None:
        return {}

    if isinstance(value, str):
        token = value.strip().lower()
        if not token:
            return {}

        canonical = BUDGET_PRESET_ALIASES.get(token, token)
        if canonical in BUDGET_PRESETS:
            return dict(BUDGET_PRESETS[canonical])

        if token.startswith("{") and token.endswith("}"):
            try:
                parsed = json.loads(token)
                if isinstance(parsed, dict):
                    return to_budget_override(parsed)
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
                return to_budget_override(parsed_pairs)

        preset_names = ", ".join(sorted(BUDGET_PRESETS.keys()))
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
