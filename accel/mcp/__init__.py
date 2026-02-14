"""MCP server utility modules.

This package contains extracted utilities from mcp_server.py
to improve maintainability and testability.
"""

from __future__ import annotations

from .coercion import (
    FALSE_LITERALS,
    TRUE_LITERALS,
    SYNC_TIMEOUT_ACTIONS,
    CONTEXT_SYNC_TIMEOUT_ACTIONS,
    BUDGET_PRESETS,
    BUDGET_PRESET_ALIASES,
    coerce_bool,
    coerce_optional_bool,
    coerce_optional_int,
    coerce_optional_float,
    coerce_sync_timeout_action,
    coerce_context_sync_timeout_action,
    coerce_events_limit,
    to_string_list,
    to_budget_override,
)
from .path_utils import (
    normalize_project_dir,
    resolve_path,
    resolve_project_storage_paths,
    normalize_relative_path,
)

__all__ = [
    # Constants
    "FALSE_LITERALS",
    "TRUE_LITERALS",
    "SYNC_TIMEOUT_ACTIONS",
    "CONTEXT_SYNC_TIMEOUT_ACTIONS",
    "BUDGET_PRESETS",
    "BUDGET_PRESET_ALIASES",
    # Coercion functions
    "coerce_bool",
    "coerce_optional_bool",
    "coerce_optional_int",
    "coerce_optional_float",
    "coerce_sync_timeout_action",
    "coerce_context_sync_timeout_action",
    "coerce_events_limit",
    "to_string_list",
    "to_budget_override",
    # Path functions
    "normalize_project_dir",
    "resolve_path",
    "resolve_project_storage_paths",
    "normalize_relative_path",
]
