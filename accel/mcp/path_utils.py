"""Path utilities for MCP server.

This module provides functions for normalizing and resolving paths
used in MCP server operations.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from ..config import resolve_effective_config
from ..harborpilot_paths import default_accel_runtime_home
from ..storage.cache import ensure_project_dirs, project_paths


def normalize_project_dir(project_value: Any) -> Path:
    """Normalize a project directory value to an absolute Path.

    Args:
        project_value: Project directory value (string or Path-like).

    Returns:
        Absolute Path to the project directory.
    """
    project_text = str(project_value or ".")
    return Path(os.path.abspath(project_text))


def resolve_path(project_dir: Path, path_value: Any) -> Path | None:
    """Resolve a path value relative to a project directory.

    Args:
        project_dir: Base project directory.
        path_value: Path value to resolve.

    Returns:
        Absolute Path or None if path_value is empty.
    """
    path_text = str(path_value or "").strip()
    if not path_text:
        return None
    path = Path(path_text)
    if not path.is_absolute():
        path = project_dir / path
    return Path(os.path.abspath(str(path)))


def resolve_project_storage_paths(project_dir: Path) -> dict[str, Path]:
    """Resolve storage paths for a project.

    Args:
        project_dir: Project directory.

    Returns:
        Dict with storage paths (index, state, logs, etc.).
    """
    config = resolve_effective_config(project_dir)
    runtime_cfg = (
        config.get("runtime", {}) if isinstance(config.get("runtime", {}), dict) else {}
    )
    accel_home_value = str(runtime_cfg.get("accel_home", "") or "").strip()
    if accel_home_value:
        accel_home = Path(accel_home_value).resolve()
    else:
        accel_home = default_accel_runtime_home(project_dir).resolve()
    paths = project_paths(accel_home, project_dir)
    ensure_project_dirs(paths)
    return paths


def normalize_relative_path(path: str) -> str:
    """Normalize a path to use forward slashes.

    Args:
        path: Path string to normalize.

    Returns:
        Normalized path with forward slashes.
    """
    return str(path or "").replace("\\", "/").strip()
