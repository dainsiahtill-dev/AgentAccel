from __future__ import annotations

import json
import os
import shlex
from pathlib import Path
from typing import Any


def _normalize_path(value: str) -> str:
    normalized = value.replace("\\", "/").strip()
    if normalized.startswith("./"):
        normalized = normalized[2:]
    return normalized


def _parse_command_tokens(command: str) -> list[str]:
    try:
        return shlex.split(command, posix=os.name != "nt")
    except ValueError:
        return [part for part in str(command).strip().split(" ") if part]


def _command_binary(command: str) -> str:
    tokens = _parse_command_tokens(command)
    if not tokens:
        return ""
    return str(tokens[0]).strip().lower()


def _extract_node_script(command: str) -> str:
    tokens = _parse_command_tokens(command)
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


def _workspace_sort_key(item: dict[str, Any]) -> tuple[int, int, str]:
    rel = str(item.get("rel", ".")).strip() or "."
    depth = 0 if rel == "." else len(Path(rel).parts)
    return (0 if rel == "." else 1, depth, rel)


def _discover_workspaces(
    project_dir: Path, marker_names: set[str], max_depth: int = 4
) -> list[dict[str, Any]]:
    skip_dirs = {
        ".git",
        ".hg",
        ".svn",
        "node_modules",
        "dist",
        "build",
        "target",
        ".next",
        ".turbo",
        ".venv",
        "venv",
        "__pycache__",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
    }
    out: list[dict[str, Any]] = []
    root_resolved = project_dir.resolve()
    for current_root, dirnames, filenames in os.walk(project_dir):
        current_path = Path(current_root)
        rel = current_path.resolve().relative_to(root_resolved).as_posix()
        depth = 0 if rel == "." else len(Path(rel).parts)
        dirnames[:] = [name for name in dirnames if name not in skip_dirs]
        if depth > max_depth:
            dirnames[:] = []
            continue
        filename_set = {str(name).strip() for name in filenames}
        if not marker_names.intersection(filename_set):
            continue
        out.append({"rel": rel, "path": current_path})
    out.sort(key=_workspace_sort_key)
    return out


def _discover_node_workspaces(project_dir: Path) -> list[dict[str, Any]]:
    workspaces = _discover_workspaces(project_dir, {"package.json"}, max_depth=5)
    for workspace in workspaces:
        scripts: set[str] = set()
        package_json_path = Path(workspace["path"]) / "package.json"
        try:
            payload = json.loads(package_json_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            payload = {}
        scripts_payload = payload.get("scripts", {})
        if isinstance(scripts_payload, dict):
            for key in scripts_payload.keys():
                script = str(key).strip().lower()
                if script:
                    scripts.add(script)
        workspace["scripts"] = scripts
    return workspaces


def _discover_python_workspaces(project_dir: Path) -> list[dict[str, Any]]:
    return _discover_workspaces(
        project_dir,
        {"pyproject.toml", "setup.py", "setup.cfg", "requirements.txt", "pytest.ini"},
        max_depth=5,
    )


def _changed_overlap_score(workspace_rel: str, changed_files: list[str]) -> int:
    rel = str(workspace_rel).strip() or "."
    if rel == ".":
        return 0
    prefix = rel.rstrip("/") + "/"
    score = 0
    for changed in changed_files:
        normalized = _normalize_path(changed)
        if normalized.startswith(prefix):
            score += 1
    return score


def _choose_workspace(
    *,
    command: str,
    changed_files: list[str],
    workspaces: list[dict[str, Any]],
    kind: str,
) -> str:
    if not workspaces:
        return "."
    candidates = list(workspaces)
    if kind == "node":
        script = _extract_node_script(command)
        if script:
            with_script = [
                item for item in candidates if script in set(item.get("scripts", set()))
            ]
            if with_script:
                candidates = with_script

    best: dict[str, Any] | None = None
    best_score = -1
    for item in candidates:
        rel = str(item.get("rel", ".")).strip() or "."
        score = _changed_overlap_score(rel, changed_files)
        if score > best_score:
            best = item
            best_score = score
        elif (
            score == best_score
            and best is not None
            and _workspace_sort_key(item) < _workspace_sort_key(best)
        ):
            best = item
    if best is not None and best_score > 0:
        return str(best.get("rel", ".")).strip() or "."

    roots = [
        item for item in candidates if str(item.get("rel", ".")).strip() in {"", "."}
    ]
    if roots:
        return "."
    candidates.sort(key=_workspace_sort_key)
    return str(candidates[0].get("rel", ".")).strip() or "."


def _wrap_command_with_workspace(command: str, workspace_rel: str) -> str:
    rel = str(workspace_rel).strip()
    if not rel or rel == ".":
        return command
    command_text = str(command or "").strip()
    if not command_text:
        return command
    if command_text.lower().startswith("cd "):
        return command
    safe_rel = rel.replace('"', '\\"')
    if os.name == "nt":
        return f'cd /d "{safe_rel}" && {command_text}'
    return f'cd "{safe_rel}" && {command_text}'


def _apply_workspace_routing(
    *,
    commands: list[str],
    config: dict[str, Any],
    changed_files: list[str],
) -> list[str]:
    runtime_cfg = dict(config.get("runtime", {}))
    if not bool(runtime_cfg.get("verify_workspace_routing_enabled", True)):
        return list(commands)

    meta_cfg = dict(config.get("meta", {}))
    project_dir_value = str(meta_cfg.get("project_dir", "")).strip()
    if not project_dir_value:
        return list(commands)
    project_dir = Path(project_dir_value)
    if not project_dir.exists():
        return list(commands)

    node_workspaces = _discover_node_workspaces(project_dir)
    python_workspaces = _discover_python_workspaces(project_dir)
    changed_normalized = [
        _normalize_path(item) for item in changed_files if str(item).strip()
    ]

    routed: list[str] = []
    for command in commands:
        binary = _command_binary(command)
        workspace_rel = "."
        if binary in {"npm", "pnpm", "yarn"}:
            workspace_rel = _choose_workspace(
                command=command,
                changed_files=changed_normalized,
                workspaces=node_workspaces,
                kind="node",
            )
        elif binary in {"python", "pytest", "ruff", "mypy"}:
            workspace_rel = _choose_workspace(
                command=command,
                changed_files=changed_normalized,
                workspaces=python_workspaces,
                kind="python",
            )
        routed.append(_wrap_command_with_workspace(command, workspace_rel))
    return routed
