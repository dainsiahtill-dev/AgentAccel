from __future__ import annotations

import json
from collections import deque
from pathlib import Path
from typing import Any

from ..storage.cache import project_paths
from ..storage.index_cache import load_index_rows


def _normalize_path(value: str) -> str:
    normalized = value.replace("\\", "/").strip()
    if normalized.startswith("./"):
        normalized = normalized[2:]
    return normalized


def _project_rel_path(path: str, project_dir: Path) -> str:
    candidate = Path(path)
    if candidate.is_absolute():
        try:
            return _normalize_path(str(candidate.resolve().relative_to(project_dir.resolve())))
        except ValueError:
            return _normalize_path(str(candidate))
    return _normalize_path(path)


def _build_module_aliases(indexed_files: set[str]) -> dict[str, set[str]]:
    aliases: dict[str, set[str]] = {}

    for file_path in indexed_files:
        path_obj = Path(file_path)
        stem_obj = path_obj.with_suffix("")

        slash_aliases = {stem_obj.as_posix(), path_obj.as_posix()}
        dotted_aliases = {".".join(stem_obj.parts), ".".join(path_obj.parts)}
        base_name = stem_obj.name
        slash_aliases.add(base_name)
        dotted_aliases.add(base_name)

        if stem_obj.name == "__init__" and len(stem_obj.parts) > 1:
            parent = Path(*stem_obj.parts[:-1])
            slash_aliases.add(parent.as_posix())
            dotted_aliases.add(".".join(parent.parts))

        if stem_obj.name == "index" and len(stem_obj.parts) > 1:
            parent = Path(*stem_obj.parts[:-1])
            slash_aliases.add(parent.as_posix())

        for alias in slash_aliases.union(dotted_aliases):
            alias_key = _normalize_path(alias).strip(".")
            if not alias_key:
                continue
            aliases.setdefault(alias_key, set()).add(file_path)

    return aliases


def _resolve_relative_target(edge_from: str, edge_to: str, indexed_files: set[str]) -> set[str]:
    source_dir = Path(edge_from).parent
    raw = edge_to.strip()

    if raw.startswith(".") and "/" not in raw and "\\" not in raw:
        level = len(raw) - len(raw.lstrip("."))
        remainder = raw[level:].replace(".", "/")
        base_dir = source_dir
        for _ in range(max(0, level - 1)):
            base_dir = base_dir.parent
        target_base = base_dir / remainder if remainder else base_dir
    else:
        target_base = source_dir / raw

    candidates: set[str] = set()
    if target_base.suffix:
        candidates.add(_normalize_path(target_base.as_posix()))
    else:
        for ext in (".py", ".ts", ".tsx", ".js", ".jsx"):
            candidates.add(_normalize_path(f"{target_base.as_posix()}{ext}"))
        for name in ("index", "__init__"):
            for ext in (".py", ".ts", ".tsx", ".js", ".jsx"):
                candidates.add(_normalize_path((target_base / f"{name}{ext}").as_posix()))

    return {item for item in candidates if item in indexed_files}


def _resolve_dependency_targets(
    edge_from: str,
    edge_to: str,
    indexed_files: set[str],
    module_aliases: dict[str, set[str]],
) -> set[str]:
    normalized_to = _normalize_path(edge_to)
    if not normalized_to:
        return set()

    if normalized_to in indexed_files:
        return {normalized_to}

    if normalized_to.startswith("./") or normalized_to.startswith("../") or normalized_to.startswith("."):
        return _resolve_relative_target(edge_from=edge_from, edge_to=normalized_to, indexed_files=indexed_files)

    if normalized_to.startswith("@/"):
        app_candidate = _normalize_path("src/frontend/src/" + normalized_to[2:])
        if app_candidate in module_aliases:
            return set(module_aliases[app_candidate])

    resolved: set[str] = set()
    alias_candidates = {
        normalized_to,
        normalized_to.replace(".", "/"),
        normalized_to.strip("/"),
    }
    for alias in alias_candidates:
        if alias in module_aliases:
            resolved.update(module_aliases[alias])

    return resolved


def _load_index_inputs(config: dict[str, Any]) -> tuple[set[str], list[dict[str, Any]], list[dict[str, Any]]]:
    meta = config.get("meta", {})
    runtime = config.get("runtime", {})

    project_dir_value = str(meta.get("project_dir", "")).strip()
    accel_home_value = str(runtime.get("accel_home", "")).strip()
    if not project_dir_value or not accel_home_value:
        return set(), [], []

    project_dir = Path(project_dir_value)
    accel_home = Path(accel_home_value).resolve()
    paths = project_paths(accel_home, project_dir)
    index_dir = paths["index"]

    manifest_path = index_dir / "manifest.json"
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            indexed_files = {
                _normalize_path(str(item))
                for item in list(manifest.get("indexed_files", []))
                if str(item).strip()
            }
        except json.JSONDecodeError:
            indexed_files = set()
    else:
        indexed_files = set()

    deps_rows = load_index_rows(index_dir=index_dir, kind="dependencies", key_field="edge_from")
    ownership_rows = load_index_rows(index_dir=index_dir, kind="test_ownership", key_field="owns_file")

    for row in deps_rows:
        edge_from = _normalize_path(str(row.get("edge_from", "")))
        if edge_from:
            indexed_files.add(edge_from)

    for row in ownership_rows:
        owns_file = _normalize_path(str(row.get("owns_file", "")))
        if owns_file:
            indexed_files.add(owns_file)

    return indexed_files, deps_rows, ownership_rows


def _build_reverse_dependency_graph(indexed_files: set[str], deps_rows: list[dict[str, Any]]) -> dict[str, set[str]]:
    reverse: dict[str, set[str]] = {}
    module_aliases = _build_module_aliases(indexed_files)

    for row in deps_rows:
        edge_from = _normalize_path(str(row.get("edge_from", "")))
        edge_to = str(row.get("edge_to", ""))
        if not edge_from or edge_from not in indexed_files:
            continue

        targets = _resolve_dependency_targets(
            edge_from=edge_from,
            edge_to=edge_to,
            indexed_files=indexed_files,
            module_aliases=module_aliases,
        )
        for target in targets:
            reverse.setdefault(target, set()).add(edge_from)

    return reverse


def _collect_impacted_files(changed_files: list[str], indexed_files: set[str], reverse_graph: dict[str, set[str]]) -> set[str]:
    seed = {item for item in changed_files if item in indexed_files}
    if not seed:
        return set()

    impacted = set(seed)
    queue: deque[str] = deque(seed)
    
    # Add protection against infinite loops
    max_iterations = len(indexed_files) * 2  # Generous upper bound
    iterations = 0

    while queue and iterations < max_iterations:
        current = queue.popleft()
        iterations += 1
        
        for dependent in reverse_graph.get(current, set()):
            if dependent in impacted:
                continue
            impacted.add(dependent)
            queue.append(dependent)
    
    # Log warning if we hit the iteration limit (potential cycle detection)
    if iterations >= max_iterations:
        import logging
        logger = logging.getLogger("accel_verify")
        logger.warning(f"Dependency graph traversal hit iteration limit ({max_iterations}), possible cycle detected")

    return impacted


def _collect_impacted_tests(
    impacted_files: set[str],
    ownership_rows: list[dict[str, Any]],
    max_tests: int,
) -> list[str]:
    if not impacted_files:
        return []

    selected: list[str] = []
    seen: set[str] = set()

    for row in ownership_rows:
        owns_file = _normalize_path(str(row.get("owns_file", "")))
        test_file = _normalize_path(str(row.get("test_file", "")))
        if not test_file or test_file in seen:
            continue
        if owns_file in impacted_files:
            seen.add(test_file)
            selected.append(test_file)
            if len(selected) >= max_tests:
                break

    return selected


def _with_targeted_pytests(command: str, target_tests: list[str]) -> str:
    if "pytest" not in command.lower() or not target_tests:
        return command

    pytest_targets = [item for item in target_tests if item.endswith(".py") or not Path(item).suffix]
    if not pytest_targets:
        return command

    quoted_targets = [f'"{item}"' if " " in item else item for item in pytest_targets]
    suffix = " ".join(quoted_targets)
    return f"{command} {suffix}".strip()


def _shard_targets(targets: list[str], shard_size: int, max_shards: int) -> list[list[str]]:
    if not targets:
        return []
    shard_size = max(1, int(shard_size))
    max_shards = max(1, int(max_shards))
    requested = max(1, (len(targets) + shard_size - 1) // shard_size)
    shard_count = min(max_shards, requested)
    shards: list[list[str]] = [[] for _ in range(shard_count)]
    for idx, target in enumerate(targets):
        shards[idx % shard_count].append(target)
    return [shard for shard in shards if shard]


def _with_targeted_pytests_shards(
    command: str,
    target_tests: list[str],
    shard_size: int,
    max_shards: int,
) -> list[str]:
    if "pytest" not in command.lower() or not target_tests:
        return [command]

    pytest_targets = [item for item in target_tests if item.endswith(".py") or not Path(item).suffix]
    if not pytest_targets:
        return [command]

    shards = _shard_targets(pytest_targets, shard_size=shard_size, max_shards=max_shards)
    if not shards:
        return [command]

    commands: list[str] = []
    for shard in shards:
        quoted_targets = [f'"{item}"' if " " in item else item for item in shard]
        suffix = " ".join(quoted_targets)
        commands.append(f"{command} {suffix}".strip())
    return commands


def select_verify_commands(
    config: dict[str, Any],
    changed_files: list[str] | None = None,
) -> list[str]:
    raw_changed_files = [item for item in (changed_files or [])]
    changed_files_low = [item.lower() for item in raw_changed_files]
    verify_cfg = config.get("verify", {})
    python_cmds = list(verify_cfg.get("python", []))
    node_cmds = list(verify_cfg.get("node", []))

    has_py = any(item.endswith(".py") for item in changed_files_low)
    has_js = any(item.endswith((".ts", ".tsx", ".js", ".jsx")) for item in changed_files_low)
    run_all = len(changed_files_low) == 0

    commands: list[str] = []

    if run_all or has_py:
        targeted_python_cmds = list(python_cmds)
        if has_py and not run_all:
            indexed_files, deps_rows, ownership_rows = _load_index_inputs(config)
            meta = config.get("meta", {})
            project_dir_value = str(meta.get("project_dir", "")).strip()
            if indexed_files and project_dir_value:
                project_dir = Path(project_dir_value)
                normalized_changed = [_project_rel_path(item, project_dir) for item in raw_changed_files]
                reverse_graph = _build_reverse_dependency_graph(indexed_files=indexed_files, deps_rows=deps_rows)
                impacted_files = _collect_impacted_files(
                    changed_files=normalized_changed,
                    indexed_files=indexed_files,
                    reverse_graph=reverse_graph,
                )
                runtime_cfg = config.get("runtime", {})
                max_tests = max(1, int(runtime_cfg.get("verify_max_target_tests", 64)))
                shard_size = max(1, int(runtime_cfg.get("verify_pytest_shard_size", 16)))
                max_shards = max(1, int(runtime_cfg.get("verify_pytest_max_shards", 6)))
                targeted_tests = _collect_impacted_tests(
                    impacted_files=impacted_files,
                    ownership_rows=ownership_rows,
                    max_tests=max_tests,
                )
                if targeted_tests:
                    expanded_python_cmds: list[str] = []
                    for command in targeted_python_cmds:
                        expanded_python_cmds.extend(
                            _with_targeted_pytests_shards(
                                command=command,
                                target_tests=targeted_tests,
                                shard_size=shard_size,
                                max_shards=max_shards,
                            )
                        )
                    targeted_python_cmds = expanded_python_cmds

        commands.extend(targeted_python_cmds)

    if run_all or has_js:
        commands.extend(node_cmds)

    return commands
