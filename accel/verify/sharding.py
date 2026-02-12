from __future__ import annotations

import json
import os
import shlex
from collections import deque
from pathlib import Path
from typing import Any

from ..storage.cache import project_paths
from ..storage.index_cache import load_index_rows
from ..storage.semantic_cache import SemanticCacheStore, make_stable_hash, normalize_changed_files


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
    dotted = normalized_to.replace("/", ".").strip(".")
    if dotted:
        parts = dotted.split(".")
        # Some dependency indexers emit symbol-level targets (e.g. src.auth.login).
        # Walk up segments to recover module-level aliases (src.auth).
        for idx in range(len(parts), 0, -1):
            candidate_dotted = ".".join(parts[:idx]).strip(".")
            candidate_slash = candidate_dotted.replace(".", "/")
            alias_candidates.add(candidate_dotted)
            alias_candidates.add(candidate_slash)
    for alias in alias_candidates:
        if alias in module_aliases:
            resolved.update(module_aliases[alias])

    return resolved


def _runtime_fingerprint(runtime_cfg: dict[str, Any]) -> str:
    payload = {
        "verify_max_target_tests": int(runtime_cfg.get("verify_max_target_tests", 64)),
        "verify_pytest_shard_size": int(runtime_cfg.get("verify_pytest_shard_size", 16)),
        "verify_pytest_max_shards": int(runtime_cfg.get("verify_pytest_max_shards", 6)),
        "verify_fail_fast": bool(runtime_cfg.get("verify_fail_fast", False)),
        "verify_workspace_routing_enabled": bool(runtime_cfg.get("verify_workspace_routing_enabled", True)),
    }
    return make_stable_hash(payload)


def _verify_config_hash(config: dict[str, Any]) -> str:
    payload = {
        "verify": config.get("verify", {}),
        "index": config.get("index", {}),
    }
    return make_stable_hash(payload)


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


def _collect_textual_dependents(project_dir: Path, changed_files: list[str], indexed_files: set[str]) -> set[str]:
    module_tokens: set[str] = set()
    for changed in changed_files:
        normalized = _normalize_path(changed)
        stem = Path(normalized).with_suffix("").as_posix()
        if stem:
            module_tokens.add(stem.replace("/", "."))
            module_tokens.add(stem.split("/")[-1])
        basename = Path(normalized).stem
        if basename:
            module_tokens.add(basename)

    if not module_tokens:
        return set()

    dependents: set[str] = set()
    for rel_path in indexed_files:
        if rel_path in changed_files:
            continue
        candidate = project_dir / rel_path
        if not candidate.exists() or not candidate.is_file():
            continue
        if candidate.suffix.lower() not in {".py", ".ts", ".tsx", ".js", ".jsx"}:
            continue
        try:
            if candidate.stat().st_size > 1_000_000:
                continue
            text = candidate.read_text(encoding="utf-8", errors="replace").lower()
        except OSError:
            continue
        if any(token and token.lower() in text for token in module_tokens):
            dependents.add(rel_path)
    return dependents


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


def _discover_workspaces(project_dir: Path, marker_names: set[str], max_depth: int = 4) -> list[dict[str, Any]]:
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
            with_script = [item for item in candidates if script in set(item.get("scripts", set()))]
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
        elif score == best_score and best is not None and _workspace_sort_key(item) < _workspace_sort_key(best):
            best = item
    if best is not None and best_score > 0:
        return str(best.get("rel", ".")).strip() or "."

    roots = [item for item in candidates if str(item.get("rel", ".")).strip() in {"", "."}]
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
    changed_normalized = [_normalize_path(item) for item in changed_files if str(item).strip()]

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

    runtime_cfg = dict(config.get("runtime", {}))
    meta_cfg = dict(config.get("meta", {}))
    project_dir_text = str(meta_cfg.get("project_dir", "")).strip()
    accel_home_text = str(runtime_cfg.get("accel_home", "")).strip()
    plan_cache_enabled = bool(runtime_cfg.get("command_plan_cache_enabled", True))
    plan_cache_ttl = max(1, int(runtime_cfg.get("command_plan_cache_ttl_seconds", 900)))
    plan_cache_max_entries = max(1, int(runtime_cfg.get("command_plan_cache_max_entries", 600)))
    cache_store: SemanticCacheStore | None = None
    cache_key = ""
    if plan_cache_enabled and project_dir_text and accel_home_text:
        changed_norm = normalize_changed_files(raw_changed_files)
        changed_fingerprint = make_stable_hash({"changed_files": changed_norm})
        runtime_hash = _runtime_fingerprint(runtime_cfg)
        config_hash = _verify_config_hash(config)
        cache_key = make_stable_hash(
            {
                "changed_fingerprint": changed_fingerprint,
                "runtime_hash": runtime_hash,
                "config_hash": config_hash,
            }
        )
        try:
            project_dir = Path(project_dir_text)
            accel_home = Path(accel_home_text).resolve()
            paths = project_paths(accel_home, project_dir)
            cache_store = SemanticCacheStore(paths["state"] / "semantic_cache.db")
            cached_commands = cache_store.get_verify_plan(cache_key)
            if isinstance(cached_commands, list):
                return cached_commands
        except OSError:
            cache_store = None

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
                if impacted_files and len(impacted_files) <= len(set(normalized_changed)):
                    textual_dependents = _collect_textual_dependents(
                        project_dir=project_dir,
                        changed_files=normalized_changed,
                        indexed_files=indexed_files,
                    )
                    impacted_files.update(textual_dependents)
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

    commands = _apply_workspace_routing(
        commands=commands,
        config=config,
        changed_files=raw_changed_files,
    )

    if cache_store is not None and cache_key:
        changed_norm = normalize_changed_files(raw_changed_files)
        cache_store.put_verify_plan(
            cache_key=cache_key,
            changed_fingerprint=make_stable_hash({"changed_files": changed_norm}),
            runtime_fingerprint=_runtime_fingerprint(runtime_cfg),
            config_hash=_verify_config_hash(config),
            commands=commands,
            ttl_seconds=plan_cache_ttl,
            max_entries=plan_cache_max_entries,
        )

    return commands
