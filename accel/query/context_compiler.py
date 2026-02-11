from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .planner import build_candidate_files, normalize_task_tokens
from .ranker import score_file
from .snippet_extractor import extract_snippet
from ..storage.cache import project_paths
from ..storage.index_cache import load_index_rows


def _group_by_file(rows: list[dict[str, Any]], key: str) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        file_value = str(row.get(key, ""))
        if not file_value:
            continue
        grouped.setdefault(file_value, []).append(row)
    return grouped


def _build_verify_plan(
    top_files: list[dict[str, Any]],
    test_ownership_rows: list[dict[str, Any]],
    changed_files: list[str],
) -> dict[str, list[str]]:
    selected_files = {item["path"] for item in top_files}
    target_tests: list[str] = []
    seen_tests: set[str] = set()
    for row in test_ownership_rows:
        owns = str(row.get("owns_file", ""))
        test_file = str(row.get("test_file", ""))
        if owns in selected_files and test_file and test_file not in seen_tests:
            seen_tests.add(test_file)
            target_tests.append(test_file)

    changed = [item.lower() for item in changed_files]
    checks: list[str] = []
    if not changed or any(item.endswith(".py") for item in changed):
        checks.append("pytest -q")
        checks.append("mypy .")
    if not changed or any(item.endswith((".ts", ".tsx", ".js", ".jsx")) for item in changed):
        checks.append("npm run typecheck")
    return {"target_tests": target_tests[:20], "target_checks": checks}


def compile_context_pack(
    project_dir: Path,
    config: dict[str, Any],
    task: str,
    changed_files: list[str] | None = None,
    hints: list[str] | None = None,
    previous_attempt_feedback: dict[str, Any] | None = None,
    budget_override: dict[str, int] | None = None,
) -> dict[str, Any]:
    changed_files = [item.replace("\\", "/") for item in (changed_files or [])]
    accel_home = Path(config["runtime"]["accel_home"]).resolve()
    paths = project_paths(accel_home, project_dir)
    index_dir = paths["index"]
    manifest_path = index_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError("Index manifest is missing. Run `accel index build` first.")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    symbols_rows = load_index_rows(index_dir=index_dir, kind="symbols", key_field="file")
    references_rows = load_index_rows(index_dir=index_dir, kind="references", key_field="file")
    deps_rows = load_index_rows(index_dir=index_dir, kind="dependencies", key_field="edge_from")
    ownership_rows = load_index_rows(index_dir=index_dir, kind="test_ownership", key_field="owns_file")

    symbols_by_file = _group_by_file(symbols_rows, "file")
    references_by_file = _group_by_file(references_rows, "file")
    deps_by_file = _group_by_file(deps_rows, "edge_from")
    tests_by_file = _group_by_file(ownership_rows, "owns_file")

    context_cfg = dict(config.get("context", {}))
    if budget_override:
        context_cfg.update(budget_override)

    top_n_files = int(context_cfg.get("top_n_files", 12))
    snippet_radius = int(context_cfg.get("snippet_radius", 40))
    max_chars = int(context_cfg.get("max_chars", 24000))
    max_snippets = int(context_cfg.get("max_snippets", 60))

    task_tokens = normalize_task_tokens(task)
    candidate_files = build_candidate_files(
        list(manifest.get("indexed_files", [])),
        changed_files=changed_files,
    )
    changed_set = set(changed_files)

    ranked = [
        score_file(
            file_path=file_path,
            task_tokens=task_tokens,
            symbols_by_file=symbols_by_file,
            references_by_file=references_by_file,
            deps_by_file=deps_by_file,
            tests_by_file=tests_by_file,
            changed_file_set=changed_set,
        )
        for file_path in candidate_files
    ]
    ranked.sort(key=lambda item: (-float(item["score"]), str(item["path"])))

    if hints:
        hints_low = [hint.lower() for hint in hints]
        for item in ranked:
            path_low = item["path"].lower()
            if any(hint in path_low for hint in hints_low):
                item["score"] = round(float(item["score"]) + 0.05, 6)
                item["reasons"] = list(dict.fromkeys(item["reasons"] + ["hint_match"]))
        ranked.sort(key=lambda item: (-float(item["score"]), str(item["path"])))

    top_files = ranked[:top_n_files]

    snippets: list[dict[str, Any]] = []
    current_chars = 0
    for top_item in top_files:
        file_path = top_item["path"]
        symbol_rows = symbols_by_file.get(file_path, [])
        snippet = extract_snippet(
            project_dir=project_dir,
            rel_path=file_path,
            task_tokens=task_tokens,
            symbol_rows=symbol_rows,
            snippet_radius=snippet_radius,
            max_chars=max_chars,
        )
        if snippet is None:
            continue
        size = len(snippet["content"])
        if current_chars + size > max_chars:
            continue
        snippets.append(snippet)
        current_chars += size
        if len(snippets) >= max_snippets:
            break

    verify_plan = _build_verify_plan(top_files, ownership_rows, changed_files)
    budget = {
        "max_chars": max_chars,
        "max_snippets": max_snippets,
        "top_n_files": top_n_files,
    }

    pack: dict[str, Any] = {
        "version": 1,
        "task": task,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "budget": budget,
        "top_files": top_files,
        "snippets": snippets,
        "verify_plan": verify_plan,
        "meta": {
            "task_tokens": task_tokens,
            "changed_files": changed_files,
            "drift_reason": "",
        },
    }
    if previous_attempt_feedback:
        pack["previous_attempt_feedback"] = previous_attempt_feedback
    return pack


def write_context_pack(output_path: Path, pack: dict[str, Any]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(pack, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
