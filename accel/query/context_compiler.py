from __future__ import annotations

import hashlib
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


def _estimate_source_bytes(project_dir: Path, indexed_files: list[str], sample_limit: int = 200) -> int:
    if not indexed_files:
        return 0
    sample = indexed_files[: max(1, int(sample_limit))]
    total = 0
    counted = 0
    for rel_path in sample:
        try:
            total += int((project_dir / rel_path).stat().st_size)
            counted += 1
        except OSError:
            continue
    if counted <= 0:
        return 0
    if len(indexed_files) > counted:
        ratio = float(len(indexed_files)) / float(counted)
        return int(total * ratio)
    return total


def _snippet_fingerprint(content: str) -> str:
    normalized = " ".join(str(content).lower().split())
    return hashlib.sha1(normalized.encode("utf-8", errors="replace")).hexdigest()


def _compact_snippet_content(content: str, max_chars: int) -> tuple[str, int]:
    original = str(content or "")
    lines = [line.rstrip() for line in original.splitlines()]

    # Drop long leading banner comments/blank lines to keep more executable context.
    leading = 0
    for line in lines:
        stripped = line.strip()
        if not stripped:
            leading += 1
            continue
        if stripped.startswith(("#", "//", "/*", "*", "*/", '"""', "'''")):
            leading += 1
            continue
        break
    if leading >= 8:
        lines = lines[leading:]

    compact_lines: list[str] = []
    blank_run = 0
    import_run = 0
    suppressed_imports = 0
    for line in lines:
        stripped_line = line.strip()
        is_import_line = stripped_line.startswith(("import ", "from ")) and "(" not in stripped_line and ")" not in stripped_line
        if is_import_line:
            import_run += 1
            # Keep a short import header but avoid spending budget on large import blocks.
            if import_run > 12:
                suppressed_imports += 1
                continue
        else:
            import_run = 0

        if not line.strip():
            blank_run += 1
            if blank_run <= 1:
                compact_lines.append("")
            continue
        blank_run = 0
        compact_lines.append(line)

    if suppressed_imports > 0:
        compact_lines.append(f"# ... [omitted {suppressed_imports} import lines]")

    compact = "\n".join(compact_lines).strip("\n")
    if len(compact) > max_chars:
        marker = "\n... [truncated]"
        keep = max(0, max_chars - len(marker))
        compact = compact[:keep].rstrip() + marker

    if not compact:
        compact = original[:max_chars]
    return compact, max(0, len(original) - len(compact))


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
    effective_file_window = max(1, min(max_snippets, top_n_files))
    per_snippet_max_chars = int(
        context_cfg.get(
            "per_snippet_max_chars",
            max(800, min(6000, int((max_chars / effective_file_window) * 2))),
        )
    )

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
    raw_snippet_chars = 0
    compacted_snippet_chars = 0
    deduped_snippet_count = 0
    seen_fingerprints: set[str] = set()
    for top_item in top_files:
        file_path = top_item["path"]
        symbol_rows = symbols_by_file.get(file_path, [])
        snippet = extract_snippet(
            project_dir=project_dir,
            rel_path=file_path,
            task_tokens=task_tokens,
            symbol_rows=symbol_rows,
            snippet_radius=snippet_radius,
            max_chars=per_snippet_max_chars,
        )
        if snippet is None:
            continue
        raw_content = str(snippet.get("content", ""))
        compact_content, _ = _compact_snippet_content(raw_content, per_snippet_max_chars)
        if not compact_content.strip():
            continue
        snippet["content"] = compact_content

        fingerprint = _snippet_fingerprint(compact_content)
        if fingerprint in seen_fingerprints:
            deduped_snippet_count += 1
            continue
        seen_fingerprints.add(fingerprint)

        raw_snippet_chars += len(raw_content)
        size = len(compact_content)
        compacted_snippet_chars += size
        if current_chars + size > max_chars:
            continue
        snippets.append(snippet)
        current_chars += size
        if len(snippets) >= max_snippets:
            break

    verify_plan = _build_verify_plan(top_files, ownership_rows, changed_files)
    counts = dict(manifest.get("counts", {}))
    source_bytes = int(counts.get("source_bytes", 0))
    source_chars_est = int(counts.get("source_chars_est", 0))
    if source_bytes <= 0:
        source_bytes = _estimate_source_bytes(project_dir, list(manifest.get("indexed_files", [])))
    if source_chars_est <= 0:
        source_chars_est = source_bytes

    budget = {
        "max_chars": max_chars,
        "max_snippets": max_snippets,
        "top_n_files": top_n_files,
        "snippet_radius": snippet_radius,
        "per_snippet_max_chars": per_snippet_max_chars,
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
            "source_bytes": source_bytes,
            "source_chars_est": source_chars_est,
            "snippet_raw_chars": raw_snippet_chars,
            "snippet_chars": compacted_snippet_chars,
            "snippet_saved_chars": max(0, raw_snippet_chars - compacted_snippet_chars),
            "snippet_deduped_count": deduped_snippet_count,
            "drift_reason": "",
        },
    }
    if previous_attempt_feedback:
        pack["previous_attempt_feedback"] = previous_attempt_feedback
    return pack


def write_context_pack(output_path: Path, pack: dict[str, Any]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(pack, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
