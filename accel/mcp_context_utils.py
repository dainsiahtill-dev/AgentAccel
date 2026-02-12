from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

JSONDict = dict[str, Any]


def _clamp_float(value: float, min_value: float, max_value: float) -> float:
    return max(float(min_value), min(float(max_value), float(value)))


def _token_reduction_ratio(context_tokens: int, baseline_tokens: int) -> float:
    baseline = int(baseline_tokens)
    if baseline <= 0:
        return 0.0
    ratio = 1.0 - (float(max(0, int(context_tokens))) / float(baseline))
    # Keep regression visible: when context_tokens > baseline_tokens, ratio becomes negative.
    return min(1.0, float(ratio))


def _estimate_changed_files_chars(
    project_dir: Path,
    changed_files: list[str],
    *,
    max_files: int = 200,
    max_total_chars: int = 2_000_000,
) -> int:
    total_chars = 0
    seen: set[str] = set()
    for rel_path in changed_files[: max(1, int(max_files))]:
        normalized = str(rel_path).replace("\\", "/").strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        file_path = (project_dir / normalized).resolve()
        if not file_path.exists() or not file_path.is_file():
            continue
        try:
            content = file_path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        total_chars += len(content)
        if total_chars >= max_total_chars:
            return int(max_total_chars)
    return int(total_chars)


def _build_context_output_pack(
    pack: JSONDict,
    *,
    snippets_only: bool,
    include_metadata: bool,
) -> JSONDict:
    if snippets_only:
        output: JSONDict = {
            "version": int(pack.get("version", 1) or 1),
            "task": str(pack.get("task", "")),
            "generated_at": str(pack.get("generated_at", "")),
            "snippets": list(pack.get("snippets", [])),
        }
        if include_metadata and isinstance(pack.get("meta"), dict):
            output["meta"] = dict(pack.get("meta", {}))
        return output

    output = dict(pack)
    if not include_metadata:
        output.pop("meta", None)
    return output


def _normalize_rel_path(value: Any) -> str:
    return str(value or "").replace("\\", "/").strip().lower()


def _apply_strict_changed_files_scope(
    pack: JSONDict,
    changed_files: list[str],
) -> tuple[JSONDict, int, int]:
    changed_set = {
        _normalize_rel_path(item) for item in changed_files if _normalize_rel_path(item)
    }
    if not changed_set:
        return dict(pack), 0, 0

    payload = dict(pack)
    filtered_top_files = 0
    filtered_snippets = 0

    top_files_raw = payload.get("top_files")
    if isinstance(top_files_raw, list):
        kept_top_files: list[JSONDict] = []
        for item in top_files_raw:
            if isinstance(item, dict):
                path_token = _normalize_rel_path(item.get("path", ""))
                if path_token and path_token in changed_set:
                    kept_top_files.append(dict(item))
                else:
                    filtered_top_files += 1
            else:
                filtered_top_files += 1
        payload["top_files"] = kept_top_files

    snippets_raw = payload.get("snippets")
    if isinstance(snippets_raw, list):
        kept_snippets: list[JSONDict] = []
        for item in snippets_raw:
            if isinstance(item, dict):
                path_token = _normalize_rel_path(item.get("path", ""))
                if path_token and path_token in changed_set:
                    kept_snippets.append(dict(item))
                else:
                    filtered_snippets += 1
            else:
                filtered_snippets += 1
        payload["snippets"] = kept_snippets

    meta_raw = payload.get("meta")
    if isinstance(meta_raw, dict):
        meta_payload = dict(meta_raw)
        meta_payload["strict_changed_files_scope"] = True
        meta_payload["strict_scope_changed_files_count"] = int(len(changed_set))
        meta_payload["strict_scope_filtered_top_files"] = int(filtered_top_files)
        meta_payload["strict_scope_filtered_snippets"] = int(filtered_snippets)
        payload["meta"] = meta_payload

    return payload, int(filtered_top_files), int(filtered_snippets)


def _resolve_changed_file_rel_path(project_dir: Path, value: Any) -> str | None:
    token = str(value or "").replace("\\", "/").strip()
    if not token:
        return None
    raw_path = Path(token)
    candidate = raw_path if raw_path.is_absolute() else (project_dir / raw_path)
    try:
        resolved = candidate.resolve()
        project_root = project_dir.resolve()
        rel = resolved.relative_to(project_root)
    except (OSError, ValueError):
        return None
    if not resolved.exists() or not resolved.is_file():
        return None
    return str(rel).replace("\\", "/")


def _build_changed_file_snippet(
    project_dir: Path, rel_path: str, max_chars: int
) -> JSONDict | None:
    rel = str(rel_path or "").replace("\\", "/").strip()
    if not rel:
        return None
    file_path = (project_dir / rel).resolve()
    try:
        content = file_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    clipped = content[: max(200, int(max_chars))]
    if not clipped:
        clipped = "[empty file]"
    end_line = max(1, int(clipped.count("\n")) + 1)
    return {
        "path": rel,
        "start_line": 1,
        "end_line": end_line,
        "symbol": "",
        "reason": "strict_changed_file_fallback",
        "content": clipped,
    }


def _ensure_strict_changed_files_presence(
    pack: JSONDict,
    *,
    project_dir: Path,
    changed_files: list[str],
) -> tuple[JSONDict, int, int]:
    payload = dict(pack)
    budget = (
        payload.get("budget", {}) if isinstance(payload.get("budget", {}), dict) else {}
    )
    top_n_files = max(1, int(budget.get("top_n_files", 8) or 8))
    max_snippets = max(1, int(budget.get("max_snippets", 30) or 30))
    per_snippet_max_chars = max(
        200, int(budget.get("per_snippet_max_chars", 2000) or 2000)
    )

    existing_rel_paths: list[str] = []
    seen: set[str] = set()
    for item in changed_files:
        rel = _resolve_changed_file_rel_path(project_dir, item)
        if not rel:
            continue
        key = _normalize_rel_path(rel)
        if key in seen:
            continue
        seen.add(key)
        existing_rel_paths.append(rel)

    if not existing_rel_paths:
        return payload, 0, 0

    top_files = payload.get("top_files")
    snippets = payload.get("snippets")
    top_files_list = list(top_files) if isinstance(top_files, list) else []
    snippets_list = list(snippets) if isinstance(snippets, list) else []

    injected_top_files = 0
    injected_snippets = 0

    if not top_files_list:
        for rel in existing_rel_paths[:top_n_files]:
            top_files_list.append(
                {
                    "path": rel,
                    "score": 1.0,
                    "reasons": ["strict_changed_file_fallback"],
                    "signals": [
                        {"signal_name": "strict_changed_file_fallback", "score": 1.0}
                    ],
                }
            )
            injected_top_files += 1
        payload["top_files"] = top_files_list

    if not snippets_list:
        for rel in existing_rel_paths[:max_snippets]:
            snippet = _build_changed_file_snippet(
                project_dir, rel, per_snippet_max_chars
            )
            if not isinstance(snippet, dict):
                continue
            snippets_list.append(snippet)
            injected_snippets += 1
        payload["snippets"] = snippets_list

    meta_raw = payload.get("meta")
    if isinstance(meta_raw, dict):
        meta_payload = dict(meta_raw)
        meta_payload["strict_scope_injected_top_files"] = int(injected_top_files)
        meta_payload["strict_scope_injected_snippets"] = int(injected_snippets)
        payload["meta"] = meta_payload

    return payload, int(injected_top_files), int(injected_snippets)


def _write_context_metadata_sidecar(out_path: Path, payload: JSONDict) -> Path:
    sidecar_path = out_path.with_suffix(".meta.json")
    token_reduction_payload = payload.get("token_reduction", {})
    token_estimator_payload = payload.get("token_estimator", {})
    sidecar_payload: JSONDict = {
        "version": 1,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "out": str(out_path),
        "output_mode": payload.get("output_mode", "full"),
        "include_metadata": bool(payload.get("include_metadata", True)),
        "budget": {
            "source": payload.get("budget_source", "auto"),
            "preset": payload.get("budget_preset", "small"),
            "reason": payload.get("budget_reason", ""),
            "effective": payload.get("budget_effective", {}),
        },
        "scope": {
            "changed_files_source": payload.get("changed_files_source", "none"),
            "changed_files_count": int(payload.get("changed_files_count", 0) or 0),
            "changed_files_used": list(payload.get("changed_files_used", [])),
            "changed_files_detection": (
                dict(payload.get("changed_files_detection", {}))
                if isinstance(payload.get("changed_files_detection"), dict)
                else {}
            ),
            "fallback_confidence": float(
                payload.get("fallback_confidence", 0.0) or 0.0
            ),
            "strict_changed_files": bool(payload.get("strict_changed_files", False)),
            "strict_scope_filtered_top_files": int(
                payload.get("strict_scope_filtered_top_files", 0) or 0
            ),
            "strict_scope_filtered_snippets": int(
                payload.get("strict_scope_filtered_snippets", 0) or 0
            ),
            "strict_scope_injected_top_files": int(
                payload.get("strict_scope_injected_top_files", 0) or 0
            ),
            "strict_scope_injected_snippets": int(
                payload.get("strict_scope_injected_snippets", 0) or 0
            ),
        },
        "estimates": {
            "context_chars": int(payload.get("context_chars", 0) or 0),
            "source_chars": int(payload.get("source_chars", 0) or 0),
            "estimated_tokens": int(payload.get("estimated_tokens", 0) or 0),
            "estimated_source_tokens": int(
                payload.get("estimated_source_tokens", 0) or 0
            ),
            "estimated_changed_files_tokens": int(
                payload.get("estimated_changed_files_tokens", 0) or 0
            ),
            "estimated_snippets_only_tokens": int(
                payload.get("estimated_snippets_only_tokens", 0) or 0
            ),
            "compression_ratio": float(payload.get("compression_ratio", 1.0) or 1.0),
            "token_reduction_ratio": float(
                payload.get("token_reduction_ratio", 0.0) or 0.0
            ),
            "token_reduction": token_reduction_payload
            if isinstance(token_reduction_payload, dict)
            else {},
            "token_estimator": token_estimator_payload
            if isinstance(token_estimator_payload, dict)
            else {},
        },
        "selected_tests_count": int(payload.get("selected_tests_count", 0) or 0),
        "selected_checks_count": int(payload.get("selected_checks_count", 0) or 0),
        "semantic_cache": {
            "enabled": bool(payload.get("semantic_cache_enabled", False)),
            "hit": bool(payload.get("semantic_cache_hit", False)),
            "mode": str(payload.get("semantic_cache_mode_used", "off")),
            "similarity": float(payload.get("semantic_cache_similarity", 0.0) or 0.0),
            "reason": str(payload.get("semantic_cache_reason", "")),
            "invalidation_reason": str(
                payload.get("semantic_cache_invalidation_reason", "")
            ),
            "safety_fingerprint": str(
                payload.get("semantic_cache_safety_fingerprint", "")
            ),
            "safety": dict(payload.get("semantic_cache_safety", {}))
            if isinstance(payload.get("semantic_cache_safety"), dict)
            else {},
        },
        "compression": {
            "rules_applied": dict(payload.get("compression_rules_applied", {})),
            "saved_chars": int(payload.get("compression_saved_chars", 0) or 0),
        },
        "constraints": {
            "mode": str(payload.get("constraint_mode", "warn")),
            "repair_count": int(payload.get("constraint_repair_count", 0) or 0),
            "warnings": list(payload.get("constraint_warnings", [])),
        },
        "warnings": list(payload.get("warnings", [])),
    }
    sidecar_path.write_text(
        json.dumps(sidecar_payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return sidecar_path
