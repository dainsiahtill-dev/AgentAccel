from __future__ import annotations

from typing import Any


def _contains_any(text: str, tokens: list[str]) -> int:
    low = text.lower()
    return sum(1 for token in tokens if token in low)


def score_file(
    file_path: str,
    task_tokens: list[str],
    symbols_by_file: dict[str, list[dict[str, Any]]],
    references_by_file: dict[str, list[dict[str, Any]]],
    deps_by_file: dict[str, list[dict[str, Any]]],
    tests_by_file: dict[str, list[dict[str, Any]]],
    changed_file_set: set[str],
) -> dict[str, Any]:
    symbol_rows = symbols_by_file.get(file_path, [])
    ref_rows = references_by_file.get(file_path, [])
    dep_rows = deps_by_file.get(file_path, [])
    test_rows = tests_by_file.get(file_path, [])

    symbol_hits = 0
    for row in symbol_rows:
        symbol_hits += _contains_any(str(row.get("symbol", "")), task_tokens)
        symbol_hits += _contains_any(str(row.get("qualified_name", "")), task_tokens)

    ref_hits = 0
    for row in ref_rows:
        ref_hits += _contains_any(str(row.get("target_symbol", "")), task_tokens)

    dep_hits = 0
    for row in dep_rows:
        dep_hits += _contains_any(str(row.get("edge_to", "")), task_tokens)

    test_hits = len(test_rows)

    symbol_match = min(1.0, symbol_hits / max(1, len(task_tokens)))
    reference_proximity = min(1.0, ref_hits / max(1, len(task_tokens)))
    dependency_impact = min(1.0, dep_hits / max(1, len(task_tokens)))
    test_relevance = min(1.0, test_hits / 3.0)

    changed_boost = 0.15 if file_path in changed_file_set else 0.0
    score = (
        0.35 * symbol_match
        + 0.25 * reference_proximity
        + 0.20 * dependency_impact
        + 0.20 * test_relevance
        + changed_boost
    )

    reasons: list[str] = []
    if symbol_match > 0:
        reasons.append("symbol_match")
    if reference_proximity > 0:
        reasons.append("reference_proximity")
    if dependency_impact > 0:
        reasons.append("dependency_impact")
    if test_relevance > 0:
        reasons.append("test_relevance")
    if file_path in changed_file_set:
        reasons.append("changed_file")
    if not reasons:
        reasons.append("baseline")

    return {
        "path": file_path,
        "score": round(score, 6),
        "reasons": reasons,
        "signals": [
            {"signal_name": "symbol_match", "score": round(symbol_match, 6)},
            {"signal_name": "reference_proximity", "score": round(reference_proximity, 6)},
            {"signal_name": "dependency_impact", "score": round(dependency_impact, 6)},
            {"signal_name": "test_relevance", "score": round(test_relevance, 6)},
            {"signal_name": "changed_boost", "score": round(changed_boost, 6)},
        ],
    }
