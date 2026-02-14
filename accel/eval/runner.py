from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .metrics import aggregate_case_metrics, recall_at_k, reciprocal_rank, symbol_hit_rate


def load_benchmark_suite(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return {"name": path.stem, "cases": payload}
    if isinstance(payload, dict):
        cases = payload.get("cases", [])
        if isinstance(cases, list):
            suite = dict(payload)
            suite["cases"] = cases
            suite.setdefault("name", path.stem)
            return suite
    raise ValueError("Benchmark suite must be an object with cases[] or a list of cases")


def _normalize_case(case: dict[str, Any], case_index: int) -> dict[str, Any]:
    case_id = str(case.get("id", "")).strip() or f"case_{case_index:03d}"
    task = str(case.get("task", "")).strip()
    if not task:
        raise ValueError(f"{case_id}: task is required")
    changed_files = case.get("changed_files", [])
    expected_files = case.get("expected_files", [])
    expected_symbols = case.get("expected_symbols", [])
    hints = case.get("hints", [])
    if not isinstance(changed_files, list):
        changed_files = []
    if not isinstance(expected_files, list):
        expected_files = []
    if not isinstance(expected_symbols, list):
        expected_symbols = []
    if not isinstance(hints, list):
        hints = []
    return {
        "id": case_id,
        "task": task,
        "changed_files": [str(item) for item in changed_files if str(item).strip()],
        "expected_files": [str(item) for item in expected_files if str(item).strip()],
        "expected_symbols": [
            str(item) for item in expected_symbols if str(item).strip()
        ],
        "hints": [str(item) for item in hints if str(item).strip()],
    }


def _collect_observed_symbols(snippets: list[dict[str, Any]]) -> list[str]:
    symbols: list[str] = []
    for snippet in snippets:
        if not isinstance(snippet, dict):
            continue
        symbol = str(snippet.get("symbol", "")).strip()
        if symbol:
            symbols.append(symbol)
        signature = str(snippet.get("signature", "")).strip()
        if signature:
            symbols.append(signature)
    return symbols


def run_benchmark_suite(
    *,
    project_dir: Path,
    config: dict[str, Any],
    suite: dict[str, Any],
    case_limit: int | None = None,
) -> dict[str, Any]:
    from ..query.context_compiler import compile_context_pack, explain_context_selection

    raw_cases = suite.get("cases", [])
    if not isinstance(raw_cases, list):
        raise ValueError("suite.cases must be a list")
    if case_limit is not None:
        raw_cases = raw_cases[: max(0, int(case_limit))]

    case_outputs: list[dict[str, Any]] = []
    metric_rows: list[dict[str, float]] = []

    for idx, raw_case in enumerate(raw_cases, start=1):
        if not isinstance(raw_case, dict):
            continue
        case = _normalize_case(raw_case, idx)
        start = time.perf_counter()
        error = ""
        selection: dict[str, Any] | None = None
        pack: dict[str, Any] | None = None
        try:
            selection = explain_context_selection(
                project_dir=project_dir,
                config=config,
                task=case["task"],
                changed_files=case["changed_files"],
                hints=case["hints"],
            )
            pack = compile_context_pack(
                project_dir=project_dir,
                config=config,
                task=case["task"],
                changed_files=case["changed_files"],
                hints=case["hints"],
            )
        except Exception as exc:  # pragma: no cover - defensive runtime guard
            error = f"{exc.__class__.__name__}: {exc}"
        elapsed_ms = round((time.perf_counter() - start) * 1000.0, 3)

        predicted_files: list[str] = []
        snippets: list[dict[str, Any]] = []
        if selection and isinstance(selection.get("selected"), list):
            predicted_files = [
                str(item.get("path", "")).strip()
                for item in selection["selected"]
                if isinstance(item, dict) and str(item.get("path", "")).strip()
            ]
        if pack and isinstance(pack.get("snippets"), list):
            snippets = [item for item in pack["snippets"] if isinstance(item, dict)]
        context_chars = int(sum(len(str(item.get("content", ""))) for item in snippets))
        observed_symbols = _collect_observed_symbols(snippets)
        metrics = {
            "recall_at_5": recall_at_k(case["expected_files"], predicted_files, 5),
            "recall_at_10": recall_at_k(case["expected_files"], predicted_files, 10),
            "mrr": reciprocal_rank(case["expected_files"], predicted_files),
            "symbol_hit_rate": symbol_hit_rate(case["expected_symbols"], observed_symbols),
            "context_chars": float(context_chars),
            "latency_ms": float(elapsed_ms),
        }
        metric_rows.append(metrics)
        case_outputs.append(
            {
                "id": case["id"],
                "task": case["task"],
                "changed_files": case["changed_files"],
                "expected_files": case["expected_files"],
                "predicted_files": predicted_files,
                "metrics": metrics,
                "context_chars": context_chars,
                "latency_ms": elapsed_ms,
                "error": error,
            }
        )

    summary = aggregate_case_metrics(metric_rows)
    return {
        "suite_name": str(suite.get("name", "benchmark_suite")),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "project_dir": str(project_dir),
        "case_count": int(len(case_outputs)),
        "summary": summary,
        "cases": case_outputs,
    }
