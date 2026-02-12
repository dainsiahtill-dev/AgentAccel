#!/usr/bin/env python3
"""Run reproducible benchmark tasks for agent-accel context generation."""

from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from accel.config import resolve_effective_config
    from accel.indexers import build_or_update_indexes
    from accel.query.context_compiler import compile_context_pack
    from accel.storage.cache import ensure_project_dirs, project_paths
    from accel.token_estimator import (
        estimate_tokens_for_text,
        estimate_tokens_from_chars,
    )
    from accel.verify.orchestrator import run_verify
except Exception as exc:  # pragma: no cover - environment bootstrap guard
    raise RuntimeError(
        "run_benchmarks.py must be executed from an environment where 'accel' is importable"
    ) from exc


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _normalize_path(path: Path) -> Path:
    return Path(os.path.abspath(str(path)))


def _run_cmd(command: list[str], cwd: Path) -> tuple[int, str]:
    try:
        proc = subprocess.run(
            command,
            cwd=str(cwd),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=20,
            check=False,
        )
        out = (proc.stdout or "").strip()
        if not out:
            out = (proc.stderr or "").strip()
        return int(proc.returncode), out
    except Exception as exc:  # pragma: no cover - defensive guard
        return 1, str(exc)


def _load_tasks(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("tasks file must be a JSON array")
    rows: list[dict[str, Any]] = []
    for item in payload:
        if not isinstance(item, dict):
            continue
        task_id = str(item.get("id", "")).strip()
        task_text = str(item.get("task", "")).strip()
        if not task_id or not task_text:
            continue
        changed_files_raw = item.get("changed_files", [])
        hints_raw = item.get("hints", [])
        expected_raw = item.get("expected_top_files", [])
        changed_files = [
            str(entry).replace("\\", "/").strip()
            for entry in changed_files_raw
            if str(entry).strip()
        ]
        hints = [str(entry).strip() for entry in hints_raw if str(entry).strip()]
        expected_top_files = [
            str(entry).replace("\\", "/").strip()
            for entry in expected_raw
            if str(entry).strip()
        ]
        rows.append(
            {
                "id": task_id,
                "task": task_text,
                "changed_files": changed_files,
                "hints": hints,
                "expected_top_files": expected_top_files,
            }
        )
    if not rows:
        raise ValueError("no valid benchmark tasks found")
    return rows


def _sum_changed_chars(project_dir: Path, changed_files: list[str]) -> int:
    total = 0
    seen: set[str] = set()
    for rel in changed_files:
        normalized = str(rel).replace("\\", "/").strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        file_path = (project_dir / normalized).resolve()
        if not file_path.exists() or not file_path.is_file():
            continue
        try:
            total += len(file_path.read_text(encoding="utf-8", errors="replace"))
        except OSError:
            continue
    return int(total)


def _load_manifest(index_dir: Path) -> dict[str, Any]:
    manifest_path = index_dir / "manifest.json"
    if not manifest_path.exists():
        return {}
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def _estimate_full_source_chars(project_dir: Path, manifest: dict[str, Any]) -> int:
    counts = manifest.get("counts", {}) if isinstance(manifest.get("counts"), dict) else {}
    source_chars_est = int(counts.get("source_chars_est", 0) or 0)
    if source_chars_est > 0:
        return source_chars_est

    indexed_files = manifest.get("indexed_files", [])
    if not isinstance(indexed_files, list):
        return 0
    total = 0
    for rel in indexed_files:
        rel_text = str(rel).replace("\\", "/").strip()
        if not rel_text:
            continue
        path = (project_dir / rel_text).resolve()
        if not path.exists() or not path.is_file():
            continue
        try:
            total += len(path.read_text(encoding="utf-8", errors="replace"))
        except OSError:
            continue
    return int(total)


def _top_file_recall(top_files: list[str], expected: list[str]) -> float | None:
    expected_set = {
        str(item).replace("\\", "/").strip().lower()
        for item in expected
        if str(item).strip()
    }
    if not expected_set:
        return None
    top_set = {
        str(item).replace("\\", "/").strip().lower() for item in top_files if str(item).strip()
    }
    matched = len(expected_set.intersection(top_set))
    return float(matched) / float(len(expected_set))


def _ensure_index(project_dir: Path, cfg: dict[str, Any], mode: str) -> dict[str, Any]:
    if mode == "build":
        return build_or_update_indexes(project_dir=project_dir, config=cfg, mode="build", full=True)
    if mode == "update":
        return build_or_update_indexes(project_dir=project_dir, config=cfg, mode="update", full=False)

    accel_home = Path(str(cfg["runtime"]["accel_home"])).resolve()
    paths = project_paths(accel_home, project_dir)
    ensure_project_dirs(paths)
    manifest = _load_manifest(paths["index"])
    if not manifest:
        return build_or_update_indexes(project_dir=project_dir, config=cfg, mode="build", full=True)
    return manifest


def _task_result(
    *,
    project_dir: Path,
    cfg: dict[str, Any],
    task: dict[str, Any],
    full_source_chars: int,
    run_verify_enabled: bool,
) -> dict[str, Any]:
    runtime_cfg = dict(cfg.get("runtime", {}))
    estimator_kwargs = {
        "backend": runtime_cfg.get("token_estimator_backend", "auto"),
        "model": runtime_cfg.get("token_estimator_model", ""),
        "encoding": runtime_cfg.get("token_estimator_encoding", "cl100k_base"),
        "calibration": runtime_cfg.get("token_estimator_calibration", 1.0),
        "fallback_chars_per_token": runtime_cfg.get("token_estimator_fallback_chars_per_token", 4.0),
    }

    started = time.perf_counter()
    pack = compile_context_pack(
        project_dir=project_dir,
        config=cfg,
        task=str(task["task"]),
        changed_files=list(task.get("changed_files", [])),
        hints=list(task.get("hints", [])),
        previous_attempt_feedback=None,
        budget_override=None,
    )
    context_build_seconds = round(time.perf_counter() - started, 6)

    pack_json_text = json.dumps(pack, ensure_ascii=False)
    context_token_est = estimate_tokens_for_text(pack_json_text, **estimator_kwargs)
    context_tokens = int(context_token_est.get("estimated_tokens", 0) or 0)
    chars_per_token = float(context_token_est.get("chars_per_token", 4.0) or 4.0)
    calibration = float(context_token_est.get("calibration", 1.0) or 1.0)

    full_tokens = int(
        estimate_tokens_from_chars(
            full_source_chars,
            chars_per_token=chars_per_token,
            calibration=calibration,
        ).get("estimated_tokens", 0)
        or 0
    )

    changed_files = list(task.get("changed_files", []))
    changed_chars = _sum_changed_chars(project_dir, changed_files)
    changed_tokens = int(
        estimate_tokens_from_chars(
            changed_chars,
            chars_per_token=chars_per_token,
            calibration=calibration,
        ).get("estimated_tokens", 0)
        or 0
    )

    reduction_vs_full = 0.0
    if full_tokens > 0:
        reduction_vs_full = max(0.0, 1.0 - (float(context_tokens) / float(full_tokens)))

    reduction_vs_changed = None
    if changed_tokens > 0:
        reduction_vs_changed = max(0.0, 1.0 - (float(context_tokens) / float(changed_tokens)))

    top_files = [
        str(row.get("path", ""))
        for row in list(pack.get("top_files", []))
        if isinstance(row, dict)
    ]
    recall = _top_file_recall(top_files, list(task.get("expected_top_files", [])))

    verify_status = None
    verify_exit_code = None
    verify_seconds = None
    if run_verify_enabled:
        verify_started = time.perf_counter()
        verify_result = run_verify(
            project_dir=project_dir,
            config=cfg,
            changed_files=changed_files,
        )
        verify_seconds = round(time.perf_counter() - verify_started, 6)
        verify_status = str(verify_result.get("status", ""))
        verify_exit_code = int(verify_result.get("exit_code", 1) or 1)

    verify_plan = pack.get("verify_plan", {}) if isinstance(pack.get("verify_plan"), dict) else {}
    selected_tests = list(verify_plan.get("target_tests", [])) if isinstance(verify_plan.get("target_tests"), list) else []
    selected_checks = list(verify_plan.get("target_checks", [])) if isinstance(verify_plan.get("target_checks"), list) else []

    return {
        "id": str(task["id"]),
        "task": str(task["task"]),
        "changed_files": changed_files,
        "hints": list(task.get("hints", [])),
        "expected_top_files": list(task.get("expected_top_files", [])),
        "top_files": top_files,
        "selected_tests_count": len(selected_tests),
        "selected_checks_count": len(selected_checks),
        "context_build_seconds": context_build_seconds,
        "context_chars": len(pack_json_text),
        "context_tokens": context_tokens,
        "full_source_chars": int(full_source_chars),
        "full_source_tokens": int(full_tokens),
        "changed_files_chars": int(changed_chars),
        "changed_files_tokens": int(changed_tokens),
        "token_reduction_vs_full_index": round(float(reduction_vs_full), 6),
        "token_reduction_vs_changed_files": round(float(reduction_vs_changed), 6)
        if reduction_vs_changed is not None
        else None,
        "top_file_recall": round(float(recall), 6) if recall is not None else None,
        "verify_status": verify_status,
        "verify_exit_code": verify_exit_code,
        "verify_seconds": verify_seconds,
    }


def _aggregate(task_rows: list[dict[str, Any]], run_verify_enabled: bool) -> dict[str, Any]:
    context_seconds = [float(item["context_build_seconds"]) for item in task_rows]
    context_tokens = [int(item["context_tokens"]) for item in task_rows]
    reductions_full = [float(item["token_reduction_vs_full_index"]) for item in task_rows]
    recall_rows: list[float] = []
    for item in task_rows:
        recall_value = item.get("top_file_recall")
        if recall_value is None:
            continue
        recall_rows.append(float(recall_value))

    summary: dict[str, Any] = {
        "tasks": len(task_rows),
        "context_build_seconds_avg": round(float(statistics.mean(context_seconds)), 6),
        "context_build_seconds_p50": round(float(statistics.median(context_seconds)), 6),
        "context_tokens_avg": round(float(statistics.mean(context_tokens)), 2),
        "token_reduction_vs_full_index_avg": round(float(statistics.mean(reductions_full)), 6),
        "top_file_recall_avg": round(float(statistics.mean(recall_rows)), 6) if recall_rows else None,
    }

    changed_reduction_rows = [
        float(item["token_reduction_vs_changed_files"])
        for item in task_rows
        if item.get("token_reduction_vs_changed_files") is not None
    ]
    summary["token_reduction_vs_changed_files_avg"] = (
        round(float(statistics.mean(changed_reduction_rows)), 6)
        if changed_reduction_rows
        else None
    )

    if run_verify_enabled:
        verify_rows = [item for item in task_rows if item.get("verify_exit_code") is not None]
        if verify_rows:
            passed = sum(1 for item in verify_rows if int(item.get("verify_exit_code", 1)) == 0)
            verify_seconds = [float(item.get("verify_seconds", 0.0) or 0.0) for item in verify_rows]
            summary["verify_pass_rate"] = round(float(passed) / float(len(verify_rows)), 6)
            summary["verify_seconds_avg"] = round(float(statistics.mean(verify_seconds)), 6)
        else:
            summary["verify_pass_rate"] = None
            summary["verify_seconds_avg"] = None

    return summary


def _format_ratio_percent(value: Any) -> str:
    if value is None:
        return "n/a"
    try:
        return f"{float(value) * 100.0:.2f}%"
    except (TypeError, ValueError):
        return "n/a"


def _format_number(value: Any, digits: int = 2) -> str:
    if value is None:
        return "n/a"
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return "n/a"


def _render_markdown_report(payload: dict[str, Any]) -> str:
    summary = payload.get("summary", {}) if isinstance(payload.get("summary"), dict) else {}
    results = payload.get("results", []) if isinstance(payload.get("results"), list) else []

    lines: list[str] = []
    lines.append("# AgentAccel Benchmark Report")
    lines.append("")
    lines.append(f"- generated_at: `{payload.get('generated_at', '')}`")
    lines.append(f"- project_dir: `{payload.get('project_dir', '')}`")
    lines.append(f"- tasks_file: `{payload.get('tasks_file', '')}`")
    lines.append(f"- index_mode: `{payload.get('index_mode', '')}`")
    lines.append(f"- run_verify: `{bool(payload.get('run_verify', False))}`")
    lines.append("")

    lines.append("## Summary")
    lines.append("")
    lines.append("| metric | value |")
    lines.append("| --- | --- |")
    lines.append(f"| tasks | {int(summary.get('tasks', 0) or 0)} |")
    lines.append(
        f"| context_build_seconds_avg | {_format_number(summary.get('context_build_seconds_avg'), 6)} |"
    )
    lines.append(
        f"| context_build_seconds_p50 | {_format_number(summary.get('context_build_seconds_p50'), 6)} |"
    )
    lines.append(f"| context_tokens_avg | {_format_number(summary.get('context_tokens_avg'), 2)} |")
    lines.append(
        f"| token_reduction_vs_full_index_avg | {_format_ratio_percent(summary.get('token_reduction_vs_full_index_avg'))} |"
    )
    lines.append(
        f"| token_reduction_vs_changed_files_avg | {_format_ratio_percent(summary.get('token_reduction_vs_changed_files_avg'))} |"
    )
    lines.append(
        f"| top_file_recall_avg | {_format_ratio_percent(summary.get('top_file_recall_avg'))} |"
    )
    if "verify_pass_rate" in summary:
        lines.append(
            f"| verify_pass_rate | {_format_ratio_percent(summary.get('verify_pass_rate'))} |"
        )
    if "verify_seconds_avg" in summary:
        lines.append(
            f"| verify_seconds_avg | {_format_number(summary.get('verify_seconds_avg'), 6)} |"
        )
    lines.append("")

    lines.append("## Per Task")
    lines.append("")
    lines.append(
        "| id | context_tokens | reduction_vs_full | reduction_vs_changed | top_file_recall | context_seconds | verify_exit_code |"
    )
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: |")
    for row in results:
        if not isinstance(row, dict):
            continue
        lines.append(
            "| "
            + f"{str(row.get('id', ''))} | "
            + f"{int(row.get('context_tokens', 0) or 0)} | "
            + f"{_format_ratio_percent(row.get('token_reduction_vs_full_index'))} | "
            + f"{_format_ratio_percent(row.get('token_reduction_vs_changed_files'))} | "
            + f"{_format_ratio_percent(row.get('top_file_recall'))} | "
            + f"{_format_number(row.get('context_build_seconds'), 6)} | "
            + (
                f"{int(row.get('verify_exit_code', 0))}"
                if row.get("verify_exit_code") is not None
                else "n/a"
            )
            + " |"
        )

    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run reproducible benchmark tasks for agent-accel")
    parser.add_argument("--project", default=".", help="Project directory (default: current directory)")
    parser.add_argument(
        "--tasks",
        default="examples/benchmarks/tasks.sample.json",
        help="Benchmark task JSON file",
    )
    parser.add_argument(
        "--out",
        default="",
        help="Output JSON path (default: examples/benchmarks/results_<ts>.json)",
    )
    parser.add_argument(
        "--out-md",
        default="",
        help="Optional Markdown report path (default: disabled)",
    )
    parser.add_argument(
        "--index-mode",
        choices=["reuse", "update", "build"],
        default="reuse",
        help="Index strategy before benchmark",
    )
    parser.add_argument(
        "--run-verify",
        action="store_true",
        help="Execute run_verify for each task to collect pass-rate metrics",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    project_dir = _normalize_path(Path(args.project))
    tasks_path = _normalize_path(Path(args.tasks))

    if not tasks_path.exists():
        raise SystemExit(f"tasks file not found: {tasks_path}")

    cfg = resolve_effective_config(project_dir)
    manifest = _ensure_index(project_dir, cfg, mode=str(args.index_mode))

    accel_home = Path(str(cfg["runtime"]["accel_home"])).resolve()
    paths = project_paths(accel_home, project_dir)
    ensure_project_dirs(paths)

    full_source_chars = _estimate_full_source_chars(project_dir, manifest)
    tasks = _load_tasks(tasks_path)

    task_rows = [
        _task_result(
            project_dir=project_dir,
            cfg=cfg,
            task=task,
            full_source_chars=full_source_chars,
            run_verify_enabled=bool(args.run_verify),
        )
        for task in tasks
    ]

    git_code, git_head = _run_cmd(["git", "rev-parse", "HEAD"], project_dir)
    git_status_code, git_status = _run_cmd(["git", "status", "--porcelain"], project_dir)

    payload = {
        "schema_version": 1,
        "generated_at": _utc_now(),
        "project_dir": str(project_dir),
        "tasks_file": str(tasks_path),
        "index_mode": str(args.index_mode),
        "run_verify": bool(args.run_verify),
        "runtime": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "token_estimator_backend": cfg.get("runtime", {}).get("token_estimator_backend", "auto"),
            "token_estimator_encoding": cfg.get("runtime", {}).get("token_estimator_encoding", "cl100k_base"),
            "token_estimator_calibration": cfg.get("runtime", {}).get("token_estimator_calibration", 1.0),
        },
        "git": {
            "head": git_head if git_code == 0 else "",
            "status": git_status if git_status_code == 0 else "",
            "dirty": bool(git_status_code == 0 and bool(git_status.strip())),
        },
        "manifest_counts": manifest.get("counts", {}) if isinstance(manifest, dict) else {},
        "results": task_rows,
        "summary": _aggregate(task_rows, run_verify_enabled=bool(args.run_verify)),
    }

    if args.out:
        out_path = _normalize_path(Path(args.out))
    else:
        nonce = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        out_path = project_dir / "examples" / "benchmarks" / f"results_{nonce}.json"

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    out_md_path = ""
    if args.out_md:
        out_md = _normalize_path(Path(args.out_md))
        out_md.parent.mkdir(parents=True, exist_ok=True)
        out_md.write_text(_render_markdown_report(payload), encoding="utf-8")
        out_md_path = str(out_md)

    print(
        json.dumps(
            {
                "status": "ok",
                "out": str(out_path),
                "out_md": out_md_path,
                "summary": payload["summary"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
