#!/usr/bin/env python3
"""Collect Phase 0 baseline metrics for HarborPilot using agent-accel outputs when available."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Ensure "accel" package is importable when script runs as a standalone file.
SCRIPT_DIR = Path(__file__).resolve().parent
ACCEL_ROOT = SCRIPT_DIR.parent
if str(ACCEL_ROOT) not in sys.path:
    sys.path.insert(0, str(ACCEL_ROOT))

# Optional imports from agent-accel package.
resolve_effective_config: Any = None
project_paths: Any = None
try:
    from accel.config import resolve_effective_config as _resolve_effective_config  # type: ignore[import-not-found,import-untyped]
    from accel.storage.cache import project_paths as _project_paths  # type: ignore[import-not-found,import-untyped]

    resolve_effective_config = _resolve_effective_config
    project_paths = _project_paths
except Exception:
    pass

RUNTIME_EXTENSIONS = {".py", ".ts", ".tsx"}
EXCLUDED_DIRS = {
    ".git",
    ".venv",
    "venv",
    "node_modules",
    "dist",
    "__pycache__",
    ".harborpilot",
    "playwright-report",
    "test-results",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
}


@dataclass
class CommandResult:
    command: list[str]
    exit_code: int
    duration_seconds: float
    stdout: str
    stderr: str


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def normalize(path: Path) -> Path:
    return Path(os.path.abspath(str(path)))


def should_skip(path: Path) -> bool:
    return any(part in EXCLUDED_DIRS for part in path.parts)


def iter_files(root: Path, suffixes: set[str] | None = None) -> list[Path]:
    if not root.exists():
        return []
    rows: list[Path] = []
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if should_skip(path):
            continue
        if suffixes is not None and path.suffix.lower() not in suffixes:
            continue
        rows.append(path)
    return rows


def count_lines(path: Path) -> int:
    try:
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            return sum(1 for _ in handle)
    except Exception:
        return -1


def count_imports(path: Path) -> int:
    total = 0
    suffix = path.suffix.lower()
    try:
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            for raw in handle:
                line = raw.lstrip()
                if suffix == ".py":
                    if line.startswith("import ") or line.startswith("from "):
                        total += 1
                else:
                    if line.startswith("import "):
                        total += 1
    except Exception:
        return -1
    return total


def rel(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root)).replace("\\", "/")
    except Exception:
        return str(path).replace("\\", "/")


def run_cmd(command: list[str], cwd: Path, timeout: int = 900) -> CommandResult:
    executable = command[0]
    resolved = shutil.which(executable)
    if os.name == "nt" and resolved is None:
        resolved = shutil.which(f"{executable}.cmd")
    if resolved:
        command = [resolved, *command[1:]]

    started = time.perf_counter()
    try:
        proc = subprocess.run(
            command,
            cwd=cwd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
        )
        return CommandResult(
            command=command,
            exit_code=int(proc.returncode),
            duration_seconds=round(time.perf_counter() - started, 2),
            stdout=proc.stdout.strip(),
            stderr=proc.stderr.strip(),
        )
    except Exception as exc:
        return CommandResult(
            command=command,
            exit_code=999,
            duration_seconds=round(time.perf_counter() - started, 2),
            stdout="",
            stderr=str(exc),
        )


def load_accel_manifest(project_root: Path) -> dict[str, Any]:
    if not callable(resolve_effective_config) or not callable(project_paths):
        return {"available": False, "reason": "accel import unavailable"}

    try:
        cfg = resolve_effective_config(project_root)
        accel_home = Path(str(cfg.get("runtime", {}).get("accel_home", ""))).resolve()
        paths = project_paths(accel_home, project_root)
        manifest_path = paths["index"] / "manifest.json"
        if not manifest_path.exists():
            return {
                "available": False,
                "reason": "manifest missing",
                "manifest_path": str(manifest_path),
            }
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
        return {
            "available": True,
            "manifest_path": str(manifest_path),
            "indexed_at": data.get("indexed_at"),
            "counts": data.get("counts", {}),
            "files": len(data.get("indexed_files", [])),
        }
    except Exception as exc:
        return {"available": False, "reason": str(exc)}


def collect(project_root: Path, include_build: bool) -> dict[str, Any]:
    runtime_roots = [
        project_root / "src" / "frontend" / "src",
        project_root / "src" / "backend",
        project_root / "src" / "electron",
    ]

    runtime_files: list[Path] = []
    for root in runtime_roots:
        runtime_files.extend(iter_files(root, RUNTIME_EXTENSIONS))

    top_files: list[dict[str, Any]] = []
    for path in runtime_files:
        top_files.append(
            {
                "path": rel(path, project_root),
                "bytes": path.stat().st_size,
                "lines": count_lines(path),
            }
        )
    top_files.sort(key=lambda row: row["bytes"], reverse=True)

    import_pressure: list[dict[str, Any]] = []
    for path in runtime_files:
        imports = count_imports(path)
        import_pressure.append(
            {
                "path": rel(path, project_root),
                "imports": imports,
            }
        )
    import_pressure = [row for row in import_pressure if row["imports"] >= 0]
    import_pressure.sort(key=lambda row: row["imports"], reverse=True)

    frontend_components = iter_files(project_root / "src" / "frontend" / "src" / "app" / "components", {".tsx", ".ts"})
    backend_routers = iter_files(project_root / "src" / "backend" / "app" / "routers", {".py"})
    backend_services = iter_files(project_root / "src" / "backend" / "app" / "services", {".py"})

    py_tests = [
        *iter_files(project_root / "tests", {".py"}),
        *iter_files(project_root / "src" / "backend" / "tests", {".py"}),
        *iter_files(project_root / "agent-accel" / "tests", {".py"}),
    ]
    py_tests = [path for path in py_tests if path.name.startswith("test_")]

    ts_tests = [
        path
        for path in iter_files(project_root, {".ts", ".tsx"})
        if ".test." in path.name or ".spec." in path.name
    ]

    git_head = run_cmd(["git", "rev-parse", "HEAD"], project_root, timeout=30)
    git_status = run_cmd(["git", "status", "--porcelain"], project_root, timeout=30)

    tool_versions = {
        "python": run_cmd(["python", "--version"], project_root, timeout=20).__dict__,
        "node": run_cmd(["node", "--version"], project_root, timeout=20).__dict__,
        "npm": run_cmd(["npm", "--version"], project_root, timeout=20).__dict__,
        "pytest": run_cmd(["pytest", "--version"], project_root, timeout=20).__dict__,
        "ruff": run_cmd(["ruff", "--version"], project_root, timeout=20).__dict__,
        "mypy": run_cmd(["mypy", "--version"], project_root, timeout=20).__dict__,
    }

    build_result: dict[str, Any] | None = None
    if include_build:
        build = run_cmd(["npm", "run", "build", "--silent"], project_root, timeout=3600)
        build_result = build.__dict__

    accel_manifest = load_accel_manifest(project_root)

    baseline = {
        "captured_at_utc": utc_now(),
        "project_root": str(project_root),
        "git": {
            "head": git_head.stdout,
            "status": git_status.stdout,
            "is_clean": git_status.exit_code == 0 and git_status.stdout == "",
        },
        "counts": {
            "runtime_files_total": len(runtime_files),
            "frontend_component_files": len(frontend_components),
            "backend_router_files": len(backend_routers),
            "backend_service_files": len(backend_services),
            "python_test_files": len(py_tests),
            "ts_test_files": len(ts_tests),
        },
        "hotspots": {
            "largest_runtime_files": top_files[:20],
            "highest_import_density": import_pressure[:20],
        },
        "tool_versions": tool_versions,
        "build_baseline": build_result,
        "accel_manifest": accel_manifest,
    }
    return baseline


def render_markdown(payload: dict[str, Any]) -> str:
    counts = payload.get("counts", {})
    hotspots = payload.get("hotspots", {})
    largest = hotspots.get("largest_runtime_files", [])
    imports = hotspots.get("highest_import_density", [])
    git_data = payload.get("git", {})
    accel_manifest = payload.get("accel_manifest", {})
    build = payload.get("build_baseline")

    lines: list[str] = []
    lines.append("# HarborPilot Phase 0 Baseline Report")
    lines.append("")
    lines.append(f"- Captured UTC: `{payload.get('captured_at_utc', '')}`")
    lines.append(f"- Git HEAD: `{git_data.get('head', '')}`")
    lines.append(f"- Git Clean: `{git_data.get('is_clean', False)}`")
    lines.append("")
    lines.append("## Structural Counts")
    lines.append("")
    lines.append(f"- Runtime files (.py/.ts/.tsx): `{counts.get('runtime_files_total', 0)}`")
    lines.append(f"- Frontend component files: `{counts.get('frontend_component_files', 0)}`")
    lines.append(f"- Backend router files: `{counts.get('backend_router_files', 0)}`")
    lines.append(f"- Backend service files: `{counts.get('backend_service_files', 0)}`")
    lines.append(f"- Python test files: `{counts.get('python_test_files', 0)}`")
    lines.append(f"- TS test/spec files: `{counts.get('ts_test_files', 0)}`")
    lines.append("")
    lines.append("## Largest Runtime Files")
    lines.append("")
    for row in largest[:15]:
        lines.append(f"- `{row['path']}` | `{row['bytes']}` bytes | `{row['lines']}` lines")
    lines.append("")
    lines.append("## Import Density (proxy for coupling)")
    lines.append("")
    for row in imports[:15]:
        lines.append(f"- `{row['path']}` | imports: `{row['imports']}`")
    lines.append("")
    lines.append("## agent-accel Index Snapshot")
    lines.append("")
    lines.append(f"- Available: `{accel_manifest.get('available', False)}`")
    lines.append(f"- Manifest: `{accel_manifest.get('manifest_path', '')}`")
    if accel_manifest.get("available"):
        lines.append(f"- Indexed At: `{accel_manifest.get('indexed_at', '')}`")
        counts_map = accel_manifest.get("counts", {})
        lines.append(f"- Symbols: `{counts_map.get('symbols', 0)}`")
        lines.append(f"- References: `{counts_map.get('references', 0)}`")
        lines.append(f"- Dependencies: `{counts_map.get('dependencies', 0)}`")
        lines.append(f"- Test Ownership Rows: `{counts_map.get('test_ownership', 0)}`")
        lines.append(f"- Indexed Files: `{accel_manifest.get('files', 0)}`")
    else:
        lines.append(f"- Reason: `{accel_manifest.get('reason', '')}`")
    lines.append("")
    lines.append("## Build Baseline")
    lines.append("")
    if build is None:
        lines.append("- Build command not executed (`--include-build` not set).")
    else:
        lines.append(f"- Command: `{ ' '.join(build.get('command', [])) }`")
        lines.append(f"- Exit Code: `{build.get('exit_code', -1)}`")
        lines.append(f"- Duration Seconds: `{build.get('duration_seconds', 0)}`")
        stdout = (build.get("stdout", "") or "")[:300]
        stderr = (build.get("stderr", "") or "")[:300]
        if stdout:
            lines.append(f"- Stdout (trimmed): `{stdout}`")
        if stderr:
            lines.append(f"- Stderr (trimmed): `{stderr}`")

    lines.append("")
    lines.append("## Notes")
    lines.append("")
    lines.append("- This report is generated from live workspace data.")
    lines.append("- Coupling and duplication here are proxy indicators; strict KPI measurement should run in Phase 5 gate.")

    return "\n".join(lines) + "\n"


def write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect HarborPilot Phase 0 baseline")
    parser.add_argument("--project", default="..", help="Project root path")
    parser.add_argument(
        "--json-out",
        default="../.harborpilot/logs/phase0_baseline_latest.json",
        help="JSON output path",
    )
    parser.add_argument(
        "--analysis-out",
        default="../.harborpilot/logs/analysis_last.json",
        help="analysis_last JSON output path",
    )
    parser.add_argument(
        "--md-out",
        default="../.harborpilot/docs/reports/phase0_baseline_latest.md",
        help="Markdown output path",
    )
    parser.add_argument("--include-build", action="store_true", help="Run npm build and capture duration")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    script_dir = Path(__file__).resolve().parent
    project_root = normalize((script_dir / args.project).resolve())

    payload = collect(project_root=project_root, include_build=bool(args.include_build))

    json_out = normalize((script_dir / args.json_out).resolve())
    analysis_out = normalize((script_dir / args.analysis_out).resolve())
    md_out = normalize((script_dir / args.md_out).resolve())

    write_json(json_out, payload)
    write_json(analysis_out, payload)
    write_text(md_out, render_markdown(payload))

    print(json.dumps({
        "status": "ok",
        "project_root": str(project_root),
        "json_out": str(json_out),
        "analysis_out": str(analysis_out),
        "md_out": str(md_out),
        "captured_at_utc": payload.get("captured_at_utc"),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
