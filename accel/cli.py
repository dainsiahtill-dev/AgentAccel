from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from .config import init_project, resolve_effective_config
from .indexers import build_or_update_indexes
from .query.context_compiler import compile_context_pack, write_context_pack
from .storage.cache import ensure_project_dirs, project_paths
from .verify.orchestrator import run_verify


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _normalize_path(path: Path) -> Path:
    return Path(os.path.abspath(str(path)))


def _detect_cuda() -> dict[str, Any]:
    nvidia_smi = shutil.which("nvidia-smi")
    if not nvidia_smi:
        return {"available": False, "reason": "nvidia-smi not found", "raw": ""}
    try:
        proc = subprocess.run(
            [nvidia_smi, "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=10,
        )
        if proc.returncode != 0:
            return {
                "available": False,
                "reason": proc.stderr.strip() or f"exit={proc.returncode}",
                "raw": proc.stdout.strip(),
            }
        gpus = [line.strip() for line in proc.stdout.splitlines() if line.strip()]
        return {"available": bool(gpus), "reason": "", "raw": gpus}
    except Exception as exc:  # pragma: no cover - defensive path
        return {"available": False, "reason": str(exc), "raw": ""}


def _print_output(payload: dict[str, Any], output: str) -> None:
    if output == "json":
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        for key, value in payload.items():
            print(f"{key}: {value}")


def _parse_budget(args: argparse.Namespace) -> dict[str, int]:
    budget: dict[str, int] = {}
    if args.max_chars is not None:
        budget["max_chars"] = int(args.max_chars)
    if args.max_snippets is not None:
        budget["max_snippets"] = int(args.max_snippets)
    if args.top_n_files is not None:
        budget["top_n_files"] = int(args.top_n_files)
    if args.snippet_radius is not None:
        budget["snippet_radius"] = int(args.snippet_radius)
    return budget


def cmd_init(args: argparse.Namespace) -> int:
    project_dir = _normalize_path(Path(args.project))
    result = init_project(project_dir, force=args.force)
    _print_output({"status": "ok", "project": str(project_dir), **result}, args.output)
    return 0


def cmd_doctor(args: argparse.Namespace) -> int:
    project_dir = _normalize_path(Path(args.project))
    cfg = resolve_effective_config(project_dir)

    doctor = {
        "status": "ok",
        "timestamp_utc": _utc_now(),
        "platform": platform.platform(),
        "system": platform.system(),
        "python": sys.version.split()[0],
        "cwd": str(Path.cwd()),
        "project_dir": str(project_dir),
        "is_wsl": bool(os.environ.get("WSL_DISTRO_NAME")),
        "accel_home": cfg["runtime"]["accel_home"],
        "tools": {
            "git": shutil.which("git") is not None,
            "rg": shutil.which("rg") is not None,
            "node": shutil.which("node") is not None,
            "npm": shutil.which("npm") is not None,
            "pytest": shutil.which("pytest") is not None,
        },
        "cuda": _detect_cuda(),
    }
    if args.print_config:
        doctor["effective_config"] = cfg
    _print_output(doctor, args.output)
    return 0


def cmd_index_build(args: argparse.Namespace) -> int:
    project_dir = _normalize_path(Path(args.project))
    cfg = resolve_effective_config(project_dir)
    manifest = build_or_update_indexes(
        project_dir=project_dir,
        config=cfg,
        mode="build",
        full=bool(args.full),
    )
    _print_output({"status": "ok", "manifest": manifest}, args.output)
    return 0


def cmd_index_update(args: argparse.Namespace) -> int:
    project_dir = _normalize_path(Path(args.project))
    cfg = resolve_effective_config(project_dir)
    manifest = build_or_update_indexes(
        project_dir=project_dir,
        config=cfg,
        mode="update",
        full=False,
    )
    _print_output({"status": "ok", "manifest": manifest}, args.output)
    return 0


def cmd_context(args: argparse.Namespace) -> int:
    project_dir = _normalize_path(Path(args.project))
    cfg = resolve_effective_config(project_dir)
    feedback: dict[str, Any] | None = None
    if args.feedback:
        feedback_path = _normalize_path(Path(args.feedback))
        if feedback_path.exists():
            feedback = json.loads(feedback_path.read_text(encoding="utf-8"))

    pack = compile_context_pack(
        project_dir=project_dir,
        config=cfg,
        task=args.task,
        changed_files=args.changed_files,
        hints=args.hints,
        previous_attempt_feedback=feedback,
        budget_override=_parse_budget(args),
    )

    accel_home = Path(cfg["runtime"]["accel_home"]).resolve()
    paths = project_paths(accel_home, project_dir)
    ensure_project_dirs(paths)
    if args.out:
        out_path = _normalize_path(Path(args.out))
    else:
        out_path = paths["context"] / f"context_pack_{uuid4().hex[:10]}.json"
    write_context_pack(out_path, pack)
    _print_output(
        {
            "status": "ok",
            "out": str(out_path),
            "top_files": len(pack.get("top_files", [])),
            "snippets": len(pack.get("snippets", [])),
        },
        args.output,
    )
    return 0


def cmd_verify(args: argparse.Namespace) -> int:
    project_dir = _normalize_path(Path(args.project))
    cfg = resolve_effective_config(project_dir)
    result = run_verify(project_dir=project_dir, config=cfg, changed_files=args.changed_files)
    _print_output(result, args.output)
    return int(result["exit_code"])


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="accel",
        description="agent-accel CLI: local code intelligence and verification orchestration",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    init_parser = sub.add_parser("init", help="Initialize accel config templates")
    init_parser.add_argument("--project", default=".", help="Project directory")
    init_parser.add_argument("--force", action="store_true", help="Overwrite existing templates")
    init_parser.add_argument("--output", choices=["text", "json"], default="text")
    init_parser.set_defaults(func=cmd_init)

    doctor_parser = sub.add_parser("doctor", help="Print environment diagnostics")
    doctor_parser.add_argument("--project", default=".", help="Project directory")
    doctor_parser.add_argument("--print-config", action="store_true", help="Include effective config")
    doctor_parser.add_argument("--output", choices=["text", "json"], default="text")
    doctor_parser.set_defaults(func=cmd_doctor)

    index_parser = sub.add_parser("index", help="Build or update indexes")
    index_sub = index_parser.add_subparsers(dest="index_command", required=True)
    build_parser_cmd = index_sub.add_parser("build", help="Build indexes")
    build_parser_cmd.add_argument("--project", default=".", help="Project directory")
    build_parser_cmd.add_argument("--full", action="store_true", help="Force full rebuild")
    build_parser_cmd.add_argument("--output", choices=["text", "json"], default="text")
    build_parser_cmd.set_defaults(func=cmd_index_build)

    update_parser_cmd = index_sub.add_parser("update", help="Incrementally update indexes")
    update_parser_cmd.add_argument("--project", default=".", help="Project directory")
    update_parser_cmd.add_argument("--output", choices=["text", "json"], default="text")
    update_parser_cmd.set_defaults(func=cmd_index_update)

    context_parser = sub.add_parser("context", help="Generate context_pack.json")
    context_parser.add_argument("--project", default=".", help="Project directory")
    context_parser.add_argument("--task", required=True, help="Task description")
    context_parser.add_argument("--out", help="Output path for context pack")
    context_parser.add_argument("--feedback", help="Optional previous feedback JSON path")
    context_parser.add_argument("--changed-files", nargs="*", default=[])
    context_parser.add_argument("--hints", nargs="*", default=[])
    context_parser.add_argument("--max-chars", type=int)
    context_parser.add_argument("--max-snippets", type=int)
    context_parser.add_argument("--top-n-files", type=int)
    context_parser.add_argument("--snippet-radius", type=int)
    context_parser.add_argument("--output", choices=["text", "json"], default="text")
    context_parser.set_defaults(func=cmd_context)

    verify_parser = sub.add_parser("verify", help="Run incremental verification")
    verify_parser.add_argument("--project", default=".", help="Project directory")
    verify_parser.add_argument("--changed-files", nargs="*", default=[])
    verify_parser.add_argument("--output", choices=["text", "json"], default="text")
    verify_parser.set_defaults(func=cmd_verify)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    exit_code = int(args.func(args))
    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
