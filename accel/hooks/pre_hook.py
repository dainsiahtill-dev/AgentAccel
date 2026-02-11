from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from ..config import resolve_effective_config
from ..query.context_compiler import compile_context_pack, write_context_pack


def main() -> None:
    parser = argparse.ArgumentParser(description="AgentAccel pre-hook: build context_pack.json")
    parser.add_argument("--project", default=".", help="Project root path")
    parser.add_argument("--task", required=True, help="Natural language task")
    parser.add_argument("--out", default="context_pack.json", help="Output file")
    parser.add_argument(
        "--changed-files",
        nargs="*",
        default=[],
        help="Optional changed files",
    )
    parser.add_argument("--hints", nargs="*", default=[], help="Optional path/symbol hints")
    args = parser.parse_args()

    project_dir = Path(os.path.abspath(str(args.project)))
    cfg = resolve_effective_config(project_dir)
    pack = compile_context_pack(
        project_dir=project_dir,
        config=cfg,
        task=args.task,
        changed_files=args.changed_files,
        hints=args.hints,
    )
    out_path = Path(os.path.abspath(str(args.out)))
    write_context_pack(out_path, pack)
    print(json.dumps({"status": "ok", "out": str(out_path)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
