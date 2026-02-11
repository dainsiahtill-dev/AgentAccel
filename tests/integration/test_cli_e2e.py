from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


def _run(cmd: list[str], cwd: Path, env: dict[str, str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        cwd=str(cwd),
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )


def test_cli_e2e_bootstrap_index_context(tmp_path: Path) -> None:
    project_dir = tmp_path / "repo"
    src_dir = project_dir / "src"
    tests_dir = project_dir / "tests"
    src_dir.mkdir(parents=True)
    tests_dir.mkdir(parents=True)
    (src_dir / "main.py").write_text(
        "def headers_input_ok(v: str) -> str:\n    return v.strip()\n",
        encoding="utf-8",
    )
    (tests_dir / "test_main.py").write_text(
        "from src.main import headers_input_ok\n\ndef test_strip():\n    assert headers_input_ok(' x ') == 'x'\n",
        encoding="utf-8",
    )

    package_root = Path(__file__).resolve().parents[2]
    env = dict(os.environ)
    existing_path = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = str(package_root) + (os.pathsep + existing_path if existing_path else "")
    env["ACCEL_HOME"] = str(tmp_path / ".accel-home")

    init_proc = _run(
        [sys.executable, "-m", "accel.cli", "init", "--project", str(project_dir), "--output", "json"],
        cwd=project_dir,
        env=env,
    )
    assert init_proc.returncode == 0, init_proc.stderr

    build_proc = _run(
        [
            sys.executable,
            "-m",
            "accel.cli",
            "index",
            "build",
            "--project",
            str(project_dir),
            "--full",
            "--output",
            "json",
        ],
        cwd=project_dir,
        env=env,
    )
    assert build_proc.returncode == 0, build_proc.stderr
    build_payload = json.loads(build_proc.stdout)
    assert build_payload["status"] == "ok"

    context_out = project_dir / "context_pack.json"
    context_proc = _run(
        [
            sys.executable,
            "-m",
            "accel.cli",
            "context",
            "--project",
            str(project_dir),
            "--task",
            "fix headers_input_ok",
            "--out",
            str(context_out),
            "--output",
            "json",
        ],
        cwd=project_dir,
        env=env,
    )
    assert context_proc.returncode == 0, context_proc.stderr
    assert context_out.exists()
