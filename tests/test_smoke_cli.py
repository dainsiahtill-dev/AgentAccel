from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from uuid import uuid4


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _run_cli_json(
    args: list[str],
    *,
    extra_env: dict[str, str] | None = None,
    timeout_seconds: int = 180,
) -> dict[str, object]:
    repo_root = _repo_root()
    env = os.environ.copy()
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    if extra_env:
        env.update(extra_env)

    proc = subprocess.run(
        [sys.executable, "-m", "accel.cli", *args, "--output", "json"],
        cwd=str(repo_root),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=max(1, int(timeout_seconds)),
        check=False,
        env=env,
    )
    assert proc.returncode == 0, proc.stderr or proc.stdout
    payload = json.loads(proc.stdout)
    assert isinstance(payload, dict)
    return payload


def test_cli_doctor_smoke() -> None:
    run_id = uuid4().hex[:10]
    accel_home = _repo_root() / ".harborpilot" / "runtime" / "pytest" / f"doctor_{run_id}"
    payload = _run_cli_json(
        ["doctor", "--project", "."],
        extra_env={"ACCEL_HOME": str(accel_home)},
    )
    assert payload.get("status") == "ok"
    semantic_runtime = payload.get("semantic_runtime")
    assert isinstance(semantic_runtime, dict)
    assert semantic_runtime.get("reason") == "removed_from_build"


def test_cli_context_output_kept_under_harborpilot() -> None:
    run_id = uuid4().hex[:10]
    accel_home = _repo_root() / ".harborpilot" / "runtime" / "pytest" / f"context_{run_id}"
    requested_out = f"context_pack_pytest_{run_id}.json"
    payload = _run_cli_json(
        [
            "context",
            "--project",
            ".",
            "--task",
            "smoke context generation",
            "--changed-files",
            "accel/cli.py",
            "--max-chars",
            "2000",
            "--max-snippets",
            "6",
            "--top-n-files",
            "3",
            "--snippet-radius",
            "8",
            "--out",
            requested_out,
        ],
        extra_env={"ACCEL_HOME": str(accel_home)},
    )
    assert payload.get("status") == "ok"
    out_value = str(payload.get("out", ""))
    assert out_value
    out_path = Path(out_value)
    assert out_path.exists()
    assert ".harborpilot" in out_path.as_posix().lower()
    assert payload.get("out_relocated_to_harborpilot") is True
