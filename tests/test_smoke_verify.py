from __future__ import annotations

from pathlib import Path
from uuid import uuid4

from accel.config import resolve_effective_config
from accel.verify.orchestrator import run_verify


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def test_run_verify_smoke_with_quick_command() -> None:
    repo_root = _repo_root()
    run_id = uuid4().hex[:10]

    cfg = resolve_effective_config(repo_root)
    cfg["runtime"]["accel_home"] = str(
        repo_root / ".harborpilot" / "runtime" / "pytest" / f"verify_{run_id}"
    )
    cfg["runtime"]["verify_workers"] = 1
    cfg["runtime"]["per_command_timeout_seconds"] = 60
    cfg["runtime"]["verify_cache_enabled"] = False
    cfg["runtime"]["verify_preflight_enabled"] = False
    cfg["verify"] = {"python": ['python -c "print(\'verify-smoke\')"']}

    result = run_verify(project_dir=repo_root, config=cfg, changed_files=["accel/cli.py"])
    assert int(result.get("exit_code", 1)) == 0
    assert str(result.get("status", "")) == "success"

    log_path = Path(str(result.get("log_path", "")))
    assert log_path.exists()
    log_text = log_path.read_text(encoding="utf-8")
    assert "VERIFICATION_START" in log_text
    assert "VERIFICATION_END" in log_text
