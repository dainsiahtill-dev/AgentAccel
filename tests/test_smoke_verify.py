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


def test_run_verify_degrades_when_node_workspace_missing_package_json() -> None:
    repo_root = _repo_root()
    run_id = uuid4().hex[:10]

    cfg = resolve_effective_config(repo_root)
    cfg["runtime"]["accel_home"] = str(
        repo_root / ".harborpilot" / "runtime" / "pytest" / f"verify_{run_id}"
    )
    cfg["runtime"]["verify_workers"] = 1
    cfg["runtime"]["per_command_timeout_seconds"] = 60
    cfg["runtime"]["verify_cache_enabled"] = False
    cfg["runtime"]["verify_preflight_enabled"] = True
    cfg["runtime"]["verify_workspace_routing_enabled"] = True
    cfg["verify"] = {"node": ["npm test --silent"]}

    result = run_verify(
        project_dir=repo_root, config=cfg, changed_files=["src/frontend/demo.ts"]
    )
    assert str(result.get("status", "")) == "degraded"
    assert int(result.get("exit_code", 1)) == 2
    assert list(result.get("results", [])) == []

    log_path = Path(str(result.get("log_path", "")))
    assert log_path.exists()
    log_text = log_path.read_text(encoding="utf-8")
    assert "node workspace missing package.json" in log_text
    assert "SKIP npm test --silent" in log_text


def test_run_verify_degrades_when_pytest_targets_root_only_in_subworkspace() -> None:
    repo_root = _repo_root()
    run_id = uuid4().hex[:10]

    cfg = resolve_effective_config(repo_root)
    cfg["runtime"]["accel_home"] = str(
        repo_root / ".harborpilot" / "runtime" / "pytest" / f"verify_{run_id}"
    )
    cfg["runtime"]["verify_workers"] = 1
    cfg["runtime"]["per_command_timeout_seconds"] = 60
    cfg["runtime"]["verify_cache_enabled"] = False
    cfg["runtime"]["verify_preflight_enabled"] = True
    cfg["verify"] = {"python": ['cd accel && python -m pytest -q tests/test_smoke_cli.py']}

    result = run_verify(project_dir=repo_root, config=cfg, changed_files=["accel/cli.py"])
    assert str(result.get("status", "")) == "degraded"
    assert int(result.get("exit_code", 1)) == 2
    assert list(result.get("results", [])) == []

    log_path = Path(str(result.get("log_path", "")))
    assert log_path.exists()
    log_text = log_path.read_text(encoding="utf-8")
    assert "pytest target only exists at project root" in log_text
