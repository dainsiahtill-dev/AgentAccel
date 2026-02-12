from __future__ import annotations

from pathlib import Path

from accel.config import init_project, resolve_effective_config
from accel.indexers import build_or_update_indexes
import accel.verify.sharding as sharding
from accel.verify.sharding import _with_targeted_pytests, _with_targeted_pytests_shards, select_verify_commands


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def test_verify_sharding_python_only() -> None:
    cfg = {
        "verify": {
            "python": ["pytest -q"],
            "node": ["npm run typecheck"],
        }
    }
    cmds = select_verify_commands(cfg, changed_files=["src/foo.py"])
    assert cmds == ["pytest -q"]


def test_verify_sharding_all_when_no_changes() -> None:
    cfg = {
        "verify": {
            "python": ["pytest -q"],
            "node": ["npm run typecheck"],
        }
    }
    cmds = select_verify_commands(cfg, changed_files=[])
    assert "pytest -q" in cmds
    assert "npm run typecheck" in cmds


def test_verify_sharding_dependency_graph_targets_impacted_tests(tmp_path: Path, monkeypatch) -> None:
    _write(
        tmp_path / "src" / "auth.py",
        "def login() -> str:\n    return 'ok'\n",
    )
    _write(
        tmp_path / "src" / "service.py",
        "from src.auth import login\n\n\ndef run_login() -> str:\n    return login()\n",
    )
    _write(
        tmp_path / "tests" / "test_auth.py",
        "from src.auth import login\n\n\ndef test_login():\n    assert login() == 'ok'\n",
    )
    _write(
        tmp_path / "tests" / "test_service.py",
        "from src.service import run_login\n\n\ndef test_run_login():\n    assert run_login() == 'ok'\n",
    )

    init_project(tmp_path)
    monkeypatch.setenv("ACCEL_HOME", str(tmp_path / ".accel-home"))

    cfg = resolve_effective_config(
        tmp_path,
        cli_overrides={"runtime": {"verify_max_target_tests": 16}},
    )
    cfg["verify"] = {
        "python": ["pytest -q", "python -m mypy ."],
        "node": [],
    }

    build_or_update_indexes(project_dir=tmp_path, config=cfg, mode="build", full=True)
    cmds = select_verify_commands(cfg, changed_files=["src/auth.py"])

    assert len(cmds) == 2
    assert cmds[0].startswith("pytest -q ")
    assert "tests/test_auth.py" in cmds[0]
    assert "tests/test_service.py" in cmds[0]
    assert cmds[1] == "python -m mypy ."


def test_with_targeted_pytests_filters_non_python_tests() -> None:
    command = _with_targeted_pytests(
        "pytest -q",
        ["tests/test_auth.py", "tests/electron/app.spec.ts"],
    )
    assert command.startswith("pytest -q ")
    assert "tests/test_auth.py" in command
    assert "tests/electron/app.spec.ts" not in command


def test_with_targeted_pytests_shards_splits_and_caps() -> None:
    targets = [f"tests/test_case_{idx}.py" for idx in range(10)]
    commands = _with_targeted_pytests_shards(
        command="pytest -q",
        target_tests=targets,
        shard_size=2,
        max_shards=3,
    )
    assert len(commands) == 3
    for target in targets:
        assert sum(1 for cmd in commands if target in cmd) == 1


def test_with_targeted_pytests_shards_ignores_non_py() -> None:
    commands = _with_targeted_pytests_shards(
        command="pytest -q",
        target_tests=["tests/a.py", "tests/b.spec.ts", "tests/c.ts"],
        shard_size=1,
        max_shards=4,
    )
    assert len(commands) == 1
    assert "tests/a.py" in commands[0]
    assert "tests/b.spec.ts" not in commands[0]
    assert "tests/c.ts" not in commands[0]


def test_verify_sharding_command_plan_cache_hits(tmp_path: Path, monkeypatch) -> None:
    project_dir = tmp_path / "cache_project"
    project_dir.mkdir(parents=True)
    cfg = {
        "verify": {"python": ["pytest -q"], "node": []},
        "runtime": {
            "accel_home": str(tmp_path / ".accel-home"),
            "command_plan_cache_enabled": True,
            "command_plan_cache_ttl_seconds": 3600,
            "command_plan_cache_max_entries": 100,
            "verify_max_target_tests": 16,
            "verify_pytest_shard_size": 8,
            "verify_pytest_max_shards": 2,
            "verify_fail_fast": False,
        },
        "meta": {"project_dir": str(project_dir)},
    }

    calls = {"count": 0}

    def fake_load_index_inputs(config):
        calls["count"] += 1
        return set(), [], []

    monkeypatch.setattr(sharding, "_load_index_inputs", fake_load_index_inputs)

    first = select_verify_commands(cfg, changed_files=["src/auth.py"])
    second = select_verify_commands(cfg, changed_files=["src/auth.py"])

    assert first == ["pytest -q"]
    assert second == ["pytest -q"]
    assert calls["count"] == 1


def test_verify_sharding_routes_node_commands_to_workspace(tmp_path: Path) -> None:
    project_dir = tmp_path / "workspace_project"
    frontend_dir = project_dir / "frontend"
    frontend_dir.mkdir(parents=True, exist_ok=True)
    (frontend_dir / "package.json").write_text(
        '{"name":"frontend","scripts":{"lint":"eslint .","typecheck":"tsc --noEmit"}}\n',
        encoding="utf-8",
    )

    cfg = {
        "verify": {"python": [], "node": ["npm run lint"]},
        "runtime": {
            "command_plan_cache_enabled": False,
            "verify_workspace_routing_enabled": True,
        },
        "meta": {"project_dir": str(project_dir)},
    }

    cmds = select_verify_commands(cfg, changed_files=["frontend/src/app.ts"])
    assert len(cmds) == 1
    routed = str(cmds[0]).lower()
    assert "npm run lint" in routed
    assert "frontend" in routed
    assert "&&" in routed


def test_verify_sharding_workspace_routing_can_be_disabled(tmp_path: Path) -> None:
    project_dir = tmp_path / "workspace_project_disable"
    frontend_dir = project_dir / "frontend"
    frontend_dir.mkdir(parents=True, exist_ok=True)
    (frontend_dir / "package.json").write_text(
        '{"name":"frontend","scripts":{"lint":"eslint ."}}\n',
        encoding="utf-8",
    )

    cfg = {
        "verify": {"python": [], "node": ["npm run lint"]},
        "runtime": {
            "command_plan_cache_enabled": False,
            "verify_workspace_routing_enabled": False,
        },
        "meta": {"project_dir": str(project_dir)},
    }

    cmds = select_verify_commands(cfg, changed_files=["frontend/src/app.ts"])
    assert cmds == ["npm run lint"]


def test_verify_sharding_respects_custom_language_profile_registry() -> None:
    cfg = {
        "language_profiles": ["go"],
        "language_profile_registry": {
            "go": {"extensions": [".go"], "verify_group": "go"},
        },
        "verify": {
            "python": ["pytest -q"],
            "go": ["go test ./..."],
        },
        "runtime": {"command_plan_cache_enabled": False},
    }
    cmds = select_verify_commands(cfg, changed_files=["pkg/service.go"])
    assert cmds == ["go test ./..."]
