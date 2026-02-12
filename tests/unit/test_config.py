from __future__ import annotations

import json
from pathlib import Path

from accel.config import init_project, resolve_effective_config


def test_init_project_writes_templates(tmp_path: Path) -> None:
    result = init_project(tmp_path)
    assert result["created"]
    assert (tmp_path / "accel.yaml").exists()
    assert (tmp_path / "accel.local.yaml.example").exists()


def test_env_overrides_take_priority(tmp_path: Path, monkeypatch) -> None:
    init_project(tmp_path)
    local_cfg = {"runtime": {"max_workers": 7}, "gpu": {"enabled": False}}
    (tmp_path / "accel.local.yaml").write_text(
        json.dumps(local_cfg, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    monkeypatch.setenv("ACCEL_MAX_WORKERS", "3")
    monkeypatch.setenv("ACCEL_GPU_ENABLED", "1")

    cfg = resolve_effective_config(tmp_path)
    assert int(cfg["runtime"]["max_workers"]) == 3
    assert bool(cfg["gpu"]["enabled"]) is True


def test_runtime_index_overrides_take_priority(tmp_path: Path, monkeypatch) -> None:
    init_project(tmp_path)
    monkeypatch.setenv("ACCEL_VERIFY_WORKERS", "9")
    monkeypatch.setenv("ACCEL_INDEX_WORKERS", "48")
    monkeypatch.setenv("ACCEL_INDEX_COMPACT_EVERY", "320")
    monkeypatch.setenv("ACCEL_VERIFY_MAX_TARGET_TESTS", "21")
    monkeypatch.setenv("ACCEL_VERIFY_PYTEST_SHARD_SIZE", "7")
    monkeypatch.setenv("ACCEL_VERIFY_PYTEST_MAX_SHARDS", "5")
    monkeypatch.setenv("ACCEL_VERIFY_FAIL_FAST", "1")
    monkeypatch.setenv("ACCEL_VERIFY_CACHE_ENABLED", "1")
    monkeypatch.setenv("ACCEL_VERIFY_CACHE_TTL_SECONDS", "77")
    monkeypatch.setenv("ACCEL_VERIFY_CACHE_MAX_ENTRIES", "33")

    cfg = resolve_effective_config(tmp_path)
    assert int(cfg["runtime"]["verify_workers"]) == 9
    assert int(cfg["runtime"]["index_workers"]) == 48
    assert int(cfg["runtime"]["index_delta_compact_every"]) == 320
    assert int(cfg["runtime"]["verify_max_target_tests"]) == 21
    assert int(cfg["runtime"]["verify_pytest_shard_size"]) == 7
    assert int(cfg["runtime"]["verify_pytest_max_shards"]) == 5
    assert bool(cfg["runtime"]["verify_fail_fast"]) is True
    assert bool(cfg["runtime"]["verify_cache_enabled"]) is True
    assert int(cfg["runtime"]["verify_cache_ttl_seconds"]) == 77
    assert int(cfg["runtime"]["verify_cache_max_entries"]) == 33


def test_runtime_workspace_and_preflight_env_overrides(tmp_path: Path, monkeypatch) -> None:
    init_project(tmp_path)
    monkeypatch.setenv("ACCEL_VERIFY_WORKSPACE_ROUTING_ENABLED", "0")
    monkeypatch.setenv("ACCEL_VERIFY_PREFLIGHT_ENABLED", "0")
    monkeypatch.setenv("ACCEL_VERIFY_PREFLIGHT_TIMEOUT_SECONDS", "9")
    monkeypatch.setenv("ACCEL_VERIFY_STALL_TIMEOUT_SECONDS", "7")
    monkeypatch.setenv("ACCEL_VERIFY_AUTO_CANCEL_ON_STALL", "1")
    monkeypatch.setenv("ACCEL_VERIFY_MAX_WALL_TIME_SECONDS", "88")

    cfg = resolve_effective_config(tmp_path)
    assert bool(cfg["runtime"]["verify_workspace_routing_enabled"]) is False
    assert bool(cfg["runtime"]["verify_preflight_enabled"]) is False
    assert int(cfg["runtime"]["verify_preflight_timeout_seconds"]) == 9
    assert float(cfg["runtime"]["verify_stall_timeout_seconds"]) == 7.0
    assert bool(cfg["runtime"]["verify_auto_cancel_on_stall"]) is True
    assert float(cfg["runtime"]["verify_max_wall_time_seconds"]) == 88.0


def test_constraint_mode_alias_normalization(tmp_path: Path, monkeypatch) -> None:
    init_project(tmp_path)
    monkeypatch.setenv("ACCEL_CONSTRAINT_MODE", "enforce")

    cfg = resolve_effective_config(tmp_path)
    assert str(cfg["runtime"]["constraint_mode"]) == "strict"
