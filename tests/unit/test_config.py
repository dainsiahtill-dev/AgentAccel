from __future__ import annotations

import json
import os
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
    monkeypatch.setenv("ACCEL_CONTEXT_RPC_TIMEOUT_SECONDS", "155")
    monkeypatch.setenv("ACCEL_SYNC_VERIFY_WAIT_SECONDS", "185")
    monkeypatch.setenv("ACCEL_SYNC_INDEX_WAIT_SECONDS", "205")
    monkeypatch.setenv("ACCEL_SYNC_CONTEXT_WAIT_SECONDS", "75")
    monkeypatch.setenv("ACCEL_SYNC_CONTEXT_TIMEOUT_ACTION", "poll")

    cfg = resolve_effective_config(tmp_path)
    assert bool(cfg["runtime"]["verify_workspace_routing_enabled"]) is False
    assert bool(cfg["runtime"]["verify_preflight_enabled"]) is False
    assert int(cfg["runtime"]["verify_preflight_timeout_seconds"]) == 9
    assert float(cfg["runtime"]["verify_stall_timeout_seconds"]) == 7.0
    assert bool(cfg["runtime"]["verify_auto_cancel_on_stall"]) is True
    assert float(cfg["runtime"]["verify_max_wall_time_seconds"]) == 88.0
    assert float(cfg["runtime"]["context_rpc_timeout_seconds"]) == 155.0
    assert float(cfg["runtime"]["sync_verify_wait_seconds"]) == 185.0
    assert float(cfg["runtime"]["sync_index_wait_seconds"]) == 205.0
    assert float(cfg["runtime"]["sync_context_wait_seconds"]) == 75.0
    assert str(cfg["runtime"]["sync_context_timeout_action"]) == "fallback_async"


def test_constraint_mode_alias_normalization(tmp_path: Path, monkeypatch) -> None:
    init_project(tmp_path)
    monkeypatch.setenv("ACCEL_CONSTRAINT_MODE", "enforce")

    cfg = resolve_effective_config(tmp_path)
    assert str(cfg["runtime"]["constraint_mode"]) == "strict"


def test_runtime_worker_defaults_support_auto(tmp_path: Path) -> None:
    init_project(tmp_path)
    local_cfg = {
        "runtime": {
            "max_workers": "auto",
            "verify_workers": "auto",
            "index_workers": "auto",
        }
    }
    (tmp_path / "accel.local.yaml").write_text(
        json.dumps(local_cfg, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    cfg = resolve_effective_config(tmp_path)
    cpu_count = max(1, int(os.cpu_count() or 1))
    assert int(cfg["runtime"]["max_workers"]) >= 1
    assert int(cfg["runtime"]["verify_workers"]) >= 1
    assert int(cfg["runtime"]["index_workers"]) >= 1
    assert int(cfg["runtime"]["max_workers"]) <= min(12, cpu_count)
    assert int(cfg["runtime"]["index_workers"]) <= min(96, cpu_count)


def test_gpu_policy_device_and_model_paths_env_overrides(tmp_path: Path, monkeypatch) -> None:
    init_project(tmp_path)
    monkeypatch.setenv("ACCEL_GPU_ENABLED", "1")
    monkeypatch.setenv("ACCEL_GPU_POLICY", "force")
    monkeypatch.setenv("ACCEL_GPU_DEVICE", "cuda:1")
    monkeypatch.setenv(
        "ACCEL_GPU_EMBEDDING_MODEL_PATH",
        r"C:\Users\dains\Models\models--BAAI--bge-m3\snapshots\5617a9f61b028005a4858fdac845db406aefb181",
    )
    monkeypatch.setenv(
        "ACCEL_GPU_RERANKER_MODEL_PATH",
        r"C:\Users\dains\Models\models--BAAI--bge-reranker-v2-m3\snapshots\953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e",
    )

    cfg = resolve_effective_config(tmp_path)
    gpu = dict(cfg.get("gpu", {}))
    assert bool(gpu.get("enabled", False)) is True
    assert str(gpu.get("policy", "")) == "force"
    assert str(gpu.get("device", "")) == "cuda:1"
    assert str(gpu.get("embedding_model_path", "")).endswith(
        r"models--BAAI--bge-m3\snapshots\5617a9f61b028005a4858fdac845db406aefb181"
    )
    assert str(gpu.get("reranker_model_path", "")).endswith(
        r"models--BAAI--bge-reranker-v2-m3\snapshots\953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e"
    )


def test_semantic_ranker_env_overrides(tmp_path: Path, monkeypatch) -> None:
    init_project(tmp_path)
    monkeypatch.setenv("ACCEL_SEMANTIC_RANKER_ENABLED", "1")
    monkeypatch.setenv("ACCEL_SEMANTIC_RANKER_PROVIDER", "auto")
    monkeypatch.setenv("ACCEL_SEMANTIC_RANKER_USE_ONNX", "1")
    monkeypatch.setenv("ACCEL_SEMANTIC_RANKER_MAX_CANDIDATES", "180")
    monkeypatch.setenv("ACCEL_SEMANTIC_RANKER_BATCH_SIZE", "12")
    monkeypatch.setenv("ACCEL_SEMANTIC_RANKER_EMBED_WEIGHT", "0.42")
    monkeypatch.setenv("ACCEL_SEMANTIC_RERANKER_ENABLED", "1")
    monkeypatch.setenv("ACCEL_SEMANTIC_RERANKER_TOP_K", "36")
    monkeypatch.setenv("ACCEL_SEMANTIC_RERANKER_WEIGHT", "0.27")

    cfg = resolve_effective_config(tmp_path)
    runtime = dict(cfg.get("runtime", {}))
    assert bool(runtime.get("semantic_ranker_enabled", False)) is True
    assert str(runtime.get("semantic_ranker_provider", "")) == "auto"
    assert bool(runtime.get("semantic_ranker_use_onnx", False)) is True
    assert int(runtime.get("semantic_ranker_max_candidates", 0)) == 180
    assert int(runtime.get("semantic_ranker_batch_size", 0)) == 12
    assert float(runtime.get("semantic_ranker_embed_weight", 0.0)) == 0.42
    assert bool(runtime.get("semantic_reranker_enabled", False)) is True
    assert int(runtime.get("semantic_reranker_top_k", 0)) == 36
    assert float(runtime.get("semantic_reranker_weight", 0.0)) == 0.27


def test_syntax_and_lexical_env_overrides(tmp_path: Path, monkeypatch) -> None:
    init_project(tmp_path)
    monkeypatch.setenv("ACCEL_SYNTAX_PARSER_ENABLED", "1")
    monkeypatch.setenv("ACCEL_SYNTAX_PARSER_PROVIDER", "auto")
    monkeypatch.setenv("ACCEL_LEXICAL_RANKER_ENABLED", "1")
    monkeypatch.setenv("ACCEL_LEXICAL_RANKER_PROVIDER", "tantivy")
    monkeypatch.setenv("ACCEL_LEXICAL_RANKER_MAX_CANDIDATES", "256")
    monkeypatch.setenv("ACCEL_LEXICAL_RANKER_WEIGHT", "0.35")

    cfg = resolve_effective_config(tmp_path)
    runtime = dict(cfg.get("runtime", {}))
    assert bool(runtime.get("syntax_parser_enabled", False)) is True
    assert str(runtime.get("syntax_parser_provider", "")) == "auto"
    assert bool(runtime.get("lexical_ranker_enabled", False)) is True
    assert str(runtime.get("lexical_ranker_provider", "")) == "tantivy"
    assert int(runtime.get("lexical_ranker_max_candidates", 0)) == 256
    assert float(runtime.get("lexical_ranker_weight", 0.0)) == 0.35


def test_syntax_and_lexical_defaults_enabled(tmp_path: Path) -> None:
    init_project(tmp_path)
    cfg = resolve_effective_config(tmp_path)
    runtime = dict(cfg.get("runtime", {}))
    assert bool(runtime.get("syntax_parser_enabled", False)) is True
    assert str(runtime.get("syntax_parser_provider", "")) == "auto"
    assert bool(runtime.get("lexical_ranker_enabled", False)) is True
    assert str(runtime.get("lexical_ranker_provider", "")) == "auto"


def test_default_accel_home_is_under_harborpilot_runtime(tmp_path: Path) -> None:
    init_project(tmp_path)
    cfg = resolve_effective_config(tmp_path)
    runtime = dict(cfg.get("runtime", {}))
    expected = (tmp_path / ".harborpilot" / "runtime" / "agent-accel").resolve()
    actual = Path(str(runtime.get("accel_home", ""))).resolve()
    assert actual == expected
