from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


DEFAULT_PROJECT_CONFIG: dict[str, Any] = {
    "version": 1,
    "project_id": "demo_project",
    "language_profiles": ["python", "typescript"],
    "index": {
        "scope_mode": "auto",
        "include": ["src/**", "accel/**", "tests/**"],
        "exclude": [
            ".git/**",
            "node_modules/**",
            "dist/**",
            "build/**",
            "target/**",
            ".venv/**",
            "venv/**",
            ".mypy_cache/**",
            ".pytest_cache/**",
            ".ruff_cache/**",
            ".next/**",
            ".turbo/**",
        ],
        "max_file_mb": 2,
    },
    "context": {
        "top_n_files": 12,
        "snippet_radius": 40,
        "max_chars": 24000,
        "max_snippets": 60,
    },
    "verify": {
        "python": ["python -m pytest -q", "python -m ruff check .", "python -m mypy --explicit-package-bases ."],
        "node": ["npm test --silent", "npm run lint", "npm run typecheck"],
    },
}

DEFAULT_LOCAL_CONFIG: dict[str, Any] = {
    "runtime": {
        "max_workers": 12,
        "verify_workers": 12,
        "index_workers": 96,
        "index_delta_compact_every": 200,
        "verify_max_target_tests": 64,
        "verify_pytest_shard_size": 16,
        "verify_pytest_max_shards": 6,
        "verify_fail_fast": False,
        "verify_cache_enabled": True,
        "verify_cache_failed_results": False,
        "verify_cache_ttl_seconds": 900,
        "verify_cache_failed_ttl_seconds": 120,
        "verify_cache_max_entries": 400,
        "verify_workspace_routing_enabled": True,
        "verify_preflight_enabled": True,
        "verify_preflight_timeout_seconds": 5,
        "sync_verify_timeout_action": "poll",
        "sync_verify_cancel_grace_seconds": 5.0,
        "token_estimator_backend": "auto",
        "token_estimator_encoding": "cl100k_base",
        "token_estimator_model": "",
        "token_estimator_calibration": 1.0,
        "token_estimator_fallback_chars_per_token": 4.0,
        "context_require_changed_files": False,
        "semantic_cache_enabled": True,
        "semantic_cache_mode": "hybrid",
        "semantic_cache_ttl_seconds": 7200,
        "semantic_cache_hybrid_threshold": 0.86,
        "semantic_cache_max_entries": 800,
        "command_plan_cache_enabled": True,
        "command_plan_cache_ttl_seconds": 900,
        "command_plan_cache_max_entries": 600,
        "constraint_mode": "warn",
        "rule_compression_enabled": True,
        "accel_home": "",
        "per_command_timeout_seconds": 1200,
        "total_verify_timeout_seconds": 3600,
    },
    "gpu": {"enabled": False, "policy": "off"},
}


def default_accel_home() -> Path:
    local_app_data = os.environ.get("LOCALAPPDATA")
    if local_app_data:
        return Path(local_app_data) / "agent-accel"
    return Path.home() / ".cache" / "agent-accel"


def _normalize_path(path: Path) -> Path:
    # Path.resolve() on some Windows/Python setups may produce duplicated segments.
    return Path(os.path.abspath(str(path)))


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged: dict[str, Any] = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _load_config_file(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return {}
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        try:
            import yaml  # type: ignore[import-untyped]

            # Add timeout protection for YAML loading
            import threading

            class TimeoutError(Exception):
                pass

            loaded = None
            timeout_error = None

            def load_yaml_with_timeout():
                nonlocal loaded, timeout_error
                try:
                    loaded = yaml.safe_load(text)
                except Exception as exc:
                    timeout_error = exc

            # Use threading for timeout (works on Windows and Unix)
            thread = threading.Thread(target=load_yaml_with_timeout)
            thread.daemon = True
            thread.start()
            thread.join(timeout=5.0)  # 5 second timeout

            if thread.is_alive():
                # Timeout occurred
                import logging
                logger = logging.getLogger("accel_config")
                logger.warning("YAML config loading timed out, using empty config")
                return {}

            if timeout_error:
                raise timeout_error

            if loaded is None:
                return {}
            if not isinstance(loaded, dict):
                raise ValueError(f"Config root must be an object: {path}")
            return loaded
        except Exception as exc:  # pragma: no cover - fallback guard
            raise ValueError(
                f"Failed to parse config file {path}. Use JSON-compatible YAML or install PyYAML."
            ) from exc
    if not isinstance(data, dict):
        raise ValueError(f"Config root must be an object: {path}")
    return data


def _normalize_max_workers(value: Any, default_value: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default_value
    return max(1, parsed)


def _normalize_positive_int(value: Any, default_value: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default_value
    return max(1, parsed)


def _normalize_positive_float(value: Any, default_value: float) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return float(default_value)
    if parsed <= 0:
        return float(default_value)
    return float(parsed)


def _normalize_bool(value: Any, default_value: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default_value
    text = str(value).strip().lower()
    return text in {"1", "true", "yes", "on"}


def _normalize_timeout_action(value: Any, default_value: str = "poll") -> str:
    token = str(value or default_value).strip().lower()
    if token in {"poll", "cancel"}:
        return token
    fallback = str(default_value or "poll").strip().lower()
    return fallback if fallback in {"poll", "cancel"} else "poll"


def _normalize_constraint_mode(value: Any, default_value: str = "warn") -> str:
    token = str(value or default_value).strip().lower()
    if token in {"off", "warn", "strict"}:
        return token
    fallback = str(default_value or "warn").strip().lower()
    return fallback if fallback in {"off", "warn", "strict"} else "warn"


def _apply_env_overrides(config: dict[str, Any]) -> dict[str, Any]:
    runtime = dict(config.get("runtime", {}))
    gpu = dict(config.get("gpu", {}))

    if os.environ.get("ACCEL_HOME"):
        runtime["accel_home"] = os.environ["ACCEL_HOME"]
    if os.environ.get("ACCEL_MAX_WORKERS"):
        runtime["max_workers"] = _normalize_max_workers(
            os.environ["ACCEL_MAX_WORKERS"], int(runtime.get("max_workers", 12))
        )
    if os.environ.get("ACCEL_VERIFY_WORKERS"):
        verify_workers_default = _normalize_positive_int(
            runtime.get("verify_workers", runtime.get("max_workers", 12)),
            12,
        )
        runtime["verify_workers"] = _normalize_positive_int(
            os.environ["ACCEL_VERIFY_WORKERS"],
            verify_workers_default,
        )
    if os.environ.get("ACCEL_INDEX_WORKERS"):
        runtime["index_workers"] = _normalize_positive_int(
            os.environ["ACCEL_INDEX_WORKERS"], int(runtime.get("index_workers", 96))
        )
    if os.environ.get("ACCEL_INDEX_COMPACT_EVERY"):
        runtime["index_delta_compact_every"] = _normalize_positive_int(
            os.environ["ACCEL_INDEX_COMPACT_EVERY"], int(runtime.get("index_delta_compact_every", 200))
        )
    if os.environ.get("ACCEL_VERIFY_MAX_TARGET_TESTS"):
        runtime["verify_max_target_tests"] = _normalize_positive_int(
            os.environ["ACCEL_VERIFY_MAX_TARGET_TESTS"], int(runtime.get("verify_max_target_tests", 64))
        )
    if os.environ.get("ACCEL_VERIFY_PYTEST_SHARD_SIZE"):
        runtime["verify_pytest_shard_size"] = _normalize_positive_int(
            os.environ["ACCEL_VERIFY_PYTEST_SHARD_SIZE"], int(runtime.get("verify_pytest_shard_size", 16))
        )
    if os.environ.get("ACCEL_VERIFY_PYTEST_MAX_SHARDS"):
        runtime["verify_pytest_max_shards"] = _normalize_positive_int(
            os.environ["ACCEL_VERIFY_PYTEST_MAX_SHARDS"], int(runtime.get("verify_pytest_max_shards", 6))
        )
    if os.environ.get("ACCEL_VERIFY_FAIL_FAST") is not None:
        runtime["verify_fail_fast"] = _normalize_bool(
            os.environ["ACCEL_VERIFY_FAIL_FAST"], bool(runtime.get("verify_fail_fast", False))
        )
    if os.environ.get("ACCEL_VERIFY_CACHE_ENABLED") is not None:
        runtime["verify_cache_enabled"] = _normalize_bool(
            os.environ["ACCEL_VERIFY_CACHE_ENABLED"], bool(runtime.get("verify_cache_enabled", True))
        )
    if os.environ.get("ACCEL_VERIFY_CACHE_FAILED_RESULTS") is not None:
        runtime["verify_cache_failed_results"] = _normalize_bool(
            os.environ["ACCEL_VERIFY_CACHE_FAILED_RESULTS"],
            bool(runtime.get("verify_cache_failed_results", False)),
        )
    if os.environ.get("ACCEL_VERIFY_CACHE_TTL_SECONDS"):
        runtime["verify_cache_ttl_seconds"] = _normalize_positive_int(
            os.environ["ACCEL_VERIFY_CACHE_TTL_SECONDS"], int(runtime.get("verify_cache_ttl_seconds", 900))
        )
    if os.environ.get("ACCEL_VERIFY_CACHE_FAILED_TTL_SECONDS"):
        runtime["verify_cache_failed_ttl_seconds"] = _normalize_positive_int(
            os.environ["ACCEL_VERIFY_CACHE_FAILED_TTL_SECONDS"],
            int(runtime.get("verify_cache_failed_ttl_seconds", 120)),
        )
    if os.environ.get("ACCEL_VERIFY_CACHE_MAX_ENTRIES"):
        runtime["verify_cache_max_entries"] = _normalize_positive_int(
            os.environ["ACCEL_VERIFY_CACHE_MAX_ENTRIES"], int(runtime.get("verify_cache_max_entries", 400))
        )
    if os.environ.get("ACCEL_VERIFY_WORKSPACE_ROUTING_ENABLED") is not None:
        runtime["verify_workspace_routing_enabled"] = _normalize_bool(
            os.environ["ACCEL_VERIFY_WORKSPACE_ROUTING_ENABLED"],
            bool(runtime.get("verify_workspace_routing_enabled", True)),
        )
    if os.environ.get("ACCEL_VERIFY_PREFLIGHT_ENABLED") is not None:
        runtime["verify_preflight_enabled"] = _normalize_bool(
            os.environ["ACCEL_VERIFY_PREFLIGHT_ENABLED"],
            bool(runtime.get("verify_preflight_enabled", True)),
        )
    if os.environ.get("ACCEL_VERIFY_PREFLIGHT_TIMEOUT_SECONDS"):
        runtime["verify_preflight_timeout_seconds"] = _normalize_positive_int(
            os.environ["ACCEL_VERIFY_PREFLIGHT_TIMEOUT_SECONDS"],
            int(runtime.get("verify_preflight_timeout_seconds", 5)),
        )
    if os.environ.get("ACCEL_TOKEN_ESTIMATOR_BACKEND"):
        runtime["token_estimator_backend"] = str(os.environ["ACCEL_TOKEN_ESTIMATOR_BACKEND"]).strip().lower()
    if os.environ.get("ACCEL_TOKEN_ESTIMATOR_ENCODING"):
        runtime["token_estimator_encoding"] = str(os.environ["ACCEL_TOKEN_ESTIMATOR_ENCODING"]).strip()
    if os.environ.get("ACCEL_TOKEN_ESTIMATOR_MODEL"):
        runtime["token_estimator_model"] = str(os.environ["ACCEL_TOKEN_ESTIMATOR_MODEL"]).strip()
    if os.environ.get("ACCEL_TOKEN_ESTIMATOR_CALIBRATION"):
        runtime["token_estimator_calibration"] = _normalize_positive_float(
            os.environ["ACCEL_TOKEN_ESTIMATOR_CALIBRATION"],
            float(runtime.get("token_estimator_calibration", 1.0)),
        )
    if os.environ.get("ACCEL_TOKEN_ESTIMATOR_FALLBACK_CHARS_PER_TOKEN"):
        runtime["token_estimator_fallback_chars_per_token"] = _normalize_positive_float(
            os.environ["ACCEL_TOKEN_ESTIMATOR_FALLBACK_CHARS_PER_TOKEN"],
            float(runtime.get("token_estimator_fallback_chars_per_token", 4.0)),
        )
    if os.environ.get("ACCEL_CONTEXT_REQUIRE_CHANGED_FILES") is not None:
        runtime["context_require_changed_files"] = _normalize_bool(
            os.environ["ACCEL_CONTEXT_REQUIRE_CHANGED_FILES"],
            bool(runtime.get("context_require_changed_files", False)),
        )
    if os.environ.get("ACCEL_SEMANTIC_CACHE_ENABLED") is not None:
        runtime["semantic_cache_enabled"] = _normalize_bool(
            os.environ["ACCEL_SEMANTIC_CACHE_ENABLED"],
            bool(runtime.get("semantic_cache_enabled", True)),
        )
    if os.environ.get("ACCEL_SEMANTIC_CACHE_MODE"):
        runtime["semantic_cache_mode"] = str(os.environ["ACCEL_SEMANTIC_CACHE_MODE"]).strip().lower()
    if os.environ.get("ACCEL_SEMANTIC_CACHE_TTL_SECONDS"):
        runtime["semantic_cache_ttl_seconds"] = _normalize_positive_int(
            os.environ["ACCEL_SEMANTIC_CACHE_TTL_SECONDS"],
            int(runtime.get("semantic_cache_ttl_seconds", 7200)),
        )
    if os.environ.get("ACCEL_SEMANTIC_CACHE_HYBRID_THRESHOLD"):
        runtime["semantic_cache_hybrid_threshold"] = _normalize_positive_float(
            os.environ["ACCEL_SEMANTIC_CACHE_HYBRID_THRESHOLD"],
            float(runtime.get("semantic_cache_hybrid_threshold", 0.86)),
        )
    if os.environ.get("ACCEL_SEMANTIC_CACHE_MAX_ENTRIES"):
        runtime["semantic_cache_max_entries"] = _normalize_positive_int(
            os.environ["ACCEL_SEMANTIC_CACHE_MAX_ENTRIES"],
            int(runtime.get("semantic_cache_max_entries", 800)),
        )
    if os.environ.get("ACCEL_COMMAND_PLAN_CACHE_ENABLED") is not None:
        runtime["command_plan_cache_enabled"] = _normalize_bool(
            os.environ["ACCEL_COMMAND_PLAN_CACHE_ENABLED"],
            bool(runtime.get("command_plan_cache_enabled", True)),
        )
    if os.environ.get("ACCEL_COMMAND_PLAN_CACHE_TTL_SECONDS"):
        runtime["command_plan_cache_ttl_seconds"] = _normalize_positive_int(
            os.environ["ACCEL_COMMAND_PLAN_CACHE_TTL_SECONDS"],
            int(runtime.get("command_plan_cache_ttl_seconds", 900)),
        )
    if os.environ.get("ACCEL_COMMAND_PLAN_CACHE_MAX_ENTRIES"):
        runtime["command_plan_cache_max_entries"] = _normalize_positive_int(
            os.environ["ACCEL_COMMAND_PLAN_CACHE_MAX_ENTRIES"],
            int(runtime.get("command_plan_cache_max_entries", 600)),
        )
    if os.environ.get("ACCEL_CONSTRAINT_MODE"):
        runtime["constraint_mode"] = _normalize_constraint_mode(
            os.environ["ACCEL_CONSTRAINT_MODE"],
            str(runtime.get("constraint_mode", "warn")),
        )
    if os.environ.get("ACCEL_RULE_COMPRESSION_ENABLED") is not None:
        runtime["rule_compression_enabled"] = _normalize_bool(
            os.environ["ACCEL_RULE_COMPRESSION_ENABLED"],
            bool(runtime.get("rule_compression_enabled", True)),
        )
    if os.environ.get("ACCEL_SYNC_VERIFY_TIMEOUT_ACTION"):
        runtime["sync_verify_timeout_action"] = _normalize_timeout_action(
            os.environ["ACCEL_SYNC_VERIFY_TIMEOUT_ACTION"],
            str(runtime.get("sync_verify_timeout_action", "poll")),
        )
    if os.environ.get("ACCEL_SYNC_VERIFY_CANCEL_GRACE_SECONDS"):
        runtime["sync_verify_cancel_grace_seconds"] = _normalize_positive_float(
            os.environ["ACCEL_SYNC_VERIFY_CANCEL_GRACE_SECONDS"],
            float(runtime.get("sync_verify_cancel_grace_seconds", 5.0)),
        )
    if os.environ.get("ACCEL_GPU_ENABLED") is not None:
        gpu["enabled"] = _normalize_bool(os.environ["ACCEL_GPU_ENABLED"], False)
    if os.environ.get("ACCEL_LOCAL_CONFIG"):
        config["meta"] = dict(config.get("meta", {}))
        config["meta"]["local_config_path"] = os.environ["ACCEL_LOCAL_CONFIG"]

    config["runtime"] = runtime
    config["gpu"] = gpu
    return config


def _validate_effective_config(config: dict[str, Any]) -> None:
    if int(config.get("version", 0)) <= 0:
        raise ValueError("version must be a positive integer")

    index = config.get("index", {})
    if not isinstance(index, dict):
        raise ValueError("index must be an object")
    index_scope_mode = str(index.get("scope_mode", "auto")).strip().lower()
    if index_scope_mode not in {"auto", "configured", "git", "git_tracked", "all"}:
        index_scope_mode = "auto"
    index["scope_mode"] = "git" if index_scope_mode == "git_tracked" else index_scope_mode
    include_raw = index.get("include", ["**/*"])
    if isinstance(include_raw, list):
        include_items = [str(item).strip() for item in include_raw if str(item).strip()]
    else:
        include_items = [str(include_raw).strip()] if str(include_raw).strip() else ["**/*"]
    index["include"] = include_items or ["**/*"]
    exclude_raw = index.get("exclude", [])
    if isinstance(exclude_raw, list):
        exclude_items = [str(item).strip() for item in exclude_raw if str(item).strip()]
    else:
        exclude_items = [str(exclude_raw).strip()] if str(exclude_raw).strip() else []
    index["exclude"] = exclude_items
    index["max_file_mb"] = _normalize_positive_int(index.get("max_file_mb", 2), default_value=2)
    index["max_files_to_scan"] = _normalize_positive_int(index.get("max_files_to_scan", 10000), default_value=10000)
    index["scan_timeout_seconds"] = _normalize_positive_int(index.get("scan_timeout_seconds", 60), default_value=60)
    config["index"] = index

    context = config.get("context", {})
    if not isinstance(context, dict):
        raise ValueError("context must be an object")
    for key in ("top_n_files", "snippet_radius", "max_chars", "max_snippets"):
        if int(context.get(key, 0)) <= 0:
            raise ValueError(f"context.{key} must be a positive integer")

    runtime = config.get("runtime", {})
    if not isinstance(runtime, dict):
        raise ValueError("runtime must be an object")
    runtime["max_workers"] = _normalize_max_workers(
        runtime.get("max_workers", 12), default_value=12
    )
    runtime["verify_workers"] = _normalize_positive_int(
        runtime.get("verify_workers", runtime.get("max_workers", 12)),
        default_value=12,
    )
    runtime["index_workers"] = _normalize_positive_int(
        runtime.get("index_workers", 96), default_value=96
    )
    runtime["index_delta_compact_every"] = _normalize_positive_int(
        runtime.get("index_delta_compact_every", 200), default_value=200
    )
    runtime["verify_max_target_tests"] = _normalize_positive_int(
        runtime.get("verify_max_target_tests", 64), default_value=64
    )
    runtime["verify_pytest_shard_size"] = _normalize_positive_int(
        runtime.get("verify_pytest_shard_size", 16), default_value=16
    )
    runtime["verify_pytest_max_shards"] = _normalize_positive_int(
        runtime.get("verify_pytest_max_shards", 6), default_value=6
    )
    runtime["verify_fail_fast"] = _normalize_bool(
        runtime.get("verify_fail_fast", False), default_value=False
    )
    runtime["verify_cache_enabled"] = _normalize_bool(
        runtime.get("verify_cache_enabled", True), default_value=True
    )
    runtime["verify_cache_failed_results"] = _normalize_bool(
        runtime.get("verify_cache_failed_results", False), default_value=False
    )
    runtime["verify_cache_ttl_seconds"] = _normalize_positive_int(
        runtime.get("verify_cache_ttl_seconds", 900), default_value=900
    )
    runtime["verify_cache_failed_ttl_seconds"] = _normalize_positive_int(
        runtime.get("verify_cache_failed_ttl_seconds", 120), default_value=120
    )
    runtime["verify_cache_max_entries"] = _normalize_positive_int(
        runtime.get("verify_cache_max_entries", 400), default_value=400
    )
    runtime["verify_workspace_routing_enabled"] = _normalize_bool(
        runtime.get("verify_workspace_routing_enabled", True),
        default_value=True,
    )
    runtime["verify_preflight_enabled"] = _normalize_bool(
        runtime.get("verify_preflight_enabled", True),
        default_value=True,
    )
    runtime["verify_preflight_timeout_seconds"] = _normalize_positive_int(
        runtime.get("verify_preflight_timeout_seconds", 5),
        default_value=5,
    )
    runtime["token_estimator_backend"] = str(runtime.get("token_estimator_backend", "auto")).strip().lower() or "auto"
    if runtime["token_estimator_backend"] not in {"auto", "tiktoken", "heuristic"}:
        runtime["token_estimator_backend"] = "auto"
    runtime["token_estimator_encoding"] = str(runtime.get("token_estimator_encoding", "cl100k_base")).strip() or "cl100k_base"
    runtime["token_estimator_model"] = str(runtime.get("token_estimator_model", "")).strip()
    runtime["token_estimator_calibration"] = _normalize_positive_float(
        runtime.get("token_estimator_calibration", 1.0),
        default_value=1.0,
    )
    runtime["token_estimator_fallback_chars_per_token"] = _normalize_positive_float(
        runtime.get("token_estimator_fallback_chars_per_token", 4.0),
        default_value=4.0,
    )
    runtime["context_require_changed_files"] = _normalize_bool(
        runtime.get("context_require_changed_files", False),
        default_value=False,
    )
    runtime["semantic_cache_enabled"] = _normalize_bool(
        runtime.get("semantic_cache_enabled", True),
        default_value=True,
    )
    runtime["semantic_cache_mode"] = str(runtime.get("semantic_cache_mode", "hybrid")).strip().lower()
    if runtime["semantic_cache_mode"] not in {"exact", "hybrid"}:
        runtime["semantic_cache_mode"] = "hybrid"
    runtime["semantic_cache_ttl_seconds"] = _normalize_positive_int(
        runtime.get("semantic_cache_ttl_seconds", 7200),
        default_value=7200,
    )
    runtime["semantic_cache_hybrid_threshold"] = _normalize_positive_float(
        runtime.get("semantic_cache_hybrid_threshold", 0.86),
        default_value=0.86,
    )
    if runtime["semantic_cache_hybrid_threshold"] > 1.0:
        runtime["semantic_cache_hybrid_threshold"] = 1.0
    runtime["semantic_cache_max_entries"] = _normalize_positive_int(
        runtime.get("semantic_cache_max_entries", 800),
        default_value=800,
    )
    runtime["command_plan_cache_enabled"] = _normalize_bool(
        runtime.get("command_plan_cache_enabled", True),
        default_value=True,
    )
    runtime["command_plan_cache_ttl_seconds"] = _normalize_positive_int(
        runtime.get("command_plan_cache_ttl_seconds", 900),
        default_value=900,
    )
    runtime["command_plan_cache_max_entries"] = _normalize_positive_int(
        runtime.get("command_plan_cache_max_entries", 600),
        default_value=600,
    )
    runtime["constraint_mode"] = _normalize_constraint_mode(
        runtime.get("constraint_mode", "warn"),
        default_value="warn",
    )
    runtime["rule_compression_enabled"] = _normalize_bool(
        runtime.get("rule_compression_enabled", True),
        default_value=True,
    )
    runtime["sync_verify_timeout_action"] = _normalize_timeout_action(
        runtime.get("sync_verify_timeout_action", "poll"),
        default_value="poll",
    )
    runtime["sync_verify_cancel_grace_seconds"] = _normalize_positive_float(
        runtime.get("sync_verify_cancel_grace_seconds", 5.0),
        default_value=5.0,
    )

    accel_home = runtime.get("accel_home")
    if not accel_home:
        runtime["accel_home"] = str(default_accel_home())
    config["runtime"] = runtime

    gpu = config.get("gpu", {})
    if not isinstance(gpu, dict):
        raise ValueError("gpu must be an object")
    gpu["enabled"] = _normalize_bool(gpu.get("enabled", False), False)
    config["gpu"] = gpu


def resolve_effective_config(
    project_dir: Path,
    cli_overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    project_dir = _normalize_path(project_dir)
    project_cfg_path = project_dir / "accel.yaml"
    env_local = os.environ.get("ACCEL_LOCAL_CONFIG")
    local_cfg_path = (
        _normalize_path(Path(env_local)) if env_local else (project_dir / "accel.local.yaml")
    )

    project_cfg = _load_config_file(project_cfg_path)
    local_cfg = _load_config_file(local_cfg_path)
    merged = _deep_merge(DEFAULT_PROJECT_CONFIG, project_cfg)
    merged = _deep_merge(merged, DEFAULT_LOCAL_CONFIG)
    merged = _deep_merge(merged, local_cfg)
    merged = _apply_env_overrides(merged)
    if cli_overrides:
        merged = _deep_merge(merged, cli_overrides)

    merged["meta"] = dict(merged.get("meta", {}))
    merged["meta"]["project_dir"] = str(project_dir)
    merged["meta"]["project_config_path"] = str(project_cfg_path)
    merged["meta"]["local_config_path"] = str(local_cfg_path)

    _validate_effective_config(merged)
    return merged


def _dump_json_as_yaml(path: Path, data: dict[str, Any]) -> None:
    # JSON is valid YAML. This keeps dependencies optional and files portable.
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def init_project(project_dir: Path, force: bool = False) -> dict[str, Any]:
    project_dir = _normalize_path(project_dir)
    project_dir.mkdir(parents=True, exist_ok=True)
    project_cfg_path = project_dir / "accel.yaml"
    local_example_path = project_dir / "accel.local.yaml.example"

    created: list[str] = []
    skipped: list[str] = []

    if force or not project_cfg_path.exists():
        _dump_json_as_yaml(project_cfg_path, DEFAULT_PROJECT_CONFIG)
        created.append(str(project_cfg_path))
    else:
        skipped.append(str(project_cfg_path))

    if force or not local_example_path.exists():
        local = json.loads(json.dumps(DEFAULT_LOCAL_CONFIG))
        local["runtime"]["accel_home"] = str(default_accel_home()).replace("\\", "/")
        _dump_json_as_yaml(local_example_path, local)
        created.append(str(local_example_path))
    else:
        skipped.append(str(local_example_path))

    gitignore_path = project_dir / ".gitignore"
    if gitignore_path.exists():
        existing = gitignore_path.read_text(encoding="utf-8")
    else:
        existing = ""
    if "accel.local.yaml" not in existing:
        suffix = "" if existing.endswith("\n") or not existing else "\n"
        gitignore_path.write_text(
            existing + suffix + "accel.local.yaml\n",
            encoding="utf-8",
        )
        created.append(str(gitignore_path))

    return {"created": created, "skipped": skipped}
