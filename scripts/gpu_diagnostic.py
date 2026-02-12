#!/usr/bin/env python3
"""GPU/semantic runtime diagnostic script for agent-accel."""

from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault("PYTHONUTF8", "1")
os.environ.setdefault("PYTHONIOENCODING", "utf-8")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _make_config(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "runtime": {
            "semantic_ranker_enabled": bool(args.semantic_enabled),
            "semantic_ranker_provider": str(args.provider),
            "semantic_ranker_use_onnx": bool(args.use_onnx),
            "semantic_ranker_batch_size": int(args.batch_size),
            "semantic_reranker_enabled": bool(args.enable_reranker),
        },
        "gpu": {
            "enabled": bool(args.gpu_enabled),
            "policy": str(args.policy),
            "device": str(args.device),
            "embedding_model_path": str(args.embedding_model_path or ""),
            "reranker_model_path": str(args.reranker_model_path or ""),
        },
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run GPU/semantic diagnostic checks for agent-accel.",
    )
    parser.add_argument(
        "--embedding-model-path",
        default=os.environ.get("ACCEL_GPU_EMBEDDING_MODEL_PATH", ""),
        help="Local embedding model path (defaults to ACCEL_GPU_EMBEDDING_MODEL_PATH).",
    )
    parser.add_argument(
        "--reranker-model-path",
        default=os.environ.get("ACCEL_GPU_RERANKER_MODEL_PATH", ""),
        help="Local reranker model path (defaults to ACCEL_GPU_RERANKER_MODEL_PATH).",
    )
    parser.add_argument(
        "--gpu-enabled",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable GPU runtime decision path in this diagnostic config.",
    )
    parser.add_argument(
        "--semantic-enabled",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable semantic ranker path in this diagnostic config.",
    )
    parser.add_argument(
        "--enable-reranker",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable reranker in runtime config (requires reranker model path).",
    )
    parser.add_argument(
        "--policy",
        choices=["off", "auto", "force"],
        default="auto",
        help="GPU policy for diagnostic runtime config.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="GPU device for diagnostic runtime config (auto/cpu/cuda[:idx]).",
    )
    parser.add_argument(
        "--provider",
        choices=["off", "auto", "flagembedding"],
        default="auto",
        help="Semantic provider for diagnostic runtime config.",
    )
    parser.add_argument(
        "--use-onnx",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable ONNX runtime path for semantic runtime.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Batch size for encode smoke test.",
    )
    parser.add_argument(
        "--run-encode",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run embedding encode smoke test when semantic runtime is ready.",
    )
    parser.add_argument(
        "--run-rerank",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Run reranker smoke test when reranker is available.",
    )
    parser.add_argument(
        "--require-gpu",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Return non-zero if effective runtime is not GPU.",
    )
    parser.add_argument(
        "--require-ready",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Return non-zero if semantic runtime reason is not 'ready'.",
    )
    parser.add_argument(
        "--require-encode-success",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Return non-zero unless encode smoke test succeeds.",
    )
    parser.add_argument(
        "--require-rerank-success",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Return non-zero unless rerank smoke test succeeds.",
    )
    parser.add_argument(
        "--output",
        choices=["json", "text"],
        default="json",
        help="Output format.",
    )
    parser.add_argument(
        "--show-traceback",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Include Python traceback when an inference step fails.",
    )
    return parser.parse_args()


def _run(args: argparse.Namespace) -> tuple[dict[str, Any], int]:
    from accel.gpu_runtime import resolve_gpu_runtime
    from accel.semantic_ranker import _build_runtime, probe_semantic_runtime

    config = _make_config(args)
    gpu_runtime = resolve_gpu_runtime(config.get("gpu", {}), raise_on_force_unavailable=False)
    semantic_runtime = probe_semantic_runtime(config)

    result: dict[str, Any] = {
        "timestamp_utc": _utc_now(),
        "input": {
            "gpu_enabled": bool(args.gpu_enabled),
            "semantic_enabled": bool(args.semantic_enabled),
            "policy": str(args.policy),
            "device": str(args.device),
            "provider": str(args.provider),
            "use_onnx": bool(args.use_onnx),
            "embedding_model_path": str(args.embedding_model_path or ""),
            "reranker_model_path": str(args.reranker_model_path or ""),
        },
        "gpu_runtime": gpu_runtime,
        "semantic_runtime": semantic_runtime,
        "smoke": {
            "encode": {"status": "not_run"},
            "rerank": {"status": "not_run"},
        },
        "checks": [],
    }

    runtime_obj: Any | None = None
    runtime_reason = str(semantic_runtime.get("reason", ""))
    if runtime_reason == "ready":
        try:
            runtime_obj, _ = _build_runtime(config, semantic_runtime)
            if runtime_obj is None:
                runtime_reason = "runtime_not_built"
        except Exception as exc:
            runtime_reason = f"runtime_build_failed:{exc.__class__.__name__}"
            if args.show_traceback:
                result["runtime_traceback"] = traceback.format_exc()
            else:
                result["runtime_error"] = str(exc)
    result["semantic_runtime_build_reason"] = runtime_reason

    if bool(args.run_encode):
        encode_block: dict[str, Any] = {"status": "skipped_not_ready"}
        if runtime_obj is not None and runtime_reason == "ready":
            try:
                texts = [
                    "agent-accel gpu diagnostic query",
                    "semantic embedding smoke test input one",
                    "semantic embedding smoke test input two",
                ]
                vectors = runtime_obj.encode_documents(texts, batch_size=max(1, int(args.batch_size)))
                vector_dim = len(vectors[0]) if vectors and isinstance(vectors[0], list) else 0
                encode_block = {
                    "status": "success",
                    "rows": len(vectors),
                    "vector_dim": int(vector_dim),
                }
            except Exception as exc:
                encode_block = {
                    "status": "failed",
                    "error": f"{exc.__class__.__name__}: {exc}",
                }
                if args.show_traceback:
                    encode_block["traceback"] = traceback.format_exc()
        result["smoke"]["encode"] = encode_block

    if bool(args.run_rerank):
        rerank_block: dict[str, Any] = {"status": "skipped_not_ready"}
        if runtime_obj is not None and runtime_reason == "ready":
            if getattr(runtime_obj, "reranker", None) is None:
                rerank_block = {"status": "skipped_no_reranker"}
            else:
                try:
                    scores = runtime_obj.rerank_documents(
                        "rank semantic relevance",
                        [
                            "doc one about semantic ranking",
                            "doc two about unrelated topic",
                        ],
                    )
                    rerank_block = {
                        "status": "success",
                        "scores": [float(item) for item in scores],
                    }
                except Exception as exc:
                    rerank_block = {
                        "status": "failed",
                        "error": f"{exc.__class__.__name__}: {exc}",
                    }
                    if args.show_traceback:
                        rerank_block["traceback"] = traceback.format_exc()
        result["smoke"]["rerank"] = rerank_block

    checks: list[dict[str, Any]] = []

    if bool(args.require_gpu):
        checks.append(
            {
                "name": "require_gpu",
                "passed": bool(gpu_runtime.get("use_gpu", False)),
                "detail": str(gpu_runtime.get("reason", "")),
            }
        )
    if bool(args.require_ready):
        checks.append(
            {
                "name": "require_ready",
                "passed": str(semantic_runtime.get("reason", "")) == "ready",
                "detail": str(semantic_runtime.get("reason", "")),
            }
        )
    if bool(args.require_encode_success):
        checks.append(
            {
                "name": "require_encode_success",
                "passed": str(result["smoke"]["encode"].get("status", "")) == "success",
                "detail": str(result["smoke"]["encode"].get("status", "")),
            }
        )
    if bool(args.require_rerank_success):
        checks.append(
            {
                "name": "require_rerank_success",
                "passed": str(result["smoke"]["rerank"].get("status", "")) == "success",
                "detail": str(result["smoke"]["rerank"].get("status", "")),
            }
        )

    result["checks"] = checks
    passed = all(bool(item.get("passed", False)) for item in checks)
    result["status"] = "ok" if passed else "failed"
    exit_code = 0 if passed else 1
    return result, exit_code


def main() -> int:
    args = _parse_args()
    payload, exit_code = _run(args)
    if args.output == "json":
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        print(f"status: {payload.get('status', '')}")
        print(f"gpu_use_gpu: {payload.get('gpu_runtime', {}).get('use_gpu', False)}")
        print(f"gpu_reason: {payload.get('gpu_runtime', {}).get('reason', '')}")
        print(f"semantic_reason: {payload.get('semantic_runtime', {}).get('reason', '')}")
        print(f"encode_status: {payload.get('smoke', {}).get('encode', {}).get('status', '')}")
        print(f"rerank_status: {payload.get('smoke', {}).get('rerank', {}).get('status', '')}")
        if payload.get("checks"):
            for item in list(payload.get("checks", [])):
                print(
                    f"check[{item.get('name', '')}]: "
                    f"{'passed' if bool(item.get('passed', False)) else 'failed'} "
                    f"({item.get('detail', '')})"
                )
    return int(exit_code)


if __name__ == "__main__":
    raise SystemExit(main())
