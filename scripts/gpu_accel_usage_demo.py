#!/usr/bin/env python3
"""End-to-end GPU acceleration demo using FlagEmbedding + optional reranker."""

from __future__ import annotations

import argparse
import importlib
import inspect
import json
import math
import os
import sys
import time
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


def _clamp_ratio(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _cosine_similarity(left: list[float], right: list[float]) -> float:
    size = min(len(left), len(right))
    if size <= 0:
        return 0.0
    dot = 0.0
    left_norm = 0.0
    right_norm = 0.0
    for idx in range(size):
        lv = float(left[idx])
        rv = float(right[idx])
        dot += lv * rv
        left_norm += lv * lv
        right_norm += rv * rv
    if left_norm <= 0.0 or right_norm <= 0.0:
        return 0.0
    return dot / math.sqrt(left_norm * right_norm)


def _normalize_scores(scores: list[float]) -> list[float]:
    if not scores:
        return []
    low = min(scores)
    high = max(scores)
    if abs(high - low) <= 1e-12:
        return [0.5 for _ in scores]
    span = high - low
    return [_clamp_ratio((float(item) - low) / span) for item in scores]


def _normalize_vector_rows(payload: Any) -> list[list[float]]:
    if hasattr(payload, "tolist"):
        payload = payload.tolist()
    if isinstance(payload, tuple):
        payload = list(payload)
    if not isinstance(payload, list):
        return []
    if payload and isinstance(payload[0], (int, float)):
        return [[float(item) for item in payload]]

    rows: list[list[float]] = []
    for row in payload:
        if hasattr(row, "tolist"):
            row = row.tolist()
        if isinstance(row, tuple):
            row = list(row)
        if not isinstance(row, list):
            continue
        rows.append([float(item) for item in row])
    return rows


def _filter_supported_kwargs(callable_obj: Any, kwargs: dict[str, Any]) -> dict[str, Any]:
    try:
        signature = inspect.signature(callable_obj)
    except (TypeError, ValueError):
        return dict(kwargs)
    parameters = dict(signature.parameters)
    if any(param.kind == inspect.Parameter.VAR_KEYWORD for param in parameters.values()):
        return dict(kwargs)
    accepted: dict[str, Any] = {}
    for key, value in kwargs.items():
        if key in parameters:
            accepted[key] = value
    return accepted


def _construct_with_fallback(ctor: Any, model_path: str, kwargs: dict[str, Any]) -> Any:
    attempts = [
        dict(kwargs),
        {key: value for key, value in kwargs.items() if key != "use_onnx"},
        {key: value for key, value in kwargs.items() if key not in {"device", "devices", "use_onnx"}},
    ]
    last_error: Exception | None = None
    for attempt in attempts:
        filtered = _filter_supported_kwargs(ctor, attempt)
        try:
            return ctor(model_path, **filtered)
        except TypeError as exc:
            last_error = exc
    if last_error is not None:
        raise last_error
    raise RuntimeError("model constructor failed with unknown error")


def _load_docs(args: argparse.Namespace) -> list[str]:
    docs = [str(item).strip() for item in list(args.doc or []) if str(item).strip()]
    if args.docs_json:
        payload = json.loads(Path(args.docs_json).read_text(encoding="utf-8"))
        if not isinstance(payload, list):
            raise ValueError("--docs-json must point to a JSON array of strings")
        docs.extend([str(item).strip() for item in payload if str(item).strip()])
    deduped: list[str] = []
    seen: set[str] = set()
    for item in docs:
        if item in seen:
            continue
        seen.add(item)
        deduped.append(item)
    return deduped


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a real inference flow demo for GPU-accelerated semantic retrieval/rerank.",
    )
    parser.add_argument("--embedding-model-path", required=True, help="Path to BGE-M3 model directory.")
    parser.add_argument("--reranker-model-path", default="", help="Path to reranker model directory.")
    parser.add_argument("--query", required=True, help="User query text for retrieval.")
    parser.add_argument(
        "--doc",
        action="append",
        default=[],
        help="Document candidate text (repeatable).",
    )
    parser.add_argument(
        "--docs-json",
        default="",
        help="Optional JSON file path containing an array of document strings.",
    )
    parser.add_argument("--gpu-enabled", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--policy", choices=["off", "auto", "force"], default="auto")
    parser.add_argument("--device", default="auto", help="auto/cpu/cuda[:idx]")
    parser.add_argument("--use-onnx", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--enable-rerank", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--embed-weight", type=float, default=0.7)
    parser.add_argument("--rerank-weight", type=float, default=0.3)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--output", choices=["json", "text"], default="json")
    return parser.parse_args()


def run_demo(args: argparse.Namespace) -> tuple[dict[str, Any], int]:
    from accel.gpu_runtime import resolve_gpu_runtime

    docs = _load_docs(args)
    if not docs:
        return {"status": "failed", "reason": "no_docs", "message": "No docs were provided."}, 2

    gpu_cfg = {
        "enabled": bool(args.gpu_enabled),
        "policy": str(args.policy),
        "device": str(args.device),
        "embedding_model_path": str(args.embedding_model_path),
        "reranker_model_path": str(args.reranker_model_path or ""),
    }
    gpu_runtime = resolve_gpu_runtime(gpu_cfg, raise_on_force_unavailable=False)
    effective_device = (
        str(gpu_runtime.get("effective_device", "cpu"))
        if bool(gpu_runtime.get("use_gpu", False))
        else "cpu"
    )
    use_fp16 = effective_device.startswith("cuda")

    try:
        flag = importlib.import_module("FlagEmbedding")
    except Exception as exc:
        return {
            "status": "failed",
            "reason": "flagembedding_import_failed",
            "error": f"{exc.__class__.__name__}: {exc}",
            "gpu_runtime": gpu_runtime,
        }, 2

    if bool(args.use_onnx):
        try:
            ort = importlib.import_module("onnxruntime")
            preload = getattr(ort, "preload_dlls", None)
            if callable(preload):
                preload()
        except Exception as exc:
            return {
                "status": "failed",
                "reason": "onnxruntime_import_failed",
                "error": f"{exc.__class__.__name__}: {exc}",
                "gpu_runtime": gpu_runtime,
            }, 2

    embedder_ctor = getattr(flag, "BGEM3FlagModel", None)
    if embedder_ctor is None:
        return {
            "status": "failed",
            "reason": "missing_BGEM3FlagModel",
            "gpu_runtime": gpu_runtime,
        }, 2

    timings: dict[str, float] = {}
    t0 = time.perf_counter()
    embedder = _construct_with_fallback(
        embedder_ctor,
        str(args.embedding_model_path),
        {
            "use_fp16": use_fp16,
            "device": effective_device,
            "devices": effective_device,
            "use_onnx": bool(args.use_onnx),
        },
    )
    timings["load_embedder_seconds"] = round(time.perf_counter() - t0, 6)

    reranker_obj: Any | None = None
    if bool(args.enable_rerank) and str(args.reranker_model_path).strip():
        reranker_ctor = getattr(flag, "FlagReranker", None)
        if reranker_ctor is not None:
            t1 = time.perf_counter()
            reranker_obj = _construct_with_fallback(
                reranker_ctor,
                str(args.reranker_model_path),
                {
                    "use_fp16": use_fp16,
                    "device": effective_device,
                    "devices": effective_device,
                    "use_onnx": bool(args.use_onnx),
                },
            )
            timings["load_reranker_seconds"] = round(time.perf_counter() - t1, 6)

    t2 = time.perf_counter()
    encode_payload: Any = embedder.encode(
        [str(args.query)] + docs,
        **_filter_supported_kwargs(embedder.encode, {"batch_size": max(1, int(args.batch_size))}),
    )
    if isinstance(encode_payload, dict):
        dense = encode_payload.get("dense_vecs")
        if dense is None:
            dense = encode_payload.get("dense_embeddings")
        if dense is None:
            dense = encode_payload.get("embeddings")
        encode_payload = dense
    vectors = _normalize_vector_rows(encode_payload)
    timings["encode_seconds"] = round(time.perf_counter() - t2, 6)

    if len(vectors) != len(docs) + 1:
        return {
            "status": "failed",
            "reason": "unexpected_embedding_shape",
            "vector_rows": len(vectors),
            "expected_rows": len(docs) + 1,
            "gpu_runtime": gpu_runtime,
        }, 2

    query_vec = vectors[0]
    doc_vecs = vectors[1:]
    embed_raw = [_cosine_similarity(query_vec, row) for row in doc_vecs]
    embed_norm = _normalize_scores(embed_raw)
    embed_weight = _clamp_ratio(float(args.embed_weight))

    rerank_raw: list[float] = []
    rerank_norm: list[float] = []
    if reranker_obj is not None:
        t3 = time.perf_counter()
        pairs = [[str(args.query), doc] for doc in docs]
        rr: Any = reranker_obj.compute_score(pairs)
        if hasattr(rr, "tolist"):
            rr = rr.tolist()
        if isinstance(rr, tuple):
            rr = list(rr)
        if isinstance(rr, list):
            rerank_raw = [float(item) for item in rr]
        elif isinstance(rr, (int, float)):
            rerank_raw = [float(rr)]
        rerank_norm = _normalize_scores(rerank_raw) if len(rerank_raw) == len(docs) else []
        timings["rerank_seconds"] = round(time.perf_counter() - t3, 6)

    rerank_weight = _clamp_ratio(float(args.rerank_weight))
    results: list[dict[str, Any]] = []
    for idx, doc in enumerate(docs):
        embed_score = embed_norm[idx] if idx < len(embed_norm) else 0.0
        rerank_score = rerank_norm[idx] if idx < len(rerank_norm) else 0.0
        if rerank_norm:
            final_score = ((1.0 - rerank_weight) * embed_score) + (
                rerank_weight * rerank_score
            )
        else:
            final_score = embed_score
        results.append(
            {
                "rank": 0,
                "doc": doc,
                "embed_raw_cosine": round(float(embed_raw[idx]), 6),
                "embed_score": round(float(embed_score), 6),
                "rerank_raw": round(float(rerank_raw[idx]), 6) if idx < len(rerank_raw) else None,
                "rerank_score": round(float(rerank_score), 6) if idx < len(rerank_norm) else None,
                "final_score": round(float(final_score), 6),
            }
        )

    results.sort(key=lambda item: (-float(item["final_score"]), str(item["doc"])))
    for idx, item in enumerate(results, start=1):
        item["rank"] = int(idx)

    top_k = max(1, int(args.top_k))
    payload = {
        "status": "ok",
        "timestamp_utc": _utc_now(),
        "runtime": {
            "gpu_runtime": gpu_runtime,
            "effective_device_for_models": effective_device,
            "use_fp16": use_fp16,
            "use_onnx": bool(args.use_onnx),
            "reranker_enabled": bool(reranker_obj is not None),
        },
        "query": str(args.query),
        "documents": int(len(docs)),
        "timings": timings,
        "weights": {
            "embed_weight": embed_weight,
            "rerank_weight": rerank_weight if rerank_norm else 0.0,
        },
        "top_k": top_k,
        "results": results[:top_k],
    }
    return payload, 0


def main() -> int:
    args = _parse_args()
    payload, exit_code = run_demo(args)
    if args.output == "json":
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        print(f"status: {payload.get('status', '')}")
        print(f"device: {payload.get('runtime', {}).get('effective_device_for_models', '')}")
        print(f"documents: {payload.get('documents', 0)}")
        for row in list(payload.get("results", [])):
            print(
                f"rank={row.get('rank', 0)} final={row.get('final_score', 0.0):.6f} "
                f"embed={row.get('embed_score', 0.0):.6f} "
                f"rerank={row.get('rerank_score', None)} doc={row.get('doc', '')}"
            )
    return int(exit_code)


if __name__ == "__main__":
    raise SystemExit(main())
