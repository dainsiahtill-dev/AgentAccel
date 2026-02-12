from __future__ import annotations

import importlib
import inspect
import math
import threading
from pathlib import Path
from typing import Any

from .gpu_runtime import resolve_gpu_runtime


_SEMANTIC_PROVIDERS = {"off", "auto", "flagembedding"}
_RUNTIME_CACHE_LOCK = threading.Lock()
_RUNTIME_CACHE: dict[str, "_FlagEmbeddingRuntime"] = {}


def normalize_semantic_provider(value: Any, default_value: str = "off") -> str:
    token = str(value or default_value).strip().lower()
    if token in _SEMANTIC_PROVIDERS:
        return token
    fallback = str(default_value or "off").strip().lower()
    return fallback if fallback in _SEMANTIC_PROVIDERS else "off"


def clamp_ratio(value: Any, default_value: float) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return float(default_value)
    if parsed < 0.0:
        return 0.0
    if parsed > 1.0:
        return 1.0
    return float(parsed)


def _cosine_similarity(left: list[float], right: list[float]) -> float:
    if not left or not right:
        return 0.0
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


def _normalize_dense_scores(scores: list[float]) -> list[float]:
    if not scores:
        return []
    low = min(scores)
    high = max(scores)
    if abs(high - low) <= 1e-9:
        return [0.5 for _ in scores]
    scale = high - low
    return [max(0.0, min(1.0, (float(value) - low) / scale)) for value in scores]


def _filter_supported_kwargs(callable_obj: Any, kwargs: dict[str, Any]) -> dict[str, Any]:
    try:
        signature = inspect.signature(callable_obj)
    except (TypeError, ValueError):
        return dict(kwargs)
    accepted: dict[str, Any] = {}
    parameters = dict(signature.parameters)
    if any(param.kind == inspect.Parameter.VAR_KEYWORD for param in parameters.values()):
        return dict(kwargs)
    for key, value in kwargs.items():
        if key in parameters:
            accepted[key] = value
    return accepted


def _try_construct(callable_obj: Any, model_path: str, kwargs: dict[str, Any]) -> Any:
    attempts = [
        dict(kwargs),
        {key: value for key, value in kwargs.items() if key != "use_onnx"},
        {key: value for key, value in kwargs.items() if key not in {"use_onnx", "device", "devices"}},
    ]
    for attempt in attempts:
        filtered = _filter_supported_kwargs(callable_obj, attempt)
        try:
            return callable_obj(model_path, **filtered)
        except TypeError:
            continue
    raise TypeError("constructor signature mismatch")


class _FlagEmbeddingRuntime:
    def __init__(
        self,
        *,
        embedder: Any,
        reranker: Any | None,
        provider: str,
        device: str,
        use_onnx: bool,
        embedding_model_path: str,
        reranker_model_path: str,
    ) -> None:
        self.embedder = embedder
        self.reranker = reranker
        self.provider = provider
        self.device = device
        self.use_onnx = bool(use_onnx)
        self.embedding_model_path = embedding_model_path
        self.reranker_model_path = reranker_model_path

    def encode_documents(self, texts: list[str], *, batch_size: int) -> list[list[float]]:
        kwargs = {"batch_size": max(1, int(batch_size))}
        try:
            encoded = self.embedder.encode(texts, **_filter_supported_kwargs(self.embedder.encode, kwargs))
        except TypeError:
            encoded = self.embedder.encode(texts)
        if isinstance(encoded, dict):
            dense = encoded.get("dense_vecs")
            if dense is None:
                dense = encoded.get("dense_embeddings")
            if dense is None:
                dense = encoded.get("embeddings")
            encoded = dense
        return _normalize_vector_rows(encoded)

    def rerank_documents(self, query: str, docs: list[str]) -> list[float]:
        if self.reranker is None:
            return []
        pairs = [[query, doc] for doc in docs]
        scores: Any = self.reranker.compute_score(pairs)
        if hasattr(scores, "tolist"):
            scores = scores.tolist()
        if isinstance(scores, tuple):
            scores = list(scores)
        if isinstance(scores, list):
            return [float(item) for item in scores]
        if isinstance(scores, (int, float)):
            return [float(scores)]
        return []


def _load_flagembedding_module() -> Any:
    return importlib.import_module("FlagEmbedding")


def _load_onnxruntime_module() -> Any:
    return importlib.import_module("onnxruntime")


def _resolve_provider(runtime_cfg: dict[str, Any]) -> str:
    provider = normalize_semantic_provider(runtime_cfg.get("semantic_ranker_provider", "off"), "off")
    if provider == "auto":
        return "flagembedding"
    return provider


def probe_semantic_runtime(config: dict[str, Any]) -> dict[str, Any]:
    runtime_cfg = dict(config.get("runtime", {}))
    gpu_cfg = dict(config.get("gpu", {}))
    enabled = bool(runtime_cfg.get("semantic_ranker_enabled", False))
    requested_provider = normalize_semantic_provider(
        runtime_cfg.get("semantic_ranker_provider", "off"),
        "off",
    )
    provider = _resolve_provider(runtime_cfg)
    use_onnx = bool(runtime_cfg.get("semantic_ranker_use_onnx", False))
    embedding_model_path = str(gpu_cfg.get("embedding_model_path", ""))
    reranker_model_path = str(gpu_cfg.get("reranker_model_path", ""))
    embedding_exists = bool(embedding_model_path and Path(embedding_model_path).exists())
    reranker_exists = bool(reranker_model_path and Path(reranker_model_path).exists())
    gpu_runtime = resolve_gpu_runtime(gpu_cfg, raise_on_force_unavailable=False)

    flagembedding_available = False
    onnxruntime_available = False
    onnx_providers: list[str] = []
    onnx_reason = ""
    try:
        _load_flagembedding_module()
        flagembedding_available = True
    except Exception as exc:
        flagembedding_available = False
        onnx_reason = f"flagembedding_unavailable:{exc.__class__.__name__}"

    if use_onnx:
        try:
            ort = _load_onnxruntime_module()
            onnxruntime_available = True
            providers = getattr(ort, "get_available_providers", None)
            if callable(providers):
                rows = providers()
                if isinstance(rows, list):
                    onnx_providers = [str(item) for item in rows]
        except Exception as exc:
            onnxruntime_available = False
            onnx_reason = f"onnxruntime_unavailable:{exc.__class__.__name__}"

    reason = "disabled_by_config"
    if enabled:
        if provider == "off":
            reason = "provider_off"
        elif provider != "flagembedding":
            reason = "provider_unsupported"
        elif not flagembedding_available:
            reason = "flagembedding_unavailable"
        elif not embedding_exists:
            reason = "embedding_model_missing"
        elif use_onnx and not onnxruntime_available:
            reason = "onnxruntime_unavailable"
        else:
            reason = "ready"

    return {
        "enabled": enabled,
        "provider_requested": requested_provider,
        "provider_resolved": provider,
        "use_onnx": use_onnx,
        "flagembedding_available": flagembedding_available,
        "onnxruntime_available": onnxruntime_available,
        "onnx_providers": onnx_providers,
        "reason": reason,
        "reason_detail": onnx_reason,
        "embedding_model_path": embedding_model_path,
        "reranker_model_path": reranker_model_path,
        "embedding_model_exists": embedding_exists,
        "reranker_model_exists": reranker_exists,
        "gpu_runtime": gpu_runtime,
    }


def _runtime_cache_key(
    *,
    embedding_model_path: str,
    reranker_model_path: str,
    device: str,
    use_onnx: bool,
    reranker_enabled: bool,
) -> str:
    return "|".join(
        [
            embedding_model_path.strip(),
            reranker_model_path.strip(),
            device.strip(),
            "onnx" if use_onnx else "native",
            "rerank" if reranker_enabled else "embed_only",
        ]
    )


def _build_runtime(config: dict[str, Any], probe: dict[str, Any]) -> tuple[_FlagEmbeddingRuntime | None, str]:
    reason = str(probe.get("reason", ""))
    if reason != "ready":
        return None, reason

    runtime_cfg = dict(config.get("runtime", {}))
    gpu_cfg = dict(config.get("gpu", {}))
    embedding_model_path = str(gpu_cfg.get("embedding_model_path", ""))
    reranker_model_path = str(gpu_cfg.get("reranker_model_path", ""))
    use_onnx = bool(runtime_cfg.get("semantic_ranker_use_onnx", False))
    reranker_enabled = bool(runtime_cfg.get("semantic_reranker_enabled", False))

    gpu_runtime = dict(probe.get("gpu_runtime", {}))
    if bool(gpu_runtime.get("use_gpu", False)):
        device = str(gpu_runtime.get("effective_device", "cpu"))
    else:
        device = "cpu"
    use_fp16 = device.startswith("cuda")

    cache_key = _runtime_cache_key(
        embedding_model_path=embedding_model_path,
        reranker_model_path=reranker_model_path,
        device=device,
        use_onnx=use_onnx,
        reranker_enabled=reranker_enabled,
    )
    with _RUNTIME_CACHE_LOCK:
        cached = _RUNTIME_CACHE.get(cache_key)
        if cached is not None:
            return cached, "ready"

    flag = _load_flagembedding_module()
    if use_onnx:
        ort = _load_onnxruntime_module()
        preload = getattr(ort, "preload_dlls", None)
        if callable(preload):
            preload()

    embedder_ctor = getattr(flag, "BGEM3FlagModel", None)
    if embedder_ctor is None:
        raise RuntimeError("FlagEmbedding.BGEM3FlagModel is unavailable")
    embedder = _try_construct(
        embedder_ctor,
        embedding_model_path,
        {
            "use_fp16": use_fp16,
            "device": device,
            "devices": device,
            "use_onnx": use_onnx,
        },
    )

    reranker_obj: Any | None = None
    if reranker_enabled and reranker_model_path:
        reranker_ctor = getattr(flag, "FlagReranker", None)
        if reranker_ctor is not None:
            reranker_obj = _try_construct(
                reranker_ctor,
                reranker_model_path,
                {
                    "use_fp16": use_fp16,
                    "device": device,
                    "devices": device,
                    "use_onnx": use_onnx,
                },
            )

    runtime = _FlagEmbeddingRuntime(
        embedder=embedder,
        reranker=reranker_obj,
        provider="flagembedding",
        device=device,
        use_onnx=use_onnx,
        embedding_model_path=embedding_model_path,
        reranker_model_path=reranker_model_path,
    )
    with _RUNTIME_CACHE_LOCK:
        _RUNTIME_CACHE[cache_key] = runtime
    return runtime, "ready"


def _build_doc_text(
    *,
    file_path: str,
    symbols_by_file: dict[str, list[dict[str, Any]]],
    references_by_file: dict[str, list[dict[str, Any]]],
    deps_by_file: dict[str, list[dict[str, Any]]],
    tests_by_file: dict[str, list[dict[str, Any]]],
) -> str:
    symbol_rows = symbols_by_file.get(file_path, [])
    ref_rows = references_by_file.get(file_path, [])
    dep_rows = deps_by_file.get(file_path, [])
    test_rows = tests_by_file.get(file_path, [])

    symbols: list[str] = []
    refs: list[str] = []
    deps: list[str] = []
    tests: list[str] = []
    for row in symbol_rows[:40]:
        symbol = str(row.get("symbol", "")).strip()
        qn = str(row.get("qualified_name", "")).strip()
        if symbol:
            symbols.append(symbol)
        if qn:
            symbols.append(qn)
    for row in ref_rows[:40]:
        target = str(row.get("target_symbol", "")).strip()
        if target:
            refs.append(target)
    for row in dep_rows[:30]:
        edge_to = str(row.get("edge_to", "")).strip()
        if edge_to:
            deps.append(edge_to)
    for row in test_rows[:20]:
        test_file = str(row.get("test_file", "")).strip()
        if test_file:
            tests.append(test_file)

    parts = [f"path: {file_path}"]
    if symbols:
        parts.append("symbols: " + " ".join(symbols[:30]))
    if refs:
        parts.append("references: " + " ".join(refs[:30]))
    if deps:
        parts.append("dependencies: " + " ".join(deps[:20]))
    if tests:
        parts.append("tests: " + " ".join(tests[:15]))
    return "\n".join(parts)


def apply_semantic_ranking(
    *,
    ranked: list[dict[str, Any]],
    config: dict[str, Any],
    task: str,
    task_tokens: list[str],
    hints: list[str] | None,
    symbols_by_file: dict[str, list[dict[str, Any]]],
    references_by_file: dict[str, list[dict[str, Any]]],
    deps_by_file: dict[str, list[dict[str, Any]]],
    tests_by_file: dict[str, list[dict[str, Any]]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    probe = probe_semantic_runtime(config)
    metadata: dict[str, Any] = {
        "enabled": bool(probe.get("enabled", False)),
        "provider_requested": str(probe.get("provider_requested", "off")),
        "provider_resolved": str(probe.get("provider_resolved", "off")),
        "reason": str(probe.get("reason", "disabled_by_config")),
        "applied": False,
        "embedding_applied": False,
        "reranker_applied": False,
        "candidate_count": 0,
        "device": str(dict(probe.get("gpu_runtime", {})).get("effective_device", "cpu")),
        "use_onnx": bool(probe.get("use_onnx", False)),
    }
    if str(probe.get("reason", "")) != "ready":
        return ranked, metadata

    try:
        runtime, runtime_reason = _build_runtime(config, probe)
    except Exception as exc:
        metadata["reason"] = f"runtime_init_failed:{exc.__class__.__name__}"
        return ranked, metadata
    if runtime is None:
        metadata["reason"] = runtime_reason
        return ranked, metadata

    runtime_cfg = dict(config.get("runtime", {}))
    max_candidates = max(1, int(runtime_cfg.get("semantic_ranker_max_candidates", 120)))
    batch_size = max(1, int(runtime_cfg.get("semantic_ranker_batch_size", 16)))
    embed_weight = clamp_ratio(runtime_cfg.get("semantic_ranker_embed_weight", 0.3), 0.3)
    rerank_enabled = bool(runtime_cfg.get("semantic_reranker_enabled", False))
    rerank_top_k = max(1, int(runtime_cfg.get("semantic_reranker_top_k", 30)))
    rerank_weight = clamp_ratio(runtime_cfg.get("semantic_reranker_weight", 0.15), 0.15)

    candidates = [dict(item) for item in ranked[:max_candidates]]
    if not candidates:
        metadata["reason"] = "no_candidates"
        return ranked, metadata

    hint_tokens = [str(item).strip() for item in (hints or []) if str(item).strip()]
    query_parts = [str(task).strip(), " ".join(task_tokens), " ".join(hint_tokens)]
    query_text = "\n".join([item for item in query_parts if item]).strip()
    if not query_text:
        metadata["reason"] = "empty_query"
        return ranked, metadata

    doc_texts = [
        _build_doc_text(
            file_path=str(item.get("path", "")),
            symbols_by_file=symbols_by_file,
            references_by_file=references_by_file,
            deps_by_file=deps_by_file,
            tests_by_file=tests_by_file,
        )
        for item in candidates
    ]

    try:
        vectors = runtime.encode_documents([query_text] + doc_texts, batch_size=batch_size)
    except Exception as exc:
        metadata["reason"] = f"embedding_failed:{exc.__class__.__name__}"
        return ranked, metadata
    if len(vectors) != len(doc_texts) + 1:
        metadata["reason"] = "embedding_shape_mismatch"
        return ranked, metadata

    query_vec = vectors[0]
    doc_vecs = vectors[1:]
    semantic_scores: list[float] = []
    for vec in doc_vecs:
        cosine = _cosine_similarity(query_vec, vec)
        semantic_scores.append(max(0.0, min(1.0, (cosine + 1.0) / 2.0)))

    for idx, item in enumerate(candidates):
        lexical_score = float(item.get("score", 0.0))
        semantic_score = semantic_scores[idx] if idx < len(semantic_scores) else 0.0
        blended = ((1.0 - embed_weight) * lexical_score) + (embed_weight * semantic_score)
        item["score"] = round(blended, 6)
        reasons = list(item.get("reasons", []))
        if "semantic_embedding" not in reasons:
            reasons.append("semantic_embedding")
        item["reasons"] = reasons
        signals = list(item.get("signals", []))
        signals.append(
            {
                "signal_name": "semantic_embedding_similarity",
                "score": round(semantic_score, 6),
            }
        )
        item["signals"] = signals

    candidates.sort(key=lambda row: (-float(row.get("score", 0.0)), str(row.get("path", ""))))
    metadata["embedding_applied"] = True

    if rerank_enabled and runtime.reranker is not None:
        head = candidates[: min(rerank_top_k, len(candidates))]
        head_docs = [
            _build_doc_text(
                file_path=str(item.get("path", "")),
                symbols_by_file=symbols_by_file,
                references_by_file=references_by_file,
                deps_by_file=deps_by_file,
                tests_by_file=tests_by_file,
            )
            for item in head
        ]
        try:
            rerank_scores_raw = runtime.rerank_documents(query_text, head_docs)
        except Exception as exc:
            metadata["reason"] = f"reranker_failed:{exc.__class__.__name__}"
            rerank_scores_raw = []
        if len(rerank_scores_raw) == len(head):
            rerank_scores = _normalize_dense_scores([float(item) for item in rerank_scores_raw])
            for idx, item in enumerate(head):
                current_score = float(item.get("score", 0.0))
                rerank_score = rerank_scores[idx]
                blended = ((1.0 - rerank_weight) * current_score) + (
                    rerank_weight * rerank_score
                )
                item["score"] = round(blended, 6)
                reasons = list(item.get("reasons", []))
                if "semantic_reranker" not in reasons:
                    reasons.append("semantic_reranker")
                item["reasons"] = reasons
                signals = list(item.get("signals", []))
                signals.append(
                    {
                        "signal_name": "semantic_reranker_score",
                        "score": round(rerank_score, 6),
                    }
                )
                item["signals"] = signals
            candidates[: len(head)] = head
            candidates.sort(key=lambda row: (-float(row.get("score", 0.0)), str(row.get("path", ""))))
            metadata["reranker_applied"] = True

    updated: dict[str, dict[str, Any]] = {
        str(item.get("path", "")): item for item in candidates if str(item.get("path", ""))
    }
    merged = [updated.get(str(item.get("path", "")), dict(item)) for item in ranked]
    merged.sort(key=lambda row: (-float(row.get("score", 0.0)), str(row.get("path", ""))))
    metadata["applied"] = True
    metadata["reason"] = "applied"
    metadata["candidate_count"] = int(len(candidates))
    metadata["provider_resolved"] = runtime.provider
    metadata["device"] = runtime.device
    metadata["use_onnx"] = runtime.use_onnx
    return merged, metadata

