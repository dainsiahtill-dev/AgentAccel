from __future__ import annotations

from typing import Any

import accel.semantic_ranker as semantic_ranker


def _base_config() -> dict[str, Any]:
    return {
        "runtime": {
            "semantic_ranker_enabled": True,
            "semantic_ranker_provider": "auto",
            "semantic_ranker_use_onnx": False,
            "semantic_ranker_max_candidates": 12,
            "semantic_ranker_batch_size": 8,
            "semantic_ranker_embed_weight": 0.5,
            "semantic_reranker_enabled": False,
            "semantic_reranker_top_k": 4,
            "semantic_reranker_weight": 0.2,
        },
        "gpu": {
            "enabled": False,
            "policy": "off",
            "device": "auto",
            "embedding_model_path": "",
            "reranker_model_path": "",
        },
    }


def test_probe_semantic_runtime_disabled_by_default() -> None:
    result = semantic_ranker.probe_semantic_runtime({"runtime": {}, "gpu": {}})
    assert bool(result.get("enabled", True)) is False
    assert str(result.get("reason", "")) == "disabled_by_config"


def test_apply_semantic_ranking_returns_original_when_probe_not_ready(monkeypatch) -> None:
    ranked = [
        {"path": "src/a.py", "score": 0.7, "reasons": ["symbol_match"], "signals": []},
        {"path": "src/b.py", "score": 0.6, "reasons": ["symbol_match"], "signals": []},
    ]
    monkeypatch.setattr(
        semantic_ranker,
        "probe_semantic_runtime",
        lambda config: {
            "enabled": True,
            "provider_requested": "auto",
            "provider_resolved": "flagembedding",
            "reason": "flagembedding_unavailable",
            "gpu_runtime": {"effective_device": "cpu"},
            "use_onnx": False,
        },
    )

    updated, meta = semantic_ranker.apply_semantic_ranking(
        ranked=ranked,
        config=_base_config(),
        task="fix semantic ranking",
        task_tokens=["semantic", "ranking"],
        hints=[],
        symbols_by_file={},
        references_by_file={},
        deps_by_file={},
        tests_by_file={},
    )
    assert updated == ranked
    assert bool(meta.get("applied", True)) is False
    assert str(meta.get("reason", "")) == "flagembedding_unavailable"


def test_apply_semantic_ranking_embedding_boost_changes_order(monkeypatch) -> None:
    class FakeRuntime:
        provider = "flagembedding"
        device = "cpu"
        use_onnx = False
        reranker = None

        def encode_documents(self, texts: list[str], *, batch_size: int) -> list[list[float]]:
            assert batch_size == 8
            # query, doc(a), doc(b)
            return [[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]]

        def rerank_documents(self, query: str, docs: list[str]) -> list[float]:
            raise AssertionError("reranker should not run")

    monkeypatch.setattr(
        semantic_ranker,
        "probe_semantic_runtime",
        lambda config: {
            "enabled": True,
            "provider_requested": "auto",
            "provider_resolved": "flagembedding",
            "reason": "ready",
            "gpu_runtime": {"effective_device": "cpu"},
            "use_onnx": False,
        },
    )
    monkeypatch.setattr(
        semantic_ranker,
        "_build_runtime",
        lambda config, probe: (FakeRuntime(), "ready"),
    )

    ranked = [
        {"path": "src/a.py", "score": 0.5, "reasons": [], "signals": []},
        {"path": "src/b.py", "score": 0.5, "reasons": [], "signals": []},
    ]
    updated, meta = semantic_ranker.apply_semantic_ranking(
        ranked=ranked,
        config=_base_config(),
        task="fix semantic ranking",
        task_tokens=["semantic", "ranking"],
        hints=[],
        symbols_by_file={},
        references_by_file={},
        deps_by_file={},
        tests_by_file={},
    )
    assert bool(meta.get("applied", False)) is True
    assert bool(meta.get("embedding_applied", False)) is True
    assert str(updated[0].get("path", "")) == "src/a.py"
    assert any(str(reason) == "semantic_embedding" for reason in updated[0].get("reasons", []))


def test_apply_semantic_ranking_reranker_blend(monkeypatch) -> None:
    class FakeRuntime:
        provider = "flagembedding"
        device = "cpu"
        use_onnx = False
        reranker = object()

        def encode_documents(self, texts: list[str], *, batch_size: int) -> list[list[float]]:
            # keep initial lexical order before rerank
            return [[1.0, 0.0], [1.0, 0.0], [0.8, 0.2]]

        def rerank_documents(self, query: str, docs: list[str]) -> list[float]:
            # second document should win after reranker blend
            return [0.1, 0.9]

    config = _base_config()
    config["runtime"]["semantic_ranker_embed_weight"] = 0.0
    config["runtime"]["semantic_reranker_enabled"] = True
    config["runtime"]["semantic_reranker_weight"] = 1.0

    monkeypatch.setattr(
        semantic_ranker,
        "probe_semantic_runtime",
        lambda cfg: {
            "enabled": True,
            "provider_requested": "auto",
            "provider_resolved": "flagembedding",
            "reason": "ready",
            "gpu_runtime": {"effective_device": "cpu"},
            "use_onnx": False,
        },
    )
    monkeypatch.setattr(
        semantic_ranker,
        "_build_runtime",
        lambda cfg, probe: (FakeRuntime(), "ready"),
    )

    ranked = [
        {"path": "src/a.py", "score": 0.9, "reasons": [], "signals": []},
        {"path": "src/b.py", "score": 0.8, "reasons": [], "signals": []},
    ]
    updated, meta = semantic_ranker.apply_semantic_ranking(
        ranked=ranked,
        config=config,
        task="rerank target",
        task_tokens=["rerank", "target"],
        hints=[],
        symbols_by_file={},
        references_by_file={},
        deps_by_file={},
        tests_by_file={},
    )
    assert bool(meta.get("reranker_applied", False)) is True
    assert str(updated[0].get("path", "")) == "src/b.py"
    assert any(str(reason) == "semantic_reranker" for reason in updated[0].get("reasons", []))
