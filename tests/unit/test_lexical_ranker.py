from __future__ import annotations

from accel.query import lexical_ranker


def test_probe_lexical_runtime_disabled_by_default() -> None:
    probe = lexical_ranker.probe_lexical_runtime({"runtime": {}})
    assert bool(probe.get("enabled", False)) is False
    assert str(probe.get("reason", "")) == "disabled_by_config"


def test_apply_lexical_ranking_bm25_blend_changes_order(monkeypatch) -> None:
    ranked = [
        {"path": "src/unrelated.py", "score": 0.6, "reasons": ["baseline"], "signals": []},
        {"path": "src/target.py", "score": 0.2, "reasons": ["baseline"], "signals": []},
    ]
    cfg = {
        "runtime": {
            "lexical_ranker_enabled": True,
            "lexical_ranker_provider": "tantivy",
            "lexical_ranker_max_candidates": 20,
            "lexical_ranker_weight": 1.0,
        }
    }

    monkeypatch.setattr(
        lexical_ranker,
        "probe_lexical_runtime",
        lambda config: {
            "enabled": True,
            "provider_requested": "tantivy",
            "provider_resolved": "tantivy",
            "tantivy_available": False,
            "reason": "ready",
            "reason_detail": "",
        },
    )

    updated, meta = lexical_ranker.apply_lexical_ranking(
        ranked=ranked,
        config=cfg,
        task="fix target parsing",
        task_tokens=["target", "parsing"],
        hints=["target"],
        symbols_by_file={
            "src/target.py": [{"symbol": "target_parser", "qualified_name": "target_parser"}],
            "src/unrelated.py": [{"symbol": "helper", "qualified_name": "helper"}],
        },
        references_by_file={},
        deps_by_file={},
        tests_by_file={},
    )

    assert bool(meta.get("applied", False)) is True
    assert str(meta.get("engine", "")) == "bm25"
    assert str(updated[0].get("path", "")) == "src/target.py"
    assert "lexical_search" in list(updated[0].get("reasons", []))


def test_apply_lexical_ranking_tantivy_failure_falls_back(monkeypatch) -> None:
    ranked = [
        {"path": "src/a.py", "score": 0.4, "reasons": ["baseline"], "signals": []},
        {"path": "src/b.py", "score": 0.3, "reasons": ["baseline"], "signals": []},
    ]
    cfg = {
        "runtime": {
            "lexical_ranker_enabled": True,
            "lexical_ranker_provider": "tantivy",
            "lexical_ranker_weight": 0.5,
        }
    }

    monkeypatch.setattr(
        lexical_ranker,
        "probe_lexical_runtime",
        lambda config: {
            "enabled": True,
            "provider_requested": "tantivy",
            "provider_resolved": "tantivy",
            "tantivy_available": True,
            "reason": "ready",
            "reason_detail": "",
        },
    )

    def fake_tantivy(*, docs, query_text, limit):
        raise RuntimeError("api mismatch")

    monkeypatch.setattr(lexical_ranker, "_score_with_tantivy", fake_tantivy)
    monkeypatch.setattr(
        lexical_ranker,
        "_score_with_bm25",
        lambda *, docs, query_tokens: {"src/a.py": 1.0, "src/b.py": 0.2},
    )

    _, meta = lexical_ranker.apply_lexical_ranking(
        ranked=ranked,
        config=cfg,
        task="a",
        task_tokens=["a"],
        hints=[],
        symbols_by_file={},
        references_by_file={},
        deps_by_file={},
        tests_by_file={},
    )

    assert bool(meta.get("applied", False)) is True
    assert str(meta.get("engine", "")) == "bm25"
    assert str(meta.get("reason", "")).startswith("tantivy_failed_fallback")
