from __future__ import annotations

from accel.eval.metrics import (
    aggregate_case_metrics,
    recall_at_k,
    reciprocal_rank,
    symbol_hit_rate,
)


def test_recall_and_mrr_metrics() -> None:
    expected = ["accel/query/context_compiler.py", "accel/query/ranker.py"]
    predicted = [
        "accel/cli.py",
        "accel/query/ranker.py",
        "accel/query/context_compiler.py",
    ]
    assert recall_at_k(expected, predicted, 1) == 0.0
    assert recall_at_k(expected, predicted, 3) == 1.0
    assert reciprocal_rank(expected, predicted) == 0.5


def test_symbol_hit_rate() -> None:
    expected = ["compile_context_pack", "score_file"]
    observed = ["score_file", "other_symbol"]
    assert symbol_hit_rate(expected, observed) == 0.5


def test_aggregate_case_metrics() -> None:
    summary = aggregate_case_metrics(
        [
            {
                "recall_at_5": 0.4,
                "recall_at_10": 0.7,
                "mrr": 0.5,
                "symbol_hit_rate": 0.2,
                "context_chars": 1400.0,
                "latency_ms": 90.0,
            },
            {
                "recall_at_5": 0.8,
                "recall_at_10": 1.0,
                "mrr": 0.75,
                "symbol_hit_rate": 0.6,
                "context_chars": 2200.0,
                "latency_ms": 150.0,
            },
        ]
    )
    assert summary["case_count"] == 2.0
    assert summary["recall_at_5"] == 0.6
    assert summary["mrr"] == 0.625
    assert summary["avg_context_chars"] == 1800.0
