from __future__ import annotations

from accel.query.ranker import score_file


def test_score_file_uses_structural_signals() -> None:
    task_tokens = ["payment", "processor", "retry"]
    symbols_by_file = {
        "accel/payment.py": [
            {
                "symbol": "PaymentProcessor",
                "qualified_name": "core.PaymentProcessor",
                "signature": "class PaymentProcessor extends BaseProcessor",
                "relation_targets": ["BaseProcessor", "RetryPolicy"],
                "line_start": 10,
                "line_end": 80,
                "kind": "class",
            },
            {
                "symbol": "run_retry",
                "qualified_name": "core.PaymentProcessor.run_retry",
                "signature": "def run_retry(payment_id: str) -> bool",
                "line_start": 90,
                "line_end": 110,
                "kind": "method",
            },
        ]
    }
    references_by_file = {
        "accel/payment.py": [
            {"source_symbol": "run_retry", "target_symbol": "retry"},
            {"source_symbol": "run_retry", "target_symbol": "processor"},
        ]
    }
    deps_by_file = {
        "accel/payment.py": [
            {"edge_to": "payment.models"},
        ]
    }
    tests_by_file = {"accel/payment.py": [{"test_file": "tests/test_payment.py"}]}

    scored = score_file(
        file_path="accel/payment.py",
        task_tokens=task_tokens,
        symbols_by_file=symbols_by_file,
        references_by_file=references_by_file,
        deps_by_file=deps_by_file,
        tests_by_file=tests_by_file,
        changed_file_set={"accel/payment.py"},
    )

    signal_names = [str(item.get("signal_name")) for item in scored.get("signals", [])]
    assert "signature_match" in signal_names
    assert "structural_match" in signal_names
    assert "syntax_unit_coverage" in signal_names
    assert "signature_match" in scored.get("reasons", [])
    assert "structural_match" in scored.get("reasons", [])
    assert float(scored["score"]) > 0.4


def test_score_file_rich_metadata_outscores_plain_metadata() -> None:
    task_tokens = ["scheduler", "async", "dispatch"]
    plain_symbols = {
        "accel/scheduler.py": [
            {
                "symbol": "dispatch",
                "qualified_name": "dispatch",
                "line_start": 10,
                "line_end": 10,
                "kind": "function",
            }
        ]
    }
    rich_symbols = {
        "accel/scheduler.py": [
            {
                "symbol": "dispatch",
                "qualified_name": "Scheduler.dispatch",
                "signature": "async def dispatch(job: Job) -> bool",
                "scope": "Scheduler",
                "attributes": ["async"],
                "relation_targets": ["JobQueue"],
                "line_start": 10,
                "line_end": 40,
                "kind": "method",
            }
        ]
    }

    shared_refs = {"accel/scheduler.py": []}
    shared_deps = {"accel/scheduler.py": []}
    shared_tests = {"accel/scheduler.py": []}

    plain_score = score_file(
        file_path="accel/scheduler.py",
        task_tokens=task_tokens,
        symbols_by_file=plain_symbols,
        references_by_file=shared_refs,
        deps_by_file=shared_deps,
        tests_by_file=shared_tests,
        changed_file_set=set(),
    )
    rich_score = score_file(
        file_path="accel/scheduler.py",
        task_tokens=task_tokens,
        symbols_by_file=rich_symbols,
        references_by_file=shared_refs,
        deps_by_file=shared_deps,
        tests_by_file=shared_tests,
        changed_file_set=set(),
    )

    assert float(rich_score["score"]) > float(plain_score["score"])
