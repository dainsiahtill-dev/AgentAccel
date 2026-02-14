from __future__ import annotations

from accel.query.adaptive_budget import classify_task_intent, resolve_adaptive_budget


def test_classify_task_intent_detects_bug_fix() -> None:
    intent = classify_task_intent(
        "Fix failing retry bug in payment processor",
        ["fix", "failing", "retry", "bug"],
    )
    assert intent == "bug_fix"


def test_resolve_adaptive_budget_scales_by_intent_and_complexity() -> None:
    budget, meta = resolve_adaptive_budget(
        context_cfg={
            "max_chars": 24000,
            "max_snippets": 60,
            "top_n_files": 12,
            "snippet_radius": 40,
        },
        runtime_cfg={
            "adaptive_budget_enabled": True,
            "adaptive_budget_min_factor": 0.65,
            "adaptive_budget_max_factor": 1.45,
        },
        task="Refactor scheduler and optimize async dispatch flow",
        task_tokens=["refactor", "scheduler", "optimize", "async", "dispatch"],
        changed_files=["a.py", "b.py", "c.py", "d.py"],
        ranked_files=[{"score": 0.9}, {"score": 0.2}, {"score": 0.1}],
        budget_override=None,
    )

    assert int(budget["max_chars"]) >= 24000
    assert int(budget["top_n_files"]) >= 12
    assert meta["intent"] == "refactor"
    assert float(meta["complexity_factor"]) > 0.0


def test_resolve_adaptive_budget_respects_overrides() -> None:
    budget, meta = resolve_adaptive_budget(
        context_cfg={
            "max_chars": 24000,
            "max_snippets": 60,
            "top_n_files": 12,
            "snippet_radius": 40,
        },
        runtime_cfg={"adaptive_budget_enabled": True},
        task="Add endpoint",
        task_tokens=["add", "endpoint"],
        changed_files=["x.py"],
        ranked_files=[{"score": 0.8}],
        budget_override={"max_chars": 11111, "top_n_files": 9},
    )

    assert int(budget["max_chars"]) == 11111
    assert int(budget["top_n_files"]) == 9
    assert sorted(meta["override_keys"]) == ["max_chars", "top_n_files"]
