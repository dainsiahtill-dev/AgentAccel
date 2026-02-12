from __future__ import annotations

from accel.query.rule_compressor import compress_snippet_content


def test_rule_compressor_applies_import_and_blank_rules() -> None:
    content = "\n".join(
        [f"import mod_{idx}" for idx in range(20)]
        + ["", "", "def run() -> None:", "    pass", "", ""]
    )
    compact, audit = compress_snippet_content(
        content,
        max_chars=2000,
        task_tokens=["run"],
        symbol="run",
    )
    rules = dict(audit.get("rules", {}))
    assert int(rules.get("trim_import_block", 0)) > 0
    assert int(rules.get("collapse_blank_runs", 0)) > 0
    assert "omitted" in compact


def test_rule_compressor_drops_low_signal_large_snippet() -> None:
    lines = ["# comment line"] * 200
    compact, audit = compress_snippet_content(
        "\n".join(lines),
        max_chars=4000,
        task_tokens=["auth", "token"],
        symbol="",
    )
    assert compact == ""
    assert bool(audit.get("dropped", False)) is True
    rules = dict(audit.get("rules", {}))
    assert int(rules.get("drop_low_signal", 0)) >= 1


def test_rule_compressor_truncates_when_over_budget() -> None:
    content = "def execute():\n" + ("    print('x')\n" * 500)
    compact, audit = compress_snippet_content(
        content,
        max_chars=300,
        task_tokens=["execute"],
        symbol="execute",
    )
    assert len(compact) <= 300
    assert compact.endswith("[truncated]")
    rules = dict(audit.get("rules", {}))
    assert int(rules.get("truncate_max_chars", 0)) >= 1
