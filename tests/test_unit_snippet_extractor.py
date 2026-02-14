from __future__ import annotations

from pathlib import Path

from accel.query.snippet_extractor import extract_snippet


def _write_file(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_extract_snippet_prefers_syntax_unit_window(tmp_path: Path) -> None:
    file_path = tmp_path / "module.py"
    _write_file(
        file_path,
        [
            "import os",
            "",
            "def helper(a: int) -> int:",
            "    value = a + 1",
            "    if value > 10:",
            "        return value",
            "    return value + 2",
            "",
            "print(helper(4))",
        ],
    )
    symbol_rows = [
        {
            "symbol": "helper",
            "kind": "function",
            "line_start": 3,
            "line_end": 7,
            "qualified_name": "helper",
            "signature": "def helper(a: int) -> int",
        }
    ]

    snippet = extract_snippet(
        project_dir=tmp_path,
        rel_path="module.py",
        task_tokens=["helper", "int"],
        symbol_rows=symbol_rows,
        snippet_radius=2,
        max_chars=10_000,
    )

    assert snippet is not None
    assert snippet["reason"] == "syntax_unit"
    assert snippet["start_line"] == 3
    assert snippet["end_line"] == 7
    assert snippet["line_start"] == 3
    assert snippet["line_end"] == 7
    assert snippet["symbol"] == "helper"
    assert "def helper" in str(snippet["content"])


def test_extract_snippet_token_fallback(tmp_path: Path) -> None:
    file_path = tmp_path / "module.py"
    _write_file(
        file_path,
        [
            "first line",
            "second line",
            "third line",
            "needle appears here",
            "fifth line",
        ],
    )

    snippet = extract_snippet(
        project_dir=tmp_path,
        rel_path="module.py",
        task_tokens=["needle"],
        symbol_rows=[],
        snippet_radius=1,
        max_chars=10_000,
    )

    assert snippet is not None
    assert snippet["start_line"] == 3
    assert snippet["end_line"] == 5
    assert snippet["reason"] == "symbol_or_token_focus"


def test_extract_snippet_truncates_large_content(tmp_path: Path) -> None:
    file_path = tmp_path / "module.py"
    _write_file(file_path, [f"line {idx:03d}" for idx in range(1, 80)])
    symbol_rows = [
        {
            "symbol": "block",
            "kind": "function",
            "line_start": 1,
            "line_end": 70,
            "qualified_name": "block",
            "signature": "def block()",
        }
    ]

    snippet = extract_snippet(
        project_dir=tmp_path,
        rel_path="module.py",
        task_tokens=["block"],
        symbol_rows=symbol_rows,
        snippet_radius=2,
        max_chars=120,
    )

    assert snippet is not None
    assert len(str(snippet["content"])) <= 120
    assert "\n...\n" in str(snippet["content"])
