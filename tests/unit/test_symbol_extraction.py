from __future__ import annotations

from pathlib import Path

from accel.indexers import symbols


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def test_extract_symbols_uses_regex_fallback_when_parser_disabled(tmp_path: Path) -> None:
    source = tmp_path / "src" / "main.ts"
    _write(source, "export function normalizeTask(input: string) { return input.trim(); }\n")

    rows = symbols.extract_symbols(
        source,
        "src/main.ts",
        "typescript",
        runtime_cfg={"syntax_parser_enabled": False, "syntax_parser_provider": "off"},
    )
    names = {str(item.get("symbol", "")) for item in rows}
    assert "normalizeTask" in names


def test_extract_symbols_prefers_tree_sitter_when_available(tmp_path: Path, monkeypatch) -> None:
    source = tmp_path / "src" / "main.ts"
    _write(source, "export function normalizeTask(input: string) { return input.trim(); }\n")

    expected = [
        {
            "symbol": "normalizeTask",
            "kind": "function",
            "lang": "typescript",
            "file": "src/main.ts",
            "line_start": 1,
            "line_end": 1,
            "qualified_name": "src/main.ts:normalizeTask",
        }
    ]
    monkeypatch.setattr(symbols, "_ts_symbols_from_tree_sitter", lambda **kwargs: expected)

    rows = symbols.extract_symbols(
        source,
        "src/main.ts",
        "typescript",
        runtime_cfg={"syntax_parser_enabled": True, "syntax_parser_provider": "tree_sitter"},
    )
    assert rows == expected


def test_extract_symbols_tree_sitter_empty_result_falls_back_to_regex(
    tmp_path: Path, monkeypatch
) -> None:
    source = tmp_path / "src" / "main.ts"
    _write(source, "export function normalizeTask(input: string) { return input.trim(); }\n")

    monkeypatch.setattr(symbols, "_ts_symbols_from_tree_sitter", lambda **kwargs: [])

    rows = symbols.extract_symbols(
        source,
        "src/main.ts",
        "typescript",
        runtime_cfg={"syntax_parser_enabled": True, "syntax_parser_provider": "auto"},
    )
    names = {str(item.get("symbol", "")) for item in rows}
    assert "normalizeTask" in names
