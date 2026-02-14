from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from accel.indexers import symbols as symbols_mod


def _write(tmp_path: Path, rel_path: str, content: str) -> Path:
    path = tmp_path / rel_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(textwrap.dedent(content).strip() + "\n", encoding="utf-8")
    return path


def test_extract_symbols_python_ast_enriched_metadata(tmp_path: Path) -> None:
    file_path = _write(
        tmp_path,
        "sample.py",
        """
        class Worker(BaseWorker):
            @decorator
            async def process(self, item: int) -> str:
                return str(item)

        VALUE: int = 3
        """,
    )

    rows = symbols_mod.extract_symbols(
        file_path,
        "sample.py",
        "python",
        runtime_cfg={"syntax_parser_enabled": False},
    )

    class_row = next(row for row in rows if row.get("symbol") == "Worker")
    assert class_row["kind"] == "class"
    assert "BaseWorker" in class_row.get("relation_targets", [])
    assert "class Worker" in str(class_row.get("signature", ""))

    method_row = next(row for row in rows if row.get("symbol") == "process")
    assert method_row["kind"] == "method"
    assert str(method_row.get("signature", "")).startswith("async def process")
    assert "async" in method_row.get("attributes", [])
    assert any("decorator" in item for item in method_row.get("decorators", []))

    variable_row = next(row for row in rows if row.get("symbol") == "VALUE")
    assert variable_row["kind"] == "variable"
    assert int(variable_row.get("scope_depth", -1)) == 0


def test_extract_symbols_typescript_text_fallback_relations(tmp_path: Path) -> None:
    file_path = _write(
        tmp_path,
        "sample.ts",
        """
        export class Service<T> extends Base implements Runner {}
        export interface Runner extends Job {}
        """,
    )

    rows = symbols_mod.extract_symbols(
        file_path,
        "sample.ts",
        "typescript",
        runtime_cfg={"syntax_parser_enabled": False},
    )

    class_row = next(row for row in rows if row.get("symbol") == "Service")
    assert class_row["kind"] == "class"
    assert "Base" in class_row.get("relation_targets", [])
    assert "Runner" in class_row.get("relation_targets", [])
    assert "T" in class_row.get("type_parameters", [])

    interface_row = next(row for row in rows if row.get("symbol") == "Runner")
    assert interface_row["kind"] == "type"
    assert "Job" in interface_row.get("relation_targets", [])


def test_extract_symbols_tree_sitter_missing_falls_back_to_ast(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    file_path = _write(
        tmp_path,
        "sample.py",
        """
        def compute(value: int) -> int:
            return value + 1
        """,
    )
    monkeypatch.setattr(symbols_mod, "_load_tree_sitter_parser", lambda *_: None)

    rows = symbols_mod.extract_symbols(
        file_path,
        "sample.py",
        "python",
        runtime_cfg={"syntax_parser_enabled": True, "syntax_parser_provider": "tree_sitter"},
    )

    function_row = next(row for row in rows if row.get("symbol") == "compute")
    assert function_row["kind"] == "function"
    assert function_row.get("syntax_source") == "ast"
