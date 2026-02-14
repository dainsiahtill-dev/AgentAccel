from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from accel.query.symbol_query import (
    _match_score,
    build_call_graph,
    get_symbol_details,
    search_symbols,
)


@pytest.fixture
def temp_index_dir():
    """Create a temporary index directory with test data."""
    with tempfile.TemporaryDirectory() as tmpdir:
        index_dir = Path(tmpdir)

        symbols = [
            {
                "symbol": "compile_context_pack",
                "qualified_name": "context_compiler.compile_context_pack",
                "kind": "function",
                "lang": "python",
                "file": "accel/query/context_compiler.py",
                "line_start": 100,
                "line_end": 150,
                "signature": "def compile_context_pack(project_dir, task, ...)",
                "scope": "",
                "relation_targets": ["build_candidate_files", "score_file"],
            },
            {
                "symbol": "build_candidate_files",
                "qualified_name": "planner.build_candidate_files",
                "kind": "function",
                "lang": "python",
                "file": "accel/query/planner.py",
                "line_start": 50,
                "line_end": 80,
                "signature": "def build_candidate_files(index_dir, task_tokens)",
            },
            {
                "symbol": "SessionReceiptStore",
                "qualified_name": "storage.session_receipts.SessionReceiptStore",
                "kind": "class",
                "lang": "python",
                "file": "accel/storage/session_receipts.py",
                "line_start": 20,
                "line_end": 100,
                "bases": ["object"],
            },
            {
                "symbol": "score_file",
                "qualified_name": "ranker.score_file",
                "kind": "function",
                "lang": "python",
                "file": "accel/query/ranker.py",
                "line_start": 30,
                "line_end": 60,
            },
            {
                "symbol": "compileTemplate",
                "qualified_name": "utils.compileTemplate",
                "kind": "function",
                "lang": "typescript",
                "file": "src/utils/template.ts",
                "line_start": 10,
                "line_end": 25,
            },
        ]

        references = [
            {
                "file": "accel/query/context_compiler.py",
                "source_symbol": "compile_context_pack",
                "target_symbol": "build_candidate_files",
                "edge_to": "accel/query/planner.py",
            },
            {
                "file": "accel/query/context_compiler.py",
                "source_symbol": "compile_context_pack",
                "target_symbol": "score_file",
                "edge_to": "accel/query/ranker.py",
            },
            {
                "file": "accel/mcp_server.py",
                "source_symbol": "_tool_context",
                "target_symbol": "compile_context_pack",
                "edge_to": "accel/query/context_compiler.py",
            },
        ]

        deps = [
            {
                "file": "accel/query/context_compiler.py",
                "target_symbol": "planner",
                "edge_to": "accel/query/planner.py",
            },
            {
                "file": "accel/query/context_compiler.py",
                "target_symbol": "ranker",
                "edge_to": "accel/query/ranker.py",
            },
        ]

        symbols_path = index_dir / "symbols.jsonl"
        symbols_path.write_text(
            "\n".join(json.dumps(row) for row in symbols) + "\n",
            encoding="utf-8",
        )

        refs_path = index_dir / "references.jsonl"
        refs_path.write_text(
            "\n".join(json.dumps(row) for row in references) + "\n",
            encoding="utf-8",
        )

        deps_path = index_dir / "deps.jsonl"
        deps_path.write_text(
            "\n".join(json.dumps(row) for row in deps) + "\n",
            encoding="utf-8",
        )

        yield index_dir


class TestMatchScore:
    def test_exact_match_symbol(self):
        score = _match_score("compile", "compile", "module.compile")
        assert score == 1.0

    def test_exact_match_qualified(self):
        score = _match_score("module.compile", "compile", "module.compile")
        assert score == 0.9

    def test_prefix_match_symbol(self):
        score = _match_score("comp", "compile", "module.compile")
        assert score == 0.7

    def test_prefix_match_qualified(self):
        score = _match_score("module", "compile", "module.compile")
        assert score == 0.5

    def test_contains_match_symbol(self):
        score = _match_score("pil", "compile", "module.compile")
        assert score == 0.3

    def test_contains_match_qualified(self):
        score = _match_score("ule", "xyz", "module.xyz")
        assert score == 0.2

    def test_no_match(self):
        score = _match_score("xyz", "abc", "def.abc")
        assert score == 0.0


class TestSearchSymbols:
    def test_search_by_prefix(self, temp_index_dir):
        results = search_symbols(temp_index_dir, "compile")
        assert len(results) >= 2
        assert results[0]["symbol"] == "compile_context_pack"
        assert results[0]["score"] > 0

    def test_search_with_kind_filter(self, temp_index_dir):
        results = search_symbols(
            temp_index_dir, "Session", symbol_kinds=["class"]
        )
        assert len(results) == 1
        assert results[0]["symbol"] == "SessionReceiptStore"
        assert results[0]["kind"] == "class"

    def test_search_with_language_filter(self, temp_index_dir):
        results = search_symbols(
            temp_index_dir, "compile", languages=["typescript"]
        )
        assert len(results) == 1
        assert results[0]["symbol"] == "compileTemplate"
        assert results[0]["lang"] == "typescript"

    def test_search_max_results(self, temp_index_dir):
        results = search_symbols(temp_index_dir, "e", max_results=2)
        assert len(results) <= 2

    def test_search_empty_query(self, temp_index_dir):
        results = search_symbols(temp_index_dir, "")
        assert results == []

    def test_search_no_match(self, temp_index_dir):
        results = search_symbols(temp_index_dir, "nonexistent_symbol_xyz")
        assert results == []

    def test_search_includes_signature(self, temp_index_dir):
        results = search_symbols(
            temp_index_dir, "compile_context_pack", include_signature=True
        )
        assert len(results) >= 1
        assert "signature" in results[0]

    def test_search_excludes_signature(self, temp_index_dir):
        results = search_symbols(
            temp_index_dir, "compile_context_pack", include_signature=False
        )
        assert len(results) >= 1
        assert "signature" not in results[0]


class TestGetSymbolDetails:
    def test_get_by_symbol_name(self, temp_index_dir):
        result = get_symbol_details(temp_index_dir, "compile_context_pack")
        assert result is not None
        assert result["symbol"] == "compile_context_pack"
        assert result["kind"] == "function"
        assert result["file"] == "accel/query/context_compiler.py"

    def test_get_by_qualified_name(self, temp_index_dir):
        result = get_symbol_details(
            temp_index_dir, "context_compiler.compile_context_pack"
        )
        assert result is not None
        assert result["symbol"] == "compile_context_pack"

    def test_get_with_file_path(self, temp_index_dir):
        result = get_symbol_details(
            temp_index_dir,
            "compile_context_pack",
            file_path="accel/query/context_compiler.py",
        )
        assert result is not None
        assert result["file"] == "accel/query/context_compiler.py"

    def test_get_nonexistent_symbol(self, temp_index_dir):
        result = get_symbol_details(temp_index_dir, "nonexistent_xyz")
        assert result is None

    def test_get_includes_relations(self, temp_index_dir):
        result = get_symbol_details(
            temp_index_dir, "compile_context_pack", include_relations=True
        )
        assert result is not None
        assert "relations" in result
        relations = result["relations"]
        assert "calls" in relations
        assert "build_candidate_files" in relations["calls"]

    def test_get_class_with_bases(self, temp_index_dir):
        result = get_symbol_details(temp_index_dir, "SessionReceiptStore")
        assert result is not None
        assert result["kind"] == "class"
        assert "bases" in result


class TestBuildCallGraph:
    def test_build_from_file(self, temp_index_dir):
        result = build_call_graph(
            temp_index_dir,
            start_file="accel/query/context_compiler.py",
            depth=2,
        )
        assert "nodes" in result
        assert "edges" in result
        assert "stats" in result
        assert result["stats"]["node_count"] >= 1

    def test_build_from_symbol(self, temp_index_dir):
        result = build_call_graph(
            temp_index_dir,
            start_symbol="compile_context_pack",
            depth=1,
        )
        assert "nodes" in result
        assert len(result["start_nodes"]) >= 1

    def test_build_direction_down(self, temp_index_dir):
        result = build_call_graph(
            temp_index_dir,
            start_file="accel/query/context_compiler.py",
            depth=2,
            direction="down",
        )
        assert result["direction"] == "down"

    def test_build_direction_up(self, temp_index_dir):
        result = build_call_graph(
            temp_index_dir,
            start_file="accel/query/ranker.py",
            depth=2,
            direction="up",
        )
        assert result["direction"] == "up"

    def test_build_depth_limit(self, temp_index_dir):
        result = build_call_graph(
            temp_index_dir,
            start_file="accel/query/context_compiler.py",
            depth=1,
        )
        assert result["depth"] == 1
        for node in result["nodes"]:
            assert node["level"] <= 1

    def test_build_empty_start(self, temp_index_dir):
        result = build_call_graph(temp_index_dir, depth=1)
        assert "nodes" in result
        assert "edges" in result
