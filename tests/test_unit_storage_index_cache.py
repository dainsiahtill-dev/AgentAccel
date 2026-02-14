"""Unit tests for storage/index_cache.py"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from accel.storage.index_cache import (
    INDEX_FILE_NAMES,
    base_path_for_kind,
    delta_path_for_base,
    load_jsonl_mmap,
    count_jsonl_lines,
    group_rows_by_key,
    flatten_grouped_rows,
    load_grouped_rows_with_delta,
    load_index_rows,
    append_delta_ops,
    write_jsonl_atomic,
    clear_delta_file,
)


@pytest.fixture
def temp_index_dir():
    """Create a temporary index directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


class TestBasePathForKind:
    def test_valid_kinds(self, temp_index_dir):
        assert base_path_for_kind(temp_index_dir, "symbols") == temp_index_dir / "symbols.jsonl"
        assert base_path_for_kind(temp_index_dir, "references") == temp_index_dir / "references.jsonl"
        assert base_path_for_kind(temp_index_dir, "dependencies") == temp_index_dir / "deps.jsonl"
        assert base_path_for_kind(temp_index_dir, "test_ownership") == temp_index_dir / "test_ownership.jsonl"

    def test_invalid_kind_raises(self, temp_index_dir):
        with pytest.raises(ValueError, match="Invalid kind"):
            base_path_for_kind(temp_index_dir, "invalid")


class TestDeltaPathForBase:
    def test_delta_path(self):
        base = Path("/some/path/symbols.jsonl")
        delta = delta_path_for_base(base)
        assert delta == Path("/some/path/symbols.delta.jsonl")

    def test_delta_path_other_file(self):
        base = Path("/some/path/deps.jsonl")
        delta = delta_path_for_base(base)
        assert delta == Path("/some/path/deps.delta.jsonl")


class TestLoadJsonlMmap:
    def test_empty_file(self, temp_index_dir):
        path = temp_index_dir / "test.jsonl"
        path.write_text("", encoding="utf-8")
        result = load_jsonl_mmap(path)
        assert result == []

    def test_nonexistent_file(self, temp_index_dir):
        path = temp_index_dir / "nonexistent.jsonl"
        result = load_jsonl_mmap(path)
        assert result == []

    def test_valid_jsonl(self, temp_index_dir):
        path = temp_index_dir / "test.jsonl"
        rows = [
            {"symbol": "func1", "kind": "function"},
            {"symbol": "Class1", "kind": "class"},
        ]
        path.write_text(
            "\n".join(json.dumps(r) for r in rows) + "\n",
            encoding="utf-8",
        )
        result = load_jsonl_mmap(path)
        assert len(result) == 2
        assert result[0]["symbol"] == "func1"
        assert result[1]["symbol"] == "Class1"

    def test_skips_invalid_json_lines(self, temp_index_dir):
        path = temp_index_dir / "test.jsonl"
        path.write_text(
            '{"symbol": "func1"}\nnot valid json\n{"symbol": "func2"}\n',
            encoding="utf-8",
        )
        result = load_jsonl_mmap(path)
        assert len(result) == 2
        assert result[0]["symbol"] == "func1"
        assert result[1]["symbol"] == "func2"

    def test_skips_empty_lines(self, temp_index_dir):
        path = temp_index_dir / "test.jsonl"
        path.write_text(
            '{"symbol": "func1"}\n\n\n{"symbol": "func2"}\n',
            encoding="utf-8",
        )
        result = load_jsonl_mmap(path)
        assert len(result) == 2


class TestCountJsonlLines:
    def test_empty_file(self, temp_index_dir):
        path = temp_index_dir / "test.jsonl"
        path.write_text("", encoding="utf-8")
        assert count_jsonl_lines(path) == 0

    def test_nonexistent_file(self, temp_index_dir):
        path = temp_index_dir / "nonexistent.jsonl"
        assert count_jsonl_lines(path) == 0

    def test_counts_lines(self, temp_index_dir):
        path = temp_index_dir / "test.jsonl"
        path.write_text("line1\nline2\nline3\n", encoding="utf-8")
        assert count_jsonl_lines(path) == 3


class TestGroupRowsByKey:
    def test_empty_rows(self):
        result = group_rows_by_key([], "file")
        assert result == {}

    def test_groups_by_key(self):
        rows = [
            {"file": "a.py", "symbol": "func1"},
            {"file": "a.py", "symbol": "func2"},
            {"file": "b.py", "symbol": "func3"},
        ]
        result = group_rows_by_key(rows, "file")
        assert len(result) == 2
        assert len(result["a.py"]) == 2
        assert len(result["b.py"]) == 1

    def test_skips_empty_keys(self):
        rows = [
            {"file": "a.py", "symbol": "func1"},
            {"file": "", "symbol": "func2"},
            {"symbol": "func3"},  # no file key
        ]
        result = group_rows_by_key(rows, "file")
        assert len(result) == 1
        assert "a.py" in result


class TestFlattenGroupedRows:
    def test_empty_grouped(self):
        result = flatten_grouped_rows({})
        assert result == []

    def test_flattens_and_sorts_by_key(self):
        grouped = {
            "b.py": [{"symbol": "func2"}],
            "a.py": [{"symbol": "func1"}],
        }
        result = flatten_grouped_rows(grouped)
        assert len(result) == 2
        # Should be sorted by key
        assert result[0]["symbol"] == "func1"
        assert result[1]["symbol"] == "func2"


class TestLoadGroupedRowsWithDelta:
    def test_base_only(self, temp_index_dir):
        base_path = temp_index_dir / "symbols.jsonl"
        rows = [
            {"file": "a.py", "symbol": "func1"},
            {"file": "b.py", "symbol": "func2"},
        ]
        base_path.write_text(
            "\n".join(json.dumps(r) for r in rows) + "\n",
            encoding="utf-8",
        )

        grouped, delta_count = load_grouped_rows_with_delta(
            temp_index_dir, "symbols", "file"
        )
        assert delta_count == 0
        assert "a.py" in grouped
        assert "b.py" in grouped

    def test_with_delete_delta(self, temp_index_dir):
        base_path = temp_index_dir / "symbols.jsonl"
        delta_path = temp_index_dir / "symbols.delta.jsonl"

        rows = [
            {"file": "a.py", "symbol": "func1"},
            {"file": "b.py", "symbol": "func2"},
        ]
        base_path.write_text(
            "\n".join(json.dumps(r) for r in rows) + "\n",
            encoding="utf-8",
        )

        delta_ops = [{"op": "delete", "key": "a.py"}]
        delta_path.write_text(
            "\n".join(json.dumps(op) for op in delta_ops) + "\n",
            encoding="utf-8",
        )

        grouped, delta_count = load_grouped_rows_with_delta(
            temp_index_dir, "symbols", "file"
        )
        assert delta_count == 1
        assert "a.py" not in grouped
        assert "b.py" in grouped

    def test_with_set_delta(self, temp_index_dir):
        base_path = temp_index_dir / "symbols.jsonl"
        delta_path = temp_index_dir / "symbols.delta.jsonl"

        rows = [{"file": "a.py", "symbol": "func1"}]
        base_path.write_text(
            "\n".join(json.dumps(r) for r in rows) + "\n",
            encoding="utf-8",
        )

        delta_ops = [
            {"op": "set", "key": "a.py", "rows": [{"file": "a.py", "symbol": "func1_updated"}]}
        ]
        delta_path.write_text(
            "\n".join(json.dumps(op) for op in delta_ops) + "\n",
            encoding="utf-8",
        )

        grouped, delta_count = load_grouped_rows_with_delta(
            temp_index_dir, "symbols", "file"
        )
        assert delta_count == 1
        assert grouped["a.py"][0]["symbol"] == "func1_updated"


class TestLoadIndexRows:
    def test_loads_and_flattens(self, temp_index_dir):
        base_path = temp_index_dir / "symbols.jsonl"
        rows = [
            {"file": "b.py", "symbol": "func2"},
            {"file": "a.py", "symbol": "func1"},
        ]
        base_path.write_text(
            "\n".join(json.dumps(r) for r in rows) + "\n",
            encoding="utf-8",
        )

        result = load_index_rows(temp_index_dir, "symbols", "file")
        assert len(result) == 2
        # Should be sorted by file key
        assert result[0]["symbol"] == "func1"
        assert result[1]["symbol"] == "func2"


class TestAppendDeltaOps:
    def test_appends_ops(self, temp_index_dir):
        ops = [
            {"op": "delete", "key": "a.py"},
            {"op": "set", "key": "b.py", "rows": [{"symbol": "func"}]},
        ]
        count = append_delta_ops(temp_index_dir, "symbols", ops)
        assert count == 2

        delta_path = temp_index_dir / "symbols.delta.jsonl"
        assert delta_path.exists()
        lines = delta_path.read_text(encoding="utf-8").strip().split("\n")
        assert len(lines) == 2

    def test_empty_ops_returns_zero(self, temp_index_dir):
        count = append_delta_ops(temp_index_dir, "symbols", [])
        assert count == 0


class TestWriteJsonlAtomic:
    def test_writes_rows(self, temp_index_dir):
        path = temp_index_dir / "output.jsonl"
        rows = [
            {"symbol": "func1"},
            {"symbol": "func2"},
        ]
        write_jsonl_atomic(path, rows)

        assert path.exists()
        loaded = load_jsonl_mmap(path)
        assert len(loaded) == 2
        assert loaded[0]["symbol"] == "func1"

    def test_creates_parent_dirs(self, temp_index_dir):
        path = temp_index_dir / "subdir" / "nested" / "output.jsonl"
        rows = [{"symbol": "func1"}]
        write_jsonl_atomic(path, rows)
        assert path.exists()

    def test_overwrites_existing(self, temp_index_dir):
        path = temp_index_dir / "output.jsonl"
        path.write_text('{"old": "data"}\n', encoding="utf-8")

        rows = [{"new": "data"}]
        write_jsonl_atomic(path, rows)

        loaded = load_jsonl_mmap(path)
        assert len(loaded) == 1
        assert "new" in loaded[0]
        assert "old" not in loaded[0]


class TestClearDeltaFile:
    def test_clears_delta(self, temp_index_dir):
        delta_path = temp_index_dir / "symbols.delta.jsonl"
        delta_path.write_text('{"op": "delete", "key": "a.py"}\n', encoding="utf-8")

        clear_delta_file(temp_index_dir, "symbols")

        content = delta_path.read_text(encoding="utf-8")
        assert content.strip() == ""

    def test_clears_nonexistent_delta(self, temp_index_dir):
        # Should not raise
        clear_delta_file(temp_index_dir, "symbols")
        delta_path = temp_index_dir / "symbols.delta.jsonl"
        assert delta_path.exists()
