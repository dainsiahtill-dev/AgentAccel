from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from accel.query.project_stats import (
    get_health_status,
    get_project_stats,
)


@pytest.fixture
def temp_index_dir():
    """Create a temporary index directory with test data."""
    with tempfile.TemporaryDirectory() as tmpdir:
        index_dir = Path(tmpdir) / "index"
        index_dir.mkdir(parents=True, exist_ok=True)

        state_dir = Path(tmpdir) / "state"
        state_dir.mkdir(parents=True, exist_ok=True)

        symbols = [
            {
                "symbol": "function_one",
                "kind": "function",
                "lang": "python",
                "file": "src/module_a.py",
            },
            {
                "symbol": "function_two",
                "kind": "function",
                "lang": "python",
                "file": "src/module_a.py",
            },
            {
                "symbol": "ClassOne",
                "kind": "class",
                "lang": "python",
                "file": "src/module_b.py",
            },
            {
                "symbol": "method_one",
                "kind": "method",
                "lang": "python",
                "file": "src/module_b.py",
            },
            {
                "symbol": "tsFunction",
                "kind": "function",
                "lang": "typescript",
                "file": "src/utils.ts",
            },
        ]

        refs = [
            {"file": "src/module_a.py", "target_symbol": "ClassOne"},
            {"file": "src/module_b.py", "target_symbol": "function_one"},
        ]

        deps = [
            {"file": "src/module_a.py", "edge_to": "src/module_b.py"},
            {"file": "src/utils.ts", "edge_to": "src/module_a.py"},
        ]

        symbols_path = index_dir / "symbols.jsonl"
        symbols_path.write_text(
            "\n".join(json.dumps(row) for row in symbols) + "\n",
            encoding="utf-8",
        )

        refs_path = index_dir / "references.jsonl"
        refs_path.write_text(
            "\n".join(json.dumps(row) for row in refs) + "\n",
            encoding="utf-8",
        )

        deps_path = index_dir / "deps.jsonl"
        deps_path.write_text(
            "\n".join(json.dumps(row) for row in deps) + "\n",
            encoding="utf-8",
        )

        yield {
            "index_dir": index_dir,
            "state_dir": state_dir,
            "project_dir": Path(tmpdir),
            "paths": {
                "index": index_dir,
                "state": state_dir,
            },
        }


@pytest.fixture
def empty_index_dir():
    """Create a temporary directory without index."""
    with tempfile.TemporaryDirectory() as tmpdir:
        index_dir = Path(tmpdir) / "index"
        index_dir.mkdir(parents=True, exist_ok=True)
        state_dir = Path(tmpdir) / "state"
        state_dir.mkdir(parents=True, exist_ok=True)

        yield {
            "index_dir": index_dir,
            "state_dir": state_dir,
            "project_dir": Path(tmpdir),
            "paths": {
                "index": index_dir,
                "state": state_dir,
            },
        }


class TestGetProjectStats:
    def test_basic_stats(self, temp_index_dir):
        result = get_project_stats(
            index_dir=temp_index_dir["index_dir"],
            project_dir=temp_index_dir["project_dir"],
        )

        assert result["status"] == "ok"
        assert "overview" in result
        assert "symbol_distribution" in result
        assert "relation_stats" in result

    def test_overview_counts(self, temp_index_dir):
        result = get_project_stats(
            index_dir=temp_index_dir["index_dir"],
            project_dir=temp_index_dir["project_dir"],
        )

        overview = result["overview"]
        assert overview["total_files"] == 3
        assert overview["total_symbols"] == 5
        assert "python" in overview["languages"]
        assert "typescript" in overview["languages"]
        assert overview["languages"]["python"] == 4
        assert overview["languages"]["typescript"] == 1

    def test_symbol_distribution(self, temp_index_dir):
        result = get_project_stats(
            index_dir=temp_index_dir["index_dir"],
            project_dir=temp_index_dir["project_dir"],
        )

        dist = result["symbol_distribution"]
        assert dist["function"] == 3
        assert dist["class"] == 1
        assert dist["method"] == 1

    def test_relation_stats(self, temp_index_dir):
        result = get_project_stats(
            index_dir=temp_index_dir["index_dir"],
            project_dir=temp_index_dir["project_dir"],
        )

        rel_stats = result["relation_stats"]
        assert rel_stats["references_count"] == 2
        assert rel_stats["dependencies_count"] == 2

    def test_last_indexed_present(self, temp_index_dir):
        result = get_project_stats(
            index_dir=temp_index_dir["index_dir"],
            project_dir=temp_index_dir["project_dir"],
        )

        assert result["overview"]["last_indexed"] is not None

    def test_no_index(self, empty_index_dir):
        result = get_project_stats(
            index_dir=empty_index_dir["index_dir"],
            project_dir=empty_index_dir["project_dir"],
        )

        assert result["status"] == "no_index"
        assert "message" in result


class TestGetHealthStatus:
    def test_healthy_status(self, temp_index_dir):
        result = get_health_status(
            index_dir=temp_index_dir["index_dir"],
            project_dir=temp_index_dir["project_dir"],
            paths=temp_index_dir["paths"],
        )

        assert result["status"] in {"healthy", "warning", "degraded"}
        assert "index" in result
        assert "cache" in result
        assert "jobs" in result

    def test_index_info(self, temp_index_dir):
        result = get_health_status(
            index_dir=temp_index_dir["index_dir"],
            project_dir=temp_index_dir["project_dir"],
            paths=temp_index_dir["paths"],
        )

        assert result["index"]["exists"] is True
        assert result["index"]["stale"] is False
        assert result["index"]["file_count"] >= 1

    def test_no_index_degraded(self, empty_index_dir):
        result = get_health_status(
            index_dir=empty_index_dir["index_dir"],
            project_dir=empty_index_dir["project_dir"],
            paths=empty_index_dir["paths"],
        )

        assert result["status"] == "degraded"
        assert result["index"]["exists"] is False
        assert "Symbol index not found" in result["issues"]

    def test_jobs_info(self, temp_index_dir):
        result = get_health_status(
            index_dir=temp_index_dir["index_dir"],
            project_dir=temp_index_dir["project_dir"],
            paths=temp_index_dir["paths"],
        )

        assert "active" in result["jobs"]
        assert "pending" in result["jobs"]
        assert result["jobs"]["active"] >= 0
        assert result["jobs"]["pending"] >= 0

    def test_paths_info(self, temp_index_dir):
        result = get_health_status(
            index_dir=temp_index_dir["index_dir"],
            project_dir=temp_index_dir["project_dir"],
            paths=temp_index_dir["paths"],
        )

        assert "paths" in result
        assert result["paths"]["project"] == str(temp_index_dir["project_dir"])
        assert result["paths"]["index"] == str(temp_index_dir["index_dir"])
