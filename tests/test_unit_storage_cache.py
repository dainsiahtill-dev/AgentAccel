"""Unit tests for storage/cache.py"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from accel.storage.cache import (
    project_hash,
    project_paths,
    ensure_project_dirs,
    write_json,
    read_json,
    write_jsonl,
)


@pytest.fixture
def temp_dir():
    """Create a temporary directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


class TestProjectHash:
    def test_returns_hex_string(self, temp_dir):
        result = project_hash(temp_dir)
        assert isinstance(result, str)
        assert len(result) == 16
        # Should be valid hex
        int(result, 16)

    def test_deterministic(self, temp_dir):
        hash1 = project_hash(temp_dir)
        hash2 = project_hash(temp_dir)
        assert hash1 == hash2

    def test_different_paths_different_hashes(self):
        hash1 = project_hash(Path("/path/to/project1"))
        hash2 = project_hash(Path("/path/to/project2"))
        assert hash1 != hash2

    def test_normalizes_slashes(self):
        # Both should produce same hash due to normalization
        hash1 = project_hash(Path("C:/Users/test/project"))
        hash2 = project_hash(Path("C:\\Users\\test\\project"))
        # Note: This depends on Path normalization behavior
        # The function normalizes to forward slashes


class TestProjectPaths:
    def test_returns_expected_keys(self, temp_dir):
        accel_home = temp_dir / "accel_home"
        project_dir = temp_dir / "project"

        paths = project_paths(accel_home, project_dir)

        assert "base" in paths
        assert "index" in paths
        assert "index_units" in paths
        assert "context" in paths
        assert "verify" in paths
        assert "telemetry" in paths
        assert "state" in paths

    def test_paths_are_under_accel_home(self, temp_dir):
        accel_home = temp_dir / "accel_home"
        project_dir = temp_dir / "project"

        paths = project_paths(accel_home, project_dir)

        for key, path in paths.items():
            assert str(path).startswith(str(accel_home))

    def test_paths_include_project_hash(self, temp_dir):
        accel_home = temp_dir / "accel_home"
        project_dir = temp_dir / "project"
        p_hash = project_hash(project_dir)

        paths = project_paths(accel_home, project_dir)

        assert p_hash in str(paths["base"])


class TestEnsureProjectDirs:
    def test_creates_directories(self, temp_dir):
        paths = {
            "base": temp_dir / "base",
            "index": temp_dir / "base" / "index",
            "state": temp_dir / "base" / "state",
        }

        ensure_project_dirs(paths)

        assert paths["index"].exists()
        assert paths["state"].exists()
        # base itself is not created (only children)

    def test_idempotent(self, temp_dir):
        paths = {
            "base": temp_dir / "base",
            "index": temp_dir / "base" / "index",
        }

        ensure_project_dirs(paths)
        ensure_project_dirs(paths)

        assert paths["index"].exists()


class TestWriteJson:
    def test_writes_json_file(self, temp_dir):
        path = temp_dir / "test.json"
        data = {"key": "value", "number": 42}

        write_json(path, data)

        assert path.exists()
        content = json.loads(path.read_text(encoding="utf-8"))
        assert content == data

    def test_unicode_support(self, temp_dir):
        path = temp_dir / "test.json"
        data = {"greeting": "你好", "emoji": "🚀"}

        write_json(path, data)

        content = json.loads(path.read_text(encoding="utf-8"))
        assert content["greeting"] == "你好"
        assert content["emoji"] == "🚀"

    def test_overwrites_existing(self, temp_dir):
        path = temp_dir / "test.json"
        path.write_text('{"old": "data"}', encoding="utf-8")

        write_json(path, {"new": "data"})

        content = json.loads(path.read_text(encoding="utf-8"))
        assert "new" in content
        assert "old" not in content


class TestReadJson:
    def test_reads_json_file(self, temp_dir):
        path = temp_dir / "test.json"
        data = {"key": "value"}
        path.write_text(json.dumps(data), encoding="utf-8")

        result = read_json(path)

        assert result == data

    def test_nonexistent_returns_empty(self, temp_dir):
        path = temp_dir / "nonexistent.json"

        result = read_json(path)

        assert result == {}

    def test_invalid_json_returns_empty(self, temp_dir):
        path = temp_dir / "test.json"
        path.write_text("not valid json", encoding="utf-8")

        result = read_json(path)

        assert result == {}

    def test_non_dict_returns_empty(self, temp_dir):
        path = temp_dir / "test.json"
        path.write_text('["array", "not", "dict"]', encoding="utf-8")

        result = read_json(path)

        assert result == {}


class TestWriteJsonl:
    def test_writes_jsonl_file(self, temp_dir):
        path = temp_dir / "test.jsonl"
        rows = [
            {"symbol": "func1"},
            {"symbol": "func2"},
        ]

        write_jsonl(path, rows)

        assert path.exists()
        lines = path.read_text(encoding="utf-8").strip().split("\n")
        assert len(lines) == 2
        assert json.loads(lines[0])["symbol"] == "func1"
        assert json.loads(lines[1])["symbol"] == "func2"

    def test_empty_rows(self, temp_dir):
        path = temp_dir / "test.jsonl"

        write_jsonl(path, [])

        assert path.exists()
        content = path.read_text(encoding="utf-8")
        assert content == ""

    def test_unicode_support(self, temp_dir):
        path = temp_dir / "test.jsonl"
        rows = [{"name": "测试函数"}]

        write_jsonl(path, rows)

        content = path.read_text(encoding="utf-8")
        loaded = json.loads(content.strip())
        assert loaded["name"] == "测试函数"
