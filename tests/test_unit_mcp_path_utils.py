"""Unit tests for mcp/path_utils.py"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from accel.mcp.path_utils import (
    normalize_project_dir,
    resolve_path,
    normalize_relative_path,
)


class TestNormalizeProjectDir:
    def test_current_dir(self):
        result = normalize_project_dir(".")
        assert result.is_absolute()
        assert result == Path.cwd()

    def test_none_returns_cwd(self):
        result = normalize_project_dir(None)
        assert result.is_absolute()
        assert result == Path.cwd()

    def test_empty_string_returns_cwd(self):
        result = normalize_project_dir("")
        assert result.is_absolute()
        assert result == Path.cwd()

    def test_absolute_path_passthrough(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            result = normalize_project_dir(tmpdir)
            assert result.is_absolute()
            assert str(result).replace("\\", "/") == tmpdir.replace("\\", "/")

    def test_relative_path_becomes_absolute(self):
        result = normalize_project_dir("some/relative/path")
        assert result.is_absolute()
        assert "some/relative/path" in str(result).replace("\\", "/")


class TestResolvePath:
    def test_empty_returns_none(self):
        project_dir = Path.cwd()
        assert resolve_path(project_dir, None) is None
        assert resolve_path(project_dir, "") is None
        assert resolve_path(project_dir, "   ") is None

    def test_relative_path_resolved(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            project_dir = Path(tmpdir)
            result = resolve_path(project_dir, "subdir/file.py")
            assert result is not None
            assert result.is_absolute()
            assert "subdir" in str(result)
            assert "file.py" in str(result)

    def test_absolute_path_unchanged(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            project_dir = Path.cwd()
            result = resolve_path(project_dir, tmpdir)
            assert result is not None
            assert result.is_absolute()
            assert str(result).replace("\\", "/") == tmpdir.replace("\\", "/")


class TestNormalizeRelativePath:
    def test_forward_slashes_unchanged(self):
        assert normalize_relative_path("src/module/file.py") == "src/module/file.py"

    def test_backslashes_converted(self):
        assert normalize_relative_path("src\\module\\file.py") == "src/module/file.py"

    def test_mixed_slashes_converted(self):
        assert normalize_relative_path("src\\module/file.py") == "src/module/file.py"

    def test_whitespace_stripped(self):
        assert normalize_relative_path("  src/file.py  ") == "src/file.py"

    def test_empty_string(self):
        assert normalize_relative_path("") == ""

    def test_none_returns_empty(self):
        assert normalize_relative_path(None) == ""
