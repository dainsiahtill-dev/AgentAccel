from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from accel.query.relation_query import (
    get_file_dependencies,
    get_inheritance_tree,
)


@pytest.fixture
def temp_index_dir():
    """Create a temporary index directory with test data."""
    with tempfile.TemporaryDirectory() as tmpdir:
        index_dir = Path(tmpdir)

        symbols = [
            {
                "symbol": "BaseModel",
                "qualified_name": "models.base.BaseModel",
                "kind": "class",
                "lang": "python",
                "file": "models/base.py",
                "line_start": 10,
                "line_end": 50,
                "bases": [],
            },
            {
                "symbol": "UserModel",
                "qualified_name": "models.user.UserModel",
                "kind": "class",
                "lang": "python",
                "file": "models/user.py",
                "line_start": 5,
                "line_end": 30,
                "bases": ["BaseModel"],
            },
            {
                "symbol": "AdminModel",
                "qualified_name": "models.admin.AdminModel",
                "kind": "class",
                "lang": "python",
                "file": "models/admin.py",
                "line_start": 5,
                "line_end": 25,
                "bases": ["UserModel"],
            },
            {
                "symbol": "IRepository",
                "qualified_name": "interfaces.IRepository",
                "kind": "class",
                "lang": "python",
                "file": "interfaces/repository.py",
                "line_start": 5,
                "line_end": 20,
                "bases": ["Protocol"],
            },
            {
                "symbol": "UserRepository",
                "qualified_name": "repos.UserRepository",
                "kind": "class",
                "lang": "python",
                "file": "repos/user_repo.py",
                "line_start": 10,
                "line_end": 50,
                "bases": ["IRepository"],
            },
            {
                "symbol": "process_data",
                "qualified_name": "utils.process_data",
                "kind": "function",
                "lang": "python",
                "file": "utils/data.py",
                "line_start": 5,
                "line_end": 15,
            },
        ]

        deps = [
            {
                "file": "models/user.py",
                "target_symbol": "BaseModel",
                "edge_to": "models/base.py",
            },
            {
                "file": "models/admin.py",
                "target_symbol": "UserModel",
                "edge_to": "models/user.py",
            },
            {
                "file": "repos/user_repo.py",
                "target_symbol": "IRepository",
                "edge_to": "interfaces/repository.py",
            },
            {
                "file": "repos/user_repo.py",
                "target_symbol": "UserModel",
                "edge_to": "models/user.py",
            },
            {
                "file": "main.py",
                "target_symbol": "UserRepository",
                "edge_to": "repos/user_repo.py",
            },
            {
                "file": "main.py",
                "target_symbol": "AdminModel",
                "edge_to": "models/admin.py",
            },
        ]

        symbols_path = index_dir / "symbols.jsonl"
        symbols_path.write_text(
            "\n".join(json.dumps(row) for row in symbols) + "\n",
            encoding="utf-8",
        )

        deps_path = index_dir / "deps.jsonl"
        deps_path.write_text(
            "\n".join(json.dumps(row) for row in deps) + "\n",
            encoding="utf-8",
        )

        yield index_dir


class TestGetInheritanceTree:
    def test_get_all_classes(self, temp_index_dir):
        result = get_inheritance_tree(temp_index_dir)
        assert result["class_count"] >= 4
        assert "tree" in result
        assert "flat_classes" in result

    def test_get_specific_class(self, temp_index_dir):
        result = get_inheritance_tree(temp_index_dir, class_name="UserModel")
        assert result["class_count"] == 1
        assert result["filter"] == "usermodel"
        assert len(result["flat_classes"]) == 1
        assert result["flat_classes"][0]["symbol"] == "UserModel"

    def test_inheritance_chain(self, temp_index_dir):
        result = get_inheritance_tree(temp_index_dir)
        flat = result["flat_classes"]

        user_model = next(
            (c for c in flat if c["symbol"] == "UserModel"), None
        )
        assert user_model is not None
        assert "BaseModel" in user_model["bases"]

        admin_model = next(
            (c for c in flat if c["symbol"] == "AdminModel"), None
        )
        assert admin_model is not None
        assert "UserModel" in admin_model["bases"]

    def test_interface_detection(self, temp_index_dir):
        result = get_inheritance_tree(temp_index_dir, include_interfaces=True)
        flat = result["flat_classes"]

        interface = next(
            (c for c in flat if c["symbol"] == "IRepository"), None
        )
        assert interface is not None
        assert interface["is_interface"] is True

    def test_exclude_interfaces(self, temp_index_dir):
        result = get_inheritance_tree(temp_index_dir, include_interfaces=False)
        flat = result["flat_classes"]

        interface = next(
            (c for c in flat if c["symbol"] == "IRepository"), None
        )
        assert interface is None

    def test_tree_structure(self, temp_index_dir):
        result = get_inheritance_tree(temp_index_dir)
        assert "tree" in result
        tree = result["tree"]
        assert isinstance(tree, list)

        for root in tree:
            assert "symbol" in root
            assert "children" in root


class TestGetFileDependencies:
    def test_get_all_dependencies(self, temp_index_dir):
        result = get_file_dependencies(temp_index_dir)
        assert "file_count" in result
        assert "files" in result
        assert result["direction"] == "both"

    def test_get_file_imports(self, temp_index_dir):
        result = get_file_dependencies(
            temp_index_dir,
            file_path="repos/user_repo.py",
            direction="imports",
        )
        assert result["file"] == "repos/user_repo.py"
        assert result["direction"] == "imports"
        assert len(result["imports"]) == 2
        assert "interfaces/repository.py" in result["imports"]
        assert "models/user.py" in result["imports"]

    def test_get_file_exported(self, temp_index_dir):
        result = get_file_dependencies(
            temp_index_dir,
            file_path="models/user.py",
            direction="exported",
        )
        assert result["file"] == "models/user.py"
        assert result["direction"] == "exported"
        assert len(result["exported_to"]) >= 1

    def test_get_file_both_directions(self, temp_index_dir):
        result = get_file_dependencies(
            temp_index_dir,
            file_path="models/user.py",
            direction="both",
        )
        assert "imports" in result
        assert "exported_to" in result
        assert result["imports_count"] >= 0
        assert result["exported_count"] >= 0

    def test_nonexistent_file(self, temp_index_dir):
        result = get_file_dependencies(
            temp_index_dir,
            file_path="nonexistent/file.py",
        )
        assert result["imports"] == []
        assert result["exported_to"] == []

    def test_total_edges(self, temp_index_dir):
        result = get_file_dependencies(temp_index_dir)
        assert result["total_edges"] == 6
