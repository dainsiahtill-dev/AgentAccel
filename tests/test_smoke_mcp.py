from __future__ import annotations

import pytest

import accel.mcp_server as mcp_server
from accel.mcp_server import SERVER_NAME, create_server


def test_create_mcp_server_smoke() -> None:
    server = create_server()
    assert server is not None
    assert SERVER_NAME == "agent-accel-mcp"
    assert hasattr(server, "run")


def test_tool_plan_and_gate_shapes_and_limits(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    fake_context_payload = {
        "status": "ok",
        "out": "context_pack.json",
        "out_meta": "context_pack.meta.json",
        "constraint_mode": "warn",
        "semantic_cache_hit": True,
        "warnings": ["fallback warning"],
        "selected_tests_count": 3,
        "selected_checks_count": 2,
        "verify_plan": {
            "target_tests": ["tests/test_alpha.py", "tests/test_beta.py"],
            "target_checks": ["pytest -q tests/test_alpha.py", "ruff check accel"],
            "selection_evidence": {"run_all": False},
        },
        "pack": {
            "top_files": [
                {"path": "accel/mcp_server.py", "score": 0.91, "reasons": ["changed"]},
                {"path": "accel/query/context_compiler.py", "score": 0.73, "reasons": ["symbol_match"]},
                {"path": "tests/test_smoke_mcp.py", "score": 0.52, "reasons": ["test_ownership"]},
            ],
            "snippets": [
                {
                    "path": "accel/mcp_server.py",
                    "line_start": 120,
                    "line_end": 180,
                    "symbol": "_tool_context",
                    "content": "def _tool_context(...): pass",
                },
                {
                    "path": "accel/query/context_compiler.py",
                    "line_start": 300,
                    "line_end": 360,
                    "symbol": "compile_context_pack",
                    "content": "def compile_context_pack(...): pass",
                },
            ],
        },
    }

    monkeypatch.setattr(
        mcp_server, "_tool_context", lambda **_: dict(fake_context_payload)
    )

    result = mcp_server._tool_plan_and_gate(
        project=str(tmp_path),
        task="Implement one-shot planning flow",
        max_affected_files=2,
        max_snippets=1,
    )

    assert result["status"] == "ok"
    assert result["mode"] == "plan_and_gate"
    assert len(result["affected_files"]) == 2
    assert result["affected_files"][0]["path"] == "accel/mcp_server.py"
    assert len(result["minimal_snippets"]) == 1
    assert result["impacted_tests"]["target_tests"] == [
        "tests/test_alpha.py",
        "tests/test_beta.py",
    ]
    assert result["impacted_tests"]["target_checks"] == [
        "pytest -q tests/test_alpha.py",
        "ruff check accel",
    ]
    assert result["receipts"]["accel_context"]["status"] == "ok"
    assert result["governance"]["workflow"] == [
        "hp_start_run",
        "hp_create_blueprint",
        "hp_create_snapshot",
        "hp_allow_implementation",
    ]
    assert "pack" not in result


def test_tool_plan_and_gate_optional_sections(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    fake_context_payload = {
        "status": "ok",
        "out": "context_pack.json",
        "out_meta": "context_pack.meta.json",
        "constraint_mode": "warn",
        "semantic_cache_hit": False,
        "warnings": [],
        "selected_tests_count": 0,
        "selected_checks_count": 0,
        "verify_plan": {"target_tests": [], "target_checks": []},
        "pack": {"top_files": [], "snippets": []},
    }
    monkeypatch.setattr(
        mcp_server, "_tool_context", lambda **_: dict(fake_context_payload)
    )

    result = mcp_server._tool_plan_and_gate(
        project=str(tmp_path),
        task="Plan only",
        include_governance=False,
        include_pack=True,
    )

    assert "governance" not in result
    assert "pack" in result
    assert result["pack"] == {"top_files": [], "snippets": []}


def test_tool_plan_and_gate_requires_task(tmp_path) -> None:
    with pytest.raises(ValueError, match="task is required"):
        mcp_server._tool_plan_and_gate(project=str(tmp_path), task="")


def test_tool_plan_and_gate_uses_precomputed_context_payload(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    fake_context_payload = {
        "status": "ok",
        "out": "context_pack.json",
        "out_meta": "context_pack.meta.json",
        "constraint_mode": "warn",
        "semantic_cache_hit": False,
        "warnings": [],
        "selected_tests_count": 1,
        "selected_checks_count": 1,
        "verify_plan": {
            "target_tests": ["tests/test_alpha.py"],
            "target_checks": ["pytest -q tests/test_alpha.py"],
            "selection_evidence": {"run_all": False},
        },
        "pack": {
            "top_files": [{"path": "accel/mcp_server.py", "score": 0.9, "reasons": []}],
            "snippets": [
                {
                    "path": "accel/mcp_server.py",
                    "line_start": 1,
                    "line_end": 3,
                    "symbol": "x",
                    "content": "pass",
                }
            ],
        },
    }

    def _unexpected_context_call(**_: object) -> dict:
        raise AssertionError("_tool_context should not be called")

    monkeypatch.setattr(mcp_server, "_tool_context", _unexpected_context_call)

    result = mcp_server._tool_plan_and_gate(
        project=str(tmp_path),
        task="reuse context result",
        context_payload=fake_context_payload,
    )

    assert result["status"] == "ok"
    assert result["affected_files"][0]["path"] == "accel/mcp_server.py"
