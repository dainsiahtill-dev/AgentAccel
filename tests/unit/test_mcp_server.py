from __future__ import annotations

import asyncio
import json
from pathlib import Path

import accel.mcp_server as mcp_server


def test_create_server_registers_core_tools_and_resources() -> None:
    server = mcp_server.create_server()
    tools = asyncio.run(server.get_tools())
    resources = asyncio.run(server.get_resources())
    templates = asyncio.run(server.get_resource_templates())

    tool_names = set(tools.keys())
    resource_uris = set(resources.keys())
    template_uris = set(templates.keys())

    assert "accel_index_build" in tool_names
    assert "accel_index_update" in tool_names
    assert "accel_context" in tool_names
    assert "accel_verify" in tool_names
    assert "agent-accel://status" in resource_uris
    assert "agent-accel://template/{kind}" in template_uris


def test_tool_context_requires_task() -> None:
    try:
        mcp_server._tool_context(task="   ")
    except ValueError as exc:
        assert str(exc) == "task is required"
    else:
        raise AssertionError("expected ValueError for empty task")


def test_tool_context_builds_pack_for_temp_project(tmp_path: Path) -> None:
    project_dir = tmp_path / "sample_project"
    (project_dir / "src").mkdir(parents=True)
    (project_dir / "src" / "sample.py").write_text(
        "def add(a: int, b: int) -> int:\n    return a + b\n",
        encoding="utf-8",
    )
    (project_dir / "accel.yaml").write_text(
        json.dumps(
            {
                "verify": {"python": ["python -c \"print('ok')\""], "node": []},
                "index": {"include": ["src/**"], "exclude": [], "max_file_mb": 2},
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    build = mcp_server._tool_index_build(project=str(project_dir), full=True)
    assert build["status"] == "ok"

    context = mcp_server._tool_context(
        project=str(project_dir),
        task="Summarize sample add implementation",
        changed_files=["src/sample.py"],
    )
    assert context["status"] == "ok"
    assert Path(context["out"]).is_file()


def test_tool_verify_runs_with_evidence_mode(tmp_path: Path) -> None:
    project_dir = tmp_path / "verify_project"
    (project_dir / "src").mkdir(parents=True)
    (project_dir / "src" / "sample.py").write_text("print('ok')\n", encoding="utf-8")
    (project_dir / "accel.yaml").write_text(
        json.dumps(
            {
                "verify": {"python": ["python -c \"print('ok')\""], "node": []},
                "index": {"include": ["src/**"], "exclude": [], "max_file_mb": 2},
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    mcp_server._tool_index_build(project=str(project_dir), full=True)
    result = mcp_server._tool_verify(
        project=str(project_dir),
        changed_files=["src/sample.py"],
        evidence_run=True,
        per_command_timeout_seconds=10,
    )

    assert result["status"] in {"success", "degraded"}
    assert int(result["exit_code"]) in {0, 2}
