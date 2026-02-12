from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_run_benchmarks_module():
    script_path = Path(__file__).resolve().parents[2] / "scripts" / "run_benchmarks.py"
    spec = importlib.util.spec_from_file_location("run_benchmarks_script", script_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_render_markdown_report_includes_summary_and_task_rows() -> None:
    module = _load_run_benchmarks_module()
    payload = {
        "generated_at": "2026-02-12T00:00:00Z",
        "project_dir": "/tmp/demo",
        "tasks_file": "/tmp/tasks.json",
        "index_mode": "update",
        "run_verify": False,
        "summary": {
            "tasks": 1,
            "context_build_seconds_avg": 0.123456,
            "context_build_seconds_p50": 0.123456,
            "context_tokens_avg": 120.0,
            "token_reduction_vs_full_index_avg": 0.5,
            "token_reduction_vs_changed_files_avg": 0.25,
            "top_file_recall_avg": 1.0,
        },
        "results": [
            {
                "id": "demo-task",
                "context_tokens": 120,
                "token_reduction_vs_full_index": 0.5,
                "token_reduction_vs_changed_files": 0.25,
                "top_file_recall": 1.0,
                "context_build_seconds": 0.123456,
                "verify_exit_code": None,
            }
        ],
    }

    markdown = module._render_markdown_report(payload)

    assert "# AgentAccel Benchmark Report" in markdown
    assert "| tasks | 1 |" in markdown
    assert "| demo-task | 120 | 50.00% | 25.00% | 100.00% | 0.123456 | n/a |" in markdown
