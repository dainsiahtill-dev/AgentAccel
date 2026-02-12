from __future__ import annotations

from pathlib import Path

import accel.cli as cli


def test_cli_parser_includes_explain_command() -> None:
    parser = cli.build_parser()
    args = parser.parse_args(
        [
            "explain",
            "--project",
            ".",
            "--task",
            "explain selection",
            "--output",
            "json",
        ]
    )
    assert args.command == "explain"
    assert args.func == cli.cmd_explain


def test_cmd_explain_outputs_json_payload(tmp_path: Path, monkeypatch, capsys) -> None:
    project_dir = tmp_path / "repo"
    project_dir.mkdir(parents=True)

    monkeypatch.setattr(
        cli,
        "resolve_effective_config",
        lambda project_dir: {"runtime": {"accel_home": str(tmp_path / ".accel-home")}},
    )
    monkeypatch.setattr(
        cli,
        "explain_context_selection",
        lambda **kwargs: {
            "version": 1,
            "schema_version": 1,
            "task": str(kwargs.get("task", "")),
            "selected": [{"path": "src/main.py", "score": 0.95}],
            "alternatives": [{"path": "src/alt.py", "score": 0.9}],
        },
    )

    args = cli.build_parser().parse_args(
        [
            "explain",
            "--project",
            str(project_dir),
            "--task",
            "find relevant files",
            "--output",
            "json",
        ]
    )
    exit_code = int(args.func(args))
    captured = capsys.readouterr().out

    assert exit_code == 0
    assert '"schema_version": 1' in captured
    assert "src/main.py" in captured
