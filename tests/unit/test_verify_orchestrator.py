from __future__ import annotations

import json
import time
from pathlib import Path

import accel.verify.orchestrator as orchestrator
from accel.verify.orchestrator import _resolve_verify_workers, run_verify, run_verify_with_callback


def test_resolve_verify_workers_prefers_verify_workers() -> None:
    assert _resolve_verify_workers({"max_workers": 12, "verify_workers": 5}) == 5


def test_resolve_verify_workers_falls_back_to_max_workers() -> None:
    assert _resolve_verify_workers({"max_workers": 7}) == 7


def test_resolve_verify_workers_normalizes_invalid_values() -> None:
    assert _resolve_verify_workers({"max_workers": 0, "verify_workers": 0}) == 1
    assert _resolve_verify_workers({"max_workers": "abc", "verify_workers": "xyz"}) == 8


def _base_config(
    tmp_path: Path,
    *,
    fail_fast: bool = False,
    cache_enabled: bool = True,
    cache_failed_results: bool = False,
) -> dict[str, object]:
    return {
        "runtime": {
            "accel_home": str(tmp_path / ".accel-home"),
            "verify_workers": 4,
            "max_workers": 4,
            "per_command_timeout_seconds": 5,
            "verify_fail_fast": fail_fast,
            "verify_cache_enabled": cache_enabled,
            "verify_cache_failed_results": cache_failed_results,
            "verify_cache_ttl_seconds": 600,
            "verify_cache_failed_ttl_seconds": 120,
            "verify_cache_max_entries": 100,
        },
        "verify": {
            "python": [],
            "node": [],
        },
    }


def test_run_verify_cache_hit_skips_runner(tmp_path: Path, monkeypatch) -> None:
    project_dir = tmp_path / "repo"
    (project_dir / "src").mkdir(parents=True, exist_ok=True)
    changed_file = project_dir / "src" / "foo.py"
    changed_file.write_text("x = 1\n", encoding="utf-8")

    commands = ["python -m ruff check ."]
    calls: list[str] = []

    def fake_select_verify_commands(config, changed_files=None):
        return list(commands)

    def fake_run_command(command: str, cwd: Path, timeout_seconds: int):
        calls.append(command)
        return {
            "command": command,
            "exit_code": 0,
            "duration_seconds": 0.01,
            "stdout": "ok",
            "stderr": "",
            "timed_out": False,
        }

    monkeypatch.setattr(orchestrator, "select_verify_commands", fake_select_verify_commands)
    monkeypatch.setattr(orchestrator, "run_command", fake_run_command)

    cfg = _base_config(tmp_path, fail_fast=False, cache_enabled=True)
    first = run_verify(project_dir=project_dir, config=cfg, changed_files=["src/foo.py"])
    second = run_verify(project_dir=project_dir, config=cfg, changed_files=["src/foo.py"])

    assert first["status"] == "success"
    assert second["status"] == "success"
    assert len(calls) == 1
    assert bool(first["results"][0]["cached"]) is False
    assert bool(second["results"][0]["cached"]) is True
    assert int(second["cache_hits"]) == 1


def test_run_verify_cache_invalidates_when_changed_file_updates(tmp_path: Path, monkeypatch) -> None:
    project_dir = tmp_path / "repo"
    (project_dir / "src").mkdir(parents=True, exist_ok=True)
    changed_file = project_dir / "src" / "foo.py"
    changed_file.write_text("x = 1\n", encoding="utf-8")

    commands = ["python -m ruff check ."]
    calls: list[str] = []

    def fake_select_verify_commands(config, changed_files=None):
        return list(commands)

    def fake_run_command(command: str, cwd: Path, timeout_seconds: int):
        calls.append(command)
        return {
            "command": command,
            "exit_code": 0,
            "duration_seconds": 0.01,
            "stdout": "ok",
            "stderr": "",
            "timed_out": False,
        }

    monkeypatch.setattr(orchestrator, "select_verify_commands", fake_select_verify_commands)
    monkeypatch.setattr(orchestrator, "run_command", fake_run_command)

    cfg = _base_config(tmp_path, fail_fast=False, cache_enabled=True)
    run_verify(project_dir=project_dir, config=cfg, changed_files=["src/foo.py"])
    time.sleep(0.002)
    changed_file.write_text("x = 2\n", encoding="utf-8")
    second = run_verify(project_dir=project_dir, config=cfg, changed_files=["src/foo.py"])

    assert len(calls) == 2
    assert bool(second["results"][0]["cached"]) is False


def test_run_verify_disables_cache_without_changed_files(tmp_path: Path, monkeypatch) -> None:
    project_dir = tmp_path / "repo"
    project_dir.mkdir(parents=True, exist_ok=True)

    commands = ["python -m ruff check ."]
    calls: list[str] = []

    def fake_select_verify_commands(config, changed_files=None):
        return list(commands)

    def fake_run_command(command: str, cwd: Path, timeout_seconds: int):
        calls.append(command)
        return {
            "command": command,
            "exit_code": 0,
            "duration_seconds": 0.01,
            "stdout": "ok",
            "stderr": "",
            "timed_out": False,
        }

    monkeypatch.setattr(orchestrator, "select_verify_commands", fake_select_verify_commands)
    monkeypatch.setattr(orchestrator, "run_command", fake_run_command)

    cfg = _base_config(tmp_path, fail_fast=False, cache_enabled=True)
    first = run_verify(project_dir=project_dir, config=cfg, changed_files=[])
    second = run_verify(project_dir=project_dir, config=cfg, changed_files=[])

    assert bool(first["cache_enabled"]) is False
    assert bool(second["cache_enabled"]) is False
    assert len(calls) == 2


def test_run_verify_fail_fast_stops_remaining_commands(tmp_path: Path, monkeypatch) -> None:
    project_dir = tmp_path / "repo"
    (project_dir / "src").mkdir(parents=True, exist_ok=True)
    (project_dir / "src" / "foo.py").write_text("x = 1\n", encoding="utf-8")

    commands = [
        'python -c "print(\'step1\')"',
        'python -c "print(\'step2\')"',
        'python -c "print(\'step3\')"',
    ]
    calls: list[str] = []

    def fake_select_verify_commands(config, changed_files=None):
        return list(commands)

    def fake_run_command(command: str, cwd: Path, timeout_seconds: int):
        calls.append(command)
        if "step1" in command:
            return {
                "command": command,
                "exit_code": 1,
                "duration_seconds": 0.01,
                "stdout": "",
                "stderr": "boom",
                "timed_out": False,
            }
        return {
            "command": command,
            "exit_code": 0,
            "duration_seconds": 0.01,
            "stdout": "ok",
            "stderr": "",
            "timed_out": False,
        }

    monkeypatch.setattr(orchestrator, "select_verify_commands", fake_select_verify_commands)
    monkeypatch.setattr(orchestrator, "run_command", fake_run_command)

    cfg = _base_config(tmp_path, fail_fast=True, cache_enabled=False)
    result = run_verify(project_dir=project_dir, config=cfg, changed_files=["src/foo.py"])

    assert result["status"] == "failed"
    assert int(result["exit_code"]) == 3
    assert calls == [commands[0]]
    assert list(result["fail_fast_skipped_commands"]) == commands[1:]


def test_run_verify_with_callback_non_fail_fast_uses_cache_key_function(tmp_path: Path, monkeypatch) -> None:
    project_dir = tmp_path / "repo_callback_non_fail_fast"
    (project_dir / "src").mkdir(parents=True, exist_ok=True)
    changed_file = project_dir / "src" / "foo.py"
    changed_file.write_text("x = 1\n", encoding="utf-8")

    commands = ['python -c "print(\'ok\')"']

    def fake_select_verify_commands(config, changed_files=None):
        return list(commands)

    def fake_run_command(command: str, cwd: Path, timeout_seconds: int):
        return {
            "command": command,
            "exit_code": 0,
            "duration_seconds": 0.01,
            "stdout": "ok",
            "stderr": "",
            "timed_out": False,
        }

    monkeypatch.setattr(orchestrator, "select_verify_commands", fake_select_verify_commands)
    monkeypatch.setattr(orchestrator, "run_command", fake_run_command)

    cfg = _base_config(tmp_path, fail_fast=False, cache_enabled=True)
    result = run_verify_with_callback(
        project_dir=project_dir,
        config=cfg,
        changed_files=["src/foo.py"],
    )

    assert result["status"] == "success"
    assert int(result["exit_code"]) == 0
    assert len(result["results"]) == 1


def test_run_verify_caches_failed_results_when_enabled(tmp_path: Path, monkeypatch) -> None:
    project_dir = tmp_path / "repo_failed_cache"
    (project_dir / "src").mkdir(parents=True, exist_ok=True)
    changed_file = project_dir / "src" / "foo.py"
    changed_file.write_text("x = 1\n", encoding="utf-8")

    command = "python -m mypy ."
    calls: list[str] = []

    def fake_select_verify_commands(config, changed_files=None):
        return [command]

    def fake_run_command(current_command: str, cwd: Path, timeout_seconds: int):
        calls.append(current_command)
        return {
            "command": current_command,
            "exit_code": 1,
            "duration_seconds": 0.01,
            "stdout": "",
            "stderr": "mypy failed",
            "timed_out": False,
        }

    monkeypatch.setattr(orchestrator, "select_verify_commands", fake_select_verify_commands)
    monkeypatch.setattr(orchestrator, "run_command", fake_run_command)

    cfg = _base_config(tmp_path, fail_fast=False, cache_enabled=True, cache_failed_results=True)
    first = run_verify(project_dir=project_dir, config=cfg, changed_files=["src/foo.py"])
    second = run_verify(project_dir=project_dir, config=cfg, changed_files=["src/foo.py"])

    assert first["status"] == "failed"
    assert second["status"] == "failed"
    assert len(calls) == 1
    assert bool(second["results"][0]["cached"]) is True
    assert str(second["results"][0]["cache_kind"]) == "failure"


def test_run_verify_jsonl_command_result_includes_structured_fields(tmp_path: Path, monkeypatch) -> None:
    project_dir = tmp_path / "repo_jsonl_fields"
    (project_dir / "src").mkdir(parents=True, exist_ok=True)
    changed_file = project_dir / "src" / "foo.py"
    changed_file.write_text("x = 1\n", encoding="utf-8")

    command = "python -m ruff check ."

    def fake_select_verify_commands(config, changed_files=None):
        return [command]

    def fake_run_command(current_command: str, cwd: Path, timeout_seconds: int):
        return {
            "command": current_command,
            "exit_code": 0,
            "duration_seconds": 0.02,
            "stdout": "ok",
            "stderr": "",
            "timed_out": False,
        }

    monkeypatch.setattr(orchestrator, "select_verify_commands", fake_select_verify_commands)
    monkeypatch.setattr(orchestrator, "run_command", fake_run_command)

    cfg = _base_config(tmp_path, fail_fast=False, cache_enabled=False, cache_failed_results=False)
    result = run_verify(project_dir=project_dir, config=cfg, changed_files=["src/foo.py"])

    jsonl_path = Path(result["jsonl_path"])
    rows = [
        line
        for line in (
            json.loads(raw)
            for raw in jsonl_path.read_text(encoding="utf-8").splitlines()
            if raw.strip()
        )
        if isinstance(line, dict)
    ]
    command_rows = [row for row in rows if row.get("event") == "command_result"]
    assert len(command_rows) == 1
    row = command_rows[0]
    assert row.get("mode") in {"parallel", "sequential", "fail_fast"}
    assert isinstance(row.get("fail_fast"), bool)
    assert isinstance(row.get("cache_hits"), int)
    assert isinstance(row.get("cache_misses"), int)
    assert isinstance(row.get("fail_fast_skipped"), bool)
    assert int(row.get("command_index", 0)) == 1
    assert int(row.get("total_commands", 0)) == 1
    assert str(row.get("cache_kind", "")) in {"success", "failure"}
