from __future__ import annotations

import sys
import time
from pathlib import Path

from accel.verify.runners import run_command


def _python_command(code: str) -> str:
    return f'"{sys.executable}" -c "{code}"'


def test_run_command_non_interactive_stdin(tmp_path: Path) -> None:
    command = _python_command("import sys; data=sys.stdin.read(); print(len(data))")

    result = run_command(command=command, cwd=tmp_path, timeout_seconds=5)

    assert bool(result["timed_out"]) is False
    assert int(result["exit_code"]) == 0
    assert str(result["stdout"]).strip() == "0"


def test_run_command_timeout_is_bounded(tmp_path: Path) -> None:
    command = _python_command("import time; time.sleep(5)")
    started = time.perf_counter()

    result = run_command(command=command, cwd=tmp_path, timeout_seconds=1)

    elapsed = time.perf_counter() - started
    assert bool(result["timed_out"]) is True
    assert int(result["exit_code"]) == 124
    assert elapsed < 6
