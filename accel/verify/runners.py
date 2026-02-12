from __future__ import annotations

import os
import signal
import subprocess
import threading
import time
from pathlib import Path
from typing import Any, Callable

OUTPUT_TAIL_LIMIT = 12000


def _normalize_output(value: str | bytes | None) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def _resolve_preexec_fn() -> Callable[[], None] | None:
    """Return a safe preexec function for Unix-like systems only."""
    if os.name == "nt":
        return None
    setsid = getattr(os, "setsid", None)
    if callable(setsid):
        return setsid
    return None


def _kill_process_tree(process: subprocess.Popen[str]) -> None:
    """Kill a process tree with enhanced Windows support and multiple fallback methods."""
    if process.poll() is not None:
        return
    
    pid = process.pid
    
    if os.name == "nt":
        # Windows-specific process tree termination with multiple fallbacks
        killed = False
        
        # Method 1: taskkill (most reliable on Windows)
        try:
            result = subprocess.run(
                ["taskkill", "/PID", str(pid), "/T", "/F"],
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                stdin=subprocess.DEVNULL,
                timeout=10,
            )
            if result.returncode == 0 or result.returncode == 128:  # 128 means process was already dead
                killed = True
        except (subprocess.TimeoutExpired, OSError, FileNotFoundError):
            pass
        
        # Method 2: wmic if taskkill failed
        if not killed:
            try:
                subprocess.run(
                    ["wmic", "process", "where", f"ParentProcessId={pid}", "delete"],
                    check=False,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    stdin=subprocess.DEVNULL,
                    timeout=5,
                )
                killed = True
            except (subprocess.TimeoutExpired, OSError, FileNotFoundError):
                pass
        
        # Method 3: PowerShell if wmic failed
        if not killed:
            try:
                subprocess.run(
                    ["powershell", "-Command", f"Stop-Process -Id {pid} -Force -ErrorAction SilentlyContinue"],
                    check=False,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    stdin=subprocess.DEVNULL,
                    timeout=5,
                )
                killed = True
            except (subprocess.TimeoutExpired, OSError, FileNotFoundError):
                pass
        
        # Method 4: Direct kill as last resort
        if not killed:
            try:
                process.kill()
                killed = True
            except (OSError, PermissionError):
                pass
    else:
        # Unix-like systems: use process group
        try:
            killpg = getattr(os, "killpg", None)
            getpgid = getattr(os, "getpgid", None)
            if callable(killpg) and callable(getpgid):
                pgid = int(getpgid(pid))
                killpg(pgid, signal.SIGTERM)
                time.sleep(0.1)  # Give it a moment to terminate
                if process.poll() is None:
                    sigkill = getattr(signal, "SIGKILL", signal.SIGTERM)
                    killpg(pgid, sigkill)
            else:
                process.terminate()
                time.sleep(0.1)
                if process.poll() is None:
                    process.kill()
        except (OSError, PermissionError):
            # Fallback to direct kill
            try:
                process.kill()
            except (OSError, PermissionError):
                pass


def _read_pipe_stream(
    pipe: Any,
    stream_name: str,
    chunks: list[str],
    output_callback: Callable[[str, str], None] | None,
) -> None:
    if pipe is None:
        return
    try:
        while True:
            line = pipe.readline()
            if line == "":
                break
            text = _normalize_output(line)
            if not text:
                continue
            chunks.append(text)
            if output_callback is not None:
                try:
                    output_callback(stream_name, text)
                except Exception:
                    pass
    except Exception:
        return
    finally:
        try:
            pipe.close()
        except Exception:
            pass


def run_command(
    command: str,
    cwd: Path,
    timeout_seconds: int,
    output_callback: Callable[[str, str], None] | None = None,
) -> dict[str, Any]:
    """Run a command with enhanced timeout handling and process cleanup."""
    started = time.perf_counter()
    process: subprocess.Popen[str] | None = None
    stdout_chunks: list[str] = []
    stderr_chunks: list[str] = []
    stdout_thread: threading.Thread | None = None
    stderr_thread: threading.Thread | None = None
    
    try:
        # Create process with enhanced settings for better timeout handling
        process = subprocess.Popen(
            command,
            cwd=str(cwd),
            shell=True,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            # Set process group for better termination on Unix
            preexec_fn=_resolve_preexec_fn(),
        )

        stdout_thread = threading.Thread(
            target=_read_pipe_stream,
            args=(process.stdout, "stdout", stdout_chunks, output_callback),
            daemon=True,
        )
        stderr_thread = threading.Thread(
            target=_read_pipe_stream,
            args=(process.stderr, "stderr", stderr_chunks, output_callback),
            daemon=True,
        )
        stdout_thread.start()
        stderr_thread.start()

        timed_out = False
        try:
            process.wait(timeout=timeout_seconds)
        except subprocess.TimeoutExpired:
            timed_out = True
            _kill_process_tree(process)
            try:
                process.wait(timeout=5)
            except (subprocess.TimeoutExpired, OSError):
                pass

        if stdout_thread is not None:
            stdout_thread.join(timeout=1.0)
        if stderr_thread is not None:
            stderr_thread.join(timeout=1.0)

        stdout_text = "".join(stdout_chunks)
        stderr_text = "".join(stderr_chunks)
        if timed_out:
            elapsed = time.perf_counter() - started
            return {
                "command": command,
                "exit_code": 124,  # Standard timeout exit code
                "duration_seconds": round(elapsed, 3),
                "stdout": _normalize_output(stdout_text)[-OUTPUT_TAIL_LIMIT:],
                "stderr": _normalize_output(stderr_text)[-OUTPUT_TAIL_LIMIT:],
                "timed_out": True,
            }
        
        # Normal completion
        elapsed = time.perf_counter() - started
        return {
            "command": command,
            "exit_code": int(process.returncode or 0),
            "duration_seconds": round(elapsed, 3),
            "stdout": _normalize_output("".join(stdout_chunks))[-OUTPUT_TAIL_LIMIT:],
            "stderr": _normalize_output("".join(stderr_chunks))[-OUTPUT_TAIL_LIMIT:],
            "timed_out": False,
        }
        
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        # Handle process creation errors
        elapsed = time.perf_counter() - started
        return {
            "command": command,
            "exit_code": 1,
            "duration_seconds": round(elapsed, 3),
            "stdout": "",
            "stderr": f"agent-accel process error: {exc}",
            "timed_out": False,
        }
    finally:
        if stdout_thread is not None and stdout_thread.is_alive():
            stdout_thread.join(timeout=0.2)
        if stderr_thread is not None and stderr_thread.is_alive():
            stderr_thread.join(timeout=0.2)
        # Ensure process is cleaned up
        if process is not None and process.poll() is None:
            try:
                _kill_process_tree(process)
            except Exception:
                pass  # Best effort cleanup
