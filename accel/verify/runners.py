from __future__ import annotations

import os
import signal
import subprocess
import time
from pathlib import Path
from typing import Any

OUTPUT_TAIL_LIMIT = 12000


def _normalize_output(value: str | bytes | None) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


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
            if hasattr(os, 'killpg'):
                os.killpg(os.getpgid(pid), signal.SIGTERM)
                time.sleep(0.1)  # Give it a moment to terminate
                if process.poll() is None:
                    os.killpg(os.getpgid(pid), signal.SIGKILL)
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


def run_command(command: str, cwd: Path, timeout_seconds: int) -> dict[str, Any]:
    """Run a command with enhanced timeout handling and process cleanup."""
    started = time.perf_counter()
    process: subprocess.Popen[str] | None = None
    stdout_text: str | bytes | None = ""
    stderr_text: str | bytes | None = ""
    
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
            # Set process group for better termination on Unix
            preexec_fn=None if os.name == "nt" else os.setsid,
        )
        
        # Use communicate with timeout
        try:
            stdout_text, stderr_text = process.communicate(timeout=timeout_seconds)
        except subprocess.TimeoutExpired as exc:
            # Enhanced timeout handling
            if process is not None:
                _kill_process_tree(process)
                try:
                    # Try to get any remaining output with a shorter timeout
                    stdout_text, stderr_text = process.communicate(timeout=5)
                except (subprocess.TimeoutExpired, OSError):
                    # If that fails, use the exception output
                    stdout_text = exc.stdout
                    stderr_text = exc.stderr
            else:
                stdout_text = exc.stdout
                stderr_text = exc.stderr
            
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
            "stdout": _normalize_output(stdout_text)[-OUTPUT_TAIL_LIMIT:],
            "stderr": _normalize_output(stderr_text)[-OUTPUT_TAIL_LIMIT:],
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
        # Ensure process is cleaned up
        if process is not None and process.poll() is None:
            try:
                _kill_process_tree(process)
            except Exception:
                pass  # Best effort cleanup
