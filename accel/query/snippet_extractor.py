from __future__ import annotations

from pathlib import Path
from typing import Any

# Maximum file size to read (10MB) to prevent blocking
MAX_FILE_SIZE_BYTES = 10 * 1024 * 1024


def extract_snippet(
    project_dir: Path,
    rel_path: str,
    task_tokens: list[str],
    symbol_rows: list[dict[str, Any]],
    snippet_radius: int,
    max_chars: int,
) -> dict[str, Any] | None:
    file_path = project_dir / rel_path
    if not file_path.exists():
        return None
    
    # Check file size to prevent blocking on large files
    try:
        file_size = file_path.stat().st_size
        if file_size > MAX_FILE_SIZE_BYTES:
            # Skip large files to prevent blocking
            return {
                "path": rel_path,
                "start_line": 1,
                "end_line": 1,
                "symbol": "",
                "reason": "file_too_large",
                "content": f"[File too large: {file_size:,} bytes, skipping to prevent blocking]",
            }
    except OSError:
        # If we can't stat the file, skip it
        return None
    
    try:
        text = file_path.read_text(encoding="utf-8", errors="replace")
    except (OSError, UnicodeDecodeError) as exc:
        # Handle file read errors gracefully
        return {
            "path": rel_path,
            "start_line": 1,
            "end_line": 1,
            "symbol": "",
            "reason": "read_error",
            "content": f"[Error reading file: {exc}]",
        }
    
    lines = text.splitlines()
    if not lines:
        return None

    focus_line = 1
    focus_symbol = ""

    if symbol_rows:
        symbol_rows_sorted = sorted(symbol_rows, key=lambda row: int(row.get("line_start", 1)))
        focus_line = int(symbol_rows_sorted[0].get("line_start", 1))
        focus_symbol = str(symbol_rows_sorted[0].get("symbol", ""))
    else:
        lowered_tokens = [token.lower() for token in task_tokens]
        for idx, line in enumerate(lines, start=1):
            low_line = line.lower()
            if any(token in low_line for token in lowered_tokens):
                focus_line = idx
                break

    start_line = max(1, focus_line - snippet_radius)
    end_line = min(len(lines), focus_line + snippet_radius)
    excerpt_lines = lines[start_line - 1 : end_line]
    content = "\n".join(excerpt_lines)
    if len(content) > max_chars:
        content = content[:max_chars]

    return {
        "path": rel_path,
        "start_line": start_line,
        "end_line": end_line,
        "symbol": focus_symbol,
        "reason": "symbol_or_token_focus",
        "content": content,
    }
