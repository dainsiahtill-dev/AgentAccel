from __future__ import annotations

import ast
import re
from pathlib import Path
from typing import Any


def _py_symbols_from_ast(tree: ast.AST, rel_path: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    def visit(node: ast.AST, scope: list[str]) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                q_name = ".".join(scope + [child.name])
                rows.append(
                    {
                        "symbol": child.name,
                        "kind": "function",
                        "lang": "python",
                        "file": rel_path,
                        "line_start": int(child.lineno),
                        "line_end": int(getattr(child, "end_lineno", child.lineno)),
                        "qualified_name": q_name,
                    }
                )
                visit(child, scope + [child.name])
            elif isinstance(child, ast.ClassDef):
                q_name = ".".join(scope + [child.name])
                rows.append(
                    {
                        "symbol": child.name,
                        "kind": "class",
                        "lang": "python",
                        "file": rel_path,
                        "line_start": int(child.lineno),
                        "line_end": int(getattr(child, "end_lineno", child.lineno)),
                        "qualified_name": q_name,
                    }
                )
                visit(child, scope + [child.name])
            else:
                visit(child, scope)

    visit(tree, [])
    return rows


TS_SYMBOL_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("function", re.compile(r"^\s*export\s+function\s+([A-Za-z_][A-Za-z0-9_]*)")),
    ("function", re.compile(r"^\s*function\s+([A-Za-z_][A-Za-z0-9_]*)")),
    ("class", re.compile(r"^\s*export\s+class\s+([A-Za-z_][A-Za-z0-9_]*)")),
    ("class", re.compile(r"^\s*class\s+([A-Za-z_][A-Za-z0-9_]*)")),
    ("type", re.compile(r"^\s*export\s+interface\s+([A-Za-z_][A-Za-z0-9_]*)")),
    ("type", re.compile(r"^\s*export\s+type\s+([A-Za-z_][A-Za-z0-9_]*)")),
    ("variable", re.compile(r"^\s*export\s+const\s+([A-Za-z_][A-Za-z0-9_]*)")),
]


def _ts_symbols_from_text(text: str, rel_path: str, lang: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_no, line in enumerate(text.splitlines(), start=1):
        for kind, pattern in TS_SYMBOL_PATTERNS:
            match = pattern.search(line)
            if match:
                symbol = match.group(1)
                rows.append(
                    {
                        "symbol": symbol,
                        "kind": kind,
                        "lang": lang,
                        "file": rel_path,
                        "line_start": line_no,
                        "line_end": line_no,
                        "qualified_name": f"{rel_path}:{symbol}",
                    }
                )
                break
    return rows


def extract_symbols(file_path: Path, rel_path: str, lang: str) -> list[dict[str, Any]]:
    # Check file size to prevent blocking
    try:
        file_size = file_path.stat().st_size
        if file_size > MAX_FILE_SIZE_BYTES:
            return [{
                "type": "error",
                "name": "file_too_large",
                "line_start": 1,
                "line_end": 1,
                "char_start": 0,
                "char_end": 0,
                "error": f"File too large: {file_size:,} bytes"
            }]
    except OSError:
        return [{
            "type": "error", 
            "name": "stat_error",
            "line_start": 1,
            "line_end": 1,
            "char_start": 0,
            "char_end": 0,
            "error": "Cannot read file stats"
        }]
    
    try:
        text = file_path.read_text(encoding="utf-8", errors="replace")
    except (OSError, UnicodeDecodeError) as exc:
        return [{
            "type": "error",
            "name": "read_error",
            "line_start": 1,
            "line_end": 1,
            "char_start": 0,
            "char_end": 0,
            "error": str(exc)
        }]
    
    if lang == "python":
        try:
            tree = ast.parse(text, filename=rel_path)
        except SyntaxError:
            return []
        return _py_symbols_from_ast(tree, rel_path)
    if lang in {"typescript", "javascript"}:
        return _ts_symbols_from_text(text, rel_path, lang)
    return []
