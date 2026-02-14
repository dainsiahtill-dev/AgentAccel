from __future__ import annotations

import ast
import re
from importlib import import_module
from pathlib import Path
from typing import Any

MAX_FILE_SIZE_BYTES = 10 * 1024 * 1024
_SYNTAX_PROVIDERS = {"off", "auto", "tree_sitter"}


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


def normalize_syntax_provider(value: Any, default_value: str = "off") -> str:
    token = str(value or default_value).strip().lower()
    if token in _SYNTAX_PROVIDERS:
        return token
    fallback = str(default_value or "off").strip().lower()
    return fallback if fallback in _SYNTAX_PROVIDERS else "off"


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


def _extract_symbol_name(node: Any, source_bytes: bytes) -> str:
    name_node = None
    child_by_field_name = getattr(node, "child_by_field_name", None)
    if callable(child_by_field_name):
        try:
            name_node = child_by_field_name("name")
        except (AttributeError, TypeError):
            name_node = None

    if name_node is None:
        for child in list(getattr(node, "children", []) or []):
            child_type = str(getattr(child, "type", ""))
            if child_type in {"identifier", "property_identifier", "type_identifier"}:
                name_node = child
                break

    if name_node is None:
        return ""

    start = int(getattr(name_node, "start_byte", 0))
    end = int(getattr(name_node, "end_byte", 0))
    if end <= start:
        return ""
    return source_bytes[start:end].decode("utf-8", errors="replace").strip()


def _append_tree_sitter_row(
    *,
    rows: list[dict[str, Any]],
    rel_path: str,
    lang: str,
    node: Any,
    symbol: str,
    kind: str,
) -> None:
    if not symbol:
        return
    start_point = tuple(getattr(node, "start_point", (0, 0)))
    end_point = tuple(getattr(node, "end_point", start_point))
    line_start = int(start_point[0]) + 1 if len(start_point) > 0 else 1
    line_end = int(end_point[0]) + 1 if len(end_point) > 0 else line_start
    rows.append(
        {
            "symbol": symbol,
            "kind": kind,
            "lang": lang,
            "file": rel_path,
            "line_start": line_start,
            "line_end": max(line_start, line_end),
            "qualified_name": f"{rel_path}:{symbol}",
        }
    )


def _load_tree_sitter_parser(lang: str, file_path: Path) -> Any | None:
    suffix = file_path.suffix.lower()
    if lang == "typescript":
        aliases = ["typescript", "tsx"] if suffix == ".tsx" else ["typescript"]
    else:
        aliases = ["jsx", "javascript"] if suffix == ".jsx" else ["javascript"]

    for module_name in ("tree_sitter_languages", "tree_sitter_language_pack"):
        try:
            module = import_module(module_name)
        except ImportError:
            continue
        parser_getter = getattr(module, "get_parser", None)
        if not callable(parser_getter):
            continue
        for alias in aliases:
            try:
                parser = parser_getter(alias)
            except (ValueError, TypeError):
                continue
            if parser is not None:
                return parser
    return None


def _ts_symbols_from_tree_sitter(
    *,
    text: str,
    file_path: Path,
    rel_path: str,
    lang: str,
) -> list[dict[str, Any]]:
    parser = _load_tree_sitter_parser(lang, file_path)
    if parser is None:
        return []
    source_bytes = text.encode("utf-8", errors="replace")
    try:
        tree = parser.parse(source_bytes)
    except (ValueError, TypeError):
        return []

    rows: list[dict[str, Any]] = []
    stack = [getattr(tree, "root_node", None)]
    while stack:
        node = stack.pop()
        if node is None:
            continue
        children = list(getattr(node, "children", []) or [])
        stack.extend(reversed(children))

        node_type = str(getattr(node, "type", ""))
        if node_type in {
            "function_declaration",
            "method_definition",
            "generator_function_declaration",
        }:
            symbol = _extract_symbol_name(node, source_bytes)
            _append_tree_sitter_row(
                rows=rows,
                rel_path=rel_path,
                lang=lang,
                node=node,
                symbol=symbol,
                kind="function",
            )
            continue
        if node_type == "class_declaration":
            symbol = _extract_symbol_name(node, source_bytes)
            _append_tree_sitter_row(
                rows=rows,
                rel_path=rel_path,
                lang=lang,
                node=node,
                symbol=symbol,
                kind="class",
            )
            continue
        if node_type in {"interface_declaration", "type_alias_declaration"}:
            symbol = _extract_symbol_name(node, source_bytes)
            _append_tree_sitter_row(
                rows=rows,
                rel_path=rel_path,
                lang=lang,
                node=node,
                symbol=symbol,
                kind="type",
            )
            continue
        if node_type in {"lexical_declaration", "variable_declaration"}:
            for child in children:
                if str(getattr(child, "type", "")) != "variable_declarator":
                    continue
                symbol = _extract_symbol_name(child, source_bytes)
                _append_tree_sitter_row(
                    rows=rows,
                    rel_path=rel_path,
                    lang=lang,
                    node=child,
                    symbol=symbol,
                    kind="variable",
                )
    return rows


def extract_symbols(
    file_path: Path,
    rel_path: str,
    lang: str,
    runtime_cfg: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    # Check file size to prevent blocking
    try:
        file_size = file_path.stat().st_size
        if file_size > MAX_FILE_SIZE_BYTES:
            return [
                {
                    "type": "error",
                    "name": "file_too_large",
                    "line_start": 1,
                    "line_end": 1,
                    "char_start": 0,
                    "char_end": 0,
                    "error": f"File too large: {file_size:,} bytes",
                }
            ]
    except OSError:
        return [
            {
                "type": "error",
                "name": "stat_error",
                "line_start": 1,
                "line_end": 1,
                "char_start": 0,
                "char_end": 0,
                "error": "Cannot read file stats",
            }
        ]

    try:
        text = file_path.read_text(encoding="utf-8", errors="replace")
    except (OSError, UnicodeDecodeError) as exc:
        return [
            {
                "type": "error",
                "name": "read_error",
                "line_start": 1,
                "line_end": 1,
                "char_start": 0,
                "char_end": 0,
                "error": str(exc),
            }
        ]

    if lang == "python":
        try:
            tree = ast.parse(text, filename=rel_path)
        except SyntaxError:
            return []
        return _py_symbols_from_ast(tree, rel_path)
    if lang in {"typescript", "javascript"}:
        runtime = dict(runtime_cfg or {})
        parser_enabled = bool(runtime.get("syntax_parser_enabled", False))
        parser_provider = normalize_syntax_provider(
            runtime.get("syntax_parser_provider", "off"),
            default_value="off",
        )
        if parser_enabled and parser_provider in {"auto", "tree_sitter"}:
            parsed_rows = _ts_symbols_from_tree_sitter(
                text=text,
                file_path=file_path,
                rel_path=rel_path,
                lang=lang,
            )
            if parsed_rows:
                return parsed_rows
        return _ts_symbols_from_text(text, rel_path, lang)
    return []
