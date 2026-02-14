# Agent-Accel Development Guide

## Build / Lint / Test Commands

```bash
# Run all tests
python -m pytest

# Run specific test file
python -m pytest tests/test_smoke_cli.py

# Run single test
python -m pytest tests/test_smoke_cli.py::test_cli_doctor_smoke -v

# Run tests in parallel (recommended for full suite)
python -m pytest -x

# Lint check (ruff)
python -m ruff check accel/

# Auto-fix lint issues
python -m ruff check --fix accel/

# Type check (mypy)
python -m mypy accel/

# Full verification pipeline
python -m ruff check accel/ && python -m mypy accel/ && python -m pytest
```

## Code Style Guidelines

### General

- **Python**: 3.11+ required
- **Line length**: Follow ruff defaults (88 chars)
- **Future annotations**: Always include `from __future__ import annotations` at top

### Imports

Order: standard library → third party → local (absolute with leading dot)

```python
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from fastmcp import FastMCP

from .config import resolve_effective_config
from .storage.cache import ensure_project_dirs
```

### Naming Conventions

- **Functions/variables**: `snake_case` (e.g., `build_index`, `max_chars`)
- **Constants**: `UPPER_SNAKE_CASE` (e.g., `DEFAULT_PROJECT_CONFIG`)
- **Classes**: `PascalCase` (e.g., `FastMCP`, `ContextCompiler`)
- **Private**: Leading underscore (e.g., `_normalize_path`, `_utc_now`)
- **Type variables**: Use `Any`, avoid cryptic single letters

### Type Annotations

- Use full type hints on all function signatures
- Return type always specified (e.g., `-> dict[str, int]`)
- Use `from __future__ import annotations` for forward references
- Optional types: `dict[str, Any] | None` (not `Optional[...]`)

### Error Handling

- Catch specific exceptions, never bare `except:`
- Prefer `try/except/else/finally` blocks
- Log errors before re-raising when context is helpful
- Use pathlib for filesystem operations (handles errors better)

```python
# Good
try:
    data = json.loads(content)
except json.JSONDecodeError as exc:
    logger.warning(f"Failed to parse JSON: {exc}")
    raise ValueError(f"Invalid JSON in {path}") from exc

# Bad
try:
    data = json.loads(content)
except:  # Never do this
    data = {}
```

### File Operations

Always use `pathlib.Path` instead of `os.path`:

```python
from pathlib import Path

# Good
config_path = Path("accel.yaml")
content = config_path.read_text(encoding="utf-8")

# Bad
with open("accel.yaml", "r") as f:  # Avoid
    content = f.read()
```

### Path Handling

- Use `Path.cwd()` for current directory
- Use `Path(__file__).resolve()` for script-relative paths
- Normalize paths with `os.path.abspath()` when needed
- Cache directory locations via `project_paths()`

### Configuration

- Store defaults in `DEFAULT_*` constants at module level
- Override via `accel.yaml` and `accel.local.yaml`
- Validate in `_validate_effective_config()`
- Access via `resolve_effective_config()` helper

### Testing

- Place tests in `tests/` directory
- Name test functions with `test_` prefix
- Use descriptive names: `test_cli_doctor_smoke` not `test_1`
- Use `_repo_root()` helper for path resolution
- Set UTF-8 encoding in test subprocesses
- Clean up via pytest fixtures, not manual deletion

### CLI Commands

All commands available via `python -m accel.cli`:

```bash
python -m accel.cli doctor --project .       # Health check
python -m accel.cli index --project .        # Build/update index
python -m accel.cli context --project .      # Generate context pack
python -m accel.cli verify --project .       # Run verification
python -m accel.cli explain --project .      # Show selection scores
```

### Project Structure

```
accel/
├── __init__.py              # Version info
├── cli.py                   # Command-line interface
├── config.py                # Configuration loading
├── mcp_server.py            # MCP server implementation
├── indexers/                # Code indexing modules
├── query/                   # Context query/planning
├── verify/                  # Verification orchestration
├── storage/                 # Caching and persistence
└── schema/                  # JSON schemas
```

### Pre-Commit Checklist

Before submitting changes:

1. `python -m ruff check accel/` - passes
2. `python -m mypy accel/` - passes
3. `python -m pytest` - all tests pass
4. CLI smoke tests pass: `python -m accel.cli doctor --project .`
5. No hardcoded paths (use `Path` objects)
6. Proper exception handling (specific, not bare)
7. Type annotations complete on public APIs

## MCP Server

The MCP server provides tool endpoints for AI agents. Run with:

```bash
python -m accel.mcp_server
```

### Core Tools

| Tool | Description |
|------|-------------|
| `accel_index_build` | Build/rebuild project index |
| `accel_index_update` | Incremental index update |
| `accel_context` | Generate context pack for a task |
| `accel_plan_and_gate` | Two-stage planning with governance |
| `accel_verify` | Run verification commands |

### Symbol Query Tools (High Token Efficiency)

| Tool | Description | Token Savings |
|------|-------------|---------------|
| `accel_symbols_search` | Search symbols by name/type/language | ~93% vs `rg` |
| `accel_symbols_get_details` | Get symbol details + relations | ~90% vs file read |
| `accel_context_get_symbol_context` | Get precise symbol context with snippets | ~90% vs multi-file read |

### Relation Analysis Tools

| Tool | Description | Token Savings |
|------|-------------|---------------|
| `accel_relations_get_call_graph` | Build call graph from symbol/file | ~94% vs manual trace |
| `accel_relations_get_inheritance_tree` | Get class hierarchy | ~95% vs AST parse |
| `accel_relations_get_dependencies` | Get file import/export relations | ~90% vs `rg import` |

### Analysis & Monitoring Tools

| Tool | Description |
|------|-------------|
| `accel_analysis_detect_patterns` | Detect design patterns (singleton/factory/observer/decorator/builder) |
| `accel_monitor_get_project_stats` | Get project statistics (files/symbols/languages) |
| `accel_monitor_get_health` | Get system health status |

See `accel/mcp_server.py` for implementation details.

### Recommended Usage Patterns

**Code Understanding Task**:
```
1. accel_monitor_get_health()                              # Verify index available
2. accel_symbols_search(query="keyword", max_results=5)    # Locate symbols
3. accel_symbols_get_details(symbol_name="...")            # Get details + relations
4. accel_context_get_symbol_context(symbol_name="...", context_type="related")
```

**Architecture Analysis Task**:
```
1. accel_monitor_get_project_stats()                       # Project overview
2. accel_relations_get_inheritance_tree()                  # Class hierarchy
3. accel_relations_get_dependencies(direction="both")      # Module dependencies
4. accel_relations_get_call_graph(file_path="...", depth=2) # Call flow
5. accel_analysis_detect_patterns()                        # Design patterns
```

### Token Optimization Principles

- **Index First**: Always ensure index is available (`accel_monitor_get_health`)
- **Symbol > File**: Prefer `accel_symbols_*` over file-level queries
- **MCP > Local**: Never use `rg`/`grep` before calling accel tools
- **Budget Aware**: Respect `max_results`, `depth`, `max_files` limits

## License

MIT License - see repository root for full text.
