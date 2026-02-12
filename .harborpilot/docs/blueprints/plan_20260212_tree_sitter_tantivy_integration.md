# Full Plan: tree-sitter-tantivy-integration
Mode: S2 Standard | Approval: Explicit (fallback token)

## Contract Snapshot
Goal: Start implementing Tree-sitter and Tantivy capabilities for this project with graceful degradation.
Acceptance Criteria:
- Tree-sitter path is integrated as optional parser and falls back safely when unavailable.
- Tantivy-based lexical reranking path is integrated as optional provider and default behavior remains unchanged.
- Config/env toggles are available for both features.
- Project quality gates pass with real-time evidence (pytest, ruff, mypy).

## Scope
- accel/indexers/symbols.py
- accel/query/context_compiler.py
- accel/config.py
- accel/config_runtime.py
- accel/query/lexical_ranker.py (new)
- tests/unit/test_symbol_extraction.py (new)
- tests/unit/test_lexical_ranker.py (new)
- tests/unit/test_config.py (extend)
- pyproject.toml

## Failure Modes
- Optional dependency missing (`tree_sitter_languages` or `tantivy`) causes runtime errors.
- Ranking output instability when lexical reranker is enabled.
- Regression risk in default path (features disabled).

## Pre-Snapshot
- Strategy A (clean git workspace):
  - record `git rev-parse HEAD`
  - record `git status --porcelain` empty output
  - append rollback_point event with commit ref

## Test Plan (Red/Green)
- `python -m pytest -q`
- `python -m ruff check .`
- `python -m mypy .`
- Focused tests for new modules:
  - tree-sitter optional fallback
  - tantivy optional fallback and score blending

## Post-Gate
- Gate Set: full
- No silent downgrade; if any command unavailable mark Verified-Pending and include degrade template.

## Rollback
- `git restore accel/indexers/symbols.py accel/query/context_compiler.py accel/config.py accel/config_runtime.py accel/query/lexical_ranker.py tests/unit/test_symbol_extraction.py tests/unit/test_lexical_ranker.py tests/unit/test_config.py pyproject.toml`
