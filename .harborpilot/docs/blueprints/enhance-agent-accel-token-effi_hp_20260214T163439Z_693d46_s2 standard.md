# Full Plan: hp_20260214T163439Z_693d46
Mode: S2 Standard | Approval: Explicit

## Contract Snapshot
Goal: Enhance agent-accel token efficiency and retrieval quality by fully leveraging Tree-sitter structural information in indexing, context selection, and scoring.
Acceptance Criteria:
- Tree-sitter symbol records include richer metadata (signature/scope/kind and language-specific structure where available).
- Context/snippet extraction prefers complete syntax units (function/class blocks) instead of shallow line windows.
- Ranking/scoring incorporates new structural signals derived from Tree-sitter output.
- Automated tests cover new metadata extraction and retrieval behavior.
- Project verification pipeline (ruff, mypy, pytest) runs and results are captured.

## Scope
- `accel/cli.py
accel/config.py
accel/config_runtime.py
accel/harborpilot_paths.py
accel/hooks/pre_hook.py
accel/indexers/__init__.py
accel/indexers/discovery.py
accel/indexers/symbols.py
accel/mcp_server.py
accel/query/context_compiler.py
accel/query/lexical_ranker.py
accel/query/snippet_extractor.py`

## Budget (Scope Budget)
- Touched Files: <= 8
- Changed LOC: <= 450
- Dependencies: no new deps

## Approach
- Implement deep Tree-sitter utilization with minimal focused edits in indexing and query path: enrich symbol records with structural metadata, add syntax-unit extraction fallback strategy, and add structural scoring blend plus tests.

## Rollback
- Snapshot + rollback.sh
