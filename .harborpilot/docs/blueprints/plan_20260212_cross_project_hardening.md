# Full Blueprint: cross_project_hardening
Mode: S2 Standard | Approval: APPROVED-FALLBACK

## Contract Snapshot
Goal: Make agent-accel reliably effective across heterogeneous projects (monorepo/workspace layouts), not only repo-specific layouts.
Acceptance Criteria:
- index auto scope can include non-src workspace files without manual hardcoding
- verify command routing supports workspace-aware execution for node/python commands
- verify preflight/degrade output is clearer for missing manifests/modules
- no breaking change to MCP tool names or required parameters
- unit gates pass on touched modules

## Scope
- accel/config.py
- accel/indexers/__init__.py
- accel/verify/sharding.py
- accel/verify/orchestrator.py
- tests/unit/test_config.py
- tests/unit/test_index_and_context.py
- tests/unit/test_verify_sharding.py
- tests/unit/test_verify_orchestrator.py
- README.md
- docs/setup_windows.md

## Failure Modes
- workspace routing selects wrong subdir in ambiguous monorepos
- preflight warnings become noisy in edge shell wrappers
- auto index scope too broad in non-git extremely large repos

## Pre-Snapshot
- dirty worktree; rollback via git revert/checkout of touched files

## Test Plan
- python -m ruff check .
- python -m mypy .
- python -m pytest -q tests/unit
- targeted behavior smoke for routing + index scope

## Post-Gate
- full static gates required; full pytest known failure may remain in external test scripts outside unit suite

## Rollback
- git checkout -- accel/config.py accel/indexers/__init__.py accel/verify/sharding.py accel/verify/orchestrator.py tests/unit/test_config.py tests/unit/test_index_and_context.py tests/unit/test_verify_sharding.py tests/unit/test_verify_orchestrator.py README.md docs/setup_windows.md
