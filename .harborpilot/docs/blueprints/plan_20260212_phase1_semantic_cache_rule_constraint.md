# Full Blueprint: phase1_semantic_cache_rule_constraint
Mode: S2 Standard | Approval: APPROVED-FALLBACK

## Contract Snapshot
Goal: Implement phase-1 semantic cache + rule compression + constrained output for accel_context and verify planning without new dependencies.
Acceptance Criteria:
- semantic cache exact+hybrid for accel_context, with TTL and config/changed-files aware keys
- verify command plan cache for changed_files scenarios
- rule compression pipeline with auditable per-rule stats
- constrained output layer for context payload and verify events summary (`warn|strict|off`)
- tests cover semantic cache, rule compression, contracts, and mcp integration

## Scope
- accel/storage/semantic_cache.py (new)
- accel/query/rule_compressor.py (new)
- accel/schema/contracts.py (new)
- accel/mcp_server.py
- accel/query/context_compiler.py
- accel/verify/sharding.py
- accel/config.py
- README.md
- docs/setup_windows.md
- tests/unit/test_semantic_cache.py (new)
- tests/unit/test_rule_compressor.py (new)
- tests/unit/test_output_constraints.py (new)
- tests/unit/test_mcp_server.py
- tests/unit/test_verify_sharding.py

## Failure Modes
- hybrid semantic cache false positives
- over-compression removes useful context
- strict contract mode rejects payload unexpectedly

## Pre-Snapshot
- git working tree not clean; rely on commit history + file-level minimal edits

## Test Plan
- targeted pytest for new/affected unit tests
- mcp tool-level regression tests

## Post-Gate
- full unit subset relevant to changed modules
- report any pre-existing baseline failures separately

## Rollback
- git checkout -- <modified files> or revert commit
