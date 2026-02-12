# Full Plan: benchmark_independent_codex
Mode: S2 Standard | Approval: APPROVED-FALLBACK

## Contract Snapshot
Goal: ???????????????CLI?MCP??????benchmark???????????
Acceptance Criteria:
- [ ] $env:PYTHONPATH='.'; python -m pytest -q
- [ ] $env:PYTHONPATH='.'; pytest -q tests/integration
- [ ] $env:PYTHONPATH='.'; ruff check .
- [ ] $env:PYTHONPATH='.'; mypy .
- [ ] python scripts/run_benchmarks.py --project . --tasks examples/benchmarks/tasks.sample.json --index-mode update --out .harborpilot/logs/benchmark_independent_codex.json --out-md .harborpilot/logs/benchmark_independent_codex.md
- [ ] ???????/????????????????
- [ ] ???????????????/?/????????????????
- [ ] ????????? full gate ????????????

## Scope
- Runtime paths may be touched only if needed to make listed commands pass:
  - accel/
  - scripts/run_benchmarks.py
  - tests/
  - docs/ (if output/schema docs need alignment)
- Evidence/log paths:
  - .harborpilot/logs/verification_benchmark_independent_codex_20260213_01.log
  - .harborpilot/logs/benchmark_independent_codex.json
  - .harborpilot/logs/benchmark_independent_codex.md

## Scope Budget
- Touched Files: <= 50
- Changed LOC: <= 2000
- Dependencies: no new deps

## Failure Modes
- Pre-existing dirty workspace can mask regression boundaries.
- Integration tests may depend on environment-specific timing/process behavior.
- Benchmark can fail from schema/output mismatch even when gates pass.

## Pre-Snapshot
- Strategy: Git rollback anchor in current dirty workspace.
- Record: HEAD + git status --porcelain in evidence.
- Rollback path: `git restore <touched files>` or snapshot rollback script if created.

## Test Plan (Red -> Green)
1. Run commanded full gate + benchmark in exact order from contract.
2. Capture first failing command and failing test/case as Red evidence.
3. Apply minimal fix.
4. Re-run affected tests, then full commanded gate, then benchmark.

## Post-Gate
- Gate Set: full (no silent downgrade).
- Required: all 5 contract commands pass with exit code 0.

## Rollback
- Primary: restore touched files to pre-change state via git.
- Fallback: restore from snapshot if non-git rollback is required.
