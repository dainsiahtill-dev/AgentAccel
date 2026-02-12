# Full Plan: priority-iter-3-timebox
Mode: S2 Standard | Approval: APPROVED-FALLBACK

## Contract Snapshot
Goal: Implement priority item #3: max verify wall-clock cap, stall-triggered auto-cancel, and automatic partial result return with unfinished items.
Acceptance Criteria:
- AC1: verification supports a configurable max wall-clock time limit and stops launching/continuing beyond limit.
- AC2: verification supports optional auto-cancel-on-stall based on stall timeout detection.
- AC3: result payload includes partial execution metadata and unfinished command list/reasons when cut short.
- AC4: MCP verify tools accept and pass through the new runtime override parameters.
- AC5: unit tests cover timeout/stall partial paths and MCP override plumbing.

## Scope
- accel/verify/orchestrator.py
- accel/mcp_server.py
- accel/config.py
- examples/accel.yaml
- tests/unit/test_verify_orchestrator.py
- tests/unit/test_mcp_server.py

## Failure Modes
- Long-running parallel verify futures exceed wall-time and return no explicit unfinished markers.
- Stall heartbeat is observed but execution never terminates, causing prolonged hangs.
- Sync and async MCP entry points diverge in supported override parameters.
- Backward compatibility regression in existing verify status/events summary paths.

## Budget (Scope Budget)
- Touched Files: <= 8
- Changed LOC: <= 900
- Dependencies: no new deps

## Pre-Snapshot
- Path A (Git clean anchor)
- `git rev-parse HEAD` captured
- `git status --porcelain` empty captured
- `git submodule status --recursive` captured (empty)

## Approach
1. Add runtime knobs for `verify_max_wall_time_seconds` and `verify_auto_cancel_on_stall`.
2. Implement orchestrator-level wall-time budgeting and automatic partial-result synthesis.
3. Add stall auto-cancel flow in callback path with explicit termination reason and unfinished command entries.
4. Expose new fields through MCP tool parameters and runtime overrides.
5. Add focused unit tests for wall-time partial, stall partial, and MCP override propagation.

## Test Plan
- `pytest -q tests/unit/test_verify_orchestrator.py`
- `pytest -q tests/unit/test_mcp_server.py -k "verify"`
- `ruff check .`
- `mypy .`

## Post-Gate
- Default `full`; if unrelated legacy failures block full suite, mark reduced with reason/risk/backfill in evidence log.

## Rollback
- `git restore accel/verify/orchestrator.py accel/mcp_server.py accel/config.py examples/accel.yaml tests/unit/test_verify_orchestrator.py tests/unit/test_mcp_server.py`
- Anchor: current HEAD before edits.
