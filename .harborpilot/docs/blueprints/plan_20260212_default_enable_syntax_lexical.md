# Mini Plan: default-enable-syntax-lexical
Mode: S1 Patch | Approval: Explicit

## Contract Snapshot
Goal: Enable new syntax parser and lexical ranker features by default so users can use them without local config edits.
Acceptance Criteria:
- Default runtime config enables syntax parser and lexical ranker with provider auto.
- Existing override env vars continue to work.
- Tests and quality gates pass.

## Scope
- accel/config.py
- accel/config_runtime.py
- tests/unit/test_config.py

## Budget (Scope Budget)
- Touched Files: <= 8
- Changed LOC: <= 200
- Dependencies: no new deps

## Approach
- Update default runtime keys to enabled + auto provider.
- Update normalization defaults to preserve enabled state.
- Add one config unit test for default behavior.

## Red
- cmd: `.\.venv\Scripts\python.exe -m pytest -q`
- expected: pass

## Implementation
- Edit: precise-string (context verified)

## Rollback
- Using Git: `git restore accel/config.py accel/config_runtime.py tests/unit/test_config.py`
