# Mini Plan: fast-loop-cache-key-fix
Mode: S1 Patch | Approval: Explicit

## Contract Snapshot
Goal: 立即修复 accel_verify fast_loop 路径报错 `cannot access local variable '_cache_key'`。
Acceptance Criteria:
- `accel_verify` / `accel_verify_start` 在 fast_loop 场景不再触发 `_cache_key` 局部变量错误。
- 增加回归测试，覆盖 `run_verify_with_callback` 的非 fail-fast 路径。

## Scope
- accel/verify/orchestrator.py
- tests/unit/test_verify_orchestrator.py

## Budget (Scope Budget)
- Touched Files: <= 3
- Changed LOC: <= 120
- Dependencies: no new deps

## Approach
- 将异常分支中的 tuple 解构变量 `_cache_key` 重命名，避免遮蔽函数 `_cache_key(...)`。
- 新增单测，确保 run_verify_with_callback 非 fail-fast 正常执行。

## Red
- cmd: `PYTHONPATH=x:/MCPs/agent-accel pytest -q tests/unit/test_verify_orchestrator.py`
- expected: 新增测试通过，原测试不回归。

## Implementation
- Edit: precise-string (context verified)

## Rollback
- Using Git
