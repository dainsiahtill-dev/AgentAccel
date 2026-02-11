# Mini Plan: cancel-heartbeat-consistency
Mode: S1 Patch | Approval: Explicit

## Contract Snapshot
Goal: 修复取消后 events 仍出现 running heartbeat 的状态流/事件流不一致问题。
Acceptance Criteria:
- cancel 后 `verify_status.state=cancelled` 时，后续新增 events 不再出现 `heartbeat state=running`。
- cancel 后不再追加 `job_completed` 等与终态冲突的 live 事件。
- 覆盖回归测试并通过。

## Scope
- accel/verify/job_manager.py
- accel/mcp_server.py
- tests/unit/test_mcp_server.py

## Budget (Scope Budget)
- Touched Files: <= 4
- Changed LOC: <= 180
- Dependencies: no new deps

## Approach
- 在 VerifyJob 增加原子 live-event 闸门（终态拒绝追加）。
- _JobCallback 与心跳线程统一通过闸门写 live 事件。
- 新增取消后事件一致性回归测试。

## Red
- cmd: `PYTHONPATH=x:/MCPs/agent-accel pytest -q tests/unit/test_mcp_server.py`
- expected: 新增 cancel 一致性测试通过，既有测试不回归。

## Implementation
- Edit: precise-string (context verified)

## Rollback
- Using Git
