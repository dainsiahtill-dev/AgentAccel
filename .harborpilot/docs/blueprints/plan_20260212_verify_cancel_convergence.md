# Mini Plan: verify-cancel-convergence
Mode: S1 Patch | Approval: Explicit User Request

## Contract Snapshot
Goal: 修复 `accel_verify_cancel` 状态机长期停留在 `cancelling` 的问题，确保最终可收敛到 `cancelled`。
Acceptance Criteria:
- `accel_verify_cancel` 返回后，`accel_verify_status` 能在短时间内稳定进入 `cancelled`。
- 取消后不会被后台线程覆盖回 `completed/failed`。
- `index/context/verify/start/status/events/cancel` 全链路至少实测一轮并给出结论。

## Scope
- `accel/mcp_server.py`
- `tests/unit/test_mcp_server.py`
- `README.md`

## Budget (Scope Budget)
- Touched Files: <= 5
- Changed LOC: <= 260
- Dependencies: no new deps

## Approach
- `cancel` 请求时立即完成状态终结（`cancelled`），并记录 `cancel_requested/finalized` 事件。
- worker 收尾时检测取消态，避免覆盖终态。
- 新增单测验证取消收敛。
- 用 FastMCP stdio 链路复测全部工具。

## Red
- cmd: `pytest -q tests/unit/test_mcp_server.py::test_verify_cancel_converges_to_cancelled`
- expected: 修改前无法稳定断言 `cancelled`，修改后通过。

## Implementation
- Edit: precise-string (context verified)

## Rollback
- Using Git: `git checkout -- accel/mcp_server.py tests/unit/test_mcp_server.py README.md`
