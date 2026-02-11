# Mini Plan: mcp-sync-nonblocking-verify
Mode: S1 Patch | Approval: Explicit User Request

## Contract Snapshot
Goal: 彻底解决 `accel_verify` 在 MCP 下的超时卡住问题，并让调用方可以拿到实时进度（不再 60 秒干等）。
Acceptance Criteria:
- `accel_verify` 调用应快速返回可追踪句柄（`job_id`），不再被 60 秒外层 timeout 截断。
- `accel_verify_status/events` 能持续给出阶段与进度变化，支持前端显示“进度条式”反馈。
- 明确解释 CPU≈0 场景的阻塞成因，并提供可复现证据。

## Scope
- `accel/mcp_server.py`
- `tests/unit/test_mcp_server.py`
- `README.md`

## Budget (Scope Budget)
- Touched Files: <= 5
- Changed LOC: <= 420
- Dependencies: no new deps

## Approach
- 将 `accel_verify` 默认行为固定为异步启动并立即返回 `started + job_id`。
- 通过 `status/events` 提供轮询进度，并增加 heartbeat/poll 提示字段。
- 加强布尔参数容错解析，避免 `None/undefined` 误触发同步等待。
- 补充回归测试覆盖：默认异步、显式同步等待、超时回退。
- 在 status 资源中暴露 `pid/module_path` 便于排查“旧进程未重载”。

## Red
- cmd: `pytest -q tests/unit/test_mcp_server.py::test_sync_verify_returns_running_when_wait_window_exceeded tests/unit/test_mcp_server.py::test_sync_verify_returns_completed_result_for_fast_job`
- expected: 旧断言失败（默认同步），表明当前行为与目标不一致。

## Implementation
- Edit: precise-string (context verified)

## Rollback
- Using Git: `git checkout -- accel/mcp_server.py tests/unit/test_mcp_server.py README.md`
