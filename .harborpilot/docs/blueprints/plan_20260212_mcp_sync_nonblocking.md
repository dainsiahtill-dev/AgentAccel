# Mini Plan: mcp-sync-nonblocking-verify
Mode: S1 Patch | Approval: Explicit User Request

## Contract Snapshot
Goal: 彻底解决 MCP 下 `accel_verify` 触发 60 秒超时后拖死后续调用的问题，让同步/异步验证都可稳定返回。
Acceptance Criteria:
- 同步 `accel_verify` 在大型仓库场景下不会无限占用 MCP 工具线程，并可在 60 秒窗口前返回可追踪结果。
- 异步链路 `accel_verify_start/status/events` 在同步调用后仍可立即使用。
- 现有小项目同步验证行为保持可用（可直接拿到完成结果）。

## Scope
- `accel/mcp_server.py`
- `tests/unit/test_mcp_server.py`

## Budget (Scope Budget)
- Touched Files: <= 4
- Changed LOC: <= 280
- Dependencies: no new deps

## Approach
- 将同步 `accel_verify` 改为复用异步 job 机制，并设置有限等待窗口（默认 < 60s）。
- 超出窗口时返回 `running + job_id`，避免阻塞 MCP 服务线程。
- 保留 `accel_verify_start/status/events` 语义并补充回归测试。
- 修复 debug logger 命名冲突，避免开启 debug 时潜在崩溃。

## Red
- cmd: `pytest -q tests/unit/test_mcp_server.py`
- expected: 新增超时场景测试在修改前失败，修改后通过。

## Implementation
- Edit: precise-string (context verified)

## Rollback
- Using Git: `git checkout -- accel/mcp_server.py tests/unit/test_mcp_server.py`
