# Mini Plan: mcp_stress_bugfix_verify_context_cancel
Mode: S1 Patch | Approval: Explicit

## Contract Snapshot
Goal: 修复在 X:\Git\Harborpilot 压测中复现的 agent-accel MCP 缺陷，仅处理 agent-accel 自身代码问题。
Acceptance Criteria:
- 修复 verify 目标测试路径在 workspace 路由下的错误映射，不再产生路径重复导致的假失败。
- 改善 accel_context 同步等待稳定性，避免被固定 45s 上限过早截断（仍保持 MCP 调用安全上限）。
- 修复 cancel 终态语义不一致，避免 cancelled 状态与 completed 语义混淆/竞态覆盖。
- 相关单元测试通过，并新增覆盖本次缺陷场景的回归测试。

## Scope
- accel/verify/sharding_workspace.py
- accel/mcp_server.py
- tests/unit/test_verify_sharding.py
- tests/unit/test_mcp_server.py

## Budget (Scope Budget)
- Touched Files: <= 8
- Changed LOC: <= 500
- Dependencies: no new deps

## Approach
- 在 workspace 路由时重写 pytest 目标路径为 workspace 相对路径。
- 为 context 同步等待引入独立 RPC 安全上限（默认高于 45s，仍低于客户端超时窗口），并保留 fallback_async 语义。
- 收紧取消竞态处理：终态取消后不再被 worker 完成态覆盖，同时优化 cancelled 状态进度语义。
- 增加对应单测，覆盖路径重写、context 上限、cancel 进度语义。

## Red
- cmd: python -m pytest -q tests/unit/test_verify_sharding.py tests/unit/test_mcp_server.py
- expected: 新增用例先失败（在修复前），修复后全通过。

## Implementation
- Edit: precise-string (context verified)

## Rollback
- Using Git (git restore --source=HEAD -- <files>)
