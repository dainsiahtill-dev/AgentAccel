# Mini Plan: context-changed-files-soft-default
Mode: S1 Patch | Approval: Explicit

## Contract Snapshot
Goal: 修复 accel_context 在未传 changed_files 时默认报错，恢复可用性并保留可配置强制模式。
Acceptance Criteria:
- 未传 changed_files 且 git 无差异时，默认不报错，可返回 context。
- 仍支持通过 runtime.context_require_changed_files=true 显式开启强制模式。
- 文档与测试同步更新。

## Scope
- accel/config.py
- accel/mcp_server.py
- tests/unit/test_mcp_server.py
- README.md

## Budget (Scope Budget)
- Touched Files: <= 4
- Changed LOC: <= 120
- Dependencies: no new deps

## Approach
- 将 context_require_changed_files 默认值改为 false。
- _tool_context 读取该开关时的默认回退也改为 false。
- 保留强制模式的 ValueError 分支不变。
- 增加默认行为单测并更新 README 说明。

## Red
- cmd: `pytest -q tests/unit/test_mcp_server.py`
- expected: 新增默认行为断言通过，既有强制模式测试继续通过。

## Implementation
- Edit: precise-string (context verified)

## Rollback
- Using Git
