# Mini Plan: all-token-optimizations
Mode: S1 Patch | Approval: Explicit

## Contract Snapshot
Goal: 重新评估当前 agent-accel MCP 对实际 token 的优化效果，并给出完善建议；并执行“全部”改进项。
Acceptance Criteria:
- 给出可量化结论 + 可执行改进点。
- 落地全部改进：默认预算策略、changed_files 缩面约束、events 摘要+最近N条、context 收益指标、snippet 去重裁剪、自适应预算。
Risk:
- 当前 token 统计按 chars/4 为工程近似值。

## Scope
- accel/mcp_server.py
- accel/query/context_compiler.py
- README.md
- tests/unit/test_mcp_server.py
- tests/unit/test_index_and_context.py

## Budget (Scope Budget)
- Touched Files: <= 8
- Changed LOC: <= 450
- Dependencies: no new deps

## Approach
- 在 accel_context 返回中增加 selected_tests_count 与 selected_checks_count。
- 对 changed_files 执行严格策略：用户未提供且 git 无法发现时返回可操作错误，避免大范围上下文退化。
- 为 accel_verify_events 增加 max_events + include_summary，默认返回摘要并可裁剪最近 N 条。
- 保持 small/tiny/medium 自适应预算策略，补充文档与单元测试。

## Red
- cmd: `pytest -q tests/unit/test_mcp_server.py`
- expected: 相关新增测试先失败（或缺失），改动后通过。

## Implementation
- Edit: precise-string (context verified)

## Rollback
- Using Git (HEAD ref captured in verification log/events)
