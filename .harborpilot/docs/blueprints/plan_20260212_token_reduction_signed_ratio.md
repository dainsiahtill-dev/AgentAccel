# Mini Plan: token-reduction-signed-ratio
Mode: S1 Patch | Approval: Explicit (user instruction)

## Contract Snapshot
Goal: 仅修复 agent-accel 代码缺陷；若属于测试端覆盖或调用口径问题则不改代码。
Acceptance Criteria:
- `accel_context` 在 `context_tokens > baseline_tokens` 时，不再把对比比率压成 `0.0`。
- 退化场景可被指标直接观察（例如 `vs_snippets_only` 出现负比率）。
- 不修改测试端流程类问题（如门禁降级、回归覆盖策略）。

## Scope
- `accel/mcp_context_utils.py`
- `tests/unit/test_mcp_server.py`
- `README.md`

## Budget (Scope Budget)
- Touched Files: <= 3
- Changed LOC: <= 80
- Dependencies: no new deps

## Approach
- 将 token 对比比率函数调整为允许负值（保留上界 1.0）。
- 增加单元测试覆盖“baseline 更小于 context”场景。
- 更新文档，明确该比率可为负值，表示相对退化。

## Red
- cmd: `pytest -q tests/unit/test_mcp_server.py -k token_reduction`
- expected: 新增用例在旧逻辑下失败，修复后通过。

## Implementation
- Edit: precise-string (context verified)

## Rollback
- `git checkout -- accel/mcp_context_utils.py tests/unit/test_mcp_server.py README.md`
