# Mini Plan: token-optimization-all
Mode: S1 Patch | Approval: Explicit User Request

## Contract Snapshot
Goal: 实现 token 优化建议的“全部”改进项，避免只看可用性，增强预算与缩面可视化和默认策略。
Acceptance Criteria:
- `accel_context` 返回包含 `estimated_tokens`、`compression_ratio` 等可量化指标。
- 默认预算策略从 fixed medium 调整为自适应（优先 tiny/small，复杂任务才升 medium）。
- `changed_files` 未传时自动发现并使用（优先 git 变更），减少上下文膨胀。
- snippets 增加去重与压缩，降低重复内容占用。
- 给出本轮实测结论与可执行改进闭环证据。

## Scope
- `accel/mcp_server.py`
- `accel/query/context_compiler.py`
- `accel/indexers/__init__.py`
- `tests/unit/test_mcp_server.py`
- `tests/unit/test_index_and_context.py`
- `README.md`

## Budget (Scope Budget)
- Touched Files: <= 8
- Changed LOC: <= 520
- Dependencies: no new deps

## Approach
- 在 MCP context 工具层新增预算自适应与 git changed_files 回填策略。
- 在 context pack 生成层加入 snippet 去重/压缩、统计字段。
- 在 index manifest 写入 source chars/bytes 基线，供压缩率计算。
- 在 context 响应返回 token 与压缩指标，并补充说明文档。
- 增加单测覆盖预算策略、兼容参数、取消收敛与缩面指标。

## Red
- cmd: `pytest -q tests/unit/test_mcp_server.py::test_tool_context_auto_budget_and_git_changed_files tests/unit/test_index_and_context.py::test_context_pack_snippet_dedupe_and_metrics`
- expected: 修改前缺失新字段/策略，测试失败；修改后通过。

## Implementation
- Edit: precise-string (context verified)

## Rollback
- Using Git: `git checkout -- accel/mcp_server.py accel/query/context_compiler.py accel/indexers/__init__.py tests/unit/test_mcp_server.py tests/unit/test_index_and_context.py README.md`
