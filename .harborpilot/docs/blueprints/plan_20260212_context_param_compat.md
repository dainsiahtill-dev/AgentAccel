# Mini Plan: context-param-compat
Mode: S1 Patch | Approval: Explicit User Request

## Contract Snapshot
Goal: 提升 `accel_context` 参数兼容性，减少因调用方传参风格差异导致的校验失败与重复调用。
Acceptance Criteria:
- `budget` 支持字符串预设（至少包含 `small`），不再因字符串直接报 schema 错误。
- `changed_files`/`hints` 支持字符串输入（逗号分隔或 JSON 数组字符串）。
- 兼容增强不影响原有对象/数组调用方式。

## Scope
- `accel/mcp_server.py`
- `tests/unit/test_mcp_server.py`
- `README.md`

## Budget (Scope Budget)
- Touched Files: <= 5
- Changed LOC: <= 220
- Dependencies: no new deps

## Approach
- 扩展 MCP tool 入参类型（`budget`、`changed_files`、`hints`、`include_pack`）。
- 增加统一解析函数：字符串列表解析、budget 预设映射与容错。
- 增加单元测试覆盖字符串入参兼容路径。
- README 增补参数兼容说明与预设示例。

## Red
- cmd: `pytest -q tests/unit/test_mcp_server.py::test_tool_context_budget_and_list_string_compat`
- expected: 修改前失败（不支持字符串 budget）；修改后通过。

## Implementation
- Edit: precise-string (context verified)

## Rollback
- Using Git: `git checkout -- accel/mcp_server.py tests/unit/test_mcp_server.py README.md`
