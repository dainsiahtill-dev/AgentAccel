# Full Plan: schema-benchmark-no-security
Mode: S2 Standard | Approval: APPROVED-FALLBACK (channel token limitation)

## Contract Snapshot
Goal: 除去安全边界全部推进（即推进 Schema 与 Benchmark harness，跳过安全边界改造）。
Acceptance Criteria:
- 补齐正式 Schema：`context pack + MCP outputs`（重点 `accel_verify_events` 事件结构）并接入契约校验。
- 新增可复现 benchmark harness（任务集输入、指标产出、结果文件输出），覆盖 token/时延/质量相关指标。
- 补齐版本治理说明（schema versioning & breaking change policy）。
- 本轮不实现安全边界功能。
- Full gate 通过：`python -m pytest -q`、`ruff check .`、`mypy .`。

## Scope
- Schema / Contract:
  - `accel/schema/contracts.py`
  - `accel/schema/context_pack.schema.json`
  - `accel/schema/index_manifest.schema.json`（必要时）
  - `accel/schema/mcp_context_response.schema.json`（新增）
  - `accel/schema/mcp_verify_events.schema.json`（新增）
  - `accel/schema/__init__.py`
- MCP integration:
  - `accel/mcp_server.py`
- Benchmark:
  - `scripts/run_benchmarks.py`（新增）
  - `examples/benchmarks/tasks.sample.json`（新增）
  - `examples/benchmarks/README.md`（新增）
- Docs:
  - `docs/schema_versioning.md`（新增）
  - `README.md`
- Tests:
  - `tests/unit/test_output_constraints.py`

## Failure Modes
- Schema 与真实输出字段不一致，导致 strict 模式误报。
- verify_events 嵌套事件结构约束过严，影响兼容性。
- benchmark harness 指标定义不稳定，导致结果不可复现。
- 文档示例与脚本参数漂移。

## Pre-Snapshot Plan
- 当前工作区非干净，使用快照路径 B。
- 创建 `.harborpilot/snapshots/snap_<ts>_<rand>/`，包含：
  - `index.json`
  - `rollback.sh`
  - `patch.diff`（当前工作区 patch）
  - `backup/`（本轮目标文件快照）

## Test Plan (Red -> Green)
- `python -m pytest -q tests/unit/test_output_constraints.py tests/unit/test_mcp_server.py`
- `python -m pytest -q`
- `ruff check .`
- `mypy .`

## Post-Gate
- Gate Set: full
- 若出现环境性失败，按 6.2 输出降级模板并标记 `Verified-Pending`。

## Rollback
- 优先执行快照目录 `rollback.sh`
- 或按快照 `backup/` 恢复目标文件并删除本轮新增文件。
