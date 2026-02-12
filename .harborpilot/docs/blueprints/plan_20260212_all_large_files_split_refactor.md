# Full Plan: all-large-files-split-refactor
Mode: S2 Standard | Approval: APPROVED-FALLBACK (channel token limitation)

## Contract Snapshot
Goal: 强制使用agent-accel MCP作为辅助工具去重构这个项目的所有代码，把大文件都要进行拆分重构。
Acceptance Criteria:
- 运行时代码中的大文件（本轮定义为 >= 500 行）全部完成拆分重构：`accel/mcp_server.py`、`accel/verify/orchestrator.py`、`accel/indexers/__init__.py`、`accel/config.py`、`accel/verify/sharding.py`。
- 拆分后保持现有 API/行为兼容。
- Full gate 通过：`pytest -q`、`ruff check .`、`mypy .`。
- 不新增依赖。

## Scope
- Runtime: `accel/mcp_server.py`, `accel/verify/orchestrator.py`, `accel/indexers/__init__.py`, `accel/config.py`, `accel/verify/sharding.py`
- New modules: `accel/*` 下新增 helper 子模块承接拆分逻辑
- Tests: 仅修正因模块重组造成的必要兼容问题
- Evidence: `.harborpilot/logs/*`, `.harborpilot/runtime/events.jsonl`

## Failure Modes
- 拆分后私有符号绑定变化导致 monkeypatch 失效。
- helper 模块引用上下文不完整导致运行时 NameError。
- 拆分后导入顺序变化触发循环依赖。
- 门禁通过但行为细节（超时/缓存/路由）出现回归。

## Pre-Snapshot Plan
- 工作区当前干净，采用 Git Anchor（Path A）。
- 记录锚点：
  - `git rev-parse HEAD`
  - `git status --porcelain`（空）
- 回滚：`git reset --hard <anchor_sha>`（仅用于本次独立变更）。

## Test Plan (Red -> Green)
- Red（基线）:
  - `pytest -q tests/unit/test_mcp_server.py tests/unit/test_verify_orchestrator.py tests/unit/test_index_and_context.py tests/unit/test_verify_sharding.py tests/unit/test_config.py`
- Green + Full Gate:
  - `pytest -q`
  - `ruff check .`
  - `mypy .`

## Post-Gate
- Gate Set: full
- 若出现环境性失败（非改动相关）则按 6.2 输出降级三行模板并标记 `Verified-Pending`。

## Rollback
- `git reset --hard 7336bd3cecbe9c35d21d923c8d18490ebed25063`
- 或 `git restore <touched_files>` 精准回滚。
