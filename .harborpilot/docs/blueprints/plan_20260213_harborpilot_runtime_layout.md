# Full Plan: harborpilot-runtime-layout
Mode: S2 Standard | Approval: Explicit-Fallback

## Contract Snapshot
Goal: 重构项目目录与路径治理，使运行时产物统一落在 `.harborpilot/` 体系，并完成 `mypy/pytest/ruff` 缓存迁移到 `.harborpilot/runtime/`。
Acceptance Criteria:
- 建立 HarborPilot 路径常量与统一解析入口，减少散落硬编码路径。
- agent-accel 运行时目录继续落在 `.harborpilot/runtime/agent-accel`，并使用统一路径解析。
- `.mypy_cache`、`.pytest_cache`、`.ruff_cache` 的配置落点迁移到 `.harborpilot/runtime/`。
- 索引/扫描排除规则与目录约定同步，避免扫描新缓存目录。
- 对 pytest 误收集风险做最小化目录治理（不影响核心运行逻辑）。

## Scope
- accel/harborpilot_paths.py (new)
- accel/config_runtime.py
- accel/mcp_server.py
- accel/config.py
- accel/indexers/discovery.py
- accel/indexers/__init__.py
- accel/verify/sharding_workspace.py
- scripts/collect_harborpilot_phase0_baseline.py
- pyproject.toml
- .gitignore
- root diagnostic test_*.py relocation

## Budget (Scope Budget)
- Touched Files: <= 30
- Changed LOC: <= 1200
- Dependencies: no new deps

## Failure Modes
- 路径常量替换不完整导致日志/状态写到旧路径。
- 缓存目录迁移后工具读取旧缓存失败（可接受，首次重建）。
- 诊断脚本移动导致文档引用失效。

## Pre-Snapshot
- Git working tree 当前非干净，使用 snapshot 路径（index.json + rollback.sh + patch.diff）。

## Test Plan
- `python -m ruff check .`
- `python -m mypy .`
- `python -m pytest -q tests`（若 tests 目录存在）
- `python -m pytest -q -k session_receipt`（增量验证）

## Post-Gate
- Gate Set: full（若环境限制则降级为 reduced 并输出 DEGRADE_REASON/RISK/BACKFILL_PLAN）

## Rollback
- 优先执行 snapshot 的 `rollback.sh` 回退到 pre-snapshot 状态。
