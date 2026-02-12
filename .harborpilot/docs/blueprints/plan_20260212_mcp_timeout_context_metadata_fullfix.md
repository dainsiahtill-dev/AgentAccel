# Full Plan: mcp_timeout_context_metadata_fullfix
Mode: S2 Standard | Approval: APPROVED-FALLBACK expected

## Contract Snapshot
Goal: 进一步完善 agent-accel MCP 当前实测暴露的问题，提升同步 verify 行为可控性、context 缩面稳定性、审计可观测性。
Acceptance Criteria:
- accel_verify 支持可配置同步等待超时动作（轮询继续或自动取消）并避免悬挂资源。
- accel_context 自动 changed_files 推断优先使用 git 变更并降低 fallback 偏差。
- accel_context 的 out 产物具备伴随元数据（token 估算/缩面比/来源）便于离线审计。
- 补齐对应单测并给出实时验证证据。
Risk:
- 若仅看可用性而不看状态收敛/元数据完整性，会高估稳定性与可审计性。

## Scope
- accel/mcp_server.py
- accel/config.py
- tests/unit/test_mcp_server.py
- README.md

## Scope Budget
- Touched Files: <= 8
- Changed LOC: <= 600
- Dependencies: no new deps

## Failure Modes
- 同步超时策略参数非法导致行为不确定。
- 超时自动取消后状态仍停留 cancelling。
- 新增 sidecar 元数据与主输出不一致。
- git 自动发现在无仓库/命令失败时回退不稳定。

## Pre-Snapshot
- Plan A (Git clean): 记录 HEAD + git status --porcelain 为空，并写入 rollback_point 事件。

## Implementation Plan
1. 为 accel_verify 增加 `sync_timeout_action`（poll|cancel）与 `sync_cancel_grace_seconds`，并在超时分支执行对应动作。
2. fast_loop 默认启用 `verify_cache_failed_results=true`（仅当用户未显式传值）。
3. 强化 `_discover_changed_files_from_git`：优先 `git status --porcelain`，包含 untracked/rename 解析，失败再回退 diff。
4. 在 `accel_context` 写出 `*.meta.json` sidecar，记录 token 估算、reduction、changed_files_source、fallback_confidence、budget 与 output mode。
5. 增补单测覆盖新增分支与产物。

## Test Plan (Red/Green)
- pytest -q tests/unit/test_mcp_server.py -k "sync_verify or context"
- pytest -q tests/unit/test_mcp_server.py

## Post-Gate
- Gate Set: full (按本次触达范围至少完整跑单测)
- 若出现环境限制则降级并按模板记录。

## Rollback
- Git rollback anchor: `git reset --hard <HEAD_SHA>` (仅用于本次变更回滚)
- 非破坏式回滚优先：`git checkout -- accel/mcp_server.py accel/config.py tests/unit/test_mcp_server.py README.md`
