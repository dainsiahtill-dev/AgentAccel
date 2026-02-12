# Full Plan: all-recommendations-fullfix
Mode: S2 Standard | Approval: APPROVED-FALLBACK (channel token limitation)

## Contract Snapshot
Goal: 建议全部解决，后续继续检测。
Acceptance Criteria:
- `accel_context` 支持 `strict_changed_files` 与 `fallback_confidence`，并在 changed_files 自动推断时给出可解释置信度。
- `accel_context` 输出三口径 token 收益：`vs_full_index`、`vs_changed_files`、`vs_snippets_only`。
- `accel_context` 提供轻量输出开关（`snippets_only` / `include_metadata`），减少响应 token。
- `accel_verify` / `accel_verify_start` 支持可选失败结果缓存（短 TTL），避免重复失败命令成本。
- verify jsonl 结构化增强：包含 `mode`、`fail_fast`、`cache_hits`、`cache_misses`、`fail_fast_skipped`、`command_index`。
- 相关单元测试覆盖新增行为并通过。

## Scope
- `accel/mcp_server.py`
- `accel/verify/orchestrator.py`
- `accel/config.py`
- `tests/unit/test_mcp_server.py`
- `tests/unit/test_verify_orchestrator.py`
- `README.md`
- `.harborpilot/logs/*` 与 `.harborpilot/runtime/events.jsonl`（审计证据）

## Failure Modes
- changed_files 仍为空且 strict 开启时误放行，导致上下文膨胀。
- fallback_confidence 误判，导致用户对缩面结果过度信任。
- snippets_only/include_metadata 互相叠加时响应字段不稳定。
- 缓存失败结果后，历史失败污染后续成功路径。
- verify jsonl 字段扩展破坏旧消费者解析。

## Pre-Snapshot Plan
- 当前工作区非干净，使用快照路径 B。
- 创建 `.harborpilot/snapshots/snap_<ts>_<rand>/`，写入：
  - `index.json`
  - `patch.diff`
  - `rollback.sh`

## Test Plan (Red -> Green)
- `pytest -q tests/unit/test_mcp_server.py -k "context or verify_events"`
- `pytest -q tests/unit/test_verify_orchestrator.py`
- `ruff check accel/mcp_server.py accel/verify/orchestrator.py accel/config.py tests/unit/test_mcp_server.py tests/unit/test_verify_orchestrator.py`

## Post-Gate
- Gate Set: full (针对本次 touched scope 的 lint + unit tests)
- 若发现仓库基线无关失败：降级为 reduced 并输出 6.2 三行模板。

## Rollback
- 优先执行快照目录中的 `rollback.sh`
- 或使用 `git apply -R <patch.diff>` 回退本次变更
