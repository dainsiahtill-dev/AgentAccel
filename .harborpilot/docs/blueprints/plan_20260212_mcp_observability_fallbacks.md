# Full Plan: mcp-observability-fallbacks
Mode: S2 Standard | Approval: Explicit user instruction (fallback format)

## Contract Snapshot
Goal: 彻底解决 agent-accel MCP 当前可用性与可观测性问题，重点覆盖超时/阻塞判断、context 缩面兜底、异步验证事件状态一致性与实时进度可见性。
Acceptance Criteria:
- `accel_context` 在未传 `changed_files` 时提供自动兜底，不因默认路径直接失败。
- `accel_verify_events.summary.latest_state` 在取消/完成后与 `accel_verify_status` 终态一致。
- `accel_verify` 异步链路在长命令阶段可返回可感知的进度字段，而非长期 `progress=0.0`。
- `index/context/verify/start/status/events/cancel` 至少完成一轮实测并给出结论。

## Scope
- `accel/mcp_server.py`
- `accel/verify/callbacks.py`
- `accel/verify/orchestrator.py`
- `tests/unit/test_mcp_server.py`
- `tests/unit/test_verify_orchestrator.py` (only if needed)
- `README.md` (only if behavior contract changes)

## Budget (Scope Budget)
- Touched Files: <= 8
- Changed LOC: <= 450
- Dependencies: no new deps

## Failure Modes
- 长命令期间 heartbeat 覆盖终态，导致 summary 状态回滚为 `running`。
- callback 签名升级后 MCP 回调未消费新字段，前端无法显示细粒度进度。
- strict 模式误配置导致 `changed_files` fallback 仍被硬拦截。

## Pre-Snapshot
- Git rollback anchor:
  - `git rev-parse HEAD`
  - `git status --porcelain`

## Test Plan (Red -> Green)
- `ruff check accel/mcp_server.py accel/verify/callbacks.py accel/verify/orchestrator.py tests/unit/test_mcp_server.py`
- `PYTHONPATH=. pytest -q tests/unit/test_mcp_server.py`
- `PYTHONPATH=. pytest -q tests/unit/test_verify_orchestrator.py` (if touched)
- MCP tool-level smoke:
  - `accel_context` (with and without `changed_files`)
  - `accel_verify_start/status/events/cancel`

## Post-Gate
- Gate Set: full
- If tool-level smoke blocked by environment, mark `Verified-Pending` and provide backfill commands.

## Rollback
- `git checkout -- accel/mcp_server.py accel/verify/callbacks.py accel/verify/orchestrator.py tests/unit/test_mcp_server.py tests/unit/test_verify_orchestrator.py README.md`
- or reset to recorded commit SHA.
