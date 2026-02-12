# Full Plan: cross_agent_stability_fix
Mode: S2 Standard | Approval: Explicit (Fallback)

## Contract Snapshot
Goal: 彻底降低跨 Codex/Agent 集成回归：避免 60s 超时、提升参数兼容、杜绝 verify 零命令假阳性、修复 context 终态观测不一致。
Acceptance Criteria:
- sync index/update/context 在默认调用下不再因 >60s 阻塞导致 MCP tool_timeout。
- semantic_cache_mode 接受 read_write 风格别名。
- verify commands=0 不再返回 success，必须显式降级/告警。
- context_events summary 在出现 context_completed/context_failed 后输出一致终态。

## Scope
- accel/mcp_server.py
- accel/verify/orchestrator.py
- tests/unit/test_mcp_server.py
- tests/unit/test_verify_orchestrator.py
- README.md

## Budget (Scope Budget)
- Touched Files: <= 15
- Changed LOC: <= 900
- Dependencies: no new deps

## Pre-Snapshot
- rollback anchor: git HEAD + events rollback_point

## Test Plan
- pytest -q tests/unit/test_mcp_server.py
- pytest -q tests/unit/test_verify_orchestrator.py
- pytest -q tests
- ruff check .
- mypy .

## Rollback
- git restore accel/mcp_server.py accel/verify/orchestrator.py tests/unit/test_mcp_server.py tests/unit/test_verify_orchestrator.py README.md
