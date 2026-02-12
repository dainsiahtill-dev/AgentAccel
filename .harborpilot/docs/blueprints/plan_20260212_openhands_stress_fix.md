# Full Plan: openhands_stress_fix
Mode: S2 Standard | Approval: Explicit (Fallback)

## Contract Snapshot
Goal: 修复 OpenHands 压测暴露的非环境问题，提升同步接口可用性与结果可解释性。
Acceptance Criteria:
- sync_wait 与同步调用窗口解耦，支持可配置 >=180s，不再硬性夹到 55s。
- accel_context 同步接口超时时自动降级异步并返回 job_id，不再直接抛错。
- verify 返回明确失败分类：项目门禁失败 vs 执行器失败。
- Windows 下为 make install-pre-commit-hooks 提供兼容执行路径。

## Scope
- accel/mcp_server.py
- accel/config.py
- accel/verify/orchestrator.py
- examples/accel.yaml
- tests/unit/test_mcp_server.py
- tests/unit/test_config.py
- tests/unit/test_verify_orchestrator.py
- README.md

## Budget (Scope Budget)
- Touched Files: <= 20
- Changed LOC: <= 1200
- Dependencies: no new deps

## Failure Modes
- 同步等待窗口过长导致调用端超时
- context 超时降级遗漏导致状态不可追踪
- 失败分类字段与既有 status 语义冲突
- Windows 兼容命令映射错误执行

## Pre-Snapshot
- Git rollback anchor: current HEAD
- events.jsonl append rollback_point

## Test Plan
- Unit:
  - tests/unit/test_mcp_server.py
  - tests/unit/test_config.py
  - tests/unit/test_verify_orchestrator.py
- Gate:
  - ruff check .
  - mypy .

## Post-Gate
- Gate Set: full
- 若失败，输出 DEGRADE_REASON/RISK/BACKFILL_PLAN 并标记 Verified-Pending

## Rollback
- git restore accel/mcp_server.py accel/config.py accel/verify/orchestrator.py examples/accel.yaml tests/unit/test_mcp_server.py tests/unit/test_config.py tests/unit/test_verify_orchestrator.py README.md
