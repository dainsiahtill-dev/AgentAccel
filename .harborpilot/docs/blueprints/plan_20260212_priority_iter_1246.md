# Full Plan: priority_iter_1246
Mode: S2 Standard | Approval: APPROVED-FALLBACK

## Contract Snapshot
Goal: 完善这个 agent-accel，单次迭代优先做 1+2+4+6。
Acceptance Criteria:
- 索引可观测性：输出已扫描文件数/总文件数/当前路径/预计剩余时间。
- 验证实时输出：accel_verify 可返回命令 stdout/stderr 摘要、命令级进度，并具备卡顿检测信号。
- 变更范围识别：changed_files 以 VCS 为一等来源，提供确定性来源链与置信度。
- 缓存可解释与失效策略：语义缓存绑定 git 提交哈希与文件修改状态，展示命中原因/失效原因与安全边界。

## Scope
- Runtime:
  - accel/mcp_server.py
  - accel/indexers/__init__.py
  - accel/verify/orchestrator.py
  - accel/verify/runners.py
  - accel/verify/callbacks.py
  - accel/storage/semantic_cache.py
- Tests:
  - tests/unit/test_mcp_server.py
  - tests/unit/test_semantic_cache.py

## Budget (Scope Budget)
- Touched Files: <= 12
- Changed LOC: <= 1400
- Dependencies: no new deps

## Failure Modes
- 事件流过密导致 MCP 轮询负担增加。
- 输出流式回调导致子进程 IO 死锁或线程泄漏。
- 缓存键策略升级导致旧缓存命中下降。

## Pre-Snapshot
- Strategy: Git clean anchor (A-path)
- Required evidence:
  - git rev-parse HEAD
  - git status --porcelain (empty)
  - git submodule status --recursive (if any)

## Test Plan (Red -> Green)
- Unit tests:
  - pytest -q tests/unit/test_mcp_server.py
  - pytest -q tests/unit/test_semantic_cache.py
  - pytest -q tests/unit/test_verify_orchestrator.py
  - pytest -q tests/unit/test_index_and_context.py
- Quality gates:
  - ruff check .
  - mypy .

## Post-Gate
- Gate Set target: full
- If degraded: emit DEGRADE_REASON / RISK / BACKFILL_PLAN and mark Verified-Pending.

## Rollback
- Preferred: git restore/checkout touched files to pre-snapshot HEAD.
- Anchor: commit SHA captured in events rollback_point.
