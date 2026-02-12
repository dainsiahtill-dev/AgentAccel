# Full Plan: post_stress_hardening
Mode: S2 Standard | Approval: Explicit/Fallback

## Contract Snapshot
Goal: 修复本次压力测试暴露的 agent-accel 非环境类问题（context 超时/异步、状态一致性、verify 预设、context 枚举校验提示）。
Acceptance Criteria:
- AC1: 新增异步 context 执行路径，避免长任务阻塞同步 RPC。
- AC2: context 支持可配置超时参数并输出清晰 timeout 结果。
- AC3: index/verify 状态输出消除 progress=100 但 running 的不一致展示。
- AC4: verify 支持 fast/full 预设并与现有参数兼容。
- AC5: context 参数枚举值在本地快速校验并返回可操作提示。
- AC6: 增加对应单元测试并通过 ruff/mypy/pytest 单测门禁。

## Scope
- accel/mcp_server.py
- tests/unit/test_mcp_server.py
- examples/accel.yaml
- (if needed) accel/config.py

## Budget (Scope Budget)
- Touched Files: <= 20
- Changed LOC: <= 1200
- Dependencies: no new deps

## Failure Modes
- Context async worker can leak running state if exception handling missing.
- New preset logic can override existing flags unexpectedly.
- Status normalization can break existing consumers if字段删除/语义突变.

## Pre-Snapshot
- Git clean anchor recorded via `git rev-parse HEAD`.

## Test Plan
- pytest -q tests/unit/test_mcp_server.py -k "context or verify or index"
- pytest -q tests/unit/test_config.py
- ruff check .
- mypy .

## Post-Gate
- Gate Set: full (unit + lint + typecheck)
- If full repo pytest has historical unrelated failures, mark reduced + evidence.

## Rollback
- git restore accel/mcp_server.py tests/unit/test_mcp_server.py examples/accel.yaml accel/config.py
