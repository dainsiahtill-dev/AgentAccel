# Mini Plan: smoke-tests-xdist
Mode: S1 Patch | Approval: Explicit

## Contract Snapshot
Goal: 按用户选择的 1,2 同时落地：补齐最小冒烟测试，并启用 pytest-xdist 后使用 -n 32 执行。
Acceptance Criteria:
- 新增可执行的最小 smoke tests，覆盖 CLI/MCP/verify 关键路径，不依赖 GPU。
- pytest 可在当前仓库收集并执行测试，不再出现“no tests collected”。
- 可使用 `python -m pytest -q -n 32` 跑通。
- 所有测试相关缓存与日志继续落在 `.harborpilot/` 体系。

## Scope
- pyproject.toml
- tests/test_smoke_cli.py
- tests/test_smoke_verify.py
- tests/test_smoke_mcp.py

## Budget (Scope Budget)
- Touched Files: <= 10
- Changed LOC: <= 500
- Dependencies: add `pytest-xdist` only (test dependency)

## Approach
- 添加轻量级 smoke tests，避免重负载与外部依赖。
- verify smoke 使用临时最小配置，命令为快速 python one-liner。
- MCP smoke 仅做 import/create_server 级健康检查。
- 使用 pytest-xdist 并发执行（32 workers），记录证据日志。

## Red
- cmd: python -m pytest -q --collect-only
- expected: collected > 0

## Implementation
- Edit: precise-string (context verified)

## Rollback
- Using existing snapshot: .harborpilot/snapshots/snap_20260213T181036Z_fab34d/rollback.sh
