# Full Plan: remove-gpu-stack
Mode: S2 Standard | Approval: Explicit

## Contract Snapshot
Goal: 彻底移除项目中的 GPU 库相关代码与入口，避免后续不稳定与阻塞。
Acceptance Criteria:
- 删除 GPU runtime 模块与 GPU 诊断脚本，不再保留 GPU 功能入口。
- 配置层移除 GPU 配置对象与相关环境变量解析。
- semantic ranker 不再触发 torch/onnx/gpu 探测链路，context 路径稳定可返回。
- 依赖声明移除 GPU 相关 extras。
- 关键命令通过增量门禁（ruff/mypy/context smoke/pytest collect）。

## Scope
- pyproject.toml
- accel/semantic_ranker.py
- accel/config_runtime.py
- accel/config.py
- accel/cli.py
- accel/gpu_runtime.py (delete)
- scripts/gpu_diagnostic.py (delete)
- scripts/gpu_accel_usage_demo.py (delete)
- README.md/docs (GPU 文档入口最小同步)

## Budget
- Touched Files: <= 20
- Changed LOC: <= 1200
- Dependencies: no new deps

## Failure Modes
- 兼容性：旧配置仍带 gpu 字段时需要容忍。
- 行为变化：semantic ranking 永久禁用，排序信号减少。
- 文档滞后：README 若未同步可能误导。

## Pre-Snapshot
- 工作区非干净，使用 snapshot 目录+patch 回滚。

## Test Plan
- python -m ruff check .
- python -m mypy .
- python -m pytest -q --collect-only
- python -m accel.cli context --project . --task "smoke" --changed-files accel/mcp_server.py --max-chars 3000 --max-snippets 8 --top-n-files 4 --snippet-radius 10 --out .harborpilot/logs/context_pack_remove_gpu_smoke.json --output json

## Rollback
- 使用 snapshot rollback.sh 恢复。
