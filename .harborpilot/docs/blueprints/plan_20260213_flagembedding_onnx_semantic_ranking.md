# Full Blueprint: flagembedding-onnx-semantic-ranking
Mode: S2 Standard | Approval: Explicit (user reply: "可以")

## Contract Snapshot
Goal: 在 agent-accel 中接入 FlagEmbedding + onnxruntime 的可配置语义检索/重排能力（默认关闭，按配置启用）。
Acceptance Criteria:
- 在不安装新依赖时，系统默认行为不变（纯词法排序），且不会报错。
- 提供配置开关控制语义 embedding 与 rerank 流程是否启用。
- 语义模块启用后，context 排序可融合语义分数并保留现有 changed-file 优先策略。
- 运行时能清晰报告“启用/回退原因”（依赖缺失、模型路径无效、设备不可用等）。
- Full gate 通过：`pytest -q`、`pytest -q tests/integration`、`ruff check .`、`mypy .`。

## Scope
- accel/query/context_compiler.py
- accel/config.py
- accel/config_runtime.py
- accel/cli.py
- accel/gpu_runtime.py
- accel/semantic_ranker.py (new)
- tests/unit/test_config.py
- tests/unit/test_semantic_ranker.py (new)
- README.md
- examples/accel.local.yaml.example
- pyproject.toml

## Budget (Scope Budget)
- Touched Files: <= 14
- Changed LOC: <= 1200
- Dependencies: no mandatory new deps; optional extras only

## Failure Modes
- FlagEmbedding API 版本差异导致实例化失败。
- onnxruntime 与 torch CUDA 运行时不匹配导致 GPU 初始化失败。
- 语义分数融合后破坏 changed-files 置顶策略。
- 缺少依赖时未正确回退，导致 context 生成失败。

## Pre-Snapshot
- 当前工作区非干净（包含已完成但未提交的 GPU mode switch 改动）。
- 采用快照路径（B）：`.harborpilot/snapshots/snap_<ts>_<rand>/`，包含 `index.json` + `patch.diff` + `rollback.sh`。

## Test Plan (Red/Green)
- `PYTHONPATH=. pytest -q tests/unit/test_semantic_ranker.py`
- `PYTHONPATH=. pytest -q tests/unit/test_config.py`
- `PYTHONPATH=. python -m pytest -q`
- `PYTHONPATH=. pytest -q tests/integration`
- `PYTHONPATH=. ruff check .`
- `PYTHONPATH=. mypy .`

## Post-Gate
- Gate Set: full

## Rollback
- 使用快照 `rollback.sh` 回滚本轮所有变更。
