# Full Blueprint: gpu-mode-switch-config-driven
Mode: S2 Standard | Approval: Explicit (user instruction: "下一步就按这个契约直接实现“配置驱动的 GPU 模式开关”")

## Contract Snapshot
Goal: 实现配置驱动的 GPU 模式开关，为后续 embedding/rerank 接入提供统一 GPU 运行时决策。
Acceptance Criteria:
- 支持配置字段：`gpu.enabled`, `gpu.policy`, `gpu.device`, `gpu.embedding_model_path`, `gpu.reranker_model_path`。
- 支持对应环境变量覆盖（至少 enabled/policy/device/path）。
- 提供统一运行时决策函数，输出 `use_gpu/effective_device/reason`，支持 `off|auto|force` 语义。
- CLI `accel doctor` 可展示该 GPU 决策结果，便于排障。
- 保持现有门禁通过（pytest + ruff + mypy）。

## Scope
- accel/config.py
- accel/config_runtime.py
- accel/cli.py
- accel/gpu_runtime.py (new)
- tests/unit/test_config.py
- tests/unit/test_gpu_runtime.py (new)
- README.md
- examples/accel.local.yaml.example

## Budget (Scope Budget)
- Touched Files: <= 10
- Changed LOC: <= 700
- Dependencies: no new deps

## Failure Modes
- `force` 语义定义不清导致在无 CUDA 环境下行为不可预期。
- 配置与环境变量优先级不一致，出现“配置已开但 runtime 仍 CPU”。
- `doctor` 输出与实际 runtime 决策不一致，导致误导排障。

## Pre-Snapshot
- 当前工作区为干净 Git 状态。
- rollback anchor:
  - `git rev-parse HEAD`
  - `git status --porcelain` (empty)

## Test Plan (Red/Green)
- `PYTHONPATH=. pytest -q tests/unit/test_config.py tests/unit/test_gpu_runtime.py`
- `PYTHONPATH=. python -m pytest -q`
- `PYTHONPATH=. pytest -q tests/integration`
- `PYTHONPATH=. ruff check .`
- `PYTHONPATH=. mypy .`

## Post-Gate
- Gate Set: full

## Rollback
- `git restore accel/config.py accel/config_runtime.py accel/cli.py accel/gpu_runtime.py tests/unit/test_config.py tests/unit/test_gpu_runtime.py README.md examples/accel.local.yaml.example`
