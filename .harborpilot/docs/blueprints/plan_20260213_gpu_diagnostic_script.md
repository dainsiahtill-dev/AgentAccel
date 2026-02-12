# Mini Plan: gpu-diagnostic-script
Mode: S1 Patch | Approval: Explicit

## Contract Snapshot
Goal: 提供一个可通过命令行执行的 Python 脚本，专门验证 agent-accel 语义/GPU链路是否可运行。
Acceptance Criteria:
- 脚本可独立执行，输出 GPU/依赖/模型可用性诊断结果。
- 支持可选执行 embedding 编码与 rerank 前向推理冒烟。
- 提供非 0 退出码用于 CI/手工门禁（例如 `--require-gpu`）。
- 不修改现有运行时代码路径。

## Scope
- `scripts/gpu_diagnostic.py` (new)

## Budget (Scope Budget)
- Touched Files: <= 3
- Changed LOC: <= 350
- Dependencies: no new deps

## Approach
- 复用 `accel.gpu_runtime.resolve_gpu_runtime` 与 `accel.semantic_ranker.probe_semantic_runtime` 输出统一诊断。
- 在 `probe=ready` 时调用内部 runtime 构造进行 encode/rerank 冒烟。
- 默认 JSON 输出；支持 `--require-gpu` 与 `--require-ready` 触发失败退出码。

## Red
- cmd: `python scripts/gpu_diagnostic.py --output json`
- expected: 输出诊断 JSON，缺依赖场景给出明确 reason，脚本可正常退出。

## Implementation
- Edit: precise-string (new file)

## Rollback
- `git restore .harborpilot/docs/blueprints/plan_20260213_gpu_diagnostic_script.md scripts/gpu_diagnostic.py`
