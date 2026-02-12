# Mini Plan: gpu-accel-usage-script
Mode: S1 Patch | Approval: Explicit

## Contract Snapshot
Goal: 提供一个可直接运行的示例脚本，演示 FlagEmbedding + onnxruntime/torch 在真实流程中的 GPU 加速关键处理代码。
Acceptance Criteria:
- 新脚本可通过命令行执行，包含 embedding、相似度计算、rerank 与融合排序流程。
- 脚本支持指定模型路径与设备策略，并输出耗时与排名结果。
- 在依赖未满足或参数缺失时给出明确错误提示，不影响主项目运行。

## Scope
- `scripts/gpu_accel_usage_demo.py` (new)

## Budget (Scope Budget)
- Touched Files: <= 3
- Changed LOC: <= 450
- Dependencies: no new deps

## Approach
- 基于现有 `accel.gpu_runtime` 的设备判断逻辑。
- 直接调用 FlagEmbedding 的 `BGEM3FlagModel` 与 `FlagReranker` 演示真实推理链路。
- 输出 JSON 结构，包含 runtime、timings、scores 与最终排名。

## Red
- cmd: `.\\.venv\\Scripts\\python.exe scripts/gpu_accel_usage_demo.py --help`
- expected: 帮助信息可正常输出。

## Implementation
- Edit: precise-string (new file)

## Rollback
- `git restore .harborpilot/docs/blueprints/plan_20260213_gpu_accel_usage_script.md scripts/gpu_accel_usage_demo.py`
