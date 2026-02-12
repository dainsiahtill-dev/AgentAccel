# Mini Plan: benchmark-report-ci
Mode: S1 Patch | Approval: Explicit (user instruction)

## Contract Snapshot
Goal: 按优先级先落地 benchmark 证据链能力：可直接产出机器可读 JSON + 人类可读 Markdown，并提供可复用 CI 入口。
Acceptance Criteria:
- `scripts/run_benchmarks.py` 支持额外输出 Markdown 报告。
- 在仓库中新增可手动触发的 benchmark CI 工作流，并上传结果工件。
- README 基准测试章节更新到新参数/新流程。

## Scope
- `scripts/run_benchmarks.py`
- `.github/workflows/benchmark-harness.yml`
- `README.md`
- `examples/benchmarks/README.md`

## Budget (Scope Budget)
- Touched Files: <= 4
- Changed LOC: <= 220
- Dependencies: no new deps

## Approach
- 在 benchmark 脚本中新增 Markdown 渲染器与 `--out-md` 参数。
- 维持 JSON 输出兼容，不改变既有字段与默认行为。
- 新增 workflow_dispatch 工作流，执行 sample benchmark 并上传 JSON/MD。

## Red
- cmd: `python scripts/run_benchmarks.py --help`
- expected: 包含 `--out-md` 参数说明。

## Implementation
- Edit: precise-string (context verified)

## Rollback
- `git checkout -- scripts/run_benchmarks.py .github/workflows/benchmark-harness.yml README.md examples/benchmarks/README.md`
