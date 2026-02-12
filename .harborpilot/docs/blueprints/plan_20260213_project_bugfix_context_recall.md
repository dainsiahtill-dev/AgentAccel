# Mini Plan: project-bugfix-context-recall
Mode: S1 Patch | Approval: Explicit

## Contract Snapshot
Goal: 只解决本项目自身 BUG（不处理纯测试脚本规范问题），修复 context 排序召回偏低。
Acceptance Criteria:
- benchmark 中 `top_file_recall_avg` 相比当前基线提升，并消除已知 0% 任务召回。
- 变更仅限 agent-accel 代码侧，不修改非必要测试脚本。
- 单元测试与质量门禁保持通过。

## Scope
- accel/query/context_compiler.py
- tests/unit/test_index_and_context.py

## Budget (Scope Budget)
- Touched Files: <= 2
- Changed LOC: <= 220
- Dependencies: no new deps

## Approach
- 在 context 排序流程中增加“changed files 强优先 + 同目录 scope affinity”。
- 保持原有 signal 体系兼容，仅补充可解释 reason/signal。
- 增加回归单测覆盖“changed file pin 到 top”行为。

## Red
- cmd: `PYTHONPATH=. pytest -q tests/unit/test_index_and_context.py`
- expected: 通过，新增 case 验证 changed-file prioritization。

## Implementation
- Edit: precise-string (context verified)

## Rollback
- Using Git
