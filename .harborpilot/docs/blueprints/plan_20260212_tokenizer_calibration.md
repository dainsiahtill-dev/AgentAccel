# Mini Plan: tokenizer-calibration
Mode: S1 Patch | Approval: Explicit

## Contract Snapshot
Goal: 把 chars/4 改成真实 tokenizer 估算并加校准系数。
Acceptance Criteria:
- `estimated_tokens` 不再固定使用 chars/4，而是优先使用真实 tokenizer。
- 支持可配置校准系数，允许按项目微调估算偏差。
- 在 tokenizer 不可用时保持可用（明确 fallback）。
- 补齐测试与文档。

## Scope
- accel/token_estimator.py (new)
- accel/config.py
- accel/mcp_server.py
- tests/unit/test_mcp_server.py
- README.md

## Budget (Scope Budget)
- Touched Files: <= 6
- Changed LOC: <= 320
- Dependencies: no new mandatory deps (optional runtime import)

## Approach
- 新增统一 token 估算模块：优先 tiktoken，失败回退 heuristic。
- runtime 增加 estimator backend/model/encoding/calibration 配置与环境变量覆盖。
- context 返回增加 tokenizer 元数据，便于审计与调优。
- 更新 README 与单测。

## Red
- cmd: `pytest -q tests/unit/test_mcp_server.py`
- expected: 新增断言先失败，改动后通过。

## Implementation
- Edit: precise-string (context verified)

## Rollback
- Using Git
