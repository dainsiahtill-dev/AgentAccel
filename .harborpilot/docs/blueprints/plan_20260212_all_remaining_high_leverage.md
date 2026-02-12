# Full Blueprint: all-remaining-high-leverage
Mode: S2 Standard | Approval: Explicit (user instruction: "全部吧")

## Contract Snapshot
Goal: 在已完成 benchmark harness 的基础上，继续完成剩余高杠杆改进：Verify Selection Evidence、Explainability、并发默认值自适应、Schema 版本化强化、Language Profile 插件化、发布变更记录。
Acceptance Criteria:
- verify 返回结构中包含可机读 `verify_selection_evidence`，并在 verify 日志/jsonl 有对应证据事件。
- CLI 提供 `accel explain`，可输出入选与落选候选文件及信号分解。
- runtime 并发相关默认值支持自适应（CPU-aware）且显式配置优先。
- context/mcp 输出包含显式 `schema_version` 字段，并更新 schema/文档。
- language profile 从硬编码扩展到“注册表驱动 + 配置扩展”机制。
- 新增发布变更记录文档（CHANGELOG）。

## Scope
- accel/verify/sharding.py
- accel/verify/orchestrator.py
- accel/query/context_compiler.py
- accel/cli.py
- accel/config.py
- accel/config_runtime.py
- accel/indexers/discovery.py
- accel/indexers/__init__.py
- accel/schema/context_pack.schema.json
- accel/schema/mcp_context_response.schema.json
- accel/schema/contracts.py
- README.md
- docs/schema_versioning.md
- tests/unit/test_verify_orchestrator.py
- tests/unit/test_config.py
- tests/unit/test_mcp_server.py
- tests/unit/test_run_benchmarks_script.py
- tests/unit/test_language_profiles.py (new)
- tests/unit/test_explain_cli.py (new)
- CHANGELOG.md (new)

## Budget (Scope Budget)
- Touched Files: <= 24
- Changed LOC: <= 1600
- Dependencies: no new deps

## Failure Modes
- Verify evidence字段破坏现有返回契约或旧测试假设。
- explain 复用排序逻辑时与 context 编译逻辑漂移。
- 并发 auto 默认影响现有性能/稳定性边界。
- language profile 注册表接入后出现空 profile 导致索引/验证选择异常。

## Pre-Snapshot
- 当前工作区非干净（已有在途改动），采用快照目录 + patch.diff 锚点。
- 快照路径：`.harborpilot/snapshots/snap_20260212T_all_remaining/`

## Test Plan (Red/Green)
- `python -m pytest -q tests/unit/test_verify_orchestrator.py`
- `python -m pytest -q tests/unit/test_config.py`
- `python -m pytest -q tests/unit/test_mcp_server.py -k "token_reduction or verify"`
- `python -m pytest -q tests/unit/test_language_profiles.py`
- `python -m pytest -q tests/unit/test_explain_cli.py`
- `python -m ruff check accel tests/unit`

## Post-Gate
- Gate Set: full（在本地可执行范围内）
- 若出现不可用门禁，按 DEGRADE 模板记录并标记 Verified-Pending。

## Rollback
- Git 回滚：`git checkout -- <touched-files>`
- 快照回滚：执行 `.harborpilot/snapshots/snap_20260212T_all_remaining/rollback.sh`
