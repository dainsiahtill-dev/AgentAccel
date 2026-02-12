# Milestone Plan: hybrid-brain-upgrade-v1
Mode: Non-Implementation Planning | Scope: Roadmap + Execution Gates

## Contract Snapshot
Goal: 先建立统一里程碑，再分阶段推进下一轮升级（前置大脑、Prompt Compression & Semantic Cache、Constrained Decoding、NPU/GPU 异构加速）。
Acceptance Criteria:
- 输出一个可执行里程碑，包含阶段目标、可量化指标、退出条件与风险边界。
- 明确“先做什么、后做什么”，并给出每阶段 DoD（Definition of Done）。
- 保留可回退策略，避免一次性大改导致稳定性回退。

## Constraints
- 当前硬件可用显存约 48GB，优先本地稳定性与吞吐。
- 现有 `accel_context` / `accel_verify*` 链路必须保持可用，不允许破坏现有 MCP 兼容性。
- 默认目标是先降 token，再追求更激进加速。

## Milestone ID
- `MS-2026Q1-HYBRID-BRAIN-V1`

## Milestone Objective
- 在不破坏当前可用性的前提下，把“上下文缩面 + 本地前置决策 + 结构化约束 + 异构执行”串成一条可审计闭环。

## Phase Breakdown
### Phase A: 前置大脑基础接入（Local SLM Gate）
Scope:
- 接入本地 `Qwen3-Coder-30B-A3B`（Ollama）到 context 前置流程。
- 增加可控开关：`brain_assist=off|auto|on`。
- 输出结构化决策：`route`、`confidence`、`compressed_task`、`inferred_changed_files`、`hints`。
DoD:
- `accel_context` 在 `brain_assist=auto` 下可稳定运行。
- 失败可自动回退到当前规则链路，失败率不高于现有基线。
- 新增指标可观测：`brain_hit_rate`、`brain_fallback_rate`、`brain_latency_p95`。

### Phase B: Prompt Compression + Semantic Cache
Scope:
- 加入提示词压缩器（本地）与语义缓存（hash + embedding / key）。
- 缓存命中策略加入 TTL 与失效规则（文件 hash 变化即失效）。
DoD:
- 平均云端输入 token 下降（目标先设 `>=25%`）。
- 缓存命中场景下 end-to-end 延迟改善（目标先设 `>=20%`）。
- 缓存误命中（错误复用）有审计证据并可快速回滚。

### Phase C: 结构化输出与约束解码
Scope:
- 为前置大脑与关键 MCP 结果定义严格 schema。
- 增强解析失败兜底（重试/降级）与字段完整性校验。
DoD:
- 非结构化响应比例显著下降（目标 `<=1%`）。
- 工具调用参数错误率下降（目标 `>=30%` 改善）。
- JSON/schema 违规可被日志精确定位。

### Phase D: 硬件感知执行（NPU/GPU 异构）
Scope:
- 加入 workload policy：根据任务类型、上下文长度、并发状态选择 CPU/GPU/NPU 路径。
- 建立资源预算与保护阈值（显存、队列、超时）。
DoD:
- 高峰期稳定性不下降（超时率不高于基线）。
- 关键路径吞吐提升（目标 `>=15%`）。
- 策略可动态回退到 CPU-only 模式。

## Global Success Metrics
- `cloud_input_tokens_per_task` 持续下降。
- `cloud_escalation_rate` 可控（高风险任务保持必要升级，低风险任务更多本地闭环）。
- `retry_rate` 下降。
- `acceptance_pass_rate` 不下降。
- `p95_latency` 不劣化。

## Guardrails
- 高风险路径（权限、支付、删除、发布）默认不走激进本地自动化，必须保守策略。
- 任一阶段出现稳定性退化，立即切回 `brain_assist=off`。
- 所有新增策略必须保留 feature flag，默认可关闭。

## Rollback Strategy
- 配置级回滚：关闭 `brain_assist` 与新策略开关。
- 路由级回滚：强制回退到当前 `accel_context` 规则路径。
- 版本级回滚：按 Git 快照恢复到里程碑前提交点。

## Suggested Execution Order
1. Phase A
2. Phase B
3. Phase C
4. Phase D

## Next Action
- 开始 Phase A 设计与最小实现（仅接入、开关、指标，不改现有默认行为）。
