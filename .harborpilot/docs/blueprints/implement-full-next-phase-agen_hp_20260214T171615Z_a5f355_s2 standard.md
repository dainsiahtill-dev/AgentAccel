# Full Plan: hp_20260214T171615Z_a5f355
Mode: S2 Standard | Approval: Explicit

## Contract Snapshot
Goal: Implement full next-phase agent-accel expansion: data-driven evaluation pipeline, semantic relation aware retrieval/ranking, and adaptive budget/token optimization integrated into context compilation.
Acceptance Criteria:
- A reproducible benchmark/evaluation workflow exists with machine-readable metrics including Recall@K, MRR, context chars, and latency.
- Context selection/ranking uses semantic relation signals beyond lexical symbol matching and exposes those signals in output.
- Context budgeting adapts based on task intent/complexity and reports effective strategy metadata.
- Snippet packing reduces redundant content using stronger similarity controls while preserving relevance.
- Automated tests cover benchmark metrics, semantic relation scoring, and adaptive budgeting behavior.
- Project verification commands (ruff, mypy, pytest) pass and policy evidence is recorded.

## Scope
- `accel/indexers/symbols.py
accel/query/lexical_ranker.py
accel/query/ranker.py
accel/query/snippet_extractor.py
accel/query/context_compiler.py
accel/cli.py
accel/query/semantic_relations.py
accel/query/adaptive_budget.py
accel/eval/__init__.py
accel/eval/metrics.py
accel/eval/runner.py
tests/test_unit_symbols.py
tests/test_unit_ranker.py
tests/test_unit_snippet_extractor.py
tests/test_unit_eval_metrics.py
tests/test_unit_semantic_relations.py
tests/test_unit_adaptive_budget.py`

## Budget (Scope Budget)
- Touched Files: <= 17
- Changed LOC: <= 1200
- Dependencies: no new deps

## Approach
- Land full extension in one pass: benchmark/eval framework, semantic relation-aware retrieval integration, adaptive budget & dedup optimization, and test coverage with strict verification.

## Rollback
- Snapshot + rollback.sh
