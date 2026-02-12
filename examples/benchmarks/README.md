# Benchmark Harness

This folder contains reproducible benchmark inputs for `agent-accel`.

## Goals

- Measure token reduction against indexed source and changed-files baselines.
- Track context generation latency and retrieval quality (`top_file_recall`).
- Optionally track verification pass-rate and timing with `--run-verify`.

## Inputs

- `tasks.sample.json`: sample task suite.
  - `id`: stable benchmark task id.
  - `task`: natural-language task prompt passed to `compile_context_pack`.
  - `changed_files`: scoped file list for the task.
  - `hints`: optional hint list.
  - `expected_top_files`: target files for recall scoring.

## Run

```bash
python scripts/run_benchmarks.py \
  --project . \
  --tasks examples/benchmarks/tasks.sample.json \
  --index-mode update \
  --out examples/benchmarks/results_local.json \
  --out-md examples/benchmarks/results_local.md
```

With verify metrics enabled:

```bash
python scripts/run_benchmarks.py \
  --project . \
  --tasks examples/benchmarks/tasks.sample.json \
  --index-mode update \
  --run-verify \
  --out examples/benchmarks/results_with_verify.json \
  --out-md examples/benchmarks/results_with_verify.md
```

## Output

Each run writes one UTF-8 JSON report and can optionally emit a Markdown report via `--out-md`:

- Metadata: generation timestamp, git head/status, runtime config snapshot.
- Per-task metrics:
  - `context_build_seconds`
  - `context_tokens`
  - `token_reduction_vs_full_index`
  - `token_reduction_vs_changed_files`
  - `top_file_recall`
  - verify metrics (`verify_status`, `verify_exit_code`, `verify_seconds`) when enabled
- Aggregate summary:
  - average/p50 context latency
  - average context tokens
  - average token reduction and recall
  - verify pass-rate and average verify duration (when enabled)

Manual CI entrypoint:

- `.github/workflows/benchmark-harness.yml` (`workflow_dispatch`) runs this harness and uploads JSON/Markdown artifacts.

## Reproducibility Notes

- Keep the task file and benchmark script versioned together.
- Prefer running on a pinned commit and clean workspace.
- Use the same `token_estimator_*` runtime knobs for apples-to-apples comparisons.
