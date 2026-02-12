# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased]

### Added
- Benchmark harness Markdown report output (`--out-md`) and CI workflow entrypoint (`.github/workflows/benchmark-harness.yml`).
- Verify selection evidence in `run_verify` / `run_verify_with_callback` outputs (`verify_selection_evidence`) and verification start jsonl event payloads.
- `accel explain` CLI command for ranking explainability (selected files + near-miss alternatives).
- Language profile registry support (`language_profile_registry`) with profile-driven extension and verify-group resolution.
- New unit tests for language profile registry and explain CLI.

### Changed
- Runtime worker defaults now support `auto` and resolve CPU-aware values for `max_workers`, `verify_workers`, and `index_workers`.
- Context pack now includes explicit `schema_version`.
- MCP context response now includes explicit `schema_version`.
- Verify plan in context pack now includes `selection_evidence`.

### Documentation
- Benchmark docs updated with Markdown report usage and CI artifact workflow.
- Schema versioning docs aligned with explicit `schema_version` output fields.
