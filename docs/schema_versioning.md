# Schema Versioning Policy

This document defines how `agent-accel` evolves JSON contracts while keeping MCP and downstream agents stable.

## Scope

Versioned schemas live under `accel/schema/`:

- `context_pack.schema.json`
- `index_manifest.schema.json`
- `mcp_context_response.schema.json`
- `mcp_verify_events.schema.json`

Contracts are enforced by `accel/schema/contracts.py` in `off | warn | strict` modes.

## Version Markers

- Every schema must include:
  - `$schema`
  - `$id`
  - top-level `title`
- Runtime payloads must carry a contract version field when applicable:
  - `context_pack.version` and `context_pack.schema_version`
  - MCP `accel_context` response `schema_version`
  - benchmark output `schema_version` for benchmark reports

## Change Types

1. Backward compatible (non-breaking)
- Add optional fields.
- Broaden accepted enum values while preserving existing semantics.
- Add metadata blocks ignored by current readers.

2. Breaking
- Remove or rename existing fields.
- Change field type (for example `string -> object`).
- Tighten validation so valid historical payloads become invalid.

## Breaking Change Procedure

1. Introduce a new schema `$id` version suffix (for example `/v2`).
2. Keep old schema contract available during migration window.
3. Add or update normalization logic in `contracts.py`.
4. Add tests for both:
   - strict failure on invalid payloads
   - warn-mode repair/fallback behavior where supported
5. Document migration in README and this file.

## Compatibility Window

- Minimum support window for previous schema major: one minor release.
- Deprecation must be announced in release notes before removal.

## Current Contract Baseline

- `context_pack.schema.json`:
  - `version` is required.
  - `schema_version` is optional but emitted by runtime payloads (`1`).
- `mcp_context_response.schema.json`:
  - `schema_version` is required and validated in strict mode.

## CI and Verification Expectations

- Schema files must be valid JSON (UTF-8).
- Contract tests must pass:
  - `tests/unit/test_output_constraints.py`
  - related MCP tests (for example `tests/unit/test_mcp_server.py`)
- Full gate remains:
  - `python -m pytest -q`
  - `ruff check .`
  - `mypy .`
