# Full Plan: hp_20260216T143817Z_dab3e4
Mode: S2 Standard | Approval: Explicit

## Contract Snapshot
Goal: Add a token-efficient MCP content search feature for debugging workflows similar to repeated rg + file-range reads.
Acceptance Criteria:
- 1) Expose a new MCP tool to search file contents by regex/text with path filters and bounded results.
2) Return concise structured matches with optional context lines to reduce repeated file reads.
3) Integrate the tool into the existing MCP server with token-safe defaults.
4) Add tests validating matching/filtering/result limits.
5) Run ruff, mypy, and pytest and capture evidence.

## Scope
- `accel/mcp_server.py,accel/query/content_search.py,tests/test_mcp_server_content_search.py,tests/test_content_search.py`

## Budget (Scope Budget)
- Touched Files: <= 4
- Changed LOC: <= 180
- Dependencies: no new deps

## Approach
- Add a bounded content-search query helper and expose it as a new MCP tool with safe defaults and tests.

## Rollback
- Snapshot + rollback.sh
