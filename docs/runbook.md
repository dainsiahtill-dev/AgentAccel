# Runbook

## Common Commands
- Full index build: `accel index build --project . --full`
- Incremental update: `accel index update --project .`
- Context generation: `accel context --project . --task "<task>"`
- Incremental verify: `accel verify --project . --changed-files ...`
- MCP server (stdio): `accel-mcp`

## MCP Integration
### Start
Run from `agent-accel` project root:

```powershell
accel-mcp
```

### Core Tools
- `accel_index_build`
- `accel_index_update`
- `accel_context`
- `accel_verify`

### Minimal Call Examples
`tools/call` for context:

```json
{
  "name": "accel_context",
  "arguments": {
    "project": "X:/Git/Harborpilot",
    "task": "refactor llm router",
    "changed_files": ["src/backend/app/routers/llm.py"],
    "include_pack": false
  }
}
```

`tools/call` for verify (evidence run):

```json
{
  "name": "accel_verify",
  "arguments": {
    "project": "X:/Git/Harborpilot",
    "changed_files": ["src/backend/app/routers/llm.py"],
    "evidence_run": true
  }
}
```

### Contract and Error Behavior
The MCP server uses FastMCP strict input validation.

- Invalid `tools/call` arguments are rejected before tool execution.
- Unknown tool names return protocol-level tool lookup errors.
- Tool runtime failures are surfaced with `ACCEL_TOOL_EXECUTION_FAILED` prefix in the error message.

When calls fail, inspect `error.message` and any `error.data` payload returned by the client SDK.

## Failure Recovery
1. Index corruption
- Delete `%ACCEL_HOME%\projects\<hash>\index\`
- Run `accel index build --project . --full`

2. Tool missing during verify
- Inspect `verify_<nonce>.log` for `DEGRADE_REASON`
- Install missing tool
- Re-run `accel verify`

3. Wrong context output
- Rebuild index with `--full`
- Check `accel doctor --print-config`

4. MCP parameter rejection
- Check `error.message` and argument schema shown in `tools/list`
- Fix payload types/required fields
- Re-send `tools/call`

## Rollback
- Revert changed code files via git:
```powershell
git restore <file1> <file2>
```
