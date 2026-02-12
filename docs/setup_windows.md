# Setup on Windows

## Prerequisites
- Windows 10/11
- Python 3.11+
- Git
- ripgrep (`rg`) optional but recommended

## Install
```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e .
```

## Initialize
```powershell
accel init --project .
accel doctor --output json
```

## Build Index and Context
```powershell
accel index build --project .
accel context --project . --task "fix headers input issue" --out context_pack.json
```

For monorepos/multi-workspace repos, keep `index.scope_mode: auto` (default) so index build uses git-tracked files first and avoids narrow `src/**`-only coverage.

## Verify
```powershell
accel verify --project . --changed-files src\foo.py src\bar.ts
```

## Phase-1 Runtime Knobs (Optional)
Create `accel.local.yaml` (or use env vars) to control semantic cache, verify plan cache, and constraints:

```yaml
runtime:
  semantic_cache_enabled: true
  semantic_cache_mode: hybrid   # exact|hybrid
  semantic_cache_ttl_seconds: 7200
  semantic_cache_hybrid_threshold: 0.86
  semantic_cache_max_entries: 800

  command_plan_cache_enabled: true
  command_plan_cache_ttl_seconds: 900
  command_plan_cache_max_entries: 600

  verify_workspace_routing_enabled: true
  verify_preflight_enabled: true
  verify_preflight_timeout_seconds: 5

  constraint_mode: warn         # off|warn|strict
  rule_compression_enabled: true
```

Tool usage examples:

```powershell
accel context --project . --task "fix verify timeout" --changed-files accel/mcp_server.py --out context_pack.json
accel verify --project . --changed-files accel/mcp_server.py
```

Strict narrowing example:

```powershell
accel context --project . --task "fix verify timeout" --changed-files accel/mcp_server.py --strict-changed-files true --constraint-mode strict --out context_pack.json
```
