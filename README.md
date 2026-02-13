# 🚀 agent-accel

<p align="center">
  <img src="docs/assets/cover.png" alt="agent-accel architecture diagram" width="100%">
</p>

<p align="center">
  <b>Standalone Local Code Intelligence & Verification Coprocessor for AI Coding Agents</b>
</p>

<p align="center">
  <a href="#features">Features</a> •
  <a href="#installation">Installation</a> •
  <a href="#quick-start">Quick Start</a> •
  <a href="#mcp-server">MCP Server</a> •
  <a href="#benchmark-harness">Benchmark Harness</a> •
  <a href="#schema-contracts--versioning">Schema Contracts</a> •
  <a href="#architecture">Architecture</a> •
  <a href="#configuration">Configuration</a>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/python-3.11%2B-blue?style=flat-square&logo=python" alt="Python 3.11+">
  <img src="https://img.shields.io/badge/license-MIT-green?style=flat-square" alt="License: MIT">
  <img src="https://img.shields.io/badge/platform-Windows%20%7C%20Linux%20%7C%20macOS-lightgrey?style=flat-square" alt="Platform">
</p>

---

## 📖 Overview

`agent-accel` is a high-performance local code intelligence tool designed specifically for AI coding agents. It reduces token consumption by **95%+** through pre-indexed, scored, and budget-constrained context generation.

### The Problem

AI coding agents often waste tokens on:
- 🔍 Full codebase scans (~500KB+ per request)
- 📊 Redundant AST parsing (~50KB per analysis)
- 🔧 Manual dependency tracing
- ⏱️ Repeated verification command discovery

### The Solution

```
┌─────────────────────────────────────────────────────────┐
│                    AI Coding Agent                      │
│  - Receives task → Generates plan → Executes → Verifies │
└────────────────────┬────────────────────────────────────┘
                     │
         ┌───────────▼───────────┐
         │    agent-accel        │ ← Local Coprocessor
         │   (Token Optimizer)   │
         │                       │
         │   Input: task + files │
         │   Output: context_pack│
         │   (≤24KB JSON)        │
         └───────────┬───────────┘
                     │
         ┌───────────▼───────────┐
         │   Agent reads JSON    │ ← No full codebase scan
         │   → Understands scope │
         │   → Executes task     │
         └───────────────────────┘
```

---

## ✨ Features

### Core Capabilities

| Feature | Description |
|---------|-------------|
| **📁 Incremental Indexing** | Symbols, references, dependencies, and test ownership tracking |
| **📦 Context Pack Generation** | Budget-constrained `.harborpilot/logs/context_pack.json` for AI consumption |
| **🔎 Explainability (`accel explain`)** | Score breakdown for selected files and near-miss alternatives |
| **✅ Verification Orchestration** | Incremental test/lint/typecheck with smart sharding |
| **🧠 Verify Selection Evidence** | Machine-readable rationale for baseline vs accelerated checks |
| **🧩 Language Profile Registry** | Profile-driven extension/verify-group mapping (built-in + custom) |
| **🔌 MCP Server** | FastMCP-based service layer with stdio transport |
| **🪝 CLI Hooks** | Pre/post execution hooks for agent integration |

### Token Optimization

| Scenario | Without agent-accel | With agent-accel | Savings |
|----------|---------------------|------------------|---------|
| Code Reading | Full scan (~500KB+) | `.harborpilot/logs/context_pack.json` (~24KB) | **~95%** |
| Symbol Understanding | AST parsing (~50KB) | `top_files` list (~2KB) | **~96%** |
| Verification Planning | Manual dependency analysis | `verify_plan` ready-to-use | **~100%** |

---

## 📦 Installation

### Prerequisites

- Python 3.11 or higher
- Git
- (Optional) [ripgrep](https://github.com/BurntSushi/ripgrep) for faster indexing

### Install from Source

```bash
# Clone the repository
git clone <repository-url>
cd agent-accel

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# or: .venv\Scripts\activate  # Windows

# Install dependencies
pip install -e .

# Optional: syntax parser + lexical search
pip install -e ".[syntax]"        # Tree-sitter parser path
pip install -e ".[search]"        # Tantivy lexical ranker path
```

### Verify Installation

```bash
# Check CLI is available
accel --help

# Check MCP server
accel-mcp --help

# Run diagnostics
accel doctor
```

---

## 🚀 Quick Start

### 1. Initialize Project

```bash
# Navigate to your project
cd /path/to/your/project

# Initialize agent-accel
accel init
```

This creates `accel.yaml` and `accel.local.yaml.example` with sensible defaults. Runtime artifacts are stored under `.harborpilot/runtime/agent-accel`.

### 2. Build Indexes

```bash
# Full index build (first time)
accel index build --full

# Incremental update (subsequent runs)
accel index update
```

### 3. Generate Context Pack

```bash
# Generate context for a specific task
accel context "Fix authentication bug in login module" \
  --changed-files "src/auth.py,src/login.py" \
  --out .harborpilot/logs/context_pack.json
```

### 4. Run Verification

```bash
# Run incremental verification
accel verify \
  --changed-files "src/auth.py,src/login.py" \
  --evidence-run
```

---

## 🔌 MCP Server

agent-accel provides a [Model Context Protocol (MCP)](https://modelcontextprotocol.io/) server for seamless AI agent integration.

### Available Tools

#### `accel_index_build`

Build indexes for the target project.

```json
{
  "project": ".",
  "full": true
}
```

#### `accel_index_update`

Incrementally update indexes for changed files.

```json
{
  "project": "."
}
```

#### `accel_context`

Generate a budgeted context pack for a task.

```json
{
  "project": ".",
  "task": "Fix authentication bug",
  "changed_files": ["src/auth.py", "src/login.py"],
  "evidence_run": true,
  "include_pack": false
}
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `project` | string | `"."` | Project directory path |
| `task` | string | required | Task description |
| `changed_files` | string[] \| string | `null` | Changed files (array, JSON-array string, or comma-separated) |
| `hints` | string[] \| string | `null` | Additional context hints (array, JSON-array string, or comma-separated) |
| `include_pack` | boolean \| string | `false` | Include full pack in response (`true/false` or boolean-like string) |
| `budget` | object \| string | `null` | Budget override object or preset (`tiny`/`small`/`medium`/`large`/`xlarge`) |
| `strict_changed_files` | boolean \| string | `null` | Strict mode: require explicit `changed_files` or git delta, then prune non-changed context items; if pruning empties context, inject minimal fallback snippets from changed files |
| `snippets_only` | boolean \| string | `false` | Output lightweight pack containing only snippets |
| `include_metadata` | boolean \| string | `true` | Include `meta` in generated pack payload |
| `semantic_cache` | boolean \| string | `true` | Enable semantic context cache lookup/store |
| `semantic_cache_mode` | string | `hybrid` | Cache lookup mode: `exact` or `hybrid` (aliases: `readwrite`, `read_write`, `read-write`, `rw`) |
| `constraint_mode` | string | `warn` | Output constraint behavior: `off` \| `warn` \| `strict` |
| `wait_for_completion` | boolean \| string | `true` | Synchronous bounded wait for final result before fallback |
| `sync_wait_seconds` | integer \| string | `null` | Override synchronous wait window for context polling |
| `sync_timeout_action` | string | `fallback_async` | Timeout action: `fallback_async` (default) or `cancel` |
| `rpc_timeout_seconds` | integer \| string | `null` | Hard timeout for context compilation worker (separate from sync wait) |

`budget` string presets are aliases for common token envelopes. Example:

```json
{
  "project": ".",
  "task": "Trace flaky verification issue",
  "changed_files": "accel/mcp_server.py,tests/unit/test_mcp_server.py",
  "hints": "[\"focus:cancellation\", \"risk:state-machine\"]",
  "budget": "small"
}
```

**Auto defaults (when omitted):**
- `budget`: adaptive policy (`tiny` for quick/small scope, `small` for daily usage, `medium` for complex scope)
- `changed_files`: auto-discovered from git diff (worktree + staged) when available
- if `changed_files` is still empty, `accel_context` continues with warning by default; set `runtime.context_require_changed_files=true` to enforce fail-fast
- sync path defaults to bounded wait; when wait window is exceeded, `accel_context` returns `status=running` + `job_id` and continues asynchronously

**Key response fields:**
- `estimated_tokens`: tokenizer-based estimate for generated context payload (calibration applied)
- `estimated_source_tokens`: source baseline estimate derived from tokenizer chars-per-token ratio
- `estimated_changed_files_tokens`: token baseline from resolved `changed_files` content
- `estimated_snippets_only_tokens`: token baseline from snippets-only payload
- `compression_ratio`: `context_chars / source_chars` (smaller is better)
- `token_reduction_ratio`: `1 - compression_ratio`
- `token_reduction`: structured three-baseline comparison:
  - `vs_full_index`
  - `vs_changed_files`
  - `vs_snippets_only`
  - note: baseline ratios can be negative when context is larger than the selected baseline
- `fallback_confidence`: confidence score for non-user changed-files inference (especially planner/index fallback)
- `output_mode`: `full` | `snippets_only`
- `selected_tests_count`: number of tests selected in `verify_plan.target_tests`
- `selected_checks_count`: number of checks selected in `verify_plan.target_checks`
- `semantic_cache_hit`: whether current context came from semantic cache
- `semantic_cache_mode_used`: `off` | `exact` | `hybrid` | `miss`
- `semantic_cache_similarity`: hybrid cache similarity score when applicable
- `compression_rules_applied`: per-rule compression counters
- `compression_saved_chars`: total chars reduced by snippet rule compression
- `constraint_repair_count`: automatic repair count produced by constraints layer
- `constraint_warnings`: repair warning messages (empty when fully compliant)
- `strict_scope_filtered_top_files` / `strict_scope_filtered_snippets`: count of non-changed items removed under strict scope
- `strict_scope_injected_top_files` / `strict_scope_injected_snippets`: count of changed-file fallback items injected when strict pruning empties context
- `budget_source` / `budget_preset`: whether budget came from user input or auto policy
- `changed_files_source`: `user` | `git_auto` | `manifest_recent` | `planner_fallback` | `index_head_fallback` | `none`
- `token_estimator`: backend/model/encoding/calibration metadata used for estimation
- `out_meta`: sidecar metadata file path (`*.meta.json`) containing estimator/baseline/source/audit fields
- `schema_version`: context response contract version marker

#### `accel_verify`

Start incremental verification with runtime override options.

```json
{
  "project": ".",
  "changed_files": ["src/auth.py"],
  "evidence_run": true,
  "fast_loop": false,
  "verify_fail_fast": false,
  "verify_workers": 8
}
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `project` | string | `"."` | Project directory path |
| `changed_files` | string[] \| string | `null` | Changed files (array or comma-separated string) |
| `evidence_run` | boolean \| string | `false` | Evidence collection mode |
| `fast_loop` | boolean \| string | `false` | Fast verification loop |
| `verify_fail_fast` | boolean \| string | `null` | Stop on first failure |
| `verify_cache_enabled` | boolean \| string | `null` | Enable verification result cache |
| `verify_cache_failed_results` | boolean \| string | `null` | Also cache failed/timed-out command results |
| `verify_workers` | integer \| string | `null` | Number of parallel workers |
| `per_command_timeout_seconds` | integer \| string | `null` | Timeout per command |
| `verify_cache_ttl_seconds` | integer \| string | `null` | TTL for successful cache entries |
| `verify_cache_failed_ttl_seconds` | integer \| string | `null` | TTL for failed cache entries (short-term) |
| `verify_cache_max_entries` | integer \| string | `null` | Max retained command cache entries |
| `command_plan_cache_enabled` | boolean \| string | `null` | Cache `select_verify_commands` planning output |
| `constraint_mode` | string | `null` | Summary/output contract mode: `off` \| `warn` \| `strict` |
| `wait_for_completion` | boolean \| string | `false` | Synchronous bounded wait for final result |
| `sync_wait_seconds` | integer \| string | `null` | Override synchronous wait window (still capped by MCP RPC-safe limit, default 45s) |
| `sync_timeout_action` | string | `poll` | Timeout action: `poll` (default) or `cancel` |
| `sync_cancel_grace_seconds` | number \| string | `null` | Extra grace window after auto-cancel request |

**Response behavior:**
- Returns quickly with `status=started` and `job_id` to avoid MCP 60s call timeouts.
- For live progress, poll:
  - `accel_verify_status(job_id)`
  - `accel_verify_events(job_id, since_seq, max_events=30, include_summary=true)` (recommended compact mode)
- Commands are auto-routed by workspace when possible (for example `frontend/` node scripts in monorepos).
- Optional preflight checks run before command execution and emit explicit degrade reasons for missing manifests/tools.
- For compatibility, a bounded synchronous wait mode is still available internally.
- If `wait_for_completion=true` and timeout occurs:
  - `sync_timeout_action=poll`: return `timed_out=true` and keep job running for async polling.
  - `sync_timeout_action=cancel`: request auto-cancel and return cancelled timeout payload (no hanging job by default path).
- `fast_loop=true` now defaults `verify_cache_failed_results=true` unless explicitly overridden, reducing repeat-failure cost in rapid loops.
- Failure outputs include `failure_kind` (`project_gate_failed` | `executor_failed` | `mixed_failed`) for clearer diagnosis.
- If verify plan resolves to zero commands, status is `degraded` (not `success`) with explicit `DEGRADE_REASON` to avoid false-positive green results.
- Verify result payload includes `verify_selection_evidence` (baseline layer vs incremental acceleration layer).

#### `accel_verify_events` (compact recommendations)

```json
{
  "job_id": "verify_xxx",
  "since_seq": 0,
  "max_events": 30,
  "include_summary": true
}
```

- `max_events`: clip to the latest N events (default 30, max 500)
- `include_summary`: include aggregate counters and latest stage/state for model-friendly ingestion
- heartbeat events may include command-level fields: `current_command`, `command_elapsed_sec`, `command_timeout_sec`, `command_progress_pct`
- command completion events may include tails: `stdout_tail`, `stderr_tail`, plus `completed` / `total`
- verify jsonl command events include structured fields: `mode`, `fail_fast`, `cache_hits`, `cache_misses`, `fail_fast_skipped`, `command_index`
- summary payload includes `state_source` and `constraint_repair_count`

Tokenizer estimation runtime knobs (via `accel.local.yaml` runtime or env):
- `token_estimator_backend`: `auto` | `tiktoken` | `heuristic`
- `token_estimator_encoding`: e.g. `cl100k_base`
- `token_estimator_model`: optional model name for encoder resolution
- `token_estimator_calibration`: positive float multiplier (default `1.0`)
- `token_estimator_fallback_chars_per_token`: fallback ratio when tokenizer is unavailable (default `4.0`)
- `semantic_cache_enabled`: enable context semantic cache (default `true`)
- `semantic_cache_mode`: `exact` | `hybrid` (default `hybrid`)
- `semantic_cache_ttl_seconds`: context cache TTL (default `7200`)
- `semantic_cache_hybrid_threshold`: hybrid similarity threshold (default `0.86`)
- `semantic_cache_max_entries`: max cached context entries (default `800`)
- `command_plan_cache_enabled`: enable verify command-plan cache (default `true`)
- `command_plan_cache_ttl_seconds`: verify plan cache TTL (default `900`)
- `command_plan_cache_max_entries`: max cached verify plans (default `600`)
- `verify_workspace_routing_enabled`: auto-route verify commands to matching workspaces (default `true`)
- `verify_preflight_enabled`: run verify preflight checks before command execution (default `true`)
- `verify_preflight_timeout_seconds`: timeout for preflight module probes (default `5`)
- `constraint_mode`: output constraint mode `off|warn|strict` (default `warn`)
- `rule_compression_enabled`: enable snippet rule compression (default `true`)
- `sync_verify_wait_seconds`: default sync wait window for `accel_verify` (default `45.0`)
- `sync_index_wait_seconds`: default sync wait window for `accel_index_build/update` (default `45.0`)
- `sync_context_wait_seconds`: default sync wait window for `accel_context` (default `45.0`)
- runtime `sync_*_wait_seconds` values are RPC-safe capped in MCP tools (default cap 45s), including per-call `sync_wait_seconds` overrides.
- `sync_verify_timeout_action`: `poll` | `cancel` (default `poll`)
- `sync_verify_cancel_grace_seconds`: grace period after auto-cancel request (default `5.0`)
- `sync_context_timeout_action`: `fallback_async` | `cancel` (default `fallback_async`)
- `max_workers` / `verify_workers` / `index_workers`: support `auto` (CPU-aware defaults, explicit values still override)

### Running MCP Server

```bash
# Start MCP server (stdio transport)
accel-mcp

# With debug logging
ACCEL_MCP_DEBUG=1 accel-mcp

# With custom max runtime (seconds)
ACCEL_MCP_MAX_RUNTIME=7200 accel-mcp
```

### MCP Configuration Example

For Claude Desktop or other MCP clients:

```json
{
  "mcpServers": {
    "agent-accel": {
      "command": "python",
      "args": ["X:/MCPs/agent-accel/scripts/accel_mcp_entry.py"],
      "env": {
        "ACCEL_MCP_DEBUG": "1"
      }
    }
  }
}
```

This launcher avoids `Transport closed` failures when client runtime does not preserve `PYTHONPATH`.

---

## 📋 Context Pack Structure

The `.harborpilot/logs/context_pack.json` output is designed for direct AI consumption:

```json
{
  "version": 1,
  "schema_version": 1,
  "task": "Fix authentication bug in login module",
  "budget": {
    "max_chars": 24000,
    "max_snippets": 60,
    "top_n_files": 12
  },
  "top_files": [
    {
      "path": "src/auth.py",
      "score": 0.95,
      "reasons": ["changed_file", "symbol_match"],
      "signals": [
        {"signal_name": "symbol_match", "score": 0.92},
        {"signal_name": "test_relevance", "score": 0.88}
      ]
    }
  ],
  "snippets": [
    {
      "path": "src/auth.py",
      "start_line": 45,
      "end_line": 85,
      "symbol": "authenticate_user",
      "content": "def authenticate_user(username, password):\n    ..."
    }
  ],
  "verify_plan": {
    "target_tests": ["tests/test_auth.py", "tests/test_login.py"],
    "target_checks": ["pytest -q tests/test_auth.py", "mypy src/auth.py"],
    "selection_evidence": {
      "run_all": false,
      "enabled_verify_groups": ["python"],
      "changed_verify_groups": ["python"],
      "layers": [
        {
          "layer": "safety_baseline",
          "commands": ["python -m pytest -q", "python -m mypy ."]
        },
        {
          "layer": "incremental_acceleration",
          "targeted_tests": ["tests/test_auth.py"]
        }
      ]
    }
  }
}
```

### AI Agent Workflow

1. **Read `top_files`** → Know which files are relevant
2. **Read `snippets`** → Get compressed code (~40 lines each)
3. **Read `verify_plan`** → Execute verification commands directly

---

## ⚙️ Configuration

### Project Configuration (`accel.yaml`)

```json
{
  "version": 1,
  "project_id": "my_project",
  "language_profiles": ["python", "typescript"],
  "language_profile_registry": {
    "python": { "extensions": [".py"], "verify_group": "python" },
    "typescript": { "extensions": [".ts", ".tsx", ".js", ".jsx"], "verify_group": "node" }
  },
  "index": {
    "scope_mode": "auto",
    "include": ["src/**", "tests/**"],
    "exclude": [".git/**", "node_modules/**", "dist/**", "build/**", "target/**", ".venv/**", "venv/**"],
    "max_file_mb": 2
  },
  "context": {
    "top_n_files": 12,
    "snippet_radius": 40,
    "max_chars": 24000,
    "max_snippets": 60
  },
  "verify": {
    "python": [
      "python -m pytest -q",
      "python -m ruff check .",
      "python -m mypy ."
    ],
    "node": [
      "npm test --silent",
      "npm run lint",
      "npm run typecheck"
    ]
  }
}
```

### Local Configuration (`accel.local.yaml`)

```json
{
  "runtime": {
    "max_workers": "auto",
    "verify_workers": "auto",
    "index_workers": "auto",
    "semantic_ranker_enabled": false,
    "semantic_ranker_provider": "off",
    "semantic_ranker_max_candidates": 120,
    "semantic_ranker_embed_weight": 0.3,
    "semantic_reranker_enabled": false,
    "semantic_reranker_top_k": 30,
    "semantic_reranker_weight": 0.15,
    "verify_fail_fast": false,
    "verify_cache_enabled": true,
    "verify_cache_ttl_seconds": 900,
    "verify_workspace_routing_enabled": true,
    "verify_preflight_enabled": true,
    "verify_preflight_timeout_seconds": 5,
    "per_command_timeout_seconds": 1200,
    "total_verify_timeout_seconds": 3600
  }
}
```

Semantic ranker runtime knobs:
- Current build note: semantic model execution is removed; these switches are compatibility placeholders and do not load external semantic/GPU runtimes.
- `semantic_ranker_enabled`: master switch (default `false`)
- `semantic_ranker_provider`: `off | auto | flagembedding` (`auto` resolves to `flagembedding`)
- `semantic_ranker_max_candidates`: top lexical candidates to re-score semantically
- `semantic_ranker_embed_weight`: blend weight for embedding similarity (`0.0~1.0`)
- `semantic_reranker_enabled`: enable second-stage reranker if reranker model is configured
- `semantic_reranker_top_k`: reranker scope size
- `semantic_reranker_weight`: blend weight for reranker score (`0.0~1.0`)

`accel doctor` now includes `semantic_runtime` with readiness/fallback reason:
- `ready`
- `disabled_by_config`
- `flagembedding_unavailable`
- `embedding_model_missing`
- `removed_from_build`

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `ACCEL_MCP_DEBUG` | Enable MCP debug logging | `0` |
| `ACCEL_MCP_MAX_RUNTIME` | Max server runtime (seconds) | `3600` |
| `ACCEL_SEMANTIC_RANKER_ENABLED` | Enable semantic ranker pipeline (`true/false`) | `false` |
| `ACCEL_SEMANTIC_RANKER_PROVIDER` | Semantic provider (`off/auto/flagembedding`) | `off` |
| `ACCEL_SEMANTIC_RANKER_MAX_CANDIDATES` | Candidate files for semantic re-score | `120` |
| `ACCEL_SEMANTIC_RANKER_BATCH_SIZE` | Embedding batch size | `16` |
| `ACCEL_SEMANTIC_RANKER_EMBED_WEIGHT` | Embedding blend weight (`0~1`) | `0.3` |
| `ACCEL_SEMANTIC_RERANKER_ENABLED` | Enable reranker stage (`true/false`) | `false` |
| `ACCEL_SEMANTIC_RERANKER_TOP_K` | Top-K files for reranker stage | `30` |
| `ACCEL_SEMANTIC_RERANKER_WEIGHT` | Reranker blend weight (`0~1`) | `0.15` |

---

## 🏗️ Architecture

```
agent-accel/
├── accel/
│   ├── mcp_server.py          # FastMCP server implementation
│   ├── cli.py                  # Command-line interface
│   ├── config.py               # Configuration management
│   ├── semantic_ranker.py      # Compatibility semantic ranker stub (runtime removed)
│   ├── indexers/               # Code indexing modules
│   │   ├── symbols.py          # Symbol extraction
│   │   ├── references.py       # Reference tracking
│   │   ├── deps.py             # Dependency analysis
│   │   └── tests_ownership.py  # Test ownership mapping
│   ├── query/                  # Context query engine
│   │   ├── planner.py          # Query planning
│   │   ├── ranker.py           # File scoring
│   │   ├── snippet_extractor.py # Code snippet extraction
│   │   └── context_compiler.py # Context pack compilation
│   ├── verify/                 # Verification orchestration
│   │   ├── orchestrator.py     # Main verification logic
│   │   ├── runners.py          # Command execution
│   │   └── sharding.py         # Test sharding
│   ├── storage/                # Data persistence
│   │   ├── cache.py            # Cache management
│   │   ├── index_cache.py      # Index storage
│   │   └── state_db.py         # State database
│   ├── hooks/                  # Agent hooks
│   │   ├── pre_hook.py         # Pre-execution hook
│   │   └── post_hook.py        # Post-execution hook
│   └── telemetry/              # Telemetry & events
│       └── events.py           # Event logging
├── accel/schema/               # JSON schemas
├── pyproject.toml              # Package configuration
└── README.md                   # This file
```

---

## 🛠️ CLI Reference

### Global Options

```bash
accel [command] [options]
```

| Option | Description |
|--------|-------------|
| `-p, --project` | Project directory (default: current) |
| `-o, --output` | Output format: `text` or `json` |

### Commands

#### `init`

Initialize agent-accel for a project.

```bash
accel init [--force]
```

#### `index build`

Build code indexes.

```bash
accel index build [--full]
```

#### `index update`

Incrementally update indexes.

```bash
accel index update
```

#### `context`

Generate context pack for a task.

```bash
accel context <task> \
  [--changed-files <files>] \
  [--hints <hints>] \
  [--out <path>] \
  [--include-pack] \
  [--max-chars <n>] \
  [--max-snippets <n>] \
  [--top-n-files <n>]
```

#### `verify`

Run incremental verification.

```bash
accel verify \
  [--changed-files <files>] \
  [--evidence-run] \
  [--fast-loop] \
  [--workers <n>] \
  [--fail-fast]
```

#### `explain`

Explain context ranking decisions and near-miss files.

```bash
accel explain \
  --task "Refactor auth session cache" \
  --changed-files accel/query/context_compiler.py tests/unit/test_context_compiler.py \
  --top-n-files 12 \
  --alternatives 8 \
  --output json
```

#### `doctor`

Run diagnostics and health checks.

```bash
accel doctor
```

---

## 🤝 Integration with AI Agents

### AGENTS.md Compliance

agent-accel complements the AGENTS.md specification:

| AGENTS.md Requirement | agent-accel Contribution |
|----------------------|--------------------------|
| Evidence Gate | ✅ Verification logs with nonce |
| UTF-8 Enforcement | ✅ All file I/O uses `encoding="utf-8"` |
| Append-Only Truth | ✅ Events written in append mode |
| Scope Budget | ✅ Hard limits (`max_chars`, `top_n_files`) |
| Defense-in-Depth | ✅ Config validation + state checks |
| Blueprint/Pre-Snapshot | ⚠️ Handled by AI agent |

### Pre/Post Hooks

Integrate with your agent workflow:

```bash
# Pre-execution: Update indexes
accel-pre-hook

# Your agent executes the task...

# Post-execution: Run verification
accel-post-hook
```

---

## 📊 Benchmark Harness

`agent-accel` includes a reproducible benchmark runner to validate token savings, context quality, and optional verification outcomes.

### Inputs

- Task suite: `examples/benchmarks/tasks.sample.json`
- Runner: `scripts/run_benchmarks.py`
- Task fields: `id`, `task`, `changed_files`, `hints`, `expected_top_files`

### Run

```bash
python scripts/run_benchmarks.py \
  --project . \
  --tasks examples/benchmarks/tasks.sample.json \
  --index-mode update \
  --out examples/benchmarks/results_local.json \
  --out-md examples/benchmarks/results_local.md
```

Optional verify metrics:

```bash
python scripts/run_benchmarks.py \
  --project . \
  --tasks examples/benchmarks/tasks.sample.json \
  --index-mode update \
  --run-verify \
  --out examples/benchmarks/results_with_verify.json \
  --out-md examples/benchmarks/results_with_verify.md
```

### Metrics

- Token metrics:
  - `context_tokens`
  - `token_reduction_vs_full_index`
  - `token_reduction_vs_changed_files`
- Quality metric:
  - `top_file_recall`
- Latency metrics:
  - `context_build_seconds`
  - `verify_seconds` (if `--run-verify`)

The repository also ships a manual GitHub Actions workflow at `.github/workflows/benchmark-harness.yml` that runs the same harness and uploads JSON/Markdown artifacts.

Benchmark details are documented in `examples/benchmarks/README.md`.

---

## 🧩 Schema Contracts & Versioning

Formal JSON schemas are shipped in `accel/schema/` and enforced by `accel/schema/contracts.py`:

- `context_pack.schema.json`
- `index_manifest.schema.json`
- `mcp_context_response.schema.json`
- `mcp_verify_events.schema.json`

Contract enforcement modes:

- `off`: skip contract checks
- `warn`: repair and return warnings
- `strict`: fail fast on schema violations

Versioning and breaking-change policy is documented in `docs/schema_versioning.md`.

Release notes are tracked in `CHANGELOG.md`.

---

## 🧪 Development

### Running Tests

```bash
# Run all tests
python -m pytest

# Run with coverage
python -m pytest --cov=accel

# Run specific test
python -m pytest tests/test_indexers.py -v
```

### Code Quality

```bash
# Format code
python -m ruff format .

# Lint
python -m ruff check .

# Type check
python -m mypy accel/
```

---

## 📄 License

MIT License - see [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- Built for [HarborPilot](https://github.com/harborpilot) AI agent framework
- Uses [FastMCP](https://github.com/jlowin/fastmcp) for MCP server implementation
- Inspired by code intelligence tools in the AI coding ecosystem

---

<p align="center">
  <sub>Built with ❤️ for AI coding agents</sub>
</p>
