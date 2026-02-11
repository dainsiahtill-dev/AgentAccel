# Architecture

`agent-accel` has three functional planes:

1. Control Plane
- CLI commands (`init`, `doctor`, `index`, `context`, `verify`)
- Hook entrypoints (`accel-pre-hook`, `accel-post-hook`)

2. Intelligence Plane
- Incremental indexing (`symbols`, `references`, `deps`, `test_ownership`)
- Context compiler (`planner`, `ranker`, `snippet_extractor`)

3. Execution Plane
- Verification orchestrator for tests/lint/typecheck
- Incremental command selection based on changed files

Runtime state is stored under `ACCEL_HOME/projects/<project_hash>/`.
