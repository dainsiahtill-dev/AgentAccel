# Blueprint: Async Verify Task Model (S2 Standard)

**Created**: 2026-02-12
**Mode**: S2 Standard
**Approval**: Explicit

## 1. Objective

解决 MCP 调用层 60 秒超时限制，通过异步任务模型让 `accel_verify` 支持：
- 长时间运行任务的启动与后台执行
- 实时状态查询 (stage, progress, elapsed, eta)
- 事件流订阅 (events with sequence)
- 任务取消 (cancellation)

## 2. Contract Snapshot (Immutable)

**Goal**: 把 `accel_verify` 改为异步任务模型，支持进度反馈和事件流。

**Acceptance Criteria**:
1. `accel_verify_start(...)` → 返回 `{ job_id }`
2. `accel_verify_status(job_id)` → 返回 `{ state, stage, progress, elapsed_sec, eta_sec, current_command }`
3. `accel_verify_events(job_id, since_seq)` → 返回 `{ events[] }`
4. `accel_verify_cancel(job_id)` → 返回确认
5. 统一的事件协议，阶段标准化
6. 兼容现有同步 `accel_verify` 调用

## 3. Current State (Observed)

### 3.1 Orchestrator (`accel/verify/orchestrator.py`)
- `run_verify()` 同步执行，无回调机制
- 已有 `_append_jsonl()` 用于事件记录
- 支持 cache、fail_fast、parallel execution

### 3.2 MCP Server (`accel/mcp_server.py`)
- `_with_timeout()` 装饰器限制 600s
- 无任务状态追踪
- FastMCP 工具装饰器模式

### 3.3 Capability Matrix
| Capability | Status | Notes |
|------------|--------|-------|
| Repo Read | Yes | Full access |
| Repo Write | Yes | Can modify source |
| Terminal | Yes | PowerShell7+ |
| Network | No | No external calls needed |

## 4. Scope (Touch Points)

### 4.1 Files Modified
1. `accel/verify/orchestrator.py` - 添加进度回调机制
2. `accel/mcp_server.py` - 添加 JobManager 和异步工具
3. `accel/verify/callbacks.py` - **新建** 进度回调接口定义

### 4.2 Not-in-Scope
- 修改其他 MCP 工具 (`accel_index_*`, `accel_context`)
- 修改 cache 逻辑或 sharding 逻辑
- Web UI 或 HTTP 服务器

## 5. Interfaces First

### 5.1 Progress Callback Protocol
```python
# accel/verify/callbacks.py (NEW)
from typing import Protocol, Any
from enum import Enum, auto

class VerifyStage(Enum):
    INIT = auto()           # 初始化
    LOAD_CACHE = auto()      # 加载缓存
    SELECT_CMDS = auto()    # 选择命令
    RUNNING = auto()         # 执行中
    PARALLEL = auto()        # 并行执行
    SEQUENTIAL = auto()      # 顺序执行 (fallback)
    COMPLETING = auto()      # 完成中
    CLEANUP = auto()        # 清理缓存

class VerifyProgressCallback(Protocol):
    def on_start(self, job_id: str, total_commands: int) -> None:
        """验证开始"""

    def on_stage_change(self, job_id: str, stage: VerifyStage) -> None:
        """阶段变更"""

    def on_command_start(self, job_id: str, command: str, index: int, total: int) -> None:
        """单个命令开始执行"""

    def on_command_complete(self, job_id: str, command: str, exit_code: int, duration: float) -> None:
        """单个命令完成"""

    def on_progress(self, job_id: str, completed: int, total: int, current_command: str) -> None:
        """进度更新 (每完成一个命令触发)"""

    def on_heartbeat(self, job_id: str, elapsed_sec: float, eta_sec: float | None, state: str) -> None:
        """心跳 (每 10 秒触发)"""

    def on_cache_hit(self, job_id: str, command: str) -> None:
        """缓存命中"""

    def on_skip(self, job_id: str, command: str, reason: str) -> None:
        """命令跳过"""

    def on_error(self, job_id: str, command: str | None, error: str) -> None:
        """错误发生"""

    def on_complete(self, job_id: str, status: str, exit_code: int) -> None:
        """验证完成"""
```

### 5.2 Job Manager Interface
```python
# accel/verify/job_manager.py (NEW)
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any
from enum import Enum

class JobState(Enum):
    PENDING = "pending"      # 等待执行
    RUNNING = "running"      # 执行中
    CANCELLING = "cancelling" # 取消中
    COMPLETED = "completed"  # 已完成
    FAILED = "failed"        # 失败
    CANCELLED = "cancelled"  # 已取消

@dataclass
class VerifyJob:
    job_id: str
    state: JobState = JobState.PENDING
    stage: str = "init"
    progress: float = 0.0
    total_commands: int = 0
    completed_commands: int = 0
    current_command: str = ""
    elapsed_sec: float = 0.0
    eta_sec: float | None = None
    start_time: datetime | None = None
    end_time: datetime | None = None
    error: str | None = None
    result: dict[str, Any] | None = None
    events: list[dict] = field(default_factory=list)
    event_seq: int = 0

    def to_status(self) -> dict[str, Any]:
        return {
            "job_id": self.job_id,
            "state": self.state.value,
            "stage": self.stage,
            "progress": round(self.progress, 2),
            "elapsed_sec": round(self.elapsed_sec, 1),
            "eta_sec": round(self.eta_sec, 1) if self.eta_sec is not None else None,
            "current_command": self.current_command,
            "total_commands": self.total_commands,
            "completed_commands": self.completed_commands,
        }
```

### 5.3 Event Protocol (JSONL)
```json
{"seq": 1, "ts": "2026-02-12T10:00:00Z", "event": "job_started", "job_id": "abc123", "total_commands": 5}
{"seq": 2, "ts": "2026-02-12T10:00:01Z", "event": "command_start", "job_id": "abc123", "command": "ruff check .", "index": 0, "total": 5}
{"seq": 3, "ts": "2026-02-12T10:00:02Z", "event": "command_complete", "job_id": "abc123", "command": "ruff check .", "exit_code": 0, "duration": 0.5}
{"seq": 4, "ts": "2026-02-12T10:00:02Z", "event": "progress", "job_id": "abc123", "completed": 1, "total": 5, "progress_pct": 20.0}
{"seq": 5, "ts": "2026-02-12T10:00:12Z", "event": "heartbeat", "job_id": "abc123", "elapsed_sec": 12.0, "eta_sec": 48.0, "state": "running"}
```

### 5.4 MCP Tool Signatures
```python
# Async Tools (NEW)
@server.tool(name="accel_verify_start", description="Start async verification job")
def accel_verify_start(
    project: str = ".",
    changed_files: list[str] | str | None = None,
    evidence_run: bool = False,
    fast_loop: bool = False,
    verify_workers: int | None = None,
    per_command_timeout_seconds: int | None = None,
    verify_fail_fast: bool | str | None = None,
    verify_cache_enabled: bool | str | None = None,
    verify_cache_ttl_seconds: int | None = None,
    verify_cache_max_entries: int | None = None,
) -> dict[str, Any]:
    """Start an async verification job. Returns job_id immediately."""

@server.tool(name="accel_verify_status", description="Get status of verification job")
def accel_verify_status(job_id: str) -> dict[str, Any]:
    """Get current status of a verification job."""

@server.tool(name="accel_verify_events", description="Get events for verification job")
def accel_verify_events(job_id: str, since_seq: int = 0) -> dict[str, Any]:
    """Get events since given sequence number."""

@server.tool(name="accel_verify_cancel", description="Cancel verification job")
def accel_verify_cancel(job_id: str) -> dict[str, Any]:
    """Cancel a running verification job."""

# Keep existing sync tool for backward compatibility
@server.tool(name="accel_verify", description="Run verification synchronously (legacy)")
def accel_verify(...) -> dict[str, Any]:
    """Existing sync tool - delegates to async implementation."""
```

## 6. State Machine / Algorithm

### 6.1 Job Lifecycle
```
PENDING → RUNNING → COMPLETED/FAILED/CANCELLED
              ↓
         CANCELLING → CANCELLED
```

### 6.2 Progress Calculation
```
progress_pct = (completed_commands / total_commands) * 100

if total_commands > 0 and elapsed_sec > 0:
    avg_time_per_cmd = elapsed_sec / completed_commands
    remaining = total_commands - completed_commands
    eta_sec = avg_time_per_cmd * remaining
else:
    eta_sec = None
```

## 7. Failure Modes

| Mode | Handler |
|------|---------|
| Job ID not found | Return error: `{"error": "job_not_found", "job_id": "..."}` |
| Job already completed | Return final status + cached events |
| Job already cancelled | Return error: `{"error": "job_already_cancelled"}` |
| Cancel during completion | Graceful shutdown, emit `job_cancelled` event |
| MCP server restart | Jobs lost (stateless) - acceptable for MVP |
| Memory pressure | Limit job history (max 100 jobs, LRU) |

## 8. Test/Harness Plan

1. **Unit: JobManager basic operations**
   ```bash
   python -c "
   from accel.verify.job_manager import JobManager, JobState
   jm = JobManager()
   job_id = jm.create_job()
   assert jm.get_job(job_id).state == JobState.PENDING
   print('JobManager basic ops: PASS')
   "
   ```

2. **Unit: Orchestrator with callback**
   ```bash
   python -c "
   from accel.verify.callbacks import VerifyProgressCallback, VerifyStage
   from accel.verify.orchestrator import run_verify_with_callback
   # Verify callback injection works
   "
   ```

3. **Integration: Async tool flow**
   ```bash
   python -c "
   # Start job -> get status -> get events -> cancel
   from accel.mcp_server import create_server
   # Test full async flow
   "
   ```

4. **Performance: Memory leak check**
   - Run 100+ jobs, verify memory stable

## 9. Observability Plan

- Logs: Use existing `_debug_log` mechanism
- Events: JSONL file per job (`verify_job_{job_id}.jsonl`)
- Metrics: `jobs_created`, `jobs_completed`, `jobs_cancelled`, `avg_duration_sec`

## 10. Refactor Budget

**Refactor Budget**: `medium`

Reason: 需要修改两个核心模块 (orchestrator, mcp_server)，添加一个新模块 (job_manager)。但改动边界清晰，不影响现有功能（仅新增异步模式）。

## 11. Rollback Plan

### 11.1 Before Implementation
- Tag current state: `git tag rollback-before-async-verify-v1`

### 11.2 During Implementation
- Feature flags: `ASYNC_VERIFY_ENABLED` env var (default: `false`)
- Keep sync `accel_verify` unchanged, new tools behind flag

### 11.3 Rollback Steps
```bash
git checkout rollback-before-async-verify-v1 -- accel/mcp_server.py accel/verify/orchestrator.py
# Remove accel/verify/callbacks.py
# Remove accel/verify/job_manager.py
```

## 12. Status

**Status**: `Planned`
**Timestamp**: 2026-02-12T10:00:00Z

## 13. Evidence

- [x] 代码分析完成 (orchestrator.py, mcp_server.py)
- [x] 接口设计完成 (callbacks.py, job_manager.py)
- [ ] Blueprint 批准
- [ ] 实现完成
- [ ] 测试通过
- [ ] 验证通过
