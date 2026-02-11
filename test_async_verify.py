import time
import threading
from pathlib import Path
from accel.verify.job_manager import JobManager, JobState
from accel.verify.callbacks import VerifyProgressCallback, VerifyStage
from accel.config import resolve_effective_config

class TestCallback(VerifyProgressCallback):
    def __init__(self):
        self.events = []
        
    def on_start(self, job_id: str, total_commands: int) -> None:
        self.events.append(('start', job_id, total_commands))
        
    def on_stage_change(self, job_id: str, stage: VerifyStage) -> None:
        self.events.append(('stage_change', job_id, stage.name))
        
    def on_command_start(self, job_id: str, command: str, index: int, total: int) -> None:
        self.events.append(('command_start', job_id, command, index, total))
        
    def on_command_complete(self, job_id: str, command: str, exit_code: int, duration: float) -> None:
        self.events.append(('command_complete', job_id, command, exit_code, duration))
        
    def on_progress(self, job_id: str, completed: int, total: int, current_command: str) -> None:
        self.events.append(('progress', job_id, completed, total, current_command))
        
    def on_heartbeat(self, job_id: str, elapsed_sec: float, eta_sec: float | None, state: str) -> None:
        self.events.append(('heartbeat', job_id, elapsed_sec, eta_sec, state))
        
    def on_cache_hit(self, job_id: str, command: str) -> None:
        self.events.append(('cache_hit', job_id, command))
        
    def on_skip(self, job_id: str, command: str, reason: str) -> None:
        self.events.append(('skip', job_id, command, reason))
        
    def on_error(self, job_id: str, command: str | None, error: str) -> None:
        self.events.append(('error', job_id, command, error))
        
    def on_complete(self, job_id: str, status: str, exit_code: int) -> None:
        self.events.append(('complete', job_id, status, exit_code))


print("=" * 60)
print("Testing Async Verify Task Model Implementation")
print("=" * 60)

print("\n1. Testing JobManager singleton...")
jm1 = JobManager()
jm2 = JobManager()
assert jm1 is jm2, "JobManager should be singleton"
print("   OK: JobManager is singleton")

print("\n2. Testing job creation and lifecycle...")
job = jm1.create_job()
print(f"   Created job: {job.job_id}")
assert job.state == JobState.PENDING
print("   OK: Job created with PENDING state")

job.mark_running("test_stage")
assert job.state == JobState.RUNNING
print("   OK: Job marked RUNNING")

job.mark_completed("success", 0)
assert job.state == JobState.COMPLETED
print("   OK: Job marked COMPLETED")

print("\n3. Testing event sequence...")
job2 = jm1.create_job()
e1 = job2.add_event("test_event", {"key": "value"})
e2 = job2.add_event("test_event2", {"key": "value2"})
assert e1['seq'] == 1
assert e2['seq'] == 2
print("   OK: Event sequence works correctly")

events = job2.get_events(0)
assert len(events) == 2
print(f"   OK: Retrieved {len(events)} events")

events_since = job2.get_events(1)
assert len(events_since) == 1
print(f"   OK: Retrieved {len(events_since)} event(s) since seq=1")

print("\n4. Testing VerifyProgressCallback protocol...")
callback = TestCallback()
assert hasattr(callback, 'on_start')
assert hasattr(callback, 'on_stage_change')
assert hasattr(callback, 'on_command_start')
assert hasattr(callback, 'on_command_complete')
assert hasattr(callback, 'on_progress')
assert hasattr(callback, 'on_heartbeat')
assert hasattr(callback, 'on_complete')
print("   OK: All callback methods defined")

print("\n5. Testing VerifyStage enum...")
stages = [s.name for s in VerifyStage]
print(f"   Available stages: {stages}")
assert 'INIT' in stages
assert 'RUNNING' in stages
assert 'PARALLEL' in stages
print("   OK: All stages defined")

print("\n6. Testing job status format...")
status = job.to_status()
required_fields = ['job_id', 'state', 'stage', 'progress', 'elapsed_sec', 'eta_sec', 'current_command', 'total_commands', 'completed_commands']
for field in required_fields:
    assert field in status, f"Status should contain '{field}'"
print(f"   Status contains all required fields")
print(f"   Sample status: {status}")
print("   OK: Job status format is correct")

print("\n" + "=" * 60)
print("All Async Verify Task Model Tests Passed!")
print("=" * 60)

print("\nSummary of new MCP tools:")
print("  - accel_verify_start: Start async verification job")
print("  - accel_verify_status: Get job status (state, progress, eta)")
print("  - accel_verify_events: Get job events with sequence")
print("  - accel_verify_cancel: Cancel running job")
print("\nThese tools bypass the 60-second MCP timeout by:")
print("  1. Starting a background thread for long-running verify")
print("  2. Returning job_id immediately")
print("  3. Allowing polling for status and events")
