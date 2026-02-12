import asyncio
import json
import time
from pathlib import Path

from accel.mcp_server import create_server

OUT = Path('.harborpilot/logs/mcp_regression_phase1_semantic_cache_rule_constraint_20260212_02.json')

async def main() -> None:
    project = Path(r"x:\Git\Harborpilot")
    tools = await create_server().get_tools()

    result = {}

    idx = tools["accel_index_update"].fn(project=str(project))
    result["index_update"] = {
        "status": idx.get("status"),
        "files": idx.get("manifest", {}).get("counts", {}).get("files"),
    }

    ctx1 = tools["accel_context"].fn(
        project=str(project),
        task="phase1 regression context call",
        budget="small",
        include_pack=False,
        constraint_mode="warn",
    )
    ctx2 = tools["accel_context"].fn(
        project=str(project),
        task="phase1 regression context call",
        budget="small",
        include_pack=False,
        constraint_mode="warn",
    )
    result["context"] = {
        "status": ctx1.get("status"),
        "token_reduction_ratio": ctx1.get("token_reduction_ratio"),
        "cache_first_hit": ctx1.get("semantic_cache_hit"),
        "cache_second_hit": ctx2.get("semantic_cache_hit"),
        "compression_saved_chars": ctx1.get("compression_saved_chars"),
        "constraint_repair_count": ctx1.get("constraint_repair_count"),
    }

    start = tools["accel_verify_start"].fn(
        project=str(project),
        changed_files="README.md",
        fast_loop=True,
        evidence_run=False,
        verify_workers=1,
    )
    job_id = start.get("job_id")
    result["verify_start"] = {"status": start.get("status"), "job_id": job_id}

    status_seen = []
    for _ in range(10):
        st = tools["accel_verify_status"].fn(job_id=job_id)
        state = st.get("state")
        status_seen.append(state)
        if state in {"completed", "failed", "cancelled"}:
            break
        time.sleep(0.2)

    cancel = tools["accel_verify_cancel"].fn(job_id=job_id)
    final_state = None
    for _ in range(20):
        st = tools["accel_verify_status"].fn(job_id=job_id)
        final_state = st.get("state")
        if final_state in {"completed", "failed", "cancelled"}:
            break
        time.sleep(0.2)

    events = tools["accel_verify_events"].fn(job_id=job_id, include_summary=True, max_events=200)
    summary = events.get("summary") or {}
    result["verify"] = {
        "status_seen": status_seen,
        "cancel_status": cancel.get("status"),
        "final_state": final_state,
        "events_latest_state": summary.get("latest_state"),
        "events_state_source": summary.get("state_source"),
        "events_count": len(events.get("events") or []),
    }

    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(result, ensure_ascii=False))

asyncio.run(main())
