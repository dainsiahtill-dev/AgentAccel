from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def append_event(
    events_path: Path,
    event_type: str,
    summary: str,
    status: str = "ok",
    extra: dict[str, Any] | None = None,
) -> None:
    events_path.parent.mkdir(parents=True, exist_ok=True)
    event: dict[str, Any] = {
        "ts": utc_now(),
        "type": event_type,
        "status": status,
        "summary": summary,
    }
    if extra:
        event.update(extra)
    with events_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event, ensure_ascii=False) + "\n")
