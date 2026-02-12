from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path

from accel.storage.semantic_cache import (
    SemanticCacheStore,
    context_changed_fingerprint,
    make_stable_hash,
    task_signature,
)


def _expire_context_row(db_path: Path, cache_key: str) -> None:
    conn = sqlite3.connect(db_path)
    try:
        expired = (datetime.now(timezone.utc) - timedelta(seconds=10)).isoformat()
        conn.execute("UPDATE context_cache SET expires_utc = ? WHERE cache_key = ?", (expired, cache_key))
        conn.commit()
    finally:
        conn.close()


def test_context_cache_exact_and_config_sensitive(tmp_path: Path) -> None:
    db_path = tmp_path / "semantic_cache.db"
    store = SemanticCacheStore(db_path)
    task_sig = task_signature(["auth", "token"], ["login"])
    changed_fp = context_changed_fingerprint(["src/auth.py"])
    budget_fp = make_stable_hash({"budget": {"max_chars": 12000}})
    config_hash = make_stable_hash({"context": {"top_n_files": 12}})
    cache_key = make_stable_hash(
        {
            "task_signature": task_sig,
            "changed_fingerprint": changed_fp,
            "budget_fingerprint": budget_fp,
            "config_hash": config_hash,
        }
    )
    payload = {"version": 1, "task": "fix auth", "top_files": [], "snippets": [], "verify_plan": {}}

    store.put_context(
        cache_key=cache_key,
        task_signature_value=task_sig,
        task_tokens=["auth", "token"],
        hint_tokens=["login"],
        changed_files=["src/auth.py"],
        changed_fingerprint=changed_fp,
        budget_fingerprint=budget_fp,
        config_hash=config_hash,
        payload=payload,
        ttl_seconds=7200,
        max_entries=100,
    )

    exact = store.get_context_exact(cache_key)
    assert isinstance(exact, dict)
    assert str(exact.get("task", "")) == "fix auth"

    different_config_key = make_stable_hash(
        {
            "task_signature": task_sig,
            "changed_fingerprint": changed_fp,
            "budget_fingerprint": budget_fp,
            "config_hash": make_stable_hash({"context": {"top_n_files": 8}}),
        }
    )
    assert store.get_context_exact(different_config_key) is None


def test_context_cache_hybrid_similarity_and_ttl_expiry(tmp_path: Path) -> None:
    db_path = tmp_path / "semantic_cache.db"
    store = SemanticCacheStore(db_path)
    task_sig = task_signature(["mcp", "verify", "timeout"], [])
    changed_fp = context_changed_fingerprint(["accel/mcp_server.py"])
    budget_fp = make_stable_hash({"budget": {"max_chars": 10000}})
    config_hash = make_stable_hash({"context": {"top_n_files": 10}})
    cache_key = make_stable_hash(
        {
            "task_signature": task_sig,
            "changed_fingerprint": changed_fp,
            "budget_fingerprint": budget_fp,
            "config_hash": config_hash,
        }
    )
    payload = {"version": 1, "task": "verify timeout", "top_files": [], "snippets": [], "verify_plan": {}}

    store.put_context(
        cache_key=cache_key,
        task_signature_value=task_sig,
        task_tokens=["mcp", "verify", "timeout"],
        hint_tokens=[],
        changed_files=["accel/mcp_server.py"],
        changed_fingerprint=changed_fp,
        budget_fingerprint=budget_fp,
        config_hash=config_hash,
        payload=payload,
        ttl_seconds=3600,
        max_entries=100,
    )

    hybrid, similarity = store.get_context_hybrid(
        task_tokens=["verify", "mcp", "progress"],
        hint_tokens=["timeout"],
        changed_files=["accel/mcp_server.py"],
        budget_fingerprint=budget_fp,
        config_hash=config_hash,
        threshold=0.5,
    )
    assert isinstance(hybrid, dict)
    assert similarity >= 0.5

    _expire_context_row(db_path, cache_key)
    assert store.get_context_exact(cache_key) is None


def test_verify_plan_cache_roundtrip(tmp_path: Path) -> None:
    db_path = tmp_path / "semantic_cache.db"
    store = SemanticCacheStore(db_path)
    cache_key = make_stable_hash({"verify_plan": "auth"})
    commands = ["pytest -q tests/test_auth.py", "python -m mypy ."]

    store.put_verify_plan(
        cache_key=cache_key,
        changed_fingerprint=make_stable_hash({"changed_files": ["src/auth.py"]}),
        runtime_fingerprint=make_stable_hash({"verify_max_target_tests": 64}),
        config_hash=make_stable_hash({"verify": {"python": commands}}),
        commands=commands,
        ttl_seconds=900,
        max_entries=50,
    )
    cached = store.get_verify_plan(cache_key)
    assert cached == commands


def test_context_cache_hybrid_respects_safety_fingerprint(tmp_path: Path) -> None:
    db_path = tmp_path / "semantic_cache.db"
    store = SemanticCacheStore(db_path)

    task_sig = task_signature(["verify", "progress"], [])
    changed_fp = context_changed_fingerprint(["accel/mcp_server.py"])
    budget_fp = make_stable_hash({"budget": {"max_chars": 9000}})
    config_hash = make_stable_hash({"context": {"top_n_files": 8}})
    cache_key = make_stable_hash(
        {
            "task_signature": task_sig,
            "changed_fingerprint": changed_fp,
            "budget_fingerprint": budget_fp,
            "config_hash": config_hash,
            "safety_fingerprint": "safe_a",
        }
    )
    store.put_context(
        cache_key=cache_key,
        task_signature_value=task_sig,
        task_tokens=["verify", "progress"],
        hint_tokens=[],
        changed_files=["accel/mcp_server.py"],
        changed_fingerprint=changed_fp,
        budget_fingerprint=budget_fp,
        config_hash=config_hash,
        payload={"version": 1, "task": "verify progress", "top_files": [], "snippets": [], "verify_plan": {}},
        ttl_seconds=3600,
        max_entries=100,
        safety_fingerprint="safe_a",
        git_head="abc123",
        changed_files_state=[{"path": "accel/mcp_server.py", "exists": True, "mtime_ns": 1, "size": 2}],
    )

    miss_payload, miss_similarity = store.get_context_hybrid(
        task_tokens=["verify", "progress"],
        hint_tokens=[],
        changed_files=["accel/mcp_server.py"],
        budget_fingerprint=budget_fp,
        config_hash=config_hash,
        threshold=0.1,
        safety_fingerprint="safe_b",
    )
    assert miss_payload is None
    assert float(miss_similarity) == 0.0

    hit_payload, hit_similarity = store.get_context_hybrid(
        task_tokens=["verify", "progress"],
        hint_tokens=[],
        changed_files=["accel/mcp_server.py"],
        budget_fingerprint=budget_fp,
        config_hash=config_hash,
        threshold=0.1,
        safety_fingerprint="safe_a",
    )
    assert isinstance(hit_payload, dict)
    assert float(hit_similarity) >= 0.1


def test_explain_context_miss_reports_git_head_invalidation(tmp_path: Path) -> None:
    db_path = tmp_path / "semantic_cache.db"
    store = SemanticCacheStore(db_path)

    task_sig = task_signature(["cache", "safety"], [])
    changed_fp = context_changed_fingerprint(["src/a.py"])
    budget_fp = make_stable_hash({"budget": {"max_chars": 8000}})
    config_hash = make_stable_hash({"context": {"top_n_files": 6}})
    cache_key = make_stable_hash(
        {
            "task_signature": task_sig,
            "changed_fingerprint": changed_fp,
            "budget_fingerprint": budget_fp,
            "config_hash": config_hash,
            "safety_fingerprint": "safe_v1",
        }
    )

    store.put_context(
        cache_key=cache_key,
        task_signature_value=task_sig,
        task_tokens=["cache", "safety"],
        hint_tokens=[],
        changed_files=["src/a.py"],
        changed_fingerprint=changed_fp,
        budget_fingerprint=budget_fp,
        config_hash=config_hash,
        payload={"version": 1, "task": "cache safety", "top_files": [], "snippets": [], "verify_plan": {}},
        ttl_seconds=3600,
        max_entries=100,
        safety_fingerprint="safe_v1",
        git_head="commit_a",
        changed_files_state=[{"path": "src/a.py", "exists": True, "mtime_ns": 10, "size": 5}],
    )

    explanation = store.explain_context_miss(
        task_signature_value=task_sig,
        budget_fingerprint=budget_fp,
        config_hash=config_hash,
        safety_fingerprint="safe_v2",
        changed_fingerprint=changed_fp,
        git_head="commit_b",
    )
    assert explanation.get("reason") == "git_head_changed"
