from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _utc_text(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat()


def _parse_utc(value: str) -> datetime | None:
    text = str(value or "").strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _normalize_path_token(value: str) -> str:
    return str(value or "").replace("\\", "/").strip().lower()


def normalize_token_list(values: list[str] | None) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for item in (values or []):
        token = str(item or "").strip().lower()
        if not token:
            continue
        if token in seen:
            continue
        seen.add(token)
        out.append(token)
    return out


def normalize_changed_files(values: list[str] | None) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for item in (values or []):
        token = _normalize_path_token(item)
        if not token:
            continue
        if token in seen:
            continue
        seen.add(token)
        out.append(token)
    return out


def make_stable_hash(payload: dict[str, Any]) -> str:
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def context_changed_fingerprint(changed_files: list[str]) -> str:
    return make_stable_hash({"changed_files": normalize_changed_files(changed_files)})


def task_signature(task_tokens: list[str], hint_tokens: list[str]) -> str:
    return make_stable_hash(
        {
            "task_tokens": normalize_token_list(task_tokens),
            "hint_tokens": normalize_token_list(hint_tokens),
        }
    )


def jaccard_similarity(left: set[str], right: set[str]) -> float:
    if not left and not right:
        return 1.0
    if not left or not right:
        return 0.0
    inter = len(left.intersection(right))
    union = len(left.union(right))
    if union <= 0:
        return 0.0
    return float(inter) / float(union)


class SemanticCacheStore:
    def __init__(self, db_path: Path) -> None:
        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path, timeout=10, isolation_level=None)
        conn.execute("PRAGMA busy_timeout = 5000")
        conn.execute("PRAGMA journal_mode = WAL")
        conn.execute("PRAGMA synchronous = NORMAL")
        return conn

    def _init_schema(self) -> None:
        with self._lock:
            conn = self._connect()
            try:
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS context_cache (
                      cache_key TEXT PRIMARY KEY,
                      task_signature TEXT NOT NULL,
                      task_tokens_json TEXT NOT NULL,
                      hint_tokens_json TEXT NOT NULL,
                      changed_files_json TEXT NOT NULL,
                      changed_fingerprint TEXT NOT NULL,
                      budget_fingerprint TEXT NOT NULL,
                      config_hash TEXT NOT NULL,
                      payload_json TEXT NOT NULL,
                      created_utc TEXT NOT NULL,
                      expires_utc TEXT NOT NULL
                    )
                    """
                )
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS verify_plan_cache (
                      cache_key TEXT PRIMARY KEY,
                      changed_fingerprint TEXT NOT NULL,
                      runtime_fingerprint TEXT NOT NULL,
                      config_hash TEXT NOT NULL,
                      commands_json TEXT NOT NULL,
                      created_utc TEXT NOT NULL,
                      expires_utc TEXT NOT NULL
                    )
                    """
                )
                conn.execute("CREATE INDEX IF NOT EXISTS idx_context_expiry ON context_cache(expires_utc)")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_verify_plan_expiry ON verify_plan_cache(expires_utc)")
            finally:
                conn.close()

    def _prune_table(self, conn: sqlite3.Connection, table_name: str, max_entries: int) -> None:
        now_text = _utc_text(_utc_now())
        conn.execute(f"DELETE FROM {table_name} WHERE expires_utc <= ?", (now_text,))
        max_keep = max(1, int(max_entries))
        count_row = conn.execute(f"SELECT COUNT(1) FROM {table_name}").fetchone()
        count = int(count_row[0]) if count_row else 0
        if count <= max_keep:
            return
        overflow = count - max_keep
        conn.execute(
            f"""
            DELETE FROM {table_name}
            WHERE cache_key IN (
              SELECT cache_key FROM {table_name}
              ORDER BY created_utc ASC
              LIMIT ?
            )
            """,
            (overflow,),
        )

    def get_context_exact(self, cache_key: str) -> dict[str, Any] | None:
        with self._lock:
            conn = self._connect()
            try:
                row = conn.execute(
                    """
                    SELECT payload_json, expires_utc
                    FROM context_cache
                    WHERE cache_key = ?
                    """,
                    (cache_key,),
                ).fetchone()
                if row is None:
                    return None
                expires = _parse_utc(str(row[1]))
                if expires is None or expires <= _utc_now():
                    conn.execute("DELETE FROM context_cache WHERE cache_key = ?", (cache_key,))
                    return None
                payload = json.loads(str(row[0]))
                if not isinstance(payload, dict):
                    return None
                return payload
            finally:
                conn.close()

    def get_context_hybrid(
        self,
        *,
        task_tokens: list[str],
        hint_tokens: list[str],
        changed_files: list[str],
        budget_fingerprint: str,
        config_hash: str,
        threshold: float,
        max_candidates: int = 80,
    ) -> tuple[dict[str, Any] | None, float]:
        task_set = set(normalize_token_list(task_tokens + hint_tokens))
        changed_set = set(normalize_changed_files(changed_files))
        min_threshold = max(0.0, min(1.0, float(threshold)))
        best_payload: dict[str, Any] | None = None
        best_score = 0.0

        with self._lock:
            conn = self._connect()
            try:
                rows = conn.execute(
                    """
                    SELECT payload_json, task_tokens_json, hint_tokens_json, changed_files_json
                    FROM context_cache
                    WHERE budget_fingerprint = ?
                      AND config_hash = ?
                      AND expires_utc > ?
                    ORDER BY created_utc DESC
                    LIMIT ?
                    """,
                    (
                        str(budget_fingerprint),
                        str(config_hash),
                        _utc_text(_utc_now()),
                        max(1, int(max_candidates)),
                    ),
                ).fetchall()
            finally:
                conn.close()

        for row in rows:
            try:
                payload = json.loads(str(row[0]))
                row_task_tokens = set(normalize_token_list(json.loads(str(row[1]))))
                row_hint_tokens = set(normalize_token_list(json.loads(str(row[2]))))
                row_changed_files = set(normalize_changed_files(json.loads(str(row[3]))))
            except (json.JSONDecodeError, TypeError, ValueError):
                continue
            if not isinstance(payload, dict):
                continue

            row_task_set = row_task_tokens.union(row_hint_tokens)
            task_score = jaccard_similarity(task_set, row_task_set)
            changed_score = jaccard_similarity(changed_set, row_changed_files)

            if changed_set or row_changed_files:
                score = (0.7 * task_score) + (0.3 * changed_score)
            else:
                score = (0.9 * task_score) + 0.1
            score = max(0.0, min(1.0, score))
            if score >= min_threshold and score > best_score:
                best_score = score
                best_payload = payload

        return best_payload, float(best_score)

    def put_context(
        self,
        *,
        cache_key: str,
        task_signature_value: str,
        task_tokens: list[str],
        hint_tokens: list[str],
        changed_files: list[str],
        changed_fingerprint: str,
        budget_fingerprint: str,
        config_hash: str,
        payload: dict[str, Any],
        ttl_seconds: int,
        max_entries: int,
    ) -> None:
        created = _utc_now()
        expires = created + timedelta(seconds=max(1, int(ttl_seconds)))
        with self._lock:
            conn = self._connect()
            try:
                conn.execute(
                    """
                    INSERT INTO context_cache (
                      cache_key,
                      task_signature,
                      task_tokens_json,
                      hint_tokens_json,
                      changed_files_json,
                      changed_fingerprint,
                      budget_fingerprint,
                      config_hash,
                      payload_json,
                      created_utc,
                      expires_utc
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(cache_key) DO UPDATE SET
                      task_signature=excluded.task_signature,
                      task_tokens_json=excluded.task_tokens_json,
                      hint_tokens_json=excluded.hint_tokens_json,
                      changed_files_json=excluded.changed_files_json,
                      changed_fingerprint=excluded.changed_fingerprint,
                      budget_fingerprint=excluded.budget_fingerprint,
                      config_hash=excluded.config_hash,
                      payload_json=excluded.payload_json,
                      created_utc=excluded.created_utc,
                      expires_utc=excluded.expires_utc
                    """,
                    (
                        str(cache_key),
                        str(task_signature_value),
                        json.dumps(normalize_token_list(task_tokens), ensure_ascii=False),
                        json.dumps(normalize_token_list(hint_tokens), ensure_ascii=False),
                        json.dumps(normalize_changed_files(changed_files), ensure_ascii=False),
                        str(changed_fingerprint),
                        str(budget_fingerprint),
                        str(config_hash),
                        json.dumps(payload, ensure_ascii=False),
                        _utc_text(created),
                        _utc_text(expires),
                    ),
                )
                self._prune_table(conn, "context_cache", max_entries=max_entries)
            finally:
                conn.close()

    def get_verify_plan(self, cache_key: str) -> list[str] | None:
        with self._lock:
            conn = self._connect()
            try:
                row = conn.execute(
                    """
                    SELECT commands_json, expires_utc
                    FROM verify_plan_cache
                    WHERE cache_key = ?
                    """,
                    (cache_key,),
                ).fetchone()
                if row is None:
                    return None
                expires = _parse_utc(str(row[1]))
                if expires is None or expires <= _utc_now():
                    conn.execute("DELETE FROM verify_plan_cache WHERE cache_key = ?", (cache_key,))
                    return None
                payload = json.loads(str(row[0]))
                if not isinstance(payload, list):
                    return None
                return [str(item) for item in payload if str(item).strip()]
            finally:
                conn.close()

    def put_verify_plan(
        self,
        *,
        cache_key: str,
        changed_fingerprint: str,
        runtime_fingerprint: str,
        config_hash: str,
        commands: list[str],
        ttl_seconds: int,
        max_entries: int,
    ) -> None:
        created = _utc_now()
        expires = created + timedelta(seconds=max(1, int(ttl_seconds)))
        normalized_commands = [str(item) for item in commands if str(item).strip()]

        with self._lock:
            conn = self._connect()
            try:
                conn.execute(
                    """
                    INSERT INTO verify_plan_cache (
                      cache_key,
                      changed_fingerprint,
                      runtime_fingerprint,
                      config_hash,
                      commands_json,
                      created_utc,
                      expires_utc
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(cache_key) DO UPDATE SET
                      changed_fingerprint=excluded.changed_fingerprint,
                      runtime_fingerprint=excluded.runtime_fingerprint,
                      config_hash=excluded.config_hash,
                      commands_json=excluded.commands_json,
                      created_utc=excluded.created_utc,
                      expires_utc=excluded.expires_utc
                    """,
                    (
                        str(cache_key),
                        str(changed_fingerprint),
                        str(runtime_fingerprint),
                        str(config_hash),
                        json.dumps(normalized_commands, ensure_ascii=False),
                        _utc_text(created),
                        _utc_text(expires),
                    ),
                )
                self._prune_table(conn, "verify_plan_cache", max_entries=max_entries)
            finally:
                conn.close()
