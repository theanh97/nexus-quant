from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


def _utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True)
class Task:
    id: int
    created_at: str
    updated_at: str
    kind: str
    status: str
    payload: Dict[str, Any]
    result: Dict[str, Any]
    error: Optional[str]
    retry_count: int = 0


class TaskStore:
    """
    Simple task bus for Orion-style orchestration.

    This is intentionally small and local-first:
    - tasks are persisted (SQLite)
    - status transitions are explicit
    - everything is auditable
    """

    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.db_path))
        self._conn.row_factory = sqlite3.Row
        self._init_schema()

    def close(self) -> None:
        try:
            self._conn.close()
        except Exception:
            pass

    def create(self, *, kind: str, payload: Dict[str, Any]) -> int:
        now = _utc_iso()
        cur = self._conn.cursor()
        cur.execute(
            "INSERT INTO tasks(created_at, updated_at, kind, status, payload_json, result_json, error) VALUES(?,?,?,?,?,?,?)",
            (now, now, kind, "pending", json.dumps(payload, sort_keys=True), "{}", None),
        )
        tid = int(cur.lastrowid)
        self._conn.commit()
        return tid

    def get(self, task_id: int) -> Task:
        cur = self._conn.cursor()
        r = cur.execute("SELECT * FROM tasks WHERE id=?", (int(task_id),)).fetchone()
        if not r:
            raise KeyError(f"Task not found: {task_id}")
        return self._row_to_task(r)

    def claim_next(self) -> Optional[Task]:
        cur = self._conn.cursor()
        r = cur.execute("SELECT * FROM tasks WHERE status='pending' ORDER BY id ASC LIMIT 1").fetchone()
        if not r:
            return None
        task_id = int(r["id"])
        now = _utc_iso()
        cur.execute("UPDATE tasks SET status='running', updated_at=? WHERE id=? AND status='pending'", (now, task_id))
        self._conn.commit()
        return self.get(task_id)

    def mark_done(self, task_id: int, result: Dict[str, Any]) -> None:
        now = _utc_iso()
        cur = self._conn.cursor()
        cur.execute(
            "UPDATE tasks SET status='done', updated_at=?, result_json=?, error=NULL WHERE id=?",
            (now, json.dumps(result, sort_keys=True), int(task_id)),
        )
        self._conn.commit()

    def mark_failed(self, task_id: int, error: str) -> None:
        now = _utc_iso()
        cur = self._conn.cursor()
        cur.execute(
            "UPDATE tasks SET status='failed', updated_at=?, error=? WHERE id=?",
            (now, str(error)[:4000], int(task_id)),
        )
        self._conn.commit()

    def recent(self, limit: int = 20) -> List[Task]:
        limit = max(1, min(int(limit), 200))
        cur = self._conn.cursor()
        rows = cur.execute("SELECT * FROM tasks ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
        return [self._row_to_task(r) for r in rows]

    def counts(self) -> Dict[str, int]:
        cur = self._conn.cursor()
        rows = cur.execute("SELECT status, COUNT(1) AS c FROM tasks GROUP BY status").fetchall()
        return {str(r["status"]): int(r["c"]) for r in rows}

    def requeue_for_retry(self, task_id: int, max_retries: int = 1) -> bool:
        """Requeue a failed task for retry. Returns True if requeued, False if max retries exceeded."""
        cur = self._conn.cursor()
        row = cur.execute("SELECT retry_count FROM tasks WHERE id=?", (int(task_id),)).fetchone()
        if not row:
            return False
        current_retries = int(row["retry_count"] or 0)
        if current_retries >= max_retries:
            return False
        now = _utc_iso()
        cur.execute(
            "UPDATE tasks SET status='pending', updated_at=?, error=NULL, retry_count=? WHERE id=?",
            (now, current_retries + 1, int(task_id)),
        )
        self._conn.commit()
        return True

    def _init_schema(self) -> None:
        cur = self._conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS tasks(
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                kind TEXT NOT NULL,
                status TEXT NOT NULL,
                payload_json TEXT NOT NULL,
                result_json TEXT NOT NULL,
                error TEXT NULL,
                retry_count INTEGER NOT NULL DEFAULT 0
            )
            """
        )
        # Migrate: add retry_count column if missing (existing DBs)
        try:
            cur.execute("SELECT retry_count FROM tasks LIMIT 1")
        except Exception:
            try:
                cur.execute("ALTER TABLE tasks ADD COLUMN retry_count INTEGER NOT NULL DEFAULT 0")
            except Exception:
                pass
        self._conn.commit()

    def _row_to_task(self, r: sqlite3.Row) -> Task:
        try:
            payload = json.loads(r["payload_json"]) or {}
        except Exception:
            payload = {}
        try:
            result = json.loads(r["result_json"]) or {}
        except Exception:
            result = {}
        try:
            retry_count = int(r["retry_count"] or 0)
        except Exception:
            retry_count = 0
        return Task(
            id=int(r["id"]),
            created_at=str(r["created_at"]),
            updated_at=str(r["updated_at"]),
            kind=str(r["kind"]),
            status=str(r["status"]),
            payload=payload,
            result=result,
            error=str(r["error"]) if r["error"] is not None else None,
            retry_count=retry_count,
        )
