from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


@dataclass(frozen=True)
class MemoryItem:
    id: int
    created_at: str
    kind: str
    tags: List[str]
    content: str
    meta: Dict[str, Any]
    run_id: Optional[str]


class MemoryStore:
    """
    Long-term memory (local SQLite).

    - Append-only semantics for core fields (we support updates, but prefer add-only).
    - Optional FTS (if available) for fast search.
    - Auto-indexes into NexusVectorStore for semantic search (if vector store attached).
    """

    def __init__(self, db_path: Path, vector_store=None) -> None:
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.db_path))
        self._conn.row_factory = sqlite3.Row
        self._init_schema()
        self._vector_store = vector_store  # Optional NexusVectorStore

    def close(self) -> None:
        try:
            self._conn.close()
        except Exception:
            pass

    def add(
        self,
        *,
        created_at: str,
        kind: str,
        tags: Optional[Sequence[str]] = None,
        content: str,
        meta: Optional[Dict[str, Any]] = None,
        run_id: Optional[str] = None,
    ) -> int:
        tags_list = sorted({t.strip() for t in (tags or []) if t and t.strip()})
        meta_obj = meta or {}

        cur = self._conn.cursor()
        cur.execute(
            "INSERT INTO memory_items(created_at, kind, tags_json, content, meta_json, run_id) VALUES(?,?,?,?,?,?)",
            (created_at, kind, json.dumps(tags_list, sort_keys=True), content, json.dumps(meta_obj, sort_keys=True), run_id),
        )
        item_id = int(cur.lastrowid)

        if self._fts_enabled():
            cur.execute(
                "INSERT INTO memory_items_fts(rowid, content) VALUES(?,?)",
                (item_id, content),
            )
        self._conn.commit()

        # Auto-index into vector store for semantic search
        if self._vector_store is not None:
            try:
                doc_id = f"memory:{item_id}"
                index_text = f"{kind} {' '.join(tags_list)} {content}"
                self._vector_store.add(doc_id, index_text, {
                    "kind": kind,
                    "tags": tags_list,
                    "created_at": created_at,
                    "run_id": run_id,
                })
            except Exception:
                pass  # Vector indexing is best-effort

        return item_id

    def recent(self, *, kind: Optional[str] = None, limit: int = 20) -> List[MemoryItem]:
        limit = max(1, min(int(limit), 200))
        cur = self._conn.cursor()
        if kind:
            rows = cur.execute(
                "SELECT id, created_at, kind, tags_json, content, meta_json, run_id FROM memory_items WHERE kind=? ORDER BY id DESC LIMIT ?",
                (kind, limit),
            ).fetchall()
        else:
            rows = cur.execute(
                "SELECT id, created_at, kind, tags_json, content, meta_json, run_id FROM memory_items ORDER BY id DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [self._row_to_item(r) for r in rows]

    def search(self, *, query: str, kind: Optional[str] = None, limit: int = 20) -> List[MemoryItem]:
        limit = max(1, min(int(limit), 200))
        q = (query or "").strip()
        if not q:
            return []

        cur = self._conn.cursor()
        if self._fts_enabled():
            if kind:
                rows = cur.execute(
                    """
                    SELECT m.id, m.created_at, m.kind, m.tags_json, m.content, m.meta_json, m.run_id
                    FROM memory_items_fts f
                    JOIN memory_items m ON m.id = f.rowid
                    WHERE f.content MATCH ? AND m.kind=?
                    ORDER BY m.id DESC
                    LIMIT ?
                    """,
                    (q, kind, limit),
                ).fetchall()
            else:
                rows = cur.execute(
                    """
                    SELECT m.id, m.created_at, m.kind, m.tags_json, m.content, m.meta_json, m.run_id
                    FROM memory_items_fts f
                    JOIN memory_items m ON m.id = f.rowid
                    WHERE f.content MATCH ?
                    ORDER BY m.id DESC
                    LIMIT ?
                    """,
                    (q, limit),
                ).fetchall()
        else:
            # Fallback: LIKE search
            like = f"%{q}%"
            if kind:
                rows = cur.execute(
                    "SELECT id, created_at, kind, tags_json, content, meta_json, run_id FROM memory_items WHERE content LIKE ? AND kind=? ORDER BY id DESC LIMIT ?",
                    (like, kind, limit),
                ).fetchall()
            else:
                rows = cur.execute(
                    "SELECT id, created_at, kind, tags_json, content, meta_json, run_id FROM memory_items WHERE content LIKE ? ORDER BY id DESC LIMIT ?",
                    (like, limit),
                ).fetchall()

        return [self._row_to_item(r) for r in rows]

    def stats(self) -> Dict[str, Any]:
        cur = self._conn.cursor()
        total = int(cur.execute("SELECT COUNT(1) AS c FROM memory_items").fetchone()["c"])
        by_kind = {r["kind"]: int(r["c"]) for r in cur.execute("SELECT kind, COUNT(1) AS c FROM memory_items GROUP BY kind").fetchall()}
        return {"total": total, "by_kind": by_kind, "fts": self._fts_enabled()}

    def _init_schema(self) -> None:
        cur = self._conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS memory_items(
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                kind TEXT NOT NULL,
                tags_json TEXT NOT NULL,
                content TEXT NOT NULL,
                meta_json TEXT NOT NULL,
                run_id TEXT NULL
            )
            """
        )
        # Best-effort FTS: if unavailable, we skip.
        try:
            cur.execute("CREATE VIRTUAL TABLE IF NOT EXISTS memory_items_fts USING fts5(content)")
        except Exception:
            pass
        self._conn.commit()

    def _fts_enabled(self) -> bool:
        cur = self._conn.cursor()
        try:
            cur.execute("SELECT 1 FROM memory_items_fts LIMIT 1")
            return True
        except Exception:
            return False

    def _row_to_item(self, r: sqlite3.Row) -> MemoryItem:
        tags = []
        meta = {}
        try:
            tags = json.loads(r["tags_json"]) or []
        except Exception:
            tags = []
        try:
            meta = json.loads(r["meta_json"]) or {}
        except Exception:
            meta = {}
        return MemoryItem(
            id=int(r["id"]),
            created_at=str(r["created_at"]),
            kind=str(r["kind"]),
            tags=[str(t) for t in tags],
            content=str(r["content"]),
            meta=meta,
            run_id=str(r["run_id"]) if r["run_id"] is not None else None,
        )

