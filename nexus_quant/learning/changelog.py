"""
NEXUS Development Changelog — Track system evolution over time.

This module logs every significant change to the system so that:
1. The system knows HOW it has evolved
2. Changes can be reviewed for conflicts with existing code
3. Impact analysis is possible (what changed, what was affected)
4. Rollback decisions have full context

Every change gets:
- timestamp
- category (feature, fix, optimization, refactor, config_change)
- description of what changed
- files affected
- impact assessment (what components are affected)
- conflicts checked (what existing things might break)

Usage:
    from nexus_quant.learning.changelog import ChangeLog

    log = ChangeLog(artifacts_dir)
    log.record(
        category="feature",
        description="Added vector store for semantic search",
        files=["memory/vector_store.py", "memory/store.py"],
        impact=["memory pipeline", "agent context", "RAG search"],
        conflicts_checked=["existing TF-IDF RAG still works", "MemoryStore API unchanged"],
    )

    # Review recent changes
    recent = log.recent(limit=10)

    # Check if a file was recently changed (conflict detection)
    changes = log.changes_for_file("memory/store.py")
"""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


def _utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class ChangeLog:
    """
    Persistent development changelog.

    Stores in SQLite for queryability. Lives in artifacts/learning/changelog.db.
    """

    def __init__(self, artifacts_dir: Path) -> None:
        self.db_path = artifacts_dir / "learning" / "changelog.db"
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.db_path))
        self._conn.row_factory = sqlite3.Row
        self._init_schema()

    def close(self) -> None:
        try:
            self._conn.close()
        except Exception:
            pass

    def record(
        self,
        *,
        category: str,
        description: str,
        files: Optional[List[str]] = None,
        impact: Optional[List[str]] = None,
        conflicts_checked: Optional[List[str]] = None,
        version: Optional[str] = None,
        source: str = "system",
    ) -> int:
        """Record a development change."""
        now = _utc_iso()
        cur = self._conn.cursor()
        cur.execute(
            """INSERT INTO changelog
               (created_at, category, description, files_json, impact_json, conflicts_json, version, source)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                now,
                category,
                description,
                json.dumps(files or [], sort_keys=True),
                json.dumps(impact or [], sort_keys=True),
                json.dumps(conflicts_checked or [], sort_keys=True),
                version,
                source,
            ),
        )
        cid = int(cur.lastrowid)
        self._conn.commit()
        return cid

    def recent(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Get recent changes."""
        cur = self._conn.cursor()
        rows = cur.execute(
            "SELECT * FROM changelog ORDER BY id DESC LIMIT ?",
            (min(limit, 200),),
        ).fetchall()
        return [self._row_to_dict(r) for r in rows]

    def changes_for_file(self, file_path: str) -> List[Dict[str, Any]]:
        """Find all changes that affected a specific file."""
        cur = self._conn.cursor()
        like = f"%{file_path}%"
        rows = cur.execute(
            "SELECT * FROM changelog WHERE files_json LIKE ? ORDER BY id DESC LIMIT 50",
            (like,),
        ).fetchall()
        return [self._row_to_dict(r) for r in rows]

    def changes_by_category(self, category: str) -> List[Dict[str, Any]]:
        """Get all changes of a specific category."""
        cur = self._conn.cursor()
        rows = cur.execute(
            "SELECT * FROM changelog WHERE category=? ORDER BY id DESC LIMIT 50",
            (category,),
        ).fetchall()
        return [self._row_to_dict(r) for r in rows]

    def summary(self) -> Dict[str, Any]:
        """Get changelog summary statistics."""
        cur = self._conn.cursor()
        total = int(cur.execute("SELECT COUNT(1) FROM changelog").fetchone()[0])
        by_category = {}
        for r in cur.execute("SELECT category, COUNT(1) AS c FROM changelog GROUP BY category").fetchall():
            by_category[str(r["category"])] = int(r["c"])

        # Most recently changed files
        recent_files = set()
        recent = cur.execute("SELECT files_json FROM changelog ORDER BY id DESC LIMIT 10").fetchall()
        for r in recent:
            try:
                files = json.loads(r["files_json"] or "[]")
                recent_files.update(files)
            except Exception:
                pass

        return {
            "total_changes": total,
            "by_category": by_category,
            "recently_touched_files": sorted(recent_files),
        }

    def _row_to_dict(self, r: sqlite3.Row) -> Dict[str, Any]:
        return {
            "id": int(r["id"]),
            "created_at": str(r["created_at"]),
            "category": str(r["category"]),
            "description": str(r["description"]),
            "files": json.loads(r["files_json"] or "[]"),
            "impact": json.loads(r["impact_json"] or "[]"),
            "conflicts_checked": json.loads(r["conflicts_json"] or "[]"),
            "version": r["version"],
            "source": str(r["source"]),
        }

    def _init_schema(self) -> None:
        cur = self._conn.cursor()
        cur.execute("""
            CREATE TABLE IF NOT EXISTS changelog (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                category TEXT NOT NULL,
                description TEXT NOT NULL,
                files_json TEXT NOT NULL DEFAULT '[]',
                impact_json TEXT NOT NULL DEFAULT '[]',
                conflicts_json TEXT NOT NULL DEFAULT '[]',
                version TEXT NULL,
                source TEXT NOT NULL DEFAULT 'system'
            )
        """)
        cur.execute("CREATE INDEX IF NOT EXISTS idx_changelog_cat ON changelog(category)")
        self._conn.commit()


def log_system_change(
    artifacts_dir: Path,
    *,
    category: str,
    description: str,
    files: Optional[List[str]] = None,
    impact: Optional[List[str]] = None,
    conflicts_checked: Optional[List[str]] = None,
    version: Optional[str] = None,
) -> int:
    """
    Convenience function to log a change without managing the ChangeLog lifecycle.

    Usage:
        from nexus_quant.learning.changelog import log_system_change
        log_system_change(
            artifacts_dir,
            category="feature",
            description="Added XYZ",
            files=["foo.py"],
        )
    """
    cl = ChangeLog(artifacts_dir)
    try:
        return cl.record(
            category=category,
            description=description,
            files=files,
            impact=impact,
            conflicts_checked=conflicts_checked,
            version=version,
        )
    finally:
        cl.close()
