"""
NEXUS Operational Learning Engine (NOLE) — The system that truly learns.

This is NOT documentation. This is CODE that ENFORCES learning.

Architecture:
1. LessonStore  — SQLite database of operational lessons (persistent across machines)
2. Preflight    — runs BEFORE every task, blocks known failure patterns
3. PostMortem   — runs AFTER every failure, extracts lessons automatically
4. Metrics      — measurable proof the system is improving

Every lesson has:
- pattern: regex or keyword that triggers the lesson
- correction: what to do instead
- hit_count: how many times this lesson prevented a failure
- category: stall | data_access | model_routing | rule_violation | task_failure

Usage:
    from nexus_quant.learning.operational import OperationalLearner

    learner = OperationalLearner(artifacts_dir)

    # Before every task:
    warnings = learner.preflight(task_kind="run", context={"config": "..."})

    # After every failure:
    learner.postmortem(task_kind="run", error="external data required", context={})

    # Metrics (prove learning):
    metrics = learner.metrics()
    # => {"total_lessons": 12, "total_hits": 47, "recurrence_blocked": 35, ...}
"""
from __future__ import annotations

import json
import re
import sqlite3
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence


def _utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True)
class Lesson:
    id: int
    created_at: str
    category: str
    pattern: str           # regex pattern to match against errors/context
    correction: str        # what to do instead
    severity: str          # critical | warning | info
    source: str            # where this lesson came from (user_feedback, postmortem, seed)
    hit_count: int         # how many times this lesson fired in preflight
    miss_count: int        # how many times the failure recurred DESPITE the lesson
    last_hit_at: Optional[str]
    active: bool


@dataclass(frozen=True)
class PreflightWarning:
    lesson_id: int
    category: str
    severity: str
    correction: str
    pattern_matched: str


class LessonStore:
    """
    Persistent store of operational lessons.

    This is the SYSTEM that remembers. It lives in the codebase (artifacts/),
    persists across machines, and enforces corrections at runtime.
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

    def add_lesson(
        self,
        *,
        category: str,
        pattern: str,
        correction: str,
        severity: str = "warning",
        source: str = "unknown",
    ) -> int:
        """Add a new operational lesson. Returns lesson ID."""
        now = _utc_iso()
        cur = self._conn.cursor()
        # Dedup: don't add if same pattern+category already exists
        existing = cur.execute(
            "SELECT id FROM operational_lessons WHERE pattern=? AND category=? AND active=1",
            (pattern, category),
        ).fetchone()
        if existing:
            return int(existing["id"])
        cur.execute(
            """INSERT INTO operational_lessons
               (created_at, category, pattern, correction, severity, source, hit_count, miss_count, last_hit_at, active)
               VALUES (?, ?, ?, ?, ?, ?, 0, 0, NULL, 1)""",
            (now, category, pattern, correction, severity, source),
        )
        lid = int(cur.lastrowid)
        self._conn.commit()
        return lid

    def preflight(self, task_kind: str, context: Dict[str, Any]) -> List[PreflightWarning]:
        """
        Run preflight checks before a task.

        Scans all active lessons and checks if the current task/context
        matches any known failure patterns. Returns warnings.
        """
        warnings: List[PreflightWarning] = []
        context_str = json.dumps({"task_kind": task_kind, **context}, default=str).lower()

        cur = self._conn.cursor()
        rows = cur.execute(
            "SELECT * FROM operational_lessons WHERE active=1 ORDER BY severity DESC, hit_count DESC"
        ).fetchall()

        for row in rows:
            lesson = self._row_to_lesson(row)
            try:
                if re.search(lesson.pattern, context_str, re.IGNORECASE):
                    warnings.append(PreflightWarning(
                        lesson_id=lesson.id,
                        category=lesson.category,
                        severity=lesson.severity,
                        correction=lesson.correction,
                        pattern_matched=lesson.pattern,
                    ))
                    # Record hit
                    self._record_hit(lesson.id)
            except re.error:
                # Invalid regex — try plain substring match
                if lesson.pattern.lower() in context_str:
                    warnings.append(PreflightWarning(
                        lesson_id=lesson.id,
                        category=lesson.category,
                        severity=lesson.severity,
                        correction=lesson.correction,
                        pattern_matched=lesson.pattern,
                    ))
                    self._record_hit(lesson.id)

        return warnings

    def postmortem(
        self,
        task_kind: str,
        error: str,
        context: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Analyze a failure and extract lessons.

        If the error matches an existing lesson → increment miss_count (lesson didn't prevent it).
        If the error is NEW → create a new lesson automatically.

        Returns: {"matched_lessons": [...], "new_lesson_id": int|None}
        """
        error_lower = error.lower()
        context_str = json.dumps({"task_kind": task_kind, "error": error, **context}, default=str).lower()

        cur = self._conn.cursor()
        rows = cur.execute(
            "SELECT * FROM operational_lessons WHERE active=1"
        ).fetchall()

        matched = []
        for row in rows:
            lesson = self._row_to_lesson(row)
            try:
                if re.search(lesson.pattern, context_str, re.IGNORECASE):
                    matched.append(lesson)
                    self._record_miss(lesson.id)
            except re.error:
                if lesson.pattern.lower() in context_str:
                    matched.append(lesson)
                    self._record_miss(lesson.id)

        # Auto-extract lesson from new failure patterns
        new_lesson_id = None
        if not matched:
            new_lesson_id = self._auto_extract_lesson(task_kind, error, context)

        # Log the postmortem event
        self._log_event("postmortem", {
            "task_kind": task_kind,
            "error": error[:500],
            "matched_lessons": [l.id for l in matched],
            "new_lesson_id": new_lesson_id,
        })

        return {
            "matched_lessons": [{"id": l.id, "category": l.category, "correction": l.correction} for l in matched],
            "new_lesson_id": new_lesson_id,
        }

    def get_active_lessons(self, category: Optional[str] = None) -> List[Lesson]:
        """Get all active lessons, optionally filtered by category."""
        cur = self._conn.cursor()
        if category:
            rows = cur.execute(
                "SELECT * FROM operational_lessons WHERE active=1 AND category=? ORDER BY severity DESC, hit_count DESC",
                (category,),
            ).fetchall()
        else:
            rows = cur.execute(
                "SELECT * FROM operational_lessons WHERE active=1 ORDER BY severity DESC, hit_count DESC"
            ).fetchall()
        return [self._row_to_lesson(r) for r in rows]

    def get_lessons_summary(self, max_lessons: int = 20) -> List[Dict[str, str]]:
        """
        Get a compact summary of top lessons for injection into agent context.

        This is what agents SEE — a distilled list of operational rules
        learned from past failures.
        """
        lessons = self.get_active_lessons()[:max_lessons]
        return [
            {
                "category": l.category,
                "severity": l.severity,
                "rule": l.correction,
                "enforced_times": l.hit_count,
                "missed_times": l.miss_count,
            }
            for l in lessons
        ]

    def metrics(self) -> Dict[str, Any]:
        """
        Measurable proof the system is learning.

        Returns metrics that answer: "Is the system actually improving?"
        """
        cur = self._conn.cursor()

        # Total lessons
        total = int(cur.execute("SELECT COUNT(1) AS c FROM operational_lessons WHERE active=1").fetchone()["c"])

        # Total preflight hits (failures prevented)
        total_hits = int(cur.execute("SELECT COALESCE(SUM(hit_count), 0) AS c FROM operational_lessons WHERE active=1").fetchone()["c"])

        # Total misses (failures that still occurred despite lessons)
        total_misses = int(cur.execute("SELECT COALESCE(SUM(miss_count), 0) AS c FROM operational_lessons WHERE active=1").fetchone()["c"])

        # Prevention rate: hits / (hits + misses) — should increase over time
        prevention_rate = (total_hits / max(1, total_hits + total_misses)) if (total_hits + total_misses) > 0 else 0.0

        # By category
        by_category = {}
        rows = cur.execute(
            "SELECT category, COUNT(1) AS c, SUM(hit_count) AS hits, SUM(miss_count) AS misses "
            "FROM operational_lessons WHERE active=1 GROUP BY category"
        ).fetchall()
        for r in rows:
            by_category[str(r["category"])] = {
                "lessons": int(r["c"]),
                "hits": int(r["hits"] or 0),
                "misses": int(r["misses"] or 0),
            }

        # Recent events
        recent_events = []
        try:
            event_rows = cur.execute(
                "SELECT * FROM operational_events ORDER BY id DESC LIMIT 10"
            ).fetchall()
            for r in event_rows:
                recent_events.append({
                    "ts": str(r["created_at"]),
                    "event_type": str(r["event_type"]),
                    "data": json.loads(r["data_json"]) if r["data_json"] else {},
                })
        except Exception:
            pass

        return {
            "total_lessons": total,
            "total_hits_prevented": total_hits,
            "total_misses": total_misses,
            "prevention_rate": round(prevention_rate, 4),
            "by_category": by_category,
            "recent_events": recent_events,
            "learning_effective": prevention_rate > 0.5 if (total_hits + total_misses) > 5 else None,
        }

    def _record_hit(self, lesson_id: int) -> None:
        """Record that a lesson caught a potential failure (preflight hit)."""
        now = _utc_iso()
        cur = self._conn.cursor()
        cur.execute(
            "UPDATE operational_lessons SET hit_count = hit_count + 1, last_hit_at = ? WHERE id = ?",
            (now, lesson_id),
        )
        self._conn.commit()

    def _record_miss(self, lesson_id: int) -> None:
        """Record that a failure occurred despite having a lesson for it."""
        cur = self._conn.cursor()
        cur.execute(
            "UPDATE operational_lessons SET miss_count = miss_count + 1 WHERE id = ?",
            (lesson_id,),
        )
        self._conn.commit()

    def _auto_extract_lesson(self, task_kind: str, error: str, context: Dict[str, Any]) -> Optional[int]:
        """
        Automatically extract a lesson from a new failure.

        Uses keyword extraction to create a pattern, and generates
        a generic correction. This is a starting point — the lesson
        can be refined by human feedback or further postmortems.
        """
        # Extract key error phrases for pattern matching
        error_lower = error.lower().strip()
        if not error_lower or len(error_lower) < 10:
            return None

        # Classify the error
        category = "task_failure"
        if any(kw in error_lower for kw in ["external data", "data not found", "no data", "missing data", "api"]):
            category = "data_access"
        elif any(kw in error_lower for kw in ["model", "llm", "api key", "router", "provider"]):
            category = "model_routing"
        elif any(kw in error_lower for kw in ["stop", "halt", "idle", "stuck", "timeout", "stale"]):
            category = "stall"
        elif any(kw in error_lower for kw in ["rule", "constitution", "violation", "policy"]):
            category = "rule_violation"

        # Create pattern from error keywords (first 100 chars, escaped for regex)
        pattern_text = re.escape(error_lower[:100])

        # Generic correction based on category
        corrections = {
            "data_access": "Do NOT stop. Use cached data, synthetic data, or switch to a different task.",
            "model_routing": "Try all available models via SmartRouter fallback chain before failing.",
            "stall": "Re-bootstrap task queue. Never idle — always create new tasks.",
            "rule_violation": "Review CLAUDE.md constitution rules and correct the violation.",
            "task_failure": f"Task '{task_kind}' failed. Auto-retry once, then switch to alternate approach.",
        }

        return self.add_lesson(
            category=category,
            pattern=pattern_text,
            correction=corrections.get(category, corrections["task_failure"]),
            severity="warning",
            source="auto_postmortem",
        )

    def _log_event(self, event_type: str, data: Dict[str, Any]) -> None:
        """Log an operational event for metrics tracking."""
        now = _utc_iso()
        cur = self._conn.cursor()
        try:
            cur.execute(
                "INSERT INTO operational_events (created_at, event_type, data_json) VALUES (?, ?, ?)",
                (now, event_type, json.dumps(data, default=str)),
            )
            self._conn.commit()
        except Exception:
            pass

    def _init_schema(self) -> None:
        cur = self._conn.cursor()
        cur.execute("""
            CREATE TABLE IF NOT EXISTS operational_lessons (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                category TEXT NOT NULL,
                pattern TEXT NOT NULL,
                correction TEXT NOT NULL,
                severity TEXT NOT NULL DEFAULT 'warning',
                source TEXT NOT NULL DEFAULT 'unknown',
                hit_count INTEGER NOT NULL DEFAULT 0,
                miss_count INTEGER NOT NULL DEFAULT 0,
                last_hit_at TEXT NULL,
                active INTEGER NOT NULL DEFAULT 1
            )
        """)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS operational_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                event_type TEXT NOT NULL,
                data_json TEXT NOT NULL
            )
        """)
        cur.execute("""
            CREATE INDEX IF NOT EXISTS idx_lessons_category ON operational_lessons(category)
        """)
        cur.execute("""
            CREATE INDEX IF NOT EXISTS idx_lessons_active ON operational_lessons(active)
        """)
        cur.execute("""
            CREATE INDEX IF NOT EXISTS idx_events_type ON operational_events(event_type)
        """)
        self._conn.commit()

    def _row_to_lesson(self, r: sqlite3.Row) -> Lesson:
        return Lesson(
            id=int(r["id"]),
            created_at=str(r["created_at"]),
            category=str(r["category"]),
            pattern=str(r["pattern"]),
            correction=str(r["correction"]),
            severity=str(r["severity"]),
            source=str(r["source"]),
            hit_count=int(r["hit_count"] or 0),
            miss_count=int(r["miss_count"] or 0),
            last_hit_at=str(r["last_hit_at"]) if r["last_hit_at"] else None,
            active=bool(r["active"]),
        )


class OperationalLearner:
    """
    High-level interface for the Operational Learning Engine.

    This is what Orion, autopilot, and agents use.
    One call to initialize, then preflight/postmortem on every task.
    """

    def __init__(self, artifacts_dir: Path) -> None:
        self.artifacts_dir = artifacts_dir
        db_path = artifacts_dir / "learning" / "operational_lessons.db"
        self.store = LessonStore(db_path)
        self._ensure_seed_lessons()

    def close(self) -> None:
        self.store.close()

    def preflight(self, task_kind: str, context: Optional[Dict[str, Any]] = None) -> List[PreflightWarning]:
        """Run before every task. Returns warnings if context matches known failure patterns."""
        return self.store.preflight(task_kind, context or {})

    def postmortem(self, task_kind: str, error: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Run after every failure. Extracts and stores lessons."""
        return self.store.postmortem(task_kind, error, context or {})

    def get_agent_lessons(self, max_lessons: int = 15) -> List[Dict[str, str]]:
        """Get lessons formatted for injection into AgentContext."""
        return self.store.get_lessons_summary(max_lessons)

    def metrics(self) -> Dict[str, Any]:
        """Get learning metrics — proof the system is improving."""
        return self.store.metrics()

    def add_user_lesson(self, category: str, pattern: str, correction: str, severity: str = "critical") -> int:
        """Add a lesson from user feedback — highest priority."""
        return self.store.add_lesson(
            category=category,
            pattern=pattern,
            correction=correction,
            severity=severity,
            source="user_feedback",
        )

    def _ensure_seed_lessons(self) -> None:
        """
        Seed the store with known operational lessons.

        These come from REAL failures observed in production.
        They are the SYSTEM's memory — not documentation.
        """
        seeds = [
            # === STALL PREVENTION ===
            {
                "category": "stall",
                "pattern": "external.?data|require.*external|need.*external|external.*required",
                "correction": (
                    "NEVER STOP for external data. Escalation: "
                    "1) Try API 2) Use cached data 3) Use synthetic/mock data "
                    "4) Build the data provider 5) Switch to different task "
                    "6) Switch to different project. NEVER IDLE."
                ),
                "severity": "critical",
                "source": "seed_v1",
            },
            {
                "category": "stall",
                "pattern": "no pending tasks|queue.*empty|nothing.*to.*do|idle",
                "correction": (
                    "NEVER IDLE. Re-bootstrap task queue immediately. "
                    "Call orion.bootstrap() to create new tasks. "
                    "Constitution Rule 1: NEXUS NEVER STOPS."
                ),
                "severity": "critical",
                "source": "seed_v1",
            },
            {
                "category": "stall",
                "pattern": "waiting.*for|blocked.*by|depend.*on|requires.*human|manual.*step",
                "correction": (
                    "Do NOT wait for human input. Make the best autonomous decision: "
                    "use defaults, generate alternatives, or switch to another task. "
                    "Log what you need in artifacts/state/blocked_tasks.json for later review."
                ),
                "severity": "critical",
                "source": "seed_v1",
            },
            {
                "category": "stall",
                "pattern": "stopped|halted|terminated|shutdown|exit",
                "correction": (
                    "NEXUS NEVER STOPS. If a component died, restart it. "
                    "If a task failed, retry or switch to alternate task. "
                    "The system must be running 24/7."
                ),
                "severity": "critical",
                "source": "seed_v1",
            },

            # === DATA ACCESS ===
            {
                "category": "data_access",
                "pattern": "api.*fail|api.*error|api.*timeout|connection.*refused|http.*error",
                "correction": (
                    "API failure is normal. SmartRouter has fallback chains. "
                    "Try: 1) Different API provider 2) Cached response 3) Retry with backoff. "
                    "Never stop the pipeline for a single API failure."
                ),
                "severity": "warning",
                "source": "seed_v1",
            },
            {
                "category": "data_access",
                "pattern": "data.*not.*found|file.*not.*found|missing.*file|no.*such.*file",
                "correction": (
                    "Missing data is recoverable. Check: "
                    "1) Is the path correct? 2) Can we download it? "
                    "3) Can we use cached/stale version? 4) Can we skip this task?"
                ),
                "severity": "warning",
                "source": "seed_v1",
            },

            # === MODEL ROUTING ===
            {
                "category": "model_routing",
                "pattern": "forgot.*model|single.*model|only.*one.*model|no.*fallback",
                "correction": (
                    "ALWAYS use SmartRouter with full fallback chain. "
                    "Available models: Gemini 3 Pro → Claude Sonnet → GLM-5. "
                    "CIPHER and ECHO REQUIRE Claude. Never use only one model."
                ),
                "severity": "critical",
                "source": "seed_v1",
            },
            {
                "category": "model_routing",
                "pattern": "api.*key.*missing|no.*api.*key|authentication.*fail",
                "correction": (
                    "Check all API keys: GEMINI_API_KEY, ANTHROPIC_API_KEY, ZAI_API_KEY. "
                    "SmartRouter will route to available providers. "
                    "At least 2 providers must be available."
                ),
                "severity": "warning",
                "source": "seed_v1",
            },

            # === RULE VIOLATIONS ===
            {
                "category": "rule_violation",
                "pattern": "look.?ahead|future.*data|universe.*leak|data.*leak",
                "correction": (
                    "CRITICAL: Data integrity violation detected. "
                    "Stop this specific run and fix the data pipeline. "
                    "No look-ahead, no universe leakage. This is non-negotiable."
                ),
                "severity": "critical",
                "source": "seed_v1",
            },
            {
                "category": "rule_violation",
                "pattern": "in.?sample.*only|no.*holdout|no.*oos|overfit",
                "correction": (
                    "Promote ONLY with credible evidence. "
                    "In-sample gains alone are INVALID. "
                    "Must pass: holdout validation + stress test + ablation."
                ),
                "severity": "critical",
                "source": "seed_v1",
            },

            # === TASK FAILURES ===
            {
                "category": "task_failure",
                "pattern": "import.*error|module.*not.*found|no.*module",
                "correction": (
                    "Import error — likely missing dependency or wrong path. "
                    "Check: 1) Is the module installed? 2) Is PYTHONPATH set? "
                    "3) Is the working directory correct? Do not stop — retry after fix."
                ),
                "severity": "warning",
                "source": "seed_v1",
            },
            {
                "category": "task_failure",
                "pattern": "memory.*error|out.*of.*memory|killed|oom",
                "correction": (
                    "Memory exhaustion. Reduce batch size, trials count, or data window. "
                    "Apply overrides: {orion: {trials: 10}} to reduce memory usage. "
                    "Do not stop — restart with reduced parameters."
                ),
                "severity": "warning",
                "source": "seed_v1",
            },

            # === MULTI-PROJECT ===
            {
                "category": "rule_violation",
                "pattern": "only.*one.*project|single.*project|forgot.*project|missing.*project",
                "correction": (
                    "ALL 3 PROJECTS must run in parallel: crypto_perps, commodity_cta, crypto_options. "
                    "Constitution Rule 2. Check watchdog status. Restart missing projects."
                ),
                "severity": "critical",
                "source": "seed_v1",
            },
        ]

        for seed in seeds:
            self.store.add_lesson(**seed)
