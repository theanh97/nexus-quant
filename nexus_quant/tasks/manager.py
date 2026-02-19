from __future__ import annotations

"""
NEXUS Task Manager — Kanban-style task tracking.
Persists to artifacts/tasks/tasks.json.

Columns: TODO / IN_PROGRESS / REVIEW / DONE
Priority: critical / high / medium / low
Category: research / practice / procedure
  - research   : Literature review, hypothesis generation, data analysis
  - practice   : Backtest runs, strategy implementation, experiments
  - procedure  : Automated workflows, brain loops, alpha loops (AI-driven)
Delegated by: human / nexus / atlas / cipher / echo / flux
"""

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

VALID_STATUSES = {"todo", "in_progress", "review", "done"}
VALID_PRIORITIES = {"critical", "high", "medium", "low"}
VALID_CATEGORIES = {"research", "practice", "procedure"}
PRIORITY_ORDER = {"critical": 0, "high": 1, "medium": 2, "low": 3}

# Human-readable category labels & icons
CATEGORY_META = {
    "research": {"label": "Research", "icon": "🔬", "color": "#6366F1"},
    "practice": {"label": "Practice", "icon": "⚗️", "color": "#10B981"},
    "procedure": {"label": "Procedures", "icon": "🤖", "color": "#F59E0B"},
}


class Task:
    __slots__ = (
        "id", "title", "description", "status", "priority",
        "assignee", "tags", "progress", "due_date",
        "created_at", "updated_at", "created_by", "result",
        "category", "delegated_by",
    )

    def __init__(
        self,
        title: str,
        description: str = "",
        status: str = "todo",
        priority: str = "medium",
        assignee: str = "NEXUS",
        tags: Optional[List[str]] = None,
        progress: int = 0,
        due_date: Optional[str] = None,
        created_by: str = "system",
        task_id: Optional[str] = None,
        created_at: Optional[str] = None,
        updated_at: Optional[str] = None,
        result: Optional[str] = None,
        category: str = "practice",
        delegated_by: str = "nexus",
    ):
        self.id = task_id or str(uuid.uuid4())[:8]
        self.title = str(title)[:200]
        self.description = str(description)[:1000]
        self.status = status if status in VALID_STATUSES else "todo"
        self.priority = priority if priority in VALID_PRIORITIES else "medium"
        self.assignee = str(assignee)[:100]
        self.tags = [str(t) for t in (tags or [])][:10]
        self.progress = max(0, min(100, int(progress)))
        self.due_date = due_date
        self.created_by = str(created_by)[:100]
        self.created_at = created_at or datetime.now(timezone.utc).isoformat()
        self.updated_at = updated_at or self.created_at
        self.result = result
        self.category = category if category in VALID_CATEGORIES else "practice"
        self.delegated_by = str(delegated_by)[:50]  # human | nexus | atlas | cipher | echo | flux

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "description": self.description,
            "status": self.status,
            "priority": self.priority,
            "assignee": self.assignee,
            "tags": self.tags,
            "progress": self.progress,
            "due_date": self.due_date,
            "created_by": self.created_by,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "result": self.result,
            "category": self.category,
            "delegated_by": self.delegated_by,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Task":
        return cls(
            title=d.get("title", "Untitled"),
            description=d.get("description", ""),
            status=d.get("status", "todo"),
            priority=d.get("priority", "medium"),
            assignee=d.get("assignee", "NEXUS"),
            tags=d.get("tags", []),
            progress=d.get("progress", 0),
            due_date=d.get("due_date"),
            created_by=d.get("created_by", "system"),
            task_id=d.get("id"),
            created_at=d.get("created_at"),
            updated_at=d.get("updated_at"),
            result=d.get("result"),
            category=d.get("category", "practice"),
            delegated_by=d.get("delegated_by", "nexus"),
        )


class TaskManager:
    """Persistent JSON-backed Kanban task manager."""

    def __init__(self, artifacts_dir: Path):
        self._tasks_dir = Path(artifacts_dir) / "tasks"
        self._tasks_dir.mkdir(parents=True, exist_ok=True)
        self._path = self._tasks_dir / "tasks.json"
        self._tasks: Dict[str, Task] = {}
        self._load()

    def _load(self) -> None:
        if self._path.exists():
            try:
                raw = json.loads(self._path.read_text("utf-8"))
                for d in raw:
                    t = Task.from_dict(d)
                    self._tasks[t.id] = t
            except Exception:
                pass

    def _save(self) -> None:
        tasks_list = [t.to_dict() for t in self._tasks.values()]
        self._path.write_text(json.dumps(tasks_list, ensure_ascii=False, indent=2), "utf-8")

    def create(
        self,
        title: str,
        description: str = "",
        priority: str = "medium",
        assignee: str = "NEXUS",
        tags: Optional[List[str]] = None,
        due_date: Optional[str] = None,
        created_by: str = "user",
        category: str = "practice",
        delegated_by: str = "nexus",
    ) -> Task:
        t = Task(
            title=title, description=description, priority=priority,
            assignee=assignee, tags=tags, due_date=due_date, created_by=created_by,
            category=category, delegated_by=delegated_by,
        )
        self._tasks[t.id] = t
        self._save()
        return t

    def get(self, task_id: str) -> Optional[Task]:
        return self._tasks.get(task_id)

    def all_tasks(self) -> List[Task]:
        tasks = list(self._tasks.values())
        tasks.sort(key=lambda t: (
            {"done": 1, "review": 0, "in_progress": 0, "todo": 0}.get(t.status, 0),
            PRIORITY_ORDER.get(t.priority, 9),
            t.created_at,
        ))
        return tasks

    def by_status(self) -> Dict[str, List[Task]]:
        columns: Dict[str, List[Task]] = {"todo": [], "in_progress": [], "review": [], "done": []}
        for t in self.all_tasks():
            col = t.status if t.status in columns else "todo"
            columns[col].append(t)
        # Sort each column by priority
        for col in columns:
            columns[col].sort(key=lambda t: PRIORITY_ORDER.get(t.priority, 9))
        return columns

    def by_category(self) -> Dict[str, List[Task]]:
        """Group active (non-done) tasks by category: research / practice / procedure."""
        cats: Dict[str, List[Task]] = {"research": [], "practice": [], "procedure": []}
        for t in self.all_tasks():
            if t.status == "done":
                continue
            cat = t.category if t.category in cats else "practice"
            cats[cat].append(t)
        return cats

    def by_category_all(self) -> Dict[str, List[Task]]:
        """Group ALL tasks (including done) by category."""
        cats: Dict[str, List[Task]] = {"research": [], "practice": [], "procedure": []}
        for t in self.all_tasks():
            cat = t.category if t.category in cats else "practice"
            cats[cat].append(t)
        return cats

    def update(self, task_id: str, **kwargs) -> Optional[Task]:
        t = self._tasks.get(task_id)
        if not t:
            return None
        allowed = {"title", "description", "status", "priority", "assignee", "tags", "progress", "due_date", "result", "category", "delegated_by"}
        for k, v in kwargs.items():
            if k not in allowed:
                continue
            if k == "status" and v not in VALID_STATUSES:
                continue
            if k == "priority" and v not in VALID_PRIORITIES:
                continue
            if k == "progress":
                v = max(0, min(100, int(v)))
            if k == "status" and v == "done" and t.progress < 100:
                t.progress = 100
            setattr(t, k, v)
        t.updated_at = datetime.now(timezone.utc).isoformat()
        self._save()
        return t

    def delete(self, task_id: str) -> bool:
        if task_id in self._tasks:
            del self._tasks[task_id]
            self._save()
            return True
        return False

    def kanban_summary(self) -> Dict[str, Any]:
        cols = self.by_status()
        return {
            "todo": len(cols["todo"]),
            "in_progress": len(cols["in_progress"]),
            "review": len(cols["review"]),
            "done": len(cols["done"]),
            "total": sum(len(v) for v in cols.values()),
            "critical": sum(1 for t in self._tasks.values() if t.priority == "critical" and t.status != "done"),
        }

    def kanban_by_category(self) -> Dict[str, Any]:
        """Returns tasks grouped by category AND status for the 3-lane view."""
        result = {}
        for cat in ("research", "practice", "procedure"):
            cols: Dict[str, List[Dict]] = {"todo": [], "in_progress": [], "review": [], "done": []}
            for t in self.all_tasks():
                if t.category == cat or (cat == "practice" and t.category not in VALID_CATEGORIES):
                    status = t.status if t.status in cols else "todo"
                    cols[status].append(t.to_dict())
            result[cat] = {
                "meta": CATEGORY_META[cat],
                "columns": cols,
                "counts": {k: len(v) for k, v in cols.items()},
            }
        return result

    def seed_defaults(self) -> None:
        """Add default NEXUS tasks if none exist."""
        if self._tasks:
            return
        defaults = [
            # Research lane — delegated by human
            {"title": "Research Optimal Factor Combinations", "description": "Identify best-performing factor combinations from academic literature", "priority": "high", "assignee": "ATLAS", "tags": ["factors", "research"], "category": "research", "delegated_by": "human"},
            {"title": "Study Walk-Forward Validation Methods", "description": "Review best practices for out-of-sample validation in crypto strategies", "priority": "medium", "assignee": "ECHO", "tags": ["validation", "methodology"], "category": "research", "delegated_by": "human"},
            {"title": "Research Crypto Regime Detection", "description": "Survey machine learning approaches for market regime classification", "priority": "medium", "assignee": "NEXUS", "tags": ["regime", "ml"], "category": "research", "delegated_by": "human"},
            # Practice lane — human-assigned experiments
            {"title": "ML Factor Sharpe > 2 Target", "description": "Improve ML factor Ridge model to reach Sharpe > 2 using new features", "priority": "high", "assignee": "CIPHER", "tags": ["ml", "factor"], "category": "practice", "delegated_by": "human"},
            {"title": "Bias Check All Strategies", "description": "Run full bias checker on all Binance strategy runs", "priority": "medium", "assignee": "ECHO", "tags": ["validation", "bias"], "category": "practice", "delegated_by": "human"},
            # Procedure lane — AI-autonomous tasks
            {"title": "Daily Research Cycle (56 sources)", "description": "Automated: fetch arXiv, Reddit, quant blogs daily — AEA loop", "priority": "high", "assignee": "NEXUS", "tags": ["research", "automation"], "category": "procedure", "delegated_by": "nexus"},
            {"title": "AlphaLoop — Champion Strategy Tracking", "description": "Automated: run all configs hourly, promote new champion if Sharpe improves", "priority": "high", "assignee": "FLUX", "tags": ["alpha", "automation"], "category": "procedure", "delegated_by": "nexus"},
            {"title": "Brain Loop — 24h Autonomous Operation", "description": "Automated: NEXUS brain runs research/backtest/reflect cycle continuously", "priority": "medium", "assignee": "NEXUS", "tags": ["brain", "loop"], "category": "procedure", "delegated_by": "nexus"},
        ]
        for d in defaults:
            self.create(**d, created_by="NEXUS")
