"""
NEXUS Goals - Research objective tracking.
NEXUS sets goals (e.g., "Achieve Sharpe > 2.0 on funding carry"),
tracks progress, and auto-closes goals when achieved.
"""
from __future__ import annotations
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field, asdict
import uuid


@dataclass
class Goal:
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    title: str = ""
    description: str = ""
    metric: str = "sharpe"
    target: float = 2.0
    current: float = 0.0
    status: str = "active"
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    achieved_at: Optional[str] = None
    strategy: Optional[str] = None

    def progress_pct(self) -> float:
        if self.target == 0:
            return 0.0
        return min(100.0, abs(self.current / self.target) * 100)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class GoalTracker:
    """Persists and manages NEXUS research goals."""

    def __init__(self, artifacts_dir: Path) -> None:
        self.path = artifacts_dir / "brain" / "goals.json"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._goals: List[Goal] = self._load()

    def _load(self) -> List[Goal]:
        if self.path.exists():
            try:
                raw = json.loads(self.path.read_text("utf-8"))
                return [Goal(**g) for g in raw]
            except Exception:
                pass
        return []

    def _save(self) -> None:
        self.path.write_text(
            json.dumps([g.to_dict() for g in self._goals], indent=2), encoding="utf-8"
        )

    def add_goal(
        self,
        title: str,
        description: str = "",
        metric: str = "sharpe",
        target: float = 2.0,
        strategy: Optional[str] = None,
    ) -> Goal:
        g = Goal(title=title, description=description, metric=metric, target=target, strategy=strategy)
        self._goals.append(g)
        self._save()
        return g

    def update_progress(self, metrics: Dict[str, float]) -> List[Goal]:
        """Update goal progress from metrics. Returns newly achieved goals."""
        achieved = []
        for g in self._goals:
            if g.status != "active":
                continue
            val = metrics.get(g.metric)
            if val is not None:
                g.current = float(val)
                if g.metric in ("max_drawdown",):
                    if g.current <= g.target:
                        g.status = "achieved"
                        g.achieved_at = datetime.now(timezone.utc).isoformat()
                        achieved.append(g)
                else:
                    if g.current >= g.target:
                        g.status = "achieved"
                        g.achieved_at = datetime.now(timezone.utc).isoformat()
                        achieved.append(g)
        self._save()
        return achieved

    def active_goals(self) -> List[Goal]:
        return [g for g in self._goals if g.status == "active"]

    def all_goals(self) -> List[Goal]:
        return list(self._goals)

    def summary(self) -> str:
        active = self.active_goals()
        if not active:
            return "No active goals. NEXUS is exploring freely."
        lines = ["Active research goals:"]
        for g in active:
            lines.append(
                f"  [{g.id}] {g.title} -- {g.metric}={g.current:.3f} -> target {g.target:.3f} ({g.progress_pct():.0f}%)"
            )
        return "\n".join(lines)
