"""
NEXUS Diary - Daily learning journal.
NEXUS writes a diary entry after each research cycle summarizing:
- What experiments were run
- What was learned
- What to do next
This creates an auditable, human-readable log of NEXUS thinking.
"""
from __future__ import annotations
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, asdict


@dataclass
class DiaryEntry:
    date: str
    experiments_run: int
    best_strategy: Optional[str]
    best_sharpe: Optional[float]
    key_learnings: List[str]
    next_plans: List[str]
    mood: str
    anomalies: List[str]
    goals_progress: str
    raw_summary: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_markdown(self) -> str:
        lines = [
            f"# NEXUS Diary -- {self.date}",
            "",
            f"**Mood:** {self.mood.upper()}",
            "",
            "## What I Did Today",
            f"Ran {self.experiments_run} experiments.",
            f"Best strategy: {self.best_strategy or 'N/A'} (Sharpe={self.best_sharpe or 'N/A'})",
            "",
            "## What I Learned",
        ]
        for item in self.key_learnings:
            lines.append(f"- {item}")
        if self.anomalies:
            lines.append("")
            lines.append("## Anomalies / Concerns")
            for a in self.anomalies:
                lines.append(f"- WARNING: {a}")
        lines.append("")
        lines.append("## Goals Progress")
        lines.append(self.goals_progress)
        lines.append("")
        lines.append("## Next Plans")
        for p in self.next_plans:
            lines.append(f"- {p}")
        return "\n".join(lines)


class NexusDiary:
    """Manages NEXUS diary entries."""

    def __init__(self, artifacts_dir: Path) -> None:
        self.diary_dir = artifacts_dir / "brain" / "diary"
        self.diary_dir.mkdir(parents=True, exist_ok=True)

    def write_entry(
        self,
        experiments_run: int,
        best_strategy: Optional[str],
        best_sharpe: Optional[float],
        key_learnings: List[str],
        next_plans: List[str],
        mood: str = "focused",
        anomalies: Optional[List[str]] = None,
        goals_progress: str = "",
        raw_summary: str = "",
    ) -> DiaryEntry:
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        entry = DiaryEntry(
            date=today,
            experiments_run=experiments_run,
            best_strategy=best_strategy,
            best_sharpe=best_sharpe,
            key_learnings=key_learnings or ["No significant findings today."],
            next_plans=next_plans or ["Continue current research direction."],
            mood=mood,
            anomalies=anomalies or [],
            goals_progress=goals_progress,
            raw_summary=raw_summary,
        )
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        (self.diary_dir / f"entry.{ts}.json").write_text(
            json.dumps(entry.to_dict(), indent=2), encoding="utf-8"
        )
        (self.diary_dir / f"entry.{ts}.md").write_text(entry.to_markdown(), encoding="utf-8")
        (self.diary_dir / "latest.json").write_text(
            json.dumps(entry.to_dict(), indent=2), encoding="utf-8"
        )
        (self.diary_dir / "latest.md").write_text(entry.to_markdown(), encoding="utf-8")
        return entry

    def latest_entry(self) -> Optional[DiaryEntry]:
        p = self.diary_dir / "latest.json"
        if p.exists():
            try:
                return DiaryEntry(**json.loads(p.read_text("utf-8")))
            except Exception:
                pass
        return None

    def recent_entries(self, limit: int = 7) -> List[DiaryEntry]:
        files = sorted(
            self.diary_dir.glob("entry.*.json"),
            key=lambda f: f.stat().st_mtime,
            reverse=True,
        )
        out = []
        for f in files[:limit]:
            try:
                out.append(DiaryEntry(**json.loads(f.read_text("utf-8"))))
            except Exception:
                continue
        return out
