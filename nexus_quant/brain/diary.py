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
from collections import Counter
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
        self.artifacts_dir = Path(artifacts_dir)
        self.diary_dir = artifacts_dir / "brain" / "diary"
        self.diary_dir.mkdir(parents=True, exist_ok=True)
        self.semantic_dir = self.artifacts_dir / "memory" / "semantic"

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
        self._maybe_consolidate(every_n=10, window=10)
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

    def _maybe_consolidate(self, *, every_n: int = 10, window: int = 10) -> None:
        """
        Every N diary entries, distill the latest window into a single semantic lesson line.
        """

        try:
            n_every = int(every_n)
            n_window = int(window)
        except Exception:
            return
        if n_every <= 0 or n_window <= 0:
            return

        try:
            total = len(list(self.diary_dir.glob("entry.*.json")))
        except Exception:
            return
        if total <= 0 or (total % n_every) != 0:
            return

        entries = self.recent_entries(limit=n_window)
        if len(entries) < n_window:
            return

        msg, evidence = self._distill(entries)
        payload = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "level": "INFO",
            "kind": "diary_consolidation",
            "strategy": str(evidence.get("best_strategy") or ""),
            "message": msg,
            "evidence": evidence,
        }

        try:
            lessons_path = self.semantic_dir / "lessons_learned.jsonl"
            lessons_path.parent.mkdir(parents=True, exist_ok=True)
            with lessons_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
        except Exception:
            return

    @staticmethod
    def _distill(entries: List[DiaryEntry]) -> tuple[str, Dict[str, Any]]:
        """
        Produce a one-line summary message plus evidence for the last N entries.
        """

        dates = [e.date for e in entries if isinstance(e.date, str) and e.date]
        date_min = min(dates) if dates else ""
        date_max = max(dates) if dates else ""
        date_range = date_min if date_min == date_max else f"{date_min}..{date_max}"

        moods = [str(e.mood or "") for e in entries if getattr(e, "mood", None)]
        mood_mode = ""
        try:
            mood_mode = Counter(moods).most_common(1)[0][0] if moods else ""
        except Exception:
            mood_mode = moods[0] if moods else ""

        best_strategy = None
        best_sharpe = None
        for e in entries:
            if e.best_sharpe is None:
                continue
            if best_sharpe is None or float(e.best_sharpe) > float(best_sharpe):
                best_sharpe = float(e.best_sharpe)
                best_strategy = e.best_strategy

        learnings: List[str] = []
        plans: List[str] = []
        anomalies: List[str] = []
        for e in entries:
            try:
                for it in (e.key_learnings or [])[:20]:
                    s = str(it).strip()
                    if s and s not in learnings:
                        learnings.append(s)
                for it in (e.next_plans or [])[:20]:
                    s = str(it).strip()
                    if s and s not in plans:
                        plans.append(s)
                for it in (e.anomalies or [])[:20]:
                    s = str(it).strip()
                    if s and s not in anomalies:
                        anomalies.append(s)
            except Exception:
                continue

        learn_s = "; ".join(learnings[:5]) if learnings else "No clear learnings captured."
        plans_s = "; ".join(plans[:3]) if plans else "Continue exploration."
        anom_s = "; ".join(anomalies[:2]) if anomalies else ""

        best_s = f"{best_strategy} (Sharpe {best_sharpe:.3f})" if best_strategy and best_sharpe is not None else "N/A"
        parts = [
            f"Diary distillation [{date_range}]",
            f"mood={mood_mode or 'unknown'}",
            f"best={best_s}",
            f"learnings={learn_s}",
            f"next={plans_s}",
        ]
        if anom_s:
            parts.append(f"anomalies={anom_s}")
        msg = " | ".join(parts).replace("\n", " ").replace("\r", " ").strip()
        if len(msg) > 600:
            msg = msg[:597].rstrip() + "..."

        evidence = {
            "entry_count": int(len(entries)),
            "date_range": date_range,
            "mood_mode": mood_mode,
            "best_strategy": best_strategy,
            "best_sharpe": best_sharpe,
            "learnings_sample": learnings[:10],
            "next_plans_sample": plans[:10],
            "anomalies_sample": anomalies[:10],
        }
        return msg, evidence
