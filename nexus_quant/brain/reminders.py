"""NEXUS Reminder Writer — generates current_reminder.md for Claude PM sessions."""
from __future__ import annotations
from pathlib import Path
from datetime import datetime, timezone
import json


def write_reminder(artifacts_dir: Path) -> None:
    """Write current NEXUS state as a reminder file. Called by brain loop (throttled 30 min)."""
    try:
        artifacts_dir = Path(artifacts_dir)

        # ── Strategy registry ──────────────────────────────────────────────
        reg_path = artifacts_dir / "memory" / "semantic" / "strategy_registry.json"
        registry: dict = {}
        if reg_path.exists():
            registry = json.loads(reg_path.read_text("utf-8")).get("strategies", {})

        best_name, best_sharpe = "unknown", 0.0
        if registry:
            best_entry = max(registry.values(), key=lambda s: s.get("best_sharpe") or -99)
            best_name = best_entry.get("name", "unknown")
            best_sharpe = best_entry.get("best_sharpe", 0.0)

        # ── Stagnant (run >5x without improvement) ─────────────────────────
        stagnant = [k for k, v in registry.items() if v.get("times_run", 0) > 5]

        # ── Last 5 lessons ─────────────────────────────────────────────────
        lessons_path = artifacts_dir / "memory" / "semantic" / "lessons_learned.jsonl"
        lessons: list[str] = []
        if lessons_path.exists():
            lines = lessons_path.read_text("utf-8").splitlines()[-5:]
            for ln in lines:
                try:
                    d = json.loads(ln)
                    lvl = d.get("level", "INFO")
                    msg = d.get("message", "")[:100]
                    lessons.append(f"[{lvl}] {msg}")
                except Exception:
                    pass

        # ── Active goals ───────────────────────────────────────────────────
        goals_path = artifacts_dir / "brain" / "goals.json"
        goals: list[str] = []
        if goals_path.exists():
            try:
                g = json.loads(goals_path.read_text("utf-8"))
                raw = g if isinstance(g, list) else g.get("goals", [])
                goals = [str(x.get("goal", x))[:80] for x in raw[:3]]
            except Exception:
                pass

        ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

        lines_out = [
            f"# NEXUS System Reminder — {ts}",
            "",
            "## Best Result",
            f"- Strategy: **{best_name}** | Sharpe: **{best_sharpe:.3f}**",
            "- 2022 bear market weakness: Sharpe 0.112 — *needs regime-adaptive fix*",
            "",
            "## Stagnant Strategies (do NOT repeat — explore new directions)",
        ]
        lines_out += [f"- {s}" for s in stagnant] if stagnant else ["- None"]
        lines_out += [
            "",
            "## Active Goals",
        ]
        lines_out += [f"- {g}" for g in goals] if goals else [
            "- Achieve Sharpe > 3.0 OOS on real Binance data",
            "- Fix 2022 bear market (regime-adaptive strategy)",
            "- ML Factor Sharpe > 2.0 after optimization",
        ]
        lines_out += [
            "",
            "## Recent Lessons",
        ]
        lines_out += [f"- {l}" for l in lessons] if lessons else ["- No lessons yet"]
        lines_out += [
            "",
            "## PM Rules (Claude)",
            "1. Check `BrainCritic.check_novelty()` before proposing any experiment",
            "2. Run `MemoryCurator.curate()` every 5 runs",
            "3. Do NOT repeat stagnant strategies — pivot to new approach",
            "4. I am PM/Brain only — Codex implements all code",
            "5. No confirmation prompts — full automation",
            "",
            "## Next Priorities",
            "- [ ] Regime-adaptive strategy (fix 2022 weakness)",
            "- [ ] ML Factor refactor (Sharpe from 0 → 2.0+)",
            "- [ ] Wire BrainCritic into brain loop (check novelty pre-experiment)",
            "",
        ]

        state_dir = artifacts_dir / "state"
        state_dir.mkdir(parents=True, exist_ok=True)
        (state_dir / "current_reminder.md").write_text("\n".join(lines_out), "utf-8")

    except Exception:
        pass  # Never crash the brain loop
