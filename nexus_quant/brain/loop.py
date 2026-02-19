from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from nexus_quant.brain.diary import NexusDiary
from nexus_quant.brain.goals import GoalTracker
from nexus_quant.brain.identity import NexusIdentity
from nexus_quant.brain.reasoning import generate_diary_synthesis
from nexus_quant.run import improve_one, run_one


class NexusAutonomousLoop:
    def __init__(self, cfg_path: Path, artifacts_dir: Path) -> None:
        self.cfg_path = Path(cfg_path)
        self.artifacts_dir = Path(artifacts_dir)
        self.state_path = self.artifacts_dir / "brain" / "loop_state.json"
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        self.state: Dict[str, Any] = self._read_json(self.state_path) or {}
        self.identity = NexusIdentity(self.artifacts_dir)
        self.goals = GoalTracker(self.artifacts_dir)
        self.diary = NexusDiary(self.artifacts_dir)

    def run_cycle(self) -> dict:
        cycle = int(self.state.get("cycle_number") or 0) + 1
        self.state["cycle_number"] = cycle

        backtest_run_id = run_one(self.cfg_path, self.artifacts_dir)
        cfg = self._read_json(self.cfg_path) or {}
        trials = int(((cfg.get("self_learn") or {}) or {}).get("trials") or 30)
        improve_run_id = improve_one(self.cfg_path, self.artifacts_dir, trials=trials)

        experiments = [
            {"kind": "backtest", "run_id": backtest_run_id},
            {"kind": "improve", "run_id": improve_run_id, "trials": trials},
        ]

        metrics = self._read_json(self.artifacts_dir / "runs" / backtest_run_id / "metrics.json") or {}
        summary = (metrics.get("summary") or {}) if isinstance(metrics, dict) else {}
        best_sharpe = self._to_float(summary.get("sharpe"))

        ledger_events = self._read_jsonl_tail(self.artifacts_dir / "ledger" / "ledger.jsonl", n=200)
        self.identity.update_from_ledger(ledger_events)

        self.goals.update_progress({k: float(v) for k, v in summary.items() if isinstance(v, (int, float))})
        goals_progress = self.goals.summary()

        synth = generate_diary_synthesis(
            experiments_run=len(experiments),
            metrics_summary=summary,
            ledger_events=ledger_events,
            goals_summary=goals_progress,
            persona=self.identity.state.persona,
        )
        mood = str(synth.get("mood") or self.identity.state.mood)
        best_strategy = self._strategy_name(cfg)

        self.diary.write_entry(
            experiments_run=len(experiments),
            best_strategy=best_strategy,
            best_sharpe=best_sharpe,
            key_learnings=list(synth.get("key_learnings") or []),
            next_plans=list(synth.get("next_plans") or []),
            mood=mood,
            goals_progress=goals_progress,
            raw_summary=str(synth.get("raw") or ""),
        )
        diary_path = str(self.artifacts_dir / "brain" / "diary" / "latest.md")

        out = {
            "cycle_number": cycle,
            "experiments": experiments,
            "best_sharpe": best_sharpe,
            "mood": mood,
            "diary_path": diary_path,
        }
        self.state["last_cycle"] = out
        self.state_path.write_text(json.dumps(self.state, indent=2, sort_keys=True), encoding="utf-8")
        return out

    @staticmethod
    def _read_json(path: Path) -> Optional[Dict[str, Any]]:
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
            return raw if isinstance(raw, dict) else None
        except Exception:
            return None

    @staticmethod
    def _read_jsonl_tail(path: Path, n: int = 50) -> List[Dict[str, Any]]:
        if not path.exists():
            return []
        out: List[Dict[str, Any]] = []
        for ln in path.read_text(encoding="utf-8").splitlines()[-n:]:
            try:
                obj = json.loads(ln)
                if isinstance(obj, dict):
                    out.append(obj)
            except Exception:
                continue
        return out

    @staticmethod
    def _to_float(x: Any) -> Optional[float]:
        try:
            return float(x) if x is not None else None
        except Exception:
            return None

    @staticmethod
    def _strategy_name(cfg: Dict[str, Any]) -> Optional[str]:
        try:
            return str((cfg.get("strategy") or {}).get("name") or "") or None
        except Exception:
            return None
