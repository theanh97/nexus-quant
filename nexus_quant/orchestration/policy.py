from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Set


def _utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _utc_ts() -> int:
    return int(datetime.now(timezone.utc).timestamp())


@dataclass(frozen=True)
class PolicyConfig:
    fast_gate_runs: int = 25
    fast_gate_seconds: int = 4 * 3600
    deep_gate_runs: int = 150
    reset_gate_runs: int = 600
    runs_per_budget_window: int = 100
    working_memory_ttl_runs: int = 150
    prior_half_life_runs: int = 300


class ResearchPolicy:
    """
    Event-driven policy controller for 24/7 autonomous loops.

    Gates are run-count based (with a time fallback for fast checks):
    - fast gate: frequent checks/reminders
    - deep gate: richer self-critique/debate pass
    - reset gate: force broad refresh/research ingest
    """

    def __init__(self, artifacts_dir: Path, cfg: PolicyConfig | None = None) -> None:
        self.artifacts_dir = Path(artifacts_dir)
        self.cfg = cfg or PolicyConfig()
        self.state_path = self.artifacts_dir / "state" / "research_policy_state.json"
        self.state_path.parent.mkdir(parents=True, exist_ok=True)

    def evaluate(self) -> Dict[str, Any]:
        now_ts = _utc_ts()
        now_iso = _utc_iso()
        run_count = self._run_count()
        state = self._load_state()

        # First run on an existing workspace: initialize baselines to current counters.
        if state.get("last_eval_run_count") is None:
            state["last_fast_run_count"] = run_count
            state["last_fast_ts"] = now_ts
            state["last_deep_run_count"] = run_count
            state["last_reset_run_count"] = run_count
            state["budget_window_start_runs"] = run_count
            state["window_runs"] = 0
            state["allow_improve"] = True
            state["improve_pause_until_runs"] = run_count
            state["last_eval_ts"] = now_iso
            state["last_eval_run_count"] = run_count
            state["config"] = {
                "fast_gate_runs": int(self.cfg.fast_gate_runs),
                "fast_gate_seconds": int(self.cfg.fast_gate_seconds),
                "deep_gate_runs": int(self.cfg.deep_gate_runs),
                "reset_gate_runs": int(self.cfg.reset_gate_runs),
                "runs_per_budget_window": int(self.cfg.runs_per_budget_window),
                "working_memory_ttl_runs": int(self.cfg.working_memory_ttl_runs),
                "prior_half_life_runs": int(self.cfg.prior_half_life_runs),
            }
            self._save_state(state)
            return {
                "ts": now_iso,
                "run_count": run_count,
                "window_runs": 0,
                "window_size": int(self.cfg.runs_per_budget_window),
                "fast_due": False,
                "deep_due": False,
                "reset_due": False,
                "allow_improve": True,
                "improve_pause_until_runs": run_count,
                "actions": [],
                "state_path": str(self.state_path),
            }

        # Reset window if run counter dropped (artifacts rotated).
        if int(state.get("budget_window_start_runs") or 0) > run_count:
            state["budget_window_start_runs"] = run_count

        window_runs = int(run_count - int(state.get("budget_window_start_runs") or 0))
        pause_until_runs = int(state.get("improve_pause_until_runs") or 0)
        budget_exhausted = window_runs >= int(self.cfg.runs_per_budget_window)
        if budget_exhausted and run_count >= pause_until_runs:
            pause_until_runs = run_count + int(self.cfg.fast_gate_runs)
            state["improve_pause_until_runs"] = pause_until_runs
            state["budget_window_start_runs"] = run_count
            window_runs = 0
        allow_improve = run_count >= pause_until_runs

        last_fast_runs = int(state.get("last_fast_run_count") or 0)
        last_fast_ts = int(state.get("last_fast_ts") or 0)
        last_deep_runs = int(state.get("last_deep_run_count") or 0)
        last_reset_runs = int(state.get("last_reset_run_count") or 0)

        fast_due = (
            run_count > 0
            and (
                (run_count - last_fast_runs) >= int(self.cfg.fast_gate_runs)
                or (now_ts - last_fast_ts) >= int(self.cfg.fast_gate_seconds)
            )
        )
        deep_due = run_count > 0 and (run_count - last_deep_runs) >= int(self.cfg.deep_gate_runs)
        reset_due = run_count > 0 and (run_count - last_reset_runs) >= int(self.cfg.reset_gate_runs)

        actions: List[Dict[str, Any]] = []
        if fast_due:
            actions.extend(
                [
                    {
                        "kind": "rules_reminder",
                        "payload": {
                            "gate": "fast",
                            "run_count": run_count,
                            "window_runs": window_runs,
                            "window_size": int(self.cfg.runs_per_budget_window),
                            "allow_improve": bool(allow_improve),
                            "working_memory_ttl_runs": int(self.cfg.working_memory_ttl_runs),
                            "prior_half_life_runs": int(self.cfg.prior_half_life_runs),
                        },
                    },
                    {"kind": "reflect", "payload": {"gate": "fast"}},
                    {"kind": "critique", "payload": {"gate": "fast"}},
                ]
            )

        if deep_due:
            actions.extend(
                [
                    {"kind": "rules_reminder", "payload": {"gate": "deep", "run_count": run_count}},
                    {"kind": "policy_review", "payload": {"mode": "deep_gate", "run_count": run_count}},
                    {"kind": "wisdom", "payload": {"gate": "deep"}},
                    {"kind": "agent_run", "payload": {"gate": "deep"}},
                    {"kind": "reflect", "payload": {"gate": "deep"}},
                    {"kind": "critique", "payload": {"gate": "deep"}},
                ]
            )

        if reset_due:
            actions.extend(
                [
                    {"kind": "rules_reminder", "payload": {"gate": "reset", "run_count": run_count}},
                    {"kind": "policy_review", "payload": {"mode": "reset_gate", "run_count": run_count}},
                    {"kind": "research_ingest", "payload": {"gate": "reset"}},
                    {"kind": "reflect", "payload": {"gate": "reset"}},
                    {"kind": "critique", "payload": {"gate": "reset"}},
                ]
            )

        if not allow_improve:
            actions.append(
                {
                    "kind": "policy_review",
                    "payload": {
                        "mode": "budget_pause",
                        "run_count": run_count,
                        "window_runs": window_runs,
                        "window_size": int(self.cfg.runs_per_budget_window),
                        "pause_until_runs": pause_until_runs,
                    },
                }
            )

        if fast_due:
            state["last_fast_run_count"] = run_count
            state["last_fast_ts"] = now_ts
        if deep_due:
            state["last_deep_run_count"] = run_count
        if reset_due:
            state["last_reset_run_count"] = run_count

        state["last_eval_ts"] = now_iso
        state["last_eval_run_count"] = run_count
        state["window_runs"] = window_runs
        state["allow_improve"] = bool(allow_improve)
        state["improve_pause_until_runs"] = int(pause_until_runs)
        state["config"] = {
            "fast_gate_runs": int(self.cfg.fast_gate_runs),
            "fast_gate_seconds": int(self.cfg.fast_gate_seconds),
            "deep_gate_runs": int(self.cfg.deep_gate_runs),
            "reset_gate_runs": int(self.cfg.reset_gate_runs),
            "runs_per_budget_window": int(self.cfg.runs_per_budget_window),
            "working_memory_ttl_runs": int(self.cfg.working_memory_ttl_runs),
            "prior_half_life_runs": int(self.cfg.prior_half_life_runs),
        }
        self._save_state(state)

        return {
            "ts": now_iso,
            "run_count": run_count,
            "window_runs": window_runs,
            "window_size": int(self.cfg.runs_per_budget_window),
            "fast_due": bool(fast_due),
            "deep_due": bool(deep_due),
            "reset_due": bool(reset_due),
            "allow_improve": bool(allow_improve),
            "improve_pause_until_runs": int(pause_until_runs),
            "actions": self._dedupe_actions(actions),
            "state_path": str(self.state_path),
        }

    def _run_count(self) -> int:
        runs_dir = self.artifacts_dir / "runs"
        if not runs_dir.exists():
            return 0
        try:
            return len([d for d in runs_dir.iterdir() if d.is_dir()])
        except Exception:
            return 0

    def _load_state(self) -> Dict[str, Any]:
        if self.state_path.exists():
            try:
                obj = json.loads(self.state_path.read_text(encoding="utf-8"))
                if isinstance(obj, dict):
                    return obj
            except Exception:
                pass
        return {
            "created_at": _utc_iso(),
            "last_fast_run_count": 0,
            "last_fast_ts": 0,
            "last_deep_run_count": 0,
            "last_reset_run_count": 0,
            "budget_window_start_runs": 0,
            "window_runs": 0,
            "allow_improve": True,
            "improve_pause_until_runs": 0,
            "last_eval_run_count": None,
        }

    def _save_state(self, state: Dict[str, Any]) -> None:
        self.state_path.write_text(json.dumps(state, indent=2, sort_keys=True), encoding="utf-8")

    @staticmethod
    def _dedupe_actions(actions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        seen: Set[str] = set()
        for a in actions:
            key = json.dumps({"kind": a.get("kind"), "payload": a.get("payload") or {}}, sort_keys=True)
            if key in seen:
                continue
            seen.add(key)
            out.append({"kind": str(a.get("kind") or ""), "payload": dict(a.get("payload") or {})})
        return out
