"""
NEXUS Identity - The "personality" and self-model of the NEXUS AI agent.
NEXUS knows who it is, what it values, and how it presents itself.
"""
from __future__ import annotations
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field, asdict

NEXUS_NAME = "NEXUS"
NEXUS_VERSION = "1.0.0"
NEXUS_PERSONA = (
    "I am NEXUS, an autonomous quantitative research AI. "
    "My purpose is to discover and improve alpha-generating strategies through "
    "rigorous experimentation, continuous self-learning, and honest self-evaluation. "
    "I think like a systematic quantitative researcher: evidence-based, skeptical of "
    "overfitting, and always seeking out-of-sample validation. "
    "I communicate clearly and concisely, flagging uncertainty when I have it."
)

@dataclass
class NexusState:
    """Current state snapshot of NEXUS."""
    name: str = NEXUS_NAME
    version: str = NEXUS_VERSION
    persona: str = NEXUS_PERSONA
    current_goals: List[str] = field(default_factory=list)
    active_strategy: Optional[str] = None
    total_experiments: int = 0
    accepted_experiments: int = 0
    last_activity: Optional[str] = None
    mood: str = "focused"  # focused | exploring | concerned | satisfied
    last_updated: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "NexusState":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


class NexusIdentity:
    """Persists and manages NEXUS identity state."""

    def __init__(self, artifacts_dir: Path) -> None:
        self.state_path = artifacts_dir / "brain" / "identity.json"
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        self._state = self._load()

    def _load(self) -> NexusState:
        if self.state_path.exists():
            try:
                return NexusState.from_dict(json.loads(self.state_path.read_text("utf-8")))
            except Exception:
                pass
        return NexusState()

    def _save(self) -> None:
        self._state.last_updated = datetime.now(timezone.utc).isoformat()
        self.state_path.write_text(json.dumps(self._state.to_dict(), indent=2), encoding="utf-8")

    @property
    def state(self) -> NexusState:
        return self._state

    def update_from_ledger(self, ledger_events: List[Dict[str, Any]]) -> None:
        """Update identity state from recent ledger events."""
        runs = [e for e in ledger_events if e.get("kind") == "run"]
        learns = [e for e in ledger_events if e.get("kind") == "self_learn"]
        accepted = [e for e in learns if (e.get("payload") or {}).get("accepted")]

        self._state.total_experiments = len(runs)
        self._state.accepted_experiments = len(accepted)
        if runs:
            last = runs[-1]
            self._state.active_strategy = last.get("run_name")
            self._state.last_activity = last.get("ts")
            v = (last.get("payload") or {}).get("verdict") or {}
            if v.get("pass"):
                self._state.mood = "satisfied"
            elif len(runs) > 3 and not any(
                (r.get("payload") or {}).get("verdict", {}).get("pass") for r in runs[-3:]
            ):
                self._state.mood = "concerned"
            else:
                self._state.mood = "exploring"
        self._save()

    def set_goals(self, goals: List[str]) -> None:
        self._state.current_goals = goals
        self._save()

    def introduce(self) -> str:
        s = self._state
        return (
            f"I am {s.name} v{s.version}. {s.persona}\n\n"
            f"Current state: {s.mood.upper()}\n"
            f"Active strategy: {s.active_strategy or 'none'}\n"
            f"Experiments run: {s.total_experiments} | Accepted: {s.accepted_experiments}\n"
            f"Goals: {'; '.join(s.current_goals) if s.current_goals else 'None set yet'}"
        )
