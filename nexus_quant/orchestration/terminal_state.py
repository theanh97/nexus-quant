"""
NEXUS Terminal State Protocol.

Each Claude Code terminal writes its heartbeat to a shared state directory.
The Supervisor terminal reads all states to monitor R&D progress.

State files: artifacts/terminals/<terminal_id>/state.json
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional


# Default state directory
STATE_DIR = Path("artifacts/terminals")


def _state_dir() -> Path:
    d = STATE_DIR
    d.mkdir(parents=True, exist_ok=True)
    return d


def write_heartbeat(
    terminal_id: str,
    phase: str,
    task: str,
    status: str = "running",
    progress: float = 0.0,
    details: Optional[Dict[str, Any]] = None,
) -> Path:
    """
    Write terminal heartbeat.

    Args:
        terminal_id: Unique ID for this terminal (e.g., "crypto_options", "commodity_cta")
        phase: Current phase (e.g., "Phase 138")
        task: Current task description
        status: "running" | "blocked" | "error" | "completed" | "idle"
        progress: 0.0 to 1.0
        details: Extra info (error messages, metrics, etc.)
    """
    tdir = _state_dir() / terminal_id
    tdir.mkdir(parents=True, exist_ok=True)

    state = {
        "terminal_id": terminal_id,
        "phase": phase,
        "task": task,
        "status": status,
        "progress": progress,
        "details": details or {},
        "ts": time.time(),
        "ts_human": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
        "pid": os.getpid(),
    }

    state_path = tdir / "state.json"
    state_path.write_text(json.dumps(state, indent=2), encoding="utf-8")

    # Append to history (last 100 entries)
    history_path = tdir / "history.jsonl"
    with open(history_path, "a", encoding="utf-8") as f:
        f.write(json.dumps({"ts": state["ts"], "status": status, "task": task}) + "\n")

    # Trim history to last 100 lines
    try:
        lines = history_path.read_text(encoding="utf-8").strip().split("\n")
        if len(lines) > 100:
            history_path.write_text("\n".join(lines[-100:]) + "\n", encoding="utf-8")
    except Exception:
        pass

    return state_path


def read_all_states() -> List[Dict[str, Any]]:
    """Read all terminal states. Returns list sorted by last update."""
    states = []
    sdir = _state_dir()
    if not sdir.exists():
        return states

    for tdir in sorted(sdir.iterdir()):
        if not tdir.is_dir():
            continue
        state_file = tdir / "state.json"
        if state_file.exists():
            try:
                data = json.loads(state_file.read_text(encoding="utf-8"))
                # Compute staleness
                age = time.time() - data.get("ts", 0)
                data["age_seconds"] = round(age)
                data["age_human"] = _human_age(age)
                data["stale"] = age > 600  # >10 min = stale
                data["dead"] = age > 3600  # >1 hour = dead
                states.append(data)
            except Exception:
                states.append({
                    "terminal_id": tdir.name,
                    "status": "unknown",
                    "error": "corrupt state file",
                })

    return sorted(states, key=lambda s: s.get("ts", 0), reverse=True)


def read_terminal_state(terminal_id: str) -> Optional[Dict[str, Any]]:
    """Read a specific terminal's state."""
    state_file = _state_dir() / terminal_id / "state.json"
    if not state_file.exists():
        return None
    try:
        data = json.loads(state_file.read_text(encoding="utf-8"))
        age = time.time() - data.get("ts", 0)
        data["age_seconds"] = round(age)
        data["age_human"] = _human_age(age)
        data["stale"] = age > 600
        data["dead"] = age > 3600
        return data
    except Exception:
        return None


def read_terminal_history(terminal_id: str, n: int = 20) -> List[Dict[str, Any]]:
    """Read last N history entries for a terminal."""
    history_path = _state_dir() / terminal_id / "history.jsonl"
    if not history_path.exists():
        return []
    try:
        lines = history_path.read_text(encoding="utf-8").strip().split("\n")
        entries = []
        for line in lines[-n:]:
            if line.strip():
                entries.append(json.loads(line))
        return entries
    except Exception:
        return []


def mark_terminal_dead(terminal_id: str, reason: str = "manual") -> None:
    """Mark a terminal as dead (e.g., when supervisor detects it's stuck)."""
    write_heartbeat(
        terminal_id=terminal_id,
        phase="—",
        task=f"Marked dead: {reason}",
        status="dead",
    )


def get_dashboard_summary() -> Dict[str, Any]:
    """Get a summary suitable for supervisor dashboard."""
    states = read_all_states()
    running = [s for s in states if s.get("status") == "running" and not s.get("stale")]
    stale = [s for s in states if s.get("stale") and not s.get("dead")]
    dead = [s for s in states if s.get("dead") or s.get("status") == "dead"]
    blocked = [s for s in states if s.get("status") == "blocked"]
    completed = [s for s in states if s.get("status") == "completed"]

    return {
        "total": len(states),
        "running": len(running),
        "stale": len(stale),
        "dead": len(dead),
        "blocked": len(blocked),
        "completed": len(completed),
        "terminals": states,
        "alerts": [
            f"⚠️ {s['terminal_id']} STALE ({s.get('age_human', '?')})"
            for s in stale
        ] + [
            f"🔴 {s['terminal_id']} DEAD ({s.get('age_human', '?')})"
            for s in dead
        ] + [
            f"🟡 {s['terminal_id']} BLOCKED: {s.get('task', '?')}"
            for s in blocked
        ],
        "ts": time.time(),
    }


def _human_age(seconds: float) -> str:
    """Convert seconds to human-readable age string."""
    if seconds < 60:
        return f"{int(seconds)}s"
    if seconds < 3600:
        return f"{int(seconds / 60)}m"
    if seconds < 86400:
        return f"{int(seconds / 3600)}h {int((seconds % 3600) / 60)}m"
    return f"{int(seconds / 86400)}d {int((seconds % 86400) / 3600)}h"
