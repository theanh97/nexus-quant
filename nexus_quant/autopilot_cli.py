from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

from .orchestration.orion import Orion, OrionConfig
from .orchestration.policy import ResearchPolicy


def _should_run_learn(artifacts_dir: Path) -> bool:
    """Check if daily research should run (>20h since last fetch)."""
    brief_path = artifacts_dir / "brain" / "daily_brief.json"
    if not brief_path.exists():
        return True
    try:
        brief = json.loads(brief_path.read_text("utf-8"))
        last_ts = brief.get("ts", "")
        if last_ts:
            last = datetime.fromisoformat(last_ts)
            age_hours = (datetime.now(timezone.utc) - last).total_seconds() / 3600
            return age_hours >= 20  # Run if >20h old (buffer for timing drift)
    except Exception:
        return True
    return True


def _get_research_status(artifacts_dir: Path) -> Dict[str, Any]:
    """Get research pipeline status for heartbeat."""
    brief_path = artifacts_dir / "brain" / "daily_brief.json"
    if not brief_path.exists():
        return {"status": "never_run", "last_fetch": None, "age_hours": -1}
    try:
        brief = json.loads(brief_path.read_text("utf-8"))
        last_ts = brief.get("ts", "")
        age_hours = -1.0
        if last_ts:
            last = datetime.fromisoformat(last_ts)
            age_hours = round((datetime.now(timezone.utc) - last).total_seconds() / 3600, 1)
        return {
            "status": "stale" if age_hours > 48 else ("due" if age_hours > 20 else "fresh"),
            "last_fetch": last_ts,
            "age_hours": age_hours,
            "sources_count": brief.get("stats", {}).get("fetched_sources", 0),
            "hypotheses_count": len(brief.get("hypotheses", [])),
            "total_items": brief.get("total_items", 0),
        }
    except Exception:
        return {"status": "error", "last_fetch": None, "age_hours": -1}


def autopilot_main(
    *,
    config_path: Path,
    artifacts_dir: Path,
    trials: int,
    bootstrap: bool,
    steps: int,
    loop: bool,
    interval_seconds: int,
    max_cycles: int,
) -> int:
    if not config_path.exists():
        raise SystemExit(f"Config not found: {config_path}")
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    orion = Orion(OrionConfig(artifacts_dir=artifacts_dir, config_path=config_path, trials=int(trials)))
    policy = ResearchPolicy(artifacts_dir=artifacts_dir)
    try:
        interval = max(5, min(int(interval_seconds), 3600))
        cycles_left = int(max_cycles) if int(max_cycles) > 0 else None

        while True:
            policy_eval = policy.evaluate()
            policy_enqueue = orion.enqueue_policy_actions(policy_eval.get("actions") or [])

            # Constitution Rule 1: NEVER STOP — always bootstrap when queue empty
            pending = [t for t in orion.task_store.recent(limit=200) if t.status == "pending"]
            if not pending:
                allow_improve = bool(policy_eval.get("allow_improve", True))
                orion.bootstrap(include_improve=allow_improve)

                # Frequency gate: remove `learn` task if research ran recently
                if not _should_run_learn(artifacts_dir):
                    # Cancel the learn task we just enqueued (it's the most recent pending)
                    fresh_pending = [t for t in orion.task_store.recent(limit=10) if t.status == "pending" and t.kind == "learn"]
                    for t in fresh_pending:
                        orion.task_store.mark_done(t.id, {"skipped": True, "reason": "research_still_fresh"})

            max_steps = max(1, min(int(steps), 200))
            last = None
            for _ in range(max_steps):
                last = orion.run_once()
                _write_heartbeat(artifacts_dir, orion, last, policy=policy_eval, policy_enqueue=policy_enqueue)
                if last.get("message") == "no pending tasks":
                    # Constitution Rule 1: Don't break — re-bootstrap and continue
                    orion.bootstrap(include_improve=bool(policy_eval.get("allow_improve", True)))
                    continue

            _write_heartbeat(
                artifacts_dir,
                orion,
                last or {"message": "idle"},
                policy=policy_eval,
                policy_enqueue=policy_enqueue,
            )

            if not loop:
                break

            if cycles_left is not None:
                cycles_left -= 1
                if cycles_left <= 0:
                    break

            time.sleep(interval)

        return 0
    finally:
        orion.close()


def _write_heartbeat(
    artifacts_dir: Path,
    orion: Orion,
    last: Dict[str, Any],
    *,
    policy: Dict[str, Any] | None = None,
    policy_enqueue: Dict[str, Any] | None = None,
) -> None:
    # Collect learning metrics — proof the system is learning
    learning_metrics = {}
    try:
        learning_metrics = orion.learner.metrics()
    except Exception:
        pass

    # Research pipeline status — proof the system is reading
    research_status = _get_research_status(artifacts_dir)

    hb = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "last": last,
        "tasks": orion.task_store.counts(),
        "policy": policy or {},
        "policy_enqueue": policy_enqueue or {},
        "learning": learning_metrics,
        "research": research_status,
    }
    p = artifacts_dir / "state" / "orion_heartbeat.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(hb, indent=2, sort_keys=True), encoding="utf-8")
