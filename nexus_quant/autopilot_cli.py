from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

from .orchestration.orion import Orion, OrionConfig
from .orchestration.policy import ResearchPolicy


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
                if bootstrap:
                    orion.bootstrap(include_improve=bool(policy_eval.get("allow_improve", True)))
                else:
                    # Even without --bootstrap flag, enqueue lightweight tasks to avoid idling
                    allow_improve = bool(policy_eval.get("allow_improve", True))
                    orion.bootstrap(include_improve=allow_improve)

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

    hb = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "last": last,
        "tasks": orion.task_store.counts(),
        "policy": policy or {},
        "policy_enqueue": policy_enqueue or {},
        "learning": learning_metrics,
    }
    p = artifacts_dir / "state" / "orion_heartbeat.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(hb, indent=2, sort_keys=True), encoding="utf-8")
