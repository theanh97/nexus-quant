from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

from .orchestration.orion import Orion, OrionConfig


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
    try:
        interval = max(5, min(int(interval_seconds), 3600))
        cycles_left = int(max_cycles) if int(max_cycles) > 0 else None

        while True:
            if bootstrap:
                pending = [t for t in orion.task_store.recent(limit=200) if t.status == "pending"]
                if not pending:
                    orion.bootstrap()

            max_steps = max(1, min(int(steps), 200))
            last = None
            for _ in range(max_steps):
                last = orion.run_once()
                _write_heartbeat(artifacts_dir, orion, last)
                if last.get("message") == "no pending tasks":
                    break

            _write_heartbeat(artifacts_dir, orion, last or {"message": "idle"})

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


def _write_heartbeat(artifacts_dir: Path, orion: Orion, last: Dict[str, Any]) -> None:
    hb = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "last": last,
        "tasks": orion.task_store.counts(),
    }
    p = artifacts_dir / "state" / "orion_heartbeat.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(hb, indent=2, sort_keys=True), encoding="utf-8")
