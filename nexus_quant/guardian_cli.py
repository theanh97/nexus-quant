from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from .orchestration.tasks import TaskStore
from .utils.time import parse_iso_utc


def guardian_main(*, artifacts_dir: Path, stale_seconds: int) -> int:
    hb_path = artifacts_dir / "state" / "orion_heartbeat.json"
    tasks_db = artifacts_dir / "state" / "tasks.db"

    now = int(datetime.now(timezone.utc).timestamp())
    stale = max(30, int(stale_seconds))

    hb = None
    if hb_path.exists():
        try:
            hb = json.loads(hb_path.read_text(encoding="utf-8"))
        except Exception:
            hb = None

    counts = {}
    if tasks_db.exists():
        s = TaskStore(tasks_db)
        try:
            counts = s.counts()
        finally:
            s.close()

    print(f"Artifacts: {artifacts_dir}")
    print(f"Heartbeat: {hb_path}")
    print(f"Tasks DB: {tasks_db}")
    print("")

    if hb is None:
        print("Guardian verdict: STALE (no heartbeat)")
        _print_counts(counts)
        print("")
        print("Action: start autopilot with `python3 -m nexus_quant autopilot --bootstrap --loop ...`")
        return 0

    ts = hb.get("ts")
    age = None
    try:
        age = now - parse_iso_utc(str(ts))
    except Exception:
        age = None

    if age is None:
        print("Guardian verdict: UNKNOWN (invalid heartbeat timestamp)")
        _print_counts(counts)
        return 0

    verdict = "OK" if age <= stale else "STALE"
    print(f"Guardian verdict: {verdict}")
    print(f"- heartbeat_age_seconds: {age}")
    print(f"- stale_threshold_seconds: {stale}")
    last = hb.get("last") or {}
    print(f"- last: {json.dumps(last, sort_keys=True)}")
    _print_counts(counts)

    if verdict == "STALE":
        print("")
        print("Action: restart autopilot (or investigate why it stopped).")

    # Anomaly detection on latest run result
    try:
        from .monitoring.anomaly import run_all_checks
        runs_dir = artifacts_dir / "runs"
        if runs_dir.exists():
            run_dirs = sorted([d for d in runs_dir.iterdir() if d.is_dir()], key=lambda d: d.stat().st_mtime)
            if run_dirs:
                import json as _json
                rp = run_dirs[-1] / "result.json"
                if rp.exists():
                    result = _json.loads(rp.read_text())
                    returns = result.get("returns") or []
                    equity = result.get("equity_curve") or []
                    if returns:
                        anomaly_result = run_all_checks(
                            equity_curve=equity,
                            returns=returns,
                            funding_rates={},
                            actual_cost_rate=0.0,
                            expected_cost_rate=0.001,
                        )
                        if anomaly_result.get("anomalies"):
                            print(f"[GUARDIAN] Anomalies: {anomaly_result['summary']}")
                            for a in anomaly_result["anomalies"]:
                                sev = a.get("severity","?").upper()
                                print(f"  [{sev}] {a.get('kind')}: {a.get('message')}")
                        else:
                            print("[GUARDIAN] No anomalies detected in latest run.")
    except Exception as e:
        print(f"[GUARDIAN] Anomaly check failed: {e}")

    return 0


def _print_counts(counts: Dict[str, Any]) -> None:
    if not counts:
        print("- tasks: (none)")
        return
    print(f"- tasks: {json.dumps(counts, sort_keys=True)}")
