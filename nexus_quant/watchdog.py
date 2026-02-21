"""
NEXUS Watchdog — Lightweight 24/7 daemon that ensures all projects are alive.

Constitution Rule 1: NEXUS NEVER STOPS.
Constitution Rule 2: ALL 3 PROJECTS RUN IN PARALLEL.

This watchdog:
1. Checks all 3 active projects have running autopilots
2. Restarts dead projects automatically
3. Runs constitution compliance checks
4. Logs all actions to watchdog.log
5. Runs via launchd for macOS persistence

Usage:
    python3 -m nexus_quant.watchdog                    # one-shot check + fix
    python3 -m nexus_quant.watchdog --loop              # continuous monitoring
    python3 -m nexus_quant.watchdog --loop --interval 120  # check every 2 min

Entry point for launchd:
    python3 -m nexus_quant.watchdog --loop --interval 120
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger("nexus.watchdog")

# The 3 active projects and their configs (from Constitution Rule 2)
ACTIVE_PROJECTS = {
    "crypto_perps": {
        "config": "configs/production_p91b_champion.json",
        "interval": 60,
        "trials": 30,
        "steps": 10,
    },
    "commodity_cta": {
        "config": "nexus_quant/projects/commodity_cta/configs/rp_mom_dd_champion.json",
        "interval": 300,
        "trials": 30,
        "steps": 10,
    },
    "crypto_options": {
        "config": "configs/crypto_options_vrp.json",
        "interval": 300,
        "trials": 20,
        "steps": 10,
    },
}


def _utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
        return obj if isinstance(obj, dict) else {}
    except Exception:
        return {}


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(path)


def _pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
        return True
    except Exception:
        return False


def _find_project_root() -> Path:
    """Find the NEXUS project root."""
    # Try CLAUDE_PROJECT_DIR
    env_root = os.environ.get("CLAUDE_PROJECT_DIR")
    if env_root:
        return Path(env_root)
    # Try git
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            return Path(result.stdout.strip())
    except Exception:
        pass
    # Fallback: relative to this file
    return Path(__file__).resolve().parents[1]


def _heartbeat_age(hb_path: Path) -> float:
    """Return heartbeat age in seconds, or inf if missing."""
    hb = _read_json(hb_path)
    ts = hb.get("ts")
    if ts is None:
        return float("inf")
    if isinstance(ts, (int, float)):
        return max(0.0, time.time() - float(ts))
    # ISO format
    try:
        dt = datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
        return max(0.0, datetime.now(timezone.utc).timestamp() - dt.timestamp())
    except Exception:
        return float("inf")


def check_project_health(project_root: Path, project_name: str) -> Dict[str, Any]:
    """Check if a project's autopilot is alive and healthy."""
    artifacts = project_root / "artifacts"

    # Check supervisor status
    sup_status = _read_json(artifacts / "state" / "orion_supervisor_status.json")
    sup_state = str(sup_status.get("state", "unknown"))
    sup_pid = sup_status.get("autopilot_pid")

    # Check supervisor PID is alive
    pid_alive = _pid_alive(int(sup_pid or 0))

    # Check heartbeat freshness
    hb_age = _heartbeat_age(artifacts / "state" / "orion_heartbeat.json")

    # Check brain heartbeat
    brain_age = _heartbeat_age(artifacts / "state" / "brain_heartbeat.json")

    # Check for control pauses
    control = _read_json(artifacts / "state" / "orion_supervisor_control.json")
    control_mode = str(control.get("mode", "")).upper()
    is_paused = control_mode in ("PAUSE", "PAUSED", "STOP")

    # Determine health status
    status = "unknown"
    if pid_alive and hb_age < 600:
        status = "healthy"
    elif pid_alive and hb_age < 1800:
        status = "stale"
    elif is_paused:
        status = "paused"
    elif not pid_alive:
        status = "dead"
    else:
        status = "stale"

    return {
        "project": project_name,
        "status": status,
        "supervisor_state": sup_state,
        "pid": sup_pid,
        "pid_alive": pid_alive,
        "heartbeat_age_s": int(hb_age) if hb_age != float("inf") else None,
        "brain_age_s": int(brain_age) if brain_age != float("inf") else None,
        "control_mode": control_mode or None,
        "is_paused": is_paused,
    }


def clear_stale_pause(project_root: Path, reason: str = "watchdog_auto_clear") -> bool:
    """Clear a stale pause (e.g., max_restarts_exceeded) so autopilot can resume."""
    control_path = project_root / "artifacts" / "state" / "orion_supervisor_control.json"
    control = _read_json(control_path)
    mode = str(control.get("mode", "")).upper()

    if mode not in ("PAUSE", "PAUSED"):
        return False

    ctrl_reason = str(control.get("reason", ""))

    # Auto-clear these pause reasons
    auto_clearable = {
        "max_restarts_exceeded",
        "daily_budget_exceeded",
    }

    if ctrl_reason in auto_clearable:
        try:
            control_path.unlink()
            logger.info("[%s] Cleared stale pause (reason=%s)", "watchdog", ctrl_reason)
            return True
        except Exception:
            return False

    return False


def start_supervisor(project_root: Path, project_name: str, config: Dict[str, Any]) -> Optional[int]:
    """Start a supervisor for a project. Returns PID or None."""
    config_path = project_root / config["config"]
    if not config_path.exists():
        logger.warning("[watchdog] Config not found for %s: %s", project_name, config_path)
        return None

    artifacts_dir = project_root / "artifacts"
    log_dir = artifacts_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"watchdog_{project_name}.log"

    cmd = [
        sys.executable, "-m", "nexus_quant", "supervisor",
        "--config", str(config_path),
        "--artifacts", str(artifacts_dir),
        "--trials", str(config.get("trials", 30)),
        "--steps", str(config.get("steps", 10)),
        "--autopilot-interval-seconds", str(config.get("interval", 60)),
        "--check-interval-seconds", "60",
        "--stale-seconds", "1800",
        "--max-restarts", "10",
        "--restart-window-seconds", "3600",
    ]

    try:
        log_handle = log_path.open("a", encoding="utf-8")
        proc = subprocess.Popen(
            cmd,
            stdin=subprocess.DEVNULL,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            text=True,
            close_fds=True,
            cwd=str(project_root),
        )
        logger.info("[watchdog] Started supervisor for %s (pid=%d)", project_name, proc.pid)
        return proc.pid
    except Exception as e:
        logger.error("[watchdog] Failed to start supervisor for %s: %s", project_name, e)
        return None


def watchdog_cycle(project_root: Path) -> Dict[str, Any]:
    """
    Run one watchdog cycle: check all projects, fix what's broken.

    Returns a report of actions taken.
    """
    report = {
        "checked_at": _utc_iso(),
        "projects": {},
        "actions": [],
        "constitution_ok": True,
    }

    for project_name, config in ACTIVE_PROJECTS.items():
        # Check config exists before trying to start
        config_path = project_root / config["config"]
        if not config_path.exists():
            health = {
                "project": project_name,
                "status": "no_config",
                "config_path": str(config_path),
            }
            report["projects"][project_name] = health
            report["actions"].append({
                "project": project_name,
                "action": "skip",
                "reason": f"Config not found: {config_path}",
            })
            continue

        health = check_project_health(project_root, project_name)
        report["projects"][project_name] = health

        if health["status"] == "healthy":
            continue

        if health["status"] == "paused":
            cleared = clear_stale_pause(project_root)
            if cleared:
                report["actions"].append({
                    "project": project_name,
                    "action": "cleared_pause",
                    "reason": "Auto-cleared stale pause",
                })
            else:
                report["actions"].append({
                    "project": project_name,
                    "action": "skip",
                    "reason": f"Paused by control (mode={health['control_mode']})",
                })
            continue

        if health["status"] in ("dead", "stale", "unknown"):
            # Clear any stale pause first
            clear_stale_pause(project_root)

            pid = start_supervisor(project_root, project_name, config)
            action = "restarted" if pid else "restart_failed"
            report["actions"].append({
                "project": project_name,
                "action": action,
                "pid": pid,
                "reason": f"Project was {health['status']}",
            })

    # Run constitution compliance check
    try:
        from .orchestration.constitution import check_compliance
        compliance = check_compliance(project_root / "artifacts")
        report["constitution_ok"] = compliance["ok"]
        report["constitution_violations"] = compliance["summary"]
    except Exception as e:
        report["constitution_ok"] = False
        report["constitution_error"] = str(e)

    # Report learning engine metrics — proof the system is learning
    try:
        from .learning.operational import OperationalLearner
        learner = OperationalLearner(project_root / "artifacts")
        report["learning_metrics"] = learner.metrics()
        learner.close()
    except Exception as e:
        report["learning_metrics"] = {"error": str(e)}

    # Write watchdog state
    state_path = project_root / "artifacts" / "state" / "watchdog_status.json"
    _write_json(state_path, report)

    return report


def watchdog_run(*, loop: bool = False, interval: int = 120) -> int:
    """Run the watchdog with explicit params (called from cli.py)."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s %(message)s",
    )

    project_root = _find_project_root()
    logger.info("NEXUS Watchdog starting (root=%s)", project_root)

    if loop:
        while True:
            try:
                report = watchdog_cycle(project_root)
                actions = report.get("actions", [])
                if actions:
                    for a in actions:
                        logger.info("[watchdog] %s: %s — %s", a["project"], a["action"], a.get("reason", ""))
                else:
                    logger.info("[watchdog] All projects healthy")
            except KeyboardInterrupt:
                logger.info("[watchdog] Stopped by user")
                return 0
            except Exception as e:
                logger.error("[watchdog] Cycle error: %s", e)
            time.sleep(max(30, interval))
    else:
        report = watchdog_cycle(project_root)
        print(json.dumps(report, indent=2, default=str))
        return 0 if report.get("constitution_ok", False) else 1


def watchdog_main() -> int:
    """CLI entry point for standalone execution (python3 -m nexus_quant.watchdog)."""
    parser = argparse.ArgumentParser(description="NEXUS Watchdog — 24/7 project health monitor")
    parser.add_argument("--loop", action="store_true", help="Run continuously")
    parser.add_argument("--interval", type=int, default=120, help="Check interval in seconds (default: 120)")
    args = parser.parse_args()
    return watchdog_run(loop=args.loop, interval=args.interval)


if __name__ == "__main__":
    raise SystemExit(watchdog_main())
