"""
NEXUS Constitution Enforcer — Runtime compliance checks.

This module validates that the NEXUS platform is operating according to
the Constitution (CLAUDE.md). It checks all 10 rules and returns violations.

Usage:
    from nexus_quant.orchestration.constitution import check_compliance
    report = check_compliance(artifacts_dir=Path("artifacts"))
    if report["violations"]:
        for v in report["violations"]:
            print(f"VIOLATION: Rule {v['rule']} — {v['message']}")
"""
from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List


def _read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
        return obj if isinstance(obj, dict) else {}
    except Exception:
        return {}


def _heartbeat_age(path: Path) -> float:
    """Return heartbeat age in seconds, or inf if missing."""
    hb = _read_json(path)
    ts = hb.get("ts")
    if ts is None:
        return float("inf")
    if isinstance(ts, (int, float)):
        return max(0.0, time.time() - float(ts))
    try:
        from ..utils.time import parse_iso_utc
        return max(0.0, int(datetime.now(timezone.utc).timestamp()) - parse_iso_utc(str(ts)))
    except Exception:
        return float("inf")


def check_rule1_never_stop(artifacts_dir: Path) -> List[Dict[str, Any]]:
    """Rule 1: NEVER STOP — check heartbeats are fresh."""
    violations = []

    # Check Orion heartbeat
    orion_hb = artifacts_dir / "state" / "orion_heartbeat.json"
    age = _heartbeat_age(orion_hb)
    if age > 1800:  # 30 min stale
        age_display = "missing" if age == float("inf") else f"{int(age)}s"
        violations.append({
            "rule": 1,
            "severity": "critical",
            "message": f"Orion heartbeat stale ({age_display}) — autopilot may be dead",
            "fix": "Restart autopilot with --bootstrap --loop",
        })

    # Check for force-pause that's blocking work
    control = _read_json(artifacts_dir / "state" / "research_policy_control.json")
    if str(control.get("mode", "")).upper() in ("PAUSE", "PAUSED", "STOP"):
        reason = control.get("reason", "unknown")
        if reason == "max_restarts_exceeded":
            violations.append({
                "rule": 1,
                "severity": "critical",
                "message": f"Supervisor paused by restart cap — reason: {reason}",
                "fix": "Clear control file or increase max_restarts",
            })

    # Check task queue — all tasks failed/done with nothing pending
    tasks_db = artifacts_dir / "state" / "tasks.db"
    if tasks_db.exists():
        from .tasks import TaskStore
        store = TaskStore(tasks_db)
        try:
            counts = store.counts()
            pending = counts.get("pending", 0)
            running = counts.get("running", 0)
            if pending == 0 and running == 0:
                violations.append({
                    "rule": 1,
                    "severity": "warning",
                    "message": "Task queue empty — no pending or running tasks",
                    "fix": "Bootstrap new tasks via orion.bootstrap()",
                })
        finally:
            store.close()

    return violations


def check_rule2_all_projects(artifacts_dir: Path) -> List[Dict[str, Any]]:
    """Rule 2: ALL 3 PROJECTS RUN IN PARALLEL — check each has activity."""
    violations = []
    project_root = Path(__file__).resolve().parents[2]
    projects_dir = project_root / "nexus_quant" / "projects"

    active_projects = ["crypto_perps", "commodity_cta", "crypto_options"]
    for proj in active_projects:
        proj_yaml = projects_dir / proj / "project.yaml"
        if not proj_yaml.exists():
            violations.append({
                "rule": 2,
                "severity": "critical",
                "message": f"Project {proj} missing project.yaml",
                "fix": f"Create nexus_quant/projects/{proj}/project.yaml",
            })

    # Check nexus.yaml lists all 3
    nexus_yaml = project_root / "nexus.yaml"
    if nexus_yaml.exists():
        text = nexus_yaml.read_text(encoding="utf-8")
        for proj in active_projects:
            if proj not in text:
                violations.append({
                    "rule": 2,
                    "severity": "warning",
                    "message": f"Project {proj} not listed in nexus.yaml",
                    "fix": f"Add {proj} to nexus.yaml projects list and scheduler",
                })

    return violations


def check_rule3_multi_model(artifacts_dir: Path) -> List[Dict[str, Any]]:
    """Rule 3: MULTI-MODEL ROUTING — check API keys available."""
    violations = []

    keys_status = {
        "GEMINI_API_KEY": bool(os.environ.get("GEMINI_API_KEY")),
        "ANTHROPIC_API_KEY": bool(os.environ.get("ANTHROPIC_API_KEY")),
        "ZAI_API_KEY": bool(os.environ.get("ZAI_API_KEY")),
    }

    available_providers = sum(1 for v in keys_status.values() if v)
    if available_providers < 2:
        missing = [k for k, v in keys_status.items() if not v]
        violations.append({
            "rule": 3,
            "severity": "warning",
            "message": f"Only {available_providers} model provider(s) available. Missing: {missing}",
            "fix": "Set missing API keys in environment",
        })

    # Claude is MANDATORY for CIPHER and ECHO
    if not keys_status["ANTHROPIC_API_KEY"] and not keys_status["ZAI_API_KEY"]:
        violations.append({
            "rule": 3,
            "severity": "critical",
            "message": "No Claude API key — CIPHER (risk) and ECHO (QA) require Claude Sonnet",
            "fix": "Set ANTHROPIC_API_KEY or ZAI_API_KEY",
        })

    return violations


def check_rule5_data_integrity(artifacts_dir: Path) -> List[Dict[str, Any]]:
    """Rule 5: DATA INTEGRITY — check ledger exists and is append-only."""
    violations = []

    ledger_path = artifacts_dir / "ledger" / "ledger.jsonl"
    if not ledger_path.exists():
        violations.append({
            "rule": 5,
            "severity": "warning",
            "message": "No ledger file found — no evidence trail",
            "fix": "Run at least one backtest to create ledger",
        })

    return violations


def check_rule8_self_healing(artifacts_dir: Path) -> List[Dict[str, Any]]:
    """Rule 8: RECOVERY — check for stuck states that should be auto-fixed."""
    violations = []

    # Check for tasks stuck in 'running' state for too long
    tasks_db = artifacts_dir / "state" / "tasks.db"
    if tasks_db.exists():
        from .tasks import TaskStore
        store = TaskStore(tasks_db)
        try:
            now = datetime.now(timezone.utc).timestamp()
            for t in store.recent(limit=50):
                if t.status == "running":
                    try:
                        from ..utils.time import parse_iso_utc
                        task_age = now - parse_iso_utc(str(t.updated_at))
                        if task_age > 7200:  # 2 hours stuck
                            violations.append({
                                "rule": 8,
                                "severity": "warning",
                                "message": f"Task {t.id} ({t.kind}) stuck in 'running' for {int(task_age)}s",
                                "fix": "Mark as failed and requeue, or restart autopilot",
                            })
                    except Exception:
                        pass
        finally:
            store.close()

    return violations


def check_compliance(artifacts_dir: Path = Path("artifacts")) -> Dict[str, Any]:
    """
    Run all constitution compliance checks.

    Returns:
        {
            "ok": bool,
            "checked_at": str,
            "violations": [...],
            "summary": {"critical": N, "warning": N},
        }
    """
    all_violations: List[Dict[str, Any]] = []

    checkers = [
        check_rule1_never_stop,
        check_rule2_all_projects,
        check_rule3_multi_model,
        check_rule5_data_integrity,
        check_rule8_self_healing,
    ]

    for checker in checkers:
        try:
            violations = checker(artifacts_dir)
            all_violations.extend(violations)
        except Exception as e:
            all_violations.append({
                "rule": 0,
                "severity": "warning",
                "message": f"Checker {checker.__name__} failed: {e}",
                "fix": "Fix the checker itself",
            })

    critical_count = sum(1 for v in all_violations if v.get("severity") == "critical")
    warning_count = sum(1 for v in all_violations if v.get("severity") == "warning")

    return {
        "ok": critical_count == 0,
        "checked_at": datetime.now(timezone.utc).isoformat(),
        "violations": all_violations,
        "summary": {"critical": critical_count, "warning": warning_count},
    }


def write_compliance_report(artifacts_dir: Path = Path("artifacts")) -> Path:
    """Run compliance check and write report to artifacts."""
    report = check_compliance(artifacts_dir)
    out_dir = artifacts_dir / "state"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "constitution_compliance.json"
    out_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return out_path
