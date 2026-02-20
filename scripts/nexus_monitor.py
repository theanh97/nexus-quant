#!/usr/bin/env python3
"""
NEXUS R&D Monitor — Lightweight 24/7 watchdog.

Uses Gemini API (FREE) or Codex CLI for analysis.
Runs as a background process, checks terminal health every 5 minutes.

Usage:
    python3 scripts/nexus_monitor.py                  # Run once
    python3 scripts/nexus_monitor.py --loop            # Continuous monitoring
    python3 scripts/nexus_monitor.py --loop --interval 300  # Every 5 min

Env vars:
    GEMINI_API_KEY  — Google Gemini API key (free tier)
    NEXUS_MONITOR_MODEL — "gemini" (default) or "codex"
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import urllib.request
import urllib.error
from pathlib import Path
from typing import Any, Dict, List, Optional

# ── Config ──────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
TERMINAL_STATE_DIR = PROJECT_ROOT / "artifacts" / "terminals"
DASHBOARD_URL = "http://localhost:8080"
MONITOR_LOG = PROJECT_ROOT / "artifacts" / "monitor.log"

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
GEMINI_MODEL = "gemini-2.0-flash"
GEMINI_URL = f"https://generativelanguage.googleapis.com/v1beta/openai/chat/completions"

CODEX_CLI = "/Applications/Codex.app/Contents/Resources/app/bin/codex"
GEMINI_CLI = "/opt/homebrew/bin/gemini"


def log(msg: str) -> None:
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    try:
        with open(MONITOR_LOG, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception:
        pass


# ── Terminal State Reader ───────────────────────────────────────────
def read_all_terminal_states() -> List[Dict[str, Any]]:
    states = []
    if not TERMINAL_STATE_DIR.exists():
        return states
    for tdir in sorted(TERMINAL_STATE_DIR.iterdir()):
        if not tdir.is_dir():
            continue
        sf = tdir / "state.json"
        if sf.exists():
            try:
                data = json.loads(sf.read_text("utf-8"))
                age = time.time() - data.get("ts", 0)
                data["age_seconds"] = round(age)
                data["stale"] = age > 600
                data["dead"] = age > 3600
                states.append(data)
            except Exception:
                states.append({"terminal_id": tdir.name, "status": "corrupt"})
    return states


def check_dashboard() -> bool:
    try:
        req = urllib.request.Request(f"{DASHBOARD_URL}/api/system_status", method="GET")
        with urllib.request.urlopen(req, timeout=5) as resp:
            return resp.status == 200
    except Exception:
        return False


def check_brain_heartbeat() -> Optional[Dict[str, Any]]:
    hb = PROJECT_ROOT / "artifacts" / "state" / "brain_heartbeat.json"
    if not hb.exists():
        return None
    try:
        data = json.loads(hb.read_text("utf-8"))
        data["age"] = time.time() - data.get("ts", 0)
        return data
    except Exception:
        return None


# ── AI Analysis (Gemini API — FREE) ────────────────────────────────
def analyze_with_gemini(context: str) -> str:
    """Use Gemini API (free) to analyze monitoring data."""
    if not GEMINI_API_KEY:
        return "[Gemini API key not set — skipping AI analysis]"

    payload = json.dumps({
        "model": GEMINI_MODEL,
        "messages": [
            {"role": "system", "content": (
                "You are NEXUS Monitor, a concise system health analyst. "
                "Analyze the monitoring data and report: "
                "1) Overall health (OK/WARNING/CRITICAL) "
                "2) Any terminals that need attention "
                "3) Recommended action (if any). "
                "Keep response under 200 words. Vietnamese OK."
            )},
            {"role": "user", "content": context},
        ],
        "max_tokens": 300,
    }).encode("utf-8")

    req = urllib.request.Request(
        f"{GEMINI_URL}?key={GEMINI_API_KEY}",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            return data["choices"][0]["message"]["content"]
    except Exception as e:
        return f"[Gemini API error: {e}]"


def analyze_with_gemini_cli(context: str) -> str:
    """Fallback: use Gemini CLI."""
    if not Path(GEMINI_CLI).exists():
        return "[Gemini CLI not available]"
    try:
        result = subprocess.run(
            [GEMINI_CLI, "-p", f"Analyze this NEXUS monitoring data concisely (under 200 words):\n{context}",
             "-m", "gemini-2.0-flash", "-o", "text"],
            capture_output=True, text=True, timeout=90,
        )
        return result.stdout.strip() if result.returncode == 0 else f"[CLI error: {result.stderr[:200]}]"
    except Exception as e:
        return f"[CLI error: {e}]"


def analyze_with_codex(context: str) -> str:
    """Alternative: use Codex CLI."""
    if not Path(CODEX_CLI).exists():
        return "[Codex CLI not available]"
    try:
        result = subprocess.run(
            [CODEX_CLI, "exec", "-c", "sandbox_permissions=disk-full-read-only,network-full-access", "-"],
            input=f"Analyze concisely (under 200 words):\n{context}",
            capture_output=True, text=True, timeout=60,
        )
        return result.stdout.strip() if result.returncode == 0 else f"[Codex error: {result.stderr[:200]}]"
    except Exception as e:
        return f"[Codex error: {e}]"


def get_ai_analysis(context: str, model: str = "gemini") -> str:
    """Get AI analysis with fallback chain: Gemini API → Gemini CLI → Codex."""
    if model == "codex":
        result = analyze_with_codex(context)
        if not result.startswith("["):
            return result

    # Gemini API first (instant, free)
    result = analyze_with_gemini(context)
    if not result.startswith("["):
        return result

    # Fallback: Gemini CLI (slow but works)
    result = analyze_with_gemini_cli(context)
    if not result.startswith("["):
        return result

    # Last fallback: Codex
    return analyze_with_codex(context)


# ── Main Monitor ────────────────────────────────────────────────────
def run_check(use_ai: bool = True, model: str = "gemini") -> Dict[str, Any]:
    """Run a single monitoring check."""
    log("=== NEXUS Monitor Check ===")

    # 1. Terminal states
    terminals = read_all_terminal_states()
    running = [t for t in terminals if t.get("status") == "running" and not t.get("stale")]
    stale = [t for t in terminals if t.get("stale") and not t.get("dead")]
    dead = [t for t in terminals if t.get("dead") or t.get("status") == "dead"]
    blocked = [t for t in terminals if t.get("status") == "blocked"]

    # 2. Dashboard
    dashboard_ok = check_dashboard()

    # 3. Brain heartbeat
    brain = check_brain_heartbeat()
    brain_ok = brain is not None and brain.get("age", 9999) < 1200

    # Build report
    report = {
        "ts": time.time(),
        "terminals": {"total": len(terminals), "running": len(running), "stale": len(stale), "dead": len(dead), "blocked": len(blocked)},
        "dashboard": "up" if dashboard_ok else "down",
        "brain": "ok" if brain_ok else ("stale" if brain else "no_heartbeat"),
        "alerts": [],
    }

    # Alerts
    for t in stale:
        alert = f"STALE: {t['terminal_id']} (last update {t.get('age_seconds', 0)}s ago)"
        report["alerts"].append(alert)
        log(f"  ⚠️  {alert}")

    for t in dead:
        alert = f"DEAD: {t['terminal_id']} (last update {t.get('age_seconds', 0)}s ago)"
        report["alerts"].append(alert)
        log(f"  🔴 {alert}")

    for t in blocked:
        alert = f"BLOCKED: {t['terminal_id']} — {t.get('task', '?')}"
        report["alerts"].append(alert)
        log(f"  🟡 {alert}")

    if not dashboard_ok:
        report["alerts"].append("Dashboard is DOWN")
        log("  🔴 Dashboard is DOWN")

    if not brain_ok:
        report["alerts"].append(f"Brain heartbeat: {report['brain']}")
        log(f"  ⚠️  Brain: {report['brain']}")

    # Summary
    if not report["alerts"]:
        log("  ✅ All systems healthy")
        report["health"] = "OK"
    elif any("DEAD" in a or "DOWN" in a for a in report["alerts"]):
        report["health"] = "CRITICAL"
    else:
        report["health"] = "WARNING"

    # AI analysis (optional)
    if use_ai and report["alerts"]:
        context = json.dumps(report, indent=2)
        log("  Requesting AI analysis...")
        analysis = get_ai_analysis(context, model=model)
        report["ai_analysis"] = analysis
        log(f"  AI: {analysis[:200]}")

    # Save report
    report_dir = PROJECT_ROOT / "artifacts" / "monitor"
    report_dir.mkdir(parents=True, exist_ok=True)
    ts_str = time.strftime("%Y%m%dT%H%M%S")
    report_path = report_dir / f"check_{ts_str}.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    # Keep only last 100 reports
    reports = sorted(report_dir.glob("check_*.json"))
    for old in reports[:-100]:
        old.unlink()

    return report


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="NEXUS R&D Monitor")
    parser.add_argument("--loop", action="store_true", help="Run continuously")
    parser.add_argument("--interval", type=int, default=300, help="Check interval (seconds)")
    parser.add_argument("--no-ai", action="store_true", help="Skip AI analysis")
    parser.add_argument("--model", default="gemini", choices=["gemini", "codex"], help="AI model for analysis")
    args = parser.parse_args()

    log(f"NEXUS Monitor started (loop={args.loop}, interval={args.interval}s, model={args.model})")

    if args.loop:
        while True:
            try:
                run_check(use_ai=not args.no_ai, model=args.model)
            except KeyboardInterrupt:
                log("Monitor stopped (Ctrl+C)")
                break
            except Exception as e:
                log(f"Monitor error: {e}")
            time.sleep(args.interval)
    else:
        result = run_check(use_ai=not args.no_ai, model=args.model)
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
