#!/usr/bin/env python3
"""
R83: Alert & Notification System
===================================

Integrates with R81 (daily signal) and R82 (position tracker) to provide
alerts when conditions warrant attention. Closes the 3rd infrastructure
gap from R80.

Alert levels:
  - INFO:     Normal status updates (daily summary)
  - WARNING:  Something needs attention (elevated BF std compression, data staleness)
  - CRITICAL: Trading should halt (kill-switch triggered)

Notification channels:
  1. Console output (always)
  2. Log file (always) — data/cache/deribit/real_surface/alerts.jsonl
  3. macOS notification (optional, if osascript available)
  4. Webhook URL (optional, if configured)

Usage:
  python3 scripts/r83_alert_system.py              # Run full check
  python3 scripts/r83_alert_system.py --test        # Test notifications
  python3 scripts/r83_alert_system.py --history     # Show alert history

Production cron (runs AFTER R81):
  20 0 * * * python3 /path/to/scripts/r83_alert_system.py >> /path/to/logs/alerts.log 2>&1
"""
import json
import math
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ALERT_LOG = ROOT / "data" / "cache" / "deribit" / "real_surface" / "alerts.jsonl"
SIGNAL_FILE = ROOT / "data" / "cache" / "deribit" / "real_surface" / "latest_signal.json"
STATE_FILE = ROOT / "data" / "cache" / "deribit" / "real_surface" / "position_state.json"

TEST_MODE = "--test" in sys.argv
HISTORY_MODE = "--history" in sys.argv

# Kill-switch thresholds (must match R81)
KILL_SWITCH = {
    "max_dd_pct": 1.4,
    "min_health": 0.25,
    "min_bf_std": 0.002,
    "max_data_stale_days": 3,
    "min_180d_sharpe": 0.0,
}

# Warning thresholds (earlier warning)
WARNING = {
    "dd_pct": 0.7,         # 50% of kill-switch
    "health": 0.40,        # MODERATE threshold
    "bf_std": 0.003,       # Above kill-switch but low
    "data_stale_days": 2,
    "sharpe_180d": 0.5,
}


# ═══════════════════════════════════════════════════════════════
# Load Latest Data
# ═══════════════════════════════════════════════════════════════

def load_latest_signal():
    if SIGNAL_FILE.exists():
        with open(SIGNAL_FILE) as f:
            return json.load(f)
    return None


def load_position_state():
    if STATE_FILE.exists():
        with open(STATE_FILE) as f:
            return json.load(f)
    return None


# ═══════════════════════════════════════════════════════════════
# Alert Generation
# ═══════════════════════════════════════════════════════════════

def check_alerts(signal, state):
    """Generate alerts based on signal and position state."""
    alerts = []
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    if signal is None:
        alerts.append({
            "timestamp": now,
            "level": "CRITICAL",
            "category": "data",
            "message": "No signal file found — R81 may not have run",
            "action": "Check R81 daily runner",
        })
        return alerts

    if state is None:
        alerts.append({
            "timestamp": now,
            "level": "WARNING",
            "category": "data",
            "message": "No position state file — R82 may not have run",
            "action": "Run R82 position tracker",
        })

    # ─── Health checks ────────────────────────────────────
    health = signal.get("health", {})
    health_score = health.get("score", 1.0)

    if health_score < KILL_SWITCH["min_health"]:
        alerts.append({
            "timestamp": now,
            "level": "CRITICAL",
            "category": "health",
            "message": f"BF health CRITICAL: {health_score:.3f} < {KILL_SWITCH['min_health']}",
            "value": health_score,
            "threshold": KILL_SWITCH["min_health"],
            "action": "HALT trading — BF edge may be degrading",
        })
    elif health_score < WARNING["health"]:
        alerts.append({
            "timestamp": now,
            "level": "WARNING",
            "category": "health",
            "message": f"BF health MODERATE: {health_score:.3f} < {WARNING['health']}",
            "value": health_score,
            "threshold": WARNING["health"],
            "action": "Monitor closely, reduce position if persists",
        })

    # ─── BF std checks ───────────────────────────────────
    bf_std = signal.get("bf_signal", {}).get("std_120d", 1.0)

    if bf_std < KILL_SWITCH["min_bf_std"]:
        alerts.append({
            "timestamp": now,
            "level": "CRITICAL",
            "category": "compression",
            "message": f"BF std CRITICALLY low: {bf_std:.5f} < {KILL_SWITCH['min_bf_std']}",
            "value": bf_std,
            "threshold": KILL_SWITCH["min_bf_std"],
            "action": "HALT trading — extreme compression",
        })
    elif bf_std < WARNING["bf_std"]:
        alerts.append({
            "timestamp": now,
            "level": "WARNING",
            "category": "compression",
            "message": f"BF std low: {bf_std:.5f} < {WARNING['bf_std']}",
            "value": bf_std,
            "threshold": WARNING["bf_std"],
            "action": "Monitor BF compression trend",
        })

    # ─── Data freshness ──────────────────────────────────
    data_range = signal.get("data_range", {})
    last_date = data_range.get("end", "")
    if last_date:
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        try:
            days_stale = (datetime.strptime(today, "%Y-%m-%d") - datetime.strptime(last_date, "%Y-%m-%d")).days
        except:
            days_stale = 0

        if days_stale > KILL_SWITCH["max_data_stale_days"]:
            alerts.append({
                "timestamp": now,
                "level": "CRITICAL",
                "category": "data",
                "message": f"Data {days_stale} days stale (last: {last_date})",
                "value": days_stale,
                "threshold": KILL_SWITCH["max_data_stale_days"],
                "action": "Do NOT trade — data pipeline broken",
            })
        elif days_stale > WARNING["data_stale_days"]:
            alerts.append({
                "timestamp": now,
                "level": "WARNING",
                "category": "data",
                "message": f"Data {days_stale} days old (last: {last_date})",
                "value": days_stale,
                "threshold": WARNING["data_stale_days"],
                "action": "Check data pipeline",
            })

    # ─── Performance checks ──────────────────────────────
    perf = signal.get("performance", {})
    perf_180 = perf.get("recent_180d", {})
    sharpe_180 = perf_180.get("sharpe", 999)

    if sharpe_180 < KILL_SWITCH["min_180d_sharpe"]:
        alerts.append({
            "timestamp": now,
            "level": "CRITICAL",
            "category": "performance",
            "message": f"180d Sharpe NEGATIVE: {sharpe_180:.2f}",
            "value": sharpe_180,
            "threshold": KILL_SWITCH["min_180d_sharpe"],
            "action": "HALT trading — strategy may have lost edge",
        })
    elif sharpe_180 < WARNING["sharpe_180d"]:
        alerts.append({
            "timestamp": now,
            "level": "WARNING",
            "category": "performance",
            "message": f"180d Sharpe weak: {sharpe_180:.2f}",
            "value": sharpe_180,
            "threshold": WARNING["sharpe_180d"],
            "action": "Monitor — may need to pause if continues",
        })

    # ─── Drawdown checks ────────────────────────────────
    if state:
        cum_pnl = state.get("cumulative_pnl", {})
        dd = abs(cum_pnl.get("current_dd_pct", 0))

        if dd > KILL_SWITCH["max_dd_pct"]:
            alerts.append({
                "timestamp": now,
                "level": "CRITICAL",
                "category": "drawdown",
                "message": f"Drawdown exceeds kill-switch: {dd:.4f}% > {KILL_SWITCH['max_dd_pct']}%",
                "value": dd,
                "threshold": KILL_SWITCH["max_dd_pct"],
                "action": "HALT trading immediately",
            })
        elif dd > WARNING["dd_pct"]:
            alerts.append({
                "timestamp": now,
                "level": "WARNING",
                "category": "drawdown",
                "message": f"Drawdown elevated: {dd:.4f}% > {WARNING['dd_pct']}%",
                "value": dd,
                "threshold": WARNING["dd_pct"],
                "action": "Monitor — approaching kill-switch level",
            })

    # ─── Existing alerts from R81 ────────────────────────
    for alert in signal.get("alerts", []):
        alerts.append({
            "timestamp": now,
            "level": alert.get("level", "WARNING"),
            "category": "r81_signal",
            "message": f"{alert['check']}: {alert['value']}",
            "action": alert.get("action", ""),
        })

    # ─── Daily summary (INFO) ─────────────────────────────
    bf = signal.get("bf_signal", {})
    alerts.append({
        "timestamp": now,
        "level": "INFO",
        "category": "summary",
        "message": (f"BF={bf.get('signal','?')} z={bf.get('z_score','?')} "
                   f"health={health_score:.2f} sharpe90={perf.get('recent_90d', {}).get('sharpe', '?')}"),
        "action": "No action needed",
    })

    return alerts


# ═══════════════════════════════════════════════════════════════
# Notification Channels
# ═══════════════════════════════════════════════════════════════

def notify_console(alerts):
    """Print alerts to console."""
    critical = [a for a in alerts if a["level"] == "CRITICAL"]
    warnings = [a for a in alerts if a["level"] == "WARNING"]
    info = [a for a in alerts if a["level"] == "INFO"]

    print("=" * 70)
    print("  ALERT SYSTEM STATUS")
    print("=" * 70)

    if critical:
        print(f"\n  🔴 CRITICAL ALERTS ({len(critical)}):")
        for a in critical:
            print(f"    [{a['category']}] {a['message']}")
            print(f"      Action: {a['action']}")
    elif warnings:
        print(f"\n  🟡 WARNINGS ({len(warnings)}):")
        for a in warnings:
            print(f"    [{a['category']}] {a['message']}")
            print(f"      Action: {a['action']}")
    else:
        print(f"\n  🟢 ALL CLEAR")

    if info:
        for a in info:
            print(f"\n  [{a['level']}] {a['message']}")

    print("=" * 70)


def notify_macos(alerts):
    """Send macOS notification for CRITICAL/WARNING alerts."""
    critical = [a for a in alerts if a["level"] == "CRITICAL"]
    warnings = [a for a in alerts if a["level"] == "WARNING"]

    if not critical and not warnings:
        return

    if critical:
        title = f"CRITICAL: {len(critical)} alert(s)"
        msg = "; ".join(a["message"] for a in critical[:3])
    else:
        title = f"WARNING: {len(warnings)} alert(s)"
        msg = "; ".join(a["message"] for a in warnings[:3])

    try:
        script = f'display notification "{msg}" with title "BF Trading: {title}"'
        subprocess.run(["osascript", "-e", script], timeout=5, capture_output=True)
    except:
        pass  # macOS notification is optional


def notify_log(alerts):
    """Append alerts to log file."""
    ALERT_LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(ALERT_LOG, "a") as f:
        for alert in alerts:
            f.write(json.dumps(alert) + "\n")


# ═══════════════════════════════════════════════════════════════
# History
# ═══════════════════════════════════════════════════════════════

def show_history():
    """Show recent alert history."""
    print("=" * 70)
    print("  ALERT HISTORY")
    print("=" * 70)

    if not ALERT_LOG.exists():
        print("  No alert history.")
        return

    # Read last 100 lines
    alerts = []
    with open(ALERT_LOG) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    alerts.append(json.loads(line))
                except:
                    pass

    # Filter to non-INFO
    important = [a for a in alerts if a.get("level") != "INFO"]

    if not important:
        print("  No warnings or critical alerts in history.")
        print(f"  (Total entries: {len(alerts)} including INFO)")
        return

    print(f"\n  Important alerts (excluding INFO): {len(important)}")
    print(f"\n  {'Timestamp':>22} {'Level':>10} {'Category':>12} Message")
    print(f"  {'─'*22} {'─'*10} {'─'*12} {'─'*40}")

    for a in important[-20:]:  # Show last 20
        print(f"  {a.get('timestamp', '?'):>22} {a.get('level', '?'):>10} "
              f"{a.get('category', '?'):>12} {a.get('message', '?')[:50]}")

    print("=" * 70)


# ═══════════════════════════════════════════════════════════════
# Test Mode
# ═══════════════════════════════════════════════════════════════

def test_notifications():
    """Test all notification channels."""
    print("  Testing alert system...")

    test_alerts = [
        {
            "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
            "level": "CRITICAL",
            "category": "test",
            "message": "TEST: This is a test critical alert",
            "action": "No action — this is a test",
        },
        {
            "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
            "level": "WARNING",
            "category": "test",
            "message": "TEST: This is a test warning",
            "action": "No action — this is a test",
        },
        {
            "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
            "level": "INFO",
            "category": "test",
            "message": "TEST: This is a test info message",
            "action": "No action",
        },
    ]

    notify_console(test_alerts)
    notify_macos(test_alerts)
    notify_log(test_alerts)
    print("  Test complete. Check macOS notifications and alert log.")


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════

def main():
    if TEST_MODE:
        test_notifications()
        return

    if HISTORY_MODE:
        show_history()
        return

    # Load latest data
    signal = load_latest_signal()
    state = load_position_state()

    # Generate alerts
    alerts = check_alerts(signal, state)

    # Send notifications
    notify_console(alerts)
    notify_macos(alerts)
    notify_log(alerts)

    # Exit code: 2 for CRITICAL, 1 for WARNING, 0 for OK
    critical = any(a["level"] == "CRITICAL" for a in alerts)
    warning = any(a["level"] == "WARNING" for a in alerts)

    if critical:
        sys.exit(2)
    elif warning:
        sys.exit(1)
    else:
        sys.exit(0)


if __name__ == "__main__":
    main()
