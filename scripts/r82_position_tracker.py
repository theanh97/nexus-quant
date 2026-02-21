#!/usr/bin/env python3
"""
R82: Position Tracker & P&L Monitor
======================================

Tracks the current BF position and cumulative P&L over time.
Designed to work with R81 (daily signal runner) to form a complete
production system.

Features:
  1. Track BF position changes (entry, reversal, hold)
  2. Compute daily mark-to-market P&L
  3. Track cumulative P&L with equity curve
  4. Position history and trade log
  5. Drawdown monitoring
  6. Performance attribution (VRP vs BF contribution)

State is persisted in a JSON file between runs.

Usage:
  python3 scripts/r82_position_tracker.py                    # Run daily update
  python3 scripts/r82_position_tracker.py --status           # Show current status
  python3 scripts/r82_position_tracker.py --history          # Show trade history
  python3 scripts/r82_position_tracker.py --reset            # Reset state (WARNING)
"""
import csv
import json
import math
import subprocess
import sys
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STATE_PATH = ROOT / "data" / "cache" / "deribit" / "real_surface" / "position_state.json"

STATUS_MODE = "--status" in sys.argv
HISTORY_MODE = "--history" in sys.argv
RESET_MODE = "--reset" in sys.argv

# Production config (must match R81)
CONFIG = {
    "w_vrp": 0.10,
    "w_bf": 0.90,
    "vrp_leverage": 2.0,
    "bf_lookback": 120,
    "bf_z_entry": 1.5,
    "bf_z_exit": 0.0,
    "bf_sensitivity": 2.5,
}


# ═══════════════════════════════════════════════════════════════
# Data Loading (same as R81)
# ═══════════════════════════════════════════════════════════════

def load_dvol_history(currency: str) -> dict:
    path = ROOT / "data" / "cache" / "deribit" / "dvol" / f"{currency}_DVOL_12h.csv"
    daily = {}
    if not path.exists():
        return daily
    with open(path) as f:
        for row in csv.DictReader(f):
            daily[row["date"][:10]] = float(row["dvol_close"]) / 100.0
    return daily


def load_price_history(currency: str) -> dict:
    prices = {}
    start_dt = datetime(2020, 1, 1, tzinfo=timezone.utc)
    end_dt = datetime(2026, 3, 1, tzinfo=timezone.utc)
    current = start_dt
    while current < end_dt:
        chunk_end = min(current + timedelta(days=365), end_dt)
        url = (f"https://www.deribit.com/api/v2/public/get_tradingview_chart_data?"
               f"instrument_name={currency}-PERPETUAL&resolution=1D"
               f"&start_timestamp={int(current.timestamp()*1000)}"
               f"&end_timestamp={int(chunk_end.timestamp()*1000)}")
        try:
            r = subprocess.run(["curl", "-s", "--max-time", "30", url],
                              capture_output=True, text=True, timeout=40)
            data = json.loads(r.stdout)
            if "result" in data:
                data = data["result"]
            if data.get("status") == "ok":
                for i, ts in enumerate(data["ticks"]):
                    dt = datetime.fromtimestamp(ts/1000, tz=timezone.utc)
                    prices[dt.strftime("%Y-%m-%d")] = data["close"][i]
        except:
            pass
        current = chunk_end
        time.sleep(0.1)
    return prices


def load_surface(currency: str) -> dict:
    path = ROOT / "data" / "cache" / "deribit" / "real_surface" / f"{currency}_daily_surface.csv"
    data = {}
    if not path.exists():
        return data
    with open(path) as f:
        for row in csv.DictReader(f):
            d = row["date"]
            entry = {}
            for field in ["butterfly_25d", "iv_atm", "skew_25d"]:
                val = row.get(field, "")
                if val and val != "None":
                    entry[field] = float(val)
            if entry:
                data[d] = entry
    return data


# ═══════════════════════════════════════════════════════════════
# State Management
# ═══════════════════════════════════════════════════════════════

def load_state():
    """Load persistent state from disk."""
    if STATE_PATH.exists():
        with open(STATE_PATH) as f:
            return json.load(f)
    return default_state()


def save_state(state):
    """Save state to disk."""
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(STATE_PATH, "w") as f:
        json.dump(state, f, indent=2)


def default_state():
    """Create default state for a fresh start."""
    return {
        "initialized": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "last_update": None,
        "current_position": {
            "bf_direction": 0,       # -1, 0, or +1
            "bf_entry_date": None,
            "bf_entry_z": None,
            "bf_entry_price": None,  # BF value at entry
        },
        "cumulative_pnl": {
            "bf_pnl": 0.0,
            "vrp_pnl": 0.0,
            "total_pnl": 0.0,
            "peak_pnl": 0.0,
            "current_dd": 0.0,
            "max_dd": 0.0,
        },
        "equity_curve": [],          # List of {"date": d, "pnl": x}
        "trade_log": [],             # List of trade events
        "daily_log": [],             # List of daily P&L records
        "stats": {
            "n_trades": 0,
            "n_days_active": 0,
            "win_days": 0,
            "loss_days": 0,
        },
    }


# ═══════════════════════════════════════════════════════════════
# Position & P&L Update
# ═══════════════════════════════════════════════════════════════

def update_state(state, dvol_hist, price_hist, surface_hist):
    """Run the full backtest from scratch and update state."""
    dates = sorted(set(dvol_hist.keys()) & set(surface_hist.keys()))
    if not dates:
        return state

    bf_vals = {}
    for d in dates:
        if d in surface_hist and "butterfly_25d" in surface_hist[d]:
            bf_vals[d] = surface_hist[d]["butterfly_25d"]

    lb = CONFIG["bf_lookback"]
    dt = 1.0 / 365.0
    position = 0.0
    entry_date = None
    entry_z = None

    # Reset running state
    equity_curve = []
    trade_log = []
    daily_log = []
    cum_bf = 0.0
    cum_vrp = 0.0
    cum_total = 0.0
    peak = 0.0
    max_dd = 0.0
    n_trades = 0
    win_days = 0
    loss_days = 0
    prev_position = 0.0

    for i in range(lb, len(dates)):
        d, dp = dates[i], dates[i-1]
        val = bf_vals.get(d)
        if val is None:
            continue

        window = [bf_vals.get(dates[j]) for j in range(i-lb, i)]
        window = [v for v in window if v is not None]
        if len(window) < lb // 2:
            continue

        mean = sum(window) / len(window)
        std = math.sqrt(sum((v - mean)**2 for v in window) / len(window))
        if std < 1e-8:
            continue

        z = (val - mean) / std

        # Position update
        old_pos = position
        if z > CONFIG["bf_z_entry"]:
            position = -1.0
        elif z < -CONFIG["bf_z_entry"]:
            position = 1.0

        # Log trade if position changed
        if position != old_pos:
            n_trades += 1
            trade_type = "REVERSAL" if old_pos != 0 else "ENTRY"
            trade_log.append({
                "date": d,
                "type": trade_type,
                "from": old_pos,
                "to": position,
                "z_score": round(z, 3),
                "bf_value": round(val, 5),
                "trade_num": n_trades,
            })
            entry_date = d
            entry_z = z

        # Compute daily P&L
        iv = dvol_hist.get(d, 0)
        f_now = bf_vals.get(d)
        f_prev = bf_vals.get(dp)

        bf_day_pnl = 0.0
        if f_now is not None and f_prev is not None and iv > 0 and position != 0:
            bf_day_pnl = position * (f_now - f_prev) * iv * math.sqrt(dt) * CONFIG["bf_sensitivity"]

        # VRP P&L
        vrp_day_pnl = 0.0
        rets = []
        for j in range(max(0, i-30), i):
            p0 = price_hist.get(dates[j])
            p1 = price_hist.get(dates[j+1]) if j+1 < len(dates) else None
            if p0 and p1 and p0 > 0:
                rets.append(math.log(p1 / p0))
        if len(rets) >= 20 and iv > 0:
            rv = math.sqrt(sum(r**2 for r in rets) / len(rets)) * math.sqrt(365)
            vrp_day_pnl = CONFIG["vrp_leverage"] * 0.5 * (iv**2 - rv**2) * dt

        # Portfolio P&L
        port_pnl = CONFIG["w_vrp"] * vrp_day_pnl + CONFIG["w_bf"] * bf_day_pnl
        cum_bf += bf_day_pnl
        cum_vrp += vrp_day_pnl
        cum_total += port_pnl

        peak = max(peak, cum_total)
        dd = cum_total - peak
        max_dd = min(max_dd, dd)

        if port_pnl > 0:
            win_days += 1
        elif port_pnl < 0:
            loss_days += 1

        daily_log.append({
            "date": d,
            "bf_pnl": round(bf_day_pnl * 10000, 2),   # bps
            "vrp_pnl": round(vrp_day_pnl * 10000, 2),  # bps
            "port_pnl": round(port_pnl * 10000, 2),     # bps
            "cum_pnl_pct": round(cum_total * 100, 4),
            "position": position,
            "z_score": round(z, 3),
        })

        equity_curve.append({
            "date": d,
            "cum_pnl_pct": round(cum_total * 100, 4),
        })

    # Update state
    state["last_update"] = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    state["current_position"] = {
        "bf_direction": position,
        "bf_entry_date": entry_date,
        "bf_entry_z": round(entry_z, 3) if entry_z else None,
        "bf_entry_price": round(bf_vals.get(entry_date, 0), 5) if entry_date else None,
    }
    state["cumulative_pnl"] = {
        "bf_pnl_pct": round(cum_bf * 100, 4),
        "vrp_pnl_pct": round(cum_vrp * 100, 4),
        "total_pnl_pct": round(cum_total * 100, 4),
        "peak_pnl_pct": round(peak * 100, 4),
        "current_dd_pct": round((cum_total - peak) * 100, 4),
        "max_dd_pct": round(max_dd * 100, 4),
    }

    # Keep only last 30 days in daily_log, all in equity_curve and trade_log
    state["daily_log"] = daily_log[-30:]
    state["equity_curve"] = equity_curve
    state["trade_log"] = trade_log
    state["stats"] = {
        "n_trades": n_trades,
        "n_days_active": win_days + loss_days,
        "win_days": win_days,
        "loss_days": loss_days,
        "hit_rate_pct": round(win_days / (win_days + loss_days) * 100, 1) if (win_days + loss_days) > 0 else 0,
        "trades_per_year": round(n_trades / (len(dates) / 365), 1) if dates else 0,
    }

    return state


# ═══════════════════════════════════════════════════════════════
# Display Functions
# ═══════════════════════════════════════════════════════════════

def show_status(state):
    """Show current position and P&L status."""
    print("=" * 70)
    print("  POSITION STATUS")
    print("=" * 70)

    pos = state.get("current_position", {})
    pnl = state.get("cumulative_pnl", {})
    stats = state.get("stats", {})

    direction_map = {-1.0: "SHORT_BUTTERFLY", 1.0: "LONG_BUTTERFLY", 0: "FLAT"}
    direction = direction_map.get(pos.get("bf_direction", 0), "UNKNOWN")

    print(f"\n  Last updated: {state.get('last_update', 'Never')}")
    print(f"\n  ── Current Position ──")
    print(f"  Direction:    {direction}")
    print(f"  Entry date:   {pos.get('bf_entry_date', 'N/A')}")
    print(f"  Entry z:      {pos.get('bf_entry_z', 'N/A')}")

    print(f"\n  ── Cumulative P&L ──")
    print(f"  BF P&L:       {pnl.get('bf_pnl_pct', 0):+.4f}%")
    print(f"  VRP P&L:      {pnl.get('vrp_pnl_pct', 0):+.4f}%")
    print(f"  Total P&L:    {pnl.get('total_pnl_pct', 0):+.4f}%")
    print(f"  Peak P&L:     {pnl.get('peak_pnl_pct', 0):+.4f}%")
    print(f"  Current DD:   {pnl.get('current_dd_pct', 0):+.4f}%")
    print(f"  Max DD:       {pnl.get('max_dd_pct', 0):+.4f}%")

    print(f"\n  ── Statistics ──")
    print(f"  Total trades: {stats.get('n_trades', 0)}")
    print(f"  Trades/year:  {stats.get('trades_per_year', 0)}")
    print(f"  Active days:  {stats.get('n_days_active', 0)}")
    print(f"  Win rate:     {stats.get('hit_rate_pct', 0)}%")

    # Recent daily P&L
    daily = state.get("daily_log", [])
    if daily:
        print(f"\n  ── Recent Daily P&L (bps) ──")
        print(f"  {'Date':>12} {'BF':>8} {'VRP':>8} {'Port':>8} {'Cum%':>10} {'Pos':>6} {'z':>6}")
        print(f"  {'─'*12} {'─'*8} {'─'*8} {'─'*8} {'─'*10} {'─'*6} {'─'*6}")
        for entry in daily[-10:]:
            print(f"  {entry['date']:>12} {entry['bf_pnl']:>8.2f} {entry['vrp_pnl']:>8.2f} "
                  f"{entry['port_pnl']:>8.2f} {entry['cum_pnl_pct']:>9.4f}% "
                  f"{entry['position']:>6.0f} {entry['z_score']:>6.2f}")

    print("=" * 70)


def show_history(state):
    """Show trade history."""
    print("=" * 70)
    print("  TRADE HISTORY")
    print("=" * 70)

    trades = state.get("trade_log", [])
    if not trades:
        print("  No trades recorded.")
        return

    print(f"\n  Total trades: {len(trades)}")
    print(f"\n  {'#':>4} {'Date':>12} {'Type':>10} {'From':>6} {'To':>6} {'Z':>7} {'BF Val':>10}")
    print(f"  {'─'*4} {'─'*12} {'─'*10} {'─'*6} {'─'*6} {'─'*7} {'─'*10}")

    for trade in trades:
        print(f"  {trade['trade_num']:>4} {trade['date']:>12} {trade['type']:>10} "
              f"{trade['from']:>6.0f} {trade['to']:>6.0f} {trade['z_score']:>7.3f} "
              f"{trade['bf_value']:>10.5f}")

    # Trade statistics
    reversals = sum(1 for t in trades if t["type"] == "REVERSAL")
    entries = sum(1 for t in trades if t["type"] == "ENTRY")
    print(f"\n  Entries: {entries}, Reversals: {reversals}")

    # Time between trades
    if len(trades) >= 2:
        gaps = []
        for i in range(1, len(trades)):
            d1 = datetime.strptime(trades[i-1]["date"], "%Y-%m-%d")
            d2 = datetime.strptime(trades[i]["date"], "%Y-%m-%d")
            gaps.append((d2 - d1).days)
        avg_gap = sum(gaps) / len(gaps)
        min_gap = min(gaps)
        max_gap = max(gaps)
        print(f"  Avg gap between trades: {avg_gap:.0f} days")
        print(f"  Min/Max gap: {min_gap}/{max_gap} days")

    print("=" * 70)


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════

def main():
    if RESET_MODE:
        state = default_state()
        save_state(state)
        print("  State reset to default.")
        return

    state = load_state()

    if STATUS_MODE:
        show_status(state)
        return

    if HISTORY_MODE:
        show_history(state)
        return

    # Full update
    print("  Loading data and updating position tracker...")
    dvol_hist = load_dvol_history("BTC")
    price_hist = load_price_history("BTC")
    surface_hist = load_surface("BTC")

    state = update_state(state, dvol_hist, price_hist, surface_hist)
    save_state(state)

    # Show status after update
    show_status(state)


if __name__ == "__main__":
    main()
