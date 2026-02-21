#!/usr/bin/env python3
"""
R81: Daily Production Runner
================================

Consolidated daily script for production deployment. Runs via cron daily.
Outputs: signal, health, alerts, position recommendation — all in one run.

Integrates:
  - R64 signal generator (VRP + BF signals)
  - R77 health indicator (BF edge monitoring)
  - R80 kill-switch checks (risk management)

Usage:
  python3 scripts/r81_daily_production_runner.py              # Full report
  python3 scripts/r81_daily_production_runner.py --json       # JSON output only
  python3 scripts/r81_daily_production_runner.py --brief      # One-line summary

Cron example (daily at 00:15 UTC):
  15 0 * * * python3 /path/to/scripts/r81_daily_production_runner.py --json >> /path/to/logs/daily_signal.jsonl
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
JSON_MODE = "--json" in sys.argv
BRIEF_MODE = "--brief" in sys.argv

# ═══════════════════════════════════════════════════════════════
# Production Config (R69/R70 — validated by R72-R79)
# ═══════════════════════════════════════════════════════════════

CONFIG = {
    "w_vrp": 0.10,
    "w_bf": 0.90,
    "vrp_leverage": 2.0,
    "bf_lookback": 120,
    "bf_z_entry": 1.5,
    "bf_z_exit": 0.0,
    "bf_sensitivity": 2.5,
    "asset": "BTC",
}

# Kill-switch thresholds (R80)
KILL_SWITCH = {
    "max_dd_pct": 1.4,           # 3× historical MaxDD at sens=2.5
    "min_health": 0.25,          # CRITICAL threshold from R77
    "health_critical_days": 30,  # Days in CRITICAL before halt
    "min_bf_std": 0.002,         # Extreme compression threshold (R78)
    "max_cost_bps": 10,          # Max acceptable execution cost
}


# ═══════════════════════════════════════════════════════════════
# Data Loading
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


def fetch_live_dvol(currency: str = "BTC"):
    url = (f"https://www.deribit.com/api/v2/public/get_volatility_index_data?"
           f"currency={currency}&resolution=3600"
           f"&start_timestamp={int((time.time() - 7200) * 1000)}"
           f"&end_timestamp={int(time.time() * 1000)}")
    try:
        r = subprocess.run(["curl", "-s", "--max-time", "15", url],
                          capture_output=True, text=True, timeout=20)
        data = json.loads(r.stdout)
        if "result" in data:
            data = data["result"]
        if "data" in data and data["data"]:
            last = data["data"][-1]
            return last[4] / 100.0
    except:
        pass
    return None


def fetch_live_price(currency: str = "BTC"):
    url = (f"https://www.deribit.com/api/v2/public/ticker?"
           f"instrument_name={currency}-PERPETUAL")
    try:
        r = subprocess.run(["curl", "-s", "--max-time", "15", url],
                          capture_output=True, text=True, timeout=20)
        data = json.loads(r.stdout)
        if "result" in data:
            result = data["result"]
            return float(result.get("last_price") or result.get("mark_price", 0))
    except:
        pass
    return None


# ═══════════════════════════════════════════════════════════════
# Signal Computation
# ═══════════════════════════════════════════════════════════════

def compute_signals(dvol_hist, price_hist, surface_hist):
    """Compute all signals and health metrics."""
    result = {
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        "config": CONFIG,
    }

    # Common dates
    dates = sorted(set(dvol_hist.keys()) & set(surface_hist.keys()))
    if not dates:
        result["error"] = "No common dates between DVOL and surface data"
        return result

    result["data_range"] = {"start": dates[0], "end": dates[-1], "n_days": len(dates)}

    # BF values
    bf_vals = {}
    for d in dates:
        if d in surface_hist and "butterfly_25d" in surface_hist[d]:
            bf_vals[d] = surface_hist[d]["butterfly_25d"]

    bf_dates = sorted(bf_vals.keys())
    lb = CONFIG["bf_lookback"]

    if len(bf_dates) < lb + 10:
        result["error"] = f"Not enough BF data ({len(bf_dates)} < {lb+10})"
        return result

    # ─── BF Signal ──────────────────────────────────────────
    latest = bf_dates[-1]
    window = [bf_vals[bf_dates[i]] for i in range(len(bf_dates)-lb, len(bf_dates))]
    bf_mean = sum(window) / len(window)
    bf_std = math.sqrt(sum((v - bf_mean)**2 for v in window) / len(window))
    bf_z = (bf_vals[latest] - bf_mean) / bf_std if bf_std > 1e-8 else 0

    if bf_z > CONFIG["bf_z_entry"]:
        bf_signal = "SHORT_BUTTERFLY"
        bf_position = -1.0
    elif bf_z < -CONFIG["bf_z_entry"]:
        bf_signal = "LONG_BUTTERFLY"
        bf_position = 1.0
    else:
        bf_signal = "HOLD"
        bf_position = None  # Keep previous

    result["bf_signal"] = {
        "signal": bf_signal,
        "z_score": round(bf_z, 3),
        "value": round(bf_vals[latest], 5),
        "mean_120d": round(bf_mean, 5),
        "std_120d": round(bf_std, 5),
        "position": bf_position,
        "date": latest,
    }

    # ─── VRP Signal ─────────────────────────────────────────
    latest_iv = dvol_hist.get(latest, 0)
    # 30d RV
    rv = 0
    rv_dates = sorted(price_hist.keys())
    for i in range(len(rv_dates)-1, -1, -1):
        if rv_dates[i] <= latest:
            rets = []
            for j in range(max(0, i-30), i):
                p0 = price_hist.get(rv_dates[j])
                p1 = price_hist.get(rv_dates[j+1]) if j+1 < len(rv_dates) else None
                if p0 and p1 and p0 > 0:
                    rets.append(math.log(p1 / p0))
            if rets:
                rv = math.sqrt(sum(r**2 for r in rets) / len(rets)) * math.sqrt(365)
            break

    vrp = latest_iv - rv
    result["vrp_signal"] = {
        "signal": "SHORT_VOL" if vrp > 0 else "NEUTRAL",
        "iv": round(latest_iv * 100, 1),
        "rv_30d": round(rv * 100, 1),
        "vrp_spread": round(vrp * 100, 1),
        "date": latest,
    }

    # ─── Portfolio Recommendation ──────────────────────────
    actions = []
    if bf_signal == "SHORT_BUTTERFLY":
        actions.append("BF: Enter SHORT butterfly (z > 1.5)")
    elif bf_signal == "LONG_BUTTERFLY":
        actions.append("BF: Enter LONG butterfly (z < -1.5)")
    else:
        actions.append("BF: HOLD current position (no z-score reversal)")

    if vrp > 0:
        actions.append("VRP: SHORT vol (VRP positive)")
    else:
        actions.append("VRP: SHORT vol (carry, even though VRP negative)")

    result["actions"] = actions

    # ─── BF Health (R77 composite) ──────────────────────────
    dt = 1.0 / 365.0
    position = 0.0
    pnl_series = []

    for i in range(lb, len(bf_dates)):
        d, dp = bf_dates[i], bf_dates[i-1]
        val = bf_vals[d]
        w = [bf_vals[bf_dates[j]] for j in range(i-lb, i)]
        m = sum(w) / len(w)
        s = math.sqrt(sum((v - m)**2 for v in w) / len(w))
        if s < 1e-8:
            continue
        zz = (val - m) / s
        if zz > CONFIG["bf_z_entry"]:
            position = -1.0
        elif zz < -CONFIG["bf_z_entry"]:
            position = 1.0

        iv = dvol_hist.get(d, 0)
        f_now = bf_vals.get(d)
        f_prev = bf_vals.get(dp)
        if f_now is not None and f_prev is not None and iv > 0 and position != 0:
            day_pnl = position * (f_now - f_prev) * iv * math.sqrt(dt) * CONFIG["bf_sensitivity"]
            pnl_series.append((d, day_pnl))

    # Health components
    health = None
    if len(pnl_series) >= 180:
        # Component 1: Rolling 90d Sharpe
        w90 = [p[1] for p in pnl_series[-90:]]
        mean90 = sum(w90) / len(w90)
        std90 = math.sqrt(sum((r - mean90)**2 for r in w90) / len(w90))
        sharpe90 = (mean90 * 365) / (std90 * math.sqrt(365)) if std90 > 0 else 0
        c1 = max(0, min(1, (sharpe90 + 2) / 8))

        # Component 2: Rolling 60d hit rate
        hits_60 = [1 if p[1] > 0 else 0 for p in pnl_series[-60:]]
        c2 = sum(hits_60) / len(hits_60)

        # Component 3: BF feature volatility (30d)
        recent_bf = [bf_vals[bf_dates[i]] for i in range(max(0, len(bf_dates)-30), len(bf_dates))]
        if recent_bf:
            bf_m = sum(recent_bf) / len(recent_bf)
            bf_vol = math.sqrt(sum((v - bf_m)**2 for v in recent_bf) / len(recent_bf))
        else:
            bf_vol = 0
        c3 = min(1, bf_vol / 0.010)

        health = (c1 + c2 + c3) / 3.0

        if health > 0.55:
            health_status = "STRONG"
        elif health > 0.40:
            health_status = "MODERATE"
        elif health > 0.25:
            health_status = "WEAK"
        else:
            health_status = "CRITICAL"

        result["health"] = {
            "score": round(health, 3),
            "status": health_status,
            "components": {
                "sharpe_90d": round(sharpe90, 2),
                "hit_rate_60d": round(c2 * 100, 1),
                "bf_vol_30d": round(bf_vol, 5),
            },
            "c1_sharpe_norm": round(c1, 3),
            "c2_hitrate": round(c2, 3),
            "c3_bf_vol": round(c3, 3),
        }

    # ─── Recent Performance ────────────────────────────────
    if len(pnl_series) >= 90:
        recent_90 = [p[1] for p in pnl_series[-90:]]
        recent_180 = [p[1] for p in pnl_series[-min(180, len(pnl_series)):]]
        full = [p[1] for p in pnl_series]

        def stats(rets):
            if len(rets) < 10:
                return {}
            m = sum(rets) / len(rets)
            s = math.sqrt(sum((r - m)**2 for r in rets) / len(rets))
            sharpe = (m * 365) / (s * math.sqrt(365)) if s > 0 else 0
            cum = 0
            peak = 0
            dd = 0
            for r in rets:
                cum += r
                peak = max(peak, cum)
                dd = min(dd, cum - peak)
            return {
                "sharpe": round(sharpe, 2),
                "ann_ret_pct": round(m * 365 * 100, 2),
                "max_dd_pct": round(dd * 100, 3),
                "n_days": len(rets),
            }

        result["performance"] = {
            "recent_90d": stats(recent_90),
            "recent_180d": stats(recent_180),
            "full_sample": stats(full),
        }

    # ─── Kill-Switch Checks ────────────────────────────────
    alerts = []

    # Check 1: Health
    if health is not None and health < KILL_SWITCH["min_health"]:
        alerts.append({
            "level": "CRITICAL",
            "check": "BF health below threshold",
            "value": round(health, 3),
            "threshold": KILL_SWITCH["min_health"],
            "action": "HALT trading if persists for 30 days",
        })

    # Check 2: BF std
    if bf_std < KILL_SWITCH["min_bf_std"]:
        alerts.append({
            "level": "WARNING",
            "check": "BF feature std extremely low",
            "value": round(bf_std, 5),
            "threshold": KILL_SWITCH["min_bf_std"],
            "action": "Monitor closely, consider reducing position",
        })

    # Check 3: 180d Sharpe negative
    if len(pnl_series) >= 180:
        w180 = [p[1] for p in pnl_series[-180:]]
        m180 = sum(w180) / len(w180)
        s180 = math.sqrt(sum((r - m180)**2 for r in w180) / len(w180))
        sharpe180 = (m180 * 365) / (s180 * math.sqrt(365)) if s180 > 0 else 0
        if sharpe180 < 0:
            alerts.append({
                "level": "CRITICAL",
                "check": "Rolling 180d Sharpe negative",
                "value": round(sharpe180, 2),
                "threshold": 0,
                "action": "HALT trading immediately",
            })

    # Check 4: Data staleness
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    days_stale = (datetime.strptime(today, "%Y-%m-%d") - datetime.strptime(latest, "%Y-%m-%d")).days
    if days_stale > 3:
        alerts.append({
            "level": "WARNING",
            "check": "Data is stale",
            "value": f"{days_stale} days old",
            "threshold": "3 days",
            "action": "Check data pipeline, do not trade on stale signals",
        })

    result["alerts"] = alerts
    result["alert_count"] = len(alerts)
    result["all_clear"] = len(alerts) == 0

    # ─── Live market data ──────────────────────────────────
    live_iv = fetch_live_dvol("BTC")
    live_price = fetch_live_price("BTC")
    if live_iv is not None:
        result["live"] = {
            "iv_pct": round(live_iv * 100, 1),
            "price": live_price,
        }

    return result


# ═══════════════════════════════════════════════════════════════
# Output Formatting
# ═══════════════════════════════════════════════════════════════

def print_brief(result):
    """One-line summary for quick checks."""
    bf = result.get("bf_signal", {})
    health = result.get("health", {})
    perf = result.get("performance", {}).get("recent_90d", {})

    status = "OK" if result.get("all_clear") else f"ALERTS({result.get('alert_count', 0)})"
    h_status = health.get("status", "?")
    h_score = health.get("score", 0)

    print(f"[{result['timestamp']}] BF={bf.get('signal','?')} z={bf.get('z_score','?')} "
          f"health={h_score:.2f}({h_status}) sharpe90={perf.get('sharpe','?')} "
          f"status={status}")


def print_full(result):
    """Full human-readable report."""
    print("=" * 70)
    print("  DAILY PRODUCTION SIGNAL REPORT")
    print("=" * 70)
    print(f"  Timestamp: {result['timestamp']}")
    print(f"  Config: {CONFIG['w_vrp']*100:.0f}/{CONFIG['w_bf']*100:.0f} VRP/BF | "
          f"sens={CONFIG['bf_sensitivity']} | z_exit={CONFIG['bf_z_exit']}")

    data = result.get("data_range", {})
    print(f"  Data: {data.get('n_days', 0)} days ({data.get('start', '?')} to {data.get('end', '?')})")

    if result.get("live"):
        print(f"  Live: BTC ${result['live']['price']:,.0f} | IV {result['live']['iv_pct']:.1f}%")

    # BF Signal
    bf = result.get("bf_signal", {})
    print(f"\n  ── BF Signal ──")
    print(f"  Signal:     {bf.get('signal', 'N/A')}")
    print(f"  Z-score:    {bf.get('z_score', 'N/A')}")
    print(f"  BF value:   {bf.get('value', 'N/A')}")
    print(f"  BF 120d μ:  {bf.get('mean_120d', 'N/A')}")
    print(f"  BF 120d σ:  {bf.get('std_120d', 'N/A')}")

    # VRP Signal
    vrp = result.get("vrp_signal", {})
    print(f"\n  ── VRP Signal ──")
    print(f"  Signal:     {vrp.get('signal', 'N/A')}")
    print(f"  IV:         {vrp.get('iv', 'N/A')}%")
    print(f"  RV (30d):   {vrp.get('rv_30d', 'N/A')}%")
    print(f"  VRP spread: {vrp.get('vrp_spread', 'N/A')}%")

    # Actions
    print(f"\n  ── Actions ──")
    for action in result.get("actions", []):
        print(f"  → {action}")

    # Health
    health = result.get("health", {})
    if health:
        print(f"\n  ── BF Health ──")
        print(f"  Score:      {health.get('score', 'N/A')} ({health.get('status', 'N/A')})")
        comps = health.get("components", {})
        print(f"  Sharpe 90d: {comps.get('sharpe_90d', 'N/A')}")
        print(f"  Hit rate:   {comps.get('hit_rate_60d', 'N/A')}%")
        print(f"  BF vol 30d: {comps.get('bf_vol_30d', 'N/A')}")

    # Performance
    perf = result.get("performance", {})
    if perf:
        print(f"\n  ── Performance ──")
        for period, stats in perf.items():
            if stats:
                print(f"  {period}: Sharpe {stats.get('sharpe', '?')} | "
                      f"Ret {stats.get('ann_ret_pct', '?')}% | "
                      f"MaxDD {stats.get('max_dd_pct', '?')}% | "
                      f"N={stats.get('n_days', '?')}")

    # Alerts
    alerts = result.get("alerts", [])
    if alerts:
        print(f"\n  ── ⚠ ALERTS ({len(alerts)}) ──")
        for alert in alerts:
            print(f"  [{alert['level']}] {alert['check']}: {alert['value']} "
                  f"(threshold: {alert['threshold']})")
            print(f"    Action: {alert['action']}")
    else:
        print(f"\n  ── ALL CLEAR ── No alerts. System healthy.")

    print("=" * 70)


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════

def main():
    # Load data
    dvol_hist = load_dvol_history("BTC")
    price_hist = load_price_history("BTC")
    surface_hist = load_surface("BTC")

    # Compute signals
    result = compute_signals(dvol_hist, price_hist, surface_hist)

    # Output
    if JSON_MODE:
        print(json.dumps(result, indent=2))
    elif BRIEF_MODE:
        print_brief(result)
    else:
        print_full(result)

    # Save latest result
    out_path = ROOT / "data" / "cache" / "deribit" / "real_surface" / "latest_signal.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)

    # Append to signal log (JSONL)
    log_path = ROOT / "data" / "cache" / "deribit" / "real_surface" / "signal_log.jsonl"
    with open(log_path, "a") as f:
        f.write(json.dumps(result) + "\n")

    return result


if __name__ == "__main__":
    main()
