#!/usr/bin/env python3
"""
R88: Multi-Asset Production Runner (BTC + ETH)
=================================================

Extends R81 (BTC-only) to run BF signals on both BTC and ETH,
then combines with optimal 70/30 allocation (R86 discovery).

Key findings from R86:
  - BF edge is structural (confirmed on ETH, score 4.0/4.0)
  - BTC-ETH BF correlation: 0.101 (nearly independent!)
  - 70/30 BTC/ETH: Sharpe 4.09 (best risk-adjusted allocation)

Usage:
  python3 scripts/r88_multi_asset_runner.py              # Full report
  python3 scripts/r88_multi_asset_runner.py --json       # JSON output
  python3 scripts/r88_multi_asset_runner.py --brief      # One-line summary

Replaces R81 in the R84 cron pipeline for multi-asset mode.
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
# Multi-Asset Config (R86 validated)
# ═══════════════════════════════════════════════════════════════

ASSETS = {
    "BTC": {"weight": 0.70},
    "ETH": {"weight": 0.30},
}

# Per-asset BF config (same params — R86 confirmed)
BF_CONFIG = {
    "bf_lookback": 120,
    "bf_z_entry": 1.5,
    "bf_z_exit": 0.0,
    "bf_sensitivity": 2.5,
}

VRP_CONFIG = {
    "vrp_leverage": 2.0,
    "w_vrp": 0.10,
    "w_bf": 0.90,
}

KILL_SWITCH = {
    "max_dd_pct": 1.4,
    "min_health": 0.25,
    "health_critical_days": 30,
    "min_bf_std": 0.002,
    "max_cost_bps": 10,
}


# ═══════════════════════════════════════════════════════════════
# Data Loading (reused from R81)
# ═══════════════════════════════════════════════════════════════

def load_dvol_history(currency):
    path = ROOT / "data" / "cache" / "deribit" / "dvol" / f"{currency}_DVOL_12h.csv"
    daily = {}
    if not path.exists():
        return daily
    with open(path) as f:
        for row in csv.DictReader(f):
            daily[row["date"][:10]] = float(row["dvol_close"]) / 100.0
    return daily


def load_surface(currency):
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


def load_price_history(currency):
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
        except Exception:
            pass
        current = chunk_end
        time.sleep(0.1)
    return prices


def fetch_live_dvol(currency):
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
            return data["data"][-1][4] / 100.0
    except Exception:
        pass
    return None


def fetch_live_price(currency):
    url = (f"https://www.deribit.com/api/v2/public/ticker?"
           f"instrument_name={currency}-PERPETUAL")
    try:
        r = subprocess.run(["curl", "-s", "--max-time", "15", url],
                          capture_output=True, text=True, timeout=20)
        data = json.loads(r.stdout)
        if "result" in data:
            result = data["result"]
            return float(result.get("last_price") or result.get("mark_price", 0))
    except Exception:
        pass
    return None


# ═══════════════════════════════════════════════════════════════
# Per-Asset Signal Computation
# ═══════════════════════════════════════════════════════════════

def compute_asset_signal(currency, dvol_hist, price_hist, surface_hist):
    """Compute signals for a single asset."""
    result = {"asset": currency}

    dates = sorted(set(dvol_hist.keys()) & set(surface_hist.keys()))
    if not dates:
        result["error"] = f"No common dates for {currency}"
        return result

    result["data_range"] = {"start": dates[0], "end": dates[-1], "n_days": len(dates)}

    # BF values
    bf_vals = {}
    for d in dates:
        if d in surface_hist and "butterfly_25d" in surface_hist[d]:
            bf_vals[d] = surface_hist[d]["butterfly_25d"]

    bf_dates = sorted(bf_vals.keys())
    lb = BF_CONFIG["bf_lookback"]

    if len(bf_dates) < lb + 10:
        result["error"] = f"Not enough BF data for {currency}"
        return result

    # BF Signal
    latest = bf_dates[-1]
    window = [bf_vals[bf_dates[i]] for i in range(len(bf_dates)-lb, len(bf_dates))]
    bf_mean = sum(window) / len(window)
    bf_std = math.sqrt(sum((v - bf_mean)**2 for v in window) / len(window))
    bf_z = (bf_vals[latest] - bf_mean) / bf_std if bf_std > 1e-8 else 0

    if bf_z > BF_CONFIG["bf_z_entry"]:
        bf_signal = "SHORT_BUTTERFLY"
        bf_position = -1.0
    elif bf_z < -BF_CONFIG["bf_z_entry"]:
        bf_signal = "LONG_BUTTERFLY"
        bf_position = 1.0
    else:
        bf_signal = "HOLD"
        bf_position = None

    result["bf_signal"] = {
        "signal": bf_signal,
        "z_score": round(bf_z, 3),
        "value": round(bf_vals[latest], 5),
        "mean_120d": round(bf_mean, 5),
        "std_120d": round(bf_std, 5),
        "position": bf_position,
        "date": latest,
    }

    # VRP Signal
    latest_iv = dvol_hist.get(latest, 0)
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

    # Health
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
        if zz > BF_CONFIG["bf_z_entry"]:
            position = -1.0
        elif zz < -BF_CONFIG["bf_z_entry"]:
            position = 1.0
        iv = dvol_hist.get(d, 0)
        f_now = bf_vals.get(d)
        f_prev = bf_vals.get(dp)
        if f_now is not None and f_prev is not None and iv > 0 and position != 0:
            day_pnl = position * (f_now - f_prev) * iv * math.sqrt(dt) * BF_CONFIG["bf_sensitivity"]
            pnl_series.append((d, day_pnl))

    health_score = None
    if len(pnl_series) >= 180:
        w90 = [p[1] for p in pnl_series[-90:]]
        mean90 = sum(w90) / len(w90)
        std90 = math.sqrt(sum((r - mean90)**2 for r in w90) / len(w90))
        sharpe90 = (mean90 * 365) / (std90 * math.sqrt(365)) if std90 > 0 else 0
        c1 = max(0, min(1, (sharpe90 + 2) / 8))
        hits_60 = [1 if p[1] > 0 else 0 for p in pnl_series[-60:]]
        c2 = sum(hits_60) / len(hits_60)
        recent_bf = [bf_vals[bf_dates[i]] for i in range(max(0, len(bf_dates)-30), len(bf_dates))]
        bf_m = sum(recent_bf) / len(recent_bf) if recent_bf else 0
        bf_vol = math.sqrt(sum((v - bf_m)**2 for v in recent_bf) / len(recent_bf)) if recent_bf else 0
        c3 = min(1, bf_vol / 0.010)
        health_score = (c1 + c2 + c3) / 3.0

        result["health"] = {
            "score": round(health_score, 3),
            "status": "STRONG" if health_score > 0.55 else "MODERATE" if health_score > 0.40 else "WEAK" if health_score > 0.25 else "CRITICAL",
            "sharpe_90d": round(sharpe90, 2),
            "hit_rate_60d": round(c2 * 100, 1),
            "bf_vol_30d": round(bf_vol, 5),
        }

    # Performance
    if len(pnl_series) >= 90:
        def compute_stats(rets):
            if len(rets) < 10:
                return {}
            m = sum(rets) / len(rets)
            s = math.sqrt(sum((r - m)**2 for r in rets) / len(rets))
            sharpe = (m * 365) / (s * math.sqrt(365)) if s > 0 else 0
            cum = peak = dd = 0
            for r in rets:
                cum += r
                peak = max(peak, cum)
                dd = min(dd, cum - peak)
            return {"sharpe": round(sharpe, 2), "ann_ret_pct": round(m * 365 * 100, 2),
                    "max_dd_pct": round(dd * 100, 3), "n_days": len(rets)}

        result["performance"] = {
            "recent_90d": compute_stats([p[1] for p in pnl_series[-90:]]),
            "full_sample": compute_stats([p[1] for p in pnl_series]),
        }

    # Alerts
    alerts = []
    if health_score is not None and health_score < KILL_SWITCH["min_health"]:
        alerts.append({"level": "CRITICAL", "check": f"{currency} BF health below threshold",
                       "value": round(health_score, 3), "threshold": KILL_SWITCH["min_health"]})
    if bf_std < KILL_SWITCH["min_bf_std"]:
        alerts.append({"level": "WARNING", "check": f"{currency} BF std extremely low",
                       "value": round(bf_std, 5), "threshold": KILL_SWITCH["min_bf_std"]})
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    days_stale = (datetime.strptime(today, "%Y-%m-%d") - datetime.strptime(latest, "%Y-%m-%d")).days
    if days_stale > 3:
        alerts.append({"level": "WARNING", "check": f"{currency} data is stale",
                       "value": f"{days_stale} days old", "threshold": "3 days"})
    result["alerts"] = alerts

    # Live data
    live_iv = fetch_live_dvol(currency)
    live_price = fetch_live_price(currency)
    if live_iv is not None:
        result["live"] = {"iv_pct": round(live_iv * 100, 1), "price": live_price}

    return result


# ═══════════════════════════════════════════════════════════════
# Multi-Asset Combiner
# ═══════════════════════════════════════════════════════════════

def combine_signals(asset_signals):
    """Combine per-asset signals into a unified portfolio view."""
    combined = {
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        "allocation": {name: cfg["weight"] for name, cfg in ASSETS.items()},
        "bf_config": BF_CONFIG,
        "vrp_config": VRP_CONFIG,
        "assets": {},
    }

    all_alerts = []
    portfolio_actions = []

    for currency, cfg in ASSETS.items():
        sig = asset_signals.get(currency, {})
        combined["assets"][currency] = sig
        weight = cfg["weight"]

        if "error" in sig:
            all_alerts.append({"level": "CRITICAL", "check": f"{currency} signal error",
                               "value": sig["error"]})
            continue

        bf = sig.get("bf_signal", {})
        vrp = sig.get("vrp_signal", {})

        action = f"{currency} ({weight*100:.0f}%): BF={bf.get('signal', '?')} z={bf.get('z_score', '?')}"
        if vrp.get("signal") == "SHORT_VOL":
            action += f", VRP=SHORT ({vrp.get('vrp_spread', '?')}%)"
        else:
            action += f", VRP=NEUTRAL ({vrp.get('vrp_spread', '?')}%)"
        portfolio_actions.append(action)

        all_alerts.extend(sig.get("alerts", []))

    combined["actions"] = portfolio_actions
    combined["alerts"] = all_alerts
    combined["alert_count"] = len(all_alerts)
    combined["all_clear"] = len(all_alerts) == 0

    # Portfolio-level health (weighted average)
    total_health = 0
    total_weight = 0
    for currency, cfg in ASSETS.items():
        sig = asset_signals.get(currency, {})
        h = sig.get("health", {}).get("score")
        if h is not None:
            total_health += h * cfg["weight"]
            total_weight += cfg["weight"]

    if total_weight > 0:
        port_health = total_health / total_weight
        combined["portfolio_health"] = {
            "score": round(port_health, 3),
            "status": "STRONG" if port_health > 0.55 else "MODERATE" if port_health > 0.40 else "WEAK" if port_health > 0.25 else "CRITICAL",
        }

    return combined


# ═══════════════════════════════════════════════════════════════
# Output
# ═══════════════════════════════════════════════════════════════

def print_brief(combined):
    """One-line summary."""
    parts = []
    for currency in ASSETS:
        sig = combined["assets"].get(currency, {})
        bf = sig.get("bf_signal", {})
        h = sig.get("health", {})
        parts.append(f"{currency}:BF={bf.get('signal','?')[0]}z={bf.get('z_score','?')}h={h.get('score','?')}")

    ph = combined.get("portfolio_health", {})
    status = "OK" if combined.get("all_clear") else f"ALERTS({combined.get('alert_count', 0)})"
    print(f"[{combined['timestamp']}] {' | '.join(parts)} | port_h={ph.get('score','?')} {status}")


def print_full(combined):
    """Full human-readable report."""
    print("=" * 70)
    print("  MULTI-ASSET PRODUCTION SIGNAL REPORT")
    print("=" * 70)
    print(f"  Timestamp:  {combined['timestamp']}")
    alloc_str = " / ".join(f"{c} {w*100:.0f}%" for c, w in combined["allocation"].items())
    print(f"  Allocation: {alloc_str}")
    print(f"  BF Config:  lb={BF_CONFIG['bf_lookback']} z_in={BF_CONFIG['bf_z_entry']} "
          f"z_out={BF_CONFIG['bf_z_exit']} sens={BF_CONFIG['bf_sensitivity']}")

    for currency in ASSETS:
        sig = combined["assets"].get(currency, {})
        weight = ASSETS[currency]["weight"]
        print(f"\n  {'─'*30}")
        print(f"  {currency} ({weight*100:.0f}% allocation)")
        print(f"  {'─'*30}")

        if "error" in sig:
            print(f"  ERROR: {sig['error']}")
            continue

        bf = sig.get("bf_signal", {})
        vrp = sig.get("vrp_signal", {})
        health = sig.get("health", {})
        perf = sig.get("performance", {})
        live = sig.get("live", {})

        print(f"  Data: {sig.get('data_range', {}).get('n_days', '?')} days "
              f"({sig.get('data_range', {}).get('start', '?')} to {sig.get('data_range', {}).get('end', '?')})")

        if live:
            print(f"  Live: ${live.get('price', 0):,.0f} | IV {live.get('iv_pct', '?')}%")

        print(f"  BF Signal:  {bf.get('signal', '?')} | z={bf.get('z_score', '?')} | "
              f"val={bf.get('value', '?')} | μ={bf.get('mean_120d', '?')} | σ={bf.get('std_120d', '?')}")
        print(f"  VRP:        {vrp.get('signal', '?')} | IV={vrp.get('iv', '?')}% | "
              f"RV={vrp.get('rv_30d', '?')}% | spread={vrp.get('vrp_spread', '?')}%")

        if health:
            print(f"  Health:     {health.get('score', '?')} ({health.get('status', '?')}) | "
                  f"Sharpe90={health.get('sharpe_90d', '?')} | "
                  f"WinRate={health.get('hit_rate_60d', '?')}%")

        if perf:
            for period, stats in perf.items():
                if stats:
                    print(f"  {period}: Sharpe {stats.get('sharpe', '?')} | "
                          f"Ret {stats.get('ann_ret_pct', '?')}% | MaxDD {stats.get('max_dd_pct', '?')}%")

    # Portfolio summary
    ph = combined.get("portfolio_health", {})
    print(f"\n  {'═'*30}")
    print(f"  PORTFOLIO")
    print(f"  {'═'*30}")
    if ph:
        print(f"  Health: {ph.get('score', '?')} ({ph.get('status', '?')})")

    print(f"\n  Actions:")
    for action in combined.get("actions", []):
        print(f"    → {action}")

    alerts = combined.get("alerts", [])
    if alerts:
        print(f"\n  ALERTS ({len(alerts)}):")
        for a in alerts:
            print(f"    [{a.get('level','?')}] {a.get('check','?')}: {a.get('value','?')}")
    else:
        print(f"\n  ALL CLEAR — No alerts across all assets.")

    print("=" * 70)


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════

def main():
    # Load data for each asset
    asset_signals = {}
    for currency in ASSETS:
        dvol_hist = load_dvol_history(currency)
        price_hist = load_price_history(currency)
        surface_hist = load_surface(currency)
        asset_signals[currency] = compute_asset_signal(currency, dvol_hist, price_hist, surface_hist)

    # Combine
    combined = combine_signals(asset_signals)

    # Output
    if JSON_MODE:
        print(json.dumps(combined, indent=2))
    elif BRIEF_MODE:
        print_brief(combined)
    else:
        print_full(combined)

    # Save latest
    out_path = ROOT / "data" / "cache" / "deribit" / "real_surface" / "latest_multi_signal.json"
    with open(out_path, "w") as f:
        json.dump(combined, f, indent=2)

    # Append to log
    log_path = ROOT / "data" / "cache" / "deribit" / "real_surface" / "multi_signal_log.jsonl"
    with open(log_path, "a") as f:
        f.write(json.dumps(combined) + "\n")

    return combined


if __name__ == "__main__":
    main()
