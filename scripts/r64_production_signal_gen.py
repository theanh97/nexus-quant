#!/usr/bin/env python3
"""
R64: Production Signal Generator (R71 — updated to R69/R70 config)
====================================================================

Live signal generator using the validated R69 production config v2:
  BTC VRP + Butterfly MR (10/90, z_exit=0.0)

R69 upgrade (vs R60):
  Sharpe: 3.76 vs 2.91 (+0.85)
  MaxDD:  0.46% vs 1.59% (-71%)
  Walk-forward: 100% positive (8/8)

Sensitivity tiers (R70 — Sharpe ~scale-invariant):
  Conservative: sens=2.5 → ~2% ann return, ~0.5% MaxDD
  Moderate:     sens=5.0 → ~4% ann return, ~0.9% MaxDD
  Aggressive:   sens=7.5 → ~6% ann return, ~1.4% MaxDD

Fetches latest data from Deribit, computes signals, outputs:
  - Current VRP signal (short straddle recommendation)
  - Current Butterfly MR signal and position
  - Combined portfolio recommendation
  - Risk metrics and equity curve
  - Historical signal accuracy

Usage:
  python3 scripts/r64_production_signal_gen.py           # Full report
  python3 scripts/r64_production_signal_gen.py --brief    # One-line signal
"""
import csv
import json
import math
import subprocess
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
BRIEF_MODE = "--brief" in sys.argv


# ═══════════════════════════════════════════════════════════════
# Configuration (R69/R70 validated production config v2)
# ═══════════════════════════════════════════════════════════════

CONFIG = {
    "w_vrp": 0.10,           # VRP weight (R69: 10% — down from R60's 50%)
    "w_bf": 0.90,            # Butterfly MR weight (R69: 90%)
    "vrp_leverage": 2.0,     # VRP leverage
    "bf_lookback": 120,      # Butterfly z-score lookback (days)
    "bf_z_entry": 1.5,       # Butterfly entry threshold
    "bf_z_exit": 0.0,        # Butterfly exit: 0.0 = hold until reversed (R68 discovery)
    "bf_sensitivity": 2.5,   # BF PnL sensitivity (R70 tiers: 2.5/5.0/7.5)
    "asset": "BTC",          # BTC ONLY
}


# ═══════════════════════════════════════════════════════════════
# Data Loading
# ═══════════════════════════════════════════════════════════════

def fetch_latest_dvol(currency: str = "BTC") -> Tuple[float, str]:
    """Fetch current DVOL from Deribit API."""
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
            dvol = last[4] / 100.0  # close
            ts = datetime.fromtimestamp(last[0] / 1000, tz=timezone.utc)
            return dvol, ts.strftime("%Y-%m-%d %H:%M UTC")
    except:
        pass
    return None, None


def fetch_latest_price(currency: str = "BTC") -> Tuple[float, str]:
    """Fetch current perpetual price from Deribit."""
    url = (f"https://www.deribit.com/api/v2/public/ticker?"
           f"instrument_name={currency}-PERPETUAL")
    try:
        r = subprocess.run(["curl", "-s", "--max-time", "15", url],
                          capture_output=True, text=True, timeout=20)
        data = json.loads(r.stdout)
        if "result" in data:
            result = data["result"]
            price = result.get("last_price") or result.get("mark_price", 0)
            return float(price), datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    except:
        pass
    return None, None


def load_dvol_history(currency: str) -> Dict[str, float]:
    """Load historical daily DVOL."""
    path = ROOT / "data" / "cache" / "deribit" / "dvol" / f"{currency}_DVOL_12h.csv"
    daily = {}
    if not path.exists():
        return daily
    with open(path) as f:
        for row in csv.DictReader(f):
            daily[row["date"][:10]] = float(row["dvol_close"]) / 100.0
    return daily


def load_price_history(currency: str) -> Dict[str, float]:
    """Load historical daily prices."""
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


def load_surface(currency: str) -> Dict[str, dict]:
    """Load historical surface data."""
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
# Signal Computation
# ═══════════════════════════════════════════════════════════════

def compute_vrp_signal(dvol: float, price_now: float, price_prev: float) -> dict:
    """Compute current VRP signal."""
    dt = 1.0 / 365.0
    rv_bar = abs(math.log(price_now / price_prev)) * math.sqrt(365)
    vrp = dvol**2 - rv_bar**2
    pnl = CONFIG["vrp_leverage"] * 0.5 * vrp * dt

    return {
        "dvol": dvol,
        "rv_bar": rv_bar,
        "vrp_spread": vrp,
        "vrp_pnl": pnl,
        "signal": "SHORT_VOL" if vrp > 0 else "FLAT",
        "strength": "STRONG" if vrp > 0.05 else "MODERATE" if vrp > 0 else "WEAK",
    }


def compute_bf_signal(surface_history: Dict[str, dict], dates: List[str],
                       dvol: float) -> dict:
    """Compute current Butterfly MR signal."""
    lb = CONFIG["bf_lookback"]
    z_entry = CONFIG["bf_z_entry"]
    z_exit = CONFIG["bf_z_exit"]

    # Get butterfly values
    bf_vals = {}
    for d in dates:
        if d in surface_history and "butterfly_25d" in surface_history[d]:
            bf_vals[d] = surface_history[d]["butterfly_25d"]

    if len(bf_vals) < lb:
        return {"signal": "NO_DATA", "z_score": None, "position": 0}

    # Current z-score
    recent_dates = sorted(bf_vals.keys())
    if len(recent_dates) < lb:
        return {"signal": "NO_DATA", "z_score": None, "position": 0}

    current_val = bf_vals[recent_dates[-1]]
    window = [bf_vals[recent_dates[i]] for i in range(max(0, len(recent_dates)-lb), len(recent_dates)-1)]
    mean = sum(window) / len(window)
    std = math.sqrt(sum((v - mean)**2 for v in window) / len(window))

    if std < 1e-8:
        return {"signal": "NO_SIGNAL", "z_score": 0, "position": 0}

    z = (current_val - mean) / std

    # Position logic (R68: z_exit=0.0 means hold until reversed — never go flat)
    if z > z_entry:
        position = -1.0
        signal = "SHORT_BUTTERFLY"
    elif z < -z_entry:
        position = 1.0
        signal = "LONG_BUTTERFLY"
    elif z_exit > 0 and abs(z) < z_exit:
        position = 0.0
        signal = "FLAT"
    else:
        position = None  # No change from previous — hold until reversal
        signal = "HOLD"

    return {
        "signal": signal,
        "z_score": round(z, 3),
        "bf_value": round(current_val, 4),
        "bf_mean": round(mean, 4),
        "bf_std": round(std, 4),
        "position": position,
        "latest_date": recent_dates[-1],
    }


def compute_backtest_equity(dvol_history, price_history, surface_history):
    """Compute full backtest equity curve."""
    dates = sorted(set(dvol_history.keys()) & set(price_history.keys()))

    # VRP PnL
    dt = 1.0 / 365.0
    vrp_pnl = {}
    for i in range(1, len(dates)):
        d, dp = dates[i], dates[i-1]
        iv = dvol_history.get(dp)
        p0, p1 = price_history.get(dp), price_history.get(d)
        if not all([iv, p0, p1]) or p0 <= 0:
            continue
        rv = abs(math.log(p1 / p0)) * math.sqrt(365)
        vrp_pnl[d] = CONFIG["vrp_leverage"] * 0.5 * (iv**2 - rv**2) * dt

    # Butterfly MR PnL
    bf_vals = {}
    for d in dates:
        if d in surface_history and "butterfly_25d" in surface_history[d]:
            bf_vals[d] = surface_history[d]["butterfly_25d"]

    bf_pnl = {}
    position = 0.0
    lb = CONFIG["bf_lookback"]
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

        if z > CONFIG["bf_z_entry"]:
            position = -1.0
        elif z < -CONFIG["bf_z_entry"]:
            position = 1.0
        elif abs(z) < CONFIG["bf_z_exit"]:
            position = 0.0

        iv = dvol_history.get(d)
        f_now, f_prev = bf_vals.get(d), bf_vals.get(dp)
        if f_now is not None and f_prev is not None and iv is not None and position != 0:
            bf_pnl[d] = position * (f_now - f_prev) * iv * math.sqrt(dt) * CONFIG["bf_sensitivity"]
        else:
            bf_pnl[d] = 0.0

    # Combined equity
    combined_dates = sorted(set(vrp_pnl.keys()) & set(bf_pnl.keys()))
    w_v, w_b = CONFIG["w_vrp"], CONFIG["w_bf"]
    equity = [0.0]
    daily_rets = []
    for d in combined_dates:
        ret = w_v * vrp_pnl[d] + w_b * bf_pnl[d]
        equity.append(equity[-1] + ret)
        daily_rets.append(ret)

    return combined_dates, equity[1:], daily_rets, vrp_pnl, bf_pnl


def calc_rolling_stats(rets, window=252):
    """Compute rolling Sharpe."""
    if len(rets) < window:
        return []
    results = []
    for i in range(window, len(rets)):
        w = rets[i-window:i]
        mean = sum(w) / len(w)
        std = math.sqrt(sum((r - mean)**2 for r in w) / len(w))
        if std > 0:
            sharpe = (mean * 365) / (std * math.sqrt(365))
        else:
            sharpe = 0
        results.append(sharpe)
    return results


# ═══════════════════════════════════════════════════════════════
# Output
# ═══════════════════════════════════════════════════════════════

def main():
    now = datetime.now(timezone.utc)

    if not BRIEF_MODE:
        print("=" * 70)
        print("R64: PRODUCTION SIGNAL GENERATOR (R69/R70 config v2)")
        print(f"  {now.strftime('%Y-%m-%d %H:%M UTC')}")
        print(f"  Config: {int(CONFIG['w_vrp']*100)}/{int(CONFIG['w_bf']*100)} VRP/BF | z_exit={CONFIG['bf_z_exit']} | sens={CONFIG['bf_sensitivity']}")
        print("=" * 70)

    # ─── Fetch live data ────────────────────────────────────────
    if not BRIEF_MODE:
        print("\n  Fetching live data...")

    dvol_now, dvol_time = fetch_latest_dvol("BTC")
    price_now, price_time = fetch_latest_price("BTC")

    if not BRIEF_MODE:
        print(f"    DVOL:  {dvol_now*100:.1f}% ({dvol_time})" if dvol_now else "    DVOL:  UNAVAILABLE")
        print(f"    Price: ${price_now:,.0f} ({price_time})" if price_now else "    Price: UNAVAILABLE")

    # ─── Load historical data ───────────────────────────────────
    if not BRIEF_MODE:
        print("  Loading historical data...")

    dvol_hist = load_dvol_history("BTC")
    price_hist = load_price_history("BTC")
    surface_hist = load_surface("BTC")

    if not BRIEF_MODE:
        print(f"    DVOL history: {len(dvol_hist)} days")
        print(f"    Price history: {len(price_hist)} days")
        print(f"    Surface history: {len(surface_hist)} days")

    # ─── Current signals ────────────────────────────────────────
    if not BRIEF_MODE:
        print("\n" + "=" * 70)
        print("  CURRENT SIGNALS")
        print("=" * 70)

    dates = sorted(set(dvol_hist.keys()) & set(price_hist.keys()))
    if not dates:
        print("  ERROR: No historical data available")
        return

    # Use latest available data if live fetch failed
    latest_date = dates[-1]
    if dvol_now is None:
        dvol_now = dvol_hist[latest_date]
    if price_now is None:
        price_now = price_hist[latest_date]

    # VRP signal
    prev_price = price_hist.get(dates[-2]) if len(dates) > 1 else price_now
    vrp_sig = compute_vrp_signal(dvol_now, price_now, prev_price)

    if not BRIEF_MODE:
        print(f"\n  ═══ VRP SIGNAL ═══")
        print(f"    DVOL (IV):     {vrp_sig['dvol']*100:.1f}%")
        print(f"    RV (1d ann):   {vrp_sig['rv_bar']*100:.1f}%")
        print(f"    VRP Spread:    {vrp_sig['vrp_spread']*100:.2f}%")
        print(f"    Daily PnL est: {vrp_sig['vrp_pnl']*10000:.2f} bps")
        print(f"    Signal:        {vrp_sig['signal']} ({vrp_sig['strength']})")

    # Butterfly signal
    bf_sig = compute_bf_signal(surface_hist, dates, dvol_now)

    if not BRIEF_MODE:
        print(f"\n  ═══ BUTTERFLY MR SIGNAL ═══")
        if bf_sig["z_score"] is not None:
            print(f"    BF value:      {bf_sig['bf_value']:.4f}")
            print(f"    BF mean:       {bf_sig['bf_mean']:.4f}")
            print(f"    BF z-score:    {bf_sig['z_score']:.3f}")
            print(f"    Signal:        {bf_sig['signal']}")
            print(f"    Data as of:    {bf_sig['latest_date']}")
        else:
            print(f"    Signal: {bf_sig['signal']}")

    # Combined recommendation
    if not BRIEF_MODE:
        print(f"\n  ═══ COMBINED RECOMMENDATION ═══")
        print(f"    Config: {int(CONFIG['w_vrp']*100)}% VRP + {int(CONFIG['w_bf']*100)}% Butterfly MR")
        print(f"    VRP:   {vrp_sig['signal']}")
        print(f"    BF:    {bf_sig['signal']}")

    if BRIEF_MODE:
        bf_pos_str = "LONG" if bf_sig.get("position", 0) == 1 else "SHORT" if bf_sig.get("position", 0) == -1 else "FLAT"
        print(f"[{now.strftime('%Y-%m-%d')}] BTC DVOL={dvol_now*100:.1f}% | VRP={vrp_sig['signal']}({vrp_sig['strength']}) | BF z={bf_sig.get('z_score', 'N/A')} {bf_pos_str}")
        return

    # ─── Backtest equity curve ──────────────────────────────────
    print("\n" + "=" * 70)
    print("  BACKTEST EQUITY CURVE (full history)")
    print("=" * 70)

    bt_dates, equity, daily_rets, vrp_pnl, bf_pnl = compute_backtest_equity(
        dvol_hist, price_hist, surface_hist
    )

    if len(daily_rets) > 30:
        mean_r = sum(daily_rets) / len(daily_rets)
        std_r = math.sqrt(sum((r - mean_r)**2 for r in daily_rets) / len(daily_rets))
        sharpe = (mean_r * 365) / (std_r * math.sqrt(365)) if std_r > 0 else 0
        ann_ret = mean_r * 365
        cum = peak = max_dd = 0.0
        for r in daily_rets:
            cum += r
            peak = max(peak, cum)
            max_dd = max(max_dd, peak - cum)
        win_rate = sum(1 for r in daily_rets if r > 0) / len(daily_rets)

        print(f"\n  Period: {bt_dates[0]} to {bt_dates[-1]} ({len(daily_rets)} days)")
        print(f"  Sharpe:     {sharpe:.2f}")
        print(f"  Ann Return: {ann_ret*100:.2f}%")
        print(f"  Ann Vol:    {std_r*math.sqrt(365)*100:.2f}%")
        print(f"  Max DD:     {max_dd*100:.2f}%")
        print(f"  Win Rate:   {win_rate*100:.1f}%")
        print(f"  Total:      {sum(daily_rets)*100:.2f}%")

        # Recent performance
        print(f"\n  ─── Recent Performance ───")
        for period_name, period_days in [("30d", 30), ("90d", 90), ("180d", 180), ("1yr", 365)]:
            if len(daily_rets) >= period_days:
                window = daily_rets[-period_days:]
                w_mean = sum(window) / len(window)
                w_std = math.sqrt(sum((r - w_mean)**2 for r in window) / len(window))
                w_sharpe = (w_mean * 365) / (w_std * math.sqrt(365)) if w_std > 0 else 0
                w_ret = w_mean * 365
                print(f"    {period_name:>4}: Sharpe {w_sharpe:5.2f}, Return {w_ret*100:6.2f}%")

        # Current drawdown
        peak_eq = max(equity)
        current_dd = (peak_eq - equity[-1]) / max(1e-10, peak_eq) if peak_eq > 0 else 0
        print(f"\n  Current drawdown: {current_dd*100:.2f}%")
        if current_dd > 0.01:
            # Find drawdown start
            peak_idx = equity.index(peak_eq)
            dd_days = len(equity) - 1 - peak_idx
            print(f"  Drawdown duration: {dd_days} days (since {bt_dates[peak_idx]})")

    # ─── VRP regime assessment ──────────────────────────────────
    print("\n" + "=" * 70)
    print("  CURRENT REGIME ASSESSMENT")
    print("=" * 70)

    # IV percentile
    dvol_vals = sorted(dvol_hist.values())
    current_iv_pctile = sum(1 for v in dvol_vals if v <= dvol_now) / len(dvol_vals)

    # Recent VRP
    recent_vrp = []
    for i in range(max(0, len(dates)-30), len(dates)-1):
        d, dp = dates[i+1], dates[i]
        iv = dvol_hist.get(dp)
        p0, p1 = price_hist.get(dp), price_hist.get(d)
        if all([iv, p0, p1]) and p0 > 0:
            rv = abs(math.log(p1 / p0)) * math.sqrt(365)
            recent_vrp.append(iv**2 - rv**2)

    avg_recent_vrp = sum(recent_vrp) / len(recent_vrp) if recent_vrp else 0

    print(f"\n  IV Level: {dvol_now*100:.1f}% (percentile: {current_iv_pctile*100:.0f}%)")
    print(f"  IV Regime: {'LOW' if current_iv_pctile < 0.30 else 'MID' if current_iv_pctile < 0.70 else 'HIGH'}")
    print(f"  Recent 30d avg VRP spread: {avg_recent_vrp*100:.2f}%")
    if avg_recent_vrp > 0.03:
        print(f"  VRP Assessment: STRONG positive — good conditions for short vol")
    elif avg_recent_vrp > 0:
        print(f"  VRP Assessment: Positive but thin — proceed with standard sizing")
    else:
        print(f"  VRP Assessment: NEGATIVE — elevated realized vol, caution")

    # ─── Summary box ────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  ╔══════════════════════════════════════════════════╗")
    print(f"  ║  BTC DVOL: {dvol_now*100:.1f}%  |  Price: ${price_now:,.0f}")
    print(f"  ║  VRP: {vrp_sig['signal']:<12} | BF: {bf_sig['signal']:<16}  ║")
    print(f"  ║  Config: {int(CONFIG['w_vrp']*100)}/{int(CONFIG['w_bf']*100)} VRP/BF  |  Sharpe: {sharpe:.2f}      ║")
    print(f"  ╚══════════════════════════════════════════════════╝")


if __name__ == "__main__":
    main()
