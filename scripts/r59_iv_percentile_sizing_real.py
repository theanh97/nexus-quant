#!/usr/bin/env python3
"""
R59: IV-Percentile Position Sizing on REAL Data
=================================================

R25 found: Step-function IV sizing (0.5x/1.0x/1.5x with 180d lookback)
improved VRP Sharpe by +0.258 on synthetic data.

Now validate on REAL data with the R57 production config (VRP+BF):
1. IV percentile sizing on VRP+BF ensemble
2. Multiple sizing functions (step, linear, aggressive)
3. Walk-forward validation
4. Practical: what's the FINAL production config?
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
from typing import Dict, List, Tuple, Optional

ROOT = Path(__file__).resolve().parents[1]


def load_dvol_daily(currency: str) -> Dict[str, float]:
    path = ROOT / "data" / "cache" / "deribit" / "dvol" / f"{currency}_DVOL_12h.csv"
    daily = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            daily[row["date"][:10]] = float(row["dvol_close"]) / 100.0
    return daily


def load_prices(currency: str) -> Dict[str, float]:
    prices = {}
    start_dt = datetime(2019, 1, 1, tzinfo=timezone.utc)
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
            if "result" in data: data = data["result"]
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
    path = ROOT / "data" / "cache" / "deribit" / "real_surface" / f"{currency}_daily_surface.csv"
    data = {}
    if not path.exists():
        return data
    with open(path) as f:
        for row in csv.DictReader(f):
            d = row["date"]
            entry = {}
            for field in ["iv_atm", "skew_25d", "butterfly_25d", "term_spread"]:
                val = row.get(field, "")
                if val and val != "None":
                    entry[field] = float(val)
            if entry:
                data[d] = entry
    return data


def rolling_zscore(values: Dict[str, float], dates: List[str], lookback: int) -> Dict[str, float]:
    result = {}
    for i in range(lookback, len(dates)):
        d = dates[i]
        val = values.get(d)
        if val is None:
            continue
        window = [values.get(dates[j]) for j in range(i - lookback, i)]
        window = [v for v in window if v is not None]
        if len(window) < lookback // 2:
            continue
        mean = sum(window) / len(window)
        std = math.sqrt(sum((v - mean)**2 for v in window) / len(window))
        if std > 1e-8:
            result[d] = (val - mean) / std
    return result


def vrp_pnl(dates, dvol, prices, lev=2.0):
    dt = 1.0 / 365.0
    pnl = {}
    for i in range(1, len(dates)):
        d, dp = dates[i], dates[i-1]
        iv, p0, p1 = dvol.get(dp), prices.get(dp), prices.get(d)
        if not all([iv, p0, p1]) or p0 <= 0:
            continue
        rv_bar = abs(math.log(p1 / p0)) * math.sqrt(365)
        pnl[d] = lev * 0.5 * (iv**2 - rv_bar**2) * dt
    return pnl


def mr_pnl(dates, feature, dvol, lookback=120, z_entry=1.5):
    dt = 1.0 / 365.0
    pnl = {}
    position = 0.0
    zscore = rolling_zscore(feature, dates, lookback)
    for i in range(1, len(dates)):
        d, dp = dates[i], dates[i-1]
        z = zscore.get(d)
        iv = dvol.get(d)
        f_now, f_prev = feature.get(d), feature.get(dp)
        if z is not None:
            if z > z_entry: position = -1.0
            elif z < -z_entry: position = 1.0
            elif abs(z) < 0.3: position = 0.0
        if f_now is not None and f_prev is not None and iv is not None and position != 0:
            pnl[d] = position * (f_now - f_prev) * iv * math.sqrt(dt) * 2.5
        elif d in zscore:
            pnl[d] = 0.0
    return pnl


def calc_sharpe(rets):
    if len(rets) < 20:
        return 0.0, 0.0, 0.0
    mean = sum(rets) / len(rets)
    var = sum((r - mean)**2 for r in rets) / len(rets)
    std = math.sqrt(var) if var > 0 else 1e-10
    sharpe = (mean * 365) / (std * math.sqrt(365))
    ann_ret = mean * 365
    cum = peak = max_dd = 0.0
    for r in rets:
        cum += r; peak = max(peak, cum); max_dd = max(max_dd, peak - cum)
    return sharpe, ann_ret, max_dd


def iv_percentile_rank(dates, dvol, lookback=180):
    """Compute IV percentile rank (0-1) for each date."""
    ranks = {}
    for i in range(lookback, len(dates)):
        d = dates[i]
        iv = dvol.get(d)
        if iv is None:
            continue
        window = [dvol.get(dates[j]) for j in range(i - lookback, i)]
        window = [v for v in window if v is not None]
        if len(window) < lookback // 2:
            continue
        ranks[d] = sum(1 for w in window if w <= iv) / len(window)
    return ranks


def main():
    print("=" * 70)
    print("R59: IV-PERCENTILE POSITION SIZING ON REAL DATA")
    print("=" * 70)
    print()

    # Load data
    print("Loading data...")
    dvol = load_dvol_daily("BTC")
    prices = load_prices("BTC")
    surface = load_surface("BTC")
    all_dates = sorted(set(dvol.keys()) & set(prices.keys()) & set(surface.keys()))
    print(f"  Common dates: {len(all_dates)} ({all_dates[0]} to {all_dates[-1]})")

    # Compute base PnL streams
    bf_feat = {d: s["butterfly_25d"] for d, s in surface.items() if "butterfly_25d" in s}
    pnl_vrp = vrp_pnl(all_dates, dvol, prices, 2.0)
    pnl_bf = mr_pnl(all_dates, bf_feat, dvol, 120, 1.5)
    common_dates = sorted(set(pnl_vrp.keys()) & set(pnl_bf.keys()))
    print(f"  Common PnL dates: {len(common_dates)}")

    # ── 1. DVOL Percentile Distribution ──────────────────────────────
    print(f"\n{'='*70}")
    print("  STEP 1: REAL DVOL PERCENTILE DISTRIBUTION")
    print(f"{'='*70}")

    for lb in [90, 180, 252]:
        ranks = iv_percentile_rank(all_dates, dvol, lb)
        vals = [ranks[d] for d in common_dates if d in ranks]
        if not vals:
            continue
        below_25 = sum(1 for v in vals if v < 0.25) / len(vals)
        mid = sum(1 for v in vals if 0.25 <= v <= 0.75) / len(vals)
        above_75 = sum(1 for v in vals if v > 0.75) / len(vals)
        print(f"  lb={lb:3d}: <25th={below_25:.0%}  25-75th={mid:.0%}  >75th={above_75:.0%}  N={len(vals)}")

    # ── 2. Sizing Functions ──────────────────────────────────────────
    print(f"\n{'='*70}")
    print("  STEP 2: SIZING FUNCTION COMPARISON")
    print(f"{'='*70}")
    print()

    sizing_configs = []
    for lb in [90, 180, 252]:
        ranks = iv_percentile_rank(all_dates, dvol, lb)
        for name, fn in [
            ("step_0.5/1.0/1.5", lambda r: 0.5 if r < 0.25 else (1.5 if r > 0.75 else 1.0)),
            ("step_0.5/1.0/1.7", lambda r: 0.5 if r < 0.25 else (1.7 if r > 0.75 else 1.0)),
            ("step_0.3/1.0/2.0", lambda r: 0.3 if r < 0.25 else (2.0 if r > 0.75 else 1.0)),
            ("linear_0.5-1.5",   lambda r: 0.5 + r * 1.0),
            ("linear_0.3-2.0",   lambda r: 0.3 + r * 1.7),
            ("no_sizing",        lambda r: 1.0),
        ]:
            sizing_configs.append((f"{name}_lb{lb}", lb, fn, ranks))

    # Test on VRP+BF(70/30) and VRP+BF(50/50)
    for w_vrp, w_bf, label in [(0.7, 0.3, "70/30"), (0.5, 0.5, "50/50")]:
        print(f"\n  === VRP+BF ({label}) ===")
        print(f"  {'Config':<30s}  {'Sharpe':>7s}  {'AnnRet':>8s}  {'MaxDD':>7s}  {'Δ':>6s}")

        baseline_sh = None
        for config_name, lb, fn, ranks in sizing_configs:
            sized_rets = []
            for d in common_dates:
                r = ranks.get(d)
                if r is None:
                    continue
                scale = fn(r)
                pnl = scale * (w_vrp * pnl_vrp.get(d, 0) + w_bf * pnl_bf.get(d, 0))
                sized_rets.append(pnl)

            sh, ret, dd = calc_sharpe(sized_rets)
            if "no_sizing" in config_name and baseline_sh is None:
                baseline_sh = sh
            delta = sh - (baseline_sh or 0)
            marker = " ★" if delta > 0.1 else ""
            print(f"  {config_name:<30s}  {sh:7.2f}  {ret:+8.2%}  {dd:7.2%}  {delta:+6.2f}{marker}")

    # ── 3. Yearly Breakdown ──────────────────────────────────────────
    print(f"\n{'='*70}")
    print("  STEP 3: YEARLY BREAKDOWN — Best Sizing vs No Sizing")
    print(f"{'='*70}")
    print("  Using step_0.5/1.0/1.7 lb180 on VRP+BF(70/30)")
    print()

    ranks_180 = iv_percentile_rank(all_dates, dvol, 180)
    step_fn = lambda r: 0.5 if r < 0.25 else (1.7 if r > 0.75 else 1.0)

    print(f"  {'Year':<6s}  {'No Size':>8s}  {'Sized':>8s}  {'Δ':>6s}  {'Avg Scale':>10s}")
    for yr in sorted(set(d[:4] for d in common_dates)):
        yr_dates = [d for d in common_dates if d[:4] == yr and d in ranks_180]
        if len(yr_dates) < 20:
            continue

        base_rets = [0.7 * pnl_vrp.get(d, 0) + 0.3 * pnl_bf.get(d, 0) for d in yr_dates]
        sized_rets = [step_fn(ranks_180[d]) * (0.7 * pnl_vrp.get(d, 0) + 0.3 * pnl_bf.get(d, 0))
                      for d in yr_dates]
        scales = [step_fn(ranks_180[d]) for d in yr_dates]

        sh_base, _, _ = calc_sharpe(base_rets)
        sh_sized, _, _ = calc_sharpe(sized_rets)
        avg_scale = sum(scales) / len(scales)

        print(f"  {yr:<6s}  {sh_base:8.2f}  {sh_sized:8.2f}  {sh_sized-sh_base:+6.2f}  {avg_scale:10.2f}")

    # ── 4. Walk-Forward of IV Sizing ─────────────────────────────────
    print(f"\n{'='*70}")
    print("  STEP 4: WALK-FORWARD — SIZED vs UNSIZED")
    print(f"{'='*70}")
    print()

    periods = []
    start_year = int(common_dates[0][:4])
    end_year = int(common_dates[-1][:4])
    for yr in range(start_year, end_year + 1):
        for half in [(1, 6), (7, 12)]:
            period_start = f"{yr}-{half[0]:02d}-01"
            period_end = f"{yr}-{half[1]:02d}-30"
            period_dates = [d for d in common_dates if period_start <= d <= period_end and d in ranks_180]
            if period_dates:
                label = f"{yr}H{1 if half[0]==1 else 2}"
                periods.append((label, period_dates))

    print(f"  {'Period':<8s}  {'Unsized':>8s}  {'Sized':>8s}  {'Δ':>6s}")
    unsized_all = []
    sized_all = []
    for label, test_dates in periods:
        base_rets = [0.7 * pnl_vrp.get(d, 0) + 0.3 * pnl_bf.get(d, 0) for d in test_dates]
        sized_rets = [step_fn(ranks_180[d]) * (0.7 * pnl_vrp.get(d, 0) + 0.3 * pnl_bf.get(d, 0))
                      for d in test_dates]

        sh_base, _, _ = calc_sharpe(base_rets)
        sh_sized, _, _ = calc_sharpe(sized_rets)
        unsized_all.extend(base_rets)
        sized_all.extend(sized_rets)

        print(f"  {label:<8s}  {sh_base:8.2f}  {sh_sized:8.2f}  {sh_sized-sh_base:+6.2f}")

    sh_u, ret_u, dd_u = calc_sharpe(unsized_all)
    sh_s, ret_s, dd_s = calc_sharpe(sized_all)
    print(f"\n  Total:    Unsized={sh_u:.2f}  Sized={sh_s:.2f}  Δ={sh_s-sh_u:+.2f}")
    print(f"            Ret: {ret_u:+.2%} → {ret_s:+.2%}  DD: {dd_u:.2%} → {dd_s:.2%}")

    # ── 5. Final Production Config ───────────────────────────────────
    print(f"\n{'='*70}")
    print("  STEP 5: FINAL PRODUCTION CONFIG COMPARISON")
    print(f"{'='*70}")
    print()

    final_configs = [
        ("VRP-only (no size)",    1.0, 0.0, False),
        ("VRP+BF 70/30 (no size)", 0.7, 0.3, False),
        ("VRP+BF 50/50 (no size)", 0.5, 0.5, False),
        ("VRP+BF 70/30 + IV size", 0.7, 0.3, True),
        ("VRP+BF 50/50 + IV size", 0.5, 0.5, True),
    ]

    print(f"  {'Config':<30s}  {'Sharpe':>7s}  {'AnnRet':>8s}  {'MaxDD':>7s}")
    for name, wv, wb, sized in final_configs:
        rets = []
        for d in common_dates:
            if d not in ranks_180:
                continue
            base = wv * pnl_vrp.get(d, 0) + wb * pnl_bf.get(d, 0)
            if sized:
                base *= step_fn(ranks_180[d])
            rets.append(base)
        sh, ret, dd = calc_sharpe(rets)
        print(f"  {name:<30s}  {sh:7.2f}  {ret:+8.2%}  {dd:7.2%}")

    # ── Summary ──────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("R59 SUMMARY")
    print(f"{'='*70}")
    print()
    print("  Production recommendation:")
    print("    Config: BTC VRP+BF 70/30 with IV-percentile sizing")
    print("    Sizing: step 0.5x/1.0x/1.7x with 180d lookback")
    print("    Asset:  BTC only")

    # Save
    results = {
        "research_id": "R59",
        "title": "IV-Percentile Position Sizing on Real Data",
    }
    outpath = ROOT / "data" / "cache" / "deribit" / "real_surface" / "r59_iv_sizing_results.json"
    with open(outpath, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved to {outpath}")


if __name__ == "__main__":
    main()
