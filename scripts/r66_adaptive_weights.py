#!/usr/bin/env python3
"""
R66: Adaptive VRP/BF Weighting
=================================

R65 showed VRP is the degradation source:
  - VRP Sharpe: 4.15 (2024H1) → -1.62 (2026H1)
  - BF Sharpe:  2.32 (2024H1) →  5.36 (2026H1) — STABLE

The fixed 50/50 weight is suboptimal. When VRP edge compresses or
turns negative, we should shift weight toward BF.

APPROACHES TESTED:
  1. Trailing VRP Sharpe gate — reduce VRP weight when recent VRP Sharpe < threshold
  2. VRP sign filter — zero VRP weight when rolling VRP is negative
  3. IV-level conditioned — lower IV → lower VRP weight (VRP scales with IV²)
  4. Optimal hindsight regime weights (upper bound)
  5. Walk-forward adaptive (realistic)
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

ROOT = Path(__file__).resolve().parents[1]


def load_dvol_daily(currency):
    path = ROOT / "data" / "cache" / "deribit" / "dvol" / f"{currency}_DVOL_12h.csv"
    daily = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            daily[row["date"][:10]] = float(row["dvol_close"]) / 100.0
    return daily


def load_prices(currency):
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


def load_surface(currency):
    path = ROOT / "data" / "cache" / "deribit" / "real_surface" / f"{currency}_daily_surface.csv"
    data = {}
    if not path.exists():
        return data
    with open(path) as f:
        for row in csv.DictReader(f):
            d = row["date"]
            entry = {}
            for field in ["butterfly_25d"]:
                val = row.get(field, "")
                if val and val != "None":
                    entry[field] = float(val)
            if entry:
                data[d] = entry
    return data


def rolling_zscore(values, dates, lookback):
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


def bf_pnl(dates, feature, dvol, lookback=120, z_entry=1.5):
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


def calc_stats(rets):
    if len(rets) < 10:
        return {"sharpe": 0.0, "ann_ret": 0.0, "max_dd": 0.0, "total_return": 0.0}
    mean = sum(rets) / len(rets)
    var = sum((r - mean)**2 for r in rets) / len(rets)
    std = math.sqrt(var) if var > 0 else 1e-10
    sharpe = (mean * 365) / (std * math.sqrt(365))
    ann_ret = mean * 365
    cum = peak = max_dd = 0.0
    for r in rets:
        cum += r; peak = max(peak, cum); max_dd = max(max_dd, peak - cum)
    return {
        "sharpe": sharpe, "ann_ret": ann_ret, "max_dd": max_dd,
        "total_return": sum(rets), "n_days": len(rets),
        "win_rate": sum(1 for r in rets if r > 0) / len(rets),
    }


def main():
    print("=" * 70)
    print("R66: ADAPTIVE VRP/BF WEIGHTING")
    print("=" * 70)
    print()

    # ═══════════════════════════════════════════════════════════════
    # Load data (same as R60)
    # ═══════════════════════════════════════════════════════════════
    print("  Loading data...")
    dvol = load_dvol_daily("BTC")
    prices = load_prices("BTC")
    surface = load_surface("BTC")
    all_dates = sorted(set(dvol.keys()) & set(prices.keys()) & set(surface.keys()))
    print(f"    {len(all_dates)} days, {all_dates[0]} to {all_dates[-1]}")

    bf_feat = {d: s["butterfly_25d"] for d, s in surface.items() if "butterfly_25d" in s}
    pnl_v = vrp_pnl(all_dates, dvol, prices, 2.0)
    pnl_b = bf_pnl(all_dates, bf_feat, dvol, 120, 1.5)
    common = sorted(set(pnl_v.keys()) & set(pnl_b.keys()))
    print(f"    {len(common)} common PnL dates")

    # ═══════════════════════════════════════════════════════════════
    # BASELINE: Static 50/50 (R60 champion)
    # ═══════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("  BASELINE: Static 50/50 (R60)")
    print("=" * 70)
    baseline_rets = [0.5 * pnl_v[d] + 0.5 * pnl_b[d] for d in common]
    bs = calc_stats(baseline_rets)
    print(f"    Sharpe: {bs['sharpe']:.4f}  Return: {bs['ann_ret']*100:.2f}%  "
          f"MaxDD: {bs['max_dd']*100:.2f}%  WinRate: {bs['win_rate']*100:.1f}%")

    # ═══════════════════════════════════════════════════════════════
    # 1. TRAILING VRP SHARPE GATE
    # ═══════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("  1. TRAILING VRP SHARPE GATE")
    print("=" * 70)
    print("  Reduce VRP weight when trailing VRP Sharpe is poor.\n")

    print(f"  {'Lookback':<10} {'Threshold':<12} {'VRP_w':<8} {'Sharpe':>8} {'Return':>10} {'MaxDD':>8}")
    best_gate = {"sharpe": 0}

    for lookback in [30, 60, 90, 120]:
        for threshold in [0.0, 0.5, 1.0]:
            for low_w in [0.0, 0.2]:
                rets = []
                for idx, d in enumerate(common):
                    # Compute trailing VRP Sharpe
                    start_idx = max(0, idx - lookback)
                    trail_vrp = [pnl_v[common[j]] for j in range(start_idx, idx)]
                    if len(trail_vrp) >= 20:
                        t_mean = sum(trail_vrp) / len(trail_vrp)
                        t_var = sum((r - t_mean)**2 for r in trail_vrp) / len(trail_vrp)
                        t_std = math.sqrt(t_var) if t_var > 0 else 1e-10
                        t_sharpe = (t_mean * 365) / (t_std * math.sqrt(365))
                        vrp_w = 0.5 if t_sharpe >= threshold else low_w
                    else:
                        vrp_w = 0.5  # default before enough data
                    bf_w = 1.0 - vrp_w
                    rets.append(vrp_w * pnl_v[d] + bf_w * pnl_b[d])
                s = calc_stats(rets)
                label = f"lb={lookback} thr={threshold} low={low_w}"
                print(f"  {lookback:<10} {threshold:<12.1f} {low_w:<8.1f} {s['sharpe']:8.4f} "
                      f"{s['ann_ret']*100:9.2f}% {s['max_dd']*100:7.2f}%")
                if s["sharpe"] > best_gate["sharpe"]:
                    best_gate = {**s, "lb": lookback, "thr": threshold, "low_w": low_w}

    print(f"\n  Best gate: lb={best_gate.get('lb')} thr={best_gate.get('thr')} "
          f"low_w={best_gate.get('low_w')} → Sharpe {best_gate['sharpe']:.4f}")

    # ═══════════════════════════════════════════════════════════════
    # 2. ROLLING VRP SIGN FILTER
    # ═══════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("  2. ROLLING VRP SIGN FILTER")
    print("=" * 70)
    print("  Zero VRP weight when trailing VRP mean is negative.\n")

    print(f"  {'Lookback':<10} {'Sharpe':>8} {'Return':>10} {'MaxDD':>8} {'%VRP-on':>10}")
    best_sign = {"sharpe": 0}

    for lookback in [20, 30, 60, 90, 120]:
        rets = []
        vrp_on_count = 0
        for idx, d in enumerate(common):
            start_idx = max(0, idx - lookback)
            trail_vrp = [pnl_v[common[j]] for j in range(start_idx, idx)]
            if len(trail_vrp) >= 10 and sum(trail_vrp) / len(trail_vrp) < 0:
                vrp_w = 0.0
            else:
                vrp_w = 0.5
                vrp_on_count += 1
            bf_w = 1.0 - vrp_w
            rets.append(vrp_w * pnl_v[d] + bf_w * pnl_b[d])
        s = calc_stats(rets)
        pct_on = vrp_on_count / len(common) * 100
        print(f"  {lookback:<10} {s['sharpe']:8.4f} {s['ann_ret']*100:9.2f}% "
              f"{s['max_dd']*100:7.2f}% {pct_on:9.1f}%")
        if s["sharpe"] > best_sign["sharpe"]:
            best_sign = {**s, "lb": lookback, "pct_on": pct_on}

    print(f"\n  Best sign filter: lb={best_sign.get('lb')} → "
          f"Sharpe {best_sign['sharpe']:.4f} (VRP on {best_sign.get('pct_on', 0):.0f}%)")

    # ═══════════════════════════════════════════════════════════════
    # 3. IV-LEVEL CONDITIONED WEIGHT
    # ═══════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("  3. IV-LEVEL CONDITIONED WEIGHT")
    print("=" * 70)
    print("  VRP PnL ~ IV². Lower IV → lower VRP weight.\n")

    print(f"  {'IV_thresh':<12} {'High_w':<8} {'Low_w':<8} {'Sharpe':>8} {'Return':>10} {'MaxDD':>8}")
    best_iv = {"sharpe": 0}

    # Compute rolling IV percentile
    for iv_pct_lb in [120, 180, 365]:
        iv_vals = {d: dvol[d] for d in common if d in dvol}
        for high_w, low_w in [(0.6, 0.3), (0.7, 0.2), (0.5, 0.1), (0.5, 0.0)]:
            rets = []
            for idx, d in enumerate(common):
                # Rolling IV percentile
                start_idx = max(0, idx - iv_pct_lb)
                trail_iv = [dvol.get(common[j]) for j in range(start_idx, idx)]
                trail_iv = [v for v in trail_iv if v is not None]
                current_iv = dvol.get(d)
                if len(trail_iv) >= 30 and current_iv:
                    pctile = sum(1 for v in trail_iv if v <= current_iv) / len(trail_iv)
                    vrp_w = high_w if pctile >= 0.5 else low_w
                else:
                    vrp_w = 0.5
                bf_w = 1.0 - vrp_w
                rets.append(vrp_w * pnl_v[d] + bf_w * pnl_b[d])
            s = calc_stats(rets)
            label = f"lb={iv_pct_lb}"
            print(f"  {label:<12} {high_w:<8.1f} {low_w:<8.1f} {s['sharpe']:8.4f} "
                  f"{s['ann_ret']*100:9.2f}% {s['max_dd']*100:7.2f}%")
            if s["sharpe"] > best_iv["sharpe"]:
                best_iv = {**s, "iv_lb": iv_pct_lb, "high_w": high_w, "low_w": low_w}

    print(f"\n  Best IV-cond: lb={best_iv.get('iv_lb')} high={best_iv.get('high_w')} "
          f"low={best_iv.get('low_w')} → Sharpe {best_iv['sharpe']:.4f}")

    # ═══════════════════════════════════════════════════════════════
    # 4. CONTINUOUS WEIGHT = f(trailing VRP Sharpe)
    # ═══════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("  4. CONTINUOUS ADAPTIVE WEIGHT")
    print("=" * 70)
    print("  VRP_w = clip(base + slope * trailing_sharpe, min_w, 0.7)\n")

    print(f"  {'Lookback':<10} {'Base':<8} {'Slope':<8} {'Min_w':<8} {'Sharpe':>8} {'Return':>10} {'MaxDD':>8}")
    best_cont = {"sharpe": 0}

    for lookback in [60, 90, 120]:
        for base in [0.3, 0.4]:
            for slope in [0.05, 0.1, 0.15]:
                for min_w in [0.0, 0.1]:
                    rets = []
                    for idx, d in enumerate(common):
                        start_idx = max(0, idx - lookback)
                        trail = [pnl_v[common[j]] for j in range(start_idx, idx)]
                        if len(trail) >= 20:
                            t_mean = sum(trail) / len(trail)
                            t_var = sum((r - t_mean)**2 for r in trail) / len(trail)
                            t_std = math.sqrt(t_var) if t_var > 0 else 1e-10
                            t_sh = (t_mean * 365) / (t_std * math.sqrt(365))
                            vrp_w = max(min_w, min(0.7, base + slope * t_sh))
                        else:
                            vrp_w = 0.5
                        bf_w = 1.0 - vrp_w
                        rets.append(vrp_w * pnl_v[d] + bf_w * pnl_b[d])
                    s = calc_stats(rets)
                    print(f"  {lookback:<10} {base:<8.2f} {slope:<8.2f} {min_w:<8.2f} "
                          f"{s['sharpe']:8.4f} {s['ann_ret']*100:9.2f}% {s['max_dd']*100:7.2f}%")
                    if s["sharpe"] > best_cont["sharpe"]:
                        best_cont = {**s, "lb": lookback, "base": base, "slope": slope, "min_w": min_w}

    print(f"\n  Best continuous: lb={best_cont.get('lb')} base={best_cont.get('base')} "
          f"slope={best_cont.get('slope')} min_w={best_cont.get('min_w')} → "
          f"Sharpe {best_cont['sharpe']:.4f}")

    # ═══════════════════════════════════════════════════════════════
    # 5. ORACLE (HINDSIGHT) — Upper bound
    # ═══════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("  5. ORACLE HINDSIGHT WEIGHTS (Upper Bound)")
    print("=" * 70)

    # Compute half-yearly oracle weights
    from itertools import product
    periods = defaultdict(list)
    for d in common:
        year = d[:4]
        half = "H1" if d[5:7] <= "06" else "H2"
        periods[f"{year}{half}"].append(d)

    print(f"\n  {'Period':<10} {'Best VRP_w':>10} {'Sharpe':>8}")
    oracle_rets = []
    for period in sorted(periods.keys()):
        days = periods[period]
        best_sh, best_w = -999, 0.5
        for w in [i/20 for i in range(0, 21)]:  # 0.0 to 1.0 in 0.05 steps
            r = [w * pnl_v[d] + (1-w) * pnl_b[d] for d in days]
            s = calc_stats(r)
            if s["sharpe"] > best_sh:
                best_sh = s["sharpe"]
                best_w = w
        print(f"  {period:<10} {best_w:10.2f} {best_sh:8.4f}")
        oracle_rets.extend([best_w * pnl_v[d] + (1-best_w) * pnl_b[d] for d in days])

    os = calc_stats(oracle_rets)
    print(f"\n  Oracle aggregate: Sharpe {os['sharpe']:.4f} Return {os['ann_ret']*100:.2f}% "
          f"MaxDD {os['max_dd']*100:.2f}%")

    # ═══════════════════════════════════════════════════════════════
    # 6. WALK-FORWARD ADAPTIVE
    # ═══════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("  6. WALK-FORWARD ADAPTIVE (REALISTIC)")
    print("=" * 70)
    print("  Train on past 180d, pick best VRP_w, apply next 30d.\n")

    for train_lb in [180, 365]:
        wf_rets = []
        wf_weights = {}
        step = 30  # rebalance every 30 days

        for start_idx in range(train_lb, len(common), step):
            # Train period
            train_days = common[start_idx - train_lb:start_idx]
            # Test period
            test_end = min(start_idx + step, len(common))
            test_days = common[start_idx:test_end]
            if not test_days:
                break

            # Find best weight on train
            best_sh, best_w = -999, 0.5
            for w in [i/10 for i in range(0, 11)]:
                r = [w * pnl_v[d] + (1-w) * pnl_b[d] for d in train_days if d in pnl_v and d in pnl_b]
                if len(r) < 30:
                    continue
                s = calc_stats(r)
                if s["sharpe"] > best_sh:
                    best_sh = s["sharpe"]
                    best_w = w

            # Apply to test
            for d in test_days:
                if d in pnl_v and d in pnl_b:
                    wf_rets.append(best_w * pnl_v[d] + (1-best_w) * pnl_b[d])
                    wf_weights[d] = best_w

        ws = calc_stats(wf_rets)
        avg_w = sum(wf_weights.values()) / len(wf_weights) if wf_weights else 0
        print(f"  Train={train_lb}d: Sharpe {ws['sharpe']:.4f}  Return {ws['ann_ret']*100:.2f}%  "
              f"MaxDD {ws['max_dd']*100:.2f}%  Avg VRP_w: {avg_w:.2f}")

        # Show recent weight path
        recent = sorted(wf_weights.keys())[-12:]
        if recent:
            print(f"    Recent weights: ", end="")
            for d in recent:
                print(f"{d[-5:]}={wf_weights[d]:.1f} ", end="")
            print()

    # ═══════════════════════════════════════════════════════════════
    # 7. COMBINED BEST — merge best signals
    # ═══════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("  7. COMBINED: VRP-SIGN + IV-COND")
    print("=" * 70)
    print("  VRP weight = 0 if trail VRP mean < 0, else IV-conditioned.\n")

    for sign_lb in [30, 60]:
        for iv_lb in [120, 180]:
            for high_w, low_w in [(0.6, 0.3), (0.5, 0.2)]:
                rets = []
                for idx, d in enumerate(common):
                    # Sign filter
                    si = max(0, idx - sign_lb)
                    trail_vrp = [pnl_v[common[j]] for j in range(si, idx)]
                    if len(trail_vrp) >= 10 and sum(trail_vrp) / len(trail_vrp) < 0:
                        vrp_w = 0.0
                    else:
                        # IV conditioned
                        si2 = max(0, idx - iv_lb)
                        trail_iv = [dvol.get(common[j]) for j in range(si2, idx)]
                        trail_iv = [v for v in trail_iv if v is not None]
                        current_iv = dvol.get(d)
                        if len(trail_iv) >= 30 and current_iv:
                            pctile = sum(1 for v in trail_iv if v <= current_iv) / len(trail_iv)
                            vrp_w = high_w if pctile >= 0.5 else low_w
                        else:
                            vrp_w = 0.5
                    bf_w = 1.0 - vrp_w
                    rets.append(vrp_w * pnl_v[d] + bf_w * pnl_b[d])
                s = calc_stats(rets)
                print(f"  sign_lb={sign_lb} iv_lb={iv_lb} hi={high_w} lo={low_w}: "
                      f"Sharpe {s['sharpe']:.4f}  Return {s['ann_ret']*100:.2f}%  MaxDD {s['max_dd']*100:.2f}%")

    # ═══════════════════════════════════════════════════════════════
    # 8. RECENT PERIOD COMPARISON
    # ═══════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("  8. RECENT PERIOD PERFORMANCE (2025-2026)")
    print("=" * 70)

    recent_dates = [d for d in common if d >= "2025-01-01"]
    if recent_dates:
        # Static 50/50
        r_base = [0.5 * pnl_v[d] + 0.5 * pnl_b[d] for d in recent_dates]
        s_base = calc_stats(r_base)

        # BF only (0/100)
        r_bf = [pnl_b[d] for d in recent_dates]
        s_bf = calc_stats(r_bf)

        # VRP only
        r_vrp = [pnl_v[d] for d in recent_dates]
        s_vrp = calc_stats(r_vrp)

        # Best sign filter
        best_slb = best_sign.get("lb", 30)
        r_sign = []
        for idx_full, d in enumerate(common):
            if d < "2025-01-01":
                continue
            idx_start = max(0, idx_full - best_slb)
            trail = [pnl_v[common[j]] for j in range(idx_start, idx_full)]
            if len(trail) >= 10 and sum(trail) / len(trail) < 0:
                r_sign.append(pnl_b[d])
            else:
                r_sign.append(0.5 * pnl_v[d] + 0.5 * pnl_b[d])
        s_sign = calc_stats(r_sign)

        # 20/80 fixed
        r_2080 = [0.2 * pnl_v[d] + 0.8 * pnl_b[d] for d in recent_dates]
        s_2080 = calc_stats(r_2080)

        print(f"\n  {'Strategy':<25} {'Sharpe':>8} {'Return':>10} {'MaxDD':>8}")
        print(f"  {'VRP only (100/0)':<25} {s_vrp['sharpe']:8.4f} {s_vrp['ann_ret']*100:9.2f}% {s_vrp['max_dd']*100:7.2f}%")
        print(f"  {'Static 50/50':<25} {s_base['sharpe']:8.4f} {s_base['ann_ret']*100:9.2f}% {s_base['max_dd']*100:7.2f}%")
        print(f"  {'Static 20/80':<25} {s_2080['sharpe']:8.4f} {s_2080['ann_ret']*100:9.2f}% {s_2080['max_dd']*100:7.2f}%")
        print(f"  {'BF only (0/100)':<25} {s_bf['sharpe']:8.4f} {s_bf['ann_ret']*100:9.2f}% {s_bf['max_dd']*100:7.2f}%")
        print(f"  {'Sign filter (lb={best_slb})':<25} {s_sign['sharpe']:8.4f} {s_sign['ann_ret']*100:9.2f}% {s_sign['max_dd']*100:7.2f}%")

    # ═══════════════════════════════════════════════════════════════
    # CONCLUSION
    # ═══════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("R66: CONCLUSION")
    print("=" * 70)

    all_best = [
        ("Static 50/50 (R60)", bs["sharpe"], bs["max_dd"]),
        ("VRP Sharpe Gate", best_gate["sharpe"], best_gate.get("max_dd", 0)),
        ("VRP Sign Filter", best_sign["sharpe"], best_sign.get("max_dd", 0)),
        ("IV-Level Cond", best_iv["sharpe"], best_iv.get("max_dd", 0)),
        ("Continuous Adaptive", best_cont["sharpe"], best_cont.get("max_dd", 0)),
        ("Oracle (hindsight)", os["sharpe"], os["max_dd"]),
    ]

    print(f"\n  {'Method':<25} {'Sharpe':>8} {'MaxDD':>8} {'vs R60':>8}")
    for name, sh, dd in sorted(all_best, key=lambda x: -x[1]):
        delta = sh - bs["sharpe"]
        print(f"  {name:<25} {sh:8.4f} {dd*100:7.2f}% {delta:+8.4f}")

    best_adaptive = max(all_best[:5], key=lambda x: x[1])
    delta_vs_r60 = best_adaptive[1] - bs["sharpe"]
    pct_to_oracle = (best_adaptive[1] - bs["sharpe"]) / (os["sharpe"] - bs["sharpe"]) * 100 if os["sharpe"] > bs["sharpe"] else 0

    if delta_vs_r60 > 0.1:
        verdict = "CONFIRMED — adaptive weighting improves on static 50/50"
    elif delta_vs_r60 > 0.03:
        verdict = "MARGINAL — small improvement, may not survive OOS"
    else:
        verdict = "NEGATIVE — adaptive weighting adds no value over static"

    print(f"\n  Best adaptive: {best_adaptive[0]} → Sharpe {best_adaptive[1]:.4f}")
    print(f"  Delta vs R60: {delta_vs_r60:+.4f}")
    print(f"  Captures {pct_to_oracle:.0f}% of oracle improvement")
    print(f"\n  VERDICT: {verdict}")

    # Save results
    out_path = ROOT / "data" / "cache" / "deribit" / "real_surface" / "r66_adaptive_weights_results.json"
    results = {
        "research_id": "R66",
        "title": "Adaptive VRP/BF Weighting",
        "baseline_sharpe": round(bs["sharpe"], 4),
        "baseline_maxdd": round(bs["max_dd"], 6),
        "best_gate": {
            "lb": best_gate.get("lb"), "thr": best_gate.get("thr"),
            "low_w": best_gate.get("low_w"), "sharpe": round(best_gate["sharpe"], 4),
        },
        "best_sign": {
            "lb": best_sign.get("lb"), "sharpe": round(best_sign["sharpe"], 4),
            "pct_on": round(best_sign.get("pct_on", 0), 1),
        },
        "best_iv_cond": {
            "iv_lb": best_iv.get("iv_lb"), "high_w": best_iv.get("high_w"),
            "low_w": best_iv.get("low_w"), "sharpe": round(best_iv["sharpe"], 4),
        },
        "best_continuous": {
            "lb": best_cont.get("lb"), "base": best_cont.get("base"),
            "slope": best_cont.get("slope"), "min_w": best_cont.get("min_w"),
            "sharpe": round(best_cont["sharpe"], 4),
        },
        "oracle_sharpe": round(os["sharpe"], 4),
        "best_adaptive_method": best_adaptive[0],
        "best_adaptive_sharpe": round(best_adaptive[1], 4),
        "delta_vs_r60": round(delta_vs_r60, 4),
        "pct_to_oracle": round(pct_to_oracle, 1),
        "verdict": verdict,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved to {out_path}")


if __name__ == "__main__":
    main()
