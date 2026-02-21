#!/usr/bin/env python3
"""
R74: iv_atm MR as Butterfly Complement
=========================================

From R73 finding: iv_atm mean-reversion showed:
  - BF PnL correlation: -0.033 (nearly perfectly uncorrelated — IDEAL)
  - In-sample best combo: Sharpe 3.903 (+0.142 vs R69 baseline)
  - Standalone Sharpe: 1.073

Key question: Is iv_atm MR a genuine complement, or is this in-sample luck?

Tests:
  1. iv_atm standalone parameter sweep (lookback, z_entry, z_exit)
  2. Deep correlation analysis (per-year, per-regime)
  3. Weighted ensemble optimization (VRP + BF + iv_atm)
  4. Walk-forward OOS validation
  5. LOYO validation
  6. Yearly performance decomposition
  7. Regime-conditional analysis (does iv_atm help in specific regimes?)
"""
import csv
import json
import math
import subprocess
import time
from collections import defaultdict
from datetime import datetime, timezone, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


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
            for field in ["butterfly_25d", "iv_atm", "skew_25d", "iv_25d_put", "iv_25d_call"]:
                val = row.get(field, "")
                if val and val != "None":
                    entry[field] = float(val)
            if entry:
                data[d] = entry
    return data


# ═══════════════════════════════════════════════════════════════
# PnL Computation
# ═══════════════════════════════════════════════════════════════

def compute_vrp_pnl(dvol_hist, price_hist, dates, leverage=2.0):
    dt = 1.0 / 365.0
    vrp_pnl = {}
    for i in range(1, len(dates)):
        d, dp = dates[i], dates[i-1]
        iv = dvol_hist.get(dp)
        p0, p1 = price_hist.get(dp), price_hist.get(d)
        if not all([iv, p0, p1]) or p0 <= 0:
            continue
        rv = abs(math.log(p1 / p0)) * math.sqrt(365)
        vrp_pnl[d] = leverage * 0.5 * (iv**2 - rv**2) * dt
    return vrp_pnl


def compute_feature_mr_pnl(dvol_hist, surface_hist, dates, feature_key,
                            lookback=120, z_entry=1.5, z_exit=0.0,
                            sensitivity=2.5):
    feat_vals = {}
    for d in dates:
        if d in surface_hist and feature_key in surface_hist[d]:
            feat_vals[d] = surface_hist[d][feature_key]

    dt = 1.0 / 365.0
    position = 0.0
    pnl = {}

    for i in range(lookback, len(dates)):
        d, dp = dates[i], dates[i-1]
        val = feat_vals.get(d)
        if val is None:
            continue

        window = [feat_vals.get(dates[j]) for j in range(i-lookback, i)]
        window = [v for v in window if v is not None]
        if len(window) < lookback // 2:
            continue

        mean = sum(window) / len(window)
        std = math.sqrt(sum((v - mean)**2 for v in window) / len(window))
        if std < 1e-8:
            continue

        z = (val - mean) / std

        if z > z_entry:
            position = -1.0
        elif z < -z_entry:
            position = 1.0
        elif z_exit > 0 and abs(z) < z_exit:
            position = 0.0

        iv = dvol_hist.get(d)
        f_now, f_prev = feat_vals.get(d), feat_vals.get(dp)
        if f_now is not None and f_prev is not None and iv is not None and position != 0:
            pnl[d] = position * (f_now - f_prev) * iv * math.sqrt(dt) * sensitivity
        else:
            pnl[d] = 0.0

    return pnl


def compute_stats(rets):
    if len(rets) < 30:
        return {"sharpe": 0, "ann_ret": 0, "max_dd": 0, "calmar": 0, "n": len(rets)}
    mean = sum(rets) / len(rets)
    std = math.sqrt(sum((r - mean)**2 for r in rets) / len(rets))
    sharpe = (mean * 365) / (std * math.sqrt(365)) if std > 0 else 0
    ann_ret = mean * 365
    cum = peak = max_dd = 0.0
    for r in rets:
        cum += r
        peak = max(peak, cum)
        max_dd = max(max_dd, peak - cum)
    calmar = ann_ret / max_dd if max_dd > 0 else 0
    return {"sharpe": sharpe, "ann_ret": ann_ret, "max_dd": max_dd, "calmar": calmar, "n": len(rets)}


def compute_correlation(x, y):
    n = len(x)
    if n < 10:
        return 0
    mx = sum(x) / n
    my = sum(y) / n
    cov = sum((a - mx) * (b - my) for a, b in zip(x, y)) / n
    sx = math.sqrt(sum((a - mx)**2 for a in x) / n)
    sy = math.sqrt(sum((b - my)**2 for b in y) / n)
    if sx < 1e-10 or sy < 1e-10:
        return 0
    return cov / (sx * sy)


# ═══════════════════════════════════════════════════════════════
# Analysis 1: iv_atm Standalone Parameter Sweep
# ═══════════════════════════════════════════════════════════════

def sweep_iv_atm_params(dvol_hist, surface_hist, dates):
    print("\n" + "=" * 70)
    print("  ANALYSIS 1: iv_atm STANDALONE PARAMETER SWEEP")
    print("=" * 70)

    results = []
    for lb in [30, 60, 90, 120, 150, 180, 252]:
        for z_in in [0.5, 1.0, 1.5, 2.0, 2.5]:
            for z_out in [0.0, 0.3, 0.5]:
                pnl = compute_feature_mr_pnl(dvol_hist, surface_hist, dates,
                                             "iv_atm", lb, z_in, z_out, 2.5)
                if len(pnl) < 100:
                    continue
                rets = [pnl[d] for d in sorted(pnl.keys())]
                stats = compute_stats(rets)
                results.append({"lb": lb, "z_in": z_in, "z_out": z_out, **stats})

    results.sort(key=lambda x: x["sharpe"], reverse=True)

    print(f"\n  Total configs: {len(results)}")
    print(f"\n  Top 15:")
    print(f"  {'LB':>4} {'z_in':>5} {'z_out':>6} {'Sharpe':>8} {'AnnRet':>10} {'MaxDD':>8}")
    for r in results[:15]:
        print(f"  {r['lb']:>4} {r['z_in']:>5.1f} {r['z_out']:>6.1f} {r['sharpe']:>8.3f} "
              f"{r['ann_ret']*100:>9.2f}% {r['max_dd']*100:>7.2f}%")

    # How many positive?
    pos = sum(1 for r in results if r["sharpe"] > 0)
    above_1 = sum(1 for r in results if r["sharpe"] > 1.0)
    print(f"\n  Sharpe > 0: {pos}/{len(results)} ({pos/len(results)*100:.0f}%)")
    print(f"  Sharpe > 1: {above_1}/{len(results)} ({above_1/len(results)*100:.0f}%)")

    return results[0] if results else None


# ═══════════════════════════════════════════════════════════════
# Analysis 2: Deep Correlation Analysis
# ═══════════════════════════════════════════════════════════════

def deep_correlation(vrp_pnl, bf_pnl, iv_pnl, dvol_hist, dates):
    print("\n" + "=" * 70)
    print("  ANALYSIS 2: DEEP CORRELATION ANALYSIS")
    print("=" * 70)

    common = sorted(set(vrp_pnl.keys()) & set(bf_pnl.keys()) & set(iv_pnl.keys()))

    v = [vrp_pnl[d] for d in common]
    b = [bf_pnl[d] for d in common]
    iv = [iv_pnl[d] for d in common]

    print(f"\n  Full-sample PnL correlations ({len(common)} days):")
    print(f"    VRP-BF:    {compute_correlation(v, b):.3f}")
    print(f"    VRP-IV:    {compute_correlation(v, iv):.3f}")
    print(f"    BF-IV:     {compute_correlation(b, iv):.3f}")

    # Per-year
    print(f"\n  Per-year BF-IV correlation:")
    by_year = defaultdict(lambda: ([], []))
    for d in common:
        by_year[d[:4]][0].append(bf_pnl[d])
        by_year[d[:4]][1].append(iv_pnl[d])

    for yr in sorted(by_year.keys()):
        c = compute_correlation(by_year[yr][0], by_year[yr][1])
        n = len(by_year[yr][0])
        print(f"    {yr}: {c:+.3f} (n={n})")

    # Per-regime (IV level buckets)
    print(f"\n  Per-IV-regime BF-IV correlation:")
    for lo, hi, label in [(0, 0.50, "IV<50%"), (0.50, 0.60, "IV 50-60%"),
                           (0.60, 0.80, "IV 60-80%"), (0.80, 2.0, "IV>80%")]:
        bf_r, iv_r = [], []
        for d in common:
            dvol = dvol_hist.get(d, 0.5)
            if lo <= dvol < hi:
                bf_r.append(bf_pnl[d])
                iv_r.append(iv_pnl[d])
        if len(bf_r) > 30:
            c = compute_correlation(bf_r, iv_r)
            print(f"    {label:>12}: {c:+.3f} (n={len(bf_r)})")


# ═══════════════════════════════════════════════════════════════
# Analysis 3: Weighted Ensemble Optimization
# ═══════════════════════════════════════════════════════════════

def optimize_ensemble(vrp_pnl, bf_pnl, iv_pnl, dates):
    print("\n" + "=" * 70)
    print("  ANALYSIS 3: WEIGHTED ENSEMBLE OPTIMIZATION")
    print("=" * 70)

    common = sorted(set(vrp_pnl.keys()) & set(bf_pnl.keys()) & set(iv_pnl.keys()))

    base_rets = [0.10 * vrp_pnl[d] + 0.90 * bf_pnl[d] for d in common]
    base_stats = compute_stats(base_rets)
    print(f"\n  Baseline R69: Sharpe={base_stats['sharpe']:.4f}")

    # Fine-grained sweep: w_vrp 5-20%, w_iv 0-30%, rest=BF
    results = []
    for w_vrp_pct in range(5, 25, 5):
        for w_iv_pct in range(0, 35, 5):
            w_bf_pct = 100 - w_vrp_pct - w_iv_pct
            if w_bf_pct < 30:
                continue
            w_v, w_b, w_i = w_vrp_pct/100, w_bf_pct/100, w_iv_pct/100

            rets = [w_v * vrp_pnl[d] + w_b * bf_pnl[d] + w_i * iv_pnl[d] for d in common]
            stats = compute_stats(rets)
            results.append({"w_vrp": w_v, "w_bf": w_b, "w_iv": w_i, **stats})

    results.sort(key=lambda x: x["sharpe"], reverse=True)

    print(f"\n  Top 15 configs (of {len(results)}):")
    print(f"  {'VRP':>6} {'BF':>6} {'IV':>6} {'Sharpe':>8} {'AnnRet':>10} {'MaxDD':>8} {'Calmar':>8}")
    for r in results[:15]:
        print(f"  {r['w_vrp']*100:>5.0f}% {r['w_bf']*100:>5.0f}% {r['w_iv']*100:>5.0f}% "
              f"{r['sharpe']:>8.3f} {r['ann_ret']*100:>9.2f}% {r['max_dd']*100:>7.2f}% {r['calmar']:>8.2f}")

    best = results[0]
    delta = best["sharpe"] - base_stats["sharpe"]
    print(f"\n  Best: VRP={best['w_vrp']:.0%} BF={best['w_bf']:.0%} IV={best['w_iv']:.0%}")
    print(f"  Sharpe={best['sharpe']:.4f} (Δ={delta:+.4f})")

    return results, base_stats


# ═══════════════════════════════════════════════════════════════
# Analysis 4: Walk-Forward OOS
# ═══════════════════════════════════════════════════════════════

def walk_forward(vrp_pnl, bf_pnl, iv_pnl, dates):
    print("\n" + "=" * 70)
    print("  ANALYSIS 4: WALK-FORWARD OOS VALIDATION")
    print("=" * 70)

    common = sorted(set(vrp_pnl.keys()) & set(bf_pnl.keys()) & set(iv_pnl.keys()))
    n = len(common)
    step = 180
    min_train = 365

    wf_oos = []
    wf_static = []

    print(f"\n  {'Period':>24} {'Train':>6} {'Best w_iv':>9} {'OOS 3-way':>12} {'Static':>8}")
    print(f"  {'─'*24} {'─'*6} {'─'*9} {'─'*12} {'─'*8}")

    i = min_train
    while i + step <= n:
        train = common[:i]
        test = common[i:i+step]

        # Optimize on train
        best_s, best_cfg = -999, (0.10, 0.90, 0.00)
        for w_vrp_pct in range(5, 25, 5):
            for w_iv_pct in range(0, 35, 5):
                w_bf_pct = 100 - w_vrp_pct - w_iv_pct
                if w_bf_pct < 30:
                    continue
                w_v, w_b, w_i = w_vrp_pct/100, w_bf_pct/100, w_iv_pct/100
                rets = [w_v * vrp_pnl[d] + w_b * bf_pnl[d] + w_i * iv_pnl[d]
                        for d in train if d in vrp_pnl and d in bf_pnl and d in iv_pnl]
                if len(rets) < 100:
                    continue
                stats = compute_stats(rets)
                if stats["sharpe"] > best_s:
                    best_s = stats["sharpe"]
                    best_cfg = (w_v, w_b, w_i)

        # Test OOS
        w_v, w_b, w_i = best_cfg
        oos_rets = [w_v * vrp_pnl[d] + w_b * bf_pnl[d] + w_i * iv_pnl[d]
                    for d in test if d in vrp_pnl and d in bf_pnl and d in iv_pnl]
        static_rets = [0.10 * vrp_pnl[d] + 0.90 * bf_pnl[d]
                       for d in test if d in vrp_pnl and d in bf_pnl]

        oos_s = compute_stats(oos_rets)
        static_s = compute_stats(static_rets)
        wf_oos.append(oos_s["sharpe"])
        wf_static.append(static_s["sharpe"])

        period = f"{test[0]} to {test[-1]}"
        print(f"  {period:>24} {len(train):>6} {w_i:>8.0%} {oos_s['sharpe']:>12.3f} {static_s['sharpe']:>8.3f}")

        i += step

    if wf_oos:
        avg_oos = sum(wf_oos) / len(wf_oos)
        avg_static = sum(wf_static) / len(wf_static)
        wins = sum(1 for o, s in zip(wf_oos, wf_static) if o > s)
        print(f"\n  WF avg (3-way):     {avg_oos:.3f}")
        print(f"  WF avg (static):    {avg_static:.3f}")
        print(f"  Delta:              {avg_oos - avg_static:+.3f}")
        print(f"  Wins:               {wins}/{len(wf_oos)}")

    return wf_oos, wf_static


# ═══════════════════════════════════════════════════════════════
# Analysis 5: LOYO Validation
# ═══════════════════════════════════════════════════════════════

def loyo(vrp_pnl, bf_pnl, iv_pnl, dates):
    print("\n" + "=" * 70)
    print("  ANALYSIS 5: LOYO VALIDATION")
    print("=" * 70)

    common = sorted(set(vrp_pnl.keys()) & set(bf_pnl.keys()) & set(iv_pnl.keys()))
    years = sorted(set(d[:4] for d in common))

    print(f"\n  {'Year':>6} {'Static':>8} {'3-way':>8} {'w_iv':>6} {'Delta':>8}")
    print(f"  {'─'*6} {'─'*8} {'─'*8} {'─'*6} {'─'*8}")

    loyo_wins = 0
    for test_yr in years:
        train = [d for d in common if d[:4] != test_yr]
        test = [d for d in common if d[:4] == test_yr]
        if len(test) < 30:
            continue

        # Optimize on train
        best_s, best_cfg = -999, (0.10, 0.90, 0.00)
        for w_vrp_pct in range(5, 25, 5):
            for w_iv_pct in range(0, 35, 5):
                w_bf_pct = 100 - w_vrp_pct - w_iv_pct
                if w_bf_pct < 30:
                    continue
                w_v, w_b, w_i = w_vrp_pct/100, w_bf_pct/100, w_iv_pct/100
                rets = [w_v * vrp_pnl[d] + w_b * bf_pnl[d] + w_i * iv_pnl[d] for d in train]
                stats = compute_stats(rets)
                if stats["sharpe"] > best_s:
                    best_s = stats["sharpe"]
                    best_cfg = (w_v, w_b, w_i)

        w_v, w_b, w_i = best_cfg
        oos_rets = [w_v * vrp_pnl[d] + w_b * bf_pnl[d] + w_i * iv_pnl[d] for d in test]
        static_rets = [0.10 * vrp_pnl[d] + 0.90 * bf_pnl[d] for d in test]

        oos_s = compute_stats(oos_rets)
        static_s = compute_stats(static_rets)
        delta = oos_s["sharpe"] - static_s["sharpe"]
        if delta > 0:
            loyo_wins += 1

        print(f"  {test_yr:>6} {static_s['sharpe']:>8.2f} {oos_s['sharpe']:>8.2f} {w_i:>5.0%} {delta:>+8.2f}")

    print(f"\n  LOYO wins: {loyo_wins}/{len(years)}")


# ═══════════════════════════════════════════════════════════════
# Analysis 6: Yearly Performance Decomposition
# ═══════════════════════════════════════════════════════════════

def yearly_decomposition(vrp_pnl, bf_pnl, iv_pnl, dates):
    print("\n" + "=" * 70)
    print("  ANALYSIS 6: YEARLY PERFORMANCE DECOMPOSITION")
    print("=" * 70)

    common = sorted(set(vrp_pnl.keys()) & set(bf_pnl.keys()) & set(iv_pnl.keys()))

    # Fixed config: 10% VRP + 75% BF + 15% IV (a reasonable test split)
    print(f"\n  Config: 10% VRP + 75% BF + 15% IV_ATM")
    print(f"\n  {'Year':>6} {'VRP':>8} {'BF':>8} {'IV_ATM':>8} {'10/90':>8} {'10/75/15':>10} {'Delta':>8}")
    print(f"  {'─'*6} {'─'*8} {'─'*8} {'─'*8} {'─'*8} {'─'*10} {'─'*8}")

    by_year = defaultdict(list)
    for d in common:
        by_year[d[:4]].append(d)

    for yr in sorted(by_year.keys()):
        yr_dates = by_year[yr]
        if len(yr_dates) < 30:
            continue

        v_rets = [vrp_pnl[d] for d in yr_dates]
        b_rets = [bf_pnl[d] for d in yr_dates]
        i_rets = [iv_pnl[d] for d in yr_dates]

        static_rets = [0.10 * vrp_pnl[d] + 0.90 * bf_pnl[d] for d in yr_dates]
        combo_rets = [0.10 * vrp_pnl[d] + 0.75 * bf_pnl[d] + 0.15 * iv_pnl[d] for d in yr_dates]

        v_s = compute_stats(v_rets)
        b_s = compute_stats(b_rets)
        i_s = compute_stats(i_rets)
        static_s = compute_stats(static_rets)
        combo_s = compute_stats(combo_rets)

        delta = combo_s["sharpe"] - static_s["sharpe"]
        print(f"  {yr:>6} {v_s['sharpe']:>8.2f} {b_s['sharpe']:>8.2f} {i_s['sharpe']:>8.2f} "
              f"{static_s['sharpe']:>8.2f} {combo_s['sharpe']:>10.2f} {delta:>+8.2f}")


# ═══════════════════════════════════════════════════════════════
# Analysis 7: iv_atm MR Signal Characteristics
# ═══════════════════════════════════════════════════════════════

def iv_atm_signal_characteristics(dvol_hist, surface_hist, dates):
    print("\n" + "=" * 70)
    print("  ANALYSIS 7: iv_atm MR SIGNAL CHARACTERISTICS")
    print("=" * 70)

    iv_vals = {}
    for d in dates:
        if d in surface_hist and "iv_atm" in surface_hist[d]:
            iv_vals[d] = surface_hist[d]["iv_atm"]

    lb = 120
    positions = {}
    z_scores = {}
    position = 0.0

    for i in range(lb, len(dates)):
        d = dates[i]
        val = iv_vals.get(d)
        if val is None:
            continue
        window = [iv_vals.get(dates[j]) for j in range(i-lb, i)]
        window = [v for v in window if v is not None]
        if len(window) < lb // 2:
            continue
        mean = sum(window) / len(window)
        std = math.sqrt(sum((v - mean)**2 for v in window) / len(window))
        if std < 1e-8:
            continue
        z = (val - mean) / std
        z_scores[d] = z

        if z > 1.5:
            position = -1.0  # Short IV → expect IV to fall (vol selling)
        elif z < -1.5:
            position = 1.0   # Long IV → expect IV to rise (vol buying)

        positions[d] = position

    if not positions:
        print("  No data")
        return

    pd = sorted(positions.keys())

    # Position distribution
    n_long = sum(1 for d in pd if positions[d] > 0)
    n_short = sum(1 for d in pd if positions[d] < 0)
    n_flat = sum(1 for d in pd if positions[d] == 0)
    print(f"\n  Position distribution:")
    print(f"    Long IV (buy vol):  {n_long:4d} ({n_long/len(pd)*100:.0f}%)")
    print(f"    Short IV (sell vol):{n_short:4d} ({n_short/len(pd)*100:.0f}%)")
    print(f"    Flat:               {n_flat:4d} ({n_flat/len(pd)*100:.0f}%)")

    # z-score distribution
    z_list = [z_scores[d] for d in sorted(z_scores.keys())]
    print(f"\n  z-score distribution:")
    print(f"    Mean: {sum(z_list)/len(z_list):.3f}")
    print(f"    Std:  {math.sqrt(sum((z-sum(z_list)/len(z_list))**2 for z in z_list)/len(z_list)):.3f}")
    print(f"    |z|>1.5: {sum(1 for z in z_list if abs(z)>1.5)/len(z_list)*100:.0f}%")

    # Trade count
    trades = 0
    prev_pos = 0
    for d in pd:
        if positions[d] != prev_pos:
            trades += 1
        prev_pos = positions[d]

    years = (datetime.strptime(pd[-1], "%Y-%m-%d") - datetime.strptime(pd[0], "%Y-%m-%d")).days / 365.25
    print(f"\n  Trade count: {trades} ({trades/years:.1f}/year)")
    print(f"  Time in position: {(n_long+n_short)/len(pd)*100:.0f}%")

    # What does "short IV" mean practically?
    print(f"\n  KEY QUESTION: What does iv_atm MR actually trade?")
    print(f"    When z>1.5: IV is ABOVE mean → SHORT IV (sell straddles) → expect IV to fall")
    print(f"    When z<-1.5: IV is BELOW mean → LONG IV (buy straddles) → expect IV to rise")
    print(f"    This is essentially a CONTRARIAN volatility bet")
    print(f"    NOTE: Overlaps with VRP concept but trades IV level, not spread")


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════

def main():
    print("=" * 70)
    print("  R74: iv_atm MR AS BUTTERFLY COMPLEMENT")
    print("=" * 70)
    print("  Loading data...")

    dvol_hist = load_dvol_history("BTC")
    price_hist = load_price_history("BTC")
    surface_hist = load_surface("BTC")

    dates = sorted(set(dvol_hist.keys()) & set(price_hist.keys()))
    print(f"  Dates: {dates[0]} to {dates[-1]} ({len(dates)} days)")

    # Compute PnL series
    vrp_pnl = compute_vrp_pnl(dvol_hist, price_hist, dates)
    bf_pnl = compute_feature_mr_pnl(dvol_hist, surface_hist, dates,
                                     "butterfly_25d", 120, 1.5, 0.0, 2.5)

    # Best iv_atm config from R73 was lb=120, z_in=1.5, z_out=0.0
    # But let's sweep first
    best_iv = sweep_iv_atm_params(dvol_hist, surface_hist, dates)

    # Use default params for rest of analysis (avoid overfitting the sweep)
    iv_pnl = compute_feature_mr_pnl(dvol_hist, surface_hist, dates,
                                     "iv_atm", 120, 1.5, 0.0, 2.5)

    print(f"\n  VRP: {len(vrp_pnl)} days, BF: {len(bf_pnl)} days, IV: {len(iv_pnl)} days")

    # Analyses
    deep_correlation(vrp_pnl, bf_pnl, iv_pnl, dvol_hist, dates)
    ensemble_results, base_stats = optimize_ensemble(vrp_pnl, bf_pnl, iv_pnl, dates)
    wf_oos, wf_static = walk_forward(vrp_pnl, bf_pnl, iv_pnl, dates)
    loyo(vrp_pnl, bf_pnl, iv_pnl, dates)
    yearly_decomposition(vrp_pnl, bf_pnl, iv_pnl, dates)
    iv_atm_signal_characteristics(dvol_hist, surface_hist, dates)

    # ─── Summary ────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  SUMMARY")
    print("=" * 70)

    best = ensemble_results[0]
    delta = best["sharpe"] - base_stats["sharpe"]

    print(f"\n  R69 Baseline: Sharpe {base_stats['sharpe']:.4f}")
    print(f"  Best 3-way:   VRP={best['w_vrp']:.0%} BF={best['w_bf']:.0%} IV={best['w_iv']:.0%} → Sharpe {best['sharpe']:.4f}")
    print(f"  Delta:        {delta:+.4f}")

    # WF summary
    if wf_oos and wf_static:
        avg_oos = sum(wf_oos) / len(wf_oos)
        avg_static = sum(wf_static) / len(wf_static)
        wf_delta = avg_oos - avg_static
        wins = sum(1 for o, s in zip(wf_oos, wf_static) if o > s)
        print(f"  WF OOS:       {avg_oos:.3f} vs static {avg_static:.3f} (Δ={wf_delta:+.3f}, wins {wins}/{len(wf_oos)})")

    if delta > 0.10 and wf_oos and sum(1 for o, s in zip(wf_oos, wf_static) if o > s) >= len(wf_oos) * 0.5:
        verdict = "UPGRADE — iv_atm improves portfolio"
    elif delta > 0:
        verdict = "MARGINAL — in-sample improvement doesn't survive OOS"
    else:
        verdict = "NEGATIVE — iv_atm does NOT improve portfolio"

    print(f"\n  VERDICT: {verdict}")

    # Save
    results = {
        "research_id": "R74",
        "title": "iv_atm MR as Butterfly Complement",
        "baseline_r69": {"sharpe": round(base_stats["sharpe"], 4)},
        "best_ensemble": {
            "w_vrp": best["w_vrp"], "w_bf": best["w_bf"], "w_iv": best["w_iv"],
            "sharpe": round(best["sharpe"], 4),
        },
        "iv_atm_standalone_sharpe": round(best_iv["sharpe"], 4) if best_iv else 0,
        "delta_sharpe": round(delta, 4),
        "verdict": verdict,
    }

    out_path = ROOT / "data" / "cache" / "deribit" / "real_surface" / "r74_iv_atm_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved to {out_path}")


if __name__ == "__main__":
    main()
