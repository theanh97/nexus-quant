#!/usr/bin/env python3
"""
R73: bf_ratio as Butterfly Complement
=======================================

From R68: bf_ratio (butterfly_25d / iv_atm) showed:
  - Standalone Sharpe 2.04 (second-best feature after butterfly_25d at 2.64)
  - Lowest VRP correlation: 0.034
  - Correlation with butterfly_25d: 0.879 (HIGH — may not add much)

Key question: Can bf_ratio COMBINED with butterfly_25d improve the portfolio?

Tests:
  1. bf_ratio standalone optimization (lookback, z_entry, z_exit)
  2. bf_ratio + butterfly_25d correlation on PnL level
  3. Weighted ensemble: VRP + BF_butterfly + BF_ratio
  4. Walk-forward OOS validation
  5. LOYO validation
  6. Compare vs R69 baseline (Sharpe 3.76)
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
    """Compute VRP PnL."""
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
    """Compute feature mean-reversion PnL.

    For raw surface features: PnL = pos * (f_now - f_prev) * IV * sqrt(dt) * sensitivity
    For ratio features: same model but on the ratio values
    """
    # Extract feature values
    feat_vals = {}
    for d in dates:
        if d in surface_hist:
            s = surface_hist[d]
            if feature_key == "bf_ratio":
                bf = s.get("butterfly_25d")
                iv = s.get("iv_atm")
                if bf is not None and iv is not None and iv > 0.01:
                    feat_vals[d] = bf / iv
            elif feature_key in s:
                feat_vals[d] = s[feature_key]

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
    """Pearson correlation between two lists."""
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
# Analysis 1: bf_ratio Standalone Optimization
# ═══════════════════════════════════════════════════════════════

def optimize_bf_ratio(dvol_hist, surface_hist, dates):
    """Sweep bf_ratio parameters."""
    print("\n" + "=" * 70)
    print("  ANALYSIS 1: bf_ratio STANDALONE OPTIMIZATION")
    print("=" * 70)

    results = []
    for lb in [60, 90, 120, 150, 180]:
        for z_in in [1.0, 1.5, 2.0]:
            for z_out in [0.0, 0.3, 0.5]:
                pnl = compute_feature_mr_pnl(dvol_hist, surface_hist, dates,
                                             "bf_ratio", lb, z_in, z_out, 2.5)
                if not pnl:
                    continue
                common_dates = sorted(pnl.keys())
                rets = [pnl[d] for d in common_dates]
                stats = compute_stats(rets)
                results.append({
                    "lb": lb, "z_in": z_in, "z_out": z_out,
                    **stats
                })

    results.sort(key=lambda x: x["sharpe"], reverse=True)

    print(f"\n  Top 10 configs (of {len(results)}):")
    print(f"  {'LB':>4} {'z_in':>5} {'z_out':>6} {'Sharpe':>8} {'AnnRet':>10} {'MaxDD':>8} {'N':>5}")
    for r in results[:10]:
        print(f"  {r['lb']:>4} {r['z_in']:>5.1f} {r['z_out']:>6.1f} {r['sharpe']:>8.3f} "
              f"{r['ann_ret']*100:>9.2f}% {r['max_dd']*100:>7.2f}% {r['n']:>5}")

    best = results[0]
    print(f"\n  Best: lb={best['lb']}, z_in={best['z_in']}, z_out={best['z_out']} → Sharpe {best['sharpe']:.3f}")

    return best


# ═══════════════════════════════════════════════════════════════
# Analysis 2: PnL Correlation Matrix
# ═══════════════════════════════════════════════════════════════

def analyze_pnl_correlations(vrp_pnl, bf_pnl, ratio_pnl, dates):
    """Compute PnL correlation matrix."""
    print("\n" + "=" * 70)
    print("  ANALYSIS 2: PnL CORRELATION MATRIX")
    print("=" * 70)

    common = sorted(set(vrp_pnl.keys()) & set(bf_pnl.keys()) & set(ratio_pnl.keys()))
    print(f"  Common dates: {len(common)}")

    v = [vrp_pnl[d] for d in common]
    b = [bf_pnl[d] for d in common]
    r = [ratio_pnl[d] for d in common]

    corr_vb = compute_correlation(v, b)
    corr_vr = compute_correlation(v, r)
    corr_br = compute_correlation(r, b)

    print(f"\n  Correlation Matrix:")
    print(f"  {'':>12} {'VRP':>8} {'BF_25d':>8} {'BF_ratio':>8}")
    print(f"  {'VRP':>12} {'1.000':>8} {corr_vb:>8.3f} {corr_vr:>8.3f}")
    print(f"  {'BF_25d':>12} {corr_vb:>8.3f} {'1.000':>8} {corr_br:>8.3f}")
    print(f"  {'BF_ratio':>12} {corr_vr:>8.3f} {corr_br:>8.3f} {'1.000':>8}")

    # Per-year correlations
    print(f"\n  Per-year BF_25d vs BF_ratio PnL correlation:")
    by_year = defaultdict(lambda: ([], []))
    for d in common:
        by_year[d[:4]][0].append(bf_pnl[d])
        by_year[d[:4]][1].append(ratio_pnl[d])

    for yr in sorted(by_year.keys()):
        c = compute_correlation(by_year[yr][0], by_year[yr][1])
        print(f"    {yr}: {c:.3f}")

    return {"vrp_bf": corr_vb, "vrp_ratio": corr_vr, "bf_ratio": corr_br}


# ═══════════════════════════════════════════════════════════════
# Analysis 3: Weighted Ensemble Optimization
# ═══════════════════════════════════════════════════════════════

def optimize_ensemble(vrp_pnl, bf_pnl, ratio_pnl, dates):
    """Sweep weights for VRP + BF_butterfly + BF_ratio."""
    print("\n" + "=" * 70)
    print("  ANALYSIS 3: WEIGHTED ENSEMBLE OPTIMIZATION")
    print("=" * 70)

    common = sorted(set(vrp_pnl.keys()) & set(bf_pnl.keys()) & set(ratio_pnl.keys()))

    # Baseline: R69 (10% VRP + 90% BF)
    base_rets = [0.10 * vrp_pnl[d] + 0.90 * bf_pnl[d] for d in common]
    base_stats = compute_stats(base_rets)
    print(f"\n  Baseline R69 (10/90 VRP/BF): Sharpe={base_stats['sharpe']:.4f}, AnnRet={base_stats['ann_ret']*100:.2f}%")

    # Sweep: w_vrp + w_bf + w_ratio = 1.0
    results = []
    for w_vrp_pct in range(0, 30, 5):  # 0-25%
        for w_ratio_pct in range(0, 55, 5):  # 0-50%
            w_bf_pct = 100 - w_vrp_pct - w_ratio_pct
            if w_bf_pct < 0:
                continue
            w_v = w_vrp_pct / 100.0
            w_b = w_bf_pct / 100.0
            w_r = w_ratio_pct / 100.0

            rets = [w_v * vrp_pnl[d] + w_b * bf_pnl[d] + w_r * ratio_pnl[d] for d in common]
            stats = compute_stats(rets)
            results.append({
                "w_vrp": w_v, "w_bf": w_b, "w_ratio": w_r,
                **stats
            })

    results.sort(key=lambda x: x["sharpe"], reverse=True)

    print(f"\n  Top 15 configs (of {len(results)}):")
    print(f"  {'VRP':>6} {'BF':>6} {'Ratio':>6} {'Sharpe':>8} {'AnnRet':>10} {'MaxDD':>8} {'Calmar':>8}")
    for r in results[:15]:
        print(f"  {r['w_vrp']*100:>5.0f}% {r['w_bf']*100:>5.0f}% {r['w_ratio']*100:>5.0f}% "
              f"{r['sharpe']:>8.3f} {r['ann_ret']*100:>9.2f}% {r['max_dd']*100:>7.2f}% {r['calmar']:>8.2f}")

    best = results[0]
    delta = best["sharpe"] - base_stats["sharpe"]
    print(f"\n  Best: VRP={best['w_vrp']:.0%} BF={best['w_bf']:.0%} Ratio={best['w_ratio']:.0%}")
    print(f"  Sharpe={best['sharpe']:.4f} (Δ={delta:+.4f} vs R69)")

    return results, base_stats


# ═══════════════════════════════════════════════════════════════
# Analysis 4: Walk-Forward OOS Validation
# ═══════════════════════════════════════════════════════════════

def walk_forward_validation(vrp_pnl, bf_pnl, ratio_pnl, dates):
    """Walk-forward validation of best ensemble config."""
    print("\n" + "=" * 70)
    print("  ANALYSIS 4: WALK-FORWARD OOS VALIDATION")
    print("=" * 70)

    common = sorted(set(vrp_pnl.keys()) & set(bf_pnl.keys()) & set(ratio_pnl.keys()))
    n = len(common)

    # Expanding window: train on [0, split], test on [split, split+180d]
    step = 180  # 6 months
    min_train = 365  # at least 1 year training

    wf_results = []
    static_results = []

    print(f"\n  {'Period':>24} {'Train':>6} {'Test':>5} {'Best w_ratio':>12} {'OOS Sharpe':>12} {'Static 10/90':>13}")
    print(f"  {'─'*24} {'─'*6} {'─'*5} {'─'*12} {'─'*12} {'─'*13}")

    i = min_train
    while i + step <= n:
        train_dates = common[:i]
        test_dates = common[i:i+step]

        # Optimize on train
        best_sharpe = -999
        best_config = (0.10, 0.90, 0.00)

        for w_vrp_pct in range(0, 25, 5):
            for w_ratio_pct in range(0, 55, 5):
                w_bf_pct = 100 - w_vrp_pct - w_ratio_pct
                if w_bf_pct < 0:
                    continue
                w_v, w_b, w_r = w_vrp_pct/100, w_bf_pct/100, w_ratio_pct/100

                rets = [w_v * vrp_pnl[d] + w_b * bf_pnl[d] + w_r * ratio_pnl[d] for d in train_dates
                        if d in vrp_pnl and d in bf_pnl and d in ratio_pnl]
                if len(rets) < 100:
                    continue
                stats = compute_stats(rets)
                if stats["sharpe"] > best_sharpe:
                    best_sharpe = stats["sharpe"]
                    best_config = (w_v, w_b, w_r)

        # Test OOS with best config
        w_v, w_b, w_r = best_config
        oos_rets = [w_v * vrp_pnl[d] + w_b * bf_pnl[d] + w_r * ratio_pnl[d]
                    for d in test_dates if d in vrp_pnl and d in bf_pnl and d in ratio_pnl]

        # Static 10/90 OOS
        static_rets = [0.10 * vrp_pnl[d] + 0.90 * bf_pnl[d]
                       for d in test_dates if d in vrp_pnl and d in bf_pnl]

        oos_stats = compute_stats(oos_rets)
        static_stats = compute_stats(static_rets)

        wf_results.append(oos_stats["sharpe"])
        static_results.append(static_stats["sharpe"])

        period = f"{test_dates[0]} to {test_dates[-1]}"
        print(f"  {period:>24} {len(train_dates):>6} {len(test_dates):>5} "
              f"{w_r:>11.0%} {oos_stats['sharpe']:>12.3f} {static_stats['sharpe']:>13.3f}")

        i += step

    if wf_results:
        avg_wf = sum(wf_results) / len(wf_results)
        avg_static = sum(static_results) / len(static_results)
        wins = sum(1 for w, s in zip(wf_results, static_results) if w > s)
        print(f"\n  WF avg (with ratio):  {avg_wf:.3f}")
        print(f"  WF avg (static 10/90): {avg_static:.3f}")
        print(f"  Delta: {avg_wf - avg_static:+.3f}")
        print(f"  Wins: {wins}/{len(wf_results)}")

    return wf_results, static_results


# ═══════════════════════════════════════════════════════════════
# Analysis 5: LOYO Validation
# ═══════════════════════════════════════════════════════════════

def loyo_validation(vrp_pnl, bf_pnl, ratio_pnl, dates):
    """Leave-One-Year-Out validation."""
    print("\n" + "=" * 70)
    print("  ANALYSIS 5: LOYO VALIDATION")
    print("=" * 70)

    common = sorted(set(vrp_pnl.keys()) & set(bf_pnl.keys()) & set(ratio_pnl.keys()))
    years = sorted(set(d[:4] for d in common))

    print(f"\n  {'Year':>6} {'Static 10/90':>14} {'Best Ratio':>12} {'w_ratio':>8} {'Delta':>8}")
    print(f"  {'─'*6} {'─'*14} {'─'*12} {'─'*8} {'─'*8}")

    for test_yr in years:
        train_dates = [d for d in common if d[:4] != test_yr]
        test_dates = [d for d in common if d[:4] == test_yr]
        if len(test_dates) < 30:
            continue

        # Optimize on train
        best_sharpe = -999
        best_config = (0.10, 0.90, 0.00)

        for w_vrp_pct in range(0, 25, 5):
            for w_ratio_pct in range(0, 55, 5):
                w_bf_pct = 100 - w_vrp_pct - w_ratio_pct
                if w_bf_pct < 0:
                    continue
                w_v, w_b, w_r = w_vrp_pct/100, w_bf_pct/100, w_ratio_pct/100

                rets = [w_v * vrp_pnl[d] + w_b * bf_pnl[d] + w_r * ratio_pnl[d]
                        for d in train_dates]
                stats = compute_stats(rets)
                if stats["sharpe"] > best_sharpe:
                    best_sharpe = stats["sharpe"]
                    best_config = (w_v, w_b, w_r)

        # Test
        w_v, w_b, w_r = best_config
        oos_rets = [w_v * vrp_pnl[d] + w_b * bf_pnl[d] + w_r * ratio_pnl[d] for d in test_dates]
        static_rets = [0.10 * vrp_pnl[d] + 0.90 * bf_pnl[d] for d in test_dates]

        oos_s = compute_stats(oos_rets)
        static_s = compute_stats(static_rets)
        delta = oos_s["sharpe"] - static_s["sharpe"]

        print(f"  {test_yr:>6} {static_s['sharpe']:>14.2f} {oos_s['sharpe']:>12.2f} {w_r:>7.0%} {delta:>+8.2f}")


# ═══════════════════════════════════════════════════════════════
# Analysis 6: Sensitivity to bf_ratio Parameters
# ═══════════════════════════════════════════════════════════════

def sensitivity_analysis(dvol_hist, surface_hist, dates, vrp_pnl, bf_pnl):
    """Test how sensitive the result is to bf_ratio parameter choice."""
    print("\n" + "=" * 70)
    print("  ANALYSIS 6: SENSITIVITY TO bf_ratio PARAMETERS")
    print("=" * 70)

    # Test different bf_ratio configs combined with 10% VRP + X% BF + Y% ratio
    configs = [
        ("Default lb=120 z_in=1.5 z_out=0.0", 120, 1.5, 0.0),
        ("Shorter lb=60", 60, 1.5, 0.0),
        ("Longer lb=180", 180, 1.5, 0.0),
        ("Tighter entry z_in=2.0", 120, 2.0, 0.0),
        ("Looser entry z_in=1.0", 120, 1.0, 0.0),
        ("With exit z_out=0.3", 120, 1.5, 0.3),
    ]

    common_dates = sorted(set(vrp_pnl.keys()) & set(bf_pnl.keys()))
    base_rets = [0.10 * vrp_pnl[d] + 0.90 * bf_pnl[d] for d in common_dates]
    base_stats = compute_stats(base_rets)

    print(f"\n  Baseline R69: Sharpe={base_stats['sharpe']:.4f}")
    print(f"\n  {'Config':>40} {'Ratio Sharpe':>14} {'Best Combo':>12} {'Delta':>8}")
    print(f"  {'─'*40} {'─'*14} {'─'*12} {'─'*8}")

    for name, lb, z_in, z_out in configs:
        ratio_pnl = compute_feature_mr_pnl(dvol_hist, surface_hist, dates,
                                           "bf_ratio", lb, z_in, z_out, 2.5)
        common = sorted(set(vrp_pnl.keys()) & set(bf_pnl.keys()) & set(ratio_pnl.keys()))
        if len(common) < 100:
            continue

        # Ratio standalone
        ratio_rets = [ratio_pnl[d] for d in common]
        ratio_stats = compute_stats(ratio_rets)

        # Best combo with ratio
        best_combo_sharpe = base_stats["sharpe"]
        for w_ratio_pct in range(5, 55, 5):
            w_bf_pct = 90 - w_ratio_pct
            if w_bf_pct < 0:
                break
            w_r = w_ratio_pct / 100
            w_b = w_bf_pct / 100
            rets = [0.10 * vrp_pnl[d] + w_b * bf_pnl[d] + w_r * ratio_pnl[d] for d in common]
            stats = compute_stats(rets)
            if stats["sharpe"] > best_combo_sharpe:
                best_combo_sharpe = stats["sharpe"]

        delta = best_combo_sharpe - base_stats["sharpe"]
        print(f"  {name:>40} {ratio_stats['sharpe']:>14.3f} {best_combo_sharpe:>12.3f} {delta:>+8.3f}")


# ═══════════════════════════════════════════════════════════════
# Analysis 7: Other Surface Features as Complements
# ═══════════════════════════════════════════════════════════════

def test_other_features(dvol_hist, surface_hist, dates, vrp_pnl, bf_pnl):
    """Test other surface features as complements to butterfly_25d."""
    print("\n" + "=" * 70)
    print("  ANALYSIS 7: OTHER SURFACE FEATURES AS COMPLEMENTS")
    print("=" * 70)

    features = [
        "bf_ratio",       # butterfly_25d / iv_atm
        "skew_25d",       # 25-delta skew
        "iv_atm",         # ATM IV level
    ]

    common_base = sorted(set(vrp_pnl.keys()) & set(bf_pnl.keys()))
    base_rets = [0.10 * vrp_pnl[d] + 0.90 * bf_pnl[d] for d in common_base]
    base_stats = compute_stats(base_rets)

    print(f"\n  Baseline R69: Sharpe={base_stats['sharpe']:.4f}")
    print(f"\n  {'Feature':>15} {'Standalone':>12} {'Corr w/BF':>10} {'Best Combo':>12} {'Delta':>8}")
    print(f"  {'─'*15} {'─'*12} {'─'*10} {'─'*12} {'─'*8}")

    for feat in features:
        feat_pnl = compute_feature_mr_pnl(dvol_hist, surface_hist, dates,
                                          feat, 120, 1.5, 0.0, 2.5)
        common = sorted(set(vrp_pnl.keys()) & set(bf_pnl.keys()) & set(feat_pnl.keys()))
        if len(common) < 100:
            print(f"  {feat:>15} {'SKIP (< 100 days)':>12}")
            continue

        # Standalone
        feat_rets = [feat_pnl[d] for d in common]
        feat_stats = compute_stats(feat_rets)

        # Correlation with BF_25d
        bf_rets = [bf_pnl[d] for d in common]
        corr = compute_correlation(feat_rets, bf_rets)

        # Best combo
        best_sharpe = base_stats["sharpe"]
        for w_feat_pct in range(5, 55, 5):
            w_bf_pct = 90 - w_feat_pct
            if w_bf_pct < 0:
                break
            w_f = w_feat_pct / 100
            w_b = w_bf_pct / 100
            rets = [0.10 * vrp_pnl[d] + w_b * bf_pnl[d] + w_f * feat_pnl[d] for d in common]
            stats = compute_stats(rets)
            if stats["sharpe"] > best_sharpe:
                best_sharpe = stats["sharpe"]

        delta = best_sharpe - base_stats["sharpe"]
        print(f"  {feat:>15} {feat_stats['sharpe']:>12.3f} {corr:>10.3f} {best_sharpe:>12.3f} {delta:>+8.3f}")


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════

def main():
    print("=" * 70)
    print("  R73: bf_ratio AS BUTTERFLY COMPLEMENT")
    print("=" * 70)
    print("  Loading data...")

    dvol_hist = load_dvol_history("BTC")
    price_hist = load_price_history("BTC")
    surface_hist = load_surface("BTC")

    dates = sorted(set(dvol_hist.keys()) & set(price_hist.keys()))
    print(f"  Dates: {dates[0]} to {dates[-1]} ({len(dates)} days)")

    # Compute base PnL series
    vrp_pnl = compute_vrp_pnl(dvol_hist, price_hist, dates)
    bf_pnl = compute_feature_mr_pnl(dvol_hist, surface_hist, dates,
                                     "butterfly_25d", 120, 1.5, 0.0, 2.5)
    ratio_pnl = compute_feature_mr_pnl(dvol_hist, surface_hist, dates,
                                        "bf_ratio", 120, 1.5, 0.0, 2.5)

    print(f"  VRP PnL: {len(vrp_pnl)} days")
    print(f"  BF PnL:  {len(bf_pnl)} days")
    print(f"  Ratio PnL: {len(ratio_pnl)} days")

    # Analysis 1: bf_ratio standalone optimization
    best_ratio = optimize_bf_ratio(dvol_hist, surface_hist, dates)

    # Analysis 2: PnL correlations
    corrs = analyze_pnl_correlations(vrp_pnl, bf_pnl, ratio_pnl, dates)

    # Analysis 3: Weighted ensemble
    ensemble_results, base_stats = optimize_ensemble(vrp_pnl, bf_pnl, ratio_pnl, dates)

    # Analysis 4: Walk-forward
    wf_results, wf_static = walk_forward_validation(vrp_pnl, bf_pnl, ratio_pnl, dates)

    # Analysis 5: LOYO
    loyo_validation(vrp_pnl, bf_pnl, ratio_pnl, dates)

    # Analysis 6: Sensitivity
    sensitivity_analysis(dvol_hist, surface_hist, dates, vrp_pnl, bf_pnl)

    # Analysis 7: Other features
    test_other_features(dvol_hist, surface_hist, dates, vrp_pnl, bf_pnl)

    # ─── Summary ────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  SUMMARY")
    print("=" * 70)

    best = ensemble_results[0]
    delta = best["sharpe"] - base_stats["sharpe"]

    print(f"\n  R69 Baseline (10/90 VRP/BF): Sharpe {base_stats['sharpe']:.4f}")
    print(f"  Best 3-way ensemble: VRP={best['w_vrp']:.0%} BF={best['w_bf']:.0%} Ratio={best['w_ratio']:.0%}")
    print(f"  Sharpe: {best['sharpe']:.4f} (Δ={delta:+.4f})")
    print(f"  BF-Ratio PnL correlation: {corrs['bf_ratio']:.3f}")

    if delta > 0.05:
        verdict = "UPGRADE — bf_ratio adds value as complement"
    elif delta > -0.05:
        verdict = "MARGINAL — bf_ratio neither helps nor hurts significantly"
    else:
        verdict = "NEGATIVE — bf_ratio does NOT improve portfolio"

    print(f"\n  VERDICT: {verdict}")

    # Save results
    results = {
        "research_id": "R73",
        "title": "bf_ratio as Butterfly Complement",
        "baseline_r69": {
            "sharpe": round(base_stats["sharpe"], 4),
            "ann_ret": round(base_stats["ann_ret"], 6),
        },
        "best_ensemble": {
            "w_vrp": best["w_vrp"], "w_bf": best["w_bf"], "w_ratio": best["w_ratio"],
            "sharpe": round(best["sharpe"], 4),
            "ann_ret": round(best["ann_ret"], 6),
        },
        "bf_ratio_standalone_sharpe": round(best_ratio["sharpe"], 4),
        "bf_ratio_bf_pnl_correlation": round(corrs["bf_ratio"], 3),
        "delta_sharpe": round(delta, 4),
        "verdict": verdict,
    }

    out_path = ROOT / "data" / "cache" / "deribit" / "real_surface" / "r73_bf_ratio_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved to {out_path}")


if __name__ == "__main__":
    main()
