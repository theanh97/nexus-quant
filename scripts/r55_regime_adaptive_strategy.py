#!/usr/bin/env python3
"""
R55: Regime-Adaptive Strategy
================================

R52 found: BTC IV structurally declining (96% → 42%), VRP edge shrinking.
R54 confirmed: Edge is real (OOS Sharpe ~2.0) but degrading in 2025.

Hypothesis: In LOW-IV regimes, VRP edge is thin → reduce VRP weight,
increase skew weight. In HIGH-IV regimes, VRP is fat → lever up VRP.

This study:
1. Define IV regimes (percentile-based using DVOL)
2. Test regime-conditional weights and leverage
3. Compare: static params vs regime-adaptive
4. Walk-forward validation of regime-adaptive

Regime definitions:
- LOW IV: DVOL < 30th percentile of trailing 180d
- MID IV: 30th-70th percentile
- HIGH IV: > 70th percentile
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


# ── Data Loading (same as R54) ───────────────────────────────────────

def load_dvol_daily(currency: str) -> Dict[str, float]:
    path = ROOT / "data" / "cache" / "deribit" / "dvol" / f"{currency}_DVOL_12h.csv"
    daily = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            daily[row["date"][:10]] = float(row["dvol_close"]) / 100.0
    return daily


def load_prices(currency: str) -> Dict[str, float]:
    import subprocess, time
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


def load_real_skew(currency: str) -> Dict[str, Optional[float]]:
    path = ROOT / "data" / "cache" / "deribit" / "real_surface" / f"{currency}_daily_surface.csv"
    data = {}
    if not path.exists():
        return data
    with open(path) as f:
        for row in csv.DictReader(f):
            val = row.get("skew_25d", "")
            if val and val != "None":
                data[row["date"]] = float(val)
    return data


# ── Regime Classification ────────────────────────────────────────────

def classify_regimes(dates: List[str], dvol: Dict[str, float],
                      lookback: int = 180,
                      low_pct: float = 30, high_pct: float = 70) -> Dict[str, str]:
    """
    Classify each date into LOW / MID / HIGH IV regime
    based on trailing `lookback`-day percentile of DVOL.
    """
    regimes = {}
    for i in range(lookback, len(dates)):
        d = dates[i]
        iv = dvol.get(d)
        if iv is None:
            continue

        # Trailing window
        window = []
        for j in range(i - lookback, i):
            v = dvol.get(dates[j])
            if v is not None:
                window.append(v)

        if len(window) < lookback // 2:
            continue

        # Percentile rank of current IV
        window_sorted = sorted(window)
        rank = sum(1 for w in window_sorted if w <= iv) / len(window_sorted) * 100

        if rank < low_pct:
            regimes[d] = "LOW"
        elif rank > high_pct:
            regimes[d] = "HIGH"
        else:
            regimes[d] = "MID"

    return regimes


# ── PnL Models ───────────────────────────────────────────────────────

def compute_vrp_pnl(dates: List[str], dvol: Dict[str, float],
                     prices: Dict[str, float], leverage: float = 1.5) -> Dict[str, float]:
    dt = 1.0 / 365.0
    pnl = {}
    for i in range(1, len(dates)):
        d = dates[i]
        dp = dates[i-1]
        iv = dvol.get(dp)
        p0 = prices.get(dp)
        p1 = prices.get(d)
        if not all([iv, p0, p1]) or p0 <= 0:
            continue
        log_ret = math.log(p1 / p0)
        rv_bar = abs(log_ret) * math.sqrt(365)
        pnl[d] = leverage * 0.5 * (iv**2 - rv_bar**2) * dt
    return pnl


def compute_skew_pnl(dates: List[str], skew: Dict[str, Optional[float]],
                      dvol: Dict[str, float], lookback: int = 90,
                      z_entry_short: float = 0.7, z_entry_long: float = 1.7) -> Dict[str, float]:
    dt = 1.0 / 365.0
    pnl = {}
    position = 0.0

    for i in range(lookback, len(dates)):
        d = dates[i]
        iv = dvol.get(d)
        s = skew.get(d)
        if s is None or iv is None:
            continue

        window_vals = []
        for j in range(i - lookback, i):
            v = skew.get(dates[j])
            if v is not None:
                window_vals.append(v)

        if len(window_vals) < lookback // 2:
            continue

        mean_s = sum(window_vals) / len(window_vals)
        std_s = math.sqrt(sum((v - mean_s)**2 for v in window_vals) / len(window_vals))
        if std_s < 1e-6:
            continue

        z = (s - mean_s) / std_s

        if z > z_entry_short:
            position = -1.0
        elif z < -z_entry_long:
            position = 1.0
        elif abs(z) < 0.3:
            position = 0.0

        s_prev = skew.get(dates[i-1])
        if s_prev is not None and position != 0:
            d_skew = s - s_prev
            pnl[d] = position * d_skew * iv * math.sqrt(dt) * 2.5
        else:
            pnl[d] = 0.0

    return pnl


def calc_sharpe(rets: List[float]) -> Tuple[float, float, float]:
    """Returns (sharpe, ann_ret, max_dd)."""
    if len(rets) < 20:
        return 0.0, 0.0, 0.0
    mean = sum(rets) / len(rets)
    var = sum((r - mean)**2 for r in rets) / len(rets)
    std = math.sqrt(var) if var > 0 else 1e-10
    sharpe = (mean * 365) / (std * math.sqrt(365))
    ann_ret = mean * 365
    cum = 0.0
    peak = 0.0
    max_dd = 0.0
    for r in rets:
        cum += r
        peak = max(peak, cum)
        max_dd = max(max_dd, peak - cum)
    return sharpe, ann_ret, max_dd


# ── Regime-Adaptive Ensemble ─────────────────────────────────────────

def compute_regime_adaptive_pnl(
    dates: List[str],
    dvol: Dict[str, float],
    prices: Dict[str, float],
    skew: Dict[str, Optional[float]],
    regimes: Dict[str, str],
    regime_params: Dict[str, dict],
    skew_lookback: int = 60,
    z_short: float = 0.7,
) -> Dict[str, Tuple[float, str]]:
    """
    Returns dict of date -> (pnl, regime).
    regime_params: {"LOW": {"w_vrp": 0.3, "lev": 1.0}, "MID": {...}, "HIGH": {...}}
    """
    dt = 1.0 / 365.0
    result = {}
    skew_position = 0.0

    for i in range(max(skew_lookback, 1), len(dates)):
        d = dates[i]
        dp = dates[i-1]
        regime = regimes.get(d)
        if regime is None:
            continue

        params = regime_params.get(regime, regime_params.get("MID", {}))
        w_vrp = params.get("w_vrp", 0.5)
        w_skew = 1.0 - w_vrp
        lev = params.get("lev", 1.5)

        # VRP PnL
        iv = dvol.get(dp)
        p0 = prices.get(dp)
        p1 = prices.get(d)
        vrp_pnl = 0.0
        if all([iv, p0, p1]) and p0 > 0:
            log_ret = math.log(p1 / p0)
            rv_bar = abs(log_ret) * math.sqrt(365)
            vrp_pnl = lev * 0.5 * (iv**2 - rv_bar**2) * dt

        # Skew PnL
        s = skew.get(d)
        iv_now = dvol.get(d)
        skew_pnl = 0.0
        if s is not None and iv_now is not None:
            window_vals = []
            for j in range(i - skew_lookback, i):
                if j >= 0:
                    v = skew.get(dates[j])
                    if v is not None:
                        window_vals.append(v)

            if len(window_vals) >= skew_lookback // 2:
                mean_s = sum(window_vals) / len(window_vals)
                std_s = math.sqrt(sum((v - mean_s)**2 for v in window_vals) / len(window_vals))
                if std_s > 1e-6:
                    z = (s - mean_s) / std_s
                    if z > z_short:
                        skew_position = -1.0
                    elif z < -1.7:
                        skew_position = 1.0
                    elif abs(z) < 0.3:
                        skew_position = 0.0

            s_prev = skew.get(dates[i-1])
            if s_prev is not None and skew_position != 0:
                d_skew = s - s_prev
                skew_pnl = skew_position * d_skew * iv_now * math.sqrt(dt) * 2.5

        total_pnl = w_vrp * vrp_pnl + w_skew * skew_pnl
        result[d] = (total_pnl, regime)

    return result


# ── Main ─────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("R55: REGIME-ADAPTIVE STRATEGY")
    print("=" * 70)
    print()

    # Load data
    print("Loading data...")
    dvol = load_dvol_daily("BTC")
    prices = load_prices("BTC")
    skew = load_real_skew("BTC")
    print(f"  DVOL: {len(dvol)} days, Prices: {len(prices)} days, Skew: {len(skew)} days")

    all_dates = sorted(set(dvol.keys()) & set(prices.keys()) & set(skew.keys()))
    print(f"  Common dates: {len(all_dates)} ({all_dates[0]} to {all_dates[-1]})")

    # ── 1. Regime Classification ─────────────────────────────────────
    print(f"\n{'='*70}")
    print("  STEP 1: REGIME CLASSIFICATION")
    print(f"{'='*70}")

    regimes = classify_regimes(all_dates, dvol, lookback=180, low_pct=30, high_pct=70)
    regime_counts = defaultdict(int)
    regime_by_year = defaultdict(lambda: defaultdict(int))
    for d, r in regimes.items():
        regime_counts[r] += 1
        regime_by_year[d[:4]][r] += 1

    print(f"\n  Regime distribution:")
    total = sum(regime_counts.values())
    for r in ["LOW", "MID", "HIGH"]:
        print(f"    {r}: {regime_counts[r]} days ({regime_counts[r]/total:.0%})")

    print(f"\n  By year:")
    print(f"  {'Year':<6s}  {'LOW':>5s}  {'MID':>5s}  {'HIGH':>5s}")
    for yr in sorted(regime_by_year.keys()):
        rc = regime_by_year[yr]
        print(f"  {yr:<6s}  {rc['LOW']:5d}  {rc['MID']:5d}  {rc['HIGH']:5d}")

    # ── 2. Per-Regime Performance (static params) ────────────────────
    print(f"\n{'='*70}")
    print("  STEP 2: PER-REGIME PERFORMANCE (R54 best static params)")
    print(f"{'='*70}")
    print("  Using w=0.7, lev=2.0, lb=60, z=1.0 (R54's best)")
    print()

    vrp_all = compute_vrp_pnl(all_dates, dvol, prices, 2.0)
    skw_all = compute_skew_pnl(all_dates, skew, dvol, 60, 1.0)

    for r_name in ["LOW", "MID", "HIGH"]:
        r_dates = sorted(d for d in regimes if regimes[d] == r_name)
        vrp_rets = [vrp_all[d] for d in r_dates if d in vrp_all]
        skw_rets = [skw_all[d] for d in r_dates if d in skw_all]
        ens_rets = [0.7 * vrp_all.get(d, 0) + 0.3 * skw_all.get(d, 0)
                    for d in r_dates if d in vrp_all]

        sh_v, ret_v, _ = calc_sharpe(vrp_rets)
        sh_s, ret_s, _ = calc_sharpe(skw_rets)
        sh_e, ret_e, dd_e = calc_sharpe(ens_rets)

        print(f"  {r_name} regime ({len(r_dates)} days):")
        print(f"    VRP:      Sharpe={sh_v:6.2f}  Ret={ret_v:+.2%}")
        print(f"    Skew:     Sharpe={sh_s:6.2f}  Ret={ret_s:+.2%}")
        print(f"    Ensemble: Sharpe={sh_e:6.2f}  Ret={ret_e:+.2%}  DD={dd_e:.2%}")
        print()

    # ── 3. Optimize Per-Regime Params ────────────────────────────────
    print(f"\n{'='*70}")
    print("  STEP 3: OPTIMIZE PER-REGIME PARAMETERS")
    print(f"{'='*70}")
    print()

    best_regime_params = {}

    for r_name in ["LOW", "MID", "HIGH"]:
        r_dates = sorted(d for d in regimes if regimes[d] == r_name)
        if len(r_dates) < 50:
            continue

        best_sh = -999
        best_p = {}

        for w_vrp in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
            for lev in [0.5, 1.0, 1.5, 2.0, 2.5]:
                vrp = compute_vrp_pnl(all_dates, dvol, prices, lev)
                # Use multiple skew params
                for lb in [30, 60, 90, 120]:
                    for zs in [0.5, 0.7, 1.0]:
                        skw = compute_skew_pnl(all_dates, skew, dvol, lb, zs)
                        w_s = 1.0 - w_vrp
                        ens_rets = [w_vrp * vrp.get(d, 0) + w_s * skw.get(d, 0)
                                    for d in r_dates if d in vrp or d in skw]
                        sh, ret, dd = calc_sharpe(ens_rets)
                        if sh > best_sh and len(ens_rets) >= 50:
                            best_sh = sh
                            best_p = {"w_vrp": w_vrp, "lev": lev,
                                      "lb": lb, "z_short": zs,
                                      "sharpe": sh, "ann_ret": ret}

        best_regime_params[r_name] = best_p
        print(f"  {r_name} optimal: w_vrp={best_p.get('w_vrp','?'):.1f}  "
              f"lev={best_p.get('lev','?')}  lb={best_p.get('lb','?')}  "
              f"z={best_p.get('z_short','?')}  Sharpe={best_p.get('sharpe',0):.2f}  "
              f"Ret={best_p.get('ann_ret',0):+.2%}")

    # ── 4. Regime-Adaptive Backtest ──────────────────────────────────
    print(f"\n{'='*70}")
    print("  STEP 4: REGIME-ADAPTIVE BACKTEST")
    print(f"{'='*70}")
    print()

    # Build regime param dict for adaptive strategy
    adaptive_params = {}
    for r_name, p in best_regime_params.items():
        adaptive_params[r_name] = {
            "w_vrp": p.get("w_vrp", 0.5),
            "lev": p.get("lev", 1.5),
        }

    # Use most common skew lb/z across regimes
    all_lbs = [p.get("lb", 60) for p in best_regime_params.values()]
    all_zs = [p.get("z_short", 0.7) for p in best_regime_params.values()]
    # Mode
    skew_lb = max(set(all_lbs), key=all_lbs.count)
    skew_z = max(set(all_zs), key=all_zs.count)

    print(f"  Adaptive params:")
    for r_name in ["LOW", "MID", "HIGH"]:
        p = adaptive_params.get(r_name, {})
        print(f"    {r_name}: w_vrp={p.get('w_vrp',0.5):.1f}  lev={p.get('lev',1.5):.1f}")
    print(f"  Shared skew: lb={skew_lb}  z={skew_z}")

    # Run adaptive strategy
    adaptive_pnl = compute_regime_adaptive_pnl(
        all_dates, dvol, prices, skew, regimes,
        adaptive_params, skew_lookback=skew_lb, z_short=skew_z
    )

    # Compare with static strategies
    print(f"\n  Comparison:")
    # Static (R54 best: w=0.7, lev=2.0)
    static_vrp = compute_vrp_pnl(all_dates, dvol, prices, 2.0)
    static_skw = compute_skew_pnl(all_dates, skew, dvol, 60, 1.0)
    static_dates = sorted(set(static_vrp.keys()) & set(static_skw.keys()) & set(regimes.keys()))
    static_rets = [0.7 * static_vrp[d] + 0.3 * static_skw[d] for d in static_dates]
    sh_static, ret_static, dd_static = calc_sharpe(static_rets)

    # Adaptive
    adaptive_dates = sorted(adaptive_pnl.keys())
    adaptive_rets = [adaptive_pnl[d][0] for d in adaptive_dates]
    sh_adaptive, ret_adaptive, dd_adaptive = calc_sharpe(adaptive_rets)

    # R53 original (w=0.5, lev=1.5)
    r53_vrp = compute_vrp_pnl(all_dates, dvol, prices, 1.5)
    r53_skw = compute_skew_pnl(all_dates, skew, dvol, 90, 0.7)
    r53_dates = sorted(set(r53_vrp.keys()) & set(r53_skw.keys()) & set(regimes.keys()))
    r53_rets = [0.5 * r53_vrp[d] + 0.5 * r53_skw[d] for d in r53_dates]
    sh_r53, ret_r53, dd_r53 = calc_sharpe(r53_rets)

    print(f"  {'Strategy':<20s}  {'Sharpe':>7s}  {'AnnRet':>8s}  {'MaxDD':>7s}  {'N':>5s}")
    print(f"  {'R53 Static':<20s}  {sh_r53:7.2f}  {ret_r53:+8.2%}  {dd_r53:7.2%}  {len(r53_rets):5d}")
    print(f"  {'R54 Best Static':<20s}  {sh_static:7.2f}  {ret_static:+8.2%}  {dd_static:7.2%}  {len(static_rets):5d}")
    print(f"  {'R55 Adaptive':<20s}  {sh_adaptive:7.2f}  {ret_adaptive:+8.2%}  {dd_adaptive:7.2%}  {len(adaptive_rets):5d}")

    # ── 5. Yearly breakdown of adaptive strategy ─────────────────────
    print(f"\n{'='*70}")
    print("  STEP 5: YEARLY BREAKDOWN")
    print(f"{'='*70}")
    print()

    by_year = defaultdict(list)
    by_year_regime = defaultdict(lambda: defaultdict(int))
    for d in adaptive_dates:
        pnl_val, regime = adaptive_pnl[d]
        by_year[d[:4]].append(pnl_val)
        by_year_regime[d[:4]][regime] += 1

    print(f"  {'Year':<6s}  {'Sharpe':>7s}  {'AnnRet':>8s}  {'N':>5s}  {'LOW':>5s}  {'MID':>5s}  {'HIGH':>5s}")
    for yr in sorted(by_year.keys()):
        rets = by_year[yr]
        sh, ret, _ = calc_sharpe(rets)
        rc = by_year_regime[yr]
        print(f"  {yr:<6s}  {sh:7.2f}  {ret:+8.2%}  {len(rets):5d}  "
              f"{rc['LOW']:5d}  {rc['MID']:5d}  {rc['HIGH']:5d}")

    # ── 6. Walk-Forward of Adaptive Strategy ─────────────────────────
    print(f"\n{'='*70}")
    print("  STEP 6: WALK-FORWARD VALIDATION OF ADAPTIVE STRATEGY")
    print(f"{'='*70}")
    print("  Expanding window: optimize regime params on train, test on next 6mo")
    print()

    # Build 6-month periods
    periods = []
    start_year = int(all_dates[0][:4])
    end_year = int(all_dates[-1][:4])
    for yr in range(start_year, end_year + 1):
        for half in [(1, 6), (7, 12)]:
            period_start = f"{yr}-{half[0]:02d}-01"
            period_end = f"{yr}-{half[1]:02d}-30"
            period_dates = [d for d in all_dates if period_start <= d <= period_end]
            if period_dates:
                label = f"{yr}H{1 if half[0]==1 else 2}"
                periods.append((label, period_dates))

    wf_results = []
    for i in range(3, len(periods)):  # Need 3+ periods for training
        # Train: all periods up to i
        train_dates = []
        for j in range(i):
            train_dates.extend(periods[j][1])
        train_dates = sorted(set(train_dates))

        test_label, test_dates = periods[i]
        if len(test_dates) < 20 or len(train_dates) < 300:
            continue

        # Classify regimes on train
        train_regimes = classify_regimes(train_dates, dvol, 180)

        # Optimize per-regime params on train
        opt_params = {}
        for r_name in ["LOW", "MID", "HIGH"]:
            r_dates_train = [d for d in train_dates if train_regimes.get(d) == r_name]
            if len(r_dates_train) < 30:
                opt_params[r_name] = {"w_vrp": 0.5, "lev": 1.5}
                continue

            best_sh = -999
            best_p = {"w_vrp": 0.5, "lev": 1.5}
            for w_vrp in [0.3, 0.5, 0.7]:
                for lev in [1.0, 1.5, 2.0]:
                    vrp = compute_vrp_pnl(train_dates, dvol, prices, lev)
                    skw = compute_skew_pnl(train_dates, skew, dvol, 60, 1.0)
                    w_s = 1.0 - w_vrp
                    rets = [w_vrp * vrp.get(d, 0) + w_s * skw.get(d, 0)
                            for d in r_dates_train if d in vrp]
                    sh, _, _ = calc_sharpe(rets)
                    if sh > best_sh and len(rets) >= 30:
                        best_sh = sh
                        best_p = {"w_vrp": w_vrp, "lev": lev}

            opt_params[r_name] = best_p

        # Test with optimized adaptive params
        test_regimes = classify_regimes(all_dates, dvol, 180)
        adaptive_test = compute_regime_adaptive_pnl(
            all_dates, dvol, prices, skew, test_regimes,
            opt_params, skew_lookback=60, z_short=1.0
        )

        test_rets = [adaptive_test[d][0] for d in test_dates
                     if d in adaptive_test]
        if len(test_rets) < 20:
            continue

        sh_test, ret_test, dd_test = calc_sharpe(test_rets)
        wf_results.append({
            "period": test_label,
            "test_sharpe": sh_test,
            "test_ann_ret": ret_test,
            "params": {r: opt_params[r] for r in opt_params}
        })

        lo = opt_params.get("LOW", {})
        mi = opt_params.get("MID", {})
        hi = opt_params.get("HIGH", {})
        print(f"  {test_label}: Sharpe={sh_test:6.2f}  Ret={ret_test:+.2%}  "
              f"[L:w={lo.get('w_vrp','?')}/l={lo.get('lev','?')} "
              f"M:w={mi.get('w_vrp','?')}/l={mi.get('lev','?')} "
              f"H:w={hi.get('w_vrp','?')}/l={hi.get('lev','?')}]")

    if wf_results:
        avg_sh = sum(r["test_sharpe"] for r in wf_results) / len(wf_results)
        pos = sum(1 for r in wf_results if r["test_sharpe"] > 0)
        print(f"\n  WF Avg Sharpe={avg_sh:.2f}  Positive={pos}/{len(wf_results)} "
              f"({pos/len(wf_results):.0%})")

    # ── Summary ──────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("R55 REGIME-ADAPTIVE STRATEGY SUMMARY")
    print(f"{'='*70}")
    print()
    print(f"  Static (R53):    Sharpe={sh_r53:.2f}  Ret={ret_r53:+.2%}")
    print(f"  Static (R54):    Sharpe={sh_static:.2f}  Ret={ret_static:+.2%}")
    print(f"  Adaptive (R55):  Sharpe={sh_adaptive:.2f}  Ret={ret_adaptive:+.2%}")
    if wf_results:
        avg_sh_wf = sum(r["test_sharpe"] for r in wf_results) / len(wf_results)
        print(f"  Adaptive WF OOS: Sharpe={avg_sh_wf:.2f}")
    print()
    print("  Per-regime optimal params:")
    for r_name in ["LOW", "MID", "HIGH"]:
        p = best_regime_params.get(r_name, {})
        print(f"    {r_name}: w_vrp={p.get('w_vrp','?')}  lev={p.get('lev','?')}  "
              f"lb={p.get('lb','?')}  z={p.get('z_short','?')}")

    # Save results
    results = {
        "research_id": "R55",
        "title": "Regime-Adaptive Strategy",
        "regime_params": best_regime_params,
        "adaptive_params": adaptive_params,
        "static_r53": {"sharpe": sh_r53, "ann_ret": ret_r53, "max_dd": dd_r53},
        "static_r54": {"sharpe": sh_static, "ann_ret": ret_static, "max_dd": dd_static},
        "adaptive": {"sharpe": sh_adaptive, "ann_ret": ret_adaptive, "max_dd": dd_adaptive},
        "walk_forward_results": wf_results if wf_results else [],
    }
    outpath = ROOT / "data" / "cache" / "deribit" / "real_surface" / "r55_regime_adaptive_results.json"
    with open(outpath, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved to {outpath}")


if __name__ == "__main__":
    main()
