#!/usr/bin/env python3
"""
R57: Walk-Forward Validation of Multi-Factor Model
====================================================

R56 found: VRP+Butterfly multi-factor Sharpe 3.25, but in-sample.
Butterfly MR (Sharpe 1.80) has only 0.055 correlation with VRP.

This study:
1. Walk-forward: optimize factor weights on expanding window, test next 6mo
2. Compare: VRP-only vs VRP+Butterfly vs full 4-factor
3. Calculate OOS Sharpe degradation for each model
4. Determine if butterfly alpha survives OOS

Key question: Is butterfly alpha REAL or artifact of lookback optimization?
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


# ── Data Loading ─────────────────────────────────────────────────────

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


# ── Strategy Models ──────────────────────────────────────────────────

def rolling_zscore(values: Dict[str, float], dates: List[str],
                    lookback: int) -> Dict[str, float]:
    result = {}
    for i in range(lookback, len(dates)):
        d = dates[i]
        val = values.get(d)
        if val is None:
            continue
        window = []
        for j in range(i - lookback, i):
            v = values.get(dates[j])
            if v is not None:
                window.append(v)
        if len(window) < lookback // 2:
            continue
        mean = sum(window) / len(window)
        std = math.sqrt(sum((v - mean)**2 for v in window) / len(window))
        if std > 1e-8:
            result[d] = (val - mean) / std
    return result


def vrp_pnl(dates: List[str], dvol: Dict[str, float],
            prices: Dict[str, float], lev: float = 2.0) -> Dict[str, float]:
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
        pnl[d] = lev * 0.5 * (iv**2 - rv_bar**2) * dt
    return pnl


def mr_pnl(dates: List[str], feature: Dict[str, float],
           dvol: Dict[str, float], lookback: int = 120,
           z_entry: float = 1.5) -> Dict[str, float]:
    dt = 1.0 / 365.0
    pnl = {}
    position = 0.0
    zscore = rolling_zscore(feature, dates, lookback)

    for i in range(1, len(dates)):
        d = dates[i]
        dp = dates[i-1]
        z = zscore.get(d)
        iv = dvol.get(d)
        f_now = feature.get(d)
        f_prev = feature.get(dp)

        if z is not None:
            if z > z_entry:
                position = -1.0
            elif z < -z_entry:
                position = 1.0
            elif abs(z) < 0.3:
                position = 0.0

        if f_now is not None and f_prev is not None and iv is not None and position != 0:
            d_feat = f_now - f_prev
            pnl[d] = position * d_feat * iv * math.sqrt(dt) * 2.5
        else:
            if d in zscore:
                pnl[d] = 0.0

    return pnl


def calc_sharpe(rets: List[float]) -> Tuple[float, float, float]:
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


# ── Main ─────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("R57: WALK-FORWARD VALIDATION OF MULTI-FACTOR MODEL")
    print("=" * 70)
    print()

    # Load data
    print("Loading data...")
    dvol = load_dvol_daily("BTC")
    prices = load_prices("BTC")
    surface = load_surface("BTC")

    all_dates = sorted(set(dvol.keys()) & set(prices.keys()) & set(surface.keys()))
    print(f"  Common dates: {len(all_dates)} ({all_dates[0]} to {all_dates[-1]})")

    # Pre-compute all factor PnL streams
    print("  Computing factor PnL streams...")
    skew_feat = {d: s["skew_25d"] for d, s in surface.items() if "skew_25d" in s}
    bf_feat = {d: s["butterfly_25d"] for d, s in surface.items() if "butterfly_25d" in s}
    ts_feat = {d: s["term_spread"] for d, s in surface.items() if "term_spread" in s}

    pnl_vrp = vrp_pnl(all_dates, dvol, prices, 2.0)
    pnl_skew = mr_pnl(all_dates, skew_feat, dvol, 120, 1.5)
    pnl_bf = mr_pnl(all_dates, bf_feat, dvol, 120, 1.5)
    pnl_ts = mr_pnl(all_dates, ts_feat, dvol, 120, 1.5)

    factors = {
        "VRP": pnl_vrp,
        "Skew": pnl_skew,
        "Butterfly": pnl_bf,
        "TermSpread": pnl_ts,
    }

    # Common dates across all factors
    common_all = sorted(set.intersection(*[set(f.keys()) for f in factors.values()]))
    print(f"  Common dates (all factors): {len(common_all)}")

    # ── Build 6-month periods ────────────────────────────────────────
    periods = []
    start_year = int(common_all[0][:4])
    end_year = int(common_all[-1][:4])
    for yr in range(start_year, end_year + 1):
        for half in [(1, 6), (7, 12)]:
            period_start = f"{yr}-{half[0]:02d}-01"
            period_end = f"{yr}-{half[1]:02d}-30"
            period_dates = [d for d in common_all if period_start <= d <= period_end]
            if period_dates:
                label = f"{yr}H{1 if half[0]==1 else 2}"
                periods.append((label, period_dates))

    # ── Model configs to test ────────────────────────────────────────
    models = {
        "VRP-only": ["VRP"],
        "VRP+Skew": ["VRP", "Skew"],
        "VRP+Butterfly": ["VRP", "Butterfly"],
        "VRP+BF+Skew": ["VRP", "Butterfly", "Skew"],
        "All 4-factor": ["VRP", "Skew", "Butterfly", "TermSpread"],
    }

    # ── Walk-Forward for each model ──────────────────────────────────

    def optimize_weights(train_dates, factor_names):
        """Grid search for best weights on training data."""
        n = len(factor_names)
        best_sh = -999
        best_w = [1.0/n] * n

        if n == 1:
            return [1.0]

        # Generate weight grid (step=0.1, sum=1)
        def gen_weights(n, steps=10):
            if n == 1:
                yield [1.0]
                return
            for w in range(1, steps):
                for sub in gen_weights(n-1, steps - w):
                    yield [w/10] + sub

        for weights in gen_weights(n):
            if abs(sum(weights) - 1.0) > 0.01:
                continue

            rets = []
            for d in train_dates:
                r = sum(w * factors[fn].get(d, 0) for w, fn in zip(weights, factor_names))
                rets.append(r)

            if len(rets) < 100:
                continue

            sh, _, _ = calc_sharpe(rets)
            if sh > best_sh:
                best_sh = sh
                best_w = weights

        return best_w

    for model_name, factor_names in models.items():
        print(f"\n{'='*70}")
        print(f"  MODEL: {model_name} ({', '.join(factor_names)})")
        print(f"{'='*70}")
        print()

        wf_results = []
        for i in range(2, len(periods)):
            # Train: all periods up to i
            train_dates = []
            for j in range(i):
                train_dates.extend(periods[j][1])
            train_dates = sorted(set(train_dates))

            test_label, test_dates = periods[i]
            if len(test_dates) < 20 or len(train_dates) < 200:
                continue

            # Optimize weights on train
            weights = optimize_weights(train_dates, factor_names)

            # Train Sharpe
            train_rets = [sum(w * factors[fn].get(d, 0) for w, fn in zip(weights, factor_names))
                         for d in train_dates]
            sh_train, _, _ = calc_sharpe(train_rets)

            # Test Sharpe
            test_rets = [sum(w * factors[fn].get(d, 0) for w, fn in zip(weights, factor_names))
                        for d in test_dates]
            sh_test, ret_test, dd_test = calc_sharpe(test_rets)

            wf_results.append({
                "period": test_label,
                "train_sharpe": sh_train,
                "test_sharpe": sh_test,
                "test_ann_ret": ret_test,
                "weights": dict(zip(factor_names, weights)),
                "n_test": len(test_rets)
            })

            w_str = " ".join(f"{fn[0]}={w:.1f}" for fn, w in zip(factor_names, weights))
            print(f"  {test_label}: Train={sh_train:6.2f}  Test={sh_test:6.2f}  "
                  f"Ret={ret_test:+.2%}  [{w_str}]")

        if wf_results:
            avg_train = sum(r["train_sharpe"] for r in wf_results) / len(wf_results)
            avg_test = sum(r["test_sharpe"] for r in wf_results) / len(wf_results)
            pos = sum(1 for r in wf_results if r["test_sharpe"] > 0)
            print(f"\n  {model_name} WF: Train={avg_train:.2f}  Test={avg_test:.2f}  "
                  f"Δ={avg_train-avg_test:+.2f}  Positive={pos}/{len(wf_results)}")

    # ── LOYO for each model ──────────────────────────────────────────
    print(f"\n{'='*70}")
    print("  LEAVE-ONE-YEAR-OUT COMPARISON")
    print(f"{'='*70}")
    print()

    years = sorted(set(d[:4] for d in common_all))
    loyo_summary = {}

    for model_name, factor_names in models.items():
        loyo_tests = []
        for test_year in years:
            train_dates = [d for d in common_all if d[:4] != test_year]
            test_dates = [d for d in common_all if d[:4] == test_year]
            if len(test_dates) < 30 or len(train_dates) < 200:
                continue

            weights = optimize_weights(train_dates, factor_names)
            test_rets = [sum(w * factors[fn].get(d, 0)
                           for w, fn in zip(weights, factor_names))
                        for d in test_dates]
            sh, ret, _ = calc_sharpe(test_rets)
            loyo_tests.append(sh)

        avg = sum(loyo_tests) / len(loyo_tests) if loyo_tests else 0
        loyo_summary[model_name] = avg
        print(f"  {model_name:<20s}  LOYO avg Sharpe={avg:.2f}  "
              f"(yearly: {' '.join(f'{s:.1f}' for s in loyo_tests)})")

    # ── Fixed weights comparison (no optimization) ───────────────────
    print(f"\n{'='*70}")
    print("  FIXED-WEIGHT COMPARISON (no optimization needed)")
    print(f"{'='*70}")
    print("  Pre-set allocation strategies:")
    print()

    fixed_configs = [
        ("100% VRP",             {"VRP": 1.0}),
        ("70% VRP + 30% BF",    {"VRP": 0.7, "Butterfly": 0.3}),
        ("60% VRP + 40% BF",    {"VRP": 0.6, "Butterfly": 0.4}),
        ("50% VRP + 50% BF",    {"VRP": 0.5, "Butterfly": 0.5}),
        ("70% VRP + 20% BF + 10% Skew", {"VRP": 0.7, "Butterfly": 0.2, "Skew": 0.1}),
        ("50% VRP + 30% BF + 10% Skew + 10% TS",
         {"VRP": 0.5, "Butterfly": 0.3, "Skew": 0.1, "TermSpread": 0.1}),
    ]

    print(f"  {'Config':<45s}  {'Sharpe':>7s}  {'AnnRet':>8s}  {'MaxDD':>7s}")
    for name, weights in fixed_configs:
        rets = []
        for d in common_all:
            r = sum(w * factors[fn].get(d, 0) for fn, w in weights.items())
            rets.append(r)
        sh, ret, dd = calc_sharpe(rets)
        print(f"  {name:<45s}  {sh:7.2f}  {ret:+8.2%}  {dd:7.2%}")

    # ── Yearly comparison of top configs ─────────────────────────────
    print(f"\n{'='*70}")
    print("  YEARLY COMPARISON: VRP-only vs VRP+BF(70/30)")
    print(f"{'='*70}")
    print()

    top_configs = [
        ("VRP-only", {"VRP": 1.0}),
        ("VRP+BF(70/30)", {"VRP": 0.7, "Butterfly": 0.3}),
        ("VRP+BF(50/50)", {"VRP": 0.5, "Butterfly": 0.5}),
    ]

    print(f"  {'Year':<6s}", end="")
    for name, _ in top_configs:
        print(f"  {name:>16s}", end="")
    print()

    for yr in sorted(set(d[:4] for d in common_all)):
        yr_dates = [d for d in common_all if d[:4] == yr]
        if len(yr_dates) < 20:
            continue
        print(f"  {yr:<6s}", end="")
        for name, weights in top_configs:
            rets = [sum(w * factors[fn].get(d, 0) for fn, w in weights.items())
                    for d in yr_dates]
            sh, _, _ = calc_sharpe(rets)
            print(f"  {sh:16.2f}", end="")
        print()

    # ── Summary ──────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("R57 WALK-FORWARD VALIDATION SUMMARY")
    print(f"{'='*70}")
    print()
    print("  LOYO avg Sharpe by model:")
    for name, sh in sorted(loyo_summary.items(), key=lambda x: -x[1]):
        print(f"    {name:<25s}  {sh:.2f}")

    # Save results
    results = {
        "research_id": "R57",
        "title": "Walk-Forward Validation of Multi-Factor Model",
        "loyo_summary": loyo_summary,
    }
    outpath = ROOT / "data" / "cache" / "deribit" / "real_surface" / "r57_multifactor_wf_results.json"
    with open(outpath, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved to {outpath}")


if __name__ == "__main__":
    main()
