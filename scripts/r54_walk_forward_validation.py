#!/usr/bin/env python3
"""
R54: Walk-Forward Out-of-Sample Validation
=============================================

CRITICAL: R53 optimized on full dataset (Sharpe 3.352).
Is this overfit? Walk-forward test:

  1. Anchored walk-forward: train on [start, T], test on [T, T+6mo]
     Slide T forward by 6 months each time
  2. Expanding-window: train on [start, T], re-optimize, test next period
  3. Leave-one-year-out: train on all years except Y, test on Y
  4. Compare: in-sample vs out-of-sample Sharpe degradation

If OOS Sharpe >> IS Sharpe, parameters are robust.
If OOS Sharpe << IS Sharpe, we're overfitting.
"""
import csv
import json
import math
import sys
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
    """Load prices from Deribit API via curl."""
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


# ── PnL Models ───────────────────────────────────────────────────────

def compute_vrp_pnl(dates: List[str], dvol: Dict[str, float],
                     prices: Dict[str, float], leverage: float = 1.5) -> Dict[str, float]:
    """VRP PnL: 0.5 * leverage * (IV^2 - RV_bar^2) * dt"""
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
    """Skew MR PnL based on z-score signals."""
    dt = 1.0 / 365.0
    pnl = {}
    position = 0.0  # -1, 0, +1

    for i in range(lookback, len(dates)):
        d = dates[i]
        iv = dvol.get(d)
        s = skew.get(d)
        if s is None or iv is None:
            continue

        # Rolling mean/std of skew
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

        # Signal
        if z > z_entry_short:
            position = -1.0
        elif z < -z_entry_long:
            position = 1.0
        elif abs(z) < 0.3:
            position = 0.0

        # PnL from skew change
        s_prev = skew.get(dates[i-1])
        if s_prev is not None and position != 0:
            d_skew = s - s_prev
            pnl[d] = position * d_skew * iv * math.sqrt(dt) * 2.5
        else:
            pnl[d] = 0.0

    return pnl


def compute_ensemble_pnl(vrp_pnl: Dict[str, float], skew_pnl: Dict[str, float],
                          w_vrp: float = 0.5) -> Dict[str, float]:
    """Weighted ensemble of VRP and Skew PnL."""
    w_skew = 1.0 - w_vrp
    common_dates = sorted(set(vrp_pnl.keys()) & set(skew_pnl.keys()))
    return {d: w_vrp * vrp_pnl[d] + w_skew * skew_pnl[d] for d in common_dates}


def calc_sharpe(pnl: Dict[str, float], dates: Optional[List[str]] = None) -> Tuple[float, float, float, int]:
    """Returns (sharpe, ann_ret, max_dd, n_days)."""
    if dates:
        rets = [pnl[d] for d in dates if d in pnl]
    else:
        rets = [pnl[d] for d in sorted(pnl.keys())]

    if len(rets) < 20:
        return 0.0, 0.0, 0.0, len(rets)

    mean = sum(rets) / len(rets)
    var = sum((r - mean)**2 for r in rets) / len(rets)
    std = math.sqrt(var) if var > 0 else 1e-10
    sharpe = (mean * 365) / (std * math.sqrt(365))
    ann_ret = mean * 365

    # Max drawdown
    cum = 0.0
    peak = 0.0
    max_dd = 0.0
    for r in rets:
        cum += r
        peak = max(peak, cum)
        dd = peak - cum
        max_dd = max(max_dd, dd)

    return sharpe, ann_ret, max_dd, len(rets)


# ── Optimization (grid search on train set) ──────────────────────────

def optimize_params(dates_train: List[str], dvol: Dict, prices: Dict,
                    skew: Dict) -> dict:
    """Grid search for best params on training data."""
    best_sharpe = -999
    best_params = {}

    for w_vrp in [0.3, 0.4, 0.5, 0.6, 0.7]:
        for lev in [1.0, 1.5, 2.0]:
            for lb in [60, 90, 120]:
                for z_s in [0.5, 0.7, 1.0]:
                    vrp = compute_vrp_pnl(dates_train, dvol, prices, lev)
                    skw = compute_skew_pnl(dates_train, skew, dvol, lb, z_s)
                    ens = compute_ensemble_pnl(vrp, skw, w_vrp)
                    sh, _, _, n = calc_sharpe(ens)
                    if n >= 100 and sh > best_sharpe:
                        best_sharpe = sh
                        best_params = {
                            "w_vrp": w_vrp, "leverage": lev,
                            "lookback": lb, "z_short": z_s,
                            "train_sharpe": sh, "train_n": n
                        }

    return best_params


# ── Main ─────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("R54: WALK-FORWARD OUT-OF-SAMPLE VALIDATION")
    print("=" * 70)
    print()

    # Load data
    print("Loading data...")
    dvol = load_dvol_daily("BTC")
    prices = load_prices("BTC")
    skew = load_real_skew("BTC")
    print(f"  DVOL: {len(dvol)} days, Prices: {len(prices)} days, Skew: {len(skew)} days")

    # Common dates (need DVOL + prices + skew)
    all_dates = sorted(set(dvol.keys()) & set(prices.keys()) & set(skew.keys()))
    print(f"  Common dates: {len(all_dates)} ({all_dates[0]} to {all_dates[-1]})")

    # ── Test 1: Leave-One-Year-Out (LOYO) ────────────────────────────
    print(f"\n{'='*70}")
    print("  TEST 1: LEAVE-ONE-YEAR-OUT (LOYO)")
    print(f"{'='*70}")
    print("  Train on all years except Y, test on Y")
    print()

    years = sorted(set(d[:4] for d in all_dates))
    loyo_results = []

    for test_year in years:
        train_dates = [d for d in all_dates if d[:4] != test_year]
        test_dates = [d for d in all_dates if d[:4] == test_year]

        if len(test_dates) < 30 or len(train_dates) < 200:
            continue

        # Optimize on train
        params = optimize_params(train_dates, dvol, prices, skew)
        if not params:
            continue

        # Evaluate on test with optimized params
        vrp_test = compute_vrp_pnl(test_dates, dvol, prices, params["leverage"])
        skw_test = compute_skew_pnl(all_dates, skew, dvol,
                                     params["lookback"], params["z_short"])
        # Filter skew PnL to test dates only
        skw_test = {d: v for d, v in skw_test.items() if d[:4] == test_year}
        ens_test = compute_ensemble_pnl(vrp_test, skw_test, params["w_vrp"])
        sh_test, ret_test, dd_test, n_test = calc_sharpe(ens_test)

        loyo_results.append({
            "year": test_year,
            "train_sharpe": params["train_sharpe"],
            "test_sharpe": sh_test,
            "test_ann_ret": ret_test,
            "test_max_dd": dd_test,
            "test_n": n_test,
            "params": params
        })

        degradation = params["train_sharpe"] - sh_test
        print(f"  {test_year}: Train={params['train_sharpe']:6.2f}  Test={sh_test:6.2f}  "
              f"Δ={degradation:+6.2f}  Ret={ret_test:+.2%}  DD={dd_test:.2%}  "
              f"n={n_test}  [w={params['w_vrp']} lev={params['leverage']} "
              f"lb={params['lookback']} z={params['z_short']}]")

    if loyo_results:
        avg_train = sum(r["train_sharpe"] for r in loyo_results) / len(loyo_results)
        avg_test = sum(r["test_sharpe"] for r in loyo_results) / len(loyo_results)
        print(f"\n  LOYO Average: Train={avg_train:.2f}  Test={avg_test:.2f}  "
              f"Degradation={avg_train - avg_test:+.2f}")

    # ── Test 2: Expanding Window Walk-Forward ────────────────────────
    print(f"\n{'='*70}")
    print("  TEST 2: EXPANDING WINDOW WALK-FORWARD (6-month steps)")
    print(f"{'='*70}")
    print("  Train on [start, T], test on [T, T+6mo], slide T by 6mo")
    print()

    # Define 6-month periods
    periods = []
    start_year = int(all_dates[0][:4])
    end_year = int(all_dates[-1][:4])
    for yr in range(start_year, end_year + 1):
        for half in [(1, 6), (7, 12)]:
            period_start = f"{yr}-{half[0]:02d}-01"
            period_end = f"{yr}-{half[1]:02d}-28" if half[1] == 2 else f"{yr}-{half[1]:02d}-30"
            period_dates = [d for d in all_dates if period_start <= d <= period_end]
            if period_dates:
                label = f"{yr}H{1 if half[0]==1 else 2}"
                periods.append((label, period_dates))

    wf_results = []
    for i in range(2, len(periods)):  # Need at least 2 periods for training
        # Train: all periods up to i
        train_dates = []
        for j in range(i):
            train_dates.extend(periods[j][1])
        train_dates = sorted(set(train_dates))

        # Test: period i
        test_label, test_dates = periods[i]
        if len(test_dates) < 20 or len(train_dates) < 200:
            continue

        # Optimize on train
        params = optimize_params(train_dates, dvol, prices, skew)
        if not params:
            continue

        # Evaluate on test
        vrp_test = compute_vrp_pnl(test_dates, dvol, prices, params["leverage"])
        skw_test = compute_skew_pnl(all_dates, skew, dvol,
                                     params["lookback"], params["z_short"])
        skw_test = {d: v for d, v in skw_test.items() if d in set(test_dates)}
        ens_test = compute_ensemble_pnl(vrp_test, skw_test, params["w_vrp"])
        sh_test, ret_test, dd_test, n_test = calc_sharpe(ens_test)

        wf_results.append({
            "period": test_label,
            "train_sharpe": params["train_sharpe"],
            "test_sharpe": sh_test,
            "test_ann_ret": ret_test,
            "test_n": n_test,
            "params": params
        })

        print(f"  {test_label}: Train={params['train_sharpe']:6.2f}  Test={sh_test:6.2f}  "
              f"Ret={ret_test:+.2%}  n={n_test}  "
              f"[w={params['w_vrp']} lev={params['leverage']}]")

    if wf_results:
        avg_train = sum(r["train_sharpe"] for r in wf_results) / len(wf_results)
        avg_test = sum(r["test_sharpe"] for r in wf_results) / len(wf_results)
        pos_periods = sum(1 for r in wf_results if r["test_sharpe"] > 0)
        print(f"\n  Walk-Forward Average: Train={avg_train:.2f}  Test={avg_test:.2f}  "
              f"Degradation={avg_train - avg_test:+.2f}")
        print(f"  Positive periods: {pos_periods}/{len(wf_results)} "
              f"({pos_periods/len(wf_results):.0%})")

    # ── Test 3: Fixed-Param Robustness ───────────────────────────────
    print(f"\n{'='*70}")
    print("  TEST 3: FIXED-PARAM ROBUSTNESS (R53 params vs alternatives)")
    print(f"{'='*70}")
    print("  How sensitive are results to parameter choices?")
    print()

    param_configs = [
        ("R53 Optimal",     0.5, 1.5, 90,  0.7),
        ("R51 Baseline",    0.4, 1.5, 60,  1.0),
        ("Conservative",    0.3, 1.0, 120, 1.0),
        ("Aggressive",      0.7, 2.0, 60,  0.5),
        ("VRP-heavy",       0.8, 2.0, 90,  0.7),
        ("Skew-heavy",      0.2, 1.0, 90,  0.7),
        ("Equal-simple",    0.5, 1.0, 90,  1.0),
    ]

    print(f"  {'Config':<18s}  {'Sharpe':>7s}  {'AnnRet':>8s}  {'MaxDD':>7s}  {'N':>5s}")
    for name, w, lev, lb, zs in param_configs:
        vrp = compute_vrp_pnl(all_dates, dvol, prices, lev)
        skw = compute_skew_pnl(all_dates, skew, dvol, lb, zs)
        ens = compute_ensemble_pnl(vrp, skw, w)
        sh, ret, dd, n = calc_sharpe(ens)
        print(f"  {name:<18s}  {sh:7.2f}  {ret:+8.2%}  {dd:7.2%}  {n:5d}")

    # ── Test 4: Rolling 1-year Sharpe with R53 params ────────────────
    print(f"\n{'='*70}")
    print("  TEST 4: ROLLING 1-YEAR SHARPE (R53 params)")
    print(f"{'='*70}")
    print()

    vrp_all = compute_vrp_pnl(all_dates, dvol, prices, 1.5)
    skw_all = compute_skew_pnl(all_dates, skew, dvol, 90, 0.7)
    ens_all = compute_ensemble_pnl(vrp_all, skw_all, 0.5)

    ens_dates = sorted(ens_all.keys())
    window = 252  # ~1 year trading days

    rolling_sharpes = []
    print(f"  {'End Date':<12s}  {'Sharpe':>7s}  {'AnnRet':>8s}")
    for i in range(window, len(ens_dates), 21):  # Monthly steps
        window_dates = ens_dates[i-window:i]
        rets = [ens_all[d] for d in window_dates]
        if len(rets) < 200:
            continue
        mean = sum(rets) / len(rets)
        var = sum((r - mean)**2 for r in rets) / len(rets)
        std = math.sqrt(var) if var > 0 else 1e-10
        sh = (mean * 365) / (std * math.sqrt(365))
        ann_ret = mean * 365
        rolling_sharpes.append((ens_dates[i], sh, ann_ret))
        print(f"  {ens_dates[i]:<12s}  {sh:7.2f}  {ann_ret:+8.2%}")

    if rolling_sharpes:
        # Stats
        sharpe_vals = [s[1] for s in rolling_sharpes]
        min_sh = min(sharpe_vals)
        max_sh = max(sharpe_vals)
        avg_sh = sum(sharpe_vals) / len(sharpe_vals)
        pct_above_1 = sum(1 for s in sharpe_vals if s > 1.0) / len(sharpe_vals)
        pct_above_2 = sum(1 for s in sharpe_vals if s > 2.0) / len(sharpe_vals)
        print(f"\n  Rolling Sharpe stats: min={min_sh:.2f}  max={max_sh:.2f}  "
              f"avg={avg_sh:.2f}")
        print(f"  Above 1.0: {pct_above_1:.0%}  Above 2.0: {pct_above_2:.0%}")

    # ── Test 5: Stability of optimal params across subsets ───────────
    print(f"\n{'='*70}")
    print("  TEST 5: PARAMETER STABILITY ACROSS TIME SUBSETS")
    print(f"{'='*70}")
    print()

    subsets = [
        ("2021-2022", [d for d in all_dates if d[:4] in ("2021", "2022")]),
        ("2022-2023", [d for d in all_dates if d[:4] in ("2022", "2023")]),
        ("2023-2024", [d for d in all_dates if d[:4] in ("2023", "2024")]),
        ("2024-2025", [d for d in all_dates if d[:4] in ("2024", "2025")]),
        ("2021-2024", [d for d in all_dates if d[:4] in ("2021","2022","2023","2024")]),
        ("Full period", all_dates),
    ]

    print(f"  {'Subset':<14s}  {'w_vrp':>6s}  {'lev':>5s}  {'lb':>4s}  {'z_s':>5s}  {'Sharpe':>7s}")
    for name, dates in subsets:
        if len(dates) < 200:
            continue
        params = optimize_params(dates, dvol, prices, skew)
        if params:
            print(f"  {name:<14s}  {params['w_vrp']:6.1f}  {params['leverage']:5.1f}  "
                  f"{params['lookback']:4d}  {params['z_short']:5.1f}  "
                  f"{params['train_sharpe']:7.2f}")

    # ── Summary ──────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("R54 WALK-FORWARD VALIDATION SUMMARY")
    print(f"{'='*70}")
    print()

    if loyo_results:
        avg_train = sum(r["train_sharpe"] for r in loyo_results) / len(loyo_results)
        avg_test = sum(r["test_sharpe"] for r in loyo_results) / len(loyo_results)
        print(f"  LOYO: Train={avg_train:.2f} → Test={avg_test:.2f} "
              f"(degradation={avg_train-avg_test:+.2f})")

    if wf_results:
        avg_train = sum(r["train_sharpe"] for r in wf_results) / len(wf_results)
        avg_test = sum(r["test_sharpe"] for r in wf_results) / len(wf_results)
        print(f"  Walk-Forward: Train={avg_train:.2f} → Test={avg_test:.2f} "
              f"(degradation={avg_train-avg_test:+.2f})")

    # Save results
    results = {
        "research_id": "R54",
        "title": "Walk-Forward Out-of-Sample Validation",
        "loyo_results": loyo_results,
        "walk_forward_results": wf_results,
    }
    if rolling_sharpes:
        results["rolling_sharpe_stats"] = {
            "min": min(s[1] for s in rolling_sharpes),
            "max": max(s[1] for s in rolling_sharpes),
            "avg": sum(s[1] for s in rolling_sharpes) / len(rolling_sharpes),
            "pct_above_1": sum(1 for s in rolling_sharpes if s[1] > 1) / len(rolling_sharpes),
            "pct_above_2": sum(1 for s in rolling_sharpes if s[1] > 2) / len(rolling_sharpes),
        }

    outpath = ROOT / "data" / "cache" / "deribit" / "real_surface" / "r54_walk_forward_results.json"
    with open(outpath, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved to {outpath}")


if __name__ == "__main__":
    main()
