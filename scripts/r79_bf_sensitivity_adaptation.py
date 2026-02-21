#!/usr/bin/env python3
"""
R79: BF Sensitivity Adaptation
==================================

R78 confirmed: BF feature variance is compressing (std 0.022→0.007) but Sharpe
remains stable because z-score normalization adapts signal quality automatically.

However, the RETURN level declines with compression because:
  PnL = position * (f_now - f_prev) * IV * sqrt(dt) * sensitivity
  As BF absolute movements shrink (with IV), raw PnL per unit decreases.

R70 showed sensitivity tiers: sens=2.5→2%, sens=5.0→4%, sens=7.5→6%.
Sharpe is ~scale-invariant but RETURN scales with sensitivity.

Key questions:
  1. Can dynamic sensitivity (higher when BF std is low) maintain stable returns?
  2. Does inverse-variance-weighted sensitivity help?
  3. What's the risk/reward of adaptive sensitivity vs static?
  4. If we used higher sensitivity (5.0 or 7.5) during low-variance periods,
     does Sharpe degrade from the "always-on higher leverage" effect?
  5. Walk-forward validation of any adaptive sensitivity approach
  6. LOYO cross-validation

Rule: Static > dynamic has been confirmed 8x. This test checks whether
sensitivity adaptation (a DIFFERENT type of dynamic) might be an exception
since it targets return LEVEL, not signal DIRECTION.
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


def compute_vrp_pnl(dvol_hist, price_hist, dates, leverage=2.0):
    """Compute VRP PnL."""
    pnl = {}
    dt = 1.0 / 365.0
    for i in range(30, len(dates)):
        d = dates[i]
        iv = dvol_hist.get(d)
        if iv is None:
            continue
        # 30d RV
        rets = []
        for j in range(i-30, i):
            p0 = price_hist.get(dates[j])
            p1 = price_hist.get(dates[j+1]) if j+1 < len(dates) else None
            if p0 and p1 and p0 > 0:
                rets.append(math.log(p1 / p0))
        if len(rets) < 20:
            continue
        rv = math.sqrt(sum(r**2 for r in rets) / len(rets)) * math.sqrt(365)
        pnl[d] = leverage * 0.5 * (iv**2 - rv**2) * dt
    return pnl


def compute_bf_pnl_adaptive(dvol_hist, surface_hist, dates, lb=120, z_entry=1.5,
                             z_exit=0.0, sensitivity_fn=None, base_sensitivity=2.5):
    """Compute BF PnL with adaptive or static sensitivity."""
    bf_vals = {}
    for d in dates:
        if d in surface_hist and "butterfly_25d" in surface_hist[d]:
            bf_vals[d] = surface_hist[d]["butterfly_25d"]

    dt = 1.0 / 365.0
    position = 0.0
    pnl = {}
    z_scores = {}

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
        z_scores[d] = z

        if z > z_entry:
            position = -1.0
        elif z < -z_entry:
            position = 1.0
        elif z_exit > 0 and abs(z) < z_exit:
            position = 0.0

        # Get sensitivity
        if sensitivity_fn is not None:
            sens = sensitivity_fn(d, std, bf_vals, dates, i)
        else:
            sens = base_sensitivity

        iv = dvol_hist.get(d)
        f_now, f_prev = bf_vals.get(d), bf_vals.get(dp)
        if f_now is not None and f_prev is not None and iv is not None and position != 0:
            day_pnl = position * (f_now - f_prev) * iv * math.sqrt(dt) * sens
            pnl[d] = day_pnl
        else:
            pnl[d] = 0.0

    return pnl, z_scores, bf_vals


def compute_stats(rets):
    if len(rets) < 20:
        return {"sharpe": 0, "ann_ret": 0, "max_dd": 0}
    mean = sum(rets) / len(rets)
    std = math.sqrt(sum((r - mean)**2 for r in rets) / len(rets))
    sharpe = (mean * 365) / (std * math.sqrt(365)) if std > 0 else 0
    ann_ret = mean * 365
    cum = peak = max_dd = 0.0
    for r in rets:
        cum += r
        peak = max(peak, cum)
        max_dd = min(max_dd, cum - peak)
    return {"sharpe": round(sharpe, 3), "ann_ret": round(ann_ret * 100, 3), "max_dd": round(max_dd * 100, 3)}


def portfolio_pnl(vrp, bf, dates, w_vrp=0.10, w_bf=0.90):
    """Combine VRP and BF into portfolio."""
    pnl = {}
    for d in dates:
        v = vrp.get(d, 0)
        b = bf.get(d, 0)
        pnl[d] = w_vrp * v + w_bf * b
    return pnl


# ═══════════════════════════════════════════════════════════════
# Analysis 1: Static Sensitivity Tiers (Baseline)
# ═══════════════════════════════════════════════════════════════

def static_sensitivity_tiers(dvol_hist, surface_hist, price_hist, dates):
    print("\n" + "=" * 70)
    print("  ANALYSIS 1: STATIC SENSITIVITY TIERS (BASELINE)")
    print("=" * 70)

    vrp_pnl = compute_vrp_pnl(dvol_hist, price_hist, dates)
    results = {}

    for sens in [2.5, 5.0, 7.5, 10.0, 15.0]:
        bf_pnl, _, _ = compute_bf_pnl_adaptive(dvol_hist, surface_hist, dates,
                                                 base_sensitivity=sens)
        port = portfolio_pnl(vrp_pnl, bf_pnl, dates)
        port_rets = [port[d] for d in sorted(port.keys()) if port[d] != 0]
        bf_rets = [bf_pnl[d] for d in sorted(bf_pnl.keys()) if bf_pnl[d] != 0]

        bf_stats = compute_stats(bf_rets)
        port_stats = compute_stats(port_rets)
        results[sens] = {"bf": bf_stats, "port": port_stats}

    print(f"\n  {'Sens':>6} {'BF Sharpe':>10} {'BF Ret%':>10} {'BF MaxDD%':>10} {'Port Sharpe':>12} {'Port Ret%':>12} {'Port MaxDD%':>12}")
    print(f"  {'─'*6} {'─'*10} {'─'*10} {'─'*10} {'─'*12} {'─'*12} {'─'*12}")

    for sens in sorted(results.keys()):
        r = results[sens]
        print(f"  {sens:>6.1f} {r['bf']['sharpe']:>10.3f} {r['bf']['ann_ret']:>10.3f} {r['bf']['max_dd']:>10.3f}"
              f" {r['port']['sharpe']:>12.3f} {r['port']['ann_ret']:>12.3f} {r['port']['max_dd']:>12.3f}")

    return results


# ═══════════════════════════════════════════════════════════════
# Analysis 2: Inverse Variance Sensitivity
# ═══════════════════════════════════════════════════════════════

def inverse_variance_sensitivity(dvol_hist, surface_hist, price_hist, dates):
    print("\n" + "=" * 70)
    print("  ANALYSIS 2: INVERSE VARIANCE SENSITIVITY")
    print("=" * 70)
    print(f"\n  Idea: When BF std is low, increase sensitivity proportionally")
    print(f"  Goal: maintain constant expected PnL magnitude regardless of BF variance")

    vrp_pnl = compute_vrp_pnl(dvol_hist, price_hist, dates)
    results = {}

    # Reference std = median std across full sample
    bf_vals = {}
    for d in dates:
        if d in surface_hist and "butterfly_25d" in surface_hist[d]:
            bf_vals[d] = surface_hist[d]["butterfly_25d"]

    bf_list = [(d, bf_vals[d]) for d in dates if d in bf_vals]
    all_rolling_std = []
    for i in range(120, len(bf_list)):
        w = [v[1] for v in bf_list[i-120:i]]
        mean = sum(w) / len(w)
        std = math.sqrt(sum((v - mean)**2 for v in w) / len(w))
        all_rolling_std.append(std)

    if not all_rolling_std:
        print("  No data")
        return {}

    ref_std = sorted(all_rolling_std)[len(all_rolling_std) // 2]  # median
    print(f"  Reference (median) BF std: {ref_std:.5f}")

    # Configs: target_sens when at median std, capped at max_sens
    configs = [
        ("inv_var_2.5_cap5", 2.5, 5.0),
        ("inv_var_2.5_cap7.5", 2.5, 7.5),
        ("inv_var_2.5_cap10", 2.5, 10.0),
        ("inv_var_5.0_cap10", 5.0, 10.0),
        ("inv_var_5.0_cap15", 5.0, 15.0),
        ("inv_var_2.5_cap15", 2.5, 15.0),
    ]

    for name, base_sens, max_sens in configs:
        def make_fn(bs, ms, rs):
            def fn(d, current_std, bf_vals, dates, idx):
                # Scale inversely with variance
                if current_std > 0:
                    raw_sens = bs * (rs / current_std)
                else:
                    raw_sens = bs
                return max(bs, min(ms, raw_sens))  # floor at base, cap at max
            return fn

        bf_pnl, _, _ = compute_bf_pnl_adaptive(dvol_hist, surface_hist, dates,
                                                 sensitivity_fn=make_fn(base_sens, max_sens, ref_std))
        port = portfolio_pnl(vrp_pnl, bf_pnl, dates)

        bf_rets = [bf_pnl[d] for d in sorted(bf_pnl.keys()) if bf_pnl[d] != 0]
        port_rets = [port[d] for d in sorted(port.keys()) if port[d] != 0]

        bf_stats = compute_stats(bf_rets)
        port_stats = compute_stats(port_rets)
        results[name] = {"bf": bf_stats, "port": port_stats, "base": base_sens, "cap": max_sens}

    print(f"\n  {'Config':>25} {'BF Sharpe':>10} {'BF Ret%':>10} {'BF MaxDD%':>10} {'Port Sharpe':>12} {'Port Ret%':>12}")
    print(f"  {'─'*25} {'─'*10} {'─'*10} {'─'*10} {'─'*12} {'─'*12}")

    for name in sorted(results.keys()):
        r = results[name]
        print(f"  {name:>25} {r['bf']['sharpe']:>10.3f} {r['bf']['ann_ret']:>10.3f} {r['bf']['max_dd']:>10.3f}"
              f" {r['port']['sharpe']:>12.3f} {r['port']['ann_ret']:>12.3f}")

    # Compare with static
    print(f"\n  Static baselines for reference:")
    for sens in [2.5, 5.0, 7.5]:
        bf_pnl, _, _ = compute_bf_pnl_adaptive(dvol_hist, surface_hist, dates, base_sensitivity=sens)
        bf_rets = [bf_pnl[d] for d in sorted(bf_pnl.keys()) if bf_pnl[d] != 0]
        port = portfolio_pnl(vrp_pnl, bf_pnl, dates)
        port_rets = [port[d] for d in sorted(port.keys()) if port[d] != 0]
        bf_s = compute_stats(bf_rets)
        port_s = compute_stats(port_rets)
        print(f"  {'static_'+str(sens):>25} {bf_s['sharpe']:>10.3f} {bf_s['ann_ret']:>10.3f} {bf_s['max_dd']:>10.3f}"
              f" {port_s['sharpe']:>12.3f} {port_s['ann_ret']:>12.3f}")

    return results


# ═══════════════════════════════════════════════════════════════
# Analysis 3: IV-Scaled Sensitivity
# ═══════════════════════════════════════════════════════════════

def iv_scaled_sensitivity(dvol_hist, surface_hist, price_hist, dates):
    print("\n" + "=" * 70)
    print("  ANALYSIS 3: IV-SCALED SENSITIVITY")
    print("=" * 70)
    print(f"\n  Idea: Since BF PnL = pos * d_bf * IV * sqrt(dt) * sens,")
    print(f"  and BF std is correlated with IV (r=0.645 from R78),")
    print(f"  scaling sens by 1/IV² could compensate for IV-driven compression")

    vrp_pnl = compute_vrp_pnl(dvol_hist, price_hist, dates)

    # Reference IV (median)
    iv_vals = sorted(dvol_hist.values())
    ref_iv = iv_vals[len(iv_vals) // 2]
    print(f"  Reference (median) IV: {ref_iv:.3f}")

    configs = [
        ("iv_scale_2.5_lin", 2.5, 1, 7.5),   # sens * (ref_iv / IV), linear
        ("iv_scale_2.5_sq", 2.5, 2, 10.0),    # sens * (ref_iv / IV)², quadratic
        ("iv_scale_5.0_lin", 5.0, 1, 15.0),
        ("iv_scale_5.0_sq", 5.0, 2, 20.0),
    ]

    results = {}
    for name, base_sens, power, max_sens in configs:
        def make_fn(bs, pw, ms, riv):
            def fn(d, current_std, bf_vals, dates, idx):
                iv = dvol_hist.get(d, riv)
                if iv > 0:
                    raw_sens = bs * (riv / iv) ** pw
                else:
                    raw_sens = bs
                return max(bs * 0.5, min(ms, raw_sens))
            return fn

        bf_pnl, _, _ = compute_bf_pnl_adaptive(dvol_hist, surface_hist, dates,
                                                 sensitivity_fn=make_fn(base_sens, power, max_sens, ref_iv))
        port = portfolio_pnl(vrp_pnl, bf_pnl, dates)

        bf_rets = [bf_pnl[d] for d in sorted(bf_pnl.keys()) if bf_pnl[d] != 0]
        port_rets = [port[d] for d in sorted(port.keys()) if port[d] != 0]

        bf_stats = compute_stats(bf_rets)
        port_stats = compute_stats(port_rets)
        results[name] = {"bf": bf_stats, "port": port_stats}

    print(f"\n  {'Config':>25} {'BF Sharpe':>10} {'BF Ret%':>10} {'BF MaxDD%':>10} {'Port Sharpe':>12} {'Port Ret%':>12}")
    print(f"  {'─'*25} {'─'*10} {'─'*10} {'─'*10} {'─'*12} {'─'*12}")

    for name in sorted(results.keys()):
        r = results[name]
        print(f"  {name:>25} {r['bf']['sharpe']:>10.3f} {r['bf']['ann_ret']:>10.3f} {r['bf']['max_dd']:>10.3f}"
              f" {r['port']['sharpe']:>12.3f} {r['port']['ann_ret']:>12.3f}")

    return results


# ═══════════════════════════════════════════════════════════════
# Analysis 4: Step-Function Sensitivity Regimes
# ═══════════════════════════════════════════════════════════════

def step_sensitivity_regimes(dvol_hist, surface_hist, price_hist, dates):
    print("\n" + "=" * 70)
    print("  ANALYSIS 4: STEP-FUNCTION SENSITIVITY REGIMES")
    print("=" * 70)
    print(f"\n  Idea: Simple regime-based sensitivity: low BF std → high sens, high BF std → low sens")

    vrp_pnl = compute_vrp_pnl(dvol_hist, price_hist, dates)

    # Compute BF std percentiles
    bf_vals = {}
    for d in dates:
        if d in surface_hist and "butterfly_25d" in surface_hist[d]:
            bf_vals[d] = surface_hist[d]["butterfly_25d"]

    bf_list = [(d, bf_vals[d]) for d in dates if d in bf_vals]
    rolling_std = {}
    for i in range(120, len(bf_list)):
        d = bf_list[i][0]
        w = [v[1] for v in bf_list[i-120:i]]
        mean = sum(w) / len(w)
        std = math.sqrt(sum((v - mean)**2 for v in w) / len(w))
        rolling_std[d] = std

    # Percentile thresholds (from full sample — look-ahead bias, but directional)
    std_vals = sorted(rolling_std.values())
    p25 = std_vals[len(std_vals) // 4]
    p50 = std_vals[len(std_vals) // 2]
    p75 = std_vals[3 * len(std_vals) // 4]
    print(f"  BF std percentiles: P25={p25:.5f}, P50={p50:.5f}, P75={p75:.5f}")

    configs = [
        ("step_2x_low", {0: 5.0, 1: 2.5}, p50),          # 2x when below median
        ("step_3x_low", {0: 7.5, 1: 2.5}, p50),          # 3x when below median
        ("step_2x_q1", {0: 5.0, 1: 2.5}, p25),           # 2x only in Q1
        ("step_grad_3", {0: 7.5, 1: 5.0, 2: 2.5}, None), # 3-tier
    ]

    results = {}
    for name, sens_map, threshold in configs:
        def make_fn(sm, th, p25=p25, p50=p50, p75=p75):
            def fn(d, current_std, bf_vals, dates, idx):
                if d not in rolling_std:
                    return 2.5
                s = rolling_std[d]
                if th is not None:
                    # Binary: below or above threshold
                    return sm[0] if s < th else sm[1]
                else:
                    # 3-tier
                    if s < p25:
                        return sm[0]
                    elif s < p75:
                        return sm[1]
                    else:
                        return sm[2]
            return fn

        bf_pnl, _, _ = compute_bf_pnl_adaptive(dvol_hist, surface_hist, dates,
                                                 sensitivity_fn=make_fn(sens_map, threshold))
        port = portfolio_pnl(vrp_pnl, bf_pnl, dates)

        bf_rets = [bf_pnl[d] for d in sorted(bf_pnl.keys()) if bf_pnl[d] != 0]
        port_rets = [port[d] for d in sorted(port.keys()) if port[d] != 0]

        bf_stats = compute_stats(bf_rets)
        port_stats = compute_stats(port_rets)
        results[name] = {"bf": bf_stats, "port": port_stats}

    print(f"\n  {'Config':>20} {'BF Sharpe':>10} {'BF Ret%':>10} {'BF MaxDD%':>10} {'Port Sharpe':>12} {'Port Ret%':>12}")
    print(f"  {'─'*20} {'─'*10} {'─'*10} {'─'*10} {'─'*12} {'─'*12}")

    for name in sorted(results.keys()):
        r = results[name]
        print(f"  {name:>20} {r['bf']['sharpe']:>10.3f} {r['bf']['ann_ret']:>10.3f} {r['bf']['max_dd']:>10.3f}"
              f" {r['port']['sharpe']:>12.3f} {r['port']['ann_ret']:>12.3f}")

    return results


# ═══════════════════════════════════════════════════════════════
# Analysis 5: Walk-Forward Validation
# ═══════════════════════════════════════════════════════════════

def walk_forward_validation(dvol_hist, surface_hist, price_hist, dates):
    print("\n" + "=" * 70)
    print("  ANALYSIS 5: WALK-FORWARD VALIDATION")
    print("=" * 70)

    # Test best adaptive configs vs static on expanding window walk-forward
    # Train: first N years. Test: next 6 months.

    vrp_pnl = compute_vrp_pnl(dvol_hist, price_hist, dates)

    # Precompute bf vals and rolling std
    bf_vals = {}
    for d in dates:
        if d in surface_hist and "butterfly_25d" in surface_hist[d]:
            bf_vals[d] = surface_hist[d]["butterfly_25d"]

    bf_list = [(d, bf_vals[d]) for d in dates if d in bf_vals]
    rolling_std = {}
    for i in range(120, len(bf_list)):
        d = bf_list[i][0]
        w = [v[1] for v in bf_list[i-120:i]]
        mean = sum(w) / len(w)
        std = math.sqrt(sum((v - mean)**2 for v in w) / len(w))
        rolling_std[d] = std

    # Test periods (6-month OOS windows)
    test_periods = [
        ("2023H1", "2023-01-01", "2023-07-01"),
        ("2023H2", "2023-07-01", "2024-01-01"),
        ("2024H1", "2024-01-01", "2024-07-01"),
        ("2024H2", "2024-07-01", "2025-01-01"),
        ("2025H1", "2025-01-01", "2025-07-01"),
        ("2025H2", "2025-07-01", "2026-01-01"),
        ("2026H1", "2026-01-01", "2026-07-01"),
    ]

    configs = {
        "static_2.5": lambda d, std, bv, dt, i: 2.5,
        "static_5.0": lambda d, std, bv, dt, i: 5.0,
        "inv_var_2.5_c7.5": None,  # will build per period
    }

    # For inv_var, we need per-period median std from training data
    print(f"\n  {'Period':>8}", end="")
    for name in ["static_2.5", "static_5.0", "inv_var_2.5_c7.5"]:
        print(f" {name:>18}", end="")
    print()
    print(f"  {'─'*8}", end="")
    for _ in range(3):
        print(f" {'─'*18}", end="")
    print()

    wf_results = defaultdict(list)
    for period_name, test_start, test_end in test_periods:
        test_dates = [d for d in dates if test_start <= d < test_end]
        if len(test_dates) < 30:
            continue

        # Training data: everything before test_start
        train_dates = [d for d in dates if d < test_start]
        train_stds = [rolling_std[d] for d in train_dates if d in rolling_std]
        if not train_stds:
            continue
        train_median_std = sorted(train_stds)[len(train_stds) // 2]

        row = f"  {period_name:>8}"
        for name in ["static_2.5", "static_5.0", "inv_var_2.5_c7.5"]:
            if name == "static_2.5":
                fn = lambda d, std, bv, dt, i: 2.5
            elif name == "static_5.0":
                fn = lambda d, std, bv, dt, i: 5.0
            else:
                def make_inv(ref):
                    def fn(d, std, bv, dt, i):
                        if std > 0:
                            raw = 2.5 * (ref / std)
                        else:
                            raw = 2.5
                        return max(2.5, min(7.5, raw))
                    return fn
                fn = make_inv(train_median_std)

            # Compute BF PnL for test period
            bf_pnl, _, _ = compute_bf_pnl_adaptive(dvol_hist, surface_hist, dates,
                                                     sensitivity_fn=fn)
            port = portfolio_pnl(vrp_pnl, bf_pnl, dates)
            test_rets = [port[d] for d in test_dates if d in port and port[d] != 0]

            if len(test_rets) >= 20:
                s = compute_stats(test_rets)
                row += f" {s['sharpe']:>8.2f} ({s['ann_ret']:>5.2f}%)"
                wf_results[name].append(s["sharpe"])
            else:
                row += f" {'N/A':>18}"

        print(row)

    # Averages
    print(f"\n  Walk-Forward Averages:")
    for name in ["static_2.5", "static_5.0", "inv_var_2.5_c7.5"]:
        vals = wf_results.get(name, [])
        if vals:
            avg = sum(vals) / len(vals)
            mn = min(vals)
            pos = sum(1 for v in vals if v > 0) / len(vals) * 100
            print(f"    {name:>20}: avg={avg:.3f}, min={mn:.3f}, positive={pos:.0f}%")

    return wf_results


# ═══════════════════════════════════════════════════════════════
# Analysis 6: LOYO Cross-Validation
# ═══════════════════════════════════════════════════════════════

def loyo_validation(dvol_hist, surface_hist, price_hist, dates):
    print("\n" + "=" * 70)
    print("  ANALYSIS 6: LEAVE-ONE-YEAR-OUT VALIDATION")
    print("=" * 70)

    vrp_pnl = compute_vrp_pnl(dvol_hist, price_hist, dates)

    bf_vals = {}
    for d in dates:
        if d in surface_hist and "butterfly_25d" in surface_hist[d]:
            bf_vals[d] = surface_hist[d]["butterfly_25d"]

    bf_list = [(d, bf_vals[d]) for d in dates if d in bf_vals]
    rolling_std = {}
    for i in range(120, len(bf_list)):
        d = bf_list[i][0]
        w = [v[1] for v in bf_list[i-120:i]]
        mean = sum(w) / len(w)
        std = math.sqrt(sum((v - mean)**2 for v in w) / len(w))
        rolling_std[d] = std

    years = sorted(set(d[:4] for d in dates))

    configs = ["static_2.5", "static_5.0", "inv_var_2.5_c7.5"]

    print(f"\n  {'Year':>6}", end="")
    for name in configs:
        print(f" {name:>18}", end="")
    print()
    print(f"  {'─'*6}", end="")
    for _ in configs:
        print(f" {'─'*18}", end="")
    print()

    loyo_results = defaultdict(list)
    for test_yr in years:
        test_dates = [d for d in dates if d[:4] == test_yr]
        if len(test_dates) < 30:
            continue

        # Training: all years except test_yr
        train_dates = [d for d in dates if d[:4] != test_yr]
        train_stds = [rolling_std[d] for d in train_dates if d in rolling_std]
        if not train_stds:
            continue
        train_median = sorted(train_stds)[len(train_stds) // 2]

        row = f"  {test_yr:>6}"
        for name in configs:
            if name == "static_2.5":
                fn = lambda d, std, bv, dt, i: 2.5
            elif name == "static_5.0":
                fn = lambda d, std, bv, dt, i: 5.0
            else:
                def make_inv(ref):
                    def fn(d, std, bv, dt, i):
                        if std > 0:
                            raw = 2.5 * (ref / std)
                        else:
                            raw = 2.5
                        return max(2.5, min(7.5, raw))
                    return fn
                fn = make_inv(train_median)

            bf_pnl, _, _ = compute_bf_pnl_adaptive(dvol_hist, surface_hist, dates,
                                                     sensitivity_fn=fn)
            port = portfolio_pnl(vrp_pnl, bf_pnl, dates)
            test_rets = [port[d] for d in test_dates if d in port and port[d] != 0]

            if len(test_rets) >= 20:
                s = compute_stats(test_rets)
                row += f" {s['sharpe']:>8.2f} ({s['ann_ret']:>5.2f}%)"
                loyo_results[name].append(s["sharpe"])
            else:
                row += f" {'N/A':>18}"

        print(row)

    print(f"\n  LOYO Averages:")
    for name in configs:
        vals = loyo_results.get(name, [])
        if vals:
            avg = sum(vals) / len(vals)
            mn = min(vals)
            print(f"    {name:>20}: avg={avg:.3f}, min={mn:.3f}")

    return loyo_results


# ═══════════════════════════════════════════════════════════════
# Analysis 7: Simple Higher Sensitivity
# ═══════════════════════════════════════════════════════════════

def simple_higher_sensitivity(dvol_hist, surface_hist, price_hist, dates):
    print("\n" + "=" * 70)
    print("  ANALYSIS 7: SIMPLE HIGHER STATIC SENSITIVITY")
    print("=" * 70)
    print(f"\n  Given that BF returns compress with IV, the simplest fix")
    print(f"  is just to use HIGHER static sensitivity. R70 showed Sharpe")
    print(f"  is ~scale-invariant. The question: is static sens=5.0 or 7.5")
    print(f"  better than any adaptive approach?")

    vrp_pnl = compute_vrp_pnl(dvol_hist, price_hist, dates)

    # By-year analysis for different static sensitivity levels
    results = {}
    for sens in [2.5, 5.0, 7.5, 10.0]:
        bf_pnl, _, _ = compute_bf_pnl_adaptive(dvol_hist, surface_hist, dates,
                                                 base_sensitivity=sens)
        port = portfolio_pnl(vrp_pnl, bf_pnl, dates)

        by_yr = defaultdict(list)
        for d in sorted(port.keys()):
            if port[d] != 0:
                by_yr[d[:4]].append(port[d])

        yr_stats = {}
        for yr, rets in sorted(by_yr.items()):
            yr_stats[yr] = compute_stats(rets)

        results[sens] = yr_stats

    print(f"\n  Portfolio Sharpe by year and sensitivity:")
    print(f"  {'Sens':>6}", end="")
    years = sorted(set(d[:4] for d in dates))
    for yr in years:
        print(f" {yr:>8}", end="")
    print()
    print(f"  {'─'*6}", end="")
    for _ in years:
        print(f" {'─'*8}", end="")
    print()

    for sens in sorted(results.keys()):
        print(f"  {sens:>6.1f}", end="")
        for yr in years:
            if yr in results[sens]:
                print(f" {results[sens][yr]['sharpe']:>8.2f}", end="")
            else:
                print(f" {'N/A':>8}", end="")
        print()

    print(f"\n  Portfolio Ann Ret% by year and sensitivity:")
    print(f"  {'Sens':>6}", end="")
    for yr in years:
        print(f" {yr:>8}", end="")
    print()
    print(f"  {'─'*6}", end="")
    for _ in years:
        print(f" {'─'*8}", end="")
    print()

    for sens in sorted(results.keys()):
        print(f"  {sens:>6.1f}", end="")
        for yr in years:
            if yr in results[sens]:
                print(f" {results[sens][yr]['ann_ret']:>7.2f}%", end="")
            else:
                print(f" {'N/A':>8}", end="")
        print()

    return results


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════

def main():
    print("=" * 70)
    print("  R79: BF SENSITIVITY ADAPTATION")
    print("=" * 70)
    print(f"\n  Context: R78 confirmed BF std is declining (0.022→0.007)")
    print(f"  but Sharpe is stable (corr=-0.088 with std).")
    print(f"  The issue is RETURN LEVEL compression, not signal quality.")
    print(f"  Can we adapt sensitivity to maintain returns?")
    print(f"\n  Standing rule: static > dynamic (8x confirmed).")
    print(f"  Testing if sensitivity adaptation is a different case.")

    # Load data
    print(f"\n  Loading data...")
    dvol_hist = load_dvol_history("BTC")
    price_hist = load_price_history("BTC")
    surface_hist = load_surface("BTC")

    common_dates = sorted(set(dvol_hist.keys()) & set(surface_hist.keys()))
    print(f"  Dates: {len(common_dates)} ({common_dates[0]} to {common_dates[-1]})")

    # Run analyses
    a1 = static_sensitivity_tiers(dvol_hist, surface_hist, price_hist, common_dates)
    a2 = inverse_variance_sensitivity(dvol_hist, surface_hist, price_hist, common_dates)
    a3 = iv_scaled_sensitivity(dvol_hist, surface_hist, price_hist, common_dates)
    a4 = step_sensitivity_regimes(dvol_hist, surface_hist, price_hist, common_dates)
    a5 = walk_forward_validation(dvol_hist, surface_hist, price_hist, common_dates)
    a6 = loyo_validation(dvol_hist, surface_hist, price_hist, common_dates)
    a7 = simple_higher_sensitivity(dvol_hist, surface_hist, price_hist, common_dates)

    # ─── Summary ────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  R79 SUMMARY: BF SENSITIVITY ADAPTATION")
    print("=" * 70)

    # Compare best adaptive vs best static
    static_5_sharpe = a1.get(5.0, {}).get("port", {}).get("sharpe", 0) if a1 else 0
    static_5_ret = a1.get(5.0, {}).get("port", {}).get("ann_ret", 0) if a1 else 0

    # Check if any adaptive beats static 5.0
    best_adaptive_name = None
    best_adaptive_sharpe = 0

    if a2:
        for name, r in a2.items():
            if r["port"]["sharpe"] > best_adaptive_sharpe:
                best_adaptive_sharpe = r["port"]["sharpe"]
                best_adaptive_name = f"inv_var_{name}"

    if a3:
        for name, r in a3.items():
            if r["port"]["sharpe"] > best_adaptive_sharpe:
                best_adaptive_sharpe = r["port"]["sharpe"]
                best_adaptive_name = f"iv_scale_{name}"

    if a4:
        for name, r in a4.items():
            if r["port"]["sharpe"] > best_adaptive_sharpe:
                best_adaptive_sharpe = r["port"]["sharpe"]
                best_adaptive_name = f"step_{name}"

    # WF comparison
    wf_static_25 = a5.get("static_2.5", []) if a5 else []
    wf_static_50 = a5.get("static_5.0", []) if a5 else []
    wf_adaptive = a5.get("inv_var_2.5_c7.5", []) if a5 else []

    wf_avg_25 = sum(wf_static_25) / len(wf_static_25) if wf_static_25 else 0
    wf_avg_50 = sum(wf_static_50) / len(wf_static_50) if wf_static_50 else 0
    wf_avg_ad = sum(wf_adaptive) / len(wf_adaptive) if wf_adaptive else 0

    print(f"\n  In-sample comparison:")
    print(f"    Static sens=2.5: Sharpe {a1.get(2.5, {}).get('port', {}).get('sharpe', 0):.3f}, "
          f"Ret {a1.get(2.5, {}).get('port', {}).get('ann_ret', 0):.3f}%")
    print(f"    Static sens=5.0: Sharpe {static_5_sharpe:.3f}, Ret {static_5_ret:.3f}%")
    print(f"    Best adaptive:   Sharpe {best_adaptive_sharpe:.3f} ({best_adaptive_name})")

    print(f"\n  Walk-Forward OOS:")
    print(f"    Static 2.5: avg Sharpe {wf_avg_25:.3f}")
    print(f"    Static 5.0: avg Sharpe {wf_avg_50:.3f}")
    print(f"    Inv-var adaptive: avg Sharpe {wf_avg_ad:.3f}")

    # Verdict
    if wf_avg_ad > wf_avg_50 + 0.1:
        verdict = "ADAPTIVE WINS — inverse-variance sensitivity improves OOS"
        recommendation = f"Use inv_var sensitivity (base=2.5, cap=7.5)"
    elif wf_avg_50 > wf_avg_25 + 0.05:
        verdict = "HIGHER STATIC WINS — just use sens=5.0 instead of 2.5"
        recommendation = f"Upgrade static sensitivity from 2.5 to 5.0"
    else:
        verdict = "NO IMPROVEMENT — current sens=2.5 is fine"
        recommendation = f"Keep current production config (sens=2.5)"

    print(f"\n  VERDICT: {verdict}")
    print(f"  RECOMMENDATION: {recommendation}")
    print(f"\n  9th test of static > dynamic rule:")
    if "ADAPTIVE WINS" in verdict:
        print(f"    EXCEPTION FOUND — sensitivity adaptation is a valid special case")
    else:
        print(f"    CONFIRMED AGAIN — 9th confirmation of static > dynamic")

    # Save
    results = {
        "research_id": "R79",
        "title": "BF Sensitivity Adaptation",
        "static_2.5_sharpe": a1.get(2.5, {}).get("port", {}).get("sharpe", 0),
        "static_5.0_sharpe": static_5_sharpe,
        "best_adaptive_sharpe": best_adaptive_sharpe,
        "best_adaptive_config": best_adaptive_name,
        "wf_static_2.5_avg": wf_avg_25,
        "wf_static_5.0_avg": wf_avg_50,
        "wf_adaptive_avg": wf_avg_ad,
        "verdict": verdict,
        "recommendation": recommendation
    }

    out_path = ROOT / "data" / "cache" / "deribit" / "real_surface" / "r79_sensitivity_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved to {out_path}")


if __name__ == "__main__":
    main()
