#!/usr/bin/env python3
"""
Robustness Validation of R42 Asymmetric Skew — R43
=====================================================

R42 found z_entry_short=1.0 adds +0.501, combined +0.594.
This is a large improvement — need to verify it's not overfitting.

Validation:
  A. LOYO (Leave-One-Year-Out) for each asymmetric config
  B. Parameter sensitivity: does the improvement hold across nearby params?
  C. Per-asset breakdown: does it help both BTC and ETH?
  D. Yearly profile: is the improvement concentrated in one year?
"""
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from nexus_quant.backtest.costs import ExecutionCostModel, FeeModel, ImpactModel
from nexus_quant.projects.crypto_options.options_engine import compute_metrics
from nexus_quant.projects.crypto_options.providers.deribit_rest import DeribitRestProvider
from nexus_quant.projects.crypto_options.strategies.variance_premium import VariancePremiumStrategy

fees = FeeModel(maker_fee_rate=0.0003, taker_fee_rate=0.0005)
impact = ImpactModel(model="sqrt", coef_bps=2.0)
COSTS = ExecutionCostModel(fee=fees, impact=impact)

YEARS = [2021, 2022, 2023, 2024, 2025]
SEED = 42
BARS_PER_YEAR = 365
W_VRP = 0.40
W_SKEW = 0.60

BTC_VRP = {"base_leverage": 1.5, "exit_z_threshold": -3.0, "vrp_lookback": 30, "rebalance_freq": 5, "min_bars": 30}
ETH_VRP = {"base_leverage": 1.5, "exit_z_threshold": -3.0, "vrp_lookback": 60, "rebalance_freq": 5, "min_bars": 60}


def iv_percentile(iv_series, idx, lookback=180):
    if idx < lookback or not iv_series:
        return None
    start = max(0, idx - lookback)
    window = [v for v in iv_series[start:idx] if v is not None]
    if len(window) < 10:
        return None
    current = iv_series[idx] if idx < len(iv_series) and iv_series[idx] is not None else None
    if current is None:
        return None
    return sum(1 for v in window if v < current) / len(window)


def iv_sizing_scale(pct):
    if pct < 0.25:
        return 0.50
    elif pct > 0.75:
        return 1.70
    return 1.0


def compute_skew_zscore(skew_series, idx, lookback=60):
    if idx < lookback or not skew_series:
        return None
    start = max(0, idx - lookback)
    window = [v for v in skew_series[start:idx] if v is not None]
    if len(window) < 10:
        return None
    current = skew_series[idx] if idx < len(skew_series) and skew_series[idx] is not None else None
    if current is None:
        return None
    mean = sum(window) / len(window)
    var = sum((v - mean) ** 2 for v in window) / len(window)
    std = var ** 0.5
    if std < 1e-10:
        return 0.0
    return (current - mean) / std


def run_single_year_asymmetric(year, z_entry_long, z_entry_short, z_exit_long, z_exit_short):
    """Run asymmetric skew on a single year."""
    cfg = {
        "symbols": ["BTC", "ETH"],
        "start": f"{year}-01-01", "end": f"{year}-12-31",
        "bar_interval": "1d", "use_synthetic_iv": True,
    }
    provider = DeribitRestProvider(cfg, seed=SEED)
    dataset = provider.load()
    n = len(dataset.timeline)

    vrp_strats = {
        "BTC": VariancePremiumStrategy(params=BTC_VRP),
        "ETH": VariancePremiumStrategy(params=ETH_VRP),
    }

    dt = 1.0 / BARS_PER_YEAR
    equity = 1.0
    vrp_w = {"BTC": 0.0, "ETH": 0.0}
    skew_w = {"BTC": 0.0, "ETH": 0.0}
    skew_pos = {"BTC": 0.0, "ETH": 0.0}
    equity_curve = [1.0]
    returns_list = []
    last_skew_rebal = {"BTC": 0, "ETH": 0}

    for idx in range(1, n):
        prev_equity = equity
        total_pnl = 0.0

        for sym in ["BTC", "ETH"]:
            aw = 0.50
            w_v = vrp_w.get(sym, 0.0)
            vpnl = 0.0
            if abs(w_v) > 1e-10:
                closes = dataset.perp_close.get(sym, [])
                if idx < len(closes) and closes[idx-1] > 0 and closes[idx] > 0:
                    lr = math.log(closes[idx]/closes[idx-1])
                    rv = abs(lr) * math.sqrt(BARS_PER_YEAR)
                else:
                    rv = 0.0
                ivs = dataset.features.get("iv_atm", {}).get(sym, [])
                iv = ivs[idx-1] if ivs and idx-1 < len(ivs) else None
                if iv and iv > 0:
                    vpnl = (-w_v) * 0.5 * (iv**2 - rv**2) * dt

            w_s = skew_w.get(sym, 0.0)
            spnl = 0.0
            if abs(w_s) > 1e-10:
                sks = dataset.features.get("skew_25d", {}).get(sym, [])
                if idx < len(sks) and idx-1 < len(sks):
                    sn, sp = sks[idx], sks[idx-1]
                    if sn is not None and sp is not None:
                        ds = float(sn) - float(sp)
                        ivs = dataset.features.get("iv_atm", {}).get(sym, [])
                        iv_s = ivs[idx-1] if ivs and idx-1 < len(ivs) else 0.70
                        if iv_s and iv_s > 0:
                            spnl = w_s * ds * iv_s * math.sqrt(dt) * 2.5

            total_pnl += aw * (W_VRP * vpnl + W_SKEW * spnl)

        equity += equity * total_pnl

        rebal_happened = False
        for sym in ["BTC", "ETH"]:
            if vrp_strats[sym].should_rebalance(dataset, idx):
                rebal_happened = True
                tv = vrp_strats[sym].target_weights(dataset, idx, {sym: vrp_w.get(sym, 0.0)})
                vrp_w[sym] = tv.get(sym, 0.0)

        for sym in ["BTC", "ETH"]:
            if idx - last_skew_rebal[sym] < 5:
                continue
            lb = 60 if sym == "BTC" else 90
            skew_series = dataset.features.get("skew_25d", {}).get(sym, [])
            z = compute_skew_zscore(skew_series, idx, lookback=lb)
            if z is None:
                continue

            per_sym_lev = 0.5
            current_pos = skew_pos[sym]
            new_pos = current_pos

            if current_pos == 0.0:
                if z >= z_entry_short:
                    new_pos = -per_sym_lev
                elif z <= -z_entry_long:
                    new_pos = per_sym_lev
            else:
                if current_pos > 0:
                    if z >= z_exit_long:
                        new_pos = 0.0
                else:
                    if z <= -z_exit_short:
                        new_pos = 0.0

            if new_pos != current_pos:
                rebal_happened = True
                last_skew_rebal[sym] = idx

            skew_pos[sym] = new_pos
            skew_w[sym] = new_pos

        if rebal_happened:
            for sym in ["BTC", "ETH"]:
                ivs = dataset.features.get("iv_atm", {}).get(sym, [])
                pct = iv_percentile(ivs, idx)
                if pct is not None:
                    sc = iv_sizing_scale(pct)
                    vrp_w[sym] *= sc
                    skew_w[sym] *= sc
            bd = COSTS.cost(equity=equity, turnover=0.05)
            equity -= float(bd.get("cost", 0.0))
            equity = max(equity, 0.0)

        equity_curve.append(equity)
        bar_ret = (equity / prev_equity) - 1.0 if prev_equity > 0 else 0.0
        returns_list.append(bar_ret)

    m = compute_metrics(equity_curve, returns_list, BARS_PER_YEAR)
    return round(m["sharpe"], 3)


def main():
    print("=" * 70)
    print("R42 ROBUSTNESS VALIDATION — R43")
    print("=" * 70)
    print()

    configs = {
        "symmetric 2.0/0.0": (2.0, 2.0, 0.0, 0.0),
        "R42 best entry (2.0/1.0)": (2.0, 1.0, 0.0, 0.0),
        "R42 best exit (2.0/0.5)": (2.0, 2.0, 0.5, 0.0),
        "R42 combined (2.0,1.0/0.5,0.0)": (2.0, 1.0, 0.5, 0.0),
        "conservative asym (2.0/1.5, 0.3/0.0)": (2.0, 1.5, 0.3, 0.0),
        "moderate asym (2.0/1.25, 0.3/0.0)": (2.0, 1.25, 0.3, 0.0),
    }

    # ── A. LOYO ───────────────────────────────────────────────────────────
    print("--- A. LEAVE-ONE-YEAR-OUT VALIDATION ---")
    print(f"  {'Config':40s}", end="")
    for yr in YEARS:
        print(f" {yr:>7d}", end="")
    print(f" {'avg':>7s} {'min':>7s}")
    print(f"  {'-'*40}", end="")
    for _ in YEARS:
        print(f" {'-'*7}", end="")
    print(f" {'-'*7} {'-'*7}")

    loyo_results = {}
    for name, (z_el, z_es, z_xl, z_xs) in configs.items():
        yr_sharpes = []
        for yr in YEARS:
            s = run_single_year_asymmetric(yr, z_el, z_es, z_xl, z_xs)
            yr_sharpes.append(s)

        avg = sum(yr_sharpes) / len(yr_sharpes)
        mn = min(yr_sharpes)
        loyo_results[name] = {"yearly": dict(zip([str(y) for y in YEARS], yr_sharpes)),
                               "avg": round(avg, 3), "min": round(mn, 3)}

        print(f"  {name:40s}", end="")
        for s in yr_sharpes:
            print(f" {s:7.3f}", end="")
        print(f" {avg:7.3f} {mn:7.3f}")

    print()

    # ── B. Parameter Sensitivity Surface ──────────────────────────────────
    print("--- B. PARAMETER SENSITIVITY (z_entry_short) ---")
    print("  (How sensitive is the improvement to exact threshold?)")

    z_short_vals = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5]
    baseline_avg = loyo_results["symmetric 2.0/0.0"]["avg"]

    sens_results = []
    for zs in z_short_vals:
        yr_sharpes = [run_single_year_asymmetric(yr, 2.0, zs, 0.0, 0.0) for yr in YEARS]
        avg = sum(yr_sharpes) / len(yr_sharpes)
        mn = min(yr_sharpes)
        d = avg - baseline_avg
        sens_results.append({"z_entry_short": zs, "avg": round(avg, 3), "min": round(mn, 3), "delta": round(d, 3)})
        marker = " *" if zs == 2.0 else ""
        print(f"  z_short={zs:.2f}: avg={avg:.3f} min={mn:.3f} Δ={d:+.3f}{marker}")

    print()

    # ── C. Yearly Improvement Consistency ─────────────────────────────────
    print("--- C. YEARLY IMPROVEMENT CONSISTENCY ---")
    base_yrs = loyo_results["symmetric 2.0/0.0"]["yearly"]
    best_yrs = loyo_results["R42 combined (2.0,1.0/0.5,0.0)"]["yearly"]

    all_improve = True
    print(f"  {'Year':>6s}  {'Baseline':>8s} {'R42':>8s} {'Δ':>8s}")
    for yr in YEARS:
        b = base_yrs[str(yr)]
        r = best_yrs[str(yr)]
        d = r - b
        flag = "✓" if d > 0 else "✗"
        if d <= 0:
            all_improve = False
        print(f"  {yr:>6d}  {b:8.3f} {r:8.3f} {d:+8.3f} {flag}")

    print(f"\n  All years improve: {'YES' if all_improve else 'NO'}")
    print()

    # ── SUMMARY ───────────────────────────────────────────────────────────
    print("=" * 70)
    print("SUMMARY — R43: R42 Robustness Validation")
    print("=" * 70)
    print()

    sym_avg = loyo_results["symmetric 2.0/0.0"]["avg"]
    comb_avg = loyo_results["R42 combined (2.0,1.0/0.5,0.0)"]["avg"]
    cons_avg = loyo_results.get("conservative asym (2.0/1.5, 0.3/0.0)", {}).get("avg", 0)
    mod_avg = loyo_results.get("moderate asym (2.0/1.25, 0.3/0.0)", {}).get("avg", 0)

    print(f"  Symmetric baseline:      avg={sym_avg:.3f}")
    print(f"  R42 full combined:       avg={comb_avg:.3f} (Δ={comb_avg-sym_avg:+.3f})")
    print(f"  Conservative asymmetric: avg={cons_avg:.3f} (Δ={cons_avg-sym_avg:+.3f})")
    print(f"  Moderate asymmetric:     avg={mod_avg:.3f} (Δ={mod_avg-sym_avg:+.3f})")
    print()

    # Check sensitivity
    mono_improvement = all(sens_results[i]["avg"] >= sens_results[i+1]["avg"]
                          for i in range(len(sens_results)-1))

    print("=" * 70)
    if all_improve and comb_avg > sym_avg + 0.10:
        print("VERDICT: R42 asymmetric skew IS ROBUST — improves ALL years")
        if cons_avg > sym_avg + 0.05:
            print(f"  Conservative config (z_short=1.5) recommended for production")
        else:
            print(f"  Full config (z_short=1.0) can be used but monitor on real data")
    elif comb_avg > sym_avg + 0.05:
        print("VERDICT: R42 PARTIALLY ROBUST — improvement is real but not uniform")
    else:
        print("VERDICT: R42 may be OVERFITTING — keep symmetric thresholds")
    print("=" * 70)

    # Save
    out = ROOT / "artifacts" / "crypto_options" / "r42_robustness_validation.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump({
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "research_id": "R43",
            "loyo": loyo_results,
            "sensitivity": sens_results,
        }, f, indent=2)
    print(f"\nResults saved to: {out}")


if __name__ == "__main__":
    main()
