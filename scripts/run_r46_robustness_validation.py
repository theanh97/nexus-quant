#!/usr/bin/env python3
"""
R46 Robustness Validation — R47
=================================

R46 found per-asset skew params improve +0.108 combined:
  - BTC: lookback=45d (from 60d), z_short=1.0 (unchanged)
  - ETH: lookback=75d (from 90d), z_short=0.75 (from 1.0)

BUT min Sharpe dropped 3.973→3.864 (2022 regression).
Need LOYO validation before integrating.

Test configs:
  1. Baseline: BTC=60d/z=1.0, ETH=90d/z=1.0
  2. R46 lookback only: BTC=45d/z=1.0, ETH=75d/z=1.0
  3. R46 z_short only: BTC=60d/z=1.0, ETH=90d/z=0.75
  4. R46 combined: BTC=45d/z=1.0, ETH=75d/z=0.75
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


def run_single_year(year, btc_lb, eth_lb, btc_z_short, eth_z_short):
    """Run a single year with specified per-asset skew params."""
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

    skew_lbs = {"BTC": btc_lb, "ETH": eth_lb}
    z_shorts = {"BTC": btc_z_short, "ETH": eth_z_short}

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
            lb = skew_lbs[sym]
            skew_series = dataset.features.get("skew_25d", {}).get(sym, [])
            z = compute_skew_zscore(skew_series, idx, lookback=lb)
            if z is None:
                continue

            per_sym_lev = 0.5
            current_pos = skew_pos[sym]
            new_pos = current_pos

            z_es = z_shorts[sym]
            if current_pos == 0.0:
                if z >= z_es:
                    new_pos = -per_sym_lev
                elif z <= -2.0:
                    new_pos = per_sym_lev
            else:
                if current_pos > 0:
                    if z >= 0.5:
                        new_pos = 0.0
                else:
                    if z <= 0.0:
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
    print("R46 ROBUSTNESS VALIDATION — R47 (LOYO)")
    print("=" * 70)
    print()

    configs = {
        "Baseline (60/90, z=1.0/1.0)": (60, 90, 1.0, 1.0),
        "R46 lookback (45/75, z=1.0/1.0)": (45, 75, 1.0, 1.0),
        "R46 z_short (60/90, z=1.0/0.75)": (60, 90, 1.0, 0.75),
        "R46 combined (45/75, z=1.0/0.75)": (45, 75, 1.0, 0.75),
        "Conservative (50/80, z=1.0/0.85)": (50, 80, 1.0, 0.85),
    }

    print(f"  {'Config':40s}", end="")
    for yr in YEARS:
        print(f" {yr:>7d}", end="")
    print(f" {'avg':>7s} {'min':>7s}")
    print(f"  {'-'*40}", end="")
    for _ in YEARS:
        print(f" {'-'*7}", end="")
    print(f" {'-'*7} {'-'*7}")

    all_results = {}
    for name, (btc_lb, eth_lb, btc_z, eth_z) in configs.items():
        yr_sharpes = []
        for yr in YEARS:
            s = run_single_year(yr, btc_lb, eth_lb, btc_z, eth_z)
            yr_sharpes.append(s)

        avg = sum(yr_sharpes) / len(yr_sharpes)
        mn = min(yr_sharpes)
        all_results[name] = {
            "yearly": dict(zip([str(y) for y in YEARS], yr_sharpes)),
            "avg": round(avg, 3),
            "min": round(mn, 3)
        }

        print(f"  {name:40s}", end="")
        for s in yr_sharpes:
            print(f" {s:7.3f}", end="")
        print(f" {avg:7.3f} {mn:7.3f}")

    print()

    # ── Yearly Improvement Check ─────────────────────────────────────────
    print("--- YEARLY IMPROVEMENT: R46 Combined vs Baseline ---")
    base = all_results["Baseline (60/90, z=1.0/1.0)"]
    comb = all_results["R46 combined (45/75, z=1.0/0.75)"]

    all_improve = True
    print(f"  {'Year':>6s}  {'Base':>8s} {'R46':>8s} {'Δ':>8s}")
    for yr in YEARS:
        b = base["yearly"][str(yr)]
        c = comb["yearly"][str(yr)]
        d = c - b
        flag = "+" if d > 0 else "-"
        if d <= 0:
            all_improve = False
        print(f"  {yr:>6d}  {b:8.3f} {c:8.3f} {d:+8.3f} {flag}")

    print(f"\n  All years improve: {'YES' if all_improve else 'NO'}")
    improving_years = sum(1 for yr in YEARS if comb["yearly"][str(yr)] > base["yearly"][str(yr)])
    print(f"  Improving years: {improving_years}/{len(YEARS)}")
    print()

    # ── Check z_short-only component ─────────────────────────────────────
    print("--- YEARLY IMPROVEMENT: z_short-only (ETH 0.75) vs Baseline ---")
    z_only = all_results["R46 z_short (60/90, z=1.0/0.75)"]

    z_all_improve = True
    print(f"  {'Year':>6s}  {'Base':>8s} {'z=0.75':>8s} {'Δ':>8s}")
    for yr in YEARS:
        b = base["yearly"][str(yr)]
        z = z_only["yearly"][str(yr)]
        d = z - b
        flag = "+" if d > 0 else "-"
        if d <= 0:
            z_all_improve = False
        print(f"  {yr:>6d}  {b:8.3f} {z:8.3f} {d:+8.3f} {flag}")

    print(f"\n  z_short-only all years improve: {'YES' if z_all_improve else 'NO'}")
    print()

    # ── SUMMARY ──────────────────────────────────────────────────────────
    print("=" * 70)
    print("SUMMARY — R47: R46 Robustness Validation")
    print("=" * 70)
    print()

    for name, r in all_results.items():
        d = r["avg"] - base["avg"]
        print(f"  {name:40s}  avg={r['avg']:.3f} min={r['min']:.3f} Δ={d:+.3f}")

    print()
    print("=" * 70)

    comb_delta = comb["avg"] - base["avg"]
    if all_improve and comb_delta > 0.05:
        print("VERDICT: R46 per-asset params ARE ROBUST — all years improve")
        print("  RECOMMENDATION: Integrate into production signal generator")
    elif improving_years >= 4 and comb_delta > 0.05:
        print(f"VERDICT: R46 PARTIALLY ROBUST — {improving_years}/5 years improve")
        print("  RECOMMENDATION: Consider z_short-only component (less risk)")
    elif comb_delta < 0.03:
        print("VERDICT: R46 is NEGLIGIBLE — keep current params")
    else:
        print(f"VERDICT: R46 improvement is MARGINAL (Δ={comb_delta:+.3f})")
        print("  Not worth production complexity for marginal gain")
    print("=" * 70)

    # Save
    out = ROOT / "artifacts" / "crypto_options" / "r46_robustness_validation.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump({
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "research_id": "R47",
            "loyo_results": all_results,
        }, f, indent=2)
    print(f"\nResults saved to: {out}")


if __name__ == "__main__":
    main()
