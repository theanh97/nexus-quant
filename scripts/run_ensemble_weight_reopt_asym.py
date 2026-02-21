#!/usr/bin/env python3
"""
Ensemble Weight Re-optimization with Asymmetric Skew — R44
============================================================

R42+R43 changed the Skew MR profile (z_entry_short=1.0 instead of 2.0).
This makes Skew MR significantly stronger (+0.501).
Do the optimal ensemble weights (VRP/Skew) shift?

R29 found the surface was flat at 35-50% VRP (40/60 confirmed).
But that was pre-asymmetric. Repeat the sweep.

Also test: per-asset weights (BTC may need different VRP/Skew ratio than ETH).
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

# Per-asset VRP params (R35 validated)
BTC_VRP = {"base_leverage": 1.5, "exit_z_threshold": -3.0, "vrp_lookback": 30, "rebalance_freq": 5, "min_bars": 30}
ETH_VRP = {"base_leverage": 1.5, "exit_z_threshold": -3.0, "vrp_lookback": 60, "rebalance_freq": 5, "min_bars": 60}

# Asymmetric skew params (R42+R43 validated)
Z_ENTRY_LONG = 2.0
Z_ENTRY_SHORT = 1.0
Z_EXIT_LONG = 0.5
Z_EXIT_SHORT = 0.0


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


def run_ensemble(w_vrp, w_skew, btc_aw=0.50, eth_aw=0.50):
    """Run full multi-asset ensemble with specified VRP/Skew weights."""
    sharpes = []
    yearly = {}

    for yr in YEARS:
        cfg = {
            "symbols": ["BTC", "ETH"],
            "start": f"{yr}-01-01", "end": f"{yr}-12-31",
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
        asset_weights = {"BTC": btc_aw, "ETH": eth_aw}

        for idx in range(1, n):
            prev_equity = equity
            total_pnl = 0.0

            for sym in ["BTC", "ETH"]:
                aw = asset_weights[sym]
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

                total_pnl += aw * (w_vrp * vpnl + w_skew * spnl)

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

                # Asymmetric entry/exit (R42+R43)
                if current_pos == 0.0:
                    if z >= Z_ENTRY_SHORT:
                        new_pos = -per_sym_lev
                    elif z <= -Z_ENTRY_LONG:
                        new_pos = per_sym_lev
                else:
                    if current_pos > 0:
                        if z >= Z_EXIT_LONG:
                            new_pos = 0.0
                    else:
                        if z <= -Z_EXIT_SHORT:
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
        sharpes.append(m["sharpe"])
        yearly[str(yr)] = round(m["sharpe"], 3)

    avg = sum(sharpes) / len(sharpes)
    mn = min(sharpes)
    return {"avg_sharpe": round(avg, 3), "min_sharpe": round(mn, 3), "yearly": yearly}


def main():
    print("=" * 70)
    print("ENSEMBLE WEIGHT RE-OPTIMIZATION — R44 (Post-R42 Asymmetric Skew)")
    print("=" * 70)
    print()

    # ── A. VRP/Skew Weight Sweep (uniform asset weights 50/50) ──────────
    print("--- A. VRP/SKEW WEIGHT SWEEP (50/50 BTC/ETH) ---")
    print(f"  {'w_VRP':>7s} {'w_Skew':>7s}  {'avg':>7s} {'min':>7s} {'Δ':>7s}")
    print(f"  {'-----':>7s} {'------':>7s}  {'---':>7s} {'---':>7s} {'---':>7s}")

    weight_pairs = [
        (0.20, 0.80), (0.25, 0.75), (0.30, 0.70), (0.35, 0.65),
        (0.40, 0.60), (0.45, 0.55), (0.50, 0.50), (0.55, 0.45),
        (0.60, 0.40), (0.70, 0.30),
    ]

    weight_results = []
    baseline = None

    for w_v, w_s in weight_pairs:
        r = run_ensemble(w_v, w_s)
        if w_v == 0.40:
            baseline = r
        weight_results.append({"w_vrp": w_v, "w_skew": w_s, **r})
        marker = " *" if w_v == 0.40 else ""
        print(f"  {w_v:7.2f} {w_s:7.2f}  {r['avg_sharpe']:7.3f} {r['min_sharpe']:7.3f}{marker}")

    # Add deltas
    for wr in weight_results:
        wr["delta"] = round(wr["avg_sharpe"] - baseline["avg_sharpe"], 3)

    # Reprint with deltas
    print()
    print(f"  {'w_VRP':>7s} {'w_Skew':>7s}  {'avg':>7s} {'min':>7s} {'Δ':>7s}")
    print(f"  {'-----':>7s} {'------':>7s}  {'---':>7s} {'---':>7s} {'---':>7s}")
    for wr in weight_results:
        marker = " ← current" if wr["w_vrp"] == 0.40 else ""
        print(f"  {wr['w_vrp']:7.2f} {wr['w_skew']:7.2f}  {wr['avg_sharpe']:7.3f} {wr['min_sharpe']:7.3f} {wr['delta']:+7.3f}{marker}")

    # Find best
    best = max(weight_results, key=lambda x: x["avg_sharpe"])
    best_min = max(weight_results, key=lambda x: x["min_sharpe"])
    print(f"\n  Best avg: VRP={best['w_vrp']:.0%}/Skew={best['w_skew']:.0%} avg={best['avg_sharpe']:.3f}")
    print(f"  Best min: VRP={best_min['w_vrp']:.0%}/Skew={best_min['w_skew']:.0%} min={best_min['min_sharpe']:.3f}")
    print()

    # ── B. Per-Asset Weight Sweep ────────────────────────────────────────
    print("--- B. PER-ASSET WEIGHT (BTC/ETH allocation) ---")
    print(f"  {'BTC%':>5s} {'ETH%':>5s}  {'avg':>7s} {'min':>7s} {'Δ':>7s}")
    print(f"  {'----':>5s} {'----':>5s}  {'---':>7s} {'---':>7s} {'---':>7s}")

    asset_pairs = [
        (0.30, 0.70), (0.35, 0.65), (0.40, 0.60), (0.45, 0.55),
        (0.50, 0.50), (0.55, 0.45), (0.60, 0.40), (0.70, 0.30),
    ]

    asset_results = []
    asset_baseline = None

    for btc, eth in asset_pairs:
        r = run_ensemble(0.40, 0.60, btc_aw=btc, eth_aw=eth)
        if btc == 0.50:
            asset_baseline = r
        asset_results.append({"btc": btc, "eth": eth, **r})

    for ar in asset_results:
        ar["delta"] = round(ar["avg_sharpe"] - asset_baseline["avg_sharpe"], 3)

    for ar in asset_results:
        marker = " ← current" if ar["btc"] == 0.50 else ""
        print(f"  {ar['btc']:5.0%} {ar['eth']:5.0%}  {ar['avg_sharpe']:7.3f} {ar['min_sharpe']:7.3f} {ar['delta']:+7.3f}{marker}")

    best_asset = max(asset_results, key=lambda x: x["avg_sharpe"])
    print(f"\n  Best: BTC={best_asset['btc']:.0%}/ETH={best_asset['eth']:.0%} avg={best_asset['avg_sharpe']:.3f}")
    print()

    # ── C. Joint Optimization (Best VRP/Skew × Best Asset) ──────────────
    print("--- C. JOINT OPTIMIZATION ---")
    top_weights = sorted(weight_results, key=lambda x: x["avg_sharpe"], reverse=True)[:3]
    top_assets = sorted(asset_results, key=lambda x: x["avg_sharpe"], reverse=True)[:3]

    joint_results = []
    print(f"  {'w_VRP':>7s} {'w_Skew':>7s} {'BTC':>5s} {'ETH':>5s}  {'avg':>7s} {'min':>7s}")
    for tw in top_weights:
        for ta in top_assets:
            r = run_ensemble(tw["w_vrp"], tw["w_skew"], btc_aw=ta["btc"], eth_aw=ta["eth"])
            joint_results.append({
                "w_vrp": tw["w_vrp"], "w_skew": tw["w_skew"],
                "btc": ta["btc"], "eth": ta["eth"], **r
            })
            print(f"  {tw['w_vrp']:7.2f} {tw['w_skew']:7.2f} {ta['btc']:5.0%} {ta['eth']:5.0%}  {r['avg_sharpe']:7.3f} {r['min_sharpe']:7.3f}")

    best_joint = max(joint_results, key=lambda x: x["avg_sharpe"])
    print(f"\n  Best joint: VRP={best_joint['w_vrp']:.0%}/Skew={best_joint['w_skew']:.0%}, "
          f"BTC={best_joint['btc']:.0%}/ETH={best_joint['eth']:.0%} → "
          f"avg={best_joint['avg_sharpe']:.3f} min={best_joint['min_sharpe']:.3f}")
    print()

    # ── SUMMARY ──────────────────────────────────────────────────────────
    print("=" * 70)
    print("SUMMARY — R44: Ensemble Weight Re-optimization")
    print("=" * 70)
    print()

    current = baseline
    print(f"  Current 40/60 VRP/Skew:  avg={current['avg_sharpe']:.3f} min={current['min_sharpe']:.3f}")
    print(f"  Best weight only:        avg={best['avg_sharpe']:.3f} min={best['min_sharpe']:.3f} "
          f"(VRP={best['w_vrp']:.0%}/Skew={best['w_skew']:.0%}, Δ={best['delta']:+.3f})")
    print(f"  Best asset only:         avg={best_asset['avg_sharpe']:.3f} "
          f"(BTC={best_asset['btc']:.0%}/ETH={best_asset['eth']:.0%})")
    print(f"  Best joint:              avg={best_joint['avg_sharpe']:.3f} min={best_joint['min_sharpe']:.3f}")
    print()

    delta_best = best["avg_sharpe"] - current["avg_sharpe"]
    delta_joint = best_joint["avg_sharpe"] - current["avg_sharpe"]

    print("=" * 70)
    if abs(delta_best) < 0.05:
        print(f"VERDICT: Weight surface STILL FLAT — 40/60 remains near-optimal (Δ={delta_best:+.3f})")
        print("  R29 conclusion confirmed: post-R42 asymmetric skew does NOT shift optimal weights")
    elif delta_best > 0.05:
        print(f"VERDICT: Optimal weights SHIFTED to VRP={best['w_vrp']:.0%}/Skew={best['w_skew']:.0%}")
        print(f"  Improvement: Δ={delta_best:+.3f}")
    else:
        print(f"VERDICT: 40/60 may be SUBOPTIMAL — best at {best['w_vrp']:.0%}/{best['w_skew']:.0%}")
    print("=" * 70)

    # Save
    out = ROOT / "artifacts" / "crypto_options" / "ensemble_weight_reopt_asym.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump({
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "research_id": "R44",
            "weight_sweep": weight_results,
            "asset_sweep": asset_results,
            "joint_optimization": joint_results,
            "best_weight": {"w_vrp": best["w_vrp"], "w_skew": best["w_skew"],
                            "avg_sharpe": best["avg_sharpe"], "min_sharpe": best["min_sharpe"]},
            "best_joint": best_joint,
        }, f, indent=2)
    print(f"\nResults saved to: {out}")


if __name__ == "__main__":
    main()
