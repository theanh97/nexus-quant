#!/usr/bin/env python3
"""
Per-Asset Skew Lookback Optimization — R46
=============================================

R35 showed ETH VRP benefits from longer lookback (60d vs 30d, +0.182).
Does the same pattern hold for Skew MR?

Current: BTC Skew lookback = 60d, ETH Skew lookback = 90d.
Test: sweep 30-180d for each asset independently.

Also test: per-asset asymmetric z-score thresholds.
Maybe ETH needs different z_entry_short than BTC?
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


def run_with_skew_lookbacks(btc_lb, eth_lb, z_entry_short_btc=1.0, z_entry_short_eth=1.0):
    """Run ensemble with per-asset skew lookbacks and asymmetric z-scores."""
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

        skew_lbs = {"BTC": btc_lb, "ETH": eth_lb}
        z_shorts = {"BTC": z_entry_short_btc, "ETH": z_entry_short_eth}

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
        sharpes.append(m["sharpe"])
        yearly[str(yr)] = round(m["sharpe"], 3)

    avg = sum(sharpes) / len(sharpes)
    mn = min(sharpes)
    return {"avg_sharpe": round(avg, 3), "min_sharpe": round(mn, 3), "yearly": yearly}


def main():
    print("=" * 70)
    print("PER-ASSET SKEW LOOKBACK OPTIMIZATION — R46")
    print("=" * 70)
    print()

    baseline = run_with_skew_lookbacks(60, 90)
    print(f"  Baseline (BTC=60d, ETH=90d): avg={baseline['avg_sharpe']:.3f} min={baseline['min_sharpe']:.3f}")
    print(f"  Yearly: {baseline['yearly']}")
    print()

    # ── A. BTC Skew Lookback Sweep ───────────────────────────────────────
    print("--- A. BTC SKEW LOOKBACK SWEEP (ETH fixed at 90d) ---")
    print(f"  {'BTC_lb':>7s}  {'avg':>7s} {'min':>7s} {'Δ':>7s}")

    btc_lbs = [30, 45, 60, 75, 90, 120, 150, 180]
    btc_results = []
    for lb in btc_lbs:
        r = run_with_skew_lookbacks(lb, 90)
        d = r["avg_sharpe"] - baseline["avg_sharpe"]
        btc_results.append({"btc_lb": lb, **r, "delta": round(d, 3)})
        marker = " ← current" if lb == 60 else ""
        print(f"  {lb:7d}  {r['avg_sharpe']:7.3f} {r['min_sharpe']:7.3f} {d:+7.3f}{marker}")

    best_btc = max(btc_results, key=lambda x: x["avg_sharpe"])
    print(f"  Best BTC lookback: {best_btc['btc_lb']}d (avg={best_btc['avg_sharpe']:.3f})")
    print()

    # ── B. ETH Skew Lookback Sweep ───────────────────────────────────────
    print("--- B. ETH SKEW LOOKBACK SWEEP (BTC fixed at 60d) ---")
    print(f"  {'ETH_lb':>7s}  {'avg':>7s} {'min':>7s} {'Δ':>7s}")

    eth_lbs = [30, 45, 60, 75, 90, 120, 150, 180]
    eth_results = []
    for lb in eth_lbs:
        r = run_with_skew_lookbacks(60, lb)
        d = r["avg_sharpe"] - baseline["avg_sharpe"]
        eth_results.append({"eth_lb": lb, **r, "delta": round(d, 3)})
        marker = " ← current" if lb == 90 else ""
        print(f"  {lb:7d}  {r['avg_sharpe']:7.3f} {r['min_sharpe']:7.3f} {d:+7.3f}{marker}")

    best_eth = max(eth_results, key=lambda x: x["avg_sharpe"])
    print(f"  Best ETH lookback: {best_eth['eth_lb']}d (avg={best_eth['avg_sharpe']:.3f})")
    print()

    # ── C. Joint Best Lookback ───────────────────────────────────────────
    print("--- C. JOINT BEST LOOKBACK ---")
    joint = run_with_skew_lookbacks(best_btc["btc_lb"], best_eth["eth_lb"])
    d = joint["avg_sharpe"] - baseline["avg_sharpe"]
    print(f"  BTC={best_btc['btc_lb']}d + ETH={best_eth['eth_lb']}d: "
          f"avg={joint['avg_sharpe']:.3f} min={joint['min_sharpe']:.3f} Δ={d:+.3f}")
    print(f"  Yearly: {joint['yearly']}")
    print()

    # ── D. Per-Asset z_entry_short Sweep ─────────────────────────────────
    print("--- D. PER-ASSET z_entry_short SWEEP ---")
    print("  (Does ETH need different z_entry_short than BTC?)")
    print(f"  {'BTC_z':>6s} {'ETH_z':>6s}  {'avg':>7s} {'min':>7s} {'Δ':>7s}")

    z_vals = [0.5, 0.75, 1.0, 1.25, 1.5]
    z_results = []
    for btc_z in z_vals:
        for eth_z in z_vals:
            r = run_with_skew_lookbacks(60, 90,
                                         z_entry_short_btc=btc_z,
                                         z_entry_short_eth=eth_z)
            d = r["avg_sharpe"] - baseline["avg_sharpe"]
            z_results.append({"btc_z": btc_z, "eth_z": eth_z, **r, "delta": round(d, 3)})

    # Print top 10
    z_results.sort(key=lambda x: x["avg_sharpe"], reverse=True)
    for zr in z_results[:10]:
        marker = " ← current" if zr["btc_z"] == 1.0 and zr["eth_z"] == 1.0 else ""
        print(f"  {zr['btc_z']:6.2f} {zr['eth_z']:6.2f}  {zr['avg_sharpe']:7.3f} {zr['min_sharpe']:7.3f} {zr['delta']:+7.3f}{marker}")

    best_z = z_results[0]
    print(f"\n  Best: BTC z_short={best_z['btc_z']}, ETH z_short={best_z['eth_z']} "
          f"(avg={best_z['avg_sharpe']:.3f})")
    print()

    # ── E. Combined Best: lookback + z_entry_short ───────────────────────
    print("--- E. COMBINED BEST: lookback + z_entry_short ---")
    combined = run_with_skew_lookbacks(
        best_btc["btc_lb"], best_eth["eth_lb"],
        z_entry_short_btc=best_z["btc_z"],
        z_entry_short_eth=best_z["eth_z"],
    )
    d = combined["avg_sharpe"] - baseline["avg_sharpe"]
    print(f"  BTC lb={best_btc['btc_lb']} z_short={best_z['btc_z']}")
    print(f"  ETH lb={best_eth['eth_lb']} z_short={best_z['eth_z']}")
    print(f"  Combined: avg={combined['avg_sharpe']:.3f} min={combined['min_sharpe']:.3f} Δ={d:+.3f}")
    print(f"  Yearly: {combined['yearly']}")
    print()

    # ── SUMMARY ──────────────────────────────────────────────────────────
    print("=" * 70)
    print("SUMMARY — R46: Per-Asset Skew Lookback Optimization")
    print("=" * 70)
    print()

    print(f"  Baseline (BTC=60d/ETH=90d, both z_short=1.0): avg={baseline['avg_sharpe']:.3f}")
    print(f"  Best BTC lookback: {best_btc['btc_lb']}d (Δ={best_btc['delta']:+.3f})")
    print(f"  Best ETH lookback: {best_eth['eth_lb']}d (Δ={best_eth['delta']:+.3f})")
    print(f"  Best per-asset z_short: BTC={best_z['btc_z']}/ETH={best_z['eth_z']} (Δ={best_z['delta']:+.3f})")
    print(f"  Combined best: avg={combined['avg_sharpe']:.3f} (Δ={d:+.3f})")
    print()

    lb_shift = abs(best_btc["delta"]) + abs(best_eth["delta"])
    z_shift = best_z["delta"]

    print("=" * 70)
    if d > 0.10:
        print(f"VERDICT: Per-asset optimization IMPROVES ensemble by Δ={d:+.3f}")
    elif d > 0.03:
        print(f"VERDICT: Per-asset optimization gives MARGINAL improvement (Δ={d:+.3f})")
    else:
        print(f"VERDICT: Per-asset optimization is NEGLIGIBLE (Δ={d:+.3f})")
        print("  Current params (BTC=60d/ETH=90d, z_short=1.0) are near-optimal")
    print("=" * 70)

    # Save
    out = ROOT / "artifacts" / "crypto_options" / "per_asset_skew_lookback.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump({
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "research_id": "R46",
            "baseline": baseline,
            "btc_lookback_sweep": btc_results,
            "eth_lookback_sweep": eth_results,
            "joint_lookback": {
                "btc_lb": best_btc["btc_lb"], "eth_lb": best_eth["eth_lb"],
                **joint
            },
            "z_entry_short_grid": z_results[:15],
            "combined_best": {
                "btc_lb": best_btc["btc_lb"], "eth_lb": best_eth["eth_lb"],
                "btc_z_short": best_z["btc_z"], "eth_z_short": best_z["eth_z"],
                **combined,
            },
        }, f, indent=2)
    print(f"\nResults saved to: {out}")


if __name__ == "__main__":
    main()
