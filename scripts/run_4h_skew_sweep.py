#!/usr/bin/env python3
"""
4-Hourly Skew MR Sweep — Intermediate Frequency Test
=====================================================
Hourly Skew MR failed (all negative) — signal too noisy.
Daily Skew MR works (avg 1.744).
Test 4-hourly as middle ground: less noise than hourly, faster than daily.

bars_per_year = 2190 (365 * 6 bars per day)
lookback: 90-360 bars (15-60 days in 4h bars)
rebalance: 6-30 bars (1-5 days in 4h bars)

Also test if 4h VRP can match hourly VRP improvement.
"""
import json
import itertools
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nexus_quant.backtest.costs import ExecutionCostModel, FeeModel, ImpactModel
from nexus_quant.projects.crypto_options.options_engine import (
    OptionsBacktestEngine, compute_metrics, run_yearly_wf,
)
from nexus_quant.projects.crypto_options.strategies.variance_premium import VariancePremiumStrategy
from nexus_quant.projects.crypto_options.strategies.skew_trade_v2 import SkewTradeV2Strategy

_FEE = FeeModel(maker_fee_rate=0.0003, taker_fee_rate=0.0005)
_IMPACT = ImpactModel(model="sqrt", coef_bps=3.0)
COSTS = ExecutionCostModel(
    fee=_FEE, execution_style="taker",
    slippage_bps=7.5, spread_bps=10.0, impact=_IMPACT, cost_multiplier=1.0,
)
COSTS_CONSERVATIVE = ExecutionCostModel(
    fee=_FEE, execution_style="taker",
    slippage_bps=7.5, spread_bps=10.0, impact=_IMPACT, cost_multiplier=1.5,
)

YEARS = [2021, 2022, 2023, 2024, 2025]

FOUR_HOUR_CFG = {
    "provider": "deribit_rest_v1",
    "symbols": ["BTC", "ETH"],
    "bar_interval": "4h",
    "cache_dir": "data/cache/deribit",
    "use_synthetic_iv": True,
    "rv_lookback_bars": 126,   # 21 days × 6 bars/day
}


def run_4h_skew_sweep():
    """Sweep Skew MR at 4-hourly frequency."""
    print(f"{'='*70}")
    print(f"  4-HOURLY SKEW MR SWEEP (bars_per_year=2190)")
    print(f"{'='*70}\n")

    grid = {
        "skew_lookback": [90, 180, 360],      # 15d, 30d, 60d in 4h bars
        "z_entry": [1.5, 2.0, 2.5],
        "z_exit": [0.0, 0.3],
        "target_leverage": [1.0],
        "rebalance_freq": [6, 12, 30],        # 1d, 2d, 5d in 4h bars
    }

    keys = list(grid.keys())
    values = list(grid.values())
    combos = list(itertools.product(*values))
    total = len(combos)
    print(f"  Grid: {total} configurations")
    print(f"  {' × '.join(f'{k}={len(v)}' for k, v in grid.items())}\n")

    results = []
    best_avg = -999
    t0 = time.time()

    for i, combo in enumerate(combos):
        params = dict(zip(keys, combo))
        params["use_vrp_filter"] = False
        params["min_bars"] = params["skew_lookback"]

        try:
            wf = run_yearly_wf(
                provider_cfg=FOUR_HOUR_CFG,
                strategy_cls=SkewTradeV2Strategy,
                strategy_params=params,
                years=YEARS,
                costs=COSTS,
                use_options_pnl=True,
                bars_per_year=2190,
                seed=42,
                skew_sensitivity_mult=2.5,
            )

            avg = wf["summary"]["avg_sharpe"]
            mn = wf["summary"]["min_sharpe"]
            passed = wf["summary"]["passed"]
            yearly = {k: v.get("sharpe", 0) for k, v in wf["yearly"].items()}

            result = {
                "rank": 0,
                "freq": "4h",
                "strategy": "skew_mr",
                "params": params,
                "avg_sharpe": avg,
                "min_sharpe": mn,
                "passed": passed,
                "yearly": yearly,
            }
            results.append(result)

            flag = ""
            if avg > best_avg:
                best_avg = avg
                flag = " ★ NEW BEST"
                if passed:
                    flag += " ★★★"
            elif passed:
                flag = " ★ PASS"

            elapsed = time.time() - t0
            rate = (i + 1) / elapsed if elapsed > 0 else 1
            eta = (total - i - 1) / rate if rate > 0 else 0
            print(f"  [{i+1:3d}/{total}] lb={params['skew_lookback']:4d} ze={params['z_entry']:.1f} "
                  f"zx={params['z_exit']:.1f} rb={params['rebalance_freq']:3d} "
                  f"→ avg={avg:+.4f} min={mn:+.4f} (ETA {eta:.0f}s){flag}")

        except Exception as e:
            print(f"  [{i+1:3d}/{total}] ERROR: {e}")
            results.append({"params": params, "avg_sharpe": -999, "error": str(e)})

    results.sort(key=lambda r: r.get("avg_sharpe", -999), reverse=True)
    for i, r in enumerate(results):
        r["rank"] = i + 1

    return results


def run_4h_vrp_sweep():
    """Quick VRP test at 4-hourly frequency."""
    print(f"\n{'='*70}")
    print(f"  4-HOURLY VRP SWEEP (bars_per_year=2190)")
    print(f"{'='*70}\n")

    configs = [
        {"vrp_lookback": 126, "exit_z_threshold": -3.0, "base_leverage": 1.5, "rebalance_freq": 30, "min_bars": 126},
        {"vrp_lookback": 180, "exit_z_threshold": -3.0, "base_leverage": 1.5, "rebalance_freq": 42, "min_bars": 180},
        {"vrp_lookback": 126, "exit_z_threshold": -3.0, "base_leverage": 1.5, "rebalance_freq": 6, "min_bars": 126},
    ]

    results = []
    for i, params in enumerate(configs):
        wf = run_yearly_wf(
            provider_cfg=FOUR_HOUR_CFG,
            strategy_cls=VariancePremiumStrategy,
            strategy_params=params,
            years=YEARS,
            costs=COSTS,
            use_options_pnl=True,
            bars_per_year=2190,
            seed=42,
        )
        s = wf["summary"]
        yearly = {k: v.get("sharpe", 0) for k, v in wf["yearly"].items()}
        flag = " ★ PASS" if s["passed"] else ""
        days = params["rebalance_freq"] / 6
        print(f"  [{i+1}/{len(configs)}] lb={params['vrp_lookback']:4d} "
              f"rb={params['rebalance_freq']:3d} ({days:.1f}d) "
              f"→ avg={s['avg_sharpe']:+.4f} min={s['min_sharpe']:+.4f}{flag}")
        yearly_str = " ".join(f"{y}={v:+.2f}" for y, v in sorted(yearly.items()))
        print(f"    {yearly_str}")

        results.append({
            "params": params,
            "avg_sharpe": s["avg_sharpe"],
            "min_sharpe": s["min_sharpe"],
            "passed": s["passed"],
            "yearly": yearly,
        })

    return results


def main():
    t0 = time.time()

    # 1. 4h Skew MR sweep
    skew_results = run_4h_skew_sweep()

    # 2. 4h VRP sweep
    vrp_results = run_4h_vrp_sweep()

    # 3. Summary
    print(f"\n{'='*70}")
    print(f"  SUMMARY")
    print(f"{'='*70}")

    n_pass = sum(1 for r in skew_results if r.get("passed", False))
    n_pos = sum(1 for r in skew_results if r.get("avg_sharpe", -999) > 0)
    print(f"\n  4h Skew MR: {len(skew_results)} configs, {n_pos} positive, {n_pass} validated")

    if skew_results and skew_results[0].get("avg_sharpe", -999) > -999:
        best = skew_results[0]
        p = best["params"]
        print(f"  Best: avg={best['avg_sharpe']:+.4f} min={best['min_sharpe']:+.4f}")
        print(f"    lb={p['skew_lookback']} ze={p['z_entry']} zx={p['z_exit']} rb={p['rebalance_freq']}")
        yearly_str = " ".join(f"{y}={s:+.2f}" for y, s in sorted(best.get("yearly", {}).items()))
        print(f"    {yearly_str}")

    print(f"\n  COMPARISON:")
    print(f"    Daily Skew MR:   avg=+1.7440 (champion)")
    print(f"    Hourly Skew MR:  avg=-0.6454 (FAILED)")
    if skew_results:
        print(f"    4h Skew MR:      avg={skew_results[0]['avg_sharpe']:+.4f}")

    print(f"\n    Daily VRP:       avg=+2.0881")
    print(f"    Hourly VRP:      avg=+3.6487")
    if vrp_results:
        best_vrp = max(vrp_results, key=lambda r: r["avg_sharpe"])
        print(f"    4h VRP:          avg={best_vrp['avg_sharpe']:+.4f}")

    # Conservative for top Skew config
    if skew_results and skew_results[0].get("avg_sharpe", -999) > 0.5:
        print(f"\n  Conservative costs for best 4h Skew MR:")
        best = skew_results[0]
        wf = run_yearly_wf(
            provider_cfg=FOUR_HOUR_CFG,
            strategy_cls=SkewTradeV2Strategy,
            strategy_params=best["params"],
            years=YEARS,
            costs=COSTS_CONSERVATIVE,
            use_options_pnl=True,
            bars_per_year=2190,
            seed=42,
            skew_sensitivity_mult=2.5,
        )
        s = wf["summary"]
        print(f"    Conservative: avg={s['avg_sharpe']:+.4f} min={s['min_sharpe']:+.4f}")

    # Save
    os.makedirs("artifacts/crypto_options", exist_ok=True)
    output = {
        "date": time.strftime("%Y-%m-%d %H:%M"),
        "skew_4h_top_20": skew_results[:20],
        "vrp_4h": vrp_results,
        "elapsed_sec": round(time.time() - t0, 1),
    }
    out_path = "artifacts/crypto_options/4h_freq_sweep_results.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\n  Results saved to {out_path}")
    print(f"  Total elapsed: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
