#!/usr/bin/env python3
"""
Butterfly MR Deep Sweep — Crypto Options Strategy #4
=====================================================
The initial sweep showed butterfly MR as the strongest Strategy #4 candidate
(avg=0.81, best yearly 2.28). This deep sweep explores:

1. Wider parameter grid (lower z_entry, varied leverage)
2. Alternative lookbacks (45, 120, 150)
3. Different rebalance frequencies (3, 7, 10)
4. Tighter z_exit values
5. VRP confirmation filter variant

Target: push avg Sharpe above 1.0 with min > 0.5 for validation.
"""
import json
import itertools
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nexus_quant.backtest.costs import ExecutionCostModel, FeeModel, ImpactModel
from nexus_quant.projects.crypto_options.options_engine import run_yearly_wf
from nexus_quant.projects.crypto_options.strategies.feature_mr import FeatureMRStrategy

PROVIDER_CFG = {
    "provider": "deribit_rest_v1",
    "symbols": ["BTC", "ETH"],
    "bar_interval": "1d",
    "cache_dir": "data/cache/deribit",
    "use_synthetic_iv": True,
    "rv_lookback_bars": 21,
}

YEARS = [2021, 2022, 2023, 2024, 2025]

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


def run_sweep():
    """Deep butterfly parameter sweep."""
    grid = {
        "lookback": [30, 45, 60, 90, 120, 150],
        "z_entry": [0.8, 1.0, 1.2, 1.5, 2.0],
        "z_exit": [0.0, 0.2, 0.3, 0.5],
        "target_leverage": [0.5, 1.0, 1.5],
        "rebalance_freq": [3, 5, 7, 10],
    }

    keys = list(grid.keys())
    values = list(grid.values())
    combos = list(itertools.product(*values))
    total = len(combos)
    print(f"Butterfly Deep Sweep: {total} configurations")
    print(f"Grid: {' × '.join(f'{k}={len(v)}' for k, v in grid.items())}")
    print()

    results = []
    best_avg = -999
    best = None
    t0 = time.time()

    for i, combo in enumerate(combos):
        params = dict(zip(keys, combo))
        params["feature_key"] = "butterfly_25d"
        params["min_bars"] = max(params["lookback"], 60)

        try:
            wf = run_yearly_wf(
                provider_cfg=PROVIDER_CFG,
                strategy_cls=FeatureMRStrategy,
                strategy_params={**params, "name": "crypto_butterfly_mr"},
                years=YEARS,
                costs=COSTS,
                use_options_pnl=True,
                bars_per_year=365,
                seed=42,
                skew_sensitivity_mult=1.0,
            )

            avg = wf["summary"]["avg_sharpe"]
            mn = wf["summary"]["min_sharpe"]
            passed = wf["summary"]["passed"]
            yearly = {k: v.get("sharpe", 0) for k, v in wf["yearly"].items()}

            result = {
                "rank": 0,
                "params": params,
                "avg_sharpe": avg,
                "min_sharpe": mn,
                "passed": passed,
                "yearly": yearly,
            }
            results.append(result)

            if avg > best_avg:
                best_avg = avg
                best = result
                flag = " ★ NEW BEST"
                if passed:
                    flag += " ★★★ VALIDATED"
            elif passed:
                flag = " ★ PASS"
            else:
                flag = ""

            if (i + 1) % 50 == 0 or flag:
                elapsed = time.time() - t0
                rate = (i + 1) / elapsed
                eta = (total - i - 1) / rate if rate > 0 else 0
                print(f"  [{i+1:4d}/{total}] lb={params['lookback']:3d} ze={params['z_entry']:.1f} "
                      f"zx={params['z_exit']:.1f} lev={params['target_leverage']:.1f} "
                      f"rb={params['rebalance_freq']:2d} → avg={avg:+.4f} min={mn:+.4f}"
                      f" (ETA {eta:.0f}s){flag}")

        except Exception as e:
            results.append({"params": params, "avg_sharpe": -999, "error": str(e)})

    # Sort and rank
    results.sort(key=lambda r: r.get("avg_sharpe", -999), reverse=True)
    for i, r in enumerate(results):
        r["rank"] = i + 1

    return results, best


def main():
    t0 = time.time()
    results, best = run_sweep()

    n_pass = sum(1 for r in results if r.get("passed", False))
    n_positive = sum(1 for r in results if r.get("avg_sharpe", -999) > 0)

    print(f"\n{'='*70}")
    print(f"  DEEP BUTTERFLY SWEEP RESULTS")
    print(f"{'='*70}")
    print(f"  Total configs: {len(results)}")
    print(f"  Positive avg Sharpe: {n_positive}")
    print(f"  Validated (avg>=1.0, min>=0.5): {n_pass}")
    print()

    # Top 20
    print("  TOP 20:")
    for r in results[:20]:
        p = r["params"]
        flag = " ★ PASS" if r.get("passed") else ""
        yearly = r.get("yearly", {})
        yearly_str = " ".join(f"{y}={s:+.2f}" for y, s in sorted(yearly.items()))
        print(f"    #{r['rank']:3d} avg={r['avg_sharpe']:+.4f} min={r['min_sharpe']:+.4f} "
              f"lb={p['lookback']:3d} ze={p['z_entry']:.1f} zx={p['z_exit']:.1f} "
              f"lev={p['target_leverage']:.1f} rb={p['rebalance_freq']:2d}{flag}")
        print(f"         {yearly_str}")

    # Bottom 5
    print("\n  BOTTOM 5:")
    for r in results[-5:]:
        p = r["params"]
        print(f"    #{r['rank']:3d} avg={r['avg_sharpe']:+.4f} min={r['min_sharpe']:+.4f} "
              f"lb={p['lookback']:3d} ze={p['z_entry']:.1f} zx={p['z_exit']:.1f}")

    # Conservative costs for best
    if best and best.get("avg_sharpe", -999) > 0.5:
        print(f"\n  Conservative costs for best config...")
        params = best["params"]
        wf_cons = run_yearly_wf(
            provider_cfg=PROVIDER_CFG,
            strategy_cls=FeatureMRStrategy,
            strategy_params={**params, "name": "crypto_butterfly_mr"},
            years=YEARS,
            costs=COSTS_CONSERVATIVE,
            use_options_pnl=True,
            bars_per_year=365,
            seed=42,
            skew_sensitivity_mult=1.0,
        )
        print(f"    Conservative: avg={wf_cons['summary']['avg_sharpe']:.4f} "
              f"min={wf_cons['summary']['min_sharpe']:.4f}")

    # Save results
    os.makedirs("artifacts/crypto_options", exist_ok=True)
    output = {
        "sweep_date": time.strftime("%Y-%m-%d %H:%M"),
        "total_configs": len(results),
        "validated": n_pass,
        "positive_sharpe": n_positive,
        "best": best,
        "top_20": results[:20],
        "elapsed_sec": round(time.time() - t0, 1),
    }
    out_path = "artifacts/crypto_options/butterfly_deep_sweep_results.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\n  Results saved to {out_path}")
    print(f"  Total elapsed: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
