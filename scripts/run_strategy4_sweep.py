#!/usr/bin/env python3
"""
Strategy #4 Candidate Sweep — Crypto Options
=============================================
Tests 3 new strategy candidates to potentially add to the VRP+Skew ensemble:

1. Butterfly MR — Mean-reversion on vol smile convexity (butterfly_25d)
2. IV Level MR — Mean-reversion on absolute IV level (iv_atm)
3. Put-Call Ratio MR — Mean-reversion on put-call ratio

Each candidate uses the FeatureMRStrategy class with different feature_key.
Engine routes to vega/feature-change P&L model via name-based routing.

Success criteria: avg Sharpe >= 1.0, min Sharpe >= 0.5 (same as previous R&D)
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

# ── Configuration ────────────────────────────────────────────────────────

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

# Realistic cost model (from configs)
COSTS = ExecutionCostModel(
    fee=_FEE,
    execution_style="taker",
    slippage_bps=7.5,
    spread_bps=10.0,
    impact=_IMPACT,
    cost_multiplier=1.0,
)

# Conservative cost model (1.5x)
COSTS_CONSERVATIVE = ExecutionCostModel(
    fee=_FEE,
    execution_style="taker",
    slippage_bps=7.5,
    spread_bps=10.0,
    impact=_IMPACT,
    cost_multiplier=1.5,
)

# ── Strategy Candidates ────────────────────────────────────────────────

# Each candidate: (name_prefix, feature_key, param_grid, sensitivity_mults)
CANDIDATES = [
    {
        "name": "butterfly_mr",
        "feature_key": "butterfly_25d",
        "description": "Butterfly MR — vol smile convexity mean-reversion",
        "param_grid": {
            "lookback": [30, 60, 90],
            "z_entry": [1.5, 2.0, 2.5],
            "z_exit": [0.0, 0.3],
            "target_leverage": [1.0],
            "rebalance_freq": [5],
        },
        "sensitivity_mults": [1.0, 1.5, 2.0, 2.5, 3.0],
    },
    {
        "name": "iv_mr",
        "feature_key": "iv_atm",
        "description": "IV Level MR — absolute IV mean-reversion",
        "param_grid": {
            "lookback": [30, 60, 90],
            "z_entry": [1.5, 2.0, 2.5],
            "z_exit": [0.0, 0.3],
            "target_leverage": [1.0],
            "rebalance_freq": [5],
        },
        "sensitivity_mults": [1.0, 1.5, 2.0, 2.5],
    },
    {
        "name": "pcr_mr",
        "feature_key": "put_call_ratio",
        "description": "Put-Call Ratio MR — sentiment mean-reversion",
        "param_grid": {
            "lookback": [30, 60, 90],
            "z_entry": [1.5, 2.0, 2.5],
            "z_exit": [0.0, 0.3],
            "target_leverage": [1.0],
            "rebalance_freq": [5],
        },
        "sensitivity_mults": [1.0, 1.5, 2.0, 2.5],
    },
]


def run_candidate_sweep(candidate):
    """Run full parameter + sensitivity sweep for a single candidate."""
    name = candidate["name"]
    feature_key = candidate["feature_key"]
    grid = candidate["param_grid"]
    sens_mults = candidate["sensitivity_mults"]

    print(f"\n{'='*70}")
    print(f"  CANDIDATE: {candidate['description']}")
    print(f"  Feature: {feature_key}")
    print(f"{'='*70}\n")

    # Generate all param combos
    keys = list(grid.keys())
    values = list(grid.values())
    combos = list(itertools.product(*values))
    total = len(combos) * len(sens_mults)
    print(f"  Grid: {len(combos)} param combos × {len(sens_mults)} sensitivity = {total} total configs")

    results = []
    best_avg = -999
    best_config = None

    for ci, combo in enumerate(combos):
        params = dict(zip(keys, combo))
        params["feature_key"] = feature_key
        params["min_bars"] = params["lookback"]  # need enough history

        for si, sens in enumerate(sens_mults):
            idx = ci * len(sens_mults) + si + 1
            strategy_name = f"crypto_{name}"

            try:
                wf = run_yearly_wf(
                    provider_cfg=PROVIDER_CFG,
                    strategy_cls=FeatureMRStrategy,
                    strategy_params={**params, "name": strategy_name},
                    years=YEARS,
                    costs=COSTS,
                    use_options_pnl=True,
                    bars_per_year=365,
                    seed=42,
                    skew_sensitivity_mult=sens,
                )

                avg = wf["summary"]["avg_sharpe"]
                mn = wf["summary"]["min_sharpe"]
                passed = wf["summary"]["passed"]

                result = {
                    "rank": 0,
                    "candidate": name,
                    "feature_key": feature_key,
                    "params": params,
                    "sensitivity_mult": sens,
                    "avg_sharpe": avg,
                    "min_sharpe": mn,
                    "passed": passed,
                    "yearly": {k: v.get("sharpe", 0) for k, v in wf["yearly"].items()},
                }
                results.append(result)

                flag = " ★ PASS" if passed else ""
                if avg > best_avg:
                    best_avg = avg
                    best_config = result
                    flag += " ★ NEW BEST"

                print(f"  [{idx:3d}/{total}] sens={sens:.1f} lb={params['lookback']} "
                      f"ze={params['z_entry']:.1f} zx={params['z_exit']:.1f} "
                      f"→ avg={avg:+.4f} min={mn:+.4f}{flag}")

            except Exception as e:
                print(f"  [{idx:3d}/{total}] ERROR: {e}")
                results.append({
                    "candidate": name,
                    "params": params,
                    "sensitivity_mult": sens,
                    "avg_sharpe": -999,
                    "error": str(e),
                })

    # Sort and rank
    results.sort(key=lambda r: r.get("avg_sharpe", -999), reverse=True)
    for i, r in enumerate(results):
        r["rank"] = i + 1

    return results, best_config


def compute_correlations(best_configs):
    """Compute return correlations between new candidates and existing strategies."""
    from nexus_quant.projects.crypto_options.providers.deribit_rest import DeribitRestProvider
    from nexus_quant.projects.crypto_options.options_engine import OptionsBacktestEngine, compute_metrics
    from nexus_quant.projects.crypto_options.strategies.variance_premium import VariancePremiumStrategy
    from nexus_quant.projects.crypto_options.strategies.skew_trade_v2 import SkewTradeV2Strategy

    print(f"\n{'='*70}")
    print(f"  CORRELATION ANALYSIS")
    print(f"{'='*70}\n")

    # Full 5-year dataset
    cfg = dict(PROVIDER_CFG)
    cfg["start"] = "2020-01-01"
    cfg["end"] = "2025-12-31"
    provider = DeribitRestProvider(cfg, seed=42)
    dataset = provider.load()

    # VRP returns
    vrp_engine = OptionsBacktestEngine(costs=COSTS, bars_per_year=365, use_options_pnl=True)
    vrp_strat = VariancePremiumStrategy(params={
        "base_leverage": 1.5, "exit_z_threshold": -3.0,
        "vrp_lookback": 30, "rebalance_freq": 5, "min_bars": 30,
    })
    vrp_result = vrp_engine.run(dataset, vrp_strat)

    # Skew v2 returns
    skew_engine = OptionsBacktestEngine(
        costs=COSTS, bars_per_year=365, use_options_pnl=True,
        skew_sensitivity_mult=2.5,
    )
    skew_strat = SkewTradeV2Strategy(params={
        "skew_lookback": 60, "z_entry": 2.0, "z_exit": 0.0,
        "target_leverage": 1.0, "rebalance_freq": 5, "min_bars": 60,
        "use_vrp_filter": False,
    })
    skew_result = skew_engine.run(dataset, skew_strat)

    all_returns = {
        "VRP": vrp_result.returns,
        "Skew_v2": skew_result.returns,
    }

    # New candidate returns
    for name, config in best_configs.items():
        if config is None:
            continue
        sens = config["sensitivity_mult"]
        params = config["params"]
        strategy_name = f"crypto_{name}"

        engine = OptionsBacktestEngine(
            costs=COSTS, bars_per_year=365, use_options_pnl=True,
            skew_sensitivity_mult=sens,
        )
        strat = FeatureMRStrategy(name=strategy_name, params=params)
        result = engine.run(dataset, strat)
        all_returns[name] = result.returns

    # Compute pairwise correlations
    names = list(all_returns.keys())
    corr_matrix = {}
    for i, n1 in enumerate(names):
        for j, n2 in enumerate(names):
            if i >= j:
                continue
            r1 = all_returns[n1]
            r2 = all_returns[n2]
            min_len = min(len(r1), len(r2))
            r1 = r1[:min_len]
            r2 = r2[:min_len]

            if min_len < 10:
                corr_matrix[f"{n1}_vs_{n2}"] = None
                continue

            mean1 = sum(r1) / len(r1)
            mean2 = sum(r2) / len(r2)
            cov = sum((a - mean1) * (b - mean2) for a, b in zip(r1, r2)) / (len(r1) - 1)
            var1 = sum((a - mean1) ** 2 for a in r1) / (len(r1) - 1)
            var2 = sum((b - mean2) ** 2 for b in r2) / (len(r2) - 1)
            std1 = var1 ** 0.5
            std2 = var2 ** 0.5
            corr = cov / (std1 * std2) if std1 > 0 and std2 > 0 else 0.0
            corr_matrix[f"{n1}_vs_{n2}"] = round(corr, 4)
            print(f"  {n1} vs {n2}: r = {corr:+.4f}")

    return corr_matrix


def main():
    t0 = time.time()
    all_results = {}
    best_configs = {}

    # ── Run each candidate sweep ──
    for candidate in CANDIDATES:
        results, best = run_candidate_sweep(candidate)
        all_results[candidate["name"]] = results
        best_configs[candidate["name"]] = best

    # ── Summary ──
    print(f"\n{'='*70}")
    print(f"  SWEEP SUMMARY — Strategy #4 Candidates")
    print(f"{'='*70}\n")

    any_validated = False
    for name, best in best_configs.items():
        if best is None:
            print(f"  {name}: NO RESULTS")
            continue
        avg = best["avg_sharpe"]
        mn = best["min_sharpe"]
        passed = best.get("passed", False)
        sens = best["sensitivity_mult"]
        params = best["params"]
        flag = "✓ VALIDATED" if passed else "✗ NOT VALIDATED"
        print(f"  {name}: avg={avg:+.4f} min={mn:+.4f} sens={sens:.1f} [{flag}]")
        print(f"    params: lb={params['lookback']} ze={params['z_entry']} zx={params['z_exit']}")
        if "yearly" in best:
            yearly_str = " ".join(f"{y}={s:+.3f}" for y, s in sorted(best["yearly"].items()))
            print(f"    yearly: {yearly_str}")
        if passed:
            any_validated = True

    # ── Correlation analysis (only for validated or top candidates) ──
    if any_validated or any(b and b.get("avg_sharpe", -999) > 0.3 for b in best_configs.values()):
        corr_matrix = compute_correlations(best_configs)
    else:
        corr_matrix = {}

    # ── Conservative costs for validated candidates ──
    conservative_results = {}
    for name, best in best_configs.items():
        if best is None or not best.get("passed", False):
            continue
        print(f"\n  Re-running {name} with conservative costs (1.5x)...")
        params = best["params"]
        sens = best["sensitivity_mult"]
        strategy_name = f"crypto_{name}"

        wf = run_yearly_wf(
            provider_cfg=PROVIDER_CFG,
            strategy_cls=FeatureMRStrategy,
            strategy_params={**params, "name": strategy_name},
            years=YEARS,
            costs=COSTS_CONSERVATIVE,
            use_options_pnl=True,
            bars_per_year=365,
            seed=42,
            skew_sensitivity_mult=sens,
        )
        conservative_results[name] = wf["summary"]
        print(f"    Conservative: avg={wf['summary']['avg_sharpe']:.4f} "
              f"min={wf['summary']['min_sharpe']:.4f}")

    # ── Save results ──
    os.makedirs("artifacts/crypto_options", exist_ok=True)
    output = {
        "sweep_date": time.strftime("%Y-%m-%d %H:%M"),
        "candidates": {name: {"results_count": len(r), "top_5": r[:5]}
                       for name, r in all_results.items()},
        "best_configs": {name: best for name, best in best_configs.items() if best is not None},
        "correlations": corr_matrix,
        "conservative": conservative_results,
        "elapsed_sec": round(time.time() - t0, 1),
    }

    out_path = "artifacts/crypto_options/strategy4_sweep_results.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\n  Results saved to {out_path}")
    print(f"  Total elapsed: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
