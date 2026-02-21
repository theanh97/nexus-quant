#!/usr/bin/env python3
"""
Butterfly Ensemble Integration Test
====================================
Tests the validated Butterfly MR strategy (#13 from deep sweep)
as Strategy #4 in the crypto options ensemble:

1. Conservative cost validation for top 3 configs
2. Correlation analysis with VRP and Skew MR
3. 3-way ensemble optimization (VRP + Skew + Butterfly)
4. Compare 2-way vs 3-way ensemble performance
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
from nexus_quant.projects.crypto_options.providers.deribit_rest import DeribitRestProvider
from nexus_quant.projects.crypto_options.strategies.variance_premium import VariancePremiumStrategy
from nexus_quant.projects.crypto_options.strategies.skew_trade_v2 import SkewTradeV2Strategy
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

# ── Top butterfly configs from deep sweep ──
BUTTERFLY_CONFIGS = {
    "fast_regime": {
        "name": "crypto_butterfly_mr",
        "feature_key": "butterfly_25d",
        "lookback": 45, "z_entry": 0.8, "z_exit": 0.3,
        "target_leverage": 0.5, "rebalance_freq": 10, "min_bars": 60,
        "_avg_sharpe": 1.1289, "_min_sharpe": 0.5340,
    },
    "slow_regime": {
        "name": "crypto_butterfly_mr",
        "feature_key": "butterfly_25d",
        "lookback": 120, "z_entry": 2.0, "z_exit": 0.3,
        "target_leverage": 0.5, "rebalance_freq": 3, "min_bars": 120,
        "_avg_sharpe": 1.0824, "_min_sharpe": 0.8120,
    },
    "slow_regime_wide_exit": {
        "name": "crypto_butterfly_mr",
        "feature_key": "butterfly_25d",
        "lookback": 120, "z_entry": 2.0, "z_exit": 0.5,
        "target_leverage": 0.5, "rebalance_freq": 3, "min_bars": 120,
        "_avg_sharpe": 1.1089, "_min_sharpe": 0.5384,
    },
}


def run_conservative_validation():
    """Run conservative cost validation for top butterfly configs."""
    print(f"{'='*70}")
    print(f"  CONSERVATIVE COST VALIDATION (1.5x)")
    print(f"{'='*70}\n")

    results = {}
    for label, params in BUTTERFLY_CONFIGS.items():
        wf = run_yearly_wf(
            provider_cfg=PROVIDER_CFG,
            strategy_cls=FeatureMRStrategy,
            strategy_params=params,
            years=YEARS,
            costs=COSTS_CONSERVATIVE,
            use_options_pnl=True,
            bars_per_year=365,
            seed=42,
            skew_sensitivity_mult=1.0,
        )
        s = wf["summary"]
        yearly = {k: v.get("sharpe", 0) for k, v in wf["yearly"].items()}
        results[label] = {"summary": s, "yearly": yearly}
        flag = "PASS" if s["passed"] else "FAIL"
        print(f"  {label:25s}: avg={s['avg_sharpe']:+.4f} min={s['min_sharpe']:+.4f} [{flag}]")
        yearly_str = " ".join(f"{y}={v:+.2f}" for y, v in sorted(yearly.items()))
        print(f"  {'':25s}  {yearly_str}")
    return results


def compute_full_correlations():
    """Compute return correlations between all strategies (full 6-year history)."""
    print(f"\n{'='*70}")
    print(f"  CORRELATION ANALYSIS (Full 2020-2025)")
    print(f"{'='*70}\n")

    cfg = dict(PROVIDER_CFG)
    cfg["start"] = "2020-01-01"
    cfg["end"] = "2025-12-31"
    provider = DeribitRestProvider(cfg, seed=42)
    dataset = provider.load()

    # VRP
    vrp_engine = OptionsBacktestEngine(costs=COSTS, bars_per_year=365, use_options_pnl=True)
    vrp_strat = VariancePremiumStrategy(params={
        "base_leverage": 1.5, "exit_z_threshold": -3.0,
        "vrp_lookback": 30, "rebalance_freq": 5, "min_bars": 30,
    })
    vrp_result = vrp_engine.run(dataset, vrp_strat)

    # Skew v2
    skew_engine = OptionsBacktestEngine(
        costs=COSTS, bars_per_year=365, use_options_pnl=True, skew_sensitivity_mult=2.5,
    )
    skew_strat = SkewTradeV2Strategy(params={
        "skew_lookback": 60, "z_entry": 2.0, "z_exit": 0.0,
        "target_leverage": 1.0, "rebalance_freq": 5, "min_bars": 60,
        "use_vrp_filter": False,
    })
    skew_result = skew_engine.run(dataset, skew_strat)

    # Butterfly (all configs)
    butterfly_results = {}
    for label, params in BUTTERFLY_CONFIGS.items():
        bf_engine = OptionsBacktestEngine(
            costs=COSTS, bars_per_year=365, use_options_pnl=True, skew_sensitivity_mult=1.0,
        )
        bf_strat = FeatureMRStrategy(name=params["name"], params=params)
        butterfly_results[label] = bf_engine.run(dataset, bf_strat)

    all_returns = {
        "VRP": vrp_result.returns,
        "Skew_v2": skew_result.returns,
    }
    for label, res in butterfly_results.items():
        all_returns[f"BF_{label}"] = res.returns

    # Pairwise correlations
    names = list(all_returns.keys())
    corr_matrix = {}
    for i, n1 in enumerate(names):
        for j, n2 in enumerate(names):
            if i >= j:
                continue
            r1, r2 = all_returns[n1], all_returns[n2]
            mn = min(len(r1), len(r2))
            r1, r2 = r1[:mn], r2[:mn]
            m1 = sum(r1) / mn
            m2 = sum(r2) / mn
            cov = sum((a - m1) * (b - m2) for a, b in zip(r1, r2)) / (mn - 1)
            v1 = sum((a - m1) ** 2 for a in r1) / (mn - 1)
            v2 = sum((b - m2) ** 2 for b in r2) / (mn - 1)
            s1, s2 = v1 ** 0.5, v2 ** 0.5
            corr = cov / (s1 * s2) if s1 > 0 and s2 > 0 else 0.0
            key = f"{n1}_vs_{n2}"
            corr_matrix[key] = round(corr, 4)
            print(f"  {n1:20s} vs {n2:20s}: r = {corr:+.4f}")

    return corr_matrix, all_returns


def run_ensemble_optimization(all_returns):
    """Test 3-way ensemble optimization across weight combos."""
    print(f"\n{'='*70}")
    print(f"  3-WAY ENSEMBLE OPTIMIZATION")
    print(f"{'='*70}\n")

    # Use slow_regime butterfly (most robust)
    vrp_rets = all_returns["VRP"]
    skew_rets = all_returns["Skew_v2"]
    bf_rets = all_returns["BF_slow_regime"]

    mn = min(len(vrp_rets), len(skew_rets), len(bf_rets))
    vrp_rets = vrp_rets[:mn]
    skew_rets = skew_rets[:mn]
    bf_rets = bf_rets[:mn]

    # Weight grid: VRP + Skew + Butterfly = 1.0
    # Test increments of 10%
    weight_combos = []
    for w_vrp in [i / 10 for i in range(1, 8)]:   # 10-70%
        for w_skew in [i / 10 for i in range(1, 8)]:  # 10-70%
            w_bf = round(1.0 - w_vrp - w_skew, 2)
            if 0.05 <= w_bf <= 0.50:
                weight_combos.append((w_vrp, w_skew, w_bf))

    print(f"  Testing {len(weight_combos)} weight combinations")

    # Also test 2-way baseline
    best_2way = None
    for w_vrp in [i / 10 for i in range(1, 10)]:
        w_skew = round(1.0 - w_vrp, 2)
        ens_rets = [w_vrp * v + w_skew * s for v, s in zip(vrp_rets, skew_rets)]
        sharpe = _compute_sharpe(ens_rets, 365)
        if best_2way is None or sharpe > best_2way[1]:
            best_2way = ((w_vrp, w_skew), sharpe)
    print(f"  2-way baseline: VRP={best_2way[0][0]:.0%} Skew={best_2way[0][1]:.0%} → Sharpe={best_2way[1]:+.4f}")

    # 3-way ensemble
    results = []
    for w_vrp, w_skew, w_bf in weight_combos:
        ens_rets = [
            w_vrp * v + w_skew * s + w_bf * b
            for v, s, b in zip(vrp_rets, skew_rets, bf_rets)
        ]
        sharpe = _compute_sharpe(ens_rets, 365)
        results.append({
            "w_vrp": w_vrp, "w_skew": w_skew, "w_bf": w_bf,
            "sharpe": sharpe,
        })
    results.sort(key=lambda r: r["sharpe"], reverse=True)

    # Per-year 3-way ensemble for top combos
    print(f"\n  TOP 15 (3-WAY) vs 2-WAY BASELINE:")
    for r in results[:15]:
        yearly = _compute_yearly_sharpe(
            vrp_rets, skew_rets, bf_rets,
            r["w_vrp"], r["w_skew"], r["w_bf"],
            365,
        )
        avg = sum(yearly.values()) / len(yearly)
        mn = min(yearly.values())
        yearly_str = " ".join(f"{y}={s:+.2f}" for y, s in sorted(yearly.items()))
        flag = " ★" if r["sharpe"] > best_2way[1] else ""
        print(f"    VRP={r['w_vrp']:.0%} Skew={r['w_skew']:.0%} BF={r['w_bf']:.0%} "
              f"→ full={r['sharpe']:+.4f} avg={avg:+.4f} min={mn:+.4f}{flag}")
        print(f"      {yearly_str}")
        r["yearly"] = yearly
        r["wf_avg"] = round(avg, 4)
        r["wf_min"] = round(mn, 4)

    return results, best_2way


def _compute_sharpe(returns, bars_per_year):
    if not returns:
        return 0.0
    n = len(returns)
    mean_r = sum(returns) / n
    var_r = sum((r - mean_r) ** 2 for r in returns) / max(n - 1, 1)
    vol = var_r ** 0.5
    return (mean_r / vol * (bars_per_year ** 0.5)) if vol > 0 else 0.0


def _compute_yearly_sharpe(vrp_rets, skew_rets, bf_rets, w_vrp, w_skew, w_bf, bars_per_year):
    """Per-year Sharpe using approximate 365-bar year segments."""
    total = min(len(vrp_rets), len(skew_rets), len(bf_rets))
    yearly = {}
    years = [2020, 2021, 2022, 2023, 2024, 2025]
    bars_per_seg = total // len(years) if total > 0 else 365

    for yi, yr in enumerate(years):
        start = yi * bars_per_seg
        end = min(start + bars_per_seg, total)
        if start >= total:
            break
        seg_rets = [
            w_vrp * vrp_rets[i] + w_skew * skew_rets[i] + w_bf * bf_rets[i]
            for i in range(start, end)
        ]
        s = _compute_sharpe(seg_rets, bars_per_year)
        yearly[str(yr)] = round(s, 4)

    return yearly


def main():
    t0 = time.time()

    # 1. Conservative validation
    cons_results = run_conservative_validation()

    # 2. Correlations
    corr_matrix, all_returns = compute_full_correlations()

    # 3. Ensemble optimization
    ens_results, baseline_2way = run_ensemble_optimization(all_returns)

    # 4. Summary
    print(f"\n{'='*70}")
    print(f"  FINAL SUMMARY")
    print(f"{'='*70}")
    print(f"  2-way champion (VRP+Skew): Sharpe={baseline_2way[1]:+.4f}")
    if ens_results:
        best_3way = ens_results[0]
        improvement = best_3way["sharpe"] - baseline_2way[1]
        print(f"  3-way champion (VRP+Skew+BF): Sharpe={best_3way['sharpe']:+.4f} "
              f"(Δ={improvement:+.4f})")
        print(f"    Weights: VRP={best_3way['w_vrp']:.0%} Skew={best_3way['w_skew']:.0%} "
              f"BF={best_3way['w_bf']:.0%}")
        if best_3way.get("wf_avg") is not None:
            print(f"    WF avg={best_3way['wf_avg']:+.4f} min={best_3way['wf_min']:+.4f}")

    # Save
    os.makedirs("artifacts/crypto_options", exist_ok=True)
    output = {
        "date": time.strftime("%Y-%m-%d %H:%M"),
        "conservative_validation": cons_results,
        "correlations": corr_matrix,
        "ensemble_2way_baseline": {"weights": best_2way[0], "sharpe": round(best_2way[1], 4)},
        "ensemble_3way_top10": ens_results[:10],
        "elapsed_sec": round(time.time() - t0, 1),
    }
    out_path = "artifacts/crypto_options/butterfly_ensemble_results.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\n  Results saved to {out_path}")
    print(f"  Total elapsed: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
