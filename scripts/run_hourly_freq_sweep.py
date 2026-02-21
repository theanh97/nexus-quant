#!/usr/bin/env python3
"""
Hourly Frequency Sweep — Crypto Options
========================================
Test Skew MR and VRP at hourly bar frequency (8760 bars/year).

Hypothesis: Skew MR may benefit from faster signal observation and
quicker rebalancing. VRP carry trade may or may not improve.

Key adjustments for hourly:
- Provider generates 24x more bars per year
- AR(1) synthetic IV coefficients auto-adjust to preserve real-time half-life
- Lookback/min_bars scaled: daily 30 → hourly 720 (30 days × 24 hours)
- Rebalance freq: test from 4-hourly to weekly

Compares with daily baseline to measure improvement.
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

# ── Cost model ───────────────────────────────────────────────────────────────
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

# ── Provider configs ─────────────────────────────────────────────────────────
HOURLY_CFG = {
    "provider": "deribit_rest_v1",
    "symbols": ["BTC", "ETH"],
    "bar_interval": "1h",
    "cache_dir": "data/cache/deribit",
    "use_synthetic_iv": True,
    "rv_lookback_bars": 504,   # 21 days × 24 hours
}

DAILY_CFG = {
    "provider": "deribit_rest_v1",
    "symbols": ["BTC", "ETH"],
    "bar_interval": "1d",
    "cache_dir": "data/cache/deribit",
    "use_synthetic_iv": True,
    "rv_lookback_bars": 21,
}


def run_daily_baselines():
    """Run daily Skew MR and VRP baselines for comparison."""
    print(f"\n{'='*70}")
    print(f"  DAILY BASELINES (bars_per_year=365)")
    print(f"{'='*70}\n")

    results = {}

    # VRP baseline (champion params)
    print("  Running VRP daily baseline...")
    vrp_wf = run_yearly_wf(
        provider_cfg=DAILY_CFG,
        strategy_cls=VariancePremiumStrategy,
        strategy_params={
            "base_leverage": 1.5, "exit_z_threshold": -3.0,
            "vrp_lookback": 30, "rebalance_freq": 5, "min_bars": 30,
        },
        years=YEARS, costs=COSTS, use_options_pnl=True,
        bars_per_year=365, seed=42,
    )
    s = vrp_wf["summary"]
    yearly = {k: v.get("sharpe", 0) for k, v in vrp_wf["yearly"].items()}
    print(f"    VRP daily:  avg={s['avg_sharpe']:+.4f} min={s['min_sharpe']:+.4f}")
    yearly_str = " ".join(f"{y}={v:+.2f}" for y, v in sorted(yearly.items()))
    print(f"                {yearly_str}")
    results["vrp_daily"] = {"summary": s, "yearly": yearly}

    # Skew MR baseline (champion params)
    print("  Running Skew MR daily baseline...")
    skew_wf = run_yearly_wf(
        provider_cfg=DAILY_CFG,
        strategy_cls=SkewTradeV2Strategy,
        strategy_params={
            "skew_lookback": 60, "z_entry": 2.0, "z_exit": 0.0,
            "target_leverage": 1.0, "rebalance_freq": 5, "min_bars": 60,
            "use_vrp_filter": False,
        },
        years=YEARS, costs=COSTS, use_options_pnl=True,
        bars_per_year=365, seed=42, skew_sensitivity_mult=2.5,
    )
    s = skew_wf["summary"]
    yearly = {k: v.get("sharpe", 0) for k, v in skew_wf["yearly"].items()}
    print(f"    Skew daily: avg={s['avg_sharpe']:+.4f} min={s['min_sharpe']:+.4f}")
    yearly_str = " ".join(f"{y}={v:+.2f}" for y, v in sorted(yearly.items()))
    print(f"                {yearly_str}")
    results["skew_daily"] = {"summary": s, "yearly": yearly}

    return results


def run_hourly_skew_sweep():
    """Sweep Skew MR at hourly frequency."""
    print(f"\n{'='*70}")
    print(f"  HOURLY SKEW MR SWEEP (bars_per_year=8760)")
    print(f"{'='*70}\n")

    grid = {
        "skew_lookback": [336, 720, 1440],      # 14d, 30d, 60d in hours
        "z_entry": [1.5, 2.0, 2.5],
        "z_exit": [0.0, 0.3],
        "target_leverage": [1.0],
        "rebalance_freq": [4, 12, 24, 120],      # 4h, 12h, daily, weekly
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
                provider_cfg=HOURLY_CFG,
                strategy_cls=SkewTradeV2Strategy,
                strategy_params=params,
                years=YEARS,
                costs=COSTS,
                use_options_pnl=True,
                bars_per_year=8760,
                seed=42,
                skew_sensitivity_mult=2.5,
            )

            avg = wf["summary"]["avg_sharpe"]
            mn = wf["summary"]["min_sharpe"]
            passed = wf["summary"]["passed"]
            yearly = {k: v.get("sharpe", 0) for k, v in wf["yearly"].items()}

            result = {
                "rank": 0,
                "freq": "1h",
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
                    flag += " ★★★ VALIDATED"
            elif passed:
                flag = " ★ PASS"

            elapsed = time.time() - t0
            rate = (i + 1) / elapsed if elapsed > 0 else 0
            eta = (total - i - 1) / rate if rate > 0 else 0
            print(f"  [{i+1:3d}/{total}] lb={params['skew_lookback']:5d} ze={params['z_entry']:.1f} "
                  f"zx={params['z_exit']:.1f} rb={params['rebalance_freq']:3d} "
                  f"→ avg={avg:+.4f} min={mn:+.4f} (ETA {eta:.0f}s){flag}")

        except Exception as e:
            print(f"  [{i+1:3d}/{total}] ERROR: {e}")
            results.append({"params": params, "avg_sharpe": -999, "error": str(e)})

    results.sort(key=lambda r: r.get("avg_sharpe", -999), reverse=True)
    for i, r in enumerate(results):
        r["rank"] = i + 1

    return results


def run_hourly_vrp_sweep():
    """Sweep VRP at hourly frequency."""
    print(f"\n{'='*70}")
    print(f"  HOURLY VRP SWEEP (bars_per_year=8760)")
    print(f"{'='*70}\n")

    grid = {
        "vrp_lookback": [504, 720],      # 21d, 30d in hours
        "exit_z_threshold": [-3.0],
        "base_leverage": [1.5],
        "rebalance_freq": [24, 120, 240],  # daily, weekly, 10-day
    }

    keys = list(grid.keys())
    values = list(grid.values())
    combos = list(itertools.product(*values))
    total = len(combos)
    print(f"  Grid: {total} configurations\n")

    results = []
    best_avg = -999
    t0 = time.time()

    for i, combo in enumerate(combos):
        params = dict(zip(keys, combo))
        params["min_bars"] = params["vrp_lookback"]

        try:
            wf = run_yearly_wf(
                provider_cfg=HOURLY_CFG,
                strategy_cls=VariancePremiumStrategy,
                strategy_params=params,
                years=YEARS,
                costs=COSTS,
                use_options_pnl=True,
                bars_per_year=8760,
                seed=42,
            )

            avg = wf["summary"]["avg_sharpe"]
            mn = wf["summary"]["min_sharpe"]
            passed = wf["summary"]["passed"]
            yearly = {k: v.get("sharpe", 0) for k, v in wf["yearly"].items()}

            result = {
                "rank": 0,
                "freq": "1h",
                "strategy": "vrp",
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
                    flag += " ★★★ VALIDATED"
            elif passed:
                flag = " ★ PASS"

            elapsed = time.time() - t0
            print(f"  [{i+1:3d}/{total}] lb={params['vrp_lookback']:4d} "
                  f"rb={params['rebalance_freq']:3d} "
                  f"→ avg={avg:+.4f} min={mn:+.4f} ({elapsed:.0f}s){flag}")

        except Exception as e:
            print(f"  [{i+1:3d}/{total}] ERROR: {e}")
            results.append({"params": params, "avg_sharpe": -999, "error": str(e)})

    results.sort(key=lambda r: r.get("avg_sharpe", -999), reverse=True)
    for i, r in enumerate(results):
        r["rank"] = i + 1

    return results


def run_hourly_ensemble(skew_results, vrp_results, daily_baselines):
    """Build hourly ensemble from best configs and compare with daily."""
    print(f"\n{'='*70}")
    print(f"  HOURLY ENSEMBLE vs DAILY ENSEMBLE")
    print(f"{'='*70}\n")

    # Get best hourly configs
    best_skew_h = next((r for r in skew_results if r.get("avg_sharpe", -999) > 0), None)
    best_vrp_h = next((r for r in vrp_results if r.get("avg_sharpe", -999) > 0), None)

    if not best_skew_h or not best_vrp_h:
        print("  No valid hourly results for ensemble.")
        return {}

    # Run both on full period for correlation + ensemble
    cfg = dict(HOURLY_CFG)
    cfg["start"] = "2020-01-01"
    cfg["end"] = "2025-12-31"

    provider = DeribitRestProvider(cfg, seed=42)
    dataset = provider.load()
    print(f"  Loaded {len(dataset.timeline)} hourly bars ({len(dataset.timeline)/8760:.1f} years)")

    # VRP hourly
    vrp_params = best_vrp_h["params"]
    vrp_engine = OptionsBacktestEngine(costs=COSTS, bars_per_year=8760, use_options_pnl=True)
    vrp_strat = VariancePremiumStrategy(params=vrp_params)
    vrp_result = vrp_engine.run(dataset, vrp_strat)
    vrp_m = compute_metrics(vrp_result.equity_curve, vrp_result.returns, 8760)
    print(f"  VRP hourly:  Sharpe={vrp_m['sharpe']:+.4f} CAGR={vrp_m['cagr']*100:.1f}%")

    # Skew hourly
    skew_params = best_skew_h["params"]
    skew_engine = OptionsBacktestEngine(
        costs=COSTS, bars_per_year=8760, use_options_pnl=True,
        skew_sensitivity_mult=2.5,
    )
    skew_strat = SkewTradeV2Strategy(params=skew_params)
    skew_result = skew_engine.run(dataset, skew_strat)
    skew_m = compute_metrics(skew_result.equity_curve, skew_result.returns, 8760)
    print(f"  Skew hourly: Sharpe={skew_m['sharpe']:+.4f} CAGR={skew_m['cagr']*100:.1f}%")

    # Correlation
    vrp_rets = vrp_result.returns
    skew_rets = skew_result.returns
    mn = min(len(vrp_rets), len(skew_rets))
    vrp_rets = vrp_rets[:mn]
    skew_rets = skew_rets[:mn]

    m1 = sum(vrp_rets) / mn
    m2 = sum(skew_rets) / mn
    cov = sum((a - m1) * (b - m2) for a, b in zip(vrp_rets, skew_rets)) / (mn - 1)
    v1 = sum((a - m1) ** 2 for a in vrp_rets) / (mn - 1)
    v2 = sum((b - m2) ** 2 for b in skew_rets) / (mn - 1)
    s1, s2 = v1 ** 0.5, v2 ** 0.5
    corr = cov / (s1 * s2) if s1 > 0 and s2 > 0 else 0.0
    print(f"  Correlation (VRP vs Skew hourly): r = {corr:+.4f}")

    # Ensemble at various weights
    print(f"\n  Ensemble weights (hourly):")
    best_ens = None
    for w_vrp in [0.1, 0.2, 0.3, 0.4, 0.5]:
        w_skew = round(1.0 - w_vrp, 2)
        ens_rets = [w_vrp * v + w_skew * s for v, s in zip(vrp_rets, skew_rets)]
        ens_eq = [1.0]
        for r in ens_rets:
            ens_eq.append(ens_eq[-1] * (1 + r))
        ens_m = compute_metrics(ens_eq, ens_rets, 8760)
        flag = ""
        if best_ens is None or ens_m["sharpe"] > best_ens["sharpe"]:
            best_ens = {"w_vrp": w_vrp, "w_skew": w_skew, **ens_m}
            flag = " ★ BEST"
        print(f"    VRP={w_vrp:.0%} Skew={w_skew:.0%}: Sharpe={ens_m['sharpe']:+.4f} "
              f"CAGR={ens_m['cagr']*100:.1f}% MDD={ens_m['max_drawdown']*100:.1f}%{flag}")

    # Compare with daily baseline
    daily_skew = daily_baselines.get("skew_daily", {}).get("summary", {})
    daily_vrp = daily_baselines.get("vrp_daily", {}).get("summary", {})
    print(f"\n  COMPARISON:")
    print(f"    Daily VRP:      avg={daily_vrp.get('avg_sharpe', 0):+.4f}")
    print(f"    Hourly VRP:     avg={best_vrp_h['avg_sharpe']:+.4f}")
    print(f"    Daily Skew MR:  avg={daily_skew.get('avg_sharpe', 0):+.4f}")
    print(f"    Hourly Skew MR: avg={best_skew_h['avg_sharpe']:+.4f}")
    print(f"    Daily Ensemble (30/70):  Sharpe≈2.615 (from wisdom)")
    if best_ens:
        print(f"    Hourly Ensemble ({best_ens['w_vrp']:.0%}/{best_ens['w_skew']:.0%}): "
              f"Sharpe={best_ens['sharpe']:+.4f}")

    return {
        "hourly_vrp_standalone": vrp_m,
        "hourly_skew_standalone": skew_m,
        "hourly_correlation": round(corr, 4),
        "hourly_best_ensemble": best_ens,
        "hourly_vrp_params": vrp_params,
        "hourly_skew_params": skew_params,
    }


def main():
    t0 = time.time()

    # 1. Daily baselines
    daily_baselines = run_daily_baselines()

    # 2. Hourly VRP sweep
    vrp_results = run_hourly_vrp_sweep()

    # 3. Hourly Skew MR sweep
    skew_results = run_hourly_skew_sweep()

    # 4. Top results
    print(f"\n{'='*70}")
    print(f"  TOP 10 HOURLY SKEW MR")
    print(f"{'='*70}")
    for r in skew_results[:10]:
        p = r["params"]
        flag = " ★ PASS" if r.get("passed") else ""
        yearly = r.get("yearly", {})
        yearly_str = " ".join(f"{y}={s:+.2f}" for y, s in sorted(yearly.items()))
        print(f"  #{r['rank']:2d} avg={r['avg_sharpe']:+.4f} min={r['min_sharpe']:+.4f} "
              f"lb={p['skew_lookback']:5d} ze={p['z_entry']:.1f} "
              f"rb={p['rebalance_freq']:3d}{flag}")
        print(f"      {yearly_str}")

    print(f"\n  TOP HOURLY VRP:")
    for r in vrp_results[:5]:
        p = r["params"]
        flag = " ★ PASS" if r.get("passed") else ""
        yearly = r.get("yearly", {})
        yearly_str = " ".join(f"{y}={s:+.2f}" for y, s in sorted(yearly.items()))
        print(f"  #{r['rank']:2d} avg={r['avg_sharpe']:+.4f} min={r['min_sharpe']:+.4f} "
              f"lb={p['vrp_lookback']:4d} rb={p['rebalance_freq']:3d}{flag}")
        print(f"      {yearly_str}")

    # 5. Ensemble comparison
    ensemble_results = run_hourly_ensemble(skew_results, vrp_results, daily_baselines)

    # 6. Save
    os.makedirs("artifacts/crypto_options", exist_ok=True)
    output = {
        "sweep_date": time.strftime("%Y-%m-%d %H:%M"),
        "daily_baselines": daily_baselines,
        "hourly_skew_top_20": skew_results[:20],
        "hourly_vrp_all": vrp_results,
        "ensemble": ensemble_results,
        "elapsed_sec": round(time.time() - t0, 1),
    }
    out_path = "artifacts/crypto_options/hourly_freq_sweep_results.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\n  Results saved to {out_path}")
    print(f"  Total elapsed: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
