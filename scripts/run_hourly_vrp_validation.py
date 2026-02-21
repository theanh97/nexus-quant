#!/usr/bin/env python3
"""
Hourly VRP Validation + Mixed-Frequency Ensemble
=================================================
Phase 2 of hourly R&D. Key findings from Phase 1:
  - Hourly VRP: Sharpe 3.65 (vs daily 2.09) — 75% improvement
  - Hourly Skew MR: ALL NEGATIVE — signal too noisy at hourly

This script:
1. Conservative cost validation for hourly VRP champion
2. Wider rebalance_freq sweep for hourly VRP (find optimal)
3. Mixed-frequency ensemble: hourly VRP + daily Skew MR
4. Full walk-forward comparison: daily ensemble vs mixed-freq
"""
import json
import math
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

HOURLY_CFG = {
    "provider": "deribit_rest_v1",
    "symbols": ["BTC", "ETH"],
    "bar_interval": "1h",
    "cache_dir": "data/cache/deribit",
    "use_synthetic_iv": True,
    "rv_lookback_bars": 504,
}

DAILY_CFG = {
    "provider": "deribit_rest_v1",
    "symbols": ["BTC", "ETH"],
    "bar_interval": "1d",
    "cache_dir": "data/cache/deribit",
    "use_synthetic_iv": True,
    "rv_lookback_bars": 21,
}

# Champion params from Phase 1
VRP_HOURLY_CHAMPION = {
    "vrp_lookback": 720,
    "exit_z_threshold": -3.0,
    "base_leverage": 1.5,
    "rebalance_freq": 240,
    "min_bars": 720,
}

VRP_DAILY_CHAMPION = {
    "base_leverage": 1.5,
    "exit_z_threshold": -3.0,
    "vrp_lookback": 30,
    "rebalance_freq": 5,
    "min_bars": 30,
}

SKEW_DAILY_CHAMPION = {
    "skew_lookback": 60,
    "z_entry": 2.0,
    "z_exit": 0.0,
    "target_leverage": 1.0,
    "rebalance_freq": 5,
    "min_bars": 60,
    "use_vrp_filter": False,
}


def run_conservative_validation():
    """Validate hourly VRP champion at conservative costs."""
    print(f"{'='*70}")
    print(f"  HOURLY VRP CONSERVATIVE COST VALIDATION")
    print(f"{'='*70}\n")

    results = {}
    for label, costs in [("realistic", COSTS), ("conservative_1.5x", COSTS_CONSERVATIVE)]:
        wf = run_yearly_wf(
            provider_cfg=HOURLY_CFG,
            strategy_cls=VariancePremiumStrategy,
            strategy_params=VRP_HOURLY_CHAMPION,
            years=YEARS,
            costs=costs,
            use_options_pnl=True,
            bars_per_year=8760,
            seed=42,
        )
        s = wf["summary"]
        yearly = {k: v.get("sharpe", 0) for k, v in wf["yearly"].items()}
        flag = "PASS" if s["passed"] else "FAIL"
        print(f"  {label:20s}: avg={s['avg_sharpe']:+.4f} min={s['min_sharpe']:+.4f} [{flag}]")
        yearly_str = " ".join(f"{y}={v:+.2f}" for y, v in sorted(yearly.items()))
        print(f"  {'':20s}  {yearly_str}")
        results[label] = {"summary": s, "yearly": yearly}

    return results


def run_hourly_vrp_wider_sweep():
    """Test more rebalance frequencies for hourly VRP."""
    print(f"\n{'='*70}")
    print(f"  HOURLY VRP — REBALANCE FREQ SWEEP")
    print(f"{'='*70}\n")

    rebalance_freqs = [48, 72, 120, 168, 240, 360, 504, 720]
    # 2d, 3d, 5d, 7d, 10d, 15d, 21d, 30d

    results = []
    for rb in rebalance_freqs:
        params = dict(VRP_HOURLY_CHAMPION)
        params["rebalance_freq"] = rb

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
        s = wf["summary"]
        yearly = {k: v.get("sharpe", 0) for k, v in wf["yearly"].items()}
        flag = " ★ PASS" if s["passed"] else ""
        days = rb / 24
        print(f"  rb={rb:4d} ({days:5.1f}d): avg={s['avg_sharpe']:+.4f} min={s['min_sharpe']:+.4f}{flag}")
        yearly_str = " ".join(f"{y}={v:+.2f}" for y, v in sorted(yearly.items()))
        print(f"  {'':20s}  {yearly_str}")

        results.append({
            "rebalance_freq": rb,
            "rebalance_days": days,
            "avg_sharpe": s["avg_sharpe"],
            "min_sharpe": s["min_sharpe"],
            "passed": s["passed"],
            "yearly": yearly,
        })

    # Also conservative for top
    results.sort(key=lambda r: r["avg_sharpe"], reverse=True)
    best = results[0]
    print(f"\n  Best: rb={best['rebalance_freq']} ({best['rebalance_days']:.1f}d) "
          f"avg={best['avg_sharpe']:+.4f}")

    # Conservative for top 3
    print(f"\n  Conservative costs for top 3:")
    for r in results[:3]:
        rb = r["rebalance_freq"]
        params = dict(VRP_HOURLY_CHAMPION)
        params["rebalance_freq"] = rb
        wf = run_yearly_wf(
            provider_cfg=HOURLY_CFG,
            strategy_cls=VariancePremiumStrategy,
            strategy_params=params,
            years=YEARS,
            costs=COSTS_CONSERVATIVE,
            use_options_pnl=True,
            bars_per_year=8760,
            seed=42,
        )
        s = wf["summary"]
        flag = "PASS" if s["passed"] else "FAIL"
        print(f"    rb={rb:4d} ({rb/24:.1f}d): avg={s['avg_sharpe']:+.4f} min={s['min_sharpe']:+.4f} [{flag}]")
        r["conservative"] = {"avg": s["avg_sharpe"], "min": s["min_sharpe"], "passed": s["passed"]}

    return results


def run_mixed_freq_ensemble():
    """Mixed-frequency ensemble: hourly VRP + daily Skew MR.

    Approach: run each strategy at its optimal frequency on yearly windows,
    then aggregate returns to daily frequency for ensemble combination.
    """
    print(f"\n{'='*70}")
    print(f"  MIXED-FREQUENCY ENSEMBLE (Hourly VRP + Daily Skew MR)")
    print(f"{'='*70}\n")

    # Per-year evaluation
    yearly_results = {}

    for yr in YEARS:
        # --- Hourly VRP ---
        h_cfg = dict(HOURLY_CFG)
        h_cfg["start"] = f"{yr}-01-01"
        h_cfg["end"] = f"{yr}-12-31"
        h_provider = DeribitRestProvider(h_cfg, seed=42)
        h_dataset = h_provider.load()

        vrp_engine = OptionsBacktestEngine(
            costs=COSTS, bars_per_year=8760, use_options_pnl=True,
        )
        vrp_strat = VariancePremiumStrategy(params=VRP_HOURLY_CHAMPION)
        vrp_result = vrp_engine.run(h_dataset, vrp_strat)
        vrp_m = compute_metrics(vrp_result.equity_curve, vrp_result.returns, 8760)

        # Aggregate hourly VRP returns to daily
        vrp_daily_rets = _aggregate_returns_to_daily(vrp_result.returns, 24)

        # --- Daily Skew MR ---
        d_cfg = dict(DAILY_CFG)
        d_cfg["start"] = f"{yr}-01-01"
        d_cfg["end"] = f"{yr}-12-31"
        d_provider = DeribitRestProvider(d_cfg, seed=42)
        d_dataset = d_provider.load()

        skew_engine = OptionsBacktestEngine(
            costs=COSTS, bars_per_year=365, use_options_pnl=True,
            skew_sensitivity_mult=2.5,
        )
        skew_strat = SkewTradeV2Strategy(params=SKEW_DAILY_CHAMPION)
        skew_result = skew_engine.run(d_dataset, skew_strat)
        skew_m = compute_metrics(skew_result.equity_curve, skew_result.returns, 365)

        skew_rets = skew_result.returns

        # Align lengths (truncate to shorter)
        mn = min(len(vrp_daily_rets), len(skew_rets))
        vrp_d = vrp_daily_rets[:mn]
        skew_d = skew_rets[:mn]

        # Ensemble at various weights
        best_ens = None
        for w_vrp in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]:
            w_skew = round(1.0 - w_vrp, 2)
            ens_rets = [w_vrp * v + w_skew * s for v, s in zip(vrp_d, skew_d)]
            ens_eq = [1.0]
            for r in ens_rets:
                ens_eq.append(ens_eq[-1] * (1 + r))
            ens_m = compute_metrics(ens_eq, ens_rets, 365)

            if best_ens is None or ens_m["sharpe"] > best_ens["sharpe"]:
                best_ens = {"w_vrp": w_vrp, "w_skew": w_skew, **ens_m}

        # Also compute standard weight combos
        ens_30_70 = _compute_ensemble(vrp_d, skew_d, 0.3, 0.7, 365)
        ens_40_60 = _compute_ensemble(vrp_d, skew_d, 0.4, 0.6, 365)
        ens_50_50 = _compute_ensemble(vrp_d, skew_d, 0.5, 0.5, 365)

        # Correlation
        if mn > 10:
            m1 = sum(vrp_d) / mn
            m2 = sum(skew_d) / mn
            cov = sum((a - m1) * (b - m2) for a, b in zip(vrp_d, skew_d)) / (mn - 1)
            v1 = sum((a - m1) ** 2 for a in vrp_d) / (mn - 1)
            v2 = sum((b - m2) ** 2 for b in skew_d) / (mn - 1)
            s1, s2 = v1 ** 0.5, v2 ** 0.5
            corr = cov / (s1 * s2) if s1 > 0 and s2 > 0 else 0.0
        else:
            corr = 0.0

        yearly_results[str(yr)] = {
            "vrp_hourly_sharpe": vrp_m["sharpe"],
            "skew_daily_sharpe": skew_m["sharpe"],
            "correlation": round(corr, 4),
            "ensemble_30_70": ens_30_70["sharpe"],
            "ensemble_40_60": ens_40_60["sharpe"],
            "ensemble_50_50": ens_50_50["sharpe"],
            "best_ensemble": best_ens,
            "n_daily_bars": mn,
        }

        print(f"  {yr}: VRP_h={vrp_m['sharpe']:+.2f} Skew_d={skew_m['sharpe']:+.2f} "
              f"corr={corr:+.3f} | ens30/70={ens_30_70['sharpe']:+.2f} "
              f"ens40/60={ens_40_60['sharpe']:+.2f} ens50/50={ens_50_50['sharpe']:+.2f} "
              f"| best={best_ens['sharpe']:+.2f} @{best_ens['w_vrp']:.0%}/{best_ens['w_skew']:.0%}")

    # Summary across years
    print(f"\n  MIXED-FREQ ENSEMBLE SUMMARY:")
    for w_label, w_vrp_val in [("30/70", 0.3), ("40/60", 0.4), ("50/50", 0.5)]:
        key = f"ensemble_{w_label.replace('/', '_')}"
        sharpes = [yearly_results[str(yr)][key] for yr in YEARS]
        avg = sum(sharpes) / len(sharpes)
        mn = min(sharpes)
        flag = "PASS" if avg >= 1.0 and mn >= 0.5 else "FAIL"
        print(f"    {w_label}: avg={avg:+.4f} min={mn:+.4f} [{flag}]")
        print(f"         {' '.join(f'{yr}={s:+.2f}' for yr, s in zip(YEARS, sharpes))}")

    # Daily-only baselines for comparison
    print(f"\n  DAILY-ONLY BASELINES (from wisdom):")
    print(f"    VRP daily:           avg=+2.0881")
    print(f"    Skew daily:          avg=+1.7440")
    print(f"    Ensemble daily 30/70: avg=+2.6150")

    return yearly_results


def _aggregate_returns_to_daily(hourly_returns, bars_per_day=24):
    """Aggregate hourly returns to daily by compounding."""
    daily_rets = []
    for i in range(0, len(hourly_returns), bars_per_day):
        chunk = hourly_returns[i:i + bars_per_day]
        if not chunk:
            break
        # Compound: (1+r1)(1+r2)...(1+rn) - 1
        compounded = 1.0
        for r in chunk:
            compounded *= (1.0 + r)
        daily_rets.append(compounded - 1.0)
    return daily_rets


def _compute_ensemble(vrp_rets, skew_rets, w_vrp, w_skew, bars_per_year):
    """Compute ensemble returns and metrics."""
    mn = min(len(vrp_rets), len(skew_rets))
    ens_rets = [w_vrp * v + w_skew * s for v, s in zip(vrp_rets[:mn], skew_rets[:mn])]
    ens_eq = [1.0]
    for r in ens_rets:
        ens_eq.append(ens_eq[-1] * (1 + r))
    return compute_metrics(ens_eq, ens_rets, bars_per_year)


def main():
    t0 = time.time()

    # 1. Conservative cost validation
    cons_results = run_conservative_validation()

    # 2. Wider rebalance sweep
    rb_results = run_hourly_vrp_wider_sweep()

    # 3. Mixed-frequency ensemble
    mixed_results = run_mixed_freq_ensemble()

    # 4. Final summary
    print(f"\n{'='*70}")
    print(f"  FINAL SUMMARY")
    print(f"{'='*70}")

    # Conservative validation
    cons_real = cons_results.get("realistic", {}).get("summary", {})
    cons_15x = cons_results.get("conservative_1.5x", {}).get("summary", {})
    print(f"\n  Hourly VRP Champion (lb=720, rb=240):")
    print(f"    Realistic costs:    avg={cons_real.get('avg_sharpe', 0):+.4f} min={cons_real.get('min_sharpe', 0):+.4f}")
    print(f"    Conservative (1.5x): avg={cons_15x.get('avg_sharpe', 0):+.4f} min={cons_15x.get('min_sharpe', 0):+.4f}")

    # Best rebalance
    if rb_results:
        best_rb = rb_results[0]
        print(f"\n  Optimal hourly VRP rebalance: rb={best_rb['rebalance_freq']} ({best_rb['rebalance_days']:.1f}d)")
        print(f"    Sharpe: avg={best_rb['avg_sharpe']:+.4f} min={best_rb['min_sharpe']:+.4f}")
        if "conservative" in best_rb:
            print(f"    Conservative: avg={best_rb['conservative']['avg']:+.4f}")

    # Mixed-freq ensemble
    if mixed_results:
        for w_label, w_key in [("30/70", "ensemble_30_70"), ("40/60", "ensemble_40_60")]:
            sharpes = [mixed_results[str(yr)][w_key] for yr in YEARS]
            avg = sum(sharpes) / len(sharpes)
            mn = min(sharpes)
            print(f"\n  Mixed-Freq Ensemble ({w_label}):")
            print(f"    avg={avg:+.4f} min={mn:+.4f}")
            print(f"    vs Daily Ensemble (30/70): avg=+2.6150")
            print(f"    Delta: {avg - 2.615:+.4f}")

    # Save
    os.makedirs("artifacts/crypto_options", exist_ok=True)
    output = {
        "date": time.strftime("%Y-%m-%d %H:%M"),
        "conservative_validation": cons_results,
        "rebalance_sweep": rb_results[:10],
        "mixed_freq_ensemble": mixed_results,
        "elapsed_sec": round(time.time() - t0, 1),
    }
    out_path = "artifacts/crypto_options/hourly_vrp_validation_results.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\n  Results saved to {out_path}")
    print(f"  Total elapsed: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
