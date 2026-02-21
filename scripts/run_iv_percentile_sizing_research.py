#!/usr/bin/env python3
"""
VRP Position Sizing by IV Percentile — Research Script
========================================================

Hypothesis: Scale VRP position size by IV percentile rank.
Higher IV = larger variance risk premium = should have more exposure.

This is DIFFERENT from regime filtering (R21):
  - R21: filter IN/OUT based on secondary signals → destroys carry (less time in market)
  - This: scale SIZE proportional to carry richness → always in market, more when better

Sizing models tested:
  1. Linear: weight *= (0.5 + 0.5 * iv_pct)  — range [0.5x, 1.0x]
  2. Aggressive: weight *= iv_pct              — range [0.0x, 1.0x]
  3. Sqrt: weight *= (0.5 + 0.5 * sqrt(iv_pct)) — range [0.5x, 1.0x]
  4. Step: weight *= {<25th: 0.5, 25-75th: 1.0, >75th: 1.5}
  5. Inverse: weight *= (1.5 - 0.5 * iv_pct)  — range [1.0x, 1.5x] (contrarian: more when IV is low)

Plus lookback sweep: 30d, 60d, 90d, 120d, 180d for IV percentile calculation.

Grid: 5 models × 5 lookbacks = 25 configs
"""
import json
import math
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from nexus_quant.backtest.costs import ExecutionCostModel, FeeModel, ImpactModel
from nexus_quant.projects.crypto_options.options_engine import (
    OptionsBacktestEngine, compute_metrics
)
from nexus_quant.projects.crypto_options.providers.deribit_rest import DeribitRestProvider
from nexus_quant.projects.crypto_options.strategies.variance_premium import VariancePremiumStrategy

# ── Cost model ─────────────────────────────────────────────────────────────

fees = FeeModel(maker_fee_rate=0.0003, taker_fee_rate=0.0005)
impact = ImpactModel(model="sqrt", coef_bps=2.0)
COSTS = ExecutionCostModel(fee=fees, impact=impact)

YEARS = [2021, 2022, 2023, 2024, 2025]
SEED = 42
BARS_PER_YEAR = 365

# Sizing models
MODELS = {
    "linear": lambda pct: 0.5 + 0.5 * pct,               # [0.5, 1.0]
    "aggressive": lambda pct: max(pct, 0.1),               # [0.1, 1.0]
    "sqrt": lambda pct: 0.5 + 0.5 * math.sqrt(pct),      # [0.5, 1.0]
    "step": lambda pct: 0.5 if pct < 0.25 else (1.5 if pct > 0.75 else 1.0),
    "inverse": lambda pct: 1.5 - 0.5 * pct,               # [1.0, 1.5] (contrarian)
}

LOOKBACKS = [30, 60, 90, 120, 180]


class IVPercentileSizedVRP(VariancePremiumStrategy):
    """VRP strategy with IV-percentile position sizing."""

    def __init__(
        self,
        params: Dict[str, Any],
        sizing_model: str = "linear",
        sizing_lookback: int = 60,
    ):
        super().__init__(params=params)
        self.sizing_model = sizing_model
        self.sizing_lookback = sizing_lookback
        self._sizing_fn = MODELS[sizing_model]

    def target_weights(self, dataset, idx: int, current_weights) -> Dict[str, float]:
        """Get base VRP weights, then scale by IV percentile."""
        base = super().target_weights(dataset, idx, current_weights)

        for sym in list(base.keys()):
            if abs(base[sym]) < 1e-10:
                continue

            # Compute IV percentile
            iv_series = dataset.features.get("iv_atm", {}).get(sym, [])
            if idx < self.sizing_lookback or not iv_series:
                continue  # not enough data, use base weight

            # Get lookback window of IV values
            start = max(0, idx - self.sizing_lookback)
            window = [v for v in iv_series[start:idx] if v is not None]
            if len(window) < 10:
                continue

            current_iv = iv_series[idx] if idx < len(iv_series) and iv_series[idx] is not None else None
            if current_iv is None:
                continue

            # Percentile rank: fraction of window below current IV
            below = sum(1 for v in window if v < current_iv)
            pct = below / len(window)

            # Apply sizing function
            scale = self._sizing_fn(pct)
            base[sym] *= scale

        return base


def run_baseline_wf() -> Dict[str, Any]:
    """Run unmodified VRP as baseline."""
    engine = OptionsBacktestEngine(costs=COSTS, bars_per_year=BARS_PER_YEAR)
    sharpes = []

    for yr in YEARS:
        cfg = {
            "symbols": ["BTC"],
            "start": f"{yr}-01-01",
            "end": f"{yr}-12-31",
            "bar_interval": "1d",
            "use_synthetic_iv": True,
        }
        provider = DeribitRestProvider(cfg, seed=SEED)
        dataset = provider.load()

        strat = VariancePremiumStrategy(params={
            "base_leverage": 1.5,
            "exit_z_threshold": -3.0,
            "vrp_lookback": 30,
            "rebalance_freq": 5,
            "min_bars": 30,
        })

        result = engine.run(dataset, strat)
        m = compute_metrics(result.equity_curve, result.returns, BARS_PER_YEAR)
        sharpes.append(m["sharpe"])

    avg = sum(sharpes) / len(sharpes)
    mn = min(sharpes)
    return {"avg_sharpe": round(avg, 3), "min_sharpe": round(mn, 3), "yearly": sharpes}


def run_sized_wf(model: str, lookback: int) -> Dict[str, Any]:
    """Run IV-percentile sized VRP."""
    engine = OptionsBacktestEngine(costs=COSTS, bars_per_year=BARS_PER_YEAR)
    sharpes = []

    for yr in YEARS:
        cfg = {
            "symbols": ["BTC"],
            "start": f"{yr}-01-01",
            "end": f"{yr}-12-31",
            "bar_interval": "1d",
            "use_synthetic_iv": True,
        }
        provider = DeribitRestProvider(cfg, seed=SEED)
        dataset = provider.load()

        strat = IVPercentileSizedVRP(
            params={
                "base_leverage": 1.5,
                "exit_z_threshold": -3.0,
                "vrp_lookback": 30,
                "rebalance_freq": 5,
                "min_bars": 30,
            },
            sizing_model=model,
            sizing_lookback=lookback,
        )

        result = engine.run(dataset, strat)
        m = compute_metrics(result.equity_curve, result.returns, BARS_PER_YEAR)
        sharpes.append(m["sharpe"])

    avg = sum(sharpes) / len(sharpes)
    mn = min(sharpes)
    return {"avg_sharpe": round(avg, 3), "min_sharpe": round(mn, 3), "yearly": sharpes}


def main():
    print("=" * 70)
    print("VRP POSITION SIZING BY IV PERCENTILE — RESEARCH")
    print("=" * 70)
    print(f"Grid: {len(MODELS)} models × {len(LOOKBACKS)} lookbacks = {len(MODELS) * len(LOOKBACKS)} configs")
    print(f"Years: {YEARS}")
    print()

    # Baseline
    print("Running baseline (unmodified VRP)...")
    baseline = run_baseline_wf()
    print(f"  Baseline: avg={baseline['avg_sharpe']:.3f} min={baseline['min_sharpe']:.3f}")
    print(f"  Yearly: {[f'{s:.3f}' for s in baseline['yearly']]}")
    print()

    # Grid search
    results = []
    best_delta = -999.0
    best_config = None

    for model in MODELS:
        for lb in LOOKBACKS:
            tag = f"{model}_lb{lb}"
            r = run_sized_wf(model, lb)
            delta = r["avg_sharpe"] - baseline["avg_sharpe"]
            results.append({
                "model": model,
                "lookback": lb,
                "tag": tag,
                "avg_sharpe": r["avg_sharpe"],
                "min_sharpe": r["min_sharpe"],
                "delta": round(delta, 3),
                "yearly": r["yearly"],
            })
            marker = "+" if delta > 0 else " "
            print(f"  {tag:25s} avg={r['avg_sharpe']:.3f} min={r['min_sharpe']:.3f} Δ={delta:+.3f} {marker}")

            if delta > best_delta:
                best_delta = delta
                best_config = results[-1]

    # Sort by delta
    results.sort(key=lambda x: x["delta"], reverse=True)

    print()
    print("=" * 70)
    print("RESULTS — Sorted by Δ vs baseline")
    print("=" * 70)
    print(f"\n  Baseline: avg={baseline['avg_sharpe']:.3f}")
    print()

    n_better = sum(1 for r in results if r["delta"] > 0)
    n_worse = sum(1 for r in results if r["delta"] < 0)
    n_equal = sum(1 for r in results if r["delta"] == 0)

    print(f"  Better than baseline: {n_better}/{len(results)}")
    print(f"  Worse than baseline:  {n_worse}/{len(results)}")
    print(f"  Equal to baseline:    {n_equal}/{len(results)}")
    print()

    # Top 5
    print("  TOP 5:")
    for r in results[:5]:
        print(f"    {r['tag']:25s} avg={r['avg_sharpe']:.3f} Δ={r['delta']:+.3f}")

    # Bottom 5
    print("\n  BOTTOM 5:")
    for r in results[-5:]:
        print(f"    {r['tag']:25s} avg={r['avg_sharpe']:.3f} Δ={r['delta']:+.3f}")

    # By model (averaged across lookbacks)
    print("\n  BY MODEL (avg across lookbacks):")
    for model in MODELS:
        model_results = [r for r in results if r["model"] == model]
        avg_delta = sum(r["delta"] for r in model_results) / len(model_results)
        best_lb = max(model_results, key=lambda x: x["delta"])
        print(f"    {model:12s} avg_Δ={avg_delta:+.3f}  best_lb={best_lb['lookback']}d Δ={best_lb['delta']:+.3f}")

    # Verdict
    print()
    print("=" * 70)
    if best_config and best_delta > 0.05:
        print(f"VERDICT: IV-percentile sizing IMPROVES VRP (Δ={best_delta:+.3f})")
        print(f"  Best: {best_config['tag']} avg={best_config['avg_sharpe']:.3f}")
    elif best_config and best_delta > 0:
        print(f"VERDICT: IV-percentile sizing is MARGINAL (best Δ={best_delta:+.3f})")
        print(f"  Best: {best_config['tag']} avg={best_config['avg_sharpe']:.3f}")
    else:
        print(f"VERDICT: IV-percentile sizing does NOT improve VRP")
        print(f"  Confirms R7: position-level overlays destroy carry")
    print("=" * 70)

    # Save results
    results_path = ROOT / "artifacts" / "crypto_options" / "iv_percentile_sizing_research.json"
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with open(results_path, "w") as f:
        json.dump({
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "baseline": baseline,
            "n_configs": len(results),
            "n_better": n_better,
            "n_worse": n_worse,
            "best_config": best_config,
            "best_delta": best_delta,
            "results": results,
        }, f, indent=2)
    print(f"\nResults saved to: {results_path}")


if __name__ == "__main__":
    main()
