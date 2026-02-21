"""
VRP Parameter Sweep — Crypto Options R&D
=========================================
Grid search over VRP strategy parameters using walk-forward validation.

Sweep dimensions:
  base_leverage:   [1.0, 1.25, 1.5, 1.75, 2.0]
  exit_z_threshold: [-1.5, -2.0, -2.5, -3.0]
  vrp_lookback:    [30, 45, 60, 90]
  rebalance_freq:  [3, 5, 7, 10]

Total: 5 × 4 × 4 × 4 = 320 combinations
Objective: avg_sharpe (WF 2021-2025, yearly windows)

Current champion: base_leverage=1.5, exit_z=-2.0, vrp_lookback=60, rebal=5
  → avg_sharpe=1.520, min_sharpe=1.273

Usage:
    python3 scripts/run_vrp_param_sweep.py
    python3 scripts/run_vrp_param_sweep.py --top 20       # show top 20 results
    python3 scripts/run_vrp_param_sweep.py --quick         # reduced grid (faster)
"""
from __future__ import annotations

import itertools
import json
import logging
import sys
import time
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("vrp_sweep")

from nexus_quant.projects.crypto_options.strategies.variance_premium import VariancePremiumStrategy
from nexus_quant.projects.crypto_options.options_engine import (
    OptionsBacktestEngine,
    compute_metrics,
    run_yearly_wf,
)
from nexus_quant.projects.crypto_options.costs.deribit_fees import (
    DERIBIT_REALISTIC,
    DERIBIT_CONSERVATIVE,
)

# ── Grid ─────────────────────────────────────────────────────────────────────

FULL_GRID = {
    "base_leverage":    [1.0, 1.25, 1.5, 1.75, 2.0],
    "exit_z_threshold": [-1.5, -2.0, -2.5, -3.0],
    "vrp_lookback":     [30, 45, 60, 90],
    "rebalance_freq":   [3, 5, 7, 10],
}

QUICK_GRID = {
    "base_leverage":    [1.0, 1.5, 2.0],
    "exit_z_threshold": [-1.5, -2.0, -3.0],
    "vrp_lookback":     [30, 60, 90],
    "rebalance_freq":   [3, 5, 10],
}

PROVIDER_CFG = {
    "provider": "deribit_rest_v1",
    "symbols": ["BTC", "ETH"],
    "bar_interval": "1d",
    "cache_dir": "data/cache/deribit",
    "use_synthetic_iv": True,
    "rv_lookback_bars": 21,
}

YEARS = [2021, 2022, 2023, 2024, 2025]


def run_sweep(grid: dict, cost_model=DERIBIT_REALISTIC, top_n: int = 20):
    """Run full parameter sweep and return sorted results."""
    keys = list(grid.keys())
    values = list(grid.values())
    combos = list(itertools.product(*values))
    total = len(combos)

    logger.info(f"VRP Parameter Sweep: {total} combinations")
    logger.info(f"Grid: {json.dumps(grid, indent=2)}")
    logger.info(f"Years: {YEARS}")
    logger.info(f"Cost model: {'REALISTIC' if cost_model is DERIBIT_REALISTIC else 'CONSERVATIVE'}")
    logger.info("=" * 70)

    results = []
    champion_avg = 0.0
    champion_params = {}

    t0 = time.time()

    for i, combo in enumerate(combos):
        params = dict(zip(keys, combo))
        params["min_bars"] = 30  # fixed

        try:
            wf = run_yearly_wf(
                provider_cfg=PROVIDER_CFG,
                strategy_cls=VariancePremiumStrategy,
                strategy_params=params,
                years=YEARS,
                costs=cost_model,
                use_options_pnl=True,
                bars_per_year=365,
                seed=42,
            )

            summary = wf["summary"]
            avg_s = summary["avg_sharpe"]
            min_s = summary["min_sharpe"]
            yearly = wf["yearly"]

            result = {
                "rank": 0,
                "params": params,
                "avg_sharpe": avg_s,
                "min_sharpe": min_s,
                "years_positive": summary["years_positive"],
                "years_above_1": summary["years_above_1"],
                "passed": summary["passed"],
                "yearly_sharpes": {yr: yearly[yr].get("sharpe", 0) for yr in yearly},
            }
            results.append(result)

            # Track champion
            if avg_s > champion_avg:
                champion_avg = avg_s
                champion_params = params.copy()

            # Progress log every 20 combos
            if (i + 1) % 20 == 0 or i == 0:
                elapsed = time.time() - t0
                rate = (i + 1) / elapsed if elapsed > 0 else 0
                eta = (total - i - 1) / rate if rate > 0 else 0
                logger.info(
                    f"  [{i+1}/{total}] avg={avg_s:.3f} min={min_s:.3f} "
                    f"| best_so_far={champion_avg:.3f} "
                    f"| {elapsed:.0f}s elapsed, ~{eta:.0f}s remaining"
                )

        except Exception as e:
            logger.warning(f"  [{i+1}/{total}] FAILED: {params} — {e}")
            results.append({
                "rank": 0,
                "params": params,
                "avg_sharpe": -999,
                "min_sharpe": -999,
                "error": str(e),
            })

    elapsed = time.time() - t0
    logger.info(f"Sweep complete: {total} combos in {elapsed:.1f}s ({elapsed/total:.2f}s/combo)")

    # Sort by avg_sharpe descending
    results.sort(key=lambda r: r["avg_sharpe"], reverse=True)
    for i, r in enumerate(results):
        r["rank"] = i + 1

    # Print top results
    logger.info("=" * 70)
    logger.info(f"TOP {top_n} RESULTS:")
    logger.info("=" * 70)
    for r in results[:top_n]:
        p = r["params"]
        yearly = r.get("yearly_sharpes", {})
        yearly_str = " ".join(f"{yr}={s:.2f}" for yr, s in sorted(yearly.items()))
        logger.info(
            f"  #{r['rank']:3d} | avg={r['avg_sharpe']:+.3f} min={r['min_sharpe']:+.3f} "
            f"| lev={p['base_leverage']:.2f} ez={p['exit_z_threshold']:.1f} "
            f"lb={p['vrp_lookback']:2d} rb={p['rebalance_freq']:2d} "
            f"| {yearly_str}"
        )

    # Baseline comparison
    logger.info("-" * 70)
    baseline_result = next(
        (r for r in results if
         r["params"].get("base_leverage") == 1.5 and
         r["params"].get("exit_z_threshold") == -2.0 and
         r["params"].get("vrp_lookback") == 60 and
         r["params"].get("rebalance_freq") == 5),
        None
    )
    if baseline_result:
        logger.info(
            f"  BASELINE (P91b champion): avg={baseline_result['avg_sharpe']:+.3f} "
            f"min={baseline_result['min_sharpe']:+.3f} rank=#{baseline_result['rank']}"
        )

    best = results[0]
    delta = best["avg_sharpe"] - (baseline_result["avg_sharpe"] if baseline_result else 1.520)
    logger.info(
        f"  BEST: avg={best['avg_sharpe']:+.3f} min={best['min_sharpe']:+.3f} "
        f"Δ={delta:+.3f} vs baseline"
    )
    logger.info(f"  BEST params: {json.dumps(best['params'])}")

    return results


def save_results(results: list, tag: str = "vrp_sweep"):
    """Save sweep results to artifacts."""
    out_dir = ROOT / "artifacts" / "crypto_options"
    out_dir.mkdir(parents=True, exist_ok=True)

    out_path = out_dir / f"{tag}_results.json"
    with open(out_path, "w") as f:
        json.dump({
            "sweep": tag,
            "total_combos": len(results),
            "top_20": results[:20],
            "all_results": results,
        }, f, indent=2)

    logger.info(f"Results saved: {out_path}")
    return out_path


def main():
    quick = "--quick" in sys.argv
    top_n = 20
    for arg in sys.argv:
        if arg.startswith("--top"):
            try:
                top_n = int(sys.argv[sys.argv.index(arg) + 1])
            except (IndexError, ValueError):
                top_n = 20

    conservative = "--conservative" in sys.argv
    cost_model = DERIBIT_CONSERVATIVE if conservative else DERIBIT_REALISTIC

    grid = QUICK_GRID if quick else FULL_GRID
    tag = "vrp_sweep_quick" if quick else "vrp_sweep_full"
    if conservative:
        tag += "_conservative"

    logger.info("=" * 70)
    logger.info("NEXUS — VRP Parameter Sweep (Crypto Options R&D)")
    logger.info(f"  Mode: {'QUICK' if quick else 'FULL'}")
    logger.info(f"  Costs: {'CONSERVATIVE' if conservative else 'REALISTIC'}")
    logger.info("=" * 70)

    results = run_sweep(grid, cost_model=cost_model, top_n=top_n)
    save_results(results, tag=tag)


if __name__ == "__main__":
    main()
