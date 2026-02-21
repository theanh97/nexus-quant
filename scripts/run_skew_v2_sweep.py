"""
Skew MR v2 Sweep — Test redesigned skew strategy with vega P&L model
=====================================================================
Tests the new skew trading strategy with proper options P&L:
  - Vega-based P&L (not directional)
  - Hold management (entry/exit z-scores)
  - VRP confirmation filter
  - Parameter sweep over signal params

Sweep dimensions:
  skew_lookback: [20, 30, 45, 60]
  z_entry: [1.0, 1.5, 2.0, 2.5]
  z_exit: [0.0, 0.3, 0.5]
  rebalance_freq: [3, 5, 7]
  use_vrp_filter: [True, False]
  target_leverage: [1.0, 1.5, 2.0]

Usage:
    python3 scripts/run_skew_v2_sweep.py
    python3 scripts/run_skew_v2_sweep.py --quick
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
logger = logging.getLogger("skew_v2_sweep")

from nexus_quant.projects.crypto_options.strategies.skew_trade import SkewTradeStrategy
from nexus_quant.projects.crypto_options.strategies.skew_trade_v2 import SkewTradeV2Strategy
from nexus_quant.projects.crypto_options.options_engine import run_yearly_wf
from nexus_quant.projects.crypto_options.costs.deribit_fees import DERIBIT_REALISTIC

PROVIDER_CFG = {
    "provider": "deribit_rest_v1",
    "symbols": ["BTC", "ETH"],
    "bar_interval": "1d",
    "cache_dir": "data/cache/deribit",
    "use_synthetic_iv": True,
    "rv_lookback_bars": 21,
}
YEARS = [2021, 2022, 2023, 2024, 2025]

FULL_GRID = {
    "skew_lookback": [20, 30, 45, 60],
    "z_entry": [1.0, 1.5, 2.0, 2.5],
    "z_exit": [0.0, 0.3, 0.5],
    "rebalance_freq": [3, 5, 7],
    "use_vrp_filter": [True, False],
    "target_leverage": [1.0, 1.5, 2.0],
}

QUICK_GRID = {
    "skew_lookback": [20, 30, 60],
    "z_entry": [1.0, 1.5, 2.0],
    "z_exit": [0.0, 0.3],
    "rebalance_freq": [3, 5],
    "use_vrp_filter": [True, False],
    "target_leverage": [1.0, 1.5],
}


def run_baselines():
    """Run v1 and basic v2 for comparison."""
    logger.info("=" * 80)
    logger.info("BASELINES")
    logger.info("=" * 80)

    results = []

    # v1 original (delta-equivalent, known to fail)
    t0 = time.time()
    try:
        wf = run_yearly_wf(
            provider_cfg=PROVIDER_CFG,
            strategy_cls=SkewTradeStrategy,
            strategy_params={"skew_lookback": 30, "z_threshold": 1.5, "target_gross_leverage": 1.2, "rebalance_freq": 5, "min_bars": 60},
            years=YEARS, costs=DERIBIT_REALISTIC, use_options_pnl=True, bars_per_year=365, seed=42,
        )
        s = wf["summary"]
        yearly = wf["yearly"]
        yearly_str = " ".join(f"{yr}={yearly[yr].get('sharpe', 0):.2f}" for yr in sorted(yearly))
        logger.info(f"  v1 Skew MR (delta-equiv):  avg={s['avg_sharpe']:+.3f} min={s['min_sharpe']:+.3f} | {yearly_str} | {time.time()-t0:.1f}s")
        results.append({"name": "v1_delta", "avg_sharpe": s["avg_sharpe"], "min_sharpe": s["min_sharpe"]})
    except Exception as e:
        logger.warning(f"  v1 Skew MR FAILED: {e}")

    # v2 with defaults (vega model)
    t0 = time.time()
    try:
        wf = run_yearly_wf(
            provider_cfg=PROVIDER_CFG,
            strategy_cls=SkewTradeV2Strategy,
            strategy_params={"skew_lookback": 30, "z_entry": 1.5, "z_exit": 0.3, "target_leverage": 1.5, "rebalance_freq": 5, "min_bars": 60, "use_vrp_filter": True},
            years=YEARS, costs=DERIBIT_REALISTIC, use_options_pnl=True, bars_per_year=365, seed=42,
        )
        s = wf["summary"]
        yearly = wf["yearly"]
        yearly_str = " ".join(f"{yr}={yearly[yr].get('sharpe', 0):.2f}" for yr in sorted(yearly))
        logger.info(f"  v2 Skew MR (vega model):   avg={s['avg_sharpe']:+.3f} min={s['min_sharpe']:+.3f} | {yearly_str} | {time.time()-t0:.1f}s")
        results.append({"name": "v2_vega", "avg_sharpe": s["avg_sharpe"], "min_sharpe": s["min_sharpe"]})
    except Exception as e:
        logger.warning(f"  v2 Skew MR FAILED: {e}")

    return results


def run_sweep(grid: dict, top_n: int = 20):
    """Run full parameter sweep."""
    keys = list(grid.keys())
    values = list(grid.values())
    combos = list(itertools.product(*values))
    total = len(combos)

    logger.info("=" * 80)
    logger.info(f"SKEW MR v2 PARAMETER SWEEP: {total} combinations")
    logger.info("=" * 80)

    results = []
    best_avg = -999.0
    t0 = time.time()

    for i, combo in enumerate(combos):
        params = dict(zip(keys, combo))
        params["min_bars"] = 60

        try:
            wf = run_yearly_wf(
                provider_cfg=PROVIDER_CFG,
                strategy_cls=SkewTradeV2Strategy,
                strategy_params=params,
                years=YEARS,
                costs=DERIBIT_REALISTIC,
                use_options_pnl=True,
                bars_per_year=365,
                seed=42,
            )
            summary = wf["summary"]
            result = {
                "rank": 0,
                "params": {k: v for k, v in params.items() if k != "min_bars"},
                "avg_sharpe": summary["avg_sharpe"],
                "min_sharpe": summary["min_sharpe"],
                "years_positive": summary["years_positive"],
                "yearly": {yr: wf["yearly"][yr].get("sharpe", 0) for yr in wf["yearly"]},
            }
            results.append(result)

            if summary["avg_sharpe"] > best_avg:
                best_avg = summary["avg_sharpe"]

            if (i + 1) % 50 == 0 or i == 0:
                elapsed = time.time() - t0
                rate = (i + 1) / elapsed if elapsed > 0 else 0
                eta = (total - i - 1) / rate if rate > 0 else 0
                logger.info(
                    f"  [{i+1}/{total}] best_so_far={best_avg:.3f} "
                    f"| {elapsed:.0f}s elapsed, ~{eta:.0f}s remaining"
                )
        except Exception as e:
            logger.warning(f"  [{i+1}/{total}] FAILED: {e}")
            results.append({"rank": 0, "params": params, "avg_sharpe": -999, "min_sharpe": -999, "error": str(e)})

    elapsed = time.time() - t0
    results.sort(key=lambda r: r["avg_sharpe"], reverse=True)
    for i, r in enumerate(results):
        r["rank"] = i + 1

    logger.info("=" * 80)
    logger.info(f"TOP {top_n} RESULTS (of {total}):")
    logger.info("=" * 80)
    for r in results[:top_n]:
        p = r["params"]
        yearly = r.get("yearly", {})
        yearly_str = " ".join(f"{yr}={s:.2f}" for yr, s in sorted(yearly.items()))
        logger.info(
            f"  #{r['rank']:3d} | avg={r['avg_sharpe']:+.3f} min={r['min_sharpe']:+.3f} "
            f"| lb={p.get('skew_lookback', '?'):2} ze={p.get('z_entry', '?'):.1f} "
            f"zx={p.get('z_exit', '?'):.1f} rb={p.get('rebalance_freq', '?'):2} "
            f"vrp={p.get('use_vrp_filter', '?')} lev={p.get('target_leverage', '?'):.1f} "
            f"| {yearly_str}"
        )

    # Check pass criteria
    passed = [r for r in results if r["avg_sharpe"] >= 0.5 and r.get("min_sharpe", -999) >= 0.0]
    logger.info("-" * 80)
    logger.info(f"PASSED (avg>=0.5, min>=0.0): {len(passed)} / {total} combos")
    if passed:
        logger.info(f"  BEST passing: avg={passed[0]['avg_sharpe']:+.3f} min={passed[0]['min_sharpe']:+.3f}")

    return results


def save_results(baselines: list, sweep: list):
    out_dir = ROOT / "artifacts" / "crypto_options"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "skew_v2_sweep_results.json"
    with open(out_path, "w") as f:
        json.dump({
            "baselines": baselines,
            "sweep_total": len(sweep),
            "top_30": sweep[:30],
            "best": sweep[0] if sweep else None,
        }, f, indent=2)
    logger.info(f"Results saved: {out_path}")


def main():
    quick = "--quick" in sys.argv
    grid = QUICK_GRID if quick else FULL_GRID

    logger.info("=" * 80)
    logger.info(f"NEXUS — Skew MR v2 Sweep ({'QUICK' if quick else 'FULL'})")
    logger.info("=" * 80)

    baselines = run_baselines()
    sweep = run_sweep(grid)
    save_results(baselines, sweep)


if __name__ == "__main__":
    main()
