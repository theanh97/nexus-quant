"""
VRP v2 Overlay Sweep — Test each overlay signal individually and combined
=========================================================================
Tests 5 configurations:
  A. v1 Baseline (no overlays, best v1 params: exit_z=-3.0, lb=30)
  B. Z-Score Scaling only
  C. IV Percentile Rank only
  D. Vol Regime only
  E. All overlays combined
  F. Full grid sweep over overlay parameters

For each, runs walk-forward validation 2021-2025.

Usage:
    python3 scripts/run_vrp_v2_overlay_sweep.py
    python3 scripts/run_vrp_v2_overlay_sweep.py --full   # extended grid
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
logger = logging.getLogger("vrp_v2_sweep")

from nexus_quant.projects.crypto_options.strategies.variance_premium import VariancePremiumStrategy
from nexus_quant.projects.crypto_options.strategies.variance_premium_v2 import VariancePremiumV2Strategy
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


def run_config(name: str, strategy_cls, params: dict) -> dict:
    """Run one configuration through walk-forward."""
    t0 = time.time()
    wf = run_yearly_wf(
        provider_cfg=PROVIDER_CFG,
        strategy_cls=strategy_cls,
        strategy_params=params,
        years=YEARS,
        costs=DERIBIT_REALISTIC,
        use_options_pnl=True,
        bars_per_year=365,
        seed=42,
    )
    elapsed = time.time() - t0
    summary = wf["summary"]
    yearly = wf["yearly"]
    yearly_str = " ".join(f"{yr}={yearly[yr].get('sharpe', 0):.2f}" for yr in sorted(yearly))

    logger.info(
        f"  {name:40s} | avg={summary['avg_sharpe']:+.3f} min={summary['min_sharpe']:+.3f} "
        f"| {yearly_str} | {elapsed:.1f}s"
    )
    return {
        "name": name,
        "params": params,
        "avg_sharpe": summary["avg_sharpe"],
        "min_sharpe": summary["min_sharpe"],
        "years_positive": summary["years_positive"],
        "passed": summary["passed"],
        "yearly": {yr: yearly[yr] for yr in yearly},
        "elapsed": round(elapsed, 1),
    }


def run_ablation():
    """Test each overlay individually and combined."""
    logger.info("=" * 90)
    logger.info("VRP v2 OVERLAY ABLATION STUDY")
    logger.info("=" * 90)

    results = []

    # A. v1 Baseline — best from sweep (exit_z=-3.0, lb=30)
    results.append(run_config(
        "A. v1 Baseline (exit_z=-3.0)",
        VariancePremiumStrategy,
        {"base_leverage": 1.5, "exit_z_threshold": -3.0, "vrp_lookback": 30, "rebalance_freq": 5, "min_bars": 30},
    ))

    # B. v1 Original (exit_z=-2.0)
    results.append(run_config(
        "B. v1 Original (exit_z=-2.0)",
        VariancePremiumStrategy,
        {"base_leverage": 1.5, "exit_z_threshold": -2.0, "vrp_lookback": 60, "rebalance_freq": 5, "min_bars": 30},
    ))

    # C. v2 Z-Score Scaling only
    results.append(run_config(
        "C. v2 Z-Scale only",
        VariancePremiumV2Strategy,
        {
            "base_leverage": 1.5, "vrp_lookback": 30, "rebalance_freq": 5, "min_bars": 30,
            "use_z_scaling": True, "z_base": 0.7, "z_slope": 0.25, "z_min": 0.0, "z_max": 1.2,
            "use_iv_pct": False, "use_vol_regime": False,
        },
    ))

    # D. v2 IV Percentile only
    results.append(run_config(
        "D. v2 IV Percentile only",
        VariancePremiumV2Strategy,
        {
            "base_leverage": 1.5, "vrp_lookback": 30, "rebalance_freq": 5, "min_bars": 30,
            "use_z_scaling": False, "use_iv_pct": True, "use_vol_regime": False,
            "iv_pct_lookback": 252, "iv_pct_low": 0.25, "iv_pct_high": 0.75,
            "iv_scale_low": 0.5, "iv_scale_high": 1.3,
        },
    ))

    # E. v2 Vol Regime only
    results.append(run_config(
        "E. v2 Vol Regime only",
        VariancePremiumV2Strategy,
        {
            "base_leverage": 1.5, "vrp_lookback": 30, "rebalance_freq": 5, "min_bars": 30,
            "use_z_scaling": False, "use_iv_pct": False, "use_vol_regime": True,
            "vol_regime_lookback": 60, "vol_high_threshold": 0.80, "vol_scale_high": 0.4,
        },
    ))

    # F. v2 Z-Scale + IV Pct (no vol regime)
    results.append(run_config(
        "F. v2 Z-Scale + IV Pct",
        VariancePremiumV2Strategy,
        {
            "base_leverage": 1.5, "vrp_lookback": 30, "rebalance_freq": 5, "min_bars": 30,
            "use_z_scaling": True, "z_base": 0.7, "z_slope": 0.25, "z_min": 0.0, "z_max": 1.2,
            "use_iv_pct": True, "iv_pct_lookback": 252, "iv_pct_low": 0.25, "iv_pct_high": 0.75,
            "iv_scale_low": 0.5, "iv_scale_high": 1.3,
            "use_vol_regime": False,
        },
    ))

    # G. v2 All overlays combined
    results.append(run_config(
        "G. v2 ALL overlays",
        VariancePremiumV2Strategy,
        {
            "base_leverage": 1.5, "vrp_lookback": 30, "rebalance_freq": 5, "min_bars": 30,
            "use_z_scaling": True, "z_base": 0.7, "z_slope": 0.25, "z_min": 0.0, "z_max": 1.2,
            "use_iv_pct": True, "iv_pct_lookback": 252, "iv_pct_low": 0.25, "iv_pct_high": 0.75,
            "iv_scale_low": 0.5, "iv_scale_high": 1.3,
            "use_vol_regime": True, "vol_regime_lookback": 60, "vol_high_threshold": 0.80, "vol_scale_high": 0.4,
        },
    ))

    # Summary
    logger.info("=" * 90)
    logger.info("ABLATION SUMMARY (sorted by avg_sharpe):")
    logger.info("=" * 90)
    results.sort(key=lambda r: r["avg_sharpe"], reverse=True)
    for i, r in enumerate(results):
        delta = r["avg_sharpe"] - results[-1]["avg_sharpe"]
        logger.info(
            f"  #{i+1} {r['name']:40s} avg={r['avg_sharpe']:+.3f} min={r['min_sharpe']:+.3f} Δ={delta:+.3f}"
        )

    return results


def run_overlay_grid():
    """Grid sweep over overlay parameters."""
    logger.info("=" * 90)
    logger.info("VRP v2 OVERLAY PARAMETER GRID SWEEP")
    logger.info("=" * 90)

    # Key params to sweep
    grid = {
        "z_base": [0.5, 0.7, 0.9],
        "z_slope": [0.15, 0.25, 0.35],
        "z_min": [0.0, 0.1, 0.2],
        "iv_pct_lookback": [120, 252],
        "iv_pct_low": [0.20, 0.30],
        "iv_pct_high": [0.70, 0.80],
        "vol_high_threshold": [0.70, 0.80, 0.90],
        "vol_scale_high": [0.3, 0.5],
    }

    keys = list(grid.keys())
    values = list(grid.values())
    combos = list(itertools.product(*values))
    total = len(combos)
    logger.info(f"Grid: {total} combinations")

    results = []
    best_avg = 0.0
    t0 = time.time()

    for i, combo in enumerate(combos):
        overlay_params = dict(zip(keys, combo))

        params = {
            "base_leverage": 1.5, "vrp_lookback": 30, "rebalance_freq": 5, "min_bars": 30,
            "use_z_scaling": True, "use_iv_pct": True, "use_vol_regime": True,
            "z_max": 1.2,
            "iv_scale_low": 0.5, "iv_scale_high": 1.3,
            **overlay_params,
        }

        try:
            wf = run_yearly_wf(
                provider_cfg=PROVIDER_CFG,
                strategy_cls=VariancePremiumV2Strategy,
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
                "params": overlay_params,
                "avg_sharpe": summary["avg_sharpe"],
                "min_sharpe": summary["min_sharpe"],
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

    elapsed = time.time() - t0
    results.sort(key=lambda r: r["avg_sharpe"], reverse=True)
    for i, r in enumerate(results):
        r["rank"] = i + 1

    logger.info("=" * 90)
    logger.info(f"OVERLAY GRID: TOP 15 (of {total})")
    logger.info("=" * 90)
    for r in results[:15]:
        p = r["params"]
        yearly = r.get("yearly", {})
        yearly_str = " ".join(f"{yr}={s:.2f}" for yr, s in sorted(yearly.items()))
        logger.info(
            f"  #{r['rank']:3d} | avg={r['avg_sharpe']:+.3f} min={r['min_sharpe']:+.3f} "
            f"| zb={p['z_base']:.1f} zs={p['z_slope']:.2f} zm={p['z_min']:.1f} "
            f"| ivlb={p['iv_pct_lookback']} ivl={p['iv_pct_low']:.2f} ivh={p['iv_pct_high']:.2f} "
            f"| vt={p['vol_high_threshold']:.2f} vs={p['vol_scale_high']:.1f} "
            f"| {yearly_str}"
        )

    return results


def save_results(ablation: list, grid: list = None):
    """Save all results."""
    out_dir = ROOT / "artifacts" / "crypto_options"
    out_dir.mkdir(parents=True, exist_ok=True)

    data = {
        "sweep": "vrp_v2_overlay",
        "ablation": ablation,
    }
    if grid:
        data["grid_top_30"] = grid[:30]
        data["grid_total"] = len(grid)
        data["best_overlay_params"] = grid[0]["params"] if grid else None

    out_path = out_dir / "vrp_v2_overlay_results.json"
    with open(out_path, "w") as f:
        json.dump(data, f, indent=2)
    logger.info(f"Results saved: {out_path}")


def main():
    full = "--full" in sys.argv

    logger.info("=" * 90)
    logger.info("NEXUS — VRP v2 Overlay Sweep (Crypto Options R&D)")
    logger.info("=" * 90)

    # Phase 1: Ablation study (quick)
    ablation = run_ablation()

    # Phase 2: Grid sweep (if --full or ablation shows improvement)
    grid = None
    best_v2 = max(r["avg_sharpe"] for r in ablation if "v2" in r["name"])
    baseline = next(r["avg_sharpe"] for r in ablation if "v1 Baseline" in r["name"])

    if full or best_v2 > baseline:
        logger.info(f"\nOverlays show improvement (+{best_v2 - baseline:.3f}). Running grid sweep...")
        grid = run_overlay_grid()
    else:
        logger.info(f"\nOverlays did NOT improve over baseline. Skipping grid sweep.")

    save_results(ablation, grid)


if __name__ == "__main__":
    main()
