#!/usr/bin/env python3
"""
Phase 118: Binance Vision Metrics Overlay R&D

Tests 5 positioning metrics (from data.binance.vision bulk data, 4.2 years)
as P91b champion overlays — same architecture as the validated vol_mom_z_168 tilt.

Metrics tested:
  1. OI momentum z-score (open interest rate of change)
  2. Global L/S ratio z-score (crowd positioning extreme)
  3. Top trader L/S position z-score (smart money positioning)
  4. Top trader L/S account z-score (smart money account counts)
  5. Taker buy/sell ratio z-score (order flow imbalance)

For each metric, we test as a "reduce leverage when extreme" overlay:
  - Compute rolling z-score of the metric's momentum
  - When z > 0 (crowded/extreme), reduce all weights by tilt_ratio
  - Same logic as the validated vol_mom_z_168 tilt

Walk-forward validation:
  IS: 2022-01-01 to 2024-12-31 (3 years)
  OOS: 2025-01-01 to 2026-02-20 (14 months)
  MIN Sharpe across annual periods is the binding constraint
"""
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

# Project setup
PROJ_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJ_ROOT))

from nexus_quant.data.providers.registry import make_provider
from nexus_quant.data.providers.binance_vision_metrics import (
    load_vision_metrics,
    merge_metrics_into_dataset,
)
from nexus_quant.backtest.engine import BacktestConfig, BacktestEngine
from nexus_quant.backtest.costs import cost_model_from_config
from nexus_quant.strategies.registry import make_strategy

# ── Configuration ──────────────────────────────────────────────────────────
SYMBOLS = [
    "BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT",
    "ADAUSDT", "DOGEUSDT", "AVAXUSDT", "DOTUSDT", "LINKUSDT",
]

# Walk-forward periods
PERIODS = {
    "2022": ("2022-01-01", "2023-01-01"),
    "2023": ("2023-01-01", "2024-01-01"),
    "2024": ("2024-01-01", "2025-01-01"),
    "2025_oos": ("2025-01-01", "2026-01-01"),
    "2026_ytd": ("2026-01-01", "2026-02-20"),
}

# P91b ensemble config
ENSEMBLE_CFG = {
    "weights": {
        "v1": 0.2747,
        "I460_bw168_k4": 0.1967,
        "I415_bw216_k4": 0.3247,
        "F144_k2": 0.2039,
    },
    "signals": {
        "v1": {"strategy": "nexus_alpha_v1_long", "params": {}},
        "I460_bw168_k4": {
            "strategy": "nexus_alpha_v1_long",
            "params": {"momentum_lookback": 460, "bandwidth": 168, "k_factor": 4},
        },
        "I415_bw216_k4": {
            "strategy": "nexus_alpha_v1_long",
            "params": {"momentum_lookback": 415, "bandwidth": 216, "k_factor": 4},
        },
        "F144_k2": {
            "strategy": "nexus_alpha_v1_long",
            "params": {"funding_lookback": 144, "k_factor": 2},
        },
    },
}

COST_CFG = {
    "maker_bps": 2.0,
    "taker_bps": 5.0,
    "funding_multiplier": 1.0,
    "slippage_bps": 1.0,
}

# Overlay variants to test
OVERLAYS = [
    # (name, metric_key, lookback, tilt_ratio, direction)
    # direction: "reduce_on_high" = reduce leverage when z > 0
    #            "reduce_on_low"  = reduce leverage when z < 0
    ("oi_mom_z168", "open_interest", 168, 0.65, "reduce_on_high"),
    ("oi_mom_z168_low", "open_interest", 168, 0.65, "reduce_on_low"),
    ("global_ls_z168", "global_ls_ratio", 168, 0.65, "reduce_on_high"),
    ("global_ls_z168_low", "global_ls_ratio", 168, 0.65, "reduce_on_low"),
    ("top_ls_pos_z168", "top_ls_position", 168, 0.65, "reduce_on_high"),
    ("top_ls_acct_z168", "top_ls_account", 168, 0.65, "reduce_on_high"),
    ("taker_ls_z168", "taker_ls_ratio", 168, 0.65, "reduce_on_high"),
    ("taker_ls_z168_low", "taker_ls_ratio", 168, 0.65, "reduce_on_low"),
    # Shorter lookback variants
    ("oi_mom_z72", "open_interest", 72, 0.65, "reduce_on_high"),
    ("global_ls_z72", "global_ls_ratio", 72, 0.65, "reduce_on_high"),
    ("taker_ls_z72", "taker_ls_ratio", 72, 0.65, "reduce_on_high"),
]


def compute_metric_z_scores(
    metric_values: dict,  # {epoch_s: float}
    timeline: list,
    lookback: int,
) -> np.ndarray:
    """
    Compute momentum z-score for a metric at each bar in the timeline.
    Same algorithm as vol_mom_z_168.
    """
    n = len(timeline)
    z_scores = np.zeros(n)

    # Build aligned array
    vals = np.zeros(n)
    last_v = 0.0
    for i, t in enumerate(timeline):
        if t in metric_values:
            last_v = metric_values[t]
        vals[i] = last_v

    # Log transform (for OI which can be very large)
    log_vals = np.log(np.maximum(vals, 1.0))

    for i in range(lookback * 2, n):
        # Momentum at bar i
        mom_i = log_vals[i] - log_vals[i - lookback]

        # Rolling momentum stats
        moms = []
        for j in range(max(lookback, i - lookback), i + 1):
            if j >= lookback:
                moms.append(log_vals[j] - log_vals[j - lookback])

        if len(moms) < 10:
            continue

        mu = float(np.mean(moms))
        sigma = float(np.std(moms))
        if sigma > 0:
            z_scores[i] = (mom_i - mu) / sigma

    return z_scores


def run_p91b_backtest(dataset, period_name: str):
    """Run P91b base ensemble and return hourly returns array."""
    cost_model = cost_model_from_config(COST_CFG)
    engine = BacktestEngine(BacktestConfig(costs=cost_model))

    # Create P91b ensemble strategy
    strat = make_strategy({
        "name": "p91b_ensemble",
        "params": {
            "ensemble_weights": ENSEMBLE_CFG["weights"],
            "signal_configs": ENSEMBLE_CFG["signals"],
        },
    })

    result = engine.run(dataset, strat)
    return np.array(result.returns, dtype=np.float64), np.array(result.equity_curve, dtype=np.float64)


def apply_overlay(
    base_returns: np.ndarray,
    z_scores: np.ndarray,
    tilt_ratio: float,
    direction: str,
) -> np.ndarray:
    """
    Apply overlay to base returns.
    When condition triggers, returns are scaled by tilt_ratio.
    """
    n = len(base_returns)
    tilted = base_returns.copy()

    for i in range(1, n):
        if direction == "reduce_on_high" and z_scores[i - 1] > 0:
            tilted[i] *= tilt_ratio
        elif direction == "reduce_on_low" and z_scores[i - 1] < 0:
            tilted[i] *= tilt_ratio

    return tilted


def compute_sharpe(returns) -> float:
    rets = np.array(returns, dtype=np.float64)
    if len(rets) < 100:
        return 0.0
    rets = rets[~np.isnan(rets)]
    if len(rets) < 100 or np.std(rets) == 0:
        return 0.0
    return float(np.mean(rets) / np.std(rets) * np.sqrt(8760))


def compute_mdd(equity) -> float:
    eq = np.array(equity, dtype=np.float64)
    peak = np.maximum.accumulate(eq)
    dd = (eq - peak) / np.maximum(peak, 1e-10)
    return float(dd.min() * 100)


def main():
    t0 = time.time()
    print("=" * 70)
    print("  PHASE 118: BINANCE VISION METRICS OVERLAY R&D")
    print("  Testing 5 positioning metrics as P91b overlays")
    print("=" * 70)
    print()

    results_dir = PROJ_ROOT / "artifacts" / "phase118_metrics"
    results_dir.mkdir(parents=True, exist_ok=True)

    all_results = {}

    for period_name, (start, end) in PERIODS.items():
        print(f"\n{'='*60}")
        print(f"  PERIOD: {period_name} ({start} → {end})")
        print(f"{'='*60}")

        # Skip 2022 for now if metrics data starts 2021-12-01
        # (we need lookback warmup)

        # 1. Load klines + funding
        print("\n  [1/3] Loading kline data from Binance...")
        provider = make_provider({
            "provider": "binance_rest_v1",
            "symbols": SYMBOLS,
            "start": start,
            "end": end,
            "bar_interval": "1h",
        }, seed=42)
        dataset = provider.load()
        n_bars = len(dataset.timeline)
        print(f"    -> {n_bars} bars loaded")

        # 2. Load vision metrics
        print("\n  [2/3] Loading Binance Vision metrics...")
        metrics = load_vision_metrics(SYMBOLS, start, end)
        dataset_with_metrics = merge_metrics_into_dataset(dataset, metrics)

        # 3. Run base P91b
        print("\n  [3/3] Running P91b base backtest...")
        base_rets, base_eq = run_p91b_backtest(dataset_with_metrics, period_name)
        base_sharpe = compute_sharpe(base_rets)
        base_mdd = compute_mdd(base_eq)
        print(f"    Base P91b: Sharpe={base_sharpe:.4f}, MDD={base_mdd:.2f}%")

        period_results = {
            "base_sharpe": round(base_sharpe, 4),
            "base_mdd": round(base_mdd, 2),
            "overlays": {},
        }

        # 4. Test each overlay
        for overlay_name, metric_key, lookback, tilt_ratio, direction in OVERLAYS:
            # Get the metric data for aggregation across symbols
            # For OI: sum across symbols (total market OI)
            # For ratios: average across symbols (market-wide sentiment)
            agg_metric: dict = {}

            for sym in SYMBOLS:
                sym_m = metrics.get(sym, {}).get(metric_key, {})
                for ts, val in sym_m.items():
                    if ts not in agg_metric:
                        agg_metric[ts] = []
                    agg_metric[ts].append(val)

            # Aggregate
            agg_values: dict = {}
            for ts, vals in agg_metric.items():
                if metric_key == "open_interest":
                    agg_values[ts] = sum(vals)  # sum OI
                else:
                    agg_values[ts] = sum(vals) / len(vals)  # mean for ratios

            # Compute z-scores
            z_scores = compute_metric_z_scores(
                agg_values, dataset.timeline, lookback
            )

            # Apply overlay
            tilted_rets = apply_overlay(base_rets, z_scores, tilt_ratio, direction)
            tilted_eq = np.cumprod(1.0 + tilted_rets)
            tilted_sharpe = compute_sharpe(tilted_rets)
            tilted_mdd = compute_mdd(tilted_eq)

            # Tilt stats
            active_bars = 0
            for i in range(1, len(z_scores)):
                if direction == "reduce_on_high" and z_scores[i - 1] > 0:
                    active_bars += 1
                elif direction == "reduce_on_low" and z_scores[i - 1] < 0:
                    active_bars += 1
            tilt_pct = active_bars / max(len(z_scores) - 1, 1) * 100

            delta = tilted_sharpe - base_sharpe
            verdict = "BETTER" if delta > 0.05 else "WORSE" if delta < -0.05 else "NEUTRAL"

            print(f"    {overlay_name:25s}: Sharpe={tilted_sharpe:+.4f} (Δ={delta:+.4f}) MDD={tilted_mdd:.2f}% tilt={tilt_pct:.1f}% → {verdict}")

            period_results["overlays"][overlay_name] = {
                "sharpe": round(tilted_sharpe, 4),
                "mdd": round(tilted_mdd, 2),
                "delta_sharpe": round(delta, 4),
                "tilt_pct": round(tilt_pct, 1),
                "verdict": verdict,
            }

        all_results[period_name] = period_results

    # ── Summary ────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  WALK-FORWARD SUMMARY")
    print("=" * 70)

    # For each overlay, compute IS avg Sharpe delta, OOS Sharpe delta, and MIN delta
    overlay_names = [o[0] for o in OVERLAYS]
    summary_rows = []

    for name in overlay_names:
        is_deltas = []
        oos_delta = None
        annual_sharpes = []

        for period_name, pr in all_results.items():
            ov = pr["overlays"].get(name, {})
            delta = ov.get("delta_sharpe", 0)

            if period_name in ("2022", "2023", "2024"):
                is_deltas.append(delta)

            if period_name == "2025_oos":
                oos_delta = delta

            if period_name in ("2022", "2023", "2024", "2025_oos"):
                annual_sharpes.append(ov.get("sharpe", 0))

        is_avg = sum(is_deltas) / len(is_deltas) if is_deltas else 0
        min_sharpe = min(annual_sharpes) if annual_sharpes else 0
        oos_d = oos_delta if oos_delta is not None else 0

        summary_rows.append({
            "name": name,
            "is_avg_delta": round(is_avg, 4),
            "oos_delta": round(oos_d, 4),
            "min_sharpe": round(min_sharpe, 4),
            "pass_wf": is_avg > 0 and oos_d > 0,
        })

    print(f"\n  {'Overlay':<25s} {'IS Δ':>8s} {'OOS Δ':>8s} {'MIN':>8s} {'WF?':>5s}")
    print(f"  {'-'*25} {'-'*8} {'-'*8} {'-'*8} {'-'*5}")
    for r in sorted(summary_rows, key=lambda x: x["oos_delta"], reverse=True):
        wf = "PASS" if r["pass_wf"] else "FAIL"
        print(f"  {r['name']:<25s} {r['is_avg_delta']:+.4f} {r['oos_delta']:+.4f} {r['min_sharpe']:.4f} {wf:>5s}")

    # Save results
    output = {
        "phase": "118_metrics_overlay",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": round(time.time() - t0, 1),
        "periods": all_results,
        "summary": summary_rows,
    }
    out_path = results_dir / "phase118_results.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Results saved → {out_path}")

    elapsed = time.time() - t0
    print(f"\n  Total time: {elapsed:.0f}s ({elapsed/60:.1f}m)")


if __name__ == "__main__":
    main()
