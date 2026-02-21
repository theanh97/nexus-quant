#!/usr/bin/env python3
"""
IV-Percentile Sizing on VRP+Skew Ensemble — Research Script
=============================================================

Follow-up to R25 (IV sizing on VRP alone: +0.258).
Tests whether IV-percentile sizing improves the COMBINED ensemble.

Approach:
  1. Run standard VRP + Skew MR ensemble (VRP 40% + Skew 60%)
  2. Scale total position by BTC IV percentile (step function)
  3. Compare with unscaled ensemble baseline

Also tests: sizing only the VRP component vs sizing both.
"""
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from nexus_quant.backtest.costs import ExecutionCostModel, FeeModel, ImpactModel
from nexus_quant.projects.crypto_options.options_engine import (
    OptionsBacktestEngine, compute_metrics
)
from nexus_quant.projects.crypto_options.providers.deribit_rest import DeribitRestProvider
from nexus_quant.projects.crypto_options.strategies.variance_premium import VariancePremiumStrategy
from nexus_quant.projects.crypto_options.strategies.skew_trade_v2 import SkewTradeV2Strategy

# ── Cost model ─────────────────────────────────────────────────────────────

fees = FeeModel(maker_fee_rate=0.0003, taker_fee_rate=0.0005)
impact = ImpactModel(model="sqrt", coef_bps=2.0)
COSTS = ExecutionCostModel(fee=fees, impact=impact)

YEARS = [2021, 2022, 2023, 2024, 2025]
SEED = 42
BARS_PER_YEAR = 365

# Ensemble weights (from wisdom v10)
W_VRP = 0.40
W_SKEW = 0.60

VRP_PARAMS = {
    "base_leverage": 1.5,
    "exit_z_threshold": -3.0,
    "vrp_lookback": 30,
    "rebalance_freq": 5,
    "min_bars": 30,
}

SKEW_PARAMS = {
    "skew_lookback": 60,
    "z_entry": 2.0,
    "z_exit": 0.0,
    "target_leverage": 1.0,
    "rebalance_freq": 5,
    "min_bars": 60,
}


def iv_percentile(iv_series: List, idx: int, lookback: int) -> Optional[float]:
    """Compute IV percentile rank over lookback window."""
    if idx < lookback or not iv_series:
        return None
    start = max(0, idx - lookback)
    window = [v for v in iv_series[start:idx] if v is not None]
    if len(window) < 10:
        return None
    current = iv_series[idx] if idx < len(iv_series) and iv_series[idx] is not None else None
    if current is None:
        return None
    below = sum(1 for v in window if v < current)
    return below / len(window)


def step_sizing(pct: float) -> float:
    """Step function: <25th→0.5x, 25-75th→1.0x, >75th→1.5x."""
    if pct < 0.25:
        return 0.5
    elif pct > 0.75:
        return 1.5
    return 1.0


def run_ensemble_wf(sizing_mode: str = "none", lookback: int = 180) -> Dict[str, Any]:
    """
    Run VRP+Skew ensemble with optional IV sizing.

    sizing_mode:
      "none": no sizing (baseline)
      "both": scale both VRP and Skew by IV percentile
      "vrp_only": scale only VRP component
    """
    sharpes = []
    yearly_detail = {}

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
        n = len(dataset.timeline)

        vrp_strat = VariancePremiumStrategy(params=VRP_PARAMS)
        skew_strat = SkewTradeV2Strategy(params=SKEW_PARAMS)

        dt = 1.0 / BARS_PER_YEAR
        equity = 1.0
        vrp_weights = {"BTC": 0.0}
        skew_weights = {"BTC": 0.0}
        equity_curve = [1.0]
        returns_list = []

        for idx in range(1, n):
            prev_equity = equity
            sym = "BTC"

            # -- VRP P&L --
            vrp_pnl = 0.0
            w_v = vrp_weights.get(sym, 0.0)
            if abs(w_v) > 1e-10:
                closes = dataset.perp_close.get(sym, [])
                if idx < len(closes) and closes[idx - 1] > 0 and closes[idx] > 0:
                    log_ret = math.log(closes[idx] / closes[idx - 1])
                    rv_bar = abs(log_ret) * math.sqrt(BARS_PER_YEAR)
                else:
                    rv_bar = 0.0
                iv_series = dataset.features.get("iv_atm", {}).get(sym, [])
                iv = None
                if iv_series and idx - 1 < len(iv_series):
                    iv = iv_series[idx - 1]
                if iv and iv > 0:
                    vrp = 0.5 * (iv ** 2 - rv_bar ** 2) * dt
                    vrp_pnl = (-w_v) * vrp

            # -- Skew P&L --
            skew_pnl = 0.0
            w_s = skew_weights.get(sym, 0.0)
            if abs(w_s) > 1e-10:
                skew_series = dataset.features.get("skew_25d", {}).get(sym, [])
                if idx < len(skew_series) and idx - 1 < len(skew_series):
                    s_now = skew_series[idx]
                    s_prev = skew_series[idx - 1]
                    if s_now is not None and s_prev is not None:
                        d_skew = float(s_now) - float(s_prev)
                        iv_series = dataset.features.get("iv_atm", {}).get(sym, [])
                        iv_s = iv_series[idx - 1] if iv_series and idx - 1 < len(iv_series) else 0.70
                        if iv_s and iv_s > 0:
                            sensitivity = iv_s * math.sqrt(dt) * 2.5  # skew_sensitivity_mult=2.5
                            skew_pnl = w_s * d_skew * sensitivity

            bar_pnl = W_VRP * vrp_pnl + W_SKEW * skew_pnl
            dp = equity * bar_pnl
            equity += dp

            # -- Rebalance --
            if vrp_strat.should_rebalance(dataset, idx):
                target_v = vrp_strat.target_weights(dataset, idx, vrp_weights)
                target_s = skew_strat.target_weights(dataset, idx, skew_weights) if skew_strat.should_rebalance(dataset, idx) else skew_weights

                # Apply IV sizing
                if sizing_mode != "none":
                    iv_series = dataset.features.get("iv_atm", {}).get(sym, [])
                    pct = iv_percentile(iv_series, idx, lookback)
                    if pct is not None:
                        scale = step_sizing(pct)
                        if sizing_mode in ("both", "vrp_only"):
                            for s in target_v:
                                target_v[s] *= scale
                        if sizing_mode == "both":
                            for s in target_s:
                                target_s[s] *= scale

                # Cost for total turnover
                old_total = {sym: W_VRP * vrp_weights.get(sym, 0) + W_SKEW * skew_weights.get(sym, 0)}
                new_total = {sym: W_VRP * target_v.get(sym, 0) + W_SKEW * target_s.get(sym, 0)}
                turnover = sum(abs(new_total[s] - old_total[s]) for s in [sym])
                bd = COSTS.cost(equity=equity, turnover=turnover)
                cost = float(bd.get("cost", 0.0))
                equity -= cost
                equity = max(equity, 0.0)

                vrp_weights = target_v
                skew_weights = target_s
            elif skew_strat.should_rebalance(dataset, idx):
                target_s = skew_strat.target_weights(dataset, idx, skew_weights)
                if sizing_mode == "both":
                    iv_series = dataset.features.get("iv_atm", {}).get(sym, [])
                    pct = iv_percentile(iv_series, idx, lookback)
                    if pct is not None:
                        scale = step_sizing(pct)
                        for s in target_s:
                            target_s[s] *= scale

                old_total = {sym: W_VRP * vrp_weights.get(sym, 0) + W_SKEW * skew_weights.get(sym, 0)}
                new_total = {sym: W_VRP * vrp_weights.get(sym, 0) + W_SKEW * target_s.get(sym, 0)}
                turnover = sum(abs(new_total[s] - old_total[s]) for s in [sym])
                bd = COSTS.cost(equity=equity, turnover=turnover)
                cost = float(bd.get("cost", 0.0))
                equity -= cost
                equity = max(equity, 0.0)

                skew_weights = target_s

            equity_curve.append(equity)
            bar_ret = (equity / prev_equity) - 1.0 if prev_equity > 0 else 0.0
            returns_list.append(bar_ret)

        m = compute_metrics(equity_curve, returns_list, BARS_PER_YEAR)
        sharpes.append(m["sharpe"])
        yearly_detail[str(yr)] = m["sharpe"]

    avg = sum(sharpes) / len(sharpes)
    mn = min(sharpes)
    return {
        "avg_sharpe": round(avg, 3),
        "min_sharpe": round(mn, 3),
        "yearly": yearly_detail,
    }


def main():
    print("=" * 70)
    print("IV-PERCENTILE SIZING ON VRP+SKEW ENSEMBLE — RESEARCH")
    print("=" * 70)
    print(f"Ensemble: VRP {W_VRP*100:.0f}% + Skew MR {W_SKEW*100:.0f}%")
    print(f"Sizing: step function (0.5x/1.0x/1.5x), 180d lookback")
    print(f"Years: {YEARS}")
    print()

    # 1. Baseline (no sizing)
    print("Running baseline (no IV sizing)...")
    baseline = run_ensemble_wf("none")
    print(f"  Baseline:   avg={baseline['avg_sharpe']:.3f} min={baseline['min_sharpe']:.3f}")
    print(f"  Yearly: {baseline['yearly']}")
    print()

    # 2. Size both components
    print("Running sized (BOTH VRP+Skew scaled)...")
    both = run_ensemble_wf("both", 180)
    d_both = both["avg_sharpe"] - baseline["avg_sharpe"]
    print(f"  Both sized: avg={both['avg_sharpe']:.3f} min={both['min_sharpe']:.3f} Δ={d_both:+.3f}")
    print(f"  Yearly: {both['yearly']}")
    print()

    # 3. Size only VRP
    print("Running sized (VRP only scaled)...")
    vrp_only = run_ensemble_wf("vrp_only", 180)
    d_vrp = vrp_only["avg_sharpe"] - baseline["avg_sharpe"]
    print(f"  VRP only:   avg={vrp_only['avg_sharpe']:.3f} min={vrp_only['min_sharpe']:.3f} Δ={d_vrp:+.3f}")
    print(f"  Yearly: {vrp_only['yearly']}")
    print()

    # 4. Try different lookbacks for "both"
    print("Lookback sweep (both sized):")
    lb_results = {}
    for lb in [60, 90, 120, 180, 240]:
        r = run_ensemble_wf("both", lb)
        d = r["avg_sharpe"] - baseline["avg_sharpe"]
        lb_results[lb] = r
        print(f"  lb={lb:3d}d  avg={r['avg_sharpe']:.3f} min={r['min_sharpe']:.3f} Δ={d:+.3f}")

    # Summary
    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  Baseline ensemble:  avg={baseline['avg_sharpe']:.3f} min={baseline['min_sharpe']:.3f}")
    print(f"  Both sized lb180:   avg={both['avg_sharpe']:.3f} min={both['min_sharpe']:.3f} Δ={d_both:+.3f}")
    print(f"  VRP-only sized:     avg={vrp_only['avg_sharpe']:.3f} min={vrp_only['min_sharpe']:.3f} Δ={d_vrp:+.3f}")

    best_lb = max(lb_results.items(), key=lambda x: x[1]["avg_sharpe"])
    best_lb_d = best_lb[1]["avg_sharpe"] - baseline["avg_sharpe"]
    print(f"  Best lookback:      lb={best_lb[0]}d avg={best_lb[1]['avg_sharpe']:.3f} Δ={best_lb_d:+.3f}")

    # Verdict
    print()
    best_d = max(d_both, d_vrp)
    if best_d > 0.1:
        print(f"VERDICT: IV sizing IMPROVES ensemble (Δ={best_d:+.3f})")
    elif best_d > 0:
        print(f"VERDICT: IV sizing is MARGINAL on ensemble (Δ={best_d:+.3f})")
    else:
        print(f"VERDICT: IV sizing does NOT improve ensemble (Δ={best_d:+.3f})")
    print("=" * 70)

    # Save
    out = ROOT / "artifacts" / "crypto_options" / "iv_percentile_ensemble_research.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump({
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "baseline": baseline,
            "both_sized_lb180": both,
            "vrp_only_sized_lb180": vrp_only,
            "lookback_sweep": {str(k): v for k, v in lb_results.items()},
        }, f, indent=2)
    print(f"\nResults saved to: {out}")


if __name__ == "__main__":
    main()
