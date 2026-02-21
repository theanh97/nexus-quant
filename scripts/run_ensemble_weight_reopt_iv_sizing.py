#!/usr/bin/env python3
"""
Ensemble Weight Re-optimization with IV Sizing — R29
=======================================================

The 40/60 VRP/Skew weights were optimized BEFORE IV sizing was added.
With R28's optimized IV sizing (step 25/75, 0.5x/1.7x, lb180), the optimal
ensemble split might be different because:
  1. IV sizing scales both strategies proportionally
  2. VRP (carry) and Skew MR (signal) respond differently to IV-based scaling
  3. Higher-leverage sizing may change the risk/return balance between strategies

Sweep:
  - VRP weight from 0.10 to 0.80 (step 0.05), Skew = 1 - VRP
  - With and without IV sizing, to see how sizing changes optimal split
  - Also test 3-tier: low/medium/high VRP weight

Grid: 15 weight configs × 2 sizing modes = 30 runs
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

# ── Config ─────────────────────────────────────────────────────────────────

fees = FeeModel(maker_fee_rate=0.0003, taker_fee_rate=0.0005)
impact = ImpactModel(model="sqrt", coef_bps=2.0)
COSTS = ExecutionCostModel(fee=fees, impact=impact)

YEARS = [2021, 2022, 2023, 2024, 2025]
SEED = 42
BARS_PER_YEAR = 365

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

# R28 optimized IV sizing
IV_LOOKBACK = 180
IV_PCT_LOW = 0.25
IV_PCT_HIGH = 0.75
IV_SCALE_LOW = 0.50
IV_SCALE_HIGH = 1.70


def iv_percentile(iv_series: List, idx: int) -> Optional[float]:
    lookback = IV_LOOKBACK
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


def iv_sizing_scale(pct: float) -> float:
    if pct < IV_PCT_LOW:
        return IV_SCALE_LOW
    elif pct > IV_PCT_HIGH:
        return IV_SCALE_HIGH
    return 1.0


def run_ensemble(w_vrp: float, w_skew: float, use_iv_sizing: bool) -> Dict[str, Any]:
    """Run VRP+Skew ensemble with given weights and optional IV sizing."""
    sharpes = []
    yearly_detail = {}
    mdd_list = []

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
                iv = iv_series[idx - 1] if iv_series and idx - 1 < len(iv_series) else None
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
                            sensitivity = iv_s * math.sqrt(dt) * 2.5
                            skew_pnl = w_s * d_skew * sensitivity

            bar_pnl = w_vrp * vrp_pnl + w_skew * skew_pnl
            dp = equity * bar_pnl
            equity += dp

            # -- Rebalance --
            if vrp_strat.should_rebalance(dataset, idx):
                target_v = vrp_strat.target_weights(dataset, idx, vrp_weights)
                target_s = skew_strat.target_weights(dataset, idx, skew_weights) if skew_strat.should_rebalance(dataset, idx) else skew_weights

                if use_iv_sizing:
                    iv_series = dataset.features.get("iv_atm", {}).get(sym, [])
                    pct = iv_percentile(iv_series, idx)
                    if pct is not None:
                        scale = iv_sizing_scale(pct)
                        for s in target_v:
                            target_v[s] *= scale
                        for s in target_s:
                            target_s[s] *= scale

                old_total = {sym: w_vrp * vrp_weights.get(sym, 0) + w_skew * skew_weights.get(sym, 0)}
                new_total = {sym: w_vrp * target_v.get(sym, 0) + w_skew * target_s.get(sym, 0)}
                turnover = sum(abs(new_total[s] - old_total[s]) for s in [sym])
                bd = COSTS.cost(equity=equity, turnover=turnover)
                cost = float(bd.get("cost", 0.0))
                equity -= cost
                equity = max(equity, 0.0)

                vrp_weights = target_v
                skew_weights = target_s

            elif skew_strat.should_rebalance(dataset, idx):
                target_s = skew_strat.target_weights(dataset, idx, skew_weights)
                if use_iv_sizing:
                    iv_series = dataset.features.get("iv_atm", {}).get(sym, [])
                    pct = iv_percentile(iv_series, idx)
                    if pct is not None:
                        scale = iv_sizing_scale(pct)
                        for s in target_s:
                            target_s[s] *= scale

                old_total = {sym: w_vrp * vrp_weights.get(sym, 0) + w_skew * skew_weights.get(sym, 0)}
                new_total = {sym: w_vrp * vrp_weights.get(sym, 0) + w_skew * target_s.get(sym, 0)}
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
        yearly_detail[str(yr)] = round(m["sharpe"], 3)
        mdd_list.append(m.get("max_drawdown", 0))

    avg = sum(sharpes) / len(sharpes)
    mn = min(sharpes)
    avg_mdd = sum(mdd_list) / len(mdd_list) if mdd_list else 0
    return {
        "avg_sharpe": round(avg, 3),
        "min_sharpe": round(mn, 3),
        "avg_mdd": round(avg_mdd, 4),
        "yearly": yearly_detail,
    }


def main():
    print("=" * 70)
    print("ENSEMBLE WEIGHT RE-OPTIMIZATION WITH IV SIZING — R29")
    print("=" * 70)
    print(f"IV sizing: step 25/75, {IV_SCALE_LOW}x/{IV_SCALE_HIGH}x, lb{IV_LOOKBACK}")
    print(f"Years: {YEARS}")
    print()

    # Weight grid
    vrp_weights = [round(w, 2) for w in [
        0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50,
        0.55, 0.60, 0.65, 0.70, 0.75, 0.80
    ]]

    # ── 1. Without IV sizing (reproduce and verify current optimal) ───────
    print("--- WITHOUT IV sizing (verify 40/60 is optimal) ---")
    print(f"  {'VRP':>5s} {'Skew':>5s}  {'avg':>7s} {'min':>7s} {'MDD':>7s}")
    print(f"  {'---':>5s} {'---':>5s}  {'---':>7s} {'---':>7s} {'---':>7s}")

    base_results = []
    for w in vrp_weights:
        ws = round(1.0 - w, 2)
        r = run_ensemble(w, ws, use_iv_sizing=False)
        base_results.append({"w_vrp": w, "w_skew": ws, **r})
        marker = " *" if w == 0.40 else ""
        print(f"  {w:5.0%} {ws:5.0%}  {r['avg_sharpe']:7.3f} {r['min_sharpe']:7.3f} {r['avg_mdd']:7.4f}{marker}")

    best_base = max(base_results, key=lambda x: x["avg_sharpe"])
    print(f"\n  Best without IV: VRP={best_base['w_vrp']:.0%} avg={best_base['avg_sharpe']:.3f}")
    print()

    # ── 2. With IV sizing (find new optimal split) ────────────────────────
    print("--- WITH IV sizing (R28: 0.5/1.7x, lb180) ---")
    print(f"  {'VRP':>5s} {'Skew':>5s}  {'avg':>7s} {'min':>7s} {'MDD':>7s} {'Δbase':>7s}")
    print(f"  {'---':>5s} {'---':>5s}  {'---':>7s} {'---':>7s} {'---':>7s} {'---':>7s}")

    iv_results = []
    for w in vrp_weights:
        ws = round(1.0 - w, 2)
        r = run_ensemble(w, ws, use_iv_sizing=True)
        # Find matching base result
        base_match = next(b for b in base_results if b["w_vrp"] == w)
        d = r["avg_sharpe"] - base_match["avg_sharpe"]
        iv_results.append({"w_vrp": w, "w_skew": ws, "delta_vs_base": round(d, 3), **r})
        marker = " *" if w == 0.40 else ""
        print(f"  {w:5.0%} {ws:5.0%}  {r['avg_sharpe']:7.3f} {r['min_sharpe']:7.3f} {r['avg_mdd']:7.4f} {d:+7.3f}{marker}")

    best_iv = max(iv_results, key=lambda x: x["avg_sharpe"])
    current_iv = next(r for r in iv_results if r["w_vrp"] == 0.40)
    print(f"\n  Best with IV: VRP={best_iv['w_vrp']:.0%} avg={best_iv['avg_sharpe']:.3f}")
    print(f"  Current (40/60): avg={current_iv['avg_sharpe']:.3f}")
    print(f"  Improvement: {best_iv['avg_sharpe'] - current_iv['avg_sharpe']:+.3f}")
    print()

    # ── 3. Pareto analysis (Sharpe vs min Sharpe) ─────────────────────────
    print("--- PARETO: Sharpe vs Min-Year Sharpe (with IV sizing) ---")
    # Sort by min_sharpe to find most consistent configs
    by_min = sorted(iv_results, key=lambda x: x["min_sharpe"], reverse=True)
    print(f"  {'VRP':>5s}  {'avg':>7s} {'min':>7s} {'MDD':>7s}")
    for r in by_min[:5]:
        print(f"  {r['w_vrp']:5.0%}  {r['avg_sharpe']:7.3f} {r['min_sharpe']:7.3f} {r['avg_mdd']:7.4f}")
    print(f"  Best min Sharpe: VRP={by_min[0]['w_vrp']:.0%} min={by_min[0]['min_sharpe']:.3f}")
    print()

    # ── SUMMARY ───────────────────────────────────────────────────────────
    print("=" * 70)
    print("SUMMARY — R29: Ensemble Weight Re-optimization")
    print("=" * 70)
    print()
    print(f"  WITHOUT IV sizing:")
    print(f"    Best: VRP={best_base['w_vrp']:.0%}/Skew={best_base['w_skew']:.0%} avg={best_base['avg_sharpe']:.3f} min={best_base['min_sharpe']:.3f}")
    print(f"    Current (40/60): avg={next(b for b in base_results if b['w_vrp'] == 0.40)['avg_sharpe']:.3f}")
    print()
    print(f"  WITH IV sizing (0.5/1.7x):")
    print(f"    Best: VRP={best_iv['w_vrp']:.0%}/Skew={best_iv['w_skew']:.0%} avg={best_iv['avg_sharpe']:.3f} min={best_iv['min_sharpe']:.3f}")
    print(f"    Current (40/60): avg={current_iv['avg_sharpe']:.3f} min={current_iv['min_sharpe']:.3f}")
    print(f"    Δ (best vs current): {best_iv['avg_sharpe'] - current_iv['avg_sharpe']:+.3f}")
    print()

    # IV sizing improvement by weight
    print("  IV SIZING IMPROVEMENT BY WEIGHT:")
    for r in iv_results:
        base_match = next(b for b in base_results if b["w_vrp"] == r["w_vrp"])
        d = r["avg_sharpe"] - base_match["avg_sharpe"]
        bar = "+" * max(int(d * 20), 0) if d > 0 else "-" * max(int(-d * 20), 0)
        print(f"    VRP={r['w_vrp']:5.0%}  Δ={d:+.3f}  {bar}")

    # Verdict
    print()
    print("=" * 70)
    if best_iv["w_vrp"] != 0.40:
        if best_iv["avg_sharpe"] > current_iv["avg_sharpe"] + 0.05:
            print(f"VERDICT: Optimal weights SHIFT with IV sizing")
            print(f"  New optimal: VRP={best_iv['w_vrp']:.0%}/Skew={best_iv['w_skew']:.0%} (was 40/60)")
            print(f"  Improvement: {best_iv['avg_sharpe'] - current_iv['avg_sharpe']:+.3f} avg Sharpe")
        else:
            print(f"VERDICT: Minor shift ({best_iv['w_vrp']:.0%} vs 40%), but difference is small")
            print(f"  Keep 40/60 for stability")
    else:
        print(f"VERDICT: 40/60 remains optimal WITH IV sizing")
        print(f"  avg={current_iv['avg_sharpe']:.3f}, min={current_iv['min_sharpe']:.3f}")
    print("=" * 70)

    # Save
    out = ROOT / "artifacts" / "crypto_options" / "ensemble_weight_reopt_iv_sizing.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump({
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "research_id": "R29",
            "iv_sizing_config": {
                "lookback": IV_LOOKBACK,
                "pct_low": IV_PCT_LOW, "pct_high": IV_PCT_HIGH,
                "scale_low": IV_SCALE_LOW, "scale_high": IV_SCALE_HIGH,
            },
            "base_results": base_results,
            "iv_results": iv_results,
            "best_base": {"w_vrp": best_base["w_vrp"], "avg_sharpe": best_base["avg_sharpe"]},
            "best_iv": {"w_vrp": best_iv["w_vrp"], "avg_sharpe": best_iv["avg_sharpe"]},
            "current_40_60_iv": {"avg_sharpe": current_iv["avg_sharpe"], "min_sharpe": current_iv["min_sharpe"]},
        }, f, indent=2)
    print(f"\nResults saved to: {out}")


if __name__ == "__main__":
    main()
