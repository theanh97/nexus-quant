#!/usr/bin/env python3
"""
IV Sizing Parameter Optimization on Ensemble — R28
=====================================================

R25 showed step function (25th/75th → 0.5x/1.5x, lb=180) improves ensemble by +0.276.
R26-R27 confirmed no other sizing approach beats IV-percentile.

Now: optimize the step function parameters to squeeze more from this winner.

Sweep:
  1. Percentile thresholds: (10/90, 15/85, 20/80, 25/75, 30/70, 33/67, 40/60)
  2. Scale factors: low=[0.3, 0.4, 0.5, 0.6, 0.7], high=[1.2, 1.3, 1.5, 1.7, 2.0]
  3. Lookback: [90, 120, 180, 240] (verify 180 is still optimal)
  4. Continuous alternatives: linear, power, logistic (on ensemble)
  5. Asymmetric: different low/high thresholds

Grid: ~100 configs (threshold × scale × lookback subsets)
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


def make_step_fn(low_pct, high_pct, low_scale, high_scale):
    """Create a step sizing function with custom thresholds."""
    def fn(pct):
        if pct < low_pct:
            return low_scale
        elif pct > high_pct:
            return high_scale
        return 1.0
    return fn


def make_linear_fn(min_scale, max_scale):
    """Linear: scale = min_scale + (max_scale - min_scale) * pct."""
    def fn(pct):
        return min_scale + (max_scale - min_scale) * pct
    return fn


def make_power_fn(min_scale, max_scale, power):
    """Power law: scale = min_scale + (max_scale - min_scale) * pct^power."""
    def fn(pct):
        return min_scale + (max_scale - min_scale) * (pct ** power)
    return fn


def make_logistic_fn(center, steepness, min_scale, max_scale):
    """Logistic: smooth step centered at 'center' percentile."""
    def fn(pct):
        x = steepness * (pct - center)
        sig = 1.0 / (1.0 + math.exp(-x))
        return min_scale + (max_scale - min_scale) * sig
    return fn


def run_ensemble(sizing_fn, lookback: int = 180) -> Dict[str, Any]:
    """Run VRP+Skew ensemble with a custom sizing function."""
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

            bar_pnl = W_VRP * vrp_pnl + W_SKEW * skew_pnl
            dp = equity * bar_pnl
            equity += dp

            # -- Rebalance --
            if vrp_strat.should_rebalance(dataset, idx):
                target_v = vrp_strat.target_weights(dataset, idx, vrp_weights)
                target_s = skew_strat.target_weights(dataset, idx, skew_weights) if skew_strat.should_rebalance(dataset, idx) else skew_weights

                # Apply IV sizing
                if sizing_fn is not None:
                    iv_series = dataset.features.get("iv_atm", {}).get(sym, [])
                    pct = iv_percentile(iv_series, idx, lookback)
                    if pct is not None:
                        scale = sizing_fn(pct)
                        for s in target_v:
                            target_v[s] *= scale
                        for s in target_s:
                            target_s[s] *= scale

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
                if sizing_fn is not None:
                    iv_series = dataset.features.get("iv_atm", {}).get(sym, [])
                    pct = iv_percentile(iv_series, idx, lookback)
                    if pct is not None:
                        scale = sizing_fn(pct)
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
        yearly_detail[str(yr)] = round(m["sharpe"], 3)

    avg = sum(sharpes) / len(sharpes)
    mn = min(sharpes)
    return {
        "avg_sharpe": round(avg, 3),
        "min_sharpe": round(mn, 3),
        "yearly": yearly_detail,
    }


def main():
    print("=" * 70)
    print("IV SIZING PARAMETER OPTIMIZATION — R28")
    print("=" * 70)
    print()

    # Baseline
    baseline = run_ensemble(sizing_fn=None)
    print(f"  Baseline (no sizing):    avg={baseline['avg_sharpe']:.3f} min={baseline['min_sharpe']:.3f}")

    # R25 champion
    r25_fn = make_step_fn(0.25, 0.75, 0.5, 1.5)
    r25 = run_ensemble(sizing_fn=r25_fn, lookback=180)
    print(f"  R25 champion (25/75):    avg={r25['avg_sharpe']:.3f} min={r25['min_sharpe']:.3f} Δ={r25['avg_sharpe']-baseline['avg_sharpe']:+.3f}")
    print()

    all_results = []

    # ── 1. Threshold sweep (fixed lb=180, scale=0.5/1.5) ─────────────────
    print("--- SWEEP 1: Percentile Thresholds (lb=180, s=0.5/1.5) ---")

    thresholds = [
        (0.10, 0.90), (0.15, 0.85), (0.20, 0.80), (0.25, 0.75),
        (0.30, 0.70), (0.33, 0.67), (0.40, 0.60),
    ]

    for lp, hp in thresholds:
        tag = f"step_{lp:.0%}/{hp:.0%}_s0.5/1.5_lb180"
        fn = make_step_fn(lp, hp, 0.5, 1.5)
        r = run_ensemble(sizing_fn=fn, lookback=180)
        d = r["avg_sharpe"] - baseline["avg_sharpe"]
        all_results.append({"tag": tag, "type": "step_threshold", **r, "delta": round(d, 3)})
        marker = "*" if r["avg_sharpe"] >= r25["avg_sharpe"] else " "
        print(f"    {tag:45s} avg={r['avg_sharpe']:.3f} min={r['min_sharpe']:.3f} Δ={d:+.3f} {marker}")

    print()

    # ── 2. Scale factor sweep (fixed lb=180, threshold=25/75) ─────────────
    print("--- SWEEP 2: Scale Factors (lb=180, thresh=25/75) ---")

    low_scales = [0.3, 0.4, 0.5, 0.6, 0.7]
    high_scales = [1.2, 1.3, 1.5, 1.7, 2.0]

    for ls in low_scales:
        for hs in high_scales:
            if ls == 0.5 and hs == 1.5:
                continue  # already tested as R25 champion
            tag = f"step_25/75_s{ls}/{hs}_lb180"
            fn = make_step_fn(0.25, 0.75, ls, hs)
            r = run_ensemble(sizing_fn=fn, lookback=180)
            d = r["avg_sharpe"] - baseline["avg_sharpe"]
            all_results.append({"tag": tag, "type": "step_scale", **r, "delta": round(d, 3)})
            marker = "*" if r["avg_sharpe"] >= r25["avg_sharpe"] else " "
            print(f"    {tag:45s} avg={r['avg_sharpe']:.3f} min={r['min_sharpe']:.3f} Δ={d:+.3f} {marker}")

    print()

    # ── 3. Lookback sweep with best params ────────────────────────────────
    print("--- SWEEP 3: Lookback (thresh=25/75, s=0.5/1.5) ---")

    lookbacks = [60, 90, 120, 150, 180, 210, 240, 300]

    for lb in lookbacks:
        if lb == 180:
            continue  # already tested
        tag = f"step_25/75_s0.5/1.5_lb{lb}"
        fn = make_step_fn(0.25, 0.75, 0.5, 1.5)
        r = run_ensemble(sizing_fn=fn, lookback=lb)
        d = r["avg_sharpe"] - baseline["avg_sharpe"]
        all_results.append({"tag": tag, "type": "step_lookback", **r, "delta": round(d, 3)})
        marker = "*" if r["avg_sharpe"] >= r25["avg_sharpe"] else " "
        print(f"    {tag:45s} avg={r['avg_sharpe']:.3f} min={r['min_sharpe']:.3f} Δ={d:+.3f} {marker}")

    print()

    # ── 4. Continuous sizing functions ────────────────────────────────────
    print("--- SWEEP 4: Continuous Sizing Functions (lb=180) ---")

    continuous_configs = [
        ("linear_0.5_1.5", make_linear_fn(0.5, 1.5)),
        ("linear_0.3_1.7", make_linear_fn(0.3, 1.7)),
        ("linear_0.6_1.4", make_linear_fn(0.6, 1.4)),
        ("power_0.5_1.5_p0.5", make_power_fn(0.5, 1.5, 0.5)),
        ("power_0.5_1.5_p2.0", make_power_fn(0.5, 1.5, 2.0)),
        ("power_0.3_1.7_p0.5", make_power_fn(0.3, 1.7, 0.5)),
        ("logistic_0.5_8_0.5_1.5", make_logistic_fn(0.5, 8, 0.5, 1.5)),
        ("logistic_0.5_4_0.5_1.5", make_logistic_fn(0.5, 4, 0.5, 1.5)),
        ("logistic_0.5_12_0.5_1.5", make_logistic_fn(0.5, 12, 0.5, 1.5)),
        ("logistic_0.5_8_0.3_1.7", make_logistic_fn(0.5, 8, 0.3, 1.7)),
    ]

    for tag, fn in continuous_configs:
        ftag = f"cont_{tag}_lb180"
        r = run_ensemble(sizing_fn=fn, lookback=180)
        d = r["avg_sharpe"] - baseline["avg_sharpe"]
        all_results.append({"tag": ftag, "type": "continuous", **r, "delta": round(d, 3)})
        marker = "*" if r["avg_sharpe"] >= r25["avg_sharpe"] else " "
        print(f"    {ftag:45s} avg={r['avg_sharpe']:.3f} min={r['min_sharpe']:.3f} Δ={d:+.3f} {marker}")

    print()

    # ── 5. Asymmetric thresholds ──────────────────────────────────────────
    print("--- SWEEP 5: Asymmetric Thresholds (lb=180, s=0.5/1.5) ---")

    asym_configs = [
        (0.20, 0.75),  # more aggressive downscaling trigger
        (0.25, 0.80),  # less aggressive upscaling trigger
        (0.15, 0.75),  # very aggressive downscaling
        (0.25, 0.85),  # very conservative upscaling
        (0.30, 0.80),  # balanced conservative
        (0.20, 0.85),  # aggressive down, conservative up
    ]

    for lp, hp in asym_configs:
        tag = f"asym_{lp:.0%}/{hp:.0%}_s0.5/1.5_lb180"
        fn = make_step_fn(lp, hp, 0.5, 1.5)
        r = run_ensemble(sizing_fn=fn, lookback=180)
        d = r["avg_sharpe"] - baseline["avg_sharpe"]
        all_results.append({"tag": tag, "type": "asymmetric", **r, "delta": round(d, 3)})
        marker = "*" if r["avg_sharpe"] >= r25["avg_sharpe"] else " "
        print(f"    {tag:45s} avg={r['avg_sharpe']:.3f} min={r['min_sharpe']:.3f} Δ={d:+.3f} {marker}")

    print()

    # ── SUMMARY ───────────────────────────────────────────────────────────
    print("=" * 70)
    print("SUMMARY — R28: IV Sizing Parameter Optimization")
    print("=" * 70)
    print()
    print(f"  Baseline:              avg={baseline['avg_sharpe']:.3f} min={baseline['min_sharpe']:.3f}")
    print(f"  R25 champion:          avg={r25['avg_sharpe']:.3f} min={r25['min_sharpe']:.3f} Δ={r25['avg_sharpe']-baseline['avg_sharpe']:+.3f}")
    print()

    # Add R25 to all results for ranking
    all_results.append({
        "tag": "R25_step_25/75_s0.5/1.5_lb180",
        "type": "step_threshold",
        **r25,
        "delta": round(r25["avg_sharpe"] - baseline["avg_sharpe"], 3),
    })

    # Sort by avg_sharpe
    all_results.sort(key=lambda x: x["avg_sharpe"], reverse=True)
    n_total = len(all_results)
    n_better = sum(1 for r in all_results if r["avg_sharpe"] > r25["avg_sharpe"])

    print(f"  Total configs tested:  {n_total}")
    print(f"  Beat R25 champion:     {n_better}/{n_total}")
    print()

    # Top 10
    print("  TOP 10:")
    for i, r in enumerate(all_results[:10]):
        d = r["avg_sharpe"] - baseline["avg_sharpe"]
        is_r25 = "← R25" if "R25" in r["tag"] else ""
        print(f"    {i+1:2d}. {r['tag']:45s} avg={r['avg_sharpe']:.3f} min={r['min_sharpe']:.3f} Δ={d:+.3f} {is_r25}")

    # Bottom 5
    print("\n  BOTTOM 5:")
    for r in all_results[-5:]:
        d = r["avg_sharpe"] - baseline["avg_sharpe"]
        print(f"      {r['tag']:45s} avg={r['avg_sharpe']:.3f} Δ={d:+.3f}")

    # By type
    print("\n  BY TYPE (avg delta):")
    for ttype in ["step_threshold", "step_scale", "step_lookback", "continuous", "asymmetric"]:
        subset = [r for r in all_results if r.get("type") == ttype]
        if subset:
            avg_d = sum(r["delta"] for r in subset) / len(subset)
            best = max(subset, key=lambda x: x["avg_sharpe"])
            print(f"    {ttype:20s} avg_Δ={avg_d:+.3f}  best: {best['tag'][:40]} avg={best['avg_sharpe']:.3f}")

    # New champion?
    champion = all_results[0]
    print()
    print("=" * 70)
    if champion["avg_sharpe"] > r25["avg_sharpe"] + 0.02:
        print(f"NEW CHAMPION: {champion['tag']}")
        print(f"  avg={champion['avg_sharpe']:.3f} min={champion['min_sharpe']:.3f} Δ={champion['delta']:+.3f}")
        print(f"  Improvement vs R25: {champion['avg_sharpe'] - r25['avg_sharpe']:+.3f}")
    else:
        print(f"R25 CONFIRMED as optimal (or near-optimal)")
        print(f"  R25: avg={r25['avg_sharpe']:.3f}  Best found: avg={champion['avg_sharpe']:.3f}")
        print(f"  Difference: {champion['avg_sharpe'] - r25['avg_sharpe']:+.3f}")
    print("=" * 70)

    # Save
    out = ROOT / "artifacts" / "crypto_options" / "iv_sizing_param_optimization.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump({
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "research_id": "R28",
            "baseline": baseline,
            "r25_champion": r25,
            "n_configs": n_total,
            "n_beat_r25": n_better,
            "new_champion": {
                "tag": champion["tag"],
                "avg_sharpe": champion["avg_sharpe"],
                "min_sharpe": champion["min_sharpe"],
                "delta": champion["delta"],
            },
            "top_10": all_results[:10],
            "all_results": all_results,
        }, f, indent=2)
    print(f"\nResults saved to: {out}")


if __name__ == "__main__":
    main()
