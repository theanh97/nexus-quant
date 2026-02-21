#!/usr/bin/env python3
"""
Smoothed RV Carry Sizing — R33
=================================

R27 showed direct carry spread (IV²-RV²) was too noisy for position sizing.
Root cause: single-day RV = |log_ret| * sqrt(365) is dominated by one day's return.

Fix: Use multi-day smoothed RV:
  - 5-day RV: sqrt(sum(log_ret_i^2, i=1..5) * 365/5)
  - 10-day RV: sqrt(sum(log_ret_i^2, i=1..10) * 365/10)
  - 21-day RV: sqrt(sum(log_ret_i^2, i=1..21) * 365/21)
  - EWMA RV: exponentially weighted with various half-lives

Then compute VRP carry = IV² - smoothed_RV² and size by percentile.

Hypothesis: Smoothed carry should be a BETTER sizing signal than IV percentile alone
because it directly measures carry richness, not just IV level.

Grid: 4 RV windows × 3 lookbacks × 2 scale factors + combinations with IV sizing
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

fees = FeeModel(maker_fee_rate=0.0003, taker_fee_rate=0.0005)
impact = ImpactModel(model="sqrt", coef_bps=2.0)
COSTS = ExecutionCostModel(fee=fees, impact=impact)

YEARS = [2021, 2022, 2023, 2024, 2025]
SEED = 42
BARS_PER_YEAR = 365
W_VRP = 0.40
W_SKEW = 0.60

VRP_PARAMS = {
    "base_leverage": 1.5, "exit_z_threshold": -3.0,
    "vrp_lookback": 30, "rebalance_freq": 5, "min_bars": 30,
}
SKEW_PARAMS = {
    "skew_lookback": 60, "z_entry": 2.0, "z_exit": 0.0,
    "target_leverage": 1.0, "rebalance_freq": 5, "min_bars": 60,
}


def iv_percentile(iv_series, idx, lookback=180):
    if idx < lookback or not iv_series:
        return None
    start = max(0, idx - lookback)
    window = [v for v in iv_series[start:idx] if v is not None]
    if len(window) < 10:
        return None
    current = iv_series[idx] if idx < len(iv_series) and iv_series[idx] is not None else None
    if current is None:
        return None
    return sum(1 for v in window if v < current) / len(window)


def iv_step_scale(pct, low=0.5, high=1.7):
    if pct < 0.25:
        return low
    elif pct > 0.75:
        return high
    return 1.0


def compute_smoothed_rv(closes, idx, rv_window):
    """Compute annualized RV using rv_window days of log returns."""
    if idx < rv_window or idx >= len(closes):
        return None
    log_rets_sq = []
    for i in range(idx - rv_window + 1, idx + 1):
        if i < 1 or i >= len(closes) or closes[i] <= 0 or closes[i - 1] <= 0:
            return None
        lr = math.log(closes[i] / closes[i - 1])
        log_rets_sq.append(lr ** 2)
    if not log_rets_sq:
        return None
    variance_daily = sum(log_rets_sq) / len(log_rets_sq)
    return math.sqrt(variance_daily * BARS_PER_YEAR)


def compute_carry(iv, smoothed_rv):
    """VRP carry = 0.5 * (IV² - RV²)."""
    if iv is None or smoothed_rv is None or iv <= 0:
        return None
    return 0.5 * (iv ** 2 - smoothed_rv ** 2)


def carry_percentile_scale(carry_history, current_carry, low_scale=0.5, high_scale=1.7):
    """Step sizing by carry percentile."""
    if len(carry_history) < 10:
        return 1.0
    below = sum(1 for v in carry_history if v < current_carry)
    pct = below / len(carry_history)
    if pct < 0.25:
        return low_scale
    elif pct > 0.75:
        return high_scale
    return 1.0


def run_ensemble(
    rv_window: int = 1,
    carry_lookback: int = 90,
    low_scale: float = 0.5,
    high_scale: float = 1.7,
    use_carry_sizing: bool = False,
    use_iv_sizing: bool = False,
) -> Dict[str, Any]:
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
        carry_history = []
        sym = "BTC"

        for idx in range(1, n):
            prev_equity = equity

            # Update carry history
            if use_carry_sizing:
                closes = dataset.perp_close.get(sym, [])
                srw = compute_smoothed_rv(closes, idx - 1, rv_window)
                iv_series = dataset.features.get("iv_atm", {}).get(sym, [])
                iv_val = iv_series[idx - 1] if iv_series and idx - 1 < len(iv_series) else None
                carry = compute_carry(iv_val, srw)
                if carry is not None:
                    carry_history.append(carry)
                    if len(carry_history) > carry_lookback:
                        carry_history = carry_history[-carry_lookback:]

            # VRP P&L
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

            # Skew P&L
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
            equity += equity * bar_pnl

            # Rebalance
            if vrp_strat.should_rebalance(dataset, idx):
                target_v = vrp_strat.target_weights(dataset, idx, vrp_weights)
                target_s = skew_strat.target_weights(dataset, idx, skew_weights) if skew_strat.should_rebalance(dataset, idx) else skew_weights

                # IV sizing (R28)
                if use_iv_sizing:
                    iv_series = dataset.features.get("iv_atm", {}).get(sym, [])
                    pct = iv_percentile(iv_series, idx)
                    if pct is not None:
                        s = iv_step_scale(pct)
                        for k in target_v:
                            target_v[k] *= s
                        for k in target_s:
                            target_s[k] *= s

                # Carry sizing (smoothed RV)
                if use_carry_sizing and len(carry_history) >= 10:
                    closes = dataset.perp_close.get(sym, [])
                    srw = compute_smoothed_rv(closes, idx, rv_window)
                    iv_series = dataset.features.get("iv_atm", {}).get(sym, [])
                    iv_val = iv_series[idx] if iv_series and idx < len(iv_series) else None
                    carry = compute_carry(iv_val, srw)
                    if carry is not None:
                        cs = carry_percentile_scale(carry_history, carry, low_scale, high_scale)
                        for k in target_v:
                            target_v[k] *= cs
                        for k in target_s:
                            target_s[k] *= cs

                old_total = {sym: W_VRP * vrp_weights.get(sym, 0) + W_SKEW * skew_weights.get(sym, 0)}
                new_total = {sym: W_VRP * target_v.get(sym, 0) + W_SKEW * target_s.get(sym, 0)}
                turnover = sum(abs(new_total[s_] - old_total[s_]) for s_ in [sym])
                bd = COSTS.cost(equity=equity, turnover=turnover)
                equity -= float(bd.get("cost", 0.0))
                equity = max(equity, 0.0)
                vrp_weights = target_v
                skew_weights = target_s

            elif skew_strat.should_rebalance(dataset, idx):
                target_s = skew_strat.target_weights(dataset, idx, skew_weights)
                if use_iv_sizing:
                    iv_series = dataset.features.get("iv_atm", {}).get(sym, [])
                    pct = iv_percentile(iv_series, idx)
                    if pct is not None:
                        s = iv_step_scale(pct)
                        for k in target_s:
                            target_s[k] *= s
                if use_carry_sizing and len(carry_history) >= 10:
                    closes = dataset.perp_close.get(sym, [])
                    srw = compute_smoothed_rv(closes, idx, rv_window)
                    iv_series = dataset.features.get("iv_atm", {}).get(sym, [])
                    iv_val = iv_series[idx] if iv_series and idx < len(iv_series) else None
                    carry = compute_carry(iv_val, srw)
                    if carry is not None:
                        cs = carry_percentile_scale(carry_history, carry, low_scale, high_scale)
                        for k in target_s:
                            target_s[k] *= cs

                old_total = {sym: W_VRP * vrp_weights.get(sym, 0) + W_SKEW * skew_weights.get(sym, 0)}
                new_total = {sym: W_VRP * vrp_weights.get(sym, 0) + W_SKEW * target_s.get(sym, 0)}
                turnover = sum(abs(new_total[s_] - old_total[s_]) for s_ in [sym])
                bd = COSTS.cost(equity=equity, turnover=turnover)
                equity -= float(bd.get("cost", 0.0))
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
    return {"avg_sharpe": round(avg, 3), "min_sharpe": round(mn, 3), "yearly": yearly_detail}


def main():
    print("=" * 70)
    print("SMOOTHED RV CARRY SIZING — R33")
    print("=" * 70)
    print()

    # Baselines
    baseline = run_ensemble(use_carry_sizing=False, use_iv_sizing=False)
    iv_only = run_ensemble(use_carry_sizing=False, use_iv_sizing=True)
    print(f"  Baseline:      avg={baseline['avg_sharpe']:.3f} min={baseline['min_sharpe']:.3f}")
    print(f"  IV-sized (R28): avg={iv_only['avg_sharpe']:.3f} min={iv_only['min_sharpe']:.3f}")
    print()

    all_results = []

    # ── 1. Smoothed RV carry sizing alone ─────────────────────────────────
    print("--- SMOOTHED RV CARRY SIZING (alone, no IV sizing) ---")
    print(f"  {'Tag':45s} {'avg':>7s} {'min':>7s} {'Δbase':>8s}")

    rv_windows = [5, 10, 21, 30]
    carry_lookbacks = [60, 90, 180]

    for rvw in rv_windows:
        for clb in carry_lookbacks:
            tag = f"carry_rv{rvw}_lb{clb}_s0.5/1.7"
            r = run_ensemble(rv_window=rvw, carry_lookback=clb,
                           low_scale=0.5, high_scale=1.7,
                           use_carry_sizing=True, use_iv_sizing=False)
            d = r["avg_sharpe"] - baseline["avg_sharpe"]
            all_results.append({"tag": tag, "rv_window": rvw, "carry_lb": clb, **r, "delta": round(d, 3)})
            marker = "+" if d > 0.01 else " "
            print(f"  {tag:45s} {r['avg_sharpe']:7.3f} {r['min_sharpe']:7.3f} {d:+8.3f} {marker}")

    print()

    # ── 2. Compare with 1-day RV (R27 approach for reference) ─────────────
    print("--- 1-DAY RV CARRY (R27 reference) ---")
    for clb in [90, 180]:
        tag = f"carry_rv1_lb{clb}_s0.5/1.7"
        r = run_ensemble(rv_window=1, carry_lookback=clb,
                       low_scale=0.5, high_scale=1.7,
                       use_carry_sizing=True, use_iv_sizing=False)
        d = r["avg_sharpe"] - baseline["avg_sharpe"]
        all_results.append({"tag": tag, "rv_window": 1, "carry_lb": clb, **r, "delta": round(d, 3)})
        print(f"  {tag:45s} {r['avg_sharpe']:7.3f} {r['min_sharpe']:7.3f} {d:+8.3f}")

    print()

    # ── 3. Combined: smoothed carry + IV sizing ──────────────────────────
    print("--- COMBINED: Smoothed carry + IV sizing ---")
    combo_results = []
    best_carry = max(all_results, key=lambda x: x["avg_sharpe"])

    combo_configs = [
        (5, 90), (10, 90), (21, 90),
        (5, 180), (10, 180), (21, 180),
    ]

    for rvw, clb in combo_configs:
        tag = f"combo_rv{rvw}_lb{clb}+IV"
        r = run_ensemble(rv_window=rvw, carry_lookback=clb,
                       low_scale=0.5, high_scale=1.7,
                       use_carry_sizing=True, use_iv_sizing=True)
        d_base = r["avg_sharpe"] - baseline["avg_sharpe"]
        d_iv = r["avg_sharpe"] - iv_only["avg_sharpe"]
        combo_results.append({
            "tag": tag, **r,
            "delta_vs_base": round(d_base, 3),
            "delta_vs_iv": round(d_iv, 3),
        })
        marker = "+" if d_iv > 0 else " "
        print(f"  {tag:45s} {r['avg_sharpe']:7.3f} Δbase={d_base:+.3f} ΔIV={d_iv:+.3f} {marker}")

    print()

    # ── SUMMARY ───────────────────────────────────────────────────────────
    print("=" * 70)
    print("SUMMARY — R33: Smoothed RV Carry Sizing")
    print("=" * 70)
    print()
    print(f"  Baseline:         avg={baseline['avg_sharpe']:.3f}")
    print(f"  IV-sized (R28):   avg={iv_only['avg_sharpe']:.3f}")
    print()

    n_better = sum(1 for r in all_results if r["delta"] > 0)
    n_total = len(all_results)
    best = max(all_results, key=lambda x: x["avg_sharpe"])

    print(f"  Carry-only configs: {n_total}")
    print(f"    Better than baseline: {n_better}/{n_total}")
    print(f"    Best: {best['tag']} avg={best['avg_sharpe']:.3f} Δ={best['delta']:+.3f}")
    print()

    # By RV window
    print("  BY RV WINDOW:")
    for rvw in [1, 5, 10, 21, 30]:
        subset = [r for r in all_results if r.get("rv_window") == rvw]
        if subset:
            avg_d = sum(r["delta"] for r in subset) / len(subset)
            b = max(subset, key=lambda x: x["delta"])
            print(f"    rv={rvw:2d}d  avg_Δ={avg_d:+.3f}  best Δ={b['delta']:+.3f}")

    # Combined
    if combo_results:
        print()
        best_c = max(combo_results, key=lambda x: x["avg_sharpe"])
        n_additive = sum(1 for r in combo_results if r["delta_vs_iv"] > 0)
        print(f"  Combined (carry + IV): {n_additive}/{len(combo_results)} additive to IV sizing")
        print(f"    Best combined: {best_c['tag']} avg={best_c['avg_sharpe']:.3f} ΔIV={best_c['delta_vs_iv']:+.3f}")

    # Verdict
    print()
    print("=" * 70)
    smoothed_better = any(r["delta"] > 0 and r.get("rv_window", 0) > 1 for r in all_results)
    rv1_better = any(r["delta"] > 0 and r.get("rv_window", 0) == 1 for r in all_results)

    if smoothed_better and not rv1_better:
        print("VERDICT: Smoothed RV FIXES carry sizing noise (R27 was right about the cause)")
    elif smoothed_better:
        print("VERDICT: Smoothed RV IMPROVES carry sizing vs 1-day RV")
    else:
        print("VERDICT: Even smoothed RV carry sizing doesn't beat baseline")
        print("  IV percentile (R28) remains the only effective sizing approach")
    print("=" * 70)

    # Save
    out = ROOT / "artifacts" / "crypto_options" / "smoothed_rv_carry_sizing.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump({
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "research_id": "R33",
            "baseline": baseline,
            "iv_sized": iv_only,
            "carry_results": all_results,
            "combo_results": combo_results,
        }, f, indent=2)
    print(f"\nResults saved to: {out}")


if __name__ == "__main__":
    main()
