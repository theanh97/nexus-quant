#!/usr/bin/env python3
"""
Rebalancing Frequency Optimization with IV Sizing — R32
==========================================================

Current: rebalance_freq=5 (every 5 days) for both VRP and Skew.
This was tuned without IV sizing (R28: 0.5x/1.7x).

With IV sizing, position sizes change by 0.5-1.7x based on IV percentile.
Questions:
  1. More frequent rebalancing → faster adaptation to IV regime changes?
  2. Less frequent → lower turnover costs from larger position swings?
  3. Should VRP and Skew have DIFFERENT rebalance frequencies?

Grid: VRP_freq × Skew_freq × [2, 3, 5, 7, 10, 14, 21]
With and without IV sizing for comparison.
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

IV_LOOKBACK = 180
IV_PCT_LOW = 0.25
IV_PCT_HIGH = 0.75
IV_SCALE_LOW = 0.50
IV_SCALE_HIGH = 1.70


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


def iv_scale(pct):
    if pct < IV_PCT_LOW:
        return IV_SCALE_LOW
    elif pct > IV_PCT_HIGH:
        return IV_SCALE_HIGH
    return 1.0


def run_ensemble(vrp_freq: int, skew_freq: int, use_iv_sizing: bool) -> Dict[str, Any]:
    sharpes = []
    yearly_detail = {}

    vrp_params = {
        "base_leverage": 1.5,
        "exit_z_threshold": -3.0,
        "vrp_lookback": 30,
        "rebalance_freq": vrp_freq,
        "min_bars": 30,
    }
    skew_params = {
        "skew_lookback": 60,
        "z_entry": 2.0,
        "z_exit": 0.0,
        "target_leverage": 1.0,
        "rebalance_freq": skew_freq,
        "min_bars": 60,
    }

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

        vrp_strat = VariancePremiumStrategy(params=vrp_params)
        skew_strat = SkewTradeV2Strategy(params=skew_params)

        dt = 1.0 / BARS_PER_YEAR
        equity = 1.0
        vrp_weights = {"BTC": 0.0}
        skew_weights = {"BTC": 0.0}
        equity_curve = [1.0]
        returns_list = []
        sym = "BTC"

        for idx in range(1, n):
            prev_equity = equity

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
            vrp_rebal = vrp_strat.should_rebalance(dataset, idx)
            skew_rebal = skew_strat.should_rebalance(dataset, idx)

            if vrp_rebal:
                target_v = vrp_strat.target_weights(dataset, idx, vrp_weights)
                target_s = skew_strat.target_weights(dataset, idx, skew_weights) if skew_rebal else skew_weights

                if use_iv_sizing:
                    iv_series = dataset.features.get("iv_atm", {}).get(sym, [])
                    pct = iv_percentile(iv_series, idx)
                    if pct is not None:
                        s = iv_scale(pct)
                        for k in target_v:
                            target_v[k] *= s
                        for k in target_s:
                            target_s[k] *= s

                old_total = {sym: W_VRP * vrp_weights.get(sym, 0) + W_SKEW * skew_weights.get(sym, 0)}
                new_total = {sym: W_VRP * target_v.get(sym, 0) + W_SKEW * target_s.get(sym, 0)}
                turnover = sum(abs(new_total[s_] - old_total[s_]) for s_ in [sym])
                bd = COSTS.cost(equity=equity, turnover=turnover)
                cost = float(bd.get("cost", 0.0))
                equity -= cost
                equity = max(equity, 0.0)
                vrp_weights = target_v
                skew_weights = target_s

            elif skew_rebal:
                target_s = skew_strat.target_weights(dataset, idx, skew_weights)
                if use_iv_sizing:
                    iv_series = dataset.features.get("iv_atm", {}).get(sym, [])
                    pct = iv_percentile(iv_series, idx)
                    if pct is not None:
                        s = iv_scale(pct)
                        for k in target_s:
                            target_s[k] *= s

                old_total = {sym: W_VRP * vrp_weights.get(sym, 0) + W_SKEW * skew_weights.get(sym, 0)}
                new_total = {sym: W_VRP * vrp_weights.get(sym, 0) + W_SKEW * target_s.get(sym, 0)}
                turnover = sum(abs(new_total[s_] - old_total[s_]) for s_ in [sym])
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
    return {"avg_sharpe": round(avg, 3), "min_sharpe": round(mn, 3), "yearly": yearly_detail}


def main():
    print("=" * 70)
    print("REBALANCING FREQUENCY OPTIMIZATION WITH IV SIZING — R32")
    print("=" * 70)
    print()

    freqs = [2, 3, 5, 7, 10, 14, 21]

    # ── 1. Symmetric frequency sweep (both VRP and Skew same freq) ────────
    print("--- SYMMETRIC FREQUENCY SWEEP (VRP_freq == Skew_freq) ---")
    print(f"  {'Freq':>6s}  {'base avg':>10s} {'base min':>10s}  {'IV avg':>10s} {'IV min':>10s}  {'Δ':>8s}")
    print(f"  {'----':>6s}  {'-'*10} {'-'*10}  {'-'*10} {'-'*10}  {'-'*8}")

    sym_base = {}
    sym_iv = {}
    for f in freqs:
        base = run_ensemble(f, f, use_iv_sizing=False)
        iv = run_ensemble(f, f, use_iv_sizing=True)
        sym_base[f] = base
        sym_iv[f] = iv
        d = iv["avg_sharpe"] - base["avg_sharpe"]
        marker = " *" if f == 5 else ""
        print(f"  {f:6d}  {base['avg_sharpe']:10.3f} {base['min_sharpe']:10.3f}  {iv['avg_sharpe']:10.3f} {iv['min_sharpe']:10.3f}  {d:+8.3f}{marker}")

    best_sym_iv = max(sym_iv.items(), key=lambda x: x[1]["avg_sharpe"])
    print(f"\n  Best symmetric: freq={best_sym_iv[0]} avg={best_sym_iv[1]['avg_sharpe']:.3f}")
    print()

    # ── 2. Asymmetric frequency sweep ─────────────────────────────────────
    print("--- ASYMMETRIC FREQUENCY SWEEP (VRP_freq ≠ Skew_freq, with IV sizing) ---")
    print(f"  {'VRP':>5s} {'Skew':>5s}  {'avg':>7s} {'min':>7s}")
    print(f"  {'---':>5s} {'---':>5s}  {'---':>7s} {'---':>7s}")

    asym_results = []
    for vf in [3, 5, 7, 10]:
        for sf in [3, 5, 7, 10]:
            if vf == sf:
                continue
            r = run_ensemble(vf, sf, use_iv_sizing=True)
            asym_results.append({"vrp_freq": vf, "skew_freq": sf, **r})
            marker = ""
            if r["avg_sharpe"] > best_sym_iv[1]["avg_sharpe"]:
                marker = " *"
            print(f"  {vf:5d} {sf:5d}  {r['avg_sharpe']:7.3f} {r['min_sharpe']:7.3f}{marker}")

    if asym_results:
        best_asym = max(asym_results, key=lambda x: x["avg_sharpe"])
        print(f"\n  Best asymmetric: VRP={best_asym['vrp_freq']} Skew={best_asym['skew_freq']} avg={best_asym['avg_sharpe']:.3f}")
    print()

    # ── SUMMARY ───────────────────────────────────────────────────────────
    print("=" * 70)
    print("SUMMARY — R32: Rebalancing Frequency with IV Sizing")
    print("=" * 70)
    print()

    current = sym_iv.get(5, {})
    print(f"  Current (freq=5):      avg={current.get('avg_sharpe', 'N/A')}")
    print(f"  Best symmetric:        freq={best_sym_iv[0]} avg={best_sym_iv[1]['avg_sharpe']:.3f}")
    if asym_results:
        best_a = max(asym_results, key=lambda x: x["avg_sharpe"])
        print(f"  Best asymmetric:       VRP={best_a['vrp_freq']} Skew={best_a['skew_freq']} avg={best_a['avg_sharpe']:.3f}")

    # Compare IV sizing Δ across frequencies
    print()
    print("  IV SIZING IMPROVEMENT BY FREQUENCY:")
    for f in freqs:
        d = sym_iv[f]["avg_sharpe"] - sym_base[f]["avg_sharpe"]
        print(f"    freq={f:2d}  Δ={d:+.3f}")

    # Verdict
    print()
    print("=" * 70)
    overall_best = best_sym_iv[1]["avg_sharpe"]
    if asym_results:
        best_a_sharpe = max(r["avg_sharpe"] for r in asym_results)
        if best_a_sharpe > overall_best:
            overall_best = best_a_sharpe

    if abs(overall_best - current.get("avg_sharpe", 0)) < 0.03:
        print(f"VERDICT: freq=5 is near-optimal — no change needed")
    elif overall_best > current.get("avg_sharpe", 0) + 0.05:
        print(f"VERDICT: Rebalancing frequency CAN be improved")
        print(f"  Best: freq={best_sym_iv[0]} gives {best_sym_iv[1]['avg_sharpe']:.3f} vs current {current.get('avg_sharpe', 0):.3f}")
    else:
        print(f"VERDICT: Marginal improvement possible, keep freq=5 for simplicity")
    print("=" * 70)

    # Save
    out = ROOT / "artifacts" / "crypto_options" / "rebalance_freq_iv_sizing.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump({
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "research_id": "R32",
            "symmetric_base": {str(k): v for k, v in sym_base.items()},
            "symmetric_iv": {str(k): v for k, v in sym_iv.items()},
            "asymmetric_iv": asym_results,
            "best_symmetric": {"freq": best_sym_iv[0], **best_sym_iv[1]},
        }, f, indent=2)
    print(f"\nResults saved to: {out}")


if __name__ == "__main__":
    main()
