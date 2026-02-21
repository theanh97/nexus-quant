#!/usr/bin/env python3
"""
Strategy Leverage Re-optimization with IV Sizing — R34
========================================================

VRP base_leverage=1.5 and Skew target_leverage=1.0 were tuned WITHOUT IV sizing.
With R28's 0.5x/1.7x sizing:
  - Low IV: effective leverage = 1.5 * 0.5 = 0.75x
  - Normal IV: effective leverage = 1.5 * 1.0 = 1.5x
  - High IV: effective leverage = 1.5 * 1.7 = 2.55x

Questions:
  1. Should base_leverage be higher (let IV sizing modulate more)?
  2. Should base_leverage be lower (limit max effective leverage)?
  3. Should Skew leverage change with IV sizing active?

Sweep: VRP_leverage × Skew_leverage with IV sizing
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
    if pct < 0.25:
        return 0.5
    elif pct > 0.75:
        return 1.7
    return 1.0


def run_ensemble(vrp_lev: float, skew_lev: float, use_iv: bool) -> Dict[str, Any]:
    sharpes = []
    yearly_detail = {}

    vrp_params = {
        "base_leverage": vrp_lev,
        "exit_z_threshold": -3.0,
        "vrp_lookback": 30,
        "rebalance_freq": 5,
        "min_bars": 30,
    }
    skew_params = {
        "skew_lookback": 60, "z_entry": 2.0, "z_exit": 0.0,
        "target_leverage": skew_lev, "rebalance_freq": 5, "min_bars": 60,
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

            if vrp_strat.should_rebalance(dataset, idx):
                target_v = vrp_strat.target_weights(dataset, idx, vrp_weights)
                target_s = skew_strat.target_weights(dataset, idx, skew_weights) if skew_strat.should_rebalance(dataset, idx) else skew_weights

                if use_iv:
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
                equity -= float(bd.get("cost", 0.0))
                equity = max(equity, 0.0)
                vrp_weights = target_v
                skew_weights = target_s

            elif skew_strat.should_rebalance(dataset, idx):
                target_s = skew_strat.target_weights(dataset, idx, skew_weights)
                if use_iv:
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
    print("LEVERAGE RE-OPTIMIZATION WITH IV SIZING — R34")
    print("=" * 70)
    print()

    current = run_ensemble(1.5, 1.0, use_iv=True)
    baseline = run_ensemble(1.5, 1.0, use_iv=False)
    print(f"  Current (VRP=1.5, Skew=1.0, no IV): avg={baseline['avg_sharpe']:.3f}")
    print(f"  Current (VRP=1.5, Skew=1.0, IV):    avg={current['avg_sharpe']:.3f}")
    print()

    # ── 1. VRP leverage sweep (Skew=1.0 fixed) ──────────────────────────
    print("--- VRP LEVERAGE SWEEP (Skew=1.0 fixed, with IV sizing) ---")
    vrp_levs = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0]
    vrp_results = []

    for lev in vrp_levs:
        r = run_ensemble(lev, 1.0, use_iv=True)
        d = r["avg_sharpe"] - current["avg_sharpe"]
        vrp_results.append({"vrp_lev": lev, **r, "delta": round(d, 3)})
        eff_max = lev * 1.7
        marker = " *" if lev == 1.5 else ""
        print(f"  VRP={lev:.2f} (max_eff={eff_max:.2f}x) avg={r['avg_sharpe']:.3f} min={r['min_sharpe']:.3f} Δ={d:+.3f}{marker}")

    print()

    # ── 2. Skew leverage sweep (VRP=1.5 fixed) ──────────────────────────
    print("--- SKEW LEVERAGE SWEEP (VRP=1.5 fixed, with IV sizing) ---")
    skew_levs = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0]
    skew_results = []

    for lev in skew_levs:
        r = run_ensemble(1.5, lev, use_iv=True)
        d = r["avg_sharpe"] - current["avg_sharpe"]
        skew_results.append({"skew_lev": lev, **r, "delta": round(d, 3)})
        marker = " *" if lev == 1.0 else ""
        print(f"  Skew={lev:.2f} avg={r['avg_sharpe']:.3f} min={r['min_sharpe']:.3f} Δ={d:+.3f}{marker}")

    print()

    # ── 3. Joint sweep (2D grid) ─────────────────────────────────────────
    print("--- JOINT LEVERAGE GRID (with IV sizing) ---")
    joint_results = []
    vrp_grid = [1.0, 1.25, 1.5, 1.75, 2.0]
    skew_grid = [0.75, 1.0, 1.25, 1.5]

    header = f"  {'VRP\\Skew':>10s}"
    for sl in skew_grid:
        header += f" {sl:10.2f}"
    print(header)

    for vl in vrp_grid:
        row = f"  {vl:10.2f}"
        for sl in skew_grid:
            r = run_ensemble(vl, sl, use_iv=True)
            joint_results.append({"vrp_lev": vl, "skew_lev": sl, **r})
            marker = "*" if vl == 1.5 and sl == 1.0 else " "
            row += f" {r['avg_sharpe']:9.3f}{marker}"
        print(row)

    best_joint = max(joint_results, key=lambda x: x["avg_sharpe"])
    print(f"\n  Best joint: VRP={best_joint['vrp_lev']:.2f} Skew={best_joint['skew_lev']:.2f} avg={best_joint['avg_sharpe']:.3f}")
    print(f"  Current:    VRP=1.50 Skew=1.00 avg={current['avg_sharpe']:.3f}")
    print(f"  Δ: {best_joint['avg_sharpe'] - current['avg_sharpe']:+.3f}")
    print()

    # ── SUMMARY ───────────────────────────────────────────────────────────
    print("=" * 70)
    print("SUMMARY — R34: Leverage Re-optimization")
    print("=" * 70)
    print()

    best_vrp = max(vrp_results, key=lambda x: x["avg_sharpe"])
    best_skew = max(skew_results, key=lambda x: x["avg_sharpe"])
    print(f"  Best VRP lev:  {best_vrp['vrp_lev']:.2f} avg={best_vrp['avg_sharpe']:.3f} (Δ={best_vrp['delta']:+.3f})")
    print(f"  Best Skew lev: {best_skew['skew_lev']:.2f} avg={best_skew['avg_sharpe']:.3f} (Δ={best_skew['delta']:+.3f})")
    print(f"  Best joint:    VRP={best_joint['vrp_lev']:.2f}/Skew={best_joint['skew_lev']:.2f} avg={best_joint['avg_sharpe']:.3f}")
    print()

    d = best_joint["avg_sharpe"] - current["avg_sharpe"]
    if d > 0.05:
        print(f"VERDICT: Leverage should be adjusted with IV sizing")
        print(f"  New: VRP={best_joint['vrp_lev']:.2f}/Skew={best_joint['skew_lev']:.2f} Δ={d:+.3f}")
    else:
        print(f"VERDICT: Current leverage (VRP=1.5, Skew=1.0) is near-optimal")
        print(f"  Best improvement: {d:+.3f} (not significant)")
    print("=" * 70)

    out = ROOT / "artifacts" / "crypto_options" / "leverage_reopt_iv_sizing.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump({
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "research_id": "R34",
            "current": current,
            "baseline_no_iv": baseline,
            "vrp_sweep": vrp_results,
            "skew_sweep": skew_results,
            "joint_grid": joint_results,
            "best_joint": {"vrp_lev": best_joint["vrp_lev"], "skew_lev": best_joint["skew_lev"],
                          "avg_sharpe": best_joint["avg_sharpe"]},
        }, f, indent=2)
    print(f"\nResults saved to: {out}")


if __name__ == "__main__":
    main()
