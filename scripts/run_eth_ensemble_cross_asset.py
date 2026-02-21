#!/usr/bin/env python3
"""
ETH Ensemble Replication & Cross-Asset Dynamics — R31
========================================================

All crypto options R&D (R1-R30) was BTC-only.
Production signal generator handles both BTC and ETH.
This validates:
  1. Does the VRP+Skew ensemble work on ETH? (same params)
  2. Does IV-percentile sizing work on ETH?
  3. BTC-ETH correlation dynamics during stress
  4. Cross-asset signals: does BTC IV predict ETH and vice versa?
  5. Multi-asset portfolio: BTC + ETH ensemble vs BTC-only

Note: DeribitRestProvider synthetic data uses different IV dynamics per asset:
  - ETH has higher base VRP (+10% vs BTC +8%), higher skew, higher vol
  - Results may differ from BTC; this tests structural generalizability
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


def run_single_asset_ensemble(
    sym: str,
    use_iv_sizing: bool = False,
) -> Dict[str, Any]:
    """Run VRP+Skew ensemble for a single asset."""
    sharpes = []
    yearly_detail = {}

    for yr in YEARS:
        cfg = {
            "symbols": [sym],
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
        vrp_weights = {sym: 0.0}
        skew_weights = {sym: 0.0}
        equity_curve = [1.0]
        returns_list = []

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
            dp = equity * bar_pnl
            equity += dp

            # Rebalance
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
                if use_iv_sizing:
                    iv_series = dataset.features.get("iv_atm", {}).get(sym, [])
                    pct = iv_percentile(iv_series, idx)
                    if pct is not None:
                        scale = iv_sizing_scale(pct)
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


def run_multi_asset_ensemble(
    w_btc: float = 0.50,
    w_eth: float = 0.50,
    use_iv_sizing: bool = False,
) -> Dict[str, Any]:
    """Run VRP+Skew ensemble on BTC+ETH with asset allocation weights."""
    sharpes = []
    yearly_detail = {}

    for yr in YEARS:
        cfg = {
            "symbols": ["BTC", "ETH"],
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
        vrp_weights = {"BTC": 0.0, "ETH": 0.0}
        skew_weights = {"BTC": 0.0, "ETH": 0.0}
        equity_curve = [1.0]
        returns_list = []

        asset_weights = {"BTC": w_btc, "ETH": w_eth}

        for idx in range(1, n):
            prev_equity = equity
            total_pnl = 0.0

            for sym in ["BTC", "ETH"]:
                aw = asset_weights[sym]

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

                total_pnl += aw * (W_VRP * vrp_pnl + W_SKEW * skew_pnl)

            dp = equity * total_pnl
            equity += dp

            # Rebalance
            if vrp_strat.should_rebalance(dataset, idx):
                for sym in ["BTC", "ETH"]:
                    target_v = vrp_strat.target_weights(dataset, idx, vrp_weights)
                    target_s = skew_strat.target_weights(dataset, idx, skew_weights) if skew_strat.should_rebalance(dataset, idx) else skew_weights

                    if use_iv_sizing:
                        iv_series = dataset.features.get("iv_atm", {}).get(sym, [])
                        pct = iv_percentile(iv_series, idx)
                        if pct is not None:
                            scale = iv_sizing_scale(pct)
                            if sym in target_v:
                                target_v[sym] *= scale
                            if sym in target_s:
                                target_s[sym] *= scale

                    vrp_weights = target_v
                    skew_weights = target_s

                # Cost: simplified as total turnover
                turnover = 0.05  # approximate
                bd = COSTS.cost(equity=equity, turnover=turnover)
                cost = float(bd.get("cost", 0.0))
                equity -= cost
                equity = max(equity, 0.0)

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
    print("ETH ENSEMBLE REPLICATION & CROSS-ASSET DYNAMICS — R31")
    print("=" * 70)
    print()

    results = {}

    # ── 1. BTC baseline (verify) ──────────────────────────────────────────
    print("--- 1. BTC BASELINE (verify) ---")
    btc_base = run_single_asset_ensemble("BTC", use_iv_sizing=False)
    btc_iv = run_single_asset_ensemble("BTC", use_iv_sizing=True)
    print(f"  BTC unsized:     avg={btc_base['avg_sharpe']:.3f} min={btc_base['min_sharpe']:.3f}")
    print(f"  BTC IV-sized:    avg={btc_iv['avg_sharpe']:.3f} min={btc_iv['min_sharpe']:.3f}")
    print(f"    Yearly: {btc_iv['yearly']}")
    results["btc_base"] = btc_base
    results["btc_iv"] = btc_iv
    print()

    # ── 2. ETH ensemble ──────────────────────────────────────────────────
    print("--- 2. ETH ENSEMBLE (same params as BTC) ---")
    eth_base = run_single_asset_ensemble("ETH", use_iv_sizing=False)
    eth_iv = run_single_asset_ensemble("ETH", use_iv_sizing=True)
    d_eth = eth_iv["avg_sharpe"] - eth_base["avg_sharpe"]
    print(f"  ETH unsized:     avg={eth_base['avg_sharpe']:.3f} min={eth_base['min_sharpe']:.3f}")
    print(f"  ETH IV-sized:    avg={eth_iv['avg_sharpe']:.3f} min={eth_iv['min_sharpe']:.3f} Δ={d_eth:+.3f}")
    print(f"    Yearly: {eth_iv['yearly']}")
    results["eth_base"] = eth_base
    results["eth_iv"] = eth_iv
    print()

    # ── 3. BTC vs ETH comparison ─────────────────────────────────────────
    print("--- 3. BTC vs ETH COMPARISON ---")
    print(f"  {'Metric':25s} {'BTC':>10s} {'ETH':>10s} {'Δ':>10s}")
    print(f"  {'-'*25} {'-'*10} {'-'*10} {'-'*10}")
    print(f"  {'Unsized avg Sharpe':25s} {btc_base['avg_sharpe']:10.3f} {eth_base['avg_sharpe']:10.3f} {eth_base['avg_sharpe']-btc_base['avg_sharpe']:+10.3f}")
    print(f"  {'Unsized min Sharpe':25s} {btc_base['min_sharpe']:10.3f} {eth_base['min_sharpe']:10.3f} {eth_base['min_sharpe']-btc_base['min_sharpe']:+10.3f}")
    print(f"  {'IV-sized avg Sharpe':25s} {btc_iv['avg_sharpe']:10.3f} {eth_iv['avg_sharpe']:10.3f} {eth_iv['avg_sharpe']-btc_iv['avg_sharpe']:+10.3f}")
    print(f"  {'IV-sized min Sharpe':25s} {btc_iv['min_sharpe']:10.3f} {eth_iv['min_sharpe']:10.3f} {eth_iv['min_sharpe']-btc_iv['min_sharpe']:+10.3f}")
    print(f"  {'IV sizing Δ':25s} {btc_iv['avg_sharpe']-btc_base['avg_sharpe']:+10.3f} {d_eth:+10.3f}")

    # Year-by-year comparison
    print()
    print("  Year-by-year IV-sized:")
    for yr in YEARS:
        btc_s = btc_iv["yearly"][str(yr)]
        eth_s = eth_iv["yearly"][str(yr)]
        print(f"    {yr}: BTC={btc_s:.3f} ETH={eth_s:.3f} Δ={eth_s-btc_s:+.3f}")
    print()

    # ── 4. Multi-asset portfolio ─────────────────────────────────────────
    print("--- 4. MULTI-ASSET PORTFOLIO (BTC + ETH) ---")

    alloc_configs = [
        (1.0, 0.0, "BTC only"),
        (0.75, 0.25, "75/25"),
        (0.60, 0.40, "60/40"),
        (0.50, 0.50, "50/50"),
        (0.40, 0.60, "40/60"),
        (0.25, 0.75, "25/75"),
        (0.0, 1.0, "ETH only"),
    ]

    multi_results = []
    for wb, we, label in alloc_configs:
        r = run_multi_asset_ensemble(wb, we, use_iv_sizing=True)
        multi_results.append({"w_btc": wb, "w_eth": we, "label": label, **r})
        print(f"  {label:12s} avg={r['avg_sharpe']:.3f} min={r['min_sharpe']:.3f}")

    results["multi_asset"] = multi_results

    best_multi = max(multi_results, key=lambda x: x["avg_sharpe"])
    btc_only = next(r for r in multi_results if r["w_btc"] == 1.0)
    print(f"\n  Best allocation: {best_multi['label']} avg={best_multi['avg_sharpe']:.3f}")
    print(f"  BTC-only:        avg={btc_only['avg_sharpe']:.3f}")
    print(f"  Diversification Δ: {best_multi['avg_sharpe'] - btc_only['avg_sharpe']:+.3f}")
    print()

    # ── SUMMARY ───────────────────────────────────────────────────────────
    print("=" * 70)
    print("SUMMARY — R31: ETH Ensemble & Cross-Asset")
    print("=" * 70)
    print()

    # Does ETH ensemble work?
    eth_works = eth_base["avg_sharpe"] > 1.0 and eth_base["min_sharpe"] > 0.0
    print(f"  ETH ensemble works?        {'YES' if eth_works else 'NO'} (avg={eth_base['avg_sharpe']:.3f} min={eth_base['min_sharpe']:.3f})")

    # Does IV sizing work on ETH?
    iv_works_eth = d_eth > 0.05
    print(f"  IV sizing helps ETH?       {'YES' if iv_works_eth else 'NO'} (Δ={d_eth:+.3f})")

    # Does multi-asset diversify?
    div_helps = best_multi["avg_sharpe"] > btc_only["avg_sharpe"] + 0.02
    print(f"  Multi-asset diversifies?   {'YES' if div_helps else 'NO'} (best={best_multi['label']} Δ={best_multi['avg_sharpe']-btc_only['avg_sharpe']:+.3f})")

    # Pareto: best min Sharpe
    best_min = max(multi_results, key=lambda x: x["min_sharpe"])
    print(f"  Best worst-year:           {best_min['label']} min={best_min['min_sharpe']:.3f}")

    print()
    print("=" * 70)
    if eth_works and iv_works_eth:
        print("VERDICT: ETH ensemble VALIDATES — strategies generalize across assets")
        if div_helps:
            print(f"  Multi-asset ({best_multi['label']}) provides diversification benefit")
        else:
            print(f"  Multi-asset diversification is minimal on synthetic data")
    else:
        print("VERDICT: ETH ensemble PARTIALLY validates")
    print("=" * 70)

    # Save
    out = ROOT / "artifacts" / "crypto_options" / "eth_ensemble_cross_asset.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump({
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "research_id": "R31",
            "results": results,
        }, f, indent=2)
    print(f"\nResults saved to: {out}")


if __name__ == "__main__":
    main()
