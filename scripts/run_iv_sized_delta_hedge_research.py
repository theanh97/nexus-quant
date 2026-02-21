#!/usr/bin/env python3
"""
IV-Sized Ensemble with Delta Hedging — Production Viability Test
=================================================================

R25 shows IV-percentile sizing improves VRP+Skew ensemble by +0.276.
But does this improvement survive delta hedging costs?

Delta hedging (R18+R19) degrades unmodified ensemble by ~-6.6% Sharpe.
If IV sizing's improvement is larger than hedge degradation, it's production-viable.

Test matrix:
  - Unhedged baseline vs IV-sized unhedged
  - 25% tolerance hedged baseline vs 25% tolerance hedged IV-sized
  - 40% tolerance hedged baseline vs 40% tolerance hedged IV-sized
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

# Delta hedge configs to test
HEDGE_CONFIGS = {
    "no_hedge": None,
    "25pct_tol": {
        "delta_tolerance": 0.25,
        "hedge_fee_bps": 2.0,
        "hedge_slippage_bps": 1.0,
        "funding_rate_annual": 0.10,
        "bidirectional_funding": True,
    },
    "40pct_tol": {
        "delta_tolerance": 0.40,
        "hedge_fee_bps": 2.0,
        "hedge_slippage_bps": 1.0,
        "funding_rate_annual": 0.10,
        "bidirectional_funding": True,
    },
}


def iv_percentile(iv_series: List, idx: int, lookback: int = 180) -> Optional[float]:
    """Compute IV percentile rank."""
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
    """R25 champion: <25th→0.5x, 25-75th→1.0x, >75th→1.5x."""
    if pct < 0.25:
        return 0.5
    elif pct > 0.75:
        return 1.5
    return 1.0


def run_ensemble(
    use_iv_sizing: bool = False,
    hedge_config: Optional[Dict] = None,
) -> Dict[str, Any]:
    """Run VRP+Skew ensemble with optional IV sizing and delta hedging."""
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

        # Delta hedge state
        use_hedge = hedge_config is not None
        if use_hedge:
            delta_tol = hedge_config["delta_tolerance"]
            h_fee = hedge_config["hedge_fee_bps"] / 10000.0
            h_slip = hedge_config["hedge_slippage_bps"] / 10000.0
            funding_annual = hedge_config["funding_rate_annual"]
            bidirectional = hedge_config["bidirectional_funding"]
            funding_per_bar = funding_annual / BARS_PER_YEAR
            hedge_position = 0.0

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

            # -- Delta hedge costs --
            if use_hedge and abs(W_VRP * w_v) > 1e-10:
                # Approximate straddle delta drift from price moves
                closes = dataset.perp_close.get(sym, [])
                if idx < len(closes) and closes[idx - 1] > 0:
                    price_ret = closes[idx] / closes[idx - 1] - 1.0
                    # Short straddle delta ≈ -2 * N'(d1) * price_ret / IV
                    # Simplified: delta drift ≈ price_ret / IV * 0.8 (BS approximation)
                    iv_h = iv_series[idx - 1] if iv_series and idx - 1 < len(iv_series) and iv_series[idx - 1] else 0.60
                    straddle_delta_drift = price_ret / max(iv_h * math.sqrt(dt), 0.01) * 0.4
                    net_delta = abs(W_VRP * w_v) * straddle_delta_drift + hedge_position

                    if abs(net_delta) > delta_tol:
                        trade_size = -net_delta
                        hcost = abs(trade_size) * (h_fee + h_slip)
                        bar_pnl -= hcost
                        hedge_position += trade_size

                    # Funding on hedge position
                    if abs(hedge_position) > 0.001:
                        if bidirectional:
                            fpnl = hedge_position * funding_per_bar
                        else:
                            fpnl = abs(hedge_position) * funding_per_bar
                        bar_pnl -= fpnl

            dp = equity * bar_pnl
            equity += dp

            # -- Rebalance --
            if vrp_strat.should_rebalance(dataset, idx):
                target_v = vrp_strat.target_weights(dataset, idx, vrp_weights)
                target_s = skew_strat.target_weights(dataset, idx, skew_weights) if skew_strat.should_rebalance(dataset, idx) else skew_weights

                # Apply IV sizing
                if use_iv_sizing:
                    iv_series = dataset.features.get("iv_atm", {}).get(sym, [])
                    pct = iv_percentile(iv_series, idx)
                    if pct is not None:
                        scale = step_sizing(pct)
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

                # Reset hedge on rebalance
                if use_hedge:
                    hedge_position = 0.0

            elif skew_strat.should_rebalance(dataset, idx):
                target_s = skew_strat.target_weights(dataset, idx, skew_weights)
                if use_iv_sizing:
                    iv_series = dataset.features.get("iv_atm", {}).get(sym, [])
                    pct = iv_percentile(iv_series, idx)
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
    print("IV-SIZED ENSEMBLE WITH DELTA HEDGING — PRODUCTION VIABILITY")
    print("=" * 70)
    print()

    results = {}

    for hedge_name, hedge_cfg in HEDGE_CONFIGS.items():
        print(f"\n--- Hedge: {hedge_name} ---")

        # Baseline (no IV sizing)
        base = run_ensemble(use_iv_sizing=False, hedge_config=hedge_cfg)
        print(f"  Baseline:   avg={base['avg_sharpe']:.3f} min={base['min_sharpe']:.3f}")
        print(f"    Yearly: {base['yearly']}")

        # IV-sized
        sized = run_ensemble(use_iv_sizing=True, hedge_config=hedge_cfg)
        delta = sized["avg_sharpe"] - base["avg_sharpe"]
        print(f"  IV-sized:   avg={sized['avg_sharpe']:.3f} min={sized['min_sharpe']:.3f} Δ={delta:+.3f}")
        print(f"    Yearly: {sized['yearly']}")

        results[hedge_name] = {
            "baseline": base,
            "iv_sized": sized,
            "delta": round(delta, 3),
        }

    # Summary
    print()
    print("=" * 70)
    print("SUMMARY — Does IV sizing survive delta hedging?")
    print("=" * 70)
    print()
    print(f"  {'Config':20s} {'Baseline':>10s} {'IV-Sized':>10s} {'Δ':>8s} {'Sizing survives?':>18s}")
    print(f"  {'-'*20} {'-'*10} {'-'*10} {'-'*8} {'-'*18}")

    for name, r in results.items():
        base_sh = r["baseline"]["avg_sharpe"]
        sized_sh = r["iv_sized"]["avg_sharpe"]
        d = r["delta"]
        survives = "YES" if d > 0.05 else ("marginal" if d > 0 else "NO")
        print(f"  {name:20s} {base_sh:10.3f} {sized_sh:10.3f} {d:+8.3f} {survives:>18s}")

    # Check if hedged IV-sized > unhedged baseline
    if results.get("no_hedge") and results.get("25pct_tol"):
        unhedged_base = results["no_hedge"]["baseline"]["avg_sharpe"]
        hedged_sized = results["25pct_tol"]["iv_sized"]["avg_sharpe"]
        hedged_base = results["25pct_tol"]["baseline"]["avg_sharpe"]
        d_net = hedged_sized - unhedged_base
        print()
        print(f"  KEY: Hedged+IV-sized ({hedged_sized:.3f}) vs Unhedged baseline ({unhedged_base:.3f}): Δ={d_net:+.3f}")
        print(f"  KEY: Hedge cost alone ({hedged_base:.3f} vs {unhedged_base:.3f}): Δ={hedged_base - unhedged_base:+.3f}")
        print(f"  KEY: IV sizing improvement survives {abs(hedged_sized - hedged_base):.3f} of {abs(hedged_base - unhedged_base):.3f} hedge drag")

    print()
    all_survive = all(r["delta"] > 0 for r in results.values())
    if all_survive:
        print("VERDICT: IV sizing improvement SURVIVES delta hedging costs")
        print("  Production-viable: use step_lb180 sizing with 25% hedge tolerance")
    else:
        print("VERDICT: IV sizing improvement is partially absorbed by hedge costs")
    print("=" * 70)

    # Save
    out = ROOT / "artifacts" / "crypto_options" / "iv_sized_delta_hedge_research.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump({
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "results": results,
        }, f, indent=2)
    print(f"\nResults saved to: {out}")


if __name__ == "__main__":
    main()
