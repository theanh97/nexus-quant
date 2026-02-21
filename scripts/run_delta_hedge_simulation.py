#!/usr/bin/env python3
"""
Delta Hedging Simulation for VRP Strategy
==========================================

The current VRP backtest assumes costless, perfect delta hedging.
In production, selling straddles requires continuous delta hedging
with perpetual futures, which adds real costs:

  1. Perpetual taker fees (5 bps per hedge trade)
  2. Slippage on hedge rebalances
  3. Funding rate cost on hedge position
  4. Gamma bleed between rebalances (imperfect hedging)

This simulation models explicit delta hedging to answer:
  "How much does VRP Sharpe degrade with realistic hedging costs?"

Model:
  - Sell ATM straddle (call + put) at start of each position
  - Track position delta using Black-Scholes as price moves
  - Rebalance hedge when delta drifts beyond tolerance band
  - Compute hedge costs from perpetual trading fees + slippage
  - Compare: perfect hedge (current model) vs explicit hedge

Expected results:
  - Sharpe degradation: 10-30% (literature estimate)
  - If VRP daily Sharpe drops from 2.088 to >1.5: still validates
  - If hourly VRP drops from 3.649 to >2.5: still validates
"""
import json
import math
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nexus_quant.backtest.costs import ExecutionCostModel, FeeModel, ImpactModel
from nexus_quant.projects.crypto_options.options_engine import compute_metrics, run_yearly_wf
from nexus_quant.projects.crypto_options.providers.deribit_rest import DeribitRestProvider
from nexus_quant.projects.crypto_options.strategies.variance_premium import VariancePremiumStrategy
from nexus_quant.projects.crypto_options.greeks import (
    bs_delta, bs_gamma, bs_theta, bs_vega, bs_price,
)

PROVIDER_CFG = {
    "provider": "deribit_rest_v1",
    "symbols": ["BTC", "ETH"],
    "bar_interval": "1d",
    "cache_dir": "data/cache/deribit",
    "use_synthetic_iv": True,
    "rv_lookback_bars": 21,
}

YEARS = [2021, 2022, 2023, 2024, 2025]

_FEE = FeeModel(maker_fee_rate=0.0003, taker_fee_rate=0.0005)
_IMPACT = ImpactModel(model="sqrt", coef_bps=3.0)
COSTS = ExecutionCostModel(
    fee=_FEE, execution_style="taker",
    slippage_bps=7.5, spread_bps=10.0, impact=_IMPACT, cost_multiplier=1.0,
)


# ── Delta Hedge Parameters ──────────────────────────────────────────────────

# Hedge rebalance triggers
HEDGE_CONFIGS = {
    "tight": {
        "delta_tolerance": 0.05,        # rebalance when |delta| > 5% of notional
        "hedge_fee_bps": 5.0,           # perpetual taker fee
        "hedge_slippage_bps": 3.0,      # slippage on hedge trade
        "funding_rate_annual": 0.10,    # 10% annual funding (bull market average)
    },
    "normal": {
        "delta_tolerance": 0.10,        # 10% tolerance
        "hedge_fee_bps": 5.0,
        "hedge_slippage_bps": 3.0,
        "funding_rate_annual": 0.10,
    },
    "wide": {
        "delta_tolerance": 0.20,        # 20% tolerance (less hedging)
        "hedge_fee_bps": 5.0,
        "hedge_slippage_bps": 3.0,
        "funding_rate_annual": 0.10,
    },
    "no_funding": {
        "delta_tolerance": 0.10,
        "hedge_fee_bps": 5.0,
        "hedge_slippage_bps": 3.0,
        "funding_rate_annual": 0.0,     # zero funding (bear market)
    },
    "high_funding": {
        "delta_tolerance": 0.10,
        "hedge_fee_bps": 5.0,
        "hedge_slippage_bps": 3.0,
        "funding_rate_annual": 0.20,    # 20% annual funding (extreme bull)
    },
}


def run_vrp_with_delta_hedge(
    year: int,
    hedge_cfg: Dict[str, float],
    bars_per_year: int = 365,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Run VRP backtest with explicit delta hedging simulation.

    Model:
    1. At each strategy rebalance, "sell" ATM straddle:
       - Short call + short put at ATM strike
       - Initial delta ≈ 0 (straddle is delta-neutral at ATM)
    2. Between rebalances, as price moves:
       - Compute new delta from BS model
       - If |delta| exceeds tolerance, rebalance hedge
       - Hedge cost = |delta_change| × notional × (fee + slippage)
    3. Funding cost on cumulative hedge position each bar
    4. Compare total PnL vs. perfect-hedge baseline

    Returns:
        dict with equity_curve, returns, metrics, hedge stats
    """
    cfg = dict(PROVIDER_CFG)
    cfg["start"] = f"{year}-01-01"
    cfg["end"] = f"{year}-12-31"

    provider = DeribitRestProvider(cfg, seed=seed)
    dataset = provider.load()

    syms = dataset.symbols
    n = len(dataset.timeline)
    dt = 1.0 / bars_per_year

    # Strategy
    strat = VariancePremiumStrategy(params={
        "base_leverage": 1.5,
        "exit_z_threshold": -3.0,
        "vrp_lookback": 30,
        "rebalance_freq": 5,
        "min_bars": 30,
    })

    # State
    equity = 1.0
    weights = {s: 0.0 for s in syms}
    equity_curve = [1.0]
    returns_list = []

    # Delta hedge tracking per symbol
    hedge_state = {s: {
        "cumulative_delta": 0.0,  # net delta exposure (from both call+put)
        "hedge_position": 0.0,   # perpetual hedge size
        "last_strike": 0.0,      # ATM strike when position opened
        "tte": 30.0 / 365.0,     # time to expiry (30 DTE rolling)
    } for s in syms}

    # Cumulative stats
    total_theta = 0.0
    total_gamma_cost = 0.0
    total_hedge_cost = 0.0
    total_funding_cost = 0.0
    total_hedge_trades = 0
    total_gamma_bleed = 0.0

    delta_tolerance = hedge_cfg["delta_tolerance"]
    hedge_fee = hedge_cfg["hedge_fee_bps"] / 10000.0
    hedge_slip = hedge_cfg["hedge_slippage_bps"] / 10000.0
    funding_per_bar = hedge_cfg["funding_rate_annual"] / bars_per_year

    for idx in range(1, n):
        prev_equity = equity
        bar_pnl = 0.0

        for sym in syms:
            w = weights.get(sym, 0.0)
            if abs(w) < 1e-10:
                hedge_state[sym]["cumulative_delta"] = 0.0
                hedge_state[sym]["hedge_position"] = 0.0
                continue

            # Get prices
            closes = dataset.perp_close.get(sym, [])
            if idx >= len(closes) or closes[idx - 1] <= 0 or closes[idx] <= 0:
                continue

            S_prev = closes[idx - 1]
            S_now = closes[idx]
            log_ret = math.log(S_now / S_prev)

            # Get IV
            iv_series = dataset.features.get("iv_atm", {}).get(sym, [])
            iv = iv_series[idx - 1] if idx - 1 < len(iv_series) and iv_series[idx - 1] else 0.70

            # ── Theta income (same as current model) ──
            rv_bar = abs(log_ret) * math.sqrt(bars_per_year)
            vrp_pnl = 0.5 * (iv ** 2 - rv_bar ** 2) * dt
            theta_pnl = (-w) * vrp_pnl  # short vol → (-w) > 0
            total_theta += theta_pnl * equity if vrp_pnl > 0 else 0
            total_gamma_cost += abs(theta_pnl * equity) if vrp_pnl < 0 else 0

            # ── Delta drift from price movement ──
            hs = hedge_state[sym]
            K = hs["last_strike"] if hs["last_strike"] > 0 else S_prev
            tte = max(hs["tte"] - dt, 1.0 / 365)  # decay TTE, min 1 day

            # Straddle delta = delta_call + delta_put
            # At ATM: delta_call ≈ 0.5, delta_put ≈ -0.5, net ≈ 0
            # As price moves: delta drifts
            r_rate = 0.0  # crypto: no risk-free rate
            try:
                delta_call = bs_delta(S_now, K, tte, r_rate, iv, "call")
                delta_put = bs_delta(S_now, K, tte, r_rate, iv, "put")
                straddle_delta = delta_call + delta_put  # net delta of short straddle
            except (ValueError, ZeroDivisionError):
                straddle_delta = 0.0

            # Position delta = straddle_delta × leverage × contracts
            # For short straddle: we sold both → position delta = -straddle_delta × |w|
            position_delta = -straddle_delta * abs(w)

            # Net delta = position delta + hedge position
            net_delta = position_delta + hs["hedge_position"]

            # ── Gamma bleed (unhedged delta × price move) ──
            # This is the cost of imperfect hedging between rebalances
            delta_pnl = net_delta * (S_now / S_prev - 1.0)
            gamma_bleed = abs(delta_pnl)
            total_gamma_bleed += gamma_bleed * equity

            # ── Check if hedge rebalance needed ──
            if abs(net_delta) > delta_tolerance:
                # Hedge trade: bring net delta back to 0
                hedge_trade_size = -net_delta  # buy/sell perps to offset
                hedge_notional = abs(hedge_trade_size) * S_now

                # Cost of hedge trade
                hedge_trade_cost = hedge_notional * (hedge_fee + hedge_slip)
                hedge_cost_pct = hedge_trade_cost / (equity * S_prev) if equity > 0 else 0

                bar_pnl -= hedge_cost_pct
                total_hedge_cost += hedge_cost_pct * equity
                total_hedge_trades += 1

                # Update hedge position
                hs["hedge_position"] += hedge_trade_size

            # ── Funding cost on hedge position ──
            if abs(hs["hedge_position"]) > 0.001:
                funding_pnl = abs(hs["hedge_position"]) * funding_per_bar
                bar_pnl -= funding_pnl
                total_funding_cost += funding_pnl * equity

            # ── Apply VRP P&L (same as current model) ──
            bar_pnl += theta_pnl

            # Update TTE
            hs["tte"] = tte

        # Apply transaction costs for strategy rebalance
        if strat.should_rebalance(dataset, idx):
            target = strat.target_weights(dataset, idx, weights)
            for s in syms:
                target.setdefault(s, 0.0)
            turnover = sum(abs(float(target.get(s, 0)) - float(weights.get(s, 0))) for s in syms)

            if turnover > 1e-6:
                bd = COSTS.cost(equity=equity, turnover=turnover)
                cost = float(bd.get("cost", 0))
                equity -= cost

            # Reset hedge state for new position
            for s in syms:
                new_w = float(target.get(s, 0))
                if abs(new_w) > 1e-10:
                    closes = dataset.perp_close.get(s, [])
                    if idx < len(closes) and closes[idx] > 0:
                        hedge_state[s]["last_strike"] = closes[idx]
                        hedge_state[s]["tte"] = 30.0 / 365.0  # reset to 30 DTE
                        hedge_state[s]["cumulative_delta"] = 0.0
                        hedge_state[s]["hedge_position"] = 0.0
                else:
                    hedge_state[s]["hedge_position"] = 0.0
                    hedge_state[s]["cumulative_delta"] = 0.0

            weights = {s: float(target.get(s, 0)) for s in syms}

        dp = equity * bar_pnl
        equity += dp
        equity = max(equity, 0.0)
        equity_curve.append(equity)
        bar_ret = (equity / prev_equity - 1.0) if prev_equity > 0 else 0.0
        returns_list.append(bar_ret)

    metrics = compute_metrics(equity_curve, returns_list, bars_per_year)

    return {
        "year": year,
        "metrics": metrics,
        "hedge_stats": {
            "total_hedge_trades": total_hedge_trades,
            "total_hedge_cost_pct": round(total_hedge_cost / max(equity_curve[0], 1e-10) * 100, 4),
            "total_funding_cost_pct": round(total_funding_cost / max(equity_curve[0], 1e-10) * 100, 4),
            "total_gamma_bleed_pct": round(total_gamma_bleed / max(equity_curve[0], 1e-10) * 100, 4),
            "avg_hedge_trades_per_bar": round(total_hedge_trades / max(n - 1, 1), 2),
        },
    }


def run_baseline_vrp(year: int, bars_per_year: int = 365, seed: int = 42) -> Dict[str, float]:
    """Run baseline VRP (no explicit hedge) for comparison."""
    from nexus_quant.projects.crypto_options.options_engine import (
        OptionsBacktestEngine, compute_metrics,
    )

    cfg = dict(PROVIDER_CFG)
    cfg["start"] = f"{year}-01-01"
    cfg["end"] = f"{year}-12-31"

    provider = DeribitRestProvider(cfg, seed=seed)
    dataset = provider.load()

    engine = OptionsBacktestEngine(costs=COSTS, bars_per_year=365, use_options_pnl=True)
    strat = VariancePremiumStrategy(params={
        "base_leverage": 1.5,
        "exit_z_threshold": -3.0,
        "vrp_lookback": 30,
        "rebalance_freq": 5,
        "min_bars": 30,
    })
    result = engine.run(dataset, strat)
    m = compute_metrics(result.equity_curve, result.returns, bars_per_year)
    return m


def main():
    t0 = time.time()

    print("=" * 70)
    print("  VRP DELTA HEDGING SIMULATION")
    print("=" * 70)
    print()

    # ── 1. Baseline (current model — perfect hedge assumed) ──
    print("  BASELINE (perfect hedge, current model):")
    baseline_sharpes = {}
    for yr in YEARS:
        m = run_baseline_vrp(yr)
        baseline_sharpes[yr] = m["sharpe"]
        print(f"    {yr}: Sharpe={m['sharpe']:+.4f}")
    avg_baseline = sum(baseline_sharpes.values()) / len(baseline_sharpes)
    min_baseline = min(baseline_sharpes.values())
    print(f"    Avg={avg_baseline:+.4f} Min={min_baseline:+.4f}")
    print()

    # ── 2. With explicit delta hedge ──
    all_results = {}
    for cfg_name, cfg in HEDGE_CONFIGS.items():
        print(f"  HEDGE CONFIG: {cfg_name}")
        print(f"    delta_tol={cfg['delta_tolerance']:.0%} "
              f"fee={cfg['hedge_fee_bps']:.0f}bps "
              f"slip={cfg['hedge_slippage_bps']:.0f}bps "
              f"funding={cfg['funding_rate_annual']:.0%}")

        yearly_sharpes = {}
        yearly_stats = {}
        for yr in YEARS:
            r = run_vrp_with_delta_hedge(yr, cfg)
            sh = r["metrics"]["sharpe"]
            yearly_sharpes[yr] = sh
            yearly_stats[yr] = r["hedge_stats"]
            print(f"    {yr}: Sharpe={sh:+.4f} "
                  f"hedges={r['hedge_stats']['total_hedge_trades']} "
                  f"hedge_cost={r['hedge_stats']['total_hedge_cost_pct']:.2f}% "
                  f"funding={r['hedge_stats']['total_funding_cost_pct']:.2f}%")

        avg_sh = sum(yearly_sharpes.values()) / len(yearly_sharpes)
        min_sh = min(yearly_sharpes.values())
        degradation = (avg_sh - avg_baseline) / abs(avg_baseline) * 100 if abs(avg_baseline) > 0 else 0
        print(f"    Avg={avg_sh:+.4f} Min={min_sh:+.4f} "
              f"(degradation: {degradation:+.1f}% vs baseline)")
        print()

        all_results[cfg_name] = {
            "config": cfg,
            "yearly_sharpes": yearly_sharpes,
            "yearly_stats": yearly_stats,
            "avg_sharpe": round(avg_sh, 4),
            "min_sharpe": round(min_sh, 4),
            "degradation_pct": round(degradation, 2),
        }

    # ── 3. Summary ──
    print("=" * 70)
    print("  SUMMARY: BASELINE vs EXPLICIT DELTA HEDGE")
    print("=" * 70)
    print(f"  {'Config':20s} {'Avg Sharpe':>12} {'Min Sharpe':>12} {'Degradation':>12}")
    print("  " + "-" * 58)
    print(f"  {'BASELINE (perfect)':20s} {avg_baseline:>12.4f} {min_baseline:>12.4f} {'---':>12}")

    for cfg_name, res in sorted(all_results.items(), key=lambda x: x[1]["avg_sharpe"], reverse=True):
        print(f"  {cfg_name:20s} {res['avg_sharpe']:>12.4f} {res['min_sharpe']:>12.4f} "
              f"{res['degradation_pct']:>+11.1f}%")

    # ── 4. Verdict ──
    best_realistic = max(all_results.values(), key=lambda x: x["avg_sharpe"])
    worst_realistic = min(all_results.values(), key=lambda x: x["avg_sharpe"])

    print()
    print(f"  VERDICT:")
    print(f"    Best realistic Sharpe: {best_realistic['avg_sharpe']:.4f} "
          f"({best_realistic['degradation_pct']:+.1f}% vs perfect hedge)")
    print(f"    Worst realistic Sharpe: {worst_realistic['avg_sharpe']:.4f} "
          f"({worst_realistic['degradation_pct']:+.1f}% vs perfect hedge)")

    if best_realistic["avg_sharpe"] >= 1.5:
        print(f"    VRP SURVIVES delta hedging costs (Sharpe >= 1.5)")
    elif best_realistic["avg_sharpe"] >= 1.0:
        print(f"    VRP MARGINAL with delta hedging costs (1.0 <= Sharpe < 1.5)")
    else:
        print(f"    VRP FAILS with delta hedging costs (Sharpe < 1.0)")

    # ── 5. Impact on ensemble ──
    # If VRP Sharpe degrades by X%, estimate ensemble impact
    normal_cfg = all_results.get("normal", best_realistic)
    vrp_degradation_factor = normal_cfg["avg_sharpe"] / avg_baseline if avg_baseline > 0 else 1.0

    # Current ensemble: VRP 40% + Skew MR 60%
    # VRP daily Sharpe 2.088 → degraded: 2.088 × factor
    degraded_daily_vrp = 2.088 * vrp_degradation_factor
    # Ensemble: 0.30 × degraded_daily_vrp + 0.70 × 1.744 (daily skew unchanged)
    degraded_ensemble = 0.30 * degraded_daily_vrp + 0.70 * 1.744
    current_ensemble = 0.30 * 2.088 + 0.70 * 1.744

    print()
    print(f"  ENSEMBLE IMPACT (daily-only, 30/70):")
    print(f"    VRP daily: {2.088:.3f} → {degraded_daily_vrp:.3f} ({vrp_degradation_factor:.0%})")
    print(f"    Ensemble: {current_ensemble:.3f} → {degraded_ensemble:.3f}")
    print()

    # Save
    os.makedirs("artifacts/crypto_options", exist_ok=True)
    output = {
        "date": time.strftime("%Y-%m-%d %H:%M"),
        "baseline": {
            "avg_sharpe": round(avg_baseline, 4),
            "min_sharpe": round(min_baseline, 4),
            "yearly": baseline_sharpes,
        },
        "hedge_results": all_results,
        "vrp_degradation_factor": round(vrp_degradation_factor, 4),
        "degraded_ensemble": round(degraded_ensemble, 4),
        "elapsed_sec": round(time.time() - t0, 1),
    }
    out_path = "artifacts/crypto_options/delta_hedge_simulation.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"  Results saved to {out_path}")
    print(f"  Total elapsed: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
