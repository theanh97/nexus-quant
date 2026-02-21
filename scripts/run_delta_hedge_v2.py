#!/usr/bin/env python3
"""
Delta Hedging Simulation v2 — Corrected Funding Model + Optimizations
======================================================================

v1 found devastating Sharpe degradation (-50% to -96%), but had a CRITICAL
flaw: it treated ALL funding as a cost (abs(hedge_position)).

In reality, funding is BIDIRECTIONAL:
  - Short perps (hedge after price rise) → RECEIVE funding in bull markets
  - Long perps (hedge after price drop) → PAY funding in bull markets
  - Net: roughly zero on average (price moves ~symmetric around straddle)

v2 improvements:
  1. Bidirectional funding model (signed position × signed funding rate)
  2. Tolerance sweep from 5% to 40% to find optimal band
  3. Maker vs taker fee comparison (2bps vs 5bps)
  4. Combined optimal config: best tolerance + maker fees + bidirectional funding
  5. Per-year funding rate calibration from actual crypto market regimes

Funding rate regimes (annualized, from historical data):
  2021: +18%  (extreme bull, high long demand)
  2022: -5%   (bear market, negative funding)
  2023: +8%   (recovery)
  2024: +12%  (bull resumption)
  2025: +10%  (moderate bull)
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
from nexus_quant.projects.crypto_options.greeks import bs_delta

PROVIDER_CFG = {
    "provider": "deribit_rest_v1",
    "symbols": ["BTC", "ETH"],
    "bar_interval": "1d",
    "cache_dir": "data/cache/deribit",
    "use_synthetic_iv": True,
    "rv_lookback_bars": 21,
}

YEARS = [2021, 2022, 2023, 2024, 2025]

# Historical annualized funding rates by year (approximate from Binance perps)
# Positive = longs pay shorts, Negative = shorts pay longs
ANNUAL_FUNDING_RATES = {
    2021: 0.18,    # extreme bull → high positive funding
    2022: -0.05,   # bear market → slightly negative
    2023: 0.08,    # recovery → moderate positive
    2024: 0.12,    # bull resumption
    2025: 0.10,    # moderate bull
}

_FEE = FeeModel(maker_fee_rate=0.0002, taker_fee_rate=0.0005)
_IMPACT = ImpactModel(model="sqrt", coef_bps=3.0)
COSTS = ExecutionCostModel(
    fee=_FEE, execution_style="taker",
    slippage_bps=7.5, spread_bps=10.0, impact=_IMPACT, cost_multiplier=1.0,
)


def run_vrp_with_hedge_v2(
    year: int,
    delta_tolerance: float = 0.10,
    hedge_fee_bps: float = 5.0,
    hedge_slippage_bps: float = 3.0,
    use_bidirectional_funding: bool = True,
    funding_override: Optional[float] = None,
    bars_per_year: int = 365,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Run VRP with explicit delta hedge — corrected funding model.

    Key change from v1: funding is SIGNED, not absolute.
    - hedge_position > 0 means LONG perps → pays positive funding
    - hedge_position < 0 means SHORT perps → receives positive funding
    - Net funding = hedge_position × funding_rate_per_bar (can be + or -)
    """
    cfg = dict(PROVIDER_CFG)
    cfg["start"] = f"{year}-01-01"
    cfg["end"] = f"{year}-12-31"

    provider = DeribitRestProvider(cfg, seed=seed)
    dataset = provider.load()

    syms = dataset.symbols
    n = len(dataset.timeline)
    dt = 1.0 / bars_per_year

    strat = VariancePremiumStrategy(params={
        "base_leverage": 1.5,
        "exit_z_threshold": -3.0,
        "vrp_lookback": 30,
        "rebalance_freq": 5,
        "min_bars": 30,
    })

    equity = 1.0
    weights = {s: 0.0 for s in syms}
    equity_curve = [1.0]
    returns_list = []

    # Delta hedge state per symbol
    hedge_state = {s: {
        "hedge_position": 0.0,
        "last_strike": 0.0,
        "tte": 30.0 / 365.0,
    } for s in syms}

    # Costs tracking
    total_hedge_cost = 0.0
    total_funding_cost = 0.0  # net (can be negative = benefit)
    total_funding_received = 0.0  # total funding income
    total_funding_paid = 0.0      # total funding expense
    total_hedge_trades = 0

    # Funding rate for this year
    if funding_override is not None:
        annual_funding = funding_override
    else:
        annual_funding = ANNUAL_FUNDING_RATES.get(year, 0.10)
    funding_per_bar = annual_funding / bars_per_year

    hedge_fee = hedge_fee_bps / 10000.0
    hedge_slip = hedge_slippage_bps / 10000.0

    for idx in range(1, n):
        prev_equity = equity
        bar_pnl = 0.0

        for sym in syms:
            w = weights.get(sym, 0.0)
            hs = hedge_state[sym]

            if abs(w) < 1e-10:
                hs["hedge_position"] = 0.0
                continue

            closes = dataset.perp_close.get(sym, [])
            if idx >= len(closes) or closes[idx - 1] <= 0 or closes[idx] <= 0:
                continue

            S_prev = closes[idx - 1]
            S_now = closes[idx]
            log_ret = math.log(S_now / S_prev)

            # IV
            iv_series = dataset.features.get("iv_atm", {}).get(sym, [])
            iv = iv_series[idx - 1] if idx - 1 < len(iv_series) and iv_series[idx - 1] else 0.70

            # ── VRP P&L (same as baseline) ──
            rv_bar = abs(log_ret) * math.sqrt(bars_per_year)
            vrp_pnl = 0.5 * (iv ** 2 - rv_bar ** 2) * dt
            theta_pnl = (-w) * vrp_pnl
            bar_pnl += theta_pnl

            # ── Delta drift ──
            K = hs["last_strike"] if hs["last_strike"] > 0 else S_prev
            tte = max(hs["tte"] - dt, 1.0 / 365)

            try:
                delta_call = bs_delta(S_now, K, tte, 0.0, iv, "call")
                delta_put = bs_delta(S_now, K, tte, 0.0, iv, "put")
                straddle_delta = delta_call + delta_put
            except (ValueError, ZeroDivisionError):
                straddle_delta = 0.0

            position_delta = -straddle_delta * abs(w)
            net_delta = position_delta + hs["hedge_position"]

            # ── Rebalance hedge if delta exceeds tolerance ──
            if abs(net_delta) > delta_tolerance:
                hedge_trade_size = -net_delta
                hedge_notional = abs(hedge_trade_size) * S_now
                hedge_trade_cost = hedge_notional * (hedge_fee + hedge_slip)
                hedge_cost_pct = hedge_trade_cost / (equity * S_prev) if equity > 0 else 0
                bar_pnl -= hedge_cost_pct
                total_hedge_cost += hedge_cost_pct * equity
                total_hedge_trades += 1
                hs["hedge_position"] += hedge_trade_size

            # ── Funding cost (CORRECTED: bidirectional) ──
            if abs(hs["hedge_position"]) > 0.001:
                if use_bidirectional_funding:
                    # Signed funding: positive position (long) pays positive funding
                    # Short position receives positive funding
                    funding_pnl = hs["hedge_position"] * funding_per_bar
                    # funding_pnl > 0 when long & positive funding (cost to us)
                    # funding_pnl < 0 when short & positive funding (income to us)
                    bar_pnl -= funding_pnl  # subtract cost (or add income)
                    if funding_pnl > 0:
                        total_funding_paid += funding_pnl * equity
                    else:
                        total_funding_received += abs(funding_pnl) * equity
                    total_funding_cost += funding_pnl * equity
                else:
                    # v1 model: always a cost (conservative)
                    funding_pnl = abs(hs["hedge_position"]) * funding_per_bar
                    bar_pnl -= funding_pnl
                    total_funding_cost += funding_pnl * equity

            hs["tte"] = tte

        # Strategy rebalance
        if strat.should_rebalance(dataset, idx):
            target = strat.target_weights(dataset, idx, weights)
            for s in syms:
                target.setdefault(s, 0.0)
            turnover = sum(abs(float(target.get(s, 0)) - float(weights.get(s, 0))) for s in syms)
            if turnover > 1e-6:
                bd = COSTS.cost(equity=equity, turnover=turnover)
                equity -= float(bd.get("cost", 0))

            for s in syms:
                new_w = float(target.get(s, 0))
                if abs(new_w) > 1e-10:
                    closes = dataset.perp_close.get(s, [])
                    if idx < len(closes) and closes[idx] > 0:
                        hedge_state[s]["last_strike"] = closes[idx]
                        hedge_state[s]["tte"] = 30.0 / 365.0
                        hedge_state[s]["hedge_position"] = 0.0
                else:
                    hedge_state[s]["hedge_position"] = 0.0

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
            "total_funding_net_pct": round(total_funding_cost / max(equity_curve[0], 1e-10) * 100, 4),
            "total_funding_paid_pct": round(total_funding_paid / max(equity_curve[0], 1e-10) * 100, 4),
            "total_funding_received_pct": round(total_funding_received / max(equity_curve[0], 1e-10) * 100, 4),
        },
    }


def run_baseline_vrp(year: int, seed: int = 42) -> Dict[str, float]:
    """Run baseline VRP (perfect hedge)."""
    from nexus_quant.projects.crypto_options.options_engine import OptionsBacktestEngine
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
    return compute_metrics(result.equity_curve, result.returns, 365)


def main():
    t0 = time.time()

    print("=" * 70)
    print("  VRP DELTA HEDGING v2 — CORRECTED FUNDING MODEL")
    print("=" * 70)
    print()

    # ── 1. Baseline ──
    print("  BASELINE (perfect hedge):")
    baseline = {}
    for yr in YEARS:
        m = run_baseline_vrp(yr)
        baseline[yr] = m["sharpe"]
        print(f"    {yr}: Sharpe={m['sharpe']:+.4f}")
    avg_bl = sum(baseline.values()) / len(baseline)
    min_bl = min(baseline.values())
    print(f"    Avg={avg_bl:+.4f} Min={min_bl:+.4f}")
    print()

    all_results = {}

    # ── 2. v1 model (always-cost funding) for comparison ──
    print("  v1 MODEL (always-cost funding, 10% tol, taker):")
    v1_sharpes = {}
    for yr in YEARS:
        r = run_vrp_with_hedge_v2(
            yr, delta_tolerance=0.10, hedge_fee_bps=5.0, hedge_slippage_bps=3.0,
            use_bidirectional_funding=False, funding_override=0.10,
        )
        v1_sharpes[yr] = r["metrics"]["sharpe"]
        print(f"    {yr}: Sharpe={r['metrics']['sharpe']:+.4f} "
              f"funding_net={r['hedge_stats']['total_funding_net_pct']:+.2f}%")
    avg_v1 = sum(v1_sharpes.values()) / len(v1_sharpes)
    print(f"    Avg={avg_v1:+.4f} (degradation: {(avg_v1-avg_bl)/abs(avg_bl)*100:+.1f}%)")
    all_results["v1_always_cost"] = {"avg_sharpe": round(avg_v1, 4), "yearly": dict(v1_sharpes)}
    print()

    # ── 3. v2 bidirectional funding, per-year rates ──
    print("  v2 BIDIRECTIONAL (per-year funding, 10% tol, taker):")
    v2_sharpes = {}
    for yr in YEARS:
        r = run_vrp_with_hedge_v2(
            yr, delta_tolerance=0.10, hedge_fee_bps=5.0, hedge_slippage_bps=3.0,
            use_bidirectional_funding=True,
        )
        v2_sharpes[yr] = r["metrics"]["sharpe"]
        hs = r["hedge_stats"]
        print(f"    {yr}: Sharpe={r['metrics']['sharpe']:+.4f} "
              f"funding_net={hs['total_funding_net_pct']:+.2f}% "
              f"(paid={hs['total_funding_paid_pct']:.2f}% recv={hs['total_funding_received_pct']:.2f}%) "
              f"rate={ANNUAL_FUNDING_RATES.get(yr, 0.10):.0%}")
    avg_v2 = sum(v2_sharpes.values()) / len(v2_sharpes)
    print(f"    Avg={avg_v2:+.4f} (degradation: {(avg_v2-avg_bl)/abs(avg_bl)*100:+.1f}%)")
    all_results["v2_bidirectional"] = {"avg_sharpe": round(avg_v2, 4), "yearly": dict(v2_sharpes)}
    print()

    # ── 4. Tolerance sweep (5% to 40%) with bidirectional funding ──
    print("  TOLERANCE SWEEP (bidirectional funding, taker):")
    print(f"    {'Tol':>6s}  {'Avg':>8s}  {'Min':>8s}  {'Hedges/yr':>10s}  {'HedgeCost':>10s}  {'FundNet':>10s}")
    print("    " + "-" * 60)
    tol_results = {}
    for tol_pct in [5, 8, 10, 15, 20, 25, 30, 40]:
        tol = tol_pct / 100.0
        yearly_sh = {}
        yearly_trades = {}
        yearly_hcost = {}
        yearly_fnet = {}
        for yr in YEARS:
            r = run_vrp_with_hedge_v2(
                yr, delta_tolerance=tol, hedge_fee_bps=5.0, hedge_slippage_bps=3.0,
                use_bidirectional_funding=True,
            )
            yearly_sh[yr] = r["metrics"]["sharpe"]
            yearly_trades[yr] = r["hedge_stats"]["total_hedge_trades"]
            yearly_hcost[yr] = r["hedge_stats"]["total_hedge_cost_pct"]
            yearly_fnet[yr] = r["hedge_stats"]["total_funding_net_pct"]
        avg_sh = sum(yearly_sh.values()) / len(yearly_sh)
        min_sh = min(yearly_sh.values())
        avg_trades = sum(yearly_trades.values()) / len(yearly_trades)
        avg_hcost = sum(yearly_hcost.values()) / len(yearly_hcost)
        avg_fnet = sum(yearly_fnet.values()) / len(yearly_fnet)
        tol_results[tol_pct] = {
            "avg_sharpe": round(avg_sh, 4),
            "min_sharpe": round(min_sh, 4),
            "avg_trades": round(avg_trades, 0),
            "avg_hedge_cost": round(avg_hcost, 2),
            "avg_funding_net": round(avg_fnet, 2),
        }
        print(f"    {tol_pct:>5d}%  {avg_sh:>+8.4f}  {min_sh:>+8.4f}  {avg_trades:>10.0f}  "
              f"{avg_hcost:>+9.2f}%  {avg_fnet:>+9.2f}%")

    best_tol = max(tol_results.items(), key=lambda x: x[1]["avg_sharpe"])
    print(f"    → OPTIMAL TOLERANCE: {best_tol[0]}% (Sharpe={best_tol[1]['avg_sharpe']:+.4f})")
    all_results["tolerance_sweep"] = tol_results
    print()

    # ── 5. Maker vs Taker at optimal tolerance ──
    opt_tol = best_tol[0] / 100.0
    print(f"  MAKER vs TAKER (tol={best_tol[0]}%, bidirectional funding):")
    for label, fee_bps, slip_bps in [
        ("taker", 5.0, 3.0),
        ("maker", 2.0, 1.0),
    ]:
        yearly_sh = {}
        for yr in YEARS:
            r = run_vrp_with_hedge_v2(
                yr, delta_tolerance=opt_tol, hedge_fee_bps=fee_bps,
                hedge_slippage_bps=slip_bps, use_bidirectional_funding=True,
            )
            yearly_sh[yr] = r["metrics"]["sharpe"]
        avg_sh = sum(yearly_sh.values()) / len(yearly_sh)
        min_sh = min(yearly_sh.values())
        deg = (avg_sh - avg_bl) / abs(avg_bl) * 100
        print(f"    {label:6s}: Avg={avg_sh:+.4f} Min={min_sh:+.4f} (degradation: {deg:+.1f}%)")
        all_results[f"optimal_{label}"] = {
            "tolerance": best_tol[0],
            "avg_sharpe": round(avg_sh, 4),
            "min_sharpe": round(min_sh, 4),
            "degradation_pct": round(deg, 2),
            "yearly": dict(yearly_sh),
        }
    print()

    # ── 6. Best config: optimal tolerance + maker + bidirectional ──
    print(f"  BEST CONFIG: tol={best_tol[0]}% + maker fees + bidirectional funding:")
    best_yearly = {}
    best_stats = {}
    for yr in YEARS:
        r = run_vrp_with_hedge_v2(
            yr, delta_tolerance=opt_tol, hedge_fee_bps=2.0,
            hedge_slippage_bps=1.0, use_bidirectional_funding=True,
        )
        best_yearly[yr] = r["metrics"]["sharpe"]
        best_stats[yr] = r["hedge_stats"]
        hs = r["hedge_stats"]
        print(f"    {yr}: Sharpe={r['metrics']['sharpe']:+.4f} "
              f"hedges={hs['total_hedge_trades']} "
              f"hcost={hs['total_hedge_cost_pct']:.2f}% "
              f"fnet={hs['total_funding_net_pct']:+.2f}%")
    avg_best = sum(best_yearly.values()) / len(best_yearly)
    min_best = min(best_yearly.values())
    deg_best = (avg_best - avg_bl) / abs(avg_bl) * 100
    print(f"    Avg={avg_best:+.4f} Min={min_best:+.4f} (degradation: {deg_best:+.1f}% vs baseline)")
    all_results["best_config"] = {
        "tolerance": best_tol[0],
        "fee": "maker",
        "funding": "bidirectional",
        "avg_sharpe": round(avg_best, 4),
        "min_sharpe": round(min_best, 4),
        "degradation_pct": round(deg_best, 2),
        "yearly": dict(best_yearly),
        "yearly_stats": {str(k): v for k, v in best_stats.items()},
    }
    print()

    # ── 7. Summary table ──
    print("=" * 70)
    print("  SUMMARY: v1 vs v2 FUNDING MODEL")
    print("=" * 70)
    print(f"  {'Config':30s} {'Avg Sharpe':>12} {'Degradation':>12}")
    print("  " + "-" * 56)
    print(f"  {'BASELINE (perfect hedge)':30s} {avg_bl:>+12.4f} {'---':>12}")
    print(f"  {'v1 always-cost 10% funding':30s} {avg_v1:>+12.4f} "
          f"{(avg_v1-avg_bl)/abs(avg_bl)*100:>+11.1f}%")
    print(f"  {'v2 bidirectional per-year':30s} {avg_v2:>+12.4f} "
          f"{(avg_v2-avg_bl)/abs(avg_bl)*100:>+11.1f}%")
    print(f"  {'v2 optimal tol={0}% taker'.format(best_tol[0]):30s} "
          f"{all_results.get('optimal_taker', {}).get('avg_sharpe', 0):>+12.4f} "
          f"{all_results.get('optimal_taker', {}).get('degradation_pct', 0):>+11.1f}%")
    print(f"  {'v2 optimal tol={0}% maker'.format(best_tol[0]):30s} "
          f"{avg_best:>+12.4f} {deg_best:>+11.1f}%")
    print()

    # ── 8. Ensemble impact ──
    # Use best realistic config for ensemble calculation
    vrp_factor = avg_best / avg_bl if avg_bl > 0 else 1.0
    degraded_daily_vrp = 2.088 * vrp_factor
    # Mixed-freq ensemble: 40% hourly VRP + 60% daily Skew MR
    # Hourly VRP Sharpe 3.649 → degraded by same factor
    degraded_hourly_vrp = 3.649 * vrp_factor
    # Daily ensemble: 30% VRP + 70% Skew MR
    current_daily_ensemble = 0.30 * 2.088 + 0.70 * 1.744
    degraded_daily_ensemble = 0.30 * degraded_daily_vrp + 0.70 * 1.744
    # Mixed-freq ensemble: 40% hourly VRP + 60% daily Skew MR
    current_mixed_ensemble = 0.40 * 3.649 + 0.60 * 1.744
    degraded_mixed_ensemble = 0.40 * degraded_hourly_vrp + 0.60 * 1.744

    print("  ENSEMBLE IMPACT (best config):")
    print(f"    VRP degradation factor: {vrp_factor:.2%}")
    print(f"    VRP daily:  {2.088:.3f} → {degraded_daily_vrp:.3f}")
    print(f"    VRP hourly: {3.649:.3f} → {degraded_hourly_vrp:.3f}")
    print(f"    Daily ensemble (30/70):  {current_daily_ensemble:.3f} → {degraded_daily_ensemble:.3f}")
    print(f"    Mixed ensemble (40/60):  {current_mixed_ensemble:.3f} → {degraded_mixed_ensemble:.3f}")
    print()

    # Verdict
    if avg_best >= 1.5:
        verdict = "VRP SURVIVES delta hedging — Sharpe >= 1.5"
    elif avg_best >= 1.0:
        verdict = "VRP VIABLE with delta hedging — Sharpe 1.0-1.5"
    elif avg_best >= 0.5:
        verdict = "VRP MARGINAL — consider reducing ensemble weight"
    else:
        verdict = "VRP FAILS — hedging costs too high"
    print(f"  VERDICT: {verdict}")
    print(f"  Mixed ensemble with hedging: {degraded_mixed_ensemble:.3f} "
          f"({'PASS' if degraded_mixed_ensemble >= 1.0 else 'FAIL'} threshold >= 1.0)")
    print()

    # Save
    os.makedirs("artifacts/crypto_options", exist_ok=True)
    output = {
        "date": time.strftime("%Y-%m-%d %H:%M"),
        "version": "v2_corrected_funding",
        "baseline": {"avg_sharpe": round(avg_bl, 4), "yearly": baseline},
        "results": all_results,
        "best_config": {
            "tolerance_pct": best_tol[0],
            "fee_type": "maker",
            "funding_model": "bidirectional_per_year",
            "avg_sharpe": round(avg_best, 4),
            "min_sharpe": round(min_best, 4),
            "degradation_pct": round(deg_best, 2),
        },
        "ensemble_impact": {
            "vrp_degradation_factor": round(vrp_factor, 4),
            "daily_ensemble_before": round(current_daily_ensemble, 4),
            "daily_ensemble_after": round(degraded_daily_ensemble, 4),
            "mixed_ensemble_before": round(current_mixed_ensemble, 4),
            "mixed_ensemble_after": round(degraded_mixed_ensemble, 4),
        },
        "verdict": verdict,
        "elapsed_sec": round(time.time() - t0, 1),
    }
    out_path = "artifacts/crypto_options/delta_hedge_v2.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"  Saved to {out_path}")
    print(f"  Total elapsed: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
