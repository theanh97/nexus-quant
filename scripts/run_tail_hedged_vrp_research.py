#!/usr/bin/env python3
"""
Tail-Hedged VRP Research
=========================

Tests whether adding OTM put protection to the VRP short straddle improves
risk-adjusted returns. This is NOT an overlay (not timing in/out) — it's a
different trade structure where VRP is always-on but with built-in tail protection.

Trade structure comparison:
    1. Naked VRP:    Sell ATM straddle (baseline)
    2. Tail-hedged:  Sell ATM straddle + Buy OTM put (10d, 15d, 25d)
    3. Iron condor:  Sell ATM straddle + Buy OTM put + Buy OTM call

Economics:
    - Naked VRP earns theta from the full straddle
    - Tail-hedged VRP pays a COST for the put protection:
        Put cost ≈ BS_price(OTM_put) per bar
        Reduces theta income but caps downside
    - Iron condor adds call protection too (caps upside risk)

PnL model:
    Naked:      PnL = 0.5 * (IV² - RV²) * dt * leverage
    Tail-hedge: PnL = 0.5 * (IV² - RV²) * dt * leverage - put_premium * dt
                + put_payoff (when spot < strike)

    Put premium (annualized) for delta-d put:
        BS_put_price(K, T=30d, sigma=IV) / S ≈ IV * N(-d1) * T^0.5
        Rough: 10d put costs ~1% of notional per 30d → ~12% annual drag
               25d put costs ~4% per 30d → ~48% annual drag

    Put payoff: max(K - S_final, 0) on crash days

Key metrics:
    - Sharpe ratio (may decrease slightly)
    - Maximum drawdown (should improve significantly)
    - Calmar ratio (should improve — better MDD-adjusted return)
    - VaR/CVaR (should improve)
    - Worst single day (should be bounded)
    - Tail ratio (gain in worst days vs loss in best days)

Configs:
    - Put delta: 5d, 10d, 15d, 25d (distance from ATM)
    - Hedge ratio: 25%, 50%, 75%, 100% (fraction of position hedged)
    - Put cost model: realistic Black-Scholes pricing
"""
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from nexus_quant.projects.crypto_options.options_engine import compute_metrics
from nexus_quant.projects.crypto_options.providers.deribit_rest import DeribitRestProvider


BARS_PER_YEAR = 365
YEARS = [2021, 2022, 2023, 2024, 2025]
BASE_LEVERAGE = 1.5
DIVIDER = "=" * 70


def ts_to_year(ts: int) -> int:
    return datetime.fromtimestamp(ts, tz=timezone.utc).year


# ── Black-Scholes Put Pricing ────────────────────────────────────────────────

def norm_cdf(x: float) -> float:
    """Standard normal CDF (Abramowitz & Stegun approximation)."""
    if x >= 0:
        t = 1.0 / (1.0 + 0.2316419 * x)
        poly = t * (0.319381530 + t * (-0.356563782 + t * (1.781477937
               + t * (-1.821255978 + t * 1.330274429))))
        return 1.0 - poly * math.exp(-0.5 * x * x) / math.sqrt(2 * math.pi)
    else:
        return 1.0 - norm_cdf(-x)


def bs_put_price(S: float, K: float, T: float, sigma: float, r: float = 0.0) -> float:
    """Black-Scholes put price."""
    if T < 1e-10 or sigma < 1e-10 or S < 1e-10:
        return max(K - S, 0.0)
    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return K * math.exp(-r * T) * norm_cdf(-d2) - S * norm_cdf(-d1)


def strike_from_delta(S: float, T: float, sigma: float, delta: float, r: float = 0.0) -> float:
    """Find put strike for a given put delta (negative convention, e.g. -0.10)."""
    # Put delta = N(d1) - 1  →  d1 = N^{-1}(1 + delta)
    # Using Newton's method
    from math import log, sqrt, exp
    target = abs(delta)  # e.g. 0.10 for 10-delta put

    # Initial guess using normal approximation
    K = S * exp(-sigma * sqrt(T) * norm_inv(target))

    for _ in range(20):
        d1 = (log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * sqrt(T))
        put_delta = norm_cdf(d1) - 1.0
        current = abs(put_delta)

        if abs(current - target) < 1e-6:
            break

        # Derivative of delta w.r.t. K
        # dDelta/dK = -N'(d1) * d(d1)/dK = N'(d1) / (K * sigma * sqrt(T))
        n_d1 = exp(-0.5 * d1 ** 2) / sqrt(2 * math.pi)
        dd_dk = n_d1 / (K * sigma * sqrt(T))
        K -= (current - target) / dd_dk

        K = max(K, S * 0.3)
        K = min(K, S * 1.0)

    return K


def norm_inv(p: float) -> float:
    """Inverse normal CDF (Beasley-Springer-Moro approximation)."""
    if p <= 0:
        return -8.0
    if p >= 1:
        return 8.0
    if p == 0.5:
        return 0.0

    # Rational approximation
    a = [-3.969683028665376e+01, 2.209460984245205e+02,
         -2.759285104469687e+02, 1.383577518672690e+02,
         -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02,
         -1.556989798598866e+02, 6.680131188771972e+01,
         -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01,
         -2.400758277161838e+00, -2.549732539343734e+00,
         4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01,
         2.445134137142996e+00, 3.754408661907416e+00]

    p_low = 0.02425
    p_high = 1 - p_low

    if p < p_low:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
               ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    elif p <= p_high:
        q = p - 0.5
        r = q * q
        return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q / \
               (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1)
    else:
        q = math.sqrt(-2 * math.log(1 - p))
        return -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
                ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)


# ── Tail-Hedged VRP Engine ───────────────────────────────────────────────────

def run_tail_hedged_vrp(
    dataset,
    put_delta: float = 0.10,
    hedge_ratio: float = 1.0,
    dte_days: float = 30.0,
) -> Tuple[List[float], List[float], Dict[str, Any]]:
    """
    Run VRP with OTM put protection.

    Args:
        dataset: MarketDataset with iv_atm, rv_realized, perp_close
        put_delta: put delta (e.g. 0.10 = 10-delta put)
        hedge_ratio: fraction of position hedged (1.0 = fully hedged)
        dte_days: option DTE for put pricing

    Returns:
        (equity_curve, returns, breakdown)
    """
    syms = dataset.symbols
    n = len(dataset.timeline)
    dt = 1.0 / BARS_PER_YEAR
    T = dte_days / 365.0

    per_sym = BASE_LEVERAGE / max(len(syms), 1)
    min_bars = 30

    equity = 1.0
    equity_curve = [1.0]
    returns_list: List[float] = []

    vrp_pnl_total = 0.0
    hedge_cost_total = 0.0
    hedge_payoff_total = 0.0

    for idx in range(1, n):
        prev_equity = equity
        bar_pnl = 0.0

        for sym in syms:
            if idx < min_bars:
                continue

            # IV for this bar
            iv_series = dataset.features.get("iv_atm", {}).get(sym, [])
            if idx >= len(iv_series) or iv_series[idx - 1] is None:
                continue
            iv = float(iv_series[idx - 1])

            # Realized vol for single bar
            closes = dataset.perp_close.get(sym, [])
            if idx >= len(closes) or closes[idx - 1] <= 0 or closes[idx] <= 0:
                continue
            log_ret = math.log(closes[idx] / closes[idx - 1])
            rv_bar = abs(log_ret) * math.sqrt(BARS_PER_YEAR)

            S = closes[idx - 1]

            # ── 1. Naked VRP PnL ──
            vrp_base = 0.5 * (iv ** 2 - rv_bar ** 2) * dt
            vrp_pnl = per_sym * vrp_base
            bar_pnl += vrp_pnl
            vrp_pnl_total += vrp_pnl * equity

            # ── 2. Put hedge cost (daily theta of OTM put) ──
            K = strike_from_delta(S, T, iv, put_delta)
            put_price = bs_put_price(S, K, T, iv)

            # Daily cost of the put = put_price / DTE (linear decay approx)
            # As fraction of spot: put_pct / DTE
            put_pct = put_price / S
            daily_hedge_cost = put_pct / dte_days

            # Apply hedge ratio and leverage
            hedge_cost = daily_hedge_cost * per_sym * hedge_ratio
            bar_pnl -= hedge_cost
            hedge_cost_total += hedge_cost * equity

            # ── 3. Put payoff on crash days ──
            # If spot drops below strike, put pays off
            S_new = closes[idx]
            if S_new < K:
                # Put payoff = (K - S_new) / S, as fraction of starting spot
                payoff_pct = (K - S_new) / S
                payoff = payoff_pct * per_sym * hedge_ratio
                bar_pnl += payoff
                hedge_payoff_total += payoff * equity

        dp = equity * bar_pnl
        equity += dp
        equity = max(equity, 0.0)

        equity_curve.append(equity)
        bar_ret = (equity / prev_equity) - 1.0 if prev_equity > 0 else 0.0
        returns_list.append(bar_ret)

    breakdown = {
        "vrp_pnl": round(vrp_pnl_total, 6),
        "hedge_cost": round(hedge_cost_total, 6),
        "hedge_payoff": round(hedge_payoff_total, 6),
        "net_hedge_cost": round(hedge_cost_total - hedge_payoff_total, 6),
        "hedge_efficiency": round(
            hedge_payoff_total / hedge_cost_total * 100, 1
        ) if hedge_cost_total > 0 else 0.0,
    }

    return equity_curve, returns_list, breakdown


def run_iron_condor_vrp(
    dataset,
    put_delta: float = 0.10,
    call_delta: float = 0.10,
    hedge_ratio: float = 1.0,
    dte_days: float = 30.0,
) -> Tuple[List[float], List[float], Dict[str, Any]]:
    """
    Run VRP with both OTM put AND OTM call protection (iron condor wrapper).
    """
    syms = dataset.symbols
    n = len(dataset.timeline)
    dt = 1.0 / BARS_PER_YEAR
    T = dte_days / 365.0

    per_sym = BASE_LEVERAGE / max(len(syms), 1)
    min_bars = 30

    equity = 1.0
    equity_curve = [1.0]
    returns_list: List[float] = []

    vrp_pnl_total = 0.0
    hedge_cost_total = 0.0
    hedge_payoff_total = 0.0

    for idx in range(1, n):
        prev_equity = equity
        bar_pnl = 0.0

        for sym in syms:
            if idx < min_bars:
                continue

            iv_series = dataset.features.get("iv_atm", {}).get(sym, [])
            if idx >= len(iv_series) or iv_series[idx - 1] is None:
                continue
            iv = float(iv_series[idx - 1])

            closes = dataset.perp_close.get(sym, [])
            if idx >= len(closes) or closes[idx - 1] <= 0 or closes[idx] <= 0:
                continue
            log_ret = math.log(closes[idx] / closes[idx - 1])
            rv_bar = abs(log_ret) * math.sqrt(BARS_PER_YEAR)

            S = closes[idx - 1]
            S_new = closes[idx]

            # VRP PnL
            vrp_base = 0.5 * (iv ** 2 - rv_bar ** 2) * dt
            vrp_pnl = per_sym * vrp_base
            bar_pnl += vrp_pnl
            vrp_pnl_total += vrp_pnl * equity

            # Put hedge
            K_put = strike_from_delta(S, T, iv, put_delta)
            put_price = bs_put_price(S, K_put, T, iv)
            put_cost = (put_price / S / dte_days) * per_sym * hedge_ratio
            bar_pnl -= put_cost
            hedge_cost_total += put_cost * equity

            if S_new < K_put:
                payoff = ((K_put - S_new) / S) * per_sym * hedge_ratio
                bar_pnl += payoff
                hedge_payoff_total += payoff * equity

            # Call hedge (symmetric)
            # Call at (1 + delta_distance) of spot
            K_call = S * (2 - K_put / S)  # symmetric around ATM
            # Call price using put-call parity approximation
            call_price = bs_put_price(S, K_call, T, iv) + S - K_call  # approximate
            call_price = max(call_price, 0.001 * S)
            call_cost = (call_price / S / dte_days) * per_sym * hedge_ratio
            bar_pnl -= call_cost
            hedge_cost_total += call_cost * equity

            if S_new > K_call:
                payoff = ((S_new - K_call) / S) * per_sym * hedge_ratio
                bar_pnl += payoff
                hedge_payoff_total += payoff * equity

        dp = equity * bar_pnl
        equity += dp
        equity = max(equity, 0.0)

        equity_curve.append(equity)
        bar_ret = (equity / prev_equity) - 1.0 if prev_equity > 0 else 0.0
        returns_list.append(bar_ret)

    breakdown = {
        "vrp_pnl": round(vrp_pnl_total, 6),
        "hedge_cost": round(hedge_cost_total, 6),
        "hedge_payoff": round(hedge_payoff_total, 6),
        "net_hedge_cost": round(hedge_cost_total - hedge_payoff_total, 6),
    }

    return equity_curve, returns_list, breakdown


def yearly_analysis(
    dataset, runner_fn, **kwargs
) -> Dict[str, Any]:
    """Run full and per-year analysis."""
    full_ec, full_ret, bd = runner_fn(dataset, **kwargs)
    full_m = compute_metrics(full_ec, full_ret, BARS_PER_YEAR)

    yearly_sharpes = {}
    yearly_mdds = {}
    for yr in YEARS:
        yr_indices = [i for i, ts in enumerate(dataset.timeline) if ts_to_year(ts) == yr]
        if len(yr_indices) < 10:
            yearly_sharpes[yr] = 0.0
            yearly_mdds[yr] = 0.0
            continue
        start, end = yr_indices[0], yr_indices[-1]
        yr_rets = full_ret[max(0, start - 1):end]
        if not yr_rets:
            yearly_sharpes[yr] = 0.0
            yearly_mdds[yr] = 0.0
            continue
        yr_ec = [1.0]
        for r in yr_rets:
            yr_ec.append(yr_ec[-1] * (1 + r))
        yr_m = compute_metrics(yr_ec, yr_rets, BARS_PER_YEAR)
        yearly_sharpes[yr] = yr_m["sharpe"]
        yearly_mdds[yr] = yr_m["max_drawdown"]

    avg_sh = sum(yearly_sharpes.values()) / len(yearly_sharpes) if yearly_sharpes else 0.0

    return {
        "full_sharpe": full_m["sharpe"],
        "full_mdd": full_m["max_drawdown"],
        "full_sortino": full_m.get("sortino", 0),
        "full_calmar": full_m.get("calmar", 0),
        "avg_sharpe": round(avg_sh, 3),
        "yearly_sharpes": yearly_sharpes,
        "yearly_mdds": yearly_mdds,
        "breakdown": bd,
    }


def tail_metrics(rets: List[float]) -> Dict[str, float]:
    """Compute tail risk metrics."""
    if not rets:
        return {}
    sorted_r = sorted(rets)
    n = len(sorted_r)
    var_95 = sorted_r[int(0.05 * n)] if n > 20 else 0
    var_99 = sorted_r[int(0.01 * n)] if n > 100 else 0
    cvar_95 = sum(sorted_r[:int(0.05 * n)]) / max(int(0.05 * n), 1) if n > 20 else 0
    worst = min(rets)
    loss_1pct = sum(1 for r in rets if r < -0.01) / n * 100
    return {
        "var_95": round(var_95 * 100, 3),
        "var_99": round(var_99 * 100, 3),
        "cvar_95": round(cvar_95 * 100, 3),
        "worst_day": round(worst * 100, 3),
        "loss_days_gt1pct": round(loss_1pct, 1),
    }


def main():
    print(DIVIDER)
    print("TAIL-HEDGED VRP RESEARCH")
    print(DIVIDER)

    print("\nLoading data...")
    cfg = {
        "provider": "deribit_rest_v1",
        "symbols": ["BTC"],
        "start": "2021-01-01",
        "end": "2025-12-31",
        "bar_interval": "1d",
        "use_synthetic_iv": True,
    }
    provider = DeribitRestProvider(cfg, seed=42)
    dataset = provider.load()
    print(f"Dataset: {len(dataset.timeline)} bars, symbols: {dataset.symbols}")

    # ── 1. Naked VRP Baseline ─────────────────────────────────────────────
    print(f"\n{DIVIDER}")
    print("1. NAKED VRP BASELINE (no hedge)")
    print(DIVIDER)

    baseline = yearly_analysis(dataset, run_tail_hedged_vrp,
                               put_delta=0.10, hedge_ratio=0.0)
    base_ec, base_ret, _ = run_tail_hedged_vrp(dataset, put_delta=0.10, hedge_ratio=0.0)
    base_tail = tail_metrics(base_ret)

    yearly_str = " ".join(f"{yr}:{s:.2f}" for yr, s in sorted(baseline["yearly_sharpes"].items()))
    print(f"  Sharpe={baseline['full_sharpe']:.3f}  Avg={baseline['avg_sharpe']:.3f}")
    print(f"  MDD={baseline['full_mdd']*100:.1f}%  Calmar={baseline['full_calmar']:.3f}  Sortino={baseline['full_sortino']:.3f}")
    print(f"  VaR95={base_tail['var_95']:.2f}%  VaR99={base_tail['var_99']:.2f}%  CVaR95={base_tail['cvar_95']:.2f}%")
    print(f"  Worst day={base_tail['worst_day']:.2f}%  Loss days>1%={base_tail['loss_days_gt1pct']:.1f}%")
    print(f"  Per-year: {yearly_str}")

    # ── 2. Put Delta Sweep ────────────────────────────────────────────────
    print(f"\n{DIVIDER}")
    print("2. PUT DELTA SWEEP (full hedge, vary OTM distance)")
    print(DIVIDER)
    print("  Deeper OTM = cheaper but less protection\n")

    deltas = [0.05, 0.10, 0.15, 0.20, 0.25]
    delta_results = {}

    print(f"  {'Config':20s} {'Sharpe':>7s} {'Avg':>6s} {'MDD':>7s} {'Calmar':>7s} {'VaR99':>7s} {'Worst':>7s} | Per-year")
    print(f"  {'-'*20} {'-'*7} {'-'*6} {'-'*7} {'-'*7} {'-'*7} {'-'*7} | {'-'*40}")

    for d in deltas:
        r = yearly_analysis(dataset, run_tail_hedged_vrp,
                           put_delta=d, hedge_ratio=1.0)
        ec, ret, _ = run_tail_hedged_vrp(dataset, put_delta=d, hedge_ratio=1.0)
        tm = tail_metrics(ret)
        delta_results[d] = {"result": r, "tail": tm}

        yr_str = " ".join(f"{yr}:{s:.2f}" for yr, s in sorted(r["yearly_sharpes"].items()))
        print(f"  put_{int(d*100):2d}d_100%      "
              f"{r['full_sharpe']:+7.3f} "
              f"{r['avg_sharpe']:+6.3f} "
              f"{r['full_mdd']*100:6.1f}% "
              f"{r['full_calmar']:7.3f} "
              f"{tm['var_99']:6.2f}% "
              f"{tm['worst_day']:6.2f}% | {yr_str}")

    # ── 3. Hedge Ratio Sweep ──────────────────────────────────────────────
    print(f"\n{DIVIDER}")
    print("3. HEDGE RATIO SWEEP (10-delta put, vary % hedged)")
    print(DIVIDER)
    print("  Lower ratio = cheaper but less protection\n")

    ratios = [0.0, 0.25, 0.50, 0.75, 1.0]
    ratio_results = {}

    print(f"  {'Config':20s} {'Sharpe':>7s} {'Avg':>6s} {'MDD':>7s} {'Calmar':>7s} {'VaR99':>7s} {'Worst':>7s} | Per-year")
    print(f"  {'-'*20} {'-'*7} {'-'*6} {'-'*7} {'-'*7} {'-'*7} {'-'*7} | {'-'*40}")

    for hr in ratios:
        r = yearly_analysis(dataset, run_tail_hedged_vrp,
                           put_delta=0.10, hedge_ratio=hr)
        ec, ret, _ = run_tail_hedged_vrp(dataset, put_delta=0.10, hedge_ratio=hr)
        tm = tail_metrics(ret)
        ratio_results[hr] = {"result": r, "tail": tm}

        yr_str = " ".join(f"{yr}:{s:.2f}" for yr, s in sorted(r["yearly_sharpes"].items()))
        print(f"  10d_hedge_{int(hr*100):3d}%     "
              f"{r['full_sharpe']:+7.3f} "
              f"{r['avg_sharpe']:+6.3f} "
              f"{r['full_mdd']*100:6.1f}% "
              f"{r['full_calmar']:7.3f} "
              f"{tm['var_99']:6.2f}% "
              f"{tm['worst_day']:6.2f}% | {yr_str}")

    # ── 4. Iron Condor Comparison ─────────────────────────────────────────
    print(f"\n{DIVIDER}")
    print("4. IRON CONDOR (both put + call protection)")
    print(DIVIDER)

    ic_configs = [
        {"put_delta": 0.10, "call_delta": 0.10, "hedge_ratio": 0.50},
        {"put_delta": 0.10, "call_delta": 0.10, "hedge_ratio": 1.0},
        {"put_delta": 0.15, "call_delta": 0.15, "hedge_ratio": 0.50},
    ]

    print(f"  {'Config':20s} {'Sharpe':>7s} {'Avg':>6s} {'MDD':>7s} {'Calmar':>7s} {'VaR99':>7s} | Per-year")
    print(f"  {'-'*20} {'-'*7} {'-'*6} {'-'*7} {'-'*7} {'-'*7} | {'-'*40}")

    for cfg_ic in ic_configs:
        r = yearly_analysis(dataset, run_iron_condor_vrp, **cfg_ic)
        ec, ret, _ = run_iron_condor_vrp(dataset, **cfg_ic)
        tm = tail_metrics(ret)
        d, cr = int(cfg_ic["put_delta"]*100), int(cfg_ic["hedge_ratio"]*100)
        name = f"IC_{d}d_{cr}%"
        yr_str = " ".join(f"{yr}:{s:.2f}" for yr, s in sorted(r["yearly_sharpes"].items()))
        print(f"  {name:20s} "
              f"{r['full_sharpe']:+7.3f} "
              f"{r['avg_sharpe']:+6.3f} "
              f"{r['full_mdd']*100:6.1f}% "
              f"{r['full_calmar']:7.3f} "
              f"{tm['var_99']:6.2f}% | {yr_str}")

    # ── 5. Cost-Benefit Analysis ──────────────────────────────────────────
    print(f"\n{DIVIDER}")
    print("5. COST-BENEFIT ANALYSIS")
    print(DIVIDER)
    print("  How much Sharpe is sacrificed per unit of tail risk improvement?\n")

    best_hedge = None
    best_calmar_improvement = 0

    for d in deltas:
        for hr in [0.25, 0.50, 0.75, 1.0]:
            r = yearly_analysis(dataset, run_tail_hedged_vrp,
                               put_delta=d, hedge_ratio=hr)
            ec, ret, bd = run_tail_hedged_vrp(dataset, put_delta=d, hedge_ratio=hr)
            tm = tail_metrics(ret)

            sharpe_cost = baseline["avg_sharpe"] - r["avg_sharpe"]
            mdd_improvement = abs(baseline["full_mdd"]) - abs(r["full_mdd"])
            calmar_change = r["full_calmar"] - baseline["full_calmar"]
            var99_improvement = abs(base_tail["var_99"]) - abs(tm["var_99"])

            if calmar_change > best_calmar_improvement:
                best_calmar_improvement = calmar_change
                best_hedge = {
                    "delta": d,
                    "ratio": hr,
                    "sharpe_cost": sharpe_cost,
                    "calmar_change": calmar_change,
                    "mdd_improvement_pct": mdd_improvement * 100,
                    "var99_improvement": var99_improvement,
                    "result": r,
                    "tail": tm,
                    "breakdown": bd,
                }

    if best_hedge:
        print(f"  Best Calmar improvement:")
        print(f"    Config: put_{int(best_hedge['delta']*100)}d, hedge={int(best_hedge['ratio']*100)}%")
        print(f"    Sharpe cost: {best_hedge['sharpe_cost']:+.3f}")
        print(f"    Calmar change: {best_hedge['calmar_change']:+.3f}")
        print(f"    MDD improvement: {best_hedge['mdd_improvement_pct']:+.1f}pp")
        print(f"    VaR99 improvement: {best_hedge['var99_improvement']:+.2f}pp")
        print(f"    Hedge cost total: {best_hedge['breakdown']['hedge_cost']:.4f}")
        print(f"    Hedge payoff total: {best_hedge['breakdown']['hedge_payoff']:.4f}")
        print(f"    Efficiency: {best_hedge['breakdown'].get('hedge_efficiency', 0):.1f}%")

    # ── Summary ───────────────────────────────────────────────────────────
    print(f"\n{DIVIDER}")
    print("SUMMARY")
    print(DIVIDER)

    print(f"\n  Baseline naked VRP: Sharpe={baseline['avg_sharpe']:.3f} MDD={baseline['full_mdd']*100:.1f}% Calmar={baseline['full_calmar']:.3f}")
    print(f"  Worst day (baseline): {base_tail['worst_day']:.2f}%")

    if best_hedge:
        bh = best_hedge
        r = bh["result"]
        print(f"\n  Best tail-hedged VRP:")
        print(f"    Config: put_{int(bh['delta']*100)}d, hedge={int(bh['ratio']*100)}%")
        print(f"    Sharpe={r['avg_sharpe']:.3f} MDD={r['full_mdd']*100:.1f}% Calmar={r['full_calmar']:.3f}")
        print(f"    Worst day: {bh['tail']['worst_day']:.2f}%")

        sharpe_delta = r["avg_sharpe"] - baseline["avg_sharpe"]
        calmar_delta = r["full_calmar"] - baseline["full_calmar"]

        if sharpe_delta > -0.10 and calmar_delta > 0.5:
            print(f"\n  FINDING: Tail hedge IMPROVES risk-adjusted returns!")
            print(f"    Sharpe cost: {sharpe_delta:+.3f} (acceptable)")
            print(f"    Calmar improvement: {calmar_delta:+.3f} (significant)")
        elif sharpe_delta > -0.30 and calmar_delta > 0:
            print(f"\n  FINDING: Tail hedge provides MODERATE improvement.")
            print(f"    Sharpe cost: {sharpe_delta:+.3f}")
            print(f"    Calmar improvement: {calmar_delta:+.3f}")
        else:
            print(f"\n  FINDING: Tail hedge COSTS too much for the protection.")
            print(f"    Sharpe cost: {sharpe_delta:+.3f} (too large)")
            print(f"    Calmar improvement: {calmar_delta:+.3f} (insufficient)")

    print(f"\n  NOTE: Synthetic data may understate crash severity.")
    print(f"  Real-world tail events could make hedging more valuable.")
    print(DIVIDER)

    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
