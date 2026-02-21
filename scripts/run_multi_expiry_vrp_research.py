#!/usr/bin/env python3
"""
Multi-Expiry VRP Research
==========================

Tests whether diversifying VRP positions across multiple expiry tenors
improves risk-adjusted returns vs. the standard single 30-day tenor.

Key hypothesis:
    Different DTEs have different gamma/theta profiles:
    - Short DTE (7d): Higher theta/gamma per day, more volatile, frequent rolling
    - Medium DTE (30d): Standard baseline
    - Long DTE (60d): Smoother PnL, less rolling, potentially lower VRP

    Splitting across tenors may:
    1. Reduce rollover gamma spikes (don't roll all at once)
    2. Capture different parts of the term structure
    3. Smooth the PnL stream if front/back IV moves aren't perfectly correlated
    4. Reduce worst-case single-bar loss (gamma concentrated at one DTE)

Tenor-specific modeling:
    - Per-tenor IV adjustment from term structure:
        IV_7d  = IV_atm + 0.50 × term_spread  (front premium)
        IV_14d = IV_atm + 0.25 × term_spread
        IV_30d = IV_atm                        (baseline)
        IV_60d = IV_atm - 0.50 × term_spread  (back discount)
    - Gamma/theta scaling: PnL × sqrt(30/DTE) for ATM gamma effect
    - Rolling: each tenor rolls every max(DTE//2, 1) bars
    - Roll costs: full turnover cost on each roll

Configs tested:
    1. Single-tenor: 7d, 14d, 30d, 60d individually
    2. Equal-weight ladder: 25% each across 4 tenors
    3. Front-heavy: 40% 7d, 30% 14d, 20% 30d, 10% 60d
    4. Back-heavy: 10% 7d, 20% 14d, 30% 30d, 40% 60d
    5. 2-tenor: 50/50 splits (7d+60d, 14d+60d, 7d+30d)
    6. Optimized: grid search over tenor weights
"""
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from nexus_quant.backtest.costs import ExecutionCostModel, FeeModel, ImpactModel
from nexus_quant.data.schema import MarketDataset
from nexus_quant.projects.crypto_options.options_engine import compute_metrics
from nexus_quant.projects.crypto_options.providers.deribit_rest import DeribitRestProvider


# ── Constants ─────────────────────────────────────────────────────────────────
BARS_PER_YEAR = 365
YEARS = [2021, 2022, 2023, 2024, 2025]
TENORS = [7, 14, 30, 60]  # DTE in days

# Term structure IV adjustment per tenor (relative to 30d ATM)
# Positive = higher IV than 30d, negative = lower
TENOR_IV_OFFSET_FACTOR = {
    7: 0.50,     # front-month premium
    14: 0.25,
    30: 0.00,    # baseline
    60: -0.50,   # back-month discount
}

# Gamma/theta scaling: ATM gamma ∝ 1/sqrt(T)
# Normalized to 30d baseline: scale = sqrt(30/DTE)
TENOR_GAMMA_SCALE = {dte: math.sqrt(30.0 / dte) for dte in TENORS}
# 7d: 2.07x, 14d: 1.46x, 30d: 1.0x, 60d: 0.71x

# Base leverage (same as production VRP)
BASE_LEVERAGE = 1.5

DIVIDER = "=" * 70


def ts_to_year(ts: int) -> int:
    return datetime.fromtimestamp(ts, tz=timezone.utc).year


def build_costs() -> ExecutionCostModel:
    """Deribit-realistic cost model."""
    fees = FeeModel(maker_fee_rate=0.0003, taker_fee_rate=0.0005)
    impact = ImpactModel(model="sqrt", coef_bps=2.0)
    return ExecutionCostModel(fee=fees, impact=impact)


def load_full_dataset() -> MarketDataset:
    """Load full 5-year synthetic dataset."""
    cfg = {
        "provider": "deribit_rest_v1",
        "symbols": ["BTC"],
        "start": "2021-01-01",
        "end": "2025-12-31",
        "bar_interval": "1d",
        "use_synthetic_iv": True,
    }
    provider = DeribitRestProvider(cfg, seed=42)
    return provider.load()


def run_multi_expiry_vrp(
    dataset: MarketDataset,
    tenor_weights: Dict[int, float],
    roll_cost_mult: float = 1.0,
) -> Tuple[List[float], List[float], Dict[str, Any]]:
    """
    Run multi-expiry VRP backtest.

    For each tenor, computes VRP PnL with tenor-specific adjustments:
      - IV adjusted by term structure offset
      - Gamma/theta scaled by sqrt(30/DTE)
      - Rolling costs at each roll frequency

    Args:
        dataset: MarketDataset with iv_atm, rv_realized, term_spread features
        tenor_weights: {DTE: weight_fraction}, must sum to 1.0
        roll_cost_mult: multiplier on roll transaction cost

    Returns:
        (equity_curve, returns_list, breakdown_dict)
    """
    syms = dataset.symbols
    n = len(dataset.timeline)
    dt = 1.0 / BARS_PER_YEAR
    costs = build_costs()

    equity = 1.0
    equity_curve = [1.0]
    returns_list: List[float] = []

    # Per-tenor state
    tenor_pnl = {dte: 0.0 for dte in tenor_weights}
    roll_cost_total = 0.0
    rebal_cost_total = 0.0

    # VRP weight per symbol (always short vol)
    per_sym = BASE_LEVERAGE / max(len(syms), 1)

    # Roll tracking: each tenor rolls at expiry (DTE bars)
    # In practice, roll ~5 days before expiry for liquidity, but model as DTE
    bars_since_roll = {dte: 0 for dte in tenor_weights}

    # Weekly rebalance (same as production VRP, every 5 bars)
    rebalance_freq = 5
    min_bars = 30

    for idx in range(1, n):
        prev_equity = equity
        bar_pnl = 0.0

        for sym in syms:
            if idx < min_bars:
                continue

            # Get IV and RV for this bar
            iv_series = dataset.features.get("iv_atm", {}).get(sym, [])
            rv_series = dataset.features.get("rv_realized", {}).get(sym, [])
            ts_series = dataset.features.get("term_spread", {}).get(sym, [])

            if idx >= len(iv_series) or iv_series[idx - 1] is None:
                continue

            iv_base = float(iv_series[idx - 1])  # 30d ATM IV

            # Realized vol for single bar (annualized)
            closes = dataset.perp_close.get(sym, [])
            if idx < len(closes) and closes[idx - 1] > 0 and closes[idx] > 0:
                log_ret = math.log(closes[idx] / closes[idx - 1])
                rv_bar = abs(log_ret) * math.sqrt(BARS_PER_YEAR)
            else:
                rv_bar = 0.0

            # Term spread for IV adjustment
            term_val = 0.0
            if ts_series and idx - 1 < len(ts_series) and ts_series[idx - 1] is not None:
                term_val = float(ts_series[idx - 1])

            # Compute PnL for each tenor
            for dte, tw in tenor_weights.items():
                if tw < 1e-10:
                    continue

                # Tenor-specific IV adjustment
                iv_offset = TENOR_IV_OFFSET_FACTOR.get(dte, 0.0) * term_val
                iv_tenor = max(0.05, iv_base + iv_offset)

                # Gamma/theta scaling (ATM gamma ∝ 1/sqrt(T))
                gamma_scale = TENOR_GAMMA_SCALE.get(dte, 1.0)

                # VRP PnL for this tenor:
                # Base: 0.5 × (IV² - RV²) × dt
                # Scaled by gamma factor and tenor weight
                vrp_base = 0.5 * (iv_tenor ** 2 - rv_bar ** 2) * dt
                vrp_pnl = vrp_base * gamma_scale

                # Weight: short vol = negative, so PnL = (-(-per_sym)) × vrp
                bar_tenor_pnl = per_sym * tw * vrp_pnl
                bar_pnl += bar_tenor_pnl
                tenor_pnl[dte] += bar_tenor_pnl * equity

                # Rolling cost: at expiry (every DTE bars), pay close+open turnover
                # This is the OPTIONS-SPECIFIC cost beyond regular rebalancing
                bars_since_roll[dte] += 1
                if bars_since_roll[dte] >= dte:
                    # Roll cost = one-way turnover for the position at this tenor
                    # (in practice, calendar spread is cheaper, so this is conservative)
                    roll_turnover = per_sym * tw  # one-way spread cost
                    rc = costs.cost(equity=equity, turnover=roll_turnover)
                    roll_c = float(rc.get("cost", 0.0)) * roll_cost_mult
                    bar_pnl -= roll_c / equity if equity > 0 else 0
                    roll_cost_total += roll_c
                    bars_since_roll[dte] = 0

        dp = equity * bar_pnl
        equity += dp
        equity = max(equity, 0.0)

        equity_curve.append(equity)
        bar_ret = (equity / prev_equity) - 1.0 if prev_equity > 0 else 0.0
        returns_list.append(bar_ret)

    breakdown = {
        "tenor_pnl": {str(dte): round(v, 6) for dte, v in tenor_pnl.items()},
        "roll_cost": round(roll_cost_total, 6),
        "rebal_cost": round(rebal_cost_total, 6),
        "gamma_scales": {str(dte): round(TENOR_GAMMA_SCALE[dte], 3) for dte in tenor_weights},
    }

    return equity_curve, returns_list, breakdown


def yearly_metrics(
    dataset: MarketDataset,
    tenor_weights: Dict[int, float],
    roll_cost_mult: float = 1.0,
) -> Dict[str, Any]:
    """Run per-year analysis and return summary metrics."""
    full_ec, full_ret, full_bd = run_multi_expiry_vrp(dataset, tenor_weights, roll_cost_mult)
    full_m = compute_metrics(full_ec, full_ret, BARS_PER_YEAR)

    # Per-year breakdown
    yearly_sharpes = {}
    yearly_mdds = {}
    for yr in YEARS:
        yr_indices = [
            i for i, ts in enumerate(dataset.timeline)
            if ts_to_year(ts) == yr
        ]
        if len(yr_indices) < 10:
            yearly_sharpes[yr] = 0.0
            yearly_mdds[yr] = 0.0
            continue

        start, end = yr_indices[0], yr_indices[-1]
        # Extract returns for this year (offset by 1 since returns start at idx=1)
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

    avg_sharpe = sum(yearly_sharpes.values()) / len(yearly_sharpes) if yearly_sharpes else 0.0
    min_sharpe = min(yearly_sharpes.values()) if yearly_sharpes else 0.0

    return {
        "full_sharpe": full_m["sharpe"],
        "full_mdd": full_m["max_drawdown"],
        "avg_sharpe": round(avg_sharpe, 3),
        "min_sharpe": round(min_sharpe, 3),
        "yearly_sharpes": yearly_sharpes,
        "yearly_mdds": yearly_mdds,
        "breakdown": full_bd,
    }


def print_result(name: str, result: Dict, baseline_avg: float = 0.0):
    """Print single result line."""
    delta = result["avg_sharpe"] - baseline_avg
    yearly_str = " ".join(
        f"{yr}:{s:.2f}" for yr, s in sorted(result["yearly_sharpes"].items())
    )
    mdd_str = f"MDD={result['full_mdd']*100:.1f}%"
    print(f"  {name:22s} Avg={result['avg_sharpe']:.3f} (Δ={delta:+.3f}) {mdd_str} | {yearly_str}")


def main():
    print(DIVIDER)
    print("MULTI-EXPIRY VRP RESEARCH")
    print(DIVIDER)

    print("\nLoading data...")
    dataset = load_full_dataset()
    print(f"Dataset: {len(dataset.timeline)} bars, symbols: {dataset.symbols}")

    print(f"\nGamma/Theta scaling by tenor:")
    for dte in TENORS:
        print(f"  {dte:2d}d: gamma_scale={TENOR_GAMMA_SCALE[dte]:.3f}x (vs 30d baseline)")

    # ── 1. Baseline: single 30d tenor ─────────────────────────────────────
    print(f"\n{DIVIDER}")
    print("1. SINGLE-TENOR COMPARISONS")
    print(DIVIDER)
    print("  Testing each DTE individually (full leverage at single tenor)\n")

    single_results = {}
    for dte in TENORS:
        weights = {dte: 1.0}
        result = yearly_metrics(dataset, weights)
        single_results[dte] = result

    baseline_avg = single_results[30]["avg_sharpe"]
    print(f"  {'Tenor':22s} {'Avg':>5s} {'(Δ)':>7s} {'MDD':>9s} | Per-year Sharpe")
    print(f"  {'-'*22} {'-'*5} {'-'*7} {'-'*9} | {'-'*40}")
    for dte in TENORS:
        print_result(f"{dte}d-only", single_results[dte], baseline_avg)

    # ── 2. Multi-tenor ladders ────────────────────────────────────────────
    print(f"\n{DIVIDER}")
    print("2. MULTI-TENOR LADDERS")
    print(DIVIDER)
    print("  Split position across multiple expiries\n")

    ladder_configs = {
        "equal_4tenor": {7: 0.25, 14: 0.25, 30: 0.25, 60: 0.25},
        "front_heavy": {7: 0.40, 14: 0.30, 30: 0.20, 60: 0.10},
        "back_heavy": {7: 0.10, 14: 0.20, 30: 0.30, 60: 0.40},
        "barbell_7_60": {7: 0.50, 60: 0.50},
        "barbell_14_60": {14: 0.50, 60: 0.50},
        "mid_spread_14_30": {14: 0.50, 30: 0.50},
        "front_pair_7_14": {7: 0.50, 14: 0.50},
    }

    ladder_results = {}
    print(f"  {'Config':22s} {'Avg':>5s} {'(Δ)':>7s} {'MDD':>9s} | Per-year Sharpe")
    print(f"  {'-'*22} {'-'*5} {'-'*7} {'-'*9} | {'-'*40}")
    for name, weights in ladder_configs.items():
        result = yearly_metrics(dataset, weights)
        ladder_results[name] = result
        print_result(name, result, baseline_avg)

    # ── 3. Impact of rolling costs ────────────────────────────────────────
    print(f"\n{DIVIDER}")
    print("3. ROLLING COST SENSITIVITY")
    print(DIVIDER)
    print("  How much do roll costs hurt short-DTE tenors?\n")

    cost_mults = [0.0, 0.5, 1.0, 2.0]
    for dte in [7, 30, 60]:
        results_by_cost = []
        for mult in cost_mults:
            r = yearly_metrics(dataset, {dte: 1.0}, roll_cost_mult=mult)
            results_by_cost.append((mult, r["avg_sharpe"], r["full_mdd"]))
        print(f"  {dte}d tenor: ", end="")
        for mult, sh, mdd in results_by_cost:
            print(f"cost×{mult:.1f}→{sh:.3f}  ", end="")
        print()

    # ── 4. Grid search for optimal tenor allocation ───────────────────────
    print(f"\n{DIVIDER}")
    print("4. TENOR ALLOCATION GRID SEARCH")
    print(DIVIDER)
    print("  Searching 2-tenor and 3-tenor allocations (step=0.10)\n")

    best_2tenor = {"name": "", "avg_sharpe": -999, "result": None}
    best_3tenor = {"name": "", "avg_sharpe": -999, "result": None}

    # 2-tenor grid
    for i, dte1 in enumerate(TENORS):
        for dte2 in TENORS[i + 1:]:
            for w1_pct in range(10, 100, 10):
                w1 = w1_pct / 100.0
                w2 = 1.0 - w1
                weights = {dte1: w1, dte2: w2}
                r = yearly_metrics(dataset, weights)
                if r["avg_sharpe"] > best_2tenor["avg_sharpe"]:
                    best_2tenor = {
                        "name": f"{dte1}d:{w1_pct}%+{dte2}d:{100-w1_pct}%",
                        "avg_sharpe": r["avg_sharpe"],
                        "result": r,
                    }

    print(f"  Best 2-tenor: {best_2tenor['name']}")
    if best_2tenor["result"]:
        print_result("best_2tenor", best_2tenor["result"], baseline_avg)

    # 3-tenor grid (step=20% for speed)
    for i, dte1 in enumerate(TENORS):
        for j, dte2 in enumerate(TENORS[i + 1:], i + 1):
            for dte3 in TENORS[j + 1:]:
                for w1_pct in range(20, 80, 20):
                    for w2_pct in range(20, 100 - w1_pct, 20):
                        w3_pct = 100 - w1_pct - w2_pct
                        if w3_pct < 10:
                            continue
                        w1, w2, w3 = w1_pct / 100, w2_pct / 100, w3_pct / 100
                        weights = {dte1: w1, dte2: w2, dte3: w3}
                        r = yearly_metrics(dataset, weights)
                        if r["avg_sharpe"] > best_3tenor["avg_sharpe"]:
                            best_3tenor = {
                                "name": f"{dte1}d:{w1_pct}%+{dte2}d:{w2_pct}%+{dte3}d:{w3_pct}%",
                                "avg_sharpe": r["avg_sharpe"],
                                "result": r,
                            }

    print(f"  Best 3-tenor: {best_3tenor['name']}")
    if best_3tenor["result"]:
        print_result("best_3tenor", best_3tenor["result"], baseline_avg)

    # ── 5. Risk comparison ────────────────────────────────────────────────
    print(f"\n{DIVIDER}")
    print("5. RISK METRICS COMPARISON")
    print(DIVIDER)
    print("  Comparing tail risk across single vs multi-expiry\n")

    risk_configs = {
        "30d_single (baseline)": {30: 1.0},
        "7d_single (max gamma)": {7: 1.0},
        "equal_4tenor_ladder": {7: 0.25, 14: 0.25, 30: 0.25, 60: 0.25},
    }

    if best_2tenor["result"]:
        # Parse best 2-tenor weights
        parts = best_2tenor["name"].split("+")
        bw = {}
        for p in parts:
            d, w = p.split(":")
            dte = int(d.replace("d", ""))
            wt = int(w.replace("%", "")) / 100.0
            bw[dte] = wt
        risk_configs["best_2tenor"] = bw

    for name, weights in risk_configs.items():
        ec, rets, _ = run_multi_expiry_vrp(dataset, weights)
        m = compute_metrics(ec, rets, BARS_PER_YEAR)

        # Tail metrics
        sorted_rets = sorted(rets)
        n_r = len(sorted_rets)
        var_95 = sorted_rets[int(0.05 * n_r)] if n_r > 20 else 0
        var_99 = sorted_rets[int(0.01 * n_r)] if n_r > 100 else 0
        cvar_95 = sum(sorted_rets[:int(0.05 * n_r)]) / max(int(0.05 * n_r), 1) if n_r > 20 else 0
        worst_day = min(rets) if rets else 0
        loss_days_gt1 = sum(1 for r in rets if r < -0.01) / max(n_r, 1) * 100

        print(f"  {name}:")
        print(f"    Sharpe={m['sharpe']:.3f}  MDD={m['max_drawdown']*100:.1f}%  Sortino={m['sortino']:.3f}")
        print(f"    VaR95={var_95*100:.2f}%  VaR99={var_99*100:.2f}%  CVaR95={cvar_95*100:.2f}%")
        print(f"    Worst day={worst_day*100:.2f}%  Loss days>1%={loss_days_gt1:.1f}%")
        print()

    # ── Summary ───────────────────────────────────────────────────────────
    print(DIVIDER)
    print("SUMMARY")
    print(DIVIDER)

    # Collect all results
    all_results = {}
    for dte in TENORS:
        all_results[f"{dte}d_single"] = single_results[dte]["avg_sharpe"]
    for name, r in ladder_results.items():
        all_results[name] = r["avg_sharpe"]
    if best_2tenor["result"]:
        all_results[f"best_2t({best_2tenor['name']})"] = best_2tenor["avg_sharpe"]
    if best_3tenor["result"]:
        all_results[f"best_3t({best_3tenor['name']})"] = best_3tenor["avg_sharpe"]

    # Sort by avg Sharpe
    ranked = sorted(all_results.items(), key=lambda x: x[1], reverse=True)
    print(f"\n  Baseline (30d single): Avg Sharpe = {baseline_avg:.3f}")
    print(f"\n  Rankings:")
    for i, (name, sh) in enumerate(ranked[:10], 1):
        delta = sh - baseline_avg
        marker = " *** BEST" if i == 1 else ""
        marker += " [BASELINE]" if "30d_single" in name else ""
        print(f"    {i:2d}. {name:40s} Avg={sh:.3f} (Δ={delta:+.3f}){marker}")

    best_name, best_sharpe = ranked[0]
    if best_sharpe > baseline_avg + 0.05:
        print(f"\n  FINDING: Multi-expiry IMPROVES on baseline by {best_sharpe - baseline_avg:+.3f}")
        print(f"  Best config: {best_name}")
    else:
        print(f"\n  FINDING: Multi-expiry does NOT meaningfully improve on 30d baseline")
        print(f"  Best improvement: {best_sharpe - baseline_avg:+.3f} — within noise threshold")
        print(f"  Recommendation: Keep single 30d tenor (simplicity, lower costs)")

    print(f"\n  IMPORTANT: Results on synthetic data. Multi-expiry benefits may")
    print(f"  be larger with REAL data (real term structure dynamics).")
    print(DIVIDER)

    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
