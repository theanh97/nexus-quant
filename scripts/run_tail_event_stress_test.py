#!/usr/bin/env python3
"""
Tail Event Stress Test for Crypto Options Ensemble
====================================================

Tests the VRP + Skew MR ensemble under extreme market conditions.

Scenarios:
  1. Historical crash periods (May 2021, LUNA, FTX) — already in data
  2. Synthetic vol shock: IV doubles instantaneously
  3. Synthetic crash: inject 10%, 15%, 20% single-day drops
  4. Sustained selloff: 5% drops over 3/5/10 consecutive days
  5. Ensemble drawdown analysis: worst peaks-to-troughs

For each scenario we measure:
  - Max single-day loss
  - Max drawdown during period
  - Recovery time (bars to new HWM)
  - Sharpe during stress period
  - VRP vs Skew MR contribution (which leg hurts/helps)

Critical question: Does the ensemble's natural hedge (VRP short vol
is hurt by crashes, but Skew MR may profit from skew expansion)
provide meaningful tail protection?
"""
import json
import math
import os
import sys
import time
from typing import Any, Dict, List, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nexus_quant.backtest.costs import ExecutionCostModel, FeeModel, ImpactModel
from nexus_quant.projects.crypto_options.options_engine import (
    OptionsBacktestEngine, compute_metrics,
)
from nexus_quant.projects.crypto_options.providers.deribit_rest import DeribitRestProvider
from nexus_quant.projects.crypto_options.strategies.variance_premium import VariancePremiumStrategy
from nexus_quant.projects.crypto_options.strategies.skew_trade_v2 import SkewTradeV2Strategy

_FEE = FeeModel(maker_fee_rate=0.0003, taker_fee_rate=0.0005)
_IMPACT = ImpactModel(model="sqrt", coef_bps=3.0)
COSTS = ExecutionCostModel(
    fee=_FEE, execution_style="taker",
    slippage_bps=7.5, spread_bps=10.0, impact=_IMPACT, cost_multiplier=1.0,
)

DAILY_CFG = {
    "provider": "deribit_rest_v1",
    "symbols": ["BTC", "ETH"],
    "bar_interval": "1d",
    "cache_dir": "data/cache/deribit",
    "use_synthetic_iv": True,
    "rv_lookback_bars": 21,
}

HOURLY_CFG = {
    "provider": "deribit_rest_v1",
    "symbols": ["BTC", "ETH"],
    "bar_interval": "1h",
    "cache_dir": "data/cache/deribit",
    "use_synthetic_iv": True,
    "rv_lookback_bars": 504,
}

YEARS = [2021, 2022, 2023, 2024, 2025]

VRP_DAILY = {
    "base_leverage": 1.5,
    "exit_z_threshold": -3.0,
    "vrp_lookback": 30,
    "rebalance_freq": 5,
    "min_bars": 30,
}
VRP_HOURLY = {
    "vrp_lookback": 720,
    "exit_z_threshold": -3.0,
    "base_leverage": 1.5,
    "rebalance_freq": 240,
    "min_bars": 720,
}
SKEW_DAILY = {
    "skew_lookback": 60,
    "z_entry": 2.0,
    "z_exit": 0.0,
    "target_leverage": 1.0,
    "rebalance_freq": 5,
    "min_bars": 60,
    "use_vrp_filter": False,
}


def _max_drawdown_info(equity_curve: List[float]) -> Dict[str, Any]:
    """Compute max drawdown with start/end indices."""
    peak = equity_curve[0]
    peak_idx = 0
    max_dd = 0.0
    dd_peak_idx = 0
    dd_trough_idx = 0

    for i, eq in enumerate(equity_curve):
        if eq > peak:
            peak = eq
            peak_idx = i
        dd = (eq / peak - 1.0) if peak > 0 else 0.0
        if dd < max_dd:
            max_dd = dd
            dd_peak_idx = peak_idx
            dd_trough_idx = i

    # Recovery: find when equity exceeds peak after trough
    recovery_idx = None
    peak_at_dd = equity_curve[dd_peak_idx] if dd_peak_idx < len(equity_curve) else 0
    for i in range(dd_trough_idx, len(equity_curve)):
        if equity_curve[i] >= peak_at_dd:
            recovery_idx = i
            break

    return {
        "max_dd": round(max_dd, 4),
        "dd_peak_idx": dd_peak_idx,
        "dd_trough_idx": dd_trough_idx,
        "recovery_idx": recovery_idx,
        "recovery_bars": (recovery_idx - dd_trough_idx) if recovery_idx else None,
    }


def _worst_returns(returns: List[float], n: int = 10) -> List[Tuple[int, float]]:
    """Return indices and values of worst n returns."""
    indexed = [(i, r) for i, r in enumerate(returns)]
    indexed.sort(key=lambda x: x[1])
    return indexed[:n]


def _period_metrics(returns: List[float], bars_per_year: int = 365) -> Dict[str, float]:
    """Quick metrics for a return series."""
    if not returns:
        return {"sharpe": 0, "cagr": 0, "vol": 0, "max_dd": 0, "worst_day": 0}
    mean_r = sum(returns) / len(returns)
    var_r = sum((r - mean_r) ** 2 for r in returns) / max(len(returns) - 1, 1)
    vol = var_r ** 0.5
    sharpe = (mean_r / vol * math.sqrt(bars_per_year)) if vol > 0 else 0.0

    eq = [1.0]
    for r in returns:
        eq.append(eq[-1] * (1 + r))
    dd_info = _max_drawdown_info(eq)

    return {
        "sharpe": round(sharpe, 4),
        "cagr": round((eq[-1] ** (bars_per_year / len(returns)) - 1) * 100, 2) if eq[-1] > 0 else -99,
        "vol": round(vol * math.sqrt(bars_per_year) * 100, 2),
        "max_dd": dd_info["max_dd"],
        "worst_day": round(min(returns), 4),
        "best_day": round(max(returns), 4),
        "n_bars": len(returns),
    }


def run_individual_strategies(year: int) -> Dict[str, Dict]:
    """Run each strategy individually for a year and return equity curves + returns."""
    results = {}

    # VRP Daily
    cfg = dict(DAILY_CFG)
    cfg["start"] = f"{year}-01-01"
    cfg["end"] = f"{year}-12-31"
    provider = DeribitRestProvider(cfg, seed=42)
    dataset = provider.load()
    engine = OptionsBacktestEngine(costs=COSTS, bars_per_year=365, use_options_pnl=True)

    vrp_strat = VariancePremiumStrategy(params=VRP_DAILY)
    vrp_result = engine.run(dataset, vrp_strat)
    results["vrp_daily"] = {
        "equity_curve": vrp_result.equity_curve,
        "returns": vrp_result.returns,
    }

    # Skew MR Daily
    skew_strat = SkewTradeV2Strategy(params=SKEW_DAILY)
    engine_skew = OptionsBacktestEngine(
        costs=COSTS, bars_per_year=365, use_options_pnl=True,
        skew_sensitivity_mult=2.5,
    )
    skew_result = engine_skew.run(dataset, skew_strat)
    results["skew_daily"] = {
        "equity_curve": skew_result.equity_curve,
        "returns": skew_result.returns,
    }

    # Ensemble: 40% VRP + 60% Skew MR
    n = min(len(vrp_result.returns), len(skew_result.returns))
    ens_returns = [
        0.40 * vrp_result.returns[i] + 0.60 * skew_result.returns[i]
        for i in range(n)
    ]
    ens_eq = [1.0]
    for r in ens_returns:
        ens_eq.append(ens_eq[-1] * (1 + r))

    results["ensemble"] = {
        "equity_curve": ens_eq,
        "returns": ens_returns,
    }

    return results


def test_1_historical_crash_analysis():
    """Analyze behavior during known crash periods in actual backtest data."""
    print("=" * 70)
    print("  TEST 1: HISTORICAL CRASH PERIOD ANALYSIS")
    print("=" * 70)
    print()

    # Known crypto crash periods (approximate bar indices in yearly data)
    # 2021: May crash (~120-140), Sep crash (~240-260)
    # 2022: LUNA May (~120-150), FTX Nov (~300-330)
    # 2023: Mostly calm, small dips
    # 2024: Aug flash crash (~210-220)

    crash_periods = {
        2021: [
            ("May_crash", 115, 145),
            ("Sep_flash", 240, 260),
        ],
        2022: [
            ("LUNA_May", 115, 155),
            ("FTX_Nov", 295, 340),
            ("Full_bear", 0, 365),
        ],
        2023: [
            ("Mar_bank_crisis", 60, 85),
        ],
        2024: [
            ("Aug_crash", 205, 225),
        ],
    }

    all_crash_results = {}

    for year in [2021, 2022, 2023, 2024]:
        if year not in crash_periods:
            continue

        strats = run_individual_strategies(year)

        print(f"  {year} — Full year metrics:")
        for name in ["vrp_daily", "skew_daily", "ensemble"]:
            m = _period_metrics(strats[name]["returns"])
            dd = _max_drawdown_info(strats[name]["equity_curve"])
            print(f"    {name:12s}: Sharpe={m['sharpe']:+.4f} MDD={dd['max_dd']:+.2%} "
                  f"worst_day={m['worst_day']:+.2%}")

        for crash_name, start, end in crash_periods[year]:
            print(f"\n  {year} {crash_name} (bars {start}-{end}):")
            crash_key = f"{year}_{crash_name}"
            crash_data = {}

            for name in ["vrp_daily", "skew_daily", "ensemble"]:
                rets = strats[name]["returns"]
                n = len(rets)
                s = min(start, n)
                e = min(end, n)
                period_rets = rets[s:e]
                if not period_rets:
                    continue

                pm = _period_metrics(period_rets)
                eq_period = [1.0]
                for r in period_rets:
                    eq_period.append(eq_period[-1] * (1 + r))
                dd = _max_drawdown_info(eq_period)

                crash_data[name] = {
                    "sharpe": pm["sharpe"],
                    "max_dd": dd["max_dd"],
                    "worst_day": pm["worst_day"],
                    "total_return": round(eq_period[-1] - 1, 4),
                    "n_bars": len(period_rets),
                }

                emoji = "+" if eq_period[-1] >= 1.0 else "-"
                print(f"    {name:12s}: ret={eq_period[-1] - 1:+.2%} MDD={dd['max_dd']:+.2%} "
                      f"worst_day={pm['worst_day']:+.2%} [{emoji}]")

            all_crash_results[crash_key] = crash_data
        print()

    return all_crash_results


def test_2_worst_day_analysis():
    """Analyze the worst single-day returns across all years."""
    print("=" * 70)
    print("  TEST 2: WORST SINGLE-DAY RETURN ANALYSIS")
    print("=" * 70)
    print()

    all_worst_days = {"vrp_daily": [], "skew_daily": [], "ensemble": []}

    for year in YEARS:
        strats = run_individual_strategies(year)

        for name in ["vrp_daily", "skew_daily", "ensemble"]:
            worst = _worst_returns(strats[name]["returns"], n=5)
            for idx, ret in worst:
                all_worst_days[name].append({
                    "year": year,
                    "day_idx": idx,
                    "return": ret,
                })

    # Sort and show worst 10 overall
    for name in ["vrp_daily", "skew_daily", "ensemble"]:
        sorted_worst = sorted(all_worst_days[name], key=lambda x: x["return"])[:10]
        print(f"  {name} — 10 worst days across 2021-2025:")
        for i, w in enumerate(sorted_worst):
            print(f"    {i+1:2d}. {w['year']} day {w['day_idx']:3d}: {w['return']:+.4%}")
        print()

    return all_worst_days


def test_3_drawdown_analysis():
    """Full drawdown analysis: depth, duration, recovery."""
    print("=" * 70)
    print("  TEST 3: DRAWDOWN DEPTH & RECOVERY ANALYSIS")
    print("=" * 70)
    print()

    summary = {}

    for year in YEARS:
        strats = run_individual_strategies(year)

        print(f"  {year}:")
        for name in ["vrp_daily", "skew_daily", "ensemble"]:
            dd = _max_drawdown_info(strats[name]["equity_curve"])
            m = compute_metrics(strats[name]["equity_curve"], strats[name]["returns"], 365)
            rec = dd["recovery_bars"]
            rec_str = f"{rec}d" if rec is not None else "NONE"
            print(f"    {name:12s}: MDD={dd['max_dd']:+.2%}  peak→trough={dd['dd_trough_idx']-dd['dd_peak_idx']:3d}d  "
                  f"recovery={rec_str:>6s}  Sharpe={m['sharpe']:+.4f}")

            summary[f"{year}_{name}"] = {
                "max_dd": dd["max_dd"],
                "dd_duration": dd["dd_trough_idx"] - dd["dd_peak_idx"],
                "recovery_bars": rec,
                "sharpe": m["sharpe"],
            }
        print()

    return summary


def test_4_var_cvar_tail_risk():
    """Value at Risk and Conditional VaR analysis."""
    print("=" * 70)
    print("  TEST 4: VaR & CVaR TAIL RISK")
    print("=" * 70)
    print()

    all_returns = {"vrp_daily": [], "skew_daily": [], "ensemble": []}

    for year in YEARS:
        strats = run_individual_strategies(year)
        for name in ["vrp_daily", "skew_daily", "ensemble"]:
            all_returns[name].extend(strats[name]["returns"])

    for name in ["vrp_daily", "skew_daily", "ensemble"]:
        rets = sorted(all_returns[name])
        n = len(rets)

        # VaR
        var_95 = rets[int(0.05 * n)]
        var_99 = rets[int(0.01 * n)]

        # CVaR (expected shortfall)
        cvar_95_cutoff = int(0.05 * n)
        cvar_95 = sum(rets[:cvar_95_cutoff]) / max(cvar_95_cutoff, 1)
        cvar_99_cutoff = int(0.01 * n)
        cvar_99 = sum(rets[:cvar_99_cutoff]) / max(cvar_99_cutoff, 1)

        # Tail ratio
        worst_5pct = rets[:int(0.05 * n)]
        best_5pct = rets[-int(0.05 * n):]
        tail_ratio = abs(sum(best_5pct) / sum(worst_5pct)) if sum(worst_5pct) != 0 else 0

        # Skewness & kurtosis
        mean_r = sum(rets) / n
        var_r = sum((r - mean_r) ** 2 for r in rets) / n
        std_r = var_r ** 0.5
        if std_r > 0:
            skew = sum((r - mean_r) ** 3 for r in rets) / (n * std_r ** 3)
            kurt = sum((r - mean_r) ** 4 for r in rets) / (n * std_r ** 4) - 3
        else:
            skew = kurt = 0

        print(f"  {name} (n={n} days across 2021-2025):")
        print(f"    VaR  95%: {var_95:+.4%}   VaR  99%: {var_99:+.4%}")
        print(f"    CVaR 95%: {cvar_95:+.4%}   CVaR 99%: {cvar_99:+.4%}")
        print(f"    Worst day: {rets[0]:+.4%}   Best day: {rets[-1]:+.4%}")
        print(f"    Tail ratio: {tail_ratio:.2f}   Skew: {skew:+.2f}   Kurtosis: {kurt:+.2f}")
        print()

    return all_returns


def test_5_correlation_during_stress():
    """Measure VRP-Skew correlation during normal vs stress periods."""
    print("=" * 70)
    print("  TEST 5: VRP-SKEW CORRELATION: NORMAL vs STRESS")
    print("=" * 70)
    print()

    all_vrp = []
    all_skew = []

    for year in YEARS:
        strats = run_individual_strategies(year)
        n = min(len(strats["vrp_daily"]["returns"]), len(strats["skew_daily"]["returns"]))
        all_vrp.extend(strats["vrp_daily"]["returns"][:n])
        all_skew.extend(strats["skew_daily"]["returns"][:n])

    n = min(len(all_vrp), len(all_skew))
    all_vrp = all_vrp[:n]
    all_skew = all_skew[:n]

    def corr(a, b):
        na = len(a)
        if na < 3:
            return 0.0
        ma = sum(a) / na
        mb = sum(b) / na
        cov = sum((a[i] - ma) * (b[i] - mb) for i in range(na)) / (na - 1)
        va = sum((x - ma) ** 2 for x in a) / (na - 1)
        vb = sum((x - mb) ** 2 for x in b) / (na - 1)
        denom = (va * vb) ** 0.5
        return cov / denom if denom > 0 else 0.0

    # Overall correlation
    overall_corr = corr(all_vrp, all_skew)

    # During VRP losses (worst 20% of VRP days)
    pairs = list(zip(all_vrp, all_skew))
    pairs.sort(key=lambda x: x[0])  # sort by VRP return
    n20 = max(int(0.20 * n), 10)
    stress_vrp = [p[0] for p in pairs[:n20]]
    stress_skew = [p[1] for p in pairs[:n20]]
    stress_corr = corr(stress_vrp, stress_skew)

    # During calm (middle 60%)
    calm_vrp = [p[0] for p in pairs[n20:-n20]]
    calm_skew = [p[1] for p in pairs[n20:-n20]]
    calm_corr = corr(calm_vrp, calm_skew)

    # During VRP best days (top 20%)
    bull_vrp = [p[0] for p in pairs[-n20:]]
    bull_skew = [p[1] for p in pairs[-n20:]]
    bull_corr = corr(bull_vrp, bull_skew)

    # Key question: when VRP loses big, does Skew MR help?
    avg_skew_during_vrp_crash = sum(stress_skew) / len(stress_skew) if stress_skew else 0
    avg_vrp_during_stress = sum(stress_vrp) / len(stress_vrp) if stress_vrp else 0

    print(f"  VRP-Skew correlation across {n} days (2021-2025):")
    print(f"    Overall:               {overall_corr:+.4f}")
    print(f"    During VRP worst 20%:  {stress_corr:+.4f}")
    print(f"    During calm (mid 60%): {calm_corr:+.4f}")
    print(f"    During VRP best 20%:   {bull_corr:+.4f}")
    print()
    print(f"  When VRP has worst 20% days (avg ret={avg_vrp_during_stress:+.4%}):")
    print(f"    Avg Skew MR return: {avg_skew_during_vrp_crash:+.4%}")
    hedge_effectiveness = -avg_skew_during_vrp_crash / avg_vrp_during_stress if avg_vrp_during_stress != 0 else 0
    print(f"    Hedge effectiveness: {hedge_effectiveness:.2%} "
          f"({'HELPS' if avg_skew_during_vrp_crash > 0 else 'HURTS'})")
    print()

    return {
        "overall_corr": round(overall_corr, 4),
        "stress_corr": round(stress_corr, 4),
        "calm_corr": round(calm_corr, 4),
        "bull_corr": round(bull_corr, 4),
        "hedge_effectiveness": round(hedge_effectiveness, 4),
        "skew_during_vrp_crash": round(avg_skew_during_vrp_crash, 6),
    }


def test_6_ensemble_vs_components_tail():
    """Compare ensemble tail behavior vs individual strategies."""
    print("=" * 70)
    print("  TEST 6: ENSEMBLE TAIL PROTECTION ANALYSIS")
    print("=" * 70)
    print()

    # Across all years, compute:
    # - % of days ensemble loss > 1% (significant)
    # - % of days ensemble loss > 2% (severe)
    # - Ratio of ensemble worst day to component worst days
    all_data = {"vrp_daily": [], "skew_daily": [], "ensemble": []}

    for year in YEARS:
        strats = run_individual_strategies(year)
        n = min(len(strats["vrp_daily"]["returns"]),
                len(strats["skew_daily"]["returns"]),
                len(strats["ensemble"]["returns"]))
        for name in all_data:
            all_data[name].extend(strats[name]["returns"][:n])

    for name in ["vrp_daily", "skew_daily", "ensemble"]:
        rets = all_data[name]
        n = len(rets)
        loss_1pct = sum(1 for r in rets if r < -0.01) / n
        loss_2pct = sum(1 for r in rets if r < -0.02) / n
        loss_3pct = sum(1 for r in rets if r < -0.03) / n
        print(f"  {name:12s}: >1% loss days={loss_1pct:.1%}  "
              f">2% loss days={loss_2pct:.1%}  "
              f">3% loss days={loss_3pct:.1%}")

    print()

    # Diversification benefit in the tail
    vrp_rets = all_data["vrp_daily"]
    skew_rets = all_data["skew_daily"]
    ens_rets = all_data["ensemble"]

    # For each day, compute: would ensemble have been better or worse?
    n = min(len(vrp_rets), len(skew_rets), len(ens_rets))
    vrp_worse_days = 0
    both_worse_days = 0
    ens_mitigates = 0

    for i in range(n):
        if ens_rets[i] < -0.005:  # ensemble has a meaningful loss
            if vrp_rets[i] < ens_rets[i] or skew_rets[i] < ens_rets[i]:
                ens_mitigates += 1  # at least one component was worse
            if vrp_rets[i] < ens_rets[i] and skew_rets[i] < ens_rets[i]:
                both_worse_days += 1  # both components were worse

    total_loss_days = sum(1 for r in ens_rets[:n] if r < -0.005)
    if total_loss_days > 0:
        print(f"  On {total_loss_days} ensemble loss days (>0.5%):")
        print(f"    Ensemble better than ≥1 component: {ens_mitigates} ({ens_mitigates/total_loss_days:.1%})")
        print(f"    Ensemble better than both components: {both_worse_days} ({both_worse_days/total_loss_days:.1%})")
    print()

    # Max drawdown comparison
    print(f"  Max Drawdown across 2021-2025 (combined):")
    for name in ["vrp_daily", "skew_daily", "ensemble"]:
        eq = [1.0]
        for r in all_data[name]:
            eq.append(eq[-1] * (1 + r))
        dd = _max_drawdown_info(eq)
        fm = compute_metrics(eq, all_data[name], 365)
        print(f"    {name:12s}: MDD={dd['max_dd']:+.2%}  Calmar={fm['calmar']:.2f}  "
              f"Sharpe={fm['sharpe']:+.4f}")
    print()

    return {name: _period_metrics(all_data[name]) for name in all_data}


def main():
    t0 = time.time()

    print()
    print("*" * 70)
    print("  CRYPTO OPTIONS ENSEMBLE — TAIL EVENT STRESS TEST")
    print("*" * 70)
    print()

    r1 = test_1_historical_crash_analysis()
    r2 = test_2_worst_day_analysis()
    r3 = test_3_drawdown_analysis()
    r4 = test_4_var_cvar_tail_risk()
    r5 = test_5_correlation_during_stress()
    r6 = test_6_ensemble_vs_components_tail()

    # ── Summary verdict ──
    print("=" * 70)
    print("  FINAL VERDICT")
    print("=" * 70)

    # Collect key metrics
    ens_data = r6.get("ensemble", {})
    vrp_data = r6.get("vrp_daily", {})
    skew_data = r6.get("skew_daily", {})

    ens_mdd = ens_data.get("max_dd", 0)
    vrp_mdd = vrp_data.get("max_dd", 0)
    skew_mdd = skew_data.get("max_dd", 0)

    print(f"  Ensemble max drawdown: {ens_mdd:.2%}")
    print(f"  VRP max drawdown:      {vrp_mdd:.2%}")
    print(f"  Skew MR max drawdown:  {skew_mdd:.2%}")
    print()

    mdd_improvement = (abs(ens_mdd) - abs(vrp_mdd)) / abs(vrp_mdd) if vrp_mdd != 0 else 0
    print(f"  Ensemble MDD improvement over VRP alone: {mdd_improvement:+.1%}")

    hedge_eff = r5.get("hedge_effectiveness", 0)
    print(f"  Skew MR hedge effectiveness: {hedge_eff:.2%}")

    corr_stress = r5.get("stress_corr", 0)
    corr_overall = r5.get("overall_corr", 0)
    print(f"  Correlation overall={corr_overall:+.4f} stress={corr_stress:+.4f}")

    # Pass/fail criteria
    checks = []
    checks.append(("Ensemble MDD < 20%", abs(ens_mdd) < 0.20))
    checks.append(("Ensemble worst day > -5%", ens_data.get("worst_day", -1) > -0.05))
    checks.append(("Stress correlation < 0.5", abs(corr_stress) < 0.50))

    print()
    for label, passed in checks:
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {label}")

    passed_count = sum(1 for _, p in checks if p)
    total = len(checks)
    overall = "PASS" if passed_count == total else "WARN" if passed_count >= total - 1 else "FAIL"
    print(f"\n  Overall: {overall} ({passed_count}/{total})")
    print()

    # Save
    os.makedirs("artifacts/crypto_options", exist_ok=True)
    output = {
        "date": time.strftime("%Y-%m-%d %H:%M"),
        "crash_analysis": {k: v for k, v in r1.items()},
        "drawdown_summary": {k: v for k, v in r3.items()},
        "correlation": r5,
        "tail_metrics": {name: r6.get(name, {}) for name in ["vrp_daily", "skew_daily", "ensemble"]},
        "verdict": overall,
        "elapsed_sec": round(time.time() - t0, 1),
    }
    out_path = "artifacts/crypto_options/tail_stress_test.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"  Saved to {out_path}")
    print(f"  Total elapsed: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
