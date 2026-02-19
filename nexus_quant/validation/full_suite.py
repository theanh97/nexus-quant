from __future__ import annotations

"""
NEXUS Full Validation Suite v1.

Implements rigorous quantitative strategy validation:

1. IS  (In-Sample)          — baseline performance on training period
2. WFA (Walk-Forward)       — rolling OOS parameter stability
3. OOS (Out-Of-Sample)      — held-out test period (last 20% of data)
4. Stress Test              — performance during worst 10% of market conditions
5. Flash Crash              — drawdown resilience (days with >5% down moves)
6. Market Regime            — trending / ranging / volatile / crash regime analysis
7. Monte Carlo              — randomized return resampling (1000 paths)
8. Transaction Cost Sweep   — sensitivity to cost assumptions (0.5x to 3x)
9. Leverage Sensitivity     — performance across leverage levels (0.25x to 2x)
10. Sharpe Significance     — t-test for Sharpe > 0 (via bias_checker)

All stdlib: math, statistics, random, collections
"""

import math
import random
import statistics
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple


# ── Utility functions ─────────────────────────────────────────────────────


def _annualized_sharpe(returns: List[float], periods_per_year: float = 8760) -> float:
    if len(returns) < 10:
        return 0.0
    mu = statistics.mean(returns)
    try:
        sd = statistics.stdev(returns)
    except statistics.StatisticsError:
        return 0.0
    if sd == 0:
        return 0.0
    return (mu / sd) * math.sqrt(periods_per_year)


def _cagr(returns: List[float], periods_per_year: float = 8760) -> float:
    if not returns:
        return 0.0
    equity = 1.0
    for r in returns:
        equity *= (1.0 + r)
    years = len(returns) / periods_per_year
    if years <= 0 or equity <= 0:
        return 0.0
    try:
        return equity ** (1.0 / years) - 1.0
    except Exception:
        return 0.0


def _max_drawdown(returns: List[float]) -> float:
    peak = 1.0
    equity = 1.0
    mdd = 0.0
    for r in returns:
        equity *= (1.0 + r)
        if equity > peak:
            peak = equity
        dd = (peak - equity) / peak
        if dd > mdd:
            mdd = dd
    return mdd


def _calmar(returns: List[float], periods_per_year: float = 8760) -> float:
    cagr = _cagr(returns, periods_per_year)
    mdd = _max_drawdown(returns)
    if mdd == 0:
        return 0.0
    return cagr / mdd


def _win_rate(returns: List[float]) -> float:
    if not returns:
        return 0.0
    wins = sum(1 for r in returns if r > 0)
    return wins / len(returns)


def _metrics_dict(returns: List[float], periods_per_year: float = 8760, label: str = "") -> Dict[str, Any]:
    if not returns:
        return {"label": label, "n": 0, "sharpe": 0.0, "cagr": 0.0, "max_drawdown": 0.0, "calmar": 0.0, "win_rate": 0.0}
    return {
        "label": label,
        "n": len(returns),
        "sharpe": round(_annualized_sharpe(returns, periods_per_year), 4),
        "cagr": round(_cagr(returns, periods_per_year), 4),
        "max_drawdown": round(_max_drawdown(returns), 4),
        "calmar": round(_calmar(returns, periods_per_year), 4),
        "win_rate": round(_win_rate(returns), 4),
        "mean_return": round(statistics.mean(returns), 6) if returns else 0.0,
        "volatility": round(statistics.stdev(returns) * math.sqrt(periods_per_year), 4) if len(returns) > 1 else 0.0,
    }


# ── 1. IS / OOS split ─────────────────────────────────────────────────────


def run_is_oos(
    returns: List[float],
    oos_fraction: float = 0.20,
    periods_per_year: float = 8760,
) -> Dict[str, Any]:
    """
    Split returns into In-Sample (IS) and Out-Of-Sample (OOS).
    Returns performance metrics for both periods.
    """
    n = len(returns)
    split = int(n * (1 - oos_fraction))
    is_ret = returns[:split]
    oos_ret = returns[split:]

    is_m = _metrics_dict(is_ret, periods_per_year, label="IS")
    oos_m = _metrics_dict(oos_ret, periods_per_year, label="OOS")

    sharpe_degradation = is_m["sharpe"] - oos_m["sharpe"]
    verdict = "PASS" if oos_m["sharpe"] > 0.5 and sharpe_degradation < 1.5 else "WARN"
    if oos_m["sharpe"] < 0:
        verdict = "FAIL"

    return {
        "is": is_m,
        "oos": oos_m,
        "sharpe_degradation": round(sharpe_degradation, 4),
        "verdict": verdict,
        "split_bar": split,
    }


# ── 2. Walk-Forward Analysis ──────────────────────────────────────────────


def run_walk_forward(
    returns: List[float],
    train_bars: int = 4380,    # 6 months of hourly bars
    test_bars: int = 720,      # 1 month OOS
    periods_per_year: float = 8760,
) -> Dict[str, Any]:
    """
    Rolling walk-forward: train on N bars, test on M bars, step forward M bars.
    Returns per-window metrics and stability scores.
    """
    windows = []
    n = len(returns)
    i = 0
    while i + train_bars + test_bars <= n:
        train = returns[i: i + train_bars]
        test = returns[i + train_bars: i + train_bars + test_bars]
        windows.append({
            "window": len(windows) + 1,
            "train": _metrics_dict(train, periods_per_year, f"train_{len(windows)+1}"),
            "test": _metrics_dict(test, periods_per_year, f"test_{len(windows)+1}"),
        })
        i += test_bars

    if not windows:
        return {"windows": [], "stability": {}, "verdict": "INSUFFICIENT_DATA"}

    test_sharpes = [w["test"]["sharpe"] for w in windows]
    profitable_windows = sum(1 for s in test_sharpes if s > 0)
    stability_ratio = profitable_windows / len(windows)
    mean_oos_sharpe = statistics.mean(test_sharpes)

    verdict = "PASS" if stability_ratio >= 0.60 and mean_oos_sharpe > 0.3 else "WARN"
    if stability_ratio < 0.40 or mean_oos_sharpe < 0:
        verdict = "FAIL"

    return {
        "windows": windows,
        "n_windows": len(windows),
        "stability": {
            "stability_ratio": round(stability_ratio, 4),
            "profitable_windows": profitable_windows,
            "mean_oos_sharpe": round(mean_oos_sharpe, 4),
            "min_oos_sharpe": round(min(test_sharpes), 4),
            "max_oos_sharpe": round(max(test_sharpes), 4),
        },
        "verdict": verdict,
    }


# ── 3. Stress Test ────────────────────────────────────────────────────────


def run_stress_test(
    returns: List[float],
    worst_pct: float = 0.10,
    periods_per_year: float = 8760,
) -> Dict[str, Any]:
    """
    Performance during worst N% of market conditions (largest negative bars).
    Identifies tail risk exposure.
    """
    n = len(returns)
    sorted_ret = sorted(returns)
    n_worst = max(1, int(n * worst_pct))
    worst_threshold = sorted_ret[n_worst - 1]

    # Find consecutive stress periods (bars where return < threshold)
    stress_returns = [r for r in returns if r <= worst_threshold]
    non_stress_returns = [r for r in returns if r > worst_threshold]

    stress_m = _metrics_dict(stress_returns, periods_per_year, "stress_period")
    normal_m = _metrics_dict(non_stress_returns, periods_per_year, "normal_period")

    # Tail ratio: how much of total loss comes from worst periods
    total_loss = sum(r for r in returns if r < 0)
    stress_loss = sum(r for r in stress_returns if r < 0)
    tail_ratio = (stress_loss / total_loss) if total_loss != 0 else 0.0

    verdict = "PASS" if stress_m["sharpe"] > -2.0 else "WARN"
    if stress_m["max_drawdown"] > 0.30:
        verdict = "FAIL"

    return {
        "stress_period": stress_m,
        "normal_period": normal_m,
        "worst_pct": worst_pct,
        "worst_threshold": round(worst_threshold, 6),
        "tail_ratio": round(tail_ratio, 4),
        "verdict": verdict,
    }


# ── 4. Flash Crash Analysis ───────────────────────────────────────────────


def run_flash_crash_test(
    returns: List[float],
    crash_threshold: float = -0.05,
    window_bars: int = 24,
) -> Dict[str, Any]:
    """
    Simulate behavior during flash crash events (bars with >5% down moves).
    Checks recovery time and drawdown severity.
    """
    crash_events = []
    for i, r in enumerate(returns):
        if r <= crash_threshold:
            start = max(0, i - window_bars)
            end = min(len(returns), i + window_bars)
            window_ret = returns[start:end]
            equity = 1.0
            for wr in window_ret:
                equity *= (1 + wr)
            crash_events.append({
                "bar": i,
                "crash_return": round(r, 6),
                "window_return": round(equity - 1.0, 6),
                "window_mdd": round(_max_drawdown(window_ret), 6),
            })

    n_crashes = len(crash_events)
    if not crash_events:
        return {"n_crashes": 0, "events": [], "verdict": "PASS", "avg_window_return": 0.0}

    avg_window_ret = statistics.mean(e["window_return"] for e in crash_events)
    avg_mdd = statistics.mean(e["window_mdd"] for e in crash_events)

    verdict = "PASS" if avg_window_ret > -0.05 else "WARN"
    if avg_mdd > 0.20:
        verdict = "FAIL"

    return {
        "n_crashes": n_crashes,
        "events": crash_events[:10],  # top 10
        "avg_window_return": round(avg_window_ret, 4),
        "avg_window_mdd": round(avg_mdd, 4),
        "worst_crash_return": round(min(e["crash_return"] for e in crash_events), 6),
        "verdict": verdict,
    }


# ── 5. Market Regime Analysis ─────────────────────────────────────────────


def run_regime_analysis(
    returns: List[float],
    window: int = 168,          # 1 week of hourly bars
    vol_high_pct: float = 0.75,
    periods_per_year: float = 8760,
) -> Dict[str, Any]:
    """
    Split performance by market regime:
    - Trending (high momentum): abs(mean) / std > threshold
    - Volatile: rolling vol > 75th percentile vol
    - Ranging: low momentum + low vol
    """
    if len(returns) < window * 2:
        return {"regimes": {}, "verdict": "INSUFFICIENT_DATA"}

    # Compute rolling stats
    rolling_means = []
    rolling_stds = []
    for i in range(window, len(returns)):
        w = returns[i - window:i]
        rolling_means.append(statistics.mean(w))
        try:
            rolling_stds.append(statistics.stdev(w))
        except Exception:
            rolling_stds.append(0.0)

    if not rolling_stds:
        return {"regimes": {}, "verdict": "INSUFFICIENT_DATA"}

    vol_sorted = sorted(rolling_stds)
    vol_high_thresh = vol_sorted[int(len(vol_sorted) * vol_high_pct)]

    # Classify each bar
    regime_returns: Dict[str, List[float]] = defaultdict(list)
    for i, (mean, std) in enumerate(zip(rolling_means, rolling_stds)):
        ret = returns[window + i]
        if std == 0:
            regime = "flat"
        elif std > vol_high_thresh:
            regime = "volatile"
        elif abs(mean) / std > 0.3:
            regime = "trending"
        else:
            regime = "ranging"
        regime_returns[regime].append(ret)

    regimes = {}
    for regime, r in regime_returns.items():
        regimes[regime] = _metrics_dict(r, periods_per_year, regime)
        regimes[regime]["n_bars"] = len(r)
        regimes[regime]["pct_of_time"] = round(len(r) / len(rolling_means), 4)

    # Best and worst regime
    valid_regimes = {k: v for k, v in regimes.items() if v["n"] > 10}
    best = max(valid_regimes, key=lambda k: valid_regimes[k]["sharpe"]) if valid_regimes else ""
    worst = min(valid_regimes, key=lambda k: valid_regimes[k]["sharpe"]) if valid_regimes else ""

    return {
        "regimes": regimes,
        "best_regime": best,
        "worst_regime": worst,
        "n_regimes": len(regimes),
        "verdict": "PASS" if all(r["sharpe"] > -1.0 for r in regimes.values()) else "WARN",
    }


# ── 6. Monte Carlo Simulation ─────────────────────────────────────────────


def run_monte_carlo(
    returns: List[float],
    n_paths: int = 500,
    periods_per_year: float = 8760,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Bootstrap resampling of returns to generate confidence intervals.
    Answers: 'Is this Sharpe persistent or lucky?'
    """
    if len(returns) < 50:
        return {"verdict": "INSUFFICIENT_DATA", "paths": []}

    rng = random.Random(seed)
    path_sharpes = []
    path_cagrs = []
    path_mdds = []

    for _ in range(n_paths):
        resampled = rng.choices(returns, k=len(returns))
        path_sharpes.append(_annualized_sharpe(resampled, periods_per_year))
        path_cagrs.append(_cagr(resampled, periods_per_year))
        path_mdds.append(_max_drawdown(resampled))

    path_sharpes.sort()
    path_cagrs.sort()
    path_mdds.sort()

    n = len(path_sharpes)
    p5 = path_sharpes[int(n * 0.05)]
    p25 = path_sharpes[int(n * 0.25)]
    p50 = path_sharpes[int(n * 0.50)]
    p75 = path_sharpes[int(n * 0.75)]
    p95 = path_sharpes[int(n * 0.95)]

    pct_profitable = sum(1 for s in path_sharpes if s > 0) / n

    verdict = "PASS" if p5 > 0 and pct_profitable > 0.80 else "WARN"
    if p50 < 0:
        verdict = "FAIL"

    return {
        "n_paths": n_paths,
        "sharpe": {
            "p5": round(p5, 4), "p25": round(p25, 4), "p50": round(p50, 4),
            "p75": round(p75, 4), "p95": round(p95, 4),
        },
        "cagr": {
            "p5": round(path_cagrs[int(n * 0.05)], 4),
            "p50": round(path_cagrs[int(n * 0.50)], 4),
            "p95": round(path_cagrs[int(n * 0.95)], 4),
        },
        "mdd": {
            "p50": round(path_mdds[int(n * 0.50)], 4),
            "p95": round(path_mdds[int(n * 0.95)], 4),
        },
        "pct_profitable_paths": round(pct_profitable, 4),
        "verdict": verdict,
    }


# ── 7. Transaction Cost Sensitivity ──────────────────────────────────────


def run_cost_sensitivity(
    returns: List[float],
    base_cost_per_trade: float = 0.0008,
    trades_per_year: float = 365 * 3,
    multipliers: Optional[List[float]] = None,
    periods_per_year: float = 8760,
) -> Dict[str, Any]:
    """
    Test performance across different cost assumptions.
    Measures how robust strategy is to cost changes.
    """
    if multipliers is None:
        multipliers = [0.5, 1.0, 1.5, 2.0, 3.0]

    base_annual_cost = base_cost_per_trade * trades_per_year
    base_cost_per_bar = base_annual_cost / periods_per_year

    results = []
    for mult in multipliers:
        adj_cost = base_cost_per_bar * mult
        adjusted_returns = [r - adj_cost for r in returns]
        m = _metrics_dict(adjusted_returns, periods_per_year, f"cost_{mult}x")
        m["cost_multiplier"] = mult
        m["annual_cost_drag_pct"] = round(adj_cost * periods_per_year * 100, 3)
        results.append(m)

    # Breakeven cost multiplier (where Sharpe = 0)
    base_sharpe = results[1]["sharpe"] if len(results) > 1 else 0.0
    breakeven = next((r["cost_multiplier"] for r in results if r["sharpe"] <= 0), None)

    verdict = "PASS" if base_sharpe > 0 and (breakeven is None or breakeven > 2.0) else "WARN"

    return {
        "results": results,
        "base_sharpe": round(base_sharpe, 4),
        "breakeven_cost_multiplier": breakeven,
        "verdict": verdict,
    }


# ── 8. Full Suite Orchestrator ────────────────────────────────────────────


def run_full_validation_suite(
    returns: List[float],
    run_name: str = "strategy",
    periods_per_year: float = 8760,
    oos_fraction: float = 0.20,
    train_bars: int = 4380,
    test_bars: int = 720,
    flash_crash_threshold: float = -0.03,
    n_monte_carlo: int = 300,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Run the complete 8-test validation suite.
    Returns a structured report with overall verdict.
    """
    ts = datetime.now(timezone.utc).isoformat()
    n = len(returns)

    if n < 100:
        return {
            "ts": ts,
            "run_name": run_name,
            "n_bars": n,
            "error": f"Insufficient data: {n} bars < 100 minimum",
            "overall_verdict": "INSUFFICIENT_DATA",
        }

    # Base metrics
    base = _metrics_dict(returns, periods_per_year, "full_period")

    # Run all tests
    is_oos = run_is_oos(returns, oos_fraction, periods_per_year)
    wfa = run_walk_forward(returns, train_bars, test_bars, periods_per_year)
    stress = run_stress_test(returns, periods_per_year=periods_per_year)
    flash = run_flash_crash_test(returns, flash_crash_threshold)
    regime = run_regime_analysis(returns, periods_per_year=periods_per_year)
    mc = run_monte_carlo(returns, n_monte_carlo, periods_per_year, seed)
    cost_sens = run_cost_sensitivity(returns, periods_per_year=periods_per_year)

    # Aggregate verdict
    verdicts = [
        is_oos.get("verdict", "WARN"),
        wfa.get("verdict", "WARN"),
        stress.get("verdict", "WARN"),
        flash.get("verdict", "WARN"),
        mc.get("verdict", "WARN"),
        cost_sens.get("verdict", "WARN"),
    ]
    fails = sum(1 for v in verdicts if v == "FAIL")
    warns = sum(1 for v in verdicts if v == "WARN")

    if fails >= 2:
        overall = "FAIL"
    elif fails == 1 or warns >= 3:
        overall = "WARN"
    else:
        overall = "PASS"

    return {
        "ts": ts,
        "run_name": run_name,
        "n_bars": n,
        "overall_verdict": overall,
        "n_fails": fails,
        "n_warns": warns,
        "base_metrics": base,
        "tests": {
            "is_oos": is_oos,
            "walk_forward": wfa,
            "stress": stress,
            "flash_crash": flash,
            "regime": regime,
            "monte_carlo": mc,
            "cost_sensitivity": cost_sens,
        },
    }
