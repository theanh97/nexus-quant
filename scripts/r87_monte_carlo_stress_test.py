#!/usr/bin/env python3
"""
R87: Monte Carlo Stress Testing
====================================

Critical pre-deployment analysis:
  1. Bootstrap confidence intervals on Sharpe
  2. Monte Carlo drawdown distribution
  3. Kill-switch hit probability
  4. Synthetic black swan scenarios
  5. Worst-case path analysis
  6. Return distribution analysis (fat tails?)
  7. Sequential drawdown risk
  8. Capital adequacy analysis

Uses actual daily P&L from R82 position tracker as the empirical distribution.
"""
import json
import math
import random
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "cache" / "deribit" / "real_surface"
OUTPUT_PATH = DATA_DIR / "r87_stress_test_results.json"

# Production kill-switch thresholds (R80)
KILL_SWITCH = {
    "max_dd_pct": 1.4,       # 3× historical MaxDD at sens=2.5
    "min_health": 0.25,      # CRITICAL threshold
}

N_SIMULATIONS = 10000
HORIZON_DAYS = 365  # 1-year forward simulation


def load_equity_curve():
    """Load equity curve from position_state.json."""
    path = DATA_DIR / "position_state.json"
    with open(path) as f:
        state = json.load(f)
    return state.get("equity_curve", [])


def load_daily_log():
    """Load daily P&L log from position_state.json."""
    path = DATA_DIR / "position_state.json"
    with open(path) as f:
        state = json.load(f)
    return state.get("daily_log", [])


def compute_daily_returns(equity_curve):
    """Compute daily returns from equity curve."""
    returns = []
    for i in range(1, len(equity_curve)):
        ret = equity_curve[i]["cum_pnl_pct"] - equity_curve[i-1]["cum_pnl_pct"]
        returns.append(ret)
    return returns


# ═══════════════════════════════════════════════════════════════
# Analysis 1: Bootstrap Sharpe Confidence Intervals
# ═══════════════════════════════════════════════════════════════

def analysis_1_bootstrap_sharpe(daily_returns):
    """Bootstrap confidence intervals for Sharpe ratio."""
    print("\n  ── Analysis 1: Bootstrap Sharpe CI ──")

    n = len(daily_returns)
    n_bootstrap = 10000
    sharpes = []

    for _ in range(n_bootstrap):
        sample = random.choices(daily_returns, k=n)
        mean_r = statistics.mean(sample)
        std_r = statistics.stdev(sample) if len(sample) > 1 else 0.001
        sharpe = (mean_r / std_r) * math.sqrt(365) if std_r > 0 else 0
        sharpes.append(sharpe)

    sharpes.sort()
    ci_90 = (sharpes[int(0.05 * n_bootstrap)], sharpes[int(0.95 * n_bootstrap)])
    ci_95 = (sharpes[int(0.025 * n_bootstrap)], sharpes[int(0.975 * n_bootstrap)])
    ci_99 = (sharpes[int(0.005 * n_bootstrap)], sharpes[int(0.995 * n_bootstrap)])

    mean_sharpe = statistics.mean(sharpes)
    median_sharpe = statistics.median(sharpes)
    prob_positive = sum(1 for s in sharpes if s > 0) / n_bootstrap * 100
    prob_gt1 = sum(1 for s in sharpes if s > 1.0) / n_bootstrap * 100
    prob_gt2 = sum(1 for s in sharpes if s > 2.0) / n_bootstrap * 100

    result = {
        "mean_sharpe": round(mean_sharpe, 2),
        "median_sharpe": round(median_sharpe, 2),
        "ci_90": [round(ci_90[0], 2), round(ci_90[1], 2)],
        "ci_95": [round(ci_95[0], 2), round(ci_95[1], 2)],
        "ci_99": [round(ci_99[0], 2), round(ci_99[1], 2)],
        "prob_positive_pct": round(prob_positive, 1),
        "prob_gt1_pct": round(prob_gt1, 1),
        "prob_gt2_pct": round(prob_gt2, 1),
    }

    print(f"    Bootstrap mean Sharpe:  {mean_sharpe:.2f}")
    print(f"    90% CI: [{ci_90[0]:.2f}, {ci_90[1]:.2f}]")
    print(f"    95% CI: [{ci_95[0]:.2f}, {ci_95[1]:.2f}]")
    print(f"    99% CI: [{ci_99[0]:.2f}, {ci_99[1]:.2f}]")
    print(f"    P(Sharpe > 0): {prob_positive:.1f}%")
    print(f"    P(Sharpe > 1): {prob_gt1:.1f}%")
    print(f"    P(Sharpe > 2): {prob_gt2:.1f}%")

    return result


# ═══════════════════════════════════════════════════════════════
# Analysis 2: Monte Carlo Drawdown Distribution
# ═══════════════════════════════════════════════════════════════

def analysis_2_mc_drawdown(daily_returns):
    """Monte Carlo simulation of 1-year drawdown distribution."""
    print("\n  ── Analysis 2: MC Drawdown Distribution (1Y Forward) ──")

    max_dds = []
    final_pnls = []

    for _ in range(N_SIMULATIONS):
        cum = 0
        peak = 0
        max_dd = 0

        for _ in range(HORIZON_DAYS):
            ret = random.choice(daily_returns)
            cum += ret
            if cum > peak:
                peak = cum
            dd = cum - peak
            if dd < max_dd:
                max_dd = dd

        max_dds.append(max_dd)
        final_pnls.append(cum)

    max_dds.sort()
    final_pnls.sort()

    # Drawdown percentiles (negative values)
    dd_pcts = {}
    for pct in [50, 75, 90, 95, 99, 99.5]:
        idx = int(pct / 100 * N_SIMULATIONS)
        dd_pcts[f"p{pct}"] = round(max_dds[idx], 4)

    # Final P&L percentiles
    pnl_pcts = {}
    for pct in [1, 5, 10, 25, 50, 75, 90, 95, 99]:
        idx = int(pct / 100 * N_SIMULATIONS)
        pnl_pcts[f"p{pct}"] = round(final_pnls[idx], 4)

    prob_dd_gt_1pct = sum(1 for dd in max_dds if dd < -1.0) / N_SIMULATIONS * 100
    prob_dd_gt_kill = sum(1 for dd in max_dds if dd < -KILL_SWITCH["max_dd_pct"]) / N_SIMULATIONS * 100
    prob_loss = sum(1 for p in final_pnls if p < 0) / N_SIMULATIONS * 100

    result = {
        "dd_percentiles": dd_pcts,
        "pnl_percentiles": pnl_pcts,
        "prob_dd_gt_1pct": round(prob_dd_gt_1pct, 2),
        "prob_dd_gt_kill_switch": round(prob_dd_gt_kill, 2),
        "prob_1y_loss": round(prob_loss, 2),
        "median_max_dd": round(statistics.median(max_dds), 4),
        "median_final_pnl": round(statistics.median(final_pnls), 4),
        "worst_dd": round(min(max_dds), 4),
        "worst_pnl": round(min(final_pnls), 4),
    }

    print(f"    Median MaxDD:          {statistics.median(max_dds):.4f}%")
    print(f"    95th pctl MaxDD:       {dd_pcts['p95']:.4f}%")
    print(f"    99th pctl MaxDD:       {dd_pcts['p99']:.4f}%")
    print(f"    Worst MaxDD:           {min(max_dds):.4f}%")
    print(f"    P(DD > 1%):            {prob_dd_gt_1pct:.2f}%")
    print(f"    P(DD > {KILL_SWITCH['max_dd_pct']}% kill):   {prob_dd_gt_kill:.2f}%")
    print(f"    P(1Y loss):            {prob_loss:.2f}%")
    print(f"    Median 1Y P&L:         {statistics.median(final_pnls):.4f}%")

    return result


# ═══════════════════════════════════════════════════════════════
# Analysis 3: Kill-Switch Hit Probability
# ═══════════════════════════════════════════════════════════════

def analysis_3_kill_switch(daily_returns):
    """Estimate probability of hitting various kill-switch levels."""
    print("\n  ── Analysis 3: Kill-Switch Hit Probability ──")

    kill_levels = [0.5, 0.75, 1.0, 1.4, 2.0, 3.0, 5.0]  # MaxDD thresholds in %
    horizons = [30, 90, 180, 365]  # Days forward

    results = {}
    print(f"\n    {'Horizon':>10}  ", end="")
    for level in kill_levels:
        print(f"  DD>{level}%", end="")
    print()
    print(f"    {'─'*10}  " + "  ".join("─" * 7 for _ in kill_levels))

    for horizon in horizons:
        hit_counts = {level: 0 for level in kill_levels}

        for _ in range(N_SIMULATIONS):
            cum = 0
            peak = 0
            max_dd = 0

            for _ in range(horizon):
                ret = random.choice(daily_returns)
                cum += ret
                if cum > peak:
                    peak = cum
                dd = cum - peak
                if dd < max_dd:
                    max_dd = dd

            for level in kill_levels:
                if max_dd < -level:
                    hit_counts[level] += 1

        probs = {level: round(hit_counts[level] / N_SIMULATIONS * 100, 2) for level in kill_levels}
        results[f"{horizon}d"] = probs

        print(f"    {horizon:>7}d  ", end="")
        for level in kill_levels:
            p = probs[level]
            print(f"  {p:>5.1f}%", end="")
        print()

    return results


# ═══════════════════════════════════════════════════════════════
# Analysis 4: Synthetic Black Swan Scenarios
# ═══════════════════════════════════════════════════════════════

def analysis_4_black_swans(daily_returns):
    """Simulate extreme scenarios."""
    print("\n  ── Analysis 4: Synthetic Black Swan Scenarios ──")

    mean_ret = statistics.mean(daily_returns)
    std_ret = statistics.stdev(daily_returns)

    # Historical extreme stats
    min_ret = min(daily_returns)
    max_ret = max(daily_returns)
    pct_1 = sorted(daily_returns)[int(0.01 * len(daily_returns))]
    pct_99 = sorted(daily_returns)[int(0.99 * len(daily_returns))]

    scenarios = {
        "March 2020 style (10 consecutive -3σ days)": {
            "n_days": 10,
            "daily_ret": mean_ret - 3 * std_ret,
        },
        "Sustained adverse (30 days at -2σ)": {
            "n_days": 30,
            "daily_ret": mean_ret - 2 * std_ret,
        },
        "Flash crash (1 day at -5σ, then normal)": {
            "n_days": 1,
            "daily_ret": mean_ret - 5 * std_ret,
        },
        "Historical worst day repeated 5x": {
            "n_days": 5,
            "daily_ret": min_ret,
        },
        "Gradual bleed (60 days at -1σ)": {
            "n_days": 60,
            "daily_ret": mean_ret - 1 * std_ret,
        },
    }

    results = {}
    print(f"\n    {'Scenario':>45}  {'Days':>5}  {'Daily':>8}  {'Impact%':>10}  {'Kill?':>6}")
    print(f"    {'─'*45}  {'─'*5}  {'─'*8}  {'─'*10}  {'─'*6}")

    for name, params in scenarios.items():
        impact = params["n_days"] * params["daily_ret"]
        hits_kill = abs(impact) > KILL_SWITCH["max_dd_pct"]

        results[name] = {
            "n_days": params["n_days"],
            "daily_ret": round(params["daily_ret"], 4),
            "total_impact_pct": round(impact, 4),
            "hits_kill_switch": hits_kill,
        }

        kill_str = "YES" if hits_kill else "no"
        print(f"    {name:>45}  {params['n_days']:>5}  {params['daily_ret']:>8.4f}  {impact:>10.4f}  {kill_str:>6}")

    # Context
    print(f"\n    Historical daily return stats:")
    print(f"      Mean:  {mean_ret:+.5f}%  Std: {std_ret:.5f}%")
    print(f"      Min:   {min_ret:+.5f}%  Max: {max_ret:+.5f}%")
    print(f"      1st pct: {pct_1:+.5f}%  99th pct: {pct_99:+.5f}%")
    print(f"      Kill-switch threshold: {KILL_SWITCH['max_dd_pct']}%")

    results["daily_stats"] = {
        "mean": round(mean_ret, 5),
        "std": round(std_ret, 5),
        "min": round(min_ret, 5),
        "max": round(max_ret, 5),
        "pct_1": round(pct_1, 5),
        "pct_99": round(pct_99, 5),
    }

    return results


# ═══════════════════════════════════════════════════════════════
# Analysis 5: Worst-Case Path Analysis
# ═══════════════════════════════════════════════════════════════

def analysis_5_worst_paths(daily_returns):
    """Find and characterize worst MC paths."""
    print("\n  ── Analysis 5: Worst-Case Path Analysis ──")

    paths = []
    for _ in range(N_SIMULATIONS):
        cum = 0
        peak = 0
        max_dd = 0
        min_cum = 0
        path_returns = []

        for _ in range(HORIZON_DAYS):
            ret = random.choice(daily_returns)
            cum += ret
            path_returns.append(ret)
            if cum > peak:
                peak = cum
            dd = cum - peak
            if dd < max_dd:
                max_dd = dd
            if cum < min_cum:
                min_cum = cum

        paths.append({
            "final_pnl": cum,
            "max_dd": max_dd,
            "min_cum": min_cum,
            "max_losing_streak": max_consecutive_losses(path_returns),
        })

    # Sort by max_dd (worst first)
    paths.sort(key=lambda p: p["max_dd"])

    worst_5 = paths[:5]
    print(f"\n    Top 5 worst paths (out of {N_SIMULATIONS}):")
    print(f"    {'#':>4}  {'MaxDD%':>10}  {'MinCum%':>10}  {'Final%':>10}  {'MaxLoseStrk':>12}")
    print(f"    {'─'*4}  {'─'*10}  {'─'*10}  {'─'*10}  {'─'*12}")
    for i, p in enumerate(worst_5):
        print(f"    {i+1:>4}  {p['max_dd']:>10.4f}  {p['min_cum']:>10.4f}  "
              f"{p['final_pnl']:>10.4f}  {p['max_losing_streak']:>12}")

    # Losing streak distribution
    streaks = [p["max_losing_streak"] for p in paths]
    avg_streak = statistics.mean(streaks)
    max_streak = max(streaks)

    result = {
        "worst_5_paths": [{k: round(v, 4) if isinstance(v, float) else v
                           for k, v in p.items()} for p in worst_5],
        "avg_max_losing_streak": round(avg_streak, 1),
        "worst_losing_streak": max_streak,
    }

    print(f"\n    Avg max losing streak: {avg_streak:.1f} days")
    print(f"    Worst losing streak:   {max_streak} days")

    return result


def max_consecutive_losses(returns):
    """Find maximum consecutive loss days."""
    max_streak = 0
    current = 0
    for r in returns:
        if r < 0:
            current += 1
            if current > max_streak:
                max_streak = current
        else:
            current = 0
    return max_streak


# ═══════════════════════════════════════════════════════════════
# Analysis 6: Return Distribution (Fat Tails?)
# ═══════════════════════════════════════════════════════════════

def analysis_6_distribution(daily_returns):
    """Analyze return distribution for fat tails."""
    print("\n  ── Analysis 6: Return Distribution Analysis ──")

    n = len(daily_returns)
    mean = statistics.mean(daily_returns)
    std = statistics.stdev(daily_returns)

    # Skewness
    skew = sum((r - mean) ** 3 for r in daily_returns) / (n * std ** 3) if std > 0 else 0

    # Kurtosis (excess)
    kurt = sum((r - mean) ** 4 for r in daily_returns) / (n * std ** 4) - 3 if std > 0 else 0

    # Jarque-Bera test statistic
    jb = n / 6 * (skew ** 2 + (kurt ** 2) / 4)
    # JB > 5.99 → reject normality at 5% level

    # Count tail events
    events_3sigma = sum(1 for r in daily_returns if abs(r - mean) > 3 * std)
    events_4sigma = sum(1 for r in daily_returns if abs(r - mean) > 4 * std)
    expected_3sigma = n * 0.0027  # Normal: 0.27%
    expected_4sigma = n * 0.0001  # Normal: 0.01%

    result = {
        "n_days": n,
        "mean": round(mean, 5),
        "std": round(std, 5),
        "skewness": round(skew, 3),
        "excess_kurtosis": round(kurt, 3),
        "jarque_bera": round(jb, 1),
        "is_normal": jb < 5.99,
        "events_3sigma": events_3sigma,
        "expected_3sigma": round(expected_3sigma, 1),
        "events_4sigma": events_4sigma,
        "expected_4sigma": round(expected_4sigma, 1),
        "fat_tail_ratio_3s": round(events_3sigma / expected_3sigma, 1) if expected_3sigma > 0 else 0,
    }

    print(f"    N days:         {n}")
    print(f"    Mean:           {mean:+.5f}%")
    print(f"    Std:            {std:.5f}%")
    print(f"    Skewness:       {skew:+.3f} {'(neg skew = left tail)' if skew < -0.5 else '(approx symmetric)' if abs(skew) < 0.5 else '(pos skew = right tail)'}")
    print(f"    Excess Kurt:    {kurt:+.3f} {'(fat tails!)' if kurt > 1 else '(thin tails)' if kurt < -0.5 else '(near normal)'}")
    print(f"    Jarque-Bera:    {jb:.1f} {'(NOT normal)' if jb > 5.99 else '(consistent with normal)'}")
    print(f"    3σ events:      {events_3sigma} (expected {expected_3sigma:.1f} under normal)")
    print(f"    4σ events:      {events_4sigma} (expected {expected_4sigma:.1f} under normal)")
    if expected_3sigma > 0:
        ratio = events_3sigma / expected_3sigma
        print(f"    Fat tail ratio: {ratio:.1f}x normal")

    return result


# ═══════════════════════════════════════════════════════════════
# Analysis 7: Sequential Drawdown Risk
# ═══════════════════════════════════════════════════════════════

def analysis_7_sequential_dd(daily_returns):
    """Analyze time-to-recovery and drawdown duration."""
    print("\n  ── Analysis 7: Sequential Drawdown Risk ──")

    # Simulate many paths and track DD durations
    dd_durations = []  # How long each drawdown lasts
    time_to_recovery = []  # Days from DD trough to new high

    for _ in range(N_SIMULATIONS):
        cum = 0
        peak = 0
        in_dd = False
        dd_start = 0
        trough = 0
        trough_day = 0

        for day in range(HORIZON_DAYS):
            ret = random.choice(daily_returns)
            cum += ret

            if cum > peak:
                if in_dd:
                    # Recovered
                    dd_durations.append(day - dd_start)
                    time_to_recovery.append(day - trough_day)
                    in_dd = False
                peak = cum
            else:
                if not in_dd:
                    dd_start = day
                    in_dd = True
                    trough = cum
                    trough_day = day
                elif cum < trough:
                    trough = cum
                    trough_day = day

        # If still in DD at end
        if in_dd:
            dd_durations.append(HORIZON_DAYS - dd_start)

    if not dd_durations:
        print("    No drawdowns observed (unlikely)")
        return None

    avg_dd_dur = statistics.mean(dd_durations)
    max_dd_dur = max(dd_durations)
    median_dd_dur = statistics.median(dd_durations)

    avg_recovery = statistics.mean(time_to_recovery) if time_to_recovery else 0
    max_recovery = max(time_to_recovery) if time_to_recovery else 0

    result = {
        "n_drawdowns": len(dd_durations),
        "avg_dd_duration_days": round(avg_dd_dur, 1),
        "median_dd_duration_days": round(median_dd_dur, 1),
        "max_dd_duration_days": max_dd_dur,
        "avg_recovery_days": round(avg_recovery, 1),
        "max_recovery_days": max_recovery,
        "pct_time_in_dd": round(sum(dd_durations) / (N_SIMULATIONS * HORIZON_DAYS) * 100, 1),
    }

    print(f"    Total drawdowns observed: {len(dd_durations):,}")
    print(f"    Avg DD duration:  {avg_dd_dur:.1f} days")
    print(f"    Median DD duration: {median_dd_dur:.1f} days")
    print(f"    Max DD duration:  {max_dd_dur} days")
    print(f"    Avg recovery:     {avg_recovery:.1f} days")
    print(f"    Max recovery:     {max_recovery} days")
    print(f"    % time in DD:     {result['pct_time_in_dd']:.1f}%")

    return result


# ═══════════════════════════════════════════════════════════════
# Analysis 8: Capital Adequacy
# ═══════════════════════════════════════════════════════════════

def analysis_8_capital_adequacy(daily_returns):
    """Determine minimum capital to survive various scenarios."""
    print("\n  ── Analysis 8: Capital Adequacy Analysis ──")

    # For different capital levels, what's the probability of ruin?
    # "Ruin" = hitting kill-switch MaxDD
    # Capital in BTC (strategy P&L is in %)

    # Run MC with 99th percentile MaxDD
    mc_max_dds = []
    for _ in range(N_SIMULATIONS):
        cum = 0
        peak = 0
        max_dd = 0
        for _ in range(HORIZON_DAYS):
            ret = random.choice(daily_returns)
            cum += ret
            if cum > peak:
                peak = cum
            dd = cum - peak
            if dd < max_dd:
                max_dd = dd
        mc_max_dds.append(abs(max_dd))

    mc_max_dds.sort()
    pct_95_dd = mc_max_dds[int(0.95 * N_SIMULATIONS)]
    pct_99_dd = mc_max_dds[int(0.99 * N_SIMULATIONS)]
    pct_999_dd = mc_max_dds[int(0.999 * N_SIMULATIONS)]

    # Expected 1Y P&L
    mc_pnls = []
    for _ in range(N_SIMULATIONS):
        cum = sum(random.choice(daily_returns) for _ in range(HORIZON_DAYS))
        mc_pnls.append(cum)
    median_pnl = statistics.median(mc_pnls)
    mean_pnl = statistics.mean(mc_pnls)

    # Capital recommendations
    btc_price = 68400  # Current price
    capital_levels = [0.1, 0.3, 0.5, 1.0, 2.0, 5.0]  # BTC

    print(f"\n    Expected 1Y P&L:  {mean_pnl:.2f}% (median {median_pnl:.2f}%)")
    print(f"    95th pctl MaxDD:  {pct_95_dd:.4f}%")
    print(f"    99th pctl MaxDD:  {pct_99_dd:.4f}%")
    print(f"    99.9th pctl MaxDD:{pct_999_dd:.4f}%")

    print(f"\n    {'Capital':>10}  {'USD':>10}  {'ExpProfit':>12}  {'99% MaxLoss':>12}  {'Verdict':>12}")
    print(f"    {'─'*10}  {'─'*10}  {'─'*12}  {'─'*12}  {'─'*12}")

    results = {}
    for btc in capital_levels:
        usd = btc * btc_price
        exp_profit = btc * btc_price * mean_pnl / 100
        max_loss_99 = btc * btc_price * pct_99_dd / 100

        if max_loss_99 < 100:
            verdict = "ADEQUATE"
        elif max_loss_99 < 500:
            verdict = "MARGINAL"
        else:
            verdict = "RISKY"

        results[f"{btc} BTC"] = {
            "usd": round(usd),
            "exp_1y_profit": round(exp_profit, 2),
            "max_loss_99pct": round(max_loss_99, 2),
            "verdict": verdict,
        }

        print(f"    {btc:>8.1f} BTC  ${usd:>9,.0f}  ${exp_profit:>11,.2f}  ${max_loss_99:>11,.2f}  {verdict:>12}")

    results["risk_metrics"] = {
        "pct_95_dd": round(pct_95_dd, 4),
        "pct_99_dd": round(pct_99_dd, 4),
        "pct_999_dd": round(pct_999_dd, 4),
        "mean_1y_pnl": round(mean_pnl, 4),
        "median_1y_pnl": round(median_pnl, 4),
    }

    return results


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════

def main():
    print("=" * 70)
    print("  R87: Monte Carlo Stress Testing")
    print("=" * 70)

    random.seed(42)  # Reproducibility

    equity_curve = load_equity_curve()
    daily_returns = compute_daily_returns(equity_curve)
    print(f"\n  Loaded {len(daily_returns)} daily returns")
    print(f"  Date range: {equity_curve[0]['date']} to {equity_curve[-1]['date']}")
    print(f"  Kill-switch MaxDD: {KILL_SWITCH['max_dd_pct']}%")
    print(f"  Simulations: {N_SIMULATIONS:,}")
    print(f"  Horizon: {HORIZON_DAYS} days")

    all_results = {}

    all_results["bootstrap_sharpe"] = analysis_1_bootstrap_sharpe(daily_returns)
    all_results["mc_drawdown"] = analysis_2_mc_drawdown(daily_returns)
    all_results["kill_switch_prob"] = analysis_3_kill_switch(daily_returns)
    all_results["black_swans"] = analysis_4_black_swans(daily_returns)
    all_results["worst_paths"] = analysis_5_worst_paths(daily_returns)
    all_results["distribution"] = analysis_6_distribution(daily_returns)
    all_results["sequential_dd"] = analysis_7_sequential_dd(daily_returns)
    all_results["capital_adequacy"] = analysis_8_capital_adequacy(daily_returns)

    # ── Overall Assessment ──
    print("\n" + "=" * 70)
    print("  OVERALL STRESS TEST ASSESSMENT")
    print("=" * 70)

    bs = all_results["bootstrap_sharpe"]
    mc = all_results["mc_drawdown"]
    dist = all_results["distribution"]
    cap = all_results["capital_adequacy"]

    checks = []
    # 1. Sharpe confidence
    if bs["ci_95"][0] > 0:
        checks.append(("PASS", "95% CI lower bound > 0"))
    else:
        checks.append(("FAIL", f"95% CI lower bound = {bs['ci_95'][0]}"))

    # 2. Kill-switch probability
    kill_prob = mc["prob_dd_gt_kill_switch"]
    if kill_prob < 1.0:
        checks.append(("PASS", f"P(DD > kill) = {kill_prob}% < 1%"))
    elif kill_prob < 5.0:
        checks.append(("WARN", f"P(DD > kill) = {kill_prob}% < 5%"))
    else:
        checks.append(("FAIL", f"P(DD > kill) = {kill_prob}% >= 5%"))

    # 3. Probability of 1Y loss
    loss_prob = mc["prob_1y_loss"]
    if loss_prob < 5.0:
        checks.append(("PASS", f"P(1Y loss) = {loss_prob}% < 5%"))
    elif loss_prob < 15.0:
        checks.append(("WARN", f"P(1Y loss) = {loss_prob}% < 15%"))
    else:
        checks.append(("FAIL", f"P(1Y loss) = {loss_prob}% >= 15%"))

    # 4. Fat tails
    if dist["excess_kurtosis"] < 3:
        checks.append(("PASS", f"Excess kurtosis = {dist['excess_kurtosis']} < 3"))
    else:
        checks.append(("WARN", f"Excess kurtosis = {dist['excess_kurtosis']} >= 3 (fat tails)"))

    # 5. Sharpe probability
    if bs["prob_gt1_pct"] > 90:
        checks.append(("PASS", f"P(Sharpe > 1) = {bs['prob_gt1_pct']}% > 90%"))
    elif bs["prob_gt1_pct"] > 75:
        checks.append(("WARN", f"P(Sharpe > 1) = {bs['prob_gt1_pct']}% > 75%"))
    else:
        checks.append(("FAIL", f"P(Sharpe > 1) = {bs['prob_gt1_pct']}% < 75%"))

    pass_count = sum(1 for c in checks if c[0] == "PASS")
    warn_count = sum(1 for c in checks if c[0] == "WARN")
    fail_count = sum(1 for c in checks if c[0] == "FAIL")

    for status, msg in checks:
        icon = "OK" if status == "PASS" else "!!" if status == "WARN" else "XX"
        print(f"    [{icon}] {msg}")

    print(f"\n    Score: {pass_count} PASS, {warn_count} WARN, {fail_count} FAIL")

    if fail_count == 0 and warn_count <= 1:
        overall = "GREEN — Strategy passes stress tests. Safe for production deployment."
    elif fail_count == 0:
        overall = "YELLOW — Strategy mostly passes. Proceed with caution, monitor closely."
    else:
        overall = "RED — Significant risks identified. Review before deployment."

    print(f"    OVERALL: {overall}")

    all_results["assessment"] = {
        "pass": pass_count,
        "warn": warn_count,
        "fail": fail_count,
        "overall": overall,
        "checks": [{"status": s, "message": m} for s, m in checks],
    }

    # Save
    with open(OUTPUT_PATH, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n  Results saved: {OUTPUT_PATH}")
    print("=" * 70)


if __name__ == "__main__":
    main()
