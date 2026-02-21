#!/usr/bin/env python3
"""
R90: Early Warning System — Can We Predict Drawdowns?
========================================================

R89 found three pre-drawdown indicators:
  1. Elevated IV (+14% above normal before BTC drawdowns)
  2. Z-score volatility (2.25x higher before drawdowns)
  3. BF feature std (25% higher before drawdowns)

This study quantifies:
  1. Predictive power of each indicator (precision/recall)
  2. Composite early warning score
  3. Can we use warnings to IMPROVE the strategy? (reduce position size)
  4. Walk-forward test of warning system
  5. False positive rate analysis
  6. Optimal warning thresholds
  7. Integration recommendation
"""
import csv
import json
import math
import statistics
from datetime import datetime
from pathlib import Path
from collections import defaultdict

ROOT = Path(__file__).resolve().parents[1]
SURFACE_DIR = ROOT / "data" / "cache" / "deribit" / "real_surface"
DVOL_DIR = ROOT / "data" / "cache" / "deribit" / "dvol"
OUTPUT_PATH = SURFACE_DIR / "r90_early_warning_results.json"

BF_CONFIG = {
    "bf_lookback": 120,
    "bf_z_entry": 1.5,
    "bf_z_exit": 0.0,
    "bf_sensitivity": 2.5,
}


def load_surface(currency):
    path = SURFACE_DIR / f"{currency}_daily_surface.csv"
    data = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            d = row["date"]
            entry = {}
            for field in ["butterfly_25d", "iv_atm", "skew_25d"]:
                val = row.get(field, "")
                if val and val != "None":
                    entry[field] = float(val)
            if "butterfly_25d" in entry:
                data[d] = entry
    return data


def load_dvol(currency):
    path = DVOL_DIR / f"{currency}_DVOL_12h.csv"
    daily = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            d = row["date"][:10]
            try:
                daily[d] = float(row["dvol_close"])
            except (ValueError, KeyError):
                pass
    return daily


def run_backtest_with_warnings(surface, dvol, start_date="2021-03-24"):
    """Run BF backtest, returning daily log with warning indicators."""
    dates = sorted(d for d in surface if "butterfly_25d" in surface[d] and d >= start_date)
    lb = BF_CONFIG["bf_lookback"]
    if len(dates) < lb + 30:
        return None

    bf_vals = []
    z_scores = []
    iv_history = []
    position = 0
    daily_log = []
    cum_pnl = 0
    peak_pnl = 0

    for i, date in enumerate(dates):
        bf = surface[date]["butterfly_25d"]
        bf_vals.append(bf)
        iv = dvol.get(date, 50)
        if iv > 1:
            iv = iv / 100
        iv_history.append(iv)

        if len(bf_vals) < lb:
            continue

        window = bf_vals[-lb:]
        mean_bf = statistics.mean(window)
        std_bf = statistics.stdev(window) if len(window) > 1 else 0.001
        if std_bf < 0.0001:
            std_bf = 0.0001
        z = (bf - mean_bf) / std_bf
        z_scores.append(z)

        prev_position = position
        if position == 0:
            if z > BF_CONFIG["bf_z_entry"]:
                position = -1
            elif z < -BF_CONFIG["bf_z_entry"]:
                position = 1
        else:
            if position == 1 and z > BF_CONFIG["bf_z_entry"]:
                position = -1
            elif position == -1 and z < -BF_CONFIG["bf_z_entry"]:
                position = 1

        dt = 1 / 365
        day_pnl = 0
        if i > 0 and prev_position != 0:
            prev_date = dates[i - 1]
            prev_bf = surface[prev_date]["butterfly_25d"]
            bf_change = bf - prev_bf
            day_pnl = prev_position * bf_change * BF_CONFIG["bf_sensitivity"] * iv * math.sqrt(dt)

        cum_pnl += day_pnl
        if cum_pnl > peak_pnl:
            peak_pnl = cum_pnl
        dd = cum_pnl - peak_pnl

        # Early warning indicators
        # 1. IV elevation: current IV vs 60d rolling mean
        if len(iv_history) >= 60:
            iv_60d_mean = statistics.mean(iv_history[-60:])
            iv_60d_std = statistics.stdev(iv_history[-60:]) if len(iv_history) >= 60 else 0.01
            iv_z = (iv - iv_60d_mean) / iv_60d_std if iv_60d_std > 0 else 0
        else:
            iv_z = 0

        # 2. Z-score volatility: 5d rolling std of z-changes
        if len(z_scores) >= 6:
            z_changes = [abs(z_scores[-j] - z_scores[-j-1]) for j in range(1, min(6, len(z_scores)))]
            z_vol = statistics.mean(z_changes) if z_changes else 0
        else:
            z_vol = 0

        # 3. BF std relative to its own rolling average
        if len(bf_vals) >= lb + 60:
            # Compute recent 60 values of 120d std
            recent_stds = []
            for k in range(max(0, len(bf_vals)-60), len(bf_vals)):
                if k >= lb:
                    w = bf_vals[k-lb:k]
                    recent_stds.append(statistics.stdev(w))
            bf_std_mean = statistics.mean(recent_stds) if recent_stds else std_bf
            bf_std_ratio = std_bf / bf_std_mean if bf_std_mean > 0 else 1
        else:
            bf_std_ratio = 1

        daily_log.append({
            "date": date,
            "day_pnl": round(day_pnl * 100, 4),
            "cum_pnl": round(cum_pnl * 100, 4),
            "drawdown": round(dd * 100, 4),
            "z_score": round(z, 3),
            "position": prev_position,
            "iv": round(iv * 100, 1),
            "bf_std": round(std_bf, 5),
            # Warning indicators
            "iv_z": round(iv_z, 3),
            "z_vol": round(z_vol, 3),
            "bf_std_ratio": round(bf_std_ratio, 3),
        })

    return daily_log


# ═══════════════════════════════════════════════════════════════
# Analysis 1: Individual Indicator Precision/Recall
# ═══════════════════════════════════════════════════════════════

def analysis_1_indicator_power(daily_log, asset):
    """Test each indicator's ability to predict 5-day forward drawdown."""
    print(f"\n  ── Analysis 1: {asset} Indicator Predictive Power ──")

    # Label days: is there a significant drawdown in the next 5 days?
    n = len(daily_log)
    labels = []
    for i in range(n):
        max_dd_5d = 0
        for j in range(i+1, min(i+6, n)):
            dd_change = daily_log[j]["cum_pnl"] - daily_log[i]["cum_pnl"]
            if dd_change < max_dd_5d:
                max_dd_5d = dd_change
        labels.append(1 if max_dd_5d < -0.05 else 0)  # >5bps drawdown = positive event

    total_positive = sum(labels)
    total_negative = n - total_positive

    indicators = {
        "iv_z": [d["iv_z"] for d in daily_log],
        "z_vol": [d["z_vol"] for d in daily_log],
        "bf_std_ratio": [d["bf_std_ratio"] for d in daily_log],
    }

    results = {}
    thresholds = {
        "iv_z": [0.5, 1.0, 1.5, 2.0],
        "z_vol": [0.2, 0.3, 0.4, 0.5, 0.7],
        "bf_std_ratio": [1.1, 1.2, 1.3, 1.5],
    }

    for name, values in indicators.items():
        print(f"\n    {name}:")
        print(f"      {'Threshold':>10}  {'TP':>5}  {'FP':>5}  {'Precision':>10}  {'Recall':>8}  {'F1':>6}")
        print(f"      {'─'*10}  {'─'*5}  {'─'*5}  {'─'*10}  {'─'*8}  {'─'*6}")

        best_f1 = 0
        best_thresh = 0
        indicator_results = []

        for thresh in thresholds.get(name, []):
            tp = sum(1 for i in range(n) if values[i] > thresh and labels[i] == 1)
            fp = sum(1 for i in range(n) if values[i] > thresh and labels[i] == 0)
            fn = sum(1 for i in range(n) if values[i] <= thresh and labels[i] == 1)

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

            indicator_results.append({
                "threshold": thresh,
                "tp": tp, "fp": fp,
                "precision": round(precision, 3),
                "recall": round(recall, 3),
                "f1": round(f1, 3),
            })

            if f1 > best_f1:
                best_f1 = f1
                best_thresh = thresh

            print(f"      {thresh:>10.1f}  {tp:>5}  {fp:>5}  {precision:>10.3f}  {recall:>8.3f}  {f1:>6.3f}")

        results[name] = {"thresholds": indicator_results, "best_threshold": best_thresh, "best_f1": round(best_f1, 3)}
        print(f"      Best: threshold={best_thresh}, F1={best_f1:.3f}")

    results["total_positive_days"] = total_positive
    results["total_negative_days"] = total_negative
    return results


# ═══════════════════════════════════════════════════════════════
# Analysis 2: Composite Warning Score
# ═══════════════════════════════════════════════════════════════

def analysis_2_composite_score(daily_log, asset):
    """Build a composite warning score from all three indicators."""
    print(f"\n  ── Analysis 2: {asset} Composite Warning Score ──")

    # Normalize each indicator to [0, 1] using empirical percentiles
    iv_z_vals = sorted([d["iv_z"] for d in daily_log])
    z_vol_vals = sorted([d["z_vol"] for d in daily_log])
    bf_std_vals = sorted([d["bf_std_ratio"] for d in daily_log])

    def percentile_rank(vals, v):
        idx = 0
        for x in vals:
            if x <= v:
                idx += 1
        return idx / len(vals)

    # Add composite score to log
    for day in daily_log:
        s1 = percentile_rank(iv_z_vals, day["iv_z"])
        s2 = percentile_rank(z_vol_vals, day["z_vol"])
        s3 = percentile_rank(bf_std_vals, day["bf_std_ratio"])
        day["warning_score"] = round((s1 + s2 + s3) / 3, 3)

    # Test warning score thresholds
    n = len(daily_log)
    results = []

    print(f"\n    {'Threshold':>10}  {'Days>':>6}  {'AvgPnL':>8}  {'WinRate':>8}  {'AvgPnL<':>8}  {'WinRate<':>9}")
    print(f"    {'─'*10}  {'─'*6}  {'─'*8}  {'─'*8}  {'─'*8}  {'─'*9}")

    for thresh in [0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9]:
        high_pnl = [d["day_pnl"] for d in daily_log if d["warning_score"] > thresh and d["position"] != 0]
        low_pnl = [d["day_pnl"] for d in daily_log if d["warning_score"] <= thresh and d["position"] != 0]

        if len(high_pnl) < 10 or len(low_pnl) < 10:
            continue

        high_avg = statistics.mean(high_pnl)
        high_wr = sum(1 for p in high_pnl if p > 0) / len(high_pnl) * 100
        low_avg = statistics.mean(low_pnl)
        low_wr = sum(1 for p in low_pnl if p > 0) / len(low_pnl) * 100

        results.append({
            "threshold": thresh,
            "n_high": len(high_pnl),
            "avg_pnl_high": round(high_avg, 4),
            "win_rate_high": round(high_wr, 1),
            "n_low": len(low_pnl),
            "avg_pnl_low": round(low_avg, 4),
            "win_rate_low": round(low_wr, 1),
        })

        print(f"    {thresh:>10.2f}  {len(high_pnl):>6}  {high_avg:>8.4f}  {high_wr:>7.1f}%  {low_avg:>8.4f}  {low_wr:>8.1f}%")

    return results


# ═══════════════════════════════════════════════════════════════
# Analysis 3: Warning-Based Position Sizing
# ═══════════════════════════════════════════════════════════════

def analysis_3_warning_sizing(daily_log, asset):
    """Can we use warnings to reduce position and improve Sharpe?"""
    print(f"\n  ── Analysis 3: {asset} Warning-Based Position Sizing ──")

    sizing_rules = [
        ("No sizing (baseline)", lambda ws: 1.0),
        ("Half on warning>0.70", lambda ws: 0.5 if ws > 0.70 else 1.0),
        ("Half on warning>0.80", lambda ws: 0.5 if ws > 0.80 else 1.0),
        ("Zero on warning>0.80", lambda ws: 0.0 if ws > 0.80 else 1.0),
        ("Graduated (1.0/0.75/0.5)", lambda ws: 0.5 if ws > 0.80 else 0.75 if ws > 0.70 else 1.0),
        ("Inverse warning", lambda ws: max(0.3, 1.0 - ws)),
    ]

    results = {}
    print(f"\n    {'Rule':>30}  {'Sharpe':>8}  {'AnnRet%':>8}  {'MaxDD%':>8}  {'WinRate':>8}  {'Delta':>8}")
    print(f"    {'─'*30}  {'─'*8}  {'─'*8}  {'─'*8}  {'─'*8}  {'─'*8}")

    baseline_sharpe = None

    for name, rule in sizing_rules:
        sized_pnl = []
        cum = 0
        peak = 0
        max_dd = 0

        for day in daily_log:
            if day["position"] == 0:
                continue
            size = rule(day["warning_score"])
            pnl = day["day_pnl"] * size
            sized_pnl.append(pnl)
            cum += pnl
            if cum > peak:
                peak = cum
            dd = cum - peak
            if dd < max_dd:
                max_dd = dd

        if len(sized_pnl) < 30:
            continue

        mean_p = statistics.mean(sized_pnl)
        std_p = statistics.stdev(sized_pnl) if len(sized_pnl) > 1 else 0.001
        sharpe = (mean_p / std_p) * math.sqrt(365) if std_p > 0 else 0
        ann_ret = mean_p * 365
        win_rate = sum(1 for p in sized_pnl if p > 0) / len(sized_pnl) * 100

        if baseline_sharpe is None:
            baseline_sharpe = sharpe

        delta = sharpe - baseline_sharpe

        results[name] = {
            "sharpe": round(sharpe, 2),
            "ann_ret_pct": round(ann_ret, 2),
            "max_dd_pct": round(max_dd, 4),
            "win_rate": round(win_rate, 1),
            "delta_sharpe": round(delta, 2),
        }

        print(f"    {name:>30}  {sharpe:>8.2f}  {ann_ret:>8.2f}  {max_dd:>8.4f}  {win_rate:>7.1f}%  {delta:>+8.2f}")

    # Key question: does any sizing beat baseline?
    best = max(results.items(), key=lambda x: x[1]["sharpe"])
    if best[1]["delta_sharpe"] > 0.1:
        verdict = f"YES — {best[0]} improves Sharpe by +{best[1]['delta_sharpe']:.2f}"
    elif best[1]["delta_sharpe"] > 0:
        verdict = f"MARGINAL — best improvement only +{best[1]['delta_sharpe']:.2f}"
    else:
        verdict = "NO — warning-based sizing does NOT improve Sharpe (10th static>dynamic!)"
    print(f"\n    Verdict: {verdict}")

    results["verdict"] = verdict
    return results


# ═══════════════════════════════════════════════════════════════
# Analysis 4: Walk-Forward Warning Test
# ═══════════════════════════════════════════════════════════════

def analysis_4_walk_forward(daily_log, asset):
    """Walk-forward test: train warning model on past data, test on next year."""
    print(f"\n  ── Analysis 4: {asset} Walk-Forward Warning Test ──")

    # Group by year
    by_year = defaultdict(list)
    for day in daily_log:
        by_year[day["date"][:4]].append(day)

    years = sorted(by_year.keys())
    results = []

    for test_year in years:
        if test_year < "2022":
            continue

        # Train on all previous years
        train_data = []
        for y in years:
            if y < test_year:
                train_data.extend(by_year[y])

        if len(train_data) < 100:
            continue

        # Compute train statistics for warning thresholds
        train_ws = [d["warning_score"] for d in train_data]
        train_p70 = sorted(train_ws)[int(0.70 * len(train_ws))]

        # Test: apply half-sizing when warning > train 70th percentile
        test_data = by_year[test_year]
        baseline_pnl = [d["day_pnl"] for d in test_data if d["position"] != 0]
        sized_pnl = [d["day_pnl"] * (0.5 if d["warning_score"] > train_p70 else 1.0)
                     for d in test_data if d["position"] != 0]

        if len(baseline_pnl) < 30:
            continue

        def calc_sharpe(pnls):
            m = statistics.mean(pnls)
            s = statistics.stdev(pnls) if len(pnls) > 1 else 0.001
            return (m / s) * math.sqrt(365) if s > 0 else 0

        base_sharpe = calc_sharpe(baseline_pnl)
        sized_sharpe = calc_sharpe(sized_pnl)

        results.append({
            "year": test_year,
            "baseline_sharpe": round(base_sharpe, 2),
            "sized_sharpe": round(sized_sharpe, 2),
            "delta": round(sized_sharpe - base_sharpe, 2),
            "threshold_used": round(train_p70, 3),
        })

    print(f"\n    {'Year':>6}  {'Baseline':>10}  {'Warning-sized':>14}  {'Delta':>8}  {'Threshold':>10}")
    print(f"    {'─'*6}  {'─'*10}  {'─'*14}  {'─'*8}  {'─'*10}")
    for r in results:
        print(f"    {r['year']:>6}  {r['baseline_sharpe']:>10.2f}  {r['sized_sharpe']:>14.2f}  "
              f"{r['delta']:>+8.2f}  {r['threshold_used']:>10.3f}")

    if results:
        avg_delta = statistics.mean([r["delta"] for r in results])
        positive = sum(1 for r in results if r["delta"] > 0)
        print(f"\n    Avg delta: {avg_delta:+.2f}, positive years: {positive}/{len(results)}")

    return results


# ═══════════════════════════════════════════════════════════════
# Analysis 5: False Positive Analysis
# ═══════════════════════════════════════════════════════════════

def analysis_5_false_positives(daily_log, asset):
    """How often do warnings fire without subsequent drawdown?"""
    print(f"\n  ── Analysis 5: {asset} False Positive Analysis ──")

    # Warning fires when score > 0.75
    threshold = 0.75
    n = len(daily_log)

    tp = 0  # Warning + drawdown follows
    fp = 0  # Warning but no drawdown
    fn = 0  # No warning but drawdown
    tn = 0  # No warning, no drawdown

    for i in range(n - 5):
        warning = daily_log[i]["warning_score"] > threshold
        # Check if 5-day forward drawdown > 5 bps
        max_dd_5d = 0
        for j in range(i+1, min(i+6, n)):
            dd_change = daily_log[j]["cum_pnl"] - daily_log[i]["cum_pnl"]
            if dd_change < max_dd_5d:
                max_dd_5d = dd_change
        actual_dd = max_dd_5d < -0.05

        if warning and actual_dd:
            tp += 1
        elif warning and not actual_dd:
            fp += 1
        elif not warning and actual_dd:
            fn += 1
        else:
            tn += 1

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    false_positive_rate = fp / (fp + tn) if (fp + tn) > 0 else 0

    result = {
        "threshold": threshold,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": round(precision, 3),
        "recall": round(recall, 3),
        "false_positive_rate": round(false_positive_rate, 3),
    }

    print(f"    Confusion matrix (threshold={threshold}):")
    print(f"                        Actual DD    No DD")
    print(f"      Warning fired:    TP={tp:>5}     FP={fp:>5}")
    print(f"      No warning:       FN={fn:>5}     TN={tn:>5}")
    print(f"\n    Precision: {precision:.3f} (when warning fires, {precision*100:.1f}% of the time DD follows)")
    print(f"    Recall:    {recall:.3f} (of all DDs, warning catches {recall*100:.1f}%)")
    print(f"    False positive rate: {false_positive_rate:.3f}")

    return result


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════

def main():
    print("=" * 70)
    print("  R90: Early Warning System — Can We Predict Drawdowns?")
    print("=" * 70)

    all_results = {}

    for currency in ["BTC", "ETH"]:
        print(f"\n  {'='*60}")
        print(f"  {currency}")
        print(f"  {'='*60}")

        surface = load_surface(currency)
        dvol = load_dvol(currency)

        daily_log = run_backtest_with_warnings(surface, dvol)
        if not daily_log:
            print(f"    FAILED: Insufficient data for {currency}")
            continue

        print(f"    Backtest: {len(daily_log)} days")

        r1 = analysis_1_indicator_power(daily_log, currency)
        r2 = analysis_2_composite_score(daily_log, currency)
        r3 = analysis_3_warning_sizing(daily_log, currency)
        r4 = analysis_4_walk_forward(daily_log, currency)
        r5 = analysis_5_false_positives(daily_log, currency)

        all_results[currency] = {
            "indicator_power": r1,
            "composite_score": r2,
            "warning_sizing": r3,
            "walk_forward": r4,
            "false_positives": r5,
        }

    # Overall verdict
    print("\n" + "=" * 70)
    print("  OVERALL VERDICT")
    print("=" * 70)

    sizing_improves = False
    for currency in ["BTC", "ETH"]:
        if currency in all_results:
            sizing = all_results[currency].get("warning_sizing", {})
            verdict = sizing.get("verdict", "")
            print(f"\n    {currency}: {verdict}")
            if "YES" in verdict:
                sizing_improves = True

    if sizing_improves:
        print("\n    RECOMMENDATION: Integrate warning-based position sizing into production.")
    else:
        print("\n    RECOMMENDATION: Use warnings for MONITORING ONLY, not position sizing.")
        print("    This is the 10th confirmation: static > dynamic at ALL levels.")
        print("    Warnings are valuable for human awareness, not automated action.")

    all_results["recommendation"] = (
        "MONITOR_ONLY" if not sizing_improves else "INTEGRATE_SIZING"
    )

    # Save
    with open(OUTPUT_PATH, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n  Results saved: {OUTPUT_PATH}")
    print("=" * 70)


if __name__ == "__main__":
    main()
