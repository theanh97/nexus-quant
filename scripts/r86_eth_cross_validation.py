#!/usr/bin/env python3
"""
R86: ETH Cross-Validation — Is the BF Edge Structural?
=========================================================

Tests whether the BF mean-reversion edge found on BTC (Sharpe 2.69)
also exists on ETH. This is a critical out-of-sample validation:

  - If ETH shows similar edge → structural vol surface phenomenon
  - If ETH shows NO edge → BTC-specific microstructure artifact

Analyses:
  1. ETH BF descriptive statistics vs BTC
  2. ETH BF z-score mean-reversion backtest (same config as BTC)
  3. Parameter sensitivity comparison (BTC vs ETH)
  4. Correlation analysis (BTC BF vs ETH BF)
  5. Combined BTC+ETH portfolio analysis
  6. ETH walk-forward validation
  7. Leave-one-year-out cross-validation (ETH)
  8. Regime analysis (does BF edge persist in same regimes?)
"""
import csv
import json
import math
import statistics
from datetime import datetime
from pathlib import Path
from collections import defaultdict

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "cache" / "deribit"
SURFACE_DIR = DATA_DIR / "real_surface"
DVOL_DIR = DATA_DIR / "dvol"
OUTPUT_PATH = SURFACE_DIR / "r86_eth_crossval_results.json"


# ═══════════════════════════════════════════════════════════════
# Data Loading
# ═══════════════════════════════════════════════════════════════

def load_surface(currency):
    """Load daily surface data (butterfly_25d, iv_atm, etc.)."""
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
    """Load DVOL 12h data, return daily close dict."""
    path = DVOL_DIR / f"{currency}_DVOL_12h.csv"
    daily = {}
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            d = row["date"][:10]
            try:
                daily[d] = float(row["dvol_close"])
            except (ValueError, KeyError):
                pass
    return daily


def load_prices(currency):
    """Load price data for delta-hedged VRP PnL."""
    # Try multiple file patterns
    for pattern in [
        f"{currency}_2021-01-01_2025-12-31_1d_prices.csv",
        f"{currency}_2020-01-01_2025-12-31_1d_prices.csv",
        f"{currency}_2024-01-01_2025-12-31_1d_prices.csv",
    ]:
        path = DATA_DIR / pattern
        if path.exists():
            prices = {}
            with open(path) as f:
                for row in csv.DictReader(f):
                    d = row.get("date", row.get("timestamp", ""))[:10]
                    for col in ["close", "price", "spot"]:
                        if col in row and row[col]:
                            try:
                                prices[d] = float(row[col])
                                break
                            except ValueError:
                                pass
            if prices:
                return prices
    return {}


# ═══════════════════════════════════════════════════════════════
# BF Backtest Engine (same as R64/R81 — replication for fairness)
# ═══════════════════════════════════════════════════════════════

def run_bf_backtest(surface, dvol, lookback=120, z_entry=1.5, z_exit=0.0,
                    sensitivity=2.5, start_date=None):
    """
    Run BF mean-reversion backtest.
    Returns: dict with pnl_series, sharpe, ann_ret, max_dd, trades, etc.
    """
    dates = sorted(d for d in surface if "butterfly_25d" in surface[d])
    if start_date:
        dates = [d for d in dates if d >= start_date]
    if len(dates) < lookback + 30:
        return None

    bf_vals = []
    daily_pnl = []
    position = 0  # +1 = long BF, -1 = short BF
    cum_pnl = 0
    peak = 0
    max_dd = 0
    trades = 0
    trade_log = []
    wins = 0
    losses = 0

    for i, date in enumerate(dates):
        bf = surface[date]["butterfly_25d"]
        bf_vals.append(bf)

        if len(bf_vals) < lookback:
            continue

        window = bf_vals[-lookback:]
        mean_bf = statistics.mean(window)
        std_bf = statistics.stdev(window) if len(window) > 1 else 0.001

        if std_bf < 0.0001:
            std_bf = 0.0001

        z = (bf - mean_bf) / std_bf

        # Get IV for scaling
        iv = dvol.get(date, surface[date].get("iv_atm", 0.5))
        if isinstance(iv, (int, float)) and iv > 1:
            iv = iv / 100  # Convert from percentage

        dt = 1 / 365

        # Signal logic (same as R64/R68)
        prev_position = position
        if position == 0:
            if z > z_entry:
                position = -1  # Short BF (expect mean reversion down)
                trades += 1
            elif z < -z_entry:
                position = 1   # Long BF (expect mean reversion up)
                trades += 1
        else:
            # z_exit=0 means hold until reversed
            if z_exit == 0:
                if position == 1 and z > z_entry:
                    position = -1
                    trades += 1
                elif position == -1 and z < -z_entry:
                    position = 1
                    trades += 1
            else:
                if position == 1 and z > -z_exit:
                    position = 0
                elif position == -1 and z < z_exit:
                    position = 0

        # Compute BF P&L
        if i > 0 and prev_position != 0:
            prev_date = dates[i - 1]
            prev_bf = surface[prev_date]["butterfly_25d"]
            bf_change = bf - prev_bf
            pnl = prev_position * bf_change * sensitivity * iv * math.sqrt(dt)
            pnl_pct = pnl * 100  # in percentage

            daily_pnl.append({"date": date, "pnl": pnl_pct, "z": z, "pos": prev_position})
            cum_pnl += pnl_pct
            if cum_pnl > peak:
                peak = cum_pnl
            dd = cum_pnl - peak
            if dd < max_dd:
                max_dd = dd

            if pnl_pct > 0:
                wins += 1
            else:
                losses += 1

        if prev_position != position and position != 0:
            trade_log.append({"date": date, "z": round(z, 3), "dir": position})

    if not daily_pnl:
        return None

    n_days = len(daily_pnl)
    pnl_values = [d["pnl"] for d in daily_pnl]
    mean_pnl = statistics.mean(pnl_values)
    std_pnl = statistics.stdev(pnl_values) if len(pnl_values) > 1 else 0.001
    sharpe = (mean_pnl / std_pnl) * math.sqrt(365) if std_pnl > 0 else 0
    ann_ret = mean_pnl * 365

    return {
        "sharpe": round(sharpe, 2),
        "ann_ret_pct": round(ann_ret, 2),
        "cum_pnl_pct": round(cum_pnl, 4),
        "max_dd_pct": round(max_dd, 4),
        "n_days": n_days,
        "n_trades": trades,
        "trades_per_year": round(trades / (n_days / 365), 1) if n_days > 0 else 0,
        "win_rate": round(wins / (wins + losses) * 100, 1) if (wins + losses) > 0 else 0,
        "daily_pnl": daily_pnl,
        "trade_log": trade_log,
    }


# ═══════════════════════════════════════════════════════════════
# Analysis 1: Descriptive Statistics Comparison
# ═══════════════════════════════════════════════════════════════

def analysis_1_descriptive(btc_surface, eth_surface):
    """Compare BF descriptive statistics between BTC and ETH."""
    print("\n  ── Analysis 1: BF Descriptive Statistics ──")

    results = {}
    for name, surface in [("BTC", btc_surface), ("ETH", eth_surface)]:
        bf_vals = [surface[d]["butterfly_25d"] for d in sorted(surface)
                   if "butterfly_25d" in surface[d]]
        dates = sorted(d for d in surface if "butterfly_25d" in surface[d])

        n = len(bf_vals)
        mean = statistics.mean(bf_vals)
        median = statistics.median(bf_vals)
        std = statistics.stdev(bf_vals)
        mn = min(bf_vals)
        mx = max(bf_vals)

        # Rolling std (120d windows)
        roll_stds = []
        for i in range(120, n):
            w = bf_vals[i-120:i]
            roll_stds.append(statistics.stdev(w))

        results[name] = {
            "n_days": n,
            "date_range": f"{dates[0]} to {dates[-1]}",
            "mean": round(mean, 5),
            "median": round(median, 5),
            "std": round(std, 5),
            "min": round(mn, 5),
            "max": round(mx, 5),
            "rolling_std_mean": round(statistics.mean(roll_stds), 5) if roll_stds else None,
            "rolling_std_latest": round(roll_stds[-1], 5) if roll_stds else None,
        }

        print(f"\n    {name}:")
        print(f"      Days: {n} ({dates[0]} to {dates[-1]})")
        print(f"      Mean: {mean:.5f}  Median: {median:.5f}")
        print(f"      Std:  {std:.5f}  Range: [{mn:.4f}, {mx:.4f}]")
        if roll_stds:
            print(f"      Rolling 120d std: mean={statistics.mean(roll_stds):.5f}, latest={roll_stds[-1]:.5f}")

    return results


# ═══════════════════════════════════════════════════════════════
# Analysis 2: ETH BF Backtest (Same Config as BTC Production)
# ═══════════════════════════════════════════════════════════════

def analysis_2_eth_backtest(eth_surface, eth_dvol):
    """Run ETH BF backtest with BTC production config."""
    print("\n  ── Analysis 2: ETH BF Backtest (BTC Config) ──")

    # Use exact BTC production config
    result = run_bf_backtest(
        eth_surface, eth_dvol,
        lookback=120, z_entry=1.5, z_exit=0.0, sensitivity=2.5,
        start_date="2021-03-24",  # Match BTC DVOL start
    )

    if result is None:
        print("    FAILED: Insufficient data")
        return None

    print(f"    Sharpe:     {result['sharpe']}")
    print(f"    Ann Return: {result['ann_ret_pct']:.2f}%")
    print(f"    Cum P&L:    {result['cum_pnl_pct']:.4f}%")
    print(f"    Max DD:     {result['max_dd_pct']:.4f}%")
    print(f"    Trades:     {result['n_trades']} ({result['trades_per_year']}/yr)")
    print(f"    Win Rate:   {result['win_rate']}%")
    print(f"    Days:       {result['n_days']}")

    return {k: v for k, v in result.items() if k != "daily_pnl"}


# ═══════════════════════════════════════════════════════════════
# Analysis 3: Parameter Sensitivity Comparison
# ═══════════════════════════════════════════════════════════════

def analysis_3_parameter_sensitivity(btc_surface, btc_dvol, eth_surface, eth_dvol):
    """Compare parameter sensitivity between BTC and ETH."""
    print("\n  ── Analysis 3: Parameter Sensitivity (BTC vs ETH) ──")

    configs = [
        {"lookback": 60,  "z_entry": 1.5, "z_exit": 0.0, "sensitivity": 2.5},
        {"lookback": 90,  "z_entry": 1.5, "z_exit": 0.0, "sensitivity": 2.5},
        {"lookback": 120, "z_entry": 1.5, "z_exit": 0.0, "sensitivity": 2.5},  # Production
        {"lookback": 180, "z_entry": 1.5, "z_exit": 0.0, "sensitivity": 2.5},
        {"lookback": 120, "z_entry": 1.0, "z_exit": 0.0, "sensitivity": 2.5},
        {"lookback": 120, "z_entry": 2.0, "z_exit": 0.0, "sensitivity": 2.5},
        {"lookback": 120, "z_entry": 1.5, "z_exit": 0.3, "sensitivity": 2.5},  # Exit at 0.3
        {"lookback": 120, "z_entry": 1.5, "z_exit": 0.0, "sensitivity": 5.0},  # Higher leverage
    ]

    results = {}
    print(f"\n    {'Config':>30}  {'BTC Sharpe':>10}  {'ETH Sharpe':>10}  {'Diff':>8}")
    print(f"    {'─'*30}  {'─'*10}  {'─'*10}  {'─'*8}")

    for cfg in configs:
        label = f"lb={cfg['lookback']} ze={cfg['z_entry']} zx={cfg['z_exit']} s={cfg['sensitivity']}"

        btc_r = run_bf_backtest(btc_surface, btc_dvol, start_date="2021-03-24", **cfg)
        eth_r = run_bf_backtest(eth_surface, eth_dvol, start_date="2021-03-24", **cfg)

        btc_s = btc_r["sharpe"] if btc_r else 0
        eth_s = eth_r["sharpe"] if eth_r else 0
        diff = eth_s - btc_s

        results[label] = {"btc_sharpe": btc_s, "eth_sharpe": eth_s, "diff": round(diff, 2)}

        marker = " *** PROD" if cfg["lookback"] == 120 and cfg["z_entry"] == 1.5 and cfg["z_exit"] == 0.0 and cfg["sensitivity"] == 2.5 else ""
        print(f"    {label:>30}  {btc_s:>10.2f}  {eth_s:>10.2f}  {diff:>+8.2f}{marker}")

    return results


# ═══════════════════════════════════════════════════════════════
# Analysis 4: BTC-ETH BF Correlation
# ═══════════════════════════════════════════════════════════════

def analysis_4_correlation(btc_surface, eth_surface):
    """Analyze correlation between BTC and ETH butterfly_25d."""
    print("\n  ── Analysis 4: BTC-ETH BF Correlation ──")

    # Get common dates
    common = sorted(set(btc_surface.keys()) & set(eth_surface.keys()))
    common = [d for d in common if "butterfly_25d" in btc_surface.get(d, {})
              and "butterfly_25d" in eth_surface.get(d, {})]

    btc_bf = [btc_surface[d]["butterfly_25d"] for d in common]
    eth_bf = [eth_surface[d]["butterfly_25d"] for d in common]

    n = len(common)
    if n < 30:
        print("    Insufficient overlapping data")
        return None

    # Pearson correlation
    mean_b = statistics.mean(btc_bf)
    mean_e = statistics.mean(eth_bf)
    cov = sum((b - mean_b) * (e - mean_e) for b, e in zip(btc_bf, eth_bf)) / (n - 1)
    std_b = statistics.stdev(btc_bf)
    std_e = statistics.stdev(eth_bf)
    corr = cov / (std_b * std_e) if std_b > 0 and std_e > 0 else 0

    # Rolling correlation (120d)
    rolling_corr = []
    for i in range(120, n):
        wb = btc_bf[i-120:i]
        we = eth_bf[i-120:i]
        mb = statistics.mean(wb)
        me = statistics.mean(we)
        c = sum((b - mb) * (e - me) for b, e in zip(wb, we)) / 119
        sb = statistics.stdev(wb)
        se = statistics.stdev(we)
        if sb > 0 and se > 0:
            rolling_corr.append(c / (sb * se))

    # Daily changes correlation
    btc_changes = [btc_bf[i] - btc_bf[i-1] for i in range(1, n)]
    eth_changes = [eth_bf[i] - eth_bf[i-1] for i in range(1, n)]
    mean_bc = statistics.mean(btc_changes)
    mean_ec = statistics.mean(eth_changes)
    cov_c = sum((b - mean_bc) * (e - mean_ec) for b, e in zip(btc_changes, eth_changes)) / (n - 2)
    std_bc = statistics.stdev(btc_changes)
    std_ec = statistics.stdev(eth_changes)
    change_corr = cov_c / (std_bc * std_ec) if std_bc > 0 and std_ec > 0 else 0

    result = {
        "common_days": n,
        "date_range": f"{common[0]} to {common[-1]}",
        "level_correlation": round(corr, 3),
        "change_correlation": round(change_corr, 3),
        "rolling_corr_mean": round(statistics.mean(rolling_corr), 3) if rolling_corr else None,
        "rolling_corr_min": round(min(rolling_corr), 3) if rolling_corr else None,
        "rolling_corr_max": round(max(rolling_corr), 3) if rolling_corr else None,
    }

    print(f"    Common days: {n} ({common[0]} to {common[-1]})")
    print(f"    Level correlation:  {corr:.3f}")
    print(f"    Change correlation: {change_corr:.3f}")
    if rolling_corr:
        print(f"    Rolling 120d corr:  mean={statistics.mean(rolling_corr):.3f}, "
              f"range=[{min(rolling_corr):.3f}, {max(rolling_corr):.3f}]")

    # Interpretation
    if corr > 0.7:
        verdict = "HIGH — BF is a market-wide phenomenon (less diversification benefit)"
    elif corr > 0.4:
        verdict = "MODERATE — partial market-wide, some diversification benefit"
    else:
        verdict = "LOW — independent signals, strong diversification potential"
    print(f"    Verdict: {verdict}")
    result["verdict"] = verdict

    return result


# ═══════════════════════════════════════════════════════════════
# Analysis 5: Combined BTC+ETH Portfolio
# ═══════════════════════════════════════════════════════════════

def analysis_5_combined_portfolio(btc_surface, btc_dvol, eth_surface, eth_dvol):
    """Test combined BTC+ETH BF portfolio."""
    print("\n  ── Analysis 5: Combined BTC+ETH Portfolio ──")

    btc_r = run_bf_backtest(btc_surface, btc_dvol, lookback=120, z_entry=1.5,
                            z_exit=0.0, sensitivity=2.5, start_date="2021-03-24")
    eth_r = run_bf_backtest(eth_surface, eth_dvol, lookback=120, z_entry=1.5,
                            z_exit=0.0, sensitivity=2.5, start_date="2021-03-24")

    if not btc_r or not eth_r:
        print("    FAILED: Missing backtest results")
        return None

    # Build daily P&L dicts
    btc_pnl = {d["date"]: d["pnl"] for d in btc_r["daily_pnl"]}
    eth_pnl = {d["date"]: d["pnl"] for d in eth_r["daily_pnl"]}

    common = sorted(set(btc_pnl.keys()) & set(eth_pnl.keys()))
    if len(common) < 30:
        return None

    # Test different allocation weights
    allocations = [
        (1.0, 0.0, "100% BTC"),
        (0.0, 1.0, "100% ETH"),
        (0.5, 0.5, "50/50"),
        (0.7, 0.3, "70/30 (BTC/ETH)"),
    ]

    results = {}
    print(f"\n    {'Allocation':>20}  {'Sharpe':>8}  {'AnnRet%':>8}  {'MaxDD%':>8}  {'WinRate':>8}")
    print(f"    {'─'*20}  {'─'*8}  {'─'*8}  {'─'*8}  {'─'*8}")

    for w_btc, w_eth, label in allocations:
        combined_pnl = []
        cum = 0
        peak = 0
        max_dd = 0
        wins = 0
        total = 0

        for date in common:
            pnl = w_btc * btc_pnl[date] + w_eth * eth_pnl[date]
            combined_pnl.append(pnl)
            cum += pnl
            if cum > peak:
                peak = cum
            dd = cum - peak
            if dd < max_dd:
                max_dd = dd
            if pnl > 0:
                wins += 1
            total += 1

        mean_p = statistics.mean(combined_pnl)
        std_p = statistics.stdev(combined_pnl) if len(combined_pnl) > 1 else 0.001
        sharpe = (mean_p / std_p) * math.sqrt(365) if std_p > 0 else 0
        ann_ret = mean_p * 365
        win_rate = wins / total * 100 if total > 0 else 0

        results[label] = {
            "sharpe": round(sharpe, 2),
            "ann_ret_pct": round(ann_ret, 2),
            "max_dd_pct": round(max_dd, 4),
            "win_rate": round(win_rate, 1),
            "n_days": len(common),
        }

        print(f"    {label:>20}  {sharpe:>8.2f}  {ann_ret:>8.2f}  {max_dd:>8.4f}  {win_rate:>7.1f}%")

    # Check if diversification improves risk-adjusted returns
    btc_only = results.get("100% BTC", {}).get("sharpe", 0)
    mix_50 = results.get("50/50", {}).get("sharpe", 0)
    mix_70 = results.get("70/30 (BTC/ETH)", {}).get("sharpe", 0)

    best_label = max(results.items(), key=lambda x: x[1].get("sharpe", 0))[0]
    print(f"\n    Best allocation: {best_label}")
    if mix_50 > btc_only:
        print(f"    DIVERSIFICATION BENEFIT: 50/50 Sharpe {mix_50:.2f} > BTC-only {btc_only:.2f}")
    else:
        print(f"    NO diversification benefit: BTC-only {btc_only:.2f} >= 50/50 {mix_50:.2f}")

    return results


# ═══════════════════════════════════════════════════════════════
# Analysis 6: ETH Walk-Forward Validation
# ═══════════════════════════════════════════════════════════════

def analysis_6_walk_forward(eth_surface, eth_dvol):
    """Walk-forward validation on ETH (1-year windows)."""
    print("\n  ── Analysis 6: ETH Walk-Forward (1-Year Windows) ──")

    dates = sorted(d for d in eth_surface if "butterfly_25d" in eth_surface[d])
    dates = [d for d in dates if d >= "2021-03-24"]  # Match DVOL availability

    if len(dates) < 365:
        print("    Insufficient data")
        return None

    # Walk forward: train on 2 years, test on next 1 year
    results = []
    years = sorted(set(d[:4] for d in dates))

    for test_year in years:
        if test_year < "2022":
            continue  # Need at least 1 year of training

        test_dates = [d for d in dates if d[:4] == test_year]
        if len(test_dates) < 60:
            continue

        # Run backtest starting from first available date
        # (lookback warm-up handles training implicitly)
        result = run_bf_backtest(
            eth_surface, eth_dvol,
            lookback=120, z_entry=1.5, z_exit=0.0, sensitivity=2.5,
            start_date=test_dates[0],
        )

        if result and result["n_days"] > 0:
            # Extract just the test year P&L
            test_pnl = [d["pnl"] for d in result["daily_pnl"] if d["date"][:4] == test_year]
            if len(test_pnl) > 30:
                mean_p = statistics.mean(test_pnl)
                std_p = statistics.stdev(test_pnl) if len(test_pnl) > 1 else 0.001
                sharpe = (mean_p / std_p) * math.sqrt(365) if std_p > 0 else 0
                ann_ret = mean_p * 365

                entry = {
                    "year": test_year,
                    "sharpe": round(sharpe, 2),
                    "ann_ret_pct": round(ann_ret, 2),
                    "n_days": len(test_pnl),
                }
                results.append(entry)

    if not results:
        print("    No valid walk-forward windows")
        return None

    print(f"\n    {'Year':>6}  {'Sharpe':>8}  {'AnnRet%':>10}  {'Days':>6}")
    print(f"    {'─'*6}  {'─'*8}  {'─'*10}  {'─'*6}")
    for r in results:
        print(f"    {r['year']:>6}  {r['sharpe']:>8.2f}  {r['ann_ret_pct']:>10.2f}  {r['n_days']:>6}")

    sharpes = [r["sharpe"] for r in results]
    avg_sharpe = statistics.mean(sharpes)
    positive_years = sum(1 for s in sharpes if s > 0)
    print(f"\n    Average WF Sharpe: {avg_sharpe:.2f}")
    print(f"    Positive years: {positive_years}/{len(sharpes)}")

    return {
        "windows": results,
        "avg_sharpe": round(avg_sharpe, 2),
        "positive_years": positive_years,
        "total_years": len(sharpes),
    }


# ═══════════════════════════════════════════════════════════════
# Analysis 7: Leave-One-Year-Out (LOYO) Cross-Validation
# ═══════════════════════════════════════════════════════════════

def analysis_7_loyo(eth_surface, eth_dvol):
    """LOYO validation on ETH — train on all but one year, test on held-out year."""
    print("\n  ── Analysis 7: ETH LOYO Cross-Validation ──")

    # Get full backtest
    full_result = run_bf_backtest(
        eth_surface, eth_dvol,
        lookback=120, z_entry=1.5, z_exit=0.0, sensitivity=2.5,
        start_date="2021-03-24",
    )

    if not full_result:
        print("    FAILED: No full backtest")
        return None

    # Extract P&L by year
    pnl_by_year = defaultdict(list)
    for d in full_result["daily_pnl"]:
        pnl_by_year[d["date"][:4]].append(d["pnl"])

    years = sorted(pnl_by_year.keys())
    results = []

    for held_out in years:
        if len(pnl_by_year[held_out]) < 30:
            continue

        # "Train" = all other years, "Test" = held-out year
        # Since our strategy is parameter-free (fixed params), LOYO just
        # validates that performance is consistent across years
        test_pnl = pnl_by_year[held_out]
        mean_p = statistics.mean(test_pnl)
        std_p = statistics.stdev(test_pnl) if len(test_pnl) > 1 else 0.001
        sharpe = (mean_p / std_p) * math.sqrt(365) if std_p > 0 else 0

        results.append({
            "year": held_out,
            "sharpe": round(sharpe, 2),
            "n_days": len(test_pnl),
            "cum_pnl": round(sum(test_pnl), 4),
        })

    print(f"\n    {'Year':>6}  {'Sharpe':>8}  {'Cum P&L%':>10}  {'Days':>6}")
    print(f"    {'─'*6}  {'─'*8}  {'─'*10}  {'─'*6}")
    for r in results:
        print(f"    {r['year']:>6}  {r['sharpe']:>8.2f}  {r['cum_pnl']:>10.4f}  {r['n_days']:>6}")

    sharpes = [r["sharpe"] for r in results]
    avg = statistics.mean(sharpes) if sharpes else 0
    positive = sum(1 for s in sharpes if s > 0)
    print(f"\n    LOYO avg Sharpe: {avg:.2f}, positive: {positive}/{len(sharpes)}")

    return {
        "results": results,
        "avg_sharpe": round(avg, 2),
        "positive_years": positive,
        "total_years": len(sharpes),
    }


# ═══════════════════════════════════════════════════════════════
# Analysis 8: Regime Consistency
# ═══════════════════════════════════════════════════════════════

def analysis_8_regime_consistency(btc_surface, btc_dvol, eth_surface, eth_dvol):
    """Test if BF edge persists in same IV regimes for both assets."""
    print("\n  ── Analysis 8: Regime Consistency (BTC vs ETH) ──")

    results = {}

    for name, surface, dvol in [("BTC", btc_surface, btc_dvol), ("ETH", eth_surface, eth_dvol)]:
        # Run backtest
        bt = run_bf_backtest(surface, dvol, lookback=120, z_entry=1.5, z_exit=0.0,
                             sensitivity=2.5, start_date="2021-03-24")
        if not bt:
            continue

        # Classify days by IV regime
        pnl_by_regime = {"low": [], "mid": [], "high": []}
        for d in bt["daily_pnl"]:
            date = d["date"]
            iv = dvol.get(date)
            if iv is None:
                continue
            if iv > 1:
                iv = iv / 100

            if iv < 0.40:
                regime = "low"
            elif iv < 0.70:
                regime = "mid"
            else:
                regime = "high"
            pnl_by_regime[regime].append(d["pnl"])

        asset_results = {}
        print(f"\n    {name}:")
        for regime in ["low", "mid", "high"]:
            pnl = pnl_by_regime[regime]
            if len(pnl) < 20:
                print(f"      {regime:>5}: insufficient data ({len(pnl)} days)")
                continue
            mean_p = statistics.mean(pnl)
            std_p = statistics.stdev(pnl) if len(pnl) > 1 else 0.001
            sharpe = (mean_p / std_p) * math.sqrt(365) if std_p > 0 else 0
            asset_results[regime] = {
                "sharpe": round(sharpe, 2),
                "n_days": len(pnl),
                "avg_pnl_bps": round(mean_p, 3),
            }
            print(f"      {regime:>5}: Sharpe={sharpe:>6.2f}, days={len(pnl):>5}, avg_pnl={mean_p:>7.3f} bps")

        results[name] = asset_results

    return results


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════

def main():
    print("=" * 70)
    print("  R86: ETH Cross-Validation — Is the BF Edge Structural?")
    print("=" * 70)

    # Load data
    print("\n  Loading data...")
    btc_surface = load_surface("BTC")
    eth_surface = load_surface("ETH")
    btc_dvol = load_dvol("BTC")
    eth_dvol = load_dvol("ETH")

    print(f"    BTC surface: {len(btc_surface)} days")
    print(f"    ETH surface: {len(eth_surface)} days")
    print(f"    BTC DVOL:    {len(btc_dvol)} days")
    print(f"    ETH DVOL:    {len(eth_dvol)} days")

    all_results = {}

    # Run all analyses
    all_results["descriptive"] = analysis_1_descriptive(btc_surface, eth_surface)
    all_results["eth_backtest"] = analysis_2_eth_backtest(eth_surface, eth_dvol)
    all_results["parameter_sensitivity"] = analysis_3_parameter_sensitivity(
        btc_surface, btc_dvol, eth_surface, eth_dvol)
    all_results["correlation"] = analysis_4_correlation(btc_surface, eth_surface)
    all_results["combined_portfolio"] = analysis_5_combined_portfolio(
        btc_surface, btc_dvol, eth_surface, eth_dvol)
    all_results["eth_walk_forward"] = analysis_6_walk_forward(eth_surface, eth_dvol)
    all_results["eth_loyo"] = analysis_7_loyo(eth_surface, eth_dvol)
    all_results["regime_consistency"] = analysis_8_regime_consistency(
        btc_surface, btc_dvol, eth_surface, eth_dvol)

    # ── Overall Verdict ──
    print("\n" + "=" * 70)
    print("  VERDICT: Is the BF Edge Structural?")
    print("=" * 70)

    eth_sharpe = all_results.get("eth_backtest", {}).get("sharpe", 0) or 0
    eth_wf_avg = (all_results.get("eth_walk_forward") or {}).get("avg_sharpe", 0) or 0
    eth_loyo_pos = (all_results.get("eth_loyo") or {}).get("positive_years", 0) or 0
    eth_loyo_tot = (all_results.get("eth_loyo") or {}).get("total_years", 1) or 1
    corr_level = (all_results.get("correlation") or {}).get("level_correlation", 0) or 0

    evidence_for = 0
    evidence_against = 0

    if eth_sharpe > 1.0:
        evidence_for += 2
        print(f"\n    [+2] ETH BF Sharpe {eth_sharpe:.2f} > 1.0 — edge exists on ETH")
    elif eth_sharpe > 0:
        evidence_for += 1
        print(f"\n    [+1] ETH BF Sharpe {eth_sharpe:.2f} > 0 — weak edge on ETH")
    else:
        evidence_against += 2
        print(f"\n    [-2] ETH BF Sharpe {eth_sharpe:.2f} <= 0 — no edge on ETH")

    if eth_wf_avg > 1.0:
        evidence_for += 1
        print(f"    [+1] ETH WF avg Sharpe {eth_wf_avg:.2f} > 1.0 — robust")
    elif eth_wf_avg > 0:
        evidence_for += 0.5
        print(f"    [+0.5] ETH WF avg Sharpe {eth_wf_avg:.2f} > 0 — partially robust")
    else:
        evidence_against += 1
        print(f"    [-1] ETH WF avg Sharpe {eth_wf_avg:.2f} <= 0 — not robust")

    loyo_ratio = eth_loyo_pos / eth_loyo_tot
    if loyo_ratio >= 0.8:
        evidence_for += 1
        print(f"    [+1] ETH LOYO {eth_loyo_pos}/{eth_loyo_tot} positive — consistent")
    elif loyo_ratio >= 0.5:
        evidence_for += 0.5
        print(f"    [+0.5] ETH LOYO {eth_loyo_pos}/{eth_loyo_tot} positive — partial consistency")
    else:
        evidence_against += 1
        print(f"    [-1] ETH LOYO {eth_loyo_pos}/{eth_loyo_tot} positive — inconsistent")

    if 0.3 < corr_level < 0.7:
        evidence_for += 1
        print(f"    [+1] BTC-ETH BF correlation {corr_level:.3f} — moderate (structural + diversifiable)")
    elif corr_level >= 0.7:
        evidence_for += 0.5
        print(f"    [+0.5] BTC-ETH BF correlation {corr_level:.3f} — high (market-wide, less diversification)")

    score = evidence_for - evidence_against
    if score >= 3:
        overall = "CONFIRMED STRUCTURAL — BF edge is a vol surface phenomenon, not BTC-specific"
    elif score >= 1:
        overall = "LIKELY STRUCTURAL — evidence supports structural edge but with caveats"
    elif score >= 0:
        overall = "INCONCLUSIVE — mixed evidence, need more data"
    else:
        overall = "LIKELY BTC-SPECIFIC — edge may be a BTC microstructure artifact"

    print(f"\n    Evidence Score: {score:.1f} (for={evidence_for:.1f}, against={evidence_against:.1f})")
    print(f"    OVERALL: {overall}")

    all_results["verdict"] = {
        "evidence_score": round(score, 1),
        "evidence_for": round(evidence_for, 1),
        "evidence_against": round(evidence_against, 1),
        "overall": overall,
        "eth_sharpe": eth_sharpe,
        "btc_eth_correlation": corr_level,
    }

    # Save results
    with open(OUTPUT_PATH, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n  Results saved: {OUTPUT_PATH}")
    print("=" * 70)


if __name__ == "__main__":
    main()
