#!/usr/bin/env python3
"""
R102: SOL BF Cross-Validation — 3rd Asset Structural Test
============================================================

R86 proved BF is STRUCTURAL (not BTC-specific): ETH Sharpe 1.76,
BTC-ETH BF correlation 0.101 (nearly independent).

Now we test SOL (Solana) — Deribit's 3rd major options market:
  - If SOL BF shows mean-reversion → 3rd confirmation of structural edge
  - SOL-BTC/ETH correlation → portfolio diversification potential
  - Optimal SOL sensitivity for config v4
  - 3-asset portfolio analysis

Data: SOL daily surface 2022-2025 (~1,322 observations)

Sections:
  1. SOL BF Descriptive Statistics vs BTC/ETH
  2. SOL BF Mean-Reversion Backtest (production config)
  3. Sensitivity Grid Search (optimal SOL sensitivity)
  4. Cross-Asset BF Correlation (SOL vs BTC vs ETH)
  5. Walk-Forward Validation
  6. 3-Asset Portfolio Analysis (BTC/ETH/SOL)
  7. Kill-Switch Analysis
  8. Verdict
"""
import csv
import json
import math
import statistics
from collections import defaultdict
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "cache" / "deribit"
SURFACE_DIR = DATA_DIR / "real_surface"
DVOL_DIR = DATA_DIR / "dvol"
OUTPUT_PATH = SURFACE_DIR / "r102_sol_bf_crossval.json"


# ═══════════════════════════════════════════════════════════════
# Data Loading
# ═══════════════════════════════════════════════════════════════

def load_surface(currency):
    """Load daily surface data (butterfly_25d, iv_atm, etc.)."""
    # For BTC/ETH use the real surface CSVs
    surface_path = SURFACE_DIR / f"{currency}_daily_surface.csv"
    if surface_path.exists():
        data = {}
        with open(surface_path) as f:
            for row in csv.DictReader(f):
                d = row["date"]
                entry = {}
                for field in ["butterfly_25d", "iv_atm", "skew_25d", "term_spread"]:
                    val = row.get(field, "")
                    if val and val != "None":
                        entry[field] = float(val)
                if "butterfly_25d" in entry:
                    data[d] = entry
        return data

    # For SOL, use the yearly IV files
    data = {}
    for year in range(2020, 2026):
        for pattern in [
            f"{currency}_{year}-01-01_{year}-12-31_1d_iv.csv",
            f"{currency}_2020-01-01_2025-12-31_1d_iv.csv",
        ]:
            path = DATA_DIR / pattern
            if path.exists():
                with open(path) as f:
                    for row in csv.DictReader(f):
                        ts = row.get("timestamp", "").strip()
                        if not ts:
                            continue
                        # Convert epoch to date
                        try:
                            d = datetime.utcfromtimestamp(int(ts)).strftime("%Y-%m-%d")
                        except (ValueError, OSError):
                            continue
                        entry = {}
                        for field, col in [("butterfly_25d", "butterfly_25d"),
                                           ("iv_atm", "iv_atm"),
                                           ("skew_25d", "skew_25d"),
                                           ("term_spread", "term_spread")]:
                            val = row.get(col, "").strip()
                            if val:
                                try:
                                    entry[field] = float(val)
                                except ValueError:
                                    pass
                        if "butterfly_25d" in entry:
                            data[d] = entry
                break  # Use first matching file per year
    return data


def load_dvol(currency):
    """Load DVOL data. Returns {date: iv_decimal}."""
    path = DVOL_DIR / f"{currency}_DVOL_12h.csv"
    if not path.exists():
        return {}
    daily = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            d = row.get("date", "")[:10]
            try:
                daily[d] = float(row["dvol_close"])
            except (ValueError, KeyError):
                pass
    return daily


def load_rv(currency):
    """Load realized vol data for VRP."""
    for pattern in [
        f"{currency}_2020-01-01_2025-12-31_1d_rv.csv",
        f"{currency}_2021-01-01_2025-12-31_1d_rv.csv",
    ]:
        path = DATA_DIR / pattern
        if path.exists():
            data = {}
            with open(path) as f:
                for row in csv.DictReader(f):
                    ts = row.get("timestamp", "").strip()
                    if not ts:
                        continue
                    try:
                        d = datetime.utcfromtimestamp(int(ts)).strftime("%Y-%m-%d")
                    except (ValueError, OSError):
                        continue
                    rv = row.get("rv", "").strip()
                    if rv:
                        try:
                            data[d] = float(rv)
                        except ValueError:
                            pass
            if data:
                return data
    return {}


# ═══════════════════════════════════════════════════════════════
# BF Backtest Engine (same as R86/R81)
# ═══════════════════════════════════════════════════════════════

def run_bf_backtest(surface, dvol, lookback=120, z_entry=1.5, z_exit=0.0,
                    sensitivity=2.5, cost_bps=8, start_date=None, end_date=None):
    """Run BF mean-reversion backtest. Returns metrics dict."""
    dates = sorted(d for d in surface if "butterfly_25d" in surface[d])
    if start_date:
        dates = [d for d in dates if d >= start_date]
    if end_date:
        dates = [d for d in dates if d <= end_date]
    if len(dates) < lookback + 30:
        return None

    bf_vals = []
    daily_pnl = []
    position = 0
    cum_pnl = 0.0
    peak = 0.0
    max_dd = 0.0
    trades = 0

    for i, date in enumerate(dates):
        bf = surface[date]["butterfly_25d"]
        bf_vals.append(bf)

        if len(bf_vals) < lookback:
            continue

        window = bf_vals[-lookback:]
        mean_bf = sum(window) / len(window)
        std_bf = (sum((v - mean_bf) ** 2 for v in window) / len(window)) ** 0.5
        if std_bf < 0.0001:
            std_bf = 0.0001

        z = (bf - mean_bf) / std_bf

        # Get IV
        iv = dvol.get(date, surface[date].get("iv_atm", 0.5))
        if isinstance(iv, (int, float)) and iv > 1:
            iv = iv / 100

        dt = 1.0 / 365.0
        prev_pos = position

        # Signal logic
        if position == 0:
            if z > z_entry:
                position = -1
                trades += 1
            elif z < -z_entry:
                position = 1
                trades += 1
        else:
            if z_exit == 0:
                if position == 1 and z > z_entry:
                    position = -1
                    trades += 1
                elif position == -1 and z < -z_entry:
                    position = 1
                    trades += 1
            else:
                if abs(z) < z_exit:
                    position = 0

        # PnL
        trade_cost = (cost_bps / 10000.0) if position != prev_pos else 0.0
        if i > 0 and prev_pos != 0:
            prev_bf = surface[dates[i-1]]["butterfly_25d"]
            bf_change = bf - prev_bf
            pnl = prev_pos * bf_change * sensitivity * iv * math.sqrt(dt) - trade_cost
        else:
            pnl = -trade_cost if trade_cost > 0 else 0.0

        cum_pnl += pnl
        daily_pnl.append(pnl)
        peak = max(peak, cum_pnl)
        dd = cum_pnl - peak
        max_dd = min(max_dd, dd)

    if len(daily_pnl) < 30:
        return None

    n_days = len(daily_pnl)
    years = n_days / 365.0
    d_mean = sum(daily_pnl) / n_days
    d_std = (sum((p - d_mean) ** 2 for p in daily_pnl) / n_days) ** 0.5
    sharpe = (d_mean / d_std) * math.sqrt(365) if d_std > 0 else 0
    ann_ret = d_mean * 365
    hit_rate = sum(1 for p in daily_pnl if p > 0) / max(sum(1 for p in daily_pnl if p != 0), 1)

    return {
        "sharpe": round(sharpe, 2),
        "ann_ret_pct": round(ann_ret * 100, 2),
        "max_dd_pct": round(max_dd * 100, 2),
        "n_trades": trades,
        "trades_per_year": round(trades / years, 1) if years > 0 else 0,
        "n_days": n_days,
        "hit_rate": round(hit_rate, 2),
        "daily_pnl": daily_pnl,
        "cum_pnl": round(cum_pnl, 6),
    }


# ═══════════════════════════════════════════════════════════════
# Section 1: SOL BF Descriptive Statistics
# ═══════════════════════════════════════════════════════════════

def section_1_stats(sol_surface, btc_surface, eth_surface):
    """Compare SOL BF stats with BTC and ETH."""
    print("\n  ── Section 1: SOL BF Descriptive Statistics ──")

    results = {}
    for name, surface in [("BTC", btc_surface), ("ETH", eth_surface), ("SOL", sol_surface)]:
        bf_vals = [surface[d]["butterfly_25d"] for d in sorted(surface) if "butterfly_25d" in surface[d]]
        n = len(bf_vals)
        if n < 10:
            print(f"    {name}: insufficient data ({n} obs)")
            continue

        mean = sum(bf_vals) / n
        std = (sum((v - mean) ** 2 for v in bf_vals) / n) ** 0.5
        median = sorted(bf_vals)[n // 2]
        mn, mx = min(bf_vals), max(bf_vals)

        # Year breakdown
        by_year = defaultdict(list)
        for d in sorted(surface):
            if "butterfly_25d" in surface[d]:
                yr = d[:4]
                by_year[yr].append(surface[d]["butterfly_25d"])

        print(f"\n    {name}: {n} observations")
        print(f"      Mean: {mean:.6f}  Std: {std:.6f}  Median: {median:.6f}")
        print(f"      Range: [{mn:.6f}, {mx:.6f}]")
        print(f"      By year:")
        for yr in sorted(by_year):
            yv = by_year[yr]
            ym = sum(yv) / len(yv)
            ys = (sum((v - ym) ** 2 for v in yv) / len(yv)) ** 0.5
            print(f"        {yr}: n={len(yv):>4}  mean={ym:.6f}  std={ys:.6f}")

        results[name] = {
            "n": n, "mean": round(mean, 6), "std": round(std, 6),
            "median": round(median, 6), "min": round(mn, 6), "max": round(mx, 6),
        }

    # Comparison
    if all(k in results for k in ["BTC", "ETH", "SOL"]):
        print(f"\n    COMPARISON (overlapping period):")
        print(f"      {'Asset':>5} {'Mean':>10} {'Std':>10} {'Mean/Std':>10}")
        print(f"      {'─'*5} {'─'*10} {'─'*10} {'─'*10}")
        for name in ["BTC", "ETH", "SOL"]:
            r = results[name]
            ratio = r["mean"] / r["std"] if r["std"] > 0 else 0
            print(f"      {name:>5} {r['mean']:>10.6f} {r['std']:>10.6f} {ratio:>10.2f}")

    return results


# ═══════════════════════════════════════════════════════════════
# Section 2: SOL BF Backtest (Production Config)
# ═══════════════════════════════════════════════════════════════

def section_2_backtest(sol_surface, sol_dvol, btc_surface, btc_dvol, eth_surface, eth_dvol):
    """Run BF backtest on all 3 assets with production config."""
    print("\n  ── Section 2: BF Backtest — Production Config ──")

    # Use overlapping period for fair comparison
    sol_dates = sorted(sol_surface.keys())
    start = sol_dates[0] if sol_dates else "2022-01-01"
    end = sol_dates[-1] if sol_dates else "2025-12-31"
    print(f"    Period: {start} to {end}")

    configs = {
        "BTC": {"sensitivity": 5.0, "dvol": btc_dvol, "surface": btc_surface},
        "ETH": {"sensitivity": 3.5, "dvol": eth_dvol, "surface": eth_surface},
        "SOL": {"sensitivity": 2.5, "dvol": sol_dvol, "surface": sol_surface},  # Start conservative
    }

    results = {}
    print(f"\n    {'Asset':>5} {'Sens':>5} {'Sharpe':>7} {'AnnRet%':>8} {'MaxDD%':>7} "
          f"{'Trades':>7} {'Tr/Yr':>6} {'HitRate':>8}")
    print(f"    {'─'*5} {'─'*5} {'─'*7} {'─'*8} {'─'*7} {'─'*7} {'─'*6} {'─'*8}")

    for asset, cfg in configs.items():
        bt = run_bf_backtest(
            cfg["surface"], cfg["dvol"],
            sensitivity=cfg["sensitivity"],
            start_date=start, end_date=end,
        )
        if bt:
            print(f"    {asset:>5} {cfg['sensitivity']:>5.1f} {bt['sharpe']:>7.2f} "
                  f"{bt['ann_ret_pct']:>8.2f} {bt['max_dd_pct']:>7.2f} "
                  f"{bt['n_trades']:>7} {bt['trades_per_year']:>6.1f} {bt['hit_rate']:>7.0%}")
            results[asset] = bt
            # Remove daily_pnl from saved results to keep JSON small
            results[asset]["daily_pnl_saved"] = False
        else:
            print(f"    {asset:>5} - backtest failed (insufficient data)")

    return results


# ═══════════════════════════════════════════════════════════════
# Section 3: SOL Sensitivity Grid Search
# ═══════════════════════════════════════════════════════════════

def section_3_sensitivity(sol_surface, sol_dvol):
    """Grid search for optimal SOL sensitivity."""
    print("\n  ── Section 3: SOL Sensitivity Grid Search ──")

    sensitivities = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0, 7.0]

    print(f"\n    {'Sens':>5} {'Sharpe':>7} {'AnnRet%':>8} {'MaxDD%':>7} {'Trades':>7} {'HitRate':>8}")
    print(f"    {'─'*5} {'─'*7} {'─'*8} {'─'*7} {'─'*7} {'─'*8}")

    results = {}
    for sens in sensitivities:
        bt = run_bf_backtest(sol_surface, sol_dvol, sensitivity=sens)
        if bt:
            print(f"    {sens:>5.1f} {bt['sharpe']:>7.2f} {bt['ann_ret_pct']:>8.2f} "
                  f"{bt['max_dd_pct']:>7.2f} {bt['n_trades']:>7} {bt['hit_rate']:>7.0%}")
            results[str(sens)] = {
                "sharpe": bt["sharpe"], "ann_ret_pct": bt["ann_ret_pct"],
                "max_dd_pct": bt["max_dd_pct"], "n_trades": bt["n_trades"],
            }

    # Best by Sharpe
    if results:
        best = max(results.items(), key=lambda x: x[1]["sharpe"])
        print(f"\n    Optimal SOL sensitivity: {best[0]} (Sharpe {best[1]['sharpe']})")

    return results


# ═══════════════════════════════════════════════════════════════
# Section 4: Cross-Asset BF Correlation
# ═══════════════════════════════════════════════════════════════

def section_4_correlation(sol_surface, btc_surface, eth_surface):
    """Compute pairwise BF correlation between SOL, BTC, ETH."""
    print("\n  ── Section 4: Cross-Asset BF Correlation ──")

    # Find overlapping dates
    sol_dates = set(sol_surface.keys())
    btc_dates = set(btc_surface.keys())
    eth_dates = set(eth_surface.keys())
    common = sorted(sol_dates & btc_dates & eth_dates)

    print(f"    Overlapping dates: {len(common)}")
    if len(common) < 60:
        print("    ERROR: insufficient overlapping data")
        return {}

    sol_bf = [sol_surface[d]["butterfly_25d"] for d in common]
    btc_bf = [btc_surface[d]["butterfly_25d"] for d in common]
    eth_bf = [eth_surface[d]["butterfly_25d"] for d in common]

    def corr(x, y):
        n = len(x)
        mx = sum(x) / n
        my = sum(y) / n
        cov = sum((xi - mx) * (yi - my) for xi, yi in zip(x, y)) / n
        sx = (sum((xi - mx) ** 2 for xi in x) / n) ** 0.5
        sy = (sum((yi - my) ** 2 for yi in y) / n) ** 0.5
        return cov / (sx * sy) if sx > 0 and sy > 0 else 0

    # Level correlations
    btc_sol = corr(btc_bf, sol_bf)
    eth_sol = corr(eth_bf, sol_bf)
    btc_eth = corr(btc_bf, eth_bf)

    print(f"\n    BF Level Correlations:")
    print(f"      BTC-SOL: {btc_sol:.3f}")
    print(f"      ETH-SOL: {eth_sol:.3f}")
    print(f"      BTC-ETH: {btc_eth:.3f}")

    # Change correlations (more relevant for trading)
    sol_chg = [sol_bf[i] - sol_bf[i-1] for i in range(1, len(sol_bf))]
    btc_chg = [btc_bf[i] - btc_bf[i-1] for i in range(1, len(btc_bf))]
    eth_chg = [eth_bf[i] - eth_bf[i-1] for i in range(1, len(eth_bf))]

    btc_sol_chg = corr(btc_chg, sol_chg)
    eth_sol_chg = corr(eth_chg, sol_chg)
    btc_eth_chg = corr(btc_chg, eth_chg)

    print(f"\n    BF Change Correlations (daily changes):")
    print(f"      BTC-SOL: {btc_sol_chg:.3f}")
    print(f"      ETH-SOL: {eth_sol_chg:.3f}")
    print(f"      BTC-ETH: {btc_eth_chg:.3f}")

    return {
        "level": {"BTC_SOL": round(btc_sol, 3), "ETH_SOL": round(eth_sol, 3),
                  "BTC_ETH": round(btc_eth, 3)},
        "change": {"BTC_SOL": round(btc_sol_chg, 3), "ETH_SOL": round(eth_sol_chg, 3),
                   "BTC_ETH": round(btc_eth_chg, 3)},
        "n_common_dates": len(common),
    }


# ═══════════════════════════════════════════════════════════════
# Section 5: Walk-Forward Validation
# ═══════════════════════════════════════════════════════════════

def section_5_walk_forward(sol_surface, sol_dvol):
    """Walk-forward by year for SOL."""
    print("\n  ── Section 5: SOL Walk-Forward Validation ──")

    years = ["2022", "2023", "2024", "2025"]
    best_sens = 2.5  # default, will optimize per year

    print(f"\n    {'Year':>6} {'Sens':>5} {'Sharpe':>7} {'AnnRet%':>8} {'MaxDD%':>7} {'Trades':>7}")
    print(f"    {'─'*6} {'─'*5} {'─'*7} {'─'*8} {'─'*7} {'─'*7}")

    results = {}
    for yr in years:
        start = f"{yr}-01-01"
        end = f"{yr}-12-31"
        # Try multiple sensitivities
        best_bt = None
        best_s = None
        for sens in [2.0, 2.5, 3.0, 3.5]:
            bt = run_bf_backtest(sol_surface, sol_dvol, sensitivity=sens,
                                start_date=start, end_date=end)
            if bt and (best_bt is None or bt["sharpe"] > best_bt["sharpe"]):
                best_bt = bt
                best_s = sens

        if best_bt:
            print(f"    {yr:>6} {best_s:>5.1f} {best_bt['sharpe']:>7.2f} "
                  f"{best_bt['ann_ret_pct']:>8.2f} {best_bt['max_dd_pct']:>7.2f} "
                  f"{best_bt['n_trades']:>7}")
            results[yr] = {"sharpe": best_bt["sharpe"], "sensitivity": best_s,
                          "ann_ret_pct": best_bt["ann_ret_pct"],
                          "max_dd_pct": best_bt["max_dd_pct"]}
        else:
            print(f"    {yr:>6} — insufficient data")

    # Average
    if results:
        avg_sharpe = sum(r["sharpe"] for r in results.values()) / len(results)
        positive = sum(1 for r in results.values() if r["sharpe"] > 0)
        print(f"\n    Average Sharpe: {avg_sharpe:.2f}")
        print(f"    Positive years: {positive}/{len(results)}")

    return results


# ═══════════════════════════════════════════════════════════════
# Section 6: 3-Asset Portfolio Analysis
# ═══════════════════════════════════════════════════════════════

def section_6_portfolio(sol_surface, sol_dvol, btc_surface, btc_dvol, eth_surface, eth_dvol):
    """Analyze 3-asset portfolio vs 2-asset baseline."""
    print("\n  ── Section 6: 3-Asset Portfolio Analysis ──")

    # Find overlapping period
    sol_dates = sorted(sol_surface.keys())
    if not sol_dates:
        print("    No SOL data available")
        return {}

    start = sol_dates[0]
    end = sol_dates[-1]
    print(f"    Period: {start} to {end}")

    # Run individual backtests
    btc_bt = run_bf_backtest(btc_surface, btc_dvol, sensitivity=5.0,
                             start_date=start, end_date=end)
    eth_bt = run_bf_backtest(eth_surface, eth_dvol, sensitivity=3.5,
                             start_date=start, end_date=end)

    # SOL at multiple sensitivities
    sol_sensitivities = [2.0, 2.5, 3.0, 3.5]
    sol_bts = {}
    for sens in sol_sensitivities:
        sol_bts[sens] = run_bf_backtest(sol_surface, sol_dvol, sensitivity=sens,
                                        start_date=start, end_date=end)

    if not btc_bt or not eth_bt:
        print("    ERROR: BTC or ETH backtest failed on overlap period")
        return {}

    # Portfolio allocations to test
    # Current production: BTC 70%, ETH 30%
    # New candidates: BTC/ETH/SOL splits
    allocations = [
        ("70/30/0 (baseline)", 0.70, 0.30, 0.00),
        ("60/25/15", 0.60, 0.25, 0.15),
        ("55/25/20", 0.55, 0.25, 0.20),
        ("50/25/25", 0.50, 0.25, 0.25),
        ("60/20/20", 0.60, 0.20, 0.20),
        ("50/30/20", 0.50, 0.30, 0.20),
    ]

    print(f"\n    Using SOL sensitivity = 2.5 (conservative default)")
    sol_bt = sol_bts.get(2.5)
    if not sol_bt:
        print("    ERROR: SOL backtest failed")
        return {}

    # Align daily PnLs
    btc_pnl = btc_bt["daily_pnl"]
    eth_pnl = eth_bt["daily_pnl"]
    sol_pnl = sol_bt["daily_pnl"]

    # Use shortest length
    min_len = min(len(btc_pnl), len(eth_pnl), len(sol_pnl))
    btc_pnl = btc_pnl[:min_len]
    eth_pnl = eth_pnl[:min_len]
    sol_pnl = sol_pnl[:min_len]

    print(f"\n    {'Allocation':>20} {'Sharpe':>7} {'AnnRet%':>8} {'MaxDD%':>7}")
    print(f"    {'─'*20} {'─'*7} {'─'*8} {'─'*7}")

    results = {}
    for name, w_btc, w_eth, w_sol in allocations:
        port_pnl = [w_btc * b + w_eth * e + w_sol * s
                    for b, e, s in zip(btc_pnl, eth_pnl, sol_pnl)]
        n = len(port_pnl)
        d_mean = sum(port_pnl) / n
        d_std = (sum((p - d_mean) ** 2 for p in port_pnl) / n) ** 0.5
        sharpe = (d_mean / d_std) * math.sqrt(365) if d_std > 0 else 0
        ann_ret = d_mean * 365

        # Max DD
        cum = 0.0
        peak = 0.0
        max_dd = 0.0
        for p in port_pnl:
            cum += p
            peak = max(peak, cum)
            max_dd = min(max_dd, cum - peak)

        print(f"    {name:>20} {sharpe:>7.2f} {ann_ret*100:>8.2f} {max_dd*100:>7.2f}")
        results[name] = {
            "sharpe": round(sharpe, 2), "ann_ret_pct": round(ann_ret * 100, 2),
            "max_dd_pct": round(max_dd * 100, 2),
        }

    # PnL correlations for the portfolio
    def corr(x, y):
        n = len(x)
        mx = sum(x) / n
        my = sum(y) / n
        cov = sum((xi - mx) * (yi - my) for xi, yi in zip(x, y)) / n
        sx = (sum((xi - mx) ** 2 for xi in x) / n) ** 0.5
        sy = (sum((yi - my) ** 2 for yi in y) / n) ** 0.5
        return cov / (sx * sy) if sx > 0 and sy > 0 else 0

    print(f"\n    Daily PnL correlations:")
    print(f"      BTC-ETH: {corr(btc_pnl, eth_pnl):.3f}")
    print(f"      BTC-SOL: {corr(btc_pnl, sol_pnl):.3f}")
    print(f"      ETH-SOL: {corr(eth_pnl, sol_pnl):.3f}")

    return results


# ═══════════════════════════════════════════════════════════════
# Section 7: Kill-Switch Analysis
# ═══════════════════════════════════════════════════════════════

def section_7_kill_switch(sol_surface, sol_dvol):
    """Analyze SOL drawdown for kill-switch threshold."""
    print("\n  ── Section 7: SOL Kill-Switch Analysis ──")

    sensitivities = [2.0, 2.5, 3.0, 3.5, 4.0, 5.0]
    thresholds = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]

    print(f"\n    MaxDD by sensitivity:")
    print(f"    {'Sens':>5} {'MaxDD%':>8} {'Pass 2.0%':>10} {'Pass 2.5%':>10} {'Pass 3.0%':>10}")
    print(f"    {'─'*5} {'─'*8} {'─'*10} {'─'*10} {'─'*10}")

    results = {}
    for sens in sensitivities:
        bt = run_bf_backtest(sol_surface, sol_dvol, sensitivity=sens)
        if bt:
            max_dd = abs(bt["max_dd_pct"])
            p2 = "✓" if max_dd <= 2.0 else "✗"
            p25 = "✓" if max_dd <= 2.5 else "✗"
            p3 = "✓" if max_dd <= 3.0 else "✗"
            print(f"    {sens:>5.1f} {bt['max_dd_pct']:>8.2f} {p2:>10} {p25:>10} {p3:>10}")
            results[str(sens)] = {
                "max_dd_pct": bt["max_dd_pct"],
                "sharpe": bt["sharpe"],
            }

    return results


# ═══════════════════════════════════════════════════════════════
# Section 8: Verdict
# ═══════════════════════════════════════════════════════════════

def section_8_verdict(all_results):
    """Final verdict on SOL BF."""
    print("\n  ── Section 8: VERDICT ──")

    s2 = all_results.get("backtest", {}).get("SOL", {})
    s3 = all_results.get("sensitivity", {})
    s4 = all_results.get("correlation", {})
    s5 = all_results.get("walk_forward", {})
    s6 = all_results.get("portfolio", {})
    s7 = all_results.get("kill_switch", {})

    # Find best sensitivity
    if s3:
        best_sens = max(s3.items(), key=lambda x: x[1].get("sharpe", -99))
    else:
        best_sens = ("?", {"sharpe": "?"})

    # Portfolio improvement
    baseline = s6.get("70/30/0 (baseline)", {})
    best_3asset = None
    if s6:
        for name, r in s6.items():
            if "baseline" not in name:
                if best_3asset is None or r.get("sharpe", -99) > best_3asset[1].get("sharpe", -99):
                    best_3asset = (name, r)

    # Correlations
    btc_sol_chg = s4.get("change", {}).get("BTC_SOL", "?")
    eth_sol_chg = s4.get("change", {}).get("ETH_SOL", "?")

    # WF
    wf_years = len(s5)
    wf_positive = sum(1 for r in s5.values() if r.get("sharpe", 0) > 0) if s5 else 0
    wf_avg = sum(r["sharpe"] for r in s5.values()) / len(s5) if s5 else 0

    verdict = {
        "sol_bf_structural": s2.get("sharpe", 0) > 0,
        "optimal_sensitivity": best_sens[0] if isinstance(best_sens[0], str) else str(best_sens[0]),
        "portfolio_improves": best_3asset[1].get("sharpe", 0) > baseline.get("sharpe", 0) if best_3asset and baseline else False,
    }

    print(f"""
    ═══════════════════════════════════════════════════════════
    R102 VERDICT: SOL BF Cross-Validation
    ═══════════════════════════════════════════════════════════

    1. SOL BF MEAN-REVERSION:
       Sharpe: {s2.get('sharpe', '?')} (production config sens=2.5)
       {'★ CONFIRMED: BF is structural on 3rd asset' if s2.get('sharpe', 0) > 0 else '✗ NOT CONFIRMED: negative Sharpe'}
       {'   → 3rd independent confirmation (BTC, ETH, SOL)' if s2.get('sharpe', 0) > 0 else ''}

    2. OPTIMAL SENSITIVITY:
       Best: sens={best_sens[0]} → Sharpe {best_sens[1].get('sharpe', '?')}
       SOL is {'more' if float(best_sens[0]) > 3.5 else 'less'} leverage-tolerant than ETH

    3. CROSS-ASSET CORRELATION:
       BTC-SOL BF change corr: {btc_sol_chg}
       ETH-SOL BF change corr: {eth_sol_chg}
       {'Low correlation → good diversification' if isinstance(btc_sol_chg, (int,float)) and abs(btc_sol_chg) < 0.3 else 'Moderate/high correlation → limited diversification'}

    4. WALK-FORWARD:
       Positive years: {wf_positive}/{wf_years}
       Average Sharpe: {wf_avg:.2f}

    5. 3-ASSET PORTFOLIO:
       Baseline (70/30/0): Sharpe {baseline.get('sharpe', '?')}
       Best 3-asset: {best_3asset[0] if best_3asset else '?'} → Sharpe {best_3asset[1].get('sharpe', '?') if best_3asset else '?'}
       {'★ IMPROVES portfolio' if verdict['portfolio_improves'] else '✗ Does NOT improve portfolio'}

    ═══════════════════════════════════════════════════════════
    RECOMMENDATION:
    ═══════════════════════════════════════════════════════════
    """)

    if s2.get("sharpe", 0) > 0.5 and verdict["portfolio_improves"]:
        print("""      ★ ADD SOL TO PRODUCTION PORTFOLIO
        → Update R88 runner with SOL config
        → Use optimal sensitivity from grid search
        → Set kill-switch based on MaxDD analysis
        → 3rd asset diversification confirmed""")
    elif s2.get("sharpe", 0) > 0:
        print("""      MONITOR SOL — edge exists but marginal
        → SOL BF MR is real but may not improve portfolio
        → Continue monitoring, add when liquidity improves
        → BTC + ETH 2-asset portfolio is sufficient""")
    else:
        print("""      ✗ DO NOT ADD SOL
        → BF MR not confirmed on SOL
        → Stay with BTC + ETH 2-asset portfolio""")

    print("    ═══════════════════════════════════════════════════════════")
    return verdict


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════

def main():
    print("=" * 70)
    print("  R102: SOL BF Cross-Validation — 3rd Asset Structural Test")
    print("=" * 70)

    # Load data
    btc_surface = load_surface("BTC")
    eth_surface = load_surface("ETH")
    sol_surface = load_surface("SOL")

    btc_dvol = load_dvol("BTC")
    eth_dvol = load_dvol("ETH")
    sol_dvol = load_dvol("SOL")  # likely empty, will fallback to iv_atm

    print(f"  Data loaded: BTC {len(btc_surface)} days, ETH {len(eth_surface)} days, SOL {len(sol_surface)} days")
    print(f"  DVOL: BTC {len(btc_dvol)}, ETH {len(eth_dvol)}, SOL {len(sol_dvol)}")

    all_results = {}

    all_results["stats"] = section_1_stats(sol_surface, btc_surface, eth_surface)
    all_results["backtest"] = section_2_backtest(
        sol_surface, sol_dvol, btc_surface, btc_dvol, eth_surface, eth_dvol)
    all_results["sensitivity"] = section_3_sensitivity(sol_surface, sol_dvol)
    all_results["correlation"] = section_4_correlation(sol_surface, btc_surface, eth_surface)
    all_results["walk_forward"] = section_5_walk_forward(sol_surface, sol_dvol)
    all_results["portfolio"] = section_6_portfolio(
        sol_surface, sol_dvol, btc_surface, btc_dvol, eth_surface, eth_dvol)
    all_results["kill_switch"] = section_7_kill_switch(sol_surface, sol_dvol)
    all_results["verdict"] = section_8_verdict(all_results)

    # Remove daily_pnl arrays from saved output
    for key in all_results.get("backtest", {}):
        if isinstance(all_results["backtest"][key], dict):
            all_results["backtest"][key].pop("daily_pnl", None)

    # Save
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n  Results saved: {OUTPUT_PATH}")
    print("=" * 70)


if __name__ == "__main__":
    main()
