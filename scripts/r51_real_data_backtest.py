#!/usr/bin/env python3
"""
R51: First REAL Sharpe Ratios — VRP + Skew backtest with actual market data
============================================================================

Uses:
  - Real DVOL (R49): ATM IV for BTC/ETH (2021-03 to 2026-02)
  - Real skew_25d (R50): From option trades (2019-03 to 2026-02)
  - Real perpetual prices: For RV computation

PnL Models (from options_engine.py):
  VRP:  0.5 * (IV² - RV_bar²) * dt  (gamma/theta model, short vol)
  Skew: w * d_skew * iv * sqrt(dt) * 2.5  (vega model, mean-reversion)

This is the moment of truth: do the strategies actually work on real data?
"""
import csv
import json
import math
import subprocess
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]


# ═══════════════════════════════════════════════════════════════════
# Data Loading
# ═══════════════════════════════════════════════════════════════════

def load_dvol_daily(currency: str) -> Dict[str, float]:
    """Load DVOL data, aggregate to daily close. Returns {date_str: iv_decimal}."""
    path = ROOT / "data" / "cache" / "deribit" / "dvol" / f"{currency}_DVOL_12h.csv"
    if not path.exists():
        print(f"  WARNING: No DVOL data for {currency}")
        return {}

    daily = {}
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            date = row["date"][:10]  # YYYY-MM-DD
            dvol = float(row["dvol_close"])
            daily[date] = dvol / 100.0  # Convert percentage to decimal

    print(f"  Loaded {len(daily)} daily DVOL points for {currency}")
    return daily


def load_real_skew(currency: str) -> Dict[str, dict]:
    """Load real skew surface data. Returns {date_str: {iv_atm, skew_25d, ...}}."""
    path = ROOT / "data" / "cache" / "deribit" / "real_surface" / f"{currency}_daily_surface.csv"
    if not path.exists():
        print(f"  WARNING: No real surface data for {currency}")
        return {}

    data = {}
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            date = row["date"]
            entry = {}
            for field in ["iv_atm", "skew_25d", "butterfly_25d", "term_spread"]:
                val = row.get(field, "")
                entry[field] = float(val) if val and val != "None" else None
            entry["interpolated"] = row.get("interpolated", "True") == "True"
            data[date] = entry

    print(f"  Loaded {len(data)} daily surface points for {currency}")
    return data


def load_prices(currency: str) -> Dict[str, float]:
    """Load daily close prices from Deribit. Returns {date_str: price}."""
    # Try to use cached DVOL data which has timestamps we can match with price data
    # Fetch fresh prices via API
    url = (f"https://www.deribit.com/api/v2/public/get_tradingview_chart_data?"
           f"instrument_name={currency}-PERPETUAL&resolution=1D")

    prices = {}
    start_dt = datetime(2019, 1, 1, tzinfo=timezone.utc)
    end_dt = datetime(2026, 3, 1, tzinfo=timezone.utc)
    current = start_dt

    while current < end_dt:
        chunk_end = min(current + timedelta(days=365), end_dt)
        start_ms = int(current.timestamp() * 1000)
        end_ms = int(chunk_end.timestamp() * 1000)

        full_url = f"{url}&start_timestamp={start_ms}&end_timestamp={end_ms}"
        try:
            result = subprocess.run(
                ["curl", "-s", "--max-time", "30", full_url],
                capture_output=True, text=True, timeout=40
            )
            data = json.loads(result.stdout)
            if "result" in data:
                data = data["result"]
            if data.get("status") == "ok" and data.get("close"):
                for i, ts in enumerate(data["ticks"]):
                    dt = datetime.fromtimestamp(ts / 1000, tz=timezone.utc)
                    prices[dt.strftime("%Y-%m-%d")] = data["close"][i]
        except Exception as e:
            print(f"    Price fetch error: {e}")

        current = chunk_end
        time.sleep(0.15)

    print(f"  Loaded {len(prices)} daily prices for {currency}")
    return prices


# ═══════════════════════════════════════════════════════════════════
# PnL Models (exact replicas from options_engine.py)
# ═══════════════════════════════════════════════════════════════════

def compute_vrp_pnl_series(
    prices: Dict[str, float],
    dvol: Dict[str, float],
    leverage: float = 1.5,
    rv_lookback: int = 30,
) -> Tuple[List[str], List[float]]:
    """
    VRP PnL: 0.5 * (IV² - RV_bar²) * dt, short vol at given leverage.

    Returns dates and daily return series.
    """
    dates = sorted(set(prices.keys()) & set(dvol.keys()))
    if len(dates) < rv_lookback + 2:
        return [], []

    dt = 1.0 / 365.0
    result_dates = []
    result_returns = []

    for i in range(1, len(dates)):
        date = dates[i]
        prev_date = dates[i - 1]

        price = prices.get(date)
        prev_price = prices.get(prev_date)
        iv = dvol.get(prev_date)  # Use previous day's IV (no lookahead)

        if not all([price, prev_price, iv]) or prev_price <= 0:
            continue

        log_ret = math.log(price / prev_price)

        # Single-bar annualized realized vol
        rv_bar = abs(log_ret) * math.sqrt(365.0)

        # VRP PnL: theta - gamma cost
        vrp_pnl = 0.5 * (iv ** 2 - rv_bar ** 2) * dt

        # Short vol at leverage → positive PnL when IV > RV
        daily_return = leverage * vrp_pnl

        result_dates.append(date)
        result_returns.append(daily_return)

    return result_dates, result_returns


def compute_skew_pnl_series(
    skew_data: Dict[str, dict],
    dvol: Optional[Dict[str, float]] = None,
    leverage: float = 1.0,
    sensitivity_mult: float = 2.5,
    skew_lookback: int = 60,
    z_entry_short: float = 1.0,
    z_entry_long: float = 2.0,
    z_exit: float = 0.0,
) -> Tuple[List[str], List[float]]:
    """
    Skew MR PnL: w * d_skew * iv * sqrt(dt) * sensitivity_mult

    Strategy: Fade extreme skew via z-score signals.
    - Short skew when z > z_entry_short (skew unusually high)
    - Long skew when z < -z_entry_long (skew unusually low)
    - Exit when z crosses z_exit

    Returns dates and daily return series.
    """
    dates = sorted(skew_data.keys())
    if len(dates) < skew_lookback + 2:
        return [], []

    dt = 1.0 / 365.0
    result_dates = []
    result_returns = []

    # Track position
    position = 0.0  # -1 = short skew, +1 = long skew, 0 = flat

    # Rolling window for z-score
    skew_history = []

    for i in range(1, len(dates)):
        date = dates[i]
        prev_date = dates[i - 1]

        entry = skew_data.get(date, {})
        prev_entry = skew_data.get(prev_date, {})

        skew_now = entry.get("skew_25d")
        skew_prev = prev_entry.get("skew_25d")

        # Get IV (prefer DVOL if available, else use surface IV)
        iv = None
        if dvol and prev_date in dvol:
            iv = dvol[prev_date]
        elif prev_entry.get("iv_atm"):
            iv = prev_entry["iv_atm"]

        if skew_now is None or skew_prev is None or iv is None or iv <= 0:
            result_dates.append(date)
            result_returns.append(0.0)
            if skew_now is not None:
                skew_history.append(skew_now)
                if len(skew_history) > skew_lookback:
                    skew_history.pop(0)
            continue

        # Update rolling history
        skew_history.append(skew_now)
        if len(skew_history) > skew_lookback:
            skew_history.pop(0)

        # PnL from current position
        d_skew = skew_now - skew_prev
        sensitivity = iv * math.sqrt(dt) * sensitivity_mult
        daily_return = position * leverage * d_skew * sensitivity

        result_dates.append(date)
        result_returns.append(daily_return)

        # Update position based on z-score
        if len(skew_history) >= skew_lookback:
            mean = sum(skew_history) / len(skew_history)
            var = sum((s - mean) ** 2 for s in skew_history) / len(skew_history)
            std = math.sqrt(var) if var > 0 else 1e-6
            z = (skew_now - mean) / std

            if position == 0:
                if z > z_entry_short:
                    position = -1.0  # Short skew (expect mean reversion down)
                elif z < -z_entry_long:
                    position = 1.0   # Long skew (expect mean reversion up)
            elif position < 0:  # Currently short skew
                if z < z_exit:
                    position = 0.0   # Exit
            elif position > 0:  # Currently long skew
                if z > -z_exit:
                    position = 0.0   # Exit

    return result_dates, result_returns


# ═══════════════════════════════════════════════════════════════════
# Statistics
# ═══════════════════════════════════════════════════════════════════

def compute_stats(dates: List[str], returns: List[float], name: str) -> dict:
    """Compute strategy statistics."""
    if not returns or len(returns) < 30:
        return {"name": name, "error": "insufficient data"}

    n = len(returns)
    mean_ret = sum(returns) / n
    var = sum((r - mean_ret) ** 2 for r in returns) / n
    std = math.sqrt(var) if var > 0 else 1e-10
    ann_ret = mean_ret * 365
    ann_std = std * math.sqrt(365)
    sharpe = ann_ret / ann_std if ann_std > 0 else 0

    # Max drawdown
    equity = 1.0
    peak = 1.0
    max_dd = 0.0
    for r in returns:
        equity *= (1 + r)
        if equity > peak:
            peak = equity
        dd = (peak - equity) / peak
        if dd > max_dd:
            max_dd = dd

    # Win rate
    wins = sum(1 for r in returns if r > 0)
    win_rate = wins / n

    # Yearly breakdown
    by_year = defaultdict(list)
    for i, d in enumerate(dates):
        by_year[d[:4]].append(returns[i])

    yearly = {}
    for yr in sorted(by_year.keys()):
        yr_rets = by_year[yr]
        yr_mean = sum(yr_rets) / len(yr_rets)
        yr_var = sum((r - yr_mean)**2 for r in yr_rets) / len(yr_rets)
        yr_std = math.sqrt(yr_var) if yr_var > 0 else 1e-10
        yr_sharpe = (yr_mean * 365) / (yr_std * math.sqrt(365))
        yearly[yr] = {
            "sharpe": round(yr_sharpe, 3),
            "ann_ret": round(yr_mean * 365, 4),
            "n_days": len(yr_rets),
        }

    # t-statistic
    t_stat = sharpe * math.sqrt(n / 365)

    return {
        "name": name,
        "sharpe": round(sharpe, 3),
        "ann_return": round(ann_ret, 4),
        "ann_vol": round(ann_std, 4),
        "max_dd": round(max_dd, 4),
        "win_rate": round(win_rate, 3),
        "t_stat": round(t_stat, 2),
        "n_days": n,
        "date_range": f"{dates[0]} to {dates[-1]}",
        "yearly": yearly,
    }


def combine_ensemble(
    vrp_dates: List[str], vrp_returns: List[float],
    skew_dates: List[str], skew_returns: List[float],
    vrp_weight: float = 0.40, skew_weight: float = 0.60,
) -> Tuple[List[str], List[float]]:
    """Combine VRP and Skew into ensemble."""
    vrp_map = dict(zip(vrp_dates, vrp_returns))
    skew_map = dict(zip(skew_dates, skew_returns))

    common_dates = sorted(set(vrp_dates) & set(skew_dates))
    ensemble_returns = []
    for d in common_dates:
        r = vrp_weight * vrp_map[d] + skew_weight * skew_map[d]
        ensemble_returns.append(r)

    return common_dates, ensemble_returns


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════

def main():
    print("=" * 70)
    print("R51: FIRST REAL SHARPE RATIOS")
    print("     VRP + Skew backtest with actual Deribit market data")
    print("=" * 70)
    print()
    print("Data sources:")
    print("  DVOL (ATM IV): Deribit DVOL index, 2021-03 to 2026-02 (R49)")
    print("  Skew_25d:      history.deribit.com option trades, 2019-03+ (R50)")
    print("  Prices:        Deribit perpetual OHLC")
    print()

    all_results = {}

    for currency in ["BTC", "ETH"]:
        print(f"\n{'='*70}")
        print(f"  {currency}")
        print(f"{'='*70}")

        # Load data
        dvol = load_dvol_daily(currency)
        skew_data = load_real_skew(currency)
        prices = load_prices(currency)

        if not prices:
            print(f"  No prices for {currency}, skipping")
            continue

        results = {}

        # ── VRP Strategy (real DVOL) ─────────────────────────────
        if dvol:
            print(f"\n  --- VRP Strategy (real DVOL, leverage=1.5) ---")
            vrp_dates, vrp_returns = compute_vrp_pnl_series(prices, dvol, leverage=1.5)
            vrp_stats = compute_stats(vrp_dates, vrp_returns, f"{currency} VRP (real DVOL)")
            results["vrp"] = vrp_stats

            print(f"  Sharpe:     {vrp_stats['sharpe']:.3f}")
            print(f"  Ann Return: {vrp_stats['ann_return']:.2%}")
            print(f"  Ann Vol:    {vrp_stats['ann_vol']:.2%}")
            print(f"  Max DD:     {vrp_stats['max_dd']:.2%}")
            print(f"  Win Rate:   {vrp_stats['win_rate']:.1%}")
            print(f"  t-stat:     {vrp_stats['t_stat']:.2f}")
            print(f"  Period:     {vrp_stats['date_range']} ({vrp_stats['n_days']} days)")
            print(f"  Yearly Sharpe:")
            for yr, ys in vrp_stats.get("yearly", {}).items():
                print(f"    {yr}: {ys['sharpe']:+.3f} (ret={ys['ann_ret']:+.2%}, n={ys['n_days']})")

        # ── Skew MR Strategy (real skew) ─────────────────────────
        if skew_data:
            # Use DVOL for IV if available, else use surface IV
            iv_source = dvol if dvol else None
            print(f"\n  --- Skew MR Strategy (real skew, leverage=1.0) ---")
            skew_dates, skew_returns = compute_skew_pnl_series(
                skew_data, dvol=iv_source, leverage=1.0,
                sensitivity_mult=2.5, skew_lookback=60,
                z_entry_short=1.0, z_entry_long=2.0, z_exit=0.0
            )
            skew_stats = compute_stats(skew_dates, skew_returns, f"{currency} Skew MR (real)")
            results["skew"] = skew_stats

            print(f"  Sharpe:     {skew_stats['sharpe']:.3f}")
            print(f"  Ann Return: {skew_stats['ann_return']:.2%}")
            print(f"  Ann Vol:    {skew_stats['ann_vol']:.2%}")
            print(f"  Max DD:     {skew_stats['max_dd']:.2%}")
            print(f"  Win Rate:   {skew_stats['win_rate']:.1%}")
            print(f"  t-stat:     {skew_stats['t_stat']:.2f}")
            print(f"  Period:     {skew_stats['date_range']} ({skew_stats['n_days']} days)")
            print(f"  Yearly Sharpe:")
            for yr, ys in skew_stats.get("yearly", {}).items():
                print(f"    {yr}: {ys['sharpe']:+.3f} (ret={ys['ann_ret']:+.2%}, n={ys['n_days']})")

        # ── Also test Skew with different params ─────────────────
        if skew_data:
            print(f"\n  --- Skew MR (real skew, RECALIBRATED: lb=30, z_entry_short=0.5) ---")
            skew_dates2, skew_returns2 = compute_skew_pnl_series(
                skew_data, dvol=iv_source, leverage=1.0,
                sensitivity_mult=2.5, skew_lookback=30,
                z_entry_short=0.5, z_entry_long=1.5, z_exit=0.0
            )
            skew_stats2 = compute_stats(skew_dates2, skew_returns2,
                                         f"{currency} Skew MR (recalibrated)")
            results["skew_recal"] = skew_stats2

            print(f"  Sharpe:     {skew_stats2['sharpe']:.3f}")
            print(f"  Ann Return: {skew_stats2['ann_return']:.2%}")
            print(f"  Max DD:     {skew_stats2['max_dd']:.2%}")
            print(f"  t-stat:     {skew_stats2['t_stat']:.2f}")

        # ── Always-short skew (no z-score) ───────────────────────
        if skew_data:
            print(f"\n  --- Skew ALWAYS SHORT (no signals, always short skew) ---")
            # Always short skew = position always -1
            dates_s = sorted(skew_data.keys())
            always_short_dates = []
            always_short_returns = []
            dt = 1.0 / 365.0
            for i in range(1, len(dates_s)):
                d = dates_s[i]
                dp = dates_s[i-1]
                e = skew_data.get(d, {})
                ep = skew_data.get(dp, {})
                s_now = e.get("skew_25d")
                s_prev = ep.get("skew_25d")
                iv = dvol.get(dp) if dvol and dp in dvol else (ep.get("iv_atm") if ep.get("iv_atm") else None)
                if s_now is None or s_prev is None or iv is None or iv <= 0:
                    continue
                d_skew = s_now - s_prev
                sensitivity = iv * math.sqrt(dt) * 2.5
                daily_ret = -1.0 * d_skew * sensitivity  # Always short
                always_short_dates.append(d)
                always_short_returns.append(daily_ret)

            as_stats = compute_stats(always_short_dates, always_short_returns,
                                      f"{currency} Skew ALWAYS SHORT")
            results["skew_always_short"] = as_stats
            print(f"  Sharpe:     {as_stats['sharpe']:.3f}")
            print(f"  Ann Return: {as_stats['ann_return']:.2%}")
            print(f"  Max DD:     {as_stats['max_dd']:.2%}")

        # ── Ensemble (40% VRP + 60% Skew) ────────────────────────
        if dvol and skew_data and "vrp" in results and "skew" in results:
            print(f"\n  --- ENSEMBLE: 40% VRP + 60% Skew (champion config) ---")
            ens_dates, ens_returns = combine_ensemble(
                vrp_dates, vrp_returns, skew_dates, skew_returns,
                vrp_weight=0.40, skew_weight=0.60
            )
            ens_stats = compute_stats(ens_dates, ens_returns,
                                       f"{currency} Ensemble 40/60 (real)")
            results["ensemble_40_60"] = ens_stats

            print(f"  Sharpe:     {ens_stats['sharpe']:.3f}")
            print(f"  Ann Return: {ens_stats['ann_return']:.2%}")
            print(f"  Ann Vol:    {ens_stats['ann_vol']:.2%}")
            print(f"  Max DD:     {ens_stats['max_dd']:.2%}")
            print(f"  Win Rate:   {ens_stats['win_rate']:.1%}")
            print(f"  t-stat:     {ens_stats['t_stat']:.2f}")
            print(f"  Period:     {ens_stats['date_range']}")
            print(f"  Yearly Sharpe:")
            for yr, ys in ens_stats.get("yearly", {}).items():
                print(f"    {yr}: {ys['sharpe']:+.3f} (ret={ys['ann_ret']:+.2%})")

            # Also try VRP-only (since skew may not work)
            print(f"\n  --- VRP ONLY at 1.5x (no skew component) ---")
            vrp_only_stats = compute_stats(vrp_dates, vrp_returns,
                                            f"{currency} VRP only 1.5x")
            print(f"  Sharpe:     {vrp_only_stats['sharpe']:.3f}")
            print(f"  Ann Return: {vrp_only_stats['ann_return']:.2%}")

        all_results[currency] = results

    # ── FINAL SUMMARY ─────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("R51 FINAL SUMMARY — FIRST REAL SHARPE RATIOS")
    print(f"{'='*70}")
    print()

    print(f"{'Strategy':<35s}  {'Sharpe':>7s}  {'AnnRet':>8s}  {'MaxDD':>7s}  {'t-stat':>6s}")
    print(f"{'-'*35}  {'-'*7}  {'-'*8}  {'-'*7}  {'-'*6}")

    for currency in ["BTC", "ETH"]:
        results = all_results.get(currency, {})
        for key in ["vrp", "skew", "skew_recal", "skew_always_short", "ensemble_40_60"]:
            s = results.get(key)
            if s and "sharpe" in s:
                print(f"  {s['name']:<33s}  {s['sharpe']:7.3f}  {s['ann_return']:8.2%}  "
                      f"{s['max_dd']:7.2%}  {s['t_stat']:6.2f}")

    # Comparison with synthetic
    print(f"\n  SYNTHETIC vs REAL comparison:")
    print(f"  {'Metric':<30s}  {'Synthetic':>10s}  {'Real':>10s}  {'Delta':>10s}")
    print(f"  {'-'*30}  {'-'*10}  {'-'*10}  {'-'*10}")

    btc = all_results.get("BTC", {})
    if "vrp" in btc:
        syn_vrp = 2.088
        real_vrp = btc["vrp"]["sharpe"]
        print(f"  {'BTC VRP Sharpe':<30s}  {syn_vrp:10.3f}  {real_vrp:10.3f}  {real_vrp-syn_vrp:+10.3f}")

    if "skew" in btc:
        syn_skew = 1.744
        real_skew = btc["skew"]["sharpe"]
        print(f"  {'BTC Skew Sharpe':<30s}  {syn_skew:10.3f}  {real_skew:10.3f}  {real_skew-syn_skew:+10.3f}")

    if "ensemble_40_60" in btc:
        syn_ens = 4.816
        real_ens = btc["ensemble_40_60"]["sharpe"]
        print(f"  {'BTC Ensemble Sharpe':<30s}  {syn_ens:10.3f}  {real_ens:10.3f}  {real_ens-syn_ens:+10.3f}")

    # Save results
    output_dir = ROOT / "data" / "cache" / "deribit" / "real_surface"
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "r51_real_backtest_results.json"
    with open(out_path, "w") as f:
        json.dump({
            "research_id": "R51",
            "title": "First Real Sharpe Ratios — VRP + Skew with actual market data",
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "data_sources": {
                "iv_atm": "Deribit DVOL (R49)",
                "skew_25d": "history.deribit.com option trades (R50)",
                "prices": "Deribit perpetual OHLC",
            },
            "results": {k: v for k, v in all_results.items()},
        }, f, indent=2)
    print(f"\n  Results saved → {out_path}")


if __name__ == "__main__":
    main()
