#!/usr/bin/env python3
"""
Fetch Deribit DVOL (Implied Volatility Index) Historical Data
=============================================================

DVOL is Deribit's 30-day forward-looking implied volatility index,
calculated from REAL option prices on the Deribit exchange.

Available since March 2021 for BTC and ETH.

This is the FIRST real IV data we can use to validate our synthetic
IV assumptions and VRP strategy.

Endpoint: public/get_volatility_index_data
Params: currency, start_timestamp, end_timestamp, resolution
Resolution: "1" (1 second), "60" (1 minute), "3600" (1 hour), "43200" (12 hours), "1D" (1 day)
"""
import csv
import json
import math
import os
import ssl
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

DERIBIT_BASE = "https://www.deribit.com/api/v2/public"


def _make_ssl_context():
    try:
        import certifi
        return ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        pass
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    return ctx


SSL_CTX = _make_ssl_context()


def _api_get(method: str, params: dict, retries: int = 3) -> dict:
    """Make a Deribit public API call with retries."""
    url = f"{DERIBIT_BASE}/{method}?{urllib.parse.urlencode(params)}"
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "NexusQuant/1.0"})
            with urllib.request.urlopen(req, context=SSL_CTX, timeout=30) as resp:
                data = json.loads(resp.read().decode())
                if "result" in data:
                    return data["result"]
                return data
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as e:
            if attempt < retries - 1:
                wait = 2 ** (attempt + 1)
                print(f"  Retry {attempt+1}/{retries} after {wait}s: {e}")
                time.sleep(wait)
            else:
                raise
    return {}


def fetch_dvol_history(currency: str, resolution: str = "3600",
                       start_date: str = "2021-03-01",
                       end_date: str = None) -> list:
    """
    Fetch DVOL OHLC data from Deribit.

    Returns list of [timestamp_ms, open, high, low, close] entries.
    """
    if end_date is None:
        end_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    start_ts = int(datetime.strptime(start_date, "%Y-%m-%d").replace(
        tzinfo=timezone.utc).timestamp() * 1000)
    end_ts = int(datetime.strptime(end_date, "%Y-%m-%d").replace(
        tzinfo=timezone.utc).timestamp() * 1000)

    all_data = []
    chunk_ms = 30 * 24 * 3600 * 1000  # 30 days per request

    current_start = start_ts
    while current_start < end_ts:
        current_end = min(current_start + chunk_ms, end_ts)

        params = {
            "currency": currency,
            "start_timestamp": current_start,
            "end_timestamp": current_end,
            "resolution": resolution,
        }

        try:
            result = _api_get("get_volatility_index_data", params)
            if isinstance(result, dict) and "data" in result:
                data = result["data"]
            elif isinstance(result, list):
                data = result
            else:
                data = []

            all_data.extend(data)
            n = len(data)
            dt_str = datetime.fromtimestamp(current_start / 1000, tz=timezone.utc).strftime("%Y-%m-%d")
            print(f"  {currency} DVOL from {dt_str}: {n} bars")

        except Exception as e:
            dt_str = datetime.fromtimestamp(current_start / 1000, tz=timezone.utc).strftime("%Y-%m-%d")
            print(f"  {currency} DVOL from {dt_str}: ERROR — {e}")

        current_start = current_end
        time.sleep(0.15)  # rate limit

    return all_data


def save_dvol_csv(currency: str, data: list, resolution: str, output_dir: Path):
    """Save DVOL data to CSV."""
    output_dir.mkdir(parents=True, exist_ok=True)
    filename = f"{currency}_DVOL_{resolution}.csv"
    filepath = output_dir / filename

    with open(filepath, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "date", "dvol_open", "dvol_high", "dvol_low", "dvol_close"])

        for row in data:
            if len(row) >= 5:
                ts_ms = row[0]
                dt = datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc)
                date_str = dt.strftime("%Y-%m-%dT%H:%M:%SZ")
                writer.writerow([ts_ms, date_str, row[1], row[2], row[3], row[4]])

    print(f"  Saved {len(data)} rows to {filepath}")
    return filepath


def fetch_perpetual_prices(currency: str, start_date: str, end_date: str,
                           resolution: str = "1D") -> list:
    """Fetch perpetual close prices for RV calculation."""
    instrument = f"{currency}-PERPETUAL"

    start_ts = int(datetime.strptime(start_date, "%Y-%m-%d").replace(
        tzinfo=timezone.utc).timestamp() * 1000)
    end_ts = int(datetime.strptime(end_date, "%Y-%m-%d").replace(
        tzinfo=timezone.utc).timestamp() * 1000)

    all_data = []
    chunk_ms = 90 * 24 * 3600 * 1000  # 90 days per chunk

    current_start = start_ts
    while current_start < end_ts:
        current_end = min(current_start + chunk_ms, end_ts)

        params = {
            "instrument_name": instrument,
            "start_timestamp": current_start,
            "end_timestamp": current_end,
            "resolution": resolution,
        }

        try:
            result = _api_get("get_tradingview_chart_data", params)
            if result and "close" in result and "ticks" in result:
                for i, ts in enumerate(result["ticks"]):
                    all_data.append({
                        "timestamp": ts,
                        "close": result["close"][i],
                        "high": result["high"][i],
                        "low": result["low"][i],
                        "open": result["open"][i],
                        "volume": result["volume"][i] if "volume" in result else 0,
                    })
        except Exception as e:
            print(f"  Price fetch error: {e}")

        current_start = current_end
        time.sleep(0.15)

    return all_data


def compute_rv(closes: list, window: int = 30) -> list:
    """Compute annualized realized volatility from close prices."""
    rv_series = [None] * len(closes)
    for i in range(window, len(closes)):
        log_returns = []
        for j in range(i - window + 1, i + 1):
            if closes[j] > 0 and closes[j-1] > 0:
                log_returns.append(math.log(closes[j] / closes[j-1]))
        if len(log_returns) >= window // 2:
            mean_r = sum(log_returns) / len(log_returns)
            var = sum((r - mean_r) ** 2 for r in log_returns) / len(log_returns)
            rv_series[i] = math.sqrt(var * 365) * 100  # annualized, in %
    return rv_series


def run_vrp_validation(dvol_data: list, price_data: list, currency: str):
    """
    Run VRP validation using REAL DVOL data.

    VRP = IV (DVOL) - RV (from prices)
    If VRP is consistently positive, our VRP strategy has a real edge.
    """
    print(f"\n{'='*70}")
    print(f"VRP VALIDATION WITH REAL DATA — {currency}")
    print(f"{'='*70}")

    # Build aligned daily series
    # DVOL: [timestamp_ms, open, high, low, close]
    dvol_by_date = {}
    for row in dvol_data:
        if len(row) >= 5:
            dt = datetime.fromtimestamp(row[0] / 1000, tz=timezone.utc)
            date_key = dt.strftime("%Y-%m-%d")
            dvol_by_date[date_key] = row[4]  # close

    # Prices
    price_by_date = {}
    for p in price_data:
        dt = datetime.fromtimestamp(p["timestamp"] / 1000, tz=timezone.utc)
        date_key = dt.strftime("%Y-%m-%d")
        price_by_date[date_key] = p["close"]

    # Find overlapping dates
    all_dates = sorted(set(dvol_by_date.keys()) & set(price_by_date.keys()))
    if len(all_dates) < 60:
        print(f"  Not enough overlapping data: {len(all_dates)} days")
        return None

    print(f"  Overlapping dates: {all_dates[0]} to {all_dates[-1]} ({len(all_dates)} days)")

    # Build aligned arrays
    dates = all_dates
    dvol_values = [dvol_by_date[d] for d in dates]
    closes = [price_by_date[d] for d in dates]

    # Compute RV (30-day window)
    rv_30 = compute_rv(closes, window=30)

    # Compute VRP = DVOL - RV
    vrp_series = []
    yearly_stats = {}

    for i in range(30, len(dates)):
        iv = dvol_values[i]  # DVOL close (already in %)
        rv = rv_30[i]
        if iv is not None and rv is not None and iv > 0 and rv > 0:
            vrp = iv - rv
            year = dates[i][:4]
            vrp_series.append({"date": dates[i], "year": year, "iv": iv, "rv": rv, "vrp": vrp})

            if year not in yearly_stats:
                yearly_stats[year] = {"vrps": [], "ivs": [], "rvs": []}
            yearly_stats[year]["vrps"].append(vrp)
            yearly_stats[year]["ivs"].append(iv)
            yearly_stats[year]["rvs"].append(rv)

    if not vrp_series:
        print("  No VRP data computed")
        return None

    # Print results
    print(f"\n  {'Year':>6s}  {'Days':>5s}  {'Avg IV%':>8s}  {'Avg RV%':>8s}  {'Avg VRP':>8s}  {'VRP>0%':>7s}  {'Med VRP':>8s}")
    print(f"  {'-'*6}  {'-'*5}  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*7}  {'-'*8}")

    all_vrps = []
    for year in sorted(yearly_stats.keys()):
        s = yearly_stats[year]
        vrps = s["vrps"]
        all_vrps.extend(vrps)
        avg_iv = sum(s["ivs"]) / len(s["ivs"])
        avg_rv = sum(s["rvs"]) / len(s["rvs"])
        avg_vrp = sum(vrps) / len(vrps)
        pct_pos = sum(1 for v in vrps if v > 0) / len(vrps) * 100
        sorted_vrps = sorted(vrps)
        med_vrp = sorted_vrps[len(sorted_vrps) // 2]

        print(f"  {year:>6s}  {len(vrps):>5d}  {avg_iv:>8.1f}  {avg_rv:>8.1f}  {avg_vrp:>+8.1f}  {pct_pos:>6.1f}%  {med_vrp:>+8.1f}")

    # Overall stats
    print(f"\n  {'TOTAL':>6s}  {len(all_vrps):>5d}  ", end="")
    all_ivs = [v["iv"] for v in vrp_series]
    all_rvs = [v["rv"] for v in vrp_series]
    avg_iv = sum(all_ivs) / len(all_ivs)
    avg_rv = sum(all_rvs) / len(all_rvs)
    avg_vrp = sum(all_vrps) / len(all_vrps)
    pct_pos = sum(1 for v in all_vrps if v > 0) / len(all_vrps) * 100
    sorted_all = sorted(all_vrps)
    med_vrp = sorted_all[len(sorted_all) // 2]

    print(f"{avg_iv:>8.1f}  {avg_rv:>8.1f}  {avg_vrp:>+8.1f}  {pct_pos:>6.1f}%  {med_vrp:>+8.1f}")

    # Statistical significance
    if len(all_vrps) > 30:
        mean_v = sum(all_vrps) / len(all_vrps)
        var_v = sum((v - mean_v) ** 2 for v in all_vrps) / len(all_vrps)
        std_v = math.sqrt(var_v)
        se = std_v / math.sqrt(len(all_vrps))
        t_stat = mean_v / se if se > 0 else 0

        print(f"\n  --- Statistical Test: VRP > 0 ---")
        print(f"  Mean VRP: {mean_v:+.2f}%")
        print(f"  Std VRP:  {std_v:.2f}%")
        print(f"  SE:       {se:.4f}%")
        print(f"  t-stat:   {t_stat:.2f}")
        print(f"  n:        {len(all_vrps)}")
        print(f"  Significant (|t| > 2): {'YES' if abs(t_stat) > 2 else 'NO'}")

    # VRP PnL simulation (simple: short vol, earn VRP daily)
    print(f"\n  --- Simple VRP Strategy PnL (daily rebalance) ---")
    equity = 1.0
    equity_curve = [1.0]
    returns_list = []

    for i in range(1, len(vrp_series)):
        # Daily VRP PnL = 0.5 * (IV² - RV²) / 365
        iv = vrp_series[i]["iv"] / 100.0  # convert to decimal
        rv = vrp_series[i]["rv"] / 100.0
        daily_pnl = 0.5 * (iv**2 - rv**2) / 365.0
        equity += equity * daily_pnl
        equity_curve.append(equity)
        ret = daily_pnl
        returns_list.append(ret)

    if returns_list:
        avg_ret = sum(returns_list) / len(returns_list)
        var_ret = sum((r - avg_ret) ** 2 for r in returns_list) / len(returns_list)
        std_ret = math.sqrt(var_ret)
        sharpe = (avg_ret / std_ret * math.sqrt(365)) if std_ret > 0 else 0

        total_ret = (equity_curve[-1] / equity_curve[0] - 1) * 100
        max_dd = 0
        peak = equity_curve[0]
        for e in equity_curve:
            if e > peak:
                peak = e
            dd = (e - peak) / peak
            if dd < max_dd:
                max_dd = dd

        print(f"  Total Return: {total_ret:+.1f}%")
        print(f"  Sharpe (ann): {sharpe:.3f}")
        print(f"  Max Drawdown: {max_dd*100:.1f}%")
        print(f"  Final Equity: {equity_curve[-1]:.4f}")

    # Compare with our synthetic assumption
    print(f"\n  --- Comparison: Real vs Synthetic VRP ---")
    print(f"  Synthetic VRP mean: {8.0 if currency == 'BTC' else 10.0:.1f}%  (AR(1) calibration)")
    print(f"  Real VRP mean:     {avg_vrp:+.1f}%  (DVOL - RV)")
    diff = avg_vrp - (8.0 if currency == "BTC" else 10.0)
    print(f"  Difference:        {diff:+.1f}%  ({'synthetic OVERESTIMATES' if diff < 0 else 'synthetic UNDERESTIMATES'})")

    return {
        "currency": currency,
        "days": len(all_vrps),
        "date_range": f"{dates[30]} to {dates[-1]}",
        "avg_iv": round(avg_iv, 2),
        "avg_rv": round(avg_rv, 2),
        "avg_vrp": round(avg_vrp, 2),
        "pct_positive": round(pct_pos, 1),
        "sharpe": round(sharpe, 3) if returns_list else None,
        "yearly": {
            year: {
                "avg_vrp": round(sum(s["vrps"]) / len(s["vrps"]), 2),
                "pct_positive": round(sum(1 for v in s["vrps"] if v > 0) / len(s["vrps"]) * 100, 1),
            }
            for year, s in yearly_stats.items()
        },
    }


def main():
    print("=" * 70)
    print("DERIBIT DVOL HISTORICAL DATA FETCH + VRP VALIDATION")
    print("=" * 70)
    print()
    print("This uses REAL implied volatility data from Deribit's DVOL index.")
    print("DVOL = 30-day forward-looking annualized IV from option prices.")
    print()

    output_dir = ROOT / "data" / "cache" / "deribit" / "dvol"

    all_results = {}

    for currency in ["BTC", "ETH"]:
        print(f"\n--- Fetching {currency} DVOL (hourly, 2021-present) ---")

        # Fetch daily DVOL (most useful for strategy validation)
        daily_data = fetch_dvol_history(
            currency=currency,
            resolution="3600",  # hourly for granularity
            start_date="2021-03-01",
        )

        if daily_data:
            save_dvol_csv(currency, daily_data, "1h", output_dir)
            print(f"  Total {currency} DVOL bars: {len(daily_data)}")
        else:
            print(f"  WARNING: No DVOL data returned for {currency}")
            continue

        # Also fetch daily resolution for simpler analysis
        print(f"\n--- Fetching {currency} DVOL (daily) ---")
        daily_1d = fetch_dvol_history(
            currency=currency,
            resolution="43200",  # 12-hour
            start_date="2021-03-01",
        )
        if daily_1d:
            save_dvol_csv(currency, daily_1d, "12h", output_dir)

        # Fetch price data for the same period
        print(f"\n--- Fetching {currency} perpetual prices ---")
        price_data = fetch_perpetual_prices(
            currency=currency,
            start_date="2021-03-01",
            end_date=datetime.now(timezone.utc).strftime("%Y-%m-%d"),
            resolution="1D",
        )
        print(f"  Total {currency} price bars: {len(price_data)}")

        # Use daily DVOL for validation (aggregate hourly to daily close)
        dvol_daily = {}
        for row in daily_data:
            if len(row) >= 5:
                dt = datetime.fromtimestamp(row[0] / 1000, tz=timezone.utc)
                date_key = dt.strftime("%Y-%m-%d")
                # Keep last hourly close as daily close
                dvol_daily[date_key] = row[4]

        daily_dvol_list = []
        for d in sorted(dvol_daily.keys()):
            daily_dvol_list.append([
                int(datetime.strptime(d, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp() * 1000),
                dvol_daily[d], dvol_daily[d], dvol_daily[d], dvol_daily[d]
            ])

        # Run VRP validation with real data
        result = run_vrp_validation(daily_dvol_list, price_data, currency)
        if result:
            all_results[currency] = result

    # Save results
    print(f"\n{'='*70}")
    print("OVERALL RESULTS")
    print(f"{'='*70}")

    for currency, r in all_results.items():
        print(f"\n  {currency}:")
        print(f"    Period: {r['date_range']} ({r['days']} days)")
        print(f"    Avg IV (DVOL): {r['avg_iv']:.1f}%")
        print(f"    Avg RV:        {r['avg_rv']:.1f}%")
        print(f"    Avg VRP:       {r['avg_vrp']:+.1f}%  (VRP>0 in {r['pct_positive']:.0f}% of days)")
        if r["sharpe"]:
            print(f"    VRP Sharpe:    {r['sharpe']:.3f}  (REAL DATA)")

    if all_results:
        # KEY COMPARISON
        print(f"\n{'='*70}")
        print("CRITICAL: REAL vs SYNTHETIC COMPARISON")
        print(f"{'='*70}")
        print()
        print("  Our synthetic model assumes:")
        print("    BTC VRP mean = 8.0% (AR(1) calibration)")
        print("    ETH VRP mean = 10.0% (AR(1) calibration)")
        print()
        print("  Real data shows:")
        for cur, r in all_results.items():
            synthetic = 8.0 if cur == "BTC" else 10.0
            print(f"    {cur} VRP mean = {r['avg_vrp']:+.1f}% (real DVOL-RV)")
            print(f"    Difference:    {r['avg_vrp'] - synthetic:+.1f}%")

        print()
        print("  If real VRP is POSITIVE and SIGNIFICANT:")
        print("    → VRP strategy has a genuine edge")
        print("    → Synthetic assumptions were approximately correct")
        print()
        print("  If real VRP is NEGATIVE or INSIGNIFICANT:")
        print("    → VRP strategy may not work in reality")
        print("    → Synthetic assumptions were WRONG")

    # Save JSON results
    out_json = output_dir / "vrp_validation_results.json"
    with open(out_json, "w") as f:
        json.dump({
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "data_source": "Deribit DVOL (public/get_volatility_index_data)",
            "data_type": "REAL implied volatility from option prices",
            "results": all_results,
        }, f, indent=2)
    print(f"\nResults saved to: {out_json}")


if __name__ == "__main__":
    main()
