#!/usr/bin/env python3
"""
R50: Fetch REAL IV Surface from history.deribit.com
====================================================
Reconstruct historical skew_25d, butterfly_25d, term_spread
from REAL option trade data (2019-2026).

Source: history.deribit.com — ALL option trades since March 2019 with IV field.
        FREE, no auth needed.

Strategy:
- Sample one day per week (Wednesdays = highest option activity)
- For each sample day: fetch option trades, extract IV surface
- Interpolate to daily frequency
- Compare real surface vs synthetic AR(1) assumptions

This completes the real data validation alongside R49 (DVOL/VRP).
"""
import json
import math
import os
import subprocess
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

DERIBIT_HISTORY = "https://history.deribit.com/api/v2/public"
OUTPUT_DIR = ROOT / "data" / "cache" / "deribit" / "real_surface"


# ═══════════════════════════════════════════════════════════════════
# HTTP via curl subprocess (reliable on macOS, avoids Python SSL bugs)
# ═══════════════════════════════════════════════════════════════════

def curl_get_json(url: str, retries: int = 3, timeout: int = 30) -> dict:
    """Fetch JSON via curl subprocess — bulletproof on macOS."""
    for attempt in range(retries):
        try:
            result = subprocess.run(
                ["curl", "-s", "--max-time", str(timeout), url],
                capture_output=True, text=True, timeout=timeout + 10
            )
            if result.returncode != 0:
                raise RuntimeError(f"curl exit {result.returncode}: {result.stderr[:200]}")
            data = json.loads(result.stdout)
            if "result" in data:
                return data["result"]
            return data
        except (json.JSONDecodeError, subprocess.TimeoutExpired, RuntimeError) as e:
            if attempt < retries - 1:
                time.sleep(2 ** (attempt + 1))
            else:
                raise
    return {}


# ═══════════════════════════════════════════════════════════════════
# Instrument parsing & delta approximation
# ═══════════════════════════════════════════════════════════════════

MONTHS = {"JAN": 1, "FEB": 2, "MAR": 3, "APR": 4, "MAY": 5, "JUN": 6,
          "JUL": 7, "AUG": 8, "SEP": 9, "OCT": 10, "NOV": 11, "DEC": 12}


def parse_instrument(name: str) -> Optional[dict]:
    """Parse BTC-28MAR25-80000-C → {currency, expiry, strike, type}."""
    parts = name.split("-")
    if len(parts) != 4 or parts[3] not in ("C", "P"):
        return None
    try:
        exp_str = parts[1]
        day = int(exp_str[:-5])
        month = MONTHS.get(exp_str[-5:-2], 0)
        year = 2000 + int(exp_str[-2:])
        if month == 0:
            return None
        return {
            "currency": parts[0],
            "expiry": datetime(year, month, day, 8, 0, tzinfo=timezone.utc),
            "strike": float(parts[2]),
            "type": parts[3],
        }
    except (ValueError, IndexError):
        return None


def approx_delta(spot: float, strike: float, iv: float, tte: float, opt_type: str) -> float:
    """Approximate Black-Scholes delta (Abramowitz & Stegun)."""
    if iv <= 0 or tte <= 0 or spot <= 0:
        return 0.0
    d1 = (math.log(spot / strike) + 0.5 * iv * iv * tte) / (iv * math.sqrt(tte))
    a = 0.2316419
    b = [0.319381530, -0.356563782, 1.781477937, -1.821255978, 1.330274429]
    t = 1.0 / (1.0 + a * abs(d1))
    nd = 1.0 - (1.0 / math.sqrt(2 * math.pi)) * math.exp(-d1 * d1 / 2) * (
        b[0]*t + b[1]*t**2 + b[2]*t**3 + b[3]*t**4 + b[4]*t**5)
    if d1 < 0:
        nd = 1.0 - nd
    return nd if opt_type == "C" else nd - 1.0


# ═══════════════════════════════════════════════════════════════════
# Core: Fetch trades & extract surface for one day
# ═══════════════════════════════════════════════════════════════════

def fetch_option_trades(currency: str, date_str: str, max_trades: int = 5000) -> list:
    """Fetch option trades for a given day from history.deribit.com."""
    dt = datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    start_ms = int(dt.timestamp() * 1000)
    end_ms = int((dt + timedelta(days=1)).timestamp() * 1000)

    all_trades = []
    current_start = start_ms

    while current_start < end_ms and len(all_trades) < max_trades:
        url = (f"{DERIBIT_HISTORY}/get_last_trades_by_currency_and_time?"
               f"currency={currency}&kind=option"
               f"&start_timestamp={current_start}&end_timestamp={end_ms}"
               f"&count=1000&sorting=asc")

        try:
            result = curl_get_json(url)
            trades = result.get("trades", [])
            if not trades:
                break
            all_trades.extend(trades)
            current_start = trades[-1]["timestamp"] + 1
            has_more = result.get("has_more", False)
            if not has_more or len(trades) < 1000:
                break
            time.sleep(0.1)
        except Exception as e:
            print(f"\n    API error: {e}")
            break

    return all_trades


def extract_surface(trades: list, spot: float, ref_date: datetime) -> dict:
    """
    Extract IV surface metrics from option trades.

    Returns: iv_atm, skew_25d, butterfly_25d, term_spread, iv_25d_put, iv_25d_call
    """
    if not trades or spot <= 0:
        return {}

    # Group trades by expiry
    by_expiry = defaultdict(list)
    for t in trades:
        info = parse_instrument(t.get("instrument_name", ""))
        if not info:
            continue
        iv = t.get("iv")
        if iv is None or iv <= 0:
            continue
        tte = (info["expiry"] - ref_date).total_seconds() / (365.25 * 86400)
        if tte < 7 / 365.25 or tte > 180 / 365.25:
            continue
        by_expiry[info["expiry"]].append({
            "strike": info["strike"],
            "type": info["type"],
            "iv": iv / 100.0,
            "tte": tte,
            "amount": t.get("amount", 0),
            "index_price": t.get("index_price", spot),
        })

    if not by_expiry:
        return {}

    # Find front-month expiry (~30 days)
    target_tte = 30 / 365.25
    expiries = sorted(by_expiry.keys())
    front_exp = min(expiries, key=lambda e: abs(
        (e - ref_date).total_seconds() / (365.25 * 86400) - target_tte))
    front_tte = (front_exp - ref_date).total_seconds() / (365.25 * 86400)
    front = by_expiry[front_exp]

    if len(front) < 3:
        return {}

    # ATM IV: volume-weighted average of nearest-to-spot trades
    front_sorted = sorted(front, key=lambda x: abs(x["strike"] - spot))
    # Take the 3 closest to ATM
    atm_candidates = front_sorted[:min(5, len(front_sorted))]
    atm_close = [t for t in atm_candidates if abs(t["strike"] - spot) / spot < 0.05]
    if not atm_close:
        atm_close = atm_candidates[:2]

    total_amt = sum(t["amount"] for t in atm_close)
    if total_amt > 0:
        iv_atm = sum(t["iv"] * t["amount"] for t in atm_close) / total_amt
    else:
        iv_atm = sum(t["iv"] for t in atm_close) / len(atm_close)

    # Find 25-delta put and call IV
    # Compute approximate delta for each trade
    for t in front:
        t["delta_approx"] = approx_delta(spot, t["strike"], t["iv"], front_tte, t["type"])

    puts = [t for t in front if t["type"] == "P"]
    calls = [t for t in front if t["type"] == "C"]

    # 25-delta put: delta ~ -0.25
    iv_25d_put = None
    if puts:
        puts_by_d = sorted(puts, key=lambda t: abs(abs(t["delta_approx"]) - 0.25))
        best_put = puts_by_d[0]
        if abs(abs(best_put["delta_approx"]) - 0.25) < 0.15:
            iv_25d_put = best_put["iv"]

    # 25-delta call: delta ~ +0.25
    iv_25d_call = None
    if calls:
        calls_by_d = sorted(calls, key=lambda t: abs(t["delta_approx"] - 0.25))
        best_call = calls_by_d[0]
        if abs(best_call["delta_approx"] - 0.25) < 0.15:
            iv_25d_call = best_call["iv"]

    # Surface metrics
    skew_25d = None
    butterfly_25d = None
    if iv_25d_put is not None and iv_25d_call is not None:
        skew_25d = iv_25d_put - iv_25d_call
        butterfly_25d = 0.5 * (iv_25d_put + iv_25d_call) - iv_atm

    # Term spread: find back-month (~60-90d)
    term_spread = None
    for exp in expiries:
        back_tte = (exp - ref_date).total_seconds() / (365.25 * 86400)
        if back_tte > front_tte * 1.5 and back_tte < 120 / 365.25:
            back_chain = by_expiry[exp]
            back_atm = sorted(back_chain, key=lambda x: abs(x["strike"] - spot))
            if back_atm:
                term_spread = iv_atm - back_atm[0]["iv"]
            break

    return {
        "iv_atm": round(iv_atm, 4),
        "iv_25d_put": round(iv_25d_put, 4) if iv_25d_put else None,
        "iv_25d_call": round(iv_25d_call, 4) if iv_25d_call else None,
        "skew_25d": round(skew_25d, 4) if skew_25d else None,
        "butterfly_25d": round(butterfly_25d, 4) if butterfly_25d else None,
        "term_spread": round(term_spread, 4) if term_spread else None,
        "front_expiry": front_exp.strftime("%Y-%m-%d"),
        "front_tte_days": round(front_tte * 365.25),
        "n_front_trades": len(front),
        "n_total_trades": len(trades),
    }


# ═══════════════════════════════════════════════════════════════════
# Fetch perpetual prices for spot reference
# ═══════════════════════════════════════════════════════════════════

def fetch_spot_prices(currency: str, start_date: str, end_date: str) -> dict:
    """Fetch daily close prices for spot reference. Returns {date_str: price}."""
    print(f"  Fetching {currency} daily prices for spot reference...")

    start_dt = datetime.strptime(start_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    end_dt = datetime.strptime(end_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)

    instrument = f"{currency}-PERPETUAL"
    prices = {}
    current = start_dt

    while current < end_dt:
        chunk_end = min(current + timedelta(days=365), end_dt)
        start_ms = int(current.timestamp() * 1000)
        end_ms = int(chunk_end.timestamp() * 1000)

        url = (f"https://www.deribit.com/api/v2/public/get_tradingview_chart_data?"
               f"instrument_name={instrument}&resolution=1D"
               f"&start_timestamp={start_ms}&end_timestamp={end_ms}")

        try:
            result = curl_get_json(url, timeout=60)
            if result.get("status") == "ok" and result.get("close"):
                ticks = result.get("ticks", [])
                closes = result["close"]
                for i, ts in enumerate(ticks):
                    dt = datetime.fromtimestamp(ts / 1000, tz=timezone.utc)
                    prices[dt.strftime("%Y-%m-%d")] = closes[i]
        except Exception as e:
            print(f"    Price fetch error for {current.strftime('%Y')}: {e}")

        current = chunk_end
        time.sleep(0.2)

    print(f"    Got {len(prices)} daily prices")
    return prices


# ═══════════════════════════════════════════════════════════════════
# Main: Weekly sampling loop
# ═══════════════════════════════════════════════════════════════════

def fetch_weekly_surfaces(currency: str, prices: dict,
                           start_date: str = "2019-03-01",
                           end_date: str = "2026-02-21") -> list:
    """Fetch IV surface for one day per week (Wednesdays)."""
    results = []
    dt = datetime.strptime(start_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    end_dt = datetime.strptime(end_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)

    # Advance to first Wednesday
    while dt.weekday() != 2:
        dt += timedelta(days=1)

    total_weeks = int((end_dt - dt).days / 7) + 1
    week_num = 0
    consecutive_fails = 0

    while dt <= end_dt:
        week_num += 1
        date_str = dt.strftime("%Y-%m-%d")
        spot = prices.get(date_str)

        # If no price on Wednesday, try Thursday then Tuesday
        if spot is None:
            for offset in [1, -1, 2, -2]:
                alt = (dt + timedelta(days=offset)).strftime("%Y-%m-%d")
                spot = prices.get(alt)
                if spot:
                    date_str = alt
                    break

        if spot is None:
            dt += timedelta(days=7)
            continue

        pct = week_num / total_weeks * 100
        print(f"  [{week_num:3d}/{total_weeks}] {date_str} spot={spot:>10,.0f} ... ",
              end="", flush=True)

        try:
            trades = fetch_option_trades(currency, date_str, max_trades=3000)
            if trades:
                ref = datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
                surface = extract_surface(trades, spot, ref)
                if surface and surface.get("iv_atm"):
                    surface["date"] = date_str
                    surface["currency"] = currency
                    surface["spot"] = round(spot, 2)
                    results.append(surface)
                    skew = surface.get("skew_25d")
                    skew_str = f"skew={skew:+.4f}" if skew else "skew=N/A"
                    print(f"ATM={surface['iv_atm']:.1%} {skew_str} "
                          f"({surface['n_front_trades']} front trades)")
                    consecutive_fails = 0
                else:
                    print(f"extraction failed ({len(trades)} trades)")
                    consecutive_fails += 1
            else:
                print("no trades")
                consecutive_fails += 1
        except Exception as e:
            print(f"ERROR: {e}")
            consecutive_fails += 1

        if consecutive_fails > 5:
            print(f"    ⚠ {consecutive_fails} consecutive failures, sleeping 5s...")
            time.sleep(5)
            consecutive_fails = 0

        dt += timedelta(days=7)
        time.sleep(0.15)

    return results


def interpolate_to_daily(weekly_data: list) -> list:
    """Linearly interpolate weekly surface data to daily frequency."""
    if len(weekly_data) < 2:
        return weekly_data

    daily = []
    for i in range(len(weekly_data) - 1):
        d0 = weekly_data[i]
        d1 = weekly_data[i + 1]
        dt0 = datetime.strptime(d0["date"], "%Y-%m-%d")
        dt1 = datetime.strptime(d1["date"], "%Y-%m-%d")
        n_days = (dt1 - dt0).days

        for day_offset in range(n_days):
            frac = day_offset / n_days
            dt = dt0 + timedelta(days=day_offset)
            row = {"date": dt.strftime("%Y-%m-%d"), "currency": d0.get("currency", "")}

            for field in ["iv_atm", "skew_25d", "butterfly_25d", "term_spread",
                          "iv_25d_put", "iv_25d_call"]:
                v0 = d0.get(field)
                v1 = d1.get(field)
                if v0 is not None and v1 is not None:
                    row[field] = round(v0 + frac * (v1 - v0), 4)
                elif v0 is not None:
                    row[field] = v0
                else:
                    row[field] = v1

            # Interpolate spot
            s0 = d0.get("spot", 0)
            s1 = d1.get("spot", 0)
            if s0 and s1:
                row["spot"] = round(s0 + frac * (s1 - s0), 2)

            row["interpolated"] = day_offset > 0
            daily.append(row)

    # Add last point
    last = weekly_data[-1].copy()
    last["interpolated"] = False
    daily.append(last)

    return daily


def analyze_real_vs_synthetic(daily_data: list, currency: str):
    """Compare real IV surface data vs synthetic AR(1) assumptions."""
    print(f"\n{'='*70}")
    print(f"R50: REAL vs SYNTHETIC IV SURFACE — {currency}")
    print(f"{'='*70}")

    # Only use non-interpolated (actual measurement) points for stats
    actual = [d for d in daily_data if not d.get("interpolated", True)]

    if not actual:
        print("  No actual data points!")
        return {}

    # Statistics for each surface metric
    stats = {}
    for field in ["iv_atm", "skew_25d", "butterfly_25d", "term_spread"]:
        vals = [d[field] for d in actual if d.get(field) is not None]
        if not vals:
            continue

        mean = sum(vals) / len(vals)
        var = sum((v - mean) ** 2 for v in vals) / len(vals)
        std = math.sqrt(var)

        # Daily changes for autocorrelation
        changes = [vals[i] - vals[i-1] for i in range(1, len(vals))]
        if changes:
            mean_chg = sum(changes) / len(changes)
            std_chg = math.sqrt(sum((c - mean_chg)**2 for c in changes) / len(changes))
        else:
            mean_chg = std_chg = 0

        # Approximate AR(1) coefficient from weekly data
        if len(vals) >= 10:
            # Regress x_t on x_{t-1} (demeaned)
            demeaned = [v - mean for v in vals]
            num = sum(demeaned[i] * demeaned[i-1] for i in range(1, len(demeaned)))
            den = sum(demeaned[i-1] ** 2 for i in range(1, len(demeaned)))
            ar1_coef = num / den if den > 0 else 0
        else:
            ar1_coef = None

        stats[field] = {
            "n": len(vals),
            "mean": round(mean, 4),
            "std": round(std, 4),
            "min": round(min(vals), 4),
            "max": round(max(vals), 4),
            "weekly_change_std": round(std_chg, 4),
            "ar1_weekly": round(ar1_coef, 3) if ar1_coef is not None else None,
        }

    # Print comparison
    synthetic_params = {
        "skew_25d": {"mean": -0.05, "ar1_daily": 0.90, "description": "AR(1) mean=-5%"},
        "butterfly_25d": {"mean": 0.01, "description": "Derived from synthetic put/call"},
        "term_spread": {"mean": 0.02, "ar1_daily": 0.90, "description": "AR(1) mean=+2%"},
    }

    print(f"\n  {'Metric':<16s}  {'N':>5s}  {'Mean':>8s}  {'Std':>8s}  {'Min':>8s}  {'Max':>8s}  {'AR1(w)':>7s}")
    print(f"  {'-'*16}  {'-'*5}  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*7}")

    for field in ["iv_atm", "skew_25d", "butterfly_25d", "term_spread"]:
        s = stats.get(field)
        if not s:
            continue
        ar1 = f"{s['ar1_weekly']:.3f}" if s['ar1_weekly'] is not None else "  N/A"
        print(f"  {field:<16s}  {s['n']:5d}  {s['mean']:8.4f}  {s['std']:8.4f}  "
              f"{s['min']:8.4f}  {s['max']:8.4f}  {ar1:>7s}")

    print(f"\n  SYNTHETIC AR(1) ASSUMPTIONS:")
    for field, sp in synthetic_params.items():
        s = stats.get(field)
        if s:
            diff = s["mean"] - sp["mean"]
            print(f"    {field:<16s}  synthetic_mean={sp['mean']:+.4f}  "
                  f"real_mean={s['mean']:+.4f}  diff={diff:+.4f}  "
                  f"({sp['description']})")

    # Yearly breakdown
    print(f"\n  YEARLY BREAKDOWN:")
    by_year = defaultdict(list)
    for d in actual:
        yr = d["date"][:4]
        by_year[yr].append(d)

    print(f"  {'Year':<6s}  {'N':>4s}  {'ATM IV':>8s}  {'Skew25d':>8s}  {'Bfly25d':>8s}  {'TermSprd':>8s}")
    for yr in sorted(by_year.keys()):
        points = by_year[yr]
        n = len(points)
        atm = [p["iv_atm"] for p in points if p.get("iv_atm")]
        skew = [p["skew_25d"] for p in points if p.get("skew_25d")]
        bfly = [p["butterfly_25d"] for p in points if p.get("butterfly_25d")]
        term = [p["term_spread"] for p in points if p.get("term_spread")]

        atm_s = f"{sum(atm)/len(atm):.4f}" if atm else "   N/A "
        skew_s = f"{sum(skew)/len(skew):+.4f}" if skew else "   N/A "
        bfly_s = f"{sum(bfly)/len(bfly):+.4f}" if bfly else "   N/A "
        term_s = f"{sum(term)/len(term):+.4f}" if term else "   N/A "

        print(f"  {yr:<6s}  {n:4d}  {atm_s:>8s}  {skew_s:>8s}  {bfly_s:>8s}  {term_s:>8s}")

    return stats


def main():
    print("=" * 70)
    print("R50: REAL IV SURFACE RECONSTRUCTION")
    print("     from history.deribit.com option trades")
    print("=" * 70)
    print()
    print("Strategy: Weekly sampling (Wednesdays) → extract ATM IV, skew, butterfly")
    print("Source:   history.deribit.com — FREE option trades with IV since 2019")
    print()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    all_results = {}
    all_daily = {}
    all_stats = {}

    for currency in ["BTC", "ETH"]:
        print(f"\n{'='*70}")
        print(f"  {currency} — Fetching spot prices")
        print(f"{'='*70}")

        # Options started on Deribit around 2019-03 for BTC, later for ETH
        start = "2019-03-01" if currency == "BTC" else "2019-06-01"
        end = "2026-02-21"

        prices = fetch_spot_prices(currency, start, end)
        if not prices:
            print(f"  No prices for {currency}, skipping")
            continue

        print(f"\n  {currency} — Fetching weekly option trade surfaces")
        weekly = fetch_weekly_surfaces(currency, prices, start, end)

        if not weekly:
            print(f"  No surface data for {currency}")
            continue

        all_results[currency] = weekly

        # Save weekly raw data
        weekly_path = OUTPUT_DIR / f"{currency}_weekly_surface.json"
        with open(weekly_path, "w") as f:
            json.dump(weekly, f, indent=2)
        print(f"\n  Saved {len(weekly)} weekly points → {weekly_path}")

        # Interpolate to daily
        daily = interpolate_to_daily(weekly)
        all_daily[currency] = daily

        daily_path = OUTPUT_DIR / f"{currency}_daily_surface.json"
        with open(daily_path, "w") as f:
            json.dump(daily, f, indent=2)
        print(f"  Saved {len(daily)} daily points → {daily_path}")

        # Also save as CSV for easy analysis
        csv_path = OUTPUT_DIR / f"{currency}_daily_surface.csv"
        with open(csv_path, "w") as f:
            fields = ["date", "currency", "iv_atm", "skew_25d", "butterfly_25d",
                      "term_spread", "iv_25d_put", "iv_25d_call", "spot", "interpolated"]
            f.write(",".join(fields) + "\n")
            for d in daily:
                vals = [str(d.get(field, "")) for field in fields]
                f.write(",".join(vals) + "\n")
        print(f"  Saved CSV → {csv_path}")

        # Analyze real vs synthetic
        stats = analyze_real_vs_synthetic(daily, currency)
        all_stats[currency] = stats

    # ── Final summary ─────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("R50 FINAL SUMMARY")
    print(f"{'='*70}")
    print()

    for currency in ["BTC", "ETH"]:
        weekly = all_results.get(currency, [])
        daily = all_daily.get(currency, [])
        if weekly:
            actual = [d for d in weekly]
            skew_vals = [d["skew_25d"] for d in actual if d.get("skew_25d")]
            print(f"  {currency}: {len(weekly)} weekly samples → {len(daily)} daily points")
            if skew_vals:
                avg_skew = sum(skew_vals) / len(skew_vals)
                print(f"    Real avg skew_25d:      {avg_skew:+.4f} ({avg_skew*100:+.1f}%)")
                syn_mean = -0.05 if currency == "BTC" else -0.06
                print(f"    Synthetic assumption:   {syn_mean:+.4f} ({syn_mean*100:+.1f}%)")
                print(f"    Difference:             {avg_skew - syn_mean:+.4f}")
            print()

    # Save complete results
    summary_path = OUTPUT_DIR / "r50_complete_results.json"
    with open(summary_path, "w") as f:
        json.dump({
            "research_id": "R50",
            "title": "Real IV Surface Reconstruction from Deribit Option Trades",
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "source": "history.deribit.com (FREE, all option trades since 2019)",
            "method": "Weekly sampling (Wednesdays) → IV surface extraction → daily interpolation",
            "stats": all_stats,
            "n_weekly": {c: len(all_results.get(c, [])) for c in ["BTC", "ETH"]},
            "n_daily": {c: len(all_daily.get(c, [])) for c in ["BTC", "ETH"]},
        }, f, indent=2)
    print(f"  Complete results → {summary_path}")
    print()
    print("NEXT STEPS:")
    print("  1. Replace synthetic IV in backtests with real DVOL (R49) + real surface (R50)")
    print("  2. Re-run ensemble optimization with real data")
    print("  3. Compare real vs synthetic Sharpe ratios")


if __name__ == "__main__":
    main()
