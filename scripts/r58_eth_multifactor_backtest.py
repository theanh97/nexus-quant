#!/usr/bin/env python3
"""
R58: ETH Multi-Factor Backtest
================================

ETH surface data now available (351 weekly → 2451 daily).
R51 showed ETH VRP-only Sharpe=1.125 (weak, negative in 2025).

Questions:
1. Does ETH butterfly MR also provide alpha (like BTC)?
2. Can multi-factor improve ETH's weak VRP?
3. Does ETH add diversification to BTC portfolio?
4. What's the optimal BTC+ETH allocation?
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
from typing import Dict, List, Tuple, Optional

ROOT = Path(__file__).resolve().parents[1]


# ── Data Loading ─────────────────────────────────────────────────────

def load_dvol_daily(currency: str) -> Dict[str, float]:
    path = ROOT / "data" / "cache" / "deribit" / "dvol" / f"{currency}_DVOL_12h.csv"
    daily = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            daily[row["date"][:10]] = float(row["dvol_close"]) / 100.0
    return daily


def load_prices(currency: str) -> Dict[str, float]:
    prices = {}
    start_dt = datetime(2019, 1, 1, tzinfo=timezone.utc)
    end_dt = datetime(2026, 3, 1, tzinfo=timezone.utc)
    current = start_dt
    while current < end_dt:
        chunk_end = min(current + timedelta(days=365), end_dt)
        url = (f"https://www.deribit.com/api/v2/public/get_tradingview_chart_data?"
               f"instrument_name={currency}-PERPETUAL&resolution=1D"
               f"&start_timestamp={int(current.timestamp()*1000)}"
               f"&end_timestamp={int(chunk_end.timestamp()*1000)}")
        try:
            r = subprocess.run(["curl", "-s", "--max-time", "30", url],
                              capture_output=True, text=True, timeout=40)
            data = json.loads(r.stdout)
            if "result" in data: data = data["result"]
            if data.get("status") == "ok":
                for i, ts in enumerate(data["ticks"]):
                    dt = datetime.fromtimestamp(ts/1000, tz=timezone.utc)
                    prices[dt.strftime("%Y-%m-%d")] = data["close"][i]
        except:
            pass
        current = chunk_end
        time.sleep(0.1)
    return prices


def load_surface(currency: str) -> Dict[str, dict]:
    path = ROOT / "data" / "cache" / "deribit" / "real_surface" / f"{currency}_daily_surface.csv"
    data = {}
    if not path.exists():
        return data
    with open(path) as f:
        for row in csv.DictReader(f):
            d = row["date"]
            entry = {}
            for field in ["iv_atm", "skew_25d", "butterfly_25d", "term_spread"]:
                val = row.get(field, "")
                if val and val != "None":
                    entry[field] = float(val)
            if entry:
                data[d] = entry
    return data


# ── Strategy Models ──────────────────────────────────────────────────

def rolling_zscore(values: Dict[str, float], dates: List[str],
                    lookback: int) -> Dict[str, float]:
    result = {}
    for i in range(lookback, len(dates)):
        d = dates[i]
        val = values.get(d)
        if val is None:
            continue
        window = []
        for j in range(i - lookback, i):
            v = values.get(dates[j])
            if v is not None:
                window.append(v)
        if len(window) < lookback // 2:
            continue
        mean = sum(window) / len(window)
        std = math.sqrt(sum((v - mean)**2 for v in window) / len(window))
        if std > 1e-8:
            result[d] = (val - mean) / std
    return result


def vrp_pnl(dates: List[str], dvol: Dict[str, float],
            prices: Dict[str, float], lev: float = 2.0) -> Dict[str, float]:
    dt = 1.0 / 365.0
    pnl = {}
    for i in range(1, len(dates)):
        d = dates[i]
        dp = dates[i-1]
        iv = dvol.get(dp)
        p0 = prices.get(dp)
        p1 = prices.get(d)
        if not all([iv, p0, p1]) or p0 <= 0:
            continue
        log_ret = math.log(p1 / p0)
        rv_bar = abs(log_ret) * math.sqrt(365)
        pnl[d] = lev * 0.5 * (iv**2 - rv_bar**2) * dt
    return pnl


def mr_pnl(dates: List[str], feature: Dict[str, float],
           dvol: Dict[str, float], lookback: int = 120,
           z_entry: float = 1.5) -> Dict[str, float]:
    dt = 1.0 / 365.0
    pnl = {}
    position = 0.0
    zscore = rolling_zscore(feature, dates, lookback)

    for i in range(1, len(dates)):
        d = dates[i]
        dp = dates[i-1]
        z = zscore.get(d)
        iv = dvol.get(d)
        f_now = feature.get(d)
        f_prev = feature.get(dp)

        if z is not None:
            if z > z_entry:
                position = -1.0
            elif z < -z_entry:
                position = 1.0
            elif abs(z) < 0.3:
                position = 0.0

        if f_now is not None and f_prev is not None and iv is not None and position != 0:
            d_feat = f_now - f_prev
            pnl[d] = position * d_feat * iv * math.sqrt(dt) * 2.5
        else:
            if d in zscore:
                pnl[d] = 0.0

    return pnl


def calc_sharpe(rets: List[float]) -> Tuple[float, float, float]:
    if len(rets) < 20:
        return 0.0, 0.0, 0.0
    mean = sum(rets) / len(rets)
    var = sum((r - mean)**2 for r in rets) / len(rets)
    std = math.sqrt(var) if var > 0 else 1e-10
    sharpe = (mean * 365) / (std * math.sqrt(365))
    ann_ret = mean * 365
    cum = 0.0
    peak = 0.0
    max_dd = 0.0
    for r in rets:
        cum += r
        peak = max(peak, cum)
        max_dd = max(max_dd, peak - cum)
    return sharpe, ann_ret, max_dd


def correlation(x: List[float], y: List[float]) -> float:
    if len(x) != len(y) or len(x) < 10:
        return 0.0
    mx = sum(x) / len(x)
    my = sum(y) / len(y)
    num = sum((xi - mx) * (yi - my) for xi, yi in zip(x, y))
    dx = math.sqrt(sum((xi - mx)**2 for xi in x))
    dy = math.sqrt(sum((yi - my)**2 for yi in y))
    if dx < 1e-10 or dy < 1e-10:
        return 0.0
    return num / (dx * dy)


# ── Main ─────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("R58: ETH MULTI-FACTOR BACKTEST")
    print("=" * 70)
    print()

    # Load data for both assets
    for currency in ["BTC", "ETH"]:
        print(f"\nLoading {currency} data...")
        dvol = load_dvol_daily(currency)
        prices = load_prices(currency)
        surface = load_surface(currency)

        all_dates = sorted(set(dvol.keys()) & set(prices.keys()) & set(surface.keys()))
        print(f"  DVOL: {len(dvol)}, Prices: {len(prices)}, Surface: {len(surface)}")
        print(f"  Common: {len(all_dates)} ({all_dates[0] if all_dates else 'N/A'} to {all_dates[-1] if all_dates else 'N/A'})")

        if len(all_dates) < 200:
            print(f"  SKIPPING: not enough data")
            continue

        # Compute all factor PnL
        skew_feat = {d: s["skew_25d"] for d, s in surface.items() if "skew_25d" in s}
        bf_feat = {d: s["butterfly_25d"] for d, s in surface.items() if "butterfly_25d" in s}
        ts_feat = {d: s["term_spread"] for d, s in surface.items() if "term_spread" in s}

        pnl_vrp = vrp_pnl(all_dates, dvol, prices, 2.0)
        pnl_skew = mr_pnl(all_dates, skew_feat, dvol, 120, 1.5)
        pnl_bf = mr_pnl(all_dates, bf_feat, dvol, 120, 1.5)
        pnl_ts = mr_pnl(all_dates, ts_feat, dvol, 120, 1.5)

        factors = {"VRP": pnl_vrp, "Skew": pnl_skew, "Butterfly": pnl_bf, "TermSpread": pnl_ts}

        # Individual factor performance
        print(f"\n  --- {currency} Individual Factor Performance ---")
        print(f"  {'Factor':<14s}  {'Sharpe':>7s}  {'AnnRet':>8s}  {'MaxDD':>7s}  {'N':>5s}")
        for fn, pnl in factors.items():
            rets = [pnl[d] for d in sorted(pnl.keys())]
            sh, ret, dd = calc_sharpe(rets)
            print(f"  {fn:<14s}  {sh:7.2f}  {ret:+8.2%}  {dd:7.2%}  {len(rets):5d}")

        # PnL correlations
        common = sorted(set.intersection(*[set(f.keys()) for f in factors.values()]))
        if len(common) < 100:
            print(f"  Not enough common dates ({len(common)})")
            continue

        print(f"\n  --- {currency} PnL Correlations ({len(common)} common days) ---")
        fnames = list(factors.keys())
        print(f"  {'':>12s}", end="")
        for fn in fnames:
            print(f"  {fn:>12s}", end="")
        print()
        for fn1 in fnames:
            print(f"  {fn1:>12s}", end="")
            for fn2 in fnames:
                x = [factors[fn1].get(d, 0) for d in common]
                y = [factors[fn2].get(d, 0) for d in common]
                print(f"  {correlation(x, y):12.3f}", end="")
            print()

        # Fixed-weight ensembles
        print(f"\n  --- {currency} Ensemble Strategies ---")
        configs = [
            ("VRP-only",                    {"VRP": 1.0}),
            ("70% VRP + 30% BF",           {"VRP": 0.7, "Butterfly": 0.3}),
            ("50% VRP + 50% BF",           {"VRP": 0.5, "Butterfly": 0.5}),
            ("50% VRP + 30% BF + 20% Skew", {"VRP": 0.5, "Butterfly": 0.3, "Skew": 0.2}),
        ]

        print(f"  {'Config':<40s}  {'Sharpe':>7s}  {'AnnRet':>8s}  {'MaxDD':>7s}")
        for name, weights in configs:
            rets = [sum(w * factors[fn].get(d, 0) for fn, w in weights.items())
                    for d in common]
            sh, ret, dd = calc_sharpe(rets)
            print(f"  {name:<40s}  {sh:7.2f}  {ret:+8.2%}  {dd:7.2%}")

        # Yearly breakdown for VRP+BF(70/30)
        print(f"\n  --- {currency} Yearly Breakdown: VRP vs VRP+BF(70/30) ---")
        print(f"  {'Year':<6s}  {'VRP':>8s}  {'VRP+BF':>8s}  {'Δ':>6s}")
        for yr in sorted(set(d[:4] for d in common)):
            yr_dates = [d for d in common if d[:4] == yr]
            if len(yr_dates) < 20:
                continue
            vrp_yr = [factors["VRP"].get(d, 0) for d in yr_dates]
            bf_yr = [0.7 * factors["VRP"].get(d, 0) + 0.3 * factors["Butterfly"].get(d, 0)
                     for d in yr_dates]
            sh_v, _, _ = calc_sharpe(vrp_yr)
            sh_b, _, _ = calc_sharpe(bf_yr)
            print(f"  {yr:<6s}  {sh_v:8.2f}  {sh_b:8.2f}  {sh_b-sh_v:+6.2f}")

    # ── Cross-Asset Correlation ──────────────────────────────────────
    print(f"\n{'='*70}")
    print("  CROSS-ASSET CORRELATION: BTC vs ETH")
    print(f"{'='*70}")
    print()

    # Load both
    btc_dvol = load_dvol_daily("BTC")
    btc_prices = load_prices("BTC")
    btc_surface = load_surface("BTC")
    eth_dvol = load_dvol_daily("ETH")
    eth_prices = load_prices("ETH")
    eth_surface = load_surface("ETH")

    btc_dates = sorted(set(btc_dvol.keys()) & set(btc_prices.keys()) & set(btc_surface.keys()))
    eth_dates = sorted(set(eth_dvol.keys()) & set(eth_prices.keys()) & set(eth_surface.keys()))
    cross_dates = sorted(set(btc_dates) & set(eth_dates))
    print(f"  Common BTC+ETH dates: {len(cross_dates)}")

    if len(cross_dates) >= 200:
        # BTC factors
        btc_bf = {d: s["butterfly_25d"] for d, s in btc_surface.items() if "butterfly_25d" in s}
        btc_vrp = vrp_pnl(cross_dates, btc_dvol, btc_prices, 2.0)
        btc_bfm = mr_pnl(cross_dates, btc_bf, btc_dvol, 120, 1.5)

        # ETH factors
        eth_bf = {d: s["butterfly_25d"] for d, s in eth_surface.items() if "butterfly_25d" in s}
        eth_vrp = vrp_pnl(cross_dates, eth_dvol, eth_prices, 2.0)
        eth_bfm = mr_pnl(cross_dates, eth_bf, eth_dvol, 120, 1.5)

        # Cross-asset PnL correlation
        cross_common = sorted(set(btc_vrp.keys()) & set(eth_vrp.keys()) &
                             set(btc_bfm.keys()) & set(eth_bfm.keys()))
        if len(cross_common) >= 100:
            streams = {
                "BTC_VRP": btc_vrp, "BTC_BF": btc_bfm,
                "ETH_VRP": eth_vrp, "ETH_BF": eth_bfm,
            }
            snames = list(streams.keys())
            print(f"\n  Cross-Asset PnL Correlations ({len(cross_common)} days):")
            print(f"  {'':>10s}", end="")
            for sn in snames:
                print(f"  {sn:>10s}", end="")
            print()
            for sn1 in snames:
                print(f"  {sn1:>10s}", end="")
                for sn2 in snames:
                    x = [streams[sn1].get(d, 0) for d in cross_common]
                    y = [streams[sn2].get(d, 0) for d in cross_common]
                    print(f"  {correlation(x, y):10.3f}", end="")
                print()

            # Cross-asset portfolio optimization
            print(f"\n  Cross-Asset Portfolio Strategies:")
            cross_configs = [
                ("BTC-only (VRP+BF 70/30)",
                 {"BTC_VRP": 0.7, "BTC_BF": 0.3}),
                ("ETH-only (VRP+BF 70/30)",
                 {"ETH_VRP": 0.7, "ETH_BF": 0.3}),
                ("70% BTC + 30% ETH (VRP+BF each)",
                 {"BTC_VRP": 0.49, "BTC_BF": 0.21, "ETH_VRP": 0.21, "ETH_BF": 0.09}),
                ("80% BTC + 20% ETH (VRP+BF each)",
                 {"BTC_VRP": 0.56, "BTC_BF": 0.24, "ETH_VRP": 0.14, "ETH_BF": 0.06}),
                ("BTC VRP + BTC BF + ETH VRP (40/30/30)",
                 {"BTC_VRP": 0.4, "BTC_BF": 0.3, "ETH_VRP": 0.3}),
            ]

            print(f"  {'Config':<50s}  {'Sharpe':>7s}  {'AnnRet':>8s}  {'MaxDD':>7s}")
            for name, weights in cross_configs:
                rets = [sum(w * streams[fn].get(d, 0) for fn, w in weights.items())
                        for d in cross_common]
                sh, ret, dd = calc_sharpe(rets)
                print(f"  {name:<50s}  {sh:7.2f}  {ret:+8.2%}  {dd:7.2%}")

    # ── Summary ──────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("R58 SUMMARY")
    print(f"{'='*70}")

    # Save results
    results = {
        "research_id": "R58",
        "title": "ETH Multi-Factor Backtest + Cross-Asset Portfolio",
    }
    outpath = ROOT / "data" / "cache" / "deribit" / "real_surface" / "r58_eth_multifactor_results.json"
    with open(outpath, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved to {outpath}")


if __name__ == "__main__":
    main()
