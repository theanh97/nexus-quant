#!/usr/bin/env python3
"""
R52: VRP + Skew Degradation Analysis (2025-2026)
==================================================

Both VRP and Skew showing weakening performance in 2025.
This study investigates:
1. Is VRP edge disappearing? Rolling VRP analysis
2. Is IV structurally lower? (making VRP edge thinner)
3. Is skew less mean-reverting? Rolling autocorrelation
4. Parameter sensitivity on recent data
5. Regime detection: are we in a new market regime?
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
from typing import Dict, List, Tuple

ROOT = Path(__file__).resolve().parents[1]


def load_dvol_daily(currency: str) -> Dict[str, float]:
    path = ROOT / "data" / "cache" / "deribit" / "dvol" / f"{currency}_DVOL_12h.csv"
    daily = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            daily[row["date"][:10]] = float(row["dvol_close"]) / 100.0
    return daily


def load_prices(currency: str) -> Dict[str, float]:
    """Load prices via API."""
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
        except: pass
        current = chunk_end
        time.sleep(0.1)
    return prices


def load_skew(currency: str) -> Dict[str, float]:
    path = ROOT / "data" / "cache" / "deribit" / "real_surface" / f"{currency}_daily_surface.csv"
    data = {}
    if not path.exists():
        return data
    with open(path) as f:
        for row in csv.DictReader(f):
            val = row.get("skew_25d", "")
            if val and val != "None":
                data[row["date"]] = float(val)
    return data


def rolling_stat(values: list, window: int) -> list:
    """Rolling mean."""
    result = [None] * len(values)
    for i in range(window - 1, len(values)):
        chunk = values[i - window + 1:i + 1]
        result[i] = sum(chunk) / len(chunk)
    return result


def main():
    print("=" * 70)
    print("R52: VRP + SKEW DEGRADATION ANALYSIS")
    print("=" * 70)
    print()

    dvol_btc = load_dvol_daily("BTC")
    dvol_eth = load_dvol_daily("ETH")
    print(f"  Loaded BTC DVOL: {len(dvol_btc)} days")
    print(f"  Loaded ETH DVOL: {len(dvol_eth)} days")

    prices_btc = load_prices("BTC")
    prices_eth = load_prices("ETH")
    print(f"  Loaded BTC prices: {len(prices_btc)} days")
    print(f"  Loaded ETH prices: {len(prices_eth)} days")

    skew_btc = load_skew("BTC")
    print(f"  Loaded BTC skew: {len(skew_btc)} days")

    for currency, dvol, prices, skew in [
        ("BTC", dvol_btc, prices_btc, skew_btc),
        ("ETH", dvol_eth, prices_eth, {}),
    ]:
        print(f"\n{'='*70}")
        print(f"  {currency} DEGRADATION ANALYSIS")
        print(f"{'='*70}")

        # ── 1. Rolling VRP (IV - RV) ─────────────────────────────
        dates = sorted(set(dvol.keys()) & set(prices.keys()))
        if len(dates) < 30:
            continue

        # Compute daily RV
        daily_vrp = {}  # date -> iv - rv_30d
        for i in range(30, len(dates)):
            d = dates[i]
            iv = dvol.get(d, 0)

            # 30-day realized vol
            rets = []
            for j in range(i - 29, i + 1):
                p0 = prices.get(dates[j - 1])
                p1 = prices.get(dates[j])
                if p0 and p1 and p0 > 0:
                    rets.append(math.log(p1 / p0))
            if len(rets) >= 20:
                rv = math.sqrt(sum(r**2 for r in rets) / len(rets)) * math.sqrt(365)
                daily_vrp[d] = iv - rv

        print(f"\n  --- Rolling VRP (IV - RV_30d) ---")
        by_quarter = defaultdict(list)
        for d, vrp in sorted(daily_vrp.items()):
            yr = d[:4]
            mo = int(d[5:7])
            q = f"{yr}Q{(mo-1)//3 + 1}"
            by_quarter[q].append(vrp)

        print(f"  {'Quarter':<8s}  {'VRP mean':>10s}  {'VRP std':>8s}  {'VRP>0%':>8s}  {'N':>4s}")
        for q in sorted(by_quarter.keys()):
            vals = by_quarter[q]
            mean = sum(vals) / len(vals)
            std = math.sqrt(sum((v - mean)**2 for v in vals) / len(vals))
            pct_pos = sum(1 for v in vals if v > 0) / len(vals)
            print(f"  {q:<8s}  {mean:10.4f}  {std:8.4f}  {pct_pos:8.1%}  {len(vals):4d}")

        # ── 2. IV Level Analysis ─────────────────────────────────
        print(f"\n  --- IV Level Analysis ---")
        by_half = defaultdict(list)
        for d in sorted(dvol.keys()):
            yr = d[:4]
            h = "H1" if int(d[5:7]) <= 6 else "H2"
            by_half[f"{yr}{h}"].append(dvol[d])

        print(f"  {'Period':<8s}  {'IV mean':>8s}  {'IV min':>8s}  {'IV max':>8s}")
        for p in sorted(by_half.keys()):
            vals = by_half[p]
            print(f"  {p:<8s}  {sum(vals)/len(vals):8.1%}  {min(vals):8.1%}  {max(vals):8.1%}")

        # ── 3. VRP PnL by quarter (rolling Sharpe) ───────────────
        print(f"\n  --- VRP PnL Analysis by Quarter ---")
        dt = 1.0 / 365.0
        pnl_by_q = defaultdict(list)
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
            vrp_pnl = 1.5 * 0.5 * (iv**2 - rv_bar**2) * dt

            yr = d[:4]
            mo = int(d[5:7])
            q = f"{yr}Q{(mo-1)//3 + 1}"
            pnl_by_q[q].append(vrp_pnl)

        print(f"  {'Quarter':<8s}  {'Sharpe':>8s}  {'AnnRet':>8s}  {'AnnVol':>8s}  {'MaxLoss':>8s}")
        for q in sorted(pnl_by_q.keys()):
            rets = pnl_by_q[q]
            if len(rets) < 10:
                continue
            m = sum(rets) / len(rets)
            v = sum((r-m)**2 for r in rets) / len(rets)
            s = math.sqrt(v) if v > 0 else 1e-10
            sharpe = (m * 365) / (s * math.sqrt(365))
            ann_ret = m * 365
            ann_vol = s * math.sqrt(365)
            max_loss = min(rets)
            print(f"  {q:<8s}  {sharpe:8.2f}  {ann_ret:8.2%}  {ann_vol:8.2%}  {max_loss:8.4f}")

        # ── 4. Skew Analysis (BTC only) ──────────────────────────
        if skew and currency == "BTC":
            print(f"\n  --- Skew Mean-Reversion Analysis ---")
            skew_dates = sorted(skew.keys())

            # Rolling 60-day autocorrelation of skew changes
            print(f"\n  Rolling 60-day skew change autocorrelation:")
            by_half_ac = defaultdict(list)
            for i in range(61, len(skew_dates)):
                d = skew_dates[i]
                window_changes = []
                for j in range(i-59, i+1):
                    s0 = skew.get(skew_dates[j-1])
                    s1 = skew.get(skew_dates[j])
                    if s0 is not None and s1 is not None:
                        window_changes.append(s1 - s0)

                if len(window_changes) >= 30:
                    # Autocorrelation of changes (negative = mean-reverting)
                    m = sum(window_changes) / len(window_changes)
                    dm = [c - m for c in window_changes]
                    if len(dm) >= 2:
                        num = sum(dm[k]*dm[k-1] for k in range(1, len(dm)))
                        den = sum(d**2 for d in dm)
                        ac1 = num / den if den > 0 else 0

                        yr = d[:4]
                        h = "H1" if int(d[5:7]) <= 6 else "H2"
                        by_half_ac[f"{yr}{h}"].append(ac1)

            print(f"  {'Period':<8s}  {'AC(1) mean':>10s}  {'N':>5s}  {'Interpretation':>20s}")
            for p in sorted(by_half_ac.keys()):
                vals = by_half_ac[p]
                m = sum(vals) / len(vals)
                interp = "MEAN-REVERTING" if m < -0.1 else ("WEAK MR" if m < 0 else "NOT MR")
                print(f"  {p:<8s}  {m:10.4f}  {len(vals):5d}  {interp:>20s}")

            # Skew level by quarter
            print(f"\n  Skew level by quarter:")
            skew_by_q = defaultdict(list)
            for d, s in skew.items():
                yr = d[:4]
                mo = int(d[5:7])
                q = f"{yr}Q{(mo-1)//3 + 1}"
                skew_by_q[q].append(s)

            print(f"  {'Quarter':<8s}  {'Mean':>8s}  {'Std':>8s}  {'Min':>8s}  {'Max':>8s}")
            for q in sorted(skew_by_q.keys()):
                vals = skew_by_q[q]
                m = sum(vals)/len(vals)
                s = math.sqrt(sum((v-m)**2 for v in vals)/len(vals))
                print(f"  {q:<8s}  {m:+8.4f}  {s:8.4f}  {min(vals):+8.4f}  {max(vals):+8.4f}")

    # ── Summary ───────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("R52 DEGRADATION ANALYSIS SUMMARY")
    print(f"{'='*70}")
    print()
    print("Key questions and answers:")
    print("  1. Is VRP edge disappearing?")
    print("     → Look at quarterly VRP (IV-RV) trend above")
    print("  2. Is IV structurally lower?")
    print("     → Look at IV level by half-year above")
    print("  3. Is skew less mean-reverting?")
    print("     → Look at autocorrelation analysis above")
    print("  4. What should we do?")
    print("     → See yearly Sharpe trends in R51 results")


if __name__ == "__main__":
    main()
