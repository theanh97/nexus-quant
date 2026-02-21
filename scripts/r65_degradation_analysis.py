#!/usr/bin/env python3
"""
R65: 2025-2026 Degradation Deep-Dive
======================================

The R60 production backtest shows:
  2021: Sharpe 5.30  ★
  2022: Sharpe 4.08  ★
  2023: Sharpe 2.31
  2024: Sharpe 2.78
  2025: Sharpe 0.98
  2026: Sharpe -1.92  ✗

Questions:
  1. Is this structural (market maturation) or cyclical (will revert)?
  2. What has changed in the BTC options market?
  3. Is the VRP still positive? What about the butterfly?
  4. How does this compare to historical drawdown periods?
  5. What would a regime-aware trader do differently?
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

ROOT = Path(__file__).resolve().parents[1]


def load_dvol_daily(currency):
    path = ROOT / "data" / "cache" / "deribit" / "dvol" / f"{currency}_DVOL_12h.csv"
    daily = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            daily[row["date"][:10]] = float(row["dvol_close"]) / 100.0
    return daily


def load_prices(currency):
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
            if "result" in data:
                data = data["result"]
            if data.get("status") == "ok":
                for i, ts in enumerate(data["ticks"]):
                    dt = datetime.fromtimestamp(ts/1000, tz=timezone.utc)
                    prices[dt.strftime("%Y-%m-%d")] = data["close"][i]
        except:
            pass
        current = chunk_end
        time.sleep(0.1)
    return prices


def load_surface(currency):
    path = ROOT / "data" / "cache" / "deribit" / "real_surface" / f"{currency}_daily_surface.csv"
    data = {}
    if not path.exists():
        return data
    with open(path) as f:
        for row in csv.DictReader(f):
            d = row["date"]
            entry = {}
            for field in ["butterfly_25d", "iv_atm", "skew_25d", "term_spread"]:
                val = row.get(field, "")
                if val and val != "None":
                    entry[field] = float(val)
            if entry:
                data[d] = entry
    return data


def calc_stats(rets):
    if len(rets) < 10:
        return {"sharpe": 0, "ann_ret": 0, "max_dd": 0, "n": len(rets)}
    mean = sum(rets) / len(rets)
    var = sum((r - mean)**2 for r in rets) / len(rets)
    std = math.sqrt(var) if var > 0 else 1e-10
    sharpe = (mean * 365) / (std * math.sqrt(365))
    ann_ret = mean * 365
    cum = peak = max_dd = 0.0
    for r in rets:
        cum += r
        peak = max(peak, cum)
        max_dd = max(max_dd, peak - cum)
    win_rate = sum(1 for r in rets if r > 0) / len(rets)
    return {"sharpe": sharpe, "ann_ret": ann_ret, "max_dd": max_dd,
            "win_rate": win_rate, "n": len(rets)}


def vrp_pnl(dates, dvol, prices, lev=2.0):
    dt = 1.0 / 365.0
    pnl = {}
    for i in range(1, len(dates)):
        d, dp = dates[i], dates[i-1]
        iv, p0, p1 = dvol.get(dp), prices.get(dp), prices.get(d)
        if not all([iv, p0, p1]) or p0 <= 0:
            continue
        rv_bar = abs(math.log(p1 / p0)) * math.sqrt(365)
        pnl[d] = lev * 0.5 * (iv**2 - rv_bar**2) * dt
    return pnl


def rolling_zscore(values, dates, lookback):
    result = {}
    for i in range(lookback, len(dates)):
        d = dates[i]
        val = values.get(d)
        if val is None:
            continue
        window = [values.get(dates[j]) for j in range(i - lookback, i)]
        window = [v for v in window if v is not None]
        if len(window) < lookback // 2:
            continue
        mean = sum(window) / len(window)
        std = math.sqrt(sum((v - mean)**2 for v in window) / len(window))
        if std > 1e-8:
            result[d] = (val - mean) / std
    return result


def bf_pnl(dates, surface, dvol, lookback=120, z_entry=1.5):
    dt = 1.0 / 365.0
    bf_vals = {d: surface[d]["butterfly_25d"] for d in dates if d in surface and "butterfly_25d" in surface[d]}
    bf_z = rolling_zscore(bf_vals, dates, lookback)
    position = 0.0
    pnl = {}
    for i in range(1, len(dates)):
        d, dp = dates[i], dates[i-1]
        z = bf_z.get(d)
        iv = dvol.get(d)
        f_now, f_prev = bf_vals.get(d), bf_vals.get(dp)
        if z is not None:
            if z > z_entry: position = -1.0
            elif z < -z_entry: position = 1.0
            elif abs(z) < 0.3: position = 0.0
        if f_now is not None and f_prev is not None and iv is not None and position != 0:
            pnl[d] = position * (f_now - f_prev) * iv * math.sqrt(dt) * 2.5
        elif d in bf_z:
            pnl[d] = 0.0
    return pnl


def main():
    print("=" * 70)
    print("R65: 2025-2026 DEGRADATION DEEP-DIVE")
    print("=" * 70)

    # Load data
    print("\n  Loading data...")
    dvol = load_dvol_daily("BTC")
    prices = load_prices("BTC")
    surface = load_surface("BTC")
    dates = sorted(set(dvol.keys()) & set(prices.keys()))
    print(f"    {len(dates)} days, {dates[0]} to {dates[-1]}")

    # ═══════════════════════════════════════════════════════════════
    # 1. IV Level Timeline
    # ═══════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("  1. IV (DVOL) LEVEL TIMELINE")
    print("=" * 70)

    # Quarterly averages
    quarterly_iv = defaultdict(list)
    for d in dates:
        q = d[:4] + "Q" + str((int(d[5:7]) - 1) // 3 + 1)
        quarterly_iv[q].append(dvol[d])

    print(f"\n  {'Quarter':<10} {'Avg IV':>8} {'Min IV':>8} {'Max IV':>8}")
    for q in sorted(quarterly_iv):
        vals = quarterly_iv[q]
        print(f"  {q:<10} {sum(vals)/len(vals)*100:7.1f}% {min(vals)*100:7.1f}% {max(vals)*100:7.1f}%")

    # ═══════════════════════════════════════════════════════════════
    # 2. VRP Spread Timeline
    # ═══════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("  2. VRP SPREAD TIMELINE (IV² - RV²)")
    print("=" * 70)

    vrp_spreads = {}
    for i in range(1, len(dates)):
        d, dp = dates[i], dates[i-1]
        iv = dvol.get(dp)
        p0, p1 = prices.get(dp), prices.get(d)
        if all([iv, p0, p1]) and p0 > 0:
            rv = abs(math.log(p1 / p0)) * math.sqrt(365)
            vrp_spreads[d] = iv**2 - rv**2

    quarterly_vrp = defaultdict(list)
    for d, v in vrp_spreads.items():
        q = d[:4] + "Q" + str((int(d[5:7]) - 1) // 3 + 1)
        quarterly_vrp[q].append(v)

    print(f"\n  {'Quarter':<10} {'Avg VRP':>10} {'%Positive':>10} {'Avg IV':>8} {'N':>4}")
    for q in sorted(quarterly_vrp):
        vals = quarterly_vrp[q]
        pct_pos = sum(1 for v in vals if v > 0) / len(vals) * 100
        avg_iv = sum(quarterly_iv[q]) / len(quarterly_iv[q]) * 100 if q in quarterly_iv else 0
        print(f"  {q:<10} {sum(vals)/len(vals)*100:9.2f}% {pct_pos:8.1f}%  {avg_iv:6.1f}%  {len(vals):3}")

    # ═══════════════════════════════════════════════════════════════
    # 3. Realized Vol Analysis
    # ═══════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("  3. REALIZED VOLATILITY ANALYSIS")
    print("=" * 70)

    # Monthly RV (30d window)
    monthly_rv = defaultdict(list)
    for i in range(30, len(dates)):
        d = dates[i]
        ym = d[:7]
        rets_window = []
        for j in range(i-30, i):
            dp = dates[j]
            p0, p1 = prices.get(dp), prices.get(dates[j+1]) if j+1 < len(dates) else None
            if p0 and p1 and p0 > 0:
                rets_window.append(math.log(p1/p0))
        if rets_window:
            rv = math.sqrt(sum(r**2 for r in rets_window) / len(rets_window)) * math.sqrt(365)
            monthly_rv[ym].append(rv)

    print(f"\n  {'Month':<10} {'30d RV':>8} {'DVOL':>8} {'VRP':>8}")
    for ym in sorted(monthly_rv):
        rv_avg = sum(monthly_rv[ym]) / len(monthly_rv[ym])
        iv_vals = [dvol[d] for d in dates if d[:7] == ym and d in dvol]
        iv_avg = sum(iv_vals) / len(iv_vals) if iv_vals else 0
        vrp = iv_avg - rv_avg  # Simplified VRP = IV - RV
        if ym >= "2024-07":
            print(f"  {ym:<10} {rv_avg*100:7.1f}% {iv_avg*100:7.1f}% {vrp*100:+6.1f}%")

    # ═══════════════════════════════════════════════════════════════
    # 4. Component Attribution
    # ═══════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("  4. COMPONENT ATTRIBUTION (VRP vs Butterfly)")
    print("=" * 70)

    vrp_daily = vrp_pnl(dates, dvol, prices)
    bf_daily = bf_pnl(dates, surface, dvol)

    # Half-yearly comparison
    periods = [
        ("2024H1", "2024-01-01", "2024-06-30"),
        ("2024H2", "2024-07-01", "2024-12-31"),
        ("2025H1", "2025-01-01", "2025-06-30"),
        ("2025H2", "2025-07-01", "2025-12-31"),
        ("2026H1", "2026-01-01", "2026-06-30"),
    ]

    print(f"\n  {'Period':<10} {'VRP Sharpe':>10} {'BF Sharpe':>10} {'VRP+BF':>10} {'VRP Ret':>10} {'BF Ret':>10}")
    for name, start, end in periods:
        v_rets = [vrp_daily[d] for d in sorted(vrp_daily) if start <= d <= end]
        b_rets = [bf_daily[d] for d in sorted(bf_daily) if start <= d <= end]
        common = [d for d in sorted(set(vrp_daily.keys()) & set(bf_daily.keys())) if start <= d <= end]
        c_rets = [0.5 * vrp_daily[d] + 0.5 * bf_daily[d] for d in common]

        v_s = calc_stats(v_rets) if len(v_rets) > 30 else {"sharpe": 0, "ann_ret": 0}
        b_s = calc_stats(b_rets) if len(b_rets) > 30 else {"sharpe": 0, "ann_ret": 0}
        c_s = calc_stats(c_rets) if len(c_rets) > 30 else {"sharpe": 0, "ann_ret": 0}

        print(f"  {name:<10} {v_s['sharpe']:10.2f} {b_s['sharpe']:10.2f} {c_s['sharpe']:10.2f} "
              f"{v_s['ann_ret']*100:9.2f}% {b_s['ann_ret']*100:9.2f}%")

    # ═══════════════════════════════════════════════════════════════
    # 5. BTC Price Action Analysis
    # ═══════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("  5. BTC PRICE ACTION — TREND vs MEAN-REVERSION")
    print("=" * 70)

    # Monthly returns and vol of returns
    monthly_data = defaultdict(list)
    for i in range(1, len(dates)):
        d = dates[i]
        dp = dates[i-1]
        p0, p1 = prices.get(dp), prices.get(d)
        if p0 and p1 and p0 > 0:
            monthly_data[d[:7]].append(math.log(p1/p0))

    print(f"\n  {'Month':<10} {'Ret':>8} {'Vol':>8} {'Abs(Ret)/Vol':>12} {'Regime':>10}")
    for ym in sorted(monthly_data):
        if ym < "2024-07":
            continue
        rets = monthly_data[ym]
        monthly_ret = sum(rets) * 100
        daily_vol = math.sqrt(sum(r**2 for r in rets) / len(rets)) * math.sqrt(365) * 100
        # Trending ratio: |sum returns| / sum |returns|
        trend_ratio = abs(sum(rets)) / (sum(abs(r) for r in rets) + 1e-10)
        regime = "TREND" if trend_ratio > 0.3 else "MR" if trend_ratio < 0.1 else "MIXED"
        print(f"  {ym:<10} {monthly_ret:+7.1f}% {daily_vol:7.1f}% {trend_ratio:11.3f} {regime:>10}")

    # ═══════════════════════════════════════════════════════════════
    # 6. Surface Feature Evolution
    # ═══════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("  6. SURFACE FEATURE EVOLUTION")
    print("=" * 70)

    # Quarterly surface stats
    quarterly_bf = defaultdict(list)
    quarterly_ts = defaultdict(list)
    quarterly_skew = defaultdict(list)

    for d in dates:
        if d not in surface:
            continue
        q = d[:4] + "Q" + str((int(d[5:7]) - 1) // 3 + 1)
        s = surface[d]
        if "butterfly_25d" in s:
            quarterly_bf[q].append(s["butterfly_25d"])
        if "term_spread" in s:
            quarterly_ts[q].append(s["term_spread"])
        if "skew_25d" in s:
            quarterly_skew[q].append(s["skew_25d"])

    print(f"\n  {'Quarter':<10} {'Butterfly':>10} {'TermSpread':>10} {'Skew25d':>10}")
    for q in sorted(set(quarterly_bf.keys()) | set(quarterly_ts.keys()) | set(quarterly_skew.keys())):
        bf = sum(quarterly_bf[q]) / len(quarterly_bf[q]) * 100 if q in quarterly_bf else 0
        ts = sum(quarterly_ts[q]) / len(quarterly_ts[q]) * 100 if q in quarterly_ts else 0
        sk = sum(quarterly_skew[q]) / len(quarterly_skew[q]) * 100 if q in quarterly_skew else 0
        print(f"  {q:<10} {bf:9.2f}% {ts:9.2f}% {sk:9.2f}%")

    # ═══════════════════════════════════════════════════════════════
    # 7. Historical Drawdown Comparison
    # ═══════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("  7. HISTORICAL DRAWDOWNS")
    print("=" * 70)

    common_dates = sorted(set(vrp_daily.keys()) & set(bf_daily.keys()))
    combined_rets = {d: 0.5 * vrp_daily[d] + 0.5 * bf_daily[d] for d in common_dates}

    # Find all drawdowns
    cum = peak = 0.0
    drawdowns = []
    dd_start = None
    for d in common_dates:
        cum += combined_rets[d]
        if cum > peak:
            if dd_start is not None and (peak - cum_at_trough) > 0.005:
                drawdowns.append((dd_start, trough_date, peak - cum_at_trough))
            peak = cum
            dd_start = None
        else:
            if dd_start is None:
                dd_start = d
                cum_at_trough = cum
                trough_date = d
            elif peak - cum > peak - cum_at_trough:
                cum_at_trough = cum
                trough_date = d

    # Sort by depth
    drawdowns.sort(key=lambda x: -x[2])

    print(f"\n  Top Drawdowns:")
    print(f"  {'Start':<12} {'Trough':<12} {'Depth':>8}")
    for start, trough, depth in drawdowns[:10]:
        print(f"  {start:<12} {trough:<12} {depth*100:7.2f}%")

    # ═══════════════════════════════════════════════════════════════
    # 8. Structural Change Assessment
    # ═══════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("  8. STRUCTURAL CHANGE ASSESSMENT")
    print("=" * 70)

    # Compare early (2021-2023) vs late (2024-2026) periods
    early_vrp = [vrp_spreads[d] for d in sorted(vrp_spreads) if d < "2024-01-01"]
    late_vrp = [vrp_spreads[d] for d in sorted(vrp_spreads) if d >= "2024-01-01"]

    early_iv = [dvol[d] for d in dates if d < "2024-01-01"]
    late_iv = [dvol[d] for d in dates if d >= "2024-01-01"]

    print(f"\n  {'Metric':<25} {'2021-2023':>12} {'2024-2026':>12} {'Change':>10}")
    print(f"  {'Avg DVOL':<25} {sum(early_iv)/len(early_iv)*100:11.1f}% {sum(late_iv)/len(late_iv)*100:11.1f}% "
          f"{(sum(late_iv)/len(late_iv) - sum(early_iv)/len(early_iv))*100:+8.1f}%")
    print(f"  {'Avg VRP Spread':<25} {sum(early_vrp)/len(early_vrp)*100:11.2f}% {sum(late_vrp)/len(late_vrp)*100:11.2f}% "
          f"{(sum(late_vrp)/len(late_vrp) - sum(early_vrp)/len(early_vrp))*100:+8.2f}%")
    print(f"  {'VRP +ve %':<25} {sum(1 for v in early_vrp if v>0)/len(early_vrp)*100:10.1f}% "
          f"{sum(1 for v in late_vrp if v>0)/len(late_vrp)*100:10.1f}% "
          f"{(sum(1 for v in late_vrp if v>0)/len(late_vrp) - sum(1 for v in early_vrp if v>0)/len(early_vrp))*100:+8.1f}%")

    early_rets_v = [vrp_daily[d] for d in sorted(vrp_daily) if d < "2024-01-01"]
    late_rets_v = [vrp_daily[d] for d in sorted(vrp_daily) if d >= "2024-01-01"]
    e_stats = calc_stats(early_rets_v)
    l_stats = calc_stats(late_rets_v)
    print(f"  {'VRP Sharpe':<25} {e_stats['sharpe']:12.2f} {l_stats['sharpe']:12.2f} "
          f"{l_stats['sharpe']-e_stats['sharpe']:+9.2f}")

    # ═══════════════════════════════════════════════════════════════
    # CONCLUSION
    # ═══════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("R65: DEGRADATION ANALYSIS CONCLUSION")
    print("=" * 70)

    # Assess structural vs cyclical
    avg_early_iv = sum(early_iv) / len(early_iv)
    avg_late_iv = sum(late_iv) / len(late_iv)
    iv_decline_pct = (avg_late_iv - avg_early_iv) / avg_early_iv * 100

    avg_early_vrp = sum(early_vrp) / len(early_vrp)
    avg_late_vrp = sum(late_vrp) / len(late_vrp)

    print(f"\n  IV Decline: {avg_early_iv*100:.1f}% → {avg_late_iv*100:.1f}% ({iv_decline_pct:+.1f}%)")
    print(f"  VRP Decline: {avg_early_vrp*100:.2f}% → {avg_late_vrp*100:.2f}%")
    print(f"  VRP Sharpe: {e_stats['sharpe']:.2f} → {l_stats['sharpe']:.2f}")

    structural_factors = []
    cyclical_factors = []

    if iv_decline_pct < -20:
        structural_factors.append(f"IV structurally lower ({iv_decline_pct:+.0f}%) — market maturation")
    else:
        cyclical_factors.append("IV decline moderate — could be cyclical")

    if avg_late_vrp > 0:
        cyclical_factors.append(f"VRP still positive ({avg_late_vrp*100:.2f}%) — edge persists")
    else:
        structural_factors.append("VRP turned negative — edge may be gone")

    if l_stats["sharpe"] > 1.0:
        cyclical_factors.append(f"Recent Sharpe still above 1.0 — profitable")
    elif l_stats["sharpe"] > 0:
        cyclical_factors.append(f"Recent Sharpe positive but thin ({l_stats['sharpe']:.2f})")
    else:
        structural_factors.append(f"Recent Sharpe negative ({l_stats['sharpe']:.2f})")

    print(f"\n  STRUCTURAL factors ({len(structural_factors)}):")
    for f in structural_factors:
        print(f"    - {f}")
    print(f"\n  CYCLICAL factors ({len(cyclical_factors)}):")
    for f in cyclical_factors:
        print(f"    - {f}")

    if len(structural_factors) >= 2:
        verdict = "LIKELY STRUCTURAL — consider strategy review"
    elif len(structural_factors) == 1:
        verdict = "MIXED — some structural elements, but edge persists"
    else:
        verdict = "LIKELY CYCLICAL — current drawdown within historical norms"

    print(f"\n  VERDICT: {verdict}")

    # Save results
    results = {
        "research_id": "R65",
        "title": "2025-2026 Degradation Deep-Dive",
        "iv_decline": {"early_avg": round(avg_early_iv, 4), "late_avg": round(avg_late_iv, 4),
                       "pct_change": round(iv_decline_pct, 2)},
        "vrp_decline": {"early_avg": round(avg_early_vrp, 4), "late_avg": round(avg_late_vrp, 4)},
        "sharpe_decline": {"early": round(e_stats["sharpe"], 4), "late": round(l_stats["sharpe"], 4)},
        "structural_factors": structural_factors,
        "cyclical_factors": cyclical_factors,
        "verdict": verdict,
    }

    out = ROOT / "data" / "cache" / "deribit" / "real_surface" / "r65_degradation_results.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved to {out}")


if __name__ == "__main__":
    main()
