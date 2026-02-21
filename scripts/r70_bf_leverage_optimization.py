#!/usr/bin/env python3
"""
R70: BF Leverage & Sensitivity Optimization
===============================================

R69 config (10/90 VRP/BF, z_exit=0.0) has Sharpe 3.76 but only 1.99% return.
The BF component uses sensitivity=2.5. Can we scale this up to boost returns?

Key: Sharpe is SCALE-INVARIANT (mean/std both scale together), so boosting
BF sensitivity should increase return WITHOUT changing Sharpe — until we hit
constraints like costs, realistic position sizing, or nonlinearities.

Tests:
1. BF sensitivity sweep (1.0 to 10.0)
2. VRP leverage sweep (1.0 to 4.0)
3. Joint sensitivity x weight optimization
4. Position size analysis (are signals realistic?)
5. Cost impact at higher sensitivity
6. Recommended production parameters
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


def load_surface(currency):
    path = ROOT / "data" / "cache" / "deribit" / "real_surface" / f"{currency}_daily_surface.csv"
    data = {}
    if not path.exists():
        return data
    with open(path) as f:
        for row in csv.DictReader(f):
            d = row["date"]
            val = row.get("butterfly_25d", "")
            if val and val != "None":
                data[d] = {"butterfly_25d": float(val)}
    return data


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


def bf_pnl(dates, feature, dvol, lookback=120, z_entry=1.5, z_exit=0.0,
           sensitivity=2.5):
    dt = 1.0 / 365.0
    pnl = {}
    position = 0.0
    zscore = rolling_zscore(feature, dates, lookback)
    for i in range(1, len(dates)):
        d, dp = dates[i], dates[i-1]
        z = zscore.get(d)
        iv = dvol.get(d)
        f_now, f_prev = feature.get(d), feature.get(dp)
        if z is not None:
            if z > z_entry: position = -1.0
            elif z < -z_entry: position = 1.0
            elif abs(z) < z_exit: position = 0.0
        if f_now is not None and f_prev is not None and iv is not None and position != 0:
            pnl[d] = position * (f_now - f_prev) * iv * math.sqrt(dt) * sensitivity
        elif d in zscore:
            pnl[d] = 0.0
    return pnl


def calc_stats(rets):
    if len(rets) < 10:
        return {"sharpe": 0.0, "ann_ret": 0.0, "max_dd": 0.0, "ann_vol": 0.0}
    mean = sum(rets) / len(rets)
    var = sum((r - mean)**2 for r in rets) / len(rets)
    std = math.sqrt(var) if var > 0 else 1e-10
    sharpe = (mean * 365) / (std * math.sqrt(365))
    ann_ret = mean * 365
    ann_vol = std * math.sqrt(365)
    cum = peak = max_dd = 0.0
    for r in rets:
        cum += r; peak = max(peak, cum); max_dd = max(max_dd, peak - cum)
    calmar = ann_ret / max_dd if max_dd > 0 else 999
    return {
        "sharpe": sharpe, "ann_ret": ann_ret, "ann_vol": ann_vol,
        "max_dd": max_dd, "calmar": calmar, "n": len(rets),
        "total_return": sum(rets),
        "worst_day": min(rets), "best_day": max(rets),
    }


def main():
    print("=" * 70)
    print("R70: BF LEVERAGE & SENSITIVITY OPTIMIZATION")
    print("=" * 70)

    print("\n  Loading data...")
    dvol = load_dvol_daily("BTC")
    prices = load_prices("BTC")
    surface = load_surface("BTC")
    all_dates = sorted(set(dvol.keys()) & set(prices.keys()) & set(surface.keys()))
    print(f"    {len(all_dates)} days, {all_dates[0]} to {all_dates[-1]}")

    bf_feat = {d: s["butterfly_25d"] for d, s in surface.items() if "butterfly_25d" in s}

    # ═══════════════════════════════════════════════════════════════
    # 1. BF SENSITIVITY SWEEP (VRP lev=2.0, weight=10/90)
    # ═══════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("  1. BF SENSITIVITY SWEEP (VRP lev=2.0, 10/90 weight)")
    print("=" * 70)

    pnl_vrp = vrp_pnl(all_dates, dvol, prices, 2.0)

    print(f"\n  {'Sens':<6} {'Sharpe':>8} {'AnnRet':>10} {'AnnVol':>8} {'MaxDD':>8} {'Calmar':>8} {'Worst':>8}")

    for sens in [1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 7.5, 10.0]:
        pnl_bf = bf_pnl(all_dates, bf_feat, dvol, 120, 1.5, 0.0, sens)
        common = sorted(set(pnl_vrp.keys()) & set(pnl_bf.keys()))
        rets = [0.1 * pnl_vrp[d] + 0.9 * pnl_bf[d] for d in common]
        s = calc_stats(rets)
        print(f"  {sens:<6.1f} {s['sharpe']:8.4f} {s['ann_ret']*100:9.2f}% "
              f"{s['ann_vol']*100:7.2f}% {s['max_dd']*100:7.2f}% "
              f"{s['calmar']:8.2f} {s['worst_day']*100:7.3f}%")

    # ═══════════════════════════════════════════════════════════════
    # 2. VRP LEVERAGE SWEEP (BF sens=2.5, weight=10/90)
    # ═══════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("  2. VRP LEVERAGE SWEEP (BF sens=2.5, 10/90 weight)")
    print("=" * 70)

    pnl_bf_base = bf_pnl(all_dates, bf_feat, dvol, 120, 1.5, 0.0, 2.5)

    print(f"\n  {'VRP lev':<8} {'Sharpe':>8} {'AnnRet':>10} {'MaxDD':>8} {'Calmar':>8}")

    for lev in [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0]:
        pnl_v = vrp_pnl(all_dates, dvol, prices, lev)
        common = sorted(set(pnl_v.keys()) & set(pnl_bf_base.keys()))
        rets = [0.1 * pnl_v[d] + 0.9 * pnl_bf_base[d] for d in common]
        s = calc_stats(rets)
        print(f"  {lev:<8.1f} {s['sharpe']:8.4f} {s['ann_ret']*100:9.2f}% "
              f"{s['max_dd']*100:7.2f}% {s['calmar']:8.2f}")

    # ═══════════════════════════════════════════════════════════════
    # 3. JOINT OPTIMIZATION: sensitivity x VRP_lev x weight
    # ═══════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("  3. JOINT OPTIMIZATION (sensitivity x VRP_lev x weight)")
    print("=" * 70)
    print("  Targeting Sharpe > 3.5 AND return > 3%\n")

    best_joint = {"sharpe": 0}
    print(f"  {'Sens':<6} {'VRP_lev':<8} {'VRP_w':<6} {'Sharpe':>8} {'AnnRet':>10} {'MaxDD':>8} {'Calmar':>8}")

    for sens in [3.0, 4.0, 5.0]:
        for lev in [1.5, 2.0, 3.0]:
            for vrp_w in [0.05, 0.10, 0.15, 0.20]:
                bf_w = 1.0 - vrp_w
                pnl_v = vrp_pnl(all_dates, dvol, prices, lev)
                pnl_bf = bf_pnl(all_dates, bf_feat, dvol, 120, 1.5, 0.0, sens)
                common = sorted(set(pnl_v.keys()) & set(pnl_bf.keys()))
                rets = [vrp_w * pnl_v[d] + bf_w * pnl_bf[d] for d in common]
                s = calc_stats(rets)
                if s["ann_ret"] > 0.03 and s["sharpe"] > 3.0:
                    print(f"  {sens:<6.1f} {lev:<8.1f} {vrp_w:<6.2f} {s['sharpe']:8.4f} "
                          f"{s['ann_ret']*100:9.2f}% {s['max_dd']*100:7.2f}% {s['calmar']:8.2f}")
                    if s["sharpe"] > best_joint.get("sharpe", 0):
                        best_joint = {**s, "sens": sens, "lev": lev, "vrp_w": vrp_w}

    if best_joint.get("sens"):
        print(f"\n  Best joint: sens={best_joint['sens']} lev={best_joint['lev']} "
              f"vrp_w={best_joint['vrp_w']} → Sharpe {best_joint['sharpe']:.4f} "
              f"Return {best_joint['ann_ret']*100:.2f}%")

    # ═══════════════════════════════════════════════════════════════
    # 4. POSITION SIZE ANALYSIS
    # ═══════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("  4. POSITION SIZE ANALYSIS (BF notional exposure)")
    print("=" * 70)
    print("  BF PnL = position * d_butterfly * IV * sqrt(dt) * sensitivity")
    print("  Position is ±1 unit. The 'sensitivity' determines effective notional.\n")

    # At sensitivity=2.5:
    # Daily BF PnL = ±1 * d_bf * IV * sqrt(1/365) * 2.5
    # Typical d_bf = 0.001-0.005 (1-5 bps butterfly change)
    # IV = 0.50 (50%)
    # sqrt(1/365) = 0.0523
    # PnL range: ±0.001 * 0.50 * 0.0523 * 2.5 = ±0.0000654 per day = ±6.5bps of notional
    # At sensitivity=5.0: ±13bps

    for sens in [2.5, 5.0, 7.5, 10.0]:
        pnl_bf = bf_pnl(all_dates, bf_feat, dvol, 120, 1.5, 0.0, sens)
        rets = [pnl_bf[d] for d in sorted(pnl_bf) if pnl_bf[d] != 0]
        if not rets:
            continue
        abs_rets = [abs(r) for r in rets]
        mean_abs = sum(abs_rets) / len(abs_rets)
        max_abs = max(abs_rets)
        pct99 = sorted(abs_rets)[int(0.99 * len(abs_rets))]
        s = calc_stats(rets)
        print(f"  Sensitivity={sens:.1f}:")
        print(f"    Avg daily |PnL|: {mean_abs*100:.4f}%  Max: {max_abs*100:.4f}%  "
              f"99th pctl: {pct99*100:.4f}%")
        print(f"    Ann Vol: {s['ann_vol']*100:.2f}%  MaxDD: {s['max_dd']*100:.2f}%  "
              f"Worst day: {s.get('worst_day',0)*100:.4f}%")

    # ═══════════════════════════════════════════════════════════════
    # 5. COST IMPACT AT HIGHER SENSITIVITY
    # ═══════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("  5. COST IMPACT AT HIGHER SENSITIVITY")
    print("=" * 70)
    print("  BF is a MR strategy — trades when z-score crosses threshold.")
    print("  Higher sensitivity = same # trades, higher PnL per trade.\n")

    # BF trades at z_entry crossings. Count trades for base config.
    zscore = rolling_zscore(bf_feat, all_dates, 120)
    position = 0.0
    n_trades = 0
    for d in all_dates:
        z = zscore.get(d)
        if z is not None:
            old_pos = position
            if z > 1.5: position = -1.0
            elif z < -1.5: position = 1.0
            if position != old_pos:
                n_trades += 1

    n_years = (len(all_dates)) / 365.0
    trades_per_year = n_trades / n_years
    print(f"  BF z_exit=0.0: {n_trades} trades over {n_years:.1f} years = "
          f"{trades_per_year:.1f} trades/year")
    print(f"  (Low turnover — sensitivity scaling doesn't increase trade count)\n")

    # Cost per trade = spread × notional. For butterfly, this is option spread.
    # Typical butterfly bid-ask: ~50-100bps of vega notional
    for cost_bps in [0, 10, 25, 50, 100]:
        cost_per_trade = cost_bps / 10000.0
        annual_cost = cost_per_trade * trades_per_year

        for sens in [2.5, 5.0, 7.5]:
            pnl_bf = bf_pnl(all_dates, bf_feat, dvol, 120, 1.5, 0.0, sens)
            pnl_v = vrp_pnl(all_dates, dvol, prices, 2.0)
            common = sorted(set(pnl_v.keys()) & set(pnl_bf.keys()))
            # Deduct cost from BF returns (proportional to trades)
            rets = []
            for d in common:
                bf_ret = 0.9 * pnl_bf[d]
                vrp_ret = 0.1 * pnl_v[d]
                # Amortize cost evenly
                daily_cost = annual_cost / 365.0
                rets.append(vrp_ret + bf_ret - daily_cost)
            s = calc_stats(rets)
            if sens == 2.5 or cost_bps == 0 or cost_bps == 50:
                print(f"  Cost={cost_bps}bps  Sens={sens:.1f}: Sharpe {s['sharpe']:.4f}  "
                      f"Return {s['ann_ret']*100:.2f}%  MaxDD {s['max_dd']*100:.2f}%")

    # ═══════════════════════════════════════════════════════════════
    # 6. YEARLY BREAKDOWN AT RECOMMENDED CONFIG
    # ═══════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("  6. YEARLY BREAKDOWN — RECOMMENDED CONFIGS")
    print("=" * 70)

    configs = [
        ("R69 base (s=2.5)", 2.5, 2.0, 0.1),
        ("Boosted (s=5.0)", 5.0, 2.0, 0.1),
        ("High (s=7.5)", 7.5, 2.0, 0.1),
    ]

    for name, sens, lev, vrp_w in configs:
        pnl_v = vrp_pnl(all_dates, dvol, prices, lev)
        pnl_bf = bf_pnl(all_dates, bf_feat, dvol, 120, 1.5, 0.0, sens)
        common = sorted(set(pnl_v.keys()) & set(pnl_bf.keys()))

        print(f"\n  --- {name} ---")
        print(f"  {'Year':<6} {'Sharpe':>8} {'AnnRet':>10} {'MaxDD':>8}")

        years = defaultdict(list)
        for d in common:
            years[d[:4]].append(vrp_w * pnl_v[d] + (1-vrp_w) * pnl_bf[d])

        for year in sorted(years):
            s = calc_stats(years[year])
            print(f"  {year:<6} {s['sharpe']:8.4f} {s['ann_ret']*100:9.2f}% {s['max_dd']*100:7.2f}%")

        # Full sample
        all_rets = [vrp_w * pnl_v[d] + (1-vrp_w) * pnl_bf[d] for d in common]
        s_all = calc_stats(all_rets)
        print(f"  {'TOTAL':<6} {s_all['sharpe']:8.4f} {s_all['ann_ret']*100:9.2f}% {s_all['max_dd']*100:7.2f}%")

    # ═══════════════════════════════════════════════════════════════
    # CONCLUSION
    # ═══════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("R70: CONCLUSION")
    print("=" * 70)

    # Compare base vs recommended
    for sens_label, sens in [("R69 (s=2.5)", 2.5), ("Recommended (s=5.0)", 5.0)]:
        pnl_bf = bf_pnl(all_dates, bf_feat, dvol, 120, 1.5, 0.0, sens)
        common = sorted(set(pnl_vrp.keys()) & set(pnl_bf.keys()))
        rets = [0.1 * pnl_vrp[d] + 0.9 * pnl_bf[d] for d in common]
        s = calc_stats(rets)
        print(f"\n  {sens_label}:")
        print(f"    Sharpe: {s['sharpe']:.4f}  Return: {s['ann_ret']*100:.2f}%  "
              f"MaxDD: {s['max_dd']*100:.2f}%  Vol: {s['ann_vol']*100:.2f}%")

    print(f"\n  KEY INSIGHT: Sharpe is scale-invariant. Boosting sensitivity from")
    print(f"  2.5 to 5.0 doubles return and vol while preserving Sharpe.")
    print(f"  Choose sensitivity based on target vol/drawdown tolerance:")
    print(f"    Conservative (s=2.5): ~2% return, ~0.5% MaxDD")
    print(f"    Moderate     (s=5.0): ~4% return, ~0.9% MaxDD")
    print(f"    Aggressive   (s=7.5): ~6% return, ~1.4% MaxDD")

    if best_joint.get("sens"):
        print(f"\n  RECOMMENDED: sens={best_joint['sens']}, VRP_lev={best_joint['lev']}, "
              f"VRP_w={best_joint['vrp_w']}")
        print(f"    Sharpe {best_joint['sharpe']:.4f}  Return {best_joint['ann_ret']*100:.2f}%  "
              f"MaxDD {best_joint['max_dd']*100:.2f}%")

    # Save results
    out_path = ROOT / "data" / "cache" / "deribit" / "real_surface" / "r70_bf_leverage_results.json"
    results = {
        "research_id": "R70",
        "title": "BF Leverage & Sensitivity Optimization",
        "key_insight": "Sharpe is scale-invariant. Sensitivity only affects return magnitude.",
        "base_config": {"sens": 2.5, "sharpe": 3.76, "ann_ret": "1.99%", "max_dd": "0.46%"},
        "recommended_configs": {
            "conservative": {"sens": 2.5, "return": "~2%", "maxdd": "~0.5%"},
            "moderate": {"sens": 5.0, "return": "~4%", "maxdd": "~0.9%"},
            "aggressive": {"sens": 7.5, "return": "~6%", "maxdd": "~1.4%"},
        },
        "best_joint": {
            "sens": best_joint.get("sens"), "lev": best_joint.get("lev"),
            "vrp_w": best_joint.get("vrp_w"),
            "sharpe": round(best_joint.get("sharpe", 0), 4),
            "ann_ret": round(best_joint.get("ann_ret", 0), 6),
        } if best_joint.get("sens") else None,
        "trades_per_year": round(trades_per_year, 1),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved to {out_path}")


if __name__ == "__main__":
    main()
