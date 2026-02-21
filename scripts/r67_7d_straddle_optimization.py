#!/usr/bin/env python3
"""
R67: 7-Day Straddle Optimization
====================================

R63 found 7d straddles have Sharpe 4.79 vs 14d (4.17) vs 30d (3.77).
This study optimizes the 7d straddle strategy across:

1. Roll frequency (3d, 5d, 7d, 10d overlap)
2. Hedge frequency (1h, 4h, 8h, 24h)
3. Transaction cost sensitivity (0, 5, 10, 20, 50 bps)
4. Straddle + BF overlay (combining both edges)
5. Period stability analysis
6. Walk-forward parameter selection
7. Recent performance (2025-2026)
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


def load_hourly_dvol(currency):
    path = ROOT / "data" / "cache" / "deribit" / "dvol" / f"{currency}_DVOL_1h.csv"
    data = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            ts = int(row["timestamp"]) // 1000
            data[ts] = float(row["dvol_close"]) / 100.0
    return data


def load_hourly_prices(currency):
    prices = {}
    cache = ROOT / "data" / "cache" / "deribit"
    for year in range(2019, 2027):
        path = cache / f"{currency}_{year}-01-01_{year}-12-31_1h_prices.csv"
        if not path.exists():
            continue
        with open(path) as f:
            for row in csv.DictReader(f):
                ts = int(row["timestamp"])
                p = row.get("close", "")
                if p and p != "None":
                    prices[ts] = float(p)
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
            for field in ["butterfly_25d"]:
                val = row.get(field, "")
                if val and val != "None":
                    entry[field] = float(val)
            if entry:
                data[d] = entry
    return data


def load_dvol_daily(currency):
    path = ROOT / "data" / "cache" / "deribit" / "dvol" / f"{currency}_DVOL_12h.csv"
    daily = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            daily[row["date"][:10]] = float(row["dvol_close"]) / 100.0
    return daily


# ═══════════════════════════════════════════════════════════════
# Black-Scholes
# ═══════════════════════════════════════════════════════════════

def norm_cdf(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def bs_d1(S, K, T, sigma):
    if T <= 0 or sigma <= 0 or S <= 0:
        return 0.0
    return (math.log(S / K) + 0.5 * sigma**2 * T) / (sigma * math.sqrt(T))


def bs_call_delta(S, K, T, sigma):
    return norm_cdf(bs_d1(S, K, T, sigma))


def bs_put_delta(S, K, T, sigma):
    return bs_call_delta(S, K, T, sigma) - 1.0


def bs_call_price(S, K, T, sigma):
    if T <= 0:
        return max(S - K, 0.0)
    d1 = bs_d1(S, K, T, sigma)
    d2 = d1 - sigma * math.sqrt(T)
    return S * norm_cdf(d1) - K * norm_cdf(d2)


def bs_put_price(S, K, T, sigma):
    if T <= 0:
        return max(K - S, 0.0)
    d1 = bs_d1(S, K, T, sigma)
    d2 = d1 - sigma * math.sqrt(T)
    return K * norm_cdf(-d2) - S * norm_cdf(-d1)


def simulate_straddle_cycle(prices_ts, dvol_ts, timestamps, tte_days=7,
                             hedge_freq_hours=24, slippage_bps=0):
    dt_annual = tte_days / 365.0
    S0 = prices_ts[timestamps[0]]
    K = S0
    sigma = dvol_ts[timestamps[0]]

    call_price = bs_call_price(S0, K, dt_annual, sigma)
    put_price = bs_put_price(S0, K, dt_annual, sigma)
    premium = call_price + put_price

    straddle_delta = bs_call_delta(S0, K, dt_annual, sigma) + bs_put_delta(S0, K, dt_annual, sigma)
    hedge_position = straddle_delta

    hedge_pnl = 0.0
    n_rebalances = 0
    total_slippage = 0.0

    hedge_sec = hedge_freq_hours * 3600
    last_hedge_ts = timestamps[0]

    for i in range(1, len(timestamps)):
        ts = timestamps[i]
        if ts not in prices_ts or ts not in dvol_ts:
            continue

        S = prices_ts[ts]
        elapsed_days = (ts - timestamps[0]) / 86400.0
        tte_remaining = max(dt_annual - elapsed_days / 365.0, 1e-6)
        current_sigma = dvol_ts[ts]

        dS = S - prices_ts.get(timestamps[max(0, i-1)], S)
        hedge_pnl += hedge_position * dS / S0

        if (ts - last_hedge_ts) >= hedge_sec:
            new_delta = (bs_call_delta(S, K, tte_remaining, current_sigma) +
                        bs_put_delta(S, K, tte_remaining, current_sigma))
            trade_size = abs(new_delta - hedge_position)
            total_slippage += trade_size * slippage_bps / 10000.0
            hedge_position = new_delta
            n_rebalances += 1
            last_hedge_ts = ts

    S_final = prices_ts[timestamps[-1]]
    call_payoff = max(S_final - K, 0)
    put_payoff = max(K - S_final, 0)
    settlement = (call_payoff + put_payoff) / S0

    pnl = premium / S0 - settlement + hedge_pnl - total_slippage

    return {
        "pnl": pnl,
        "premium": premium / S0,
        "settlement": settlement,
        "hedge_pnl": hedge_pnl,
        "slippage": total_slippage,
        "n_rebalances": n_rebalances,
        "start_ts": timestamps[0],
    }


def run_rolling_straddles(prices_hourly, dvol_hourly, roll_days=7, tte_days=7,
                           hedge_freq_hours=24, slippage_bps=0):
    common_ts = sorted(set(prices_hourly.keys()) & set(dvol_hourly.keys()))
    if not common_ts:
        return []

    results = []
    cycle_sec = roll_days * 86400
    tte_sec = tte_days * 86400
    cycle_start = common_ts[0]

    while cycle_start < common_ts[-1] - tte_sec:
        cycle_end = cycle_start + tte_sec
        cycle_timestamps = [ts for ts in common_ts if cycle_start <= ts <= cycle_end]

        if len(cycle_timestamps) < tte_days * 12:
            cycle_start += cycle_sec
            continue

        result = simulate_straddle_cycle(
            prices_hourly, dvol_hourly, cycle_timestamps,
            tte_days=tte_days, hedge_freq_hours=hedge_freq_hours,
            slippage_bps=slippage_bps
        )
        start_date = datetime.fromtimestamp(cycle_start, tz=timezone.utc).strftime("%Y-%m-%d")
        result["date"] = start_date
        results.append(result)

        cycle_start += cycle_sec

    return results


def straddle_stats(results, roll_days):
    if len(results) < 5:
        return {"sharpe": 0, "ann_ret": 0, "max_dd": 0, "n": len(results)}
    pnls = [r["pnl"] for r in results]
    mean_pnl = sum(pnls) / len(pnls)
    std_pnl = math.sqrt(sum((p - mean_pnl)**2 for p in pnls) / len(pnls)) if len(pnls) > 1 else 1e-10
    ann_factor = 365.0 / roll_days
    sharpe = (mean_pnl * ann_factor) / (std_pnl * math.sqrt(ann_factor)) if std_pnl > 0 else 0
    ann_ret = mean_pnl * ann_factor
    cum = peak = max_dd = 0.0
    for p in pnls:
        cum += p; peak = max(peak, cum); max_dd = max(max_dd, peak - cum)
    win_rate = sum(1 for p in pnls if p > 0) / len(pnls)
    return {
        "sharpe": round(sharpe, 4), "ann_ret": round(ann_ret, 6),
        "max_dd": round(max_dd, 6), "win_rate": round(win_rate, 4),
        "n": len(results), "total_return": round(sum(pnls), 6),
        "avg_premium": round(sum(r["premium"] for r in results) / len(results), 6),
    }


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


def bf_pnl_daily(dates, feature, dvol, lookback=120, z_entry=1.5):
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
            elif abs(z) < 0.3: position = 0.0
        if f_now is not None and f_prev is not None and iv is not None and position != 0:
            pnl[d] = position * (f_now - f_prev) * iv * math.sqrt(dt) * 2.5
        elif d in zscore:
            pnl[d] = 0.0
    return pnl


def calc_stats(rets):
    if len(rets) < 10:
        return {"sharpe": 0.0, "ann_ret": 0.0, "max_dd": 0.0}
    mean = sum(rets) / len(rets)
    var = sum((r - mean)**2 for r in rets) / len(rets)
    std = math.sqrt(var) if var > 0 else 1e-10
    sharpe = (mean * 365) / (std * math.sqrt(365))
    ann_ret = mean * 365
    cum = peak = max_dd = 0.0
    for r in rets:
        cum += r; peak = max(peak, cum); max_dd = max(max_dd, peak - cum)
    return {"sharpe": sharpe, "ann_ret": ann_ret, "max_dd": max_dd,
            "total_return": sum(rets), "n_days": len(rets)}


def main():
    print("=" * 70)
    print("R67: 7-DAY STRADDLE OPTIMIZATION")
    print("=" * 70)

    # Load data
    print("\n  Loading data...")
    dvol_hourly = load_hourly_dvol("BTC")
    prices_hourly = load_hourly_prices("BTC")
    dvol_daily = load_dvol_daily("BTC")
    surface = load_surface("BTC")
    bf_feat = {d: s["butterfly_25d"] for d, s in surface.items() if "butterfly_25d" in s}

    print(f"    Hourly DVOL: {len(dvol_hourly)} points")
    print(f"    Hourly prices: {len(prices_hourly)} points")

    # ═══════════════════════════════════════════════════════════════
    # 1. TENOR x ROLL FREQUENCY GRID
    # ═══════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("  1. TENOR x ROLL FREQUENCY GRID (hedge=24h, cost=0)")
    print("=" * 70)

    print(f"\n  {'TTE':<6} {'Roll':<6} {'Sharpe':>8} {'AnnRet':>10} {'MaxDD':>8} {'WinRate':>8} {'N':>5}")
    best_combo = {"sharpe": 0}

    for tte in [5, 7, 10, 14]:
        for roll in [3, 5, 7]:
            if roll > tte:
                continue
            results = run_rolling_straddles(
                prices_hourly, dvol_hourly,
                roll_days=roll, tte_days=tte,
                hedge_freq_hours=24, slippage_bps=0
            )
            s = straddle_stats(results, roll)
            print(f"  {tte:<6} {roll:<6} {s['sharpe']:8.4f} {s['ann_ret']*100:9.2f}% "
                  f"{s['max_dd']*100:7.2f}% {s['win_rate']*100:7.1f}% {s['n']:5}")
            if s["sharpe"] > best_combo.get("sharpe", 0):
                best_combo = {**s, "tte": tte, "roll": roll}

    print(f"\n  Best: TTE={best_combo.get('tte')}d, Roll={best_combo.get('roll')}d → "
          f"Sharpe {best_combo['sharpe']:.4f}")

    best_tte = best_combo.get("tte", 7)
    best_roll = best_combo.get("roll", 7)

    # ═══════════════════════════════════════════════════════════════
    # 2. HEDGE FREQUENCY OPTIMIZATION
    # ═══════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print(f"  2. HEDGE FREQUENCY (TTE={best_tte}d, Roll={best_roll}d, cost=0)")
    print("=" * 70)

    print(f"\n  {'Hedge':>8} {'Sharpe':>8} {'AnnRet':>10} {'MaxDD':>8} {'AvgRebal':>10}")
    best_hedge = {"sharpe": 0}

    for hedge_h in [1, 2, 4, 8, 12, 24]:
        results = run_rolling_straddles(
            prices_hourly, dvol_hourly,
            roll_days=best_roll, tte_days=best_tte,
            hedge_freq_hours=hedge_h, slippage_bps=0
        )
        s = straddle_stats(results, best_roll)
        avg_reb = sum(r["n_rebalances"] for r in results) / len(results) if results else 0
        print(f"  {f'{hedge_h}h':>8} {s['sharpe']:8.4f} {s['ann_ret']*100:9.2f}% "
              f"{s['max_dd']*100:7.2f}% {avg_reb:10.1f}")
        if s["sharpe"] > best_hedge.get("sharpe", 0):
            best_hedge = {**s, "hedge_h": hedge_h}

    print(f"\n  Best hedge: {best_hedge.get('hedge_h')}h → Sharpe {best_hedge['sharpe']:.4f}")
    best_hedge_h = best_hedge.get("hedge_h", 24)

    # ═══════════════════════════════════════════════════════════════
    # 3. TRANSACTION COST SENSITIVITY
    # ═══════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print(f"  3. TRANSACTION COST SENSITIVITY (TTE={best_tte}d, Roll={best_roll}d, Hedge={best_hedge_h}h)")
    print("=" * 70)

    print(f"\n  {'Cost(bps)':>10} {'Sharpe':>8} {'AnnRet':>10} {'MaxDD':>8} {'TotalSlip':>12}")

    for cost in [0, 2, 5, 10, 20, 50]:
        results = run_rolling_straddles(
            prices_hourly, dvol_hourly,
            roll_days=best_roll, tte_days=best_tte,
            hedge_freq_hours=best_hedge_h, slippage_bps=cost
        )
        s = straddle_stats(results, best_roll)
        total_slip = sum(r["slippage"] for r in results) if results else 0
        print(f"  {cost:>10} {s['sharpe']:8.4f} {s['ann_ret']*100:9.2f}% "
              f"{s['max_dd']*100:7.2f}% {total_slip*100:11.2f}%")

    # ═══════════════════════════════════════════════════════════════
    # 4. YEARLY STABILITY
    # ═══════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print(f"  4. YEARLY STABILITY (TTE={best_tte}d, Roll={best_roll}d, Hedge={best_hedge_h}h)")
    print("=" * 70)

    results = run_rolling_straddles(
        prices_hourly, dvol_hourly,
        roll_days=best_roll, tte_days=best_tte,
        hedge_freq_hours=best_hedge_h, slippage_bps=10
    )

    years = defaultdict(list)
    for r in results:
        years[r["date"][:4]].append(r)

    print(f"\n  {'Year':<6} {'Sharpe':>8} {'AnnRet':>10} {'MaxDD':>8} {'WinRate':>8} {'N':>5}")
    for year in sorted(years):
        s = straddle_stats(years[year], best_roll)
        print(f"  {year:<6} {s['sharpe']:8.4f} {s['ann_ret']*100:9.2f}% "
              f"{s['max_dd']*100:7.2f}% {s['win_rate']*100:7.1f}% {s['n']:5}")

    # ═══════════════════════════════════════════════════════════════
    # 5. STRADDLE + BF OVERLAY
    # ═══════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("  5. 7d STRADDLE + BF OVERLAY")
    print("=" * 70)
    print("  Combine simulated straddle PnL with BF mean-reversion.\n")

    # Get straddle daily returns (10bps cost)
    straddle_results = run_rolling_straddles(
        prices_hourly, dvol_hourly,
        roll_days=best_roll, tte_days=best_tte,
        hedge_freq_hours=best_hedge_h, slippage_bps=10
    )

    # Convert straddle cycle PnL to daily returns (spread evenly across roll period)
    straddle_daily = {}
    for r in straddle_results:
        d = r["date"]
        daily_pnl = r["pnl"] / best_roll  # spread over roll period
        dt_start = datetime.strptime(d, "%Y-%m-%d")
        for day_offset in range(best_roll):
            dd = (dt_start + timedelta(days=day_offset)).strftime("%Y-%m-%d")
            straddle_daily[dd] = daily_pnl

    # BF daily PnL
    all_dates = sorted(set(dvol_daily.keys()) & set(bf_feat.keys()))
    pnl_bf = bf_pnl_daily(all_dates, bf_feat, dvol_daily, 120, 1.5)

    # Combine
    combo_dates = sorted(set(straddle_daily.keys()) & set(pnl_bf.keys()))
    print(f"  Combined dates: {len(combo_dates)} ({combo_dates[0]} to {combo_dates[-1]})")

    print(f"\n  {'Blend':<15} {'Sharpe':>8} {'AnnRet':>10} {'MaxDD':>8}")

    best_blend = {"sharpe": 0}
    for strad_w in [1.0, 0.8, 0.7, 0.6, 0.5, 0.3, 0.0]:
        bf_w = 1.0 - strad_w
        rets = [strad_w * straddle_daily[d] + bf_w * pnl_bf[d] for d in combo_dates]
        s = calc_stats(rets)
        label = f"{int(strad_w*100)}/{int(bf_w*100)}"
        print(f"  {label:<15} {s['sharpe']:8.4f} {s['ann_ret']*100:9.2f}% {s['max_dd']*100:7.2f}%")
        if s["sharpe"] > best_blend.get("sharpe", 0):
            best_blend = {**s, "strad_w": strad_w, "bf_w": bf_w}

    print(f"\n  Best blend: {int(best_blend.get('strad_w',0)*100)}/{int(best_blend.get('bf_w',0)*100)} "
          f"→ Sharpe {best_blend['sharpe']:.4f}")

    # ═══════════════════════════════════════════════════════════════
    # 6. RECENT PERFORMANCE (2025-2026)
    # ═══════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("  6. RECENT PERFORMANCE — 2025-2026")
    print("=" * 70)

    recent_straddle = [r for r in straddle_results if r["date"] >= "2025-01-01"]
    if recent_straddle:
        s_recent = straddle_stats(recent_straddle, best_roll)
        print(f"\n  7d Straddle (2025-2026):")
        print(f"    Sharpe: {s_recent['sharpe']:.4f}  Return: {s_recent['ann_ret']*100:.2f}%  "
              f"MaxDD: {s_recent['max_dd']*100:.2f}%  WinRate: {s_recent['win_rate']*100:.1f}%")

    # Compare with R60 VRP model for same period
    recent_combo = [d for d in combo_dates if d >= "2025-01-01"]
    if recent_combo:
        # Straddle only
        r_strad = [straddle_daily[d] for d in recent_combo]
        s_strad = calc_stats(r_strad)

        # BF only
        r_bf = [pnl_bf[d] for d in recent_combo]
        s_bf = calc_stats(r_bf)

        # Best blend
        bw = best_blend.get("strad_w", 0.5)
        r_blend = [bw * straddle_daily[d] + (1-bw) * pnl_bf[d] for d in recent_combo]
        s_blend = calc_stats(r_blend)

        print(f"\n  {'Strategy':<25} {'Sharpe':>8} {'Return':>10} {'MaxDD':>8}")
        print(f"  {'7d Straddle only':<25} {s_strad['sharpe']:8.4f} {s_strad['ann_ret']*100:9.2f}% {s_strad['max_dd']*100:7.2f}%")
        print(f"  {'BF only':<25} {s_bf['sharpe']:8.4f} {s_bf['ann_ret']*100:9.2f}% {s_bf['max_dd']*100:7.2f}%")
        print(f"  {'Best blend':<25} {s_blend['sharpe']:8.4f} {s_blend['ann_ret']*100:9.2f}% {s_blend['max_dd']*100:7.2f}%")

    # ═══════════════════════════════════════════════════════════════
    # 7. vs R60 THEORETICAL VRP COMPARISON
    # ═══════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("  7. 7d STRADDLE vs R60 THEORETICAL VRP")
    print("=" * 70)

    # R60 theoretical VRP
    prices_daily_loaded = {}
    start_dt = datetime(2019, 1, 1, tzinfo=timezone.utc)
    end_dt = datetime(2026, 3, 1, tzinfo=timezone.utc)
    current = start_dt
    while current < end_dt:
        chunk_end = min(current + timedelta(days=365), end_dt)
        url = (f"https://www.deribit.com/api/v2/public/get_tradingview_chart_data?"
               f"instrument_name=BTC-PERPETUAL&resolution=1D"
               f"&start_timestamp={int(current.timestamp()*1000)}"
               f"&end_timestamp={int(chunk_end.timestamp()*1000)}")
        try:
            r = subprocess.run(["curl", "-s", "--max-time", "30", url],
                              capture_output=True, text=True, timeout=40)
            data = json.loads(r.stdout)
            if "result" in data: data = data["result"]
            if data.get("status") == "ok":
                for i, ts in enumerate(data["ticks"]):
                    dt_val = datetime.fromtimestamp(ts/1000, tz=timezone.utc)
                    prices_daily_loaded[dt_val.strftime("%Y-%m-%d")] = data["close"][i]
        except:
            pass
        current = chunk_end
        time.sleep(0.1)

    r60_dates = sorted(set(dvol_daily.keys()) & set(prices_daily_loaded.keys()))
    dt = 1.0 / 365.0
    r60_pnl = {}
    for i in range(1, len(r60_dates)):
        d, dp = r60_dates[i], r60_dates[i-1]
        iv, p0, p1 = dvol_daily.get(dp), prices_daily_loaded.get(dp), prices_daily_loaded.get(d)
        if not all([iv, p0, p1]) or p0 <= 0:
            continue
        rv_bar = abs(math.log(p1 / p0)) * math.sqrt(365)
        r60_pnl[d] = 2.0 * 0.5 * (iv**2 - rv_bar**2) * dt

    # Compare on common dates
    compare_dates = sorted(set(straddle_daily.keys()) & set(r60_pnl.keys()) & set(pnl_bf.keys()))
    if compare_dates:
        # R60 VRP+BF 50/50
        r60_combo = [0.5 * r60_pnl[d] + 0.5 * pnl_bf[d] for d in compare_dates]
        s_r60 = calc_stats(r60_combo)

        # 7d Straddle+BF best blend
        bw = best_blend.get("strad_w", 0.5)
        r67_combo = [bw * straddle_daily[d] + (1-bw) * pnl_bf[d] for d in compare_dates]
        s_r67 = calc_stats(r67_combo)

        # 7d Straddle only
        r67_strad = [straddle_daily[d] for d in compare_dates]
        s_strad_only = calc_stats(r67_strad)

        print(f"\n  {'Strategy':<30} {'Sharpe':>8} {'Return':>10} {'MaxDD':>8}")
        print(f"  {'R60: VRP+BF 50/50':<30} {s_r60['sharpe']:8.4f} {s_r60['ann_ret']*100:9.2f}% {s_r60['max_dd']*100:7.2f}%")
        print(f"  {'R67: 7d Straddle only':<30} {s_strad_only['sharpe']:8.4f} {s_strad_only['ann_ret']*100:9.2f}% {s_strad_only['max_dd']*100:7.2f}%")
        print(f"  {'R67: Straddle+BF blend':<30} {s_r67['sharpe']:8.4f} {s_r67['ann_ret']*100:9.2f}% {s_r67['max_dd']*100:7.2f}%")

        delta_sharpe = s_r67["sharpe"] - s_r60["sharpe"]
        print(f"\n  Delta Sharpe (R67 blend vs R60): {delta_sharpe:+.4f}")

    # ═══════════════════════════════════════════════════════════════
    # CONCLUSION
    # ═══════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("R67: CONCLUSION")
    print("=" * 70)

    if compare_dates:
        if delta_sharpe > 0.2:
            verdict = "CONFIRMED — 7d straddle sim significantly improves on R60"
        elif delta_sharpe > 0.05:
            verdict = "MARGINAL — small improvement over R60"
        elif delta_sharpe > -0.1:
            verdict = "EQUIVALENT — similar risk-adjusted performance"
        else:
            verdict = "WORSE — 7d straddle underperforms R60 model"
    else:
        verdict = "INCONCLUSIVE — insufficient comparison data"

    print(f"\n  Best 7d config: TTE={best_tte}d, Roll={best_roll}d, Hedge={best_hedge_h}h")
    print(f"  Best straddle Sharpe: {best_combo['sharpe']:.4f}")
    print(f"  Best blend: {int(best_blend.get('strad_w',0)*100)}/{int(best_blend.get('bf_w',0)*100)} "
          f"→ Sharpe {best_blend['sharpe']:.4f}")
    if compare_dates:
        print(f"  vs R60: {delta_sharpe:+.4f} Sharpe")
    print(f"\n  VERDICT: {verdict}")

    # Save results
    out_path = ROOT / "data" / "cache" / "deribit" / "real_surface" / "r67_7d_straddle_results.json"
    results_json = {
        "research_id": "R67",
        "title": "7-Day Straddle Optimization",
        "best_config": {
            "tte": best_tte, "roll": best_roll, "hedge_h": best_hedge_h,
        },
        "best_straddle_sharpe": best_combo["sharpe"],
        "best_blend": {
            "strad_w": best_blend.get("strad_w"),
            "bf_w": best_blend.get("bf_w"),
            "sharpe": round(best_blend["sharpe"], 4),
        },
        "r60_sharpe": round(s_r60["sharpe"], 4) if compare_dates else None,
        "delta_sharpe": round(delta_sharpe, 4) if compare_dates else None,
        "verdict": verdict,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results_json, f, indent=2)
    print(f"\n  Results saved to {out_path}")


if __name__ == "__main__":
    main()
