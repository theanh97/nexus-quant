#!/usr/bin/env python3
"""
R63: Delta Hedging Simulation with Real Market Data
=====================================================

The VRP model assumes continuous delta hedging: PnL = 0.5*(IV²-RV²)*dt.
In practice, we hedge discretely. This study quantifies:

1. Hedging error at various frequencies (1h, 4h, 8h, daily)
2. Slippage and funding cost impact
3. PnL distribution: model vs simulated
4. Gamma PnL decomposition
5. Straddle implementation simulation

Using real hourly BTC prices and DVOL data.

Key question: How much does discrete hedging erode the theoretical VRP edge?
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


def load_hourly_dvol(currency: str) -> Dict[int, float]:
    path = ROOT / "data" / "cache" / "deribit" / "dvol" / f"{currency}_DVOL_1h.csv"
    data = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            ts = int(row["timestamp"]) // 1000
            data[ts] = float(row["dvol_close"]) / 100.0
    return data


def load_hourly_prices(currency: str) -> Dict[int, float]:
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


def load_daily_dvol(currency: str) -> Dict[str, float]:
    path = ROOT / "data" / "cache" / "deribit" / "dvol" / f"{currency}_DVOL_12h.csv"
    daily = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            daily[row["date"][:10]] = float(row["dvol_close"]) / 100.0
    return daily


def load_daily_prices(currency: str) -> Dict[str, float]:
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


def calc_stats(rets):
    if len(rets) < 10:
        return {"sharpe": 0, "ann_ret": 0, "max_dd": 0, "n": len(rets)}
    mean = sum(rets) / len(rets)
    var = sum((r - mean)**2 for r in rets) / len(rets)
    std = math.sqrt(var) if var > 0 else 1e-10
    sharpe = (mean * 365) / (std * math.sqrt(365))
    ann_ret = mean * 365
    ann_vol = std * math.sqrt(365)
    cum = peak = max_dd = 0.0
    for r in rets:
        cum += r
        peak = max(peak, cum)
        max_dd = max(max_dd, peak - cum)
    win_rate = sum(1 for r in rets if r > 0) / len(rets)
    calmar = ann_ret / max_dd if max_dd > 0 else 999
    return {
        "sharpe": sharpe, "ann_ret": ann_ret, "ann_vol": ann_vol,
        "max_dd": max_dd, "win_rate": win_rate, "n": len(rets),
        "calmar": calmar, "total_return": sum(rets),
    }


# ═══════════════════════════════════════════════════════════════
# Black-Scholes Greeks for delta hedging simulation
# ═══════════════════════════════════════════════════════════════

def bs_d1(S, K, T, sigma, r=0.0):
    if T <= 0 or sigma <= 0 or S <= 0:
        return 0.0
    return (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))


def norm_cdf(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def bs_call_delta(S, K, T, sigma, r=0.0):
    d1 = bs_d1(S, K, T, sigma, r)
    return norm_cdf(d1)


def bs_put_delta(S, K, T, sigma, r=0.0):
    return bs_call_delta(S, K, T, sigma, r) - 1.0


def bs_gamma(S, K, T, sigma, r=0.0):
    if T <= 0 or sigma <= 0 or S <= 0:
        return 0.0
    d1 = bs_d1(S, K, T, sigma, r)
    return math.exp(-d1**2 / 2.0) / (S * sigma * math.sqrt(2 * math.pi * T))


def bs_theta(S, K, T, sigma, r=0.0):
    """Theta per calendar day (negative for long options)."""
    if T <= 0 or sigma <= 0 or S <= 0:
        return 0.0
    d1 = bs_d1(S, K, T, sigma, r)
    pdf_d1 = math.exp(-d1**2 / 2.0) / math.sqrt(2 * math.pi)
    # Theta for call (same for ATM straddle per unit)
    theta = -(S * sigma * pdf_d1) / (2 * math.sqrt(T))
    return theta / 365.0  # per calendar day


def bs_call_price(S, K, T, sigma, r=0.0):
    if T <= 0:
        return max(S - K, 0.0)
    d1 = bs_d1(S, K, T, sigma, r)
    d2 = d1 - sigma * math.sqrt(T)
    return S * norm_cdf(d1) - K * math.exp(-r * T) * norm_cdf(d2)


def bs_put_price(S, K, T, sigma, r=0.0):
    if T <= 0:
        return max(K - S, 0.0)
    d1 = bs_d1(S, K, T, sigma, r)
    d2 = d1 - sigma * math.sqrt(T)
    return K * math.exp(-r * T) * norm_cdf(-d2) - S * norm_cdf(-d1)


# ═══════════════════════════════════════════════════════════════
# Straddle Simulation
# ═══════════════════════════════════════════════════════════════

def simulate_straddle_cycle(prices_ts, dvol_ts, timestamps, tte_days=30,
                             hedge_freq_hours=24, slippage_bps=0, funding_rate_daily=0):
    """
    Simulate a single short straddle cycle:
    1. Sell ATM straddle at entry
    2. Delta-hedge at hedge_freq_hours intervals
    3. Settle at expiry

    Returns: dict with PnL components
    """
    dt_annual = tte_days / 365.0
    S0 = prices_ts[timestamps[0]]
    K = S0  # ATM
    sigma = dvol_ts[timestamps[0]]

    # Premium collected (short straddle)
    call_price = bs_call_price(S0, K, dt_annual, sigma)
    put_price = bs_put_price(S0, K, dt_annual, sigma)
    premium = call_price + put_price

    # Initial delta (straddle delta ≈ call_delta + put_delta)
    straddle_delta = bs_call_delta(S0, K, dt_annual, sigma) + bs_put_delta(S0, K, dt_annual, sigma)
    hedge_position = straddle_delta  # We're SHORT the straddle, so we BUY delta

    # Track PnL components
    hedge_pnl = 0.0
    n_rebalances = 0
    total_slippage = 0.0
    total_funding = 0.0

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

        # Mark-to-market of hedge position (we're LONG hedge_position shares)
        dS = S - prices_ts.get(timestamps[max(0, i-1)], S)
        hedge_pnl += hedge_position * dS / S0  # normalized by initial spot

        # Funding cost on hedge position
        daily_funding = abs(hedge_position) * funding_rate_daily / 365.0
        hours_since_last = (ts - timestamps[max(0, i-1)]) / 3600.0
        total_funding += daily_funding * (hours_since_last / 24.0)

        # Rebalance delta at frequency
        if (ts - last_hedge_ts) >= hedge_sec:
            new_delta = (bs_call_delta(S, K, tte_remaining, current_sigma) +
                        bs_put_delta(S, K, tte_remaining, current_sigma))
            trade_size = abs(new_delta - hedge_position)
            slippage = trade_size * slippage_bps / 10000.0
            total_slippage += slippage
            hedge_position = new_delta
            n_rebalances += 1
            last_hedge_ts = ts

    # Expiry settlement
    S_final = prices_ts[timestamps[-1]]
    call_payoff = max(S_final - K, 0)
    put_payoff = max(K - S_final, 0)
    settlement = (call_payoff + put_payoff) / S0  # normalized

    # Total PnL for short straddle + hedge
    pnl = premium / S0 - settlement + hedge_pnl - total_slippage - total_funding

    return {
        "pnl": pnl,
        "premium": premium / S0,
        "settlement": settlement,
        "hedge_pnl": hedge_pnl,
        "slippage": total_slippage,
        "funding": total_funding,
        "n_rebalances": n_rebalances,
    }


def run_rolling_straddles(prices_hourly, dvol_hourly, roll_days=7, tte_days=30,
                           hedge_freq_hours=24, slippage_bps=0, funding_rate=0):
    """
    Run rolling short straddles:
    - Every roll_days, sell a new tte_days straddle
    - Delta-hedge at hedge_freq_hours
    """
    # Get sorted timestamps
    common_ts = sorted(set(prices_hourly.keys()) & set(dvol_hourly.keys()))
    if not common_ts:
        return []

    results = []
    cycle_sec = roll_days * 86400
    tte_sec = tte_days * 86400

    # Start first cycle
    cycle_start = common_ts[0]

    while cycle_start < common_ts[-1] - tte_sec:
        cycle_end = cycle_start + tte_sec
        cycle_timestamps = [ts for ts in common_ts if cycle_start <= ts <= cycle_end]

        if len(cycle_timestamps) < tte_days * 12:  # Need at least half the hours
            cycle_start += cycle_sec
            continue

        result = simulate_straddle_cycle(
            prices_hourly, dvol_hourly, cycle_timestamps,
            tte_days=tte_days, hedge_freq_hours=hedge_freq_hours,
            slippage_bps=slippage_bps, funding_rate_daily=funding_rate
        )
        start_date = datetime.fromtimestamp(cycle_start, tz=timezone.utc).strftime("%Y-%m-%d")
        result["date"] = start_date
        results.append(result)

        cycle_start += cycle_sec

    return results


def main():
    print("=" * 70)
    print("R63: DELTA HEDGING SIMULATION WITH REAL DATA")
    print("=" * 70)

    # Load data
    print("\n  Loading data...")
    dvol_hourly = load_hourly_dvol("BTC")
    prices_hourly = load_hourly_prices("BTC")
    dvol_daily = load_daily_dvol("BTC")
    prices_daily = load_daily_prices("BTC")

    print(f"    Hourly DVOL: {len(dvol_hourly)} points")
    print(f"    Hourly prices: {len(prices_hourly)} points")
    print(f"    Daily DVOL: {len(dvol_daily)} points")
    print(f"    Daily prices: {len(prices_daily)} points")

    # ═══════════════════════════════════════════════════════════════
    # TEST 1: Theoretical vs simulated PnL comparison
    # ═══════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("  TEST 1: THEORETICAL (R60) vs SIMULATED STRADDLE PnL")
    print("=" * 70)

    # Theoretical daily VRP (R60 model)
    dt = 1.0 / 365.0
    dates = sorted(set(dvol_daily.keys()) & set(prices_daily.keys()))
    theoretical_pnl = {}
    for i in range(1, len(dates)):
        d, dp = dates[i], dates[i-1]
        iv, p0, p1 = dvol_daily.get(dp), prices_daily.get(dp), prices_daily.get(d)
        if not all([iv, p0, p1]) or p0 <= 0:
            continue
        rv_bar = abs(math.log(p1 / p0)) * math.sqrt(365)
        theoretical_pnl[d] = 2.0 * 0.5 * (iv**2 - rv_bar**2) * dt

    theoretical_rets = [theoretical_pnl[d] for d in sorted(theoretical_pnl)]
    t_stats = calc_stats(theoretical_rets)

    print(f"\n  Theoretical VRP (R60 model):")
    print(f"    Sharpe:     {t_stats['sharpe']:.2f}")
    print(f"    Ann Return: {t_stats['ann_ret']*100:.2f}%")
    print(f"    Max DD:     {t_stats['max_dd']*100:.2f}%")

    # Simulated straddles at various hedge frequencies
    print(f"\n  Running simulated straddles (30d, weekly roll)...")

    for hedge_h, label in [(1, "1h"), (4, "4h"), (8, "8h"), (24, "24h"), (168, "Weekly")]:
        results = run_rolling_straddles(
            prices_hourly, dvol_hourly,
            roll_days=7, tte_days=30,
            hedge_freq_hours=hedge_h
        )
        if not results:
            continue

        # Convert to daily-equivalent returns (annualize from 7-day returns)
        pnls = [r["pnl"] for r in results]
        premiums = [r["premium"] for r in results]
        settlements = [r["settlement"] for r in results]
        hedge_pnls = [r["hedge_pnl"] for r in results]
        slippages = [r["slippage"] for r in results]
        n_rebs = [r["n_rebalances"] for r in results]

        # Stats on cycle-level returns
        mean_pnl = sum(pnls) / len(pnls)
        std_pnl = math.sqrt(sum((p - mean_pnl)**2 for p in pnls) / len(pnls)) if len(pnls) > 1 else 1
        ann_factor = 365.0 / 7.0  # 52 cycles per year
        sharpe_sim = (mean_pnl * ann_factor) / (std_pnl * math.sqrt(ann_factor)) if std_pnl > 0 else 0
        ann_ret_sim = mean_pnl * ann_factor
        win_rate = sum(1 for p in pnls if p > 0) / len(pnls)

        print(f"\n  ═══ Hedge freq: {label} ═══")
        print(f"    Sharpe:       {sharpe_sim:.2f}")
        print(f"    Ann Return:   {ann_ret_sim*100:.2f}%")
        print(f"    Win Rate:     {win_rate*100:.1f}%")
        print(f"    Cycles:       {len(results)}")
        print(f"    Avg premium:  {sum(premiums)/len(premiums)*100:.2f}%")
        print(f"    Avg settle:   {sum(settlements)/len(settlements)*100:.2f}%")
        print(f"    Avg hedge PnL:{sum(hedge_pnls)/len(hedge_pnls)*100:.4f}%")
        print(f"    Avg rebalances: {sum(n_rebs)/len(n_rebs):.0f}")

    # ═══════════════════════════════════════════════════════════════
    # TEST 2: Transaction cost impact on simulated straddles
    # ═══════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("  TEST 2: TRANSACTION COST IMPACT")
    print("=" * 70)

    print(f"\n  {'Hedge':<8} {'Slip(bps)':<10} {'Sharpe':>8} {'Return':>8} {'WinRate':>8}")
    for hedge_h, label in [(8, "8h"), (24, "24h")]:
        for slip in [0, 2, 5, 10, 20]:
            results = run_rolling_straddles(
                prices_hourly, dvol_hourly,
                roll_days=7, tte_days=30,
                hedge_freq_hours=hedge_h, slippage_bps=slip
            )
            if not results:
                continue
            pnls = [r["pnl"] for r in results]
            mean_pnl = sum(pnls) / len(pnls)
            std_pnl = math.sqrt(sum((p - mean_pnl)**2 for p in pnls) / len(pnls)) if len(pnls) > 1 else 1
            ann_factor = 365.0 / 7.0
            sharpe = (mean_pnl * ann_factor) / (std_pnl * math.sqrt(ann_factor)) if std_pnl > 0 else 0
            ann_ret = mean_pnl * ann_factor
            win_rate = sum(1 for p in pnls if p > 0) / len(pnls)
            print(f"  {label:<8} {slip:<10} {sharpe:8.2f} {ann_ret*100:7.2f}% {win_rate*100:6.1f}%")
        print()

    # ═══════════════════════════════════════════════════════════════
    # TEST 3: Funding rate impact
    # ═══════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("  TEST 3: FUNDING RATE IMPACT (perpetual futures hedging)")
    print("=" * 70)

    print(f"\n  {'Fund Rate':<12} {'Sharpe':>8} {'Return':>8}")
    for funding in [0, 0.0001, 0.0003, 0.0005, 0.001]:
        results = run_rolling_straddles(
            prices_hourly, dvol_hourly,
            roll_days=7, tte_days=30,
            hedge_freq_hours=24, funding_rate=funding
        )
        if not results:
            continue
        pnls = [r["pnl"] for r in results]
        mean_pnl = sum(pnls) / len(pnls)
        std_pnl = math.sqrt(sum((p - mean_pnl)**2 for p in pnls) / len(pnls)) if len(pnls) > 1 else 1
        ann_factor = 365.0 / 7.0
        sharpe = (mean_pnl * ann_factor) / (std_pnl * math.sqrt(ann_factor)) if std_pnl > 0 else 0
        ann_ret = mean_pnl * ann_factor
        rate_pct = funding * 365 * 100
        print(f"  {rate_pct:5.1f}% ann  {sharpe:8.2f} {ann_ret*100:7.2f}%")

    # ═══════════════════════════════════════════════════════════════
    # TEST 4: Expiry selection (7d vs 14d vs 30d vs 60d)
    # ═══════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("  TEST 4: EXPIRY SELECTION (straddle tenor)")
    print("=" * 70)

    print(f"\n  {'TTE':<8} {'Roll':<8} {'Sharpe':>8} {'Return':>8} {'Premium':>8} {'WinRate':>8}")
    for tte, roll in [(7, 3), (14, 7), (30, 7), (30, 14), (60, 14), (60, 30)]:
        results = run_rolling_straddles(
            prices_hourly, dvol_hourly,
            roll_days=roll, tte_days=tte,
            hedge_freq_hours=24
        )
        if not results:
            continue
        pnls = [r["pnl"] for r in results]
        premiums = [r["premium"] for r in results]
        mean_pnl = sum(pnls) / len(pnls)
        std_pnl = math.sqrt(sum((p - mean_pnl)**2 for p in pnls) / len(pnls)) if len(pnls) > 1 else 1
        ann_factor = 365.0 / roll
        sharpe = (mean_pnl * ann_factor) / (std_pnl * math.sqrt(ann_factor)) if std_pnl > 0 else 0
        ann_ret = mean_pnl * ann_factor
        win_rate = sum(1 for p in pnls if p > 0) / len(pnls)
        avg_premium = sum(premiums) / len(premiums)
        print(f"  {tte}d      {roll}d     {sharpe:8.2f} {ann_ret*100:7.2f}% {avg_premium*100:7.2f}% {win_rate*100:6.1f}%")

    # ═══════════════════════════════════════════════════════════════
    # TEST 5: Yearly breakdown of simulated vs theoretical
    # ═══════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("  TEST 5: YEARLY BREAKDOWN (simulated 24h hedge)")
    print("=" * 70)

    sim_results = run_rolling_straddles(
        prices_hourly, dvol_hourly,
        roll_days=7, tte_days=30,
        hedge_freq_hours=24
    )

    yearly_sim = defaultdict(list)
    for r in sim_results:
        yr = r["date"][:4]
        yearly_sim[yr].append(r["pnl"])

    yearly_theo = defaultdict(list)
    for d in sorted(theoretical_pnl):
        yearly_theo[d[:4]].append(theoretical_pnl[d])

    print(f"\n  {'Year':<8} {'Sim Sharpe':>10} {'Theo Sharpe':>12} {'Gap':>8}")
    for yr in sorted(set(yearly_sim.keys()) | set(yearly_theo.keys())):
        s_pnls = yearly_sim.get(yr, [])
        t_pnls = yearly_theo.get(yr, [])

        if len(s_pnls) > 5:
            sm = sum(s_pnls) / len(s_pnls)
            ss = math.sqrt(sum((p - sm)**2 for p in s_pnls) / len(s_pnls)) if len(s_pnls) > 1 else 1
            ann_f = 365.0 / 7.0
            s_sharpe = (sm * ann_f) / (ss * math.sqrt(ann_f)) if ss > 0 else 0
        else:
            s_sharpe = 0

        t_stats_yr = calc_stats(t_pnls) if len(t_pnls) > 30 else {"sharpe": 0}
        gap = s_sharpe - t_stats_yr["sharpe"]
        print(f"  {yr:<8} {s_sharpe:10.2f} {t_stats_yr['sharpe']:12.2f} {gap:+7.2f}")

    # ═══════════════════════════════════════════════════════════════
    # CONCLUSION
    # ═══════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("R63: DELTA HEDGING SIMULATION CONCLUSION")
    print("=" * 70)

    # Summarize key sim result (24h hedge, 30d straddle, weekly roll)
    base_results = run_rolling_straddles(
        prices_hourly, dvol_hourly,
        roll_days=7, tte_days=30, hedge_freq_hours=24
    )
    base_pnls = [r["pnl"] for r in base_results]
    bm = sum(base_pnls) / len(base_pnls)
    bs = math.sqrt(sum((p - bm)**2 for p in base_pnls) / len(base_pnls))
    ann_f = 365.0 / 7.0
    sim_sharpe = (bm * ann_f) / (bs * math.sqrt(ann_f)) if bs > 0 else 0
    sim_ann_ret = bm * ann_f

    print(f"\n  Theoretical VRP (R60): Sharpe {t_stats['sharpe']:.2f}, Return {t_stats['ann_ret']*100:.2f}%")
    print(f"  Simulated straddle:    Sharpe {sim_sharpe:.2f}, Return {sim_ann_ret*100:.2f}%")
    gap = sim_sharpe - t_stats["sharpe"]
    gap_pct = gap / abs(t_stats["sharpe"]) * 100 if t_stats["sharpe"] != 0 else 0
    print(f"  Gap: {gap:+.2f} ({gap_pct:+.1f}%)")

    if abs(gap) < 0.5:
        verdict = "THEORETICAL MODEL IS ACCURATE — discrete hedging has minimal impact"
    elif gap < -0.5:
        verdict = "HEDGING EROSION SIGNIFICANT — theoretical model overestimates"
    else:
        verdict = "SIMULATION OUTPERFORMS THEORY — convexity benefits from discrete hedging"

    print(f"\n  VERDICT: {verdict}")

    # Save results
    results = {
        "research_id": "R63",
        "title": "Delta Hedging Simulation with Real Data",
        "theoretical": {
            "sharpe": round(t_stats["sharpe"], 4),
            "ann_ret": round(t_stats["ann_ret"], 6),
        },
        "simulated_24h": {
            "sharpe": round(sim_sharpe, 4),
            "ann_ret": round(sim_ann_ret, 6),
            "n_cycles": len(base_results),
            "avg_premium": round(sum(r["premium"] for r in base_results) / len(base_results), 6),
        },
        "gap": round(gap, 4),
        "gap_pct": round(gap_pct, 2),
        "verdict": verdict,
    }

    out = ROOT / "data" / "cache" / "deribit" / "real_surface" / "r63_delta_hedging_results.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved to {out}")


if __name__ == "__main__":
    main()
