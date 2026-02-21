#!/usr/bin/env python3
"""
R62: Multi-Expiry VRP with Real Term Structure
================================================

The production VRP uses a single 30-day DVOL. The real surface has
term_spread = front_atm_iv - back_atm_iv (~30d vs ~60-90d).

Tests:
  1. Term-structure-conditioned VRP: scale VRP by term spread signal
  2. Back-month VRP: use longer-dated IV (lower gamma, more theta)
  3. Blended VRP: weighted average of front/back VRP
  4. Term spread as additional alpha factor
  5. Walk-forward validation

Hypothesis: multi-expiry information improves VRP timing.
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


def load_surface(currency: str) -> Dict[str, dict]:
    path = ROOT / "data" / "cache" / "deribit" / "real_surface" / f"{currency}_daily_surface.csv"
    data = {}
    if not path.exists():
        return data
    with open(path) as f:
        for row in csv.DictReader(f):
            d = row["date"]
            entry = {}
            for field in ["iv_atm", "term_spread", "butterfly_25d", "skew_25d"]:
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
    ann_vol = std * math.sqrt(365)
    cum = peak = max_dd = 0.0
    for r in rets:
        cum += r
        peak = max(peak, cum)
        max_dd = max(max_dd, peak - cum)
    win_rate = sum(1 for r in rets if r > 0) / len(rets)
    t_stat = (mean / std) * math.sqrt(len(rets)) if std > 0 else 0
    calmar = ann_ret / max_dd if max_dd > 0 else 999
    return {
        "sharpe": sharpe, "ann_ret": ann_ret, "ann_vol": ann_vol,
        "max_dd": max_dd, "win_rate": win_rate, "t_stat": t_stat,
        "n": len(rets), "calmar": calmar, "total_return": sum(rets),
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


def rolling_percentile(values, dates, lookback):
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
        below = sum(1 for w in window if w <= val)
        result[d] = below / len(window)
    return result


# ─── VRP Models ─────────────────────────────────────────────────

def vrp_pnl_basic(dates, dvol, prices, lev=2.0):
    """Standard daily VRP (R60 production)."""
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


def vrp_pnl_back_month(dates, surface, prices, dvol, lev=2.0):
    """
    VRP using back-month IV instead of front-month DVOL.
    back_iv = front_iv - term_spread (since ts = front - back, so back = front - ts)
    """
    dt = 1.0 / 365.0
    pnl = {}
    for i in range(1, len(dates)):
        d, dp = dates[i], dates[i-1]
        s = surface.get(dp, {})
        front_iv = dvol.get(dp)
        ts = s.get("term_spread")
        p0, p1 = prices.get(dp), prices.get(d)
        if not all([front_iv, ts is not None, p0, p1]) or p0 <= 0:
            continue
        back_iv = front_iv - ts  # back = front - (front - back)
        if back_iv <= 0:
            continue
        rv_bar = abs(math.log(p1 / p0)) * math.sqrt(365)
        pnl[d] = lev * 0.5 * (back_iv**2 - rv_bar**2) * dt
    return pnl


def vrp_pnl_blended(dates, surface, prices, dvol, lev=2.0, w_front=0.5):
    """Blended VRP: weighted average of front and back month IV."""
    dt = 1.0 / 365.0
    pnl = {}
    w_back = 1.0 - w_front
    for i in range(1, len(dates)):
        d, dp = dates[i], dates[i-1]
        s = surface.get(dp, {})
        front_iv = dvol.get(dp)
        ts = s.get("term_spread")
        p0, p1 = prices.get(dp), prices.get(d)
        if not all([front_iv, ts is not None, p0, p1]) or p0 <= 0:
            continue
        back_iv = front_iv - ts
        if back_iv <= 0:
            continue
        blended_iv = w_front * front_iv + w_back * back_iv
        rv_bar = abs(math.log(p1 / p0)) * math.sqrt(365)
        pnl[d] = lev * 0.5 * (blended_iv**2 - rv_bar**2) * dt
    return pnl


def vrp_pnl_ts_conditioned(dates, surface, prices, dvol, lev=2.0, lookback=90):
    """
    VRP conditioned on term structure signal.
    Contango (front > back, ts > 0): normal, full VRP exposure
    Backwardation (front < back, ts < 0): stress, reduce/flip VRP
    """
    dt = 1.0 / 365.0
    ts_vals = {}
    for d in dates:
        s = surface.get(d, {})
        if "term_spread" in s:
            ts_vals[d] = s["term_spread"]

    ts_z = rolling_zscore(ts_vals, dates, lookback)
    pnl = {}
    for i in range(1, len(dates)):
        d, dp = dates[i], dates[i-1]
        iv, p0, p1 = dvol.get(dp), prices.get(dp), prices.get(d)
        if not all([iv, p0, p1]) or p0 <= 0:
            continue
        rv_bar = abs(math.log(p1 / p0)) * math.sqrt(365)
        base_pnl = lev * 0.5 * (iv**2 - rv_bar**2) * dt

        # Scale by term structure signal
        z = ts_z.get(d)
        if z is not None:
            if z > 1.0:  # Strong contango → VRP is wide → full exposure
                scale = 1.5
            elif z > 0:  # Mild contango
                scale = 1.0
            elif z > -1.0:  # Mild backwardation
                scale = 0.5
            else:  # Strong backwardation → stress → reduce
                scale = 0.0
        else:
            scale = 1.0
        pnl[d] = base_pnl * scale
    return pnl


def vrp_pnl_ts_dynamic(dates, surface, prices, dvol, lev=2.0, lookback=90):
    """
    Dynamic VRP: use front IV in contango, back IV in backwardation.
    Intuition: in backwardation, back IV is higher → more theta, less gamma risk.
    """
    dt = 1.0 / 365.0
    ts_vals = {}
    for d in dates:
        s = surface.get(d, {})
        if "term_spread" in s:
            ts_vals[d] = s["term_spread"]

    ts_pctile = rolling_percentile(ts_vals, dates, lookback)
    pnl = {}
    for i in range(1, len(dates)):
        d, dp = dates[i], dates[i-1]
        s = surface.get(dp, {})
        front_iv = dvol.get(dp)
        ts = s.get("term_spread")
        p0, p1 = prices.get(dp), prices.get(d)
        if not all([front_iv, ts is not None, p0, p1]) or p0 <= 0:
            continue

        back_iv = front_iv - ts
        if back_iv <= 0:
            continue

        pctile = ts_pctile.get(d)
        if pctile is not None:
            # Low percentile = backwardation → use more back IV
            # High percentile = contango → use front IV
            w_front = max(0.3, min(0.9, pctile))
        else:
            w_front = 0.5
        iv = w_front * front_iv + (1.0 - w_front) * back_iv

        rv_bar = abs(math.log(p1 / p0)) * math.sqrt(365)
        pnl[d] = lev * 0.5 * (iv**2 - rv_bar**2) * dt
    return pnl


def term_spread_mr_pnl(dates, surface, dvol, lookback=120, z_entry=1.5):
    """Term spread mean-reversion strategy (pure alpha)."""
    dt = 1.0 / 365.0
    ts_vals = {}
    for d in dates:
        s = surface.get(d, {})
        if "term_spread" in s:
            ts_vals[d] = s["term_spread"]

    ts_z = rolling_zscore(ts_vals, dates, lookback)
    position = 0.0
    pnl = {}
    for i in range(1, len(dates)):
        d, dp = dates[i], dates[i-1]
        z = ts_z.get(d)
        iv = dvol.get(d)
        ts_now = ts_vals.get(d)
        ts_prev = ts_vals.get(dp)

        if z is not None:
            if z > z_entry:
                position = -1.0  # Contango too steep → revert
            elif z < -z_entry:
                position = 1.0   # Backwardation too steep → revert
            elif abs(z) < 0.3:
                position = 0.0

        if ts_now is not None and ts_prev is not None and iv is not None and position != 0:
            pnl[d] = position * (ts_now - ts_prev) * iv * math.sqrt(dt) * 2.5
        elif d in ts_z:
            pnl[d] = 0.0
    return pnl


def bf_pnl(dates, surface, dvol, lookback=120, z_entry=1.5):
    """Butterfly MR PnL (same as R60)."""
    dt = 1.0 / 365.0
    bf_vals = {}
    for d in dates:
        s = surface.get(d, {})
        if "butterfly_25d" in s:
            bf_vals[d] = s["butterfly_25d"]

    bf_z = rolling_zscore(bf_vals, dates, lookback)
    position = 0.0
    pnl = {}
    for i in range(1, len(dates)):
        d, dp = dates[i], dates[i-1]
        z = bf_z.get(d)
        iv = dvol.get(d)
        f_now, f_prev = bf_vals.get(d), bf_vals.get(dp)

        if z is not None:
            if z > z_entry:
                position = -1.0
            elif z < -z_entry:
                position = 1.0
            elif abs(z) < 0.3:
                position = 0.0

        if f_now is not None and f_prev is not None and iv is not None and position != 0:
            pnl[d] = position * (f_now - f_prev) * iv * math.sqrt(dt) * 2.5
        elif d in bf_z:
            pnl[d] = 0.0
    return pnl


def walk_forward(pnl_dict, periods):
    """Walk-forward stats per period."""
    results = []
    for name, start, end in periods:
        rets = [pnl_dict[d] for d in sorted(pnl_dict) if start <= d <= end]
        if len(rets) < 30:
            continue
        stats = calc_stats(rets)
        results.append({"period": name, "sharpe": round(stats["sharpe"], 2),
                        "ann_ret": round(stats["ann_ret"] * 100, 2)})
    return results


def main():
    print("=" * 70)
    print("R62: MULTI-EXPIRY VRP WITH REAL TERM STRUCTURE")
    print("=" * 70)

    # Load data
    print("\n  Loading data...")
    dvol = load_dvol_daily("BTC")
    prices = load_prices("BTC")
    surface = load_surface("BTC")

    # Get common dates
    all_dates = sorted(set(dvol.keys()) & set(prices.keys()))
    surface_dates = sorted(set(all_dates) & set(surface.keys()))

    # Filter for dates with term_spread
    ts_dates = [d for d in surface_dates if "term_spread" in surface[d]]
    print(f"    DVOL: {len(dvol)} days")
    print(f"    Prices: {len(prices)} days")
    print(f"    Surface: {len(surface)} days")
    print(f"    With term_spread: {len(ts_dates)} days")
    print(f"    Date range: {ts_dates[0]} to {ts_dates[-1]}")

    # ─── Term spread statistics ─────────────────────────────────
    print("\n" + "=" * 70)
    print("  TERM SPREAD STATISTICS")
    print("=" * 70)

    ts_values = [surface[d]["term_spread"] for d in ts_dates]
    ts_mean = sum(ts_values) / len(ts_values)
    ts_std = math.sqrt(sum((v - ts_mean)**2 for v in ts_values) / len(ts_values))
    ts_sorted = sorted(ts_values)
    pct25 = ts_sorted[int(0.25 * len(ts_sorted))]
    pct50 = ts_sorted[int(0.50 * len(ts_sorted))]
    pct75 = ts_sorted[int(0.75 * len(ts_sorted))]
    positive_pct = sum(1 for v in ts_values if v > 0) / len(ts_values)

    print(f"\n  Mean:       {ts_mean:.4f} ({ts_mean*100:.2f}%)")
    print(f"  Std:        {ts_std:.4f}")
    print(f"  25th pctile:{pct25:.4f}")
    print(f"  Median:     {pct50:.4f}")
    print(f"  75th pctile:{pct75:.4f}")
    print(f"  Min:        {min(ts_values):.4f}")
    print(f"  Max:        {max(ts_values):.4f}")
    print(f"  Positive (contango): {positive_pct*100:.1f}%")

    # Correlation with DVOL
    dvol_vals = [dvol[d] for d in ts_dates if d in dvol]
    ts_vals_aligned = [surface[d]["term_spread"] for d in ts_dates if d in dvol]
    if len(dvol_vals) == len(ts_vals_aligned) and len(dvol_vals) > 10:
        dvol_mean = sum(dvol_vals) / len(dvol_vals)
        ts_mean_a = sum(ts_vals_aligned) / len(ts_vals_aligned)
        cov = sum((d - dvol_mean) * (t - ts_mean_a) for d, t in zip(dvol_vals, ts_vals_aligned)) / len(dvol_vals)
        dvol_std = math.sqrt(sum((d - dvol_mean)**2 for d in dvol_vals) / len(dvol_vals))
        ts_std_a = math.sqrt(sum((t - ts_mean_a)**2 for t in ts_vals_aligned) / len(ts_vals_aligned))
        corr = cov / (dvol_std * ts_std_a) if dvol_std > 0 and ts_std_a > 0 else 0
        print(f"  Corr(TermSpread, DVOL): {corr:.3f}")

    # ═══════════════════════════════════════════════════════════════
    # TEST 1: VRP model comparison
    # ═══════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("  TEST 1: VRP MODEL COMPARISON")
    print("=" * 70)

    models = {}

    # Baseline: standard VRP
    pnl = vrp_pnl_basic(all_dates, dvol, prices)
    models["VRP (standard)"] = pnl

    # Back-month VRP
    pnl = vrp_pnl_back_month(all_dates, surface, prices, dvol)
    models["VRP (back-month)"] = pnl

    # Blended VRP at various weights
    for w in [0.7, 0.5, 0.3]:
        pnl = vrp_pnl_blended(all_dates, surface, prices, dvol, w_front=w)
        models[f"VRP (blend {int(w*100)}/{int((1-w)*100)})"] = pnl

    # Term-structure conditioned VRP
    for lb in [60, 90, 120]:
        pnl = vrp_pnl_ts_conditioned(all_dates, surface, prices, dvol, lookback=lb)
        models[f"VRP (TS-cond lb={lb})"] = pnl

    # Dynamic VRP
    for lb in [60, 90, 120]:
        pnl = vrp_pnl_ts_dynamic(all_dates, surface, prices, dvol, lookback=lb)
        models[f"VRP (dynamic lb={lb})"] = pnl

    print(f"\n  {'Model':<28} {'Sharpe':>8} {'Return':>8} {'MaxDD':>8} {'Calmar':>8} {'N':>6}")
    for name, pnl in models.items():
        rets = [pnl[d] for d in sorted(pnl)]
        stats = calc_stats(rets)
        print(f"  {name:<28} {stats['sharpe']:8.2f} {stats['ann_ret']*100:7.2f}% "
              f"{stats['max_dd']*100:7.2f}% {stats['calmar']:8.2f} {stats['n']:6}")

    # ═══════════════════════════════════════════════════════════════
    # TEST 2: Term spread as additional alpha (MR strategy)
    # ═══════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("  TEST 2: TERM SPREAD MEAN-REVERSION AS ALPHA")
    print("=" * 70)

    ts_mr_results = {}
    for lb in [60, 90, 120, 180]:
        for z in [1.0, 1.5, 2.0]:
            pnl = term_spread_mr_pnl(all_dates, surface, dvol, lookback=lb, z_entry=z)
            rets = [pnl[d] for d in sorted(pnl)]
            stats = calc_stats(rets)
            key = f"TS-MR lb={lb} z={z}"
            ts_mr_results[key] = {"stats": stats, "pnl": pnl}

    print(f"\n  {'Config':<24} {'Sharpe':>8} {'Return':>8} {'MaxDD':>8} {'Calmar':>8}")
    for key, r in ts_mr_results.items():
        s = r["stats"]
        print(f"  {key:<24} {s['sharpe']:8.2f} {s['ann_ret']*100:7.2f}% "
              f"{s['max_dd']*100:7.2f}% {s['calmar']:8.2f}")

    # ═══════════════════════════════════════════════════════════════
    # TEST 3: VRP + BF + TS multi-factor ensemble
    # ═══════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("  TEST 3: MULTI-FACTOR ENSEMBLE (VRP + BF + TS)")
    print("=" * 70)

    # Get base PnLs
    vrp_base = vrp_pnl_basic(all_dates, dvol, prices)
    bf_base = bf_pnl(all_dates, surface, dvol)

    # Best TS-MR
    best_ts_key = max(ts_mr_results.items(), key=lambda x: x[1]["stats"]["sharpe"])
    ts_best_pnl = best_ts_key[1]["pnl"]
    print(f"\n  Best TS-MR: {best_ts_key[0]} (Sharpe {best_ts_key[1]['stats']['sharpe']:.2f})")

    # PnL correlation
    common = sorted(set(vrp_base.keys()) & set(bf_base.keys()) & set(ts_best_pnl.keys()))
    if len(common) > 100:
        v = [vrp_base[d] for d in common]
        b = [bf_base[d] for d in common]
        t = [ts_best_pnl[d] for d in common]

        def corr(x, y):
            mx, my = sum(x)/len(x), sum(y)/len(y)
            cov = sum((xi - mx) * (yi - my) for xi, yi in zip(x, y)) / len(x)
            sx = math.sqrt(sum((xi - mx)**2 for xi in x) / len(x))
            sy = math.sqrt(sum((yi - my)**2 for yi in y) / len(y))
            return cov / (sx * sy) if sx > 0 and sy > 0 else 0

        print(f"\n  PnL Correlations:")
        print(f"    VRP vs BF:  {corr(v, b):.3f}")
        print(f"    VRP vs TS:  {corr(v, t):.3f}")
        print(f"    BF vs TS:   {corr(b, t):.3f}")

    # Ensemble weights
    print(f"\n  {'Weights (VRP/BF/TS)':<28} {'Sharpe':>8} {'Return':>8} {'MaxDD':>8} {'Calmar':>8}")
    ensemble_results = {}
    for w_vrp, w_bf, w_ts in [
        (1.0, 0.0, 0.0),    # VRP only
        (0.5, 0.5, 0.0),    # R60 champion
        (0.7, 0.3, 0.0),    # R60 alt
        (0.5, 0.3, 0.2),    # Triple
        (0.4, 0.4, 0.2),    # Triple balanced
        (0.5, 0.25, 0.25),  # Triple VRP-heavy
        (0.33, 0.33, 0.34), # Equal weight
        (0.4, 0.3, 0.3),    # Slight VRP tilt
    ]:
        combined = {}
        for d in common:
            combined[d] = w_vrp * vrp_base[d] + w_bf * bf_base[d] + w_ts * ts_best_pnl[d]
        rets = [combined[d] for d in sorted(combined)]
        stats = calc_stats(rets)
        label = f"{int(w_vrp*100)}/{int(w_bf*100)}/{int(w_ts*100)}"
        ensemble_results[label] = {"stats": stats, "pnl": combined}
        print(f"  {label:<28} {stats['sharpe']:8.2f} {stats['ann_ret']*100:7.2f}% "
              f"{stats['max_dd']*100:7.2f}% {stats['calmar']:8.2f}")

    # ═══════════════════════════════════════════════════════════════
    # TEST 4: Walk-forward validation
    # ═══════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("  TEST 4: WALK-FORWARD VALIDATION")
    print("=" * 70)

    periods = [
        ("2021H2", "2021-07-01", "2021-12-31"),
        ("2022H1", "2022-01-01", "2022-06-30"),
        ("2022H2", "2022-07-01", "2022-12-31"),
        ("2023H1", "2023-01-01", "2023-06-30"),
        ("2023H2", "2023-07-01", "2023-12-31"),
        ("2024H1", "2024-01-01", "2024-06-30"),
        ("2024H2", "2024-07-01", "2024-12-31"),
        ("2025H1", "2025-01-01", "2025-06-30"),
        ("2025H2", "2025-07-01", "2025-12-31"),
        ("2026H1", "2026-01-01", "2026-06-30"),
    ]

    # Compare R60 champion (VRP 50/BF 50) vs best triple
    best_triple_key = max(
        [(k, v) for k, v in ensemble_results.items() if k.count("/") == 2 and k not in ["100/0/0", "50/50/0", "70/30/0"]],
        key=lambda x: x[1]["stats"]["sharpe"]
    )
    best_triple_label = best_triple_key[0]
    best_triple_pnl = best_triple_key[1]["pnl"]

    r60_pnl = ensemble_results.get("50/50/0", {}).get("pnl", {})

    print(f"\n  Comparing: R60 (50/50/0) vs Best Triple ({best_triple_label})")
    print(f"\n  {'Period':<10} {'R60':>8} {'Triple':>8} {'Delta':>8}")
    t_wins = 0
    for name, start, end in periods:
        r60_rets = [r60_pnl[d] for d in sorted(r60_pnl) if start <= d <= end]
        tri_rets = [best_triple_pnl[d] for d in sorted(best_triple_pnl) if start <= d <= end]
        if len(r60_rets) < 30 or len(tri_rets) < 30:
            continue
        r60_s = calc_stats(r60_rets)["sharpe"]
        tri_s = calc_stats(tri_rets)["sharpe"]
        delta = tri_s - r60_s
        if delta > 0:
            t_wins += 1
        flag = "+" if delta > 0 else "-"
        print(f"  {name:<10} {r60_s:8.2f} {tri_s:8.2f} {delta:+7.2f} {flag}")

    wf_total = sum(1 for name, start, end in periods
                   if len([d for d in sorted(r60_pnl) if start <= d <= end]) >= 30)
    print(f"\n  Triple wins: {t_wins}/{wf_total} periods ({t_wins/max(1,wf_total)*100:.0f}%)")

    # ═══════════════════════════════════════════════════════════════
    # TEST 5: Yearly breakdown
    # ═══════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("  TEST 5: YEARLY BREAKDOWN")
    print("=" * 70)

    years = sorted(set(d[:4] for d in common))
    print(f"\n  {'Year':<8} {'VRP-only':>10} {'VRP+BF':>10} {'VRP+BF+TS':>10}")
    for yr in years:
        vrp_rets = [vrp_base[d] for d in sorted(vrp_base) if d[:4] == yr and d in vrp_base]
        r60_rets = [r60_pnl[d] for d in sorted(r60_pnl) if d[:4] == yr]
        tri_rets = [best_triple_pnl[d] for d in sorted(best_triple_pnl) if d[:4] == yr]

        v_s = calc_stats(vrp_rets)["sharpe"] if len(vrp_rets) > 30 else 0
        r_s = calc_stats(r60_rets)["sharpe"] if len(r60_rets) > 30 else 0
        t_s = calc_stats(tri_rets)["sharpe"] if len(tri_rets) > 30 else 0
        print(f"  {yr:<8} {v_s:10.2f} {r_s:10.2f} {t_s:10.2f}")

    # ═══════════════════════════════════════════════════════════════
    # CONCLUSION
    # ═══════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("R62: MULTI-EXPIRY VRP CONCLUSION")
    print("=" * 70)

    r60_stats = ensemble_results.get("50/50/0", {}).get("stats", {})
    best_t_stats = best_triple_key[1]["stats"]
    delta = best_t_stats.get("sharpe", 0) - r60_stats.get("sharpe", 0)

    print(f"\n  R60 Champion (VRP+BF 50/50):  Sharpe {r60_stats.get('sharpe', 0):.2f}")
    print(f"  Best Triple ({best_triple_label}):  Sharpe {best_t_stats.get('sharpe', 0):.2f}")
    print(f"  Delta: {delta:+.2f}")
    print(f"  Triple wins {t_wins}/{wf_total} walk-forward periods")

    if delta > 0.1 and t_wins >= wf_total * 0.6:
        verdict = f"TERM SPREAD IMPROVES — upgrade to {best_triple_label} triple factor"
    elif delta > -0.1:
        verdict = "MARGINAL — term spread adds negligible value"
    else:
        verdict = "TERM SPREAD HURTS — stick with VRP+BF"

    print(f"\n  VERDICT: {verdict}")

    # Save results
    results = {
        "research_id": "R62",
        "title": "Multi-Expiry VRP with Real Term Structure",
        "term_spread_stats": {
            "mean": round(ts_mean, 4), "std": round(ts_std, 4),
            "pct_contango": round(positive_pct, 3),
        },
        "model_comparison": {
            name: {"sharpe": round(calc_stats([pnl[d] for d in sorted(pnl)])["sharpe"], 4)}
            for name, pnl in models.items()
        },
        "best_ts_mr": best_ts_key[0],
        "best_ts_mr_sharpe": round(best_ts_key[1]["stats"]["sharpe"], 4),
        "ensemble_comparison": {
            k: {"sharpe": round(v["stats"]["sharpe"], 4), "max_dd": round(v["stats"]["max_dd"], 6)}
            for k, v in ensemble_results.items()
        },
        "r60_sharpe": round(r60_stats.get("sharpe", 0), 4),
        "best_triple": best_triple_label,
        "best_triple_sharpe": round(best_t_stats.get("sharpe", 0), 4),
        "delta_sharpe": round(delta, 4),
        "walk_forward_triple_wins": f"{t_wins}/{wf_total}",
        "verdict": verdict,
    }

    out = ROOT / "data" / "cache" / "deribit" / "real_surface" / "r62_multi_expiry_results.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved to {out}")


if __name__ == "__main__":
    main()
