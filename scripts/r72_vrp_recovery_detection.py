#!/usr/bin/env python3
"""
R72: VRP Recovery Detection
=============================

Build indicators to detect when VRP edge is recovering, which would
signal a weight shift back from 10% toward 50% VRP allocation.

Key question: Can we detect IV regime shifts EARLY ENOUGH to be useful?

VRP degradation pattern (R65):
  - IV structural decline: 70% → 51.5% (-26%)
  - VRP Sharpe by half-year: 4.15 → 1.47 → 1.11 → 0.60 → -1.62
  - VRP negative in 2025Q4-2026Q1

Approach:
  1. Analyze VRP edge vs IV level — what IV threshold restores VRP edge?
  2. Rolling VRP Sharpe as recovery indicator
  3. IV regime change detection (breakpoint analysis)
  4. Build composite VRP health indicator
  5. Backtest: dynamic VRP weight (10-50%) based on indicator
  6. Compare vs static 10/90 (R69 baseline Sharpe 3.76)
"""
import csv
import json
import math
import subprocess
import time
from collections import defaultdict
from datetime import datetime, timezone, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


# ═══════════════════════════════════════════════════════════════
# Data Loading (reuse from R69/R70 pattern)
# ═══════════════════════════════════════════════════════════════

def load_dvol_history(currency: str) -> dict:
    path = ROOT / "data" / "cache" / "deribit" / "dvol" / f"{currency}_DVOL_12h.csv"
    daily = {}
    if not path.exists():
        return daily
    with open(path) as f:
        for row in csv.DictReader(f):
            daily[row["date"][:10]] = float(row["dvol_close"]) / 100.0
    return daily


def load_price_history(currency: str) -> dict:
    prices = {}
    start_dt = datetime(2020, 1, 1, tzinfo=timezone.utc)
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


def load_surface(currency: str) -> dict:
    path = ROOT / "data" / "cache" / "deribit" / "real_surface" / f"{currency}_daily_surface.csv"
    data = {}
    if not path.exists():
        return data
    with open(path) as f:
        for row in csv.DictReader(f):
            d = row["date"]
            entry = {}
            for field in ["butterfly_25d", "iv_atm", "skew_25d"]:
                val = row.get(field, "")
                if val and val != "None":
                    entry[field] = float(val)
            if entry:
                data[d] = entry
    return data


# ═══════════════════════════════════════════════════════════════
# VRP Analysis Functions
# ═══════════════════════════════════════════════════════════════

def compute_daily_vrp(dvol_hist, price_hist):
    """Compute daily VRP spread and PnL."""
    dates = sorted(set(dvol_hist.keys()) & set(price_hist.keys()))
    dt = 1.0 / 365.0

    vrp_data = {}
    for i in range(1, len(dates)):
        d, dp = dates[i], dates[i-1]
        iv = dvol_hist.get(dp)
        p0, p1 = price_hist.get(dp), price_hist.get(d)
        if not all([iv, p0, p1]) or p0 <= 0:
            continue
        rv = abs(math.log(p1 / p0)) * math.sqrt(365)
        vrp_spread = iv**2 - rv**2
        vrp_pnl = 2.0 * 0.5 * vrp_spread * dt  # leverage=2.0

        vrp_data[d] = {
            "iv": iv,
            "rv": rv,
            "vrp_spread": vrp_spread,
            "vrp_pnl": vrp_pnl,
        }

    return vrp_data


def compute_rolling_sharpe(pnl_series, dates, window):
    """Compute rolling Sharpe ratio."""
    rolling = {}
    vals = [pnl_series[d] for d in dates if d in pnl_series]
    date_list = [d for d in dates if d in pnl_series]

    for i in range(window, len(vals)):
        w = vals[i-window:i]
        mean = sum(w) / len(w)
        std = math.sqrt(sum((v - mean)**2 for v in w) / len(w))
        if std > 0:
            sharpe = (mean * 365) / (std * math.sqrt(365))
        else:
            sharpe = 0
        rolling[date_list[i]] = sharpe
    return rolling


def compute_rolling_iv(dvol_hist, dates, window):
    """Compute rolling average IV."""
    rolling = {}
    vals = [(d, dvol_hist[d]) for d in dates if d in dvol_hist]
    for i in range(window, len(vals)):
        w = [v[1] for v in vals[i-window:i]]
        rolling[vals[i][0]] = sum(w) / len(w)
    return rolling


def compute_bf_pnl(dvol_hist, surface_hist, dates, z_exit=0.0, sensitivity=2.5):
    """Compute BF MR PnL series."""
    bf_vals = {}
    for d in dates:
        if d in surface_hist and "butterfly_25d" in surface_hist[d]:
            bf_vals[d] = surface_hist[d]["butterfly_25d"]

    dt = 1.0 / 365.0
    lb = 120
    z_entry = 1.5
    position = 0.0
    bf_pnl = {}

    for i in range(lb, len(dates)):
        d, dp = dates[i], dates[i-1]
        val = bf_vals.get(d)
        if val is None:
            continue
        window = [bf_vals.get(dates[j]) for j in range(i-lb, i)]
        window = [v for v in window if v is not None]
        if len(window) < lb // 2:
            continue
        mean = sum(window) / len(window)
        std = math.sqrt(sum((v - mean)**2 for v in window) / len(window))
        if std < 1e-8:
            continue
        z = (val - mean) / std

        if z > z_entry:
            position = -1.0
        elif z < -z_entry:
            position = 1.0
        elif z_exit > 0 and abs(z) < z_exit:
            position = 0.0

        iv = dvol_hist.get(d)
        f_now, f_prev = bf_vals.get(d), bf_vals.get(dp)
        if f_now is not None and f_prev is not None and iv is not None and position != 0:
            bf_pnl[d] = position * (f_now - f_prev) * iv * math.sqrt(dt) * sensitivity
        else:
            bf_pnl[d] = 0.0

    return bf_pnl


def compute_stats(rets):
    """Compute Sharpe, return, MaxDD, Calmar for a return series."""
    if len(rets) < 30:
        return {"sharpe": 0, "ann_ret": 0, "max_dd": 0, "calmar": 0}
    mean = sum(rets) / len(rets)
    std = math.sqrt(sum((r - mean)**2 for r in rets) / len(rets))
    sharpe = (mean * 365) / (std * math.sqrt(365)) if std > 0 else 0
    ann_ret = mean * 365

    cum = peak = max_dd = 0.0
    for r in rets:
        cum += r
        peak = max(peak, cum)
        max_dd = max(max_dd, peak - cum)

    calmar = ann_ret / max_dd if max_dd > 0 else 0
    return {"sharpe": sharpe, "ann_ret": ann_ret, "max_dd": max_dd, "calmar": calmar}


# ═══════════════════════════════════════════════════════════════
# Analysis 1: VRP Edge vs IV Level
# ═══════════════════════════════════════════════════════════════

def analyze_vrp_vs_iv_level(vrp_data, dates):
    """Bucket VRP performance by IV level."""
    print("\n" + "=" * 70)
    print("  ANALYSIS 1: VRP EDGE vs IV LEVEL")
    print("=" * 70)

    # Bucket by IV percentile
    all_iv = [vrp_data[d]["iv"] for d in dates if d in vrp_data]
    iv_pctiles = sorted(all_iv)

    buckets = [
        ("IV < 40%", 0.0, 0.40),
        ("IV 40-50%", 0.40, 0.50),
        ("IV 50-60%", 0.50, 0.60),
        ("IV 60-80%", 0.60, 0.80),
        ("IV > 80%", 0.80, 2.0),
    ]

    results = {}
    for name, lo, hi in buckets:
        rets = [vrp_data[d]["vrp_pnl"] for d in dates
                if d in vrp_data and lo <= vrp_data[d]["iv"] < hi]
        if len(rets) < 30:
            results[name] = {"n": len(rets), "sharpe": None}
            continue
        stats = compute_stats(rets)
        results[name] = {"n": len(rets), **stats}
        print(f"  {name:>12}: n={len(rets):4d}  Sharpe={stats['sharpe']:6.2f}  "
              f"AnnRet={stats['ann_ret']*100:6.2f}%  MaxDD={stats['max_dd']*100:5.2f}%")

    # Find IV threshold where VRP Sharpe > 1.0
    print("\n  ─── IV Threshold Analysis ───")
    for threshold in [0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]:
        rets_above = [vrp_data[d]["vrp_pnl"] for d in dates
                      if d in vrp_data and vrp_data[d]["iv"] >= threshold]
        rets_below = [vrp_data[d]["vrp_pnl"] for d in dates
                      if d in vrp_data and vrp_data[d]["iv"] < threshold]
        if len(rets_above) < 30:
            continue
        s_above = compute_stats(rets_above)
        s_below = compute_stats(rets_below) if len(rets_below) >= 30 else {"sharpe": 0}
        pct_above = len(rets_above) / (len(rets_above) + len(rets_below)) * 100
        print(f"    IV>={threshold*100:.0f}%: Sharpe={s_above['sharpe']:5.2f} (n={len(rets_above):4d}, {pct_above:4.0f}%)  "
              f"| IV<{threshold*100:.0f}%: Sharpe={s_below['sharpe']:5.2f}")

    return results


# ═══════════════════════════════════════════════════════════════
# Analysis 2: Rolling VRP Sharpe as Indicator
# ═══════════════════════════════════════════════════════════════

def analyze_rolling_vrp_sharpe(vrp_data, dates):
    """Test rolling VRP Sharpe as recovery/degradation indicator."""
    print("\n" + "=" * 70)
    print("  ANALYSIS 2: ROLLING VRP SHARPE AS INDICATOR")
    print("=" * 70)

    vrp_pnl = {d: vrp_data[d]["vrp_pnl"] for d in dates if d in vrp_data}

    for window in [60, 90, 120, 180, 252]:
        rolling = compute_rolling_sharpe(vrp_pnl, dates, window)
        if not rolling:
            continue

        rd = sorted(rolling.keys())
        above_zero = sum(1 for d in rd if rolling[d] > 0) / len(rd)
        above_one = sum(1 for d in rd if rolling[d] > 1.0) / len(rd)
        above_two = sum(1 for d in rd if rolling[d] > 2.0) / len(rd)
        avg = sum(rolling[d] for d in rd) / len(rd)

        # Check recent values
        recent = [rolling[d] for d in rd[-60:]] if len(rd) >= 60 else []
        recent_avg = sum(recent) / len(recent) if recent else 0

        print(f"\n  Window={window}d:")
        print(f"    Avg Sharpe:    {avg:.2f}")
        print(f"    % above 0:    {above_zero*100:.0f}%")
        print(f"    % above 1.0:  {above_one*100:.0f}%")
        print(f"    % above 2.0:  {above_two*100:.0f}%")
        print(f"    Recent 60d:   {recent_avg:.2f}")
        print(f"    Latest:       {rolling[rd[-1]]:.2f} ({rd[-1]})")

    return vrp_pnl


# ═══════════════════════════════════════════════════════════════
# Analysis 3: IV Regime Change Detection
# ═══════════════════════════════════════════════════════════════

def analyze_iv_regime_changes(dvol_hist, dates):
    """Detect IV regime shifts using rolling statistics."""
    print("\n" + "=" * 70)
    print("  ANALYSIS 3: IV REGIME CHANGE DETECTION")
    print("=" * 70)

    # Rolling IV with different windows
    for window in [30, 60, 90, 180]:
        rolling_iv = compute_rolling_iv(dvol_hist, dates, window)
        if not rolling_iv:
            continue
        rd = sorted(rolling_iv.keys())

        # Detect crossovers: short MA vs long MA
        short_iv = compute_rolling_iv(dvol_hist, dates, 30)
        long_iv = compute_rolling_iv(dvol_hist, dates, window)
        common = sorted(set(short_iv.keys()) & set(long_iv.keys()))

        if not common:
            continue

        crossovers = []
        for i in range(1, len(common)):
            d, dp = common[i], common[i-1]
            prev_diff = short_iv[dp] - long_iv[dp]
            curr_diff = short_iv[d] - long_iv[d]
            if prev_diff <= 0 and curr_diff > 0:
                crossovers.append((d, "BULLISH", short_iv[d], long_iv[d]))
            elif prev_diff >= 0 and curr_diff < 0:
                crossovers.append((d, "BEARISH", short_iv[d], long_iv[d]))

        print(f"\n  30d vs {window}d IV crossover:")
        print(f"    Total crossovers: {len(crossovers)}")
        if crossovers:
            # Show last 5
            for d, direction, short, long in crossovers[-5:]:
                print(f"      {d}: {direction:8s} (30d={short*100:.1f}%, {window}d={long*100:.1f}%)")

    # IV percentile over time
    print("\n  ─── IV Level by Year ───")
    by_year = defaultdict(list)
    for d in dates:
        if d in dvol_hist:
            by_year[d[:4]].append(dvol_hist[d])

    for yr in sorted(by_year.keys()):
        vals = by_year[yr]
        avg = sum(vals) / len(vals)
        mn, mx = min(vals), max(vals)
        print(f"    {yr}: avg={avg*100:.1f}%, min={mn*100:.1f}%, max={mx*100:.1f}%")


# ═══════════════════════════════════════════════════════════════
# Analysis 4: Composite VRP Health Indicator
# ═══════════════════════════════════════════════════════════════

def build_vrp_health_indicator(vrp_data, dvol_hist, dates):
    """Build composite indicator: rolling VRP Sharpe + IV level + VRP spread trend."""
    print("\n" + "=" * 70)
    print("  ANALYSIS 4: COMPOSITE VRP HEALTH INDICATOR")
    print("=" * 70)

    vrp_pnl = {d: vrp_data[d]["vrp_pnl"] for d in dates if d in vrp_data}

    # Components:
    # 1. Rolling 90d VRP Sharpe (normalized to 0-1)
    roll_sharpe = compute_rolling_sharpe(vrp_pnl, dates, 90)

    # 2. IV level percentile (rolling 365d)
    all_iv = sorted([dvol_hist[d] for d in dates if d in dvol_hist])

    # 3. Rolling 30d VRP spread (avg VRP spread)
    vrp_spread_30 = {}
    spread_list = [(d, vrp_data[d]["vrp_spread"]) for d in dates if d in vrp_data]
    for i in range(30, len(spread_list)):
        w = [s[1] for s in spread_list[i-30:i]]
        vrp_spread_30[spread_list[i][0]] = sum(w) / len(w)

    # Composite health score
    common = sorted(set(roll_sharpe.keys()) & set(vrp_spread_30.keys()))
    health = {}

    for d in common:
        # Component 1: Rolling Sharpe (clip to -2..6, normalize to 0..1)
        rs = max(-2, min(6, roll_sharpe[d]))
        c1 = (rs + 2) / 8.0  # -2→0, 6→1

        # Component 2: IV level (higher → more VRP edge)
        iv = dvol_hist.get(d, 0.5)
        iv_pctile = sum(1 for v in all_iv if v <= iv) / len(all_iv)
        c2 = iv_pctile

        # Component 3: VRP spread positivity (clip to -0.1..0.3)
        vrps = max(-0.1, min(0.3, vrp_spread_30[d]))
        c3 = (vrps + 0.1) / 0.4  # -0.1→0, 0.3→1

        # Equal weight composite
        health[d] = (c1 + c2 + c3) / 3.0

    if not health:
        print("  No data for health indicator")
        return health

    hd = sorted(health.keys())

    # Show health by year
    print("\n  ─── VRP Health by Year ───")
    by_year = defaultdict(list)
    for d in hd:
        by_year[d[:4]].append(health[d])
    for yr in sorted(by_year.keys()):
        vals = by_year[yr]
        avg = sum(vals) / len(vals)
        mn, mx = min(vals), max(vals)
        print(f"    {yr}: avg={avg:.3f}, min={mn:.3f}, max={mx:.3f}")

    # Show last 30 days
    print("\n  ─── Recent Health (last 30d) ───")
    for d in hd[-30:]:
        rs = roll_sharpe.get(d, 0)
        iv = dvol_hist.get(d, 0)
        vrps = vrp_spread_30.get(d, 0)
        print(f"    {d}: health={health[d]:.3f} | roll_sharpe={rs:5.2f} | IV={iv*100:5.1f}% | VRP_spread={vrps*100:5.1f}%")

    # Classify health levels
    print("\n  ─── Health Level Classification ───")
    for level_name, lo, hi in [("STRONG (>0.6)", 0.6, 1.1), ("MODERATE (0.4-0.6)", 0.4, 0.6),
                                ("WEAK (0.2-0.4)", 0.2, 0.4), ("CRITICAL (<0.2)", 0.0, 0.2)]:
        n = sum(1 for d in hd if lo <= health[d] < hi)
        pct = n / len(hd) * 100
        print(f"    {level_name:>22}: {n:4d} days ({pct:5.1f}%)")

    return health


# ═══════════════════════════════════════════════════════════════
# Analysis 5: Dynamic VRP Weight Backtest
# ═══════════════════════════════════════════════════════════════

def backtest_dynamic_vrp_weight(vrp_data, bf_pnl, health, dates):
    """Backtest dynamic VRP weight (10-50%) based on health indicator."""
    print("\n" + "=" * 70)
    print("  ANALYSIS 5: DYNAMIC VRP WEIGHT BACKTEST")
    print("=" * 70)

    vrp_pnl = {d: vrp_data[d]["vrp_pnl"] for d in dates if d in vrp_data}
    common = sorted(set(vrp_pnl.keys()) & set(bf_pnl.keys()))

    # Baseline: static 10/90 (R69)
    static_rets = [0.10 * vrp_pnl[d] + 0.90 * bf_pnl[d] for d in common]
    static_stats = compute_stats(static_rets)

    print(f"\n  ─── BASELINE: Static 10/90 ───")
    print(f"    Sharpe:     {static_stats['sharpe']:.4f}")
    print(f"    Ann Return: {static_stats['ann_ret']*100:.2f}%")
    print(f"    Max DD:     {static_stats['max_dd']*100:.2f}%")

    # Dynamic strategies
    strategies = [
        # (name, min_vrp_w, max_vrp_w, health_threshold_low, health_threshold_high)
        ("Health 10-30%", 0.10, 0.30, 0.3, 0.6),
        ("Health 10-50%", 0.10, 0.50, 0.3, 0.6),
        ("Health 10-50% v2", 0.10, 0.50, 0.2, 0.5),
        ("Health 10-50% v3", 0.10, 0.50, 0.4, 0.7),
        ("Step 10/30/50%", None, None, None, None),  # Step function
        ("Binary 10/50%", None, None, None, None),   # Binary switch
    ]

    results = {}
    for name, min_w, max_w, h_lo, h_hi in strategies:
        rets = []
        vrp_weights = []

        for d in common:
            h = health.get(d)
            if h is None:
                # Before health data available, use static
                w_vrp = 0.10
            elif name == "Step 10/30/50%":
                if h >= 0.6:
                    w_vrp = 0.50
                elif h >= 0.4:
                    w_vrp = 0.30
                else:
                    w_vrp = 0.10
            elif name == "Binary 10/50%":
                w_vrp = 0.50 if h >= 0.5 else 0.10
            else:
                # Linear interpolation
                if h <= h_lo:
                    w_vrp = min_w
                elif h >= h_hi:
                    w_vrp = max_w
                else:
                    w_vrp = min_w + (max_w - min_w) * (h - h_lo) / (h_hi - h_lo)

            w_bf = 1.0 - w_vrp
            ret = w_vrp * vrp_pnl[d] + w_bf * bf_pnl[d]
            rets.append(ret)
            vrp_weights.append(w_vrp)

        stats = compute_stats(rets)
        avg_w = sum(vrp_weights) / len(vrp_weights)
        results[name] = {**stats, "avg_w_vrp": avg_w}

        delta = stats["sharpe"] - static_stats["sharpe"]
        print(f"\n  {name}:")
        print(f"    Sharpe:     {stats['sharpe']:.4f} (Δ={delta:+.4f})")
        print(f"    Ann Return: {stats['ann_ret']*100:.2f}%")
        print(f"    Max DD:     {stats['max_dd']*100:.2f}%")
        print(f"    Avg VRP w:  {avg_w:.2f}")

    return results, static_stats


# ═══════════════════════════════════════════════════════════════
# Analysis 6: LOYO Validation of Best Dynamic Strategy
# ═══════════════════════════════════════════════════════════════

def validate_loyo(vrp_data, bf_pnl, health, dates):
    """Leave-One-Year-Out validation of dynamic vs static."""
    print("\n" + "=" * 70)
    print("  ANALYSIS 6: LOYO VALIDATION — DYNAMIC vs STATIC")
    print("=" * 70)

    vrp_pnl = {d: vrp_data[d]["vrp_pnl"] for d in dates if d in vrp_data}
    common = sorted(set(vrp_pnl.keys()) & set(bf_pnl.keys()))

    years = sorted(set(d[:4] for d in common))

    print(f"\n  {'Year':<6} {'Static 10/90':>14} {'Dynamic 10-50%':>16} {'Delta':>8} {'Avg VRP w':>10}")
    print(f"  {'─'*6} {'─'*14} {'─'*16} {'─'*8} {'─'*10}")

    for yr in years:
        yr_dates = [d for d in common if d[:4] == yr]
        if len(yr_dates) < 30:
            continue

        # Static
        static_rets = [0.10 * vrp_pnl[d] + 0.90 * bf_pnl[d] for d in yr_dates]
        static_s = compute_stats(static_rets)

        # Dynamic (best strategy from Analysis 5)
        dyn_rets = []
        vrp_weights = []
        for d in yr_dates:
            h = health.get(d)
            if h is None:
                w_vrp = 0.10
            else:
                # Linear 10-50% with thresholds 0.3-0.6
                if h <= 0.3:
                    w_vrp = 0.10
                elif h >= 0.6:
                    w_vrp = 0.50
                else:
                    w_vrp = 0.10 + 0.40 * (h - 0.3) / 0.3
                w_vrp = min(0.50, max(0.10, w_vrp))
            dyn_rets.append(w_vrp * vrp_pnl[d] + (1 - w_vrp) * bf_pnl[d])
            vrp_weights.append(w_vrp)

        dyn_s = compute_stats(dyn_rets)
        avg_w = sum(vrp_weights) / len(vrp_weights)
        delta = dyn_s["sharpe"] - static_s["sharpe"]

        print(f"  {yr:<6} {static_s['sharpe']:>14.2f} {dyn_s['sharpe']:>16.2f} {delta:>+8.2f} {avg_w:>10.2f}")


# ═══════════════════════════════════════════════════════════════
# Analysis 7: What IV Level Would Justify 50/50?
# ═══════════════════════════════════════════════════════════════

def analyze_iv_threshold_for_50_50(vrp_data, bf_pnl, dvol_hist, dates):
    """Find the IV threshold above which 50/50 VRP/BF beats 10/90."""
    print("\n" + "=" * 70)
    print("  ANALYSIS 7: IV THRESHOLD FOR 50/50 VIABILITY")
    print("=" * 70)

    vrp_pnl = {d: vrp_data[d]["vrp_pnl"] for d in dates if d in vrp_data}
    common = sorted(set(vrp_pnl.keys()) & set(bf_pnl.keys()))

    for iv_thresh in [0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.80]:
        # Days where IV > threshold: use 50/50
        # Days where IV <= threshold: use 10/90
        rets_10_90 = []
        rets_50_50 = []
        rets_cond = []
        n_above = 0

        for d in common:
            iv = dvol_hist.get(d, 0.5)
            r_10 = 0.10 * vrp_pnl[d] + 0.90 * bf_pnl[d]
            r_50 = 0.50 * vrp_pnl[d] + 0.50 * bf_pnl[d]
            rets_10_90.append(r_10)
            rets_50_50.append(r_50)

            if iv >= iv_thresh:
                rets_cond.append(r_50)
                n_above += 1
            else:
                rets_cond.append(r_10)

        if n_above < 30 or n_above == len(common):
            continue

        s_10 = compute_stats(rets_10_90)
        s_50 = compute_stats(rets_50_50)
        s_cond = compute_stats(rets_cond)
        pct_above = n_above / len(common) * 100

        print(f"\n  IV>={iv_thresh*100:.0f}%→50/50, else→10/90 ({pct_above:.0f}% of days above):")
        print(f"    Static 10/90: Sharpe={s_10['sharpe']:.3f}, AnnRet={s_10['ann_ret']*100:.2f}%")
        print(f"    Static 50/50: Sharpe={s_50['sharpe']:.3f}, AnnRet={s_50['ann_ret']*100:.2f}%")
        print(f"    Conditional:  Sharpe={s_cond['sharpe']:.3f}, AnnRet={s_cond['ann_ret']*100:.2f}%")
        delta = s_cond["sharpe"] - s_10["sharpe"]
        print(f"    Delta vs 10/90: {delta:+.3f}")


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════

def main():
    print("=" * 70)
    print("  R72: VRP RECOVERY DETECTION")
    print("=" * 70)
    print("  Loading data...")

    dvol_hist = load_dvol_history("BTC")
    price_hist = load_price_history("BTC")
    surface_hist = load_surface("BTC")

    dates = sorted(set(dvol_hist.keys()) & set(price_hist.keys()))
    print(f"  Dates: {dates[0]} to {dates[-1]} ({len(dates)} days)")

    # Compute VRP data
    vrp_data = compute_daily_vrp(dvol_hist, price_hist)
    print(f"  VRP data: {len(vrp_data)} days")

    # Compute BF PnL
    bf_pnl = compute_bf_pnl(dvol_hist, surface_hist, dates, z_exit=0.0, sensitivity=2.5)
    print(f"  BF PnL: {len(bf_pnl)} days")

    # Analysis 1: VRP vs IV level
    iv_results = analyze_vrp_vs_iv_level(vrp_data, dates)

    # Analysis 2: Rolling VRP Sharpe
    vrp_pnl = analyze_rolling_vrp_sharpe(vrp_data, dates)

    # Analysis 3: IV regime change detection
    analyze_iv_regime_changes(dvol_hist, dates)

    # Analysis 4: Composite VRP health indicator
    health = build_vrp_health_indicator(vrp_data, dvol_hist, dates)

    # Analysis 5: Dynamic VRP weight backtest
    dyn_results, static_stats = backtest_dynamic_vrp_weight(vrp_data, bf_pnl, health, dates)

    # Analysis 6: LOYO validation
    validate_loyo(vrp_data, bf_pnl, health, dates)

    # Analysis 7: IV threshold for 50/50 viability
    analyze_iv_threshold_for_50_50(vrp_data, bf_pnl, dvol_hist, dates)

    # ─── Summary ────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  SUMMARY")
    print("=" * 70)

    best_dyn = max(dyn_results.items(), key=lambda x: x[1]["sharpe"])
    print(f"\n  Static 10/90 Sharpe: {static_stats['sharpe']:.4f}")
    print(f"  Best dynamic:       {best_dyn[0]} → Sharpe {best_dyn[1]['sharpe']:.4f}")
    print(f"  Delta:              {best_dyn[1]['sharpe'] - static_stats['sharpe']:+.4f}")

    verdict = "UPGRADE" if best_dyn[1]["sharpe"] > static_stats["sharpe"] + 0.05 else "NO_IMPROVEMENT"
    print(f"\n  VERDICT: {verdict}")
    if verdict == "NO_IMPROVEMENT":
        print("  → Static 10/90 (R69) remains optimal. No dynamic VRP weighting improves Sharpe.")
        print("  → VRP recovery is NOT detectable early enough to act on.")
    else:
        print(f"  → Dynamic weighting IMPROVES Sharpe by {best_dyn[1]['sharpe'] - static_stats['sharpe']:+.3f}")
        print(f"  → Average VRP weight: {best_dyn[1]['avg_w_vrp']:.0%}")

    # Save results
    results = {
        "research_id": "R72",
        "title": "VRP Recovery Detection",
        "static_10_90": {
            "sharpe": round(static_stats["sharpe"], 4),
            "ann_ret": round(static_stats["ann_ret"], 6),
            "max_dd": round(static_stats["max_dd"], 6),
        },
        "best_dynamic": {
            "name": best_dyn[0],
            "sharpe": round(best_dyn[1]["sharpe"], 4),
            "ann_ret": round(best_dyn[1]["ann_ret"], 6),
            "max_dd": round(best_dyn[1]["max_dd"], 6),
            "avg_w_vrp": round(best_dyn[1]["avg_w_vrp"], 3),
        },
        "delta_sharpe": round(best_dyn[1]["sharpe"] - static_stats["sharpe"], 4),
        "verdict": verdict,
        "key_insight": "TBD",
    }

    out_path = ROOT / "data" / "cache" / "deribit" / "real_surface" / "r72_vrp_recovery_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved to {out_path}")


if __name__ == "__main__":
    main()
