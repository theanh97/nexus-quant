#!/usr/bin/env python3
"""
R78: BF Structural Compression Analysis
==========================================

R77 flagged a concern: butterfly_25d feature std is declining (0.022→0.007 from 2021→2026).
This is analogous to the IV structural decline that caused VRP degradation (R65).

Key questions:
  1. Is the BF std decline statistically significant or just cyclical?
  2. What's driving it? (IV level decline, market maturation, or something else?)
  3. If BF std → 0, does the BF edge disappear?
  4. What's the BF std threshold below which the strategy breaks?
  5. Can we decompose BF into components to identify the compression source?
  6. What's the BF compression forecast?
  7. Are there any structural breaks in the BF time series?

Production impact: If BF edge decays like VRP, our Sharpe 3.76 portfolio has no fallback.
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


def compute_bf_pnl(dvol_hist, surface_hist, dates, lb=120, z_entry=1.5,
                    z_exit=0.0, sensitivity=2.5):
    """Compute BF PnL series."""
    bf_vals = {}
    for d in dates:
        if d in surface_hist and "butterfly_25d" in surface_hist[d]:
            bf_vals[d] = surface_hist[d]["butterfly_25d"]

    dt = 1.0 / 365.0
    position = 0.0
    pnl = {}
    z_scores = {}

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
        z_scores[d] = z

        if z > z_entry:
            position = -1.0
        elif z < -z_entry:
            position = 1.0
        elif z_exit > 0 and abs(z) < z_exit:
            position = 0.0

        iv = dvol_hist.get(d)
        f_now, f_prev = bf_vals.get(d), bf_vals.get(dp)
        if f_now is not None and f_prev is not None and iv is not None and position != 0:
            day_pnl = position * (f_now - f_prev) * iv * math.sqrt(dt) * sensitivity
            pnl[d] = day_pnl
        else:
            pnl[d] = 0.0

    return pnl, z_scores, bf_vals


def compute_stats(rets):
    if len(rets) < 20:
        return {"sharpe": 0, "ann_ret": 0, "max_dd": 0}
    mean = sum(rets) / len(rets)
    std = math.sqrt(sum((r - mean)**2 for r in rets) / len(rets))
    sharpe = (mean * 365) / (std * math.sqrt(365)) if std > 0 else 0
    ann_ret = mean * 365
    cum = peak = max_dd = 0.0
    for r in rets:
        cum += r
        peak = max(peak, cum)
        max_dd = min(max_dd, cum - peak)
    return {"sharpe": round(sharpe, 3), "ann_ret": round(ann_ret * 100, 3), "max_dd": round(max_dd * 100, 3)}


# ═══════════════════════════════════════════════════════════════
# Analysis 1: BF Std Decline — Significance Testing
# ═══════════════════════════════════════════════════════════════

def bf_std_significance(bf_vals, dates):
    print("\n" + "=" * 70)
    print("  ANALYSIS 1: BF STD DECLINE — STATISTICAL SIGNIFICANCE")
    print("=" * 70)

    # Compute rolling 90d std
    bf_list = [(d, bf_vals[d]) for d in dates if d in bf_vals]
    rolling_std = {}
    for i in range(90, len(bf_list)):
        d = bf_list[i][0]
        w = [v[1] for v in bf_list[i-90:i]]
        mean = sum(w) / len(w)
        std = math.sqrt(sum((v - mean)**2 for v in w) / len(w))
        rolling_std[d] = std

    # By half-year
    by_half = defaultdict(list)
    for d, v in rolling_std.items():
        yr = int(d[:4])
        half = "H1" if int(d[5:7]) <= 6 else "H2"
        by_half[f"{yr}{half}"].append(v)

    print(f"\n  {'Period':>8} {'Mean Std':>10} {'Std of Std':>12} {'N':>6} {'Trend':>10}")
    print(f"  {'─'*8} {'─'*10} {'─'*12} {'─'*6} {'─'*10}")

    prev_mean = None
    std_trajectory = []
    for period in sorted(by_half.keys()):
        vals = by_half[period]
        mean = sum(vals) / len(vals)
        std_of_std = math.sqrt(sum((v - mean)**2 for v in vals) / len(vals))
        trend = f"{mean - prev_mean:+.5f}" if prev_mean is not None else "—"
        print(f"  {period:>8} {mean:>10.5f} {std_of_std:>12.5f} {len(vals):>6} {trend:>10}")
        std_trajectory.append(mean)
        prev_mean = mean

    # Linear regression: std vs time
    n = len(std_trajectory)
    if n >= 4:
        x = list(range(n))
        y = std_trajectory
        x_mean = sum(x) / n
        y_mean = sum(y) / n
        ss_xy = sum((x[i] - x_mean) * (y[i] - y_mean) for i in range(n))
        ss_xx = sum((x[i] - x_mean)**2 for i in range(n))
        if ss_xx > 0:
            slope = ss_xy / ss_xx
            intercept = y_mean - slope * x_mean
            # R²
            ss_yy = sum((y[i] - y_mean)**2 for i in range(n))
            y_hat = [intercept + slope * x[i] for i in range(n)]
            ss_res = sum((y[i] - y_hat[i])**2 for i in range(n))
            r_sq = 1 - ss_res / ss_yy if ss_yy > 0 else 0
            # t-statistic for slope
            mse = ss_res / (n - 2) if n > 2 else 0
            se_slope = math.sqrt(mse / ss_xx) if ss_xx > 0 and mse > 0 else 0
            t_stat = slope / se_slope if se_slope > 0 else 0
            # Annualized slope (2 halves per year)
            ann_slope = slope * 2  # per half → per year

            print(f"\n  Linear regression (BF std vs time):")
            print(f"    Slope per half-year: {slope:.6f}")
            print(f"    Annualized slope:    {ann_slope:.6f} per year")
            print(f"    R²:                  {r_sq:.3f}")
            print(f"    t-statistic:         {t_stat:.2f}")
            print(f"    Significant:         {'YES' if abs(t_stat) > 2.0 else 'NO'} (|t| > 2.0)")

            # Extrapolate: when does BF std hit zero?
            if slope < 0:
                half_to_zero = -intercept / slope if slope != 0 else float('inf')
                years_to_zero = half_to_zero / 2  # halves → years
                yr_from_start = 2021 + years_to_zero
                print(f"\n    ⚠ At current rate, BF std → 0 in ~{years_to_zero:.1f} years ({yr_from_start:.0f})")
            else:
                print(f"\n    BF std is NOT declining — no concern")

            return {
                "slope_per_half": slope,
                "ann_slope": ann_slope,
                "r_squared": r_sq,
                "t_stat": t_stat,
                "significant": abs(t_stat) > 2.0,
                "trajectory": std_trajectory
            }

    return {"trajectory": std_trajectory}


# ═══════════════════════════════════════════════════════════════
# Analysis 2: What's Driving the Compression?
# ═══════════════════════════════════════════════════════════════

def compression_drivers(surface_hist, dvol_hist, price_hist, dates):
    print("\n" + "=" * 70)
    print("  ANALYSIS 2: COMPRESSION DRIVERS — WHAT'S CAUSING BF STD DECLINE?")
    print("=" * 70)

    # Butterfly_25d = call_25d + put_25d - 2*iv_atm
    # If IV declines, the absolute butterfly value can shrink even if
    # the normalized (butterfly/IV) ratio stays constant.
    # Also check: IV variance, price volatility, and market maturation

    # Component 1: BF absolute level vs BF/IV ratio
    bf_abs = []
    bf_ratio = []
    iv_levels = []
    rv_levels = []

    for d in dates:
        if d in surface_hist and "butterfly_25d" in surface_hist[d] and "iv_atm" in surface_hist[d]:
            bf = surface_hist[d]["butterfly_25d"]
            iv = surface_hist[d]["iv_atm"]
            bf_abs.append((d, bf))
            if iv > 0:
                bf_ratio.append((d, bf / iv))
            iv_levels.append((d, iv))

    # Compute RV from prices
    price_dates = sorted(d for d in dates if d in price_hist)
    for i in range(30, len(price_dates)):
        d = price_dates[i]
        rets = []
        for j in range(i-30, i):
            p0 = price_hist.get(price_dates[j])
            p1 = price_hist.get(price_dates[j+1]) if j+1 < len(price_dates) else None
            if p0 and p1 and p0 > 0:
                rets.append(math.log(p1 / p0))
        if rets:
            rv = math.sqrt(sum(r**2 for r in rets) / len(rets)) * math.sqrt(365)
            rv_levels.append((d, rv))

    # By year: BF absolute std vs BF/IV ratio std
    def yearly_std(series):
        by_yr = defaultdict(list)
        for d, v in series:
            by_yr[d[:4]].append(v)
        result = {}
        for yr, vals in sorted(by_yr.items()):
            mean = sum(vals) / len(vals)
            std = math.sqrt(sum((v - mean)**2 for v in vals) / len(vals))
            result[yr] = {"mean": mean, "std": std, "n": len(vals)}
        return result

    bf_abs_yearly = yearly_std(bf_abs)
    bf_ratio_yearly = yearly_std(bf_ratio)
    iv_yearly = yearly_std(iv_levels)
    rv_yearly = yearly_std(rv_levels)

    print(f"\n  {'Year':>6} {'BF Std':>8} {'BF/IV Std':>10} {'IV Mean':>10} {'IV Std':>8} {'RV Std':>8}")
    print(f"  {'─'*6} {'─'*8} {'─'*10} {'─'*10} {'─'*8} {'─'*8}")

    for yr in sorted(bf_abs_yearly.keys()):
        bf_s = bf_abs_yearly[yr]["std"]
        ratio_s = bf_ratio_yearly.get(yr, {}).get("std", 0)
        iv_m = iv_yearly.get(yr, {}).get("mean", 0)
        iv_s = iv_yearly.get(yr, {}).get("std", 0)
        rv_s = rv_yearly.get(yr, {}).get("std", 0)
        print(f"  {yr:>6} {bf_s:>8.5f} {ratio_s:>10.5f} {iv_m:>10.3f} {iv_s:>8.3f} {rv_s:>8.3f}")

    # Correlation: BF std vs IV level (rolling basis)
    bf_90d_std = {}
    iv_90d_mean = {}
    for i in range(90, len(bf_abs)):
        d = bf_abs[i][0]
        w = [v[1] for v in bf_abs[i-90:i]]
        mean = sum(w) / len(w)
        bf_90d_std[d] = math.sqrt(sum((v - mean)**2 for v in w) / len(w))

    for i in range(90, len(iv_levels)):
        d = iv_levels[i][0]
        w = [v[1] for v in iv_levels[i-90:i]]
        iv_90d_mean[d] = sum(w) / len(w)

    # Correlation
    common = sorted(set(bf_90d_std.keys()) & set(iv_90d_mean.keys()))
    if len(common) >= 30:
        x = [iv_90d_mean[d] for d in common]
        y = [bf_90d_std[d] for d in common]
        x_m = sum(x) / len(x)
        y_m = sum(y) / len(y)
        cov = sum((x[i] - x_m) * (y[i] - y_m) for i in range(len(common))) / len(common)
        sx = math.sqrt(sum((v - x_m)**2 for v in x) / len(x))
        sy = math.sqrt(sum((v - y_m)**2 for v in y) / len(y))
        corr = cov / (sx * sy) if sx > 0 and sy > 0 else 0
        print(f"\n  Correlation(90d BF std, 90d IV mean) = {corr:.3f}")
        if corr > 0.5:
            print(f"  → BF compression is DRIVEN BY IV decline (r={corr:.3f})")
            print(f"  → As IV structurally declines, butterfly absolute spread compresses")
        elif corr > 0.2:
            print(f"  → BF compression PARTIALLY linked to IV decline")
        else:
            print(f"  → BF compression is INDEPENDENT of IV — structural change in smile shape")

    # Check BF/IV ratio stability — is the NORMALIZED butterfly stable?
    ratio_90d_std = {}
    for i in range(90, len(bf_ratio)):
        d = bf_ratio[i][0]
        w = [v[1] for v in bf_ratio[i-90:i]]
        mean = sum(w) / len(w)
        ratio_90d_std[d] = math.sqrt(sum((v - mean)**2 for v in w) / len(w))

    by_half_ratio = defaultdict(list)
    for d, v in ratio_90d_std.items():
        yr = int(d[:4])
        half = "H1" if int(d[5:7]) <= 6 else "H2"
        by_half_ratio[f"{yr}{half}"].append(v)

    print(f"\n  Normalized BF/IV ratio std by half-year:")
    print(f"  {'Period':>8} {'BF/IV Std':>12}")
    print(f"  {'─'*8} {'─'*12}")
    for period in sorted(by_half_ratio.keys()):
        vals = by_half_ratio[period]
        mean = sum(vals) / len(vals)
        print(f"  {period:>8} {mean:>12.5f}")

    return {"corr_bf_std_iv": corr if len(common) >= 30 else None}


# ═══════════════════════════════════════════════════════════════
# Analysis 3: BF Sharpe vs BF Feature Variance
# ═══════════════════════════════════════════════════════════════

def sharpe_vs_variance(pnl, bf_vals, dates):
    print("\n" + "=" * 70)
    print("  ANALYSIS 3: BF SHARPE vs FEATURE VARIANCE — THRESHOLD ANALYSIS")
    print("=" * 70)

    # Question: at what BF std level does the strategy break?
    # Segment data into low/mid/high BF variance periods

    bf_list = [(d, bf_vals[d]) for d in dates if d in bf_vals]
    rolling_std = {}
    for i in range(60, len(bf_list)):
        d = bf_list[i][0]
        w = [v[1] for v in bf_list[i-60:i]]
        mean = sum(w) / len(w)
        rolling_std[d] = math.sqrt(sum((v - mean)**2 for v in w) / len(w))

    # Segment into quintiles
    std_vals = sorted(rolling_std.values())
    n = len(std_vals)
    quintiles = [std_vals[min(int(n * q / 5), n - 1)] for q in range(6)]

    print(f"\n  BF 60d rolling std quintiles:")
    for q in range(5):
        print(f"    Q{q+1}: {quintiles[q]:.5f} to {quintiles[q+1]:.5f}")

    # Compute Sharpe for each quintile
    print(f"\n  {'Quintile':>10} {'Std Range':>16} {'Sharpe':>8} {'Ann Ret':>10} {'N Days':>8} {'Hit%':>8}")
    print(f"  {'─'*10} {'─'*16} {'─'*8} {'─'*10} {'─'*8} {'─'*8}")

    quintile_results = {}
    for q in range(5):
        lo, hi = quintiles[q], quintiles[q+1]
        q_rets = []
        for d in dates:
            if d in rolling_std and d in pnl:
                s = rolling_std[d]
                if (q < 4 and lo <= s < hi) or (q == 4 and lo <= s <= hi):
                    q_rets.append(pnl[d])

        if len(q_rets) >= 30:
            stats = compute_stats(q_rets)
            hit = sum(1 for r in q_rets if r > 0) / len(q_rets) * 100
            print(f"  {'Q'+str(q+1):>10} {lo:.5f}-{hi:.5f} {stats['sharpe']:>8.2f} {stats['ann_ret']:>9.3f}% {len(q_rets):>8} {hit:>7.1f}%")
            quintile_results[f"Q{q+1}"] = {
                "std_range": f"{lo:.5f}-{hi:.5f}",
                "sharpe": stats["sharpe"],
                "n_days": len(q_rets)
            }

    # Finer grid at low end — find the breaking point
    print(f"\n  Fine grid: BF std thresholds where Sharpe degrades")
    print(f"  {'Threshold':>12} {'Sharpe (below)':>16} {'Sharpe (above)':>16} {'N below':>10} {'N above':>10}")
    print(f"  {'─'*12} {'─'*16} {'─'*16} {'─'*10} {'─'*10}")

    thresholds = [0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.010, 0.012, 0.015]
    threshold_results = {}
    for th in thresholds:
        below_rets = [pnl[d] for d in dates if d in rolling_std and d in pnl and rolling_std[d] < th]
        above_rets = [pnl[d] for d in dates if d in rolling_std and d in pnl and rolling_std[d] >= th]

        if len(below_rets) >= 20 and len(above_rets) >= 20:
            s_below = compute_stats(below_rets)
            s_above = compute_stats(above_rets)
            print(f"  {th:>12.4f} {s_below['sharpe']:>16.2f} {s_above['sharpe']:>16.2f} {len(below_rets):>10} {len(above_rets):>10}")
            threshold_results[f"{th:.4f}"] = {
                "sharpe_below": s_below["sharpe"],
                "sharpe_above": s_above["sharpe"],
                "n_below": len(below_rets),
                "n_above": len(above_rets)
            }

    return {"quintiles": quintile_results, "thresholds": threshold_results}


# ═══════════════════════════════════════════════════════════════
# Analysis 4: BF Decomposition — Smile Shape Evolution
# ═══════════════════════════════════════════════════════════════

def smile_decomposition(surface_hist, dates):
    print("\n" + "=" * 70)
    print("  ANALYSIS 4: SMILE SHAPE EVOLUTION — BF COMPONENT DECOMPOSITION")
    print("=" * 70)

    # butterfly_25d = call_25d_iv + put_25d_iv - 2 * iv_atm
    # When butterfly > 0: wings are above ATM (convex smile)
    # butterfly_25d = iv_atm + skew + wing_premium  (approx)
    #
    # If BF is compressing, it could be:
    # a) Wings flattening (less convexity) — structural market change
    # b) IV level declining (less absolute spread)
    # c) Both

    # We can check: butterfly/IV² (controls for IV level quadratically)
    # and butterfly/IV (linear normalization)

    by_year = defaultdict(list)
    bf_levels = defaultdict(list)
    bf_iv_ratio = defaultdict(list)
    bf_iv2_ratio = defaultdict(list)
    skew_levels = defaultdict(list)
    iv_levels = defaultdict(list)

    for d in dates:
        if d not in surface_hist:
            continue
        s = surface_hist[d]
        bf = s.get("butterfly_25d")
        iv = s.get("iv_atm")
        sk = s.get("skew_25d")
        if bf is not None and iv is not None and iv > 0:
            yr = d[:4]
            by_year[yr].append(d)
            bf_levels[yr].append(bf)
            bf_iv_ratio[yr].append(bf / iv)
            bf_iv2_ratio[yr].append(bf / (iv * iv))
            if sk is not None:
                skew_levels[yr].append(sk)
            iv_levels[yr].append(iv)

    print(f"\n  {'Year':>6} {'BF Mean':>10} {'BF/IV':>10} {'BF/IV²':>10} {'IV Mean':>10} {'Skew Mean':>10} {'N':>6}")
    print(f"  {'─'*6} {'─'*10} {'─'*10} {'─'*10} {'─'*10} {'─'*10} {'─'*6}")

    results = {}
    for yr in sorted(by_year.keys()):
        bf_m = sum(bf_levels[yr]) / len(bf_levels[yr])
        ratio_m = sum(bf_iv_ratio[yr]) / len(bf_iv_ratio[yr])
        ratio2_m = sum(bf_iv2_ratio[yr]) / len(bf_iv2_ratio[yr])
        iv_m = sum(iv_levels[yr]) / len(iv_levels[yr])
        sk_m = sum(skew_levels[yr]) / len(skew_levels[yr]) if skew_levels[yr] else 0
        print(f"  {yr:>6} {bf_m:>10.5f} {ratio_m:>10.4f} {ratio2_m:>10.3f} {iv_m:>10.3f} {sk_m:>10.3f} {len(bf_levels[yr]):>6}")
        results[yr] = {
            "bf_mean": bf_m,
            "bf_iv_ratio": ratio_m,
            "bf_iv2_ratio": ratio2_m,
            "iv_mean": iv_m,
            "skew_mean": sk_m
        }

    # Key test: is BF/IV (normalized) also declining?
    ratio_stds = {}
    for yr in sorted(by_year.keys()):
        vals = bf_iv_ratio[yr]
        mean = sum(vals) / len(vals)
        std = math.sqrt(sum((v - mean)**2 for v in vals) / len(vals))
        ratio_stds[yr] = std

    print(f"\n  BF/IV ratio std by year (tests IV-normalized compression):")
    print(f"  {'Year':>6} {'BF Abs Std':>12} {'BF/IV Std':>12} {'Ratio':>10}")
    print(f"  {'─'*6} {'─'*12} {'─'*12} {'─'*10}")
    for yr in sorted(ratio_stds.keys()):
        bf_vals_yr = bf_levels[yr]
        bf_m = sum(bf_vals_yr) / len(bf_vals_yr)
        bf_std = math.sqrt(sum((v - bf_m)**2 for v in bf_vals_yr) / len(bf_vals_yr))
        ratio_std = ratio_stds[yr]
        ratio = bf_std / ratio_std if ratio_std > 0 else 0
        print(f"  {yr:>6} {bf_std:>12.5f} {ratio_std:>12.5f} {ratio:>10.3f}")

    return results


# ═══════════════════════════════════════════════════════════════
# Analysis 5: Counterfactual — BF on Normalized Feature
# ═══════════════════════════════════════════════════════════════

def bf_normalized_backtest(dvol_hist, surface_hist, dates):
    print("\n" + "=" * 70)
    print("  ANALYSIS 5: BF ON NORMALIZED FEATURE (butterfly/IV)")
    print("=" * 70)

    # If compression is driven by IV decline, then using butterfly/IV
    # as the feature (instead of raw butterfly) should be immune.
    # Compare: raw BF MR vs normalized BF MR

    configs = [
        ("raw_bf", "butterfly_25d", False),
        ("bf_div_iv", "butterfly_25d", True),  # butterfly / iv_atm
    ]

    for name, feat, normalize in configs:
        bf_vals = {}
        for d in dates:
            if d in surface_hist and feat in surface_hist[d]:
                val = surface_hist[d][feat]
                if normalize and "iv_atm" in surface_hist[d] and surface_hist[d]["iv_atm"] > 0:
                    val = val / surface_hist[d]["iv_atm"]
                elif normalize:
                    continue
                bf_vals[d] = val

        # Run BF MR strategy
        lb, z_entry, z_exit, sensitivity = 120, 1.5, 0.0, 2.5
        dt = 1.0 / 365.0
        position = 0.0
        pnl = {}

        active_dates = sorted(d for d in dates if d in bf_vals)
        for i in range(lb, len(active_dates)):
            d, dp = active_dates[i], active_dates[i-1]
            val = bf_vals[d]

            window = [bf_vals.get(active_dates[j]) for j in range(i-lb, i)]
            window = [v for v in window if v is not None]
            if len(window) < lb // 2:
                continue

            mean = sum(window) / len(window)
            std = math.sqrt(sum((v - mean)**2 for v in window) / len(window))
            if std < 1e-10:
                continue

            z = (val - mean) / std

            if z > z_entry:
                position = -1.0
            elif z < -z_entry:
                position = 1.0
            elif z_exit > 0 and abs(z) < z_exit:
                position = 0.0

            # PnL still uses RAW butterfly for execution
            iv = dvol_hist.get(d)
            raw_now = surface_hist.get(d, {}).get("butterfly_25d")
            raw_prev = surface_hist.get(dp, {}).get("butterfly_25d")
            if raw_now is not None and raw_prev is not None and iv is not None and position != 0:
                day_pnl = position * (raw_now - raw_prev) * iv * math.sqrt(dt) * sensitivity
                pnl[d] = day_pnl

        rets = [pnl[d] for d in sorted(pnl.keys())]
        stats = compute_stats(rets)

        # By year
        by_yr = defaultdict(list)
        for d, r in pnl.items():
            by_yr[d[:4]].append(r)

        print(f"\n  {name}: Sharpe {stats['sharpe']:.3f}, Ann Ret {stats['ann_ret']:.3f}%, MaxDD {stats['max_dd']:.3f}%")
        print(f"    By year: ", end="")
        for yr in sorted(by_yr.keys()):
            yr_stats = compute_stats(by_yr[yr])
            print(f"{yr}={yr_stats['sharpe']:.2f}  ", end="")
        print()

    # Also test: using the z-score from normalized feature but PnL from raw
    # This is important because the SIGNAL uses normalized data but EXECUTION uses raw
    print(f"\n  Note: bf_div_iv uses normalized feature for Z-SCORE SIGNALS but")
    print(f"        PnL execution still uses raw butterfly movement × IV × sqrt(dt)")
    print(f"        This tests whether the compression affects signal quality")


# ═══════════════════════════════════════════════════════════════
# Analysis 6: Compression Forecast & Risk Assessment
# ═══════════════════════════════════════════════════════════════

def compression_forecast(bf_vals, pnl, dates):
    print("\n" + "=" * 70)
    print("  ANALYSIS 6: COMPRESSION FORECAST & RISK ASSESSMENT")
    print("=" * 70)

    # Compute trailing BF std for each half-year
    bf_list = [(d, bf_vals[d]) for d in dates if d in bf_vals]

    # Compute 180d rolling std
    rolling_std = {}
    for i in range(180, len(bf_list)):
        d = bf_list[i][0]
        w = [v[1] for v in bf_list[i-180:i]]
        mean = sum(w) / len(w)
        rolling_std[d] = math.sqrt(sum((v - mean)**2 for v in w) / len(w))

    # Also compute 180d rolling PnL Sharpe
    pnl_list = [(d, pnl[d]) for d in sorted(pnl.keys()) if d in pnl]
    rolling_sharpe = {}
    for i in range(180, len(pnl_list)):
        d = pnl_list[i][0]
        w = [v[1] for v in pnl_list[i-180:i]]
        if len(w) < 60:
            continue
        mean = sum(w) / len(w)
        std = math.sqrt(sum((r - mean)**2 for r in w) / len(w))
        if std > 0:
            rolling_sharpe[d] = (mean * 365) / (std * math.sqrt(365))

    # Scatter: rolling std vs rolling sharpe
    common = sorted(set(rolling_std.keys()) & set(rolling_sharpe.keys()))
    if len(common) >= 30:
        x = [rolling_std[d] for d in common]
        y = [rolling_sharpe[d] for d in common]
        x_m = sum(x) / len(x)
        y_m = sum(y) / len(y)
        cov = sum((x[i] - x_m) * (y[i] - y_m) for i in range(len(common))) / len(common)
        sx = math.sqrt(sum((v - x_m)**2 for v in x) / len(x))
        sy = math.sqrt(sum((v - y_m)**2 for v in y) / len(y))
        corr = cov / (sx * sy) if sx > 0 and sy > 0 else 0

        print(f"\n  Correlation(180d BF std, 180d BF Sharpe) = {corr:.3f}")
        if corr > 0.3:
            print(f"  → POSITIVE correlation: lower variance → lower Sharpe → RISK")
        elif corr > -0.1:
            print(f"  → WEAK correlation: BF Sharpe is NOT directly linked to BF std → GOOD")
        else:
            print(f"  → NEGATIVE correlation: BF works better in low-variance periods → VERY GOOD")

    # Risk scenarios
    print(f"\n  Compression scenarios:")
    print(f"  {'Scenario':>30} {'BF Std':>10} {'When':>14} {'Risk':>8}")
    print(f"  {'─'*30} {'─'*10} {'─'*14} {'─'*8}")

    # Current BF std
    recent_dates = [d for d in sorted(rolling_std.keys())][-90:]
    if recent_dates:
        recent_std = sum(rolling_std[d] for d in recent_dates) / len(recent_dates)
    else:
        recent_std = 0

    scenarios = [
        ("Current", recent_std, "Now", "—"),
        ("Mild compression", recent_std * 0.7, "~6-12 months", "LOW"),
        ("Moderate (VRP-like)", recent_std * 0.5, "~1-2 years", "MEDIUM"),
        ("Severe (IV→30%)", recent_std * 0.3, "~2-3 years", "HIGH"),
        ("Extreme (smile flat)", recent_std * 0.1, "~3-5 years", "CRITICAL"),
    ]

    for name, std_val, when, risk in scenarios:
        print(f"  {name:>30} {std_val:>10.5f} {when:>14} {risk:>8}")

    # Does BF MR degrade at RECENT low-std values?
    print(f"\n  Recent BF std: {recent_std:.5f}")

    # Check BF Sharpe in recent periods
    recent_pnl = [pnl[d] for d in sorted(pnl.keys()) if d >= "2025-01-01"]
    if len(recent_pnl) >= 30:
        recent_stats = compute_stats(recent_pnl)
        print(f"  Recent BF Sharpe (2025+): {recent_stats['sharpe']:.3f}")
        print(f"  Recent BF Ann Ret (2025+): {recent_stats['ann_ret']:.3f}%")

    return {
        "corr_std_sharpe": corr if len(common) >= 30 else None,
        "recent_std": recent_std
    }


# ═══════════════════════════════════════════════════════════════
# Analysis 7: Structural Break Detection
# ═══════════════════════════════════════════════════════════════

def structural_break_detection(bf_vals, dates):
    print("\n" + "=" * 70)
    print("  ANALYSIS 7: STRUCTURAL BREAK DETECTION")
    print("=" * 70)

    # CUSUM test: detect structural breaks in BF mean and variance
    bf_list = [(d, bf_vals[d]) for d in dates if d in bf_vals]
    if len(bf_list) < 100:
        print("  Not enough data")
        return {}

    # Mean CUSUM
    vals = [v[1] for v in bf_list]
    overall_mean = sum(vals) / len(vals)

    cusum = []
    s = 0
    max_cusum = 0
    max_cusum_idx = 0
    for i, v in enumerate(vals):
        s += (v - overall_mean)
        cusum.append(s)
        if abs(s) > abs(max_cusum):
            max_cusum = s
            max_cusum_idx = i

    break_date = bf_list[max_cusum_idx][0]
    print(f"\n  Mean CUSUM break point: {break_date} (index {max_cusum_idx}/{len(vals)})")

    # Compare mean before and after break
    before = vals[:max_cusum_idx]
    after = vals[max_cusum_idx:]
    if before and after:
        mean_before = sum(before) / len(before)
        mean_after = sum(after) / len(after)
        std_before = math.sqrt(sum((v - mean_before)**2 for v in before) / len(before))
        std_after = math.sqrt(sum((v - mean_after)**2 for v in after) / len(after))

        # Two-sample t-test
        n1, n2 = len(before), len(after)
        se = math.sqrt(std_before**2/n1 + std_after**2/n2) if n1 > 0 and n2 > 0 else 1
        t_stat = (mean_before - mean_after) / se if se > 0 else 0

        print(f"    Before: mean={mean_before:.5f}, std={std_before:.5f}, n={n1}")
        print(f"    After:  mean={mean_after:.5f}, std={std_after:.5f}, n={n2}")
        print(f"    Mean shift: {mean_after - mean_before:+.5f}")
        print(f"    Std shift:  {std_after - std_before:+.5f}")
        print(f"    t-stat (mean): {t_stat:.2f}")

    # Variance break: sliding window comparison
    print(f"\n  Variance break detection (rolling 120d window):")
    window_size = 120
    var_changes = []
    for i in range(window_size, len(vals) - window_size, 30):
        d = bf_list[i][0]
        w1 = vals[i-window_size:i]
        w2 = vals[i:i+window_size]
        std1 = math.sqrt(sum((v - sum(w1)/len(w1))**2 for v in w1) / len(w1))
        std2 = math.sqrt(sum((v - sum(w2)/len(w2))**2 for v in w2) / len(w2))
        ratio = std2 / std1 if std1 > 0 else 1
        var_changes.append((d, std1, std2, ratio))

    print(f"  {'Date':>12} {'Std (before)':>14} {'Std (after)':>14} {'Ratio':>8}")
    print(f"  {'─'*12} {'─'*14} {'─'*14} {'─'*8}")
    for d, s1, s2, r in var_changes:
        flag = " ← BREAK" if r < 0.5 or r > 2.0 else ""
        print(f"  {d:>12} {s1:>14.5f} {s2:>14.5f} {r:>8.3f}{flag}")

    return {"mean_break_date": break_date, "variance_changes": [(d, r) for d, _, _, r in var_changes]}


# ═══════════════════════════════════════════════════════════════
# Analysis 8: Cross-Market Comparison (ETH BF as benchmark)
# ═══════════════════════════════════════════════════════════════

def cross_market_comparison(dates):
    print("\n" + "=" * 70)
    print("  ANALYSIS 8: CROSS-MARKET COMPARISON (BTC vs ETH BF COMPRESSION)")
    print("=" * 70)

    # Load ETH surface for comparison
    btc_surface = load_surface("BTC")
    eth_surface = load_surface("ETH")

    # Compare BF std decline across both assets
    for label, surface in [("BTC", btc_surface), ("ETH", eth_surface)]:
        bf_by_year = defaultdict(list)
        for d in sorted(surface.keys()):
            if "butterfly_25d" in surface[d]:
                bf_by_year[d[:4]].append(surface[d]["butterfly_25d"])

        if not bf_by_year:
            print(f"  {label}: No data")
            continue

        print(f"\n  {label} butterfly_25d by year:")
        print(f"    {'Year':>6} {'Mean':>10} {'Std':>10} {'N':>6}")
        print(f"    {'─'*6} {'─'*10} {'─'*10} {'─'*6}")
        for yr in sorted(bf_by_year.keys()):
            vals = bf_by_year[yr]
            mean = sum(vals) / len(vals)
            std = math.sqrt(sum((v - mean)**2 for v in vals) / len(vals))
            print(f"    {yr:>6} {mean:>10.5f} {std:>10.5f} {len(vals):>6}")

    # If BOTH BTC and ETH show compression → market-wide structural change
    # If only BTC → BTC-specific
    print(f"\n  Interpretation:")
    print(f"    If BOTH BTC+ETH show BF compression → market-wide (structural)")
    print(f"    If only BTC → BTC-specific (could be maturation)")
    print(f"    If neither ETH → BTC compression is noise")


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════

def main():
    print("=" * 70)
    print("  R78: BF STRUCTURAL COMPRESSION ANALYSIS")
    print("=" * 70)
    print(f"\n  Question: Is the declining BF feature std a genuine risk")
    print(f"  to the production strategy (Sharpe 3.76)?")
    print(f"\n  R77 flagged: BF std declining from 0.022→0.007 (2021→2026)")
    print(f"  VRP degraded from Sharpe 3.66→1.59 when IV declined -26%")
    print(f"  If BF follows the same pattern, our portfolio has no fallback.")

    # Load data
    print(f"\n  Loading data...")
    dvol_hist = load_dvol_history("BTC")
    price_hist = load_price_history("BTC")
    surface_hist = load_surface("BTC")

    common_dates = sorted(set(dvol_hist.keys()) & set(surface_hist.keys()))
    print(f"  Dates: {len(common_dates)} common ({common_dates[0]} to {common_dates[-1]})")

    # Compute BF PnL
    pnl, z_scores, bf_vals = compute_bf_pnl(dvol_hist, surface_hist, common_dates)
    print(f"  BF PnL days: {len(pnl)}")

    # Full sample stats
    rets = [pnl[d] for d in sorted(pnl.keys())]
    full_stats = compute_stats(rets)
    print(f"  Full sample: Sharpe {full_stats['sharpe']}, Ann Ret {full_stats['ann_ret']}%, MaxDD {full_stats['max_dd']}%")

    # Run analyses
    a1 = bf_std_significance(bf_vals, common_dates)
    a2 = compression_drivers(surface_hist, dvol_hist, price_hist, common_dates)
    a3 = sharpe_vs_variance(pnl, bf_vals, common_dates)
    a4 = smile_decomposition(surface_hist, common_dates)
    bf_normalized_backtest(dvol_hist, surface_hist, common_dates)
    a6 = compression_forecast(bf_vals, pnl, common_dates)
    a7 = structural_break_detection(bf_vals, common_dates)
    cross_market_comparison(common_dates)

    # ─── Summary ────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  R78 SUMMARY: BF STRUCTURAL COMPRESSION ANALYSIS")
    print("=" * 70)

    is_significant = a1.get("significant", False)
    corr_bf_iv = a2.get("corr_bf_std_iv")
    corr_std_sharpe = a6.get("corr_std_sharpe")
    recent_std = a6.get("recent_std", 0)

    # Determine verdict
    if is_significant and corr_std_sharpe and corr_std_sharpe > 0.3:
        verdict = "WARNING — BF compression is significant AND linked to Sharpe"
        risk_level = "HIGH"
    elif is_significant and (corr_std_sharpe is None or corr_std_sharpe <= 0.3):
        verdict = "CAUTION — BF compression is real but NOT yet affecting Sharpe"
        risk_level = "MODERATE"
    elif not is_significant:
        verdict = "BENIGN — BF compression is NOT statistically significant"
        risk_level = "LOW"
    else:
        verdict = "INCONCLUSIVE — need more data"
        risk_level = "UNKNOWN"

    print(f"\n  VERDICT: {verdict}")
    print(f"  Risk Level: {risk_level}")
    print(f"\n  Key findings:")
    print(f"    1. BF std decline statistically significant: {is_significant}")
    if corr_bf_iv is not None:
        print(f"    2. BF compression driven by IV decline: corr={corr_bf_iv:.3f}")
    if corr_std_sharpe is not None:
        print(f"    3. BF Sharpe linked to BF std: corr={corr_std_sharpe:.3f}")
    print(f"    4. Recent BF std: {recent_std:.5f}")

    print(f"\n  Implications:")
    print(f"    - If compression is IV-driven, using BF/IV (normalized) may help")
    print(f"    - If compression is structural, BF edge may decay like VRP")
    print(f"    - Production config should be monitored via R77 health indicator")
    print(f"    - Consider sensitivity tier (R70): higher sensitivity compensates for less variance")

    # Save results
    results = {
        "research_id": "R78",
        "title": "BF Structural Compression Analysis",
        "full_sample_sharpe": full_stats["sharpe"],
        "std_decline_significant": is_significant,
        "std_decline_t_stat": a1.get("t_stat"),
        "corr_bf_std_iv": corr_bf_iv,
        "corr_std_sharpe": corr_std_sharpe,
        "recent_bf_std": recent_std,
        "risk_level": risk_level,
        "verdict": verdict
    }

    out_path = ROOT / "data" / "cache" / "deribit" / "real_surface" / "r78_bf_compression_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved to {out_path}")


if __name__ == "__main__":
    main()
