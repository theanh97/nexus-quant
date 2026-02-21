#!/usr/bin/env python3
"""
R53: BTC-Only Ensemble Optimization on REAL Data
==================================================

Given R51 results (BTC VRP=2.905, Skew=1.551, Ensemble 40/60=3.094)
and R52 findings (ETH broken, IV declining), optimize:

1. VRP/Skew weight allocation (sweep 10/90 to 90/10)
2. VRP leverage (sweep 1.0 to 3.0)
3. Skew parameters on real data (lookback, z-thresholds)
4. BTC-only vs multi-asset decision

Uses exact PnL models from R51.
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


def load_dvol(currency):
    path = ROOT / "data" / "cache" / "deribit" / "dvol" / f"{currency}_DVOL_12h.csv"
    d = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            d[row["date"][:10]] = float(row["dvol_close"]) / 100.0
    return d


def load_skew(currency):
    path = ROOT / "data" / "cache" / "deribit" / "real_surface" / f"{currency}_daily_surface.csv"
    d = {}
    if not path.exists(): return d
    with open(path) as f:
        for row in csv.DictReader(f):
            val = row.get("skew_25d", "")
            if val and val != "None":
                d[row["date"]] = float(val)
    return d


def load_prices(currency):
    prices = {}
    start = datetime(2020, 1, 1, tzinfo=timezone.utc)
    end = datetime(2026, 3, 1, tzinfo=timezone.utc)
    current = start
    while current < end:
        chunk_end = min(current + timedelta(days=365), end)
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


def vrp_returns(prices, dvol, leverage=1.5):
    dates = sorted(set(prices.keys()) & set(dvol.keys()))
    dt = 1.0 / 365.0
    r_dates, r_rets = [], []
    for i in range(1, len(dates)):
        d, dp = dates[i], dates[i-1]
        iv, p0, p1 = dvol.get(dp), prices.get(dp), prices.get(d)
        if not all([iv, p0, p1]) or p0 <= 0: continue
        lr = math.log(p1 / p0)
        rv_bar = abs(lr) * math.sqrt(365)
        pnl = leverage * 0.5 * (iv**2 - rv_bar**2) * dt
        r_dates.append(d)
        r_rets.append(pnl)
    return r_dates, r_rets


def skew_returns(skew_data, dvol, leverage=1.0, sens=2.5, lookback=60,
                  z_short=1.0, z_long=2.0, z_exit=0.0):
    dates = sorted(skew_data.keys())
    dt = 1.0 / 365.0
    r_dates, r_rets = [], []
    pos = 0.0
    hist = []

    for i in range(1, len(dates)):
        d, dp = dates[i], dates[i-1]
        s_now = skew_data.get(d)
        s_prev = skew_data.get(dp)
        iv = dvol.get(dp) if dvol and dp in dvol else None
        if iv is None:
            # Use a fallback from the skew data (iv_atm column not available here)
            iv = 0.50  # rough default

        if s_now is None or s_prev is None or iv <= 0:
            if s_now is not None:
                hist.append(s_now)
                if len(hist) > lookback: hist.pop(0)
            r_dates.append(d)
            r_rets.append(0.0)
            continue

        hist.append(s_now)
        if len(hist) > lookback: hist.pop(0)

        d_skew = s_now - s_prev
        sensitivity = iv * math.sqrt(dt) * sens
        ret = pos * leverage * d_skew * sensitivity
        r_dates.append(d)
        r_rets.append(ret)

        if len(hist) >= lookback:
            m = sum(hist) / len(hist)
            v = sum((s - m)**2 for s in hist) / len(hist)
            std = math.sqrt(v) if v > 0 else 1e-6
            z = (s_now - m) / std
            if pos == 0:
                if z > z_short: pos = -1.0
                elif z < -z_long: pos = 1.0
            elif pos < 0:
                if z < z_exit: pos = 0.0
            elif pos > 0:
                if z > -z_exit: pos = 0.0

    return r_dates, r_rets


def sharpe(returns, ann_factor=365):
    if not returns or len(returns) < 30: return -999
    m = sum(returns) / len(returns)
    v = sum((r-m)**2 for r in returns) / len(returns)
    s = math.sqrt(v) if v > 0 else 1e-10
    return (m * ann_factor) / (s * math.sqrt(ann_factor))


def max_dd(returns):
    eq = 1.0
    peak = 1.0
    mdd = 0.0
    for r in returns:
        eq *= (1 + r)
        if eq > peak: peak = eq
        dd = (peak - eq) / peak
        if dd > mdd: mdd = dd
    return mdd


def combine(vrp_d, vrp_r, skew_d, skew_r, w_vrp, w_skew):
    vm = dict(zip(vrp_d, vrp_r))
    sm = dict(zip(skew_d, skew_r))
    common = sorted(set(vrp_d) & set(skew_d))
    return common, [w_vrp * vm[d] + w_skew * sm[d] for d in common]


def main():
    print("=" * 70)
    print("R53: BTC-ONLY ENSEMBLE OPTIMIZATION ON REAL DATA")
    print("=" * 70)

    dvol = load_dvol("BTC")
    skew_data = load_skew("BTC")
    prices = load_prices("BTC")
    print(f"  Data: DVOL={len(dvol)}, Skew={len(skew_data)}, Prices={len(prices)}")

    # Base VRP returns
    vrp_d, vrp_r = vrp_returns(prices, dvol, leverage=1.5)
    print(f"  VRP base: Sharpe={sharpe(vrp_r):.3f}, N={len(vrp_r)}")

    # ── 1. Weight Allocation Sweep ────────────────────────────────
    print(f"\n{'='*70}")
    print("1. WEIGHT ALLOCATION SWEEP (VRP leverage=1.5, Skew leverage=1.0)")
    print(f"{'='*70}")

    # Base skew returns
    skew_d, skew_r = skew_returns(skew_data, dvol, leverage=1.0, lookback=60,
                                    z_short=1.0, z_long=2.0, z_exit=0.0)

    print(f"\n  {'W_VRP':>6s}  {'W_Skew':>6s}  {'Sharpe':>7s}  {'AnnRet':>8s}  {'MaxDD':>7s}")
    best_weight = None
    best_sharpe = -999
    for w_vrp_pct in range(10, 100, 10):
        w_vrp = w_vrp_pct / 100.0
        w_skew = 1.0 - w_vrp
        ed, er = combine(vrp_d, vrp_r, skew_d, skew_r, w_vrp, w_skew)
        s = sharpe(er)
        ann = sum(er) / len(er) * 365 if er else 0
        mdd = max_dd(er)
        print(f"  {w_vrp:6.0%}  {w_skew:6.0%}  {s:7.3f}  {ann:8.2%}  {mdd:7.2%}")
        if s > best_sharpe:
            best_sharpe = s
            best_weight = w_vrp

    print(f"\n  Best weight allocation: VRP={best_weight:.0%}, Skew={1-best_weight:.0%} (Sharpe={best_sharpe:.3f})")

    # ── 2. VRP Leverage Sweep ─────────────────────────────────────
    print(f"\n{'='*70}")
    print("2. VRP LEVERAGE SWEEP (with best weight allocation)")
    print(f"{'='*70}")

    print(f"\n  {'Leverage':>8s}  {'Sharpe':>7s}  {'AnnRet':>8s}  {'MaxDD':>7s}")
    best_lev = None
    best_lev_sharpe = -999
    for lev_x10 in range(10, 35, 5):
        lev = lev_x10 / 10.0
        vd, vr = vrp_returns(prices, dvol, leverage=lev)
        ed, er = combine(vd, vr, skew_d, skew_r, best_weight, 1-best_weight)
        s = sharpe(er)
        ann = sum(er) / len(er) * 365 if er else 0
        mdd = max_dd(er)
        print(f"  {lev:8.1f}  {s:7.3f}  {ann:8.2%}  {mdd:7.2%}")
        if s > best_lev_sharpe:
            best_lev_sharpe = s
            best_lev = lev

    print(f"\n  Best VRP leverage: {best_lev:.1f}x (Sharpe={best_lev_sharpe:.3f})")

    # ── 3. Skew Parameter Sweep ───────────────────────────────────
    print(f"\n{'='*70}")
    print("3. SKEW PARAMETER SWEEP (lookback × z_entry_short)")
    print(f"{'='*70}")

    vd, vr = vrp_returns(prices, dvol, leverage=best_lev)

    print(f"\n  {'LB':>4s}  {'z_short':>7s}  {'z_long':>6s}  {'Sharpe':>7s}  {'AnnRet':>8s}  {'MaxDD':>7s}")
    best_skew_params = None
    best_skew_sharpe = -999
    for lb in [30, 45, 60, 90, 120]:
        for z_short_x10 in [5, 7, 10, 13, 15, 20]:
            z_short = z_short_x10 / 10.0
            z_long = min(z_short + 1.0, 3.0)
            sd, sr = skew_returns(skew_data, dvol, leverage=1.0, lookback=lb,
                                   z_short=z_short, z_long=z_long, z_exit=0.0)
            ed, er = combine(vd, vr, sd, sr, best_weight, 1-best_weight)
            s = sharpe(er)
            ann = sum(er) / len(er) * 365 if er else 0
            mdd = max_dd(er)
            if s > best_skew_sharpe:
                best_skew_sharpe = s
                best_skew_params = (lb, z_short, z_long)
                print(f"  {lb:4d}  {z_short:7.1f}  {z_long:6.1f}  {s:7.3f}  {ann:8.2%}  {mdd:7.2%}  ★")
            elif s > best_skew_sharpe - 0.3:
                print(f"  {lb:4d}  {z_short:7.1f}  {z_long:6.1f}  {s:7.3f}  {ann:8.2%}  {mdd:7.2%}")

    if best_skew_params:
        print(f"\n  Best skew params: lookback={best_skew_params[0]}, "
              f"z_short={best_skew_params[1]:.1f}, z_long={best_skew_params[2]:.1f} "
              f"(Sharpe={best_skew_sharpe:.3f})")

    # ── 4. Final Optimized vs Baseline ────────────────────────────
    print(f"\n{'='*70}")
    print("4. FINAL: OPTIMIZED vs BASELINE")
    print(f"{'='*70}")

    # Baseline (R51 config)
    vd_base, vr_base = vrp_returns(prices, dvol, leverage=1.5)
    sd_base, sr_base = skew_returns(skew_data, dvol, leverage=1.0, lookback=60,
                                      z_short=1.0, z_long=2.0, z_exit=0.0)
    ed_base, er_base = combine(vd_base, vr_base, sd_base, sr_base, 0.4, 0.6)

    # Optimized
    if best_skew_params:
        lb, zs, zl = best_skew_params
    else:
        lb, zs, zl = 60, 1.0, 2.0
    vd_opt, vr_opt = vrp_returns(prices, dvol, leverage=best_lev or 1.5)
    sd_opt, sr_opt = skew_returns(skew_data, dvol, leverage=1.0, lookback=lb,
                                    z_short=zs, z_long=zl, z_exit=0.0)
    ed_opt, er_opt = combine(vd_opt, vr_opt, sd_opt, sr_opt,
                              best_weight or 0.4, 1-(best_weight or 0.4))

    # Yearly comparison
    def yearly_sharpe(dates, returns):
        by_yr = defaultdict(list)
        for i, d in enumerate(dates):
            by_yr[d[:4]].append(returns[i])
        result = {}
        for yr in sorted(by_yr):
            result[yr] = sharpe(by_yr[yr])
        return result

    base_yr = yearly_sharpe(ed_base, er_base)
    opt_yr = yearly_sharpe(ed_opt, er_opt)

    print(f"\n  {'Config':<25s}  {'Sharpe':>7s}  {'AnnRet':>8s}  {'MaxDD':>7s}")
    print(f"  {'-'*25}  {'-'*7}  {'-'*8}  {'-'*7}")
    s_base = sharpe(er_base)
    s_opt = sharpe(er_opt)
    ann_base = sum(er_base)/len(er_base)*365
    ann_opt = sum(er_opt)/len(er_opt)*365
    print(f"  {'Baseline 40/60 1.5x':<25s}  {s_base:7.3f}  {ann_base:8.2%}  {max_dd(er_base):7.2%}")
    print(f"  {'Optimized':<25s}  {s_opt:7.3f}  {ann_opt:8.2%}  {max_dd(er_opt):7.2%}")
    print(f"  {'Delta':<25s}  {s_opt-s_base:+7.3f}  {ann_opt-ann_base:+8.2%}")

    print(f"\n  Yearly comparison:")
    print(f"  {'Year':<6s}  {'Baseline':>8s}  {'Optimized':>9s}  {'Delta':>7s}")
    for yr in sorted(set(list(base_yr.keys()) + list(opt_yr.keys()))):
        b = base_yr.get(yr, 0)
        o = opt_yr.get(yr, 0)
        print(f"  {yr:<6s}  {b:8.3f}  {o:9.3f}  {o-b:+7.3f}")

    # ── 5. VRP-Only (no skew) for comparison ─────────────────────
    print(f"\n  --- VRP-ONLY at {best_lev or 1.5:.1f}x ---")
    vd_solo, vr_solo = vrp_returns(prices, dvol, leverage=best_lev or 1.5)
    s_solo = sharpe(vr_solo)
    ann_solo = sum(vr_solo)/len(vr_solo)*365
    print(f"  Sharpe: {s_solo:.3f}, AnnRet: {ann_solo:.2%}, MaxDD: {max_dd(vr_solo):.2%}")

    # Save
    out = ROOT / "data" / "cache" / "deribit" / "real_surface" / "r53_optimization_results.json"
    with open(out, "w") as f:
        json.dump({
            "research_id": "R53",
            "best_weight_vrp": best_weight,
            "best_vrp_leverage": best_lev,
            "best_skew_params": {"lookback": lb, "z_short": zs, "z_long": zl},
            "baseline_sharpe": round(s_base, 3),
            "optimized_sharpe": round(s_opt, 3),
            "delta": round(s_opt - s_base, 3),
        }, f, indent=2)
    print(f"\n  Results → {out}")


if __name__ == "__main__":
    main()
