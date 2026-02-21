#!/usr/bin/env python3
"""
R68: Butterfly-Focused Strategy Optimization
==============================================

BF (Butterfly MR) is the ONLY surviving edge in the current regime:
  - VRP Sharpe: 4.15 (2024H1) → -1.62 (2026H1) — collapsed
  - BF Sharpe:  2.32 (2024H1) →  5.36 (2026H1) — STABLE

This study:
1. Optimizes BF lookback and z-score parameters
2. Tests ALL surface features as MR signals (butterfly, skew, term_spread, etc.)
3. Multi-feature BF ensemble
4. BF standalone period stability
5. BF with reduced VRP (20/80, 10/90, 0/100)
6. New composite features (butterfly ratio, skew-butterfly spread, etc.)
7. Walk-forward validation of BF standalone

Available real surface features:
  - butterfly_25d, skew_25d, term_spread, iv_atm, iv_25d_put, iv_25d_call, spot
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
            entry = {}
            for field in ["butterfly_25d", "skew_25d", "term_spread", "iv_atm",
                          "iv_25d_put", "iv_25d_call", "spot"]:
                val = row.get(field, "")
                if val and val != "None":
                    try:
                        entry[field] = float(val)
                    except ValueError:
                        pass
            if entry:
                data[d] = entry
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


def feature_mr_pnl(dates, feature, dvol, lookback=120, z_entry=1.5, z_exit=0.3,
                    sensitivity=2.5):
    """General mean-reversion PnL on any surface feature."""
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


def calc_stats(rets):
    if len(rets) < 10:
        return {"sharpe": 0.0, "ann_ret": 0.0, "max_dd": 0.0, "n": len(rets)}
    mean = sum(rets) / len(rets)
    var = sum((r - mean)**2 for r in rets) / len(rets)
    std = math.sqrt(var) if var > 0 else 1e-10
    sharpe = (mean * 365) / (std * math.sqrt(365))
    ann_ret = mean * 365
    cum = peak = max_dd = 0.0
    for r in rets:
        cum += r; peak = max(peak, cum); max_dd = max(max_dd, peak - cum)
    win_rate = sum(1 for r in rets if r > 0) / len(rets)
    return {"sharpe": sharpe, "ann_ret": ann_ret, "max_dd": max_dd,
            "win_rate": win_rate, "n": len(rets), "total_return": sum(rets)}


def main():
    print("=" * 70)
    print("R68: BUTTERFLY-FOCUSED STRATEGY OPTIMIZATION")
    print("=" * 70)

    # Load data
    print("\n  Loading data...")
    dvol = load_dvol_daily("BTC")
    prices = load_prices("BTC")
    surface = load_surface("BTC")
    all_dates = sorted(set(dvol.keys()) & set(prices.keys()) & set(surface.keys()))
    print(f"    {len(all_dates)} days, {all_dates[0]} to {all_dates[-1]}")

    # Extract features
    features = {}
    for feat_name in ["butterfly_25d", "skew_25d", "term_spread", "iv_atm",
                      "iv_25d_put", "iv_25d_call"]:
        features[feat_name] = {d: surface[d][feat_name] for d in all_dates
                               if feat_name in surface[d]}
        n = len(features[feat_name])
        vals = list(features[feat_name].values())
        if vals:
            print(f"    {feat_name}: {n} days, mean={sum(vals)/len(vals)*100:.2f}%, "
                  f"std={math.sqrt(sum((v-sum(vals)/len(vals))**2 for v in vals)/len(vals))*100:.2f}%")

    # Create composite features
    # 1. Butterfly ratio = butterfly / iv_atm (normalized convexity)
    features["bf_ratio"] = {}
    for d in all_dates:
        if "butterfly_25d" in surface[d] and "iv_atm" in surface[d] and surface[d]["iv_atm"] > 0.01:
            features["bf_ratio"][d] = surface[d]["butterfly_25d"] / surface[d]["iv_atm"]

    # 2. Skew-butterfly spread
    features["skew_bf_spread"] = {}
    for d in all_dates:
        if "skew_25d" in surface[d] and "butterfly_25d" in surface[d]:
            features["skew_bf_spread"][d] = surface[d]["skew_25d"] - surface[d]["butterfly_25d"]

    # 3. Put-call IV spread (different from skew — absolute diff)
    features["put_call_spread"] = {}
    for d in all_dates:
        if "iv_25d_put" in surface[d] and "iv_25d_call" in surface[d]:
            features["put_call_spread"][d] = surface[d]["iv_25d_put"] - surface[d]["iv_25d_call"]

    # 4. IV ATM change (momentum/mean-reversion)
    features["iv_atm_5d_change"] = {}
    for i in range(5, len(all_dates)):
        d, dp = all_dates[i], all_dates[i-5]
        if "iv_atm" in surface[d] and "iv_atm" in surface[dp]:
            features["iv_atm_5d_change"][d] = surface[d]["iv_atm"] - surface[dp]["iv_atm"]

    print(f"    Composite features created: bf_ratio, skew_bf_spread, put_call_spread, iv_atm_5d_change")

    # ═══════════════════════════════════════════════════════════════
    # 1. BF PARAMETER SWEEP (lookback x z_entry x z_exit)
    # ═══════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("  1. BF PARAMETER SWEEP (butterfly_25d)")
    print("=" * 70)

    bf_feat = features["butterfly_25d"]
    print(f"\n  {'LB':<6} {'Z_in':<6} {'Z_out':<6} {'Sharpe':>8} {'AnnRet':>10} {'MaxDD':>8} {'WinRate':>8}")
    best_bf = {"sharpe": 0}

    for lb in [60, 90, 120, 180, 240]:
        for z_in in [1.0, 1.25, 1.5, 2.0]:
            for z_out in [0.0, 0.3, 0.5]:
                if z_out >= z_in:
                    continue
                pnl = feature_mr_pnl(all_dates, bf_feat, dvol, lb, z_in, z_out)
                rets = [pnl[d] for d in sorted(pnl)]
                s = calc_stats(rets)
                print(f"  {lb:<6} {z_in:<6.2f} {z_out:<6.1f} {s['sharpe']:8.4f} "
                      f"{s['ann_ret']*100:9.2f}% {s['max_dd']*100:7.2f}% {s.get('win_rate', 0)*100:7.1f}%")
                if s["sharpe"] > best_bf.get("sharpe", 0):
                    best_bf = {**s, "lb": lb, "z_in": z_in, "z_out": z_out}

    print(f"\n  Best BF: lb={best_bf.get('lb')} z_in={best_bf.get('z_in')} "
          f"z_out={best_bf.get('z_out')} → Sharpe {best_bf['sharpe']:.4f}")

    # ═══════════════════════════════════════════════════════════════
    # 2. ALL FEATURES AS MR SIGNALS
    # ═══════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("  2. ALL SURFACE FEATURES AS MR SIGNALS (lb=120, z=1.5)")
    print("=" * 70)

    feature_results = {}
    print(f"\n  {'Feature':<25} {'Sharpe':>8} {'AnnRet':>10} {'MaxDD':>8} {'WinRate':>8} {'N':>6}")

    for feat_name in ["butterfly_25d", "skew_25d", "term_spread", "iv_atm",
                      "iv_25d_put", "iv_25d_call",
                      "bf_ratio", "skew_bf_spread", "put_call_spread", "iv_atm_5d_change"]:
        feat_data = features.get(feat_name, {})
        if len(feat_data) < 200:
            print(f"  {feat_name:<25} — insufficient data ({len(feat_data)} days)")
            continue
        pnl = feature_mr_pnl(all_dates, feat_data, dvol, 120, 1.5, 0.3)
        rets = [pnl[d] for d in sorted(pnl)]
        s = calc_stats(rets)
        feature_results[feat_name] = s
        print(f"  {feat_name:<25} {s['sharpe']:8.4f} {s['ann_ret']*100:9.2f}% "
              f"{s['max_dd']*100:7.2f}% {s.get('win_rate', 0)*100:7.1f}% {s['n']:6}")

    # ═══════════════════════════════════════════════════════════════
    # 3. OPTIMIZE BEST NEW FEATURES
    # ═══════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("  3. OPTIMIZE TOP FEATURES")
    print("=" * 70)

    # Find top features (Sharpe > 0.5)
    top_features = sorted(feature_results.items(), key=lambda x: -x[1]["sharpe"])[:5]

    for feat_name, base_stats in top_features:
        print(f"\n  --- {feat_name} (base Sharpe: {base_stats['sharpe']:.4f}) ---")
        feat_data = features[feat_name]
        best_this = {"sharpe": 0}

        for lb in [60, 90, 120, 180]:
            for z_in in [1.0, 1.5, 2.0]:
                pnl = feature_mr_pnl(all_dates, feat_data, dvol, lb, z_in, 0.3)
                rets = [pnl[d] for d in sorted(pnl)]
                s = calc_stats(rets)
                if s["sharpe"] > best_this.get("sharpe", 0):
                    best_this = {**s, "lb": lb, "z_in": z_in}

        print(f"    Best: lb={best_this.get('lb')} z={best_this.get('z_in')} → "
              f"Sharpe {best_this['sharpe']:.4f}")

    # ═══════════════════════════════════════════════════════════════
    # 4. MULTI-FEATURE BF ENSEMBLE
    # ═══════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("  4. MULTI-FEATURE BF ENSEMBLE")
    print("=" * 70)
    print("  Combine top features with equal weight.\n")

    # Get PnL for top features at their default params
    feature_pnls = {}
    for feat_name in ["butterfly_25d", "skew_25d", "term_spread", "bf_ratio",
                      "skew_bf_spread", "put_call_spread"]:
        feat_data = features.get(feat_name, {})
        if len(feat_data) < 200:
            continue
        pnl = feature_mr_pnl(all_dates, feat_data, dvol, 120, 1.5, 0.3)
        if len(pnl) > 100:
            feature_pnls[feat_name] = pnl

    # All pairs and triples
    feat_names = list(feature_pnls.keys())
    print(f"  Features: {', '.join(feat_names)}")

    print(f"\n  {'Ensemble':<40} {'Sharpe':>8} {'AnnRet':>10} {'MaxDD':>8}")

    # Singles
    for fn in feat_names:
        pnl = feature_pnls[fn]
        rets = [pnl[d] for d in sorted(pnl)]
        s = calc_stats(rets)
        print(f"  {fn:<40} {s['sharpe']:8.4f} {s['ann_ret']*100:9.2f}% {s['max_dd']*100:7.2f}%")

    # Pairs
    best_pair = {"sharpe": 0}
    for i in range(len(feat_names)):
        for j in range(i+1, len(feat_names)):
            fn1, fn2 = feat_names[i], feat_names[j]
            p1, p2 = feature_pnls[fn1], feature_pnls[fn2]
            common = sorted(set(p1.keys()) & set(p2.keys()))
            if len(common) < 200:
                continue
            rets = [0.5*p1[d] + 0.5*p2[d] for d in common]
            s = calc_stats(rets)
            label = f"{fn1[:12]}+{fn2[:12]}"
            print(f"  {label:<40} {s['sharpe']:8.4f} {s['ann_ret']*100:9.2f}% {s['max_dd']*100:7.2f}%")
            if s["sharpe"] > best_pair.get("sharpe", 0):
                best_pair = {**s, "f1": fn1, "f2": fn2}

    if best_pair.get("f1"):
        print(f"\n  Best pair: {best_pair.get('f1')} + {best_pair.get('f2')} → "
              f"Sharpe {best_pair['sharpe']:.4f}")

    # ═══════════════════════════════════════════════════════════════
    # 5. BF STANDALONE PERIOD STABILITY
    # ═══════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("  5. BF STANDALONE — YEARLY STABILITY")
    print("=" * 70)

    # Use best BF params
    best_lb = best_bf.get("lb", 120)
    best_z = best_bf.get("z_in", 1.5)
    best_zo = best_bf.get("z_out", 0.3)
    pnl_bf = feature_mr_pnl(all_dates, bf_feat, dvol, best_lb, best_z, best_zo)

    years = defaultdict(list)
    for d in sorted(pnl_bf):
        years[d[:4]].append(pnl_bf[d])

    print(f"\n  BF params: lb={best_lb}, z_in={best_z}, z_out={best_zo}")
    print(f"\n  {'Year':<6} {'Sharpe':>8} {'AnnRet':>10} {'MaxDD':>8} {'WinRate':>8} {'N':>5}")
    for year in sorted(years):
        s = calc_stats(years[year])
        print(f"  {year:<6} {s['sharpe']:8.4f} {s['ann_ret']*100:9.2f}% "
              f"{s['max_dd']*100:7.2f}% {s.get('win_rate',0)*100:7.1f}% {len(years[year]):5}")

    # Half-yearly
    print(f"\n  {'Period':<8} {'Sharpe':>8} {'AnnRet':>10} {'MaxDD':>8}")
    halves = defaultdict(list)
    for d in sorted(pnl_bf):
        year = d[:4]
        half = "H1" if d[5:7] <= "06" else "H2"
        halves[f"{year}{half}"].append(pnl_bf[d])

    for period in sorted(halves):
        s = calc_stats(halves[period])
        print(f"  {period:<8} {s['sharpe']:8.4f} {s['ann_ret']*100:9.2f}% {s['max_dd']*100:7.2f}%")

    # ═══════════════════════════════════════════════════════════════
    # 6. VRP+BF WEIGHT OPTIMIZATION FOR CURRENT REGIME
    # ═══════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("  6. VRP+BF WEIGHT GRID (FULL SAMPLE vs RECENT)")
    print("=" * 70)

    pnl_vrp = vrp_pnl(all_dates, dvol, prices, 2.0)
    common = sorted(set(pnl_vrp.keys()) & set(pnl_bf.keys()))

    print(f"\n  Full Sample ({common[0]} to {common[-1]}):")
    print(f"  {'VRP/BF':<10} {'Sharpe':>8} {'AnnRet':>10} {'MaxDD':>8}")
    for vrp_w in [1.0, 0.8, 0.7, 0.5, 0.3, 0.2, 0.1, 0.0]:
        bf_w = 1.0 - vrp_w
        rets = [vrp_w * pnl_vrp[d] + bf_w * pnl_bf[d] for d in common]
        s = calc_stats(rets)
        print(f"  {int(vrp_w*100)}/{int(bf_w*100):<10} {s['sharpe']:8.4f} "
              f"{s['ann_ret']*100:9.2f}% {s['max_dd']*100:7.2f}%")

    # Recent period (2025-2026)
    recent = [d for d in common if d >= "2025-01-01"]
    if recent:
        print(f"\n  Recent ({recent[0]} to {recent[-1]}):")
        print(f"  {'VRP/BF':<10} {'Sharpe':>8} {'AnnRet':>10} {'MaxDD':>8}")
        for vrp_w in [1.0, 0.5, 0.3, 0.2, 0.1, 0.0]:
            bf_w = 1.0 - vrp_w
            rets = [vrp_w * pnl_vrp[d] + bf_w * pnl_bf[d] for d in recent]
            s = calc_stats(rets)
            print(f"  {int(vrp_w*100)}/{int(bf_w*100):<10} {s['sharpe']:8.4f} "
                  f"{s['ann_ret']*100:9.2f}% {s['max_dd']*100:7.2f}%")

    # ═══════════════════════════════════════════════════════════════
    # 7. WALK-FORWARD BF STANDALONE
    # ═══════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("  7. WALK-FORWARD VALIDATION — BF STANDALONE")
    print("=" * 70)
    print("  Expanding window: train on [start, T], test on [T, T+6mo]\n")

    bf_dates = sorted(pnl_bf.keys())
    wf_rets = []
    wf_periods = []

    # Start walk-forward after 1 year of data
    min_train = 365
    step = 180  # 6-month test periods

    for test_start_idx in range(min_train, len(bf_dates), step):
        test_end_idx = min(test_start_idx + step, len(bf_dates))
        train_dates = bf_dates[:test_start_idx]
        test_dates = bf_dates[test_start_idx:test_end_idx]

        if len(test_dates) < 30:
            break

        # Optimize on train set
        best_train_sh = -999
        best_params = (120, 1.5, 0.3)
        for lb in [60, 90, 120, 180]:
            for z_in in [1.0, 1.5, 2.0]:
                pnl_train = feature_mr_pnl(train_dates, bf_feat, dvol, lb, z_in, 0.3)
                train_rets = [pnl_train[d] for d in train_dates if d in pnl_train]
                if len(train_rets) < 60:
                    continue
                s_train = calc_stats(train_rets)
                if s_train["sharpe"] > best_train_sh:
                    best_train_sh = s_train["sharpe"]
                    best_params = (lb, z_in, 0.3)

        # Test with best params
        pnl_test = feature_mr_pnl(test_dates, bf_feat, dvol, *best_params)
        test_rets = [pnl_test[d] for d in test_dates if d in pnl_test]
        if not test_rets:
            continue

        s_test = calc_stats(test_rets)
        wf_rets.extend(test_rets)
        period = f"{test_dates[0][:7]} to {test_dates[-1][:7]}"
        wf_periods.append((period, s_test["sharpe"], best_params))
        print(f"  {period}: Sharpe {s_test['sharpe']:6.2f}  params=lb{best_params[0]}/z{best_params[1]}")

    if wf_rets:
        wf_stats = calc_stats(wf_rets)
        print(f"\n  Walk-Forward OOS: Sharpe {wf_stats['sharpe']:.4f}  "
              f"Return {wf_stats['ann_ret']*100:.2f}%  MaxDD {wf_stats['max_dd']*100:.2f}%")
        print(f"  Positive periods: {sum(1 for _,sh,_ in wf_periods if sh > 0)}/{len(wf_periods)}")

    # ═══════════════════════════════════════════════════════════════
    # 8. FEATURE CORRELATION ANALYSIS
    # ═══════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("  8. FEATURE PnL CORRELATION MATRIX")
    print("=" * 70)

    # Compute pairwise correlations of feature MR PnLs
    pnl_dict = {}
    for feat_name in ["butterfly_25d", "skew_25d", "term_spread", "bf_ratio", "put_call_spread"]:
        feat_data = features.get(feat_name, {})
        if len(feat_data) > 200:
            pnl_dict[feat_name] = feature_mr_pnl(all_dates, feat_data, dvol, 120, 1.5, 0.3)

    if len(pnl_dict) >= 2:
        pnl_names = list(pnl_dict.keys())
        print(f"\n  {'':>15}", end="")
        for name in pnl_names:
            print(f"  {name[:8]:>8}", end="")
        print()

        for i, n1 in enumerate(pnl_names):
            print(f"  {n1[:15]:>15}", end="")
            for j, n2 in enumerate(pnl_names):
                p1, p2 = pnl_dict[n1], pnl_dict[n2]
                common_d = sorted(set(p1.keys()) & set(p2.keys()))
                if len(common_d) < 100:
                    print(f"  {'N/A':>8}", end="")
                    continue
                r1 = [p1[d] for d in common_d]
                r2 = [p2[d] for d in common_d]
                m1, m2 = sum(r1)/len(r1), sum(r2)/len(r2)
                cov = sum((a-m1)*(b-m2) for a, b in zip(r1, r2)) / len(r1)
                s1 = math.sqrt(sum((a-m1)**2 for a in r1) / len(r1))
                s2 = math.sqrt(sum((b-m2)**2 for b in r2) / len(r2))
                corr = cov / (s1 * s2) if s1 > 0 and s2 > 0 else 0
                print(f"  {corr:8.3f}", end="")
            print()

    # Also show VRP correlation with each feature
    print(f"\n  VRP correlation with features:")
    for feat_name in pnl_names:
        p_bf_feat = pnl_dict[feat_name]
        common_d = sorted(set(pnl_vrp.keys()) & set(p_bf_feat.keys()))
        if len(common_d) < 100:
            continue
        r1 = [pnl_vrp[d] for d in common_d]
        r2 = [p_bf_feat[d] for d in common_d]
        m1, m2 = sum(r1)/len(r1), sum(r2)/len(r2)
        cov = sum((a-m1)*(b-m2) for a, b in zip(r1, r2)) / len(r1)
        s1 = math.sqrt(sum((a-m1)**2 for a in r1) / len(r1))
        s2 = math.sqrt(sum((b-m2)**2 for b in r2) / len(r2))
        corr = cov / (s1 * s2) if s1 > 0 and s2 > 0 else 0
        print(f"    VRP vs {feat_name}: {corr:.3f}")

    # ═══════════════════════════════════════════════════════════════
    # CONCLUSION
    # ═══════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("R68: CONCLUSION")
    print("=" * 70)

    # R60 baseline
    pnl_r60 = {d: 0.5 * pnl_vrp[d] + 0.5 * pnl_bf[d] for d in common}
    r60_rets = [pnl_r60[d] for d in common]
    r60_stats = calc_stats(r60_rets)

    # Best BF standalone
    bf_standalone_rets = [pnl_bf[d] for d in sorted(pnl_bf)]
    bf_standalone_stats = calc_stats(bf_standalone_rets)

    print(f"\n  {'Strategy':<30} {'Sharpe':>8} {'AnnRet':>10} {'MaxDD':>8}")
    print(f"  {'R60 VRP+BF 50/50':<30} {r60_stats['sharpe']:8.4f} {r60_stats['ann_ret']*100:9.2f}% {r60_stats['max_dd']*100:7.2f}%")
    print(f"  {'BF standalone (optimized)':<30} {best_bf['sharpe']:8.4f} {best_bf.get('ann_ret',0)*100:9.2f}% {best_bf.get('max_dd',0)*100:7.2f}%")
    if best_pair.get("f1"):
        print(f"  {'Best feature pair':<30} {best_pair['sharpe']:8.4f} {best_pair.get('ann_ret',0)*100:9.2f}% {best_pair.get('max_dd',0)*100:7.2f}%")
    if wf_rets:
        print(f"  {'BF walk-forward OOS':<30} {wf_stats['sharpe']:8.4f} {wf_stats['ann_ret']*100:9.2f}% {wf_stats['max_dd']*100:7.2f}%")

    # Verdict
    bf_vs_r60 = best_bf["sharpe"] - r60_stats["sharpe"]
    if best_bf["sharpe"] > 2.0:
        verdict = f"BF standalone viable: Sharpe {best_bf['sharpe']:.2f} — can run without VRP"
    elif best_bf["sharpe"] > 1.5:
        verdict = f"BF standalone acceptable: Sharpe {best_bf['sharpe']:.2f} — usable as primary signal"
    else:
        verdict = f"BF standalone too weak: Sharpe {best_bf['sharpe']:.2f} — still needs VRP diversification"

    print(f"\n  VERDICT: {verdict}")

    # Save results
    out_path = ROOT / "data" / "cache" / "deribit" / "real_surface" / "r68_bf_optimization_results.json"
    results = {
        "research_id": "R68",
        "title": "Butterfly-Focused Strategy Optimization",
        "best_bf_params": {"lb": best_bf.get("lb"), "z_in": best_bf.get("z_in"), "z_out": best_bf.get("z_out")},
        "best_bf_sharpe": round(best_bf["sharpe"], 4),
        "best_pair": {
            "f1": best_pair.get("f1"), "f2": best_pair.get("f2"),
            "sharpe": round(best_pair.get("sharpe", 0), 4),
        } if best_pair.get("f1") else None,
        "r60_sharpe": round(r60_stats["sharpe"], 4),
        "wf_oos_sharpe": round(wf_stats["sharpe"], 4) if wf_rets else None,
        "feature_sharpes": {fn: round(s["sharpe"], 4) for fn, s in feature_results.items()},
        "verdict": verdict,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved to {out_path}")


if __name__ == "__main__":
    main()
