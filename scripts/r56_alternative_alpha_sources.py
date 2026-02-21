#!/usr/bin/env python3
"""
R56: Alternative Alpha Sources from Real IV Surface
=====================================================

R55 found: Skew MR is weak on real data (optimizer pushes w_vrp → 0.7-0.9).
Real skew mean=+1.4% (vs synthetic -5%), making MR signals weaker.

We have 4 surface features from R50:
  1. iv_atm (used in VRP — proven)
  2. skew_25d (used in Skew MR — weak on real data)
  3. butterfly_25d (unused — potential new alpha)
  4. term_spread (unused — potential new alpha)

This study investigates:
1. Do butterfly/term_spread have predictive power?
2. Can we build MR or trend-following strategies on these?
3. What's the correlation structure across features?
4. Can a multi-factor model improve on VRP-only?
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
from typing import Dict, List, Tuple, Optional

ROOT = Path(__file__).resolve().parents[1]


# ── Data Loading ─────────────────────────────────────────────────────

def load_dvol_daily(currency: str) -> Dict[str, float]:
    path = ROOT / "data" / "cache" / "deribit" / "dvol" / f"{currency}_DVOL_12h.csv"
    daily = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            daily[row["date"][:10]] = float(row["dvol_close"]) / 100.0
    return daily


def load_prices(currency: str) -> Dict[str, float]:
    import subprocess, time
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


def load_surface(currency: str) -> Dict[str, dict]:
    """Load full daily surface data (all features)."""
    path = ROOT / "data" / "cache" / "deribit" / "real_surface" / f"{currency}_daily_surface.csv"
    data = {}
    if not path.exists():
        return data
    with open(path) as f:
        for row in csv.DictReader(f):
            d = row["date"]
            entry = {}
            for field in ["iv_atm", "skew_25d", "butterfly_25d", "term_spread"]:
                val = row.get(field, "")
                if val and val != "None":
                    entry[field] = float(val)
            if entry:
                data[d] = entry
    return data


# ── Statistical Helpers ──────────────────────────────────────────────

def rolling_zscore(values: Dict[str, float], dates: List[str],
                    lookback: int) -> Dict[str, float]:
    """Compute rolling z-score of a time series."""
    result = {}
    for i in range(lookback, len(dates)):
        d = dates[i]
        val = values.get(d)
        if val is None:
            continue

        window = []
        for j in range(i - lookback, i):
            v = values.get(dates[j])
            if v is not None:
                window.append(v)

        if len(window) < lookback // 2:
            continue

        mean = sum(window) / len(window)
        std = math.sqrt(sum((v - mean)**2 for v in window) / len(window))
        if std > 1e-8:
            result[d] = (val - mean) / std
    return result


def calc_sharpe(rets: List[float]) -> Tuple[float, float, float]:
    if len(rets) < 20:
        return 0.0, 0.0, 0.0
    mean = sum(rets) / len(rets)
    var = sum((r - mean)**2 for r in rets) / len(rets)
    std = math.sqrt(var) if var > 0 else 1e-10
    sharpe = (mean * 365) / (std * math.sqrt(365))
    ann_ret = mean * 365
    cum = 0.0
    peak = 0.0
    max_dd = 0.0
    for r in rets:
        cum += r
        peak = max(peak, cum)
        max_dd = max(max_dd, peak - cum)
    return sharpe, ann_ret, max_dd


def correlation(x: List[float], y: List[float]) -> float:
    """Pearson correlation."""
    if len(x) != len(y) or len(x) < 10:
        return 0.0
    mx = sum(x) / len(x)
    my = sum(y) / len(y)
    num = sum((xi - mx) * (yi - my) for xi, yi in zip(x, y))
    dx = math.sqrt(sum((xi - mx)**2 for xi in x))
    dy = math.sqrt(sum((yi - my)**2 for yi in y))
    if dx < 1e-10 or dy < 1e-10:
        return 0.0
    return num / (dx * dy)


# ── Strategy Models ──────────────────────────────────────────────────

def vrp_pnl(dates: List[str], dvol: Dict[str, float],
            prices: Dict[str, float], lev: float = 2.0) -> Dict[str, float]:
    dt = 1.0 / 365.0
    pnl = {}
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
        pnl[d] = lev * 0.5 * (iv**2 - rv_bar**2) * dt
    return pnl


def mr_strategy(dates: List[str], feature: Dict[str, float],
                dvol: Dict[str, float], lookback: int = 60,
                z_entry: float = 1.0) -> Dict[str, float]:
    """Generic mean-reversion strategy on any feature.
    Short when z > z_entry, long when z < -z_entry.
    PnL from feature changes * IV * sqrt(dt) * scale.
    """
    dt = 1.0 / 365.0
    pnl = {}
    position = 0.0
    zscore = rolling_zscore(feature, dates, lookback)

    for i in range(1, len(dates)):
        d = dates[i]
        dp = dates[i-1]
        z = zscore.get(d)
        iv = dvol.get(d)
        f_now = feature.get(d)
        f_prev = feature.get(dp)

        if z is not None:
            if z > z_entry:
                position = -1.0
            elif z < -z_entry:
                position = 1.0
            elif abs(z) < 0.3:
                position = 0.0

        if f_now is not None and f_prev is not None and iv is not None and position != 0:
            d_feat = f_now - f_prev
            pnl[d] = position * d_feat * iv * math.sqrt(dt) * 2.5
        else:
            if d in zscore:
                pnl[d] = 0.0

    return pnl


def term_structure_strategy(dates: List[str], surface: Dict[str, dict],
                             dvol: Dict[str, float],
                             lookback: int = 60) -> Dict[str, float]:
    """
    Term spread trading: When term spread z-score is extreme,
    bet on normalization.
    """
    ts = {d: s["term_spread"] for d, s in surface.items() if "term_spread" in s}
    return mr_strategy(dates, ts, dvol, lookback)


def butterfly_strategy(dates: List[str], surface: Dict[str, dict],
                        dvol: Dict[str, float],
                        lookback: int = 60) -> Dict[str, float]:
    """
    Butterfly trading: Mean-revert butterfly (wings vs body).
    """
    bf = {d: s["butterfly_25d"] for d, s in surface.items() if "butterfly_25d" in s}
    return mr_strategy(dates, bf, dvol, lookback)


# ── Main ─────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("R56: ALTERNATIVE ALPHA SOURCES FROM REAL IV SURFACE")
    print("=" * 70)
    print()

    # Load data
    print("Loading data...")
    dvol = load_dvol_daily("BTC")
    prices = load_prices("BTC")
    surface = load_surface("BTC")
    print(f"  DVOL: {len(dvol)} days, Prices: {len(prices)} days")
    print(f"  Surface: {len(surface)} days")

    # Check feature availability
    features_count = defaultdict(int)
    for d, s in surface.items():
        for f in s:
            features_count[f] += 1
    for f, c in sorted(features_count.items()):
        print(f"    {f}: {c} days")

    all_dates = sorted(set(dvol.keys()) & set(prices.keys()) & set(surface.keys()))
    print(f"  Common dates: {len(all_dates)} ({all_dates[0]} to {all_dates[-1]})")

    # ── 1. Feature Statistics ────────────────────────────────────────
    print(f"\n{'='*70}")
    print("  STEP 1: FEATURE STATISTICS")
    print(f"{'='*70}")
    print()

    for feat_name in ["iv_atm", "skew_25d", "butterfly_25d", "term_spread"]:
        vals = [surface[d][feat_name] for d in all_dates
                if feat_name in surface.get(d, {})]
        if not vals:
            print(f"  {feat_name}: NO DATA")
            continue

        mean = sum(vals) / len(vals)
        std = math.sqrt(sum((v - mean)**2 for v in vals) / len(vals))
        q25 = sorted(vals)[len(vals)//4]
        q75 = sorted(vals)[3*len(vals)//4]

        # Autocorrelation (1-day)
        ac1 = 0
        if len(vals) > 10:
            dm = [v - mean for v in vals]
            num = sum(dm[i]*dm[i-1] for i in range(1, len(dm)))
            den = sum(d**2 for d in dm)
            ac1 = num / den if den > 0 else 0

        # Change autocorrelation (MR indicator)
        changes = [vals[i] - vals[i-1] for i in range(1, len(vals))]
        ch_ac1 = 0
        if len(changes) > 10:
            cm = sum(changes) / len(changes)
            cdm = [c - cm for c in changes]
            cnum = sum(cdm[i]*cdm[i-1] for i in range(1, len(cdm)))
            cden = sum(d**2 for d in cdm)
            ch_ac1 = cnum / cden if cden > 0 else 0

        print(f"  {feat_name}:")
        print(f"    N={len(vals)}  Mean={mean:+.4f}  Std={std:.4f}  "
              f"Q25={q25:+.4f}  Q75={q75:+.4f}")
        print(f"    Level AC(1)={ac1:.3f}  Change AC(1)={ch_ac1:.3f} "
              f"{'← MEAN-REVERTING' if ch_ac1 < -0.1 else '← weak MR' if ch_ac1 < 0 else '← NOT MR'}")
        print()

    # ── 2. Feature Correlations ──────────────────────────────────────
    print(f"\n{'='*70}")
    print("  STEP 2: FEATURE CORRELATIONS (daily changes)")
    print(f"{'='*70}")
    print()

    feat_names = ["iv_atm", "skew_25d", "butterfly_25d", "term_spread"]
    # Compute daily changes
    feat_changes = {}
    for fn in feat_names:
        changes = {}
        for i in range(1, len(all_dates)):
            d = all_dates[i]
            dp = all_dates[i-1]
            v1 = surface.get(d, {}).get(fn)
            v0 = surface.get(dp, {}).get(fn)
            if v1 is not None and v0 is not None:
                changes[d] = v1 - v0
        feat_changes[fn] = changes

    # Pairwise correlations
    print(f"  {'':>16s}", end="")
    for fn in feat_names:
        print(f"  {fn:>14s}", end="")
    print()

    for fn1 in feat_names:
        print(f"  {fn1:>16s}", end="")
        for fn2 in feat_names:
            common = sorted(set(feat_changes[fn1].keys()) & set(feat_changes[fn2].keys()))
            if len(common) < 20:
                print(f"  {'N/A':>14s}", end="")
                continue
            x = [feat_changes[fn1][d] for d in common]
            y = [feat_changes[fn2][d] for d in common]
            corr = correlation(x, y)
            print(f"  {corr:14.3f}", end="")
        print()

    # ── 3. Individual Strategy Performance ───────────────────────────
    print(f"\n{'='*70}")
    print("  STEP 3: INDIVIDUAL STRATEGY PERFORMANCE")
    print(f"{'='*70}")
    print()

    # VRP (baseline)
    vrp = vrp_pnl(all_dates, dvol, prices, 2.0)
    vrp_rets = [vrp[d] for d in sorted(vrp.keys())]
    sh_vrp, ret_vrp, dd_vrp = calc_sharpe(vrp_rets)
    print(f"  VRP (lev=2.0):        Sharpe={sh_vrp:6.2f}  Ret={ret_vrp:+.2%}  DD={dd_vrp:.2%}  N={len(vrp_rets)}")

    # Skew MR
    skew_feat = {d: s["skew_25d"] for d, s in surface.items() if "skew_25d" in s}
    strategies = {}
    for lb in [30, 60, 90, 120]:
        for ze in [0.5, 0.7, 1.0, 1.5]:
            skew_pnl = mr_strategy(all_dates, skew_feat, dvol, lb, ze)
            if len(skew_pnl) > 100:
                rets = [skew_pnl[d] for d in sorted(skew_pnl.keys())]
                sh, ret, dd = calc_sharpe(rets)
                strategies[f"Skew_lb{lb}_z{ze}"] = (sh, ret, dd, len(rets))

    # Find best skew
    best_skew = max(strategies.items(), key=lambda x: x[1][0])
    print(f"  Best Skew MR ({best_skew[0]}): Sharpe={best_skew[1][0]:6.2f}  "
          f"Ret={best_skew[1][1]:+.2%}  DD={best_skew[1][2]:.2%}")

    # Butterfly MR
    bf_feat = {d: s["butterfly_25d"] for d, s in surface.items() if "butterfly_25d" in s}
    bf_strategies = {}
    for lb in [30, 60, 90, 120]:
        for ze in [0.5, 0.7, 1.0, 1.5]:
            bf_pnl = mr_strategy(all_dates, bf_feat, dvol, lb, ze)
            if len(bf_pnl) > 100:
                rets = [bf_pnl[d] for d in sorted(bf_pnl.keys())]
                sh, ret, dd = calc_sharpe(rets)
                bf_strategies[f"BF_lb{lb}_z{ze}"] = (sh, ret, dd, len(rets))

    if bf_strategies:
        best_bf = max(bf_strategies.items(), key=lambda x: x[1][0])
        print(f"  Best Butterfly MR ({best_bf[0]}): Sharpe={best_bf[1][0]:6.2f}  "
              f"Ret={best_bf[1][1]:+.2%}  DD={best_bf[1][2]:.2%}")
    else:
        print(f"  Butterfly MR: NO DATA")

    # Term Structure MR
    ts_feat = {d: s["term_spread"] for d, s in surface.items() if "term_spread" in s}
    ts_strategies = {}
    for lb in [30, 60, 90, 120]:
        for ze in [0.5, 0.7, 1.0, 1.5]:
            ts_pnl = mr_strategy(all_dates, ts_feat, dvol, lb, ze)
            if len(ts_pnl) > 100:
                rets = [ts_pnl[d] for d in sorted(ts_pnl.keys())]
                sh, ret, dd = calc_sharpe(rets)
                ts_strategies[f"TS_lb{lb}_z{ze}"] = (sh, ret, dd, len(rets))

    if ts_strategies:
        best_ts = max(ts_strategies.items(), key=lambda x: x[1][0])
        print(f"  Best Term Spread MR ({best_ts[0]}): Sharpe={best_ts[1][0]:6.2f}  "
              f"Ret={best_ts[1][1]:+.2%}  DD={best_ts[1][2]:.2%}")
    else:
        print(f"  Term Spread MR: NO DATA")

    # ── 4. PnL Correlation Matrix (daily PnL streams) ───────────────
    print(f"\n{'='*70}")
    print("  STEP 4: PnL CORRELATION MATRIX")
    print(f"{'='*70}")
    print("  Low correlation → good diversification potential")
    print()

    # Compute PnL streams with best params
    pnl_streams = {"VRP": vrp}

    # Best skew params
    best_skew_name = best_skew[0]
    parts = best_skew_name.split("_")
    sk_lb = int(parts[1].replace("lb", ""))
    sk_ze = float(parts[2].replace("z", ""))
    pnl_streams["Skew"] = mr_strategy(all_dates, skew_feat, dvol, sk_lb, sk_ze)

    if bf_strategies:
        best_bf_name = best_bf[0]
        parts = best_bf_name.split("_")
        bf_lb = int(parts[1].replace("lb", ""))
        bf_ze = float(parts[2].replace("z", ""))
        pnl_streams["Butterfly"] = mr_strategy(all_dates, bf_feat, dvol, bf_lb, bf_ze)

    if ts_strategies:
        best_ts_name = best_ts[0]
        parts = best_ts_name.split("_")
        ts_lb = int(parts[1].replace("lb", ""))
        ts_ze = float(parts[2].replace("z", ""))
        pnl_streams["TermSpread"] = mr_strategy(all_dates, ts_feat, dvol, ts_lb, ts_ze)

    stream_names = list(pnl_streams.keys())
    print(f"  {'':>12s}", end="")
    for sn in stream_names:
        print(f"  {sn:>12s}", end="")
    print()

    for sn1 in stream_names:
        print(f"  {sn1:>12s}", end="")
        for sn2 in stream_names:
            common = sorted(set(pnl_streams[sn1].keys()) & set(pnl_streams[sn2].keys()))
            if len(common) < 20:
                print(f"  {'N/A':>12s}", end="")
                continue
            x = [pnl_streams[sn1][d] for d in common]
            y = [pnl_streams[sn2][d] for d in common]
            corr = correlation(x, y)
            print(f"  {corr:12.3f}", end="")
        print()

    # ── 5. Multi-Factor Ensemble ─────────────────────────────────────
    print(f"\n{'='*70}")
    print("  STEP 5: MULTI-FACTOR ENSEMBLE OPTIMIZATION")
    print(f"{'='*70}")
    print()

    # Common dates across all streams
    common_dates = sorted(set.intersection(*[set(s.keys()) for s in pnl_streams.values()]))
    print(f"  Common dates for all factors: {len(common_dates)}")

    if len(common_dates) < 200:
        print("  Not enough common dates for meaningful optimization")
    else:
        # Grid search for optimal weights
        best_sh = -999
        best_weights = {}

        n_factors = len(stream_names)
        if n_factors == 2:
            weight_combos = [(w/10, 1-w/10) for w in range(1, 10)]
        elif n_factors == 3:
            weight_combos = []
            for w1 in range(1, 9):
                for w2 in range(1, 10 - w1):
                    w3 = 10 - w1 - w2
                    weight_combos.append((w1/10, w2/10, w3/10))
        elif n_factors == 4:
            weight_combos = []
            for w1 in range(1, 8):
                for w2 in range(1, 9 - w1):
                    for w3 in range(1, 10 - w1 - w2):
                        w4 = 10 - w1 - w2 - w3
                        weight_combos.append((w1/10, w2/10, w3/10, w4/10))
        else:
            weight_combos = [(1.0,)]

        print(f"  Testing {len(weight_combos)} weight combinations...")

        results_all = []
        for weights in weight_combos:
            ens_rets = []
            for d in common_dates:
                r = sum(w * pnl_streams[sn].get(d, 0) for w, sn in zip(weights, stream_names))
                ens_rets.append(r)

            sh, ret, dd = calc_sharpe(ens_rets)
            results_all.append((weights, sh, ret, dd))

            if sh > best_sh:
                best_sh = sh
                best_weights = dict(zip(stream_names, weights))

        # Sort and show top 10
        results_all.sort(key=lambda x: x[1], reverse=True)
        print(f"\n  Top 10 weight combinations:")
        print(f"  {'Weights':<40s}  {'Sharpe':>7s}  {'AnnRet':>8s}  {'MaxDD':>7s}")
        for weights, sh, ret, dd in results_all[:10]:
            w_str = "  ".join(f"{sn}={w:.1f}" for sn, w in zip(stream_names, weights))
            print(f"  {w_str:<40s}  {sh:7.2f}  {ret:+8.2%}  {dd:7.2%}")

        # Compare with VRP-only
        vrp_only = [vrp.get(d, 0) for d in common_dates]
        sh_v, ret_v, dd_v = calc_sharpe(vrp_only)
        print(f"\n  VRP-only on same dates: Sharpe={sh_v:.2f}  Ret={ret_v:+.2%}  DD={dd_v:.2%}")
        print(f"  Best multi-factor:     Sharpe={best_sh:.2f}  Weights={best_weights}")

    # ── 6. Yearly breakdown of multi-factor ──────────────────────────
    print(f"\n{'='*70}")
    print("  STEP 6: YEARLY BREAKDOWN — MULTI-FACTOR vs VRP-ONLY")
    print(f"{'='*70}")
    print()

    if len(common_dates) >= 200 and best_weights:
        print(f"  {'Year':<6s}  {'VRP Sharpe':>10s}  {'MF Sharpe':>10s}  {'Δ':>6s}  {'MF Ret':>8s}")
        for yr in sorted(set(d[:4] for d in common_dates)):
            yr_dates = [d for d in common_dates if d[:4] == yr]
            if len(yr_dates) < 20:
                continue

            vrp_yr = [vrp.get(d, 0) for d in yr_dates]
            sh_v_yr, ret_v_yr, _ = calc_sharpe(vrp_yr)

            mf_yr = [sum(best_weights[sn] * pnl_streams[sn].get(d, 0)
                        for sn in stream_names) for d in yr_dates]
            sh_mf_yr, ret_mf_yr, _ = calc_sharpe(mf_yr)

            delta = sh_mf_yr - sh_v_yr
            print(f"  {yr:<6s}  {sh_v_yr:10.2f}  {sh_mf_yr:10.2f}  {delta:+6.2f}  {ret_mf_yr:+8.2%}")

    # ── Summary ──────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("R56 SUMMARY")
    print(f"{'='*70}")
    print()
    print("  Feature statistics:")
    for feat_name in feat_names:
        vals = [surface[d][feat_name] for d in all_dates
                if feat_name in surface.get(d, {})]
        if vals:
            mean = sum(vals) / len(vals)
            print(f"    {feat_name}: N={len(vals)}  Mean={mean:+.4f}")
    print()
    print("  Individual strategies (best params):")
    print(f"    VRP:         Sharpe={sh_vrp:.2f}")
    if strategies:
        print(f"    Skew MR:     Sharpe={best_skew[1][0]:.2f} ({best_skew[0]})")
    if bf_strategies:
        print(f"    Butterfly:   Sharpe={best_bf[1][0]:.2f} ({best_bf[0]})")
    if ts_strategies:
        print(f"    TermSpread:  Sharpe={best_ts[1][0]:.2f} ({best_ts[0]})")
    if best_weights:
        print(f"\n  Best multi-factor: Sharpe={best_sh:.2f}")
        print(f"    Weights: {best_weights}")

    # Save results
    results = {
        "research_id": "R56",
        "title": "Alternative Alpha Sources from Real IV Surface",
        "vrp_sharpe": sh_vrp,
        "best_skew": {"name": best_skew[0], "sharpe": best_skew[1][0]} if strategies else None,
        "best_butterfly": {"name": best_bf[0], "sharpe": best_bf[1][0]} if bf_strategies else None,
        "best_term_spread": {"name": best_ts[0], "sharpe": best_ts[1][0]} if ts_strategies else None,
        "best_multi_factor": {"sharpe": best_sh, "weights": best_weights} if best_weights else None,
    }
    outpath = ROOT / "data" / "cache" / "deribit" / "real_surface" / "r56_alternative_alpha_results.json"
    with open(outpath, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved to {outpath}")


if __name__ == "__main__":
    main()
