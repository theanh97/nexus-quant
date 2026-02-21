#!/usr/bin/env python3
"""
Asymmetric Skew Z-Score Thresholds — R42
==========================================

Current Skew MR: z_entry=2.0, z_exit=0.0 (symmetric).
Skew has natural negative bias (puts > calls in crypto).

Questions:
  1. Is the skew z-score distribution symmetric?
  2. Should entry thresholds differ for positive vs negative z?
  3. Can asymmetric exits improve: tighter exit on losing side?
  4. Should BTC and ETH have different skew thresholds?

Tests:
  A. Skew z-score distribution analysis (skewness, kurtosis, quantiles)
  B. Asymmetric entry thresholds: different z_entry for long/short
  C. Asymmetric exit: different z_exit for profitable vs losing trades
  D. Per-asset asymmetric thresholds
  E. Combined: best asymmetric + IV sizing
"""
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from nexus_quant.backtest.costs import ExecutionCostModel, FeeModel, ImpactModel
from nexus_quant.projects.crypto_options.options_engine import compute_metrics
from nexus_quant.projects.crypto_options.providers.deribit_rest import DeribitRestProvider
from nexus_quant.projects.crypto_options.strategies.variance_premium import VariancePremiumStrategy
from nexus_quant.projects.crypto_options.strategies.skew_trade_v2 import SkewTradeV2Strategy

fees = FeeModel(maker_fee_rate=0.0003, taker_fee_rate=0.0005)
impact = ImpactModel(model="sqrt", coef_bps=2.0)
COSTS = ExecutionCostModel(fee=fees, impact=impact)

YEARS = [2021, 2022, 2023, 2024, 2025]
SEED = 42
BARS_PER_YEAR = 365
W_VRP = 0.40
W_SKEW = 0.60

BTC_VRP = {"base_leverage": 1.5, "exit_z_threshold": -3.0, "vrp_lookback": 30, "rebalance_freq": 5, "min_bars": 30}
ETH_VRP = {"base_leverage": 1.5, "exit_z_threshold": -3.0, "vrp_lookback": 60, "rebalance_freq": 5, "min_bars": 60}

IV_LOOKBACK = 180


def iv_percentile(iv_series, idx, lookback=180):
    if idx < lookback or not iv_series:
        return None
    start = max(0, idx - lookback)
    window = [v for v in iv_series[start:idx] if v is not None]
    if len(window) < 10:
        return None
    current = iv_series[idx] if idx < len(iv_series) and iv_series[idx] is not None else None
    if current is None:
        return None
    return sum(1 for v in window if v < current) / len(window)


def iv_sizing_scale(pct):
    if pct < 0.25:
        return 0.50
    elif pct > 0.75:
        return 1.70
    return 1.0


def compute_skew_zscore(skew_series, idx, lookback=60):
    """Compute z-score of skew at index idx."""
    if idx < lookback or not skew_series:
        return None
    start = max(0, idx - lookback)
    window = [v for v in skew_series[start:idx] if v is not None]
    if len(window) < 10:
        return None
    current = skew_series[idx] if idx < len(skew_series) and skew_series[idx] is not None else None
    if current is None:
        return None
    mean = sum(window) / len(window)
    var = sum((v - mean) ** 2 for v in window) / len(window)
    std = var ** 0.5
    if std < 1e-10:
        return 0.0
    return (current - mean) / std


def run_asymmetric_skew(
    z_entry_long, z_entry_short, z_exit_long, z_exit_short,
    skew_lookback=60, use_iv_sizing=True,
):
    """Run multi-asset ensemble with asymmetric skew thresholds."""
    sharpes = []
    yearly_detail = {}

    for yr in YEARS:
        cfg = {
            "symbols": ["BTC", "ETH"],
            "start": f"{yr}-01-01", "end": f"{yr}-12-31",
            "bar_interval": "1d", "use_synthetic_iv": True,
        }
        provider = DeribitRestProvider(cfg, seed=SEED)
        dataset = provider.load()
        n = len(dataset.timeline)

        vrp_strats = {
            "BTC": VariancePremiumStrategy(params=BTC_VRP),
            "ETH": VariancePremiumStrategy(params=ETH_VRP),
        }

        dt = 1.0 / BARS_PER_YEAR
        equity = 1.0
        vrp_w = {"BTC": 0.0, "ETH": 0.0}
        skew_w = {"BTC": 0.0, "ETH": 0.0}
        skew_pos = {"BTC": 0.0, "ETH": 0.0}  # track direction
        equity_curve = [1.0]
        returns_list = []
        last_skew_rebal = {"BTC": 0, "ETH": 0}

        for idx in range(1, n):
            prev_equity = equity
            total_pnl = 0.0

            for sym in ["BTC", "ETH"]:
                aw = 0.50
                w_v = vrp_w.get(sym, 0.0)
                vpnl = 0.0
                if abs(w_v) > 1e-10:
                    closes = dataset.perp_close.get(sym, [])
                    if idx < len(closes) and closes[idx-1] > 0 and closes[idx] > 0:
                        lr = math.log(closes[idx]/closes[idx-1])
                        rv = abs(lr) * math.sqrt(BARS_PER_YEAR)
                    else:
                        rv = 0.0
                    ivs = dataset.features.get("iv_atm", {}).get(sym, [])
                    iv = ivs[idx-1] if ivs and idx-1 < len(ivs) else None
                    if iv and iv > 0:
                        vpnl = (-w_v) * 0.5 * (iv**2 - rv**2) * dt

                w_s = skew_w.get(sym, 0.0)
                spnl = 0.0
                if abs(w_s) > 1e-10:
                    sks = dataset.features.get("skew_25d", {}).get(sym, [])
                    if idx < len(sks) and idx-1 < len(sks):
                        sn, sp = sks[idx], sks[idx-1]
                        if sn is not None and sp is not None:
                            ds = float(sn) - float(sp)
                            ivs = dataset.features.get("iv_atm", {}).get(sym, [])
                            iv_s = ivs[idx-1] if ivs and idx-1 < len(ivs) else 0.70
                            if iv_s and iv_s > 0:
                                spnl = w_s * ds * iv_s * math.sqrt(dt) * 2.5

                total_pnl += aw * (W_VRP * vpnl + W_SKEW * spnl)

            equity += equity * total_pnl

            # Rebalance VRP
            rebal_happened = False
            for sym in ["BTC", "ETH"]:
                if vrp_strats[sym].should_rebalance(dataset, idx):
                    rebal_happened = True
                    tv = vrp_strats[sym].target_weights(dataset, idx, {sym: vrp_w.get(sym, 0.0)})
                    vrp_w[sym] = tv.get(sym, 0.0)

            # Rebalance Skew (asymmetric logic)
            for sym in ["BTC", "ETH"]:
                if idx - last_skew_rebal[sym] < 5:
                    continue
                lb = skew_lookback if sym == "BTC" else 90

                skew_series = dataset.features.get("skew_25d", {}).get(sym, [])
                z = compute_skew_zscore(skew_series, idx, lookback=lb)
                if z is None:
                    continue

                per_sym_lev = 1.0 / 2  # target_leverage / num_symbols
                current_pos = skew_pos[sym]

                new_pos = current_pos
                if current_pos == 0.0:
                    # Entry: asymmetric thresholds
                    if z >= z_entry_short:
                        new_pos = -per_sym_lev  # sell skew (z is high)
                    elif z <= -z_entry_long:
                        new_pos = per_sym_lev   # buy skew (z is low)
                else:
                    # Exit: asymmetric
                    if current_pos > 0:
                        # Long position: exit when z crosses z_exit_long
                        if z >= z_exit_long:
                            new_pos = 0.0
                    else:
                        # Short position: exit when z crosses -z_exit_short
                        if z <= -z_exit_short:
                            new_pos = 0.0

                if new_pos != current_pos:
                    rebal_happened = True
                    last_skew_rebal[sym] = idx

                skew_pos[sym] = new_pos
                skew_w[sym] = new_pos

            # Apply IV sizing
            if use_iv_sizing and rebal_happened:
                for sym in ["BTC", "ETH"]:
                    ivs = dataset.features.get("iv_atm", {}).get(sym, [])
                    pct = iv_percentile(ivs, idx)
                    if pct is not None:
                        sc = iv_sizing_scale(pct)
                        vrp_w[sym] *= sc
                        skew_w[sym] *= sc

            if rebal_happened:
                bd = COSTS.cost(equity=equity, turnover=0.05)
                equity -= float(bd.get("cost", 0.0))
                equity = max(equity, 0.0)

            equity_curve.append(equity)
            bar_ret = (equity / prev_equity) - 1.0 if prev_equity > 0 else 0.0
            returns_list.append(bar_ret)

        m = compute_metrics(equity_curve, returns_list, BARS_PER_YEAR)
        sharpes.append(m["sharpe"])
        yearly_detail[str(yr)] = round(m["sharpe"], 3)

    avg = sum(sharpes) / len(sharpes)
    mn = min(sharpes)
    return {"avg_sharpe": round(avg, 3), "min_sharpe": round(mn, 3), "yearly": yearly_detail}


def main():
    print("=" * 70)
    print("ASYMMETRIC SKEW Z-SCORE THRESHOLDS — R42")
    print("=" * 70)
    print()

    # ── A. Skew Z-Score Distribution ──────────────────────────────────────
    print("--- A. SKEW Z-SCORE DISTRIBUTION ---")
    all_z = {"BTC": [], "ETH": []}
    for yr in YEARS:
        cfg = {"symbols": ["BTC", "ETH"], "start": f"{yr}-01-01", "end": f"{yr}-12-31",
               "bar_interval": "1d", "use_synthetic_iv": True}
        provider = DeribitRestProvider(cfg, seed=SEED)
        dataset = provider.load()
        for sym in ["BTC", "ETH"]:
            skew_series = dataset.features.get("skew_25d", {}).get(sym, [])
            lb = 60 if sym == "BTC" else 90
            for idx in range(lb, len(skew_series)):
                z = compute_skew_zscore(skew_series, idx, lookback=lb)
                if z is not None:
                    all_z[sym].append(z)

    for sym in ["BTC", "ETH"]:
        zs = all_z[sym]
        if zs:
            n = len(zs)
            mean = sum(zs) / n
            std = (sum((z - mean) ** 2 for z in zs) / n) ** 0.5
            skewness = sum((z - mean) ** 3 for z in zs) / (n * std ** 3) if std > 0 else 0
            kurtosis = sum((z - mean) ** 4 for z in zs) / (n * std ** 4) - 3 if std > 0 else 0
            sorted_z = sorted(zs)
            q05 = sorted_z[int(0.05 * n)]
            q25 = sorted_z[int(0.25 * n)]
            q75 = sorted_z[int(0.75 * n)]
            q95 = sorted_z[int(0.95 * n)]
            pct_above_2 = sum(1 for z in zs if z > 2.0) / n * 100
            pct_below_m2 = sum(1 for z in zs if z < -2.0) / n * 100

            print(f"  {sym}: mean={mean:.3f} std={std:.3f} skew={skewness:.3f} kurt={kurtosis:.3f}")
            print(f"        q05={q05:.2f} q25={q25:.2f} q75={q75:.2f} q95={q95:.2f}")
            print(f"        z>+2: {pct_above_2:.1f}%  z<-2: {pct_below_m2:.1f}%")
    print()

    # ── B. Baseline (symmetric) ───────────────────────────────────────────
    print("--- B. BASELINE (symmetric z_entry=2.0, z_exit=0.0) ---")
    baseline = run_asymmetric_skew(2.0, 2.0, 0.0, 0.0)
    print(f"  Baseline: avg={baseline['avg_sharpe']:.3f} min={baseline['min_sharpe']:.3f}")
    print()

    # ── C. Asymmetric Entry Thresholds ────────────────────────────────────
    print("--- C. ASYMMETRIC ENTRY THRESHOLDS ---")
    print("  (z_entry_long = threshold for buying skew, z_entry_short = for selling)")

    entry_configs = [
        (2.0, 2.0, "symmetric 2.0/2.0"),
        (1.5, 2.0, "easier long 1.5/2.0"),
        (2.0, 1.5, "easier short 2.0/1.5"),
        (1.5, 2.5, "easy long 1.5 / hard short 2.5"),
        (2.5, 1.5, "hard long 2.5 / easy short 1.5"),
        (1.5, 1.5, "both easy 1.5/1.5"),
        (2.5, 2.5, "both hard 2.5/2.5"),
        (1.0, 2.0, "very easy long 1.0/2.0"),
        (2.0, 1.0, "very easy short 2.0/1.0"),
        (1.5, 3.0, "easy long / very hard short"),
        (3.0, 1.5, "very hard long / easy short"),
    ]

    entry_results = []
    for z_long, z_short, label in entry_configs:
        r = run_asymmetric_skew(z_long, z_short, 0.0, 0.0)
        d = r["avg_sharpe"] - baseline["avg_sharpe"]
        entry_results.append({"z_long": z_long, "z_short": z_short, "label": label, **r, "delta": round(d, 3)})
        marker = " *" if z_long == 2.0 and z_short == 2.0 else ""
        print(f"  {label:35s} avg={r['avg_sharpe']:.3f} min={r['min_sharpe']:.3f} Δ={d:+.3f}{marker}")

    best_entry = max(entry_results, key=lambda x: x["avg_sharpe"])
    print(f"\n  Best entry: {best_entry['label']} avg={best_entry['avg_sharpe']:.3f}")
    print()

    # ── D. Asymmetric Exit Thresholds ─────────────────────────────────────
    print("--- D. ASYMMETRIC EXIT THRESHOLDS ---")
    print("  (z_exit_long = exit long pos when z crosses this, z_exit_short = for short)")

    exit_configs = [
        (0.0, 0.0, "symmetric 0.0/0.0"),
        (0.5, 0.0, "tighter long exit 0.5/0.0"),
        (0.0, 0.5, "tighter short exit 0.0/0.5"),
        (0.5, 0.5, "both tighter 0.5/0.5"),
        (-0.5, 0.0, "wider long exit -0.5/0.0"),
        (0.0, -0.5, "wider short exit 0.0/-0.5"),
        (-0.5, -0.5, "both wider -0.5/-0.5"),
        (1.0, 0.0, "very tight long exit 1.0/0.0"),
        (0.0, 1.0, "very tight short exit 0.0/1.0"),
    ]

    exit_results = []
    for z_exit_l, z_exit_s, label in exit_configs:
        r = run_asymmetric_skew(2.0, 2.0, z_exit_l, z_exit_s)
        d = r["avg_sharpe"] - baseline["avg_sharpe"]
        exit_results.append({"z_exit_long": z_exit_l, "z_exit_short": z_exit_s, "label": label, **r, "delta": round(d, 3)})
        marker = " *" if z_exit_l == 0.0 and z_exit_s == 0.0 else ""
        print(f"  {label:35s} avg={r['avg_sharpe']:.3f} min={r['min_sharpe']:.3f} Δ={d:+.3f}{marker}")

    best_exit = max(exit_results, key=lambda x: x["avg_sharpe"])
    print(f"\n  Best exit: {best_exit['label']} avg={best_exit['avg_sharpe']:.3f}")
    print()

    # ── E. Combined Best ──────────────────────────────────────────────────
    print("--- E. COMBINED BEST ---")
    if best_entry["z_long"] != 2.0 or best_entry["z_short"] != 2.0 or \
       best_exit["z_exit_long"] != 0.0 or best_exit["z_exit_short"] != 0.0:
        combined = run_asymmetric_skew(
            best_entry["z_long"], best_entry["z_short"],
            best_exit["z_exit_long"], best_exit["z_exit_short"],
        )
        d = combined["avg_sharpe"] - baseline["avg_sharpe"]
        print(f"  Entry: long={best_entry['z_long']:.1f} short={best_entry['z_short']:.1f}")
        print(f"  Exit:  long={best_exit['z_exit_long']:.1f} short={best_exit['z_exit_short']:.1f}")
        print(f"  Combined: avg={combined['avg_sharpe']:.3f} min={combined['min_sharpe']:.3f} Δ={d:+.3f}")
    else:
        print(f"  No improvement from asymmetric thresholds — symmetric baseline is optimal")
        combined = baseline
    print()

    # ── SUMMARY ───────────────────────────────────────────────────────────
    print("=" * 70)
    print("SUMMARY — R42: Asymmetric Skew Z-Score Thresholds")
    print("=" * 70)
    print()

    print(f"  Baseline (symmetric 2.0/0.0): avg={baseline['avg_sharpe']:.3f}")
    print(f"  Best entry: {best_entry['label']} (Δ={best_entry['delta']:+.3f})")
    print(f"  Best exit:  {best_exit['label']} (Δ={best_exit['delta']:+.3f})")
    print()

    total_d = max(best_entry["delta"], best_exit["delta"], 0)
    print("=" * 70)
    if total_d > 0.05:
        print(f"VERDICT: Asymmetric thresholds HELP (+{total_d:.3f})")
    elif total_d > -0.03:
        print(f"VERDICT: Asymmetric thresholds MARGINAL (best Δ={total_d:+.3f})")
    else:
        print(f"VERDICT: Symmetric thresholds are optimal — no improvement from asymmetry")
    print("=" * 70)

    # Save
    out = ROOT / "artifacts" / "crypto_options" / "skew_asymmetric_zscore.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump({
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "research_id": "R42",
            "baseline": baseline,
            "entry_results": entry_results,
            "exit_results": exit_results,
        }, f, indent=2)
    print(f"\nResults saved to: {out}")


if __name__ == "__main__":
    main()
