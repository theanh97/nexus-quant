#!/usr/bin/env python3
"""
Dynamic BTC/ETH Asset Allocation by IV Regime — R38
====================================================

R31 found 50/50 BTC/ETH is best for avg Sharpe (3.174), 50/50 best for min.
R12 showed static > dynamic for strategy weights.

But cross-asset allocation is different: if ETH IV is extremely high vs BTC,
ETH VRP carry is much richer → should we tilt toward ETH?

Tests:
  A. Static allocation benchmark (30/70, 40/60, 50/50, 60/40, 70/30)
  B. IV-ratio tilt: allocate more to asset with higher IV percentile
  C. VRP-spread tilt: allocate more to asset with wider carry spread
  D. Volatility-parity: allocate inversely proportional to realized vol
  E. Combined: IV-ratio tilt + vol-parity

All with R35 ETH params (VRP lb=60) and conservative IV sizing (0.5/1.7x).
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

# BTC params
BTC_VRP = {"base_leverage": 1.5, "exit_z_threshold": -3.0, "vrp_lookback": 30, "rebalance_freq": 5, "min_bars": 30}
BTC_SKEW = {"skew_lookback": 60, "z_entry": 2.0, "z_exit": 0.0, "target_leverage": 1.0, "rebalance_freq": 5, "min_bars": 60}

# ETH params (R35 optimized)
ETH_VRP = {"base_leverage": 1.5, "exit_z_threshold": -3.0, "vrp_lookback": 60, "rebalance_freq": 5, "min_bars": 60}
ETH_SKEW = {"skew_lookback": 90, "z_entry": 2.0, "z_exit": 0.0, "target_leverage": 1.0, "rebalance_freq": 5, "min_bars": 90}

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


def iv_sizing_scale(pct, low_scale=0.50, high_scale=1.70):
    if pct < 0.25:
        return low_scale
    elif pct > 0.75:
        return high_scale
    return 1.0


def compute_rv(closes, idx, window=30):
    """Compute annualized realized vol over lookback window."""
    if idx < window + 1 or len(closes) < idx + 1:
        return None
    log_rets = []
    for i in range(idx - window, idx):
        if closes[i] > 0 and closes[i + 1] > 0:
            log_rets.append(math.log(closes[i + 1] / closes[i]))
    if len(log_rets) < 10:
        return None
    var = sum(r ** 2 for r in log_rets) / len(log_rets)
    return math.sqrt(var * 365)


def run_dynamic_allocation(
    alloc_method: str,
    alloc_params: dict = None,
) -> Dict[str, Any]:
    """Run multi-asset ensemble with dynamic asset allocation."""
    if alloc_params is None:
        alloc_params = {}

    sharpes = []
    yearly_detail = {}

    for yr in YEARS:
        cfg = {
            "symbols": ["BTC", "ETH"],
            "start": f"{yr}-01-01",
            "end": f"{yr}-12-31",
            "bar_interval": "1d",
            "use_synthetic_iv": True,
        }
        provider = DeribitRestProvider(cfg, seed=SEED)
        dataset = provider.load()
        n = len(dataset.timeline)

        strats = {
            "BTC": {
                "vrp": VariancePremiumStrategy(params=BTC_VRP),
                "skew": SkewTradeV2Strategy(params=BTC_SKEW),
            },
            "ETH": {
                "vrp": VariancePremiumStrategy(params=ETH_VRP),
                "skew": SkewTradeV2Strategy(params=ETH_SKEW),
            },
        }

        dt = 1.0 / BARS_PER_YEAR
        equity = 1.0
        vrp_weights = {"BTC": 0.0, "ETH": 0.0}
        skew_weights = {"BTC": 0.0, "ETH": 0.0}
        equity_curve = [1.0]
        returns_list = []

        # Static allocation defaults
        w_btc = alloc_params.get("w_btc", 0.50)
        w_eth = alloc_params.get("w_eth", 0.50)

        for idx in range(1, n):
            prev_equity = equity

            # Dynamic allocation logic (computed before P&L)
            if alloc_method == "static":
                pass  # keep w_btc, w_eth as-is

            elif alloc_method == "iv_ratio_tilt":
                btc_iv_series = dataset.features.get("iv_atm", {}).get("BTC", [])
                eth_iv_series = dataset.features.get("iv_atm", {}).get("ETH", [])
                btc_pct = iv_percentile(btc_iv_series, idx)
                eth_pct = iv_percentile(eth_iv_series, idx)
                if btc_pct is not None and eth_pct is not None:
                    # Tilt toward asset with higher IV percentile (richer carry)
                    tilt_strength = alloc_params.get("tilt_strength", 0.20)
                    base = 0.50
                    diff = eth_pct - btc_pct  # positive = ETH IV richer
                    w_eth = base + diff * tilt_strength
                    w_eth = max(0.20, min(0.80, w_eth))
                    w_btc = 1.0 - w_eth

            elif alloc_method == "vrp_spread_tilt":
                # Tilt toward asset with wider IV-RV spread
                btc_iv_series = dataset.features.get("iv_atm", {}).get("BTC", [])
                eth_iv_series = dataset.features.get("iv_atm", {}).get("ETH", [])
                btc_closes = dataset.perp_close.get("BTC", [])
                eth_closes = dataset.perp_close.get("ETH", [])

                btc_iv = btc_iv_series[idx] if idx < len(btc_iv_series) else None
                eth_iv = eth_iv_series[idx] if idx < len(eth_iv_series) else None
                btc_rv = compute_rv(btc_closes, idx, window=30)
                eth_rv = compute_rv(eth_closes, idx, window=30)

                if btc_iv and eth_iv and btc_rv and eth_rv:
                    btc_spread = btc_iv ** 2 - btc_rv ** 2
                    eth_spread = eth_iv ** 2 - eth_rv ** 2
                    total_spread = abs(btc_spread) + abs(eth_spread)
                    if total_spread > 1e-6:
                        tilt_strength = alloc_params.get("tilt_strength", 0.30)
                        base = 0.50
                        # Allocate proportionally to spread magnitude
                        eth_share = abs(eth_spread) / total_spread
                        diff = eth_share - 0.50
                        w_eth = base + diff * tilt_strength
                        w_eth = max(0.20, min(0.80, w_eth))
                        w_btc = 1.0 - w_eth

            elif alloc_method == "vol_parity":
                # Inverse volatility weighting
                btc_closes = dataset.perp_close.get("BTC", [])
                eth_closes = dataset.perp_close.get("ETH", [])
                btc_rv = compute_rv(btc_closes, idx, window=alloc_params.get("rv_window", 30))
                eth_rv = compute_rv(eth_closes, idx, window=alloc_params.get("rv_window", 30))
                if btc_rv and eth_rv and btc_rv > 0 and eth_rv > 0:
                    inv_btc = 1.0 / btc_rv
                    inv_eth = 1.0 / eth_rv
                    total_inv = inv_btc + inv_eth
                    w_btc = inv_btc / total_inv
                    w_eth = inv_eth / total_inv

            elif alloc_method == "momentum_tilt":
                # Tilt toward asset with better recent performance (30d return)
                btc_closes = dataset.perp_close.get("BTC", [])
                eth_closes = dataset.perp_close.get("ETH", [])
                window = alloc_params.get("momentum_window", 30)
                if idx > window:
                    btc_mom = math.log(btc_closes[idx] / btc_closes[idx - window]) if btc_closes[idx] > 0 and btc_closes[idx - window] > 0 else 0
                    eth_mom = math.log(eth_closes[idx] / eth_closes[idx - window]) if eth_closes[idx] > 0 and eth_closes[idx - window] > 0 else 0
                    tilt = alloc_params.get("tilt_strength", 0.20)
                    diff = eth_mom - btc_mom
                    w_eth = 0.50 + diff * tilt
                    w_eth = max(0.20, min(0.80, w_eth))
                    w_btc = 1.0 - w_eth

            asset_weights = {"BTC": w_btc, "ETH": w_eth}

            # P&L computation
            total_pnl = 0.0
            for sym in ["BTC", "ETH"]:
                aw = asset_weights[sym]
                w_v = vrp_weights.get(sym, 0.0)
                vpnl = 0.0
                if abs(w_v) > 1e-10:
                    closes = dataset.perp_close.get(sym, [])
                    if idx < len(closes) and closes[idx - 1] > 0 and closes[idx] > 0:
                        lr = math.log(closes[idx] / closes[idx - 1])
                        rv = abs(lr) * math.sqrt(BARS_PER_YEAR)
                    else:
                        rv = 0.0
                    ivs = dataset.features.get("iv_atm", {}).get(sym, [])
                    iv = ivs[idx - 1] if ivs and idx - 1 < len(ivs) else None
                    if iv and iv > 0:
                        vpnl = (-w_v) * 0.5 * (iv ** 2 - rv ** 2) * dt

                w_s = skew_weights.get(sym, 0.0)
                spnl = 0.0
                if abs(w_s) > 1e-10:
                    sks = dataset.features.get("skew_25d", {}).get(sym, [])
                    if idx < len(sks) and idx - 1 < len(sks):
                        sn, sp = sks[idx], sks[idx - 1]
                        if sn is not None and sp is not None:
                            ds = float(sn) - float(sp)
                            ivs = dataset.features.get("iv_atm", {}).get(sym, [])
                            iv_s = ivs[idx - 1] if ivs and idx - 1 < len(ivs) else 0.70
                            if iv_s and iv_s > 0:
                                spnl = w_s * ds * iv_s * math.sqrt(dt) * 2.5

                total_pnl += aw * (W_VRP * vpnl + W_SKEW * spnl)

            equity += equity * total_pnl

            # Rebalance
            rebal_happened = False
            for sym in ["BTC", "ETH"]:
                vrp_rebal = strats[sym]["vrp"].should_rebalance(dataset, idx)
                skew_rebal = strats[sym]["skew"].should_rebalance(dataset, idx)
                if vrp_rebal or skew_rebal:
                    rebal_happened = True
                    if vrp_rebal:
                        tv = strats[sym]["vrp"].target_weights(dataset, idx, {sym: vrp_weights.get(sym, 0.0)})
                        vrp_weights[sym] = tv.get(sym, 0.0)
                    if skew_rebal:
                        ts = strats[sym]["skew"].target_weights(dataset, idx, {sym: skew_weights.get(sym, 0.0)})
                        skew_weights[sym] = ts.get(sym, 0.0)
                    ivs = dataset.features.get("iv_atm", {}).get(sym, [])
                    pct = iv_percentile(ivs, idx)
                    if pct is not None:
                        sc = iv_sizing_scale(pct)
                        vrp_weights[sym] *= sc
                        skew_weights[sym] *= sc

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
    print("DYNAMIC BTC/ETH ASSET ALLOCATION — R38")
    print("=" * 70)
    print()

    results = {}

    # ── A. Static Baselines ────────────────────────────────────────────────
    print("--- A. STATIC ALLOCATION BASELINES ---")
    static_configs = [
        (0.30, 0.70, "30/70"),
        (0.40, 0.60, "40/60"),
        (0.50, 0.50, "50/50"),
        (0.60, 0.40, "60/40"),
        (0.70, 0.30, "70/30"),
    ]

    static_results = []
    for wb, we, label in static_configs:
        r = run_dynamic_allocation("static", {"w_btc": wb, "w_eth": we})
        static_results.append({"label": label, "w_btc": wb, "w_eth": we, **r})
        print(f"  {label:8s}  avg={r['avg_sharpe']:.3f} min={r['min_sharpe']:.3f}")

    best_static = max(static_results, key=lambda x: x["avg_sharpe"])
    results["static"] = static_results
    print(f"\n  Best static: {best_static['label']} avg={best_static['avg_sharpe']:.3f}")
    print()

    # ── B. IV-Ratio Tilt ──────────────────────────────────────────────────
    print("--- B. IV-RATIO TILT ---")
    print("  (Tilt toward asset with higher IV percentile)")

    iv_tilt_results = []
    for strength in [0.10, 0.20, 0.30, 0.50, 0.70]:
        r = run_dynamic_allocation("iv_ratio_tilt", {"tilt_strength": strength})
        d = r["avg_sharpe"] - best_static["avg_sharpe"]
        iv_tilt_results.append({"tilt_strength": strength, **r, "delta": round(d, 3)})
        print(f"  tilt={strength:.2f}  avg={r['avg_sharpe']:.3f} min={r['min_sharpe']:.3f} Δ={d:+.3f}")

    best_iv_tilt = max(iv_tilt_results, key=lambda x: x["avg_sharpe"])
    results["iv_tilt"] = iv_tilt_results
    print(f"\n  Best IV tilt: str={best_iv_tilt['tilt_strength']:.2f} avg={best_iv_tilt['avg_sharpe']:.3f}")
    print()

    # ── C. VRP-Spread Tilt ────────────────────────────────────────────────
    print("--- C. VRP-SPREAD TILT ---")
    print("  (Tilt toward asset with wider carry spread)")

    vrp_tilt_results = []
    for strength in [0.20, 0.30, 0.50, 0.70, 1.00]:
        r = run_dynamic_allocation("vrp_spread_tilt", {"tilt_strength": strength})
        d = r["avg_sharpe"] - best_static["avg_sharpe"]
        vrp_tilt_results.append({"tilt_strength": strength, **r, "delta": round(d, 3)})
        print(f"  tilt={strength:.2f}  avg={r['avg_sharpe']:.3f} min={r['min_sharpe']:.3f} Δ={d:+.3f}")

    best_vrp_tilt = max(vrp_tilt_results, key=lambda x: x["avg_sharpe"])
    results["vrp_tilt"] = vrp_tilt_results
    print(f"\n  Best VRP tilt: str={best_vrp_tilt['tilt_strength']:.2f} avg={best_vrp_tilt['avg_sharpe']:.3f}")
    print()

    # ── D. Volatility Parity ──────────────────────────────────────────────
    print("--- D. VOLATILITY PARITY ---")
    print("  (Inverse RV weighting)")

    vp_results = []
    for window in [20, 30, 60, 90]:
        r = run_dynamic_allocation("vol_parity", {"rv_window": window})
        d = r["avg_sharpe"] - best_static["avg_sharpe"]
        vp_results.append({"rv_window": window, **r, "delta": round(d, 3)})
        print(f"  rv_win={window:3d}  avg={r['avg_sharpe']:.3f} min={r['min_sharpe']:.3f} Δ={d:+.3f}")

    best_vp = max(vp_results, key=lambda x: x["avg_sharpe"])
    results["vol_parity"] = vp_results
    print(f"\n  Best vol-parity: win={best_vp['rv_window']} avg={best_vp['avg_sharpe']:.3f}")
    print()

    # ── E. Momentum Tilt ──────────────────────────────────────────────────
    print("--- E. MOMENTUM TILT ---")
    print("  (Tilt toward asset with better recent return)")

    mom_results = []
    for window in [15, 30, 60, 90]:
        for strength in [0.20, 0.50]:
            r = run_dynamic_allocation("momentum_tilt", {"momentum_window": window, "tilt_strength": strength})
            d = r["avg_sharpe"] - best_static["avg_sharpe"]
            mom_results.append({"momentum_window": window, "tilt_strength": strength, **r, "delta": round(d, 3)})
            print(f"  win={window:3d} str={strength:.1f}  avg={r['avg_sharpe']:.3f} min={r['min_sharpe']:.3f} Δ={d:+.3f}")

    best_mom = max(mom_results, key=lambda x: x["avg_sharpe"])
    results["momentum_tilt"] = mom_results
    print(f"\n  Best momentum: win={best_mom['momentum_window']} str={best_mom['tilt_strength']:.1f} avg={best_mom['avg_sharpe']:.3f}")
    print()

    # ── SUMMARY ───────────────────────────────────────────────────────────
    print("=" * 70)
    print("SUMMARY — R38: Dynamic BTC/ETH Asset Allocation")
    print("=" * 70)
    print()

    print(f"  Best static:    {best_static['label']:10s} avg={best_static['avg_sharpe']:.3f} min={best_static['min_sharpe']:.3f}")
    print(f"  Best IV tilt:   str={best_iv_tilt['tilt_strength']:.2f}      avg={best_iv_tilt['avg_sharpe']:.3f} min={best_iv_tilt['min_sharpe']:.3f} Δ={best_iv_tilt['delta']:+.3f}")
    print(f"  Best VRP tilt:  str={best_vrp_tilt['tilt_strength']:.2f}      avg={best_vrp_tilt['avg_sharpe']:.3f} min={best_vrp_tilt['min_sharpe']:.3f} Δ={best_vrp_tilt['delta']:+.3f}")
    print(f"  Best vol-par:   win={best_vp['rv_window']:3d}       avg={best_vp['avg_sharpe']:.3f} min={best_vp['min_sharpe']:.3f} Δ={best_vp['delta']:+.3f}")
    print(f"  Best momentum:  win={best_mom['momentum_window']:3d}       avg={best_mom['avg_sharpe']:.3f} min={best_mom['min_sharpe']:.3f} Δ={best_mom['delta']:+.3f}")

    # All dynamic methods
    all_dynamic = [best_iv_tilt, best_vrp_tilt, best_vp, best_mom]
    best_dynamic = max(all_dynamic, key=lambda x: x["avg_sharpe"])

    print()
    print("=" * 70)
    d_vs_static = best_dynamic["avg_sharpe"] - best_static["avg_sharpe"]
    if d_vs_static > 0.05:
        print(f"VERDICT: Dynamic allocation IMPROVES over static (Δ={d_vs_static:+.3f})")
    elif d_vs_static > -0.03:
        print(f"VERDICT: Dynamic allocation is MARGINAL (Δ={d_vs_static:+.3f})")
    else:
        print(f"VERDICT: Static allocation wins — dynamic destroys Sharpe (Δ={d_vs_static:+.3f})")
    print(f"  Confirms R12: {'static preferred' if d_vs_static <= 0.05 else 'dynamic helps for cross-asset'}")
    print("=" * 70)

    # Save
    out = ROOT / "artifacts" / "crypto_options" / "dynamic_asset_allocation.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump({
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "research_id": "R38",
            "results": results,
            "best_static": {"label": best_static["label"], "avg": best_static["avg_sharpe"]},
            "best_dynamic_delta": d_vs_static,
        }, f, indent=2)
    print(f"\nResults saved to: {out}")


if __name__ == "__main__":
    main()
