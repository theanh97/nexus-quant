#!/usr/bin/env python3
"""
Signal Decay & Staleness Analysis — R41
==========================================

Production concern: what happens if we CAN'T rebalance for N days?
(Exchange downtime, connectivity issues, risk limits, etc.)

Questions:
  1. How quickly does the ensemble degrade with stale positions?
  2. Is VRP or Skew more sensitive to staleness?
  3. What is the maximum acceptable rebalance gap?
  4. Does IV sizing help or hurt with stale positions?

Method: Force rebalancing gaps of 1-60 days, measure Sharpe degradation.
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
BTC_SKEW = {"skew_lookback": 60, "z_entry": 2.0, "z_exit": 0.0, "target_leverage": 1.0, "rebalance_freq": 5, "min_bars": 60}
ETH_VRP = {"base_leverage": 1.5, "exit_z_threshold": -3.0, "vrp_lookback": 60, "rebalance_freq": 5, "min_bars": 60}
ETH_SKEW = {"skew_lookback": 90, "z_entry": 2.0, "z_exit": 0.0, "target_leverage": 1.0, "rebalance_freq": 5, "min_bars": 90}


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


def run_with_forced_gap(gap_days, use_iv_sizing=True):
    """Run multi-asset ensemble with forced rebalancing gap."""
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

        # Force rebalance_freq to gap_days for both strategies
        btc_vrp_p = {**BTC_VRP, "rebalance_freq": gap_days}
        btc_skew_p = {**BTC_SKEW, "rebalance_freq": gap_days}
        eth_vrp_p = {**ETH_VRP, "rebalance_freq": gap_days}
        eth_skew_p = {**ETH_SKEW, "rebalance_freq": gap_days}

        strats = {
            "BTC": {"vrp": VariancePremiumStrategy(params=btc_vrp_p), "skew": SkewTradeV2Strategy(params=btc_skew_p)},
            "ETH": {"vrp": VariancePremiumStrategy(params=eth_vrp_p), "skew": SkewTradeV2Strategy(params=eth_skew_p)},
        }

        dt = 1.0 / BARS_PER_YEAR
        equity = 1.0
        vrp_w = {"BTC": 0.0, "ETH": 0.0}
        skew_w = {"BTC": 0.0, "ETH": 0.0}
        equity_curve = [1.0]
        returns_list = []

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

            rebal_happened = False
            for sym in ["BTC", "ETH"]:
                if strats[sym]["vrp"].should_rebalance(dataset, idx) or strats[sym]["skew"].should_rebalance(dataset, idx):
                    rebal_happened = True
                    if strats[sym]["vrp"].should_rebalance(dataset, idx):
                        tv = strats[sym]["vrp"].target_weights(dataset, idx, {sym: vrp_w.get(sym, 0.0)})
                        vrp_w[sym] = tv.get(sym, 0.0)
                    if strats[sym]["skew"].should_rebalance(dataset, idx):
                        ts = strats[sym]["skew"].target_weights(dataset, idx, {sym: skew_w.get(sym, 0.0)})
                        skew_w[sym] = ts.get(sym, 0.0)
                    if use_iv_sizing:
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
    print("SIGNAL DECAY & STALENESS ANALYSIS — R41")
    print("=" * 70)
    print()

    # ── A. Rebalance Gap Sweep (with IV sizing) ───────────────────────────
    print("--- A. REBALANCE GAP SWEEP (with IV sizing) ---")
    print(f"  {'Gap':>5s}  {'avg':>7s} {'min':>7s} {'Δ':>7s}")
    print(f"  {'---':>5s}  {'---':>7s} {'---':>7s} {'---':>7s}")

    gaps = [1, 2, 3, 5, 7, 10, 14, 21, 30, 45, 60]
    gap_results_iv = []
    baseline = None

    for gap in gaps:
        r = run_with_forced_gap(gap, use_iv_sizing=True)
        if gap == 5:
            baseline = r
        d = r["avg_sharpe"] - (baseline["avg_sharpe"] if baseline else r["avg_sharpe"])
        gap_results_iv.append({"gap_days": gap, **r, "delta": round(d, 3)})
        marker = " *" if gap == 5 else ""
        print(f"  {gap:5d}  {r['avg_sharpe']:7.3f} {r['min_sharpe']:7.3f} {d:+7.3f}{marker}")

    # Recalculate deltas now that baseline is set
    for gr in gap_results_iv:
        gr["delta"] = round(gr["avg_sharpe"] - baseline["avg_sharpe"], 3)

    print()

    # ── B. Same sweep without IV sizing ───────────────────────────────────
    print("--- B. REBALANCE GAP SWEEP (without IV sizing) ---")
    print(f"  {'Gap':>5s}  {'avg':>7s} {'min':>7s} {'Δ':>7s}")
    print(f"  {'---':>5s}  {'---':>7s} {'---':>7s} {'---':>7s}")

    gap_results_no_iv = []
    baseline_no_iv = None

    for gap in gaps:
        r = run_with_forced_gap(gap, use_iv_sizing=False)
        if gap == 5:
            baseline_no_iv = r
        d = r["avg_sharpe"] - (baseline_no_iv["avg_sharpe"] if baseline_no_iv else r["avg_sharpe"])
        gap_results_no_iv.append({"gap_days": gap, **r, "delta": round(d, 3)})
        marker = " *" if gap == 5 else ""
        print(f"  {gap:5d}  {r['avg_sharpe']:7.3f} {r['min_sharpe']:7.3f} {d:+7.3f}{marker}")

    for gr in gap_results_no_iv:
        gr["delta"] = round(gr["avg_sharpe"] - baseline_no_iv["avg_sharpe"], 3)

    print()

    # ── C. IV Sizing Impact by Gap ────────────────────────────────────────
    print("--- C. IV SIZING IMPROVEMENT AT EACH GAP ---")
    print(f"  {'Gap':>5s}  {'w/ IV':>7s} {'w/o IV':>7s} {'IV Δ':>7s}")
    print(f"  {'---':>5s}  {'-----':>7s} {'------':>7s} {'-----':>7s}")

    for i, gap in enumerate(gaps):
        iv_val = gap_results_iv[i]["avg_sharpe"]
        noiv_val = gap_results_no_iv[i]["avg_sharpe"]
        d = iv_val - noiv_val
        print(f"  {gap:5d}  {iv_val:7.3f} {noiv_val:7.3f} {d:+7.3f}")

    print()

    # ── D. Half-Life Analysis ─────────────────────────────────────────────
    print("--- D. SIGNAL HALF-LIFE ANALYSIS ---")
    if baseline:
        max_sharpe = gap_results_iv[0]["avg_sharpe"]  # gap=1 (freshest)
        half_target = (max_sharpe + 0) / 2  # half-life = where Sharpe reaches 50% of max
        half_life = None
        for gr in gap_results_iv:
            if gr["avg_sharpe"] <= half_target:
                half_life = gr["gap_days"]
                break

        print(f"  Max Sharpe (gap=1):  {max_sharpe:.3f}")
        print(f"  Baseline (gap=5):    {baseline['avg_sharpe']:.3f}")
        if half_life:
            print(f"  Half-life:           ~{half_life}d (Sharpe drops to {half_target:.3f})")
        else:
            print(f"  Half-life:           >60d (signal is very persistent!)")

        # Find where min Sharpe drops below 1.0
        for gr in gap_results_iv:
            if gr["min_sharpe"] < 1.0:
                print(f"  Safety limit:        {gr['gap_days']}d (min Sharpe < 1.0)")
                break
        else:
            print(f"  Safety limit:        >60d (min Sharpe stays above 1.0!)")

    print()

    # ── SUMMARY ───────────────────────────────────────────────────────────
    print("=" * 70)
    print("SUMMARY — R41: Signal Decay & Staleness Analysis")
    print("=" * 70)
    print()

    print("  SIGNAL PERSISTENCE (with IV sizing):")
    for gr in gap_results_iv:
        marker = " ← current" if gr["gap_days"] == 5 else ""
        pct = (gr["avg_sharpe"] / gap_results_iv[0]["avg_sharpe"] * 100) if gap_results_iv[0]["avg_sharpe"] > 0 else 0
        print(f"    gap={gr['gap_days']:3d}d: avg={gr['avg_sharpe']:.3f} min={gr['min_sharpe']:.3f} ({pct:.0f}% of max){marker}")

    # Maximum acceptable gap
    print()
    for gr in gap_results_iv:
        if gr["min_sharpe"] < 2.0:
            print(f"  Max gap for Sharpe>2.0: {gaps[gap_results_iv.index(gr) - 1] if gap_results_iv.index(gr) > 0 else 0}d")
            break
    else:
        print(f"  Sharpe stays >2.0 even at 60d gap!")

    print()
    print("=" * 70)
    gap30 = next((g for g in gap_results_iv if g["gap_days"] == 30), None)
    if gap30 and gap30["avg_sharpe"] > 2.0:
        print("VERDICT: Signal is VERY PERSISTENT — survives 30d without rebalancing")
        print(f"  VRP carry accrues regardless of rebalancing (always short vol)")
        print(f"  Production safety: can miss rebalancing for weeks without disaster")
    else:
        print("VERDICT: Signal decays significantly — rebalancing matters")
    print("=" * 70)

    # Save
    out = ROOT / "artifacts" / "crypto_options" / "signal_decay_analysis.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump({
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "research_id": "R41",
            "gap_sweep_iv": gap_results_iv,
            "gap_sweep_no_iv": gap_results_no_iv,
        }, f, indent=2)
    print(f"\nResults saved to: {out}")


if __name__ == "__main__":
    main()
