#!/usr/bin/env python3
"""
Cross-Strategy Signal Interaction — R45
=========================================

VRP and Skew MR are essentially uncorrelated (r=0.014), but there may
be temporal dependencies. Does VRP z-score predict Skew MR success?

Tests:
  A. VRP z-gating on Skew MR: Only enter Skew trades when VRP z is in
     a specific regime (e.g., only sell skew when VRP z > 0 = vol is expensive).
  B. Conditional Skew sizing by VRP z: Scale Skew position by VRP z magnitude.
  C. Temporal lag: Does lagging Skew signal by 1-3 days improve entry timing?
  D. Signal agreement bonus: When both VRP and Skew z are extreme, upweight.

All tests use the full multi-asset ensemble with R42 asymmetric skew.
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

Z_ENTRY_LONG = 2.0
Z_ENTRY_SHORT = 1.0
Z_EXIT_LONG = 0.5
Z_EXIT_SHORT = 0.0


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


def compute_zscore(series, idx, lookback):
    if idx < lookback or not series:
        return None
    start = max(0, idx - lookback)
    window = [v for v in series[start:idx] if v is not None]
    if len(window) < 10:
        return None
    current = series[idx] if idx < len(series) and series[idx] is not None else None
    if current is None:
        return None
    mean = sum(window) / len(window)
    var = sum((v - mean) ** 2 for v in window) / len(window)
    std = var ** 0.5
    if std < 1e-10:
        return 0.0
    return (current - mean) / std


def compute_vrp_zscore(dataset, sym, idx, lookback=30):
    """Compute VRP z-score (IV - RV spread z-score)."""
    ivs = dataset.features.get("iv_atm", {}).get(sym, [])
    closes = dataset.perp_close.get(sym, [])
    if idx < lookback or not ivs or not closes:
        return None

    vrp_vals = []
    for i in range(max(1, idx - lookback), idx + 1):
        if i < len(ivs) and i < len(closes) and i > 0:
            iv = ivs[i]
            if iv is None or closes[i-1] <= 0 or closes[i] <= 0:
                continue
            lr = math.log(closes[i] / closes[i-1])
            rv = abs(lr) * math.sqrt(BARS_PER_YEAR)
            vrp_vals.append(iv - rv)

    if len(vrp_vals) < 10:
        return None
    current = vrp_vals[-1]
    mean = sum(vrp_vals) / len(vrp_vals)
    var = sum((v - mean) ** 2 for v in vrp_vals) / len(vrp_vals)
    std = var ** 0.5
    if std < 1e-10:
        return 0.0
    return (current - mean) / std


def run_with_gating(mode="none", gate_threshold=0.0, skew_scale_by_vrp=False,
                    lag_days=0, agreement_bonus=1.0):
    """Run ensemble with cross-strategy signal gating.

    mode:
      "none" = baseline (no gating)
      "vrp_gate_sell" = only sell skew when VRP z > gate_threshold
      "vrp_gate_buy" = only buy skew when VRP z < -gate_threshold
      "vrp_gate_both" = both above gates active
      "vrp_scale" = scale Skew position by VRP z magnitude
    """
    sharpes = []
    yearly = {}

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
        skew_pos = {"BTC": 0.0, "ETH": 0.0}
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

            # VRP rebalance
            rebal_happened = False
            for sym in ["BTC", "ETH"]:
                if vrp_strats[sym].should_rebalance(dataset, idx):
                    rebal_happened = True
                    tv = vrp_strats[sym].target_weights(dataset, idx, {sym: vrp_w.get(sym, 0.0)})
                    vrp_w[sym] = tv.get(sym, 0.0)

            # Skew MR with gating
            for sym in ["BTC", "ETH"]:
                if idx - last_skew_rebal[sym] < 5:
                    continue
                lb = 60 if sym == "BTC" else 90
                skew_series = dataset.features.get("skew_25d", {}).get(sym, [])

                # Apply lag
                skew_idx = max(0, idx - lag_days)
                z = compute_zscore(skew_series, skew_idx, lookback=lb)
                if z is None:
                    continue

                # Compute VRP z for gating
                vrp_lb = 30 if sym == "BTC" else 60
                vrp_z = compute_vrp_zscore(dataset, sym, idx, lookback=vrp_lb)

                per_sym_lev = 0.5
                current_pos = skew_pos[sym]
                new_pos = current_pos

                # Entry logic with gating
                if current_pos == 0.0:
                    can_sell = True
                    can_buy = True

                    if mode in ("vrp_gate_sell", "vrp_gate_both"):
                        if vrp_z is None or vrp_z <= gate_threshold:
                            can_sell = False
                    if mode in ("vrp_gate_buy", "vrp_gate_both"):
                        if vrp_z is None or vrp_z >= -gate_threshold:
                            can_buy = False

                    if z >= Z_ENTRY_SHORT and can_sell:
                        new_pos = -per_sym_lev
                    elif z <= -Z_ENTRY_LONG and can_buy:
                        new_pos = per_sym_lev

                    # Agreement bonus: if both signals agree on direction
                    if agreement_bonus != 1.0 and vrp_z is not None:
                        if new_pos < 0 and vrp_z > 0:  # sell skew + vol is expensive
                            new_pos *= agreement_bonus
                        elif new_pos > 0 and vrp_z < 0:  # buy skew + vol is cheap
                            new_pos *= agreement_bonus

                    # VRP-scaled sizing
                    if skew_scale_by_vrp and vrp_z is not None and new_pos != 0:
                        scale = min(max(0.5 + 0.25 * abs(vrp_z), 0.5), 2.0)
                        new_pos *= scale

                else:
                    # Exit logic unchanged
                    if current_pos > 0:
                        if z >= Z_EXIT_LONG:
                            new_pos = 0.0
                    else:
                        if z <= -Z_EXIT_SHORT:
                            new_pos = 0.0

                if new_pos != current_pos:
                    rebal_happened = True
                    last_skew_rebal[sym] = idx

                skew_pos[sym] = new_pos
                skew_w[sym] = new_pos

            if rebal_happened:
                for sym in ["BTC", "ETH"]:
                    ivs = dataset.features.get("iv_atm", {}).get(sym, [])
                    pct = iv_percentile(ivs, idx)
                    if pct is not None:
                        sc = iv_sizing_scale(pct)
                        vrp_w[sym] *= sc
                        skew_w[sym] *= sc
                bd = COSTS.cost(equity=equity, turnover=0.05)
                equity -= float(bd.get("cost", 0.0))
                equity = max(equity, 0.0)

            equity_curve.append(equity)
            bar_ret = (equity / prev_equity) - 1.0 if prev_equity > 0 else 0.0
            returns_list.append(bar_ret)

        m = compute_metrics(equity_curve, returns_list, BARS_PER_YEAR)
        sharpes.append(m["sharpe"])
        yearly[str(yr)] = round(m["sharpe"], 3)

    avg = sum(sharpes) / len(sharpes)
    mn = min(sharpes)
    return {"avg_sharpe": round(avg, 3), "min_sharpe": round(mn, 3), "yearly": yearly}


def main():
    print("=" * 70)
    print("CROSS-STRATEGY SIGNAL INTERACTION — R45")
    print("=" * 70)
    print()

    # ── Baseline ─────────────────────────────────────────────────────────
    baseline = run_with_gating(mode="none")
    print(f"  Baseline (no gating): avg={baseline['avg_sharpe']:.3f} min={baseline['min_sharpe']:.3f}")
    print(f"  Yearly: {baseline['yearly']}")
    print()

    all_results = {"baseline": baseline}

    # ── A. VRP Z-Score Gating on Skew MR ─────────────────────────────────
    print("--- A. VRP Z-GATING ON SKEW MR ENTRY ---")
    print(f"  {'Gate Mode':30s}  {'thresh':>6s}  {'avg':>7s} {'min':>7s} {'Δ':>7s}")
    print(f"  {'-'*30}  {'-'*6}  {'-'*7} {'-'*7} {'-'*7}")

    gate_configs = [
        ("vrp_gate_sell", 0.0),
        ("vrp_gate_sell", 0.5),
        ("vrp_gate_sell", 1.0),
        ("vrp_gate_buy", 0.0),
        ("vrp_gate_buy", 0.5),
        ("vrp_gate_buy", 1.0),
        ("vrp_gate_both", 0.0),
        ("vrp_gate_both", 0.5),
        ("vrp_gate_both", 1.0),
    ]

    gate_results = []
    for mode, thresh in gate_configs:
        r = run_with_gating(mode=mode, gate_threshold=thresh)
        d = r["avg_sharpe"] - baseline["avg_sharpe"]
        gate_results.append({"mode": mode, "threshold": thresh, **r, "delta": round(d, 3)})
        label = f"{mode} (t={thresh})"
        print(f"  {label:30s}  {thresh:6.1f}  {r['avg_sharpe']:7.3f} {r['min_sharpe']:7.3f} {d:+7.3f}")

    all_results["gate_results"] = gate_results
    print()

    # ── B. VRP Z-Based Skew Sizing ───────────────────────────────────────
    print("--- B. VRP Z-BASED SKEW SIZING ---")
    r_vrp_scale = run_with_gating(mode="none", skew_scale_by_vrp=True)
    d = r_vrp_scale["avg_sharpe"] - baseline["avg_sharpe"]
    print(f"  VRP-scaled Skew sizing: avg={r_vrp_scale['avg_sharpe']:.3f} min={r_vrp_scale['min_sharpe']:.3f} Δ={d:+.3f}")
    all_results["vrp_scaled_sizing"] = {**r_vrp_scale, "delta": round(d, 3)}
    print()

    # ── C. Temporal Lag ──────────────────────────────────────────────────
    print("--- C. TEMPORAL LAG (Skew signal delayed) ---")
    print(f"  {'Lag':>5s}  {'avg':>7s} {'min':>7s} {'Δ':>7s}")
    print(f"  {'---':>5s}  {'---':>7s} {'---':>7s} {'---':>7s}")

    lag_results = []
    for lag in [0, 1, 2, 3, 5, 7]:
        r = run_with_gating(mode="none", lag_days=lag)
        d = r["avg_sharpe"] - baseline["avg_sharpe"]
        lag_results.append({"lag_days": lag, **r, "delta": round(d, 3)})
        marker = " *" if lag == 0 else ""
        print(f"  {lag:5d}  {r['avg_sharpe']:7.3f} {r['min_sharpe']:7.3f} {d:+7.3f}{marker}")

    all_results["lag_results"] = lag_results
    print()

    # ── D. Signal Agreement Bonus ────────────────────────────────────────
    print("--- D. SIGNAL AGREEMENT BONUS ---")
    print("  (Scale Skew position up when VRP z agrees on direction)")
    print(f"  {'Bonus':>7s}  {'avg':>7s} {'min':>7s} {'Δ':>7s}")
    print(f"  {'-----':>7s}  {'---':>7s} {'---':>7s} {'---':>7s}")

    bonus_results = []
    for bonus in [1.0, 1.25, 1.5, 1.75, 2.0]:
        r = run_with_gating(mode="none", agreement_bonus=bonus)
        d = r["avg_sharpe"] - baseline["avg_sharpe"]
        bonus_results.append({"bonus": bonus, **r, "delta": round(d, 3)})
        marker = " *" if bonus == 1.0 else ""
        print(f"  {bonus:7.2f}  {r['avg_sharpe']:7.3f} {r['min_sharpe']:7.3f} {d:+7.3f}{marker}")

    all_results["bonus_results"] = bonus_results
    print()

    # ── SUMMARY ──────────────────────────────────────────────────────────
    print("=" * 70)
    print("SUMMARY — R45: Cross-Strategy Signal Interaction")
    print("=" * 70)
    print()

    best_gate = max(gate_results, key=lambda x: x["avg_sharpe"])
    best_lag = max(lag_results, key=lambda x: x["avg_sharpe"])
    best_bonus = max(bonus_results, key=lambda x: x["avg_sharpe"])

    print(f"  Baseline:             avg={baseline['avg_sharpe']:.3f}")
    print(f"  Best gate:            avg={best_gate['avg_sharpe']:.3f} Δ={best_gate['delta']:+.3f} "
          f"({best_gate['mode']} t={best_gate['threshold']})")
    print(f"  VRP-scaled sizing:    avg={r_vrp_scale['avg_sharpe']:.3f} Δ={r_vrp_scale['avg_sharpe']-baseline['avg_sharpe']:+.3f}")
    print(f"  Best lag:             avg={best_lag['avg_sharpe']:.3f} Δ={best_lag['delta']:+.3f} "
          f"(lag={best_lag['lag_days']}d)")
    print(f"  Best agreement bonus: avg={best_bonus['avg_sharpe']:.3f} Δ={best_bonus['delta']:+.3f} "
          f"(bonus={best_bonus['bonus']}x)")
    print()

    any_improvement = (best_gate["delta"] > 0.05 or best_lag["delta"] > 0.05 or
                       best_bonus["delta"] > 0.05 or
                       r_vrp_scale["avg_sharpe"] - baseline["avg_sharpe"] > 0.05)

    print("=" * 70)
    if any_improvement:
        print("VERDICT: Cross-strategy interaction IMPROVES ensemble")
        if best_gate["delta"] > 0.05:
            print(f"  VRP gating adds {best_gate['delta']:+.3f}")
    else:
        print("VERDICT: Cross-strategy interaction is MARGINAL — signals are independent")
        print("  VRP and Skew MR operate on different mechanisms (carry vs mean-reversion)")
        print("  Combining at portfolio level (static weights) is already optimal")
    print("=" * 70)

    # Save
    out = ROOT / "artifacts" / "crypto_options" / "cross_strategy_signal_gating.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump({
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "research_id": "R45",
            **all_results,
        }, f, indent=2)
    print(f"\nResults saved to: {out}")


if __name__ == "__main__":
    main()
