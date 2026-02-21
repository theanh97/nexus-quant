#!/usr/bin/env python3
"""
Position-Level Risk Management — R48
======================================

All overlay/filter approaches have failed (R7, R21, R26, R45).
But we haven't tested simple position-level risk limits:

Tests:
  A. Maximum portfolio drawdown trigger: if equity drops X% from peak,
     flatten all positions for N bars.
  B. Per-bar loss limit: if single-bar PnL < -X%, cut exposure next bar.
  C. Running Sharpe circuit breaker: if rolling Sharpe drops below X,
     reduce position size until it recovers.
  D. Volatility-scaled position sizing: instead of IV percentile,
     use recent realized vol to cap maximum loss exposure.

Hypothesis: These will likely hurt (like R7/R21/R26) since VRP is
always-on carry. But we should verify with data.
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

Z_ENTRY_SHORT = 1.0
Z_ENTRY_LONG = 2.0
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


def compute_skew_zscore(skew_series, idx, lookback=60):
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


def run_with_risk_limits(dd_trigger=None, dd_cooldown=10, bar_loss_limit=None,
                         rolling_sharpe_floor=None, rolling_window=60,
                         vol_cap_mult=None, vol_window=20):
    """Run ensemble with position-level risk management.

    dd_trigger: flatten all when equity drops this % from peak (e.g., 0.02 = 2%)
    bar_loss_limit: if single bar PnL < -this %, reduce position next bar
    rolling_sharpe_floor: if rolling Sharpe < this, scale down positions
    vol_cap_mult: cap position size so max 1-day loss < vol_cap_mult × daily_vol
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
        peak_equity = 1.0
        vrp_w = {"BTC": 0.0, "ETH": 0.0}
        skew_w = {"BTC": 0.0, "ETH": 0.0}
        skew_pos = {"BTC": 0.0, "ETH": 0.0}
        equity_curve = [1.0]
        returns_list = []
        last_skew_rebal = {"BTC": 0, "ETH": 0}
        dd_cooldown_counter = 0
        risk_scale = 1.0

        for idx in range(1, n):
            prev_equity = equity

            # Risk scale adjustments
            risk_scale = 1.0

            # A: Drawdown trigger
            if dd_trigger is not None:
                dd = (equity - peak_equity) / peak_equity if peak_equity > 0 else 0
                if dd < -dd_trigger:
                    dd_cooldown_counter = dd_cooldown
                if dd_cooldown_counter > 0:
                    risk_scale = 0.0  # flat
                    dd_cooldown_counter -= 1

            # B: Per-bar loss limit
            if bar_loss_limit is not None and len(returns_list) > 0:
                last_ret = returns_list[-1]
                if last_ret < -bar_loss_limit:
                    risk_scale *= 0.5

            # C: Rolling Sharpe floor
            if rolling_sharpe_floor is not None and len(returns_list) >= rolling_window:
                recent = returns_list[-rolling_window:]
                avg_r = sum(recent) / len(recent)
                var_r = sum((r - avg_r)**2 for r in recent) / len(recent)
                std_r = var_r ** 0.5
                if std_r > 1e-10:
                    rolling_sr = (avg_r / std_r) * math.sqrt(BARS_PER_YEAR)
                    if rolling_sr < rolling_sharpe_floor:
                        ratio = max(rolling_sr / rolling_sharpe_floor, 0.25)
                        risk_scale *= ratio

            # D: Vol-cap
            if vol_cap_mult is not None and len(returns_list) >= vol_window:
                recent = returns_list[-vol_window:]
                var_r = sum(r**2 for r in recent) / len(recent)
                daily_vol = var_r ** 0.5
                if daily_vol > 1e-10:
                    max_scale = vol_cap_mult / (daily_vol * math.sqrt(BARS_PER_YEAR))
                    risk_scale = min(risk_scale, max(max_scale, 0.25))

            # Apply risk scale
            scaled_vrp_w = {s: vrp_w[s] * risk_scale for s in vrp_w}
            scaled_skew_w = {s: skew_w[s] * risk_scale for s in skew_w}

            total_pnl = 0.0
            for sym in ["BTC", "ETH"]:
                aw = 0.50
                w_v = scaled_vrp_w.get(sym, 0.0)
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

                w_s = scaled_skew_w.get(sym, 0.0)
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
            peak_equity = max(peak_equity, equity)

            # Normal rebalancing
            rebal_happened = False
            for sym in ["BTC", "ETH"]:
                if vrp_strats[sym].should_rebalance(dataset, idx):
                    rebal_happened = True
                    tv = vrp_strats[sym].target_weights(dataset, idx, {sym: vrp_w.get(sym, 0.0)})
                    vrp_w[sym] = tv.get(sym, 0.0)

            for sym in ["BTC", "ETH"]:
                if idx - last_skew_rebal[sym] < 5:
                    continue
                lb = 60 if sym == "BTC" else 90
                skew_series = dataset.features.get("skew_25d", {}).get(sym, [])
                z = compute_skew_zscore(skew_series, idx, lookback=lb)
                if z is None:
                    continue

                per_sym_lev = 0.5
                current_pos = skew_pos[sym]
                new_pos = current_pos

                if current_pos == 0.0:
                    if z >= Z_ENTRY_SHORT:
                        new_pos = -per_sym_lev
                    elif z <= -Z_ENTRY_LONG:
                        new_pos = per_sym_lev
                else:
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
    print("POSITION-LEVEL RISK MANAGEMENT — R48")
    print("=" * 70)
    print()

    baseline = run_with_risk_limits()
    print(f"  Baseline: avg={baseline['avg_sharpe']:.3f} min={baseline['min_sharpe']:.3f}")
    print(f"  Yearly: {baseline['yearly']}")
    print()

    all_results = {"baseline": baseline}

    # ── A. Drawdown Trigger ──────────────────────────────────────────────
    print("--- A. DRAWDOWN TRIGGER (flatten for N bars after DD) ---")
    print(f"  {'DD%':>5s} {'cool':>5s}  {'avg':>7s} {'min':>7s} {'Δ':>7s}")

    dd_configs = [
        (0.005, 5), (0.005, 10), (0.005, 20),
        (0.01, 5), (0.01, 10), (0.01, 20),
        (0.02, 5), (0.02, 10), (0.02, 20),
        (0.03, 10), (0.05, 10),
    ]

    dd_results = []
    for dd, cool in dd_configs:
        r = run_with_risk_limits(dd_trigger=dd, dd_cooldown=cool)
        d = r["avg_sharpe"] - baseline["avg_sharpe"]
        dd_results.append({"dd_pct": dd, "cooldown": cool, **r, "delta": round(d, 3)})
        print(f"  {dd*100:5.1f} {cool:5d}  {r['avg_sharpe']:7.3f} {r['min_sharpe']:7.3f} {d:+7.3f}")

    all_results["dd_results"] = dd_results
    print()

    # ── B. Per-Bar Loss Limit ────────────────────────────────────────────
    print("--- B. PER-BAR LOSS LIMIT (reduce after large single-bar loss) ---")
    print(f"  {'limit%':>7s}  {'avg':>7s} {'min':>7s} {'Δ':>7s}")

    bar_limits = [0.001, 0.002, 0.005, 0.01, 0.02]
    bar_results = []
    for lim in bar_limits:
        r = run_with_risk_limits(bar_loss_limit=lim)
        d = r["avg_sharpe"] - baseline["avg_sharpe"]
        bar_results.append({"limit_pct": lim, **r, "delta": round(d, 3)})
        print(f"  {lim*100:7.2f}  {r['avg_sharpe']:7.3f} {r['min_sharpe']:7.3f} {d:+7.3f}")

    all_results["bar_results"] = bar_results
    print()

    # ── C. Rolling Sharpe Floor ──────────────────────────────────────────
    print("--- C. ROLLING SHARPE FLOOR (scale down when rolling Sharpe is low) ---")
    print(f"  {'floor':>6s} {'win':>4s}  {'avg':>7s} {'min':>7s} {'Δ':>7s}")

    sharpe_configs = [
        (0.5, 30), (0.5, 60), (1.0, 30), (1.0, 60),
        (2.0, 30), (2.0, 60), (3.0, 60),
    ]

    sharpe_results = []
    for floor, win in sharpe_configs:
        r = run_with_risk_limits(rolling_sharpe_floor=floor, rolling_window=win)
        d = r["avg_sharpe"] - baseline["avg_sharpe"]
        sharpe_results.append({"floor": floor, "window": win, **r, "delta": round(d, 3)})
        print(f"  {floor:6.1f} {win:4d}  {r['avg_sharpe']:7.3f} {r['min_sharpe']:7.3f} {d:+7.3f}")

    all_results["sharpe_results"] = sharpe_results
    print()

    # ── D. Vol-Cap Sizing ────────────────────────────────────────────────
    print("--- D. VOL-CAP SIZING (cap position so max daily loss < X × vol) ---")
    print(f"  {'mult':>5s} {'win':>4s}  {'avg':>7s} {'min':>7s} {'Δ':>7s}")

    vol_configs = [
        (0.5, 10), (0.5, 20), (1.0, 10), (1.0, 20),
        (2.0, 20), (3.0, 20),
    ]

    vol_results = []
    for mult, win in vol_configs:
        r = run_with_risk_limits(vol_cap_mult=mult, vol_window=win)
        d = r["avg_sharpe"] - baseline["avg_sharpe"]
        vol_results.append({"mult": mult, "window": win, **r, "delta": round(d, 3)})
        print(f"  {mult:5.1f} {win:4d}  {r['avg_sharpe']:7.3f} {r['min_sharpe']:7.3f} {d:+7.3f}")

    all_results["vol_results"] = vol_results
    print()

    # ── SUMMARY ──────────────────────────────────────────────────────────
    print("=" * 70)
    print("SUMMARY — R48: Position-Level Risk Management")
    print("=" * 70)
    print()

    best_dd = max(dd_results, key=lambda x: x["avg_sharpe"])
    best_bar = max(bar_results, key=lambda x: x["avg_sharpe"])
    best_sr = max(sharpe_results, key=lambda x: x["avg_sharpe"])
    best_vol = max(vol_results, key=lambda x: x["avg_sharpe"])

    print(f"  Baseline:            avg={baseline['avg_sharpe']:.3f}")
    print(f"  Best DD trigger:     avg={best_dd['avg_sharpe']:.3f} Δ={best_dd['delta']:+.3f}")
    print(f"  Best bar limit:      avg={best_bar['avg_sharpe']:.3f} Δ={best_bar['delta']:+.3f}")
    print(f"  Best Sharpe floor:   avg={best_sr['avg_sharpe']:.3f} Δ={best_sr['delta']:+.3f}")
    print(f"  Best vol-cap:        avg={best_vol['avg_sharpe']:.3f} Δ={best_vol['delta']:+.3f}")
    print()

    any_improve = any(x["delta"] > 0.03 for x in [best_dd, best_bar, best_sr, best_vol])
    all_deltas = [best_dd["delta"], best_bar["delta"], best_sr["delta"], best_vol["delta"]]

    n_configs = len(dd_configs) + len(bar_limits) + len(sharpe_configs) + len(vol_configs)
    n_improve = sum(1 for r in dd_results + bar_results + sharpe_results + vol_results if r["delta"] > 0)

    print("=" * 70)
    if any_improve:
        print("VERDICT: Some risk management HELPS")
    else:
        print(f"VERDICT: Position risk management DOES NOT HELP")
        print(f"  Tested {n_configs} configs, {n_improve}/{n_configs} improve (marginal)")
        print(f"  Confirms: carry trades must be always fully invested (R7, R21, R26)")
        print(f"  VRP MDD -1.5% — too small for stop-losses to add value")
    print("=" * 70)

    # Save
    out = ROOT / "artifacts" / "crypto_options" / "position_risk_management.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump({
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "research_id": "R48",
            **all_results,
        }, f, indent=2)
    print(f"\nResults saved to: {out}")


if __name__ == "__main__":
    main()
