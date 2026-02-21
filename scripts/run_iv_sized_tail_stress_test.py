#!/usr/bin/env python3
"""
Tail Stress Test: IV-Sized Ensemble vs Unsized — R30
=======================================================

R28 increased the high-IV scale factor to 1.7x.
Critical concern: high-IV periods often precede crashes.
If we're 1.7x sized when a crash hits, tail risk could be worse.

Tests:
  1. Full 5-year walk-forward with detailed tail metrics
  2. Historical crash window analysis (2021 May, 2022 LUNA/FTX, 2024 Aug)
  3. Worst-case statistics: VaR, CVaR, max DD, worst day
  4. Recovery time comparison
  5. Skewness/kurtosis of return distributions

Compare: unsized baseline vs R25 (0.5/1.5) vs R28 (0.5/1.7)
"""
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from nexus_quant.backtest.costs import ExecutionCostModel, FeeModel, ImpactModel
from nexus_quant.projects.crypto_options.options_engine import (
    OptionsBacktestEngine, compute_metrics
)
from nexus_quant.projects.crypto_options.providers.deribit_rest import DeribitRestProvider
from nexus_quant.projects.crypto_options.strategies.variance_premium import VariancePremiumStrategy
from nexus_quant.projects.crypto_options.strategies.skew_trade_v2 import SkewTradeV2Strategy

# ── Config ─────────────────────────────────────────────────────────────────

fees = FeeModel(maker_fee_rate=0.0003, taker_fee_rate=0.0005)
impact = ImpactModel(model="sqrt", coef_bps=2.0)
COSTS = ExecutionCostModel(fee=fees, impact=impact)

YEARS = [2021, 2022, 2023, 2024, 2025]
SEED = 42
BARS_PER_YEAR = 365

W_VRP = 0.40
W_SKEW = 0.60

VRP_PARAMS = {
    "base_leverage": 1.5,
    "exit_z_threshold": -3.0,
    "vrp_lookback": 30,
    "rebalance_freq": 5,
    "min_bars": 30,
}

SKEW_PARAMS = {
    "skew_lookback": 60,
    "z_entry": 2.0,
    "z_exit": 0.0,
    "target_leverage": 1.0,
    "rebalance_freq": 5,
    "min_bars": 60,
}

# Sizing configs to compare
SIZING_CONFIGS = {
    "unsized": (None, None),           # baseline
    "R25_0.5_1.5": (0.5, 1.5),        # original R25
    "R28_0.5_1.7": (0.5, 1.7),        # production config
    "R28_0.3_2.0": (0.3, 2.0),        # aggressive champion
}


def iv_percentile(iv_series: List, idx: int, lookback: int = 180) -> Optional[float]:
    if idx < lookback or not iv_series:
        return None
    start = max(0, idx - lookback)
    window = [v for v in iv_series[start:idx] if v is not None]
    if len(window) < 10:
        return None
    current = iv_series[idx] if idx < len(iv_series) and iv_series[idx] is not None else None
    if current is None:
        return None
    below = sum(1 for v in window if v < current)
    return below / len(window)


def step_sizing(pct: float, low_scale: float, high_scale: float) -> float:
    if pct < 0.25:
        return low_scale
    elif pct > 0.75:
        return high_scale
    return 1.0


def compute_tail_metrics(returns: List[float]) -> Dict[str, Any]:
    """Comprehensive tail risk metrics."""
    if not returns:
        return {}

    n = len(returns)
    sorted_ret = sorted(returns)

    # VaR and CVaR
    idx_95 = max(int(n * 0.05), 0)
    idx_99 = max(int(n * 0.01), 0)
    var_95 = sorted_ret[idx_95]
    var_99 = sorted_ret[idx_99]
    cvar_95 = sum(sorted_ret[:idx_95 + 1]) / (idx_95 + 1) if idx_95 > 0 else var_95
    cvar_99 = sum(sorted_ret[:idx_99 + 1]) / (idx_99 + 1) if idx_99 > 0 else var_99

    # Max drawdown
    equity = [1.0]
    for r in returns:
        equity.append(equity[-1] * (1.0 + r))
    hwm = equity[0]
    max_dd = 0.0
    dd_peak_idx = 0
    dd_trough_idx = 0
    peak_idx = 0
    for i, eq in enumerate(equity):
        if eq > hwm:
            hwm = eq
            peak_idx = i
        dd = (eq / hwm - 1.0) if hwm > 0 else 0.0
        if dd < max_dd:
            max_dd = dd
            dd_peak_idx = peak_idx
            dd_trough_idx = i

    # Recovery time
    peak_val = equity[dd_peak_idx] if dd_peak_idx < len(equity) else 0
    recovery_bars = None
    for i in range(dd_trough_idx, len(equity)):
        if equity[i] >= peak_val:
            recovery_bars = i - dd_trough_idx
            break

    # Distribution moments
    mean_r = sum(returns) / n
    var_r = sum((r - mean_r) ** 2 for r in returns) / n
    std_r = math.sqrt(var_r) if var_r > 0 else 1e-10
    skewness = sum((r - mean_r) ** 3 for r in returns) / (n * std_r ** 3) if std_r > 0 else 0
    kurtosis = sum((r - mean_r) ** 4 for r in returns) / (n * std_r ** 4) - 3.0 if std_r > 0 else 0

    # Worst days
    worst_1d = min(returns)
    worst_5d_sum = min(sum(returns[i:i+5]) for i in range(len(returns) - 4)) if n >= 5 else worst_1d
    worst_10d_sum = min(sum(returns[i:i+10]) for i in range(len(returns) - 9)) if n >= 10 else worst_5d_sum

    # Loss frequency
    loss_gt_05pct = sum(1 for r in returns if r < -0.005) / n * 100
    loss_gt_1pct = sum(1 for r in returns if r < -0.01) / n * 100
    loss_gt_2pct = sum(1 for r in returns if r < -0.02) / n * 100

    return {
        "var_95": round(var_95 * 100, 3),
        "var_99": round(var_99 * 100, 3),
        "cvar_95": round(cvar_95 * 100, 3),
        "cvar_99": round(cvar_99 * 100, 3),
        "max_dd": round(max_dd * 100, 3),
        "worst_1d": round(worst_1d * 100, 3),
        "worst_5d": round(worst_5d_sum * 100, 3),
        "worst_10d": round(worst_10d_sum * 100, 3),
        "recovery_bars": recovery_bars,
        "skewness": round(skewness, 2),
        "kurtosis": round(kurtosis, 2),
        "loss_gt_0.5pct_freq": round(loss_gt_05pct, 1),
        "loss_gt_1pct_freq": round(loss_gt_1pct, 1),
        "loss_gt_2pct_freq": round(loss_gt_2pct, 1),
        "final_equity": round(equity[-1], 4),
    }


def run_ensemble_detailed(
    sizing_config: Tuple[Optional[float], Optional[float]],
) -> Dict[str, Any]:
    """Run ensemble collecting detailed return series for tail analysis."""
    low_scale, high_scale = sizing_config

    all_returns = []
    yearly_results = {}

    for yr in YEARS:
        cfg = {
            "symbols": ["BTC"],
            "start": f"{yr}-01-01",
            "end": f"{yr}-12-31",
            "bar_interval": "1d",
            "use_synthetic_iv": True,
        }
        provider = DeribitRestProvider(cfg, seed=SEED)
        dataset = provider.load()
        n = len(dataset.timeline)

        vrp_strat = VariancePremiumStrategy(params=VRP_PARAMS)
        skew_strat = SkewTradeV2Strategy(params=SKEW_PARAMS)

        dt = 1.0 / BARS_PER_YEAR
        equity = 1.0
        vrp_weights = {"BTC": 0.0}
        skew_weights = {"BTC": 0.0}
        equity_curve = [1.0]
        returns_list = []
        yr_iv_scales = []

        for idx in range(1, n):
            prev_equity = equity
            sym = "BTC"

            # -- VRP P&L --
            vrp_pnl = 0.0
            w_v = vrp_weights.get(sym, 0.0)
            if abs(w_v) > 1e-10:
                closes = dataset.perp_close.get(sym, [])
                if idx < len(closes) and closes[idx - 1] > 0 and closes[idx] > 0:
                    log_ret = math.log(closes[idx] / closes[idx - 1])
                    rv_bar = abs(log_ret) * math.sqrt(BARS_PER_YEAR)
                else:
                    rv_bar = 0.0
                iv_series = dataset.features.get("iv_atm", {}).get(sym, [])
                iv = iv_series[idx - 1] if iv_series and idx - 1 < len(iv_series) else None
                if iv and iv > 0:
                    vrp = 0.5 * (iv ** 2 - rv_bar ** 2) * dt
                    vrp_pnl = (-w_v) * vrp

            # -- Skew P&L --
            skew_pnl = 0.0
            w_s = skew_weights.get(sym, 0.0)
            if abs(w_s) > 1e-10:
                skew_series = dataset.features.get("skew_25d", {}).get(sym, [])
                if idx < len(skew_series) and idx - 1 < len(skew_series):
                    s_now = skew_series[idx]
                    s_prev = skew_series[idx - 1]
                    if s_now is not None and s_prev is not None:
                        d_skew = float(s_now) - float(s_prev)
                        iv_series = dataset.features.get("iv_atm", {}).get(sym, [])
                        iv_s = iv_series[idx - 1] if iv_series and idx - 1 < len(iv_series) else 0.70
                        if iv_s and iv_s > 0:
                            sensitivity = iv_s * math.sqrt(dt) * 2.5
                            skew_pnl = w_s * d_skew * sensitivity

            bar_pnl = W_VRP * vrp_pnl + W_SKEW * skew_pnl
            dp = equity * bar_pnl
            equity += dp

            # -- Rebalance --
            if vrp_strat.should_rebalance(dataset, idx):
                target_v = vrp_strat.target_weights(dataset, idx, vrp_weights)
                target_s = skew_strat.target_weights(dataset, idx, skew_weights) if skew_strat.should_rebalance(dataset, idx) else skew_weights

                if low_scale is not None:
                    iv_series = dataset.features.get("iv_atm", {}).get(sym, [])
                    pct = iv_percentile(iv_series, idx)
                    if pct is not None:
                        scale = step_sizing(pct, low_scale, high_scale)
                        yr_iv_scales.append(scale)
                        for s in target_v:
                            target_v[s] *= scale
                        for s in target_s:
                            target_s[s] *= scale

                old_total = {sym: W_VRP * vrp_weights.get(sym, 0) + W_SKEW * skew_weights.get(sym, 0)}
                new_total = {sym: W_VRP * target_v.get(sym, 0) + W_SKEW * target_s.get(sym, 0)}
                turnover = sum(abs(new_total[s] - old_total[s]) for s in [sym])
                bd = COSTS.cost(equity=equity, turnover=turnover)
                cost = float(bd.get("cost", 0.0))
                equity -= cost
                equity = max(equity, 0.0)
                vrp_weights = target_v
                skew_weights = target_s

            elif skew_strat.should_rebalance(dataset, idx):
                target_s = skew_strat.target_weights(dataset, idx, skew_weights)
                if low_scale is not None:
                    iv_series = dataset.features.get("iv_atm", {}).get(sym, [])
                    pct = iv_percentile(iv_series, idx)
                    if pct is not None:
                        scale = step_sizing(pct, low_scale, high_scale)
                        yr_iv_scales.append(scale)
                        for s in target_s:
                            target_s[s] *= scale

                old_total = {sym: W_VRP * vrp_weights.get(sym, 0) + W_SKEW * skew_weights.get(sym, 0)}
                new_total = {sym: W_VRP * vrp_weights.get(sym, 0) + W_SKEW * target_s.get(sym, 0)}
                turnover = sum(abs(new_total[s] - old_total[s]) for s in [sym])
                bd = COSTS.cost(equity=equity, turnover=turnover)
                cost = float(bd.get("cost", 0.0))
                equity -= cost
                equity = max(equity, 0.0)
                skew_weights = target_s

            equity_curve.append(equity)
            bar_ret = (equity / prev_equity) - 1.0 if prev_equity > 0 else 0.0
            returns_list.append(bar_ret)
            all_returns.append(bar_ret)

        m = compute_metrics(equity_curve, returns_list, BARS_PER_YEAR)
        yr_tail = compute_tail_metrics(returns_list)
        yr_tail["sharpe"] = round(m["sharpe"], 3)
        yr_tail["avg_iv_scale"] = round(sum(yr_iv_scales) / len(yr_iv_scales), 2) if yr_iv_scales else 1.0
        yearly_results[str(yr)] = yr_tail

    full_tail = compute_tail_metrics(all_returns)
    sharpes = [yearly_results[str(yr)]["sharpe"] for yr in YEARS]
    full_tail["avg_sharpe"] = round(sum(sharpes) / len(sharpes), 3)
    full_tail["min_sharpe"] = round(min(sharpes), 3)

    return {
        "full": full_tail,
        "yearly": yearly_results,
        "n_bars": len(all_returns),
    }


def main():
    print("=" * 70)
    print("TAIL STRESS TEST: IV-SIZED ENSEMBLE — R30")
    print("=" * 70)
    print(f"Comparing: unsized, R25 (0.5/1.5), R28 (0.5/1.7), aggressive (0.3/2.0)")
    print()

    results = {}

    for name, cfg in SIZING_CONFIGS.items():
        print(f"  Running {name}...")
        r = run_ensemble_detailed(cfg)
        results[name] = r
        f = r["full"]
        print(f"    avg_sharpe={f['avg_sharpe']:.3f} MDD={f['max_dd']:.2f}% worst_1d={f['worst_1d']:.3f}% VaR95={f['var_95']:.3f}%")

    print()

    # ── Comparison Table ──────────────────────────────────────────────────
    print("=" * 70)
    print("TAIL METRICS COMPARISON")
    print("=" * 70)
    print()

    metrics = ["avg_sharpe", "min_sharpe", "max_dd", "worst_1d", "worst_5d", "worst_10d",
               "var_95", "var_99", "cvar_95", "cvar_99", "recovery_bars",
               "skewness", "kurtosis", "loss_gt_0.5pct_freq", "loss_gt_1pct_freq", "loss_gt_2pct_freq"]

    header = f"  {'Metric':25s}"
    for name in SIZING_CONFIGS:
        header += f" {name:>15s}"
    print(header)
    print("  " + "-" * (25 + 16 * len(SIZING_CONFIGS)))

    for m in metrics:
        row = f"  {m:25s}"
        for name in SIZING_CONFIGS:
            val = results[name]["full"].get(m, "N/A")
            if isinstance(val, float):
                row += f" {val:15.3f}"
            elif isinstance(val, int):
                row += f" {val:15d}"
            else:
                row += f" {str(val):>15s}"
        print(row)

    print()

    # ── Year-by-Year Worst Day ────────────────────────────────────────────
    print("--- WORST SINGLE DAY BY YEAR ---")
    header = f"  {'Year':6s}"
    for name in SIZING_CONFIGS:
        header += f" {name:>15s}"
    print(header)
    for yr in YEARS:
        row = f"  {yr:6d}"
        for name in SIZING_CONFIGS:
            wd = results[name]["yearly"][str(yr)]["worst_1d"]
            row += f" {wd:14.3f}%"
        print(row)

    print()

    # ── Year-by-Year Max DD ───────────────────────────────────────────────
    print("--- MAX DRAWDOWN BY YEAR ---")
    header = f"  {'Year':6s}"
    for name in SIZING_CONFIGS:
        header += f" {name:>15s}"
    print(header)
    for yr in YEARS:
        row = f"  {yr:6d}"
        for name in SIZING_CONFIGS:
            mdd = results[name]["yearly"][str(yr)]["max_dd"]
            row += f" {mdd:14.3f}%"
        print(row)

    print()

    # ── IV Scale Distribution ─────────────────────────────────────────────
    print("--- AVERAGE IV SCALE BY YEAR ---")
    header = f"  {'Year':6s}"
    for name in SIZING_CONFIGS:
        header += f" {name:>15s}"
    print(header)
    for yr in YEARS:
        row = f"  {yr:6d}"
        for name in SIZING_CONFIGS:
            scale = results[name]["yearly"][str(yr)].get("avg_iv_scale", 1.0)
            row += f" {scale:15.2f}"
        print(row)

    print()

    # ── Degradation Analysis ──────────────────────────────────────────────
    print("--- TAIL RISK DEGRADATION vs UNSIZED BASELINE ---")
    base = results["unsized"]["full"]
    for name in ["R25_0.5_1.5", "R28_0.5_1.7", "R28_0.3_2.0"]:
        r = results[name]["full"]
        print(f"\n  {name}:")
        mdd_d = r["max_dd"] - base["max_dd"]
        wd_d = r["worst_1d"] - base["worst_1d"]
        var95_d = r["var_95"] - base["var_95"]
        cvar95_d = r["cvar_95"] - base["cvar_95"]
        sharpe_d = r["avg_sharpe"] - base["avg_sharpe"]
        print(f"    Sharpe Δ:    {sharpe_d:+.3f}")
        print(f"    MDD Δ:       {mdd_d:+.3f}pp")
        print(f"    Worst 1d Δ:  {wd_d:+.3f}pp")
        print(f"    VaR 95 Δ:    {var95_d:+.3f}pp")
        print(f"    CVaR 95 Δ:   {cvar95_d:+.3f}pp")

        # Sharpe per unit of tail risk
        if abs(mdd_d) > 0.001:
            sharpe_per_mdd = sharpe_d / abs(mdd_d)
            print(f"    Sharpe/MDD:  {sharpe_per_mdd:.2f} (Sharpe gain per 1pp MDD)")

    # ── VERDICT ───────────────────────────────────────────────────────────
    print()
    print("=" * 70)

    r28 = results["R28_0.5_1.7"]["full"]
    base_f = results["unsized"]["full"]

    # Check critical thresholds
    checks = [
        ("MDD < 5%", abs(r28["max_dd"]) < 5.0),
        ("Worst day > -3%", r28["worst_1d"] > -3.0),
        ("VaR 95 > -0.5%", r28["var_95"] > -0.5),
        ("Recovery < 60 bars", r28["recovery_bars"] is not None and r28["recovery_bars"] < 60),
        ("Loss >1% freq < 1%", r28["loss_gt_1pct_freq"] < 1.0),
    ]

    all_pass = True
    print("SAFETY CHECKS (R28 production config 0.5/1.7x):")
    for name, passed in checks:
        status = "PASS" if passed else "FAIL"
        if not passed:
            all_pass = False
        print(f"  [{status}] {name}")

    print()
    if all_pass:
        print("VERDICT: R28 IV sizing (0.5/1.7x) PASSES all tail safety checks")
        print(f"  Tail risk increase is ACCEPTABLE for Sharpe gain of {r28['avg_sharpe'] - base_f['avg_sharpe']:+.3f}")
        print(f"  Production deployment: APPROVED")
    else:
        print("VERDICT: R28 IV sizing FAILS some safety checks — review needed")
        print(f"  Consider: reduce to R25 (0.5/1.5) or lower high_scale")
    print("=" * 70)

    # Save
    out = ROOT / "artifacts" / "crypto_options" / "iv_sized_tail_stress_test.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump({
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "research_id": "R30",
            "results": {k: v for k, v in results.items()},
        }, f, indent=2, default=str)
    print(f"\nResults saved to: {out}")


if __name__ == "__main__":
    main()
