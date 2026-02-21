#!/usr/bin/env python3
"""
Equity-Curve Position Scaling on VRP+Skew Ensemble — R26
==========================================================

Prior research showed:
  - Market-signal overlays (regime filters R21) DESTROY carry Sharpe
  - IV-percentile SIZING (R25) IMPROVES carry (+0.276 on ensemble)
  - IV sizing scales by expected carry richness, not by timing in/out

Key question: Do equity-curve-based overlays also destroy carry?
  - They respond to PORTFOLIO STATE (drawdown), not market signals
  - Classic systematic trading technique (Pardo, Vince, etc.)
  - Fundamentally different from regime filters

Test matrix:
  1. Drawdown threshold scaling: reduce when DD > X%
  2. Moving average envelope: reduce when equity < MA
  3. Time-since-HWM: reduce if no new HWM in N bars
  4. Combined: drawdown + IV sizing together

Configurations: ~30 total across 4 approaches
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


# ── IV Sizing (from R25) ──────────────────────────────────────────────────

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


def step_sizing(pct: float) -> float:
    if pct < 0.25:
        return 0.5
    elif pct > 0.75:
        return 1.5
    return 1.0


# ── Equity-Curve Scaling Functions ─────────────────────────────────────────

def dd_scale(equity: float, hwm: float, dd_threshold: float, scale_factor: float) -> float:
    """Scale down when drawdown exceeds threshold."""
    if hwm <= 0:
        return 1.0
    dd = (equity - hwm) / hwm
    if dd < -dd_threshold:
        return scale_factor
    return 1.0


def ma_envelope_scale(equity_curve: List[float], ma_lookback: int, scale_factor: float) -> float:
    """Scale down when equity is below its moving average."""
    if len(equity_curve) < ma_lookback:
        return 1.0
    ma = sum(equity_curve[-ma_lookback:]) / ma_lookback
    if equity_curve[-1] < ma:
        return scale_factor
    return 1.0


def time_since_hwm_scale(bars_since_hwm: int, patience: int, scale_factor: float) -> float:
    """Scale down if no new HWM in `patience` bars."""
    if bars_since_hwm > patience:
        return scale_factor
    return 1.0


def graduated_dd_scale(equity: float, hwm: float, thresholds: List[tuple]) -> float:
    """Graduated scaling: more drawdown → more scaling down.
    thresholds: [(dd_pct, scale), ...] sorted by dd_pct ascending
    """
    if hwm <= 0:
        return 1.0
    dd = (equity - hwm) / hwm
    scale = 1.0
    for dd_thresh, s in thresholds:
        if dd < -dd_thresh:
            scale = s
    return scale


# ── Backtest Engine ────────────────────────────────────────────────────────

def run_ensemble(
    ec_mode: str = "none",
    ec_params: Optional[Dict] = None,
    use_iv_sizing: bool = False,
) -> Dict[str, Any]:
    """
    Run VRP+Skew ensemble with optional equity-curve scaling.

    ec_mode:
      "none": no equity-curve overlay
      "dd_threshold": reduce when DD > threshold
      "ma_envelope": reduce when equity < MA
      "time_hwm": reduce when no new HWM in N bars
      "graduated_dd": multi-step drawdown scaling
    """
    sharpes = []
    yearly_detail = {}
    all_dd_events = []  # track how often scaling triggers

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
        hwm = 1.0
        bars_since_hwm = 0
        vrp_weights = {"BTC": 0.0}
        skew_weights = {"BTC": 0.0}
        equity_curve = [1.0]
        returns_list = []
        scale_triggered_bars = 0

        for idx in range(1, n):
            prev_equity = equity
            sym = "BTC"

            # -- Compute equity-curve scale --
            ec_scale = 1.0
            if ec_mode == "dd_threshold" and ec_params:
                ec_scale = dd_scale(
                    equity, hwm,
                    ec_params["dd_threshold"],
                    ec_params["scale_factor"]
                )
            elif ec_mode == "ma_envelope" and ec_params:
                ec_scale = ma_envelope_scale(
                    equity_curve,
                    ec_params["ma_lookback"],
                    ec_params["scale_factor"]
                )
            elif ec_mode == "time_hwm" and ec_params:
                ec_scale = time_since_hwm_scale(
                    bars_since_hwm,
                    ec_params["patience"],
                    ec_params["scale_factor"]
                )
            elif ec_mode == "graduated_dd" and ec_params:
                ec_scale = graduated_dd_scale(
                    equity, hwm,
                    ec_params["thresholds"]
                )

            if ec_scale < 1.0:
                scale_triggered_bars += 1

            # -- VRP P&L --
            vrp_pnl = 0.0
            w_v = vrp_weights.get(sym, 0.0) * ec_scale
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
            w_s = skew_weights.get(sym, 0.0) * ec_scale
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

            # -- Update HWM tracking --
            if equity > hwm:
                hwm = equity
                bars_since_hwm = 0
            else:
                bars_since_hwm += 1

            # -- Rebalance --
            if vrp_strat.should_rebalance(dataset, idx):
                target_v = vrp_strat.target_weights(dataset, idx, vrp_weights)
                target_s = skew_strat.target_weights(dataset, idx, skew_weights) if skew_strat.should_rebalance(dataset, idx) else skew_weights

                # Apply IV sizing (R25)
                if use_iv_sizing:
                    iv_series = dataset.features.get("iv_atm", {}).get(sym, [])
                    pct = iv_percentile(iv_series, idx)
                    if pct is not None:
                        scale = step_sizing(pct)
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
                if use_iv_sizing:
                    iv_series = dataset.features.get("iv_atm", {}).get(sym, [])
                    pct = iv_percentile(iv_series, idx)
                    if pct is not None:
                        scale = step_sizing(pct)
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

        m = compute_metrics(equity_curve, returns_list, BARS_PER_YEAR)
        sharpes.append(m["sharpe"])
        yearly_detail[str(yr)] = round(m["sharpe"], 3)
        all_dd_events.append({
            "year": yr,
            "triggered_bars": scale_triggered_bars,
            "total_bars": n - 1,
            "trigger_pct": round(100 * scale_triggered_bars / max(n - 1, 1), 1),
            "final_equity": round(equity, 4),
            "max_dd": round(m.get("max_drawdown", 0), 4),
        })

    avg = sum(sharpes) / len(sharpes)
    mn = min(sharpes)
    return {
        "avg_sharpe": round(avg, 3),
        "min_sharpe": round(mn, 3),
        "yearly": yearly_detail,
        "dd_events": all_dd_events,
    }


def main():
    print("=" * 70)
    print("EQUITY-CURVE POSITION SCALING — R26")
    print("=" * 70)
    print(f"Ensemble: VRP {W_VRP*100:.0f}% + Skew MR {W_SKEW*100:.0f}%")
    print(f"Years: {YEARS}")
    print()

    # ── 1. Baselines ──────────────────────────────────────────────────────
    print("--- BASELINES ---")

    baseline = run_ensemble(ec_mode="none", use_iv_sizing=False)
    print(f"  Baseline (no sizing, no EC):  avg={baseline['avg_sharpe']:.3f} min={baseline['min_sharpe']:.3f}")

    iv_sized = run_ensemble(ec_mode="none", use_iv_sizing=True)
    print(f"  IV-sized only (R25):          avg={iv_sized['avg_sharpe']:.3f} min={iv_sized['min_sharpe']:.3f}")
    print()

    results = {}
    results["baseline"] = baseline
    results["iv_sized"] = iv_sized

    # ── 2. Drawdown threshold scaling ─────────────────────────────────────
    print("--- APPROACH 1: Drawdown Threshold Scaling ---")
    print("  Scale down when DD > threshold")

    dd_configs = [
        {"dd_threshold": 0.005, "scale_factor": 0.5, "tag": "dd0.5%_s0.5"},
        {"dd_threshold": 0.005, "scale_factor": 0.25, "tag": "dd0.5%_s0.25"},
        {"dd_threshold": 0.01,  "scale_factor": 0.5,  "tag": "dd1.0%_s0.5"},
        {"dd_threshold": 0.01,  "scale_factor": 0.25, "tag": "dd1.0%_s0.25"},
        {"dd_threshold": 0.02,  "scale_factor": 0.5,  "tag": "dd2.0%_s0.5"},
        {"dd_threshold": 0.02,  "scale_factor": 0.25, "tag": "dd2.0%_s0.25"},
        {"dd_threshold": 0.005, "scale_factor": 0.0,  "tag": "dd0.5%_s0.0"},  # full exit
        {"dd_threshold": 0.01,  "scale_factor": 0.0,  "tag": "dd1.0%_s0.0"},  # full exit
    ]

    dd_results = []
    for cfg in dd_configs:
        tag = cfg.pop("tag")
        r = run_ensemble(ec_mode="dd_threshold", ec_params=cfg, use_iv_sizing=False)
        d = r["avg_sharpe"] - baseline["avg_sharpe"]
        trigger_pcts = [e["trigger_pct"] for e in r["dd_events"]]
        avg_trigger = sum(trigger_pcts) / len(trigger_pcts) if trigger_pcts else 0
        dd_results.append({
            "tag": tag, **r, "delta": round(d, 3), "avg_trigger_pct": round(avg_trigger, 1)
        })
        marker = "+" if d > 0 else " "
        print(f"    {tag:20s} avg={r['avg_sharpe']:.3f} min={r['min_sharpe']:.3f} Δ={d:+.3f} trig={avg_trigger:.0f}% {marker}")
        cfg["tag"] = tag  # restore

    results["dd_threshold"] = dd_results
    print()

    # ── 3. Moving average envelope ────────────────────────────────────────
    print("--- APPROACH 2: Moving Average Envelope ---")
    print("  Scale down when equity < MA(lookback)")

    ma_configs = [
        {"ma_lookback": 10, "scale_factor": 0.5, "tag": "ma10_s0.5"},
        {"ma_lookback": 20, "scale_factor": 0.5, "tag": "ma20_s0.5"},
        {"ma_lookback": 30, "scale_factor": 0.5, "tag": "ma30_s0.5"},
        {"ma_lookback": 60, "scale_factor": 0.5, "tag": "ma60_s0.5"},
        {"ma_lookback": 10, "scale_factor": 0.25, "tag": "ma10_s0.25"},
        {"ma_lookback": 20, "scale_factor": 0.25, "tag": "ma20_s0.25"},
        {"ma_lookback": 30, "scale_factor": 0.25, "tag": "ma30_s0.25"},
    ]

    ma_results = []
    for cfg in ma_configs:
        tag = cfg.pop("tag")
        r = run_ensemble(ec_mode="ma_envelope", ec_params=cfg, use_iv_sizing=False)
        d = r["avg_sharpe"] - baseline["avg_sharpe"]
        trigger_pcts = [e["trigger_pct"] for e in r["dd_events"]]
        avg_trigger = sum(trigger_pcts) / len(trigger_pcts) if trigger_pcts else 0
        ma_results.append({
            "tag": tag, **r, "delta": round(d, 3), "avg_trigger_pct": round(avg_trigger, 1)
        })
        marker = "+" if d > 0 else " "
        print(f"    {tag:20s} avg={r['avg_sharpe']:.3f} min={r['min_sharpe']:.3f} Δ={d:+.3f} trig={avg_trigger:.0f}% {marker}")
        cfg["tag"] = tag

    results["ma_envelope"] = ma_results
    print()

    # ── 4. Time-since-HWM ─────────────────────────────────────────────────
    print("--- APPROACH 3: Time-Since-HWM Patience ---")
    print("  Scale down when no new HWM in N bars")

    hwm_configs = [
        {"patience": 10, "scale_factor": 0.5, "tag": "hwm10_s0.5"},
        {"patience": 20, "scale_factor": 0.5, "tag": "hwm20_s0.5"},
        {"patience": 30, "scale_factor": 0.5, "tag": "hwm30_s0.5"},
        {"patience": 60, "scale_factor": 0.5, "tag": "hwm60_s0.5"},
        {"patience": 10, "scale_factor": 0.25, "tag": "hwm10_s0.25"},
        {"patience": 20, "scale_factor": 0.25, "tag": "hwm20_s0.25"},
    ]

    hwm_results = []
    for cfg in hwm_configs:
        tag = cfg.pop("tag")
        r = run_ensemble(ec_mode="time_hwm", ec_params=cfg, use_iv_sizing=False)
        d = r["avg_sharpe"] - baseline["avg_sharpe"]
        trigger_pcts = [e["trigger_pct"] for e in r["dd_events"]]
        avg_trigger = sum(trigger_pcts) / len(trigger_pcts) if trigger_pcts else 0
        hwm_results.append({
            "tag": tag, **r, "delta": round(d, 3), "avg_trigger_pct": round(avg_trigger, 1)
        })
        marker = "+" if d > 0 else " "
        print(f"    {tag:20s} avg={r['avg_sharpe']:.3f} min={r['min_sharpe']:.3f} Δ={d:+.3f} trig={avg_trigger:.0f}% {marker}")
        cfg["tag"] = tag

    results["time_hwm"] = hwm_results
    print()

    # ── 5. Graduated DD scaling ───────────────────────────────────────────
    print("--- APPROACH 4: Graduated Drawdown Scaling ---")
    print("  Multiple thresholds: deeper DD → more scaling")

    grad_configs = [
        {
            "thresholds": [(0.005, 0.75), (0.01, 0.5), (0.02, 0.25)],
            "tag": "grad_0.5/1/2%"
        },
        {
            "thresholds": [(0.005, 0.5), (0.01, 0.25), (0.02, 0.0)],
            "tag": "grad_agg_0.5/1/2%"
        },
        {
            "thresholds": [(0.01, 0.75), (0.02, 0.5)],
            "tag": "grad_0.1/2%"
        },
    ]

    grad_results = []
    for cfg in grad_configs:
        tag = cfg["tag"]
        params = {"thresholds": cfg["thresholds"]}
        r = run_ensemble(ec_mode="graduated_dd", ec_params=params, use_iv_sizing=False)
        d = r["avg_sharpe"] - baseline["avg_sharpe"]
        trigger_pcts = [e["trigger_pct"] for e in r["dd_events"]]
        avg_trigger = sum(trigger_pcts) / len(trigger_pcts) if trigger_pcts else 0
        grad_results.append({
            "tag": tag, **r, "delta": round(d, 3), "avg_trigger_pct": round(avg_trigger, 1)
        })
        marker = "+" if d > 0 else " "
        print(f"    {tag:20s} avg={r['avg_sharpe']:.3f} min={r['min_sharpe']:.3f} Δ={d:+.3f} trig={avg_trigger:.0f}% {marker}")

    results["graduated_dd"] = grad_results
    print()

    # ── 6. Combined: best EC + IV sizing ──────────────────────────────────
    print("--- APPROACH 5: Best EC Methods + IV Sizing (R25) ---")
    print("  Test whether EC scaling and IV sizing are complementary")

    # Pick a few representative EC configs to combine with IV sizing
    combo_configs = [
        ("dd1.0%_s0.5", "dd_threshold", {"dd_threshold": 0.01, "scale_factor": 0.5}),
        ("ma20_s0.5", "ma_envelope", {"ma_lookback": 20, "scale_factor": 0.5}),
        ("hwm20_s0.5", "time_hwm", {"patience": 20, "scale_factor": 0.5}),
        ("grad_0.1/2%", "graduated_dd", {"thresholds": [(0.01, 0.75), (0.02, 0.5)]}),
    ]

    combo_results = []
    for tag, mode, params in combo_configs:
        combo_tag = f"{tag}+IV"
        r = run_ensemble(ec_mode=mode, ec_params=params, use_iv_sizing=True)
        d_vs_base = r["avg_sharpe"] - baseline["avg_sharpe"]
        d_vs_iv = r["avg_sharpe"] - iv_sized["avg_sharpe"]
        trigger_pcts = [e["trigger_pct"] for e in r["dd_events"]]
        avg_trigger = sum(trigger_pcts) / len(trigger_pcts) if trigger_pcts else 0
        combo_results.append({
            "tag": combo_tag, **r,
            "delta_vs_base": round(d_vs_base, 3),
            "delta_vs_iv_only": round(d_vs_iv, 3),
            "avg_trigger_pct": round(avg_trigger, 1),
        })
        marker = "+" if d_vs_iv > 0 else " "
        print(f"    {combo_tag:25s} avg={r['avg_sharpe']:.3f} min={r['min_sharpe']:.3f} Δbase={d_vs_base:+.3f} ΔIV={d_vs_iv:+.3f} trig={avg_trigger:.0f}% {marker}")

    results["combined_ec_iv"] = combo_results
    print()

    # ── SUMMARY ───────────────────────────────────────────────────────────
    print("=" * 70)
    print("SUMMARY — R26: Equity-Curve Position Scaling")
    print("=" * 70)
    print()
    print(f"  Baseline (no overlay):    avg={baseline['avg_sharpe']:.3f} min={baseline['min_sharpe']:.3f}")
    print(f"  IV-sized only (R25):      avg={iv_sized['avg_sharpe']:.3f} min={iv_sized['min_sharpe']:.3f}")
    print()

    # Collect all EC-only results
    all_ec = dd_results + ma_results + hwm_results + grad_results
    n_better = sum(1 for r in all_ec if r["delta"] > 0)
    n_worse = sum(1 for r in all_ec if r["delta"] < 0)
    n_total = len(all_ec)

    print(f"  EC-only configs tested:   {n_total}")
    print(f"    Better than baseline:   {n_better}/{n_total}")
    print(f"    Worse than baseline:    {n_worse}/{n_total}")
    print()

    # Best and worst EC-only
    best_ec = max(all_ec, key=lambda x: x["avg_sharpe"])
    worst_ec = min(all_ec, key=lambda x: x["avg_sharpe"])
    print(f"  Best EC-only:  {best_ec['tag']:25s} avg={best_ec['avg_sharpe']:.3f} Δ={best_ec['delta']:+.3f}")
    print(f"  Worst EC-only: {worst_ec['tag']:25s} avg={worst_ec['avg_sharpe']:.3f} Δ={worst_ec['delta']:+.3f}")
    print()

    # By approach
    print("  BY APPROACH (avg delta):")
    for name, res_list in [("DD threshold", dd_results), ("MA envelope", ma_results),
                           ("Time-HWM", hwm_results), ("Graduated DD", grad_results)]:
        if res_list:
            avg_d = sum(r["delta"] for r in res_list) / len(res_list)
            best = max(res_list, key=lambda x: x["delta"])
            print(f"    {name:16s} avg_Δ={avg_d:+.3f}  best: {best['tag']} Δ={best['delta']:+.3f}")
    print()

    # Combined results
    if combo_results:
        print("  COMBINED (EC + IV sizing):")
        for r in combo_results:
            print(f"    {r['tag']:25s} avg={r['avg_sharpe']:.3f} ΔIV={r['delta_vs_iv_only']:+.3f}")
        best_combo = max(combo_results, key=lambda x: x["avg_sharpe"])
        print(f"\n  Best combined:  {best_combo['tag']:25s} avg={best_combo['avg_sharpe']:.3f}")
    print()

    # Verdict
    print("=" * 70)
    if n_better > n_worse:
        print(f"VERDICT: Equity-curve scaling IMPROVES ensemble ({n_better}/{n_total} configs better)")
        print(f"  Best: {best_ec['tag']} Δ={best_ec['delta']:+.3f}")
        if combo_results:
            best_c = max(combo_results, key=lambda x: x["avg_sharpe"])
            if best_c["delta_vs_iv_only"] > 0:
                print(f"  EC + IV sizing is COMPLEMENTARY: {best_c['tag']} ΔIV={best_c['delta_vs_iv_only']:+.3f}")
            else:
                print(f"  EC + IV sizing is NOT complementary (ΔIV={best_c['delta_vs_iv_only']:+.3f})")
    elif n_better == 0:
        print(f"VERDICT: Equity-curve scaling DESTROYS ensemble (0/{n_total} configs improve)")
        print("  Confirms: carry trades should be ALWAYS FULLY INVESTED")
        print("  Any overlay that reduces time-in-market hurts carry")
    else:
        print(f"VERDICT: Equity-curve scaling is MIXED ({n_better}/{n_total} better)")
        print(f"  Best: {best_ec['tag']} Δ={best_ec['delta']:+.3f}")
    print("=" * 70)

    # Save
    out = ROOT / "artifacts" / "crypto_options" / "equity_curve_scaling_research.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump({
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "research_id": "R26",
            "baseline": baseline,
            "iv_sized_baseline": iv_sized,
            "n_configs": n_total,
            "n_better": n_better,
            "n_worse": n_worse,
            "best_ec_only": {"tag": best_ec["tag"], "avg_sharpe": best_ec["avg_sharpe"], "delta": best_ec["delta"]},
            "worst_ec_only": {"tag": worst_ec["tag"], "avg_sharpe": worst_ec["avg_sharpe"], "delta": worst_ec["delta"]},
            "dd_threshold_results": dd_results,
            "ma_envelope_results": ma_results,
            "time_hwm_results": hwm_results,
            "graduated_dd_results": grad_results,
            "combined_ec_iv_results": combo_results,
        }, f, indent=2)
    print(f"\nResults saved to: {out}")


if __name__ == "__main__":
    main()
