#!/usr/bin/env python3
"""
VRP Carry Magnitude Sizing on Ensemble — R27
===============================================

R25 showed IV-percentile sizing improves ensemble by +0.276.
R26 showed equity-curve overlays destroy carry (0/24 improve).

Key insight: sizing that scales by EXPECTED CARRY RICHNESS works.
IV percentile is an indirect proxy. Can we do better with DIRECT carry measures?

VRP carry = 0.5 * (IV² - RV²) — the actual theta income per bar.
When carry is historically high → increase exposure (more to earn).
When carry is historically low → decrease (less available premium).

Test matrix:
  1. VRP carry percentile (like IV percentile but on the actual spread)
  2. VRP carry Z-score (normalized magnitude)
  3. IV-RV ratio (simple carry richness ratio)
  4. Combined: carry sizing + IV sizing

Also test: does carry sizing on VRP-only component (not Skew) work better?
Since carry is specific to VRP, not Skew MR.

Configurations: ~35 total
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


# ── IV-Percentile Sizing (R25 baseline) ───────────────────────────────────

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


def step_sizing_iv(pct: float) -> float:
    if pct < 0.25:
        return 0.5
    elif pct > 0.75:
        return 1.5
    return 1.0


# ── Carry-Based Sizing Functions ──────────────────────────────────────────

def compute_vrp_carry(dataset, sym: str, idx: int) -> Optional[float]:
    """Compute instantaneous VRP carry = 0.5 * (IV² - RV²)."""
    iv_series = dataset.features.get("iv_atm", {}).get(sym, [])
    if not iv_series or idx >= len(iv_series) or iv_series[idx] is None:
        return None
    iv = iv_series[idx]
    if iv <= 0:
        return None

    closes = dataset.perp_close.get(sym, [])
    if idx < 1 or idx >= len(closes) or closes[idx] <= 0 or closes[idx - 1] <= 0:
        return None

    log_ret = math.log(closes[idx] / closes[idx - 1])
    rv_bar = abs(log_ret) * math.sqrt(BARS_PER_YEAR)

    return 0.5 * (iv ** 2 - rv_bar ** 2)


def carry_percentile_scale(carry_history: List[float], current_carry: float,
                           low_pct: float = 0.25, high_pct: float = 0.75,
                           low_scale: float = 0.5, high_scale: float = 1.5) -> float:
    """Scale by carry percentile rank."""
    if len(carry_history) < 10:
        return 1.0
    below = sum(1 for v in carry_history if v < current_carry)
    pct = below / len(carry_history)
    if pct < low_pct:
        return low_scale
    elif pct > high_pct:
        return high_scale
    return 1.0


def carry_zscore_scale(carry_history: List[float], current_carry: float,
                       low_z: float = -1.0, high_z: float = 1.0,
                       low_scale: float = 0.5, high_scale: float = 1.5) -> float:
    """Scale by carry Z-score."""
    if len(carry_history) < 10:
        return 1.0
    mean = sum(carry_history) / len(carry_history)
    var = sum((v - mean) ** 2 for v in carry_history) / len(carry_history)
    std = math.sqrt(var) if var > 0 else 0.01
    z = (current_carry - mean) / std
    if z < low_z:
        return low_scale
    elif z > high_z:
        return high_scale
    return 1.0


def iv_rv_ratio_scale(iv: float, rv: float,
                      low_ratio: float = 0.8, high_ratio: float = 1.5,
                      low_scale: float = 0.5, high_scale: float = 1.5) -> float:
    """Scale by IV/RV ratio. >1 means IV is rich, <1 means cheap."""
    if rv <= 0.01:
        rv = 0.01
    ratio = iv / rv
    if ratio < low_ratio:
        return low_scale
    elif ratio > high_ratio:
        return high_scale
    return 1.0


# ── Backtest Engine ────────────────────────────────────────────────────────

def run_ensemble(
    sizing_mode: str = "none",
    sizing_params: Optional[Dict] = None,
    apply_to: str = "both",  # "both", "vrp_only"
    use_iv_sizing: bool = False,
) -> Dict[str, Any]:
    """
    Run VRP+Skew ensemble with optional carry-based sizing.

    sizing_mode:
      "none": baseline
      "carry_pct": carry percentile sizing
      "carry_z": carry Z-score sizing
      "iv_rv_ratio": IV/RV ratio sizing
    apply_to:
      "both": size both VRP and Skew components
      "vrp_only": only size VRP component (carry-specific)
    """
    sharpes = []
    yearly_detail = {}

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

        # Track carry history for percentile/z-score
        carry_history = []
        lookback = sizing_params.get("lookback", 90) if sizing_params else 90

        for idx in range(1, n):
            prev_equity = equity
            sym = "BTC"

            # Update carry history
            carry = compute_vrp_carry(dataset, sym, idx - 1)
            if carry is not None:
                carry_history.append(carry)
                # Keep only lookback window
                if len(carry_history) > lookback:
                    carry_history = carry_history[-lookback:]

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

                # Apply IV sizing first (R25)
                if use_iv_sizing:
                    iv_series = dataset.features.get("iv_atm", {}).get(sym, [])
                    pct = iv_percentile(iv_series, idx)
                    if pct is not None:
                        scale = step_sizing_iv(pct)
                        for s in target_v:
                            target_v[s] *= scale
                        for s in target_s:
                            target_s[s] *= scale

                # Apply carry-based sizing
                if sizing_mode != "none" and sizing_params and len(carry_history) >= 10:
                    current_carry = compute_vrp_carry(dataset, sym, idx)
                    if current_carry is not None:
                        if sizing_mode == "carry_pct":
                            scale = carry_percentile_scale(
                                carry_history, current_carry,
                                sizing_params.get("low_pct", 0.25),
                                sizing_params.get("high_pct", 0.75),
                                sizing_params.get("low_scale", 0.5),
                                sizing_params.get("high_scale", 1.5),
                            )
                        elif sizing_mode == "carry_z":
                            scale = carry_zscore_scale(
                                carry_history, current_carry,
                                sizing_params.get("low_z", -1.0),
                                sizing_params.get("high_z", 1.0),
                                sizing_params.get("low_scale", 0.5),
                                sizing_params.get("high_scale", 1.5),
                            )
                        elif sizing_mode == "iv_rv_ratio":
                            iv_series = dataset.features.get("iv_atm", {}).get(sym, [])
                            iv_curr = iv_series[idx] if iv_series and idx < len(iv_series) else None
                            closes = dataset.perp_close.get(sym, [])
                            rv_curr = 0.0
                            if idx > 0 and idx < len(closes) and closes[idx] > 0 and closes[idx - 1] > 0:
                                lr = math.log(closes[idx] / closes[idx - 1])
                                rv_curr = abs(lr) * math.sqrt(BARS_PER_YEAR)
                            if iv_curr and iv_curr > 0:
                                scale = iv_rv_ratio_scale(
                                    iv_curr, rv_curr,
                                    sizing_params.get("low_ratio", 0.8),
                                    sizing_params.get("high_ratio", 1.5),
                                    sizing_params.get("low_scale", 0.5),
                                    sizing_params.get("high_scale", 1.5),
                                )
                            else:
                                scale = 1.0
                        else:
                            scale = 1.0

                        if apply_to == "both":
                            for s in target_v:
                                target_v[s] *= scale
                            for s in target_s:
                                target_s[s] *= scale
                        elif apply_to == "vrp_only":
                            for s in target_v:
                                target_v[s] *= scale

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
                        scale = step_sizing_iv(pct)
                        for s in target_s:
                            target_s[s] *= scale

                # Also apply carry sizing to Skew if "both"
                if sizing_mode != "none" and sizing_params and apply_to == "both" and len(carry_history) >= 10:
                    current_carry = compute_vrp_carry(dataset, sym, idx)
                    if current_carry is not None:
                        if sizing_mode == "carry_pct":
                            scale = carry_percentile_scale(
                                carry_history, current_carry,
                                sizing_params.get("low_pct", 0.25),
                                sizing_params.get("high_pct", 0.75),
                                sizing_params.get("low_scale", 0.5),
                                sizing_params.get("high_scale", 1.5),
                            )
                        elif sizing_mode == "carry_z":
                            scale = carry_zscore_scale(
                                carry_history, current_carry,
                                sizing_params.get("low_z", -1.0),
                                sizing_params.get("high_z", 1.0),
                                sizing_params.get("low_scale", 0.5),
                                sizing_params.get("high_scale", 1.5),
                            )
                        elif sizing_mode == "iv_rv_ratio":
                            iv_series = dataset.features.get("iv_atm", {}).get(sym, [])
                            iv_curr = iv_series[idx] if iv_series and idx < len(iv_series) else None
                            closes = dataset.perp_close.get(sym, [])
                            rv_curr = 0.0
                            if idx > 0 and idx < len(closes) and closes[idx] > 0 and closes[idx - 1] > 0:
                                lr = math.log(closes[idx] / closes[idx - 1])
                                rv_curr = abs(lr) * math.sqrt(BARS_PER_YEAR)
                            if iv_curr and iv_curr > 0:
                                scale = iv_rv_ratio_scale(
                                    iv_curr, rv_curr,
                                    sizing_params.get("low_ratio", 0.8),
                                    sizing_params.get("high_ratio", 1.5),
                                    sizing_params.get("low_scale", 0.5),
                                    sizing_params.get("high_scale", 1.5),
                                )
                            else:
                                scale = 1.0
                        else:
                            scale = 1.0
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

    avg = sum(sharpes) / len(sharpes)
    mn = min(sharpes)
    return {
        "avg_sharpe": round(avg, 3),
        "min_sharpe": round(mn, 3),
        "yearly": yearly_detail,
    }


def main():
    print("=" * 70)
    print("VRP CARRY MAGNITUDE SIZING — R27")
    print("=" * 70)
    print(f"Ensemble: VRP {W_VRP*100:.0f}% + Skew MR {W_SKEW*100:.0f}%")
    print(f"Years: {YEARS}")
    print()

    # ── Baselines ─────────────────────────────────────────────────────────
    print("--- BASELINES ---")
    baseline = run_ensemble(sizing_mode="none", use_iv_sizing=False)
    print(f"  Baseline:            avg={baseline['avg_sharpe']:.3f} min={baseline['min_sharpe']:.3f}")
    print(f"    Yearly: {baseline['yearly']}")

    iv_sized = run_ensemble(sizing_mode="none", use_iv_sizing=True)
    print(f"  IV-sized (R25):      avg={iv_sized['avg_sharpe']:.3f} min={iv_sized['min_sharpe']:.3f}")
    print(f"    Yearly: {iv_sized['yearly']}")
    print()

    all_results = []

    # ── 1. Carry Percentile Sizing ────────────────────────────────────────
    print("--- APPROACH 1: VRP Carry Percentile Sizing ---")

    carry_pct_configs = [
        # (lookback, low_pct, high_pct, low_scale, high_scale, apply_to)
        (90,  0.25, 0.75, 0.5, 1.5, "both"),
        (90,  0.25, 0.75, 0.5, 1.5, "vrp_only"),
        (180, 0.25, 0.75, 0.5, 1.5, "both"),
        (180, 0.25, 0.75, 0.5, 1.5, "vrp_only"),
        (60,  0.25, 0.75, 0.5, 1.5, "both"),
        (60,  0.25, 0.75, 0.5, 1.5, "vrp_only"),
        # More aggressive
        (90,  0.20, 0.80, 0.3, 1.7, "both"),
        (180, 0.20, 0.80, 0.3, 1.7, "both"),
        # Less aggressive
        (90,  0.33, 0.67, 0.7, 1.3, "both"),
        (180, 0.33, 0.67, 0.7, 1.3, "both"),
    ]

    for lb, lp, hp, ls, hs, apply in carry_pct_configs:
        tag = f"cpct_lb{lb}_{lp:.0%}-{hp:.0%}_s{ls}-{hs}_{apply}"
        params = {"lookback": lb, "low_pct": lp, "high_pct": hp, "low_scale": ls, "high_scale": hs}
        r = run_ensemble(sizing_mode="carry_pct", sizing_params=params, apply_to=apply)
        d = r["avg_sharpe"] - baseline["avg_sharpe"]
        all_results.append({"tag": tag, "approach": "carry_pct", **r, "delta": round(d, 3)})
        marker = "+" if d > 0.01 else " "
        print(f"    {tag:50s} avg={r['avg_sharpe']:.3f} Δ={d:+.3f} {marker}")

    print()

    # ── 2. Carry Z-Score Sizing ───────────────────────────────────────────
    print("--- APPROACH 2: VRP Carry Z-Score Sizing ---")

    carry_z_configs = [
        (90,  -1.0, 1.0, 0.5, 1.5, "both"),
        (90,  -1.0, 1.0, 0.5, 1.5, "vrp_only"),
        (180, -1.0, 1.0, 0.5, 1.5, "both"),
        (180, -1.0, 1.0, 0.5, 1.5, "vrp_only"),
        (60,  -1.0, 1.0, 0.5, 1.5, "both"),
        # Wider thresholds
        (90,  -1.5, 1.5, 0.5, 1.5, "both"),
        (180, -1.5, 1.5, 0.5, 1.5, "both"),
        # Narrower thresholds
        (90,  -0.5, 0.5, 0.5, 1.5, "both"),
        (180, -0.5, 0.5, 0.5, 1.5, "both"),
    ]

    for lb, lz, hz, ls, hs, apply in carry_z_configs:
        tag = f"cz_lb{lb}_z{lz}/{hz}_s{ls}-{hs}_{apply}"
        params = {"lookback": lb, "low_z": lz, "high_z": hz, "low_scale": ls, "high_scale": hs}
        r = run_ensemble(sizing_mode="carry_z", sizing_params=params, apply_to=apply)
        d = r["avg_sharpe"] - baseline["avg_sharpe"]
        all_results.append({"tag": tag, "approach": "carry_z", **r, "delta": round(d, 3)})
        marker = "+" if d > 0.01 else " "
        print(f"    {tag:50s} avg={r['avg_sharpe']:.3f} Δ={d:+.3f} {marker}")

    print()

    # ── 3. IV/RV Ratio Sizing ─────────────────────────────────────────────
    print("--- APPROACH 3: IV/RV Ratio Sizing ---")

    ratio_configs = [
        (0.8, 1.5, 0.5, 1.5, "both"),
        (0.8, 1.5, 0.5, 1.5, "vrp_only"),
        (1.0, 2.0, 0.5, 1.5, "both"),
        (1.0, 2.0, 0.5, 1.5, "vrp_only"),
        (0.7, 1.3, 0.5, 1.5, "both"),
        # More aggressive
        (0.8, 1.5, 0.3, 1.7, "both"),
        (1.0, 2.0, 0.3, 1.7, "both"),
    ]

    for lr, hr, ls, hs, apply in ratio_configs:
        tag = f"ivrv_r{lr}-{hr}_s{ls}-{hs}_{apply}"
        params = {"low_ratio": lr, "high_ratio": hr, "low_scale": ls, "high_scale": hs, "lookback": 90}
        r = run_ensemble(sizing_mode="iv_rv_ratio", sizing_params=params, apply_to=apply)
        d = r["avg_sharpe"] - baseline["avg_sharpe"]
        all_results.append({"tag": tag, "approach": "iv_rv_ratio", **r, "delta": round(d, 3)})
        marker = "+" if d > 0.01 else " "
        print(f"    {tag:50s} avg={r['avg_sharpe']:.3f} Δ={d:+.3f} {marker}")

    print()

    # ── 4. Combined: Carry + IV Sizing ────────────────────────────────────
    print("--- APPROACH 4: Carry Sizing + IV Sizing (R25) ---")

    combo_configs = [
        ("carry_pct", {"lookback": 180, "low_pct": 0.25, "high_pct": 0.75, "low_scale": 0.5, "high_scale": 1.5}, "both"),
        ("carry_pct", {"lookback": 90, "low_pct": 0.25, "high_pct": 0.75, "low_scale": 0.5, "high_scale": 1.5}, "vrp_only"),
        ("carry_z", {"lookback": 180, "low_z": -1.0, "high_z": 1.0, "low_scale": 0.5, "high_scale": 1.5}, "both"),
        ("carry_z", {"lookback": 90, "low_z": -1.0, "high_z": 1.0, "low_scale": 0.5, "high_scale": 1.5}, "vrp_only"),
        ("iv_rv_ratio", {"lookback": 90, "low_ratio": 0.8, "high_ratio": 1.5, "low_scale": 0.5, "high_scale": 1.5}, "both"),
    ]

    combo_results = []
    for mode, params, apply in combo_configs:
        tag = f"combo_{mode}_{apply}+IV"
        r = run_ensemble(sizing_mode=mode, sizing_params=params, apply_to=apply, use_iv_sizing=True)
        d_base = r["avg_sharpe"] - baseline["avg_sharpe"]
        d_iv = r["avg_sharpe"] - iv_sized["avg_sharpe"]
        combo_results.append({
            "tag": tag, **r,
            "delta_vs_base": round(d_base, 3),
            "delta_vs_iv": round(d_iv, 3),
        })
        marker = "+" if d_iv > 0 else " "
        print(f"    {tag:50s} avg={r['avg_sharpe']:.3f} Δbase={d_base:+.3f} ΔIV={d_iv:+.3f} {marker}")

    print()

    # ── SUMMARY ───────────────────────────────────────────────────────────
    print("=" * 70)
    print("SUMMARY — R27: VRP Carry Magnitude Sizing")
    print("=" * 70)
    print()
    print(f"  Baseline:            avg={baseline['avg_sharpe']:.3f} min={baseline['min_sharpe']:.3f}")
    print(f"  IV-sized (R25):      avg={iv_sized['avg_sharpe']:.3f} min={iv_sized['min_sharpe']:.3f}")
    print()

    n_better = sum(1 for r in all_results if r["delta"] > 0)
    n_worse = sum(1 for r in all_results if r["delta"] < 0)
    n_total = len(all_results)

    print(f"  Carry sizing configs: {n_total}")
    print(f"    Better than baseline: {n_better}/{n_total}")
    print(f"    Worse than baseline:  {n_worse}/{n_total}")
    print()

    # Best/worst
    best = max(all_results, key=lambda x: x["avg_sharpe"])
    worst = min(all_results, key=lambda x: x["avg_sharpe"])
    print(f"  Best:  {best['tag']:50s} avg={best['avg_sharpe']:.3f} Δ={best['delta']:+.3f}")
    print(f"  Worst: {worst['tag']:50s} avg={worst['avg_sharpe']:.3f} Δ={worst['delta']:+.3f}")
    print()

    # By approach
    print("  BY APPROACH:")
    for approach in ["carry_pct", "carry_z", "iv_rv_ratio"]:
        subset = [r for r in all_results if r["approach"] == approach]
        if subset:
            avg_d = sum(r["delta"] for r in subset) / len(subset)
            b = max(subset, key=lambda x: x["delta"])
            n_b = sum(1 for r in subset if r["delta"] > 0)
            print(f"    {approach:15s} avg_Δ={avg_d:+.3f} {n_b}/{len(subset)} better  best: Δ={b['delta']:+.3f}")
    print()

    # Compare best carry sizing vs IV sizing
    if best["avg_sharpe"] > iv_sized["avg_sharpe"]:
        print(f"  Carry sizing ({best['avg_sharpe']:.3f}) > IV sizing ({iv_sized['avg_sharpe']:.3f})")
    elif best["avg_sharpe"] > baseline["avg_sharpe"]:
        print(f"  Carry sizing ({best['avg_sharpe']:.3f}) < IV sizing ({iv_sized['avg_sharpe']:.3f}) but > baseline ({baseline['avg_sharpe']:.3f})")
    else:
        print(f"  Carry sizing ({best['avg_sharpe']:.3f}) <= baseline ({baseline['avg_sharpe']:.3f})")

    # Combined results
    if combo_results:
        print()
        print("  COMBINED (carry + IV sizing):")
        best_c = max(combo_results, key=lambda x: x["avg_sharpe"])
        for r in combo_results:
            marker = "***" if r["tag"] == best_c["tag"] else ""
            print(f"    {r['tag']:50s} avg={r['avg_sharpe']:.3f} ΔIV={r['delta_vs_iv']:+.3f} {marker}")

        if best_c["delta_vs_iv"] > 0.05:
            print(f"\n  Combined > IV-only by {best_c['delta_vs_iv']:+.3f} — carry sizing is ADDITIVE")
        elif best_c["delta_vs_iv"] > 0:
            print(f"\n  Combined > IV-only by {best_c['delta_vs_iv']:+.3f} — marginal improvement")
        else:
            print(f"\n  Combined <= IV-only — carry sizing is REDUNDANT with IV sizing")

    # Verdict
    print()
    print("=" * 70)
    if n_better > n_total * 0.6 and best["delta"] > 0.1:
        print(f"VERDICT: Carry sizing IMPROVES ensemble (Δ={best['delta']:+.3f})")
    elif n_better > n_total * 0.3:
        print(f"VERDICT: Carry sizing is MIXED — some configs help ({n_better}/{n_total})")
    elif n_better > 0:
        print(f"VERDICT: Carry sizing is MARGINAL — few configs help ({n_better}/{n_total})")
    else:
        print(f"VERDICT: Carry sizing DOES NOT improve ensemble (0/{n_total})")

    # Compare with IV sizing
    if best["avg_sharpe"] > iv_sized["avg_sharpe"]:
        print(f"  Carry sizing BEATS IV sizing: {best['avg_sharpe']:.3f} vs {iv_sized['avg_sharpe']:.3f}")
        print(f"  Recommend: replace IV sizing with carry sizing")
    else:
        print(f"  IV sizing remains champion: {iv_sized['avg_sharpe']:.3f} vs carry best {best['avg_sharpe']:.3f}")
        print(f"  R25 step_lb180 is still the best sizing approach")
    print("=" * 70)

    # Save
    out = ROOT / "artifacts" / "crypto_options" / "vrp_carry_sizing_research.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump({
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "research_id": "R27",
            "baseline": baseline,
            "iv_sized_baseline": iv_sized,
            "n_configs": n_total,
            "n_better": n_better,
            "n_worse": n_worse,
            "best": {"tag": best["tag"], "avg_sharpe": best["avg_sharpe"], "delta": best["delta"]},
            "all_results": all_results,
            "combined_results": combo_results,
        }, f, indent=2)
    print(f"\nResults saved to: {out}")


if __name__ == "__main__":
    main()
