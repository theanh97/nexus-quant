#!/usr/bin/env python3
"""
Per-Asset Parameter Optimization for Multi-Asset Ensemble — R35
=================================================================

R31 found BTC+ETH multi-asset ensemble gives avg=3.174 (+0.989 vs BTC-only).
But used IDENTICAL params for both assets.

ETH has:
  - Higher base IV (64.9% vs 47.8% for BTC on 2026-02-20)
  - Higher VRP spread (synthetic: +10% vs +8%)
  - Higher skew values
  - Different vol dynamics

Questions:
  1. Should ETH VRP use different leverage/lookback than BTC?
  2. Should ETH Skew use different lookback/thresholds?
  3. Should IV sizing thresholds differ per asset?
  4. Does cross-asset IV signal improve sizing? (BTC IV → ETH sizing)
  5. Optimized per-asset params vs uniform: how much improvement?

Sweep plan:
  A. ETH VRP leverage sweep (BTC at 1.5)
  B. ETH VRP lookback sweep (BTC at 30)
  C. ETH Skew lookback sweep (BTC at 60)
  D. Per-asset IV sizing thresholds
  E. Cross-asset IV sizing
  F. Full optimized multi-asset ensemble
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

# Default params (R28-optimized for BTC)
BTC_VRP_PARAMS = {
    "base_leverage": 1.5, "exit_z_threshold": -3.0,
    "vrp_lookback": 30, "rebalance_freq": 5, "min_bars": 30,
}
BTC_SKEW_PARAMS = {
    "skew_lookback": 60, "z_entry": 2.0, "z_exit": 0.0,
    "target_leverage": 1.0, "rebalance_freq": 5, "min_bars": 60,
}

IV_LOOKBACK = 180
IV_PCT_LOW = 0.25
IV_PCT_HIGH = 0.75
IV_SCALE_LOW = 0.50
IV_SCALE_HIGH = 1.70


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
    return sum(1 for v in window if v < current) / len(window)


def iv_sizing_scale(pct: float, low_thresh=0.25, high_thresh=0.75,
                    low_scale=0.50, high_scale=1.70) -> float:
    if pct < low_thresh:
        return low_scale
    elif pct > high_thresh:
        return high_scale
    return 1.0


def run_multi_asset(
    btc_vrp_params: dict,
    btc_skew_params: dict,
    eth_vrp_params: dict,
    eth_skew_params: dict,
    w_btc: float = 0.50,
    w_eth: float = 0.50,
    use_iv_sizing: bool = True,
    iv_config: Optional[Dict] = None,
    cross_asset_iv: bool = False,
) -> Dict[str, Any]:
    """Run multi-asset ensemble with per-asset params."""
    sharpes = []
    yearly_detail = {}

    # Per-asset IV sizing config
    if iv_config is None:
        iv_config = {
            "BTC": {"lookback": 180, "low_thresh": 0.25, "high_thresh": 0.75,
                     "low_scale": 0.50, "high_scale": 1.70},
            "ETH": {"lookback": 180, "low_thresh": 0.25, "high_thresh": 0.75,
                     "low_scale": 0.50, "high_scale": 1.70},
        }

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

        # Per-asset strategy instances
        strats = {
            "BTC": {
                "vrp": VariancePremiumStrategy(params=btc_vrp_params),
                "skew": SkewTradeV2Strategy(params=btc_skew_params),
            },
            "ETH": {
                "vrp": VariancePremiumStrategy(params=eth_vrp_params),
                "skew": SkewTradeV2Strategy(params=eth_skew_params),
            },
        }

        dt = 1.0 / BARS_PER_YEAR
        equity = 1.0
        vrp_weights = {"BTC": 0.0, "ETH": 0.0}
        skew_weights = {"BTC": 0.0, "ETH": 0.0}
        equity_curve = [1.0]
        returns_list = []

        asset_weights = {"BTC": w_btc, "ETH": w_eth}

        for idx in range(1, n):
            prev_equity = equity
            total_pnl = 0.0

            for sym in ["BTC", "ETH"]:
                aw = asset_weights[sym]

                # VRP P&L
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

                # Skew P&L
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

                total_pnl += aw * (W_VRP * vrp_pnl + W_SKEW * skew_pnl)

            equity += equity * total_pnl

            # Rebalance — check per-asset
            rebal_happened = False
            for sym in ["BTC", "ETH"]:
                vrp_rebal = strats[sym]["vrp"].should_rebalance(dataset, idx)
                skew_rebal = strats[sym]["skew"].should_rebalance(dataset, idx)

                if vrp_rebal or skew_rebal:
                    rebal_happened = True
                    old_v = vrp_weights.get(sym, 0.0)
                    old_s = skew_weights.get(sym, 0.0)

                    if vrp_rebal:
                        target_v = strats[sym]["vrp"].target_weights(dataset, idx, {sym: vrp_weights.get(sym, 0.0)})
                        vrp_weights[sym] = target_v.get(sym, 0.0)
                    if skew_rebal:
                        target_s = strats[sym]["skew"].target_weights(dataset, idx, {sym: skew_weights.get(sym, 0.0)})
                        skew_weights[sym] = target_s.get(sym, 0.0)

                    # Apply IV sizing
                    if use_iv_sizing:
                        ic = iv_config.get(sym, iv_config.get("BTC", {}))

                        if cross_asset_iv:
                            # Use OTHER asset's IV for sizing
                            other_sym = "ETH" if sym == "BTC" else "BTC"
                            iv_series = dataset.features.get("iv_atm", {}).get(other_sym, [])
                        else:
                            iv_series = dataset.features.get("iv_atm", {}).get(sym, [])

                        pct = iv_percentile(iv_series, idx, lookback=ic.get("lookback", 180))
                        if pct is not None:
                            scale = iv_sizing_scale(
                                pct,
                                low_thresh=ic.get("low_thresh", 0.25),
                                high_thresh=ic.get("high_thresh", 0.75),
                                low_scale=ic.get("low_scale", 0.50),
                                high_scale=ic.get("high_scale", 1.70),
                            )
                            vrp_weights[sym] *= scale
                            skew_weights[sym] *= scale

            if rebal_happened:
                # Approximate turnover cost
                turnover = 0.05
                bd = COSTS.cost(equity=equity, turnover=turnover)
                cost = float(bd.get("cost", 0.0))
                equity -= cost
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
    print("PER-ASSET PARAMETER OPTIMIZATION — R35")
    print("=" * 70)
    print()

    # ── BASELINE: uniform params, 50/50 BTC/ETH ────────────────────────────
    print("--- BASELINE: Uniform params, 50/50 BTC/ETH, IV-sized ---")
    baseline = run_multi_asset(
        BTC_VRP_PARAMS, BTC_SKEW_PARAMS,
        BTC_VRP_PARAMS, BTC_SKEW_PARAMS,  # ETH uses BTC params
        w_btc=0.50, w_eth=0.50, use_iv_sizing=True,
    )
    print(f"  Baseline: avg={baseline['avg_sharpe']:.3f} min={baseline['min_sharpe']:.3f}")
    print(f"  Yearly: {baseline['yearly']}")
    print()

    # ── A. ETH VRP Leverage Sweep ──────────────────────────────────────────
    print("--- A. ETH VRP LEVERAGE SWEEP (BTC=1.5 fixed) ---")
    eth_lev_results = []
    for lev in [0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5]:
        eth_vrp = {**BTC_VRP_PARAMS, "base_leverage": lev}
        r = run_multi_asset(
            BTC_VRP_PARAMS, BTC_SKEW_PARAMS,
            eth_vrp, BTC_SKEW_PARAMS,
            w_btc=0.50, w_eth=0.50, use_iv_sizing=True,
        )
        d = r["avg_sharpe"] - baseline["avg_sharpe"]
        eth_lev_results.append({"eth_vrp_lev": lev, **r, "delta": round(d, 3)})
        marker = " *" if lev == 1.5 else ""
        print(f"  ETH VRP lev={lev:.2f}  avg={r['avg_sharpe']:.3f} min={r['min_sharpe']:.3f} Δ={d:+.3f}{marker}")

    best_lev = max(eth_lev_results, key=lambda x: x["avg_sharpe"])
    print(f"\n  Best ETH VRP lev: {best_lev['eth_vrp_lev']:.2f} avg={best_lev['avg_sharpe']:.3f}")
    print()

    # ── B. ETH VRP Lookback Sweep ──────────────────────────────────────────
    print("--- B. ETH VRP LOOKBACK SWEEP (BTC lb=30 fixed) ---")
    eth_lb_results = []
    for lb in [15, 20, 30, 45, 60, 90]:
        eth_vrp = {**BTC_VRP_PARAMS, "vrp_lookback": lb, "min_bars": lb}
        r = run_multi_asset(
            BTC_VRP_PARAMS, BTC_SKEW_PARAMS,
            eth_vrp, BTC_SKEW_PARAMS,
            w_btc=0.50, w_eth=0.50, use_iv_sizing=True,
        )
        d = r["avg_sharpe"] - baseline["avg_sharpe"]
        eth_lb_results.append({"eth_vrp_lb": lb, **r, "delta": round(d, 3)})
        marker = " *" if lb == 30 else ""
        print(f"  ETH VRP lb={lb:3d}  avg={r['avg_sharpe']:.3f} min={r['min_sharpe']:.3f} Δ={d:+.3f}{marker}")

    best_lb = max(eth_lb_results, key=lambda x: x["avg_sharpe"])
    print(f"\n  Best ETH VRP lb: {best_lb['eth_vrp_lb']} avg={best_lb['avg_sharpe']:.3f}")
    print()

    # ── C. ETH Skew Lookback Sweep ─────────────────────────────────────────
    print("--- C. ETH SKEW LOOKBACK SWEEP (BTC lb=60 fixed) ---")
    eth_skew_lb_results = []
    for lb in [30, 45, 60, 90, 120]:
        eth_skew = {**BTC_SKEW_PARAMS, "skew_lookback": lb, "min_bars": lb}
        r = run_multi_asset(
            BTC_VRP_PARAMS, BTC_SKEW_PARAMS,
            BTC_VRP_PARAMS, eth_skew,
            w_btc=0.50, w_eth=0.50, use_iv_sizing=True,
        )
        d = r["avg_sharpe"] - baseline["avg_sharpe"]
        eth_skew_lb_results.append({"eth_skew_lb": lb, **r, "delta": round(d, 3)})
        marker = " *" if lb == 60 else ""
        print(f"  ETH Skew lb={lb:3d}  avg={r['avg_sharpe']:.3f} min={r['min_sharpe']:.3f} Δ={d:+.3f}{marker}")

    best_skew_lb = max(eth_skew_lb_results, key=lambda x: x["avg_sharpe"])
    print(f"\n  Best ETH Skew lb: {best_skew_lb['eth_skew_lb']} avg={best_skew_lb['avg_sharpe']:.3f}")
    print()

    # ── D. Per-Asset IV Sizing Thresholds ──────────────────────────────────
    print("--- D. PER-ASSET IV SIZING THRESHOLDS ---")
    iv_configs_to_test = [
        ("uniform 25/75", {
            "BTC": {"lookback": 180, "low_thresh": 0.25, "high_thresh": 0.75, "low_scale": 0.50, "high_scale": 1.70},
            "ETH": {"lookback": 180, "low_thresh": 0.25, "high_thresh": 0.75, "low_scale": 0.50, "high_scale": 1.70},
        }),
        ("ETH wider 20/80", {
            "BTC": {"lookback": 180, "low_thresh": 0.25, "high_thresh": 0.75, "low_scale": 0.50, "high_scale": 1.70},
            "ETH": {"lookback": 180, "low_thresh": 0.20, "high_thresh": 0.80, "low_scale": 0.50, "high_scale": 1.70},
        }),
        ("ETH narrower 30/70", {
            "BTC": {"lookback": 180, "low_thresh": 0.25, "high_thresh": 0.75, "low_scale": 0.50, "high_scale": 1.70},
            "ETH": {"lookback": 180, "low_thresh": 0.30, "high_thresh": 0.70, "low_scale": 0.50, "high_scale": 1.70},
        }),
        ("ETH higher scale 0.5/2.0", {
            "BTC": {"lookback": 180, "low_thresh": 0.25, "high_thresh": 0.75, "low_scale": 0.50, "high_scale": 1.70},
            "ETH": {"lookback": 180, "low_thresh": 0.25, "high_thresh": 0.75, "low_scale": 0.50, "high_scale": 2.00},
        }),
        ("ETH lower scale 0.5/1.5", {
            "BTC": {"lookback": 180, "low_thresh": 0.25, "high_thresh": 0.75, "low_scale": 0.50, "high_scale": 1.70},
            "ETH": {"lookback": 180, "low_thresh": 0.25, "high_thresh": 0.75, "low_scale": 0.50, "high_scale": 1.50},
        }),
        ("ETH shorter lb=120", {
            "BTC": {"lookback": 180, "low_thresh": 0.25, "high_thresh": 0.75, "low_scale": 0.50, "high_scale": 1.70},
            "ETH": {"lookback": 120, "low_thresh": 0.25, "high_thresh": 0.75, "low_scale": 0.50, "high_scale": 1.70},
        }),
        ("both aggressive 0.3/2.0", {
            "BTC": {"lookback": 180, "low_thresh": 0.25, "high_thresh": 0.75, "low_scale": 0.30, "high_scale": 2.00},
            "ETH": {"lookback": 180, "low_thresh": 0.25, "high_thresh": 0.75, "low_scale": 0.30, "high_scale": 2.00},
        }),
        ("ETH aggressive BTC conserv", {
            "BTC": {"lookback": 180, "low_thresh": 0.25, "high_thresh": 0.75, "low_scale": 0.50, "high_scale": 1.70},
            "ETH": {"lookback": 180, "low_thresh": 0.25, "high_thresh": 0.75, "low_scale": 0.30, "high_scale": 2.00},
        }),
    ]

    iv_results = []
    for label, ic in iv_configs_to_test:
        r = run_multi_asset(
            BTC_VRP_PARAMS, BTC_SKEW_PARAMS,
            BTC_VRP_PARAMS, BTC_SKEW_PARAMS,
            w_btc=0.50, w_eth=0.50, use_iv_sizing=True, iv_config=ic,
        )
        d = r["avg_sharpe"] - baseline["avg_sharpe"]
        iv_results.append({"label": label, **r, "delta": round(d, 3)})
        marker = " *" if label == "uniform 25/75" else ""
        print(f"  {label:30s} avg={r['avg_sharpe']:.3f} min={r['min_sharpe']:.3f} Δ={d:+.3f}{marker}")

    best_iv = max(iv_results, key=lambda x: x["avg_sharpe"])
    print(f"\n  Best IV config: {best_iv['label']} avg={best_iv['avg_sharpe']:.3f}")
    print()

    # ── E. Cross-Asset IV Sizing ───────────────────────────────────────────
    print("--- E. CROSS-ASSET IV SIZING ---")
    print("  (Use OTHER asset's IV percentile for sizing)")

    cross_configs = [
        ("own IV (baseline)", False),
        ("cross-asset IV", True),
    ]
    cross_results = []
    for label, cross in cross_configs:
        r = run_multi_asset(
            BTC_VRP_PARAMS, BTC_SKEW_PARAMS,
            BTC_VRP_PARAMS, BTC_SKEW_PARAMS,
            w_btc=0.50, w_eth=0.50, use_iv_sizing=True,
            cross_asset_iv=cross,
        )
        d = r["avg_sharpe"] - baseline["avg_sharpe"]
        cross_results.append({"label": label, "cross": cross, **r, "delta": round(d, 3)})
        print(f"  {label:30s} avg={r['avg_sharpe']:.3f} min={r['min_sharpe']:.3f} Δ={d:+.3f}")
    print()

    # ── F. Optimized Multi-Asset Ensemble ──────────────────────────────────
    print("--- F. OPTIMIZED MULTI-ASSET ENSEMBLE ---")
    print("  (Best params from A-E combined)")

    # Use best ETH VRP leverage from A
    best_eth_lev_val = best_lev["eth_vrp_lev"]
    # Use best ETH VRP lookback from B
    best_eth_lb_val = best_lb["eth_vrp_lb"]
    # Use best ETH Skew lookback from C
    best_eth_skew_lb_val = best_skew_lb["eth_skew_lb"]

    opt_eth_vrp = {
        **BTC_VRP_PARAMS,
        "base_leverage": best_eth_lev_val,
        "vrp_lookback": best_eth_lb_val,
        "min_bars": best_eth_lb_val,
    }
    opt_eth_skew = {
        **BTC_SKEW_PARAMS,
        "skew_lookback": best_eth_skew_lb_val,
        "min_bars": best_eth_skew_lb_val,
    }

    # Also find best IV config
    best_iv_label = best_iv["label"]
    best_iv_config = None
    for label, ic in iv_configs_to_test:
        if label == best_iv_label:
            best_iv_config = ic
            break

    # Test different BTC/ETH allocations with optimized params
    print(f"\n  ETH VRP: lev={best_eth_lev_val}, lb={best_eth_lb_val}")
    print(f"  ETH Skew: lb={best_eth_skew_lb_val}")
    print(f"  IV config: {best_iv_label}")
    print()

    alloc_configs = [
        (0.60, 0.40, "60/40"),
        (0.50, 0.50, "50/50"),
        (0.40, 0.60, "40/60"),
        (0.30, 0.70, "30/70"),
    ]

    optimized_results = []
    for wb, we, label in alloc_configs:
        r = run_multi_asset(
            BTC_VRP_PARAMS, BTC_SKEW_PARAMS,
            opt_eth_vrp, opt_eth_skew,
            w_btc=wb, w_eth=we, use_iv_sizing=True,
            iv_config=best_iv_config,
        )
        d = r["avg_sharpe"] - baseline["avg_sharpe"]
        optimized_results.append({"label": label, "w_btc": wb, "w_eth": we, **r, "delta": round(d, 3)})
        print(f"  {label:8s}  avg={r['avg_sharpe']:.3f} min={r['min_sharpe']:.3f} Δ={d:+.3f}")

    best_opt = max(optimized_results, key=lambda x: x["avg_sharpe"])

    # Also test uniform params for comparison
    uniform_same_alloc = run_multi_asset(
        BTC_VRP_PARAMS, BTC_SKEW_PARAMS,
        BTC_VRP_PARAMS, BTC_SKEW_PARAMS,
        w_btc=best_opt["w_btc"], w_eth=best_opt["w_eth"],
        use_iv_sizing=True,
    )

    print()
    print(f"  Best optimized: {best_opt['label']} avg={best_opt['avg_sharpe']:.3f}")
    print(f"  Same alloc uniform: avg={uniform_same_alloc['avg_sharpe']:.3f}")
    print(f"  Per-asset gain: {best_opt['avg_sharpe'] - uniform_same_alloc['avg_sharpe']:+.3f}")
    print()

    # ── SUMMARY ───────────────────────────────────────────────────────────
    print("=" * 70)
    print("SUMMARY — R35: Per-Asset Parameter Optimization")
    print("=" * 70)
    print()

    print(f"  Baseline (uniform, 50/50): avg={baseline['avg_sharpe']:.3f} min={baseline['min_sharpe']:.3f}")
    print()

    print("  SWEEP RESULTS:")
    print(f"    A. Best ETH VRP lev:     {best_lev['eth_vrp_lev']:.2f} (Δ={best_lev['delta']:+.3f})")
    print(f"    B. Best ETH VRP lb:      {best_lb['eth_vrp_lb']}d (Δ={best_lb['delta']:+.3f})")
    print(f"    C. Best ETH Skew lb:     {best_skew_lb['eth_skew_lb']}d (Δ={best_skew_lb['delta']:+.3f})")
    print(f"    D. Best IV config:       {best_iv['label']} (Δ={best_iv['delta']:+.3f})")

    best_cross = max(cross_results, key=lambda x: x["avg_sharpe"])
    print(f"    E. Cross-asset IV:       {best_cross['label']} (Δ={best_cross['delta']:+.3f})")
    print()

    print(f"  OPTIMIZED ENSEMBLE:")
    print(f"    Allocation:     {best_opt['label']}")
    print(f"    avg Sharpe:     {best_opt['avg_sharpe']:.3f}")
    print(f"    min Sharpe:     {best_opt['min_sharpe']:.3f}")
    print(f"    vs baseline:    {best_opt['delta']:+.3f}")
    print(f"    per-asset gain: {best_opt['avg_sharpe'] - uniform_same_alloc['avg_sharpe']:+.3f}")
    print()

    # Verdict
    total_improvement = best_opt["avg_sharpe"] - baseline["avg_sharpe"]
    per_asset_gain = best_opt["avg_sharpe"] - uniform_same_alloc["avg_sharpe"]

    print("=" * 70)
    if per_asset_gain > 0.05:
        print(f"VERDICT: Per-asset optimization HELPS (+{per_asset_gain:.3f} from param tuning)")
        print(f"  ETH params: VRP lev={best_eth_lev_val}, VRP lb={best_eth_lb_val}, Skew lb={best_eth_skew_lb_val}")
    elif total_improvement > 0.05:
        print(f"VERDICT: Allocation matters more than per-asset params")
        print(f"  Total improvement Δ={total_improvement:+.3f}, param tuning only {per_asset_gain:+.3f}")
    else:
        print(f"VERDICT: Uniform params are near-optimal for multi-asset ensemble")
        print(f"  Per-asset tuning: {per_asset_gain:+.3f} (not significant)")
    print("=" * 70)

    # Save
    out = ROOT / "artifacts" / "crypto_options" / "per_asset_param_optimization.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump({
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "research_id": "R35",
            "baseline": baseline,
            "eth_vrp_leverage_sweep": eth_lev_results,
            "eth_vrp_lookback_sweep": eth_lb_results,
            "eth_skew_lookback_sweep": eth_skew_lb_results,
            "iv_sizing_configs": iv_results,
            "cross_asset_iv": cross_results,
            "optimized_ensemble": optimized_results,
            "best_optimized": {
                "allocation": best_opt["label"],
                "avg_sharpe": best_opt["avg_sharpe"],
                "min_sharpe": best_opt["min_sharpe"],
                "eth_vrp_leverage": best_eth_lev_val,
                "eth_vrp_lookback": best_eth_lb_val,
                "eth_skew_lookback": best_eth_skew_lb_val,
                "iv_config": best_iv_label,
            },
        }, f, indent=2)
    print(f"\nResults saved to: {out}")


if __name__ == "__main__":
    main()
