#!/usr/bin/env python3
"""
Transaction Cost Sensitivity & Production Readiness Audit — R40
================================================================

All backtests used fixed costs:
  maker_fee=0.03%, taker_fee=0.05%, impact=2bps sqrt
Real Deribit costs may differ. How sensitive is the ensemble?

Tests:
  A. Fee sensitivity: 0x to 5x baseline fees
  B. Impact sensitivity: 0-10bps impact coefficient
  C. Turnover sensitivity: what if rebalancing causes more turnover?
  D. Combined worst-case: 3x fees + 5bps impact
  E. Production configuration summary
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


def run_with_costs(maker_rate, taker_rate, impact_bps, turnover_mult=1.0):
    """Run multi-asset ensemble with specified cost model."""
    fees = FeeModel(maker_fee_rate=maker_rate, taker_fee_rate=taker_rate)
    impact = ImpactModel(model="sqrt", coef_bps=impact_bps)
    costs = ExecutionCostModel(fee=fees, impact=impact)

    sharpes = []
    yearly_detail = {}
    total_costs_pct = 0.0
    total_trades = 0

    for yr in YEARS:
        cfg = {
            "symbols": ["BTC", "ETH"],
            "start": f"{yr}-01-01", "end": f"{yr}-12-31",
            "bar_interval": "1d", "use_synthetic_iv": True,
        }
        provider = DeribitRestProvider(cfg, seed=SEED)
        dataset = provider.load()
        n = len(dataset.timeline)

        strats = {
            "BTC": {"vrp": VariancePremiumStrategy(params=BTC_VRP), "skew": SkewTradeV2Strategy(params=BTC_SKEW)},
            "ETH": {"vrp": VariancePremiumStrategy(params=ETH_VRP), "skew": SkewTradeV2Strategy(params=ETH_SKEW)},
        }

        dt = 1.0 / BARS_PER_YEAR
        equity = 1.0
        vrp_w = {"BTC": 0.0, "ETH": 0.0}
        skew_w = {"BTC": 0.0, "ETH": 0.0}
        equity_curve = [1.0]
        returns_list = []
        yr_costs = 0.0
        yr_trades = 0

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
                    ivs = dataset.features.get("iv_atm", {}).get(sym, [])
                    pct = iv_percentile(ivs, idx)
                    if pct is not None:
                        sc = iv_sizing_scale(pct)
                        vrp_w[sym] *= sc
                        skew_w[sym] *= sc

            if rebal_happened:
                turnover = 0.05 * turnover_mult
                bd = costs.cost(equity=equity, turnover=turnover)
                cost = float(bd.get("cost", 0.0))
                equity -= cost
                equity = max(equity, 0.0)
                yr_costs += cost
                yr_trades += 1

            equity_curve.append(equity)
            bar_ret = (equity / prev_equity) - 1.0 if prev_equity > 0 else 0.0
            returns_list.append(bar_ret)

        m = compute_metrics(equity_curve, returns_list, BARS_PER_YEAR)
        sharpes.append(m["sharpe"])
        yearly_detail[str(yr)] = round(m["sharpe"], 3)
        total_costs_pct += yr_costs / equity * 100
        total_trades += yr_trades

    avg = sum(sharpes) / len(sharpes)
    mn = min(sharpes)
    avg_cost_pct = total_costs_pct / len(YEARS)
    avg_trades = total_trades / len(YEARS)
    return {
        "avg_sharpe": round(avg, 3),
        "min_sharpe": round(mn, 3),
        "yearly": yearly_detail,
        "avg_annual_cost_pct": round(avg_cost_pct, 3),
        "avg_trades_per_year": round(avg_trades, 1),
    }


def main():
    print("=" * 70)
    print("TRANSACTION COST SENSITIVITY & PRODUCTION AUDIT — R40")
    print("=" * 70)
    print()

    # Baseline
    baseline = run_with_costs(0.0003, 0.0005, 2.0)
    zero_cost = run_with_costs(0.0, 0.0, 0.0)
    print(f"  Zero-cost ensemble:    avg={zero_cost['avg_sharpe']:.3f} min={zero_cost['min_sharpe']:.3f}")
    print(f"  Baseline costs:        avg={baseline['avg_sharpe']:.3f} min={baseline['min_sharpe']:.3f}")
    print(f"  Cost drag:             Δ={baseline['avg_sharpe'] - zero_cost['avg_sharpe']:+.3f}")
    print(f"  Avg annual cost:       {baseline['avg_annual_cost_pct']:.3f}%")
    print(f"  Avg trades/year:       {baseline['avg_trades_per_year']:.0f}")
    print()

    results = {"baseline": baseline, "zero_cost": zero_cost}

    # ── A. Fee Sensitivity ─────────────────────────────────────────────────
    print("--- A. FEE SENSITIVITY ---")
    print("  (Scale maker/taker fees from 0x to 5x baseline)")

    fee_results = []
    for mult in [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0]:
        r = run_with_costs(0.0003 * mult, 0.0005 * mult, 2.0)
        d = r["avg_sharpe"] - baseline["avg_sharpe"]
        fee_results.append({"fee_mult": mult, **r, "delta": round(d, 3)})
        marker = " *" if mult == 1.0 else ""
        print(f"  fees×{mult:.1f}: avg={r['avg_sharpe']:.3f} min={r['min_sharpe']:.3f} cost={r['avg_annual_cost_pct']:.3f}% Δ={d:+.3f}{marker}")

    results["fee_sensitivity"] = fee_results
    print()

    # ── B. Impact Sensitivity ──────────────────────────────────────────────
    print("--- B. IMPACT SENSITIVITY ---")
    print("  (Impact coefficient from 0 to 10bps)")

    impact_results = []
    for bps in [0.0, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0]:
        r = run_with_costs(0.0003, 0.0005, bps)
        d = r["avg_sharpe"] - baseline["avg_sharpe"]
        impact_results.append({"impact_bps": bps, **r, "delta": round(d, 3)})
        marker = " *" if bps == 2.0 else ""
        print(f"  impact={bps:.0f}bps: avg={r['avg_sharpe']:.3f} min={r['min_sharpe']:.3f} Δ={d:+.3f}{marker}")

    results["impact_sensitivity"] = impact_results
    print()

    # ── C. Turnover Sensitivity ────────────────────────────────────────────
    print("--- C. TURNOVER SENSITIVITY ---")
    print("  (Turnover multiplier 0.5x to 5x)")

    turnover_results = []
    for mult in [0.5, 1.0, 1.5, 2.0, 3.0, 5.0]:
        r = run_with_costs(0.0003, 0.0005, 2.0, turnover_mult=mult)
        d = r["avg_sharpe"] - baseline["avg_sharpe"]
        turnover_results.append({"turnover_mult": mult, **r, "delta": round(d, 3)})
        marker = " *" if mult == 1.0 else ""
        print(f"  turnover×{mult:.1f}: avg={r['avg_sharpe']:.3f} min={r['min_sharpe']:.3f} cost={r['avg_annual_cost_pct']:.3f}% Δ={d:+.3f}{marker}")

    results["turnover_sensitivity"] = turnover_results
    print()

    # ── D. Worst-Case Scenarios ────────────────────────────────────────────
    print("--- D. WORST-CASE SCENARIOS ---")

    worst_cases = [
        ("2x fees + 3bps impact", 0.0006, 0.0010, 3.0, 1.0),
        ("3x fees + 5bps impact", 0.0009, 0.0015, 5.0, 1.0),
        ("2x fees + 2x turnover", 0.0006, 0.0010, 2.0, 2.0),
        ("3x fees + 5bps + 2x turn", 0.0009, 0.0015, 5.0, 2.0),
        ("5x everything", 0.0015, 0.0025, 10.0, 5.0),
    ]

    worst_results = []
    for label, mk, tk, bps, tm in worst_cases:
        r = run_with_costs(mk, tk, bps, turnover_mult=tm)
        d = r["avg_sharpe"] - baseline["avg_sharpe"]
        worst_results.append({"label": label, **r, "delta": round(d, 3)})
        survives = "PASS" if r["min_sharpe"] > 1.0 else "FAIL"
        print(f"  {label:30s} avg={r['avg_sharpe']:.3f} min={r['min_sharpe']:.3f} cost={r['avg_annual_cost_pct']:.3f}% [{survives}]")

    results["worst_case"] = worst_results
    print()

    # ── E. Production Configuration Summary ────────────────────────────────
    print("=" * 70)
    print("PRODUCTION CONFIGURATION SUMMARY — R40")
    print("=" * 70)
    print()

    print("  STRATEGY ENSEMBLE:")
    print("    VRP 40% + Skew MR 60% (mixed-frequency)")
    print("    BTC: VRP lb=30d, Skew lb=60d")
    print("    ETH: VRP lb=60d (R35), Skew lb=90d")
    print("    IV sizing: step 0.5/1.0/1.7x at 25/75th pct, 180d lookback")
    print("    Rebalance: every 5 days")
    print("    Leverage: VRP=1.5x, Skew=1.0x")
    print()

    print("  MULTI-ASSET ALLOCATION:")
    print("    BTC 50% + ETH 50% (static)")
    print("    Dynamic allocation marginal (R38)")
    print()

    print("  BACKTEST RESULTS (R35 params, conservative IV sizing):")
    print(f"    avg Sharpe: {baseline['avg_sharpe']:.3f}")
    print(f"    min Sharpe: {baseline['min_sharpe']:.3f}")
    print(f"    Yearly: {baseline['yearly']}")
    print()

    print("  COST ROBUSTNESS:")
    # Find break-even fee multiplier (where Sharpe drops below 1.0)
    for fr in fee_results:
        if fr["min_sharpe"] < 1.0:
            print(f"    Break-even fee mult: ~{fr['fee_mult']:.1f}x")
            break
    else:
        print(f"    Sharpe stays > 1.0 even at 5x fees!")

    for wr in worst_results:
        if wr["min_sharpe"] > 1.0:
            print(f"    Worst surviving scenario: {wr['label']} (min={wr['min_sharpe']:.3f})")
        else:
            print(f"    Fails: {wr['label']} (min={wr['min_sharpe']:.3f})")

    print()
    print("  OUTSTANDING ITEMS:")
    print("    [ ] Real Deribit data validation (~April 2026, 60+ days)")
    print("    [ ] Validate IV sizing with real IV dynamics")
    print("    [ ] Multi-expiry VRP with real term structure")
    print("    [ ] Production monitoring/alerting framework")
    print()

    # Overall verdict
    print("=" * 70)
    survives_3x = any(w["label"] == "3x fees + 5bps impact" and w["min_sharpe"] > 1.0 for w in worst_results)
    if survives_3x:
        print("VERDICT: Ensemble is COST-ROBUST — survives 3x fee + 5bps impact")
    else:
        print("VERDICT: Ensemble is COST-SENSITIVE — may degrade with higher real costs")
    print("=" * 70)

    # Save
    out = ROOT / "artifacts" / "crypto_options" / "cost_sensitivity_production_audit.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump({
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "research_id": "R40",
            "results": results,
            "production_config": {
                "ensemble": "VRP 40% + Skew MR 60%",
                "btc_vrp_lb": 30, "eth_vrp_lb": 60,
                "btc_skew_lb": 60, "eth_skew_lb": 90,
                "iv_sizing": "step 0.5/1.0/1.7x at 25/75th pct, lb=180",
                "rebalance_freq": 5,
                "asset_allocation": "50/50 BTC/ETH (static)",
                "avg_sharpe": baseline["avg_sharpe"],
                "min_sharpe": baseline["min_sharpe"],
            },
        }, f, indent=2)
    print(f"\nResults saved to: {out}")


if __name__ == "__main__":
    main()
