#!/usr/bin/env python3
"""
Monte Carlo Vol Shock Stress Test
===================================

Tests the VRP + Skew MR ensemble under synthetic extreme vol scenarios
that exceed historical experience. The historical tail test showed MDD -1.51%
across 2021-2025 — but what if we get a COVID-scale or worse event?

Scenarios injected into actual backtest data:
  1. IV doubles (×2) with -10% crash over 3 days
  2. IV triples (×3) with -20% crash over 5 days (COVID-March 2020 style)
  3. IV ×4 with -30% crash over 7 days (black swan)
  4. Vol spike WITHOUT crash: IV doubles but price stays flat (fear without move)
  5. Sustained vol: IV stays elevated (×2) for 30 days then normalizes

For each scenario:
  - Run 200 Monte Carlo paths with random shock injection timing
  - Measure: MDD distribution, worst-case loss, recovery time
  - Compare VRP alone vs ensemble
  - Determine if ensemble SURVIVES or BREAKS

Key insight we're testing: VRP short vol PnL = 0.5 * (IV² - RV²) * dt
  - When IV doubles but RV quadruples (crash): massive loss
  - When IV doubles and price is flat (fear only): VRP profits more
  - Ensemble with Skew MR should cushion via uncorrelated returns
"""
import json
import math
import os
import random
import sys
import time
from typing import Any, Dict, List, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nexus_quant.backtest.costs import ExecutionCostModel, FeeModel, ImpactModel
from nexus_quant.projects.crypto_options.options_engine import (
    OptionsBacktestEngine, compute_metrics,
)
from nexus_quant.projects.crypto_options.providers.deribit_rest import DeribitRestProvider
from nexus_quant.projects.crypto_options.strategies.variance_premium import VariancePremiumStrategy
from nexus_quant.projects.crypto_options.strategies.skew_trade_v2 import SkewTradeV2Strategy

_FEE = FeeModel(maker_fee_rate=0.0003, taker_fee_rate=0.0005)
_IMPACT = ImpactModel(model="sqrt", coef_bps=3.0)
COSTS = ExecutionCostModel(
    fee=_FEE, execution_style="taker",
    slippage_bps=7.5, spread_bps=10.0, impact=_IMPACT, cost_multiplier=1.0,
)

DAILY_CFG = {
    "provider": "deribit_rest_v1",
    "symbols": ["BTC", "ETH"],
    "bar_interval": "1d",
    "cache_dir": "data/cache/deribit",
    "use_synthetic_iv": True,
    "rv_lookback_bars": 21,
}

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
    "use_vrp_filter": False,
}


# ── Vol Shock Scenarios ─────────────────────────────────────────────

SCENARIOS = {
    "crash_10pct_iv2x": {
        "description": "IV doubles, -10% crash over 3 days",
        "iv_mult": 2.0,
        "crash_pct": -0.10,
        "crash_days": 3,
        "iv_recovery_days": 14,
        "note": "Moderate crash (May 2021 scale)",
    },
    "crash_20pct_iv3x": {
        "description": "IV triples, -20% crash over 5 days",
        "iv_mult": 3.0,
        "crash_pct": -0.20,
        "crash_days": 5,
        "iv_recovery_days": 21,
        "note": "Severe crash (COVID March 2020 scale)",
    },
    "crash_30pct_iv4x": {
        "description": "IV ×4, -30% crash over 7 days",
        "iv_mult": 4.0,
        "crash_pct": -0.30,
        "crash_days": 7,
        "iv_recovery_days": 30,
        "note": "Black swan — exceeds historical",
    },
    "fear_no_crash": {
        "description": "IV doubles, price stays flat",
        "iv_mult": 2.0,
        "crash_pct": 0.0,
        "crash_days": 0,
        "iv_recovery_days": 14,
        "note": "Fear-driven IV spike without price move",
    },
    "sustained_vol": {
        "description": "IV doubles for 30 days then normalizes",
        "iv_mult": 2.0,
        "crash_pct": -0.05,
        "crash_days": 2,
        "iv_recovery_days": 30,
        "note": "Prolonged high-vol regime (LUNA period)",
    },
}


def _inject_shock_into_returns(
    vrp_returns: List[float],
    skew_returns: List[float],
    shock_day: int,
    scenario: Dict,
    bars_per_year: int = 365,
) -> Tuple[List[float], List[float]]:
    """
    Inject a vol shock into return streams with REALISTIC IV lag.

    Critical insight: during a crash, IV does NOT spike instantly.
    The sequence is:
      Day 0-2: Price crashes, RV >> current IV → VRP LOSES heavily
      Day 2+: Market reprices IV upward → IV catches up to RV
      Day 5+: IV overshoots RV (fear premium) → VRP starts profiting
      Recovery: IV decays slowly while RV normalizes → VRP profits from carry

    This lag is what makes short vol dangerous: the gamma loss hits
    BEFORE the theta benefit of elevated IV kicks in.

    VRP P&L per day = leverage × 0.5 × (IV² - RV_bar²) × dt
    """
    n = len(vrp_returns)
    mod_vrp = list(vrp_returns)
    mod_skew = list(skew_returns)

    iv_mult = scenario["iv_mult"]
    crash_pct = scenario["crash_pct"]
    crash_days = scenario["crash_days"]
    recovery_days = scenario["iv_recovery_days"]
    dt = 1.0 / bars_per_year
    leverage = 1.5
    base_iv = 0.70  # typical BTC IV

    # ── Phase 1: Crash days (IV LAGS the crash) ──
    if crash_days > 0 and crash_pct != 0:
        daily_crash = crash_pct / crash_days

        for d in range(crash_days):
            idx = shock_day + d
            if idx >= n:
                break

            # RV spikes immediately with the crash
            rv_bar_ann = abs(daily_crash) * math.sqrt(bars_per_year)

            # IV lags: ramps up gradually over crash period
            # Day 0: IV still at base level (hasn't repriced yet)
            # Day N: IV reaches shocked level
            iv_lag_factor = d / max(crash_days - 1, 1)  # 0 to 1
            # First half of crash: IV barely moves. Second half: IV catches up.
            iv_lag_factor = iv_lag_factor ** 2  # quadratic lag (slow then fast)
            current_iv = base_iv * (1.0 + (iv_mult - 1.0) * iv_lag_factor)

            # VRP P&L: when RV >> IV → big loss
            vrp_pnl = leverage * 0.5 * (current_iv ** 2 - rv_bar_ann ** 2) * dt
            mod_vrp[idx] = vrp_pnl

            # Skew MR: skew spikes during crash → model as correlated loss
            # Skew MR loses when skew moves against position (spike = bad for short skew)
            mod_skew[idx] = -abs(daily_crash) * 0.20  # 20% of crash magnitude

    # ── Phase 2: Aftermath (elevated IV, normal RV → VRP profits) ──
    for d in range(recovery_days):
        idx = shock_day + crash_days + d
        if idx >= n:
            break

        # IV decays exponentially back to normal
        decay = math.exp(-d / (recovery_days * 0.4))  # faster than linear
        current_iv = base_iv * (1.0 + (iv_mult - 1.0) * decay)

        # RV normalizes quickly (crash is over)
        # Post-crash vol elevated but decaying: ~3-4% daily moves early, normalizing
        base_daily_move = 0.025
        rv_elevation = 1.0 + 2.0 * math.exp(-d / 5.0)  # elevated for ~5 days
        daily_move = base_daily_move * rv_elevation
        rv_bar_ann = daily_move * math.sqrt(bars_per_year)

        vrp_pnl = leverage * 0.5 * (current_iv ** 2 - rv_bar_ann ** 2) * dt
        mod_vrp[idx] = vrp_pnl

        # Skew MR during recovery: skew mean-reverts → profit
        mod_skew[idx] = 0.002 * decay  # small positive during recovery

    # ── Phase 0: Fear-only scenario (no crash) ──
    if crash_days == 0 and crash_pct == 0.0:
        # IV spikes over 3 days, no price movement
        for d in range(min(3, recovery_days)):
            idx = shock_day + d
            if idx >= n:
                break
            ramp = (d + 1) / 3.0
            current_iv = base_iv * (1.0 + (iv_mult - 1.0) * ramp)
            rv_bar_ann = 0.025 * math.sqrt(bars_per_year)  # normal RV
            vrp_pnl = leverage * 0.5 * (current_iv ** 2 - rv_bar_ann ** 2) * dt
            mod_vrp[idx] = vrp_pnl
            mod_skew[idx] = 0.001 * ramp  # mild skew spike

        # Then decay
        for d in range(3, recovery_days):
            idx = shock_day + d
            if idx >= n:
                break
            decay = math.exp(-(d - 3) / (recovery_days * 0.4))
            current_iv = base_iv * (1.0 + (iv_mult - 1.0) * decay)
            rv_bar_ann = 0.025 * math.sqrt(bars_per_year)
            vrp_pnl = leverage * 0.5 * (current_iv ** 2 - rv_bar_ann ** 2) * dt
            mod_vrp[idx] = vrp_pnl
            mod_skew[idx] = 0.001 * decay

    return mod_vrp, mod_skew


def run_mc_scenario(
    scenario_name: str,
    scenario: Dict,
    n_paths: int = 200,
    seed_base: int = 42,
) -> Dict[str, Any]:
    """Run Monte Carlo simulation for one scenario."""
    # Get baseline return streams for each year
    all_vrp_returns = []
    all_skew_returns = []

    for year in [2021, 2022, 2023, 2024, 2025]:
        cfg = dict(DAILY_CFG)
        cfg["start"] = f"{year}-01-01"
        cfg["end"] = f"{year}-12-31"
        provider = DeribitRestProvider(cfg, seed=42)
        dataset = provider.load()

        engine_vrp = OptionsBacktestEngine(costs=COSTS, bars_per_year=365, use_options_pnl=True)
        engine_skew = OptionsBacktestEngine(
            costs=COSTS, bars_per_year=365, use_options_pnl=True, skew_sensitivity_mult=2.5,
        )

        vrp_result = engine_vrp.run(dataset, VariancePremiumStrategy(params=VRP_PARAMS))
        skew_result = engine_skew.run(dataset, SkewTradeV2Strategy(params=SKEW_PARAMS))

        all_vrp_returns.extend(vrp_result.returns)
        all_skew_returns.extend(skew_result.returns)

    n_total = min(len(all_vrp_returns), len(all_skew_returns))
    all_vrp_returns = all_vrp_returns[:n_total]
    all_skew_returns = all_skew_returns[:n_total]

    rng = random.Random(seed_base)

    # Buffers for shock injection
    shock_window = scenario.get("crash_days", 0) + scenario.get("iv_recovery_days", 14) + 5
    max_start = n_total - shock_window - 10

    # MC paths
    vrp_mdds = []
    skew_mdds = []
    ens_mdds = []
    vrp_worst_days = []
    ens_worst_days = []
    vrp_shock_period_returns = []
    ens_shock_period_returns = []
    recovery_bars_list = []

    for path in range(n_paths):
        # Random shock injection point
        shock_day = rng.randint(60, max(60, max_start))

        # Inject shock
        mod_vrp, mod_skew = _inject_shock_into_returns(
            all_vrp_returns, all_skew_returns, shock_day, scenario,
        )

        # Ensemble: 40% VRP + 60% Skew MR
        ens_returns = [0.40 * mod_vrp[i] + 0.60 * mod_skew[i] for i in range(n_total)]

        # Compute equity curves
        vrp_eq = [1.0]
        skew_eq = [1.0]
        ens_eq = [1.0]
        for i in range(n_total):
            vrp_eq.append(vrp_eq[-1] * (1 + mod_vrp[i]))
            skew_eq.append(skew_eq[-1] * (1 + mod_skew[i]))
            ens_eq.append(ens_eq[-1] * (1 + ens_returns[i]))

        # Max drawdown
        def _mdd(eq):
            peak = eq[0]
            dd = 0
            for v in eq:
                if v > peak:
                    peak = v
                d = (v / peak - 1) if peak > 0 else 0
                if d < dd:
                    dd = d
            return dd

        vrp_mdds.append(_mdd(vrp_eq))
        skew_mdds.append(_mdd(skew_eq))
        ens_mdds.append(_mdd(ens_eq))

        # Worst single day
        vrp_worst_days.append(min(mod_vrp))
        ens_worst_days.append(min(ens_returns))

        # Shock-period return (from shock_day to shock_day + shock_window)
        end_idx = min(shock_day + shock_window, n_total)
        shock_vrp_eq = [1.0]
        shock_ens_eq = [1.0]
        for i in range(shock_day, end_idx):
            shock_vrp_eq.append(shock_vrp_eq[-1] * (1 + mod_vrp[i]))
            shock_ens_eq.append(shock_ens_eq[-1] * (1 + ens_returns[i]))
        vrp_shock_period_returns.append(shock_vrp_eq[-1] - 1)
        ens_shock_period_returns.append(shock_ens_eq[-1] - 1)

        # Recovery: bars from shock_day until equity exceeds pre-shock level
        pre_shock_eq = ens_eq[shock_day]
        recovery = None
        for i in range(shock_day + 1, len(ens_eq)):
            if ens_eq[i] >= pre_shock_eq:
                recovery = i - shock_day
                break
        recovery_bars_list.append(recovery)

    # Statistics
    def _pct(vals, p):
        s = sorted(vals)
        idx = int(p / 100 * len(s))
        return s[min(idx, len(s) - 1)]

    def _avg(vals):
        return sum(vals) / len(vals) if vals else 0

    valid_recovery = [r for r in recovery_bars_list if r is not None]

    return {
        "scenario": scenario_name,
        "description": scenario["description"],
        "n_paths": n_paths,
        "vrp": {
            "mdd_median": round(_pct(vrp_mdds, 50) * 100, 2),
            "mdd_p95": round(_pct(vrp_mdds, 5) * 100, 2),  # 5th percentile = worst 5%
            "mdd_worst": round(min(vrp_mdds) * 100, 2),
            "worst_day_median": round(_pct(vrp_worst_days, 50) * 100, 2),
            "worst_day_p99": round(_pct(vrp_worst_days, 1) * 100, 2),
            "shock_period_return_avg": round(_avg(vrp_shock_period_returns) * 100, 2),
        },
        "ensemble": {
            "mdd_median": round(_pct(ens_mdds, 50) * 100, 2),
            "mdd_p95": round(_pct(ens_mdds, 5) * 100, 2),
            "mdd_worst": round(min(ens_mdds) * 100, 2),
            "worst_day_median": round(_pct(ens_worst_days, 50) * 100, 2),
            "worst_day_p99": round(_pct(ens_worst_days, 1) * 100, 2),
            "shock_period_return_avg": round(_avg(ens_shock_period_returns) * 100, 2),
            "recovery_median": round(_pct(valid_recovery, 50), 0) if valid_recovery else None,
            "recovery_p95": round(_pct(valid_recovery, 95), 0) if valid_recovery else None,
            "pct_recovered": round(len(valid_recovery) / n_paths * 100, 1),
        },
    }


def main():
    t0 = time.time()

    print()
    print("*" * 70)
    print("  MONTE CARLO VOL SHOCK STRESS TEST")
    print("  200 paths × 5 scenarios = 1000 simulations")
    print("*" * 70)
    print()

    all_results = {}

    for name, scenario in SCENARIOS.items():
        print(f"  Scenario: {name}")
        print(f"    {scenario['description']} ({scenario['note']})")

        result = run_mc_scenario(name, scenario, n_paths=200)

        v = result["vrp"]
        e = result["ensemble"]

        print(f"    VRP:      MDD median={v['mdd_median']:+.2f}% p95={v['mdd_p95']:+.2f}% "
              f"worst={v['mdd_worst']:+.2f}%")
        print(f"    Ensemble: MDD median={e['mdd_median']:+.2f}% p95={e['mdd_p95']:+.2f}% "
              f"worst={e['mdd_worst']:+.2f}%")
        print(f"    VRP worst day: median={v['worst_day_median']:+.2f}% p99={v['worst_day_p99']:+.2f}%")
        print(f"    Ens worst day: median={e['worst_day_median']:+.2f}% p99={e['worst_day_p99']:+.2f}%")
        print(f"    Shock period: VRP avg={v['shock_period_return_avg']:+.2f}% "
              f"Ens avg={e['shock_period_return_avg']:+.2f}%")
        if e["recovery_median"] is not None:
            print(f"    Recovery: median={e['recovery_median']:.0f}d p95={e['recovery_p95']:.0f}d "
                  f"recovered={e['pct_recovered']:.1f}%")
        else:
            print(f"    Recovery: NOT RECOVERED in simulation window")
        print()

        all_results[name] = result

    # ── Summary ──
    print("=" * 70)
    print("  SUMMARY TABLE")
    print("=" * 70)
    print(f"  {'Scenario':25s} {'Ens MDD p95':>12s} {'Ens Worst Day':>14s} {'Recovery':>10s} {'Verdict':>8s}")
    print("  " + "-" * 72)

    verdicts = {}
    for name, r in all_results.items():
        e = r["ensemble"]
        mdd = e["mdd_p95"]
        wd = e["worst_day_p99"]
        rec = f"{e['recovery_median']:.0f}d" if e["recovery_median"] is not None else "N/A"

        # Verdict: PASS if MDD p95 < 15% and worst day p99 > -10%
        v_pass = abs(mdd) < 15.0 and abs(wd) < 10.0
        verdict = "PASS" if v_pass else "FAIL"
        verdicts[name] = verdict

        print(f"  {name:25s} {mdd:>+11.2f}% {wd:>+13.2f}% {rec:>10s} {verdict:>8s}")

    # Overall
    pass_count = sum(1 for v in verdicts.values() if v == "PASS")
    total = len(verdicts)
    overall = "PASS" if pass_count == total else "WARN" if pass_count >= total - 1 else "FAIL"
    print(f"\n  Overall: {overall} ({pass_count}/{total})")
    print()

    # Key insight: during crash, VRP loses from gamma but recovers during
    # IV normalization (theta income). Ensemble smooths the path.
    # The critical scenario is whether the DRAWDOWN forces a liquidation
    # before the recovery.

    # ── Risk overlay interaction ──
    print("  RISK OVERLAY INTERACTION:")
    print("    Drawdown tiers: 3%→75%, 5%→50%, 8%→25%, 10%→halt")
    for name, r in all_results.items():
        e = r["ensemble"]
        mdd_p95 = abs(e["mdd_p95"])
        if mdd_p95 < 3:
            overlay_action = "No action"
        elif mdd_p95 < 5:
            overlay_action = "Scale to 75%"
        elif mdd_p95 < 8:
            overlay_action = "Scale to 50%"
        elif mdd_p95 < 10:
            overlay_action = "Scale to 25%"
        else:
            overlay_action = "EMERGENCY HALT"
        print(f"    {name:25s}: MDD p95={e['mdd_p95']:+.2f}% → {overlay_action}")
    print()

    # Save
    os.makedirs("artifacts/crypto_options", exist_ok=True)
    output = {
        "date": time.strftime("%Y-%m-%d %H:%M"),
        "n_paths_per_scenario": 200,
        "scenarios": all_results,
        "verdicts": verdicts,
        "overall_verdict": overall,
        "elapsed_sec": round(time.time() - t0, 1),
    }
    out_path = "artifacts/crypto_options/monte_carlo_vol_shock.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"  Saved to {out_path}")
    print(f"  Total elapsed: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
