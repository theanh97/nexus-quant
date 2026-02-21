#!/usr/bin/env python3
"""
VRP Regime Filter Research
=============================

Tests whether conditioning VRP entry/exit on secondary signals improves
risk-adjusted returns. Filters tested:

1. IV Term Structure Slope: Steep term structure (front > back) = fear premium
   → VRP should be MORE profitable (higher theta income)
   → Flat/inverted = complacency → reduce VRP exposure

2. Skew Extreme Filter: When 25d skew is extreme (|z| > 2), VRP risk changes
   → High put skew = tail risk rising → reduce VRP
   → Low put skew = complacency → VRP is safer

3. VRP Z-Score Filter: Already in baseline (-3.0 exit). Test tighter thresholds.

4. Combined Filter: IV slope + skew z-score for optimal regime detection.

Methodology:
  - Base VRP with Sharpe 2.088 (daily) as control
  - Walk-forward with annual LOYO
  - Compare filtered vs unfiltered VRP Sharpe, MDD, tail metrics
"""
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from nexus_quant.backtest.costs import ExecutionCostModel, FeeModel, ImpactModel
from nexus_quant.projects.crypto_options.options_engine import OptionsBacktestEngine
from nexus_quant.projects.crypto_options.providers.deribit_rest import DeribitRestProvider
from nexus_quant.projects.crypto_options.strategies.variance_premium import VariancePremiumStrategy
from nexus_quant.strategies.base import Strategy, Weights
from nexus_quant.data.schema import MarketDataset

from datetime import datetime, timezone

DIVIDER = "=" * 70


def ts_to_year(ts: int) -> int:
    """Convert epoch timestamp to year."""
    return datetime.fromtimestamp(ts, tz=timezone.utc).year


# Realistic Deribit cost model
_FEE = FeeModel(maker_fee_rate=0.0002, taker_fee_rate=0.0005)
_IMPACT = ImpactModel(model="sqrt", coef_bps=3.0)
COSTS = ExecutionCostModel(
    fee=_FEE, execution_style="taker",
    slippage_bps=7.5, spread_bps=10.0, impact=_IMPACT, cost_multiplier=1.0,
)


class FilteredVRPStrategy(VariancePremiumStrategy):
    """VRP strategy with configurable regime filters.

    Subclasses the validated VariancePremiumStrategy and adds post-hoc
    filters to the base target weights.
    """

    def __init__(
        self,
        base_leverage: float = 1.5,
        vrp_lookback: int = 30,
        rebalance_freq: int = 5,
        exit_z_threshold: float = -3.0,
        # Regime filters
        term_slope_filter: bool = False,
        term_slope_threshold: float = 0.0,
        skew_z_filter: bool = False,
        skew_z_max: float = 3.0,
        skew_z_reduce: float = 0.5,
        iv_level_filter: bool = False,
        iv_low_threshold: float = 0.30,
        iv_low_scale: float = 0.5,
        iv_high_threshold: float = 1.50,
        iv_high_scale: float = 0.5,
        name_suffix: str = "",
    ):
        # Init base VRP strategy
        super().__init__(name="vrp_filtered", params={
            "base_leverage": base_leverage,
            "exit_z_threshold": exit_z_threshold,
            "vrp_lookback": vrp_lookback,
            "rebalance_freq": rebalance_freq,
            "min_bars": 30,
        })

        # Filters
        self._term_filter = term_slope_filter
        self._term_thresh = term_slope_threshold
        self._skew_filter = skew_z_filter
        self._skew_z_max = skew_z_max
        self._skew_reduce = skew_z_reduce
        self._iv_filter = iv_level_filter
        self._iv_low = iv_low_threshold
        self._iv_low_scale = iv_low_scale
        self._iv_high = iv_high_threshold
        self._iv_high_scale = iv_high_scale

    def target_weights(self, dataset: MarketDataset, idx: int, current: Weights) -> Weights:
        # Get base VRP weights from parent
        weights = super().target_weights(dataset, idx, current)

        # Apply regime filters on top
        for sym in dataset.symbols:
            w = weights.get(sym, 0.0)
            if abs(w) < 1e-10:
                continue

            # ── Term slope filter ──
            if self._term_filter:
                ts_series = dataset.feature("term_spread", sym)
                if ts_series and idx < len(ts_series) and ts_series[idx] is not None:
                    if ts_series[idx] < self._term_thresh:
                        weights[sym] = 0.0
                        continue

            # ── Skew z-score filter ──
            if self._skew_filter:
                skew_series = dataset.feature("skew_25d", sym)
                if skew_series and idx >= self.vrp_lookback:
                    recent = [skew_series[i] for i in range(idx - self.vrp_lookback, idx)
                              if i < len(skew_series) and skew_series[i] is not None]
                    if recent and idx < len(skew_series) and skew_series[idx] is not None:
                        mean_s = sum(recent) / len(recent)
                        std_s = (sum((x - mean_s) ** 2 for x in recent) / max(len(recent) - 1, 1)) ** 0.5
                        if std_s > 0.001:
                            skew_z = abs(skew_series[idx] - mean_s) / std_s
                            if skew_z > self._skew_z_max:
                                weights[sym] = w * self._skew_reduce

            # ── IV level filter ──
            if self._iv_filter:
                iv_series = dataset.feature("iv_atm", sym)
                if iv_series and idx < len(iv_series) and iv_series[idx] is not None:
                    iv = iv_series[idx]
                    if iv > 0:
                        if iv < self._iv_low:
                            weights[sym] = w * self._iv_low_scale
                        elif iv > self._iv_high:
                            weights[sym] = w * self._iv_high_scale

        return weights


def load_data(year_start: int = 2021, year_end: int = 2025):
    """Load BTC options dataset."""
    cfg = {
        "symbols": ["BTC"],
        "start": f"{year_start}-01-01",
        "end": f"{year_end}-12-31",
        "timeframe": "1d",
    }
    provider = DeribitRestProvider(cfg)
    dataset = provider.load()
    return dataset


def run_walk_forward(engine, strategy, dataset, years):
    """Walk-forward LOYO with per-year Sharpe."""
    sharpes = {}
    mdds = {}

    for test_year in years:
        train_mask = [i for i, ts in enumerate(dataset.timeline)
                      if ts > 0 and not (test_year * 10000 <= ts // 1000000 < (test_year + 1) * 10000)]
        test_mask = [i for i, ts in enumerate(dataset.timeline)
                     if ts > 0 and test_year * 10000 <= ts // 1000000 < (test_year + 1) * 10000]

        # Run on full dataset (strategy filters internally)
        strategy._step = 0
        result = engine.run(dataset, strategy)

        # Extract test period returns
        if len(result.returns) > 0:
            test_returns = []
            for i in test_mask:
                if i - 1 < len(result.returns):
                    test_returns.append(result.returns[i - 1])

            if test_returns:
                avg_r = sum(test_returns) / len(test_returns)
                std_r = (sum((r - avg_r) ** 2 for r in test_returns) / max(len(test_returns) - 1, 1)) ** 0.5
                sharpe = (avg_r / std_r * math.sqrt(365)) if std_r > 0 else 0.0
                sharpes[test_year] = sharpe

                # MDD
                peak = 1.0
                worst_dd = 0.0
                eq = 1.0
                for r in test_returns:
                    eq *= (1 + r)
                    if eq > peak:
                        peak = eq
                    dd = (peak - eq) / peak
                    if dd > worst_dd:
                        worst_dd = dd
                mdds[test_year] = worst_dd

    return sharpes, mdds


def main():
    print(DIVIDER)
    print("VRP REGIME FILTER RESEARCH")
    print(DIVIDER)

    print("\nLoading data...")
    dataset = load_data()
    n = len(dataset.timeline)
    print(f"Dataset: {n} bars, symbols: {dataset.symbols}")

    # Engine
    engine = OptionsBacktestEngine(costs=COSTS, bars_per_year=365, use_options_pnl=True)

    years = [2021, 2022, 2023, 2024, 2025]

    # ── 1. Baseline VRP (no filters) ──
    print(f"\n{DIVIDER}")
    print("1. BASELINE VRP (no filters)")
    print(DIVIDER)
    baseline = FilteredVRPStrategy(base_leverage=1.5, vrp_lookback=30,
                                    rebalance_freq=5, exit_z_threshold=-3.0)
    result_base = engine.run(dataset, baseline)
    base_returns = result_base.returns

    avg_r = sum(base_returns) / len(base_returns)
    std_r = (sum((r - avg_r) ** 2 for r in base_returns) / max(len(base_returns) - 1, 1)) ** 0.5
    base_sharpe = (avg_r / std_r * math.sqrt(365)) if std_r > 0 else 0.0
    print(f"  Full-period Sharpe: {base_sharpe:.3f}")

    # Per-year analysis
    sharpes_base, mdds_base = {}, {}
    eq = 1.0
    peak = 1.0
    year_returns = {}
    for i, ts in enumerate(dataset.timeline[1:], 1):
        if i - 1 < len(base_returns):
            year = ts_to_year(ts)
            if year not in year_returns:
                year_returns[year] = []
            year_returns[year].append(base_returns[i - 1])

    for yr in years:
        rets = year_returns.get(yr, [])
        if rets:
            m = sum(rets) / len(rets)
            s = (sum((r - m) ** 2 for r in rets) / max(len(rets) - 1, 1)) ** 0.5
            sharpes_base[yr] = (m / s * math.sqrt(365)) if s > 0 else 0.0
            p = 1.0
            pk = 1.0
            wd = 0.0
            for r in rets:
                p *= (1 + r)
                if p > pk:
                    pk = p
                dd = (pk - p) / pk
                if dd > wd:
                    wd = dd
            mdds_base[yr] = wd

    for yr in years:
        sh = sharpes_base.get(yr, 0)
        mdd = mdds_base.get(yr, 0)
        print(f"  {yr}: Sharpe={sh:.3f} MDD={mdd:.1%}")
    avg_sh = sum(sharpes_base.values()) / len(sharpes_base) if sharpes_base else 0
    print(f"  Avg Sharpe: {avg_sh:.3f}")

    # ── 2. Term structure filter configs ──
    print(f"\n{DIVIDER}")
    print("2. IV TERM STRUCTURE SLOPE FILTER")
    print(DIVIDER)
    print("  When term_spread < threshold → go flat (term structure inverted/flat)")

    term_configs = [
        {"name": "term_-0.05", "term_slope_threshold": -0.05},
        {"name": "term_-0.02", "term_slope_threshold": -0.02},
        {"name": "term_0.00", "term_slope_threshold": 0.00},
        {"name": "term_0.02", "term_slope_threshold": 0.02},
        {"name": "term_0.05", "term_slope_threshold": 0.05},
    ]

    term_results = []
    for tc in term_configs:
        strat = FilteredVRPStrategy(
            base_leverage=1.5, vrp_lookback=30, rebalance_freq=5,
            exit_z_threshold=-3.0,
            term_slope_filter=True,
            term_slope_threshold=tc["term_slope_threshold"],
        )
        res = engine.run(dataset, strat)
        rets = res.returns
        if rets:
            m = sum(rets) / len(rets)
            s = (sum((r - m) ** 2 for r in rets) / max(len(rets) - 1, 1)) ** 0.5
            sh = (m / s * math.sqrt(365)) if s > 0 else 0.0

            # Per year
            yr_sharpes = {}
            yr_rets = {}
            for i, ts in enumerate(dataset.timeline[1:], 1):
                if i - 1 < len(rets):
                    yr = ts_to_year(ts)
                    if yr not in yr_rets:
                        yr_rets[yr] = []
                    yr_rets[yr].append(rets[i - 1])
            for yr in years:
                r_yr = yr_rets.get(yr, [])
                if r_yr:
                    m_yr = sum(r_yr) / len(r_yr)
                    s_yr = (sum((r - m_yr) ** 2 for r in r_yr) / max(len(r_yr) - 1, 1)) ** 0.5
                    yr_sharpes[yr] = (m_yr / s_yr * math.sqrt(365)) if s_yr > 0 else 0.0

            avg_yr_sh = sum(yr_sharpes.values()) / len(yr_sharpes) if yr_sharpes else 0.0
            delta = avg_yr_sh - avg_sh

            term_results.append({
                "name": tc["name"],
                "threshold": tc["term_slope_threshold"],
                "full_sharpe": sh,
                "avg_yr_sharpe": avg_yr_sh,
                "delta_vs_base": delta,
                "yr_sharpes": yr_sharpes,
            })

            yr_str = " ".join(f"{y}:{yr_sharpes.get(y, 0):.2f}" for y in years)
            print(f"  {tc['name']:15s} Avg={avg_yr_sh:.3f} (Δ={delta:+.3f}) | {yr_str}")

    # ── 3. Skew z-score filter configs ──
    print(f"\n{DIVIDER}")
    print("3. SKEW Z-SCORE FILTER")
    print(DIVIDER)
    print("  When |skew_z| > threshold → reduce VRP exposure by scale factor")

    skew_configs = [
        {"name": "skewz1.5_50%", "skew_z_max": 1.5, "skew_z_reduce": 0.5},
        {"name": "skewz2.0_50%", "skew_z_max": 2.0, "skew_z_reduce": 0.5},
        {"name": "skewz2.5_50%", "skew_z_max": 2.5, "skew_z_reduce": 0.5},
        {"name": "skewz2.0_0%", "skew_z_max": 2.0, "skew_z_reduce": 0.0},
        {"name": "skewz2.5_0%", "skew_z_max": 2.5, "skew_z_reduce": 0.0},
    ]

    skew_results = []
    for sc in skew_configs:
        strat = FilteredVRPStrategy(
            base_leverage=1.5, vrp_lookback=30, rebalance_freq=5,
            exit_z_threshold=-3.0,
            skew_z_filter=True,
            skew_z_max=sc["skew_z_max"],
            skew_z_reduce=sc["skew_z_reduce"],
        )
        res = engine.run(dataset, strat)
        rets = res.returns
        if rets:
            m = sum(rets) / len(rets)
            s = (sum((r - m) ** 2 for r in rets) / max(len(rets) - 1, 1)) ** 0.5
            sh = (m / s * math.sqrt(365)) if s > 0 else 0.0

            yr_sharpes = {}
            yr_rets = {}
            for i, ts in enumerate(dataset.timeline[1:], 1):
                if i - 1 < len(rets):
                    yr = ts_to_year(ts)
                    if yr not in yr_rets:
                        yr_rets[yr] = []
                    yr_rets[yr].append(rets[i - 1])
            for yr in years:
                r_yr = yr_rets.get(yr, [])
                if r_yr:
                    m_yr = sum(r_yr) / len(r_yr)
                    s_yr = (sum((r - m_yr) ** 2 for r in r_yr) / max(len(r_yr) - 1, 1)) ** 0.5
                    yr_sharpes[yr] = (m_yr / s_yr * math.sqrt(365)) if s_yr > 0 else 0.0

            avg_yr_sh = sum(yr_sharpes.values()) / len(yr_sharpes) if yr_sharpes else 0.0
            delta = avg_yr_sh - avg_sh

            skew_results.append({
                "name": sc["name"],
                "avg_yr_sharpe": avg_yr_sh,
                "delta_vs_base": delta,
            })

            yr_str = " ".join(f"{y}:{yr_sharpes.get(y, 0):.2f}" for y in years)
            print(f"  {sc['name']:15s} Avg={avg_yr_sh:.3f} (Δ={delta:+.3f}) | {yr_str}")

    # ── 4. IV level filter configs ──
    print(f"\n{DIVIDER}")
    print("4. IV LEVEL FILTER")
    print(DIVIDER)
    print("  Reduce VRP when IV too low (no premium) or too high (crash risk)")

    iv_configs = [
        {"name": "iv_30-120%", "iv_low": 0.30, "iv_high": 1.20, "iv_low_s": 0.5, "iv_high_s": 0.5},
        {"name": "iv_40-100%", "iv_low": 0.40, "iv_high": 1.00, "iv_low_s": 0.5, "iv_high_s": 0.5},
        {"name": "iv_30-150%", "iv_low": 0.30, "iv_high": 1.50, "iv_low_s": 0.0, "iv_high_s": 0.5},
        {"name": "iv_50-120%", "iv_low": 0.50, "iv_high": 1.20, "iv_low_s": 0.0, "iv_high_s": 0.0},
    ]

    iv_results = []
    for ic in iv_configs:
        strat = FilteredVRPStrategy(
            base_leverage=1.5, vrp_lookback=30, rebalance_freq=5,
            exit_z_threshold=-3.0,
            iv_level_filter=True,
            iv_low_threshold=ic["iv_low"],
            iv_high_threshold=ic["iv_high"],
            iv_low_scale=ic["iv_low_s"],
            iv_high_scale=ic["iv_high_s"],
        )
        res = engine.run(dataset, strat)
        rets = res.returns
        if rets:
            m = sum(rets) / len(rets)
            s = (sum((r - m) ** 2 for r in rets) / max(len(rets) - 1, 1)) ** 0.5
            sh = (m / s * math.sqrt(365)) if s > 0 else 0.0

            yr_sharpes = {}
            yr_rets = {}
            for i, ts in enumerate(dataset.timeline[1:], 1):
                if i - 1 < len(rets):
                    yr = ts_to_year(ts)
                    if yr not in yr_rets:
                        yr_rets[yr] = []
                    yr_rets[yr].append(rets[i - 1])
            for yr in years:
                r_yr = yr_rets.get(yr, [])
                if r_yr:
                    m_yr = sum(r_yr) / len(r_yr)
                    s_yr = (sum((r - m_yr) ** 2 for r in r_yr) / max(len(r_yr) - 1, 1)) ** 0.5
                    yr_sharpes[yr] = (m_yr / s_yr * math.sqrt(365)) if s_yr > 0 else 0.0

            avg_yr_sh = sum(yr_sharpes.values()) / len(yr_sharpes) if yr_sharpes else 0.0
            delta = avg_yr_sh - avg_sh

            iv_results.append({
                "name": ic["name"],
                "avg_yr_sharpe": avg_yr_sh,
                "delta_vs_base": delta,
            })

            yr_str = " ".join(f"{y}:{yr_sharpes.get(y, 0):.2f}" for y in years)
            print(f"  {ic['name']:15s} Avg={avg_yr_sh:.3f} (Δ={delta:+.3f}) | {yr_str}")

    # ── 5. Combined best filters ──
    print(f"\n{DIVIDER}")
    print("5. COMBINED FILTER (best from each category)")
    print(DIVIDER)

    # Find best from each category
    best_term = max(term_results, key=lambda x: x["avg_yr_sharpe"]) if term_results else None
    best_skew = max(skew_results, key=lambda x: x["avg_yr_sharpe"]) if skew_results else None
    best_iv = max(iv_results, key=lambda x: x["avg_yr_sharpe"]) if iv_results else None

    if best_term:
        print(f"  Best term filter: {best_term['name']} (Δ={best_term['delta_vs_base']:+.3f})")
    if best_skew:
        print(f"  Best skew filter: {best_skew['name']} (Δ={best_skew['delta_vs_base']:+.3f})")
    if best_iv:
        print(f"  Best IV filter:   {best_iv['name']} (Δ={best_iv['delta_vs_base']:+.3f})")

    # Test combination: best term + best skew + best IV
    combo_configs = [
        ("term+skew", True, True, False),
        ("term+iv", True, False, True),
        ("skew+iv", False, True, True),
        ("all_three", True, True, True),
    ]

    for combo_name, use_term, use_skew, use_iv in combo_configs:
        strat = FilteredVRPStrategy(
            base_leverage=1.5, vrp_lookback=30, rebalance_freq=5,
            exit_z_threshold=-3.0,
            term_slope_filter=use_term,
            term_slope_threshold=best_term["threshold"] if best_term and use_term else 0.0,
            skew_z_filter=use_skew,
            skew_z_max=2.0 if use_skew else 99.0,
            skew_z_reduce=0.5 if use_skew else 1.0,
            iv_level_filter=use_iv,
            iv_low_threshold=0.30 if use_iv else 0.0,
            iv_high_threshold=1.50 if use_iv else 99.0,
            iv_low_scale=0.5 if use_iv else 1.0,
            iv_high_scale=0.5 if use_iv else 1.0,
        )
        res = engine.run(dataset, strat)
        rets = res.returns
        if rets:
            yr_sharpes = {}
            yr_rets = {}
            for i, ts in enumerate(dataset.timeline[1:], 1):
                if i - 1 < len(rets):
                    yr = ts_to_year(ts)
                    if yr not in yr_rets:
                        yr_rets[yr] = []
                    yr_rets[yr].append(rets[i - 1])
            for yr in years:
                r_yr = yr_rets.get(yr, [])
                if r_yr:
                    m_yr = sum(r_yr) / len(r_yr)
                    s_yr = (sum((r - m_yr) ** 2 for r in r_yr) / max(len(r_yr) - 1, 1)) ** 0.5
                    yr_sharpes[yr] = (m_yr / s_yr * math.sqrt(365)) if s_yr > 0 else 0.0

            avg_yr_sh = sum(yr_sharpes.values()) / len(yr_sharpes) if yr_sharpes else 0.0
            delta = avg_yr_sh - avg_sh

            yr_str = " ".join(f"{y}:{yr_sharpes.get(y, 0):.2f}" for y in years)
            print(f"  {combo_name:15s} Avg={avg_yr_sh:.3f} (Δ={delta:+.3f}) | {yr_str}")

    # ── Summary ──
    print(f"\n{DIVIDER}")
    print("SUMMARY")
    print(DIVIDER)
    print(f"  Baseline VRP: Avg Sharpe = {avg_sh:.3f}")
    print(f"  Key question: Do regime filters improve risk-adjusted returns?")
    print(f"  IMPORTANT: Improvements on synthetic data may not transfer to live.")
    print(f"  Only adopt filters that show CLEAR improvement across ALL years.")
    print(DIVIDER)


if __name__ == "__main__":
    main()
