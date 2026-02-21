#!/usr/bin/env python3
"""
Cross-Asset Signal Sweep — Crypto Options
==========================================
Tests the hypothesis that BTC vol/skew signals lead ETH vol/skew dynamics.

Strategy concepts:
1. BTC-leads-ETH Skew: Use BTC skew z-score to trade ETH skew
2. BTC-leads-ETH VRP: Use BTC VRP z-score to time ETH vol selling
3. Spread Trade: Trade the BTC-ETH skew differential
4. Vol Ratio: Trade when BTC/ETH IV ratio is extreme

These are fundamentally different from single-asset strategies because
they exploit cross-asset information, potentially generating orthogonal alpha.
"""
import json
import itertools
import math
import os
import statistics
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nexus_quant.backtest.costs import ExecutionCostModel, FeeModel, ImpactModel
from nexus_quant.projects.crypto_options.options_engine import (
    OptionsBacktestEngine, compute_metrics,
)
from nexus_quant.projects.crypto_options.providers.deribit_rest import DeribitRestProvider
from nexus_quant.data.schema import MarketDataset
from nexus_quant.strategies.base import Strategy, Weights

PROVIDER_CFG = {
    "provider": "deribit_rest_v1",
    "symbols": ["BTC", "ETH"],
    "bar_interval": "1d",
    "cache_dir": "data/cache/deribit",
    "use_synthetic_iv": True,
    "rv_lookback_bars": 21,
}

YEARS = [2021, 2022, 2023, 2024, 2025]

_FEE = FeeModel(maker_fee_rate=0.0003, taker_fee_rate=0.0005)
_IMPACT = ImpactModel(model="sqrt", coef_bps=3.0)
COSTS = ExecutionCostModel(
    fee=_FEE, execution_style="taker",
    slippage_bps=7.5, spread_bps=10.0, impact=_IMPACT, cost_multiplier=1.0,
)
COSTS_CONSERVATIVE = ExecutionCostModel(
    fee=_FEE, execution_style="taker",
    slippage_bps=7.5, spread_bps=10.0, impact=_IMPACT, cost_multiplier=1.5,
)


# ── Cross-Asset Strategy Classes ──────────────────────────────────────────

class CrossAssetSkewStrategy(Strategy):
    """Trade ETH skew based on BTC skew z-score.

    When BTC skew is extreme → use as leading indicator for ETH.
    BTC skew high (z > threshold) → ETH skew likely to follow → short ETH skew
    BTC skew low (z < -threshold) → ETH skew likely to follow → long ETH skew

    This captures the lead-lag relationship between BTC and ETH options markets.
    """

    def __init__(self, name="crypto_cross_skew", params=None):
        params = params or {}
        super().__init__(name, params)
        self.leader = str(params.get("leader", "BTC"))
        self.follower = str(params.get("follower", "ETH"))
        self.lookback = int(params.get("lookback", 60))
        self.z_entry = float(params.get("z_entry", 2.0))
        self.z_exit = float(params.get("z_exit", 0.0))
        self.target_leverage = float(params.get("target_leverage", 1.0))
        self.rebalance_freq = int(params.get("rebalance_freq", 5))
        self.min_bars = int(params.get("min_bars", 60))
        self.feature_key = str(params.get("feature_key", "skew_25d"))
        self._positions = {}

    def should_rebalance(self, dataset, idx):
        if idx < self.min_bars:
            return False
        return (idx % self.rebalance_freq) == 0

    def target_weights(self, dataset, idx, current):
        weights = {}
        # Get LEADER's z-score
        z_leader = self._get_zscore(dataset, self.leader, idx)

        for sym in dataset.symbols:
            if sym == self.leader:
                # Don't trade leader, only follower
                weights[sym] = 0.0
                continue

            current_pos = self._positions.get(sym, 0.0)
            if z_leader is None:
                weights[sym] = 0.0
                self._positions[sym] = 0.0
                continue

            lev = self.target_leverage
            new_pos = self._compute_position(z_leader, current_pos, lev)
            weights[sym] = new_pos
            self._positions[sym] = new_pos

        return weights

    def _compute_position(self, z, current_pos, lev):
        if current_pos == 0.0:
            if z >= self.z_entry:
                return -lev  # Leader skew high → short follower skew
            elif z <= -self.z_entry:
                return lev   # Leader skew low → long follower skew
            return 0.0
        else:
            if abs(z) <= self.z_exit:
                return 0.0
            elif current_pos < 0 and z <= 0:
                return 0.0
            elif current_pos > 0 and z >= 0:
                return 0.0
            return current_pos

    def _get_zscore(self, dataset, sym, idx):
        series = dataset.feature(self.feature_key, sym)
        if series is None:
            return None
        start = max(0, idx - self.lookback)
        history = [float(v) for i in range(start, min(idx + 1, len(series)))
                   if (v := series[i]) is not None]
        if len(history) < 10:
            return None
        current = history[-1]
        mean = sum(history) / len(history)
        std = statistics.pstdev(history)
        if std < 1e-6:
            return 0.0
        return (current - mean) / std


class SkewSpreadStrategy(Strategy):
    """Trade the BTC-ETH skew differential (spread).

    When BTC skew >> ETH skew (spread extreme high) → short BTC skew, long ETH skew
    When ETH skew >> BTC skew (spread extreme low) → long BTC skew, short ETH skew

    This is a relative value trade on skew across assets.
    """

    def __init__(self, name="crypto_skew_spread", params=None):
        params = params or {}
        super().__init__(name, params)
        self.lookback = int(params.get("lookback", 60))
        self.z_entry = float(params.get("z_entry", 2.0))
        self.z_exit = float(params.get("z_exit", 0.0))
        self.target_leverage = float(params.get("target_leverage", 1.0))
        self.rebalance_freq = int(params.get("rebalance_freq", 5))
        self.min_bars = int(params.get("min_bars", 60))
        self._position = 0.0  # +1 = long BTC/short ETH, -1 = reverse

    def should_rebalance(self, dataset, idx):
        if idx < self.min_bars:
            return False
        return (idx % self.rebalance_freq) == 0

    def target_weights(self, dataset, idx, current):
        z = self._get_spread_zscore(dataset, idx)
        if z is None:
            self._position = 0.0
            return {s: 0.0 for s in dataset.symbols}

        lev = self.target_leverage

        if self._position == 0.0:
            if z >= self.z_entry:
                # BTC skew >> ETH → short BTC skew, long ETH skew
                self._position = -1.0
            elif z <= -self.z_entry:
                # ETH skew >> BTC → long BTC skew, short ETH skew
                self._position = 1.0
        else:
            if abs(z) <= self.z_exit:
                self._position = 0.0
            elif self._position < 0 and z <= 0:
                self._position = 0.0
            elif self._position > 0 and z >= 0:
                self._position = 0.0

        weights = {}
        for sym in dataset.symbols:
            if sym == "BTC":
                weights[sym] = self._position * lev
            elif sym == "ETH":
                weights[sym] = -self._position * lev
            else:
                weights[sym] = 0.0
        return weights

    def _get_spread_zscore(self, dataset, idx):
        btc_skew = dataset.feature("skew_25d", "BTC")
        eth_skew = dataset.feature("skew_25d", "ETH")
        if btc_skew is None or eth_skew is None:
            return None

        start = max(0, idx - self.lookback)
        spreads = []
        for i in range(start, min(idx + 1, len(btc_skew), len(eth_skew))):
            b, e = btc_skew[i], eth_skew[i]
            if b is not None and e is not None:
                spreads.append(float(b) - float(e))

        if len(spreads) < 10:
            return None
        current = spreads[-1]
        mean = sum(spreads) / len(spreads)
        std = statistics.pstdev(spreads)
        if std < 1e-6:
            return 0.0
        return (current - mean) / std


class VolRatioStrategy(Strategy):
    """Trade the BTC/ETH IV ratio.

    When BTC_IV / ETH_IV is extreme high → BTC vol expensive → short BTC, long ETH
    When ratio is extreme low → ETH vol expensive → short ETH, long BTC

    Trades the relative IV level rather than absolute.
    """

    def __init__(self, name="crypto_skew_vol_ratio", params=None):
        params = params or {}
        super().__init__(name, params)
        self.lookback = int(params.get("lookback", 60))
        self.z_entry = float(params.get("z_entry", 2.0))
        self.z_exit = float(params.get("z_exit", 0.0))
        self.target_leverage = float(params.get("target_leverage", 1.0))
        self.rebalance_freq = int(params.get("rebalance_freq", 5))
        self.min_bars = int(params.get("min_bars", 60))
        self._position = 0.0

    def should_rebalance(self, dataset, idx):
        if idx < self.min_bars:
            return False
        return (idx % self.rebalance_freq) == 0

    def target_weights(self, dataset, idx, current):
        z = self._get_ratio_zscore(dataset, idx)
        if z is None:
            self._position = 0.0
            return {s: 0.0 for s in dataset.symbols}

        lev = self.target_leverage

        if self._position == 0.0:
            if z >= self.z_entry:
                self._position = -1.0  # Ratio high → short BTC vol, long ETH vol
            elif z <= -self.z_entry:
                self._position = 1.0   # Ratio low → long BTC vol, short ETH vol
        else:
            if abs(z) <= self.z_exit:
                self._position = 0.0
            elif self._position < 0 and z <= 0:
                self._position = 0.0
            elif self._position > 0 and z >= 0:
                self._position = 0.0

        weights = {}
        for sym in dataset.symbols:
            if sym == "BTC":
                weights[sym] = self._position * lev
            elif sym == "ETH":
                weights[sym] = -self._position * lev
            else:
                weights[sym] = 0.0
        return weights

    def _get_ratio_zscore(self, dataset, idx):
        btc_iv = dataset.feature("iv_atm", "BTC")
        eth_iv = dataset.feature("iv_atm", "ETH")
        if btc_iv is None or eth_iv is None:
            return None

        start = max(0, idx - self.lookback)
        ratios = []
        for i in range(start, min(idx + 1, len(btc_iv), len(eth_iv))):
            b, e = btc_iv[i], eth_iv[i]
            if b is not None and e is not None and float(e) > 0.01:
                ratios.append(float(b) / float(e))

        if len(ratios) < 10:
            return None
        current = ratios[-1]
        mean = sum(ratios) / len(ratios)
        std = statistics.pstdev(ratios)
        if std < 1e-6:
            return 0.0
        return (current - mean) / std


# ── Sweep Function ─────────────────────────────────────────────────────

def run_yearly_cross_asset_wf(strategy_cls, strategy_params, sens=1.0):
    """Per-year walk-forward for cross-asset strategies."""
    from nexus_quant.projects.crypto_options.options_engine import run_yearly_wf as _wf
    # Cross-asset strategies use "skew" in name → engine routes to _run_skew
    return _wf(
        provider_cfg=PROVIDER_CFG,
        strategy_cls=strategy_cls,
        strategy_params=strategy_params,
        years=YEARS,
        costs=COSTS,
        use_options_pnl=True,
        bars_per_year=365,
        seed=42,
        skew_sensitivity_mult=sens,
    )


def sweep_strategy(label, strategy_cls, param_grid, sens_vals=[1.0, 2.0, 2.5]):
    """Run parameter sweep for a strategy."""
    keys = list(param_grid.keys())
    values = list(param_grid.values())
    combos = list(itertools.product(*values))
    total = len(combos) * len(sens_vals)

    print(f"\n{'='*70}")
    print(f"  {label}: {len(combos)} param combos × {len(sens_vals)} sensitivity = {total} configs")
    print(f"{'='*70}\n")

    results = []
    best_avg = -999
    best = None

    for ci, combo in enumerate(combos):
        params = dict(zip(keys, combo))
        params["min_bars"] = max(params.get("lookback", 60), 60)

        for si, sens in enumerate(sens_vals):
            idx = ci * len(sens_vals) + si + 1
            try:
                wf = run_yearly_cross_asset_wf(strategy_cls, params, sens)
                avg = wf["summary"]["avg_sharpe"]
                mn = wf["summary"]["min_sharpe"]
                passed = wf["summary"]["passed"]

                result = {
                    "params": params, "sensitivity": sens,
                    "avg_sharpe": avg, "min_sharpe": mn, "passed": passed,
                    "yearly": {k: v.get("sharpe", 0) for k, v in wf["yearly"].items()},
                }
                results.append(result)

                flag = ""
                if avg > best_avg:
                    best_avg = avg
                    best = result
                    flag = " ★ NEW BEST"
                if passed:
                    flag += " ★★★ VALIDATED"

                if flag or idx % 20 == 0:
                    print(f"  [{idx:3d}/{total}] sens={sens:.1f} lb={params.get('lookback', '?')} "
                          f"ze={params.get('z_entry', '?')} → avg={avg:+.4f} min={mn:+.4f}{flag}")

            except Exception as e:
                results.append({"params": params, "sensitivity": sens, "avg_sharpe": -999, "error": str(e)})

    results.sort(key=lambda r: r.get("avg_sharpe", -999), reverse=True)
    return results, best


def compute_correlations(best_configs):
    """Compute correlations between new cross-asset strategies and existing ones."""
    from nexus_quant.projects.crypto_options.strategies.variance_premium import VariancePremiumStrategy
    from nexus_quant.projects.crypto_options.strategies.skew_trade_v2 import SkewTradeV2Strategy

    print(f"\n{'='*70}")
    print(f"  CORRELATION ANALYSIS")
    print(f"{'='*70}\n")

    cfg = dict(PROVIDER_CFG)
    cfg["start"] = "2020-01-01"
    cfg["end"] = "2025-12-31"
    provider = DeribitRestProvider(cfg, seed=42)
    dataset = provider.load()

    # Existing strategies
    vrp_eng = OptionsBacktestEngine(costs=COSTS, bars_per_year=365, use_options_pnl=True)
    vrp = VariancePremiumStrategy(params={
        "base_leverage": 1.5, "exit_z_threshold": -3.0,
        "vrp_lookback": 30, "rebalance_freq": 5, "min_bars": 30,
    })
    vrp_r = vrp_eng.run(dataset, vrp)

    skew_eng = OptionsBacktestEngine(costs=COSTS, bars_per_year=365, use_options_pnl=True, skew_sensitivity_mult=2.5)
    skew = SkewTradeV2Strategy(params={
        "skew_lookback": 60, "z_entry": 2.0, "z_exit": 0.0,
        "target_leverage": 1.0, "rebalance_freq": 5, "min_bars": 60,
        "use_vrp_filter": False,
    })
    skew_r = skew_eng.run(dataset, skew)

    all_returns = {"VRP": vrp_r.returns, "Skew_v2": skew_r.returns}

    # New cross-asset strategies
    for label, (cls, params, sens) in best_configs.items():
        eng = OptionsBacktestEngine(costs=COSTS, bars_per_year=365, use_options_pnl=True, skew_sensitivity_mult=sens)
        strat = cls(params=params)
        result = eng.run(dataset, strat)
        all_returns[label] = result.returns

    # Pairwise
    names = list(all_returns.keys())
    corr = {}
    for i, n1 in enumerate(names):
        for j, n2 in enumerate(names):
            if i >= j:
                continue
            r1, r2 = all_returns[n1], all_returns[n2]
            mn = min(len(r1), len(r2))
            r1, r2 = r1[:mn], r2[:mn]
            m1 = sum(r1) / mn
            m2 = sum(r2) / mn
            cov = sum((a - m1) * (b - m2) for a, b in zip(r1, r2)) / max(mn - 1, 1)
            v1 = sum((a - m1)**2 for a in r1) / max(mn - 1, 1)
            v2 = sum((b - m2)**2 for b in r2) / max(mn - 1, 1)
            s1, s2 = v1**0.5, v2**0.5
            c = cov / (s1 * s2) if s1 > 0 and s2 > 0 else 0.0
            corr[f"{n1}_vs_{n2}"] = round(c, 4)
            print(f"  {n1:20s} vs {n2:20s}: r = {c:+.4f}")

    return corr


def main():
    t0 = time.time()

    # 1. BTC-leads-ETH Skew
    cross_skew_results, cross_skew_best = sweep_strategy(
        "Cross-Asset Skew (BTC→ETH)",
        CrossAssetSkewStrategy,
        {
            "lookback": [30, 60, 90, 120],
            "z_entry": [1.0, 1.5, 2.0, 2.5],
            "z_exit": [0.0, 0.3],
            "target_leverage": [0.5, 1.0],
            "rebalance_freq": [5, 10],
            "leader": ["BTC"],
            "follower": ["ETH"],
            "feature_key": ["skew_25d"],
        },
        sens_vals=[1.0, 2.5],
    )

    # 2. Skew Spread (BTC-ETH differential)
    spread_results, spread_best = sweep_strategy(
        "Skew Spread (BTC-ETH)",
        SkewSpreadStrategy,
        {
            "lookback": [30, 60, 90, 120],
            "z_entry": [1.0, 1.5, 2.0, 2.5],
            "z_exit": [0.0, 0.3],
            "target_leverage": [0.5, 1.0],
            "rebalance_freq": [5, 10],
        },
        sens_vals=[1.0, 2.5],
    )

    # 3. Vol Ratio (BTC/ETH IV ratio)
    vol_ratio_results, vol_ratio_best = sweep_strategy(
        "Vol Ratio (BTC/ETH)",
        VolRatioStrategy,
        {
            "lookback": [30, 60, 90, 120],
            "z_entry": [1.0, 1.5, 2.0, 2.5],
            "z_exit": [0.0, 0.3],
            "target_leverage": [0.5, 1.0],
            "rebalance_freq": [5, 10],
        },
        sens_vals=[1.0, 2.5],
    )

    # Summary
    print(f"\n{'='*70}")
    print(f"  CROSS-ASSET SWEEP SUMMARY")
    print(f"{'='*70}\n")

    all_bests = {
        "cross_skew": (CrossAssetSkewStrategy, cross_skew_best),
        "skew_spread": (SkewSpreadStrategy, spread_best),
        "vol_ratio": (VolRatioStrategy, vol_ratio_best),
    }

    any_positive = False
    corr_configs = {}
    for label, (cls, best) in all_bests.items():
        if best is None:
            print(f"  {label}: NO RESULTS")
            continue
        avg = best["avg_sharpe"]
        mn = best["min_sharpe"]
        passed = best.get("passed", False)
        flag = "VALIDATED" if passed else "NOT VALIDATED"
        print(f"  {label}: avg={avg:+.4f} min={mn:+.4f} [{flag}]")
        if "yearly" in best:
            yearly_str = " ".join(f"{y}={s:+.3f}" for y, s in sorted(best["yearly"].items()))
            print(f"    {yearly_str}")
        if avg > 0:
            any_positive = True
            corr_configs[label] = (cls, best["params"], best.get("sensitivity", 1.0))

    # Correlations for positive strategies
    corr_matrix = {}
    if any_positive:
        corr_matrix = compute_correlations(corr_configs)

    # Save
    os.makedirs("artifacts/crypto_options", exist_ok=True)
    output = {
        "date": time.strftime("%Y-%m-%d %H:%M"),
        "cross_skew": {"total": len(cross_skew_results), "top5": cross_skew_results[:5], "best": cross_skew_best},
        "skew_spread": {"total": len(spread_results), "top5": spread_results[:5], "best": spread_best},
        "vol_ratio": {"total": len(vol_ratio_results), "top5": vol_ratio_results[:5], "best": vol_ratio_best},
        "correlations": corr_matrix,
        "elapsed_sec": round(time.time() - t0, 1),
    }
    out_path = "artifacts/crypto_options/cross_asset_sweep_results.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\n  Results saved to {out_path}")
    print(f"  Total elapsed: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
