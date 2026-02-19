"""
Ensemble Strategy — dynamically combines multiple sub-strategies.

Uses recent out-of-sample performance to weight sub-strategies:
- Strategies with higher recent Sharpe get higher weight
- Minimum weight floor to ensure diversification
- Exponential decay weighting (recent performance matters more)

Sub-strategies are run independently, then weights are combined.
"""
from __future__ import annotations

import math
import statistics
from typing import Any, Dict, List, Optional

from .base import Strategy, Weights
from ..data.schema import MarketDataset


class EnsembleV1Strategy(Strategy):
    """
    Dynamic ensemble meta-strategy.

    Runs multiple sub-strategies independently, then blends their target weights
    using Sharpe-based performance weighting computed over a rolling window.
    A minimum strategy weight floor ensures all sub-strategies contribute.
    """

    def __init__(
        self,
        sub_strategies: List[Strategy],
        params: Dict[str, Any],
    ) -> None:
        super().__init__(name="ensemble_v1", params=params)
        self.sub_strategies: List[Strategy] = sub_strategies

        # Per-strategy running return log: strategy index -> list of bar returns
        self._strategy_returns: Dict[int, List[float]] = {
            i: [] for i in range(len(sub_strategies))
        }
        # Previous bar weights per sub-strategy (to compute implied PnL)
        self._prev_weights: Dict[int, Weights] = {
            i: {} for i in range(len(sub_strategies))
        }
        self._last_rebalance_idx: int = -1

    # ------------------------------------------------------------------
    # Parameters
    # ------------------------------------------------------------------

    @property
    def rebalance_interval_bars(self) -> int:
        return int(self.params.get("rebalance_interval_bars") or 168)

    @property
    def target_gross_leverage(self) -> float:
        return float(self.params.get("target_gross_leverage") or 0.30)

    @property
    def perf_window_bars(self) -> int:
        return int(self.params.get("perf_window_bars") or 720)

    @property
    def min_strategy_weight(self) -> float:
        return float(self.params.get("min_strategy_weight") or 0.10)

    @property
    def warmup_bars(self) -> int:
        return int(self.params.get("warmup_bars") or 720)

    # ------------------------------------------------------------------
    # Strategy interface
    # ------------------------------------------------------------------

    def should_rebalance(self, dataset: MarketDataset, idx: int) -> bool:
        if idx < self.warmup_bars:
            return False
        interval = max(1, self.rebalance_interval_bars)
        return idx % interval == 0

    def target_weights(
        self,
        dataset: MarketDataset,
        idx: int,
        current: Weights,
    ) -> Weights:
        if not self.sub_strategies:
            return {s: 0.0 for s in dataset.symbols}

        # 1. Collect weights from each sub-strategy
        sub_weights: List[Weights] = []
        for i, strat in enumerate(self.sub_strategies):
            try:
                if strat.should_rebalance(dataset, idx):
                    w = strat.target_weights(dataset, idx, self._prev_weights[i])
                    self._prev_weights[i] = w
                else:
                    w = self._prev_weights[i]
            except Exception:
                w = self._prev_weights[i]
            sub_weights.append(w)

        # 2. Update per-strategy implied returns from previous bar
        if idx > 0:
            self._update_strategy_returns(dataset, idx, sub_weights)

        # 3. Compute Sharpe-based strategy allocation weights
        strategy_allocation = self._compute_strategy_allocation()

        # 4. Blend sub-strategy weights into a single portfolio
        blended = self._blend_weights(sub_weights, strategy_allocation, dataset.symbols)

        # 5. Rescale to target gross leverage
        return self._rescale(blended, self.target_gross_leverage, dataset.symbols)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _update_strategy_returns(
        self,
        dataset: MarketDataset,
        idx: int,
        current_weights: List[Weights],
    ) -> None:
        """
        Compute and record the implied one-bar PnL for each sub-strategy
        based on its previous holdings and the latest bar returns.
        """
        for i, strat_weights in enumerate(current_weights):
            bar_ret = 0.0
            for sym in dataset.symbols:
                w = strat_weights.get(sym, 0.0)
                if abs(w) < 1e-10:
                    continue
                c_prev = dataset.perp_close[sym][idx - 1]
                c_curr = dataset.perp_close[sym][idx]
                if c_prev > 0:
                    sym_ret = (c_curr / c_prev) - 1.0
                    bar_ret += w * sym_ret
            self._strategy_returns[i].append(bar_ret)
            # Keep only the rolling window
            window = self.perf_window_bars
            if len(self._strategy_returns[i]) > window:
                self._strategy_returns[i] = self._strategy_returns[i][-window:]

    def _compute_sharpe(self, returns: List[float]) -> float:
        """Compute annualised Sharpe ratio from a list of bar returns."""
        if len(returns) < 2:
            return 0.0
        try:
            mu = statistics.mean(returns)
            sigma = statistics.pstdev(returns)
        except Exception:
            return 0.0
        if sigma <= 0.0:
            return 0.0
        # Annualise (assuming hourly bars -> ~8760 bars/year)
        bars_per_year = 8760.0
        return (mu / sigma) * math.sqrt(bars_per_year)

    def _compute_strategy_allocation(self) -> List[float]:
        """
        Compute normalised allocation weights across sub-strategies using
        Sharpe-based weighting with a minimum weight floor.

        Returns
        -------
        List of floats summing to 1.0; one entry per sub-strategy.
        """
        n = len(self.sub_strategies)
        if n == 0:
            return []
        if n == 1:
            return [1.0]

        sharpes = [
            self._compute_sharpe(self._strategy_returns[i]) for i in range(n)
        ]

        # Positive Sharpe contributions only
        pos_sharpes = [max(0.0, s) for s in sharpes]
        total_pos = sum(pos_sharpes)

        min_w = max(0.0, min(self.min_strategy_weight, 1.0 / n))

        if total_pos <= 0.0:
            # All non-positive Sharpe: equal weight fallback
            return [1.0 / n] * n

        # Sharpe-proportional weights
        raw_weights = [s / total_pos for s in pos_sharpes]

        # Apply minimum weight floor then renormalise
        floored = [max(min_w, w) for w in raw_weights]
        total_floored = sum(floored)
        if total_floored <= 0.0:
            return [1.0 / n] * n
        return [w / total_floored for w in floored]

    def _blend_weights(
        self,
        sub_weights: List[Weights],
        strategy_allocation: List[float],
        symbols: List[str],
    ) -> Dict[str, float]:
        """
        Blend sub-strategy weight dicts using strategy_allocation weights.

        blended[sym] = sum_i(strategy_allocation[i] * sub_weights[i][sym])
        """
        blended: Dict[str, float] = {s: 0.0 for s in symbols}
        for i, alloc in enumerate(strategy_allocation):
            if alloc <= 0.0:
                continue
            w_dict = sub_weights[i] if i < len(sub_weights) else {}
            for sym in symbols:
                blended[sym] += alloc * w_dict.get(sym, 0.0)
        return blended

    @staticmethod
    def _rescale(
        weights: Dict[str, float],
        target_gross: float,
        symbols: List[str],
    ) -> Dict[str, float]:
        """Scale weights so sum-of-abs equals target_gross."""
        gross = sum(abs(weights.get(s, 0.0)) for s in symbols)
        if gross <= 0.0:
            return {s: 0.0 for s in symbols}
        scale = target_gross / gross
        return {s: weights.get(s, 0.0) * scale for s in symbols}

    def describe(self) -> Dict[str, Any]:
        base = super().describe()
        base["sub_strategies"] = [s.describe() for s in self.sub_strategies]
        return base
