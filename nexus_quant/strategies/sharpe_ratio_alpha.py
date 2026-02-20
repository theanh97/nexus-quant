"""
Sharpe Ratio Alpha — Cross-sectional ranking by historical risk-adjusted returns.

Economic Foundation:
  Raw momentum treats a 10% return equally regardless of whether it came with 5%
  or 30% volatility. Risk-adjusted momentum (historical Sharpe) ranks assets by
  the quality of their returns — high Sharpe = consistent, low-risk outperformance.

  In cross-section: assets with higher historical Sharpe ratios attract momentum
  capital more sustainably than high-return/high-vol assets that may revert.

  Orthogonal to V1 because:
  - V1 uses raw momentum return (not risk-adjusted)
  - V1's MR component is separate; here risk-adjustment is embedded in the signal
  - High-vol coins that V1 longs (high momentum) may rank poorly here

Signal:
  1. Compute hourly return series per symbol over lookback_bars
  2. Sharpe proxy = mean(returns) / stdev(returns) × sqrt(N)
  3. Cross-sectional z-score
  4. Long top-k (highest risk-adj momentum), Short bottom-k

Parameters:
  k_per_side              int   = 2
  lookback_bars           int   = 168   (7 days — short enough to be responsive)
  vol_lookback_bars       int   = 168
  target_gross_leverage   float = 0.35
  rebalance_interval_bars int   = 48
"""
from __future__ import annotations

import math
import statistics
from typing import Any, Dict, List

from ._math import normalize_dollar_neutral, trailing_vol, zscores
from .base import Strategy, Weights
from ..data.schema import MarketDataset


class SharpeRatioAlphaStrategy(Strategy):
    """Cross-sectional Sharpe ratio ranking: quality momentum."""

    def __init__(self, params: Dict[str, Any]) -> None:
        super().__init__(name="sharpe_ratio_alpha", params=params)

    def _p(self, key: str, default: Any) -> Any:
        v = self.params.get(key)
        if v is None:
            return default
        try:
            if isinstance(default, bool):
                return bool(v)
            if isinstance(default, int):
                return int(v)
            if isinstance(default, float):
                return float(v)
        except (TypeError, ValueError):
            pass
        return v

    def should_rebalance(self, dataset: MarketDataset, idx: int) -> bool:
        lookback = self._p("lookback_bars", 168)
        interval = self._p("rebalance_interval_bars", 48)
        warmup = lookback + 10
        if idx <= warmup:
            return False
        return idx % max(1, interval) == 0

    def _historical_sharpe(self, closes: List[float], idx: int, lookback: int) -> float:
        """Compute Sharpe proxy over past lookback bars."""
        start = max(1, idx - lookback)
        end = min(idx, len(closes))
        rets: List[float] = []
        for i in range(start, end):
            c0 = closes[i - 1]
            c1 = closes[i]
            if c0 > 0:
                rets.append((c1 / c0) - 1.0)
        if len(rets) < 10:
            return 0.0
        mu = statistics.mean(rets)
        sd = statistics.pstdev(rets)
        if sd < 1e-12:
            return 0.0
        # Annualize (hourly → annual): sqrt(8760) ≈ 93.6
        return (mu / sd) * math.sqrt(len(rets))

    def target_weights(self, dataset: MarketDataset, idx: int, current: Weights) -> Weights:
        syms = dataset.symbols
        k = max(1, min(self._p("k_per_side", 2), len(syms) // 2))
        lookback = self._p("lookback_bars", 168)
        vol_lb = self._p("vol_lookback_bars", 168)
        target_gross = self._p("target_gross_leverage", 0.35)

        # Historical Sharpe per symbol
        sharpe_raw: Dict[str, float] = {}
        for s in syms:
            sharpe_raw[s] = self._historical_sharpe(dataset.perp_close[s], idx, lookback)

        # Cross-sectional z-score
        sharpe_z = zscores(sharpe_raw)
        score = {s: float(sharpe_z.get(s, 0.0)) for s in syms}

        ranked = sorted(syms, key=lambda s: score[s], reverse=True)
        long_syms = ranked[:k]
        short_syms = ranked[-k:]

        long_syms = [s for s in long_syms if s not in short_syms]
        short_syms = [s for s in short_syms if s not in long_syms]

        if not long_syms or not short_syms:
            return {s: 0.0 for s in syms}

        inv_vol: Dict[str, float] = {}
        for s in set(long_syms) | set(short_syms):
            v = trailing_vol(dataset.perp_close[s], end_idx=idx, lookback_bars=vol_lb)
            inv_vol[s] = (1.0 / v) if v > 0 else 1.0

        w = normalize_dollar_neutral(long_syms, short_syms, inv_vol, target_gross)
        out = {s: 0.0 for s in syms}
        out.update(w)
        return out
