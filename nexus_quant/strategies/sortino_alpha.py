"""
Sortino Ratio Alpha — Cross-sectional ranking by historical Sortino ratio.

Economic Foundation:
  Plain Sharpe Ratio penalizes both upside and downside volatility equally.
  The Sortino Ratio penalizes only DOWNSIDE volatility (returns below zero),
  which is more appropriate for crypto assets that have:
  - Positively skewed returns (occasional large upside)
  - Irregular downside crashes

  Cross-sectionally: assets with high historical Sortino ratios have
  good risk-adjusted returns with low downside tail risk — these are the
  "quality momentum" candidates.

  Orthogonal to Sharpe Ratio (Phase 60) in that it penalizes differently:
  - High-vol assets that are mostly positive → same ranking in both
  - High-vol assets with mixed direction → different ranking (Sortino favors
    if downside vol is low even if total vol is high)

Signal:
  1. Compute hourly return series per symbol over lookback_bars
  2. Sortino = mean(returns) / downside_stdev × sqrt(N)
     where downside_stdev = stdev of returns below 0
  3. Cross-sectional z-score → rank by Sortino
  4. Long top-k (highest risk-adj momentum), Short bottom-k

Parameters:
  k_per_side              int   = 2
  lookback_bars           int   = 336   (14 days, same as Sharpe champion)
  target_return           float = 0.0   (MAR: minimum acceptable return)
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


class SortinoAlphaStrategy(Strategy):
    """Sortino Ratio cross-section: quality momentum with downside-only penalty."""

    def __init__(self, params: Dict[str, Any]) -> None:
        super().__init__(name="sortino_alpha", params=params)

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
        lookback = self._p("lookback_bars", 336)
        interval = self._p("rebalance_interval_bars", 48)
        warmup = lookback + 10
        if idx <= warmup:
            return False
        return idx % max(1, interval) == 0

    def _sortino(self, closes: List[float], idx: int, lookback: int, mar: float) -> float:
        """
        Compute Sortino ratio over past lookback bars.
        mar: minimum acceptable return (typically 0.0)
        """
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

        # Downside deviation: RMS of returns below MAR
        downside_rets = [r - mar for r in rets if r < mar]
        if len(downside_rets) < 3:
            # No downside — all returns >= MAR
            # High positive Sortino (asset never lost money)
            # Use a large value proportional to mean
            return mu * math.sqrt(len(rets)) * 10.0 if mu > 0 else 0.0

        # Downside variance
        downside_var = sum(dr ** 2 for dr in downside_rets) / len(rets)  # use full N denominator
        if downside_var < 1e-16:
            return mu * math.sqrt(len(rets)) * 10.0 if mu > 0 else 0.0

        downside_std = math.sqrt(downside_var)
        # Sortino proxy (scaled same as Sharpe for comparability)
        return (mu / downside_std) * math.sqrt(len(rets))

    def target_weights(self, dataset: MarketDataset, idx: int, current: Weights) -> Weights:
        syms = dataset.symbols
        k = max(1, min(self._p("k_per_side", 2), len(syms) // 2))
        lookback = self._p("lookback_bars", 336)
        mar = self._p("target_return", 0.0)
        vol_lb = self._p("vol_lookback_bars", 168)
        target_gross = self._p("target_gross_leverage", 0.35)

        # Sortino ratio per symbol
        sortino_raw: Dict[str, float] = {}
        for s in syms:
            sortino_raw[s] = self._sortino(dataset.perp_close[s], idx, lookback, mar)

        # Cross-sectional z-score
        sortino_z = zscores(sortino_raw)
        score = {s: float(sortino_z.get(s, 0.0)) for s in syms}

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
