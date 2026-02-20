"""
Momentum Breakout Alpha — Binary threshold signal
===================================================
Economic Foundation:
  Standard cross-sectional momentum uses LINEAR rankings of returns.
  But the alpha generation is NON-LINEAR: coins that break above a
  meaningful threshold (e.g., >0% return over lookback) have qualitatively
  different dynamics than those just below.

  This strategy applies a THRESHOLD-BASED approach:
  - Only coins with returns above a positive threshold are long candidates
  - Only coins with returns below a negative threshold are short candidates
  - Coins in the "dead zone" (between thresholds) are ignored
  - Within the active set, rank by magnitude for sizing

  Why this might be orthogonal:
  - Linear momentum says coin at +5% is "a bit better" than coin at +3%
  - Breakout says: if both are above threshold, they're both strong; if below, both weak
  - The threshold creates a different basket composition than linear ranking

Strategy:
  1. Compute return over lookback for each symbol
  2. Apply threshold filter: only keep |return| > threshold
  3. Long: top-k among positive-threshold coins
  4. Short: bottom-k among negative-threshold coins
  5. If insufficient candidates on one side, reduce exposure

Parameters:
  lookback_bars           int   = 336   Return window
  threshold               float = 0.05  Min |return| to be active (5%)
  k_per_side              int   = 2
  vol_lookback_bars       int   = 168
  target_gross_leverage   float = 0.30
  rebalance_interval_bars int   = 48
"""
from __future__ import annotations

import statistics
from typing import Any, Dict, List

from ._math import normalize_dollar_neutral, trailing_vol
from .base import Strategy, Weights
from ..data.schema import MarketDataset


class MomentumBreakoutAlphaStrategy(Strategy):
    """Momentum with binary threshold activation."""

    def __init__(self, params: Dict[str, Any]) -> None:
        super().__init__(name="momentum_breakout_alpha", params=params)

    def _p(self, key: str, default: Any) -> Any:
        v = self.params.get(key)
        if v is None:
            return default
        try:
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
        if idx <= lookback + 10:
            return False
        return idx % max(1, interval) == 0

    def target_weights(self, dataset: MarketDataset, idx: int, current: Weights) -> Weights:
        syms = dataset.symbols
        lookback = self._p("lookback_bars", 336)
        threshold = self._p("threshold", 0.05)
        k = max(1, min(self._p("k_per_side", 2), len(syms) // 2))
        vol_lb = self._p("vol_lookback_bars", 168)
        target_gross = self._p("target_gross_leverage", 0.30)

        # Compute returns
        rets: Dict[str, float] = {}
        for s in syms:
            closes = dataset.perp_close[s]
            i1 = min(idx - 1, len(closes) - 1)
            i0 = max(0, idx - 1 - lookback)
            c1 = float(closes[i1]) if i1 >= 0 else 0.0
            c0 = float(closes[i0]) if i0 >= 0 and i0 < len(closes) else 0.0
            rets[s] = (c1 / c0 - 1.0) if c0 > 0 else 0.0

        # Filter by threshold
        long_candidates = [(s, rets[s]) for s in syms if rets[s] > threshold]
        short_candidates = [(s, rets[s]) for s in syms if rets[s] < -threshold]

        # Sort by magnitude
        long_candidates.sort(key=lambda x: x[1], reverse=True)
        short_candidates.sort(key=lambda x: x[1])

        long_syms = [s for s, _ in long_candidates[:k]]
        short_syms = [s for s, _ in short_candidates[:k]]

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
