"""
Price Level Alpha — Distance from N-bar high (anchoring bias momentum).

Economic Foundation:
  Anchoring effect (Kahneman/Tversky): investors use recent price highs as
  reference points. Assets near their N-bar high attract attention, new buyers,
  and breakout-chasing momentum. Assets far below their high face selling pressure
  from underwater holders (resistance at cost basis levels).

  52-week high effect is well-documented in equities (George & Hwang 2004).
  Adapted here for crypto with shorter lookback (crypto cycles faster).

Signal:
  1. Compute rolling N-bar high (max price over past level_lookback_bars)
  2. Score = current_price / rolling_high  → ranges 0 to 1
     (1.0 = at the high, 0.5 = halfway to high, 0.1 = far below high)
  3. Cross-sectional ranking: LONG closest-to-high, SHORT furthest-from-high

  Rationale: Unlike raw momentum (rate of change), this captures the
  POSITION in the price range — even a sideways coin at its high outranks
  a rebounding coin that is still far below its peak.

Parameters:
  k_per_side              int   = 2
  level_lookback_bars     int   = 504   (21 days = 3 weeks of hourly bars)
  vol_lookback_bars       int   = 168
  target_gross_leverage   float = 0.30
  rebalance_interval_bars int   = 48
"""
from __future__ import annotations

from typing import Any, Dict, List

from ._math import normalize_dollar_neutral, trailing_vol, zscores
from .base import Strategy, Weights
from ..data.schema import MarketDataset


class PriceLevelAlphaStrategy(Strategy):
    """Long assets near N-bar high, short assets far below their high."""

    def __init__(self, params: Dict[str, Any]) -> None:
        super().__init__(name="price_level_alpha", params=params)

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
        lookback = self._p("level_lookback_bars", 504)
        interval = self._p("rebalance_interval_bars", 48)
        warmup = lookback + 10
        if idx <= warmup:
            return False
        return idx % max(1, interval) == 0

    def _dist_from_high(self, closes: List[float], idx: int, lookback: int) -> float:
        """Compute current price as fraction of rolling N-bar high."""
        start = max(0, idx - lookback)
        end = min(idx, len(closes))
        if start >= end:
            return 0.5  # neutral
        window = closes[start:end]
        high = max(window)
        current = closes[min(idx - 1, len(closes) - 1)]
        if high <= 0:
            return 0.5
        return float(current) / float(high)

    def target_weights(self, dataset: MarketDataset, idx: int, current: Weights) -> Weights:
        syms = dataset.symbols
        k = max(1, min(self._p("k_per_side", 2), len(syms) // 2))
        lookback = self._p("level_lookback_bars", 504)
        vol_lb = self._p("vol_lookback_bars", 168)
        target_gross = self._p("target_gross_leverage", 0.30)

        # Distance from rolling high for each symbol
        dist_raw: Dict[str, float] = {}
        for s in syms:
            dist_raw[s] = self._dist_from_high(dataset.perp_close[s], idx, lookback)

        # Cross-sectional z-score (higher = closer to high = bullish)
        dist_z = zscores(dist_raw)
        score = {s: float(dist_z.get(s, 0.0)) for s in syms}

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
