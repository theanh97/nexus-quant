"""
Skip-Gram Momentum Alpha — Momentum with short-term reversal avoidance.

Economic Foundation:
  Research shows 1-5 day reversal in crypto markets: the most recent returns
  often reverse in the short run while medium-term momentum persists.

  "Skip-gram" technique (borrowed from NLP): measure the signal window but
  EXCLUDE the most recent skip_bars period. This avoids picking up on
  temporary price dislocations that are likely to revert.

  Formula:
    signal = (price[idx - skip_bars] / price[idx - lookback_bars]) - 1
    (i.e., return from -lookback to -skip, ignoring the most recent skip bars)

  Academic basis: Jegadeesh & Titman (1993) use 12-month lookback skipping
  1 month. Liu & Timmermann (2013) find 1-week reversal in crypto.

Strategy:
  Cross-sectional ranking by skip-gram return
  Long top-k, Short bottom-k (vol-weighted, dollar-neutral)

Parameters:
  k_per_side              int   = 2
  lookback_bars           int   = 437   full lookback window
  skip_bars               int   = 24    skip most recent N bars (1 day)
  vol_lookback_bars       int   = 168
  target_gross_leverage   float = 0.30
  rebalance_interval_bars int   = 48
"""
from __future__ import annotations

from typing import Any, Dict

from ._math import normalize_dollar_neutral, trailing_vol, zscores
from .base import Strategy, Weights
from ..data.schema import MarketDataset


class SkipGramMomentumAlphaStrategy(Strategy):
    """Momentum with short-term reversal skip."""

    def __init__(self, params: Dict[str, Any]) -> None:
        super().__init__(name="skip_gram_momentum_alpha", params=params)

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
        lookback = self._p("lookback_bars", 437)
        interval = self._p("rebalance_interval_bars", 48)
        warmup = lookback + 10
        if idx <= warmup:
            return False
        return idx % max(1, interval) == 0

    def _skip_gram_return(self, closes, idx: int, lookback: int, skip: int) -> float:
        """Return from (idx - lookback) to (idx - skip), skipping recent period."""
        i_end = max(0, idx - 1 - skip)       # end = current - skip
        i_start = max(0, idx - 1 - lookback)  # start = current - lookback
        if i_start >= i_end or i_start >= len(closes) or i_end >= len(closes):
            return 0.0
        c0 = float(closes[i_start])
        c1 = float(closes[i_end])
        if c0 <= 0:
            return 0.0
        return (c1 / c0) - 1.0

    def target_weights(self, dataset: MarketDataset, idx: int, current: Weights) -> Weights:
        syms = dataset.symbols
        k = max(1, min(self._p("k_per_side", 2), len(syms) // 2))
        lookback = self._p("lookback_bars", 437)
        skip = self._p("skip_bars", 24)
        vol_lb = self._p("vol_lookback_bars", 168)
        target_gross = self._p("target_gross_leverage", 0.30)

        scores_raw = {}
        for s in syms:
            scores_raw[s] = self._skip_gram_return(dataset.perp_close[s], idx, lookback, skip)

        score = {s: float(v) for s, v in zscores(scores_raw).items()}

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
