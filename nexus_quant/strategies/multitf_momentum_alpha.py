"""
Multi-Timeframe Momentum Alpha — blend of momentum signals at 4 horizons.

Economic Foundation:
  Single-timeframe momentum is noisy. Blending signals at 24h, 72h, 168h, 336h
  captures persistent cross-sectional momentum that agrees across multiple
  frequencies — higher conviction = better signal quality.

  Orthogonal to V1 because V1 uses exactly one horizon (168h). This strategy
  uses ALL four horizons with different weights, capturing both short-term and
  long-term momentum simultaneously.

Signal:
  1. Compute momentum return for each symbol at 4 horizons (24h, 72h, 168h, 336h)
  2. Cross-sectional z-score at each horizon
  3. Weighted blend: slower horizons get higher weight (longer persistence)
  4. Long top-k, Short bottom-k, inverse-vol weighted

Parameters:
  k_per_side                int   = 2
  w_24h                     float = 0.10   (short-term, noisy → low weight)
  w_72h                     float = 0.20   (3-day swing)
  w_168h                    float = 0.35   (weekly trend)
  w_336h                    float = 0.35   (2-week trend, persistent)
  vol_lookback_bars         int   = 168
  target_gross_leverage     float = 0.35
  rebalance_interval_bars   int   = 48
"""
from __future__ import annotations

from typing import Any, Dict

from ._math import normalize_dollar_neutral, trailing_vol, zscores
from .base import Strategy, Weights
from ..data.schema import MarketDataset


class MultiTFMomentumAlphaStrategy(Strategy):
    """Multi-timeframe momentum blend across 24h/72h/168h/336h horizons."""

    def __init__(self, params: Dict[str, Any]) -> None:
        super().__init__(name="multitf_momentum_alpha", params=params)

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
        interval = self._p("rebalance_interval_bars", 48)
        warmup = 336 + 10  # longest lookback
        if idx <= warmup:
            return False
        return idx % max(1, interval) == 0

    def _momentum(self, closes, idx: int, lookback: int) -> float:
        i1 = min(idx - 1, len(closes) - 1)
        i0 = max(0, idx - 1 - lookback)
        c1 = float(closes[i1]) if i1 >= 0 else 0.0
        c0 = float(closes[i0]) if i0 >= 0 and i0 < len(closes) else 0.0
        return (c1 / c0 - 1.0) if c0 > 0 else 0.0

    def target_weights(self, dataset: MarketDataset, idx: int, current: Weights) -> Weights:
        syms = dataset.symbols
        k = max(1, min(self._p("k_per_side", 2), len(syms) // 2))
        w24 = self._p("w_24h", 0.10)
        w72 = self._p("w_72h", 0.20)
        w168 = self._p("w_168h", 0.35)
        w336 = self._p("w_336h", 0.35)
        vol_lb = self._p("vol_lookback_bars", 168)
        target_gross = self._p("target_gross_leverage", 0.35)

        # Momentum at each horizon
        mom24: Dict[str, float] = {}
        mom72: Dict[str, float] = {}
        mom168: Dict[str, float] = {}
        mom336: Dict[str, float] = {}

        for s in syms:
            c = dataset.perp_close[s]
            mom24[s] = self._momentum(c, idx, 24)
            mom72[s] = self._momentum(c, idx, 72)
            mom168[s] = self._momentum(c, idx, 168)
            mom336[s] = self._momentum(c, idx, 336)

        # Cross-sectional z-scores at each horizon
        z24 = zscores(mom24)
        z72 = zscores(mom72)
        z168 = zscores(mom168)
        z336 = zscores(mom336)

        # Weighted blend
        score: Dict[str, float] = {}
        for s in syms:
            score[s] = (
                w24 * z24.get(s, 0.0)
                + w72 * z72.get(s, 0.0)
                + w168 * z168.get(s, 0.0)
                + w336 * z336.get(s, 0.0)
            )

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
