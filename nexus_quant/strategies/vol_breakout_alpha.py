"""
Volatility Breakout Alpha — Trade breakouts after vol compression.

Hypothesis: After vol compression, breakout direction predicts trend.

Signal:
  1. Compute vol ratio = vol_short / vol_long
  2. When ratio < threshold (vol compression) → prepare for breakout
  3. Direction = sign of recent return during breakout
  4. Score = (1 - vol_ratio) × return_direction
     High score = compressed vol + positive breakout → long
     Low score = compressed vol + negative breakout → short
  5. When vol is not compressed (ratio > threshold), score → 0 (no trade)

Why orthogonal to V1: Regime-timing signal based on vol structure, not price levels.

Parameters:
  k_per_side              int   = 2
  vol_short_bars          int   = 24    short-term vol window (1 day)
  vol_long_bars           int   = 168   long-term vol window (1 week)
  compression_threshold   float = 0.7   vol ratio below this = compressed
  return_lookback_bars    int   = 12    recent return for breakout direction
  vol_lookback_bars       int   = 168   for inverse-vol weighting
  target_gross_leverage   float = 0.30
  rebalance_interval_bars int   = 24
"""
from __future__ import annotations

import statistics
from typing import Any, Dict, List

from ._math import normalize_dollar_neutral, trailing_vol, zscores
from .base import Strategy, Weights
from ..data.schema import MarketDataset


class VolBreakoutAlphaStrategy(Strategy):
    """Volatility breakout: trade direction after vol compression."""

    def __init__(self, params: Dict[str, Any]) -> None:
        super().__init__(name="vol_breakout_alpha", params=params)

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
        vol_long = self._p("vol_long_bars", 168)
        interval = self._p("rebalance_interval_bars", 24)
        warmup = vol_long + 10
        if idx <= warmup:
            return False
        return idx % max(1, interval) == 0

    def target_weights(self, dataset: MarketDataset, idx: int, current: Weights) -> Weights:
        syms = dataset.symbols
        k = max(1, min(self._p("k_per_side", 2), len(syms) // 2))
        vol_short_bars = self._p("vol_short_bars", 24)
        vol_long_bars = self._p("vol_long_bars", 168)
        compression_thr = self._p("compression_threshold", 0.7)
        ret_lb = self._p("return_lookback_bars", 12)
        target_gross = self._p("target_gross_leverage", 0.30)
        vol_lb = self._p("vol_lookback_bars", 168)

        signal: Dict[str, float] = {}
        for s in syms:
            c = dataset.perp_close[s]

            # Compute short-term and long-term vol
            v_short = trailing_vol(c, end_idx=idx, lookback_bars=vol_short_bars)
            v_long = trailing_vol(c, end_idx=idx, lookback_bars=vol_long_bars)

            # Vol ratio
            if v_long < 1e-10:
                vol_ratio = 1.0
            else:
                vol_ratio = v_short / v_long

            # Recent return for breakout direction
            i1 = min(idx - 1, len(c) - 1)
            i0 = max(0, idx - 1 - ret_lb)
            c1 = float(c[i1]) if i1 >= 0 else 0.0
            c0 = float(c[i0]) if i0 >= 0 and i0 < len(c) else 0.0
            ret = (c1 / c0 - 1.0) if c0 > 0 else 0.0

            # Breakout score: stronger when vol is more compressed
            if vol_ratio < compression_thr:
                # Compression detected: trade breakout direction
                compression_strength = max(0.0, 1.0 - vol_ratio)
                signal[s] = compression_strength * ret
            else:
                # No compression: weaker signal (still use slight mean-reversion)
                signal[s] = 0.1 * ret

        sz = zscores(signal)
        signal_z = {s: float(sz.get(s, 0.0)) for s in syms}

        ranked = sorted(syms, key=lambda s: signal_z.get(s, 0.0), reverse=True)
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
