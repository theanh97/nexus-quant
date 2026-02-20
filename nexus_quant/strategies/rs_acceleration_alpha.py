"""
Relative Strength Acceleration Alpha — Trade acceleration in relative strength.

Hypothesis: Assets accelerating in relative strength outperform.

Signal:
  1. RS_short = rank(return over short window) among universe
  2. RS_long = rank(return over long window) among universe
  3. Acceleration = RS_short - RS_long
  4. Long top-k accelerating, short bottom-k decelerating

Why orthogonal to V1: Measures CHANGE in momentum ranking, not momentum itself.
An asset can have negative momentum but positive acceleration (improving).

Parameters:
  k_per_side              int   = 2
  rs_short_bars           int   = 72    short-term RS window (3 days)
  rs_long_bars            int   = 336   long-term RS window (14 days)
  vol_lookback_bars       int   = 168
  target_gross_leverage   float = 0.30
  rebalance_interval_bars int   = 48
"""
from __future__ import annotations

from typing import Any, Dict, List

from ._math import normalize_dollar_neutral, trailing_vol, zscores
from .base import Strategy, Weights
from ..data.schema import MarketDataset


class RSAccelerationAlphaStrategy(Strategy):
    """Relative strength acceleration: trade improving vs deteriorating assets."""

    def __init__(self, params: Dict[str, Any]) -> None:
        super().__init__(name="rs_acceleration_alpha", params=params)

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
        rs_long = self._p("rs_long_bars", 336)
        interval = self._p("rebalance_interval_bars", 48)
        warmup = rs_long + 10
        if idx <= warmup:
            return False
        return idx % max(1, interval) == 0

    def _compute_returns(self, dataset: MarketDataset, idx: int, lookback: int) -> Dict[str, float]:
        """Compute returns over lookback for all symbols."""
        rets: Dict[str, float] = {}
        for s in dataset.symbols:
            c = dataset.perp_close[s]
            i1 = min(idx - 1, len(c) - 1)
            i0 = max(0, idx - 1 - lookback)
            c1 = float(c[i1]) if i1 >= 0 else 0.0
            c0 = float(c[i0]) if i0 >= 0 and i0 < len(c) else 0.0
            rets[s] = (c1 / c0 - 1.0) if c0 > 0 else 0.0
        return rets

    def _rank(self, values: Dict[str, float]) -> Dict[str, float]:
        """Rank values cross-sectionally (0 = worst, 1 = best)."""
        if not values:
            return {}
        sorted_syms = sorted(values.keys(), key=lambda s: values[s])
        n = len(sorted_syms)
        if n <= 1:
            return {s: 0.5 for s in values}
        return {s: i / (n - 1) for i, s in enumerate(sorted_syms)}

    def target_weights(self, dataset: MarketDataset, idx: int, current: Weights) -> Weights:
        syms = dataset.symbols
        k = max(1, min(self._p("k_per_side", 2), len(syms) // 2))
        rs_short = self._p("rs_short_bars", 72)
        rs_long = self._p("rs_long_bars", 336)
        target_gross = self._p("target_gross_leverage", 0.30)
        vol_lb = self._p("vol_lookback_bars", 168)

        # Compute returns over short and long windows
        rets_short = self._compute_returns(dataset, idx, rs_short)
        rets_long = self._compute_returns(dataset, idx, rs_long)

        # Rank cross-sectionally
        rank_short = self._rank(rets_short)
        rank_long = self._rank(rets_long)

        # Acceleration = short rank - long rank
        # Positive = improving relative strength
        # Negative = deteriorating relative strength
        accel: Dict[str, float] = {}
        for s in syms:
            accel[s] = rank_short.get(s, 0.5) - rank_long.get(s, 0.5)

        sz = zscores(accel)
        signal_z = {s: float(sz.get(s, 0.0)) for s in syms}

        ranked = sorted(syms, key=lambda s: signal_z.get(s, 0.0), reverse=True)
        long_syms = ranked[:k]     # accelerating → long
        short_syms = ranked[-k:]   # decelerating → short

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
