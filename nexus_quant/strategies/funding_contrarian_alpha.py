"""
Funding Contrarian Alpha — Short crowded trades.

Hypothesis: Extremely high funding + high momentum = crowded long → reversal coming.
            Extremely low funding + low momentum = capitulation short → bounce coming.

This is the OPPOSITE of V1's carry signal. V1 says "low funding → long" (level-based).
This says "extreme funding in SAME direction as momentum → reversal" (contrarian).

Signal:
  1. Compute funding z-score (carry_z) and momentum z-score (mom_z)
  2. Crowding score = carry_z * mom_z (both high = crowded)
  3. INVERSE the signal: short the most crowded, long the least crowded
  4. This captures mean-reversion of crowded positions

Why orthogonal to V1: V1 uses carry and momentum in the SAME direction.
This uses them CONTRARIAN when they're both extreme.

Parameters:
  k_per_side              int   = 2
  momentum_lookback_bars  int   = 168
  vol_lookback_bars       int   = 168
  extreme_threshold       float = 0.5   only trade when crowding > threshold
  target_gross_leverage   float = 0.30
  rebalance_interval_bars int   = 48
"""
from __future__ import annotations

from typing import Any, Dict

from ._math import normalize_dollar_neutral, trailing_vol, zscores
from .base import Strategy, Weights
from ..data.schema import MarketDataset


class FundingContrarianAlphaStrategy(Strategy):
    """Funding contrarian: fade crowded trades."""

    def __init__(self, params: Dict[str, Any]) -> None:
        super().__init__(name="funding_contrarian_alpha", params=params)

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
        mom_lb = self._p("momentum_lookback_bars", 168)
        interval = self._p("rebalance_interval_bars", 48)
        warmup = mom_lb + 10
        if idx <= warmup:
            return False
        return idx % max(1, interval) == 0

    def target_weights(self, dataset: MarketDataset, idx: int, current: Weights) -> Weights:
        syms = dataset.symbols
        k = max(1, min(self._p("k_per_side", 2), len(syms) // 2))
        mom_lb = self._p("momentum_lookback_bars", 168)
        target_gross = self._p("target_gross_leverage", 0.30)
        vol_lb = self._p("vol_lookback_bars", 168)

        ts = dataset.timeline[idx]

        # Funding z-score (same as V1 carry)
        f_raw = {s: float(dataset.last_funding_rate_before(s, ts)) for s in syms}
        fz = zscores(f_raw)

        # Momentum z-score
        mom_raw: Dict[str, float] = {}
        for s in syms:
            c = dataset.perp_close[s]
            i1 = min(idx - 1, len(c) - 1)
            i0 = max(0, idx - 1 - mom_lb)
            c1 = float(c[i1]) if i1 >= 0 else 0.0
            c0 = float(c[i0]) if i0 >= 0 and i0 < len(c) else 0.0
            mom_raw[s] = (c1 / c0 - 1.0) if c0 > 0 else 0.0
        mz = zscores(mom_raw)

        # Crowding score = funding_z × momentum_z
        # High positive = both high (crowded long) → short it
        # High negative = one extreme positive, one extreme negative (not crowded)
        # We SHORT the most crowded (highest funding × momentum)
        signal: Dict[str, float] = {}
        for s in syms:
            crowding = float(fz.get(s, 0.0)) * float(mz.get(s, 0.0))
            signal[s] = -crowding  # INVERT: fade crowded positions

        sz = zscores(signal)
        signal_z = {s: float(sz.get(s, 0.0)) for s in syms}

        ranked = sorted(syms, key=lambda s: signal_z.get(s, 0.0), reverse=True)
        long_syms = ranked[:k]     # least crowded → long
        short_syms = ranked[-k:]   # most crowded → short

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
