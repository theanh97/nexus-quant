"""
Volume Reversal Alpha — Capitulation / climax detection.

Hypothesis: Extreme volume + negative return = capitulation = buy.
            Extreme volume + positive return = distribution/climax = sell.

Signal:
  1. Compute volume z-score (rolling window)
  2. Compute return sign over lookback
  3. Capitulation score = volume_zscore × (-return_sign)
  4. High score → capitulation → long; Low score → climax → short
  5. Cross-sectional ranking, top-k long / bottom-k short

Why orthogonal to V1: Volume-based, not price-momentum or carry.

Parameters:
  k_per_side              int   = 2
  volume_lookback_bars    int   = 168   rolling volume z-score window
  return_lookback_bars    int   = 24    return sign lookback
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


class VolumeReversalAlphaStrategy(Strategy):
    """Volume reversal: buy capitulation, sell climax."""

    def __init__(self, params: Dict[str, Any]) -> None:
        super().__init__(name="volume_reversal_alpha", params=params)

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
        vol_lb = self._p("volume_lookback_bars", 168)
        interval = self._p("rebalance_interval_bars", 24)
        warmup = vol_lb + 10
        if idx <= warmup:
            return False
        return idx % max(1, interval) == 0

    def _volume_zscore(self, dataset: MarketDataset, symbol: str, idx: int) -> float:
        """Rolling z-score of current volume vs trailing window."""
        vol_lb = self._p("volume_lookback_bars", 168)
        volumes = dataset.perp_volume
        if volumes is None:
            return 0.0
        vol_series = volumes.get(symbol)
        if vol_series is None:
            return 0.0

        end = min(idx, len(vol_series))
        start = max(0, end - vol_lb)
        window = [float(vol_series[i]) for i in range(start, end) if i < len(vol_series)]

        if len(window) < 20:
            return 0.0

        current = window[-1] if window else 0.0
        mu = statistics.mean(window)
        sd = statistics.pstdev(window)
        if sd < 1e-10:
            return 0.0
        return (current - mu) / sd

    def target_weights(self, dataset: MarketDataset, idx: int, current: Weights) -> Weights:
        syms = dataset.symbols
        k = max(1, min(self._p("k_per_side", 2), len(syms) // 2))
        ret_lb = self._p("return_lookback_bars", 24)
        target_gross = self._p("target_gross_leverage", 0.30)
        vol_lb = self._p("vol_lookback_bars", 168)

        # Check if volume data is available
        if dataset.perp_volume is None:
            return {s: 0.0 for s in syms}

        signal: Dict[str, float] = {}
        for s in syms:
            # Volume z-score
            vz = self._volume_zscore(dataset, s, idx)

            # Return over lookback
            c = dataset.perp_close[s]
            i1 = min(idx - 1, len(c) - 1)
            i0 = max(0, idx - 1 - ret_lb)
            c1 = float(c[i1]) if i1 >= 0 else 0.0
            c0 = float(c[i0]) if i0 >= 0 and i0 < len(c) else 0.0
            ret = (c1 / c0 - 1.0) if c0 > 0 else 0.0

            # Capitulation score: high volume × negative return = buy opportunity
            # volume_zscore × (-return) → high when volume spikes on selloff
            signal[s] = vz * (-ret)

        sz = zscores(signal)
        signal_z = {s: float(sz.get(s, 0.0)) for s in syms}

        ranked = sorted(syms, key=lambda s: signal_z.get(s, 0.0), reverse=True)
        long_syms = ranked[:k]     # highest capitulation score → long
        short_syms = ranked[-k:]   # lowest (distribution/climax) → short

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
