"""
Volume-Regime Momentum Alpha — Volume anomaly × momentum interaction
=====================================================================
Economic Foundation:
  Abnormal volume precedes price moves. When a coin has:
  - High relative volume (vs its own trailing avg) + positive momentum → strong trend
  - High relative volume + negative momentum → capitulation (reversal candidate)
  - Low volume + any momentum → noise, ignore

  The KEY DIFFERENCE from pure momentum:
  - Pure momentum ranks ALL coins by return regardless of volume
  - This signal ONLY ranks coins with abnormal volume (conviction filter)
  - Coins with normal/low volume get zero score → trades only conviction moves

Strategy:
  1. For each symbol: compute volume z-score = (vol_now - vol_mean) / vol_std
  2. Momentum = cumulative return over mom_lookback
  3. Signal = momentum × max(0, vol_z - vol_threshold)
     (Only activate when volume is elevated above threshold)
  4. Cross-sectional rank of activated signals
  5. Long top-k, short bottom-k

Parameters:
  mom_lookback_bars       int   = 168   Momentum window
  vol_avg_bars            int   = 336   Volume averaging window
  vol_threshold           float = 0.5   Minimum vol z-score to activate
  k_per_side              int   = 2
  vol_lookback_bars       int   = 168
  target_gross_leverage   float = 0.30
  rebalance_interval_bars int   = 48
"""
from __future__ import annotations

import statistics
from typing import Any, Dict, List

from ._math import normalize_dollar_neutral, trailing_vol, zscores
from .base import Strategy, Weights
from ..data.schema import MarketDataset


class VolRegimeMomAlphaStrategy(Strategy):
    """Volume-regime conditional momentum."""

    def __init__(self, params: Dict[str, Any]) -> None:
        super().__init__(name="vol_regime_mom_alpha", params=params)

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
        vol_avg = self._p("vol_avg_bars", 336)
        interval = self._p("rebalance_interval_bars", 48)
        if idx <= vol_avg + 10:
            return False
        return idx % max(1, interval) == 0

    def target_weights(self, dataset: MarketDataset, idx: int, current: Weights) -> Weights:
        syms = dataset.symbols
        mom_lb = self._p("mom_lookback_bars", 168)
        vol_avg = self._p("vol_avg_bars", 336)
        vol_thresh = self._p("vol_threshold", 0.5)
        k = max(1, min(self._p("k_per_side", 2), len(syms) // 2))
        vol_lb = self._p("vol_lookback_bars", 168)
        target_gross = self._p("target_gross_leverage", 0.30)

        scores: Dict[str, float] = {}

        for s in syms:
            closes = dataset.perp_close[s]
            volumes = dataset.perp_volume.get(s, []) if dataset.perp_volume else []

            # Momentum
            i1 = min(idx - 1, len(closes) - 1)
            i0 = max(0, idx - 1 - mom_lb)
            c1 = float(closes[i1]) if i1 >= 0 else 0.0
            c0 = float(closes[i0]) if i0 >= 0 and i0 < len(closes) else 0.0
            mom = (c1 / c0 - 1.0) if c0 > 0 else 0.0

            # Volume z-score
            vol_z = 0.0
            if volumes and len(volumes) >= vol_avg and idx > vol_avg:
                vol_window = [float(volumes[t]) for t in range(max(0, idx - vol_avg), idx)]
                if len(vol_window) > 10:
                    recent_vol = float(volumes[idx - 1]) if idx - 1 < len(volumes) else 0.0
                    vol_mean = sum(vol_window) / len(vol_window)
                    vol_std = statistics.pstdev(vol_window)
                    if vol_std > 0:
                        vol_z = (recent_vol - vol_mean) / vol_std

            # Interaction: momentum × volume activation
            activation = max(0.0, vol_z - vol_thresh)
            scores[s] = mom * activation

        # If no activated signals, return flat
        if all(v == 0.0 for v in scores.values()):
            return {s: 0.0 for s in syms}

        # Cross-sectional z-scores
        sz = zscores(scores)
        ranked = sorted(syms, key=lambda s: sz.get(s, 0.0), reverse=True)

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
