"""
Amihud Illiquidity Alpha — Cross-sectional illiquidity momentum signal.

Economic Foundation:
  Amihud (2002): Illiquidity = |return| / dollar_volume. Illiquid assets have
  higher price impact per unit of trading — a dollar of trading moves the price
  more. Cross-sectionally: more illiquid assets earn a liquidity premium and
  amplify momentum effects (less efficient price discovery → trends persist).

  Adaptation for crypto:
  - Use |hourly_return| / perp_volume as the bar-level illiquidity measure
  - Average over lookback_bars for stability
  - Combine illiquidity with momentum direction:
      score = amihud_z × momentum_z  (interaction term)
    This means: LONG illiquid assets with positive momentum
                SHORT illiquid assets with negative momentum
    Pure illiquidity alone would just bet on small-cap assets.

Parameters:
  k_per_side              int   = 2
  lookback_bars           int   = 168   for Amihud estimation
  mom_lookback_bars       int   = 168   for momentum direction
  use_interaction         bool  = True  multiply Amihud × momentum
  vol_lookback_bars       int   = 168
  target_gross_leverage   float = 0.30
  rebalance_interval_bars int   = 48
"""
from __future__ import annotations

import math
from typing import Any, Dict, List

from ._math import normalize_dollar_neutral, trailing_vol, zscores
from .base import Strategy, Weights
from ..data.schema import MarketDataset


class AmihudIlliquidityAlphaStrategy(Strategy):
    """Amihud illiquidity × momentum interaction: amplified cross-sectional signal."""

    def __init__(self, params: Dict[str, Any]) -> None:
        super().__init__(name="amihud_illiquidity_alpha", params=params)

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
        lookback = max(self._p("lookback_bars", 168), self._p("mom_lookback_bars", 168))
        interval = self._p("rebalance_interval_bars", 48)
        warmup = lookback + 10
        if idx <= warmup:
            return False
        return idx % max(1, interval) == 0

    def _amihud(
        self,
        closes: List[float],
        volume: List[float],
        idx: int,
        lookback: int,
    ) -> float:
        """
        Amihud illiquidity proxy: mean(|ret_t| / vol_t) over lookback bars.
        Returns 0.0 if volume data unavailable.
        """
        start = max(1, idx - lookback)
        end = min(idx, len(closes), len(volume))
        measures: List[float] = []
        for i in range(start, end):
            c0 = closes[i - 1]
            c1 = closes[i]
            v = volume[i]
            if c0 <= 0 or v <= 0:
                continue
            ret = abs((c1 / c0) - 1.0)
            measures.append(ret / v)
        if not measures:
            return 0.0
        return sum(measures) / len(measures)

    def _momentum(self, closes: List[float], idx: int, lookback: int) -> float:
        i1 = min(idx - 1, len(closes) - 1)
        i0 = max(0, idx - 1 - lookback)
        c1 = float(closes[i1]) if i1 >= 0 else 0.0
        c0 = float(closes[i0]) if i0 >= 0 and i0 < len(closes) else 0.0
        return (c1 / c0 - 1.0) if c0 > 0 else 0.0

    def target_weights(self, dataset: MarketDataset, idx: int, current: Weights) -> Weights:
        syms = dataset.symbols
        k = max(1, min(self._p("k_per_side", 2), len(syms) // 2))
        lookback = self._p("lookback_bars", 168)
        mom_lb = self._p("mom_lookback_bars", 168)
        use_interaction = self._p("use_interaction", True)
        vol_lb = self._p("vol_lookback_bars", 168)
        target_gross = self._p("target_gross_leverage", 0.30)

        # Amihud illiquidity per symbol
        amihud_raw: Dict[str, float] = {}
        mom_raw: Dict[str, float] = {}
        has_volume = dataset.perp_volume is not None

        for s in syms:
            closes = dataset.perp_close[s]
            if has_volume and dataset.perp_volume.get(s):
                volume = dataset.perp_volume[s]
                amihud_raw[s] = self._amihud(closes, volume, idx, lookback)
            else:
                amihud_raw[s] = 0.0
            mom_raw[s] = self._momentum(closes, idx, mom_lb)

        # Z-scores
        amihud_z = zscores(amihud_raw)
        mom_z = zscores(mom_raw)

        # Score: interaction (illiquidity amplifies momentum direction)
        # or pure amihud z-score if use_interaction is False
        score: Dict[str, float] = {}
        for s in syms:
            az = amihud_z.get(s, 0.0)
            mz = mom_z.get(s, 0.0)
            if use_interaction:
                # Interaction: high illiquidity AND positive momentum = strong long
                # Use product then re-center
                score[s] = az * mz
            else:
                # Pure illiquidity momentum: trade in direction of illiquidity ranking
                score[s] = az

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
