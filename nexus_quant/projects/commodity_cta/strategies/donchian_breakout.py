"""
Strategy: Donchian Channel Breakout (Turtle Trading)
=====================================================
Phase 141 v2: Fixed for narrow commodity universe.

Key fixes vs v1:
  - Continuous channel position signal (not just breakout)
  - Lower vol target (0.08) for 8-instrument universe
  - Net leverage constraint
  - Safe division handling
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional

from nexus_quant.strategies.base import Strategy, Weights
from nexus_quant.data.schema import MarketDataset
from .trend_following import _get, _normalise
from .tsmom import _clip, _clip_abs, _apply_net_limit

_EPS = 1e-10


class DonchianBreakoutStrategy(Strategy):
    """
    Multi-channel Donchian breakout with continuous position signal.
    """

    def __init__(
        self,
        name: str = "cta_donchian",
        params: Optional[Dict[str, Any]] = None,
    ) -> None:
        p = params or {}
        super().__init__(name, p)

        # Channel weights
        self.w_fast: float = float(p.get("w_fast", 0.20))
        self.w_medium: float = float(p.get("w_medium", 0.40))
        self.w_slow: float = float(p.get("w_slow", 0.40))

        # Vol target per instrument
        self.vol_target: float = float(p.get("vol_target", 0.08))

        # Portfolio constraints
        self.max_gross_leverage: float = float(p.get("max_gross_leverage", 1.5))
        self.max_position: float = float(p.get("max_position", 0.20))
        self.max_net_leverage: float = float(p.get("max_net_leverage", 0.8))

        self.warmup: int = int(p.get("warmup", 130))
        self.rebalance_freq: int = int(p.get("rebalance_freq", 10))
        self._last_rebalance: int = -1

    def should_rebalance(self, dataset: MarketDataset, idx: int) -> bool:
        if idx < self.warmup:
            return False
        if self._last_rebalance < 0 or (idx - self._last_rebalance) >= self.rebalance_freq:
            self._last_rebalance = idx
            return True
        return False

    def target_weights(
        self, dataset: MarketDataset, idx: int, current: Weights
    ) -> Weights:
        if idx < self.warmup:
            return {}

        symbols = dataset.symbols
        dc20_h = dataset.features.get("donchian_20_high", {})
        dc20_l = dataset.features.get("donchian_20_low", {})
        dc55_h = dataset.features.get("donchian_55_high", {})
        dc55_l = dataset.features.get("donchian_55_low", {})
        dc120_h = dataset.features.get("donchian_120_high", {})
        dc120_l = dataset.features.get("donchian_120_low", {})
        rv = dataset.features.get("rv_20d", {})

        raw_weights: Dict[str, float] = {}

        for sym in symbols:
            price = dataset.close(sym, idx)
            if price <= 0:
                continue

            # Continuous channel position signals
            signals = []
            horizon_weights = []

            s20 = _channel_position(price, _get(dc20_h, sym, idx), _get(dc20_l, sym, idx))
            if s20 is not None and self.w_fast > _EPS:
                signals.append(s20)
                horizon_weights.append(self.w_fast)

            s55 = _channel_position(price, _get(dc55_h, sym, idx), _get(dc55_l, sym, idx))
            if s55 is not None and self.w_medium > _EPS:
                signals.append(s55)
                horizon_weights.append(self.w_medium)

            s120 = _channel_position(price, _get(dc120_h, sym, idx), _get(dc120_l, sym, idx))
            if s120 is not None and self.w_slow > _EPS:
                signals.append(s120)
                horizon_weights.append(self.w_slow)

            if len(signals) < 2:
                continue

            total_w = sum(horizon_weights)
            if total_w < _EPS:
                continue
            combo = sum(s * w for s, w in zip(signals, horizon_weights)) / total_w

            if abs(combo) < 0.15:
                continue

            # Vol-targeting
            vol = _get(rv, sym, idx)
            if vol is None or vol < _EPS:
                vol = 0.20
            weight = combo * (self.vol_target / max(vol, _EPS))
            weight = _clip_abs(weight, self.max_position)
            raw_weights[sym] = weight

        if not raw_weights:
            return {}

        weights = _normalise(raw_weights, self.max_gross_leverage, self.max_position)
        weights = _apply_net_limit(weights, self.max_net_leverage)
        return weights


def _channel_position(
    price: float,
    high: Optional[float],
    low: Optional[float],
) -> Optional[float]:
    """
    Continuous Donchian channel position: [-1, +1].
    +1 at channel high, -1 at channel low, 0 at midpoint.
    Returns None if channel data missing.
    """
    if high is None or low is None or high <= 0 or low <= 0:
        return None
    if high <= low:
        return None
    mid = (high + low) / 2.0
    half_range = (high - low) / 2.0
    if half_range < _EPS:
        return None
    return _clip((price - mid) / half_range)


__all__ = ["DonchianBreakoutStrategy"]
