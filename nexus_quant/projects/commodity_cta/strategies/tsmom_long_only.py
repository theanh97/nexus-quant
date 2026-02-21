"""
Strategy: TSMOM Long-Only Momentum Timing
==========================================
Phase 141b: Key insight from diagnostic — commodity basket = Sharpe 0.60.
Long-short CTA DESTROYS value by fighting the positive drift.

Approach: Long-only momentum timing
  - In uptrend: fully invested (capture the +10% CAGR drift)
  - In downtrend: go FLAT (not short — avoid fighting drift)
  - Between: scale position with conviction

Signal: multi-scale TSMOM (same as CTA v2)
  combo = weighted avg of clip(tsmom_k / scale, -1, +1) for k in [21,63,126,252]

Position: LONG-ONLY
  weight = max(0, combo) * vol_target / realized_vol

This is equivalent to an "enhanced buy-and-hold" with momentum timing.
Expected: Sharpe > 0.6 (buy-hold baseline) + timing alpha.
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional

from nexus_quant.strategies.base import Strategy, Weights
from nexus_quant.data.schema import MarketDataset
from .trend_following import _get, _normalise

_EPS = 1e-10


class TSMOMLongOnlyStrategy(Strategy):
    """Long-only momentum timing. Go flat in downtrends, invested in uptrends."""

    def __init__(
        self,
        name: str = "cta_tsmom_lo",
        params: Optional[Dict[str, Any]] = None,
    ) -> None:
        p = params or {}
        super().__init__(name, p)

        self.w_21: float = float(p.get("w_21", 0.10))
        self.w_63: float = float(p.get("w_63", 0.20))
        self.w_126: float = float(p.get("w_126", 0.35))
        self.w_252: float = float(p.get("w_252", 0.35))

        # Base weight per instrument when fully invested
        self.base_weight: float = float(p.get("base_weight", 0.125))  # 1/8 for 8 instruments

        # Signal scaling
        self.signal_scale: float = float(p.get("signal_scale", 2.0))

        # Portfolio constraints
        self.max_gross_leverage: float = float(p.get("max_gross_leverage", 1.0))
        self.max_position: float = float(p.get("max_position", 0.20))

        self.warmup: int = int(p.get("warmup", 270))
        self.rebalance_freq: int = int(p.get("rebalance_freq", 21))
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

        tsmom_21 = dataset.features.get("tsmom_21d", {})
        tsmom_63 = dataset.features.get("tsmom_63d", {})
        tsmom_126 = dataset.features.get("tsmom_126d", {})
        tsmom_252 = dataset.features.get("tsmom_252d", {})

        raw_weights: Dict[str, float] = {}

        for sym in dataset.symbols:
            s21 = _get(tsmom_21, sym, idx)
            s63 = _get(tsmom_63, sym, idx)
            s126 = _get(tsmom_126, sym, idx)
            s252 = _get(tsmom_252, sym, idx)

            raw_signals = []
            horizon_weights = []

            if s21 is not None:
                raw_signals.append(_clip(s21 / self.signal_scale))
                horizon_weights.append(self.w_21)
            if s63 is not None:
                raw_signals.append(_clip(s63 / self.signal_scale))
                horizon_weights.append(self.w_63)
            if s126 is not None:
                raw_signals.append(_clip(s126 / self.signal_scale))
                horizon_weights.append(self.w_126)
            if s252 is not None:
                raw_signals.append(_clip(s252 / self.signal_scale))
                horizon_weights.append(self.w_252)

            if len(raw_signals) < 2:
                continue

            total_w = sum(horizon_weights)
            if total_w < _EPS:
                continue
            combo = sum(s * w for s, w in zip(raw_signals, horizon_weights)) / total_w

            # LONG-ONLY: max(0, combo) — flat when bearish
            long_signal = max(0.0, combo)
            if long_signal < 0.05:
                continue

            weight = long_signal * self.base_weight
            weight = min(weight, self.max_position)
            raw_weights[sym] = weight

        if not raw_weights:
            return {}

        return _normalise(raw_weights, self.max_gross_leverage, self.max_position)


def _clip(x: float, lo: float = -1.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))


__all__ = ["TSMOMLongOnlyStrategy"]
