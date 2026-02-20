"""
Strategy #2: Carry / Roll Yield (Contango vs Backwardation Proxy)
=================================================================
Since Yahoo only provides front-month continuous data (no term structure),
we approximate carry using price signals:

  carry_signal = -zscore_60d * sign(mom_120d)

Rationale:
  - Backwardation: price below 60d average in an uptrend → positive carry → LONG
  - Contango   : price above 60d average in an uptrend → negative carry → SHORT
  - Momentum filter (mom_120d sign) ensures we only trade carry in trending mkts

Additionally layered:
  - EWMA signal filter: only trade when medium-term trend confirms direction
  - Equal risk contribution sizing (inverse-vol)

Rebalance: weekly (carry signals are slow-moving)
"""
from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from nexus_quant.strategies.base import Strategy, Weights
from nexus_quant.data.schema import MarketDataset
from .trend_following import _get, _safe_price, _normalise

_EPS = 1e-10


class CarryRollStrategy(Strategy):
    """
    Carry proxy using price mean-reversion within trend filter.
    Weekly rebalance, equal risk contribution sizing.
    """

    def __init__(
        self, name: str = "cta_carry", params: Optional[Dict[str, Any]] = None
    ) -> None:
        p = params or {}
        super().__init__(name, p)

        self.max_gross_leverage: float = float(p.get("max_gross_leverage", 1.5))
        self.max_position: float = float(p.get("max_position", 0.20))
        self.warmup: int = int(p.get("warmup", 130))

        # Carry signal weight vs momentum filter strength
        self.carry_weight: float = float(p.get("carry_weight", 1.0))

        # Only take positions where |mom_120d| > this threshold
        self.trend_threshold: float = float(p.get("trend_threshold", 0.0))

        # Weekly rebalance: rebalance every N bars
        self.rebalance_freq: int = int(p.get("rebalance_freq", 5))

        self._last_rebalance: int = -1

    # ── Rebalance cadence ────────────────────────────────────────────────────

    def should_rebalance(self, dataset: MarketDataset, idx: int) -> bool:
        if idx < self.warmup:
            return False
        # Weekly: rebalance if it's been rebalance_freq bars since last rebalance
        if self._last_rebalance < 0 or (idx - self._last_rebalance) >= self.rebalance_freq:
            self._last_rebalance = idx
            return True
        return False

    # ── Weight computation ───────────────────────────────────────────────────

    def target_weights(
        self, dataset: MarketDataset, idx: int, current: Weights
    ) -> Weights:
        if idx < self.warmup:
            return {}

        symbols = dataset.symbols
        zscore60 = dataset.features.get("zscore_60d", {})
        mom120 = dataset.features.get("mom_120d", {})
        ewma_sig = dataset.features.get("ewma_signal", {})
        rv = dataset.features.get("rv_20d", {})

        # ── 1. Carry signal per symbol ───────────────────────────────────────
        carry_scores: Dict[str, float] = {}
        for sym in symbols:
            z60 = _get(zscore60, sym, idx)
            m120 = _get(mom120, sym, idx)
            ewma = _get(ewma_sig, sym, idx)
            if z60 is None or m120 is None:
                continue

            # Trend filter: only trade in confirmed trend
            if abs(m120) < self.trend_threshold:
                continue

            # Carry proxy: mean-reversion within trend
            #   - In uptrend (m120 > 0): price below avg → backwardation → LONG (score > 0)
            #   - In downtrend (m120 < 0): price above avg → contango → SHORT (score < 0)
            trend_sign = 1.0 if m120 > 0 else -1.0
            carry = -z60 * trend_sign

            # EWMA confirmation: dampen against-trend signals
            if ewma is not None:
                ewma_sign = 1.0 if ewma > 0 else -1.0
                if ewma_sign != trend_sign:
                    carry *= 0.5  # reduce strength when EWMA disagrees

            carry_scores[sym] = carry

        if not carry_scores:
            return {}

        # ── 2. Cross-sectional z-score ───────────────────────────────────────
        vals = list(carry_scores.values())
        mn = sum(vals) / len(vals)
        std = math.sqrt(sum((v - mn) ** 2 for v in vals) / len(vals)) + _EPS
        z_scores = {sym: (carry_scores[sym] - mn) / std for sym in carry_scores}

        # ── 3. Equal risk contribution (inverse realized vol) ─────────────────
        raw_weights: Dict[str, float] = {}
        for sym, z in z_scores.items():
            rv_val = _get(rv, sym, idx)
            inv_vol = 1.0 / (rv_val + _EPS) if rv_val and rv_val > 0 else 1.0
            raw_weights[sym] = z * inv_vol

        return _normalise(raw_weights, self.max_gross_leverage, self.max_position)
