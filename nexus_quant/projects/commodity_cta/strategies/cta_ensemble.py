"""
Strategy #4: CTA Ensemble
==========================
Static-weight blend of all 3 CTA component strategies.

Component weights (lessons from crypto: static > adaptive):
  Trend (40%) + Carry (30%) + MomValue (30%)

Vol tilt overlay (knowledge transfer from crypto research):
  - Compute vol_mom_z (volume momentum z-score)
  - When vol_mom_z > threshold → market is crowded → reduce gross leverage
  - Leverage multiplier: r = max(r_min, 1 - tilt_strength * vol_mom_z_excess)
  - Default: r=0.65 when crowded (same as crypto champion)

Rebalance: daily (sub-components computed on every bar)
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional

from nexus_quant.strategies.base import Strategy, Weights
from nexus_quant.data.schema import MarketDataset
from .trend_following import _get, _normalise, TrendFollowingStrategy
from .carry_roll import CarryRollStrategy
from .momentum_value import MomentumValueStrategy

_EPS = 1e-10


class CTAEnsembleStrategy(Strategy):
    """
    Static-weight ensemble with volume momentum tilt overlay.
    Computes all component signals daily and blends them.
    """

    def __init__(
        self,
        name: str = "cta_ensemble",
        params: Optional[Dict[str, Any]] = None,
    ) -> None:
        p = params or {}
        super().__init__(name, p)

        # Component weights
        self.w_trend: float = float(p.get("w_trend", 0.40))
        self.w_carry: float = float(p.get("w_carry", 0.30))
        self.w_mom_value: float = float(p.get("w_mom_value", 0.30))

        # Portfolio sizing
        self.max_gross_leverage: float = float(p.get("max_gross_leverage", 2.0))
        self.max_position: float = float(p.get("max_position", 0.25))

        # Warmup (longest sub-strategy warmup + buffer)
        self.warmup: int = int(p.get("warmup", 265))

        # Vol tilt overlay (transfer from crypto)
        self.vol_tilt_threshold: float = float(p.get("vol_tilt_threshold", 1.5))
        self.vol_tilt_r_min: float = float(p.get("vol_tilt_r_min", 0.65))
        self.vol_tilt_strength: float = float(p.get("vol_tilt_strength", 0.15))

        # Instantiate sub-strategies with matching params
        self._trend = TrendFollowingStrategy(
            name="cta_trend_sub",
            params={
                "max_gross_leverage": 1.0,  # normalised — ensemble controls total lev
                "max_position": 0.5,
                "warmup": 125,
            },
        )
        self._carry = CarryRollStrategy(
            name="cta_carry_sub",
            params={
                "max_gross_leverage": 1.0,
                "max_position": 0.5,
                "warmup": 130,
                "rebalance_freq": 1,  # compute every bar; ensemble controls cadence
            },
        )
        self._mom_val = MomentumValueStrategy(
            name="cta_mv_sub",
            params={
                "max_gross_leverage": 1.0,
                "max_position": 0.5,
                "warmup": 260,
                "rebalance_freq": 1,  # compute every bar
            },
        )

    # ── Rebalance cadence ────────────────────────────────────────────────────

    def should_rebalance(self, dataset: MarketDataset, idx: int) -> bool:
        return idx >= self.warmup

    # ── Weight computation ───────────────────────────────────────────────────

    def target_weights(
        self, dataset: MarketDataset, idx: int, current: Weights
    ) -> Weights:
        if idx < self.warmup:
            return {}

        symbols = dataset.symbols

        # ── 1. Get component weights ─────────────────────────────────────────
        # Each sub-strategy returns weights with max_gross=1 (signal-only)
        trend_w = self._trend.target_weights(dataset, idx, current) if idx >= 125 else {}
        carry_w = self._carry.target_weights(dataset, idx, current) if idx >= 130 else {}
        mv_w = self._mom_val.target_weights(dataset, idx, current) if idx >= 260 else {}

        # ── 2. Blend components ──────────────────────────────────────────────
        blended: Dict[str, float] = {}
        for sym in symbols:
            t = trend_w.get(sym, 0.0)
            c = carry_w.get(sym, 0.0)
            mv = mv_w.get(sym, 0.0)
            blended[sym] = self.w_trend * t + self.w_carry * c + self.w_mom_value * mv

        # Remove near-zero weights
        blended = {sym: w for sym, w in blended.items() if abs(w) > _EPS}
        if not blended:
            return {}

        # ── 3. Vol tilt overlay ──────────────────────────────────────────────
        vol_factor = self._compute_vol_tilt(dataset, idx, symbols)

        # ── 4. Normalise to target gross leverage (with vol factor) ───────────
        target_gross = self.max_gross_leverage * vol_factor
        weights = _normalise(blended, target_gross, self.max_position)

        return weights

    # ── Vol tilt helper ───────────────────────────────────────────────────────

    def _compute_vol_tilt(
        self,
        dataset: MarketDataset,
        idx: int,
        symbols: List[str],
    ) -> float:
        """
        Compute leverage multiplier based on cross-market volume momentum z-score.

        When volume is accelerating across commodities (crowded trend following),
        reduce gross leverage to protect against crowded unwind.

        Returns: float in [vol_tilt_r_min, 1.0]
        """
        vol_mom_z = dataset.features.get("vol_mom_z", {})
        if not vol_mom_z:
            return 1.0

        # Average vol_mom_z across all symbols
        vals = []
        for sym in symbols:
            v = _get(vol_mom_z, sym, idx)
            if v is not None:
                vals.append(v)

        if not vals:
            return 1.0

        avg_z = sum(vals) / len(vals)

        if avg_z <= self.vol_tilt_threshold:
            return 1.0

        # Reduce leverage proportionally above threshold
        excess = avg_z - self.vol_tilt_threshold
        factor = 1.0 - self.vol_tilt_strength * excess
        return max(self.vol_tilt_r_min, min(1.0, factor))


# ── Strategy registration ─────────────────────────────────────────────────────

__all__ = ["CTAEnsembleStrategy"]
