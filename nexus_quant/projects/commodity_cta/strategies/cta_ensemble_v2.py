"""
Strategy: CTA Ensemble v2 — TSMOM + Donchian + MomValue (no carry)
===================================================================
Phase 141 v2: Conservative sizing for narrow commodity universe.
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional

from nexus_quant.strategies.base import Strategy, Weights
from nexus_quant.data.schema import MarketDataset
from .trend_following import _get, _normalise
from .tsmom import TSMOMStrategy, _apply_net_limit
from .donchian_breakout import DonchianBreakoutStrategy
from .momentum_value import MomentumValueStrategy

_EPS = 1e-10


class CTAEnsembleV2Strategy(Strategy):
    """
    CTA v2: TSMOM(50%) + Donchian(30%) + MomValue(20%) with vol tilt.
    No carry. Conservative sizing. Net leverage limited.
    """

    def __init__(
        self,
        name: str = "cta_ensemble_v2",
        params: Optional[Dict[str, Any]] = None,
    ) -> None:
        p = params or {}
        super().__init__(name, p)

        self.w_tsmom: float = float(p.get("w_tsmom", 0.50))
        self.w_donchian: float = float(p.get("w_donchian", 0.30))
        self.w_mom_value: float = float(p.get("w_mom_value", 0.20))

        self.max_gross_leverage: float = float(p.get("max_gross_leverage", 1.5))
        self.max_position: float = float(p.get("max_position", 0.20))
        self.max_net_leverage: float = float(p.get("max_net_leverage", 0.8))

        self.warmup: int = int(p.get("warmup", 280))
        self.rebalance_freq: int = int(p.get("rebalance_freq", 10))
        self._last_rebalance: int = -1

        # Vol tilt overlay
        self.vol_tilt_threshold: float = float(p.get("vol_tilt_threshold", 1.5))
        self.vol_tilt_r_min: float = float(p.get("vol_tilt_r_min", 0.65))
        self.vol_tilt_strength: float = float(p.get("vol_tilt_strength", 0.15))

        # Sub-strategies with conservative sizing
        self._tsmom = TSMOMStrategy(
            name="tsmom_sub",
            params={
                "max_gross_leverage": 1.0,
                "max_position": 0.5,
                "max_net_leverage": 10.0,  # no net limit at sub-strategy level
                "vol_target": 0.08,
                "signal_scale": 2.0,
                "warmup": 270,
                "rebalance_freq": 1,
                "w_21": float(p.get("tsmom_w_21", 0.10)),
                "w_63": float(p.get("tsmom_w_63", 0.20)),
                "w_126": float(p.get("tsmom_w_126", 0.35)),
                "w_252": float(p.get("tsmom_w_252", 0.35)),
            },
        )
        self._donchian = DonchianBreakoutStrategy(
            name="donchian_sub",
            params={
                "max_gross_leverage": 1.0,
                "max_position": 0.5,
                "max_net_leverage": 10.0,
                "vol_target": 0.08,
                "warmup": 130,
                "rebalance_freq": 1,
            },
        )
        self._mom_val = MomentumValueStrategy(
            name="mv_sub",
            params={
                "max_gross_leverage": 1.0,
                "max_position": 0.5,
                "warmup": 260,
                "rebalance_freq": 1,
            },
        )

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

        tsmom_w = self._tsmom.target_weights(dataset, idx, current) if idx >= 270 else {}
        donchian_w = self._donchian.target_weights(dataset, idx, current) if idx >= 130 else {}
        mv_w = self._mom_val.target_weights(dataset, idx, current) if idx >= 260 else {}

        blended: Dict[str, float] = {}
        for sym in symbols:
            t = tsmom_w.get(sym, 0.0)
            d = donchian_w.get(sym, 0.0)
            mv = mv_w.get(sym, 0.0)
            blended[sym] = self.w_tsmom * t + self.w_donchian * d + self.w_mom_value * mv

        blended = {sym: w for sym, w in blended.items() if abs(w) > _EPS}
        if not blended:
            return {}

        vol_factor = self._compute_vol_tilt(dataset, idx, symbols)
        target_gross = self.max_gross_leverage * vol_factor

        weights = _normalise(blended, target_gross, self.max_position)
        weights = _apply_net_limit(weights, self.max_net_leverage)
        return weights

    def _compute_vol_tilt(
        self, dataset: MarketDataset, idx: int, symbols: List[str],
    ) -> float:
        vol_mom_z = dataset.features.get("vol_mom_z", {})
        if not vol_mom_z:
            return 1.0
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
        excess = avg_z - self.vol_tilt_threshold
        factor = 1.0 - self.vol_tilt_strength * excess
        return max(self.vol_tilt_r_min, min(1.0, factor))


__all__ = ["CTAEnsembleV2Strategy"]
