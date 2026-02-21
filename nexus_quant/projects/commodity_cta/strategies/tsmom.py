"""
Strategy: Time-Series Momentum (TSMOM) — Moskowitz, Ooi, Pedersen (2012)
=========================================================================
Phase 141 v2: Fixed for narrow commodity universe (8-13 instruments).

Key fixes vs v1:
  - CONTINUOUS signal (clip(tsmom/2, -1, +1)) instead of binary sign
    → scales position with conviction, reduces whipsaw in choppy markets
  - Lower vol target (0.08) for concentrated commodity portfolio
    → academic TSMOM uses 40% across 58 instruments ≈ 0.7% per instrument
    → with 8 instruments, 0.08 per instrument → ~0.64 gross (reasonable)
  - Max net leverage constraint to prevent directional concentration
  - Signal decay: require 2+ horizons to confirm (avoid single-horizon noise)
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional

from nexus_quant.strategies.base import Strategy, Weights
from nexus_quant.data.schema import MarketDataset
from .trend_following import _get, _normalise

_EPS = 1e-10


class TSMOMStrategy(Strategy):
    """
    Multi-scale time-series momentum with continuous signal and vol-targeting.
    """

    def __init__(
        self,
        name: str = "cta_tsmom",
        params: Optional[Dict[str, Any]] = None,
    ) -> None:
        p = params or {}
        super().__init__(name, p)

        # Lookback horizon weights (commodity-optimal: heavier on 6m/12m)
        self.w_21: float = float(p.get("w_21", 0.10))
        self.w_63: float = float(p.get("w_63", 0.20))
        self.w_126: float = float(p.get("w_126", 0.35))
        self.w_252: float = float(p.get("w_252", 0.35))

        # Vol target per instrument (conservative for 8-instrument universe)
        self.vol_target: float = float(p.get("vol_target", 0.08))

        # Portfolio constraints
        self.max_gross_leverage: float = float(p.get("max_gross_leverage", 1.5))
        self.max_position: float = float(p.get("max_position", 0.20))
        self.max_net_leverage: float = float(p.get("max_net_leverage", 0.8))

        # Signal scaling: clip(tsmom / signal_scale, -1, +1)
        self.signal_scale: float = float(p.get("signal_scale", 2.0))

        # Warmup
        self.warmup: int = int(p.get("warmup", 270))

        # Monthly rebalancing
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

        symbols = dataset.symbols

        tsmom_21 = dataset.features.get("tsmom_21d", {})
        tsmom_63 = dataset.features.get("tsmom_63d", {})
        tsmom_126 = dataset.features.get("tsmom_126d", {})
        tsmom_252 = dataset.features.get("tsmom_252d", {})
        rv = dataset.features.get("rv_20d", {})

        raw_weights: Dict[str, float] = {}

        for sym in symbols:
            s21 = _get(tsmom_21, sym, idx)
            s63 = _get(tsmom_63, sym, idx)
            s126 = _get(tsmom_126, sym, idx)
            s252 = _get(tsmom_252, sym, idx)

            # Continuous signal: clip(tsmom / scale, -1, +1)
            # Weights down uncertain signals, amplifies strong ones
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

            # Weighted blend of continuous signals
            total_w = sum(horizon_weights)
            if total_w < _EPS:
                continue
            combo = sum(s * w for s, w in zip(raw_signals, horizon_weights)) / total_w

            # Require some minimum conviction
            if abs(combo) < 0.1:
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

        # Apply net leverage constraint
        weights = _normalise(raw_weights, self.max_gross_leverage, self.max_position)
        weights = _apply_net_limit(weights, self.max_net_leverage)
        return weights


def _clip(x: float, lo: float = -1.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))


def _clip_abs(x: float, max_abs: float) -> float:
    return math.copysign(min(abs(x), max_abs), x) if abs(x) > _EPS else 0.0


def _apply_net_limit(weights: Dict[str, float], max_net: float) -> Dict[str, float]:
    """Scale weights to enforce max net leverage (prevents directional concentration)."""
    net = sum(weights.values())
    if abs(net) <= max_net:
        return weights
    # Scale all weights so net = max_net * sign(net)
    gross = sum(abs(v) for v in weights.values())
    if gross < _EPS:
        return weights
    # Reduce the dominant side proportionally
    target_net = math.copysign(max_net, net)
    excess = net - target_net
    # Distribute excess reduction among positions in the dominant direction
    dom_sign = 1.0 if net > 0 else -1.0
    dom_total = sum(v for v in weights.values() if math.copysign(1.0, v) == dom_sign)
    if abs(dom_total) < _EPS:
        return weights
    result = {}
    for sym, w in weights.items():
        if math.copysign(1.0, w) == dom_sign:
            result[sym] = w - excess * (w / dom_total)
        else:
            result[sym] = w
    return result


__all__ = ["TSMOMStrategy"]
