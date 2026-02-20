"""
Strategy #1: Multi-Timeframe Trend Following
=============================================
Classic CTA trend signal with NEXUS improvements.

Signal construction:
  - 3 timeframes: 20d (20%), 60d (30%), 120d (50%) log-momentum
  - Cross-sectional z-score normalisation
  - Position sizing: inverse-ATR (risk parity across commodities)
  - Max gross leverage: 2.0x
  - Vol target: 10% annualised (inherent via ATR sizing)
  - Rebalance: daily

Key insight: long-only is NOT the default in commodity CTA.
Both long and short positions are taken based on trend direction.
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional

from nexus_quant.strategies.base import Strategy, Weights
from nexus_quant.data.schema import MarketDataset

_EPS = 1e-10


class TrendFollowingStrategy(Strategy):
    """Multi-timeframe momentum with ATR-based risk parity sizing."""

    def __init__(self, name: str = "cta_trend", params: Optional[Dict[str, Any]] = None) -> None:
        p = params or {}
        super().__init__(name, p)

        # Timeframe weights (must sum to 1)
        self.w20: float = float(p.get("w20", 0.20))
        self.w60: float = float(p.get("w60", 0.30))
        self.w120: float = float(p.get("w120", 0.50))

        # Sizing
        self.max_gross_leverage: float = float(p.get("max_gross_leverage", 2.0))
        self.max_position: float = float(p.get("max_position", 0.25))

        # Warmup: need at least 120 bars of history
        self.warmup: int = int(p.get("warmup", 125))

        # Signal threshold — symbols with |signal| < this get zero weight
        self.signal_threshold: float = float(p.get("signal_threshold", 0.1))

    # ── Rebalance logic ──────────────────────────────────────────────────────

    def should_rebalance(self, dataset: MarketDataset, idx: int) -> bool:
        return idx >= self.warmup

    # ── Weight computation ───────────────────────────────────────────────────

    def target_weights(
        self, dataset: MarketDataset, idx: int, current: Weights
    ) -> Weights:
        if idx < self.warmup:
            return {}

        symbols = dataset.symbols
        mom20 = dataset.features.get("mom_20d", {})
        mom60 = dataset.features.get("mom_60d", {})
        mom120 = dataset.features.get("mom_120d", {})
        atr = dataset.features.get("atr_14", {})

        # ── 1. Composite momentum score per symbol ───────────────────────────
        raw_scores: Dict[str, float] = {}
        for sym in symbols:
            m20 = _get(mom20, sym, idx)
            m60 = _get(mom60, sym, idx)
            m120 = _get(mom120, sym, idx)
            if m20 is None or m60 is None or m120 is None:
                continue
            raw_scores[sym] = self.w20 * m20 + self.w60 * m60 + self.w120 * m120

        if not raw_scores:
            return {}

        # ── 2. Cross-sectional z-score ───────────────────────────────────────
        vals = list(raw_scores.values())
        mn = sum(vals) / len(vals)
        std = math.sqrt(sum((v - mn) ** 2 for v in vals) / len(vals)) + _EPS
        z_scores = {sym: (raw_scores[sym] - mn) / std for sym in raw_scores}

        # ── 3. ATR-based inverse-vol sizing ──────────────────────────────────
        raw_weights: Dict[str, float] = {}
        for sym, z in z_scores.items():
            if abs(z) < self.signal_threshold:
                continue
            atr_val = _get(atr, sym, idx)
            price = _safe_price(dataset, sym, idx)
            if atr_val is None or atr_val <= 0 or price is None or price <= 0:
                # Fallback: unit weight (sign only)
                raw_weights[sym] = math.copysign(1.0, z)
                continue
            # ATR as fraction of price = normalised risk
            atr_pct = atr_val / price
            # Inverse-vol weight: larger when commodity is less volatile
            inv_risk = 1.0 / (atr_pct + _EPS)
            raw_weights[sym] = math.copysign(inv_risk, z)

        if not raw_weights:
            return {}

        # ── 4. Normalise to target gross leverage ─────────────────────────────
        weights = _normalise(raw_weights, self.max_gross_leverage, self.max_position)
        return weights


# ── Helpers ───────────────────────────────────────────────────────────────────


def _get(
    feat: Dict[str, List[float]],
    sym: str,
    idx: int,
) -> Optional[float]:
    """Safely get feature[sym][idx]."""
    arr = feat.get(sym)
    if arr is None or idx >= len(arr):
        return None
    v = arr[idx]
    return v if (v == v and math.isfinite(v)) else None


def _safe_price(dataset: MarketDataset, sym: str, idx: int) -> Optional[float]:
    prices = dataset.perp_close.get(sym)
    if prices is None or idx >= len(prices):
        return None
    v = prices[idx]
    return v if (v == v and v > 0) else None


def _normalise(
    weights: Dict[str, float],
    max_gross: float,
    max_pos: float,
) -> Dict[str, float]:
    """Scale weights so sum(|w|) <= max_gross and max(|w|) <= max_pos."""
    gross = sum(abs(v) for v in weights.values())
    if gross < _EPS:
        return {}
    scale = min(max_gross / gross, 1.0)
    out = {sym: w * scale for sym, w in weights.items()}
    # Cap individual positions
    for sym, w in out.items():
        if abs(w) > max_pos:
            out[sym] = math.copysign(max_pos, w)
    return out
