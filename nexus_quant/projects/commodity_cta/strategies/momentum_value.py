"""
Strategy #3: Momentum + Value Combo
=====================================
Cross-sectional combination of trend momentum and long-run value signals.

Signal construction:
  combined = 0.6 * momentum_rank + 0.4 * value_rank

  - Momentum: 120-day log return, cross-sectional rank (0→1)
  - Value   : distance from 252-day mean (z-score), inverted
              (cheap = positive value score → LONG)

Portfolio construction:
  - Long top 4 commodities (highest combined score)
  - Short bottom 4 commodities (lowest combined score)
  - Equal risk contribution sizing

Rebalance: monthly (~21 bars)
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

from nexus_quant.strategies.base import Strategy, Weights
from nexus_quant.data.schema import MarketDataset
from .trend_following import _get, _normalise

_EPS = 1e-10


class MomentumValueStrategy(Strategy):
    """Momentum + long-run value combo. Monthly rebalance."""

    def __init__(
        self,
        name: str = "cta_mom_value",
        params: Optional[Dict[str, Any]] = None,
    ) -> None:
        p = params or {}
        super().__init__(name, p)

        self.mom_weight: float = float(p.get("mom_weight", 0.6))
        self.val_weight: float = float(p.get("val_weight", 0.4))
        self.n_long: int = int(p.get("n_long", 4))
        self.n_short: int = int(p.get("n_short", 4))
        self.max_gross_leverage: float = float(p.get("max_gross_leverage", 1.5))
        self.max_position: float = float(p.get("max_position", 0.25))
        self.warmup: int = int(p.get("warmup", 260))  # need 252-day z-score
        self.rebalance_freq: int = int(p.get("rebalance_freq", 21))  # monthly
        self._last_rebalance: int = -1

    # ── Rebalance cadence ────────────────────────────────────────────────────

    def should_rebalance(self, dataset: MarketDataset, idx: int) -> bool:
        if idx < self.warmup:
            return False
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
        mom120 = dataset.features.get("mom_120d", {})
        zscore252 = dataset.features.get("zscore_252d", {})
        rv = dataset.features.get("rv_20d", {})

        # ── 1. Collect raw signals ───────────────────────────────────────────
        mom_raw: Dict[str, float] = {}
        val_raw: Dict[str, float] = {}

        for sym in symbols:
            m = _get(mom120, sym, idx)
            z = _get(zscore252, sym, idx)
            if m is not None:
                mom_raw[sym] = m
            if z is not None:
                # Value = cheap when below long-run mean (negative z-score)
                val_raw[sym] = -z

        common_syms = [s for s in symbols if s in mom_raw and s in val_raw]
        if len(common_syms) < self.n_long + self.n_short:
            return {}

        # ── 2. Cross-sectional rank (0 = bottom, 1 = top) ───────────────────
        mom_rank = _cross_rank(mom_raw, common_syms)
        val_rank = _cross_rank(val_raw, common_syms)

        # ── 3. Combined score ────────────────────────────────────────────────
        combined = {
            sym: self.mom_weight * mom_rank[sym] + self.val_weight * val_rank[sym]
            for sym in common_syms
        }

        # ── 4. Select top N long / bottom N short ────────────────────────────
        ranked = sorted(common_syms, key=lambda s: combined[s], reverse=True)
        longs = ranked[: self.n_long]
        shorts = ranked[-self.n_short :]

        # ── 5. Equal risk contribution sizing ────────────────────────────────
        raw_weights: Dict[str, float] = {}
        for sym in longs:
            rv_val = _get(rv, sym, idx)
            inv_vol = 1.0 / (rv_val + _EPS) if rv_val and rv_val > 0 else 1.0
            raw_weights[sym] = +inv_vol

        for sym in shorts:
            rv_val = _get(rv, sym, idx)
            inv_vol = 1.0 / (rv_val + _EPS) if rv_val and rv_val > 0 else 1.0
            raw_weights[sym] = -inv_vol

        return _normalise(raw_weights, self.max_gross_leverage, self.max_position)


# ── Helpers ───────────────────────────────────────────────────────────────────


def _cross_rank(scores: Dict[str, float], syms: List[str]) -> Dict[str, float]:
    """Rank symbols by score and normalise to [0, 1]."""
    n = len(syms)
    if n == 0:
        return {}
    ordered = sorted(syms, key=lambda s: scores.get(s, 0.0))
    return {sym: i / (n - 1) if n > 1 else 0.5 for i, sym in enumerate(ordered)}
