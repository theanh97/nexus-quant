"""
Breadth-Adaptive Ensemble — P145 Champion

Cross-sectional momentum breadth as real-time regime classifier.
Switches blend weights between prod (defensive) and p143b (momentum-heavy)
based on what % of symbols have positive N-bar return.

From Phase 145 (2026-02-21):
  OBJ=1.8851 vs baseline=1.5672 (+0.3179)
  LOYO: 4/5 wins, avg_delta=+0.4283
  Best thresholds: p_low=0.33, p_high=0.67
  Window: robust (84h, 168h, 336h all give same result)

Weight sets:
  PROD weights (defensive, V1-heavy):
    v1=0.2747, i460bw168=0.1967, i415bw216=0.3247, f144=0.2039
  P143B weights (momentum-heavy, I415-heavy):
    v1=0.05, i460bw168=0.25, i415bw216=0.45, f144=0.25

Regime logic:
  breadth = (# symbols with positive breadth_window_bars return) / n_symbols
  if breadth >= p_high: use P143B (momentum regime)
  if breadth <= p_low:  use PROD  (defensive regime)
  otherwise: interpolate linearly between PROD and P143B
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

from .base import Strategy, Weights
from .nexus_alpha import NexusAlphaV1Strategy
from .idio_momentum_alpha import IdioMomentumAlphaStrategy
from .funding_momentum_alpha import FundingMomentumAlphaStrategy
from ..data.schema import MarketDataset

_EPS = 1e-10

# Default weight sets (validated in Phase 145)
_PROD_WEIGHTS = {
    "v1": 0.2747,
    "i460": 0.1967,
    "i415": 0.3247,
    "f144": 0.2039,
}

_P143B_WEIGHTS = {
    "v1": 0.05,
    "i460": 0.25,
    "i415": 0.45,
    "f144": 0.25,
}


def _blend_weights(w_prod: Dict[str, float], w_p143b: Dict[str, float], alpha: float) -> Dict[str, float]:
    """Interpolate: alpha=0 → prod, alpha=1 → p143b."""
    return {k: (1.0 - alpha) * w_prod[k] + alpha * w_p143b[k] for k in w_prod}


class BreadthAdaptiveEnsembleStrategy(Strategy):
    """
    Phase 145 champion: breadth-based regime classifier + adaptive weight switching.

    Extends P91b ensemble with real-time breadth signal for weight interpolation.
    """

    def __init__(self, params: Dict[str, Any]) -> None:
        super().__init__(name="breadth_adaptive_ensemble", params=params)

        # Breadth classifier parameters
        self._breadth_window: int = int(params.get("breadth_window_bars", 84))
        self._p_low: float = float(params.get("p_low", 0.33))
        self._p_high: float = float(params.get("p_high", 0.67))

        # Weight sets (can be overridden via params)
        prod_w = params.get("prod_weights", _PROD_WEIGHTS)
        p143b_w = params.get("p143b_weights", _P143B_WEIGHTS)
        self._prod_w: Dict[str, float] = dict(prod_w)
        self._p143b_w: Dict[str, float] = dict(p143b_w)

        # Target gross leverage
        self._target_leverage: float = float(params.get("target_gross_leverage", 0.35))

        # Sub-strategy instances (shared across weight sets)
        v1_p = dict(params.get("v1_params") or {})
        i460_p = dict(params.get("i460_params") or {})
        i415_p = dict(params.get("i415_params") or {})
        f144_p = dict(params.get("f144_params") or {})

        self._subs: List[Tuple[str, Strategy]] = [
            ("v1",  NexusAlphaV1Strategy(params=v1_p)),
            ("i460", IdioMomentumAlphaStrategy(params=i460_p)),
            ("i415", IdioMomentumAlphaStrategy(params=i415_p)),
            ("f144", FundingMomentumAlphaStrategy(params=f144_p)),
        ]
        self._cached: Dict[str, Weights] = {k: {} for k, _ in self._subs}

        # Regime tracking
        self._last_breadth: float = 0.5
        self._last_alpha: float = 0.5

    # ── Regime computation ─────────────────────────────────────────────────

    def _compute_breadth(self, dataset: MarketDataset, idx: int) -> float:
        """
        Compute cross-sectional momentum breadth.
        breadth = fraction of symbols with positive N-bar return.
        Range: [0, 1]. Higher = more bullish / momentum regime.
        """
        win = self._breadth_window
        if idx < win + 1:
            return 0.5  # neutral during warmup

        n_pos = 0
        n_total = 0
        for sym in dataset.symbols:
            closes = dataset.perp_close[sym]
            c_now = float(closes[idx - 1]) if idx - 1 < len(closes) else None
            c_back = float(closes[idx - 1 - win]) if (idx - 1 - win) >= 0 and (idx - 1 - win) < len(closes) else None
            if c_now is not None and c_back is not None and c_back > _EPS:
                n_total += 1
                if c_now > c_back:
                    n_pos += 1

        return n_pos / n_total if n_total > 0 else 0.5

    def _compute_alpha(self, breadth: float) -> float:
        """
        Map breadth → alpha for weight interpolation.
        alpha=0: use PROD weights, alpha=1: use P143B weights.
        """
        if breadth >= self._p_high:
            return 1.0  # full momentum regime → p143b
        if breadth <= self._p_low:
            return 0.0  # defensive regime → prod
        # Linear interpolation between p_low and p_high
        return (breadth - self._p_low) / (self._p_high - self._p_low)

    # ── Rebalance cadence ──────────────────────────────────────────────────

    def should_rebalance(self, dataset: MarketDataset, idx: int) -> bool:
        # Update breadth state every bar (low cost, continuous)
        self._last_breadth = self._compute_breadth(dataset, idx)
        self._last_alpha = self._compute_alpha(self._last_breadth)
        # Rebalance when any sub-strategy wants to
        return any(s.should_rebalance(dataset, idx) for _, s in self._subs)

    # ── Weight computation ─────────────────────────────────────────────────

    def target_weights(
        self,
        dataset: MarketDataset,
        idx: int,
        current: Weights,
    ) -> Weights:
        syms = dataset.symbols

        # Update cached weights for each sub-strategy that wants to rebalance
        for key, strat in self._subs:
            if strat.should_rebalance(dataset, idx):
                self._cached[key] = strat.target_weights(dataset, idx, current)

        # Compute adaptive blend weights based on current regime
        blend = _blend_weights(self._prod_w, self._p143b_w, self._last_alpha)

        # Blend sub-strategy positions
        blended: Dict[str, float] = {s: 0.0 for s in syms}
        for key, _ in self._subs:
            bw = blend.get(key, 0.0)
            sub_w = self._cached.get(key, {})
            for s in syms:
                blended[s] += bw * sub_w.get(s, 0.0)

        # Rescale to target gross leverage
        gross = sum(abs(w) for w in blended.values())
        if gross <= _EPS:
            return {s: 0.0 for s in syms}

        scale = self._target_leverage / gross
        return {s: blended[s] * scale for s in syms}

    def describe(self) -> Dict[str, Any]:
        base = super().describe()
        base.update({
            "breadth_window_bars": self._breadth_window,
            "p_low": self._p_low,
            "p_high": self._p_high,
            "last_breadth": self._last_breadth,
            "last_alpha": self._last_alpha,
            "regime": "p143b" if self._last_alpha >= 1.0 else ("prod" if self._last_alpha <= 0.0 else "mixed"),
        })
        return base


__all__ = ["BreadthAdaptiveEnsembleStrategy"]
