"""
P91b Champion Ensemble — Static 4-signal blend.

Combines four independent alpha sources with fixed weights
(discovered through Phases 59-92, validated in-sample and out-of-sample):

  1. NexusAlpha V1       (27.47%)  — carry + momentum + MR
  2. IdioMomentum i460   (19.67%)  — beta-hedged momentum (lb=460, bw=168, k=4)
  3. IdioMomentum i415   (32.47%)  — beta-hedged momentum (lb=415, bw=216, k=4)
  4. FundingMomentum f144 (20.39%) — contrarian cumulative funding (lb=144, k=2)

Blending:
  final_weight[sym] = sum(w_i * sub_weight_i[sym]) for all 4 signals
  then rescale to target_gross_leverage.

Each sub-strategy runs independently with its own rebalance schedule.
The ensemble rebalances when ANY sub-strategy wants to rebalance.
"""
from __future__ import annotations

from typing import Any, Dict, List, Tuple

from .base import Strategy, Weights
from .nexus_alpha import NexusAlphaV1Strategy
from .idio_momentum_alpha import IdioMomentumAlphaStrategy
from .funding_momentum_alpha import FundingMomentumAlphaStrategy
from ..data.schema import MarketDataset


class P91bEnsembleStrategy(Strategy):
    """
    P91b Champion: static 4-signal weight-blending ensemble.
    """

    def __init__(self, params: Dict[str, Any]) -> None:
        super().__init__(name="p91b_ensemble", params=params)

        # Ensemble weights (static, from Phase 92 optimization)
        self._blend: Dict[str, float] = {
            "v1": float(params.get("w_v1", 0.2747)),
            "i460": float(params.get("w_i460", 0.1967)),
            "i415": float(params.get("w_i415", 0.3247)),
            "f144": float(params.get("w_f144", 0.2039)),
        }

        # Sub-strategy instances
        v1_p = dict(params.get("v1_params") or {})
        i460_p = dict(params.get("i460_params") or {})
        i415_p = dict(params.get("i415_params") or {})
        f144_p = dict(params.get("f144_params") or {})

        self._subs: List[Tuple[str, Strategy]] = [
            ("v1", NexusAlphaV1Strategy(params=v1_p)),
            ("i460", IdioMomentumAlphaStrategy(params=i460_p)),
            ("i415", IdioMomentumAlphaStrategy(params=i415_p)),
            ("f144", FundingMomentumAlphaStrategy(params=f144_p)),
        ]

        # Cache last weights per sub-strategy
        self._cached: Dict[str, Weights] = {k: {} for k, _ in self._subs}

    @property
    def target_gross_leverage(self) -> float:
        return float(self.params.get("target_gross_leverage", 0.35))

    def should_rebalance(self, dataset: MarketDataset, idx: int) -> bool:
        return any(s.should_rebalance(dataset, idx) for _, s in self._subs)

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

        # Blend: sum(blend_weight * sub_weights)
        blended: Dict[str, float] = {s: 0.0 for s in syms}
        for key, _ in self._subs:
            bw = self._blend.get(key, 0.0)
            sub_w = self._cached.get(key, {})
            for s in syms:
                blended[s] += bw * sub_w.get(s, 0.0)

        # Rescale to target gross leverage
        gross = sum(abs(w) for w in blended.values())
        if gross <= 1e-10:
            return {s: 0.0 for s in syms}

        scale = self.target_gross_leverage / gross
        return {s: blended[s] * scale for s in syms}

    def describe(self) -> Dict[str, Any]:
        base = super().describe()
        base["blend_weights"] = dict(self._blend)
        base["sub_strategies"] = {k: s.describe() for k, s in self._subs}
        return base
