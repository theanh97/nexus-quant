"""
NEXUS Ensemble V1 -- Simple Weight-Blending of NexusAlphaV1 + OrderflowAlphaV1.

Rationale
---------
NexusAlphaV1 (carry/momentum/MR) is strong in bull markets (2021 Sharpe=2.047)
but weak in bear markets (2022 Sharpe=0.112).  OrderflowAlphaV1 (taker buy/sell
imbalance) is regime-invariant (2022 Sharpe=1.619).  Blending the two should
provide robust performance across all regimes.

Blending Method
---------------
  final_weight[sym] = alpha * v1_weight[sym] + (1 - alpha) * orderflow_weight[sym]

Default alpha = 0.5 (equal blend).  Each sub-strategy runs independently with
its own parameters and rebalance schedule.  The ensemble rebalances whenever
EITHER sub-strategy wants to rebalance.

After blending, the combined weights are rescaled to target_gross_leverage
to maintain consistent risk exposure.

Parameters
----------
alpha                       : float = 0.5    blend coefficient (1.0 = pure V1, 0.0 = pure OF)
target_gross_leverage       : float = 0.35   rescale blended portfolio to this gross

v1_params                   : dict           parameters passed to NexusAlphaV1Strategy
orderflow_params            : dict           parameters passed to OrderflowAlphaV1Strategy
"""
from __future__ import annotations

from typing import Any, Dict

from .base import Strategy, Weights
from .nexus_alpha import NexusAlphaV1Strategy
from .orderflow_alpha import OrderflowAlphaV1Strategy
from ..data.schema import MarketDataset


class NexusEnsembleV1Strategy(Strategy):
    """
    Simple weight-blending ensemble of NexusAlphaV1 + OrderflowAlphaV1.

    Runs both sub-strategies independently to compute target weights, then
    blends them: final = alpha * v1 + (1 - alpha) * orderflow.  Rescales
    to target gross leverage.
    """

    def __init__(self, params: Dict[str, Any]) -> None:
        super().__init__(name="nexus_ensemble_v1", params=params)

        # Extract sub-strategy params (separate dicts so they don't clash)
        v1_params = dict(params.get("v1_params") or {})
        of_params = dict(params.get("orderflow_params") or {})

        # Create independent sub-strategy instances
        self._v1 = NexusAlphaV1Strategy(params=v1_params)
        self._of = OrderflowAlphaV1Strategy(params=of_params)

        # Cache last weights from each sub-strategy (carry forward between rebalances)
        self._v1_weights: Weights = {}
        self._of_weights: Weights = {}

    # ------------------------------------------------------------------
    # Parameters
    # ------------------------------------------------------------------

    @property
    def alpha(self) -> float:
        """Blend coefficient: 1.0 = pure V1, 0.0 = pure orderflow."""
        return float(self.params.get("alpha", 0.5))

    @property
    def target_gross_leverage(self) -> float:
        return float(self.params.get("target_gross_leverage", 0.35))

    # ------------------------------------------------------------------
    # Rebalance: fire when EITHER sub-strategy wants to rebalance
    # ------------------------------------------------------------------

    def should_rebalance(self, dataset: MarketDataset, idx: int) -> bool:
        v1_wants = self._v1.should_rebalance(dataset, idx)
        of_wants = self._of.should_rebalance(dataset, idx)
        return v1_wants or of_wants

    # ------------------------------------------------------------------
    # Target weights: blend then rescale
    # ------------------------------------------------------------------

    def target_weights(
        self,
        dataset: MarketDataset,
        idx: int,
        current: Weights,
    ) -> Weights:
        syms = dataset.symbols
        a = self.alpha

        # Get V1 weights (update only when V1 wants to rebalance)
        if self._v1.should_rebalance(dataset, idx):
            self._v1_weights = self._v1.target_weights(dataset, idx, current)
        # Get orderflow weights (update only when OF wants to rebalance)
        if self._of.should_rebalance(dataset, idx):
            self._of_weights = self._of.target_weights(dataset, idx, current)

        # Blend: final = alpha * v1 + (1 - alpha) * orderflow
        blended: Dict[str, float] = {}
        for s in syms:
            v1_w = self._v1_weights.get(s, 0.0)
            of_w = self._of_weights.get(s, 0.0)
            blended[s] = a * v1_w + (1.0 - a) * of_w

        # Rescale to target gross leverage
        gross = sum(abs(w) for w in blended.values())
        if gross <= 1e-10:
            return {s: 0.0 for s in syms}

        target = self.target_gross_leverage
        scale = target / gross
        return {s: blended[s] * scale for s in syms}

    # ------------------------------------------------------------------
    # Describe (for logging / serialization)
    # ------------------------------------------------------------------

    def describe(self) -> Dict[str, Any]:
        base = super().describe()
        base["sub_strategies"] = {
            "v1": self._v1.describe(),
            "orderflow": self._of.describe(),
        }
        base["alpha"] = self.alpha
        return base
