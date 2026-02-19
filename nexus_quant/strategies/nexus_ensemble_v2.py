"""
NEXUS Ensemble V2 -- 3-Strategy Weight-Blending Ensemble.

Blends three independent alpha sources:
  1. NexusAlphaV1    (carry + momentum + MR)     -- strong in bull markets
  2. OrderflowAlphaV1 (taker buy/sell imbalance)  -- regime-invariant
  3. PositioningAlphaV1 (contrarian sentiment)    -- complementary sentiment signal

Blending Method
---------------
  final_weight[sym] = alpha1 * v1_weight[sym]
                    + alpha2 * orderflow_weight[sym]
                    + alpha3 * positioning_weight[sym]

Default: equal weight alpha1 = alpha2 = alpha3 = 1/3.

Each sub-strategy runs independently with its own parameters and rebalance
schedule.  The ensemble rebalances whenever ANY sub-strategy wants to
rebalance.

After blending, the combined weights are rescaled to target_gross_leverage
to maintain consistent risk exposure.

Parameters
----------
alpha1                      : float = 0.333  NexusAlphaV1 blend weight
alpha2                      : float = 0.333  OrderflowAlphaV1 blend weight
alpha3                      : float = 0.334  PositioningAlphaV1 blend weight
target_gross_leverage       : float = 0.35   rescale blended portfolio to this gross

v1_params                   : dict           parameters passed to NexusAlphaV1Strategy
orderflow_params            : dict           parameters passed to OrderflowAlphaV1Strategy
positioning_params          : dict           parameters passed to PositioningAlphaV1Strategy
"""
from __future__ import annotations

from typing import Any, Dict

from .base import Strategy, Weights
from .nexus_alpha import NexusAlphaV1Strategy
from .orderflow_alpha import OrderflowAlphaV1Strategy
from .positioning_alpha import PositioningAlphaV1Strategy
from ..data.schema import MarketDataset


class NexusEnsembleV2Strategy(Strategy):
    """
    3-strategy weight-blending ensemble of NexusAlphaV1 + OrderflowAlphaV1
    + PositioningAlphaV1.

    Runs all three sub-strategies independently to compute target weights,
    then blends them:
      final = alpha1 * v1 + alpha2 * orderflow + alpha3 * positioning

    Rescales to target gross leverage.
    """

    def __init__(self, params: Dict[str, Any]) -> None:
        super().__init__(name="nexus_ensemble_v2", params=params)

        # Extract sub-strategy params (separate dicts so they don't clash)
        v1_params = dict(params.get("v1_params") or {})
        of_params = dict(params.get("orderflow_params") or {})
        pos_params = dict(params.get("positioning_params") or {})

        # Create independent sub-strategy instances
        self._v1 = NexusAlphaV1Strategy(params=v1_params)
        self._of = OrderflowAlphaV1Strategy(params=of_params)
        self._pos = PositioningAlphaV1Strategy(params=pos_params)

        # Cache last weights from each sub-strategy (carry forward between rebalances)
        self._v1_weights: Weights = {}
        self._of_weights: Weights = {}
        self._pos_weights: Weights = {}

    # ------------------------------------------------------------------
    # Parameters
    # ------------------------------------------------------------------

    @property
    def alpha1(self) -> float:
        """Blend weight for NexusAlphaV1."""
        return float(self.params.get("alpha1", 1.0 / 3.0))

    @property
    def alpha2(self) -> float:
        """Blend weight for OrderflowAlphaV1."""
        return float(self.params.get("alpha2", 1.0 / 3.0))

    @property
    def alpha3(self) -> float:
        """Blend weight for PositioningAlphaV1."""
        return float(self.params.get("alpha3", 1.0 - self.alpha1 - self.alpha2))

    @property
    def target_gross_leverage(self) -> float:
        return float(self.params.get("target_gross_leverage", 0.35))

    # ------------------------------------------------------------------
    # Rebalance: fire when ANY sub-strategy wants to rebalance
    # ------------------------------------------------------------------

    def should_rebalance(self, dataset: MarketDataset, idx: int) -> bool:
        v1_wants = self._v1.should_rebalance(dataset, idx)
        of_wants = self._of.should_rebalance(dataset, idx)
        pos_wants = self._pos.should_rebalance(dataset, idx)
        return v1_wants or of_wants or pos_wants

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
        a1 = self.alpha1
        a2 = self.alpha2
        a3 = self.alpha3

        # Get V1 weights (update only when V1 wants to rebalance)
        if self._v1.should_rebalance(dataset, idx):
            self._v1_weights = self._v1.target_weights(dataset, idx, current)
        # Get orderflow weights (update only when OF wants to rebalance)
        if self._of.should_rebalance(dataset, idx):
            self._of_weights = self._of.target_weights(dataset, idx, current)
        # Get positioning weights (update only when POS wants to rebalance)
        if self._pos.should_rebalance(dataset, idx):
            self._pos_weights = self._pos.target_weights(dataset, idx, current)

        # Blend: final = alpha1 * v1 + alpha2 * orderflow + alpha3 * positioning
        blended: Dict[str, float] = {}
        for s in syms:
            v1_w = self._v1_weights.get(s, 0.0)
            of_w = self._of_weights.get(s, 0.0)
            pos_w = self._pos_weights.get(s, 0.0)
            blended[s] = a1 * v1_w + a2 * of_w + a3 * pos_w

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
            "positioning": self._pos.describe(),
        }
        base["alpha1"] = self.alpha1
        base["alpha2"] = self.alpha2
        base["alpha3"] = self.alpha3
        return base
