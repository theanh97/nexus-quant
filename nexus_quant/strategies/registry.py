from __future__ import annotations

from typing import Any, Dict

from .base import Strategy
from .funding_carry import FundingCarryPerpV1Strategy
from .momentum import MomentumCrossSectionV1Strategy
from .mean_reversion import MeanReversionCrossSectionV1Strategy


def make_strategy(strategy_cfg: Dict[str, Any]) -> Strategy:
    name = str(strategy_cfg.get("name") or "")
    params = dict(strategy_cfg.get("params") or {})

    if name == "funding_carry_perp_v1":
        return FundingCarryPerpV1Strategy(params=params)
    if name == "momentum_xs_v1":
        return MomentumCrossSectionV1Strategy(params=params)
    if name == "mean_reversion_xs_v1":
        return MeanReversionCrossSectionV1Strategy(params=params)

    raise ValueError(f"Unknown strategy: {name}")

