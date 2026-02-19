from __future__ import annotations

from typing import Any, Dict

from .base import Strategy
from .funding_carry import FundingCarryPerpV1Strategy
from .momentum import MomentumCrossSectionV1Strategy
from .mean_reversion import MeanReversionCrossSectionV1Strategy
from .multi_factor import MultiFactorCrossSectionV1Strategy
from .ml_factor import MLFactorCrossSectionV1Strategy
from .ensemble import EnsembleV1Strategy
from .tsmom import TimeSerisMomentumV1Strategy
from .combined_carry_mom import CombinedCarryMomentumV1Strategy
from .nexus_alpha import NexusAlphaV1Strategy
from .nexus_alpha_v2 import NexusAlphaV2Strategy
from .ml_factor_v3 import MLFactorV3Strategy
from .regime_adaptive import NexusAlphaV1RegimeStrategy, NexusAlphaV1VolScaledStrategy
from .orderflow_alpha import OrderflowAlphaV1Strategy
from .positioning_alpha import PositioningAlphaV1Strategy
from .nexus_ensemble_v1 import NexusEnsembleV1Strategy
from .nexus_ensemble_v2 import NexusEnsembleV2Strategy
from .funding_carry_alpha import FundingCarryAlphaStrategy
from .low_vol_alpha import LowVolAlphaStrategy
from .regime_switch_ensemble import RegimeSwitchEnsembleStrategy
from .regime_mixer import RegimeMixerStrategy
from .dispersion_alpha import DispersionAlphaStrategy


def make_strategy(strategy_cfg: Dict[str, Any]) -> Strategy:
    name = str(strategy_cfg.get("name") or "")
    params = dict(strategy_cfg.get("params") or {})

    if name == "funding_carry_perp_v1":
        return FundingCarryPerpV1Strategy(params=params)
    if name == "momentum_xs_v1":
        return MomentumCrossSectionV1Strategy(params=params)
    if name == "mean_reversion_xs_v1":
        return MeanReversionCrossSectionV1Strategy(params=params)
    if name == "multi_factor_xs_v1":
        return MultiFactorCrossSectionV1Strategy(params=params)
    if name == "ml_factor_xs_v1":
        return MLFactorCrossSectionV1Strategy(params=params)
    if name == "tsmom_v1":
        return TimeSerisMomentumV1Strategy(params=params)
    if name == "combined_carry_mom_v1":
        return CombinedCarryMomentumV1Strategy(params=params)
    if name == "nexus_alpha_v1":
        return NexusAlphaV1Strategy(params=params)
    if name == "nexus_alpha_v2":
        return NexusAlphaV2Strategy(params=params)
    if name == "nexus_alpha_v1_regime":
        return NexusAlphaV1RegimeStrategy(params=params)
    if name == "nexus_alpha_v1_vol_scaled":
        return NexusAlphaV1VolScaledStrategy(params=params)

    if name == "ml_factor_v3":
        return MLFactorV3Strategy(params=params)

    if name == "orderflow_alpha_v1":
        return OrderflowAlphaV1Strategy(params=params)

    if name == "positioning_alpha_v1":
        return PositioningAlphaV1Strategy(params=params)

    if name == "nexus_ensemble_v1":
        return NexusEnsembleV1Strategy(params=params)

    if name == "nexus_ensemble_v2":
        return NexusEnsembleV2Strategy(params=params)

    if name == "funding_carry_alpha":
        return FundingCarryAlphaStrategy(params=params)

    if name == "low_vol_alpha":
        return LowVolAlphaStrategy(params=params)

    if name == "regime_switch_ensemble":
        return RegimeSwitchEnsembleStrategy(params=params)

    if name == "regime_mixer":
        return RegimeMixerStrategy(params=params)

    if name == "dispersion_alpha":
        return DispersionAlphaStrategy(params=params)

    if name == "ensemble_v1":
        sub_cfgs = params.pop("sub_strategies", [])
        sub_strats = [make_strategy(sc) for sc in sub_cfgs]
        return EnsembleV1Strategy(sub_strategies=sub_strats, params=params)

    raise ValueError(f"Unknown strategy: {name}")
