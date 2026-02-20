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
from .lead_lag_alpha import LeadLagAlphaStrategy
from .volume_reversal_alpha import VolumeReversalAlphaStrategy
from .basis_momentum_alpha import BasisMomentumAlphaStrategy
from .vol_breakout_alpha import VolBreakoutAlphaStrategy
from .rs_acceleration_alpha import RSAccelerationAlphaStrategy
from .taker_buy_alpha import TakerBuyAlphaStrategy
from .funding_contrarian_alpha import FundingContrarianAlphaStrategy
from .mean_reversion_funding_alpha import MeanReversionFundingAlphaStrategy
from .hybrid_alpha import HybridAlphaStrategy
from .multitf_momentum_alpha import MultiTFMomentumAlphaStrategy
from .sharpe_ratio_alpha import SharpeRatioAlphaStrategy
from .price_level_alpha import PriceLevelAlphaStrategy
from .amihud_illiquidity_alpha import AmihudIlliquidityAlphaStrategy
from .ewma_sharpe_alpha import EWMASharpeAlphaStrategy
from .sortino_alpha import SortinoAlphaStrategy
from .idio_momentum_alpha import IdioMomentumAlphaStrategy
from .skip_gram_momentum_alpha import SkipGramMomentumAlphaStrategy
from .pure_momentum_alpha import PureMomentumAlphaStrategy
from .funding_momentum_alpha import FundingMomentumAlphaStrategy
from .vol_adjusted_momentum_alpha import VolAdjustedMomentumAlphaStrategy
from .funding_vol_alpha import FundingVolAlphaStrategy


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

    if name == "lead_lag_alpha":
        return LeadLagAlphaStrategy(params=params)

    if name == "volume_reversal_alpha":
        return VolumeReversalAlphaStrategy(params=params)

    if name == "basis_momentum_alpha":
        return BasisMomentumAlphaStrategy(params=params)

    if name == "vol_breakout_alpha":
        return VolBreakoutAlphaStrategy(params=params)

    if name == "rs_acceleration_alpha":
        return RSAccelerationAlphaStrategy(params=params)

    if name == "taker_buy_alpha":
        return TakerBuyAlphaStrategy(params=params)

    if name == "funding_contrarian_alpha":
        return FundingContrarianAlphaStrategy(params=params)

    if name == "mr_funding_alpha":
        return MeanReversionFundingAlphaStrategy(params=params)

    if name == "hybrid_alpha":
        return HybridAlphaStrategy(params=params)

    if name == "multitf_momentum_alpha":
        return MultiTFMomentumAlphaStrategy(params=params)

    if name == "sharpe_ratio_alpha":
        return SharpeRatioAlphaStrategy(params=params)

    if name == "price_level_alpha":
        return PriceLevelAlphaStrategy(params=params)

    if name == "amihud_illiquidity_alpha":
        return AmihudIlliquidityAlphaStrategy(params=params)

    if name == "ewma_sharpe_alpha":
        return EWMASharpeAlphaStrategy(params=params)

    if name == "sortino_alpha":
        return SortinoAlphaStrategy(params=params)

    if name == "idio_momentum_alpha":
        return IdioMomentumAlphaStrategy(params=params)

    if name == "skip_gram_momentum_alpha":
        return SkipGramMomentumAlphaStrategy(params=params)

    if name == "pure_momentum_alpha":
        return PureMomentumAlphaStrategy(params=params)

    if name == "funding_momentum_alpha":
        return FundingMomentumAlphaStrategy(params=params)

    if name == "vol_adjusted_momentum_alpha":
        return VolAdjustedMomentumAlphaStrategy(params=params)

    if name == "funding_vol_alpha":
        return FundingVolAlphaStrategy(params=params)

    if name == "ensemble_v1":
        sub_cfgs = params.pop("sub_strategies", [])
        sub_strats = [make_strategy(sc) for sc in sub_cfgs]
        return EnsembleV1Strategy(sub_strategies=sub_strats, params=params)

    raise ValueError(f"Unknown strategy: {name}")
