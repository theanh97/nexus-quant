from __future__ import annotations

import importlib
import logging
from typing import Any, Callable, Dict, List, Optional, Type

from .base import Strategy

logger = logging.getLogger("nexus.strategies.registry")

# ── Dynamic Strategy Registry ────────────────────────────────────────────
# Strategies can be registered dynamically from project directories.
# This allows new markets (FX, options) to add strategies without modifying
# this file. Legacy strategies (crypto) are still imported directly below.

_DYNAMIC_REGISTRY: Dict[str, Callable[..., Strategy]] = {}


def register_strategy(name: str, factory: Callable[..., Strategy]) -> None:
    """Register a strategy factory function by name."""
    _DYNAMIC_REGISTRY[name] = factory
    logger.debug("Registered strategy: %s", name)


def register_strategy_class(name: str, cls: Type[Strategy]) -> None:
    """Register a Strategy subclass by name."""
    _DYNAMIC_REGISTRY[name] = lambda params: cls(params=params)


def list_registered() -> List[str]:
    """List all registered strategy names (dynamic + legacy)."""
    return sorted(set(list(_DYNAMIC_REGISTRY.keys()) + _LEGACY_NAMES))


def load_project_strategies(project_name: str) -> int:
    """
    Dynamically load strategies from a project's strategies/ directory.
    Returns number of strategies loaded.
    """
    count = 0
    try:
        mod = importlib.import_module(f"nexus_quant.projects.{project_name}")
        project_dir = getattr(mod, "__path__", None)
        if not project_dir:
            return 0
        # Try importing the project's strategy module
        try:
            strat_mod = importlib.import_module(f"nexus_quant.projects.{project_name}.strategies")
            # Look for STRATEGIES dict: {"name": StrategyClass}
            strat_dict = getattr(strat_mod, "STRATEGIES", {})
            for sname, scls in strat_dict.items():
                register_strategy_class(sname, scls)
                count += 1
        except ImportError:
            pass
    except ImportError:
        pass
    if count:
        logger.info("Loaded %d strategies from project %s", count, project_name)
    return count
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
from .pair_spread_alpha import PairSpreadAlphaStrategy
from .vol_regime_mom_alpha import VolRegimeMomAlphaStrategy
from .momentum_breakout_alpha import MomentumBreakoutAlphaStrategy
from .p91b_ensemble import P91bEnsembleStrategy
from .breadth_adaptive_ensemble import BreadthAdaptiveEnsembleStrategy


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

    if name == "pair_spread_alpha":
        return PairSpreadAlphaStrategy(params=params)

    if name == "vol_regime_mom_alpha":
        return VolRegimeMomAlphaStrategy(params=params)

    if name == "momentum_breakout_alpha":
        return MomentumBreakoutAlphaStrategy(params=params)

    if name == "p91b_ensemble":
        return P91bEnsembleStrategy(params=params)

    if name == "breadth_adaptive_ensemble":
        return BreadthAdaptiveEnsembleStrategy(params=params)

    if name == "ensemble_v1":
        sub_cfgs = params.pop("sub_strategies", [])
        sub_strats = [make_strategy(sc) for sc in sub_cfgs]
        return EnsembleV1Strategy(sub_strategies=sub_strats, params=params)

    # ── Dynamic registry lookup (project-contributed strategies) ──
    if name in _DYNAMIC_REGISTRY:
        return _DYNAMIC_REGISTRY[name](params)

    raise ValueError(f"Unknown strategy: {name}")


# Legacy strategy names (for list_registered())
_LEGACY_NAMES = [
    "funding_carry_perp_v1", "momentum_xs_v1", "mean_reversion_xs_v1",
    "multi_factor_xs_v1", "ml_factor_xs_v1", "tsmom_v1", "combined_carry_mom_v1",
    "nexus_alpha_v1", "nexus_alpha_v2", "nexus_alpha_v1_regime",
    "nexus_alpha_v1_vol_scaled", "ml_factor_v3", "orderflow_alpha_v1",
    "positioning_alpha_v1", "nexus_ensemble_v1", "nexus_ensemble_v2",
    "funding_carry_alpha", "low_vol_alpha", "regime_switch_ensemble",
    "regime_mixer", "dispersion_alpha", "lead_lag_alpha", "volume_reversal_alpha",
    "basis_momentum_alpha", "vol_breakout_alpha", "rs_acceleration_alpha",
    "taker_buy_alpha", "funding_contrarian_alpha", "mr_funding_alpha",
    "hybrid_alpha", "multitf_momentum_alpha", "sharpe_ratio_alpha",
    "price_level_alpha", "amihud_illiquidity_alpha", "ewma_sharpe_alpha",
    "sortino_alpha", "idio_momentum_alpha", "skip_gram_momentum_alpha",
    "pure_momentum_alpha", "funding_momentum_alpha", "vol_adjusted_momentum_alpha",
    "funding_vol_alpha", "pair_spread_alpha", "vol_regime_mom_alpha",
    "momentum_breakout_alpha", "p91b_ensemble", "ensemble_v1",
]
