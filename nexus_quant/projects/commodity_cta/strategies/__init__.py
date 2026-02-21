"""
Commodity CTA Strategies
=========================
Dynamic strategy registry for the commodity_cta project.
"""
from .trend_following import TrendFollowingStrategy
from .carry_roll import CarryRollStrategy
from .momentum_value import MomentumValueStrategy
from .cta_ensemble import CTAEnsembleStrategy
from .tsmom import TSMOMStrategy
from .donchian_breakout import DonchianBreakoutStrategy
from .cta_ensemble_v2 import CTAEnsembleV2Strategy
from .rp_mom_dd import RPMomDDStrategy

# Registry for dynamic discovery by NEXUS platform
STRATEGIES = {
    "cta_trend": TrendFollowingStrategy,
    "cta_carry": CarryRollStrategy,
    "cta_mom_value": MomentumValueStrategy,
    "cta_ensemble": CTAEnsembleStrategy,
    "cta_tsmom": TSMOMStrategy,
    "cta_donchian": DonchianBreakoutStrategy,
    "cta_ensemble_v2": CTAEnsembleV2Strategy,
    "rp_mom_dd": RPMomDDStrategy,
}

__all__ = [
    "TrendFollowingStrategy",
    "CarryRollStrategy",
    "MomentumValueStrategy",
    "CTAEnsembleStrategy",
    "TSMOMStrategy",
    "DonchianBreakoutStrategy",
    "CTAEnsembleV2Strategy",
    "RPMomDDStrategy",
    "STRATEGIES",
]
