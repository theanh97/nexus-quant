"""
Commodity CTA Strategies
=========================
Dynamic strategy registry for the commodity_cta project.
"""
from .trend_following import TrendFollowingStrategy
from .carry_roll import CarryRollStrategy
from .momentum_value import MomentumValueStrategy
from .cta_ensemble import CTAEnsembleStrategy

# Registry for dynamic discovery by NEXUS platform
STRATEGIES = {
    "cta_trend": TrendFollowingStrategy,
    "cta_carry": CarryRollStrategy,
    "cta_mom_value": MomentumValueStrategy,
    "cta_ensemble": CTAEnsembleStrategy,
}

__all__ = [
    "TrendFollowingStrategy",
    "CarryRollStrategy",
    "MomentumValueStrategy",
    "CTAEnsembleStrategy",
    "STRATEGIES",
]
