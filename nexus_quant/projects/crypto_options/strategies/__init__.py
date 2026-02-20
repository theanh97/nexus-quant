"""
Crypto Options Strategies — Dynamic Registration

Strategies:
    crypto_vrp: Variance Risk Premium — short IV when IV >> RV
    crypto_skew_mr: Skew Mean-Reversion — fade extreme put/call skew
    crypto_term_structure: Term Structure Calendar Spread — mean-revert front/back IV spread
"""
from .variance_premium import VariancePremiumStrategy
from .skew_trade import SkewTradeStrategy
from .term_structure import TermStructureStrategy

STRATEGIES = {
    "crypto_vrp": VariancePremiumStrategy,
    "crypto_skew_mr": SkewTradeStrategy,
    "crypto_term_structure": TermStructureStrategy,
}

__all__ = [
    "VariancePremiumStrategy",
    "SkewTradeStrategy",
    "TermStructureStrategy",
    "STRATEGIES",
]
