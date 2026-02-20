"""
Commodity Futures Cost Model
==============================
CME/ICE futures commission and slippage model.

CME E-mini futures (ES, NQ, etc.): ~$2.25/contract RT
As % of notional at standard lot sizes:
  - WTI (CL): 1000 bbl × ~$80 = $80,000 notional → $2.25 / $80,000 = ~0.003%
  - Gold (GC): 100 oz × ~$1,800 = $180,000 notional → $2.25 / $180,000 = ~0.001%

Effective rates (rounded up conservatively):
  commission : 0.01% (1 bp) — applies to round-trip turnover
  slippage   : 0.05% (5 bps) — daily bars = low frequency, typically >1 tick
  spread     : 0.01% (1 bp) — bid-ask spread, tight for liquid futures

Total round-trip cost: ~0.07% (7 bps per rebalance)
Very cheap vs crypto (typically 10-30 bps round-trip).
"""
from __future__ import annotations

from nexus_quant.backtest.costs import ExecutionCostModel, FeeModel, ImpactModel

# ── Default commodity futures cost model ──────────────────────────────────────

COMMODITY_FUTURES_COSTS = {
    "maker_fee_rate": 0.0001,    # 1 bp commission (maker)
    "taker_fee_rate": 0.0001,    # 1 bp commission (taker — same for futures)
    "slippage_bps": 5.0,         # 5 bps slippage (conservative for daily bars)
    "spread_bps": 1.0,           # 1 bp bid-ask spread
    "cost_multiplier": 1.0,
}


def commodity_futures_cost_model(
    slippage_bps: float = 5.0,
    commission_bps: float = 1.0,
    spread_bps: float = 1.0,
    cost_multiplier: float = 1.0,
) -> ExecutionCostModel:
    """
    Build an ExecutionCostModel suitable for commodity futures.

    Args:
        slippage_bps    : Market impact / slippage in basis points (default 5)
        commission_bps  : Round-trip commission in basis points (default 1)
        spread_bps      : Bid-ask spread in basis points (default 1)
        cost_multiplier : Scale all costs (for sensitivity testing)

    Returns:
        ExecutionCostModel instance
    """
    fee_rate = commission_bps / 10_000.0
    return ExecutionCostModel(
        fee=FeeModel(
            maker_fee_rate=fee_rate,
            taker_fee_rate=fee_rate,
        ),
        execution_style="taker",
        slippage_bps=slippage_bps,
        spread_bps=spread_bps,
        impact=ImpactModel(model="none", coef_bps=0.0),
        cost_multiplier=cost_multiplier,
    )
