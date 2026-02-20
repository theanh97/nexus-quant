"""
Deribit Options Fee Model

Deribit fee schedule (as of 2024):
    Options:
        Maker: 0.03% of underlying value (capped at 12.5% of option price)
        Taker: 0.05% of underlying value (capped at 12.5% of option price)
        Min fee: 0.0001 BTC per contract (or 0.0003 ETH)
    Perpetuals (for delta hedging):
        Maker: -0.01% (rebate)
        Taker: +0.05%
    Settlement: No fee

Reference: https://www.deribit.com/kb/fees
"""
from __future__ import annotations

from typing import Any, Dict

from nexus_quant.backtest.costs import (
    ExecutionCostModel,
    FeeModel,
    ImpactModel,
)


class DeribitOptionsCostModel(ExecutionCostModel):
    """Deribit options-specific cost model.

    Uses ExecutionCostModel with Deribit's fee schedule.
    Options costs are charged on underlying notional, not option premium.

    Deribit fee schedule:
        Taker fee: 0.05% of underlying (5 bps)
        Maker fee: 0.03% of underlying (3 bps)

    Additional costs for realistic execution:
        Slippage: ~2-5 bps on underlying (bid-ask spread)
        Delta hedging (perpetuals): additional 5 bps per rebalance

    NOTE: We use a simplified model treating options PnL as delta-equivalent
    exposure on the underlying. The fee is applied to the underlying notional
    * delta when executing option trades.
    """

    # No special class body needed — use ExecutionCostModel directly


def make_deribit_cost_model(
    execution_style: str = "taker",
    include_hedge_cost: bool = True,
    cost_multiplier: float = 1.0,
) -> ExecutionCostModel:
    """Create Deribit options cost model.

    Args:
        execution_style: "maker" or "taker"
        include_hedge_cost: if True, add delta-hedge (perpetual) costs
        cost_multiplier: scale all costs by this factor (for sensitivity analysis)

    Returns:
        ExecutionCostModel configured for Deribit options
    """
    # Deribit options fees (on underlying notional)
    fee = FeeModel(
        maker_fee_rate=0.0003,    # 3 bps
        taker_fee_rate=0.0005,    # 5 bps
    )

    # Bid-ask spread for options (rough estimate: ~5-10 bps on underlying)
    spread_bps = 10.0 if execution_style == "taker" else 5.0

    # Slippage for size (crypto options markets are illiquid vs perps)
    slippage_bps = 5.0   # 5 bps base slippage

    # Delta-hedge cost: when we trade options, we delta-hedge with perpetuals
    # Perpetual taker fee: 5 bps. Add if hedging is enabled.
    if include_hedge_cost:
        # Model: hedge turnover ≈ 50% of option turnover (avg delta ~0.5)
        # Cost: 0.5 * 5 bps = 2.5 bps additional
        slippage_bps += 2.5

    # Market impact: options market thinner than perps
    impact = ImpactModel(model="sqrt", coef_bps=3.0)

    return ExecutionCostModel(
        fee=fee,
        execution_style=execution_style,
        slippage_bps=slippage_bps,
        spread_bps=spread_bps,
        impact=impact,
        cost_multiplier=cost_multiplier,
    )


# ── Pre-built cost models ─────────────────────────────────────────────────────

# Conservative (worst-case): full taker + high slippage + impact
DERIBIT_CONSERVATIVE = make_deribit_cost_model(
    execution_style="taker",
    include_hedge_cost=True,
    cost_multiplier=1.5,
)

# Realistic: taker fees + moderate slippage
DERIBIT_REALISTIC = make_deribit_cost_model(
    execution_style="taker",
    include_hedge_cost=True,
    cost_multiplier=1.0,
)

# Optimistic: maker fees + minimal slippage
DERIBIT_OPTIMISTIC = make_deribit_cost_model(
    execution_style="maker",
    include_hedge_cost=False,
    cost_multiplier=0.7,
)


def cost_model_from_name(name: str) -> ExecutionCostModel:
    """Get a cost model by name string."""
    models = {
        "deribit_conservative": DERIBIT_CONSERVATIVE,
        "deribit_realistic": DERIBIT_REALISTIC,
        "deribit_optimistic": DERIBIT_OPTIMISTIC,
        "deribit_options": DERIBIT_REALISTIC,   # default alias
    }
    return models.get(name.lower(), DERIBIT_REALISTIC)


def cost_model_from_config(cfg: Dict[str, Any]) -> ExecutionCostModel:
    """Build cost model from a config dict (from JSON backtest config)."""
    if "cost_model_name" in cfg:
        return cost_model_from_name(cfg["cost_model_name"])

    return make_deribit_cost_model(
        execution_style=cfg.get("execution_style", "taker"),
        include_hedge_cost=cfg.get("include_hedge_cost", True),
        cost_multiplier=float(cfg.get("cost_multiplier", 1.0)),
    )
