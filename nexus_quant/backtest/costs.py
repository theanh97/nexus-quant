from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Any, Dict, Optional


def _clamp(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else hi if x > hi else x


@dataclass(frozen=True)
class FeeModel:
    """
    Maker/taker fee model (rate as fraction of notional, e.g. 0.0004 = 4 bps).
    """

    maker_fee_rate: float
    taker_fee_rate: float

    def rate(self, execution_style: str) -> float:
        style = (execution_style or "").strip().lower()
        if style == "maker":
            return float(self.maker_fee_rate)
        return float(self.taker_fee_rate)


@dataclass(frozen=True)
class ImpactModel:
    """
    Minimal slippage/impact model used as an extra *rate* on traded notional.

    - model="none": rate=0
    - model="sqrt": impact_bps = coef_bps * sqrt(turnover)
    """

    model: str = "none"
    coef_bps: float = 0.0

    def rate(self, turnover: float) -> float:
        m = (self.model or "none").strip().lower()
        if m == "none":
            return 0.0
        if m == "sqrt":
            t = max(0.0, float(turnover))
            bps = float(self.coef_bps) * math.sqrt(t)
            bps = _clamp(bps, 0.0, 10_000.0)
            return bps / 10_000.0
        # Unknown -> be conservative: no impact model rather than hallucinating a formula.
        return 0.0


@dataclass(frozen=True)
class ExecutionCostModel:
    """
    Transaction cost model:
    - fee (maker/taker)
    - slippage (bps)
    - spread (bps, charged as half-spread for taker-style execution)
    - optional impact model

    Everything is deterministic and simple by design (stdlib-only).
    """

    fee: FeeModel
    execution_style: str = "taker"  # maker|taker
    slippage_bps: float = 0.0
    spread_bps: float = 0.0
    impact: ImpactModel = ImpactModel()
    cost_multiplier: float = 1.0

    def with_multiplier(self, mult: float) -> "ExecutionCostModel":
        return replace(self, cost_multiplier=float(self.cost_multiplier) * float(mult))

    def cost(self, *, equity: float, turnover: float) -> Dict[str, float]:
        """
        Return a cost breakdown. Total cost should be subtracted from equity.

        Model: cost = equity * turnover * rate
        rate = fee_rate + slippage_rate + spread_rate + impact_rate
        """
        eq = max(0.0, float(equity))
        t = max(0.0, float(turnover))
        mult = max(0.0, float(self.cost_multiplier))

        fee_rate = max(0.0, float(self.fee.rate(self.execution_style)))
        slip_rate = max(0.0, float(self.slippage_bps) / 10_000.0)

        # If we assume taker-style execution, half-spread is a reasonable first-order cost.
        spr = max(0.0, float(self.spread_bps) / 10_000.0)
        spread_rate = spr * 0.5 if (self.execution_style or "").strip().lower() != "maker" else 0.0

        impact_rate = max(0.0, float(self.impact.rate(t)))

        fee_cost = eq * t * fee_rate * mult
        slippage_cost = eq * t * slip_rate * mult
        spread_cost = eq * t * spread_rate * mult
        impact_cost = eq * t * impact_rate * mult
        total = fee_cost + slippage_cost + spread_cost + impact_cost
        return {
            "fee_cost": float(fee_cost),
            "slippage_cost": float(slippage_cost),
            "spread_cost": float(spread_cost),
            "impact_cost": float(impact_cost),
            "cost": float(total),
            "fee_rate": float(fee_rate),
            "slippage_rate": float(slip_rate),
            "spread_rate": float(spread_rate),
            "impact_rate": float(impact_rate),
            "cost_multiplier": float(mult),
        }


def cost_model_from_config(
    costs_cfg: Dict[str, Any],
    execution_cfg: Optional[Dict[str, Any]] = None,
    venue_cfg: Optional[Dict[str, Any]] = None,
) -> ExecutionCostModel:
    """
    Backward compatible:
    - legacy: costs.fee_rate + costs.slippage_rate
    - newer: execution.style + execution.slippage_bps/spread_bps + costs.maker_fee_rate/taker_fee_rate
    """
    execution_cfg = execution_cfg or {}
    venue_cfg = venue_cfg or {}

    style = str(execution_cfg.get("style") or costs_cfg.get("execution_style") or "taker")
    style = style.strip().lower()
    if style not in {"maker", "taker"}:
        style = "taker"

    maker_fee = costs_cfg.get("maker_fee_rate")
    taker_fee = costs_cfg.get("taker_fee_rate")
    if maker_fee is None:
        maker_fee = venue_cfg.get("maker_fee_rate", venue_cfg.get("maker_fee"))
    if taker_fee is None:
        taker_fee = venue_cfg.get("taker_fee_rate", venue_cfg.get("taker_fee"))
    fee_rate_legacy = costs_cfg.get("fee_rate")
    if maker_fee is None and taker_fee is None:
        # If only fee_rate exists, treat it as the fee for the selected execution style.
        # (Don't guess maker/taker relationship; keep it conservative/auditable.)
        fr = max(0.0, float(fee_rate_legacy or 0.0))
        maker_fee = fr
        taker_fee = fr
    fee = FeeModel(maker_fee_rate=max(0.0, float(maker_fee or 0.0)), taker_fee_rate=max(0.0, float(taker_fee or 0.0)))

    # Slippage/spread can be given either in bps or rate (legacy).
    sl_bps = execution_cfg.get("slippage_bps")
    if sl_bps is None:
        sl_rate = max(0.0, float(costs_cfg.get("slippage_rate") or 0.0))
        sl_bps = sl_rate * 10_000.0

    spread_bps = float(execution_cfg.get("spread_bps") or 0.0)

    impact_cfg = execution_cfg.get("impact") or {}
    impact = ImpactModel(model=str(impact_cfg.get("model") or "none"), coef_bps=float(impact_cfg.get("coef_bps") or impact_cfg.get("coef") or 0.0))

    mult = float(costs_cfg.get("cost_multiplier") or 1.0)
    return ExecutionCostModel(
        fee=fee,
        execution_style=style,
        slippage_bps=float(sl_bps or 0.0),
        spread_bps=float(spread_bps),
        impact=impact,
        cost_multiplier=float(mult),
    )
