"""
NEXUS Portfolio Risk Overlay
===============================

Cross-venue, portfolio-level risk management that monitors aggregate
drawdown and can trigger position scaling or liquidation across all
NEXUS projects simultaneously.

Risk Layers:
  1. Portfolio Drawdown: If combined equity drops >X% from HWM, scale positions
  2. Correlation Spike: If real-time cross-strategy correlation exceeds threshold,
     reduce total leverage (diversification benefit eroding)
  3. Per-Venue Risk: Ensure each venue stays within its margin/leverage limits
  4. Emergency: Kill switch that closes all positions across all venues

Design:
  The overlay sits BETWEEN the signal aggregator and the executors.
  It receives portfolio-level target weights and can scale them down
  (never up) before passing to execution.

Usage:
    from nexus_quant.portfolio.risk_overlay import PortfolioRiskOverlay
    overlay = PortfolioRiskOverlay()
    safe_weights = overlay.apply(venue_targets, portfolio_equity)
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger("nexus.risk_overlay")

PROJ_ROOT = Path(__file__).resolve().parents[2]
RISK_LOG = PROJ_ROOT / "artifacts" / "portfolio" / "risk_log.jsonl"
RISK_STATE = PROJ_ROOT / "artifacts" / "portfolio" / "risk_state.json"


@dataclass
class RiskState:
    """Persistent risk monitoring state."""
    hwm_equity: float = 0.0          # high-water mark
    current_equity: float = 0.0
    drawdown_pct: float = 0.0        # current drawdown from HWM
    scale_factor: float = 1.0        # position scaling (1.0 = no reduction)
    emergency_halt: bool = False
    last_update: str = ""
    drawdown_history: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "hwm_equity": round(self.hwm_equity, 2),
            "current_equity": round(self.current_equity, 2),
            "drawdown_pct": round(self.drawdown_pct, 4),
            "scale_factor": round(self.scale_factor, 4),
            "emergency_halt": self.emergency_halt,
            "last_update": self.last_update,
            "drawdown_history_len": len(self.drawdown_history),
        }

    def save(self) -> None:
        RISK_STATE.parent.mkdir(parents=True, exist_ok=True)
        with open(RISK_STATE, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls) -> "RiskState":
        if RISK_STATE.exists():
            try:
                with open(RISK_STATE) as f:
                    d = json.load(f)
                return cls(
                    hwm_equity=d.get("hwm_equity", 0),
                    current_equity=d.get("current_equity", 0),
                    drawdown_pct=d.get("drawdown_pct", 0),
                    scale_factor=d.get("scale_factor", 1.0),
                    emergency_halt=d.get("emergency_halt", False),
                    last_update=d.get("last_update", ""),
                )
            except Exception:
                pass
        return cls()


@dataclass
class RiskDecision:
    """Output of the risk overlay."""
    scale_factor: float               # 1.0 = no change, 0.5 = halve positions, 0.0 = close all
    venue_targets: Dict[str, Dict[str, float]]  # scaled targets
    reasons: List[str]                # why positions were scaled
    drawdown_pct: float
    emergency: bool


# ── Risk Parameters ──────────────────────────────────────────────────────────

# Drawdown thresholds → scale factors
# If portfolio drawdown exceeds threshold, multiply all weights by scale
DRAWDOWN_TIERS = [
    (0.03, 0.75),   # -3% DD → reduce to 75% of target
    (0.05, 0.50),   # -5% DD → reduce to 50%
    (0.08, 0.25),   # -8% DD → reduce to 25%
    (0.10, 0.00),   # -10% DD → close all positions
]

# Maximum gross leverage across all venues combined
MAX_PORTFOLIO_GROSS_LEVERAGE = 2.0

# Per-venue maximum gross leverage
MAX_VENUE_LEVERAGE = {
    "binance": 1.0,    # perps: max 1x gross
    "deribit": 1.5,    # options: max 1.5x gross (straddle is ~2 deltas)
}

# Correlation spike threshold: if estimated real-time correlation > this,
# reduce leverage (diversification benefit is eroding)
CORRELATION_SPIKE_THRESHOLD = 0.50
CORRELATION_SCALE = 0.70  # multiply weights by this when corr spikes


class PortfolioRiskOverlay:
    """
    Applies portfolio-level risk controls to venue targets.

    Sits between NexusSignalAggregator and executors.
    Can only REDUCE positions (never increase beyond signal targets).
    """

    def __init__(
        self,
        initial_equity: float = 100000.0,
        drawdown_tiers: Optional[List] = None,
        max_gross_leverage: float = MAX_PORTFOLIO_GROSS_LEVERAGE,
    ) -> None:
        self.initial_equity = initial_equity
        self.drawdown_tiers = drawdown_tiers or DRAWDOWN_TIERS
        self.max_gross_leverage = max_gross_leverage
        self.state = RiskState.load()
        if self.state.hwm_equity == 0:
            self.state.hwm_equity = initial_equity
            self.state.current_equity = initial_equity

    def apply(
        self,
        venue_targets: Dict[str, Dict[str, float]],
        current_equity: Optional[float] = None,
        real_time_correlation: Optional[float] = None,
    ) -> RiskDecision:
        """
        Apply risk overlay to venue targets.

        Args:
            venue_targets: {venue: {symbol: weight}} from aggregator
            current_equity: current portfolio equity (sum across venues)
            real_time_correlation: estimated cross-strategy correlation

        Returns:
            RiskDecision with potentially scaled-down weights
        """
        now = datetime.now(timezone.utc).isoformat()
        reasons: List[str] = []
        scale = 1.0
        emergency = False

        # Check emergency halt
        if self.state.emergency_halt:
            return RiskDecision(
                scale_factor=0.0,
                venue_targets={v: {s: 0.0 for s in t} for v, t in venue_targets.items()},
                reasons=["EMERGENCY HALT ACTIVE — manual reset required"],
                drawdown_pct=self.state.drawdown_pct,
                emergency=True,
            )

        # ── 1. Drawdown Check ──
        if current_equity is not None:
            self.state.current_equity = current_equity
            if current_equity > self.state.hwm_equity:
                self.state.hwm_equity = current_equity

            if self.state.hwm_equity > 0:
                dd = (self.state.hwm_equity - current_equity) / self.state.hwm_equity
                self.state.drawdown_pct = dd

                for threshold, dd_scale in self.drawdown_tiers:
                    if dd >= threshold:
                        if dd_scale < scale:
                            scale = dd_scale
                            reasons.append(
                                f"Drawdown {dd:.1%} >= {threshold:.1%} -> scale={dd_scale:.0%}"
                            )
                            if dd_scale == 0.0:
                                emergency = True

        # ── 2. Correlation Spike Check ──
        if real_time_correlation is not None:
            if real_time_correlation > CORRELATION_SPIKE_THRESHOLD:
                corr_scale = CORRELATION_SCALE
                if corr_scale < scale:
                    scale = corr_scale
                    reasons.append(
                        f"Correlation spike {real_time_correlation:.2f} > "
                        f"{CORRELATION_SPIKE_THRESHOLD:.2f} -> scale={corr_scale:.0%}"
                    )

        # ── 3. Apply Scale to Targets ──
        scaled_targets: Dict[str, Dict[str, float]] = {}
        total_gross = 0.0

        for venue, targets in venue_targets.items():
            scaled = {}
            for symbol, weight in targets.items():
                scaled[symbol] = round(weight * scale, 6)
            scaled_targets[venue] = scaled
            total_gross += sum(abs(w) for w in scaled.values())

        # ── 4. Portfolio Gross Leverage Cap ──
        if total_gross > self.max_gross_leverage and total_gross > 0:
            cap_scale = self.max_gross_leverage / total_gross
            for venue in scaled_targets:
                for symbol in scaled_targets[venue]:
                    scaled_targets[venue][symbol] = round(
                        scaled_targets[venue][symbol] * cap_scale, 6
                    )
            reasons.append(
                f"Gross leverage {total_gross:.2f} > max {self.max_gross_leverage:.2f} "
                f"-> capped (scale={cap_scale:.2f})"
            )
            scale *= cap_scale

        # ── 5. Per-Venue Leverage Cap ──
        for venue, max_lev in MAX_VENUE_LEVERAGE.items():
            if venue in scaled_targets:
                venue_gross = sum(abs(w) for w in scaled_targets[venue].values())
                if venue_gross > max_lev and venue_gross > 0:
                    venue_cap = max_lev / venue_gross
                    for symbol in scaled_targets[venue]:
                        scaled_targets[venue][symbol] = round(
                            scaled_targets[venue][symbol] * venue_cap, 6
                        )
                    reasons.append(
                        f"Venue {venue} leverage {venue_gross:.2f} > max {max_lev:.2f} "
                        f"-> capped"
                    )

        # ── 6. Update State ──
        self.state.scale_factor = scale
        self.state.emergency_halt = emergency
        self.state.last_update = now
        self.state.save()

        if not reasons:
            reasons.append("No risk triggers — full position size")

        decision = RiskDecision(
            scale_factor=scale,
            venue_targets=scaled_targets,
            reasons=reasons,
            drawdown_pct=self.state.drawdown_pct,
            emergency=emergency,
        )

        self._log_decision(decision)
        return decision

    def reset_emergency(self) -> None:
        """Manually reset emergency halt."""
        self.state.emergency_halt = False
        self.state.save()
        logger.warning("Emergency halt RESET")

    def reset_hwm(self, new_equity: float) -> None:
        """Reset high-water mark (e.g. after capital injection)."""
        self.state.hwm_equity = new_equity
        self.state.current_equity = new_equity
        self.state.drawdown_pct = 0.0
        self.state.save()
        logger.info("HWM reset to $%.2f", new_equity)

    def status(self) -> str:
        """Human-readable risk status."""
        lines = []
        lines.append("NEXUS Portfolio Risk Overlay")
        lines.append(f"  HWM: ${self.state.hwm_equity:,.2f}")
        lines.append(f"  Current: ${self.state.current_equity:,.2f}")
        lines.append(f"  Drawdown: {self.state.drawdown_pct:.2%}")
        lines.append(f"  Scale factor: {self.state.scale_factor:.2f}")
        lines.append(f"  Emergency halt: {self.state.emergency_halt}")
        lines.append(f"  Last update: {self.state.last_update}")
        lines.append("")
        lines.append("  Drawdown Tiers:")
        for thresh, sc in self.drawdown_tiers:
            lines.append(f"    {thresh:5.1%} -> scale to {sc:.0%}")
        lines.append(f"  Max portfolio leverage: {self.max_gross_leverage:.1f}")
        for v, ml in MAX_VENUE_LEVERAGE.items():
            lines.append(f"  Max {v} leverage: {ml:.1f}")
        return "\n".join(lines)

    def _log_decision(self, decision: RiskDecision) -> None:
        """Log risk decision."""
        RISK_LOG.parent.mkdir(parents=True, exist_ok=True)
        try:
            entry = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "scale_factor": decision.scale_factor,
                "drawdown_pct": decision.drawdown_pct,
                "emergency": decision.emergency,
                "reasons": decision.reasons,
            }
            with open(RISK_LOG, "a") as f:
                f.write(json.dumps(entry) + "\n")
        except Exception as e:
            logger.error("Failed to log risk decision: %s", e)
