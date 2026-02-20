"""
Risk Gate — circuit breakers for live trading.

Checks halt/warning conditions from production config before each rebalance.
If any halt condition is breached, trading is stopped immediately.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

log = logging.getLogger("nexus.risk")


@dataclass
class RiskCheck:
    """Result of a single risk check."""
    name: str
    value: float
    threshold: float
    passed: bool
    severity: str  # "ok" | "warning" | "halt"
    message: str = ""


@dataclass
class RiskReport:
    """Full risk assessment for a rebalance cycle."""
    timestamp: str
    checks: List[RiskCheck]
    can_trade: bool
    halt_reason: str = ""
    warnings: List[str] = field(default_factory=list)


class RiskGate:
    """
    Pre-trade risk gate. Evaluates halt and warning conditions.

    Config (from production_p91b_champion.json):
      halt_conditions:
        rolling_30d_sharpe_below: -1.0
        max_drawdown_pct: 15.0
        max_daily_loss_pct: 5.0
        correlation_with_btc_above: 0.60
        consecutive_losing_days: 14
      warning_conditions:
        rolling_30d_sharpe_below: 0.0
        drawdown_pct_above: 8.0
        daily_turnover_above_pct: 50.0
    """

    def __init__(
        self,
        halt_conditions: Optional[Dict[str, float]] = None,
        warning_conditions: Optional[Dict[str, float]] = None,
        state_dir: Optional[Path] = None,
    ):
        self._halt = halt_conditions or {
            "max_drawdown_pct": 15.0,
            "max_daily_loss_pct": 5.0,
            "consecutive_losing_days": 14,
        }
        self._warn = warning_conditions or {
            "drawdown_pct_above": 8.0,
            "daily_turnover_above_pct": 50.0,
        }
        self._state_dir = state_dir or Path("artifacts/execution")
        self._state_dir.mkdir(parents=True, exist_ok=True)
        self._state_file = self._state_dir / "risk_state.json"
        self._state = self._load_state()

    # ------------------------------------------------------------------
    # State persistence
    # ------------------------------------------------------------------

    def _load_state(self) -> Dict[str, Any]:
        """Load risk tracking state (peak equity, daily PnL, etc.)."""
        if self._state_file.exists():
            try:
                return json.loads(self._state_file.read_text())
            except Exception:
                pass
        return {
            "peak_equity": 0.0,
            "daily_returns": [],  # last 30 days
            "consecutive_losing_days": 0,
            "last_balance": 0.0,
            "last_date": "",
            "halted": False,
            "halt_reason": "",
        }

    def _save_state(self) -> None:
        try:
            self._state_file.write_text(json.dumps(self._state, indent=2))
        except Exception:
            pass

    # ------------------------------------------------------------------
    # State updates
    # ------------------------------------------------------------------

    def update_equity(self, balance: float) -> None:
        """Update equity tracking after each rebalance."""
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")

        # Update peak
        if balance > self._state.get("peak_equity", 0):
            self._state["peak_equity"] = balance

        # Daily return
        last_bal = self._state.get("last_balance", 0)
        if last_bal > 0 and self._state.get("last_date") != today:
            daily_ret = (balance - last_bal) / last_bal
            self._state.setdefault("daily_returns", []).append({
                "date": today,
                "return": round(daily_ret, 6),
                "balance": round(balance, 2),
            })
            # Keep last 30 days
            self._state["daily_returns"] = self._state["daily_returns"][-30:]

            # Consecutive losing days
            if daily_ret < 0:
                self._state["consecutive_losing_days"] = \
                    self._state.get("consecutive_losing_days", 0) + 1
            else:
                self._state["consecutive_losing_days"] = 0

        self._state["last_balance"] = balance
        self._state["last_date"] = today
        self._save_state()

    # ------------------------------------------------------------------
    # Risk evaluation
    # ------------------------------------------------------------------

    def evaluate(self, balance: float, daily_turnover_pct: float = 0.0) -> RiskReport:
        """
        Run all risk checks. Returns RiskReport with can_trade flag.
        """
        ts = datetime.now(timezone.utc).isoformat()
        checks: List[RiskCheck] = []
        warnings: List[str] = []
        halt_reason = ""

        # Already halted?
        if self._state.get("halted"):
            return RiskReport(
                timestamp=ts,
                checks=[],
                can_trade=False,
                halt_reason=f"Previously halted: {self._state.get('halt_reason', '?')}",
            )

        peak = self._state.get("peak_equity", balance)
        if peak <= 0:
            peak = balance

        # --- HALT checks ---

        # 1. Max drawdown
        if peak > 0:
            dd_pct = ((peak - balance) / peak) * 100
            threshold = self._halt.get("max_drawdown_pct", 15.0)
            passed = dd_pct < threshold
            checks.append(RiskCheck(
                name="max_drawdown",
                value=round(dd_pct, 2),
                threshold=threshold,
                passed=passed,
                severity="halt" if not passed else "ok",
                message=f"Drawdown {dd_pct:.1f}% {'< ' if passed else '>= '}{threshold}%",
            ))
            if not passed:
                halt_reason = f"Max drawdown breached: {dd_pct:.1f}% >= {threshold}%"

        # 2. Max daily loss
        daily_rets = self._state.get("daily_returns", [])
        if daily_rets:
            last_ret = daily_rets[-1].get("return", 0) * 100
            threshold = self._halt.get("max_daily_loss_pct", 5.0)
            passed = last_ret > -threshold
            checks.append(RiskCheck(
                name="max_daily_loss",
                value=round(last_ret, 2),
                threshold=-threshold,
                passed=passed,
                severity="halt" if not passed else "ok",
                message=f"Daily return {last_ret:.2f}% {'> ' if passed else '<= '}-{threshold}%",
            ))
            if not passed:
                halt_reason = f"Daily loss breached: {last_ret:.2f}% <= -{threshold}%"

        # 3. Consecutive losing days
        consec = self._state.get("consecutive_losing_days", 0)
        threshold = int(self._halt.get("consecutive_losing_days", 14))
        passed = consec < threshold
        checks.append(RiskCheck(
            name="consecutive_losing_days",
            value=consec,
            threshold=threshold,
            passed=passed,
            severity="halt" if not passed else "ok",
            message=f"{consec} consecutive losing days {'< ' if passed else '>= '}{threshold}",
        ))
        if not passed:
            halt_reason = f"Consecutive losing days: {consec} >= {threshold}"

        # --- WARNING checks ---

        # Drawdown warning
        if peak > 0:
            dd_pct = ((peak - balance) / peak) * 100
            warn_thresh = self._warn.get("drawdown_pct_above", 8.0)
            if dd_pct >= warn_thresh:
                warnings.append(f"Drawdown warning: {dd_pct:.1f}% >= {warn_thresh}%")

        # Turnover warning
        turn_thresh = self._warn.get("daily_turnover_above_pct", 50.0)
        if daily_turnover_pct >= turn_thresh:
            warnings.append(f"Turnover warning: {daily_turnover_pct:.1f}% >= {turn_thresh}%")

        # If halted, persist
        can_trade = not bool(halt_reason)
        if not can_trade:
            self._state["halted"] = True
            self._state["halt_reason"] = halt_reason
            self._save_state()
            log.critical("TRADING HALTED: %s", halt_reason)

        for w in warnings:
            log.warning(w)

        return RiskReport(
            timestamp=ts,
            checks=checks,
            can_trade=can_trade,
            halt_reason=halt_reason,
            warnings=warnings,
        )

    def reset_halt(self) -> None:
        """Manually reset halt state (requires human confirmation)."""
        self._state["halted"] = False
        self._state["halt_reason"] = ""
        self._save_state()
        log.info("Risk halt RESET by operator")

    def is_halted(self) -> bool:
        return self._state.get("halted", False)
