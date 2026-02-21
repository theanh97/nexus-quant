"""
Crypto Options Paper State Tracker
====================================

Tracks hypothetical P&L for the VRP + Skew MR ensemble in paper trading mode.
Unlike perps (simple price returns), options P&L uses model-based pricing:

  VRP P&L:     0.5 * (IV² - RV²) * dt   (gamma/theta carry)
  Skew MR P&L: vega * Δskew_z            (vega-based mean reversion)

Persists state between cycles. Called by the NEXUS runner in paper/dry_run mode.

State:  artifacts/crypto_options/paper_state.json
Log:    artifacts/crypto_options/paper_pnl_log.jsonl

Usage:
    from nexus_quant.projects.crypto_options.paper_state import OptionsPaperTracker
    tracker = OptionsPaperTracker()
    tracker.update(signal, market_snapshot)
    print(tracker.status_report())
"""
from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger("nexus.options_paper")

PROJ_ROOT = Path(__file__).resolve().parents[3]
STATE_PATH = PROJ_ROOT / "artifacts" / "crypto_options" / "paper_state.json"
PNL_LOG = PROJ_ROOT / "artifacts" / "crypto_options" / "paper_pnl_log.jsonl"

# Model constants (from backtest engine)
DT_DAILY = 1.0 / 365.0


@dataclass
class PositionState:
    """Single symbol position state with strategy breakdown."""
    symbol: str
    weight: float                   # net target weight from ensemble
    vrp_weight: float               # VRP component weight
    skew_weight: float              # Skew MR component weight
    entry_iv: Optional[float]       # IV at entry (for VRP mark-to-market)
    entry_skew: Optional[float]     # Skew at entry (for Skew MR mark-to-market)
    entry_price: Optional[float]    # Spot price at entry
    entry_epoch: int                # epoch when position was opened

    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "weight": round(self.weight, 6),
            "vrp_weight": round(self.vrp_weight, 6),
            "skew_weight": round(self.skew_weight, 6),
            "entry_iv": round(self.entry_iv, 4) if self.entry_iv else None,
            "entry_skew": round(self.entry_skew, 4) if self.entry_skew else None,
            "entry_price": round(self.entry_price, 2) if self.entry_price else None,
            "entry_epoch": self.entry_epoch,
        }


@dataclass
class OptionsPaperState:
    """Full paper trading state for crypto_options."""
    equity: float = 100_000.0            # hypothetical starting equity (USD)
    positions: Dict[str, PositionState] = field(default_factory=dict)
    cumulative_pnl: float = 0.0          # total P&L since inception
    vrp_pnl: float = 0.0                 # VRP component P&L
    skew_pnl: float = 0.0                # Skew MR component P&L
    n_cycles: int = 0                    # number of update cycles
    hwm_equity: float = 100_000.0        # high-water mark for drawdown
    max_drawdown: float = 0.0            # worst drawdown observed
    pnl_history: List[Dict[str, Any]] = field(default_factory=list)
    created_at: str = ""
    updated_at: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "equity": round(self.equity, 2),
            "positions": {k: v.to_dict() for k, v in self.positions.items()},
            "cumulative_pnl": round(self.cumulative_pnl, 2),
            "vrp_pnl": round(self.vrp_pnl, 2),
            "skew_pnl": round(self.skew_pnl, 2),
            "n_cycles": self.n_cycles,
            "hwm_equity": round(self.hwm_equity, 2),
            "max_drawdown": round(self.max_drawdown, 6),
            "pnl_history": self.pnl_history[-200:],  # keep last 200
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    def save(self) -> None:
        STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(STATE_PATH, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls) -> Optional["OptionsPaperState"]:
        if not STATE_PATH.exists():
            return None
        try:
            with open(STATE_PATH) as f:
                d = json.load(f)
            state = cls(
                equity=d.get("equity", 100_000.0),
                cumulative_pnl=d.get("cumulative_pnl", 0.0),
                vrp_pnl=d.get("vrp_pnl", 0.0),
                skew_pnl=d.get("skew_pnl", 0.0),
                n_cycles=d.get("n_cycles", 0),
                hwm_equity=d.get("hwm_equity", 100_000.0),
                max_drawdown=d.get("max_drawdown", 0.0),
                pnl_history=d.get("pnl_history", []),
                created_at=d.get("created_at", ""),
                updated_at=d.get("updated_at", ""),
            )
            # Restore positions
            for sym, pos_d in d.get("positions", {}).items():
                state.positions[sym] = PositionState(
                    symbol=pos_d["symbol"],
                    weight=pos_d["weight"],
                    vrp_weight=pos_d["vrp_weight"],
                    skew_weight=pos_d["skew_weight"],
                    entry_iv=pos_d.get("entry_iv"),
                    entry_skew=pos_d.get("entry_skew"),
                    entry_price=pos_d.get("entry_price"),
                    entry_epoch=pos_d.get("entry_epoch", 0),
                )
            return state
        except Exception as e:
            logger.warning("Failed to load paper state: %s", e)
            return None


class OptionsPaperTracker:
    """
    Tracks hypothetical options P&L cycle-by-cycle.

    Called after each signal generation with the signal dict and
    current market snapshot. Computes model-based P&L:

    - VRP: theta carry = 0.5 * w * (IV² - RV²) * dt per day
    - Skew MR: vega P&L = w * Δskew (change in 25d skew level)

    P&L is scaled by equity and weights to produce USD returns.
    """

    def __init__(self, initial_equity: float = 100_000.0) -> None:
        self.state = OptionsPaperState.load() or OptionsPaperState(
            equity=initial_equity,
            hwm_equity=initial_equity,
            created_at=_now_iso(),
        )

    def update(
        self,
        signal: Dict[str, Any],
        market_snapshot: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Process one cycle update.

        Args:
            signal: Output from OptionsSignalGenerator.generate()
            market_snapshot: Current market data per symbol
                Expected keys per symbol: iv_atm, skew_25d, rv_24h, price

        Returns:
            Dict with cycle P&L breakdown.
        """
        now = _now_iso()
        epoch = int(datetime.now(timezone.utc).timestamp())
        snapshot = market_snapshot or signal.get("market_snapshot", {})

        cycle_vrp_pnl = 0.0
        cycle_skew_pnl = 0.0
        sym_details: Dict[str, Dict[str, Any]] = {}

        target_weights = signal.get("target_weights", {})
        vrp_signal = signal.get("vrp_signal", {})
        skew_signal = signal.get("skew_signal", {})

        for sym in target_weights:
            mkt = snapshot.get(sym, {})
            curr_iv = mkt.get("iv_atm")
            curr_rv = mkt.get("rv_24h")
            curr_skew = mkt.get("skew_25d")
            curr_price = mkt.get("price")

            old_pos = self.state.positions.get(sym)
            sym_vrp = 0.0
            sym_skew = 0.0

            # -- VRP P&L: theta carry --
            if old_pos and old_pos.vrp_weight != 0.0 and curr_iv is not None and curr_rv is not None:
                # PnL = -weight * 0.5 * (IV² - RV²) * dt
                # (negative weight = short vol → positive when IV > RV)
                iv2 = curr_iv ** 2
                rv2 = curr_rv ** 2
                sym_vrp = (-old_pos.vrp_weight) * 0.5 * (iv2 - rv2) * DT_DAILY
                # Scale by equity
                sym_vrp *= self.state.equity

            # -- Skew MR P&L: vega on skew change --
            if old_pos and old_pos.skew_weight != 0.0 and curr_skew is not None and old_pos.entry_skew is not None:
                # PnL = -weight * Δskew (short skew profits when skew reverts)
                delta_skew = curr_skew - old_pos.entry_skew
                sym_skew = (-old_pos.skew_weight) * delta_skew
                sym_skew *= self.state.equity

            cycle_vrp_pnl += sym_vrp
            cycle_skew_pnl += sym_skew

            sym_details[sym] = {
                "vrp_pnl": round(sym_vrp, 2),
                "skew_pnl": round(sym_skew, 2),
                "total_pnl": round(sym_vrp + sym_skew, 2),
                "iv": curr_iv,
                "rv": curr_rv,
                "skew": curr_skew,
                "price": curr_price,
            }

            # -- Update position state for next cycle --
            new_vrp_w = vrp_signal.get(sym, 0.0)
            new_skew_w = skew_signal.get(sym, 0.0)
            new_weight = target_weights.get(sym, 0.0)

            if abs(new_weight) > 1e-6:
                self.state.positions[sym] = PositionState(
                    symbol=sym,
                    weight=new_weight,
                    vrp_weight=new_vrp_w,
                    skew_weight=new_skew_w,
                    entry_iv=curr_iv,
                    entry_skew=curr_skew,
                    entry_price=curr_price,
                    entry_epoch=epoch,
                )
            else:
                # Flat — remove position
                self.state.positions.pop(sym, None)

        # -- Update equity --
        total_pnl = cycle_vrp_pnl + cycle_skew_pnl
        self.state.equity += total_pnl
        self.state.cumulative_pnl += total_pnl
        self.state.vrp_pnl += cycle_vrp_pnl
        self.state.skew_pnl += cycle_skew_pnl
        self.state.n_cycles += 1

        # -- Drawdown tracking --
        if self.state.equity > self.state.hwm_equity:
            self.state.hwm_equity = self.state.equity
        if self.state.hwm_equity > 0:
            dd = (self.state.equity - self.state.hwm_equity) / self.state.hwm_equity
            if dd < self.state.max_drawdown:
                self.state.max_drawdown = dd

        # -- History entry --
        entry = {
            "epoch": epoch,
            "timestamp": now,
            "equity": round(self.state.equity, 2),
            "cycle_pnl": round(total_pnl, 2),
            "vrp_pnl": round(cycle_vrp_pnl, 2),
            "skew_pnl": round(cycle_skew_pnl, 2),
            "gross_leverage": signal.get("gross_leverage", 0.0),
            "confidence": signal.get("confidence", "?"),
            "symbols": sym_details,
        }
        self.state.pnl_history.append(entry)
        self.state.updated_at = now

        # Persist
        self.state.save()
        self._log_event(entry)

        return entry

    def status_report(self) -> str:
        """Human-readable status report."""
        s = self.state
        lines = []
        lines.append("=" * 60)
        lines.append("CRYPTO OPTIONS — PAPER TRADING STATUS")
        lines.append("=" * 60)
        lines.append("")
        lines.append(f"  Equity:       ${s.equity:>12,.2f}")
        lines.append(f"  HWM:          ${s.hwm_equity:>12,.2f}")
        lines.append(f"  Total P&L:    ${s.cumulative_pnl:>+12,.2f}")
        lines.append(f"    VRP P&L:    ${s.vrp_pnl:>+12,.2f}")
        lines.append(f"    Skew P&L:   ${s.skew_pnl:>+12,.2f}")
        lines.append(f"  Max Drawdown: {s.max_drawdown:>12.2%}")
        lines.append(f"  Cycles:       {s.n_cycles:>12,}")
        lines.append(f"  Created:      {s.created_at}")
        lines.append(f"  Updated:      {s.updated_at}")
        lines.append("")

        if s.positions:
            lines.append("POSITIONS:")
            for sym, pos in sorted(s.positions.items()):
                lines.append(
                    f"  {sym:6s}  net={pos.weight:+.4f}  "
                    f"vrp={pos.vrp_weight:+.4f}  skew={pos.skew_weight:+.4f}  "
                    f"entry_iv={pos.entry_iv or 0:.2f}"
                )
        else:
            lines.append("POSITIONS: none (flat)")

        # Recent P&L
        if s.pnl_history:
            lines.append("")
            lines.append("RECENT P&L (last 5 cycles):")
            for entry in s.pnl_history[-5:]:
                ts = entry.get("timestamp", "?")[:19]
                pnl = entry.get("cycle_pnl", 0)
                eq = entry.get("equity", 0)
                lines.append(f"  {ts}  P&L=${pnl:>+8,.2f}  equity=${eq:>12,.2f}")

        lines.append("")
        lines.append("=" * 60)
        return "\n".join(lines)

    def reset(self, initial_equity: float = 100_000.0) -> None:
        """Reset paper state (wipes history)."""
        self.state = OptionsPaperState(
            equity=initial_equity,
            hwm_equity=initial_equity,
            created_at=_now_iso(),
        )
        self.state.save()
        logger.info("Paper state reset to $%,.2f", initial_equity)

    def _log_event(self, entry: Dict[str, Any]) -> None:
        """Append to P&L JSONL log."""
        PNL_LOG.parent.mkdir(parents=True, exist_ok=True)
        try:
            with open(PNL_LOG, "a") as f:
                f.write(json.dumps(entry, default=str) + "\n")
        except Exception:
            pass


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# ── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Options Paper State Tracker")
    parser.add_argument("--status", action="store_true", help="Show paper trading status")
    parser.add_argument("--reset", action="store_true", help="Reset paper state")
    parser.add_argument("--equity", type=float, default=100_000.0,
                        help="Initial equity for reset")
    args = parser.parse_args()

    if args.reset:
        tracker = OptionsPaperTracker(initial_equity=args.equity)
        tracker.reset(args.equity)
        print(f"Paper state reset to ${args.equity:,.2f}")
    else:
        tracker = OptionsPaperTracker()
        print(tracker.status_report())
