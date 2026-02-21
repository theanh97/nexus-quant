"""
NEXUS Unified Trading Runner
================================

Single entry point for multi-project, multi-venue trading.
Orchestrates the full pipeline:

  1. Generate signals from each project independently
  2. Aggregate via NexusSignalAggregator (portfolio-level weights)
  3. Apply PortfolioRiskOverlay (drawdown control, leverage caps)
  4. Dispatch to venue-specific executors (Binance, Deribit)

Modes:
  - PAPER: Signals only, track hypothetical P&L
  - DRY_RUN: Full pipeline, log intended orders (no real execution)
  - LIVE: Real execution on all venues (requires API keys)

Usage:
    python3 -m nexus_quant.live.nexus_runner --mode paper
    python3 -m nexus_quant.live.nexus_runner --mode dry_run
    python3 -m nexus_quant.live.nexus_runner --status
"""
from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger("nexus.runner")

PROJ_ROOT = Path(__file__).resolve().parents[2]
NEXUS_LOG = PROJ_ROOT / "artifacts" / "portfolio" / "nexus_runner_log.jsonl"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _log(msg: str) -> None:
    print(f"[NEXUS] {msg}", flush=True)


def run_cycle(mode: str = "paper") -> Dict[str, Any]:
    """
    Run one complete NEXUS trading cycle across all projects.

    Returns dict with cycle results.
    """
    from ..portfolio.aggregator import NexusSignalAggregator
    from ..portfolio.risk_overlay import PortfolioRiskOverlay

    cycle_start = time.time()
    result: Dict[str, Any] = {
        "mode": mode,
        "timestamp": _now_iso(),
        "signals": {},
        "allocation": None,
        "risk": None,
        "orders": {},
        "error": None,
    }

    # ── 1. Aggregate Signals ──
    _log(f"Mode: {mode.upper()} | Aggregating signals...")
    try:
        agg = NexusSignalAggregator()
        portfolio = agg.aggregate()

        result["allocation"] = {
            "weights": portfolio.project_allocations,
            "expected_sharpe": portfolio.expected_sharpe,
            "total_gross": portfolio.total_gross_leverage,
        }
        result["signals"] = {
            k: {
                "confidence": v.confidence,
                "age": v.age_seconds,
                "gross": v.gross_leverage,
            }
            for k, v in portfolio.signals.items()
        }

        _log(f"  Allocation: {portfolio.project_allocations}")
        _log(f"  Expected Sharpe: {portfolio.expected_sharpe:.3f}")
        _log(f"  Gross leverage: {portfolio.total_gross_leverage:.4f}")

        for venue, targets in portfolio.venue_targets.items():
            active = {s: w for s, w in targets.items() if abs(w) > 0.0001}
            if active:
                _log(f"  {venue}: {len(active)} active positions")
            else:
                _log(f"  {venue}: no active positions")

    except Exception as e:
        result["error"] = f"Aggregation failed: {e}"
        _log(f"  ERROR: {e}")
        _log_event(result)
        return result

    # ── 2. Paper Mode — track hypothetical P&L, no execution ──
    if mode == "paper":
        result["orders"] = {"mode": "paper", "note": "Signals aggregated, no execution"}

        # Update options paper P&L tracker
        try:
            from ..projects.crypto_options.paper_state import OptionsPaperTracker
            from ..projects.crypto_options.signal_generator import OptionsSignalGenerator

            opts_gen = OptionsSignalGenerator()
            opts_signal = opts_gen.generate()
            tracker = OptionsPaperTracker()
            paper_result = tracker.update(opts_signal)
            result["options_paper"] = {
                "equity": tracker.state.equity,
                "cycle_pnl": paper_result["cycle_pnl"],
                "cumulative_pnl": tracker.state.cumulative_pnl,
            }
            _log(f"  Options paper: equity=${tracker.state.equity:,.2f} "
                 f"cycle_pnl=${paper_result['cycle_pnl']:+,.2f}")
        except Exception as e:
            _log(f"  Options paper tracker: {e}")

        _log("  Paper mode -- signals logged, no orders placed")
        _log_event(result)
        return result

    # ── 3. Dispatch to Executors ──
    for venue, targets in portfolio.venue_targets.items():
        active = {s: w for s, w in targets.items() if abs(w) > 0.0001}
        if not active:
            continue

        try:
            if venue == "binance":
                orders = _execute_binance(active, mode)
                result["orders"]["binance"] = orders
            elif venue == "deribit":
                orders = _execute_deribit(active, mode)
                result["orders"]["deribit"] = orders
            else:
                _log(f"  Unknown venue: {venue}")
        except Exception as e:
            _log(f"  Execution error ({venue}): {e}")
            result["orders"][venue] = {"error": str(e)}

    elapsed = time.time() - cycle_start
    result["elapsed_seconds"] = round(elapsed, 1)
    _log(f"  Cycle complete in {elapsed:.1f}s")
    _log_event(result)
    return result


def _execute_binance(targets: Dict[str, float], mode: str) -> List[Dict]:
    """Execute on Binance."""
    from .binance_executor import BinanceExecutor

    dry_run = mode in ("dry_run", "paper")
    testnet = mode == "testnet"

    executor = BinanceExecutor(
        testnet=testnet,
        dry_run=dry_run,
    )

    _log(f"  Binance: reconciling {len(targets)} positions ({mode})")
    orders = executor.reconcile(targets)
    order_dicts = [o.to_dict() for o in orders]
    for o in orders:
        _log(f"    {o.side} {o.symbol}: qty={o.quantity:.6f} status={o.status}")
    return order_dicts


def _execute_deribit(targets: Dict[str, float], mode: str) -> List[Dict]:
    """Execute on Deribit."""
    from .deribit_executor import DeribitExecutor

    dry_run = mode in ("dry_run", "paper")
    testnet = mode in ("testnet", "dry_run")

    executor = DeribitExecutor(
        testnet=testnet,
        dry_run=dry_run,
    )

    _log(f"  Deribit: reconciling {len(targets)} positions ({mode})")
    orders = executor.reconcile(targets)
    order_dicts = [o.to_dict() for o in orders]
    for o in orders:
        _log(f"    {o.direction} {o.instrument_name}: ${o.amount:,.0f} status={o.status}")
    return order_dicts


def _log_event(event: Dict[str, Any]) -> None:
    """Log cycle event."""
    NEXUS_LOG.parent.mkdir(parents=True, exist_ok=True)
    try:
        with open(NEXUS_LOG, "a") as f:
            f.write(json.dumps(event, default=str) + "\n")
    except Exception:
        pass


def status_report() -> str:
    """Full NEXUS status report."""
    from ..portfolio.aggregator import NexusSignalAggregator
    from ..portfolio.risk_overlay import PortfolioRiskOverlay
    from ..portfolio.optimizer import PortfolioOptimizer

    lines = []
    lines.append("=" * 70)
    lines.append("NEXUS UNIFIED TRADING PLATFORM — STATUS")
    lines.append("=" * 70)
    lines.append("")

    # Portfolio optimization
    opt = PortfolioOptimizer()
    results = opt.optimize(step=0.05, correlation_override=0.20)
    best = results[0]
    perps = next(s for s in opt.strategies if s.name == "crypto_perps")
    opts = next(s for s in opt.strategies if s.name == "crypto_options")

    lines.append("STRATEGY PROFILES:")
    lines.append(f"  crypto_perps:    avg Sharpe={perps.avg_sharpe:.3f} [{perps.status}]")
    lines.append(f"  crypto_options:  avg Sharpe={opts.avg_sharpe:.3f} [{opts.status}]")
    lines.append(f"  Portfolio (25/75): avg Sharpe={best.avg_sharpe:.3f} "
                 f"vol={best.annual_vol_pct:.1f}%")
    lines.append("")

    # Signal aggregator
    agg = NexusSignalAggregator(risk_overlay=False)  # skip risk overlay for status
    lines.append("SIGNAL STATUS:")
    for project in agg.portfolio_weights:
        sig = agg.read_latest_signal(project)
        if sig is None:
            lines.append(f"  {project:25s} NO SIGNAL")
        else:
            from ..portfolio.aggregator import MAX_SIGNAL_AGE
            max_age = MAX_SIGNAL_AGE.get(project, 86400)
            stale = "STALE" if sig.age_seconds > max_age else "OK"
            if sig.confidence == "insufficient_data":
                stale = "COLLECTING"
            lines.append(f"  {project:25s} [{stale:10s}] "
                         f"age={sig.age_seconds:>7,}s  "
                         f"confidence={sig.confidence:17s} "
                         f"gross={sig.gross_leverage:.4f}")
    lines.append("")

    # Risk overlay
    overlay = PortfolioRiskOverlay()
    lines.append("RISK OVERLAY:")
    lines.append(f"  HWM: ${overlay.state.hwm_equity:,.2f}")
    lines.append(f"  Current equity: ${overlay.state.current_equity:,.2f}")
    lines.append(f"  Drawdown: {overlay.state.drawdown_pct:.2%}")
    lines.append(f"  Scale factor: {overlay.state.scale_factor:.2f}")
    lines.append(f"  Emergency halt: {overlay.state.emergency_halt}")
    lines.append("")

    # Recent runner log
    if NEXUS_LOG.exists():
        try:
            last_line = None
            with open(NEXUS_LOG) as f:
                for line in f:
                    if line.strip():
                        last_line = line
            if last_line:
                last = json.loads(last_line)
                lines.append(f"LAST CYCLE: {last.get('timestamp', '?')}")
                lines.append(f"  Mode: {last.get('mode', '?')}")
                if last.get("error"):
                    lines.append(f"  Error: {last['error']}")
                lines.append("")
        except Exception:
            pass

    lines.append("EXECUTION VENUES:")
    lines.append("  binance:  crypto_perps (USDM futures)")
    lines.append("  deribit:  crypto_options (options + perps delta hedge)")
    lines.append("")
    lines.append("=" * 70)

    return "\n".join(lines)


# ── CLI ──────────────────────────────────────────────────────────────────────

def main():
    import argparse
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    parser = argparse.ArgumentParser(description="NEXUS Unified Trading Runner")
    parser.add_argument("--mode", default="paper",
                        choices=["paper", "dry_run", "testnet", "live"],
                        help="Trading mode")
    parser.add_argument("--status", action="store_true", help="Show status")
    parser.add_argument("--loop", action="store_true", help="Run continuous loop")
    parser.add_argument("--interval", type=int, default=3600,
                        help="Loop interval in seconds")
    args = parser.parse_args()

    if args.status:
        print(status_report())
        return

    if args.loop:
        _log(f"Starting {args.mode.upper()} loop (interval={args.interval}s)")
        _log("=" * 60)
        cycle = 0
        while True:
            cycle += 1
            _log(f"\n--- Cycle {cycle} @ {_now_iso()} ---")
            try:
                run_cycle(mode=args.mode)
            except Exception as e:
                _log(f"  Cycle {cycle} EXCEPTION: {e}")
            _log(f"  Sleeping {args.interval}s...")
            time.sleep(args.interval)
    else:
        run_cycle(mode=args.mode)


if __name__ == "__main__":
    main()
