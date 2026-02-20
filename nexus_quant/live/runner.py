"""
Live Trading Runner — Orchestrates signal generation and execution.

Modes:
  - PAPER: Generate signals only, track hypothetical P&L (no real orders)
  - DRY_RUN: Generate signals + log what would be traded (no real orders)
  - TESTNET: Execute on Binance testnet (real API calls, fake money)
  - LIVE: Execute on Binance mainnet (REAL MONEY — use with extreme caution)

Usage:
    python3 -m nexus_quant trade --mode paper     # Paper trading (default)
    python3 -m nexus_quant trade --mode dry_run    # Log trades without executing
    python3 -m nexus_quant trade --mode testnet    # Binance testnet
    python3 -m nexus_quant trade --mode live       # REAL TRADING
"""
from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

PROJ_ROOT = Path(__file__).resolve().parents[2]
RUNNER_LOG = PROJ_ROOT / "artifacts" / "live" / "runner_log.jsonl"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _log(msg: str) -> None:
    print(f"[TRADE] {msg}", flush=True)


def _log_event(event: Dict[str, Any]) -> None:
    RUNNER_LOG.parent.mkdir(parents=True, exist_ok=True)
    try:
        with open(RUNNER_LOG, "a") as f:
            f.write(json.dumps(event, ensure_ascii=False) + "\n")
    except Exception:
        pass


def run_cycle(
    mode: str = "paper",
    config_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Run one trading cycle:
    1. Generate signal (fetch data + compute target weights)
    2. Check risk gates
    3. Execute trades (or log them in paper/dry_run mode)
    4. Log results
    """
    from .signal_generator import SignalGenerator
    from .binance_executor import BinanceExecutor

    cycle_start = time.time()
    result: Dict[str, Any] = {
        "mode": mode,
        "timestamp": _now_iso(),
        "signal": None,
        "risk_check": None,
        "orders": [],
        "error": None,
    }

    # 1. Generate signal
    _log(f"Mode: {mode.upper()} | Generating signal...")
    try:
        gen = SignalGenerator.from_production_config(config_path)
        signal = gen.generate()
        result["signal"] = {
            "gross_leverage": signal.gross_leverage,
            "net_exposure": signal.net_exposure,
            "vol_tilt_active": signal.vol_tilt_active,
            "vol_tilt_z": signal.vol_tilt_z_score,
            "n_trades": len(signal.trades_needed),
        }
        _log(f"  Signal OK: gross={signal.gross_leverage:.4f}, "
             f"tilt={'ON' if signal.vol_tilt_active else 'OFF'}, "
             f"trades={len(signal.trades_needed)}")
    except Exception as e:
        result["error"] = f"Signal generation failed: {e}"
        _log(f"  ERROR: {e}")
        _log_event(result)
        return result

    # 2. Paper mode — just signal, no execution
    if mode == "paper":
        result["orders"] = [{"mode": "paper", "note": "Signal logged, no execution"}]
        _log("  Paper mode — signal logged, no orders placed")
        _log_event(result)
        return result

    # 3. Initialize executor
    executor = BinanceExecutor(
        testnet=(mode == "testnet"),
        dry_run=(mode == "dry_run"),
    )

    # 4. Check risk gates (if we have API access)
    if mode in ("testnet", "live") and executor.api_key:
        try:
            account = executor.get_account()
            risk = executor.check_risk_gates(account)
            result["risk_check"] = risk
            if risk["halt"]:
                _log(f"  RISK HALT: {risk['reasons']}")
                result["error"] = f"Risk halt: {risk['reasons']}"
                _log_event(result)
                return result
            if risk["warnings"]:
                _log(f"  Risk warnings: {risk['warnings']}")
        except Exception as e:
            _log(f"  Risk check failed (continuing): {e}")

    # 5. Execute trades
    if not signal.trades_needed:
        _log("  No trades needed")
        _log_event(result)
        return result

    _log(f"  Executing {len(signal.trades_needed)} trades...")
    try:
        orders = executor.reconcile(signal.target_weights)
        result["orders"] = [o.to_dict() for o in orders]
        for o in orders:
            _log(f"    {o.side} {o.symbol}: qty={o.quantity:.6f}, "
                 f"price={o.price:.2f}, status={o.status}")
    except Exception as e:
        result["error"] = f"Execution failed: {e}"
        _log(f"  EXECUTION ERROR: {e}")

    elapsed = time.time() - cycle_start
    result["elapsed_seconds"] = round(elapsed, 1)
    _log(f"  Cycle complete in {elapsed:.1f}s")
    _log_event(result)
    return result


def run_loop(
    mode: str = "paper",
    config_path: Optional[str] = None,
    interval_seconds: int = 3600,
    max_cycles: int = 0,
) -> None:
    """Run continuous trading loop."""
    _log(f"Starting {mode.upper()} trading loop (interval={interval_seconds}s)")
    _log("=" * 60)

    cycle = 0
    while True:
        cycle += 1
        if max_cycles > 0 and cycle > max_cycles:
            _log(f"Max cycles ({max_cycles}) reached. Stopping.")
            break

        _log(f"\n--- Cycle {cycle} @ {_now_iso()} ---")
        try:
            result = run_cycle(mode=mode, config_path=config_path)
            if result.get("error"):
                _log(f"  Cycle {cycle} had error: {result['error']}")
        except Exception as e:
            _log(f"  Cycle {cycle} EXCEPTION: {e}")

        if max_cycles > 0 and cycle >= max_cycles:
            break

        _log(f"  Sleeping {interval_seconds}s until next cycle...")
        time.sleep(interval_seconds)
