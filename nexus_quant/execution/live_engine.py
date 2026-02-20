"""
Live Trading Engine — main orchestration loop for NEXUS P91b Champion.

Ties together:
  1. SignalGenerator  (nexus_quant.live.signal_generator) — strategy target weights
  2. BinanceFuturesClient  (execution.binance_client) — authenticated Binance API
  3. PositionManager  (execution.position_manager) — order generation + execution
  4. RiskGate  (execution.risk_gate) — circuit breakers

Runs on an hourly schedule, aligned to the top of each UTC hour.

Usage:
  # CLI
  python3 -m nexus_quant live --config configs/production_p91b_champion.json
  python3 -m nexus_quant live --config configs/production_p91b_champion.json --dry-run
  python3 -m nexus_quant live --config configs/production_p91b_champion.json --testnet

  # Programmatic
  engine = LiveEngine.from_config("configs/production_p91b_champion.json")
  engine.run_loop()
"""
from __future__ import annotations

import json
import logging
import signal
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from .binance_client import BinanceFuturesClient
from .position_manager import PositionManager, RebalanceResult
from .risk_gate import RiskGate, RiskReport

log = logging.getLogger("nexus.live")

PROJ_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJ_ROOT / "configs" / "production_p91b_champion.json"
STATE_DIR = PROJ_ROOT / "artifacts" / "execution"
ENGINE_LOG = STATE_DIR / "engine_log.jsonl"


class LiveEngine:
    """
    Main live trading loop.

    Each cycle:
      1. Generate target weights from P91b champion strategy
      2. Run risk gate checks
      3. If can_trade → execute rebalance
      4. Update equity tracking
      5. Log everything
      6. Sleep until next hour
    """

    def __init__(
        self,
        config: Dict[str, Any],
        client: BinanceFuturesClient,
        dry_run: bool = False,
    ) -> None:
        self.config = config
        self.client = client
        self.dry_run = dry_run
        self._running = False

        # Extract config sections
        symbols = config["data"]["symbols"]
        risk_cfg = config.get("risk", {})
        monitoring = config.get("monitoring", {})
        operational = config.get("operational", {})

        # Position manager
        self.pos_manager = PositionManager(
            client=client,
            symbols=symbols,
            max_gross_leverage=risk_cfg.get("max_gross_leverage", 0.70),
            max_position_pct=risk_cfg.get("max_position_per_symbol", 0.30),
            max_turnover_pct=risk_cfg.get("max_turnover_per_rebalance", 0.80),
            min_order_usd=10.0,
            log_dir=STATE_DIR,
        )

        # Risk gate
        self.risk_gate = RiskGate(
            halt_conditions=monitoring.get("halt_conditions"),
            warning_conditions=monitoring.get("warning_conditions"),
            state_dir=STATE_DIR,
        )

        # Timing
        self.interval_seconds = operational.get("data_refresh_interval_minutes", 60) * 60
        self.max_retries = operational.get("max_retries_on_api_error", 3)
        self.retry_delay = operational.get("retry_delay_seconds", 5)

        # Signal generator (lazy init — imports strategy modules)
        self._signal_gen = None

        # Mock balance for dry-run without API keys
        self._mock_balance = 100000.0

        # State
        STATE_DIR.mkdir(parents=True, exist_ok=True)

    @classmethod
    def from_config(
        cls,
        config_path: str,
        dry_run: bool = False,
        testnet: bool = False,
    ) -> "LiveEngine":
        """Create engine from production config file."""
        with open(config_path) as f:
            config = json.load(f)

        client = BinanceFuturesClient(testnet=testnet)
        return cls(config=config, client=client, dry_run=dry_run)

    @property
    def signal_gen(self):
        """Lazy-init signal generator (heavy import)."""
        if self._signal_gen is None:
            from ..live.signal_generator import SignalGenerator
            ens = self.config["ensemble"]
            vol = self.config.get("volume_tilt", {})
            self._signal_gen = SignalGenerator(
                symbols=self.config["data"]["symbols"],
                ensemble_weights=ens["weights"],
                signal_configs=ens["signals"],
                vol_tilt_lookback=vol.get("lookback_bars", 168),
                vol_tilt_ratio=vol.get("tilt_ratio", 0.65),
                vol_tilt_enabled=vol.get("enabled", True),
                warmup_bars=self.config.get("operational", {}).get(
                    "warmup_bars_required", 600
                ),
            )
        return self._signal_gen

    # ------------------------------------------------------------------
    # Pre-flight checks
    # ------------------------------------------------------------------

    def preflight(self) -> Dict[str, Any]:
        """
        Run pre-flight checks before starting live trading.
        Returns dict with status and any issues found.
        """
        issues: List[str] = []
        warnings: List[str] = []

        has_keys = bool(self.client._api_key and self.client._api_secret)

        if has_keys:
            # 1. Check API connectivity
            try:
                if self.client.ping():
                    log.info("Binance API: OK")
                else:
                    issues.append("Binance API: ping failed (check keys)")
            except Exception as e:
                issues.append(f"Binance API: {e}")

            # 2. Check balance
            try:
                balance = self.client.get_balance()
                if balance < 10:
                    issues.append(f"Balance too low: ${balance:.2f} (min $10)")
                elif balance < 1000:
                    warnings.append(f"Low balance: ${balance:.2f} (recommended $10K+)")
                else:
                    log.info("Balance: $%.2f", balance)
            except Exception as e:
                issues.append(f"Balance check failed: {e}")

            # 3. Check leverage settings
            try:
                for sym in self.config["data"]["symbols"][:2]:
                    self.client.set_margin_type(sym, "CROSSED")
                    self.client.set_leverage(sym, 1)
                log.info("Leverage/margin: OK (1x CROSSED)")
            except Exception as e:
                warnings.append(f"Leverage setup: {e}")
        else:
            if self.dry_run:
                warnings.append("No API keys — dry-run will use mock balance ($100K)")
                log.info("No API keys set — using mock balance for dry-run")
            else:
                issues.append("BINANCE_API_KEY / BINANCE_API_SECRET not set")

        # 4. Check risk gate state
        if self.risk_gate.is_halted():
            issues.append("Risk gate is HALTED — call reset_halt() first")

        # 5. Dry run mode
        if self.dry_run:
            warnings.append("DRY RUN mode — no real orders will be placed")

        status = "PASS" if not issues else "FAIL"
        result = {
            "status": status,
            "issues": issues,
            "warnings": warnings,
            "balance": balance if "balance" in dir() else 0,
            "dry_run": self.dry_run,
            "testnet": self.client._base_url != "https://fapi.binance.com",
        }

        if issues:
            log.error("Pre-flight FAILED: %s", "; ".join(issues))
        else:
            log.info("Pre-flight PASSED (%d warnings)", len(warnings))

        return result

    # ------------------------------------------------------------------
    # Single cycle
    # ------------------------------------------------------------------

    def run_cycle(self) -> Dict[str, Any]:
        """
        Execute one full trading cycle:
          1. Generate signal (target weights)
          2. Risk check
          3. Execute rebalance (or dry-run)
          4. Update equity tracking
          5. Return cycle report
        """
        ts = datetime.now(timezone.utc)
        cycle_start = time.time()
        report: Dict[str, Any] = {
            "timestamp": ts.isoformat(),
            "status": "unknown",
        }

        # Step 1: Generate signal
        log.info("Generating signal...")
        try:
            sig = self.signal_gen.generate()
            target_weights = sig.target_weights
            report["signal"] = {
                "vol_tilt_active": sig.vol_tilt_active,
                "vol_tilt_z": round(sig.vol_tilt_z_score, 4),
                "gross_leverage": round(sig.gross_leverage, 4),
                "net_exposure": round(sig.net_exposure, 4),
                "trades_needed": len(sig.trades_needed),
                "data_bars": sig.meta.get("data_bars"),
            }
            log.info(
                "Signal: gross=%.4f net=%.4f vol_tilt=%s trades=%d",
                sig.gross_leverage, sig.net_exposure,
                sig.vol_tilt_active, len(sig.trades_needed),
            )
        except Exception as e:
            report["status"] = "signal_error"
            report["error"] = str(e)
            log.error("Signal generation failed: %s", e)
            self._log_cycle(report)
            return report

        # Step 2: Get balance + risk check
        try:
            if self.dry_run and not self.client._api_key:
                # Dry-run without API keys: use mock balance
                balance = self._mock_balance
                log.info("Dry-run (no API key): using mock balance $%.2f", balance)
            else:
                balance = self.client.get_balance()
            report["balance_usd"] = round(balance, 2)

            risk_report = self.risk_gate.evaluate(balance)
            report["risk"] = {
                "can_trade": risk_report.can_trade,
                "halt_reason": risk_report.halt_reason,
                "warnings": risk_report.warnings,
                "checks": [
                    {"name": c.name, "value": c.value,
                     "threshold": c.threshold, "severity": c.severity}
                    for c in risk_report.checks
                ],
            }

            if not risk_report.can_trade:
                report["status"] = "halted"
                log.critical("HALTED: %s", risk_report.halt_reason)
                self._log_cycle(report)
                return report

            for w in risk_report.warnings:
                log.warning("Risk: %s", w)

        except Exception as e:
            report["status"] = "risk_error"
            report["error"] = str(e)
            log.error("Risk check failed: %s", e)
            self._log_cycle(report)
            return report

        # Step 3: Execute rebalance
        try:
            if self.dry_run and not self.client._api_key:
                # Dry-run without API keys: simulate rebalance from signal data
                rebal = self._mock_rebalance(target_weights, balance, sig)
            else:
                rebal = self.pos_manager.execute_rebalance(
                    target_weights=target_weights,
                    dry_run=self.dry_run,
                )
            report["rebalance"] = {
                "orders_planned": rebal.orders_planned,
                "orders_executed": rebal.orders_executed,
                "orders_failed": rebal.orders_failed,
                "turnover_usd": round(rebal.total_turnover_usd, 2),
                "fills": rebal.fills,
                "errors": rebal.errors,
                "skipped": rebal.skipped,
                "skip_reason": rebal.skip_reason,
            }

            if rebal.orders_failed > 0:
                log.warning(
                    "Rebalance: %d/%d orders failed",
                    rebal.orders_failed, rebal.orders_planned,
                )
            else:
                log.info(
                    "Rebalance: %d orders, $%.0f turnover%s",
                    rebal.orders_executed, rebal.total_turnover_usd,
                    " (DRY RUN)" if self.dry_run else "",
                )

        except Exception as e:
            report["status"] = "execution_error"
            report["error"] = str(e)
            log.error("Rebalance failed: %s", e)
            self._log_cycle(report)
            return report

        # Step 4: Update equity tracking
        try:
            if self.dry_run:
                post_balance = balance
            else:
                post_balance = self.client.get_balance()
            self.risk_gate.update_equity(post_balance)
            report["post_balance_usd"] = round(post_balance, 2)
        except Exception as e:
            log.warning("Equity update failed: %s", e)

        report["status"] = "ok"
        report["elapsed_seconds"] = round(time.time() - cycle_start, 1)

        self._log_cycle(report)
        return report

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def run_loop(self) -> None:
        """
        Run the live trading loop continuously.
        Aligns to the top of each UTC hour.
        """
        self._running = True

        # Handle graceful shutdown
        def _shutdown(signum, frame):
            log.info("Shutdown signal received — stopping after current cycle")
            self._running = False

        signal.signal(signal.SIGINT, _shutdown)
        signal.signal(signal.SIGTERM, _shutdown)

        mode = "DRY-RUN" if self.dry_run else "LIVE"
        log.info("=" * 60)
        log.info("NEXUS LIVE ENGINE — %s MODE", mode)
        log.info("Config: %s", self.config.get("run_name", "?"))
        log.info("Symbols: %s", ", ".join(self.config["data"]["symbols"]))
        log.info("Interval: %ds", self.interval_seconds)
        log.info("=" * 60)

        # Pre-flight
        pf = self.preflight()
        if pf["status"] == "FAIL" and not self.dry_run:
            log.error("Pre-flight failed — aborting")
            for issue in pf["issues"]:
                log.error("  - %s", issue)
            return

        cycle_num = 0
        while self._running:
            cycle_num += 1
            log.info("--- Cycle %d @ %s ---", cycle_num,
                     datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"))

            try:
                report = self.run_cycle()
                log.info(
                    "Cycle %d: status=%s orders=%s turnover=$%s",
                    cycle_num,
                    report.get("status"),
                    report.get("rebalance", {}).get("orders_executed", "?"),
                    report.get("rebalance", {}).get("turnover_usd", "?"),
                )
            except Exception as e:
                log.error("Cycle %d crashed: %s", cycle_num, e)
                log.error(traceback.format_exc())

            if not self._running:
                break

            # Sleep until next hour (align to :00)
            sleep_sec = self._seconds_until_next_hour()
            log.info("Sleeping %.0f seconds until next hour...", sleep_sec)
            self._interruptible_sleep(sleep_sec)

        log.info("Live engine stopped after %d cycles", cycle_num)

    def stop(self) -> None:
        """Graceful stop — finishes current cycle then exits."""
        self._running = False

    # ------------------------------------------------------------------
    # Emergency
    # ------------------------------------------------------------------

    def emergency_close_all(self) -> List[Dict[str, Any]]:
        """Emergency: close all positions immediately."""
        log.critical("EMERGENCY CLOSE ALL — closing all positions")
        results = []

        positions = self.client.get_positions(self.config["data"]["symbols"])
        for pos in positions:
            side = "SELL" if pos.quantity > 0 else "BUY"
            qty = abs(pos.quantity)
            try:
                if self.dry_run:
                    log.info("DRY RUN: would close %s %.6f %s", side, qty, pos.symbol)
                    results.append({
                        "symbol": pos.symbol, "side": side,
                        "quantity": qty, "status": "DRY_RUN",
                    })
                else:
                    order = self.client.place_order(
                        symbol=pos.symbol, side=side, quantity=qty,
                        order_type="MARKET", reduce_only=True,
                    )
                    results.append({
                        "symbol": order.symbol, "side": order.side,
                        "quantity": order.quantity, "status": order.status,
                    })
                    log.info("Closed %s: %s %.6f", pos.symbol, order.status, order.quantity)
            except Exception as e:
                log.error("Failed to close %s: %s", pos.symbol, e)
                results.append({
                    "symbol": pos.symbol, "side": side,
                    "quantity": qty, "status": f"ERROR: {e}",
                })

        return results

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _mock_rebalance(
        self, target_weights: Dict[str, float], balance: float, sig
    ) -> RebalanceResult:
        """Simulate rebalance without any Binance API calls (for dry-run without keys)."""
        ts = datetime.now(timezone.utc).isoformat()
        fills = []
        for s, w in target_weights.items():
            if abs(w) > 0.001:
                notional = abs(w) * balance
                price = sig.prices.get(s, 0)
                qty = notional / price if price > 0 else 0
                fills.append({
                    "symbol": s,
                    "side": "BUY" if w > 0 else "SELL",
                    "quantity": round(qty, 6),
                    "notional_usd": round(notional, 2),
                    "status": "DRY_RUN_MOCK",
                })

        turnover = sum(f["notional_usd"] for f in fills)
        return RebalanceResult(
            timestamp=ts,
            balance_usd=balance,
            target_weights=target_weights,
            current_weights={s: 0.0 for s in target_weights},
            orders_planned=len(fills),
            orders_executed=len(fills),
            orders_failed=0,
            total_turnover_usd=turnover,
            fills=fills,
        )

    def _seconds_until_next_hour(self) -> float:
        """Seconds until the top of the next UTC hour (+ 60s buffer)."""
        now = datetime.now(timezone.utc)
        # Next hour
        seconds_past = now.minute * 60 + now.second
        remaining = 3600 - seconds_past + 60  # 60s buffer for data settlement
        return max(remaining, 10)

    def _interruptible_sleep(self, total: float) -> None:
        """Sleep in small increments so we can respond to shutdown signals."""
        step = 5.0
        elapsed = 0.0
        while elapsed < total and self._running:
            time.sleep(min(step, total - elapsed))
            elapsed += step

    def _log_cycle(self, report: Dict[str, Any]) -> None:
        """Append cycle report to JSONL log."""
        try:
            ENGINE_LOG.parent.mkdir(parents=True, exist_ok=True)
            with open(ENGINE_LOG, "a") as f:
                f.write(json.dumps(report, default=str) + "\n")
        except Exception:
            pass

    def status(self) -> Dict[str, Any]:
        """Return current engine status summary."""
        halted = self.risk_gate.is_halted()
        try:
            balance = self.client.get_balance()
        except Exception:
            balance = 0.0

        try:
            positions = self.client.get_positions(self.config["data"]["symbols"])
            pos_summary = [
                {"symbol": p.symbol, "side": p.side, "qty": p.quantity,
                 "notional": p.notional, "pnl": p.unrealized_pnl}
                for p in positions
            ]
        except Exception:
            pos_summary = []

        return {
            "engine": "live" if not self.dry_run else "dry_run",
            "halted": halted,
            "balance_usd": round(balance, 2),
            "positions": pos_summary,
            "symbols": self.config["data"]["symbols"],
            "config": self.config.get("run_name", "?"),
            "interval_seconds": self.interval_seconds,
        }


# ------------------------------------------------------------------
# CLI entry point
# ------------------------------------------------------------------

def live_main(
    config_path: str,
    dry_run: bool = False,
    testnet: bool = False,
    once: bool = False,
) -> int:
    """Entry point for `nexus_quant live` CLI command."""
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        stream=sys.stdout,
    )

    mode = []
    if dry_run:
        mode.append("DRY-RUN")
    if testnet:
        mode.append("TESTNET")
    mode_str = " + ".join(mode) if mode else "LIVE"

    print(f"\n{'='*60}", flush=True)
    print(f"  NEXUS LIVE ENGINE — {mode_str}", flush=True)
    print(f"  Config: {config_path}", flush=True)
    print(f"{'='*60}\n", flush=True)

    try:
        engine = LiveEngine.from_config(
            config_path=config_path,
            dry_run=dry_run,
            testnet=testnet,
        )
    except Exception as e:
        print(f"[ERROR] Failed to initialize: {e}", flush=True)
        return 1

    if once:
        report = engine.run_cycle()
        print(json.dumps(report, indent=2, default=str), flush=True)
        return 0 if report.get("status") == "ok" else 1

    engine.run_loop()
    return 0
