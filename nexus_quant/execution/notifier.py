"""
Trade Notifier — Telegram alerts for live trading events.

Reads NEXUS_TELEGRAM_TOKEN and NEXUS_TELEGRAM_CHAT_ID from env.
If not set, all calls are no-ops (silent fallback).

Events notified:
  - Engine start / stop
  - Each rebalance cycle (orders, turnover, leverage)
  - Risk warnings
  - Risk HALT (critical)
  - Emergency close
"""
from __future__ import annotations

import json
import logging
import os
import urllib.request
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

log = logging.getLogger("nexus.notifier")


class TradeNotifier:
    """Sends formatted Telegram alerts for live trading events."""

    def __init__(self) -> None:
        self.token = os.environ.get("NEXUS_TELEGRAM_TOKEN", "")
        self.chat_id = os.environ.get("NEXUS_TELEGRAM_CHAT_ID", "")
        self.enabled = bool(self.token and self.chat_id)
        if not self.enabled:
            log.info("Telegram notifier disabled (no token/chat_id)")

    def _send(self, text: str) -> bool:
        if not self.enabled:
            return False
        try:
            url = f"https://api.telegram.org/bot{self.token}/sendMessage"
            payload = json.dumps({
                "chat_id": self.chat_id,
                "text": text[:4096],
                "parse_mode": "HTML",
                "disable_web_page_preview": True,
            }).encode("utf-8")
            req = urllib.request.Request(
                url, data=payload,
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=10) as resp:
                return resp.status == 200
        except Exception as e:
            log.warning("Telegram send failed: %s", e)
            return False

    # ------------------------------------------------------------------
    # Event methods
    # ------------------------------------------------------------------

    def engine_start(self, mode: str, config_name: str, symbols: List[str]) -> None:
        self._send(
            f"<b>NEXUS Live Engine Started</b>\n"
            f"Mode: <code>{mode}</code>\n"
            f"Config: <code>{config_name}</code>\n"
            f"Symbols: {len(symbols)} coins\n"
            f"Time: {_now_short()}"
        )

    def engine_stop(self, cycles: int) -> None:
        self._send(
            f"<b>NEXUS Live Engine Stopped</b>\n"
            f"Total cycles: {cycles}\n"
            f"Time: {_now_short()}"
        )

    def cycle_complete(self, report: Dict[str, Any]) -> None:
        """Send summary after each rebalance cycle."""
        status = report.get("status", "?")
        sig = report.get("signal", {})
        rebal = report.get("rebalance", {})
        balance = report.get("balance_usd", 0)
        elapsed = report.get("elapsed_seconds", 0)

        icon = {"ok": "OK", "halted": "HALTED", "signal_error": "SIG ERR",
                "risk_error": "RISK ERR", "execution_error": "EXEC ERR"}.get(status, status.upper())

        lines = [
            f"<b>Cycle [{icon}]</b> @ {_now_short()}",
            f"Balance: <b>${balance:,.0f}</b>",
        ]

        if sig:
            tilt = "ON" if sig.get("vol_tilt_active") else "OFF"
            pvol = "ON" if sig.get("price_vol_active") else "OFF"
            lines.append(
                f"Gross: {sig.get('gross_leverage', 0):.3f} | "
                f"Net: {sig.get('net_exposure', 0):.3f} | "
                f"Tilt: {tilt} (z={sig.get('vol_tilt_z', 0):.2f})"
            )
            lines.append(
                f"PriceVol: {pvol} (vol={sig.get('price_vol', 0):.2f})"
            )

        if rebal:
            lines.append(
                f"Orders: {rebal.get('orders_executed', 0)}/{rebal.get('orders_planned', 0)} | "
                f"Turnover: ${rebal.get('turnover_usd', 0):,.0f}"
            )
            if rebal.get("orders_failed", 0) > 0:
                lines.append(f"Failed: {rebal['orders_failed']} orders")

        if report.get("error"):
            lines.append(f"Error: <code>{str(report['error'])[:200]}</code>")

        lines.append(f"({elapsed:.1f}s)")
        self._send("\n".join(lines))

    def risk_warning(self, warnings: List[str]) -> None:
        if not warnings:
            return
        msg = "<b>Risk Warning</b>\n" + "\n".join(f"- {w}" for w in warnings)
        self._send(msg)

    def risk_halt(self, reason: str) -> None:
        self._send(
            f"<b>TRADING HALTED</b>\n"
            f"Reason: {reason}\n"
            f"Time: {_now_short()}\n\n"
            f"Manual reset required to resume."
        )

    def emergency_close(self, results: List[Dict[str, Any]]) -> None:
        lines = [f"<b>EMERGENCY CLOSE ALL</b>"]
        for r in results:
            lines.append(
                f"  {r.get('side', '?')} {r.get('symbol', '?')} "
                f"qty={r.get('quantity', 0):.6f} → {r.get('status', '?')}"
            )
        self._send("\n".join(lines))


def _now_short() -> str:
    return datetime.now(timezone.utc).strftime("%H:%M UTC %b %d")
