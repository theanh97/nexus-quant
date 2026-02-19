from __future__ import annotations

"""
NEXUS Telegram Bot — direct human-AI interaction without terminal.

Setup (one-time):
1. Message @BotFather on Telegram → /newbot → get TOKEN
2. export NEXUS_TELEGRAM_TOKEN="<token>"
3. Get your chat_id: message the bot, then:
   curl https://api.telegram.org/bot<TOKEN>/getUpdates
4. export NEXUS_TELEGRAM_CHAT_ID="<your_chat_id>"
5. Run: python3 -m nexus_quant telegram --artifacts artifacts

Bot commands:
  /status   — current strategy performance
  /market   — live BTC/ETH/SOL funding rates
  /alerts   — recent alert feed
  /research — trigger research cycle
  /run      — run backtest
  /improve  — self-learning optimization
  /chat <msg> — talk to GLM-5 via NEXUS brain
  /help     — show all commands
"""

import json
import os
import time
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
import urllib.request


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class TelegramBot:
    """Long-polling Telegram bot. Pure stdlib — no external dependencies."""

    def __init__(self, artifacts_dir: Path, config_path: Optional[Path] = None):
        self.artifacts_dir = Path(artifacts_dir)
        self.config_path = config_path
        self.token = os.environ.get("NEXUS_TELEGRAM_TOKEN", "")
        self.chat_id = os.environ.get("NEXUS_TELEGRAM_CHAT_ID", "")
        self._offset = 0
        self._running = False
        self._log = self.artifacts_dir / "comms" / "telegram_log.jsonl"
        self._log.parent.mkdir(parents=True, exist_ok=True)

    # ── Telegram API helpers ────────────────────────────────────────────────

    def _api(self, method: str, data: Optional[Dict] = None, timeout: int = 30) -> Optional[Dict]:
        if not self.token:
            return None
        url = f"https://api.telegram.org/bot{self.token}/{method}"
        try:
            body = json.dumps(data or {}).encode("utf-8") if data else None
            req = urllib.request.Request(
                url, data=body,
                headers={"Content-Type": "application/json"} if body else {},
            )
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read())
        except Exception:
            return None

    def send(self, text: str, chat_id: Optional[str] = None, parse_mode: str = "HTML") -> bool:
        """Send a message. Returns True if successful."""
        cid = chat_id or self.chat_id
        if not cid or not self.token:
            return False
        result = self._api("sendMessage", {
            "chat_id": cid,
            "text": text[:4096],
            "parse_mode": parse_mode,
            "disable_web_page_preview": True,
        })
        try:
            with open(self._log, "a", encoding="utf-8") as f:
                f.write(json.dumps({"ts": _now_iso(), "dir": "out", "text": text[:200]}) + "\n")
        except Exception:
            pass
        return bool(result and result.get("ok"))

    def notify(self, text: str, level: str = "info") -> None:
        """Convenience for AlertEngine / Orion notifications."""
        icon = {"info": "ℹ️", "warning": "⚠️", "critical": "🚨"}.get(level, "📢")
        self.send(f"{icon} <b>NEXUS</b>\n{text}")

    def _get_updates(self) -> List[Dict]:
        result = self._api("getUpdates", {"offset": self._offset, "timeout": 25, "limit": 10}, timeout=35)
        if not result or not result.get("ok"):
            return []
        updates = result.get("result", [])
        if updates:
            self._offset = updates[-1]["update_id"] + 1
        return updates

    # ── Command dispatch ────────────────────────────────────────────────────

    def _handle(self, text: str, chat_id: str) -> None:
        text = (text or "").strip()
        parts = text.split(None, 1)
        cmd = (parts[0] or "").lower().lstrip("/").split("@")[0]
        args = parts[1] if len(parts) > 1 else ""

        handlers = {
            "start": self._cmd_help,
            "help": self._cmd_help,
            "status": self._cmd_status,
            "market": self._cmd_market,
            "alerts": self._cmd_alerts,
            "research": lambda c: threading.Thread(target=self._run_research, args=(c,), daemon=True).start(),
            "run": lambda c: threading.Thread(target=self._run_backtest, args=(c,), daemon=True).start(),
            "improve": lambda c: threading.Thread(target=self._run_improve, args=(c,), daemon=True).start(),
        }

        if cmd in handlers:
            handlers[cmd](chat_id)
        elif cmd == "chat":
            threading.Thread(target=self._run_chat, args=(chat_id, args or text), daemon=True).start()
        elif not text.startswith("/"):
            # Natural language — send to GLM-5
            threading.Thread(target=self._run_chat, args=(chat_id, text), daemon=True).start()
        else:
            self.send(f"Unknown command <code>{cmd}</code>. Try /help", chat_id=chat_id)

    # ── Command implementations ─────────────────────────────────────────────

    def _cmd_help(self, chat_id: str) -> None:
        self.send(
            "🤖 <b>NEXUS Quant AI</b>\n\n"
            "<b>Available commands:</b>\n"
            "/status — Current performance metrics\n"
            "/market — Live BTC/ETH/SOL funding rates\n"
            "/alerts — Recent alerts (drawdown, funding extremes)\n"
            "/research — Trigger autonomous research cycle\n"
            "/run — Run backtest with latest config\n"
            "/improve — Run self-learning optimization\n"
            "/chat &lt;msg&gt; — Chat with GLM-5 (NEXUS brain)\n"
            "/help — This message\n\n"
            "💡 <i>Or just type anything to chat naturally!</i>",
            chat_id=chat_id,
        )

    def _cmd_status(self, chat_id: str) -> None:
        try:
            runs_dir = self.artifacts_dir / "runs"
            if not runs_dir.exists():
                self.send("⚠️ No backtest results yet. Run /run to start.", chat_id=chat_id)
                return
            dirs = sorted(
                [d for d in runs_dir.iterdir() if d.is_dir()],
                key=lambda d: d.stat().st_mtime, reverse=True,
            )
            for d in dirs:
                mp = d / "metrics.json"
                if not mp.exists():
                    continue
                m = json.loads(mp.read_text("utf-8"))
                s = m.get("summary", {})
                sharpe = s.get("sharpe", 0)
                cagr = s.get("cagr", 0)
                mdd = s.get("max_drawdown", 0)
                wr = s.get("win_rate", 0)
                total_ret = s.get("total_return", 0)
                verdict = "✅ PASS" if sharpe > 1.0 else "⚠️ WARN" if sharpe > 0 else "❌ FAIL"
                self.send(
                    f"📊 <b>NEXUS Performance</b>\n"
                    f"<code>{d.name[:45]}</code>\n\n"
                    f"Sharpe:     <b>{sharpe:+.3f}</b> {verdict}\n"
                    f"CAGR:       <b>{cagr:+.2%}</b>\n"
                    f"Max DD:     <b>{mdd:.2%}</b>\n"
                    f"Win Rate:   <b>{wr:.2%}</b>\n"
                    f"Total Ret:  <b>{total_ret:+.2%}</b>",
                    chat_id=chat_id,
                )
                return
            self.send("No metrics found.", chat_id=chat_id)
        except Exception as e:
            self.send(f"❌ Status error: {e}", chat_id=chat_id)

    def _cmd_market(self, chat_id: str) -> None:
        try:
            from ..tools.web_research import fetch_binance_funding_rates
            lines = ["📡 <b>Live Funding Rates (Binance)</b>\n"]
            for sym in ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT"]:
                try:
                    rates = fetch_binance_funding_rates(sym, limit=1)
                    if rates:
                        rate = float(rates[-1].get("fundingRate", 0))
                        pct = rate * 100
                        icon = "🔴" if rate > 0.001 else "🟢" if rate < -0.001 else "⚪"
                        signal = "📉 SHORT carry" if rate > 0 else "📈 LONG carry"
                        ann = rate * 3 * 365 * 100
                        lines.append(f"{icon} {sym}: <b>{pct:+.4f}%</b> ({ann:+.1f}%/yr) → {signal}")
                except Exception:
                    continue
            self.send("\n".join(lines), chat_id=chat_id)
        except Exception as e:
            self.send(f"❌ Market error: {e}", chat_id=chat_id)

    def _cmd_alerts(self, chat_id: str) -> None:
        try:
            ap = self.artifacts_dir / "monitoring" / "alerts.jsonl"
            if not ap.exists():
                self.send("✅ No alerts on record.", chat_id=chat_id)
                return
            lines = ap.read_text("utf-8").splitlines()[-5:]
            alerts = []
            for ln in lines:
                try:
                    alerts.append(json.loads(ln))
                except Exception:
                    continue
            if not alerts:
                self.send("✅ No recent alerts.", chat_id=chat_id)
                return
            msg = "🚨 <b>Recent Alerts</b>\n\n"
            for a in reversed(alerts):
                icon = {"critical": "🔴", "warning": "🟡", "info": "🔵"}.get(a.get("level", ""), "⚪")
                ts = (a.get("ts") or "")[:16].replace("T", " ")
                msg += f"{icon} [{ts}] {a.get('message', '')[:80]}\n"
            self.send(msg, chat_id=chat_id)
        except Exception as e:
            self.send(f"❌ Alerts error: {e}", chat_id=chat_id)

    # ── Background task runners ─────────────────────────────────────────────

    def _run_research(self, chat_id: str) -> None:
        self.send("🔬 Research cycle starting...", chat_id=chat_id)
        try:
            from ..orchestration.research_cycle import ResearchCycle
            rc = ResearchCycle(self.artifacts_dir, self.config_path)
            report = rc.run()
            n_papers = len(report.get("arxiv_findings", []))
            n_tasks = len(report.get("tasks_created", []))
            n_proposals = report.get("strategy_proposals", 0)
            dur = round(report.get("duration_sec", 0), 1)
            self.send(
                f"✅ <b>Research Complete</b> ({dur}s)\n"
                f"📚 Papers found: {n_papers}\n"
                f"✅ Kanban tasks: {n_tasks}\n"
                f"🧪 Proposals: {n_proposals}",
                chat_id=chat_id,
            )
        except Exception as e:
            self.send(f"❌ Research failed: {e}", chat_id=chat_id)

    def _run_backtest(self, chat_id: str) -> None:
        self.send("⚙️ Backtest running...", chat_id=chat_id)
        try:
            from ..run import run_one
            cfg = str(self.config_path or Path("configs") / "run_synthetic_funding.json")
            result = run_one(cfg, artifacts_dir=str(self.artifacts_dir))
            s = (result.get("metrics") or {}).get("summary", {})
            sharpe = s.get("sharpe", 0)
            cagr = s.get("cagr", 0)
            icon = "✅" if sharpe > 1 else "⚠️" if sharpe > 0 else "❌"
            self.send(
                f"{icon} <b>Backtest Done</b>\n"
                f"Sharpe: <b>{sharpe:+.3f}</b>\n"
                f"CAGR: <b>{cagr:+.2%}</b>",
                chat_id=chat_id,
            )
        except Exception as e:
            self.send(f"❌ Backtest failed: {e}", chat_id=chat_id)

    def _run_improve(self, chat_id: str) -> None:
        self.send("🧬 Self-learning started (may take 5-10 min)...", chat_id=chat_id)
        try:
            from ..run import improve_one
            cfg = str(self.config_path or Path("configs") / "run_synthetic_funding.json")
            result = improve_one(cfg, artifacts_dir=str(self.artifacts_dir), trials=20)
            accepted = result.get("accepted", False)
            icon = "✅" if accepted else "⚠️"
            self.send(f"{icon} <b>Improve Done</b>\nNew params accepted: {accepted}", chat_id=chat_id)
        except Exception as e:
            self.send(f"❌ Improve failed: {e}", chat_id=chat_id)

    def _run_chat(self, chat_id: str, message: str) -> None:
        if not message:
            self.send("Please provide a message after /chat", chat_id=chat_id)
            return
        try:
            import anthropic
            api_key = os.environ.get("ZAI_API_KEY") or os.environ.get("ANTHROPIC_API_KEY", "")
            base_url = os.environ.get("ZAI_ANTHROPIC_BASE_URL")
            model = os.environ.get("ZAI_DEFAULT_MODEL", "glm-4-flash")
            if not api_key:
                self.send("⚠️ No AI API key configured.", chat_id=chat_id)
                return
            kwargs: Dict[str, Any] = {"api_key": api_key, "max_retries": 1}
            if base_url:
                kwargs["base_url"] = base_url.rstrip("/")
            client = anthropic.Anthropic(**kwargs)
            resp = client.messages.create(
                model=model,
                max_tokens=600,
                system=(
                    "You are NEXUS — an autonomous quant trading AI managing a portfolio "
                    "of crypto perpetual futures strategies. You are precise, data-driven, "
                    "and professional. Keep responses under 400 characters for Telegram."
                ),
                messages=[{"role": "user", "content": message}],
            )
            answer = str(resp.content[0].text) if resp.content else "No response."
            self.send(f"🤖 {answer}", chat_id=chat_id)
        except Exception as e:
            self.send(f"❌ Chat error: {e}", chat_id=chat_id)

    # ── Main loop ────────────────────────────────────────────────────────────

    def start_polling(self) -> None:
        """Start long-polling loop (blocking). Ctrl+C to stop."""
        if not self.token:
            print("[TelegramBot] ❌ NEXUS_TELEGRAM_TOKEN not set.")
            print("[TelegramBot] Get a token from @BotFather then:")
            print("[TelegramBot]   export NEXUS_TELEGRAM_TOKEN='<token>'")
            print("[TelegramBot]   export NEXUS_TELEGRAM_CHAT_ID='<your_id>'")
            return

        self._running = True
        print(f"[TelegramBot] ✅ Starting — chat_id={self.chat_id or '(all)'}")
        self.send("🚀 <b>NEXUS Online</b>\nAutonomous quant AI ready.\nType /help for commands.")

        while self._running:
            try:
                updates = self._get_updates()
                for upd in updates:
                    msg = upd.get("message") or {}
                    text = msg.get("text", "")
                    cid = str(msg.get("chat", {}).get("id", ""))
                    if text and cid:
                        try:
                            self._handle(text, cid)
                        except Exception:
                            pass
                    try:
                        with open(self._log, "a", encoding="utf-8") as f:
                            f.write(json.dumps({"ts": _now_iso(), "dir": "in", "text": text[:100], "chat_id": cid}) + "\n")
                    except Exception:
                        pass
            except KeyboardInterrupt:
                break
            except Exception:
                time.sleep(5)

    def stop(self) -> None:
        self._running = False
        self.send("🔴 <b>NEXUS Offline</b>")
