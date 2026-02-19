"""
NEXUS Notifier - Send alerts and messages to humans.
Currently supports: console log, file log, Telegram (optional).
"""
from __future__ import annotations
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class NexusNotifier:
    """Multi-channel notifier for NEXUS alerts."""

    def __init__(self, artifacts_dir: Path) -> None:
        self.log_path = artifacts_dir / "brain" / "notifications.jsonl"
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._telegram_token = os.environ.get("NEXUS_TELEGRAM_TOKEN")
        self._telegram_chat_id = os.environ.get("NEXUS_TELEGRAM_CHAT_ID")

    def notify(self, message: str, level: str = "info", channel: str = "all") -> None:
        """Send notification to configured channels."""
        event = {"ts": _utc_now(), "level": level, "message": message}
        # Always log to file
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(event) + "\n")
        # Print to console
        prefix = {"info": "[i]", "warning": "[!]", "critical": "[CRIT]", "success": "[OK]"}.get(level, "[*]")
        print(f"[NEXUS] {prefix} {message}")
        # Telegram if configured
        if self._telegram_token and self._telegram_chat_id and channel in ("all", "telegram"):
            self._send_telegram(f"{prefix} {message}")

    def _send_telegram(self, text: str) -> None:
        try:
            import urllib.request
            url = f"https://api.telegram.org/bot{self._telegram_token}/sendMessage"
            data = json.dumps({"chat_id": self._telegram_chat_id, "text": text, "parse_mode": "Markdown"}).encode()
            req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
            urllib.request.urlopen(req, timeout=5)
        except Exception as e:
            print(f"[NEXUS Telegram] Error: {e}")

    def recent_notifications(self, limit: int = 20):
        if not self.log_path.exists():
            return []
        lines = self.log_path.read_text("utf-8").splitlines()[-limit:]
        out = []
        for ln in lines:
            try:
                out.append(json.loads(ln))
            except Exception:
                continue
        return list(reversed(out))
