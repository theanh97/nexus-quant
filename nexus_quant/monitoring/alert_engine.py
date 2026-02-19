from __future__ import annotations

"""
NEXUS Alert Engine - Threshold-based automated alerting.

Monitors:
- Funding rate extremes (carry opportunities / risk)
- Sharpe degradation
- Drawdown thresholds
- Market regime changes
- Memory/task anomalies

Sends alerts via NexusNotifier (console + file + Telegram).
Saves alerts to artifacts/monitoring/alerts.jsonl.
"""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# ── Thresholds (configurable) ───────────────────────────────────────────────

FUNDING_RATE_HIGH = 0.001      # >0.1% per 8h = HIGH positive (pay funding)
FUNDING_RATE_EXTREME = 0.003   # >0.3% per 8h = EXTREME (deleverage signal)
FUNDING_RATE_NEGATIVE = -0.001 # <-0.1% = long receive funding (carry opportunity)
SHARPE_DEGRADATION_THRESH = 0.5  # Sharpe drops by more than 0.5 → alert
DRAWDOWN_WARN_THRESH = 0.05    # 5% drawdown → warning
DRAWDOWN_CRIT_THRESH = 0.10    # 10% drawdown → critical


class AlertEngine:
    """
    Monitors NEXUS system state and emits structured alerts.
    Designed to run after each research cycle or on-demand.
    """

    def __init__(self, artifacts_dir: Path, notifier=None):
        self.artifacts_dir = Path(artifacts_dir)
        self._alerts_path = self.artifacts_dir / "monitoring" / "alerts.jsonl"
        self._alerts_path.parent.mkdir(parents=True, exist_ok=True)
        self._notifier = notifier

    def _emit(self, level: str, category: str, message: str, data: Dict[str, Any]) -> Dict[str, Any]:
        alert = {
            "ts": _now_iso(),
            "level": level,  # info | warning | critical
            "category": category,
            "message": message,
            "data": data,
        }
        with open(self._alerts_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(alert, ensure_ascii=False) + "\n")
        if self._notifier:
            try:
                self._notifier.notify(f"[{category}] {message}", level=level)
            except Exception:
                pass
        return alert

    def check_funding_rates(self, market_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Check funding rate extremes for carry signals."""
        alerts: List[Dict[str, Any]] = []
        for sym, info in market_data.items():
            if sym == "top_tickers":
                continue
            rate = float(info.get("latest_rate") or info.get("rate") or 0)
            rate_pct = rate * 100

            if abs(rate) >= FUNDING_RATE_EXTREME:
                level = "critical"
                msg = f"{sym} funding EXTREME: {rate_pct:+.4f}% per 8h ({rate_pct*3*365:.1f}% APR). {'DELEVERAGE longs' if rate > 0 else 'DELEVERAGE shorts'}."
            elif rate >= FUNDING_RATE_HIGH:
                level = "warning"
                msg = f"{sym} funding HIGH: {rate_pct:+.4f}%. Short perp carry opportunity."
            elif rate <= FUNDING_RATE_NEGATIVE:
                level = "info"
                msg = f"{sym} funding NEGATIVE: {rate_pct:+.4f}%. Long perp carry opportunity."
            else:
                continue

            alerts.append(self._emit(level, "funding_rate", msg, {"symbol": sym, "rate": rate, "rate_pct": rate_pct}))
        return alerts

    def check_sharpe(self, current_sharpe: Optional[float], baseline_sharpe: Optional[float]) -> Optional[Dict[str, Any]]:
        """Alert if Sharpe degraded significantly from baseline."""
        if current_sharpe is None or baseline_sharpe is None:
            return None
        degradation = baseline_sharpe - current_sharpe
        if degradation >= SHARPE_DEGRADATION_THRESH:
            level = "critical" if degradation >= SHARPE_DEGRADATION_THRESH * 2 else "warning"
            msg = f"Sharpe degraded {degradation:.2f} points (baseline: {baseline_sharpe:.2f} → current: {current_sharpe:.2f})"
            return self._emit(level, "sharpe_degradation", msg, {
                "current_sharpe": current_sharpe,
                "baseline_sharpe": baseline_sharpe,
                "degradation": degradation,
            })
        return None

    def check_drawdown(self, max_drawdown: Optional[float]) -> Optional[Dict[str, Any]]:
        """Alert on significant drawdown."""
        if max_drawdown is None:
            return None
        dd = abs(max_drawdown)
        if dd >= DRAWDOWN_CRIT_THRESH:
            msg = f"CRITICAL drawdown: {dd:.1%}. Risk limits approaching."
            return self._emit("critical", "drawdown", msg, {"max_drawdown": max_drawdown})
        if dd >= DRAWDOWN_WARN_THRESH:
            msg = f"Drawdown warning: {dd:.1%}."
            return self._emit("warning", "drawdown", msg, {"max_drawdown": max_drawdown})
        return None

    def check_task_overload(self) -> Optional[Dict[str, Any]]:
        """Alert if too many critical tasks are piling up."""
        try:
            from ..tasks.manager import TaskManager
            tm = TaskManager(self.artifacts_dir)
            summary = tm.kanban_summary()
            critical = summary.get("critical", 0)
            in_progress = summary.get("in_progress", 0)
            if critical >= 3:
                msg = f"Task board overloaded: {critical} critical tasks unresolved, {in_progress} in progress."
                return self._emit("warning", "task_overload", msg, summary)
        except Exception:
            pass
        return None

    def run_full_check(self, market_data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Run all alert checks and return summary."""
        all_alerts: List[Dict[str, Any]] = []

        # Funding rate checks
        if market_data:
            all_alerts.extend(self.check_funding_rates(market_data))

        # Latest metrics checks
        try:
            runs_dir = self.artifacts_dir / "runs"
            if runs_dir.exists():
                dirs = sorted(runs_dir.iterdir(), key=lambda d: d.stat().st_mtime, reverse=True)
                for d in dirs[:1]:
                    mp = d / "metrics.json"
                    if mp.exists():
                        m = json.loads(mp.read_text("utf-8"))
                        s = m.get("summary") or {}
                        sharpe = s.get("sharpe")
                        mdd = s.get("max_drawdown")
                        dd_alert = self.check_drawdown(mdd)
                        if dd_alert:
                            all_alerts.append(dd_alert)
        except Exception:
            pass

        # Task overload
        task_alert = self.check_task_overload()
        if task_alert:
            all_alerts.append(task_alert)

        return {
            "ts": _now_iso(),
            "total_alerts": len(all_alerts),
            "critical": sum(1 for a in all_alerts if a["level"] == "critical"),
            "warnings": sum(1 for a in all_alerts if a["level"] == "warning"),
            "alerts": all_alerts,
        }

    def recent_alerts(self, limit: int = 30) -> List[Dict[str, Any]]:
        """Return most recent alerts from file."""
        if not self._alerts_path.exists():
            return []
        lines = self._alerts_path.read_text("utf-8").splitlines()[-limit:]
        out = []
        for ln in lines:
            try:
                out.append(json.loads(ln))
            except Exception:
                continue
        return list(reversed(out))
