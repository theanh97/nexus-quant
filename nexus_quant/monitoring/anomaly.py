from __future__ import annotations

import math
import statistics
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


class AnomalyKind:
    SLIPPAGE_SPIKE = "slippage_spike"
    DRAWDOWN_BREACH = "drawdown_breach"
    PERF_DRIFT = "perf_drift"
    FUNDING_ANOMALY = "funding_anomaly"
    REGIME_CHANGE = "regime_change"


@dataclass
class AnomalyEvent:
    kind: str
    severity: str  # "warn" | "critical"
    value: float
    threshold: float
    message: str
    ts: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def check_slippage_spike(
    actual_cost_rate: float,
    expected_cost_rate: float,
    spike_multiplier: float = 3.0,
    critical_multiplier: float = 6.0,
) -> Optional[AnomalyEvent]:
    if expected_cost_rate <= 0:
        return None
    ratio = actual_cost_rate / expected_cost_rate
    if ratio >= critical_multiplier:
        return AnomalyEvent(
            kind=AnomalyKind.SLIPPAGE_SPIKE,
            severity="critical",
            value=round(ratio, 4),
            threshold=critical_multiplier,
            message=f"Slippage {ratio:.1f}x expected (critical threshold {critical_multiplier}x)",
            ts=_utc_iso(),
        )
    if ratio >= spike_multiplier:
        return AnomalyEvent(
            kind=AnomalyKind.SLIPPAGE_SPIKE,
            severity="warn",
            value=round(ratio, 4),
            threshold=spike_multiplier,
            message=f"Slippage {ratio:.1f}x expected (warn threshold {spike_multiplier}x)",
            ts=_utc_iso(),
        )
    return None


def check_drawdown_breach(
    equity_curve: List[float],
    warn_threshold: float = 0.15,
    critical_threshold: float = 0.25,
) -> Optional[AnomalyEvent]:
    if len(equity_curve) < 2:
        return None
    peak = float("-inf")
    current_dd = 0.0
    for x in equity_curve:
        ex = float(x)
        if ex > peak:
            peak = ex
        if peak > 0:
            dd = 1.0 - (ex / peak)
            if dd > current_dd:
                current_dd = dd

    if current_dd >= critical_threshold:
        return AnomalyEvent(
            kind=AnomalyKind.DRAWDOWN_BREACH,
            severity="critical",
            value=round(current_dd, 6),
            threshold=critical_threshold,
            message=f"Drawdown {current_dd*100:.1f}% breached critical threshold {critical_threshold*100:.0f}%",
            ts=_utc_iso(),
        )
    if current_dd >= warn_threshold:
        return AnomalyEvent(
            kind=AnomalyKind.DRAWDOWN_BREACH,
            severity="warn",
            value=round(current_dd, 6),
            threshold=warn_threshold,
            message=f"Drawdown {current_dd*100:.1f}% breached warn threshold {warn_threshold*100:.0f}%",
            ts=_utc_iso(),
        )
    return None


def _sharpe_simple(returns: List[float]) -> float:
    if len(returns) < 3:
        return 0.0
    mu = statistics.mean(returns)
    sd = statistics.pstdev(returns)
    if sd == 0:
        return 0.0
    return mu / sd * math.sqrt(float(len(returns)))


def check_performance_drift(
    returns: List[float],
    short_window: int = 168,
    long_window: int = 720,
    degradation_threshold: float = 0.5,
    critical_degradation: float = 0.8,
) -> Optional[AnomalyEvent]:
    if len(returns) < long_window:
        return None
    short_ret = returns[-short_window:]
    long_ret = returns[-long_window:]
    sh_short = _sharpe_simple(short_ret)
    sh_long = _sharpe_simple(long_ret)
    if sh_long <= 0:
        return None
    ratio = sh_short / sh_long  # < 1 means degrading

    if ratio <= (1.0 - critical_degradation):
        return AnomalyEvent(
            kind=AnomalyKind.PERF_DRIFT,
            severity="critical",
            value=round(ratio, 4),
            threshold=1.0 - critical_degradation,
            message=f"Performance drift critical: short Sharpe {sh_short:.2f} vs long {sh_long:.2f} (ratio {ratio:.2f})",
            ts=_utc_iso(),
        )
    if ratio <= (1.0 - degradation_threshold):
        return AnomalyEvent(
            kind=AnomalyKind.PERF_DRIFT,
            severity="warn",
            value=round(ratio, 4),
            threshold=1.0 - degradation_threshold,
            message=f"Performance drift warn: short Sharpe {sh_short:.2f} vs long {sh_long:.2f} (ratio {ratio:.2f})",
            ts=_utc_iso(),
        )
    return None


def check_funding_anomaly(
    funding_rates: Dict[str, float],
    warn_threshold: float = 0.003,
    critical_threshold: float = 0.01,
) -> List[AnomalyEvent]:
    events: List[AnomalyEvent] = []
    for sym, rate in funding_rates.items():
        abs_rate = abs(float(rate))
        if abs_rate >= critical_threshold:
            events.append(AnomalyEvent(
                kind=AnomalyKind.FUNDING_ANOMALY,
                severity="critical",
                value=round(rate, 8),
                threshold=critical_threshold,
                message=f"Extreme funding rate {sym}: {rate*100:.3f}% (critical threshold {critical_threshold*100:.2f}%)",
                ts=_utc_iso(),
            ))
        elif abs_rate >= warn_threshold:
            events.append(AnomalyEvent(
                kind=AnomalyKind.FUNDING_ANOMALY,
                severity="warn",
                value=round(rate, 8),
                threshold=warn_threshold,
                message=f"High funding rate {sym}: {rate*100:.3f}% (warn threshold {warn_threshold*100:.2f}%)",
                ts=_utc_iso(),
            ))
    return events


def check_regime_change(
    returns: List[float],
    short_window: int = 24,
    long_window: int = 168,
    vol_spike_multiplier: float = 2.0,
) -> Optional[AnomalyEvent]:
    if len(returns) < long_window:
        return None
    short_ret = returns[-short_window:]
    long_ret = returns[-long_window:]
    if len(short_ret) < 2 or len(long_ret) < 2:
        return None
    vol_short = statistics.pstdev(short_ret)
    vol_long = statistics.pstdev(long_ret)
    if vol_long <= 0:
        return None
    ratio = vol_short / vol_long
    if ratio >= vol_spike_multiplier:
        severity = "critical" if ratio >= vol_spike_multiplier * 1.5 else "warn"
        return AnomalyEvent(
            kind=AnomalyKind.REGIME_CHANGE,
            severity=severity,
            value=round(ratio, 4),
            threshold=vol_spike_multiplier,
            message=f"Volatility spike: recent vol {vol_short:.4f} is {ratio:.1f}x long-term {vol_long:.4f}",
            ts=_utc_iso(),
        )
    return None


def run_all_checks(
    equity_curve: List[float],
    returns: List[float],
    funding_rates: Dict[str, float],
    actual_cost_rate: float,
    expected_cost_rate: float,
    thresholds: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    thresholds = thresholds or {}
    anomalies: List[AnomalyEvent] = []

    e = check_slippage_spike(
        actual_cost_rate=actual_cost_rate,
        expected_cost_rate=expected_cost_rate,
        spike_multiplier=float(thresholds.get("slippage_spike_multiplier", 3.0)),
        critical_multiplier=float(thresholds.get("slippage_critical_multiplier", 6.0)),
    )
    if e:
        anomalies.append(e)

    e = check_drawdown_breach(
        equity_curve=equity_curve,
        warn_threshold=float(thresholds.get("drawdown_warn", 0.15)),
        critical_threshold=float(thresholds.get("drawdown_critical", 0.25)),
    )
    if e:
        anomalies.append(e)

    e = check_performance_drift(
        returns=returns,
        short_window=int(thresholds.get("perf_short_window", 168)),
        long_window=int(thresholds.get("perf_long_window", 720)),
        degradation_threshold=float(thresholds.get("perf_degradation", 0.5)),
    )
    if e:
        anomalies.append(e)

    funding_events = check_funding_anomaly(
        funding_rates=funding_rates,
        warn_threshold=float(thresholds.get("funding_warn", 0.003)),
        critical_threshold=float(thresholds.get("funding_critical", 0.01)),
    )
    anomalies.extend(funding_events)

    e = check_regime_change(
        returns=returns,
        vol_spike_multiplier=float(thresholds.get("regime_vol_multiplier", 2.0)),
    )
    if e:
        anomalies.append(e)

    has_critical = any(a.severity == "critical" for a in anomalies)
    summary_parts = []
    if not anomalies:
        summary_parts.append("No anomalies detected.")
    else:
        critical_list = [a.kind for a in anomalies if a.severity == "critical"]
        warn_list = [a.kind for a in anomalies if a.severity == "warn"]
        if critical_list:
            summary_parts.append(f"CRITICAL: {', '.join(critical_list)}")
        if warn_list:
            summary_parts.append(f"WARN: {', '.join(warn_list)}")

    return {
        "anomalies": [a.to_dict() for a in anomalies],
        "count": len(anomalies),
        "has_critical": has_critical,
        "summary": " | ".join(summary_parts),
    }
