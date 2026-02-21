"""
NEXUS Cross-Project Signal Aggregator
=======================================

Unifies signals from multiple NEXUS projects into a single portfolio-level
allocation. Reads latest signals from each project, applies portfolio weights
from the optimizer, and outputs target positions per venue.

Projects:
  - crypto_perps: Binance USDM perpetual futures (V1+I460+I415+F144 ensemble)
  - crypto_options: Deribit options (hourly VRP 40% + daily Skew MR 60%)

Architecture:
  Each project produces its own signal independently (different frequencies,
  data sources, strategies). The aggregator:
  1. Reads the latest signal from each project's signal log
  2. Applies portfolio-level allocation weights (from PortfolioOptimizer)
  3. Scales each project's target weights by its portfolio allocation
  4. Outputs per-venue target weights ready for execution

Usage:
    from nexus_quant.portfolio.aggregator import NexusSignalAggregator
    agg = NexusSignalAggregator()
    portfolio = agg.aggregate()
    print(portfolio)
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger("nexus.aggregator")

PROJ_ROOT = Path(__file__).resolve().parents[2]

# Signal log paths for each project
SIGNAL_LOGS = {
    "crypto_perps": PROJ_ROOT / "artifacts" / "live" / "signals_log.jsonl",
    "crypto_options": PROJ_ROOT / "artifacts" / "crypto_options" / "signals_log.jsonl",
}

# Portfolio allocation output
PORTFOLIO_LOG = PROJ_ROOT / "artifacts" / "portfolio" / "portfolio_log.jsonl"


@dataclass
class ProjectSignal:
    """Latest signal from a single project."""
    project: str
    timestamp: str
    epoch: int
    target_weights: Dict[str, float]
    gross_leverage: float
    confidence: str
    age_seconds: int  # seconds since signal was generated
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PortfolioAllocation:
    """Unified portfolio allocation across all projects."""
    timestamp: str
    epoch: int

    # Per-project allocations
    project_allocations: Dict[str, float]  # {project_name: portfolio_weight}

    # Per-venue target weights (what to actually execute)
    venue_targets: Dict[str, Dict[str, float]]
    # e.g. {"binance": {"BTCUSDT": 0.05, "ETHUSDT": -0.03},
    #        "deribit": {"BTC": -0.15, "ETH": -0.10}}

    # Risk metrics
    total_gross_leverage: float
    total_net_exposure: float

    # Source signals
    signals: Dict[str, ProjectSignal]

    # Portfolio optimization context
    expected_sharpe: float
    expected_vol_pct: float
    correlation_assumption: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "epoch": self.epoch,
            "project_allocations": self.project_allocations,
            "venue_targets": self.venue_targets,
            "total_gross_leverage": round(self.total_gross_leverage, 4),
            "total_net_exposure": round(self.total_net_exposure, 4),
            "signals": {
                k: {
                    "project": v.project,
                    "timestamp": v.timestamp,
                    "target_weights": v.target_weights,
                    "gross_leverage": round(v.gross_leverage, 4),
                    "confidence": v.confidence,
                    "age_seconds": v.age_seconds,
                }
                for k, v in self.signals.items()
            },
            "expected_sharpe": round(self.expected_sharpe, 3),
            "expected_vol_pct": round(self.expected_vol_pct, 2),
            "correlation_assumption": self.correlation_assumption,
        }


# Venue mapping: which exchange handles which project
VENUE_MAP = {
    "crypto_perps": "binance",
    "crypto_options": "deribit",
}

# Maximum signal age before it's considered stale (seconds)
MAX_SIGNAL_AGE = {
    "crypto_perps": 7200,    # 2 hours (hourly signals)
    "crypto_options": 86400,  # 24 hours (daily signals)
}


class NexusSignalAggregator:
    """
    Aggregates signals from all NEXUS projects into portfolio-level targets.

    The aggregator does NOT re-run strategies — it reads the latest signals
    produced independently by each project's signal generator.
    """

    def __init__(
        self,
        portfolio_weights: Optional[Dict[str, float]] = None,
        signal_logs: Optional[Dict[str, Path]] = None,
        correlation: float = 0.20,
        risk_overlay: bool = True,
        initial_equity: float = 100000.0,
    ) -> None:
        self.signal_logs = signal_logs or dict(SIGNAL_LOGS)
        self.correlation = correlation

        # Default portfolio weights from optimizer (max Sharpe allocation)
        # Will be computed properly in aggregate() using PortfolioOptimizer
        if portfolio_weights is not None:
            self.portfolio_weights = portfolio_weights
        else:
            self.portfolio_weights = self._optimize_weights()

        # Risk overlay (optional but recommended)
        self._risk_overlay = None
        if risk_overlay:
            from .risk_overlay import PortfolioRiskOverlay
            self._risk_overlay = PortfolioRiskOverlay(
                initial_equity=initial_equity
            )

    def _optimize_weights(self) -> Dict[str, float]:
        """Run PortfolioOptimizer to get optimal allocation weights."""
        from .optimizer import PortfolioOptimizer
        opt = PortfolioOptimizer()
        results = opt.optimize(step=0.05, correlation_override=self.correlation)
        if results:
            return dict(results[0].weights)
        return {"crypto_perps": 0.50, "crypto_options": 0.50}

    def read_latest_signal(self, project: str) -> Optional[ProjectSignal]:
        """Read the latest signal from a project's signal log."""
        log_path = self.signal_logs.get(project)
        if not log_path or not log_path.exists():
            logger.warning("No signal log for %s at %s", project, log_path)
            return None

        # Read last line of JSONL
        last_line = None
        try:
            with open(log_path) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        last_line = line
        except Exception as e:
            logger.error("Failed to read signal log for %s: %s", project, e)
            return None

        if not last_line:
            return None

        try:
            data = json.loads(last_line)
        except json.JSONDecodeError as e:
            logger.error("Bad JSON in %s signal log: %s", project, e)
            return None

        now = int(time.time())
        epoch = data.get("epoch", 0)
        age = now - epoch if epoch > 0 else 999999

        # Extract target weights (different field names per project)
        target_weights = data.get("target_weights", {})

        # Confidence
        confidence = data.get("confidence", "unknown")
        if project == "crypto_perps":
            # Perps always has confidence if signal exists
            confidence = "high" if target_weights else "no_signal"

        return ProjectSignal(
            project=project,
            timestamp=data.get("timestamp", ""),
            epoch=epoch,
            target_weights=target_weights,
            gross_leverage=data.get("gross_leverage", 0.0),
            confidence=confidence,
            age_seconds=age,
            meta={k: v for k, v in data.items()
                  if k not in ("target_weights", "timestamp", "epoch",
                               "gross_leverage", "confidence")},
        )

    def aggregate(self) -> PortfolioAllocation:
        """
        Read latest signals from all projects and compute unified portfolio.

        Each project's target weights are scaled by its portfolio allocation:
            venue_weight = project_weight × portfolio_allocation

        If a project has no signal or stale signal, its allocation is
        redistributed to active projects.
        """
        now = datetime.now(tz=timezone.utc)
        now_epoch = int(now.timestamp())

        # 1. Read latest signals
        signals: Dict[str, ProjectSignal] = {}
        active_projects: List[str] = []

        for project in self.portfolio_weights:
            sig = self.read_latest_signal(project)
            if sig is not None:
                max_age = MAX_SIGNAL_AGE.get(project, 86400)
                if sig.age_seconds <= max_age and sig.confidence != "insufficient_data":
                    signals[project] = sig
                    active_projects.append(project)
                    logger.info("[%s] Signal OK (age=%ds, confidence=%s)",
                                project, sig.age_seconds, sig.confidence)
                else:
                    signals[project] = sig  # keep for reporting
                    logger.warning("[%s] Signal STALE or insufficient (age=%ds, confidence=%s)",
                                   project, sig.age_seconds, sig.confidence)
            else:
                logger.warning("[%s] No signal available", project)

        # 2. Compute effective allocations (redistribute from inactive projects)
        effective_weights = {}
        if active_projects:
            total_active = sum(self.portfolio_weights.get(p, 0) for p in active_projects)
            if total_active > 0:
                for p in active_projects:
                    effective_weights[p] = self.portfolio_weights.get(p, 0) / total_active
            else:
                for p in active_projects:
                    effective_weights[p] = 1.0 / len(active_projects)
        else:
            logger.warning("No active project signals — portfolio goes to cash")
            effective_weights = {p: 0.0 for p in self.portfolio_weights}

        # 3. Compute per-venue target weights
        venue_targets: Dict[str, Dict[str, float]] = {}
        total_gross = 0.0
        total_net = 0.0

        for project in active_projects:
            sig = signals[project]
            alloc = effective_weights.get(project, 0.0)
            venue = VENUE_MAP.get(project, "unknown")

            if venue not in venue_targets:
                venue_targets[venue] = {}

            for symbol, weight in sig.target_weights.items():
                scaled_weight = weight * alloc
                if symbol in venue_targets[venue]:
                    venue_targets[venue][symbol] += scaled_weight
                else:
                    venue_targets[venue][symbol] = scaled_weight
                total_gross += abs(scaled_weight)
                total_net += scaled_weight

        # Round venue targets
        for venue in venue_targets:
            venue_targets[venue] = {
                s: round(w, 6) for s, w in venue_targets[venue].items()
            }

        # 4. Compute expected portfolio metrics
        from .optimizer import PortfolioOptimizer
        opt = PortfolioOptimizer()
        avg_ret, port_vol, yr_sharpe = opt.portfolio_stats(
            effective_weights, correlation_override=self.correlation
        )
        expected_sharpe = (
            sum(yr_sharpe.values()) / len(yr_sharpe) if yr_sharpe else 0.0
        )

        # 5. Apply risk overlay (can only reduce positions)
        risk_reasons: List[str] = []
        if self._risk_overlay is not None:
            risk_decision = self._risk_overlay.apply(
                venue_targets=venue_targets,
                current_equity=None,  # equity updated externally
            )
            if risk_decision.scale_factor < 1.0:
                venue_targets = risk_decision.venue_targets
                total_gross = sum(
                    abs(w) for t in venue_targets.values() for w in t.values()
                )
                total_net = sum(
                    w for t in venue_targets.values() for w in t.values()
                )
                risk_reasons = risk_decision.reasons
                logger.info("Risk overlay applied: scale=%.2f reasons=%s",
                            risk_decision.scale_factor, risk_reasons)

        allocation = PortfolioAllocation(
            timestamp=now.isoformat(),
            epoch=now_epoch,
            project_allocations=effective_weights,
            venue_targets=venue_targets,
            total_gross_leverage=total_gross,
            total_net_exposure=total_net,
            signals=signals,
            expected_sharpe=expected_sharpe,
            expected_vol_pct=port_vol,
            correlation_assumption=self.correlation,
        )

        # 6. Log
        self._log_allocation(allocation)

        return allocation

    def _log_allocation(self, alloc: PortfolioAllocation) -> None:
        """Append portfolio allocation to JSONL log."""
        PORTFOLIO_LOG.parent.mkdir(parents=True, exist_ok=True)
        try:
            with open(PORTFOLIO_LOG, "a") as f:
                f.write(json.dumps(alloc.to_dict(), default=str) + "\n")
        except Exception as e:
            logger.error("Failed to log portfolio allocation: %s", e)

    def status(self) -> str:
        """Human-readable status report."""
        lines = []
        lines.append("=" * 70)
        lines.append("NEXUS CROSS-PROJECT SIGNAL AGGREGATOR")
        lines.append("=" * 70)
        lines.append("")

        lines.append("Portfolio Allocation Weights:")
        for p, w in sorted(self.portfolio_weights.items()):
            venue = VENUE_MAP.get(p, "?")
            lines.append(f"  {p:25s} {w:6.1%}  (venue: {venue})")
        lines.append(f"  Correlation assumption: {self.correlation:.2f}")
        lines.append("")

        lines.append("Latest Signals:")
        for project in self.portfolio_weights:
            sig = self.read_latest_signal(project)
            if sig is None:
                lines.append(f"  {project:25s} NO SIGNAL")
                continue
            max_age = MAX_SIGNAL_AGE.get(project, 86400)
            stale = sig.age_seconds > max_age
            status = "STALE" if stale else "OK"
            if sig.confidence == "insufficient_data":
                status = "COLLECTING"
            lines.append(
                f"  {project:25s} [{status}] "
                f"age={sig.age_seconds:,}s "
                f"confidence={sig.confidence} "
                f"gross={sig.gross_leverage:.4f}"
            )
            for sym, w in sorted(sig.target_weights.items()):
                if abs(w) > 0.0001:
                    lines.append(f"    {sym:12s} {w:+.4f}")

        lines.append("")

        # Run optimizer for context
        from .optimizer import PortfolioOptimizer
        opt = PortfolioOptimizer()
        avg_ret, port_vol, yr_sharpe = opt.portfolio_stats(
            self.portfolio_weights, correlation_override=self.correlation
        )
        avg_sh = sum(yr_sharpe.values()) / len(yr_sharpe) if yr_sharpe else 0.0
        min_sh = min(yr_sharpe.values()) if yr_sharpe else 0.0
        yr_str = " ".join(f"{yr}:{sh:.2f}" for yr, sh in sorted(yr_sharpe.items()))

        lines.append("Portfolio Expected Performance:")
        lines.append(f"  Avg Sharpe: {avg_sh:.3f}  Min Sharpe: {min_sh:.3f}")
        lines.append(f"  Annual Vol: {port_vol:.1f}%  Annual Return: {avg_ret:.1f}%")
        lines.append(f"  Per-year: {yr_str}")
        lines.append("")
        lines.append("=" * 70)

        return "\n".join(lines)
