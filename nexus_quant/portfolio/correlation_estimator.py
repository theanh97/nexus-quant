"""
Real-Time Cross-Project Correlation Estimator
===============================================

Estimates the rolling correlation between NEXUS project returns in real-time.
Used by the PortfolioRiskOverlay to detect correlation spikes (diversification
benefit eroding).

Methods:
  1. EWMA (Exponentially Weighted Moving Average) — primary
     - Fast-reacting to regime changes
     - Half-life configurable (default: 20 periods)
  2. Rolling window — secondary (for validation)
     - Fixed window (default: 60 periods)
     - More stable but slower to react

Architecture:
  The estimator reads return logs from each project and computes pairwise
  correlations. It can also be updated incrementally with new return observations.

Key insight from R&D (rule R20): VRP-Skew correlation DROPS during stress
(0.01 vs 0.04 overall). This is ideal for diversification — but cross-project
correlation (perps vs options) may behave differently.

Usage:
    from nexus_quant.portfolio.correlation_estimator import CorrelationEstimator
    est = CorrelationEstimator(half_life=20)
    est.update("crypto_perps", 0.0012)
    est.update("crypto_options", -0.0005)
    corr = est.current_correlation()
"""
from __future__ import annotations

import json
import logging
import math
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("nexus.correlation")

PROJ_ROOT = Path(__file__).resolve().parents[2]
CORR_STATE = PROJ_ROOT / "artifacts" / "portfolio" / "correlation_state.json"
CORR_LOG = PROJ_ROOT / "artifacts" / "portfolio" / "correlation_log.jsonl"


@dataclass
class CorrelationState:
    """Persistent state for EWMA correlation estimation."""
    # EWMA moments (for each project)
    ewma_mean: Dict[str, float] = field(default_factory=dict)
    ewma_var: Dict[str, float] = field(default_factory=dict)

    # Cross-project EWMA covariance
    ewma_cov: Dict[str, float] = field(default_factory=dict)

    # Return history (for rolling window backup)
    return_history: Dict[str, List[float]] = field(default_factory=dict)

    # Current estimates
    current_correlation: float = 0.0
    last_update: str = ""
    n_updates: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ewma_mean": self.ewma_mean,
            "ewma_var": {k: round(v, 8) for k, v in self.ewma_var.items()},
            "ewma_cov": {k: round(v, 8) for k, v in self.ewma_cov.items()},
            "current_correlation": round(self.current_correlation, 4),
            "last_update": self.last_update,
            "n_updates": self.n_updates,
            "history_lengths": {k: len(v) for k, v in self.return_history.items()},
        }

    def save(self) -> None:
        CORR_STATE.parent.mkdir(parents=True, exist_ok=True)
        with open(CORR_STATE, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls) -> "CorrelationState":
        if CORR_STATE.exists():
            try:
                with open(CORR_STATE) as f:
                    d = json.load(f)
                state = cls(
                    ewma_mean=d.get("ewma_mean", {}),
                    ewma_var=d.get("ewma_var", {}),
                    ewma_cov=d.get("ewma_cov", {}),
                    current_correlation=d.get("current_correlation", 0.0),
                    last_update=d.get("last_update", ""),
                    n_updates=d.get("n_updates", 0),
                )
                return state
            except Exception:
                pass
        return cls()


class CorrelationEstimator:
    """
    Real-time cross-project correlation estimator.

    Uses EWMA for fast reaction to regime changes + rolling window for stability.
    """

    def __init__(
        self,
        projects: Optional[List[str]] = None,
        half_life: int = 20,
        rolling_window: int = 60,
        max_history: int = 500,
    ) -> None:
        self.projects = projects or ["crypto_perps", "crypto_options"]
        self.half_life = half_life
        self.rolling_window = rolling_window
        self.max_history = max_history

        # EWMA decay factor: alpha = 1 - exp(-ln2 / half_life)
        self.alpha = 1.0 - math.exp(-math.log(2) / half_life)

        self.state = CorrelationState.load()

        # Initialize EWMA moments if needed
        for p in self.projects:
            if p not in self.state.ewma_mean:
                self.state.ewma_mean[p] = 0.0
            if p not in self.state.ewma_var:
                self.state.ewma_var[p] = 0.0001  # small initial variance
            if p not in self.state.return_history:
                self.state.return_history[p] = []

        # Initialize covariance for each pair
        for i, p1 in enumerate(self.projects):
            for p2 in self.projects[i + 1:]:
                pair_key = f"{p1}:{p2}"
                if pair_key not in self.state.ewma_cov:
                    self.state.ewma_cov[pair_key] = 0.0

    def update(self, project: str, return_value: float) -> None:
        """Update EWMA moments with a new return observation."""
        if project not in self.projects:
            return

        # Store return in history
        self.state.return_history[project].append(return_value)
        if len(self.state.return_history[project]) > self.max_history:
            self.state.return_history[project] = \
                self.state.return_history[project][-self.max_history:]

        # Update EWMA mean and variance
        old_mean = self.state.ewma_mean[project]
        new_mean = (1 - self.alpha) * old_mean + self.alpha * return_value
        self.state.ewma_mean[project] = new_mean

        deviation = return_value - old_mean
        old_var = self.state.ewma_var[project]
        new_var = (1 - self.alpha) * old_var + self.alpha * deviation ** 2
        self.state.ewma_var[project] = max(new_var, 1e-10)

        self.state.n_updates += 1

    def update_pair(
        self, returns: Dict[str, float],
    ) -> float:
        """
        Update with simultaneous returns from all projects.

        This is the preferred method — ensures covariance is computed
        from synchronized observations.

        Args:
            returns: {project_name: return_value}

        Returns:
            Current estimated correlation.
        """
        # Update individual moments
        for project, ret in returns.items():
            self.update(project, ret)

        # Update cross-covariance for each pair
        for i, p1 in enumerate(self.projects):
            for p2 in self.projects[i + 1:]:
                if p1 in returns and p2 in returns:
                    pair_key = f"{p1}:{p2}"
                    dev1 = returns[p1] - self.state.ewma_mean[p1]
                    dev2 = returns[p2] - self.state.ewma_mean[p2]
                    old_cov = self.state.ewma_cov.get(pair_key, 0.0)
                    new_cov = (1 - self.alpha) * old_cov + self.alpha * dev1 * dev2
                    self.state.ewma_cov[pair_key] = new_cov

        # Compute correlation
        corr = self._compute_correlation()
        self.state.current_correlation = corr
        self.state.last_update = datetime.now(timezone.utc).isoformat()

        return corr

    def _compute_correlation(self) -> float:
        """Compute pairwise EWMA correlation (average across all pairs)."""
        correlations = []

        for i, p1 in enumerate(self.projects):
            for p2 in self.projects[i + 1:]:
                pair_key = f"{p1}:{p2}"
                cov = self.state.ewma_cov.get(pair_key, 0.0)
                var1 = self.state.ewma_var.get(p1, 1e-10)
                var2 = self.state.ewma_var.get(p2, 1e-10)
                denom = math.sqrt(var1 * var2)
                if denom > 1e-10:
                    rho = cov / denom
                    rho = max(-1.0, min(1.0, rho))  # clamp
                    correlations.append(rho)

        return sum(correlations) / len(correlations) if correlations else 0.0

    def current_correlation(self) -> float:
        """Get current estimated correlation."""
        return self.state.current_correlation

    def rolling_correlation(self) -> Optional[float]:
        """Compute correlation from rolling window (validation method)."""
        if len(self.projects) < 2:
            return None

        p1, p2 = self.projects[0], self.projects[1]
        h1 = self.state.return_history.get(p1, [])
        h2 = self.state.return_history.get(p2, [])

        n = min(len(h1), len(h2), self.rolling_window)
        if n < 10:
            return None

        r1 = h1[-n:]
        r2 = h2[-n:]
        m1 = sum(r1) / n
        m2 = sum(r2) / n

        cov = sum((r1[i] - m1) * (r2[i] - m2) for i in range(n)) / (n - 1)
        v1 = sum((x - m1) ** 2 for x in r1) / (n - 1)
        v2 = sum((x - m2) ** 2 for x in r2) / (n - 1)
        denom = math.sqrt(v1 * v2)

        if denom < 1e-10:
            return 0.0
        return max(-1.0, min(1.0, cov / denom))

    def is_spiking(self, threshold: float = 0.50) -> bool:
        """Check if correlation has spiked above threshold."""
        return abs(self.state.current_correlation) > threshold

    def save(self) -> None:
        """Persist state."""
        self.state.save()

    def log_observation(self) -> None:
        """Log current correlation to JSONL."""
        CORR_LOG.parent.mkdir(parents=True, exist_ok=True)
        try:
            entry = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "ewma_correlation": round(self.state.current_correlation, 4),
                "rolling_correlation": self.rolling_correlation(),
                "n_updates": self.state.n_updates,
                "ewma_vars": {k: round(v, 8) for k, v in self.state.ewma_var.items()},
            }
            with open(CORR_LOG, "a") as f:
                f.write(json.dumps(entry) + "\n")
        except Exception as e:
            logger.error("Failed to log correlation: %s", e)

    def status(self) -> str:
        """Human-readable status."""
        lines = []
        lines.append("NEXUS Cross-Project Correlation Estimator")
        lines.append(f"  Projects: {', '.join(self.projects)}")
        lines.append(f"  EWMA half-life: {self.half_life} periods")
        lines.append(f"  Rolling window: {self.rolling_window} periods")
        lines.append(f"  EWMA correlation: {self.state.current_correlation:+.4f}")
        roll = self.rolling_correlation()
        if roll is not None:
            lines.append(f"  Rolling correlation: {roll:+.4f}")
        lines.append(f"  Spike threshold: 0.50")
        lines.append(f"  Currently spiking: {self.is_spiking()}")
        lines.append(f"  Updates: {self.state.n_updates}")
        lines.append(f"  Last update: {self.state.last_update}")
        for p in self.projects:
            n = len(self.state.return_history.get(p, []))
            lines.append(f"  {p}: {n} observations, "
                        f"mean={self.state.ewma_mean.get(p, 0):.6f}, "
                        f"var={self.state.ewma_var.get(p, 0):.8f}")
        return "\n".join(lines)
