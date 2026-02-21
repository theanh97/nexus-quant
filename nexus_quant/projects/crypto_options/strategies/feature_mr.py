"""
Strategy: Generic Feature Mean-Reversion
=========================================
Generalized mean-reversion strategy on any configurable feature.
Supports: butterfly_25d, iv_atm, put_call_ratio, or any other feature
available in the MarketDataset.

Uses the same hold management and z-score logic as SkewTradeV2Strategy,
but with a configurable feature_key instead of hardcoded skew_25d.

P&L model is determined by the engine routing (name-based):
    - "butterfly" → vega/feature-change model (via _run_skew)
    - "iv_mr"     → vega/IV-change model (via _run_skew)
    - "pcr"       → vega/PCR-change model (via _run_skew)
"""
from __future__ import annotations

import statistics
from typing import Any, Dict, List, Optional

from nexus_quant.data.schema import MarketDataset
from nexus_quant.strategies.base import Strategy, Weights


class FeatureMRStrategy(Strategy):
    """Generic feature mean-reversion strategy.

    Fades extreme z-scores on any configurable feature.
    Uses hold management (enter at extreme z, exit at z~0).

    Strategy weights represent feature exposure:
        weight > 0 → "long feature" → expects feature to increase
        weight < 0 → "short feature" → expects feature to decrease
    """

    def __init__(self, name: str = "feature_mr", params: Dict[str, Any] = None) -> None:
        params = params or {}
        super().__init__(name, params)

        # Feature configuration
        self.feature_key: str = str(params.get("feature_key", "butterfly_25d"))

        # Signal parameters
        self.lookback: int = int(params.get("lookback", 60))
        self.z_entry: float = float(params.get("z_entry", 2.0))
        self.z_exit: float = float(params.get("z_exit", 0.0))
        self.target_leverage: float = float(params.get("target_leverage", 1.0))
        self.rebalance_freq: int = int(params.get("rebalance_freq", 5))
        self.min_bars: int = int(params.get("min_bars", 60))

        # Position tracking for hold management
        self._positions: Dict[str, float] = {}

    def should_rebalance(self, dataset: MarketDataset, idx: int) -> bool:
        if idx < self.min_bars:
            return False
        return (idx % self.rebalance_freq) == 0

    def target_weights(
        self, dataset: MarketDataset, idx: int, current: Weights
    ) -> Weights:
        syms = dataset.symbols
        n = max(len(syms), 1)
        per_sym = self.target_leverage / n
        weights: Weights = {}

        for sym in syms:
            z = self._get_feature_zscore(dataset, sym, idx)
            current_pos = self._positions.get(sym, 0.0)

            if z is None:
                weights[sym] = 0.0
                self._positions[sym] = 0.0
                continue

            new_pos = self._compute_position(z, current_pos, per_sym)
            weights[sym] = new_pos
            self._positions[sym] = new_pos

        return weights

    def _compute_position(self, z: float, current_pos: float, per_sym: float) -> float:
        """Compute target position based on z-score and current position.

        z > z_entry: feature is HIGH → SHORT feature (expect reversion down)
            → weight < 0
        z < -z_entry: feature is LOW → LONG feature (expect reversion up)
            → weight > 0
        |z| < z_exit: close position (feature near mean)
        """
        if current_pos == 0.0:
            # No position → check entry
            if z >= self.z_entry:
                return -per_sym
            elif z <= -self.z_entry:
                return per_sym
            else:
                return 0.0
        else:
            # Have position → check exit
            if abs(z) <= self.z_exit:
                return 0.0
            elif current_pos < 0 and z <= 0:
                # Was short, feature dropped below mean → exit
                return 0.0
            elif current_pos > 0 and z >= 0:
                # Was long, feature rose above mean → exit
                return 0.0
            else:
                return current_pos

    def _get_feature_zscore(
        self, dataset: MarketDataset, sym: str, idx: int
    ) -> Optional[float]:
        """Z-score of current feature value vs rolling mean/std."""
        series = dataset.feature(self.feature_key, sym)
        if series is None:
            return None

        start = max(0, idx - self.lookback)
        history: List[float] = []
        for i in range(start, min(idx + 1, len(series))):
            v = series[i]
            if v is not None:
                history.append(float(v))

        if len(history) < 10:
            return None

        current = history[-1]
        mean = sum(history) / len(history)
        std = statistics.pstdev(history)
        if std < 1e-6:
            return 0.0
        return (current - mean) / std
