"""
Strategy #2: Skew Mean-Reversion

Signal: 25-delta skew (put IV - call IV) deviates from rolling mean → fade the deviation

Economics:
    - Skew represents crash risk premium (put protection demand)
    - When skew is VERY HIGH: market is fearful, buying protection → mean reverts
    - When skew is VERY LOW: calls are expensive → market frothy → mean reverts
    - Fade extreme skew moves as mean-reverting signal

Implementation (delta-equivalent simplified approach):
    - Signal = skew_z = (current_skew - mean_skew_30d) / std_skew_30d
    - skew_z > 1.5: skew too expensive → sell puts, buy calls → LONG underlying
      (put sellers profit when puts revert to fair value = crash didn't happen)
    - skew_z < -1.5: calls too expensive → FLAT or slight short
    - Delta-hedge with underlying (implicit in weight representation)

Parameters:
    skew_lookback: bars for rolling mean/std of skew (default 30*24 = 30 days hourly)
    z_threshold: z-score threshold to enter position (default 1.5)
    target_gross_leverage: max position size (default 1.2)
    rebalance_freq: bars between rebalances (default 24 = daily)
    min_bars: warmup period before trading (default 60)
"""
from __future__ import annotations

import statistics
from typing import Any, Dict, List, Optional

from nexus_quant.data.schema import MarketDataset
from nexus_quant.strategies.base import Strategy, Weights


class SkewTradeStrategy(Strategy):
    """Skew mean-reversion strategy on crypto options.

    Fades extreme moves in the 25-delta put/call skew.
    When puts are very expensive (skew spike), we go long underlying
    (equivalent to selling overpriced put protection).
    """

    def __init__(self, name: str = "crypto_skew_mr", params: Dict[str, Any] = None) -> None:
        params = params or {}
        super().__init__(name, params)

        self.skew_lookback: int = int(params.get("skew_lookback", 30))
        self.z_threshold: float = float(params.get("z_threshold", 1.5))
        self.target_leverage: float = float(params.get("target_gross_leverage", 1.2))
        self.rebalance_freq: int = int(params.get("rebalance_freq", 24))
        self.min_bars: int = int(params.get("min_bars", 60))
        self.use_butterfly: bool = bool(params.get("use_butterfly", False))

    def should_rebalance(self, dataset: MarketDataset, idx: int) -> bool:
        if idx < self.min_bars:
            return False
        return (idx % self.rebalance_freq) == 0

    def target_weights(
        self, dataset: MarketDataset, idx: int, current: Weights
    ) -> Weights:
        syms = dataset.symbols
        weights: Weights = {s: 0.0 for s in syms}

        signals: Dict[str, float] = {}
        for sym in syms:
            z = self._get_skew_zscore(dataset, sym, idx)
            if z is not None:
                signals[sym] = z

        if not signals:
            return weights

        # Only trade extreme z-scores
        traded_syms = {s: z for s, z in signals.items() if abs(z) >= self.z_threshold}
        if not traded_syms:
            return weights

        # Normalize and allocate
        total_signal = sum(abs(z) for z in traded_syms.values())

        for sym in syms:
            z = traded_syms.get(sym)
            if z is None:
                weights[sym] = 0.0
                continue

            # Skew too high (puts expensive, fear spike):
            #   → sell puts → profit from crash NOT happening → LONG underlying
            # Skew too low (calls expensive, euphoria):
            #   → sell calls → profit from rally failing → SHORT underlying
            # Direction: negative sign on z (mean-revert)
            direction = -1.0 if z > 0 else 1.0
            mag = (abs(z) / total_signal) * self.target_leverage
            weights[sym] = direction * mag

        return weights

    def _get_skew_zscore(
        self, dataset: MarketDataset, sym: str, idx: int
    ) -> Optional[float]:
        """Z-score of current skew vs rolling mean/std."""
        feature = "skew_25d"
        if self.use_butterfly:
            feature = "butterfly_25d"

        skew_series = dataset.feature(feature, sym)
        if skew_series is None:
            return None

        # Get rolling history
        start = max(0, idx - self.skew_lookback)
        history: List[float] = []
        for i in range(start, min(idx + 1, len(skew_series))):
            v = skew_series[i]
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

    def _get_current_skew(
        self, dataset: MarketDataset, sym: str, idx: int
    ) -> Optional[float]:
        """Get current 25-delta skew value."""
        skew_series = dataset.feature("skew_25d", sym)
        if skew_series is None or idx >= len(skew_series):
            return None
        v = skew_series[idx]
        return float(v) if v is not None else None
