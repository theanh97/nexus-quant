"""
Strategy #1: Variance Risk Premium (VRP)

Economics:
    VRP = IV_atm - RV_realized > 0 in ~80% of all periods for crypto.
    Short vol is the DEFAULT position. The goal is to REDUCE exposure only
    during true vol spikes (RV >> IV), not to time entry/exit daily.

Design (carry-like):
    - Base: always SHORT vol at target_leverage (collect premium constantly)
    - Scale: reduce exposure when VRP z-score goes very negative (vol spike)
    - Rebalance: weekly (not daily) to minimize turnover costs

VRP PnL per bar (collected by engine):
    PnL = (IV - RV_rolling) * dt * |weight|
    Annual expectation: (IV - RV) * |weight| ≈ 7-12% at 1.5x leverage

Parameters:
    base_leverage: default short vol leverage (default 1.5 = 0.75 per symbol)
    exit_z_threshold: reduce to 0 only when z < this value (default -1.5 = extreme spike)
    vrp_lookback: bars for rolling VRP history (default 60)
    rebalance_freq: bars between rebalances (default 5 = weekly for daily bars)
    min_bars: warmup (default 30)
"""
from __future__ import annotations

import statistics
from typing import Any, Dict, List, Optional

from nexus_quant.data.schema import MarketDataset
from nexus_quant.strategies.base import Strategy, Weights


class VariancePremiumStrategy(Strategy):
    """Short variance premium — carry-style, always short vol.

    Default position: always short vol at base_leverage.
    Reduce to 0 only when VRP z-score < exit_z_threshold (vol spike = cover).
    Weekly rebalancing to minimize costs.
    """

    def __init__(self, name: str = "crypto_vrp", params: Dict[str, Any] = None) -> None:
        params = params or {}
        super().__init__(name, params)

        self.base_leverage: float = float(params.get("base_leverage", 1.5))
        self.exit_z_threshold: float = float(params.get("exit_z_threshold", -1.5))
        self.vrp_lookback: int = int(params.get("vrp_lookback", 60))
        self.rebalance_freq: int = int(params.get("rebalance_freq", 5))
        self.min_bars: int = int(params.get("min_bars", 30))

    def should_rebalance(self, dataset: MarketDataset, idx: int) -> bool:
        if idx < self.min_bars:
            return False
        return (idx % self.rebalance_freq) == 0

    def target_weights(
        self, dataset: MarketDataset, idx: int, current: Weights
    ) -> Weights:
        syms = dataset.symbols
        n = max(len(syms), 1)
        per_sym = self.base_leverage / n
        weights: Weights = {}

        for sym in syms:
            z = self._get_vrp_zscore(dataset, sym, idx)

            if z is None:
                # Insufficient history: flat
                weights[sym] = 0.0
            elif z < self.exit_z_threshold:
                # Extreme vol spike (VRP very negative): cover short, go flat
                weights[sym] = 0.0
            else:
                # Normal: short vol
                # Optionally scale by z-score magnitude for more premium when VRP high
                # scale = min(1.0, max(0.5, 0.5 + z * 0.25))  # 0.5x to 1.0x
                weights[sym] = -per_sym  # fixed short

        return weights

    def _get_vrp_zscore(
        self, dataset: MarketDataset, sym: str, idx: int
    ) -> Optional[float]:
        """Time-series VRP z-score vs this symbol's own rolling history."""
        iv_series = dataset.feature("iv_atm", sym)
        rv_series = dataset.feature("rv_realized", sym)

        if iv_series is None or rv_series is None:
            return None

        history: List[float] = []
        start = max(0, idx - self.vrp_lookback)
        for i in range(start, min(idx + 1, len(iv_series), len(rv_series))):
            iv = iv_series[i]
            rv = rv_series[i]
            if iv is not None and rv is not None:
                history.append(float(iv) - float(rv))

        if len(history) < max(10, self.vrp_lookback // 6):
            return None

        current_vrp = history[-1]
        mean_vrp = sum(history) / len(history)
        std_vrp = statistics.pstdev(history)

        if std_vrp < 1e-6:
            return 0.0

        return (current_vrp - mean_vrp) / std_vrp
