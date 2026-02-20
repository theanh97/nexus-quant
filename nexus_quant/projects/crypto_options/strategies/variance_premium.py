"""
Strategy #1: Variance Risk Premium (VRP)

Signal: IV_atm - RV_realized_21d → short vol when IV >> RV

Economics:
    Option sellers systematically earn a premium because:
    1. Buyers pay for insurance (crash protection)
    2. Market makers demand a spread
    3. IV consistently overstates future realized vol by ~5-10 vol points

Implementation (delta-equivalent simplified approach):
    - Signal = VRP = IV_atm - RV_realized
    - Z-score VRP across symbols (cross-sectional)
    - Short vol (negative delta-equivalent weight) when VRP is high
    - Flat when VRP signal is weak
    - Model: sell premium → profit from theta + vol mean reversion

Parameters:
    vrp_threshold: min VRP to enter position (default 0.05 = 5 vol points)
    vrp_lookback: bars for VRP z-score normalization (default 60)
    target_gross_leverage: max sum(abs(weights)) (default 1.5)
    rebalance_freq: bars between rebalances (default 24 = daily for 1h bars)
    min_bars: min bars before trading (default 30)
    vol_scale: if True, scale by inverse vol (risk parity across symbols)
"""
from __future__ import annotations

import statistics
from typing import Any, Dict, List, Optional

from nexus_quant.data.schema import MarketDataset
from nexus_quant.strategies.base import Strategy, Weights


class VariancePremiumStrategy(Strategy):
    """Short variance premium on crypto options (BTC/ETH).

    Uses VRP (IV_atm - RV_realized) as the primary signal.
    Outputs delta-equivalent weights on the underlying.
    """

    def __init__(self, name: str = "crypto_vrp", params: Dict[str, Any] = None) -> None:
        params = params or {}
        super().__init__(name, params)

        self.vrp_threshold: float = float(params.get("vrp_threshold", 0.03))
        self.vrp_lookback: int = int(params.get("vrp_lookback", 60))
        self.target_leverage: float = float(params.get("target_gross_leverage", 1.5))
        self.rebalance_freq: int = int(params.get("rebalance_freq", 24))
        self.min_bars: int = int(params.get("min_bars", 30))
        self.vol_scale: bool = bool(params.get("vol_scale", True))

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
            vrp = self._get_vrp(dataset, sym, idx)
            if vrp is None:
                continue
            signals[sym] = vrp

        if not signals:
            return weights

        # Z-score VRP across symbols for cross-sectional ranking
        if len(signals) > 1:
            vals = list(signals.values())
            mean = sum(vals) / len(vals)
            std = statistics.pstdev(vals)
            if std > 0:
                signals = {s: (v - mean) / std for s, v in signals.items()}
            else:
                signals = {s: 0.0 for s in signals}

        # Signal → weights
        # High VRP → short vol → negative weight (sell premium)
        # Low VRP → flat (don't fight market when IV is cheap)
        active_syms = [s for s in signals if abs(signals[s]) > 0.5]
        if not active_syms:
            return weights

        total_signal = sum(abs(signals[s]) for s in active_syms)
        if total_signal <= 0:
            return weights

        for sym in syms:
            sig = signals.get(sym, 0.0)
            if abs(sig) <= 0.5:  # below threshold — stay flat
                weights[sym] = 0.0
                continue

            # Directional: short vol = negative weight (sell premium = short underlying exposure)
            # Positive VRP → IV > RV → sell premium → short delta-equivalent
            raw_weight = -(sig / total_signal) * self.target_leverage
            weights[sym] = raw_weight

        # Apply vol scaling (inverse vol weighting for risk parity)
        if self.vol_scale:
            weights = self._apply_vol_scale(dataset, idx, weights, syms)

        # Clamp total leverage
        total = sum(abs(w) for w in weights.values())
        if total > self.target_leverage * 1.1:
            scale = (self.target_leverage * 1.1) / total
            weights = {s: w * scale for s, w in weights.items()}

        return weights

    def _get_vrp(
        self, dataset: MarketDataset, sym: str, idx: int
    ) -> Optional[float]:
        """Get current VRP = IV_atm - RV_realized at bar idx."""
        iv_series = dataset.feature("iv_atm", sym)
        rv_series = dataset.feature("rv_realized", sym)

        if iv_series is None or rv_series is None:
            return None
        if idx >= len(iv_series) or idx >= len(rv_series):
            return None

        iv = iv_series[idx]
        rv = rv_series[idx]
        if iv is None or rv is None:
            return None

        return float(iv) - float(rv)

    def _get_vrp_zscore(
        self, dataset: MarketDataset, sym: str, idx: int
    ) -> Optional[float]:
        """Get z-scored VRP relative to recent history."""
        # Compute rolling VRP history
        vrp_history: List[float] = []
        start = max(0, idx - self.vrp_lookback)
        for i in range(start, idx + 1):
            v = self._get_vrp(dataset, sym, i)
            if v is not None:
                vrp_history.append(v)

        if len(vrp_history) < 10:
            return None

        current = vrp_history[-1]
        mean = sum(vrp_history) / len(vrp_history)
        std = statistics.pstdev(vrp_history)
        if std < 1e-6:
            return 0.0
        return (current - mean) / std

    def _apply_vol_scale(
        self,
        dataset: MarketDataset,
        idx: int,
        weights: Weights,
        syms: List[str],
    ) -> Weights:
        """Scale weights by inverse realized vol (risk parity)."""
        inv_vols: Dict[str, float] = {}
        for sym in syms:
            rv_series = dataset.feature("rv_realized", sym)
            if rv_series and idx < len(rv_series) and rv_series[idx] is not None:
                rv = float(rv_series[idx])
                inv_vols[sym] = 1.0 / max(rv, 0.10)
            else:
                # Fallback: compute from price history
                closes = dataset.perp_close.get(sym, [])
                if len(closes) > 21:
                    window = [
                        closes[i] / closes[i - 1] - 1
                        for i in range(max(1, idx - 21), idx + 1)
                        if closes[i - 1] > 0
                    ]
                    if len(window) > 5:
                        rv = statistics.pstdev(window) * (8760 ** 0.5)
                        inv_vols[sym] = 1.0 / max(rv, 0.10)
                    else:
                        inv_vols[sym] = 1.0
                else:
                    inv_vols[sym] = 1.0

        if not inv_vols:
            return weights

        # Normalize inv_vol weights
        total_inv = sum(inv_vols.values())
        norm_inv = {s: v / total_inv for s, v in inv_vols.items()}

        # Scale: keep direction, scale magnitude by inverse vol
        scaled: Weights = {}
        for sym in syms:
            w = weights.get(sym, 0.0)
            if w == 0.0:
                scaled[sym] = 0.0
            else:
                sign = 1.0 if w > 0 else -1.0
                scaled[sym] = sign * abs(w) * (norm_inv.get(sym, 0.0) * len(syms))
        return scaled
