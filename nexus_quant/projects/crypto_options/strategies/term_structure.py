"""
Strategy #3: Term Structure Calendar Spread

Signal: Front-month IV vs back-month IV spread → mean-reverts around equilibrium

Economics:
    - Normal: front IV slightly above back IV (contango) due to gamma premium
    - Stress: front IV spikes above back IV dramatically (backwardation → panic)
    - Post-stress: front IV collapses back to normal → term spread mean-reverts
    - Trade: when term spread is extreme, bet on mean reversion

Implementation (delta-equivalent simplified approach):
    - Signal = term_spread_z = (current_spread - mean_spread_60d) / std_spread_60d
    - High term_spread_z (front >> back, unusual): sell front vol = sell underlying
      (steeply inverted term structure = panic selling = expect vol fade)
    - Low term_spread_z (back >> front, backwardation extreme): buy front vol = buy underlying
      (unusual backwardation = forced hedging = tend to revert)
    - Rebalance: weekly (slower signal)

Parameters:
    term_lookback: bars for rolling mean/std of term spread (default 60*24 = 60 days hourly)
    z_threshold: z-score to enter (default 1.5)
    target_gross_leverage: max position (default 1.0)
    rebalance_freq: bars between rebalances (default 168 = weekly at 1h bars)
    min_bars: warmup period (default 90)
"""
from __future__ import annotations

import statistics
from typing import Any, Dict, List, Optional

from nexus_quant.data.schema import MarketDataset
from nexus_quant.strategies.base import Strategy, Weights


class TermStructureStrategy(Strategy):
    """Calendar spread / term structure strategy on crypto options.

    Trades mean-reversion of the front-vs-back-month IV spread.
    High spread (front >> back) = panic vol → sell underlying.
    Low/inverted spread → forced hedging → buy underlying.
    """

    def __init__(
        self, name: str = "crypto_term_structure", params: Dict[str, Any] = None
    ) -> None:
        params = params or {}
        super().__init__(name, params)

        self.term_lookback: int = int(params.get("term_lookback", 60))
        self.z_threshold: float = float(params.get("z_threshold", 1.2))
        self.target_leverage: float = float(params.get("target_gross_leverage", 1.0))
        self.rebalance_freq: int = int(params.get("rebalance_freq", 168))
        self.min_bars: int = int(params.get("min_bars", 90))
        self.use_vrp_confirm: bool = bool(params.get("use_vrp_confirm", False))

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
            z = self._get_term_zscore(dataset, sym, idx)
            if z is not None:
                signals[sym] = z

        if not signals:
            return weights

        # Only trade extreme z-scores
        traded_syms = {s: z for s, z in signals.items() if abs(z) >= self.z_threshold}
        if not traded_syms:
            return weights

        # Optional confirmation: VRP must also signal the same direction
        if self.use_vrp_confirm:
            traded_syms = self._apply_vrp_filter(dataset, idx, traded_syms)
        if not traded_syms:
            return weights

        total_signal = sum(abs(z) for z in traded_syms.values())
        if total_signal <= 0:
            return weights

        for sym in syms:
            z = traded_syms.get(sym)
            if z is None:
                weights[sym] = 0.0
                continue

            # High term spread (z>0): front vol elevated = panic = short underlying
            # Low term spread (z<0): backwardation = buy underlying
            # Direction: negative sign (high spread = bad = short)
            direction = -1.0 if z > 0 else 1.0
            mag = (abs(z) / total_signal) * self.target_leverage
            weights[sym] = direction * mag

        return weights

    def _get_term_zscore(
        self, dataset: MarketDataset, sym: str, idx: int
    ) -> Optional[float]:
        """Z-score of current term spread vs rolling history."""
        term_series = dataset.feature("term_spread", sym)
        if term_series is None:
            return None

        start = max(0, idx - self.term_lookback)
        history: List[float] = []
        for i in range(start, min(idx + 1, len(term_series))):
            v = term_series[i]
            if v is not None:
                history.append(float(v))

        if len(history) < 20:
            return None

        current = history[-1]
        mean = sum(history) / len(history)
        std = statistics.pstdev(history)
        if std < 1e-6:
            return 0.0
        return (current - mean) / std

    def _apply_vrp_filter(
        self,
        dataset: MarketDataset,
        idx: int,
        signals: Dict[str, float],
    ) -> Dict[str, float]:
        """Filter signals that conflict with VRP direction."""
        filtered: Dict[str, float] = {}
        for sym, z in signals.items():
            vrp_series = dataset.feature("vrp", sym)
            if vrp_series is None or idx >= len(vrp_series):
                filtered[sym] = z
                continue
            vrp = vrp_series[idx]
            if vrp is None:
                filtered[sym] = z
                continue
            vrp_val = float(vrp)
            # High term_z (→ short) confirmed if VRP also positive (IV > RV)
            # Low term_z (→ long) confirmed if VRP very low (IV ≈ RV, fair)
            if z > 0 and vrp_val > 0.02:
                filtered[sym] = z  # Confirmed: sell premium
            elif z < 0 and vrp_val < 0.05:
                filtered[sym] = z  # Confirmed: buy when vol cheap
            # Otherwise: skip (conflicting signals)
        return filtered
