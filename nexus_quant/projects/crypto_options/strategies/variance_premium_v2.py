"""
Strategy: Variance Risk Premium v2 (VRP + Overlays)
=====================================================
Extension of VRP v1 with three overlay signals for smarter position sizing:

1. VRP Z-Score Scaling: Smooth leverage between 0.3x-1.2x based on z-score
   (replaces hard cutoff at exit_z_threshold)
2. IV Percentile Rank: Boost leverage when IV is high (more premium to collect),
   reduce when IV is at historical lows (less premium, danger of vol expansion)
3. Vol Regime: Scale down in high-vol regimes (gamma costs eat theta),
   full size in low/medium vol

Combined signal:
    weight = -base_leverage * z_scale * iv_scale * vol_scale / n_symbols

Economics: Same carry trade as v1, but modulates position size based on:
    - How much premium exists (z_scale)
    - How expensive options are (iv_scale)
    - How volatile the market is (vol_scale)

Expected improvement: Better drawdown control during vol spikes while
maintaining full exposure during premium-rich calm periods.
"""
from __future__ import annotations

import math
import statistics
from typing import Any, Dict, List, Optional

from nexus_quant.data.schema import MarketDataset
from nexus_quant.strategies.base import Strategy, Weights


def _clamp(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else hi if x > hi else x


def _percentile_rank(value: float, history: List[float]) -> float:
    """Compute percentile rank of value within history (0.0 = lowest, 1.0 = highest)."""
    if not history:
        return 0.5
    count_below = sum(1 for h in history if h < value)
    count_equal = sum(1 for h in history if h == value)
    return (count_below + 0.5 * count_equal) / len(history)


class VariancePremiumV2Strategy(Strategy):
    """VRP v2 — carry-style short vol with overlay signals.

    Three overlays modulate position size:
    1. z_scale: smooth function of VRP z-score (0.3x to 1.2x)
    2. iv_scale: IV percentile rank filter (0.5x to 1.3x)
    3. vol_scale: vol regime ceiling (0.4x to 1.0x)
    """

    def __init__(self, name: str = "crypto_vrp_v2", params: Dict[str, Any] = None) -> None:
        params = params or {}
        super().__init__(name, params)

        # Base parameters (same as v1)
        self.base_leverage: float = float(params.get("base_leverage", 1.5))
        self.vrp_lookback: int = int(params.get("vrp_lookback", 30))
        self.rebalance_freq: int = int(params.get("rebalance_freq", 5))
        self.min_bars: int = int(params.get("min_bars", 30))

        # Overlay 1: VRP z-score scaling
        self.use_z_scaling: bool = bool(params.get("use_z_scaling", True))
        self.z_base: float = float(params.get("z_base", 0.7))     # scale at z=0
        self.z_slope: float = float(params.get("z_slope", 0.25))   # scale per z unit
        self.z_min: float = float(params.get("z_min", 0.0))        # min scale (0 = can go flat)
        self.z_max: float = float(params.get("z_max", 1.2))        # max scale (cap leverage boost)

        # Overlay 2: IV percentile rank
        self.use_iv_pct: bool = bool(params.get("use_iv_pct", True))
        self.iv_pct_lookback: int = int(params.get("iv_pct_lookback", 252))
        self.iv_pct_low: float = float(params.get("iv_pct_low", 0.25))   # below this → reduce
        self.iv_pct_high: float = float(params.get("iv_pct_high", 0.75)) # above this → boost
        self.iv_scale_low: float = float(params.get("iv_scale_low", 0.5))  # scale when IV low
        self.iv_scale_high: float = float(params.get("iv_scale_high", 1.3)) # scale when IV high

        # Overlay 3: Vol regime
        self.use_vol_regime: bool = bool(params.get("use_vol_regime", True))
        self.vol_regime_lookback: int = int(params.get("vol_regime_lookback", 60))
        self.vol_high_threshold: float = float(params.get("vol_high_threshold", 0.80))  # pct rank
        self.vol_scale_high: float = float(params.get("vol_scale_high", 0.4))  # scale in high vol

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
            # Get overlay scales
            z_scale = self._z_score_scale(dataset, sym, idx)
            iv_scale = self._iv_percentile_scale(dataset, sym, idx)
            vol_scale = self._vol_regime_scale(dataset, sym, idx)

            combined = z_scale * iv_scale * vol_scale
            weights[sym] = -per_sym * combined

        return weights

    # ── Overlay 1: VRP Z-Score Scaling ──────────────────────────────────────

    def _z_score_scale(self, dataset: MarketDataset, sym: str, idx: int) -> float:
        """Smooth scale from z_min to z_max based on VRP z-score.

        z >> 0 (VRP very positive, lots of premium): scale up
        z << 0 (VRP negative, vol spike): scale down toward 0
        """
        if not self.use_z_scaling:
            return 1.0

        z = self._get_vrp_zscore(dataset, sym, idx)
        if z is None:
            return 0.0  # no data → flat

        scale = self.z_base + self.z_slope * z
        return _clamp(scale, self.z_min, self.z_max)

    # ── Overlay 2: IV Percentile Rank ───────────────────────────────────────

    def _iv_percentile_scale(self, dataset: MarketDataset, sym: str, idx: int) -> float:
        """Scale based on where IV sits relative to recent history.

        High IV percentile → more premium to collect → boost
        Low IV percentile → less premium, risk of vol expansion → reduce
        """
        if not self.use_iv_pct:
            return 1.0

        iv_series = dataset.feature("iv_atm", sym)
        if iv_series is None:
            return 1.0

        start = max(0, idx - self.iv_pct_lookback)
        history = []
        for i in range(start, min(idx + 1, len(iv_series))):
            v = iv_series[i]
            if v is not None:
                history.append(float(v))

        if len(history) < 20:
            return 1.0

        current_iv = history[-1]
        pct = _percentile_rank(current_iv, history)

        if pct < self.iv_pct_low:
            # IV is low → less premium → scale down
            frac = pct / self.iv_pct_low  # 0 to 1
            return self.iv_scale_low + frac * (1.0 - self.iv_scale_low)
        elif pct > self.iv_pct_high:
            # IV is high → more premium → scale up
            frac = (pct - self.iv_pct_high) / (1.0 - self.iv_pct_high)
            return 1.0 + frac * (self.iv_scale_high - 1.0)
        else:
            return 1.0  # neutral zone

    # ── Overlay 3: Vol Regime ───────────────────────────────────────────────

    def _vol_regime_scale(self, dataset: MarketDataset, sym: str, idx: int) -> float:
        """Reduce leverage in high-vol regimes (gamma costs eat theta).

        Uses realized vol percentile rank to detect regime.
        """
        if not self.use_vol_regime:
            return 1.0

        rv_series = dataset.feature("rv_realized", sym)
        if rv_series is None:
            return 1.0

        start = max(0, idx - self.vol_regime_lookback)
        history = []
        for i in range(start, min(idx + 1, len(rv_series))):
            v = rv_series[i]
            if v is not None:
                history.append(float(v))

        if len(history) < 15:
            return 1.0

        current_rv = history[-1]
        pct = _percentile_rank(current_rv, history)

        if pct > self.vol_high_threshold:
            # High vol regime → scale down
            frac = (pct - self.vol_high_threshold) / (1.0 - self.vol_high_threshold)
            return 1.0 - frac * (1.0 - self.vol_scale_high)
        else:
            return 1.0  # normal regime

    # ── VRP Z-Score (shared) ────────────────────────────────────────────────

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
