"""
Dispersion Alpha — Cross-sectional dispersion timing strategy.

Economic Foundation:
  Cross-sectional return dispersion measures how differently assets are
  performing relative to each other. High dispersion = high idiosyncratic
  moves = good environment for stock-picking / factor strategies.
  Low dispersion = everything moves together = beta-driven market.

  In crypto, dispersion tends to be HIGH during:
  - Recovery/rotation periods (some coins rally, others lag)
  - Alt-season (alts outperform BTC)
  - Choppy/sideways markets with sector rotation

  Dispersion tends to be LOW during:
  - Sharp sell-offs (everything drops together)
  - Strong bull runs (everything pumps together)

Signal:
  1. Compute trailing cross-sectional dispersion of returns (72h window)
  2. When dispersion is above median → use momentum + MR signals
  3. When dispersion is below median → reduce exposure (low conviction)
  4. Score = momentum(72h) z-score * dispersion_weight
     (shorter horizon than V1's 168h — targets faster rotations)

  This strategy is complementary to V1 because:
  - V1 uses 168h momentum (slow trend following)
  - Dispersion Alpha uses 72h momentum (faster rotation capture)
  - V1 reduces leverage in bear; Dispersion reduces in low-dispersion
  - In 2025 sideways: dispersion may still be high → Dispersion trades
    while V1's momentum filter is flat

Parameters:
  k_per_side                int   = 2
  momentum_lookback_bars    int   = 72     (3 days — faster rotation)
  dispersion_lookback_bars  int   = 72     (3 days for dispersion calc)
  dispersion_threshold      float = 0.0    (0 = always trade; >0 = minimum dispersion)
  target_gross_leverage     float = 0.25   (lower than V1 — complementary allocation)
  rebalance_interval_bars   int   = 24     (daily)
"""
from __future__ import annotations

import statistics
from typing import Any, Dict, List

from ._math import normalize_dollar_neutral, trailing_vol, zscores
from .base import Strategy, Weights
from ..data.schema import MarketDataset


class DispersionAlphaStrategy(Strategy):
    """Cross-sectional dispersion timing: trade when assets diverge."""

    def __init__(self, params: Dict[str, Any]) -> None:
        super().__init__(name="dispersion_alpha", params=params)

    def _p(self, key: str, default: Any) -> Any:
        v = self.params.get(key)
        if v is None:
            return default
        try:
            if isinstance(default, int):
                return int(v)
            if isinstance(default, float):
                return float(v)
        except (TypeError, ValueError):
            pass
        return v

    def should_rebalance(self, dataset: MarketDataset, idx: int) -> bool:
        mom_lb = self._p("momentum_lookback_bars", 72)
        disp_lb = self._p("dispersion_lookback_bars", 72)
        interval = self._p("rebalance_interval_bars", 24)
        warmup = max(mom_lb, disp_lb) + 10
        if idx <= warmup:
            return False
        return idx % max(1, interval) == 0

    def _compute_dispersion(self, dataset: MarketDataset, idx: int) -> float:
        """Cross-sectional return dispersion over lookback window."""
        disp_lb = self._p("dispersion_lookback_bars", 72)
        syms = dataset.symbols
        if idx < disp_lb + 1:
            return 0.0

        # Compute return for each symbol over lookback
        rets: List[float] = []
        for s in syms:
            c = dataset.perp_close[s]
            i1 = min(idx - 1, len(c) - 1)
            i0 = max(0, idx - 1 - disp_lb)
            c1 = float(c[i1]) if i1 >= 0 else 0.0
            c0 = float(c[i0]) if i0 >= 0 and i0 < len(c) else 0.0
            if c0 > 0:
                rets.append(c1 / c0 - 1.0)

        if len(rets) < 3:
            return 0.0
        return statistics.pstdev(rets)

    def _compute_dispersion_zscore(self, dataset: MarketDataset, idx: int) -> float:
        """Z-score of current dispersion vs rolling history."""
        disp_lb = self._p("dispersion_lookback_bars", 72)
        history_window = 720  # 30 days of dispersion history

        dispersions: List[float] = []
        for t in range(max(disp_lb + 10, idx - history_window), idx + 1):
            d = 0.0
            rets = []
            for s in dataset.symbols:
                c = dataset.perp_close[s]
                i1 = min(t - 1, len(c) - 1)
                i0 = max(0, t - 1 - disp_lb)
                c1 = float(c[i1]) if i1 >= 0 else 0.0
                c0 = float(c[i0]) if i0 >= 0 and i0 < len(c) else 0.0
                if c0 > 0:
                    rets.append(c1 / c0 - 1.0)
            if len(rets) >= 3:
                d = statistics.pstdev(rets)
            dispersions.append(d)

        if len(dispersions) < 10:
            return 0.0

        mu = statistics.mean(dispersions)
        sd = statistics.pstdev(dispersions)
        if sd < 1e-10:
            return 0.0
        return (dispersions[-1] - mu) / sd

    def target_weights(self, dataset: MarketDataset, idx: int, current: Weights) -> Weights:
        syms = dataset.symbols
        k = max(1, min(self._p("k_per_side", 2), len(syms) // 2))
        mom_lb = self._p("momentum_lookback_bars", 72)
        target_gross = self._p("target_gross_leverage", 0.25)
        disp_thr = self._p("dispersion_threshold", 0.0)

        # Compute current dispersion
        disp = self._compute_dispersion(dataset, idx)

        # If dispersion is below threshold → zero exposure (no edge)
        if disp_thr > 0 and disp < disp_thr:
            return {s: 0.0 for s in syms}

        # Scale leverage by dispersion z-score (high dispersion → more leverage)
        disp_z = self._compute_dispersion_zscore(dataset, idx)
        # Map z-score to leverage scale: z=0 → 0.6x, z=1 → 1.0x, z=-1 → 0.2x
        scale = max(0.1, min(1.5, 0.6 + 0.4 * disp_z))
        effective_leverage = target_gross * scale

        # Signal: short-term momentum (72h)
        mom_raw: Dict[str, float] = {}
        for s in syms:
            c = dataset.perp_close[s]
            i1 = min(idx - 1, len(c) - 1)
            i0 = max(0, idx - 1 - mom_lb)
            c1 = float(c[i1]) if i1 >= 0 else 0.0
            c0 = float(c[i0]) if i0 >= 0 and i0 < len(c) else 0.0
            mom_raw[s] = (c1 / c0 - 1.0) if c0 > 0 else 0.0

        mz = zscores(mom_raw)
        signal = {s: float(mz.get(s, 0.0)) for s in syms}

        ranked = sorted(syms, key=lambda s: signal.get(s, 0.0), reverse=True)
        long_syms = ranked[:k]
        short_syms = ranked[-k:]

        long_syms = [s for s in long_syms if s not in short_syms]
        short_syms = [s for s in short_syms if s not in long_syms]

        if not long_syms or not short_syms:
            return {s: 0.0 for s in syms}

        inv_vol: Dict[str, float] = {}
        vol_lb = self._p("momentum_lookback_bars", 72)
        for s in set(long_syms) | set(short_syms):
            v = trailing_vol(dataset.perp_close[s], end_idx=idx, lookback_bars=vol_lb)
            inv_vol[s] = (1.0 / v) if v > 0 else 1.0

        w = normalize_dollar_neutral(long_syms, short_syms, inv_vol, effective_leverage)
        out = {s: 0.0 for s in syms}
        out.update(w)
        return out
