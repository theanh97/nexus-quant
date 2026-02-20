"""
EWMA Sharpe Ratio Alpha — Exponentially Weighted Moving Average Sharpe.

Economic Foundation:
  Plain Sharpe Ratio Alpha (Phase 60) uses equal-weight over all N bars.
  EWMA weighting decays older returns exponentially — recent returns get
  more weight, making the signal more responsive to regime changes while
  still using a long lookback window for stability.

  For crypto markets that shift quickly, EWMA is often superior to
  equal-weight because it adapts faster to changing momentum regimes.

EWMA formula:
  weighted_mean = sum(lambda^(N-1-t) * r_t) / sum(lambda^(N-1-t))
  where lambda ∈ (0,1) controls the decay rate
  half-life ≈ log(0.5) / log(lambda)

  lambda=0.99 → half-life ~69 bars (≈3 days hourly)
  lambda=0.98 → half-life ~34 bars (≈1.4 days hourly)
  lambda=0.97 → half-life ~23 bars (≈1 day hourly)

Signal:
  1. Compute EWMA mean and EWMA variance of hourly returns per symbol
  2. EWMA Sharpe = ewma_mean / sqrt(ewma_variance) × sqrt(N)
  3. Cross-sectional z-score → rank by EWMA Sharpe
  4. Long top-k, Short bottom-k, inverse-vol weighted

Parameters:
  k_per_side              int   = 2
  lookback_bars           int   = 336   (same window as sr_336h)
  ewma_lambda             float = 0.98  (decay factor; 0.99=slow, 0.95=fast)
  vol_lookback_bars       int   = 168
  target_gross_leverage   float = 0.35
  rebalance_interval_bars int   = 48
"""
from __future__ import annotations

import math
from typing import Any, Dict, List

from ._math import normalize_dollar_neutral, trailing_vol, zscores
from .base import Strategy, Weights
from ..data.schema import MarketDataset


class EWMASharpeAlphaStrategy(Strategy):
    """EWMA Sharpe Ratio: exponentially weighted risk-adjusted momentum."""

    def __init__(self, params: Dict[str, Any]) -> None:
        super().__init__(name="ewma_sharpe_alpha", params=params)

    def _p(self, key: str, default: Any) -> Any:
        v = self.params.get(key)
        if v is None:
            return default
        try:
            if isinstance(default, bool):
                return bool(v)
            if isinstance(default, int):
                return int(v)
            if isinstance(default, float):
                return float(v)
        except (TypeError, ValueError):
            pass
        return v

    def should_rebalance(self, dataset: MarketDataset, idx: int) -> bool:
        lookback = self._p("lookback_bars", 336)
        interval = self._p("rebalance_interval_bars", 48)
        warmup = lookback + 10
        if idx <= warmup:
            return False
        return idx % max(1, interval) == 0

    def _ewma_sharpe(self, closes: List[float], idx: int, lookback: int, lam: float) -> float:
        """
        Compute EWMA Sharpe ratio over past lookback bars.
        lam is the exponential decay factor (e.g. 0.98).
        """
        start = max(1, idx - lookback)
        end = min(idx, len(closes))

        # Compute returns and EWMA weights
        rets: List[float] = []
        for i in range(start, end):
            c0 = closes[i - 1]
            c1 = closes[i]
            if c0 > 0:
                rets.append((c1 / c0) - 1.0)

        n = len(rets)
        if n < 10:
            return 0.0

        # EWMA weights: most recent gets weight 1, older gets lambda, lambda^2, ...
        # Weight for index i (0=oldest, n-1=newest): lambda^(n-1-i)
        weights = [lam ** (n - 1 - i) for i in range(n)]
        w_sum = sum(weights)

        if w_sum < 1e-12:
            return 0.0

        # Weighted mean
        w_mean = sum(weights[i] * rets[i] for i in range(n)) / w_sum

        # Weighted variance
        w_var = sum(weights[i] * (rets[i] - w_mean) ** 2 for i in range(n)) / w_sum

        if w_var < 1e-16:
            return 0.0

        # Sharpe proxy (annualized with sqrt(n) ≈ sqrt(8760/1) for hourly)
        # Use sqrt(n) to normalize to the window length
        return (w_mean / math.sqrt(w_var)) * math.sqrt(n)

    def target_weights(self, dataset: MarketDataset, idx: int, current: Weights) -> Weights:
        syms = dataset.symbols
        k = max(1, min(self._p("k_per_side", 2), len(syms) // 2))
        lookback = self._p("lookback_bars", 336)
        lam = self._p("ewma_lambda", 0.98)
        vol_lb = self._p("vol_lookback_bars", 168)
        target_gross = self._p("target_gross_leverage", 0.35)

        # EWMA Sharpe per symbol
        sharpe_raw: Dict[str, float] = {}
        for s in syms:
            sharpe_raw[s] = self._ewma_sharpe(dataset.perp_close[s], idx, lookback, lam)

        # Cross-sectional z-score
        sharpe_z = zscores(sharpe_raw)
        score = {s: float(sharpe_z.get(s, 0.0)) for s in syms}

        ranked = sorted(syms, key=lambda s: score[s], reverse=True)
        long_syms = ranked[:k]
        short_syms = ranked[-k:]

        long_syms = [s for s in long_syms if s not in short_syms]
        short_syms = [s for s in short_syms if s not in long_syms]

        if not long_syms or not short_syms:
            return {s: 0.0 for s in syms}

        inv_vol: Dict[str, float] = {}
        for s in set(long_syms) | set(short_syms):
            v = trailing_vol(dataset.perp_close[s], end_idx=idx, lookback_bars=vol_lb)
            inv_vol[s] = (1.0 / v) if v > 0 else 1.0

        w = normalize_dollar_neutral(long_syms, short_syms, inv_vol, target_gross)
        out = {s: 0.0 for s in syms}
        out.update(w)
        return out
