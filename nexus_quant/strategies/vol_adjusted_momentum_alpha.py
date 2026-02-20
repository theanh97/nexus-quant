"""
Vol-Adjusted Momentum Alpha — Cross-sectional ranking by endpoint return / realized vol.

Economic Foundation:
  This is the Information Ratio (IR) decomposition of momentum:
    score = total_return_over_lb / trailing_realized_vol

  Distinct from SR Alpha (Sharpe Ratio Alpha):
    - SR Alpha  : mean(bar_returns) / stdev(bar_returns) * sqrt(N) → path Sharpe
    - Vol-Adj Mom: endpoint_return / realized_vol → endpoint IR

  These diverge when: a coin has volatile intra-period price swings but strong net return
  (high vol-adj score) vs steady trending behavior (high SR score).

  Intuition:
    - SR Alpha rewards consistency of the trend (low path variance, steady gains)
    - Vol-Adj Momentum rewards efficiency of the overall gain (high net return per unit of risk)
    - A coin that crashes 30% then recovers 50% has high vol-adj score but low SR score

  Expected correlation with SR Alpha: ~0.65-0.80 (same lookback)
  Expected correlation with Pure Momentum: ~0.70-0.85 (same numerator, vol-normalizes denominator)

  Key question: Does dividing by realized vol improve rank stability over raw momentum?
  If yes → vol-normalization helps even at the return level (not just per-bar level)
  If no → vol normalization is already captured by SR Alpha

Parameters:
  k_per_side              int   = 2
  lookback_bars           int   = 437   consistent with discovered optimum
  vol_lookback_bars       int   = 168   for position sizing inv-vol weights
  signal_vol_bars         int   = 437   vol window for signal construction (same as return lookback)
  target_gross_leverage   float = 0.30
  rebalance_interval_bars int   = 48
"""
from __future__ import annotations

import math
from typing import Any, Dict

from ._math import normalize_dollar_neutral, trailing_vol, zscores
from .base import Strategy, Weights
from ..data.schema import MarketDataset


class VolAdjustedMomentumAlphaStrategy(Strategy):
    """Cross-sectional ranking by total_return / realized_vol (endpoint IR)."""

    def __init__(self, params: Dict[str, Any]) -> None:
        super().__init__(name="vol_adjusted_momentum_alpha", params=params)

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
            if isinstance(default, str):
                return str(v)
        except (TypeError, ValueError):
            pass
        return v

    def should_rebalance(self, dataset: MarketDataset, idx: int) -> bool:
        lookback = self._p("lookback_bars", 437)
        interval = self._p("rebalance_interval_bars", 48)
        warmup = lookback + 10
        if idx <= warmup:
            return False
        return idx % max(1, interval) == 0

    def _vol_adjusted_score(self, closes, idx: int, lookback: int, sig_vol_bars: int) -> float:
        """Compute total_return / realized_vol over lookback bars."""
        i1 = min(idx - 1, len(closes) - 1)
        i0 = max(0, idx - 1 - lookback)
        if i0 >= i1 or i0 >= len(closes) or i1 >= len(closes):
            return 0.0
        c0 = float(closes[i0])
        c1 = float(closes[i1])
        if c0 <= 0:
            return 0.0
        total_ret = (c1 / c0) - 1.0
        # Realized vol over signal window (same or separate window)
        rv = trailing_vol(closes, end_idx=idx, lookback_bars=sig_vol_bars)
        if rv <= 0 or not math.isfinite(rv):
            # Fallback to raw return if vol is zero/invalid
            return total_ret
        return total_ret / rv

    def target_weights(self, dataset: MarketDataset, idx: int, current: Weights) -> Weights:
        syms = dataset.symbols
        k = max(1, min(self._p("k_per_side", 2), len(syms) // 2))
        lookback = self._p("lookback_bars", 437)
        sig_vol_bars = self._p("signal_vol_bars", 437)
        vol_lb = self._p("vol_lookback_bars", 168)
        target_gross = self._p("target_gross_leverage", 0.30)

        raw_scores: Dict[str, float] = {}
        for s in syms:
            raw_scores[s] = self._vol_adjusted_score(
                dataset.perp_close[s], idx, lookback, sig_vol_bars
            )

        score = {s: float(v) for s, v in zscores(raw_scores).items()}

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
