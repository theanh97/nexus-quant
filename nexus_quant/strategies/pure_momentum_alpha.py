"""
Pure Momentum Alpha — Raw cross-sectional return momentum without adjustments.

Economic Foundation:
  The simplest momentum signal: rank coins by cumulative return over N bars.
  No beta-hedging (unlike Idio Momentum), no Sharpe normalization (unlike SR Alpha).

  This serves as the baseline comparator:
    - vs Idio Momentum: tests whether beta-hedging helps
    - vs Sharpe Ratio Alpha: tests whether vol normalization helps
    - vs V1-Long: tests isolated momentum without carry/MR components

  Academic consensus: Pure price momentum (Jegadeesh & Titman 1993) is one of the
  most robust anomalies in financial markets. In crypto: Liu et al. (2022) confirm
  3-week cross-sectional momentum with highest significance.

  If this outperforms Idio Momentum → beta-hedging actually hurts
  If this underperforms → beta-hedging genuinely adds value

Parameters:
  k_per_side              int   = 2
  lookback_bars           int   = 437
  vol_lookback_bars       int   = 168
  target_gross_leverage   float = 0.30
  rebalance_interval_bars int   = 48
"""
from __future__ import annotations

from typing import Any, Dict

from ._math import normalize_dollar_neutral, trailing_vol, zscores
from .base import Strategy, Weights
from ..data.schema import MarketDataset


class PureMomentumAlphaStrategy(Strategy):
    """Pure raw cross-sectional price momentum."""

    def __init__(self, params: Dict[str, Any]) -> None:
        super().__init__(name="pure_momentum_alpha", params=params)

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
        lookback = self._p("lookback_bars", 437)
        interval = self._p("rebalance_interval_bars", 48)
        warmup = lookback + 10
        if idx <= warmup:
            return False
        return idx % max(1, interval) == 0

    def _total_return(self, closes, idx: int, lookback: int) -> float:
        i1 = min(idx - 1, len(closes) - 1)
        i0 = max(0, idx - 1 - lookback)
        c1 = float(closes[i1]) if i1 >= 0 else 0.0
        c0 = float(closes[i0]) if i0 >= 0 and i0 < len(closes) else 0.0
        return (c1 / c0 - 1.0) if c0 > 0 else 0.0

    def target_weights(self, dataset: MarketDataset, idx: int, current: Weights) -> Weights:
        syms = dataset.symbols
        k = max(1, min(self._p("k_per_side", 2), len(syms) // 2))
        lookback = self._p("lookback_bars", 437)
        vol_lb = self._p("vol_lookback_bars", 168)
        target_gross = self._p("target_gross_leverage", 0.30)

        raw_scores = {s: self._total_return(dataset.perp_close[s], idx, lookback) for s in syms}
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
