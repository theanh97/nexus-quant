"""
Funding Momentum Alpha — Cross-sectional ranking by perpetual funding rates.

Economic Foundation:
  Perpetual futures funding rates reflect the imbalance between longs and shorts.
  High funding (positive) = longs are paying shorts = crowded long positioning.

  Two competing hypotheses:
  1. CONTRARIAN: High funding → crowded long → reversal risk → go SHORT high-funding coins
  2. MOMENTUM: High funding → strong demand → continued outperformance → go LONG

  Academic evidence (Liu et al., 2021): Funding rate predicts FUTURE returns with
  contrarian sign — high current funding leads to lower future returns.
  → Strategy: SHORT coins with high cumulative funding, LONG coins with negative funding.

  Why NOT already captured by V1-Long:
  - V1-Long uses funding as one component with fixed weight (w_carry)
  - This signal uses CUMULATIVE funding over lookback as the ONLY ranking criterion
  - It's a "pure funding" version to test isolated funding signal strength

  Key difference from V1-Long's carry component:
  - V1: uses instantaneous funding rate (most recent bar)
  - This: cumulative funding over 48-168 bars (captures persistent over-funding)

Parameters:
  k_per_side              int   = 2
  funding_lookback_bars   int   = 168   window for cumulative funding
  direction               str   = "contrarian"  or "momentum"
  vol_lookback_bars       int   = 168
  target_gross_leverage   float = 0.25  (lower since funding signal is weaker)
  rebalance_interval_bars int   = 24    (daily rebalance — funding changes faster)
"""
from __future__ import annotations

import math
from typing import Any, Dict

from ._math import normalize_dollar_neutral, trailing_vol, zscores
from .base import Strategy, Weights
from ..data.schema import MarketDataset


class FundingMomentumAlphaStrategy(Strategy):
    """Cross-sectional momentum based on cumulative funding rates."""

    def __init__(self, params: Dict[str, Any]) -> None:
        super().__init__(name="funding_momentum_alpha", params=params)

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
        lookback = self._p("funding_lookback_bars", 168)
        interval = self._p("rebalance_interval_bars", 24)
        warmup = lookback + 10
        if idx <= warmup:
            return False
        return idx % max(1, interval) == 0

    def _cumulative_funding(self, dataset: MarketDataset, symbol: str, idx: int, lookback: int) -> float:
        """Cumulative funding rate over past lookback bars using timeline-aligned access.
        Funding settles every 8h on Binance, so sample at 8-bar intervals."""
        n_samples = max(1, lookback // 8)
        total = 0.0
        count = 0
        tl = getattr(dataset, "timeline", None)
        if tl is None:
            return 0.0
        for offset in range(n_samples):
            back_idx = max(0, idx - offset * 8)
            if back_idx >= len(tl):
                continue
            ts = tl[back_idx]
            rate = dataset.last_funding_rate_before(symbol, ts)
            if rate is not None and math.isfinite(float(rate)):
                total += float(rate)
                count += 1
        return total if count > 0 else 0.0

    def target_weights(self, dataset: MarketDataset, idx: int, current: Weights) -> Weights:
        syms = dataset.symbols
        k = max(1, min(self._p("k_per_side", 2), len(syms) // 2))
        funding_lb = self._p("funding_lookback_bars", 168)
        direction = self._p("direction", "contrarian")
        vol_lb = self._p("vol_lookback_bars", 168)
        target_gross = self._p("target_gross_leverage", 0.25)

        # Check funding and timeline availability
        if not hasattr(dataset, "funding") or not hasattr(dataset, "timeline"):
            return {s: 0.0 for s in syms}
        if not any(dataset.funding.get(s) for s in syms):
            return {s: 0.0 for s in syms}

        fund_scores: Dict[str, float] = {}
        for s in syms:
            fund_scores[s] = self._cumulative_funding(dataset, s, idx, funding_lb)

        # Rank: contrarian = short high-funding (longs to be squeezed)
        #        momentum = long high-funding (strong demand continues)
        sign = -1.0 if direction == "contrarian" else 1.0

        score = {}
        raw_z = zscores(fund_scores)
        for s in syms:
            score[s] = sign * float(raw_z.get(s, 0.0))

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
