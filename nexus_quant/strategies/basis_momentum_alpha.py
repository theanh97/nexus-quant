"""
Basis Momentum Alpha — Funding rate CHANGE predicts returns.

Hypothesis: CHANGE in funding rate predicts future returns (not level).
  - Rising funding → increasing speculation → trend continues short-term
  - Falling funding → decreasing speculation → reversal coming

Signal:
  1. Compute ΔFunding = avg_funding(recent_N) - avg_funding(older_N)
  2. Cross-sectional ranking by ΔFunding
  3. Long top-k (rising funding = trend continuation)
  4. Short bottom-k (falling funding)

Why orthogonal to V1: Funding CHANGE is different from funding LEVEL (carry).
V1 uses carry = -funding_level. This uses delta_funding = funding_change.

Parameters:
  k_per_side              int   = 2
  funding_fast_n          int   = 3     recent N settlements for fast average
  funding_slow_n          int   = 10    older N settlements for slow average
  delta_lookback_bars     int   = 72    lookback for delta funding (alternative)
  vol_lookback_bars       int   = 168
  target_gross_leverage   float = 0.30
  rebalance_interval_bars int   = 48
"""
from __future__ import annotations

import statistics
from typing import Any, Dict, List

from ._math import normalize_dollar_neutral, trailing_vol, zscores
from .base import Strategy, Weights
from ..data.schema import MarketDataset


class BasisMomentumAlphaStrategy(Strategy):
    """Basis momentum: trade funding rate changes, not levels."""

    def __init__(self, params: Dict[str, Any]) -> None:
        super().__init__(name="basis_momentum_alpha", params=params)

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
        interval = self._p("rebalance_interval_bars", 48)
        warmup = 200  # need enough funding history
        if idx <= warmup:
            return False
        return idx % max(1, interval) == 0

    def _delta_funding(self, dataset: MarketDataset, symbol: str, ts: int) -> float:
        """Compute change in funding rate: fast_avg - slow_avg."""
        fast_n = self._p("funding_fast_n", 3)
        slow_n = self._p("funding_slow_n", 10)

        fund_dict = dataset.funding.get(symbol)
        if not fund_dict:
            return 0.0

        all_times = sorted(t for t in fund_dict if t < ts)
        if len(all_times) < slow_n:
            return 0.0

        recent = all_times[-slow_n:]
        rates = [fund_dict[t] for t in recent]
        fast_avg = sum(rates[-fast_n:]) / fast_n
        slow_avg = sum(rates) / len(rates)
        return fast_avg - slow_avg

    def target_weights(self, dataset: MarketDataset, idx: int, current: Weights) -> Weights:
        syms = dataset.symbols
        k = max(1, min(self._p("k_per_side", 2), len(syms) // 2))
        target_gross = self._p("target_gross_leverage", 0.30)
        vol_lb = self._p("vol_lookback_bars", 168)

        ts = dataset.timeline[idx]

        # Signal: delta funding rate (rising → long, falling → short)
        signal: Dict[str, float] = {}
        for s in syms:
            signal[s] = self._delta_funding(dataset, s, ts)

        sz = zscores(signal)
        signal_z = {s: float(sz.get(s, 0.0)) for s in syms}

        ranked = sorted(syms, key=lambda s: signal_z.get(s, 0.0), reverse=True)
        long_syms = ranked[:k]     # rising funding → trend continuation → long
        short_syms = ranked[-k:]   # falling funding → reversal → short

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
