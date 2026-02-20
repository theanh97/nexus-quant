"""
Mean-Reversion + Funding Filter Alpha — Buy dips in underfunded assets.

Hypothesis: Short-term losers with LOW funding = genuine dip → buy.
            Short-term losers with HIGH funding = still crowded → don't buy.
            Short-term winners with HIGH funding = overbought → sell.

Combines mean-reversion (price reversal) with funding as a quality filter.
Only trades when funding confirms the MR signal.

Why orthogonal to V1: V1 uses carry and momentum as ADDITIVE signals.
This uses funding as a GATE on mean-reversion (multiplicative filter).

Parameters:
  k_per_side              int   = 2
  mr_lookback_bars        int   = 48    short-term return for MR signal
  vol_lookback_bars       int   = 168
  target_gross_leverage   float = 0.30
  rebalance_interval_bars int   = 24
"""
from __future__ import annotations

from typing import Any, Dict

from ._math import normalize_dollar_neutral, trailing_vol, zscores
from .base import Strategy, Weights
from ..data.schema import MarketDataset


class MeanReversionFundingAlphaStrategy(Strategy):
    """MR + funding filter: buy dips in underfunded, sell rips in overfunded."""

    def __init__(self, params: Dict[str, Any]) -> None:
        super().__init__(name="mr_funding_alpha", params=params)

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
        mr_lb = self._p("mr_lookback_bars", 48)
        interval = self._p("rebalance_interval_bars", 24)
        warmup = max(mr_lb, 168) + 10
        if idx <= warmup:
            return False
        return idx % max(1, interval) == 0

    def target_weights(self, dataset: MarketDataset, idx: int, current: Weights) -> Weights:
        syms = dataset.symbols
        k = max(1, min(self._p("k_per_side", 2), len(syms) // 2))
        mr_lb = self._p("mr_lookback_bars", 48)
        target_gross = self._p("target_gross_leverage", 0.30)
        vol_lb = self._p("vol_lookback_bars", 168)

        ts = dataset.timeline[idx]

        # Funding z-score
        f_raw = {s: float(dataset.last_funding_rate_before(s, ts)) for s in syms}
        fz = zscores(f_raw)

        # Short-term return (for MR)
        mr_raw: Dict[str, float] = {}
        for s in syms:
            c = dataset.perp_close[s]
            i1 = min(idx - 1, len(c) - 1)
            i0 = max(0, idx - 1 - mr_lb)
            c1 = float(c[i1]) if i1 >= 0 else 0.0
            c0 = float(c[i0]) if i0 >= 0 and i0 < len(c) else 0.0
            mr_raw[s] = (c1 / c0 - 1.0) if c0 > 0 else 0.0
        mr_z = zscores(mr_raw)

        # Signal = MR (inverted return) × funding quality filter
        # MR signal: recent losers → positive (buy dip)
        # Funding filter: low funding → positive (not crowded)
        # Combined: buy losers with low funding, short winners with high funding
        signal: Dict[str, float] = {}
        for s in syms:
            mr_signal = -float(mr_z.get(s, 0.0))      # inverted: losers = positive
            funding_signal = -float(fz.get(s, 0.0))    # inverted: low funding = positive

            # Multiplicative gating: signal is strong only when both agree
            # If both positive (loser + low funding) → strong buy
            # If both negative (winner + high funding) → strong sell
            if mr_signal * funding_signal > 0:
                signal[s] = mr_signal + funding_signal  # reinforcing
            else:
                signal[s] = 0.3 * (mr_signal + funding_signal)  # dampened when conflicting

        sz = zscores(signal)
        signal_z = {s: float(sz.get(s, 0.0)) for s in syms}

        ranked = sorted(syms, key=lambda s: signal_z.get(s, 0.0), reverse=True)
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
