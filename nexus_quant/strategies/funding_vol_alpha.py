"""
Funding Volatility Alpha — Cross-sectional ranking by funding rate VOLATILITY.

Economic Foundation:
  Perpetual funding rates reflect crowded positioning. But while fund_cont_168
  captures the LEVEL of crowding (cumulative sum), this captures the DISPERSION
  (standard deviation) of funding rates over a lookback period.

  High funding rate volatility = funding was swinging wildly = UNSTABLE crowding:
  - Positions are being opened and squeezed more frequently
  - Less persistence in the crowding regime
  - This could signal: after a squeeze, the next period is calmer (reversion)

  Two competing hypotheses:
  1. CONTRARIAN: High funding vol → crowded positions were squeezed → now cleaner
     → LONG high-funding-vol coins (they just got squeezed, now at fair value)
  2. MOMENTUM: High funding vol → unstable structure, still being squeezed
     → SHORT high-funding-vol coins (more instability coming)

  Academic: Related to "volatility of illiquidity premium" in traditional markets.
  High premium volatility often predicts mean reversion of the premium.
  → We test CONTRARIAN direction first.

Parameters:
  k_per_side                int   = 2
  funding_lookback_bars     int   = 168   window for computing funding rate volatility
  direction                 str   = "contrarian"  or "momentum"
  vol_lookback_bars         int   = 168
  target_gross_leverage     float = 0.25
  rebalance_interval_bars   int   = 24
"""
from __future__ import annotations

import math
from typing import Any, Dict

from ._math import normalize_dollar_neutral, trailing_vol, zscores
from .base import Strategy, Weights
from ..data.schema import MarketDataset


class FundingVolAlphaStrategy(Strategy):
    """Funding rate volatility cross-sectional alpha.

    Ranks coins by the standard deviation of their funding rates over
    the lookback window. Contrarian direction = long coins where funding
    has been most volatile (they've been squeezed already).
    """

    def __init__(self, params: Dict[str, Any]) -> None:
        super().__init__(name="funding_vol_alpha", params=params)

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

    def _funding_volatility(
        self, dataset: MarketDataset, symbol: str, idx: int, lookback: int,
    ) -> float:
        """Standard deviation of funding rates over lookback bars."""
        n_samples = max(2, lookback // 8)
        rates = []
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
                rates.append(float(rate))

        if len(rates) < 3:
            return 0.0
        mu = sum(rates) / len(rates)
        var = sum((r - mu) ** 2 for r in rates) / len(rates)
        return math.sqrt(var) if var > 0 else 0.0

    def target_weights(self, dataset: MarketDataset, idx: int, current: Weights) -> Weights:
        syms = dataset.symbols
        k = max(1, min(self._p("k_per_side", 2), len(syms) // 2))
        funding_lb = self._p("funding_lookback_bars", 168)
        direction = self._p("direction", "contrarian")
        vol_lb = self._p("vol_lookback_bars", 168)
        target_gross = self._p("target_gross_leverage", 0.25)

        if not hasattr(dataset, "funding") or not hasattr(dataset, "timeline"):
            return {s: 0.0 for s in syms}
        if not any(dataset.funding.get(s) for s in syms):
            return {s: 0.0 for s in syms}

        # Compute funding rate volatility per symbol
        vol_scores: Dict[str, float] = {}
        for s in syms:
            vol_scores[s] = self._funding_volatility(dataset, s, idx, funding_lb)

        # Rank by volatility
        # contrarian = LONG high-vol (squeezed, now clean) → score = +vol
        # momentum   = SHORT high-vol (still unstable) → score = -vol
        sign = 1.0 if direction == "contrarian" else -1.0

        raw_z = zscores(vol_scores)
        score = {s: sign * float(raw_z.get(s, 0.0)) for s in syms}

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
