"""
Pair Spread Mean-Reversion Alpha — Statistical Arbitrage
=========================================================
Economic Foundation:
  Crypto alts are highly correlated (BTC beta ~0.8-1.5). When a PAIR of coins
  diverges from its historical spread (e.g., SOL outperforms ETH by 2σ over
  the trailing window), mean-reversion implies convergence.

  This is fundamentally different from cross-sectional momentum:
  - Momentum ranks ALL coins and goes long/short the extremes
  - Pair spread finds DIVERGENT PAIRS and bets on convergence
  - The signal is the Z-SCORE OF THE SPREAD, not the direction of returns

Strategy:
  1. For each pair (i, j): compute log price ratio = log(P_i / P_j)
  2. Rolling z-score of the spread over lookback_bars
  3. If z > threshold: short i, long j (spread will contract)
  4. If z < -threshold: long i, short j (spread will expand)
  5. Aggregate pair signals into per-symbol scores
  6. Normalize to dollar-neutral portfolio

Parameters:
  lookback_bars           int   = 168   Rolling window for mean/std of spread
  entry_z                 float = 1.5   Z-score threshold to enter
  vol_lookback_bars       int   = 168
  target_gross_leverage   float = 0.30
  rebalance_interval_bars int   = 24
"""
from __future__ import annotations

import math
import statistics
from typing import Any, Dict, List

from ._math import normalize_dollar_neutral, trailing_vol
from .base import Strategy, Weights
from ..data.schema import MarketDataset


class PairSpreadAlphaStrategy(Strategy):
    """Mean-reversion on pair spreads — statistical arbitrage."""

    def __init__(self, params: Dict[str, Any]) -> None:
        super().__init__(name="pair_spread_alpha", params=params)

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
        lookback = self._p("lookback_bars", 168)
        interval = self._p("rebalance_interval_bars", 24)
        if idx <= lookback + 10:
            return False
        return idx % max(1, interval) == 0

    def target_weights(self, dataset: MarketDataset, idx: int, current: Weights) -> Weights:
        syms = dataset.symbols
        lookback = self._p("lookback_bars", 168)
        entry_z = self._p("entry_z", 1.5)
        vol_lb = self._p("vol_lookback_bars", 168)
        target_gross = self._p("target_gross_leverage", 0.30)

        # Compute log price ratios for all pairs
        scores: Dict[str, float] = {s: 0.0 for s in syms}
        n_pairs = 0

        for i in range(len(syms)):
            for j in range(i + 1, len(syms)):
                s_i, s_j = syms[i], syms[j]
                closes_i = dataset.perp_close[s_i]
                closes_j = dataset.perp_close[s_j]

                # Build spread = log(P_i / P_j) over lookback
                start = max(0, idx - lookback)
                end = idx
                if end > len(closes_i) or end > len(closes_j):
                    continue

                spread = []
                for t in range(start, end):
                    if closes_i[t] > 0 and closes_j[t] > 0:
                        spread.append(math.log(closes_i[t] / closes_j[t]))

                if len(spread) < 20:
                    continue

                # Z-score of current spread
                mu = sum(spread) / len(spread)
                sd = statistics.pstdev(spread)
                if sd < 1e-10:
                    continue

                z = (spread[-1] - mu) / sd
                n_pairs += 1

                # Signal: if z > threshold, spread is too wide → short i, long j
                if z > entry_z:
                    scores[s_i] -= z
                    scores[s_j] += z
                elif z < -entry_z:
                    scores[s_i] -= z  # z is negative, so this adds
                    scores[s_j] += z  # this subtracts

        if n_pairs == 0 or all(v == 0.0 for v in scores.values()):
            return {s: 0.0 for s in syms}

        # Rank by aggregated pair scores
        k = max(1, min(self._p("k_per_side", 2), len(syms) // 2))
        ranked = sorted(syms, key=lambda s: scores[s], reverse=True)

        # Only take positions if there's enough signal
        long_syms = [s for s in ranked[:k] if scores[s] > 0]
        short_syms = [s for s in ranked[-k:] if scores[s] < 0]

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
