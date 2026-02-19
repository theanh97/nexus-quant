"""
NEXUS Orderflow Alpha V1 — Taker Buy/Sell Volume Imbalance Strategy.

Alpha Source: Orderflow Imbalance
─────────────────────────────────
Binance klines include taker buy base volume (index 9) alongside total volume
(index 5). This gives us a free, granular measure of aggressive buying vs
selling pressure at each bar:

    taker_sell_vol = total_vol - taker_buy_vol
    imbalance      = (taker_buy_vol - taker_sell_vol) / total_vol
                   = (2 * taker_buy_vol / total_vol) - 1

imbalance ranges from -1 (all aggressive selling) to +1 (all aggressive buying).

Strategy:
  1. Compute rolling N-bar average orderflow imbalance per symbol.
  2. Optionally confirm with a momentum filter (168h trend direction):
     only go LONG on symbols with positive momentum, SHORT on negative momentum.
  3. Cross-sectional ranking: LONG top-k imbalance, SHORT bottom-k imbalance.
  4. Dollar-neutral, inverse-vol weighted, with configurable leverage.

This signal is ORTHOGONAL to carry, momentum, and mean-reversion — it captures
microstructure information about who is crossing the spread.

Parameters
──────────
orderflow_lookback_bars     : int   = 48    rolling window for imbalance
momentum_filter_bars        : int   = 168   momentum lookback for confirmation
use_momentum_filter         : bool  = True  require momentum agreement?
k_per_side                  : int   = 2     positions per side
target_gross_leverage       : float = 0.35  total gross leverage
vol_lookback_bars           : int   = 168   for inverse-vol weighting
rebalance_interval_bars     : int   = 24    rebalance frequency
"""
from __future__ import annotations

import math
import statistics
from typing import Any, Dict, List, Optional

from ._math import normalize_dollar_neutral, trailing_vol, zscores
from .base import Strategy, Weights
from ..data.schema import MarketDataset


class OrderflowAlphaV1Strategy(Strategy):
    """
    Orderflow Alpha V1 — Taker Buy/Sell Volume Imbalance.

    Ranks symbols by rolling average orderflow imbalance (aggressive buy vs sell
    pressure). Optionally filters by momentum direction for confirmation.
    """

    def __init__(self, params: Dict[str, Any]) -> None:
        super().__init__(name="orderflow_alpha_v1", params=params)

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

    # ------------------------------------------------------------------
    # Orderflow imbalance computation
    # ------------------------------------------------------------------

    def _rolling_imbalance(
        self,
        dataset: MarketDataset,
        symbol: str,
        idx: int,
        lookback: int,
    ) -> float:
        """
        Compute rolling average orderflow imbalance over the last `lookback` bars.

        imbalance_bar = (2 * taker_buy_vol / total_vol) - 1
        Returns mean imbalance over [idx-lookback, idx).

        If taker_buy_volume is not available, falls back to 0.0.
        """
        if dataset.taker_buy_volume is None or dataset.perp_volume is None:
            return 0.0

        tbv = dataset.taker_buy_volume.get(symbol)
        vol = dataset.perp_volume.get(symbol)
        if tbv is None or vol is None:
            return 0.0

        start = max(0, idx - lookback)
        end = idx  # exclusive (current bar not included — avoid lookahead)

        if start >= end:
            return 0.0

        imbalances: List[float] = []
        for i in range(start, end):
            if i >= len(vol) or i >= len(tbv):
                continue
            total_v = vol[i]
            taker_buy_v = tbv[i]
            if total_v <= 0.0:
                continue
            # imbalance = (taker_buy - taker_sell) / total
            # = (taker_buy - (total - taker_buy)) / total
            # = (2 * taker_buy / total) - 1
            imb = (2.0 * taker_buy_v / total_v) - 1.0
            # Clamp to [-1, 1] for safety
            imb = max(-1.0, min(1.0, imb))
            imbalances.append(imb)

        if not imbalances:
            return 0.0

        return sum(imbalances) / len(imbalances)

    # ------------------------------------------------------------------
    # Rebalance timing
    # ------------------------------------------------------------------

    def should_rebalance(self, dataset: MarketDataset, idx: int) -> bool:
        of_lb = self._p("orderflow_lookback_bars", 48)
        mom_lb = self._p("momentum_filter_bars", 168)
        interval = self._p("rebalance_interval_bars", 24)
        warmup = max(of_lb, mom_lb) + 2
        if idx <= warmup:
            return False
        return idx % max(1, interval) == 0

    # ------------------------------------------------------------------
    # Target weights
    # ------------------------------------------------------------------

    def target_weights(
        self,
        dataset: MarketDataset,
        idx: int,
        current: Weights,
    ) -> Weights:
        syms = dataset.symbols
        k = max(1, min(self._p("k_per_side", 2), len(syms) // 2))

        of_lb       = self._p("orderflow_lookback_bars", 48)
        mom_lb      = self._p("momentum_filter_bars", 168)
        use_mom     = self._p("use_momentum_filter", True)
        vol_lb      = self._p("vol_lookback_bars", 168)
        target_gross = self._p("target_gross_leverage", 0.35)

        # ── Signal 1: Orderflow imbalance ──────────────────────────────
        of_raw: Dict[str, float] = {}
        for s in syms:
            of_raw[s] = self._rolling_imbalance(dataset, s, idx, of_lb)

        of_z = zscores(of_raw)
        orderflow = {s: float(of_z.get(s, 0.0)) for s in syms}

        # ── Signal 2 (optional): Momentum filter ──────────────────────
        # Compute raw momentum return over mom_lb bars
        mom_raw: Dict[str, float] = {}
        for s in syms:
            c = dataset.perp_close[s]
            i1 = min(idx - 1, len(c) - 1)
            i0 = max(0, idx - 1 - mom_lb)
            c1 = c[i1] if i1 >= 0 else 0.0
            c0 = c[i0] if i0 >= 0 and i0 < len(c) else 0.0
            mom_raw[s] = (c1 / c0 - 1.0) if c0 > 0 else 0.0

        # ── Composite scoring ──────────────────────────────────────────
        # Primary signal = orderflow imbalance z-score
        # If momentum filter is on, we penalize conflicting signals:
        # LONG candidate must have positive momentum, SHORT must have negative
        score: Dict[str, float] = {}
        for s in syms:
            score[s] = orderflow[s]

        # Rank by composite score
        ranked = sorted(syms, key=lambda s: score[s], reverse=True)

        if use_mom:
            # LONG: top orderflow + positive momentum
            long_cands = [s for s in ranked if mom_raw[s] > 0.0][:k]
            # SHORT: bottom orderflow + negative momentum
            short_cands = [s for s in reversed(ranked) if mom_raw[s] < 0.0][:k]
        else:
            long_cands = ranked[:k]
            short_cands = ranked[-k:]

        # Avoid overlap
        long_syms = [s for s in long_cands if s not in short_cands]
        short_syms = [s for s in short_cands if s not in long_syms]

        if not long_syms or not short_syms:
            return {s: 0.0 for s in syms}

        # ── Portfolio construction (inverse-vol weighted) ──────────────
        all_active = list(set(long_syms) | set(short_syms))
        inv_vol: Dict[str, float] = {}
        for s in all_active:
            v = trailing_vol(dataset.perp_close[s], end_idx=idx, lookback_bars=vol_lb)
            inv_vol[s] = (1.0 / v) if v > 0 else 1.0

        w = normalize_dollar_neutral(long_syms, short_syms, inv_vol, target_gross)

        out = {s: 0.0 for s in syms}
        out.update(w)
        return out
