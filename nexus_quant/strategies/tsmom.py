from __future__ import annotations
"""
Time-Series Momentum (TSMOM) Strategy.

Academic reference: Moskowitz, Ooi, Pedersen (2012) "Time Series Momentum"
Adapted for crypto perpetual futures.

Logic per symbol (INDEPENDENT - not cross-sectional):
  - Calculate backward-looking return over lookback_bars
  - Position = sign(return) x vol_target / realized_vol
  - Long if return > 0, Short if return < 0, scaled by volatility target
  - Optional: skip if |return| < skip_threshold (flat market filter)

This strategy works ACROSS market regimes (long in bull, short in bear).
Expected Sharpe: 0.8-1.5 documented in crypto.
"""

import math
from typing import Any, Dict

from .base import Strategy, Weights
from ._math import trailing_vol
from ..data.schema import MarketDataset

# Annualized volatility target expressed per bar.
# vol_target is annualized (e.g. 0.15 = 15% p.a.).
# Crypto perp data is typically hourly (8760 bars/year).
_BARS_PER_YEAR = 8760.0


class TimeSerisMomentumV1Strategy(Strategy):
    """
    Per-asset time-series momentum with volatility scaling.

    Each symbol is treated INDEPENDENTLY:
      raw_weight = sign(lookback_return) x (vol_target_per_bar / realized_vol_per_bar)

    Weights are then clipped and rescaled to target_gross_leverage.

    Parameters
    ----------
    lookback_bars : int, default 168
        Bars used to measure the directional signal (default 7 days x 24 h).
    vol_lookback_bars : int, default 168
        Bars used to estimate realized volatility for scaling.
    vol_target : float, default 0.15
        Annualized volatility target (0.15 = 15% p.a.).  Converted internally
        to per-bar target via sqrt(bars_per_year).
    target_gross_leverage : float, default 0.5
        After vol-scaling, rescale the entire book so that
        sum(|w|) == target_gross_leverage.
    max_weight_per_symbol : float, default 0.30
        Hard cap on absolute weight for any single symbol before the global
        leverage rescaling step.
    skip_threshold : float, default 0.02
        If |lookback_return| < skip_threshold the symbol is set flat.
        Useful to avoid trading in choppy / directionless markets.
        Set to 0.0 to disable.
    rebalance_interval_bars : int, default 24
        How often (in bars) the strategy rebalances after warmup.
    """

    def __init__(self, params: Dict[str, Any]) -> None:
        super().__init__(name="tsmom_v1", params=params)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _p_int(self, key: str, default: int) -> int:
        v = self.params.get(key)
        return int(v) if v is not None else default

    def _p_float(self, key: str, default: float) -> float:
        v = self.params.get(key)
        return float(v) if v is not None else default

    # ------------------------------------------------------------------
    # Strategy interface
    # ------------------------------------------------------------------

    def should_rebalance(self, dataset: MarketDataset, idx: int) -> bool:
        lookback = self._p_int("lookback_bars", 168)
        interval = self._p_int("rebalance_interval_bars", 24)

        # Need at least lookback bars of history before issuing the first
        # signal (warm-up period).
        if idx <= lookback + 1:
            return False

        return idx % max(1, interval) == 0

    def target_weights(self, dataset: MarketDataset, idx: int, current: Weights) -> Weights:
        lookback = self._p_int("lookback_bars", 168)
        vol_lookback = self._p_int("vol_lookback_bars", 168)
        vol_target_ann = self._p_float("vol_target", 0.15)
        target_gross = self._p_float("target_gross_leverage", 0.5)
        max_w = self._p_float("max_weight_per_symbol", 0.30)
        skip_thresh = self._p_float("skip_threshold", 0.02)

        # Per-bar volatility target (annualised to per-bar)
        vol_target_pb = vol_target_ann / math.sqrt(_BARS_PER_YEAR)

        raw: Dict[str, float] = {}

        for s in dataset.symbols:
            closes = dataset.perp_close.get(s, [])
            n = len(closes)

            # Need enough history for both the signal and the vol estimate
            needed = max(lookback, vol_lookback) + 2
            if idx < needed or n <= idx:
                raw[s] = 0.0
                continue

            # -- Directional signal (use t-1 to avoid lookahead) --
            c_now = closes[idx - 1]
            c_past = closes[max(0, idx - 1 - lookback)]
            lookback_return = (c_now / c_past) - 1.0 if c_past != 0.0 else 0.0

            # Flat-market filter
            if abs(lookback_return) < skip_thresh:
                raw[s] = 0.0
                continue

            direction = 1.0 if lookback_return > 0.0 else -1.0

            # -- Volatility scaling --
            vol_pb = trailing_vol(closes, end_idx=idx, lookback_bars=vol_lookback)
            vol_pb = max(vol_pb, 1e-9)  # never divide by zero

            # raw weight: scale so expected per-bar vol == vol_target_pb
            w = direction * (vol_target_pb / vol_pb)

            # Hard cap per symbol
            w = max(-max_w, min(max_w, w))

            raw[s] = w

        # ------------------------------------------------------------------
        # Global rescaling to target_gross_leverage
        # ------------------------------------------------------------------
        gross = sum(abs(v) for v in raw.values())
        if gross > 0.0:
            scale = target_gross / gross
            raw = {s: v * scale for s, v in raw.items()}

        # Ensure every symbol has an entry
        out: Weights = {s: 0.0 for s in dataset.symbols}
        out.update(raw)
        return out
