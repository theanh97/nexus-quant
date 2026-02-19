"""
Low-Volatility Defensive Alpha — Quality/Defensive factor for bear/sideways markets.

Economic Foundation:
  In bear and sideways crypto markets, high-volatility names embed
  liquidation risk and convexity premium — they tend to underperform.
  Low-vol names behave like "quality" assets: they hold value better
  during stress and attract flight-to-safety flows.

  This is a well-known anomaly in traditional equities (low-vol anomaly,
  Ang et al. 2006) and has been documented in crypto (e.g., Bianchi 2020).

Signal:
  1. Compute trailing realized volatility per symbol (7-day = 168h lookback)
  2. Cross-sectional z-score
  3. Long bottom-k (lowest vol = "quality"), Short top-k (highest vol = "fragile")
  4. Dollar-neutral, equal-weighted (NOT inverse-vol since that would double-down)

Expected Sharpe: 0.4-0.9 (GPT-5.2 estimate, conservative)
Best regime: bear/sideways (flight-to-quality dominates)

Parameters:
  k_per_side                    int   = 2
  vol_lookback_bars             int   = 168    (7 days for realized vol)
  target_gross_leverage         float = 0.35
  rebalance_interval_bars       int   = 24     (daily, slow-moving factor)
"""
from __future__ import annotations

from typing import Any, Dict

from ._math import normalize_dollar_neutral, trailing_vol, zscores
from .base import Strategy, Weights
from ..data.schema import MarketDataset


class LowVolAlphaStrategy(Strategy):
    """Long low-vol, Short high-vol: defensive quality factor."""

    def __init__(self, params: Dict[str, Any]) -> None:
        super().__init__(name="low_vol_alpha", params=params)

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
        vol_lb = self._p("vol_lookback_bars", 168)
        interval = self._p("rebalance_interval_bars", 24)
        warmup = vol_lb + 10
        if idx <= warmup:
            return False
        return idx % max(1, interval) == 0

    def target_weights(self, dataset: MarketDataset, idx: int, current: Weights) -> Weights:
        syms = dataset.symbols
        k = max(1, min(self._p("k_per_side", 2), len(syms) // 2))
        vol_lb = self._p("vol_lookback_bars", 168)
        target_gross = self._p("target_gross_leverage", 0.35)

        # ── Signal: Realized volatility ──────────────────────────────
        vol_raw: Dict[str, float] = {}
        for s in syms:
            v = trailing_vol(dataset.perp_close[s], end_idx=idx, lookback_bars=vol_lb)
            if v > 0:
                vol_raw[s] = v

        if len(vol_raw) < 2 * k:
            return {s: 0.0 for s in syms}

        # Z-score cross-sectionally
        vz = zscores(vol_raw)

        # Signal: NEGATE vol z-score (low vol = positive signal = LONG)
        signal = {s: -float(vz.get(s, 0.0)) for s in syms}

        ranked = sorted(syms, key=lambda s: signal.get(s, 0.0), reverse=True)
        long_syms = ranked[:k]
        short_syms = ranked[-k:]

        long_syms = [s for s in long_syms if s not in short_syms]
        short_syms = [s for s in short_syms if s not in long_syms]

        if not long_syms or not short_syms:
            return {s: 0.0 for s in syms}

        # Equal weight (NOT inverse-vol, since that would double-down the signal)
        inv_vol = {s: 1.0 for s in set(long_syms) | set(short_syms)}

        w = normalize_dollar_neutral(long_syms, short_syms, inv_vol, target_gross)
        out = {s: 0.0 for s in syms}
        out.update(w)
        return out
