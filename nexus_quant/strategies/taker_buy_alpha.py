"""
Taker Buy Ratio Alpha — Directional aggression signal.

Hypothesis: Taker buy ratio (buy volume / total volume) measures conviction.
  - High ratio = aggressive buying = bullish conviction
  - Low ratio = aggressive selling = bearish conviction
  - Cross-sectional: long highest buy ratio, short lowest

Why orthogonal to V1: Volume-flow based, not price/funding based.

Parameters:
  k_per_side              int   = 2
  ratio_lookback_bars     int   = 48    window to compute average buy ratio
  vol_lookback_bars       int   = 168
  target_gross_leverage   float = 0.30
  rebalance_interval_bars int   = 24
"""
from __future__ import annotations

from typing import Any, Dict, List

from ._math import normalize_dollar_neutral, trailing_vol, zscores
from .base import Strategy, Weights
from ..data.schema import MarketDataset


class TakerBuyAlphaStrategy(Strategy):
    """Taker buy ratio: long aggressive buyers, short aggressive sellers."""

    def __init__(self, params: Dict[str, Any]) -> None:
        super().__init__(name="taker_buy_alpha", params=params)

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
        lb = self._p("ratio_lookback_bars", 48)
        interval = self._p("rebalance_interval_bars", 24)
        warmup = lb + 10
        if idx <= warmup:
            return False
        return idx % max(1, interval) == 0

    def target_weights(self, dataset: MarketDataset, idx: int, current: Weights) -> Weights:
        syms = dataset.symbols
        k = max(1, min(self._p("k_per_side", 2), len(syms) // 2))
        lb = self._p("ratio_lookback_bars", 48)
        target_gross = self._p("target_gross_leverage", 0.30)
        vol_lb = self._p("vol_lookback_bars", 168)

        if dataset.taker_buy_volume is None or dataset.perp_volume is None:
            return {s: 0.0 for s in syms}

        signal: Dict[str, float] = {}
        for s in syms:
            tbv = dataset.taker_buy_volume.get(s)
            tv = dataset.perp_volume.get(s)
            if tbv is None or tv is None:
                signal[s] = 0.0
                continue

            # Compute average taker buy ratio over lookback
            end = min(idx, len(tbv), len(tv))
            start = max(0, end - lb)
            ratios: List[float] = []
            for i in range(start, end):
                if i < len(tv) and i < len(tbv) and float(tv[i]) > 0:
                    ratios.append(float(tbv[i]) / float(tv[i]))

            if ratios:
                # Signal = average buy ratio - 0.5 (centered at neutral)
                signal[s] = sum(ratios) / len(ratios) - 0.5
            else:
                signal[s] = 0.0

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
