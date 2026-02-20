"""
Hybrid Alpha — Multi-signal combination of the best orthogonal signals.

Combines:
  1. Vol Breakout: vol compression → breakout direction
  2. Volume Reversal: capitulation detection (volume × return)
  3. Funding Contrarian: fade crowded positions (funding × momentum)

Each sub-signal is z-scored cross-sectionally, then combined with weights.
This provides diversification across signal types (vol structure, volume flow, positioning).

Parameters:
  k_per_side              int   = 2
  w_vol_breakout          float = 0.40   weight for vol breakout signal
  w_vol_reversal          float = 0.30   weight for volume reversal signal
  w_fund_contrarian       float = 0.30   weight for funding contrarian signal
  vol_short_bars          int   = 24     vol breakout: short-term vol
  vol_long_bars           int   = 168    vol breakout: long-term vol
  compression_threshold   float = 0.7    vol breakout: compression threshold
  return_lookback_bars    int   = 48     vol breakout + vol reversal: return lookback
  volume_lookback_bars    int   = 168    vol reversal: volume z-score window
  momentum_lookback_bars  int   = 336    funding contrarian: momentum window
  vol_lookback_bars       int   = 168    for inverse-vol weighting
  target_gross_leverage   float = 0.30
  rebalance_interval_bars int   = 48
"""
from __future__ import annotations

import statistics
from typing import Any, Dict, List

from ._math import normalize_dollar_neutral, trailing_vol, zscores
from .base import Strategy, Weights
from ..data.schema import MarketDataset


class HybridAlphaStrategy(Strategy):
    """Hybrid signal: vol breakout + volume reversal + funding contrarian."""

    def __init__(self, params: Dict[str, Any]) -> None:
        super().__init__(name="hybrid_alpha", params=params)

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
        vol_long = self._p("vol_long_bars", 168)
        mom_lb = self._p("momentum_lookback_bars", 336)
        interval = self._p("rebalance_interval_bars", 48)
        warmup = max(vol_long, mom_lb) + 10
        if idx <= warmup:
            return False
        return idx % max(1, interval) == 0

    def _signal_vol_breakout(self, dataset: MarketDataset, idx: int) -> Dict[str, float]:
        """Vol breakout sub-signal."""
        vol_short_bars = self._p("vol_short_bars", 24)
        vol_long_bars = self._p("vol_long_bars", 168)
        compression_thr = self._p("compression_threshold", 0.7)
        ret_lb = self._p("return_lookback_bars", 48)

        signal: Dict[str, float] = {}
        for s in dataset.symbols:
            c = dataset.perp_close[s]
            v_short = trailing_vol(c, end_idx=idx, lookback_bars=vol_short_bars)
            v_long = trailing_vol(c, end_idx=idx, lookback_bars=vol_long_bars)
            vol_ratio = v_short / v_long if v_long > 1e-10 else 1.0

            i1 = min(idx - 1, len(c) - 1)
            i0 = max(0, idx - 1 - ret_lb)
            c1 = float(c[i1]) if i1 >= 0 else 0.0
            c0 = float(c[i0]) if i0 >= 0 and i0 < len(c) else 0.0
            ret = (c1 / c0 - 1.0) if c0 > 0 else 0.0

            if vol_ratio < compression_thr:
                compression_strength = max(0.0, 1.0 - vol_ratio)
                signal[s] = compression_strength * ret
            else:
                signal[s] = 0.1 * ret

        return zscores(signal)

    def _signal_vol_reversal(self, dataset: MarketDataset, idx: int) -> Dict[str, float]:
        """Volume reversal sub-signal."""
        vol_lb = self._p("volume_lookback_bars", 168)
        ret_lb = self._p("return_lookback_bars", 48)

        if dataset.perp_volume is None:
            return {s: 0.0 for s in dataset.symbols}

        signal: Dict[str, float] = {}
        for s in dataset.symbols:
            # Volume z-score
            vol_series = dataset.perp_volume.get(s)
            vz = 0.0
            if vol_series is not None:
                end = min(idx, len(vol_series))
                start = max(0, end - vol_lb)
                window = [float(vol_series[i]) for i in range(start, end) if i < len(vol_series)]
                if len(window) >= 20:
                    current = window[-1]
                    mu = statistics.mean(window)
                    sd = statistics.pstdev(window)
                    if sd > 1e-10:
                        vz = (current - mu) / sd

            # Return
            c = dataset.perp_close[s]
            i1 = min(idx - 1, len(c) - 1)
            i0 = max(0, idx - 1 - ret_lb)
            c1 = float(c[i1]) if i1 >= 0 else 0.0
            c0 = float(c[i0]) if i0 >= 0 and i0 < len(c) else 0.0
            ret = (c1 / c0 - 1.0) if c0 > 0 else 0.0

            signal[s] = vz * (-ret)

        return zscores(signal)

    def _signal_funding_contrarian(self, dataset: MarketDataset, idx: int) -> Dict[str, float]:
        """Funding contrarian sub-signal."""
        mom_lb = self._p("momentum_lookback_bars", 336)
        ts = dataset.timeline[idx]

        f_raw = {s: float(dataset.last_funding_rate_before(s, ts)) for s in dataset.symbols}
        fz = zscores(f_raw)

        mom_raw: Dict[str, float] = {}
        for s in dataset.symbols:
            c = dataset.perp_close[s]
            i1 = min(idx - 1, len(c) - 1)
            i0 = max(0, idx - 1 - mom_lb)
            c1 = float(c[i1]) if i1 >= 0 else 0.0
            c0 = float(c[i0]) if i0 >= 0 and i0 < len(c) else 0.0
            mom_raw[s] = (c1 / c0 - 1.0) if c0 > 0 else 0.0
        mz = zscores(mom_raw)

        signal: Dict[str, float] = {}
        for s in dataset.symbols:
            crowding = float(fz.get(s, 0.0)) * float(mz.get(s, 0.0))
            signal[s] = -crowding

        return zscores(signal)

    def target_weights(self, dataset: MarketDataset, idx: int, current: Weights) -> Weights:
        syms = dataset.symbols
        k = max(1, min(self._p("k_per_side", 2), len(syms) // 2))
        target_gross = self._p("target_gross_leverage", 0.30)
        vol_lb = self._p("vol_lookback_bars", 168)

        w_vb = self._p("w_vol_breakout", 0.40)
        w_vr = self._p("w_vol_reversal", 0.30)
        w_fc = self._p("w_fund_contrarian", 0.30)

        # Compute sub-signals
        sig_vb = self._signal_vol_breakout(dataset, idx)
        sig_vr = self._signal_vol_reversal(dataset, idx)
        sig_fc = self._signal_funding_contrarian(dataset, idx)

        # Composite
        composite: Dict[str, float] = {}
        for s in syms:
            composite[s] = (
                w_vb * float(sig_vb.get(s, 0.0))
                + w_vr * float(sig_vr.get(s, 0.0))
                + w_fc * float(sig_fc.get(s, 0.0))
            )

        ranked = sorted(syms, key=lambda s: composite.get(s, 0.0), reverse=True)
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
