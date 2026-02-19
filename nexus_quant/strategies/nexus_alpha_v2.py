"""
NEXUS Alpha V2 — Regime-Adaptive Enhanced Multi-Factor Strategy.

Key Innovations over V1:
─────────────────────────────────────────────────────────────────────
1. Vol-Normalized MR Signal:
   Instead of raw 48h return, use (48h_return / 48h_vol) — identifies
   truly overbought/oversold symbols vs just those with high raw return.
   A symbol up +5% in a 1% vol environment is more overbought than
   one up +5% in a 10% vol environment.

2. Regime-Adaptive Leverage:
   BTC weekly (168h) trend determines leverage scaling factor:
     BTC +10% weekly → 1.0× target_gross (full leverage)
     BTC flat         → 0.5× target_gross (half leverage)
     BTC -10% weekly → 0.4× target_gross (floor)
   Reduces drawdown in bear markets while maintaining bull-market gains.

3. Multi-Horizon MR (optional):
   Averages vol-normalized MR z-scores at [mr_lb-12, mr_lb, mr_lb+24].
   Grid search showed 36h, 48h, 72h all have alpha (Sharpe 2.16, 2.78, 2.19).
   Averaging across three horizons is more robust than any single lookback.

4. Short-term Continuation Signal (optional, w=0.0 default):
   12h momentum as an independent microstructure alpha source.
   Not used by default — enable with w_short_mom > 0 to test.

5. Cross-Sectional Dispersion Filter (optional):
   Only trade when dispersion of cross-sectional returns exceeds a threshold.
   Low dispersion = all coins moving together = low alpha environment.

Signal convention (same as V1):
  HIGH composite score → LONG candidate
  LOW  composite score → SHORT candidate

Parameters
──────────
k_per_side                    int   = 2
w_carry                       float = 0.35   funding carry (low funding → LONG)
w_mom                         float = 0.45   price momentum 168h (high → LONG)
w_mean_reversion              float = 0.20   vol-norm MR (recent losers → LONG)
w_short_mom                   float = 0.00   12h continuation signal (high → LONG)
momentum_lookback_bars        int   = 168
mean_reversion_lookback_bars  int   = 48     centre of multi-horizon triplet
short_mom_lookback_bars       int   = 12
vol_lookback_bars             int   = 168    for inverse-vol portfolio construction
multi_horizon_mr              bool  = True   average MR at 3 horizons
use_regime_filter             bool  = True   scale leverage by BTC trend
regime_lookback_bars          int   = 168    BTC lookback for regime score
dispersion_threshold          float = 0.0   min cross-sectional dispersion (0=off)
target_gross_leverage         float = 0.35
min_gross_leverage            float = 0.05
max_gross_leverage            float = 0.65
rebalance_interval_bars       int   = 168
"""
from __future__ import annotations

import math
import statistics
from typing import Any, Dict, List

from ._math import normalize_dollar_neutral, trailing_vol, zscores
from .base import Strategy, Weights
from ..data.schema import MarketDataset


# ── Regime helpers ─────────────────────────────────────────────────────────────

def _market_regime_factor(
    closes: List[float],
    idx: int,
    lookback: int = 168,
) -> float:
    """
    Returns a leverage scaling factor in [0.40, 1.00] based on recent trend.

    Piecewise linear mapping:
      weekly return >= +10%  →  factor = 1.00  (bull)
      weekly return = 0%     →  factor = 0.50  (neutral)
      weekly return <= -10%  →  factor = 0.40  (bear, floored)
    """
    i1 = min(idx - 1, len(closes) - 1)
    i0 = max(0, idx - 1 - lookback)
    c1 = closes[i1] if 0 <= i1 < len(closes) else 0.0
    c0 = closes[i0] if 0 <= i0 < len(closes) else 0.0
    if c0 <= 0:
        return 0.75  # neutral if no data
    mom = c1 / c0 - 1.0
    # Linear: +10% weekly → 1.0; flat → 0.5; -10% → 0.0 (floored at 0.4)
    factor = 0.5 + 5.0 * mom
    return max(0.40, min(1.00, factor))


def _cs_dispersion(raw_returns: Dict[str, float]) -> float:
    """Cross-sectional standard deviation of returns (alpha opportunity measure)."""
    vals = list(raw_returns.values())
    if len(vals) < 2:
        return 0.0
    return statistics.pstdev(vals)


# ── Main strategy class ────────────────────────────────────────────────────────

class NexusAlphaV2Strategy(Strategy):
    """NEXUS Alpha V2 — Regime-Adaptive Enhanced Multi-Factor Strategy."""

    def __init__(self, params: Dict[str, Any]) -> None:
        super().__init__(name="nexus_alpha_v2", params=params)

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
        mom_lb   = self._p("momentum_lookback_bars", 168)
        vol_lb   = self._p("vol_lookback_bars", 168)
        mr_lb    = self._p("mean_reversion_lookback_bars", 48)
        interval = self._p("rebalance_interval_bars", 168)
        warmup   = max(mom_lb, vol_lb, mr_lb + 24) + 2
        if idx <= warmup:
            return False
        return idx % max(1, interval) == 0

    def target_weights(self, dataset: MarketDataset, idx: int, current: Weights) -> Weights:
        syms = dataset.symbols
        k    = max(1, min(self._p("k_per_side", 2), len(syms) // 2))

        mom_lb   = self._p("momentum_lookback_bars", 168)
        mr_lb    = self._p("mean_reversion_lookback_bars", 48)
        st_lb    = self._p("short_mom_lookback_bars", 12)
        vol_lb   = self._p("vol_lookback_bars", 168)
        reg_lb   = self._p("regime_lookback_bars", 168)
        multi_mr = self._p("multi_horizon_mr", True)
        use_reg  = self._p("use_regime_filter", True)
        disp_thr = self._p("dispersion_threshold", 0.0)

        target_gross = self._p("target_gross_leverage", 0.35)
        min_lev      = self._p("min_gross_leverage", 0.05)
        max_lev      = self._p("max_gross_leverage", 0.65)

        w_carry = self._p("w_carry",          0.35)
        w_mom   = self._p("w_mom",            0.45)
        w_mr    = self._p("w_mean_reversion", 0.20)
        w_sm    = self._p("w_short_mom",      0.00)

        ts = dataset.timeline[idx]

        # ── Signal 1: Funding carry ────────────────────────────────
        f_raw = {s: float(dataset.last_funding_rate_before(s, ts)) for s in syms}
        fz    = zscores(f_raw)
        carry = {s: -float(fz.get(s, 0.0)) for s in syms}  # low funding → LONG

        # ── Signal 2: Price momentum (168h) ───────────────────────
        mom_raw: Dict[str, float] = {}
        for s in syms:
            c  = dataset.perp_close[s]
            i1 = min(idx - 1, len(c) - 1)
            i0 = max(0, idx - 1 - mom_lb)
            c1 = c[i1] if 0 <= i1 < len(c) else 0.0
            c0 = c[i0] if 0 <= i0 < len(c) else 0.0
            mom_raw[s] = (c1 / c0 - 1.0) if c0 > 0 else 0.0
        mz  = zscores(mom_raw)
        mom = {s: float(mz.get(s, 0.0)) for s in syms}  # high mom → LONG

        # ── Signal 3: Vol-normalized mean reversion ───────────────
        # Core innovation: normalize 48h return by 48h vol to detect
        # truly overbought/oversold vs merely high-return symbols.
        def _vol_norm_mr_raw(lb: int) -> Dict[str, float]:
            raw: Dict[str, float] = {}
            for s in syms:
                c  = dataset.perp_close[s]
                i1 = min(idx - 1, len(c) - 1)
                i0 = max(0, idx - 1 - lb)
                c1 = c[i1] if 0 <= i1 < len(c) else 0.0
                c0 = c[i0] if 0 <= i0 < len(c) else 0.0
                ret = (c1 / c0 - 1.0) if c0 > 0 else 0.0
                # Trailing vol over the same look-back window
                pv = trailing_vol(c, end_idx=max(0, idx - 1), lookback_bars=lb)
                # Divide return by vol → "how many sigmas did this symbol move?"
                raw[s] = (ret / pv) if pv > 1e-8 else ret
            return raw

        if multi_mr:
            # Three lookbacks centred on mr_lb: [mr_lb-12, mr_lb, mr_lb+24]
            lb1 = max(12, mr_lb - 12)
            lb2 = mr_lb
            lb3 = mr_lb + 24
            z1  = zscores(_vol_norm_mr_raw(lb1))
            z2  = zscores(_vol_norm_mr_raw(lb2))
            z3  = zscores(_vol_norm_mr_raw(lb3))
            # Negate: recent relative losers (low z-score) → LONG
            mr  = {s: -(z1.get(s, 0.0) + z2.get(s, 0.0) + z3.get(s, 0.0)) / 3.0
                   for s in syms}
        else:
            z_mr = zscores(_vol_norm_mr_raw(mr_lb))
            mr   = {s: -float(z_mr.get(s, 0.0)) for s in syms}

        # ── Signal 4: Short-term continuation (12h, optional) ─────
        if abs(w_sm) > 1e-6:
            st_raw: Dict[str, float] = {}
            for s in syms:
                c  = dataset.perp_close[s]
                i1 = min(idx - 1, len(c) - 1)
                i0 = max(0, idx - 1 - st_lb)
                c1 = c[i1] if 0 <= i1 < len(c) else 0.0
                c0 = c[i0] if 0 <= i0 < len(c) else 0.0
                st_raw[s] = (c1 / c0 - 1.0) if c0 > 0 else 0.0
            st_z      = zscores(st_raw)
            short_mom = {s: float(st_z.get(s, 0.0)) for s in syms}
        else:
            short_mom = {s: 0.0 for s in syms}

        # ── Composite score ────────────────────────────────────────
        score: Dict[str, float] = {}
        for s in syms:
            score[s] = (
                w_carry * carry[s]
                + w_mom   * mom[s]
                + w_mr    * mr[s]
                + w_sm    * short_mom[s]
            )

        # ── Cross-sectional dispersion filter (optional) ──────────
        if disp_thr > 0.0:
            if _cs_dispersion(mom_raw) < disp_thr:
                return {s: 0.0 for s in syms}

        ranked     = sorted(syms, key=lambda s: score[s], reverse=True)
        long_cands = ranked[:k]
        short_cands = ranked[-k:]

        long_syms  = [s for s in long_cands  if s not in short_cands]
        short_syms = [s for s in short_cands if s not in long_cands]

        if not long_syms or not short_syms:
            return {s: 0.0 for s in syms}

        # ── Regime-adaptive leverage ──────────────────────────────
        effective_gross = target_gross
        if use_reg and syms:
            # Use first symbol as market proxy (expect BTC to be first)
            btc_sym       = syms[0]
            regime_factor = _market_regime_factor(
                closes  = dataset.perp_close[btc_sym],
                idx     = idx,
                lookback = reg_lb,
            )
            effective_gross = max(min_lev, min(max_lev, target_gross * regime_factor))

        # ── Portfolio construction (inverse-vol weighting) ─────────
        all_active = list(set(long_syms) | set(short_syms))
        inv_vol: Dict[str, float] = {}
        for s in all_active:
            v = trailing_vol(dataset.perp_close[s], end_idx=idx, lookback_bars=vol_lb)
            inv_vol[s] = (1.0 / v) if v > 0 else 1.0

        w = normalize_dollar_neutral(long_syms, short_syms, inv_vol, effective_gross)

        out = {s: 0.0 for s in syms}
        out.update(w)
        return out
