from __future__ import annotations
"""
Combined Funding Carry + Momentum Strategy.

Combines two alpha signals:
1. Funding rate carry (cross-sectional): short high-funding, long low-funding
2. Cross-sectional momentum: long top performers, short bottom performers

Key innovation: Both signals must AGREE for a position:
- LONG candidate: LOW/NEGATIVE funding AND POSITIVE momentum
- SHORT candidate: HIGH/POSITIVE funding AND NEGATIVE momentum
- If signals disagree: REDUCE position size (conflict penalty)

This avoids the 2024 bull market disaster where pure carry shorts
high-funding = high-momentum = strong bull coins.

Expected improvement vs pure carry: Sharpe 1.5-2.5
"""

from typing import Any, Dict, List

from .base import Strategy, Weights
from ._math import normalize_dollar_neutral, trailing_vol, zscores
from ..data.schema import MarketDataset


class CombinedCarryMomentumV1Strategy(Strategy):
    """
    Cross-sectional combination of funding-rate carry and price momentum.

    Signal construction
    -------------------
    For each symbol s at rebalance bar idx (using t-1 data to avoid lookahead):

        fz[s]   = cross-sectional z-score of last_funding_rate_before(ts)
        mz[s]   = cross-sectional z-score of price return over momentum_bars

        carry_signal[s]   = -fz[s]
            (negative funding z => we want to be LONG; positive => SHORT)
        mom_signal[s]     = mz[s]
            (positive momentum z => we want to be LONG)

        agreement[s]      = carry_signal[s] * mom_signal[s]
            (positive => both signals agree on direction)

        final_score[s]    = w_carry * carry_signal[s]
                          + w_mom   * mom_signal[s]
                          + w_confirm * agreement[s]

    Position selection
    ------------------
    - Rank symbols by final_score descending.
    - Long top k_per_side, short bottom k_per_side.
    - If strict_agreement=True, zero out any position where agreement < 0.
    - Inverse-vol weight within each side (if risk_weighting="inverse_vol").
    - Scale to target_gross_leverage using normalize_dollar_neutral.

    Parameters
    ----------
    k_per_side : int, default 2
    momentum_bars : int, default 168
    vol_lookback_bars : int, default 168
    w_carry : float, default 0.4
    w_mom : float, default 0.4
    w_confirm : float, default 0.2
    target_gross_leverage : float, default 0.30
    risk_weighting : str, default "inverse_vol"
    rebalance_interval_bars : int, default 168
    strict_agreement : bool, default False
        If True, remove any symbol from long/short lists where the two
        primary signals point in opposite directions.
    """

    def __init__(self, params: Dict[str, Any]) -> None:
        super().__init__(name="combined_carry_mom_v1", params=params)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _p_int(self, key: str, default: int) -> int:
        v = self.params.get(key)
        return int(v) if v is not None else default

    def _p_float(self, key: str, default: float) -> float:
        v = self.params.get(key)
        return float(v) if v is not None else default

    def _p_bool(self, key: str, default: bool) -> bool:
        v = self.params.get(key)
        return bool(v) if v is not None else default

    def _p_str(self, key: str, default: str) -> str:
        v = self.params.get(key)
        return str(v) if v is not None else default

    # ------------------------------------------------------------------
    # Strategy interface
    # ------------------------------------------------------------------

    def should_rebalance(self, dataset: MarketDataset, idx: int) -> bool:
        mom_lb = self._p_int("momentum_bars", 168)
        vol_lb = self._p_int("vol_lookback_bars", 168)
        interval = self._p_int("rebalance_interval_bars", 168)

        warmup = max(mom_lb, vol_lb) + 2
        if idx <= warmup:
            return False

        return idx % max(1, interval) == 0

    def target_weights(self, dataset: MarketDataset, idx: int, current: Weights) -> Weights:
        k = self._p_int("k_per_side", 2)
        k = max(1, min(k, max(1, len(dataset.symbols) // 2)))

        mom_lb = self._p_int("momentum_bars", 168)
        vol_lb = self._p_int("vol_lookback_bars", 168)
        w_carry = self._p_float("w_carry", 0.4)
        w_mom = self._p_float("w_mom", 0.4)
        w_confirm = self._p_float("w_confirm", 0.2)
        target_gross = self._p_float("target_gross_leverage", 0.30)
        risk_weighting = self._p_str("risk_weighting", "inverse_vol")
        strict = self._p_bool("strict_agreement", False)

        ts = dataset.timeline[idx]

        # ------------------------------------------------------------------
        # 1. Funding z-scores (cross-sectional, lagged)
        # ------------------------------------------------------------------
        f_raw = {s: dataset.last_funding_rate_before(s, ts) for s in dataset.symbols}
        fz = zscores(f_raw)

        # carry_signal: negative fz means low/negative funding => want LONG
        carry_signal: Dict[str, float] = {s: -float(fz.get(s, 0.0)) for s in dataset.symbols}

        # ------------------------------------------------------------------
        # 2. Momentum z-scores (cross-sectional, lagged)
        # ------------------------------------------------------------------
        mom_raw: Dict[str, float] = {}
        for s in dataset.symbols:
            closes = dataset.perp_close.get(s, [])
            if len(closes) > idx and idx > mom_lb:
                c_now = closes[idx - 1]
                c_past = closes[max(0, idx - 1 - mom_lb)]
                mom_raw[s] = (c_now / c_past) - 1.0 if c_past != 0.0 else 0.0
            else:
                mom_raw[s] = 0.0
        mz = zscores(mom_raw)

        mom_signal: Dict[str, float] = {s: float(mz.get(s, 0.0)) for s in dataset.symbols}

        # ------------------------------------------------------------------
        # 3. Agreement score and combined final score
        # ------------------------------------------------------------------
        agreement: Dict[str, float] = {
            s: carry_signal[s] * mom_signal[s] for s in dataset.symbols
        }

        final_score: Dict[str, float] = {
            s: (
                w_carry * carry_signal[s]
                + w_mom * mom_signal[s]
                + w_confirm * agreement[s]
            )
            for s in dataset.symbols
        }

        # ------------------------------------------------------------------
        # 4. Select top/bottom k symbols
        # ------------------------------------------------------------------
        ranked: List[str] = sorted(dataset.symbols, key=lambda s: final_score[s], reverse=True)
        long_candidates: List[str] = ranked[:k]
        short_candidates: List[str] = ranked[-k:]

        # strict_agreement filter: remove conflicting-signal positions
        if strict:
            long_candidates = [s for s in long_candidates if agreement[s] >= 0.0]
            short_candidates = [s for s in short_candidates if agreement[s] >= 0.0]

        # Avoid overlap (edge case: small universe)
        long_syms: List[str] = [s for s in long_candidates if s not in short_candidates]
        short_syms: List[str] = [s for s in short_candidates if s not in long_candidates]

        # ------------------------------------------------------------------
        # 5. Inverse-vol weighting
        # ------------------------------------------------------------------
        all_syms = list(set(long_syms) | set(short_syms))
        inv_vol: Dict[str, float] = {}
        if risk_weighting == "inverse_vol":
            for s in all_syms:
                vol = trailing_vol(dataset.perp_close[s], end_idx=idx, lookback_bars=vol_lb)
                inv_vol[s] = (1.0 / vol) if vol > 0.0 else 1.0
        else:
            for s in all_syms:
                inv_vol[s] = 1.0

        # ------------------------------------------------------------------
        # 6. Dollar-neutral allocation
        # ------------------------------------------------------------------
        w = normalize_dollar_neutral(
            long_syms=long_syms,
            short_syms=short_syms,
            inv_vol=inv_vol,
            target_gross_leverage=target_gross,
        )

        # Ensure every symbol has an entry
        out: Weights = {s: 0.0 for s in dataset.symbols}
        out.update(w)
        return out
