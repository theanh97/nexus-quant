"""
Funding Carry Alpha — Pure carry trade for sideways/bear crypto markets.

Economic Foundation:
  In crypto perpetual futures, funding rates are paid every 8 hours.
  - Positive funding → longs pay shorts (market is crowded-long)
  - Negative funding → shorts pay longs (market is crowded-short)

  Strategy: Long symbols with LOW/negative funding (receive payments),
  Short symbols with HIGH/positive funding (collect carry + expect
  crowded-long reversion).

  This is economically DIFFERENT from momentum+MR (V1's alpha source).
  Carry alpha is regime-INdependent: works in sideways and bear markets
  because funding is the dominant P&L driver when prices are flat.

Signal:
  1. Compute rolling average funding rate per symbol (last 24h = 3 periods)
  2. Cross-sectional z-score
  3. Long bottom-k (lowest funding), Short top-k (highest funding)
  4. Dollar-neutral, inverse-vol weighted

Expected Sharpe: 0.5-1.5 OOS (GLM-5 estimates 1.2-1.8, GPT-5.2 estimates 0.5-1.0)
Best regime: sideways/bear (funding is reliable income stream)

Parameters:
  k_per_side                    int   = 2
  funding_lookback_periods      int   = 3     (3 x 8h = 24h rolling avg)
  vol_lookback_bars             int   = 168
  target_gross_leverage         float = 0.35
  rebalance_interval_bars       int   = 8     (every funding period)
  risk_weighting                str   = "inverse_vol"
"""
from __future__ import annotations

from typing import Any, Dict

from ._math import normalize_dollar_neutral, trailing_vol, zscores
from .base import Strategy, Weights
from ..data.schema import MarketDataset


class FundingCarryAlphaStrategy(Strategy):
    """Pure funding carry: Long low-funding, Short high-funding."""

    def __init__(self, params: Dict[str, Any]) -> None:
        super().__init__(name="funding_carry_alpha", params=params)

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
        interval = self._p("rebalance_interval_bars", 8)
        warmup = max(vol_lb, 48) + 2
        if idx <= warmup:
            return False
        return idx % max(1, interval) == 0

    def target_weights(self, dataset: MarketDataset, idx: int, current: Weights) -> Weights:
        syms = dataset.symbols
        k = max(1, min(self._p("k_per_side", 2), len(syms) // 2))
        vol_lb = self._p("vol_lookback_bars", 168)
        target_gross = self._p("target_gross_leverage", 0.35)
        funding_lb = self._p("funding_lookback_periods", 3)
        risk_wt = str(self.params.get("risk_weighting") or "inverse_vol")

        ts = dataset.timeline[idx]

        # ── Signal: Rolling average funding rate ──────────────────────
        # Collect last N funding rate observations per symbol
        funding_raw: Dict[str, float] = {}
        for s in syms:
            # Get the most recent funding rate
            fr = float(dataset.last_funding_rate_before(s, ts))
            # For rolling average, we look back at prior bars' funding
            # Since funding is every 8h and we have 1h bars, approximate
            # by sampling funding at idx, idx-8, idx-16, etc.
            rates = []
            for offset in range(funding_lb):
                back_idx = max(0, idx - offset * 8)
                if back_idx >= 0 and back_idx < len(dataset.timeline):
                    back_ts = dataset.timeline[back_idx]
                    r = float(dataset.last_funding_rate_before(s, back_ts))
                    rates.append(r)
            if rates:
                funding_raw[s] = sum(rates) / len(rates)
            else:
                funding_raw[s] = fr

        if len(funding_raw) < 2 * k:
            return {s: 0.0 for s in syms}

        # Z-score cross-sectionally
        fz = zscores(funding_raw)

        # Signal: NEGATE funding z-score
        # Low funding → positive signal → LONG (receive funding)
        # High funding → negative signal → SHORT (crowded-long reversion)
        signal = {s: -float(fz.get(s, 0.0)) for s in syms}

        ranked = sorted(syms, key=lambda s: signal[s], reverse=True)
        long_syms = ranked[:k]
        short_syms = ranked[-k:]

        # Ensure no overlap
        long_syms = [s for s in long_syms if s not in short_syms]
        short_syms = [s for s in short_syms if s not in long_syms]

        if not long_syms or not short_syms:
            return {s: 0.0 for s in syms}

        # ── Portfolio construction ────────────────────────────────────
        inv_vol: Dict[str, float] = {}
        all_active = list(set(long_syms) | set(short_syms))
        for s in all_active:
            if risk_wt == "inverse_vol":
                v = trailing_vol(dataset.perp_close[s], end_idx=idx, lookback_bars=vol_lb)
                inv_vol[s] = (1.0 / v) if v > 0 else 1.0
            else:
                inv_vol[s] = 1.0

        w = normalize_dollar_neutral(long_syms, short_syms, inv_vol, target_gross)
        out = {s: 0.0 for s in syms}
        out.update(w)
        return out
