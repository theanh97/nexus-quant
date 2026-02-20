"""
Idiosyncratic Momentum Alpha — Beta-hedged cross-sectional momentum.

Economic Foundation:
  Raw cross-sectional momentum is dominated by the BTC factor: when BTC rises,
  most alts rise proportionally to their beta. True alpha comes from
  IDIOSYNCRATIC performance — the component of each coin's return that is
  unexplained by BTC exposure.

  Formula:
    residual_return = total_return - beta_btc × btc_return
    beta_btc = OLS slope of coin_hourly_returns vs btc_hourly_returns

  Cross-sectionally: coins with higher idiosyncratic (beta-neutral) returns
  continue to outperform their peers even after controlling for BTC risk.

  This is orthogonal to all existing signals because:
  - V1-Long uses raw returns (doesn't remove BTC factor)
  - Sharpe Ratio uses total returns (same issue)
  - Vol Breakout doesn't use returns at all
  - This is the first pure "alpha extraction" signal

Strategy:
  1. Compute rolling OLS beta of each alt vs BTC over lookback_bars
  2. Compute BTC return over the same period
  3. Residual = coin_return - beta × btc_return (idiosyncratic component)
  4. Cross-sectional z-score of residuals
  5. Long top-k (highest idiosyncratic momentum), Short bottom-k

Parameters:
  k_per_side              int   = 2
  lookback_bars           int   = 336   OLS estimation window
  beta_window_bars        int   = 168   rolling window for beta estimation
  vol_lookback_bars       int   = 168
  target_gross_leverage   float = 0.30
  rebalance_interval_bars int   = 48
"""
from __future__ import annotations

import statistics
from typing import Any, Dict, List, Optional

from ._math import normalize_dollar_neutral, trailing_vol, zscores
from .base import Strategy, Weights
from ..data.schema import MarketDataset


def _simple_ols_beta(x: List[float], y: List[float]) -> float:
    """OLS beta: covariance(x,y) / variance(x). No intercept."""
    n = min(len(x), len(y))
    if n < 10:
        return 1.0  # default beta
    x, y = x[:n], y[:n]
    mx = sum(x) / n
    my = sum(y) / n
    cov = sum((x[i] - mx) * (y[i] - my) for i in range(n)) / n
    var_x = sum((xi - mx) ** 2 for xi in x) / n
    if var_x < 1e-12:
        return 1.0
    return cov / var_x


class IdioMomentumAlphaStrategy(Strategy):
    """Idiosyncratic momentum: beta-hedged cross-sectional alpha."""

    BTC_SYMBOL = "BTCUSDT"

    def __init__(self, params: Dict[str, Any]) -> None:
        super().__init__(name="idio_momentum_alpha", params=params)

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
        lookback = max(self._p("lookback_bars", 336), self._p("beta_window_bars", 168))
        interval = self._p("rebalance_interval_bars", 48)
        warmup = lookback + 10
        if idx <= warmup:
            return False
        return idx % max(1, interval) == 0

    def _hourly_returns(self, closes: List[float], idx: int, lookback: int) -> List[float]:
        """Return series of hourly returns over past lookback bars."""
        start = max(1, idx - lookback)
        end = min(idx, len(closes))
        rets = []
        for i in range(start, end):
            c0 = closes[i - 1]
            c1 = closes[i]
            if c0 > 0:
                rets.append((c1 / c0) - 1.0)
            else:
                rets.append(0.0)
        return rets

    def _total_return(self, closes: List[float], idx: int, lookback: int) -> float:
        """Cumulative return over past lookback bars."""
        i1 = min(idx - 1, len(closes) - 1)
        i0 = max(0, idx - 1 - lookback)
        c1 = float(closes[i1]) if i1 >= 0 else 0.0
        c0 = float(closes[i0]) if i0 >= 0 and i0 < len(closes) else 0.0
        return (c1 / c0 - 1.0) if c0 > 0 else 0.0

    def target_weights(self, dataset: MarketDataset, idx: int, current: Weights) -> Weights:
        syms = dataset.symbols
        k = max(1, min(self._p("k_per_side", 2), len(syms) // 2))
        lookback = self._p("lookback_bars", 336)
        beta_window = self._p("beta_window_bars", 168)
        vol_lb = self._p("vol_lookback_bars", 168)
        target_gross = self._p("target_gross_leverage", 0.30)

        # Need BTC as the market factor
        btc_sym = self.BTC_SYMBOL
        if btc_sym not in dataset.perp_close:
            # Fallback: use raw momentum if BTC not available
            mom_raw = {}
            for s in syms:
                mom_raw[s] = self._total_return(dataset.perp_close[s], idx, lookback)
            mz = zscores(mom_raw)
            score = {s: float(mz.get(s, 0.0)) for s in syms}
        else:
            # Compute BTC hourly returns for beta estimation
            btc_closes = dataset.perp_close[btc_sym]
            btc_hr_rets = self._hourly_returns(btc_closes, idx, beta_window)
            btc_total_ret = self._total_return(btc_closes, idx, lookback)

            # For each symbol: compute beta and idiosyncratic return
            idio_raw: Dict[str, float] = {}
            for s in syms:
                closes = dataset.perp_close[s]
                coin_hr_rets = self._hourly_returns(closes, idx, beta_window)

                # OLS beta
                n = min(len(btc_hr_rets), len(coin_hr_rets))
                if n >= 20:
                    beta = _simple_ols_beta(btc_hr_rets[:n], coin_hr_rets[:n])
                else:
                    beta = 1.0  # assume market beta = 1 if insufficient data

                # Clamp beta to [0.1, 3.0] to avoid extreme values
                beta = max(0.1, min(3.0, beta))

                # Total return of coin over lookback
                coin_total_ret = self._total_return(closes, idx, lookback)

                # Idiosyncratic component: coin return - beta * btc return
                idio_raw[s] = coin_total_ret - beta * btc_total_ret

            idio_z = zscores(idio_raw)
            score = {s: float(idio_z.get(s, 0.0)) for s in syms}

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
