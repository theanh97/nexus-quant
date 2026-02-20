"""
Lead-Lag Alpha — BTC leads altcoin moves.

Hypothesis: BTC price moves lead altcoin moves by ~4-24 hours.

Signal:
  1. Compute BTC return over last N bars (e.g., 4h, 8h, 12h, 24h)
  2. Compute each altcoin's beta to BTC over trailing window
  3. If BTC up → go long altcoins with highest beta to BTC
  4. If BTC down → go short altcoins with highest beta
  5. Exclude BTC from traded universe (it's the signal, not the trade)

Why orthogonal to V1: Cross-asset signal, not self-referential momentum/MR.

Parameters:
  k_per_side              int   = 2
  btc_lookback_bars       int   = 12    BTC return lookback (signal)
  beta_lookback_bars      int   = 168   trailing beta estimation window
  vol_lookback_bars       int   = 168   for inverse-vol weighting
  target_gross_leverage   float = 0.30
  rebalance_interval_bars int   = 24
  btc_symbol              str   = "BTCUSDT"
"""
from __future__ import annotations

import math
import statistics
from typing import Any, Dict, List

from ._math import normalize_dollar_neutral, trailing_vol, zscores
from .base import Strategy, Weights
from ..data.schema import MarketDataset


class LeadLagAlphaStrategy(Strategy):
    """BTC lead-lag: trade altcoins based on BTC momentum direction × altcoin beta."""

    def __init__(self, params: Dict[str, Any]) -> None:
        super().__init__(name="lead_lag_alpha", params=params)

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
        beta_lb = self._p("beta_lookback_bars", 168)
        interval = self._p("rebalance_interval_bars", 24)
        warmup = beta_lb + 10
        if idx <= warmup:
            return False
        return idx % max(1, interval) == 0

    def _compute_btc_return(self, dataset: MarketDataset, idx: int) -> float:
        """BTC return over btc_lookback_bars."""
        btc_sym = self._p("btc_symbol", "BTCUSDT")
        btc_lb = self._p("btc_lookback_bars", 12)
        c = dataset.perp_close.get(btc_sym)
        if c is None:
            return 0.0
        i1 = min(idx - 1, len(c) - 1)
        i0 = max(0, idx - 1 - btc_lb)
        c1 = float(c[i1]) if i1 >= 0 else 0.0
        c0 = float(c[i0]) if i0 >= 0 and i0 < len(c) else 0.0
        if c0 <= 0:
            return 0.0
        return c1 / c0 - 1.0

    def _compute_beta(self, dataset: MarketDataset, symbol: str, idx: int) -> float:
        """Trailing beta of symbol to BTC."""
        btc_sym = self._p("btc_symbol", "BTCUSDT")
        beta_lb = self._p("beta_lookback_bars", 168)

        btc_c = dataset.perp_close.get(btc_sym)
        sym_c = dataset.perp_close.get(symbol)
        if btc_c is None or sym_c is None:
            return 1.0

        start = max(1, idx - 1 - beta_lb)
        end = min(idx - 1, len(btc_c) - 1, len(sym_c) - 1)

        btc_rets: List[float] = []
        sym_rets: List[float] = []
        for i in range(start, end + 1):
            if i < 1 or i >= len(btc_c) or i >= len(sym_c):
                continue
            bc0, bc1 = float(btc_c[i - 1]), float(btc_c[i])
            sc0, sc1 = float(sym_c[i - 1]), float(sym_c[i])
            if bc0 > 0 and sc0 > 0:
                btc_rets.append(bc1 / bc0 - 1.0)
                sym_rets.append(sc1 / sc0 - 1.0)

        if len(btc_rets) < 20:
            return 1.0

        # beta = cov(sym, btc) / var(btc)
        mu_btc = statistics.mean(btc_rets)
        mu_sym = statistics.mean(sym_rets)
        cov = sum((br - mu_btc) * (sr - mu_sym) for br, sr in zip(btc_rets, sym_rets)) / len(btc_rets)
        var_btc = sum((br - mu_btc) ** 2 for br in btc_rets) / len(btc_rets)

        if var_btc < 1e-12:
            return 1.0
        return cov / var_btc

    def target_weights(self, dataset: MarketDataset, idx: int, current: Weights) -> Weights:
        syms = dataset.symbols
        btc_sym = self._p("btc_symbol", "BTCUSDT")
        k = max(1, min(self._p("k_per_side", 2), (len(syms) - 1) // 2))
        target_gross = self._p("target_gross_leverage", 0.30)
        vol_lb = self._p("vol_lookback_bars", 168)

        # BTC return = lead signal
        btc_ret = self._compute_btc_return(dataset, idx)

        # Compute beta for each altcoin (exclude BTC)
        alt_syms = [s for s in syms if s != btc_sym]
        if len(alt_syms) < 2 * k:
            return {s: 0.0 for s in syms}

        betas: Dict[str, float] = {}
        for s in alt_syms:
            betas[s] = self._compute_beta(dataset, s, idx)

        # Signal: btc_return × beta
        # BTC up + high beta → long (will catch up)
        # BTC down + high beta → short (will fall harder)
        signal: Dict[str, float] = {}
        for s in alt_syms:
            signal[s] = btc_ret * betas[s]

        sz = zscores(signal)
        signal_z = {s: float(sz.get(s, 0.0)) for s in alt_syms}

        ranked = sorted(alt_syms, key=lambda s: signal_z.get(s, 0.0), reverse=True)
        long_syms = ranked[:k]
        short_syms = ranked[-k:]

        long_syms = [s for s in long_syms if s not in short_syms]
        short_syms = [s for s in short_syms if s not in long_syms]

        if not long_syms or not short_syms:
            return {s: 0.0 for s in syms}

        # Inverse-vol weighting
        inv_vol: Dict[str, float] = {}
        for s in set(long_syms) | set(short_syms):
            v = trailing_vol(dataset.perp_close[s], end_idx=idx, lookback_bars=vol_lb)
            inv_vol[s] = (1.0 / v) if v > 0 else 1.0

        w = normalize_dollar_neutral(long_syms, short_syms, inv_vol, target_gross)
        out = {s: 0.0 for s in syms}
        out.update(w)
        return out
