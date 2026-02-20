"""
Commodity CTA Feature Engineering
===================================
Pre-computes all features from daily OHLCV data.
Called once by YahooFuturesProvider.load() and stored in MarketDataset.features.

Features computed per symbol:
  Trend   : mom_20d, mom_60d, mom_120d, ewma_signal
  Vol     : atr_14, rv_20d, vol_regime, vol_mom_z
  Value   : rsi_14, zscore_60d, zscore_252d
  Cross   : sector_momentum (group average)

All values are floats; 0.0 where history is insufficient.

Performance: Uses incremental (O(n)) rolling statistics where possible.
"""
from __future__ import annotations

import math
from collections import deque
from typing import Any, Dict, Deque, List, Optional

_NAN = float("nan")


# ── Main entry point ─────────────────────────────────────────────────────────


def compute_features(
    aligned: Dict[str, Dict[str, List[float]]],
    symbols: List[str],
    timeline: List[int],
) -> Dict[str, Any]:
    """
    Compute all commodity features from aligned OHLCV data.

    Args:
        aligned  : {symbol: {"open": [...], "high": [...], "low": [...],
                             "close": [...], "volume": [...]}}
        symbols  : list of valid symbols
        timeline : list of Unix timestamps (one per bar)

    Returns:
        features dict suitable for MarketDataset.features
    """
    n = len(timeline)

    # Per-symbol feature arrays
    atr_14: Dict[str, List[float]] = {}
    rv_20d: Dict[str, List[float]] = {}
    vol_regime: Dict[str, List[float]] = {}
    vol_mom_z: Dict[str, List[float]] = {}
    mom_20d: Dict[str, List[float]] = {}
    mom_60d: Dict[str, List[float]] = {}
    mom_120d: Dict[str, List[float]] = {}
    ewma_signal: Dict[str, List[float]] = {}
    rsi_14: Dict[str, List[float]] = {}
    zscore_60d: Dict[str, List[float]] = {}
    zscore_252d: Dict[str, List[float]] = {}
    # Raw OHLCV re-exported for strategy use
    volume_feat: Dict[str, List[float]] = {}
    open_feat: Dict[str, List[float]] = {}
    high_feat: Dict[str, List[float]] = {}
    low_feat: Dict[str, List[float]] = {}

    for sym in symbols:
        cl = aligned[sym]["close"]
        hi = aligned[sym]["high"]
        lo = aligned[sym]["low"]
        op = aligned[sym]["open"]
        vo = aligned[sym]["volume"]

        # Re-export OHLCV
        volume_feat[sym] = list(vo)
        open_feat[sym] = list(op)
        high_feat[sym] = list(hi)
        low_feat[sym] = list(lo)

        # ATR 14
        atr_14[sym] = _compute_atr(cl, hi, lo, period=14)

        # Realized vol 20d (annualised)
        rv_20d[sym] = _compute_rv(cl, window=20, annualise=True)

        # Vol regime percentile (60-bar rolling: 0=low, 1=high)
        vol_regime[sym] = _compute_rolling_pct(rv_20d[sym], window=60)

        # Volume momentum z-score (short 20d vs long 60d)
        vol_mom_z[sym] = _compute_vol_mom_z(vo, short=20, long=60)

        # Momentum (log return over N days)
        mom_20d[sym] = _compute_momentum(cl, 20)
        mom_60d[sym] = _compute_momentum(cl, 60)
        mom_120d[sym] = _compute_momentum(cl, 120)

        # EWMA crossover signal (12 vs 26-day EWMA)
        ewma_signal[sym] = _compute_ewma_cross(cl, fast=12, slow=26)

        # RSI 14
        rsi_14[sym] = _compute_rsi(cl, period=14)

        # Z-score vs 60-day and 252-day mean/std
        zscore_60d[sym] = _compute_zscore(cl, window=60)
        zscore_252d[sym] = _compute_zscore(cl, window=252)

    # Cross-commodity: sector momentum (equal-weight avg of mom_120d within sector)
    sector_momentum = _compute_sector_momentum(mom_120d, symbols)

    return {
        # OHLCV
        "volume": volume_feat,
        "open": open_feat,
        "high": high_feat,
        "low": low_feat,
        # Trend
        "mom_20d": mom_20d,
        "mom_60d": mom_60d,
        "mom_120d": mom_120d,
        "ewma_signal": ewma_signal,
        # Volatility
        "atr_14": atr_14,
        "rv_20d": rv_20d,
        "vol_regime": vol_regime,
        "vol_mom_z": vol_mom_z,
        # Value / mean-reversion
        "rsi_14": rsi_14,
        "zscore_60d": zscore_60d,
        "zscore_252d": zscore_252d,
        # Cross-commodity
        "sector_momentum": sector_momentum,
    }


# ── Indicator helpers ─────────────────────────────────────────────────────────


def _safe(v: float, fallback: float = 0.0) -> float:
    """Return v if finite, else fallback."""
    return v if (v == v and math.isfinite(v)) else fallback


def _compute_atr(
    close: List[float],
    high: List[float],
    low: List[float],
    period: int = 14,
) -> List[float]:
    """Average True Range (Wilder smoothing)."""
    n = len(close)
    result = [0.0] * n
    if n < period + 1:
        return result

    # Compute True Ranges
    trs = []
    for i in range(1, n):
        h = _safe(high[i], close[i])
        lo = _safe(low[i], close[i])
        pc = _safe(close[i - 1], close[i])
        tr = max(h - lo, abs(h - pc), abs(lo - pc))
        trs.append(tr)

    # First ATR = simple mean of first `period` TRs
    if len(trs) < period:
        return result

    atr = sum(trs[:period]) / period
    result[period] = atr
    for i in range(period + 1, n):
        atr = (atr * (period - 1) + trs[i - 1]) / period
        result[i] = atr

    return result


def _compute_rv(
    close: List[float],
    window: int = 20,
    annualise: bool = True,
) -> List[float]:
    """Realized volatility = rolling std of log returns, annualised. O(n) incremental."""
    n = len(close)
    result = [0.0] * n
    scale = math.sqrt(252) if annualise else 1.0

    # Build log-return series
    log_rets: List[float] = [0.0]
    for i in range(1, n):
        c0 = _safe(close[i - 1])
        c1 = _safe(close[i])
        log_rets.append(math.log(c1 / c0) if c0 > 0 and c1 > 0 else 0.0)

    # Rolling std with incremental deque
    buf: Deque[float] = deque()
    s1, s2 = 0.0, 0.0

    for i in range(n):
        v = log_rets[i]
        buf.append(v)
        s1 += v
        s2 += v * v
        if len(buf) > window:
            old = buf.popleft()
            s1 -= old
            s2 -= old * old
        w = len(buf)
        if w >= window:
            mn = s1 / w
            var = s2 / w - mn * mn
            result[i] = math.sqrt(max(var, 0.0)) * scale

    return result


def _compute_rolling_pct(
    series: List[float],
    window: int = 60,
) -> List[float]:
    """Percentile rank of each value in rolling window (0–1). O(n*w/2) average."""
    n = len(series)
    result = [0.5] * n
    for i in range(window, n):
        v = series[i]
        chunk = series[i - window : i + 1]
        rank = sum(1 for x in chunk if x <= v) / len(chunk)
        result[i] = rank
    return result


def _compute_vol_mom_z(
    volume: List[float],
    short: int = 20,
    long: int = 60,
) -> List[float]:
    """
    Volume momentum z-score: (short_ma - long_ma) / long_std.
    Positive = rising volume = potentially crowded market.
    Transfer from crypto (vol_mom_z_168) → commodity equivalent.
    """
    n = len(volume)
    result = [0.0] * n
    eps = 1e-10

    for i in range(long, n):
        short_chunk = volume[i - short : i]
        long_chunk = volume[i - long : i]
        short_ma = sum(short_chunk) / len(short_chunk) if short_chunk else 0.0
        long_ma = sum(long_chunk) / len(long_chunk) if long_chunk else 0.0
        long_mn = long_ma
        long_std = math.sqrt(
            sum((v - long_mn) ** 2 for v in long_chunk) / len(long_chunk)
        ) if long_chunk else eps
        result[i] = (short_ma - long_ma) / (long_std + eps)

    return result


def _compute_momentum(close: List[float], lookback: int) -> List[float]:
    """Log momentum: log(close[i] / close[i - lookback])."""
    n = len(close)
    result = [0.0] * n
    for i in range(lookback, n):
        c0 = _safe(close[i - lookback])
        c1 = _safe(close[i])
        if c0 > 0 and c1 > 0:
            result[i] = math.log(c1 / c0)
    return result


def _compute_ewma_cross(
    close: List[float],
    fast: int = 12,
    slow: int = 26,
) -> List[float]:
    """
    EWMA crossover signal.
    Returns (fast_ema - slow_ema) / slow_ema.  Positive = uptrend.
    """
    n = len(close)
    result = [0.0] * n
    if n < slow:
        return result

    k_fast = 2.0 / (fast + 1)
    k_slow = 2.0 / (slow + 1)

    ema_f = _safe(close[0], 0.0)
    ema_s = _safe(close[0], 0.0)

    for i in range(1, n):
        c = _safe(close[i], close[i - 1] if i > 0 else 0.0)
        ema_f = c * k_fast + ema_f * (1 - k_fast)
        ema_s = c * k_slow + ema_s * (1 - k_slow)
        if i >= slow and ema_s > 0:
            result[i] = (ema_f - ema_s) / ema_s

    return result


def _compute_rsi(close: List[float], period: int = 14) -> List[float]:
    """Wilder RSI (0–100). Normalised to [-1, 1] for strategy use."""
    n = len(close)
    result = [0.0] * n
    if n < period + 1:
        return result

    gains = []
    losses = []
    for i in range(1, n):
        delta = _safe(close[i]) - _safe(close[i - 1])
        gains.append(max(delta, 0.0))
        losses.append(max(-delta, 0.0))

    if len(gains) < period:
        return result

    avg_gain = sum(gains[:period]) / period
    avg_loss = sum(losses[:period]) / period

    for i in range(period, n):
        avg_gain = (avg_gain * (period - 1) + gains[i - 1]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i - 1]) / period
        if avg_loss == 0:
            raw_rsi = 100.0
        else:
            rs = avg_gain / avg_loss
            raw_rsi = 100.0 - 100.0 / (1.0 + rs)
        # Normalise: 0→-1, 50→0, 100→+1
        result[i] = (raw_rsi - 50.0) / 50.0

    return result


def _compute_zscore(close: List[float], window: int = 60) -> List[float]:
    """
    Rolling z-score using incremental Welford/deque method: O(n).
    (close[i] - rolling_mean) / rolling_std
    """
    n = len(close)
    result = [0.0] * n
    eps = 1e-10

    buf: Deque[float] = deque()
    s1 = 0.0   # sum of values
    s2 = 0.0   # sum of squares

    for i in range(n):
        v = _safe(close[i])
        buf.append(v)
        s1 += v
        s2 += v * v

        if len(buf) > window + 1:
            old = buf.popleft()
            s1 -= old
            s2 -= old * old

        w = len(buf)
        if w >= window:
            mn = s1 / w
            var = s2 / w - mn * mn
            std = math.sqrt(max(var, 0.0))
            result[i] = (v - mn) / (std + eps)

    return result


def _compute_sector_momentum(
    mom_120d: Dict[str, List[float]],
    symbols: List[str],
) -> Dict[str, List[float]]:
    """
    Sector momentum: for each symbol, return the equal-weight average
    mom_120d of all OTHER symbols in the same sector.
    """
    from nexus_quant.projects.commodity_cta.providers.yahoo_futures import SECTOR_MAP

    sector_to_syms: Dict[str, List[str]] = {}
    for sym in symbols:
        sec = SECTOR_MAP.get(sym, "other")
        sector_to_syms.setdefault(sec, []).append(sym)

    result: Dict[str, List[float]] = {}
    n = len(next(iter(mom_120d.values()))) if mom_120d else 0

    for sym in symbols:
        sec = SECTOR_MAP.get(sym, "other")
        peers = [s for s in sector_to_syms.get(sec, []) if s != sym and s in mom_120d]
        arr = [0.0] * n
        if peers:
            for i in range(n):
                vals = [mom_120d[p][i] for p in peers]
                arr[i] = sum(vals) / len(vals)
        result[sym] = arr

    return result
