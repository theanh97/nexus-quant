from __future__ import annotations

import math
from typing import Any, Dict, List


# ---------------------------------------------------------------------------
# Regime labels
# ---------------------------------------------------------------------------

class MarketRegime:
    TRENDING  = "trending"
    RANGING   = "ranging"
    VOLATILE  = "volatile"
    UNKNOWN   = "unknown"


# ---------------------------------------------------------------------------
# Internal statistics helpers (stdlib only)
# ---------------------------------------------------------------------------

def _mean(data: List[float]) -> float:
    return sum(data) / len(data) if data else 0.0


def _stdev(data: List[float]) -> float:
    """Population standard deviation."""
    if len(data) < 2:
        return 0.0
    mu = _mean(data)
    return math.sqrt(sum((x - mu) ** 2 for x in data) / len(data))


def _autocorr_lag1(data: List[float]) -> float:
    """
    Lag-1 autocorrelation of *data* using the standard Pearson formula
    on (data[:-1], data[1:]).
    Returns 0.0 if the series is too short or has zero variance.
    """
    n = len(data)
    if n < 3:
        return 0.0

    x = data[:-1]
    y = data[1:]

    mu_x = _mean(x)
    mu_y = _mean(y)

    cov = sum((xi - mu_x) * (yi - mu_y) for xi, yi in zip(x, y)) / (n - 1)
    sx  = _stdev(x)
    sy  = _stdev(y)

    if sx == 0.0 or sy == 0.0:
        return 0.0
    return cov / (sx * sy)


# ---------------------------------------------------------------------------
# Core regime detection
# ---------------------------------------------------------------------------

def detect_regime(
    returns: List[float],
    lookback: int = 168,            # 1 week of hourly bars
    vol_multiplier_threshold: float = 1.5,
    autocorr_threshold: float = 0.1,
) -> Dict[str, Any]:
    """
    Detect the current market regime from a return series.

    Logic (applied in priority order)
    ----------------------------------
    1. VOLATILE  – recent vol / longer-term vol exceeds *vol_multiplier_threshold*.
    2. TRENDING  – lag-1 autocorrelation  >  +*autocorr_threshold*
                   (returns tend to persist, i.e. momentum).
    3. RANGING   – lag-1 autocorrelation  < -*autocorr_threshold*
                   (returns tend to mean-revert).
    4. UNKNOWN   – insufficient data or inconclusive signal.

    Parameters
    ----------
    returns                  : Full return series up to *now*.
    lookback                 : Number of recent periods used as the
                               "recent" window for vol and autocorr.
    vol_multiplier_threshold : Ratio above which regime is VOLATILE.
    autocorr_threshold       : Absolute autocorrelation threshold for
                               TRENDING / RANGING classification.

    Returns
    -------
    Dict with keys:
        regime    – MarketRegime constant string.
        vol_ratio – recent_vol / longer_vol (NaN if not computable).
        autocorr_1 – Lag-1 autocorrelation of recent returns.
        evidence   – Human-readable explanation string.
    """
    nan = float("nan")

    if len(returns) < lookback:
        return {
            "regime": MarketRegime.UNKNOWN,
            "vol_ratio": nan,
            "autocorr_1": nan,
            "evidence": f"Insufficient data: have {len(returns)} periods, need {lookback}.",
        }

    recent   = returns[-lookback:]
    longer   = returns[-(lookback * 2) :] if len(returns) >= lookback * 2 else returns

    recent_vol = _stdev(recent)
    longer_vol = _stdev(longer)

    if longer_vol == 0.0:
        vol_ratio = nan
    else:
        vol_ratio = recent_vol / longer_vol

    autocorr = _autocorr_lag1(recent)

    # --- Decision tree --------------------------------------------------
    if not math.isnan(vol_ratio) and vol_ratio > vol_multiplier_threshold:
        regime   = MarketRegime.VOLATILE
        evidence = (
            f"Recent vol ({recent_vol:.6f}) is {vol_ratio:.2f}x the "
            f"longer-term vol ({longer_vol:.6f}), exceeding threshold "
            f"{vol_multiplier_threshold:.2f}."
        )
    elif autocorr > autocorr_threshold:
        regime   = MarketRegime.TRENDING
        evidence = (
            f"Lag-1 autocorrelation = {autocorr:.4f} > +{autocorr_threshold} "
            f"suggests persistent (trending) returns."
        )
    elif autocorr < -autocorr_threshold:
        regime   = MarketRegime.RANGING
        evidence = (
            f"Lag-1 autocorrelation = {autocorr:.4f} < -{autocorr_threshold} "
            f"suggests mean-reverting (ranging) returns."
        )
    else:
        regime   = MarketRegime.UNKNOWN
        evidence = (
            f"No clear signal: vol_ratio={vol_ratio:.3f}, "
            f"autocorr_1={autocorr:.4f}."
        )

    return {
        "regime":     regime,
        "vol_ratio":  vol_ratio,
        "autocorr_1": autocorr,
        "evidence":   evidence,
    }


# ---------------------------------------------------------------------------
# Rolling regime series
# ---------------------------------------------------------------------------

def rolling_regime_series(
    returns: List[float],
    lookback: int = 168,
    step: int = 24,
) -> List[Dict[str, Any]]:
    """
    Compute regime snapshots across the return series at every *step* periods.

    Parameters
    ----------
    returns  : Full return series.
    lookback : Look-back window passed to detect_regime.
    step     : How many periods to advance between snapshots.

    Returns
    -------
    List of dicts, each with an additional "end_index" key indicating
    which period the snapshot refers to (inclusive).
    """
    results: List[Dict[str, Any]] = []
    n = len(returns)

    # Start at the first index where we have at least *lookback* history.
    start = lookback - 1
    if start >= n:
        return results

    for end_idx in range(start, n, step):
        snapshot = detect_regime(
            returns[: end_idx + 1],
            lookback=lookback,
        )
        snapshot["end_index"] = end_idx
        results.append(snapshot)

    return results
