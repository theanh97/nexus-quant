from __future__ import annotations

import math
import statistics
from typing import Any, Dict, List, Tuple


# ---------------------------------------------------------------------------
# Core VaR / CVaR helpers
# ---------------------------------------------------------------------------

def historical_var(returns: List[float], confidence: float = 0.95) -> float:
    """
    Historical Value at Risk at the given confidence level.

    Sorts returns ascending (worst first) and takes the (1 - confidence)
    quantile.  Returns a *positive* number representing the loss magnitude,
    so a return of -0.03 maps to a VaR of +0.03.

    Parameters
    ----------
    returns    : List of period returns (can be negative).
    confidence : Confidence level, e.g. 0.95 for 95 % VaR.

    Returns
    -------
    float  Positive loss magnitude at the given confidence level.
    """
    if not returns:
        return 0.0

    sorted_r = sorted(returns)           # ascending: worst losses first
    n = len(sorted_r)
    # Index of the (1-confidence) quantile; floor so we are conservative
    idx = max(0, math.floor((1.0 - confidence) * n) - 1)
    return -sorted_r[idx]               # negate so positive = loss


def historical_cvar(returns: List[float], confidence: float = 0.95) -> float:
    """
    Conditional VaR (Expected Shortfall) at the given confidence level.

    Computes the mean of all returns that lie *below* the VaR quantile cut-off,
    then negates so the result is a positive loss magnitude.

    Parameters
    ----------
    returns    : List of period returns.
    confidence : Confidence level matching the VaR calculation.

    Returns
    -------
    float  Positive expected loss in the tail.
    """
    if not returns:
        return 0.0

    sorted_r = sorted(returns)
    n = len(sorted_r)
    cutoff_idx = max(1, math.floor((1.0 - confidence) * n))
    tail = sorted_r[:cutoff_idx]        # the worst (1-confidence) fraction
    if not tail:
        return 0.0
    return -(sum(tail) / len(tail))


def rolling_var_series(
    returns: List[float],
    window: int = 252,
    confidence: float = 0.95,
) -> List[float]:
    """
    Rolling VaR computed over a sliding window.

    The first (window - 1) positions are filled with NaN so the output
    has the same length as *returns*.

    Parameters
    ----------
    returns    : Full return series.
    window     : Look-back window size in periods.
    confidence : VaR confidence level.

    Returns
    -------
    List[float]  Rolling VaR series (positive loss magnitude), NaN-padded.
    """
    n = len(returns)
    result: List[float] = [float("nan")] * n
    for i in range(window - 1, n):
        window_slice = returns[i - window + 1 : i + 1]
        result[i] = historical_var(window_slice, confidence)
    return result


# ---------------------------------------------------------------------------
# Higher-order statistics via stdlib only
# ---------------------------------------------------------------------------

def _mean(data: List[float]) -> float:
    return sum(data) / len(data) if data else 0.0


def _variance(data: List[float]) -> float:
    """Population variance."""
    if len(data) < 2:
        return 0.0
    mu = _mean(data)
    return sum((x - mu) ** 2 for x in data) / len(data)


def _skewness(data: List[float]) -> float:
    """Sample skewness (Fisher-Pearson standardised third moment)."""
    n = len(data)
    if n < 3:
        return 0.0
    mu = _mean(data)
    sigma = math.sqrt(_variance(data))
    if sigma == 0.0:
        return 0.0
    third_moment = sum((x - mu) ** 3 for x in data) / n
    return third_moment / (sigma ** 3)


def _kurtosis(data: List[float]) -> float:
    """Excess kurtosis (normal distribution = 0)."""
    n = len(data)
    if n < 4:
        return 0.0
    mu = _mean(data)
    sigma = math.sqrt(_variance(data))
    if sigma == 0.0:
        return 0.0
    fourth_moment = sum((x - mu) ** 4 for x in data) / n
    return (fourth_moment / (sigma ** 4)) - 3.0


# ---------------------------------------------------------------------------
# Full report
# ---------------------------------------------------------------------------

def var_report(
    returns: List[float],
    equity_curve: List[float],
    periods_per_year: float = 8760.0,
) -> Dict[str, Any]:
    """
    Build a comprehensive VaR / risk report from a return series.

    Parameters
    ----------
    returns          : List of period returns (e.g. hourly).
    equity_curve     : Corresponding equity / NAV curve (same length).
    periods_per_year : Number of periods in a calendar year; 8760 for hourly.

    Returns
    -------
    Dict with keys:
        var_95           – 95 % historical VaR (positive loss)
        var_99           – 99 % historical VaR
        cvar_95          – 95 % CVaR / Expected Shortfall
        cvar_99          – 99 % CVaR
        max_loss_1d      – Worst single-period return (as negative number)
        worst_5_periods  – List of (index, return) for the 5 worst periods
        skewness         – Return distribution skewness
        kurtosis         – Excess kurtosis
        annualised_vol   – Annualised volatility of returns
        mean_return      – Mean period return
        sharpe_approx    – Approximate annualised Sharpe (rf = 0)
    """
    if not returns:
        return {
            "var_95": 0.0, "var_99": 0.0,
            "cvar_95": 0.0, "cvar_99": 0.0,
            "max_loss_1d": 0.0, "worst_5_periods": [],
            "skewness": 0.0, "kurtosis": 0.0,
            "annualised_vol": 0.0, "mean_return": 0.0,
            "sharpe_approx": 0.0,
        }

    var_95 = historical_var(returns, 0.95)
    var_99 = historical_var(returns, 0.99)
    cvar_95 = historical_cvar(returns, 0.95)
    cvar_99 = historical_cvar(returns, 0.99)

    max_loss_1d = min(returns)

    indexed = list(enumerate(returns))
    worst_5: List[Tuple[int, float]] = sorted(indexed, key=lambda t: t[1])[:5]

    skew = _skewness(returns)
    kurt = _kurtosis(returns)

    period_vol = math.sqrt(_variance(returns))
    annualised_vol = period_vol * math.sqrt(periods_per_year)

    mean_r = _mean(returns)
    sharpe = (mean_r * periods_per_year) / annualised_vol if annualised_vol > 0 else 0.0

    return {
        "var_95": var_95,
        "var_99": var_99,
        "cvar_95": cvar_95,
        "cvar_99": cvar_99,
        "max_loss_1d": max_loss_1d,
        "worst_5_periods": worst_5,
        "skewness": skew,
        "kurtosis": kurt,
        "annualised_vol": annualised_vol,
        "mean_return": mean_r,
        "sharpe_approx": sharpe,
    }
