from __future__ import annotations

import math
import statistics
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


def max_drawdown(equity_curve: List[float]) -> float:
    eq = np.asarray(equity_curve, dtype=np.float64)
    if len(eq) < 2:
        return 0.0
    peak = np.maximum.accumulate(eq)
    mask = peak > 0
    dd = np.where(mask, (eq / peak) - 1.0, 0.0)
    return float(-dd.min()) if dd.min() < 0 else 0.0


def cagr(equity_start: float, equity_end: float, periods: int, periods_per_year: float) -> float:
    if equity_start <= 0 or equity_end <= 0:
        return 0.0
    if periods_per_year <= 0:
        return 0.0
    years = float(periods) / float(periods_per_year)
    if years <= 0:
        return 0.0
    return float((equity_end / equity_start) ** (1.0 / years) - 1.0)


def sharpe(returns: List[float], periods_per_year: float) -> float:
    r = np.asarray(returns, dtype=np.float64)
    if len(r) < 2:
        return 0.0
    sd = float(r.std(ddof=1))
    if sd == 0.0:
        return 0.0
    return float(r.mean() / sd * math.sqrt(float(periods_per_year)))


def sortino(returns: List[float], periods_per_year: float) -> float:
    r = np.asarray(returns, dtype=np.float64)
    if len(r) == 0:
        return 0.0
    mu = float(r.mean())
    neg = r[r < 0]
    if len(neg) < 2:
        return 0.0
    sd = float(neg.std(ddof=1))
    if sd == 0.0:
        return 0.0
    return float(mu / sd * math.sqrt(float(periods_per_year)))


def volatility(returns: List[float], periods_per_year: float) -> float:
    r = np.asarray(returns, dtype=np.float64)
    if len(r) < 2:
        return 0.0
    return float(r.std(ddof=1) * math.sqrt(float(periods_per_year)))


def beta_and_corr(strategy_returns: List[float], benchmark_returns: List[float]) -> Dict[str, float]:
    n = min(len(strategy_returns), len(benchmark_returns))
    if n < 3:
        return {"beta": 0.0, "corr": 0.0}
    a = np.asarray(strategy_returns[:n], dtype=np.float64)
    b = np.asarray(benchmark_returns[:n], dtype=np.float64)
    cov_mat = np.cov(a, b, ddof=0)
    va, vb, cov = cov_mat[0, 0], cov_mat[1, 1], cov_mat[0, 1]
    beta = cov / vb if vb > 0 else 0.0
    corr = cov / math.sqrt(va * vb) if va > 0 and vb > 0 else 0.0
    return {"beta": float(beta), "corr": float(corr)}


def equity_from_returns(returns: List[float], start: float = 1.0) -> List[float]:
    r = np.asarray(returns, dtype=np.float64)
    eq = np.empty(len(r) + 1, dtype=np.float64)
    eq[0] = float(start)
    np.cumprod(1.0 + r, out=eq[1:])
    eq[1:] *= float(start)
    return eq.tolist()


def summarize(
    returns: List[float],
    equity_curve: List[float],
    periods_per_year: float,
    trades: Optional[List[Dict[str, Any]]] = None,
    benchmark_returns: Optional[List[float]] = None,
) -> Dict[str, Any]:
    periods = len(returns)
    eq0 = float(equity_curve[0]) if equity_curve else 1.0
    eq1 = float(equity_curve[-1]) if equity_curve else 1.0

    mdd = max_drawdown(equity_curve)
    cg = cagr(eq0, eq1, periods=periods, periods_per_year=periods_per_year)
    calmar = (cg / mdd) if mdd > 0 else 0.0
    sh = sharpe(returns, periods_per_year=periods_per_year)
    so = sortino(returns, periods_per_year=periods_per_year)
    vol = volatility(returns, periods_per_year=periods_per_year)

    r_arr = np.asarray(returns, dtype=np.float64)
    win = float((r_arr > 0).sum() / periods) if periods > 0 else 0.0

    t_total = 0.0
    t_avg = 0.0
    t_max = 0.0
    if trades:
        t_total = sum(float(x.get("turnover", 0.0)) for x in trades)
        t_avg = t_total / float(len(trades)) if trades else 0.0
        t_max = max((float(x.get("turnover", 0.0)) for x in trades), default=0.0)

    out: Dict[str, Any] = {
        "periods": periods,
        "equity_start": eq0,
        "equity_end": eq1,
        "total_return": (eq1 / eq0) - 1.0 if eq0 != 0 else 0.0,
        "cagr": cg,
        "volatility": vol,
        "sharpe": sh,
        "sortino": so,
        "max_drawdown": mdd,
        "calmar": calmar,
        "win_rate": win,
        "turnover_total": t_total,
        "turnover_avg": t_avg,
        "turnover_max": t_max,
    }

    if benchmark_returns is not None:
        out.update(beta_and_corr(strategy_returns=returns, benchmark_returns=benchmark_returns))

    return out
