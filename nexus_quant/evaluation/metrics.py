from __future__ import annotations

import math
import statistics
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple


def max_drawdown(equity_curve: List[float]) -> float:
    peak = float("-inf")
    mdd = 0.0
    for x in equity_curve:
        ex = float(x)
        if ex > peak:
            peak = ex
        if peak > 0:
            dd = (ex / peak) - 1.0
            if dd < -mdd:
                mdd = -dd
    return float(mdd)


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
    if not returns:
        return 0.0
    mu = statistics.mean(returns)
    sd = statistics.pstdev(returns) if len(returns) > 1 else 0.0
    if sd == 0.0:
        return 0.0
    return float(mu / sd * math.sqrt(float(periods_per_year)))


def sortino(returns: List[float], periods_per_year: float) -> float:
    if not returns:
        return 0.0
    mu = statistics.mean(returns)
    neg = [r for r in returns if r < 0]
    if len(neg) < 2:
        return 0.0
    sd = statistics.pstdev(neg)
    if sd == 0.0:
        return 0.0
    return float(mu / sd * math.sqrt(float(periods_per_year)))


def volatility(returns: List[float], periods_per_year: float) -> float:
    if len(returns) < 2:
        return 0.0
    sd = statistics.pstdev(returns)
    return float(sd * math.sqrt(float(periods_per_year)))


def beta_and_corr(strategy_returns: List[float], benchmark_returns: List[float]) -> Dict[str, float]:
    n = min(len(strategy_returns), len(benchmark_returns))
    if n < 3:
        return {"beta": 0.0, "corr": 0.0}
    a = strategy_returns[:n]
    b = benchmark_returns[:n]
    ma = statistics.mean(a)
    mb = statistics.mean(b)
    cov = sum((a[i] - ma) * (b[i] - mb) for i in range(n)) / float(n)
    vb = sum((b[i] - mb) ** 2 for i in range(n)) / float(n)
    va = sum((a[i] - ma) ** 2 for i in range(n)) / float(n)
    beta = cov / vb if vb > 0 else 0.0
    corr = cov / math.sqrt(va * vb) if va > 0 and vb > 0 else 0.0
    return {"beta": float(beta), "corr": float(corr)}


def equity_from_returns(returns: List[float], start: float = 1.0) -> List[float]:
    eq = [float(start)]
    x = float(start)
    for r in returns:
        x *= 1.0 + float(r)
        eq.append(x)
    return eq


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

    win = 0.0
    if periods > 0:
        win = sum(1 for r in returns if r > 0) / float(periods)

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
