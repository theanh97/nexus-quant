from __future__ import annotations

import statistics
from typing import Dict, Iterable, List, Tuple


def zscores(values: Dict[str, float]) -> Dict[str, float]:
    xs = list(values.values())
    if not xs:
        return {}
    mu = statistics.mean(xs)
    sd = statistics.pstdev(xs) if len(xs) > 1 else 0.0
    if sd == 0.0:
        return {k: 0.0 for k in values.keys()}
    return {k: (v - mu) / sd for k, v in values.items()}


def trailing_returns(closes: List[float]) -> List[float]:
    out = []
    for i in range(1, len(closes)):
        c0 = closes[i - 1]
        c1 = closes[i]
        if c0 == 0:
            out.append(0.0)
        else:
            out.append((c1 / c0) - 1.0)
    return out


def trailing_vol(closes: List[float], end_idx: int, lookback_bars: int) -> float:
    """
    Volatility estimate over [end_idx-lookback_bars, end_idx) using close-to-close returns.
    end_idx is the index in closes (exclusive).
    """
    if lookback_bars <= 1:
        return 0.0
    start = max(1, end_idx - lookback_bars)
    rs = []
    for i in range(start, end_idx):
        c0 = closes[i - 1]
        c1 = closes[i]
        if c0 != 0:
            rs.append((c1 / c0) - 1.0)
    if len(rs) < 2:
        return 0.0
    return statistics.pstdev(rs)


def normalize_dollar_neutral(
    long_syms: List[str],
    short_syms: List[str],
    inv_vol: Dict[str, float],
    target_gross_leverage: float,
) -> Dict[str, float]:
    """
    Allocate weights to achieve:
    - sum(abs(w)) == target_gross_leverage (approximately)
    - sum(w) == 0 (dollar-neutral) if both sides non-empty
    Weights per side are proportional to inv_vol.
    """
    out: Dict[str, float] = {}

    if target_gross_leverage <= 0:
        return {s: 0.0 for s in (long_syms + short_syms)}

    if not long_syms and not short_syms:
        return {}

    if not long_syms or not short_syms:
        # Fall back: all weight on the available side (still bounded by leverage)
        syms = long_syms or short_syms
        side = 1.0 if long_syms else -1.0
        weights = _proportional(inv_vol, syms)
        for s, w in weights.items():
            out[s] = side * w * target_gross_leverage
        return out

    long_budget = target_gross_leverage / 2.0
    short_budget = target_gross_leverage / 2.0

    long_w = _proportional(inv_vol, long_syms)
    short_w = _proportional(inv_vol, short_syms)

    for s, w in long_w.items():
        out[s] = w * long_budget
    for s, w in short_w.items():
        out[s] = -w * short_budget
    return out


def _proportional(weights: Dict[str, float], syms: List[str]) -> Dict[str, float]:
    xs = [max(0.0, float(weights.get(s, 0.0))) for s in syms]
    ssum = sum(xs)
    if ssum <= 0:
        # Equal weights
        n = max(1, len(syms))
        return {s: 1.0 / n for s in syms}
    return {s: max(0.0, float(weights.get(s, 0.0))) / ssum for s in syms}

