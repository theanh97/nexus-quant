from __future__ import annotations

import math
from typing import Any, Dict, List, Tuple


# ---------------------------------------------------------------------------
# Pearson correlation (stdlib only)
# ---------------------------------------------------------------------------

def pairwise_correlation(a: List[float], b: List[float]) -> float:
    """
    Pearson correlation coefficient between two return series.

    Uses only the overlapping length (min(len(a), len(b))).
    Returns 0.0 if either series has zero variance or is too short.

    Parameters
    ----------
    a, b : Return series of equal (or truncated-to-equal) length.

    Returns
    -------
    float in [-1, 1].
    """
    n = min(len(a), len(b))
    if n < 2:
        return 0.0

    a = a[-n:]
    b = b[-n:]

    mu_a = sum(a) / n
    mu_b = sum(b) / n

    cov = sum((ai - mu_a) * (bi - mu_b) for ai, bi in zip(a, b))
    var_a = sum((ai - mu_a) ** 2 for ai in a)
    var_b = sum((bi - mu_b) ** 2 for bi in b)

    denom = math.sqrt(var_a * var_b)
    if denom == 0.0:
        return 0.0
    return cov / denom


# ---------------------------------------------------------------------------
# Rolling correlation matrix
# ---------------------------------------------------------------------------

def rolling_correlation_matrix(
    price_series: Dict[str, List[float]],
    window: int = 168,
    high_corr_threshold: float = 0.8,
) -> Dict[str, Any]:
    """
    Compute a pairwise correlation matrix over the most recent *window*
    periods for all symbols in *price_series*.

    The function converts raw price levels to log-returns internally, so
    you may pass either price series or pre-computed return series (the
    log-return of a return series is still meaningful as a difference signal,
    but passing price series is the intended usage).

    Parameters
    ----------
    price_series        : Mapping of symbol -> list of prices (or returns).
    window              : Number of most-recent observations to use.
    high_corr_threshold : Pairs with |corr| above this value are flagged.

    Returns
    -------
    Dict with keys:
        matrix               – symbol -> symbol -> Pearson correlation float.
        max_pair_corr        – Highest absolute pairwise correlation.
        mean_abs_corr        – Mean of all upper-triangle |correlations|.
        high_corr_pairs      – List of (sym_a, sym_b, corr) where |corr| > threshold.
        concentration_warning – True when any pair exceeds the threshold.
    """
    symbols = list(price_series.keys())
    n_sym   = len(symbols)

    # ---- Convert price levels to log-returns over the window ----
    returns_map: Dict[str, List[float]] = {}
    for sym, prices in price_series.items():
        p = prices[-window - 1 :]          # need one extra for differencing
        if len(p) < 2:
            returns_map[sym] = []
        else:
            log_rets: List[float] = []
            for i in range(1, len(p)):
                if p[i - 1] > 0 and p[i] > 0:
                    log_rets.append(math.log(p[i] / p[i - 1]))
                else:
                    log_rets.append(0.0)
            returns_map[sym] = log_rets

    # ---- Build full matrix (including diagonal = 1.0) ----
    matrix: Dict[str, Dict[str, float]] = {s: {} for s in symbols}
    for s in symbols:
        matrix[s][s] = 1.0

    upper_triangle_corrs: List[float] = []
    high_corr_pairs: List[Tuple[str, str, float]] = []
    max_pair_corr: float = 0.0

    for i in range(n_sym):
        for j in range(i + 1, n_sym):
            sym_a = symbols[i]
            sym_b = symbols[j]
            corr  = pairwise_correlation(returns_map[sym_a], returns_map[sym_b])
            matrix[sym_a][sym_b] = corr
            matrix[sym_b][sym_a] = corr

            abs_corr = abs(corr)
            upper_triangle_corrs.append(abs_corr)
            if abs_corr > max_pair_corr:
                max_pair_corr = abs_corr
            if abs_corr > high_corr_threshold:
                high_corr_pairs.append((sym_a, sym_b, corr))

    if upper_triangle_corrs:
        mean_abs_corr = sum(upper_triangle_corrs) / len(upper_triangle_corrs)
    else:
        mean_abs_corr = 0.0

    # Sort high-corr pairs by descending |corr|
    high_corr_pairs.sort(key=lambda t: abs(t[2]), reverse=True)

    return {
        "matrix":                matrix,
        "max_pair_corr":         max_pair_corr,
        "mean_abs_corr":         mean_abs_corr,
        "high_corr_pairs":       high_corr_pairs,
        "concentration_warning": len(high_corr_pairs) > 0,
    }
