"""
Portfolio Optimization Module — stdlib only (no numpy/scipy).

Implements:
1. Risk Parity (Equal Risk Contribution) — iterative algorithm
2. Mean-Variance Optimization (with Ledoit-Wolf-style shrinkage) — analytical
3. Minimum Variance — special case of MV
4. Kelly Criterion position sizing

All implementations use pure Python stdlib (math, statistics, itertools).
Can be used as a plug-in for any strategy position sizing.
"""
from __future__ import annotations

import math
import statistics
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Internal linear-algebra helpers (pure Python, no numpy)
# ---------------------------------------------------------------------------

def _mat_identity(n: int) -> List[List[float]]:
    """Return n x n identity matrix."""
    return [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]


def _mat_scale(mat: List[List[float]], scalar: float) -> List[List[float]]:
    return [[mat[i][j] * scalar for j in range(len(mat[0]))] for i in range(len(mat))]


def _mat_add(a: List[List[float]], b: List[List[float]]) -> List[List[float]]:
    n = len(a)
    return [[a[i][j] + b[i][j] for j in range(n)] for i in range(n)]


def _mat_diag(mat: List[List[float]]) -> List[List[float]]:
    n = len(mat)
    return [[mat[i][j] if i == j else 0.0 for j in range(n)] for i in range(n)]


def _mat_inv_2x2(mat: List[List[float]]) -> Optional[List[List[float]]]:
    a, b = mat[0][0], mat[0][1]
    c, d = mat[1][0], mat[1][1]
    det = a * d - b * c
    if abs(det) < 1e-15:
        return None
    inv_det = 1.0 / det
    return [[d * inv_det, -b * inv_det], [-c * inv_det, a * inv_det]]


def _mat_inv_3x3(mat: List[List[float]]) -> Optional[List[List[float]]]:
    m = mat
    det = (
        m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
        - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
        + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0])
    )
    if abs(det) < 1e-15:
        return None
    inv_det = 1.0 / det
    return [
        [
            (m[1][1] * m[2][2] - m[1][2] * m[2][1]) * inv_det,
            (m[0][2] * m[2][1] - m[0][1] * m[2][2]) * inv_det,
            (m[0][1] * m[1][2] - m[0][2] * m[1][1]) * inv_det,
        ],
        [
            (m[1][2] * m[2][0] - m[1][0] * m[2][2]) * inv_det,
            (m[0][0] * m[2][2] - m[0][2] * m[2][0]) * inv_det,
            (m[0][2] * m[1][0] - m[0][0] * m[1][2]) * inv_det,
        ],
        [
            (m[1][0] * m[2][1] - m[1][1] * m[2][0]) * inv_det,
            (m[0][1] * m[2][0] - m[0][0] * m[2][1]) * inv_det,
            (m[0][0] * m[1][1] - m[0][1] * m[1][0]) * inv_det,
        ],
    ]


def _mat_inv_diag(mat: List[List[float]]) -> List[List[float]]:
    n = len(mat)
    result = [[0.0] * n for _ in range(n)]
    for i in range(n):
        diag_val = mat[i][i]
        result[i][i] = 1.0 / diag_val if abs(diag_val) > 1e-15 else 0.0
    return result


def _mat_vec_mul(mat: List[List[float]], vec: List[float]) -> List[float]:
    return [sum(mat[i][j] * vec[j] for j in range(len(vec))) for i in range(len(mat))]


def _invert_matrix(mat: List[List[float]]) -> List[List[float]]:
    n = len(mat)
    if n == 1:
        val = mat[0][0]
        return [[1.0 / val if abs(val) > 1e-15 else 0.0]]
    if n == 2:
        inv = _mat_inv_2x2(mat)
        return inv if inv is not None else _mat_inv_diag(_mat_diag(mat))
    if n == 3:
        inv = _mat_inv_3x3(mat)
        return inv if inv is not None else _mat_inv_diag(_mat_diag(mat))
    # n >= 4: diagonal (shrinkage-like) approximation
    return _mat_inv_diag(_mat_diag(mat))


# ---------------------------------------------------------------------------
# Covariance computation
# ---------------------------------------------------------------------------

def compute_rolling_cov(
    returns_dict: Dict[str, List[float]],
    window: int = 168,
) -> Dict[str, Dict[str, float]]:
    symbols = list(returns_dict.keys())
    n = len(symbols)
    cov: Dict[str, Dict[str, float]] = {s: {t: 0.0 for t in symbols} for s in symbols}

    if n == 0:
        return cov

    trimmed: Dict[str, List[float]] = {}
    for sym in symbols:
        series = returns_dict[sym]
        trimmed[sym] = series[-window:] if len(series) >= window else list(series)

    min_len = min(len(trimmed[s]) for s in symbols)
    if min_len < 2:
        for sym in symbols:
            try:
                v = statistics.pvariance(trimmed[sym]) if trimmed[sym] else 1e-8
            except Exception:
                v = 1e-8
            cov[sym][sym] = v if v > 0 else 1e-8
        return cov

    aligned: Dict[str, List[float]] = {s: trimmed[s][-min_len:] for s in symbols}
    means: Dict[str, float] = {s: statistics.mean(aligned[s]) for s in symbols}
    denom = min_len - 1

    for i, s_i in enumerate(symbols):
        for j, s_j in enumerate(symbols):
            if j < i:
                cov[s_i][s_j] = cov[s_j][s_i]
                continue
            dev_i = [aligned[s_i][k] - means[s_i] for k in range(min_len)]
            dev_j = [aligned[s_j][k] - means[s_j] for k in range(min_len)]
            cov[s_i][s_j] = sum(dev_i[k] * dev_j[k] for k in range(min_len)) / denom

    return cov


# ---------------------------------------------------------------------------
# Kelly Criterion
# ---------------------------------------------------------------------------

def kelly_fraction(
    mean_ret: float,
    variance: float,
    kelly_frac: float = 0.25,
) -> float:
    if variance <= 0.0:
        return 0.0
    full_kelly = mean_ret / variance
    return max(-1.0, min(1.0, kelly_frac * full_kelly))


# ---------------------------------------------------------------------------
# Risk Parity Optimizer
# ---------------------------------------------------------------------------

class RiskParityOptimizer:
    def __init__(self, n_iter: int = 10) -> None:
        self.n_iter = n_iter

    def optimize(
        self,
        vol_dict: Dict[str, float],
        target_gross: float = 1.0,
    ) -> Dict[str, float]:
        if not vol_dict or target_gross <= 0.0:
            return {s: 0.0 for s in vol_dict}

        symbols = [s for s, v in vol_dict.items() if v > 0.0]
        zero_syms = [s for s, v in vol_dict.items() if v <= 0.0]
        n = len(symbols)

        if n == 0:
            return {s: 0.0 for s in vol_dict}

        vols = [vol_dict[s] for s in symbols]
        inv_vols = [1.0 / v for v in vols]
        weights = self._normalize(inv_vols, target_gross)

        for _ in range(self.n_iter):
            weights = self._erc_step(weights, vols, target_gross)

        out: Dict[str, float] = {}
        for s in zero_syms:
            out[s] = 0.0
        for s, w in zip(symbols, weights):
            out[s] = w
        return out

    @staticmethod
    def _normalize(raw: List[float], target: float) -> List[float]:
        total = sum(raw)
        if total <= 0.0:
            n = max(1, len(raw))
            return [target / n] * len(raw)
        scale = target / total
        return [x * scale for x in raw]

    @staticmethod
    def _erc_step(weights: List[float], vols: List[float], target_gross: float) -> List[float]:
        n = len(weights)
        port_var = sum(weights[i] ** 2 * vols[i] ** 2 for i in range(n))
        if port_var <= 0.0:
            inv_vols = [1.0 / v if v > 0 else 0.0 for v in vols]
            return RiskParityOptimizer._normalize(inv_vols, target_gross)
        port_vol = math.sqrt(port_var)
        mrc = [(weights[i] * vols[i] ** 2) / port_vol for i in range(n)]
        new_w_raw = [
            weights[i] / mrc[i] if mrc[i] > 1e-15 else 1.0 / vols[i]
            for i in range(n)
        ]
        return RiskParityOptimizer._normalize(new_w_raw, target_gross)


# ---------------------------------------------------------------------------
# Mean-Variance Optimizer
# ---------------------------------------------------------------------------

class MeanVarianceOptimizer:
    def __init__(self, shrinkage_alpha: float = 0.2) -> None:
        self.shrinkage_alpha = max(0.0, min(1.0, shrinkage_alpha))

    def optimize(
        self,
        expected_returns: Dict[str, float],
        cov_matrix: Dict[str, Dict[str, float]],
        target_gross: float = 1.0,
        max_weight: float = 0.4,
    ) -> Dict[str, float]:
        symbols = list(expected_returns.keys())
        n = len(symbols)
        zero_out: Dict[str, float] = {s: 0.0 for s in symbols}

        if n == 0 or target_gross <= 0.0:
            return zero_out

        raw_cov = [
            [cov_matrix.get(symbols[i], {}).get(symbols[j], 0.0) for j in range(n)]
            for i in range(n)
        ]
        for i in range(n):
            if raw_cov[i][i] <= 0.0:
                raw_cov[i][i] = 1e-8

        diag_cov = _mat_diag(raw_cov)
        shrunk_cov = _mat_add(
            _mat_scale(raw_cov, 1.0 - self.shrinkage_alpha),
            _mat_scale(diag_cov, self.shrinkage_alpha),
        )

        try:
            inv_cov = _invert_matrix(shrunk_cov)
        except Exception:
            inv_cov = _mat_inv_diag(diag_cov)

        mu = [expected_returns[s] for s in symbols]
        w_raw = _mat_vec_mul(inv_cov, mu)

        abs_cap = max_weight * target_gross
        w_capped = [max(-abs_cap, min(abs_cap, w_raw[i])) for i in range(n)]

        gross = sum(abs(w) for w in w_capped)
        if gross <= 0.0:
            return zero_out

        scale = target_gross / gross
        return {symbols[i]: w_capped[i] * scale for i in range(n)}

    def minimum_variance(
        self,
        cov_matrix: Dict[str, Dict[str, float]],
        target_gross: float = 1.0,
        max_weight: float = 0.4,
        long_only: bool = True,
    ) -> Dict[str, float]:
        symbols = list(cov_matrix.keys())
        uniform_mu = {s: 1.0 for s in symbols}
        weights = self.optimize(
            expected_returns=uniform_mu,
            cov_matrix=cov_matrix,
            target_gross=target_gross,
            max_weight=max_weight,
        )
        if long_only:
            raw = {s: max(0.0, w) for s, w in weights.items()}
            gross = sum(raw.values())
            if gross <= 0.0:
                n = max(1, len(symbols))
                return {s: target_gross / n for s in symbols}
            scale = target_gross / gross
            return {s: v * scale for s, v in raw.items()}
        return weights
