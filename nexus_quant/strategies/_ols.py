from __future__ import annotations

"""
Rolling regression for ML Factor strategy.

Uses numpy + scikit-learn Ridge regression when available (100x faster).
Falls back to pure stdlib Gaussian elimination if numpy is not installed.

API is identical to the original — drop-in replacement.
"""

from typing import List, Optional

# ── Try to import numpy/sklearn ────────────────────────────────────────────
try:
    import numpy as _np
    from sklearn.linear_model import Ridge as _Ridge
    _HAS_NUMPY = True
except ImportError:
    _HAS_NUMPY = False


# ── Stdlib fallback (original implementation) ─────────────────────────────

def _solve_gaussian(A: List[List[float]], b: List[float]) -> Optional[List[float]]:
    n = len(A)
    M = [A[i][:] + [b[i]] for i in range(n)]
    for col in range(n):
        max_row = max(range(col, n), key=lambda r: abs(M[r][col]))
        M[col], M[max_row] = M[max_row], M[col]
        pivot = M[col][col]
        if abs(pivot) < 1e-12:
            return None
        for r in range(col + 1, n):
            factor = M[r][col] / pivot
            for c in range(col, n + 1):
                M[r][c] -= factor * M[col][c]
    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        s = M[i][n]
        for j in range(i + 1, n):
            s -= M[i][j] * x[j]
        denom = M[i][i]
        if abs(denom) < 1e-12:
            return None
        x[i] = s / denom
    return x


def _ols_fit_stdlib(X: List[List[float]], y: List[float]) -> Optional[List[float]]:
    n, k = len(X), len(X[0])
    if n < k:
        return None
    XtX = [[0.0] * k for _ in range(k)]
    Xty = [0.0] * k
    for i in range(n):
        xi, yi = X[i], y[i]
        for a in range(k):
            Xty[a] += xi[a] * yi
            for b in range(k):
                XtX[a][b] += xi[a] * xi[b]
    lam = 1e-4  # small ridge for stability
    for a in range(k):
        XtX[a][a] += lam
    return _solve_gaussian(XtX, Xty)


# ── Numpy/sklearn fast path ────────────────────────────────────────────────

def _ols_fit_numpy(X_arr: "_np.ndarray", y_arr: "_np.ndarray", alpha: float = 1e-3) -> Optional["_np.ndarray"]:
    """Ridge regression via sklearn. alpha = L2 penalty (default 1e-3 = weak regularization)."""
    try:
        model = _Ridge(alpha=alpha, fit_intercept=False, max_iter=200)
        model.fit(X_arr, y_arr)
        return model.coef_
    except Exception:
        return None


# ── Public API (compatible with original) ────────────────────────────────

def ols_fit(X: List[List[float]], y: List[float]) -> Optional[List[float]]:
    """
    Fit Ridge regression (or OLS with L2 if sklearn unavailable).

    X: N x K matrix (each row = one sample)
    y: N-vector of targets
    Returns K-vector beta, or None if singular/failed.
    """
    if not X or not y or len(X) != len(y):
        return None
    if _HAS_NUMPY:
        try:
            X_arr = _np.array(X, dtype=_np.float64)
            y_arr = _np.array(y, dtype=_np.float64)
            coef = _ols_fit_numpy(X_arr, y_arr)
            if coef is not None:
                return coef.tolist()
        except Exception:
            pass
    return _ols_fit_stdlib(X, y)


def ols_predict(X_row: List[float], beta: List[float]) -> float:
    if _HAS_NUMPY:
        return float(_np.dot(X_row, beta))
    return sum(x * b for x, b in zip(X_row, beta))


class RollingOLS:
    """
    Rolling window regression: maintains (X, y) buffer, refits every
    `refit_every` new observations.

    When numpy/sklearn is available, uses Ridge regression (fast + well-regularized).
    Falls back to pure-stdlib OLS with small ridge penalty otherwise.
    """

    def __init__(
        self,
        n_features: int,
        window: int = 504,
        refit_every: int = 24,
        min_samples: int = 60,
        ridge_alpha: float = 1e-3,
    ) -> None:
        self.n_features = n_features
        self.window = window
        self.refit_every = refit_every
        self.min_samples = min_samples
        self.ridge_alpha = ridge_alpha

        # Use numpy arrays as ring buffers when available
        if _HAS_NUMPY:
            self._X_buf: Optional["_np.ndarray"] = None  # (window, n_features)
            self._y_buf: Optional["_np.ndarray"] = None  # (window,)
            self._buf_n: int = 0     # total items added
            self._buf_size: int = 0  # current items in buffer (min of window and buf_n)
        else:
            self._X: List[List[float]] = []
            self._y: List[float] = []

        self._betas: Optional[List[float]] = None
        self._since_last_fit: int = 0

    def update(self, x_row: List[float], y_val: float) -> None:
        if len(x_row) != self.n_features:
            raise ValueError(f"Expected {self.n_features} features, got {len(x_row)}")

        if _HAS_NUMPY:
            # Initialize ring buffers on first call
            if self._X_buf is None:
                self._X_buf = _np.empty((self.window, self.n_features), dtype=_np.float64)
                self._y_buf = _np.empty(self.window, dtype=_np.float64)
            slot = self._buf_n % self.window
            self._X_buf[slot] = x_row
            self._y_buf[slot] = y_val
            self._buf_n += 1
            self._buf_size = min(self._buf_n, self.window)
        else:
            self._X.append(list(x_row))
            self._y.append(float(y_val))
            if len(self._X) > self.window:
                self._X.pop(0)
                self._y.pop(0)

        self._since_last_fit += 1
        if self._since_last_fit >= self.refit_every:
            self._refit()

    def _get_current_data(self):
        """Return (X_array, y_array) for current buffer contents."""
        if _HAS_NUMPY:
            n = self._buf_size
            if n < self.min_samples:
                return None, None
            # Return current window slice (ring buffer → ordered)
            if self._buf_n <= self.window:
                # Not yet wrapped around
                return self._X_buf[:n], self._y_buf[:n]
            else:
                # Ring has wrapped; order oldest→newest
                end = self._buf_n % self.window
                idx = _np.arange(n)
                order = (end + idx) % self.window
                return self._X_buf[order], self._y_buf[order]
        else:
            if len(self._X) < self.min_samples:
                return None, None
            return self._X, self._y

    def _refit(self) -> None:
        X, y = self._get_current_data()
        if X is not None:
            if _HAS_NUMPY:
                coef = _ols_fit_numpy(X, y, alpha=self.ridge_alpha)
                self._betas = coef.tolist() if coef is not None else None
            else:
                self._betas = _ols_fit_stdlib(list(X), list(y))
        self._since_last_fit = 0

    def predict(self, x_row: List[float]) -> Optional[float]:
        if self._betas is None:
            return None
        if len(x_row) != self.n_features:
            return None
        return ols_predict(x_row, self._betas)

    @property
    def is_ready(self) -> bool:
        n = self._buf_size if _HAS_NUMPY else len(self._X) if hasattr(self, '_X') else 0
        return self._betas is not None and n >= self.min_samples

    @property
    def betas(self) -> Optional[List[float]]:
        return self._betas

    @property
    def n_samples(self) -> int:
        if _HAS_NUMPY:
            return self._buf_size
        return len(self._X) if hasattr(self, '_X') else 0
