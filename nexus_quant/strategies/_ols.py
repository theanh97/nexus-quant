from __future__ import annotations

"""
Minimal rolling Ordinary Least Squares (OLS) for factor model.
Pure stdlib - no numpy, no scipy.

Features: [f1, f2, ..., fN] -> target_return prediction.
Gaussian elimination for small K (< 20 features).
"""

from typing import List, Optional


def _matmul(A: List[List[float]], B: List[List[float]]) -> List[List[float]]:
    rows_a, cols_a = len(A), len(A[0])
    rows_b, cols_b = len(B), len(B[0])
    assert cols_a == rows_b
    C = [[0.0] * cols_b for _ in range(rows_a)]
    for i in range(rows_a):
        for j in range(cols_b):
            s = 0.0
            for k in range(cols_a):
                s += A[i][k] * B[k][j]
            C[i][j] = s
    return C


def _transpose(A: List[List[float]]) -> List[List[float]]:
    rows, cols = len(A), len(A[0])
    return [[A[r][c] for r in range(rows)] for c in range(cols)]


def _solve_gaussian(A: List[List[float]], b: List[float]) -> Optional[List[float]]:
    """Solve A @ x = b using Gaussian elimination with partial pivoting."""
    n = len(A)
    # Build augmented matrix [A | b]
    M = [A[i][:] + [b[i]] for i in range(n)]

    for col in range(n):
        # Partial pivot
        max_row = max(range(col, n), key=lambda r: abs(M[r][col]))
        M[col], M[max_row] = M[max_row], M[col]
        pivot = M[col][col]
        if abs(pivot) < 1e-12:
            return None  # Singular
        for r in range(col + 1, n):
            factor = M[r][col] / pivot
            for c in range(col, n + 1):
                M[r][c] -= factor * M[col][c]

    # Back substitution
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


def ols_fit(X: List[List[float]], y: List[float]) -> Optional[List[float]]:
    """
    Fit OLS: minimize ||y - X @ beta||^2.

    X: N x K matrix (each row is one sample, columns are features)
    y: N-vector of targets
    Returns K-vector beta, or None if singular.
    """
    if not X or not y or len(X) != len(y):
        return None
    n = len(X)
    k = len(X[0])
    if n < k:
        return None  # Under-determined

    # XtX = X.T @ X  (K x K)
    # Xty = X.T @ y  (K)
    XtX = [[0.0] * k for _ in range(k)]
    Xty = [0.0] * k
    for i in range(n):
        xi = X[i]
        yi = y[i]
        for a in range(k):
            Xty[a] += xi[a] * yi
            for b in range(k):
                XtX[a][b] += xi[a] * xi[b]

    # Add tiny ridge regularisation for numerical stability
    lam = 1e-8
    for a in range(k):
        XtX[a][a] += lam

    return _solve_gaussian(XtX, Xty)


def ols_predict(X_row: List[float], beta: List[float]) -> float:
    """Dot product X_row · beta."""
    return sum(x * b for x, b in zip(X_row, beta))


class RollingOLS:
    """
    Maintains a rolling window of (X, y) observations.
    Refits OLS every `refit_every` new observations added.
    """

    def __init__(
        self,
        n_features: int,
        window: int = 504,
        refit_every: int = 24,
        min_samples: int = 60,
    ) -> None:
        self.n_features = n_features
        self.window = window
        self.refit_every = refit_every
        self.min_samples = min_samples

        self._X: List[List[float]] = []
        self._y: List[float] = []
        self._betas: Optional[List[float]] = None
        self._since_last_fit: int = 0

    def update(self, x_row: List[float], y_val: float) -> None:
        """Add new (features, target) observation."""
        if len(x_row) != self.n_features:
            raise ValueError(f"Expected {self.n_features} features, got {len(x_row)}")
        self._X.append(list(x_row))
        self._y.append(float(y_val))
        # Rolling window: drop oldest
        if len(self._X) > self.window:
            self._X.pop(0)
            self._y.pop(0)
        self._since_last_fit += 1
        if self._since_last_fit >= self.refit_every:
            self._refit()

    def _refit(self) -> None:
        if len(self._X) >= self.min_samples:
            self._betas = ols_fit(self._X, self._y)
        self._since_last_fit = 0

    def predict(self, x_row: List[float]) -> Optional[float]:
        """Predict using current beta. Returns None if not ready."""
        if self._betas is None:
            return None
        if len(x_row) != self.n_features:
            return None
        return ols_predict(x_row, self._betas)

    @property
    def is_ready(self) -> bool:
        return self._betas is not None and len(self._X) >= self.min_samples

    @property
    def betas(self) -> Optional[List[float]]:
        return self._betas

    @property
    def n_samples(self) -> int:
        return len(self._X)
