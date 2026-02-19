from __future__ import annotations

"""
ML Factor V3 - Per-Symbol Time-Series Binary Direction Prediction

Replaces cross-sectional OLS (which fails with only 5-10 symbols) with
per-symbol rolling logistic regression that classifies the DIRECTION of
the next 24h return (+1 up / -1 down).

Features (5, all causal):
  0. norm_ret_24h  : (close[t-1] - close[t-25]) / close[t-25]
  1. norm_ret_168h : (close[t-1] - close[t-169]) / close[t-169]
  2. funding_z     : latest_funding / std(last_30_funding_rates), clipped [-5,5]
  3. vol_ratio     : std(last_24h_rets) / std(last_168h_rets), clipped [0.1, 10]
  4. price_vs_sma  : (close[t-1] - mean(last_168_closes)) / mean(last_168_closes)

Label: +1 if close[t+24] > close[t-1], else -1

Model: batch logistic regression (SGD on rolling window, refit every N bars).
No external libraries — stdlib + math only.

Score = sigmoid(dot(theta, features)) - 0.5  in (-0.5, +0.5)
Portfolio: top-k score -> LONG, bottom-k score -> SHORT (dollar-neutral, inv-vol weighted)
"""

import math
import statistics
from collections import deque
from typing import Any, Deque, Dict, List, Optional, Tuple

from ._math import normalize_dollar_neutral, trailing_vol
from .base import Strategy, Weights
from ..data.schema import MarketDataset


# ---------------------------------------------------------------------------
# Numerics helpers (stdlib-only)
# ---------------------------------------------------------------------------

_N_FEATURES = 5


def _sigmoid(x: float) -> float:
    """Numerically stable sigmoid."""
    x = max(-500.0, min(500.0, x))
    return 1.0 / (1.0 + math.exp(-x))


def _dot(a: List[float], b: List[float]) -> float:
    return sum(ai * bi for ai, bi in zip(a, b))


def _logistic_loss_grad(theta: List[float], x: List[float], y: float) -> List[float]:
    """Gradient of logistic loss wrt theta: -y * x / (1 + exp(y * dot(theta, x)))"""
    margin = y * _dot(theta, x)
    margin = max(-500.0, min(500.0, margin))
    factor = -y / (1.0 + math.exp(margin))
    return [factor * xi for xi in x]


def _batch_logistic_fit(
    samples: List[Tuple[List[float], float]],
    n_features: int,
    n_epochs: int = 5,
    lr: float = 0.01,
    l2: float = 1e-4,
) -> List[float]:
    """
    Fit logistic regression via full-batch SGD on the given samples.
    Returns theta (weight vector, length n_features).
    """
    theta = [0.0] * n_features
    n = len(samples)
    if n == 0:
        return theta
    for _ in range(n_epochs):
        for x, y in samples:
            grad = _logistic_loss_grad(theta, x, y)
            # L2 regularisation + gradient step
            theta = [
                theta[i] * (1.0 - lr * l2) - lr * grad[i]
                for i in range(n_features)
            ]
    return theta


# ---------------------------------------------------------------------------
# Per-symbol model state
# ---------------------------------------------------------------------------

class _SymbolModel:
    """
    Maintains a rolling training buffer for one symbol and a logistic weight vector.
    """

    def __init__(
        self,
        window: int,
        refit_every: int,
        lr: float,
        min_warmup: int,
    ) -> None:
        self.window = window
        self.refit_every = refit_every
        self.lr = lr
        self.min_warmup = min_warmup

        # Rolling buffer of (features, label) pairs
        self._buf: Deque[Tuple[List[float], float]] = deque(maxlen=window)
        # Current weight vector
        self.theta: List[float] = [0.0] * _N_FEATURES
        # How many times model has been fed a sample
        self._n_seen: int = 0
        # Counter since last refit
        self._since_refit: int = 0
        # Whether model has been trained at least once
        self.is_ready: bool = False

    def push(self, features: List[float], label: float) -> None:
        """Add one (features, label) training pair and optionally refit."""
        self._buf.append((features, label))
        self._n_seen += 1
        self._since_refit += 1

        if self._n_seen < self.min_warmup:
            return  # not enough data yet

        if self._since_refit >= self.refit_every:
            self._refit()
            self._since_refit = 0

    def _refit(self) -> None:
        samples = list(self._buf)
        if len(samples) < 10:
            return
        self.theta = _batch_logistic_fit(
            samples,
            n_features=_N_FEATURES,
            n_epochs=5,
            lr=self.lr,
            l2=1e-4,
        )
        self.is_ready = True

    def predict_score(self, features: List[float]) -> float:
        """Return sigmoid(dot(theta, features)) - 0.5 in (-0.5, +0.5)."""
        return _sigmoid(_dot(self.theta, features)) - 0.5


# ---------------------------------------------------------------------------
# Feature extraction helpers
# ---------------------------------------------------------------------------

def _safe_norm_return(closes: List[float], end_idx: int, lookback: int) -> float:
    """(close[end_idx] - close[end_idx - lookback]) / close[end_idx - lookback]"""
    start = end_idx - lookback
    if start < 0 or end_idx >= len(closes):
        return 0.0
    c0 = closes[start]
    c1 = closes[end_idx]
    if c0 == 0.0:
        return 0.0
    return (c1 - c0) / c0


def _rolling_std(values: List[float], end_idx: int, lookback: int) -> float:
    """Population std of closes[end_idx-lookback:end_idx] returns."""
    start = max(0, end_idx - lookback)
    window = values[start:end_idx]
    if len(window) < 2:
        return 0.0
    mu = sum(window) / len(window)
    var = sum((v - mu) ** 2 for v in window) / len(window)
    return math.sqrt(var) if var > 0 else 0.0


def _rolling_mean(values: List[float], end_idx: int, lookback: int) -> float:
    start = max(0, end_idx - lookback)
    window = values[start:end_idx]
    if not window:
        return 0.0
    return sum(window) / len(window)


def _returns_list(closes: List[float], end_idx: int, lookback: int) -> List[float]:
    """Bar-to-bar returns for closes[end_idx-lookback-1:end_idx]."""
    start = max(1, end_idx - lookback)
    out = []
    for i in range(start, min(end_idx, len(closes))):
        c0 = closes[i - 1]
        c1 = closes[i]
        if c0 != 0:
            out.append((c1 - c0) / c0)
    return out


def _std_of_list(vals: List[float]) -> float:
    if len(vals) < 2:
        return 0.0
    mu = sum(vals) / len(vals)
    var = sum((v - mu) ** 2 for v in vals) / len(vals)
    return math.sqrt(var) if var > 0 else 0.0


def _build_features(
    dataset: MarketDataset,
    symbol: str,
    idx: int,
    funding_history: List[float],
) -> Optional[List[float]]:
    """
    Build 5-feature vector at bar idx.
    Uses close[idx-1] as the most-recent observable (causal, no lookahead).
    Returns None if not enough history.
    """
    min_needed = 170  # need 168h lookback + 1 lag + safety
    if idx < min_needed:
        return None

    closes = dataset.perp_close[symbol]
    if len(closes) <= idx:
        return None

    t = idx - 1  # last observable bar

    # Feature 0: norm_ret_24h
    f0 = _safe_norm_return(closes, t, 24)

    # Feature 1: norm_ret_168h
    f1 = _safe_norm_return(closes, t, 168)

    # Feature 2: funding_z
    if len(funding_history) >= 2:
        latest_f = funding_history[-1] if funding_history else 0.0
        std_f = _std_of_list(funding_history[-30:])
        f2 = (latest_f / std_f) if std_f > 0 else 0.0
        f2 = max(-5.0, min(5.0, f2))
    else:
        ts = dataset.timeline[idx]
        f2 = float(dataset.last_funding_rate_before(symbol, ts))
        f2 = max(-5.0, min(5.0, f2))

    # Feature 3: vol_ratio (short-window vol / long-window vol)
    rets_24 = _returns_list(closes, t + 1, 24)   # last 24 bar returns ending at t
    rets_168 = _returns_list(closes, t + 1, 168)
    std_24 = _std_of_list(rets_24)
    std_168 = _std_of_list(rets_168)
    if std_168 > 0:
        f3 = std_24 / std_168
    else:
        f3 = 1.0
    f3 = max(0.1, min(10.0, f3))

    # Feature 4: price_vs_sma
    sma_168 = _rolling_mean(closes, t + 1, 168)  # mean of last 168 closes up to t
    if sma_168 > 0:
        f4 = (closes[t] - sma_168) / sma_168
    else:
        f4 = 0.0

    return [f0, f1, f2, f3, f4]


# ---------------------------------------------------------------------------
# Strategy
# ---------------------------------------------------------------------------

class MLFactorV3Strategy(Strategy):
    """
    Per-symbol time-series logistic regression for next-24h direction prediction.

    Params:
      k_per_side              : int   = 2
      ols_window              : int   = 720   (training buffer size)
      ols_refit_every         : int   = 24    (refit every N bars)
      min_warmup_bars         : int   = 336   (bars before first prediction)
      learning_rate           : float = 0.01  (SGD lr)
      vol_lookback_bars       : int   = 168
      rebalance_interval_bars : int   = 24
      target_gross_leverage   : float = 0.3
      risk_weighting          : str   = "inverse_vol"
    """

    def __init__(self, params: Dict[str, Any]) -> None:
        super().__init__(name="ml_factor_v3", params=params)

        self._ols_window = int(params.get("ols_window") or 720)
        self._ols_refit_every = int(params.get("ols_refit_every") or 24)
        self._min_warmup = int(params.get("min_warmup_bars") or 336)
        self._lr = float(params.get("learning_rate") or 0.01)

        # Per-symbol models (created lazily)
        self._models: Dict[str, _SymbolModel] = {}

        # Per-symbol funding history (for funding_z feature)
        self._funding_hist: Dict[str, List[float]] = {}

        # Guard: last idx we pushed training samples at
        self._last_push_idx: int = -1

    def _get_model(self, symbol: str) -> _SymbolModel:
        if symbol not in self._models:
            self._models[symbol] = _SymbolModel(
                window=self._ols_window,
                refit_every=self._ols_refit_every,
                lr=self._lr,
                min_warmup=self._min_warmup,
            )
        return self._models[symbol]

    def _update_models(self, dataset: MarketDataset, idx: int) -> None:
        """
        At bar idx, feed realized label for features built at idx-24.
        Label = +1 if close[idx-1] > close[idx-25], else -1.
        (We use a 24-bar lookahead from the feature point idx-24.)
        Only run once per bar (idempotent guard).
        """
        if idx == self._last_push_idx:
            return
        self._last_push_idx = idx

        # Update funding history for all symbols
        ts = dataset.timeline[idx]
        for symbol in dataset.symbols:
            if symbol not in self._funding_hist:
                self._funding_hist[symbol] = []
            rate = float(dataset.last_funding_rate_before(symbol, ts))
            # Append only if this is a new non-zero rate or list is small
            hist = self._funding_hist[symbol]
            if not hist or hist[-1] != rate:
                hist.append(rate)
                if len(hist) > 60:
                    hist.pop(0)

        # Need at least 24+169+1 = 194 bars to have features and a realized label
        if idx < 195:
            return

        # The feature point is idx-24 (features were observed there)
        feature_idx = idx - 24

        for symbol in dataset.symbols:
            closes = dataset.perp_close[symbol]
            if len(closes) <= idx:
                continue

            # Realized label: was next-24h return positive?
            # close[idx-1] vs close[feature_idx - 1]  (i.e. close[idx-25])
            c_future = closes[idx - 1]   # close at feature_idx + 23 (24h ahead)
            c_now = closes[feature_idx - 1] if feature_idx >= 1 else closes[0]
            if c_now == 0.0:
                continue
            label = 1.0 if c_future > c_now else -1.0

            # Build features at feature_idx (causal)
            feats = _build_features(
                dataset, symbol, feature_idx, self._funding_hist.get(symbol, [])
            )
            if feats is None:
                continue

            self._get_model(symbol).push(feats, label)

    def should_rebalance(self, dataset: MarketDataset, idx: int) -> bool:
        # Always update training data every bar
        self._update_models(dataset, idx)

        if idx < self._min_warmup:
            return False

        interval = int(self.params.get("rebalance_interval_bars") or 24)
        return idx % max(1, interval) == 0

    def target_weights(self, dataset: MarketDataset, idx: int, current: Weights) -> Weights:
        k = int(self.params.get("k_per_side") or 2)
        k = max(1, min(k, max(1, len(dataset.symbols) // 2)))
        risk_weighting = str(self.params.get("risk_weighting") or "inverse_vol")
        vol_lookback = int(self.params.get("vol_lookback_bars") or 168)
        target_gross = float(self.params.get("target_gross_leverage") or 0.3)

        # Score each symbol
        scores: Dict[str, float] = {}
        for symbol in dataset.symbols:
            model = self._get_model(symbol)
            if not model.is_ready:
                continue
            feats = _build_features(
                dataset, symbol, idx, self._funding_hist.get(symbol, [])
            )
            if feats is None:
                continue
            score = model.predict_score(feats)
            if not math.isnan(score) and not math.isinf(score):
                scores[symbol] = score

        # Need at least 2*k symbols with valid scores
        if len(scores) < 2 * k:
            return {s: 0.0 for s in dataset.symbols}

        ranked = sorted(scores.keys(), key=lambda s: scores[s], reverse=True)
        long_syms = ranked[:k]
        short_syms = ranked[-k:]

        # Avoid overlap (shouldn't happen but guard anyway)
        long_set = set(long_syms)
        short_set = set(short_syms)
        if long_set & short_set:
            return {s: 0.0 for s in dataset.symbols}

        # Inverse-vol weights
        inv_vol: Dict[str, float] = {}
        if risk_weighting == "inverse_vol":
            for s in set(long_syms + short_syms):
                closes = dataset.perp_close[s]
                vol = trailing_vol(closes, end_idx=idx, lookback_bars=vol_lookback)
                inv_vol[s] = (1.0 / vol) if vol > 0 else 1.0
        else:
            for s in set(long_syms + short_syms):
                inv_vol[s] = 1.0

        w = normalize_dollar_neutral(
            long_syms=long_syms,
            short_syms=short_syms,
            inv_vol=inv_vol,
            target_gross_leverage=target_gross,
        )
        out = {s: 0.0 for s in dataset.symbols}
        out.update(w)
        return out
