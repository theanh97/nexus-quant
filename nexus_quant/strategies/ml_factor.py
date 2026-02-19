from __future__ import annotations

"""
ML Factor Cross-Sectional Strategy V1.

Uses per-symbol rolling OLS to predict next-period returns from lagged factors.
Constructs a dollar-neutral portfolio: long top-k predicted, short bottom-k predicted.

Factors (all lagged, using idx-1 to avoid look-ahead bias):
  1. momentum_1w  : 168-bar return
  2. momentum_1d  : 24-bar return
  3. mean_rev_4h  : 4-bar return (inverted sign = mean-reversion signal)
  4. funding      : last funding rate before current timestamp
  5. basis        : perp/spot basis proxy (0 if spot not available)
  6. vol          : 72-bar trailing volatility
  7. intercept    : 1.0
"""

from typing import Any, Dict, Optional

from ._math import normalize_dollar_neutral, trailing_vol
from ._ols import RollingOLS
from .base import Strategy, Weights
from ..data.schema import MarketDataset

_N_FEATURES = 7
_FEAT_MOM_1W = 0
_FEAT_MOM_1D = 1
_FEAT_MR_4H = 2
_FEAT_FUNDING = 3
_FEAT_BASIS = 4
_FEAT_VOL = 5
_FEAT_INTERCEPT = 6


def _safe_ret(series: list, end_idx: int, lookback: int) -> float:
    if end_idx <= 0 or end_idx >= len(series):
        return 0.0
    c1 = float(series[end_idx])
    c0 = float(series[max(0, end_idx - lookback)])
    if c0 == 0:
        return 0.0
    return (c1 / c0) - 1.0


class MLFactorCrossSectionV1Strategy(Strategy):
    """
    Rolling OLS cross-sectional factor model.

    Params:
      k_per_side          : int   = 2   (long/short legs)
      ols_window          : int   = 504 (rolling OLS window)
      ols_refit_every     : int   = 24  (refit interval in bars)
      ols_min_samples     : int   = 60  (minimum samples before first prediction)
      min_warmup_bars     : int   = 600 (bars before any trading)
      risk_weighting      : str   = "equal" | "inverse_vol"
      vol_lookback_bars   : int   = 72
      target_gross_leverage: float = 1.0
      rebalance_interval_bars: int = 24
    """

    def __init__(self, params: Dict[str, Any]) -> None:
        super().__init__(name="ml_factor_xs_v1", params=params)
        ols_window = int(self.params.get("ols_window") or 504)
        ols_refit = int(self.params.get("ols_refit_every") or 24)
        ols_min = int(self.params.get("ols_min_samples") or 60)
        # Per-symbol models (created lazily on first call)
        self._models: Dict[str, RollingOLS] = {}
        self._ols_window = ols_window
        self._ols_refit = ols_refit
        self._ols_min = ols_min
        # Track last idx we updated models at (to avoid double-update)
        self._last_update_idx: int = -1

    def _get_model(self, symbol: str) -> RollingOLS:
        if symbol not in self._models:
            self._models[symbol] = RollingOLS(
                n_features=_N_FEATURES,
                window=self._ols_window,
                refit_every=self._ols_refit,
                min_samples=self._ols_min,
            )
        return self._models[symbol]

    def _build_features(self, dataset: MarketDataset, symbol: str, idx: int) -> Optional[list]:
        """Build feature vector at bar idx (using idx-1 for lag). Returns None if not enough data."""
        min_lookback = 168 + 2
        if idx < min_lookback:
            return None
        closes = dataset.perp_close[symbol]
        ts = dataset.timeline[idx]

        mom_1w = _safe_ret(closes, idx - 1, 168)
        mom_1d = _safe_ret(closes, idx - 1, 24)
        mr_4h = -_safe_ret(closes, idx - 1, 4)  # inverted = mean-reversion
        funding = float(dataset.last_funding_rate_before(symbol, ts))
        basis = float(dataset.basis(symbol, idx - 1)) if dataset.spot_close is not None else 0.0
        vol = trailing_vol(closes, end_idx=idx - 1, lookback_bars=int(self.params.get("vol_lookback_bars") or 72))

        return [mom_1w, mom_1d, mr_4h, funding, basis, vol, 1.0]

    def _update_models(self, dataset: MarketDataset, idx: int) -> None:
        """
        Feed realized return (bar idx-1 -> idx) as target for features at idx-1.
        Called once per target_weights invocation.
        """
        if idx <= 168 or idx == self._last_update_idx:
            return
        self._last_update_idx = idx

        for symbol in dataset.symbols:
            closes = dataset.perp_close[symbol]
            if idx >= len(closes) or idx < 2:
                continue
            # Realized return from bar idx-1 -> idx
            c1 = float(closes[idx])
            c0 = float(closes[idx - 1])
            if c0 == 0:
                continue
            realized_ret = (c1 / c0) - 1.0

            # Features at idx-1 (the predictor side)
            feats = self._build_features(dataset, symbol, idx - 1)
            if feats is None:
                continue
            self._get_model(symbol).update(feats, realized_ret)

    def should_rebalance(self, dataset: MarketDataset, idx: int) -> bool:
        min_warmup = int(self.params.get("min_warmup_bars") or 600)
        if idx < min_warmup:
            return False
        interval = int(self.params.get("rebalance_interval_bars") or 24)
        return idx % max(1, interval) == 0

    def target_weights(self, dataset: MarketDataset, idx: int, current: Weights) -> Weights:
        # Feed realized returns into models
        self._update_models(dataset, idx)

        k = int(self.params.get("k_per_side") or 2)
        k = max(1, min(k, max(1, len(dataset.symbols) // 2)))
        risk_weighting = str(self.params.get("risk_weighting") or "equal")
        vol_lookback = int(self.params.get("vol_lookback_bars") or 72)
        target_gross = float(self.params.get("target_gross_leverage") or 1.0)

        ts = dataset.timeline[idx]

        # Get predictions for each symbol
        predictions: Dict[str, float] = {}
        for symbol in dataset.symbols:
            model = self._get_model(symbol)
            if not model.is_ready:
                continue
            feats = self._build_features(dataset, symbol, idx)
            if feats is None:
                continue
            pred = model.predict(feats)
            if pred is not None and not (pred != pred):  # NaN check
                predictions[symbol] = pred

        # Need enough symbols with predictions
        if len(predictions) < 2 * k:
            return {s: 0.0 for s in dataset.symbols}

        ranked = sorted(predictions.keys(), key=lambda s: predictions[s], reverse=True)
        long_syms = ranked[-k:]   # predicted highest return -> long
        short_syms = ranked[:k]   # predicted lowest return -> short

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
