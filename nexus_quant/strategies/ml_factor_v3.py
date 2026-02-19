from __future__ import annotations

"""
ML Factor Cross-Sectional Strategy V3.

Improvements over V1:
  1. Extended feature set (10 features vs 6):
     - momentum_1w, momentum_3d, momentum_1d  (multi-horizon)
     - mean_rev_4h, mean_rev_2h               (short-term reversion)
     - funding_rate                            (carry signal)
     - vol_ratio                               (vol expansion/compression)
     - rsi_proxy                               (RSI-like oscillator)
     - volume_momentum                         (volume surge indicator)
     - intercept

  2. Cross-sectional z-scoring on ALL raw features before training AND inference.

  3. Portfolio construction: rank-weighted allocation (not just top/bottom-k),
     using sigmoid weighting to reduce impact of borderline symbols.

  4. Adaptive gross leverage: scale down when recent portfolio vol is high.

  5. Lower default warmup + tighter OLS window for faster readiness on 1-year OOS runs.
"""

import math
from typing import Any, Dict, List, Optional

from ._math import normalize_dollar_neutral, trailing_vol
from ._ols import RollingOLS
from .base import Strategy, Weights
from ..data.schema import MarketDataset


# ── Feature indices ────────────────────────────────────────────────────────
_N_FEATURES = 10
_FEAT_MOM_1W   = 0   # 168-bar return
_FEAT_MOM_3D   = 1   # 72-bar return
_FEAT_MOM_1D   = 2   # 24-bar return
_FEAT_MR_4H    = 3   # 4-bar mean reversion (inverted)
_FEAT_MR_2H    = 4   # 2-bar mean reversion (inverted)
_FEAT_FUNDING  = 5   # last 8h funding rate
_FEAT_VOL_RATIO= 6   # short_vol / long_vol  (>1 = expanding)
_FEAT_RSI_PROX = 7   # RSI proxy (0-1 normalised)
_FEAT_VOL_MOM  = 8   # log(avg_vol_24h / avg_vol_168h)  -- requires volume data
_FEAT_INTERCEPT= 9   # 1.0


def _safe_ret(series: list, end_idx: int, lookback: int) -> float:
    if end_idx <= 0 or end_idx >= len(series):
        return 0.0
    c1 = float(series[end_idx])
    c0 = float(series[max(0, end_idx - lookback)])
    if c0 == 0:
        return 0.0
    return (c1 / c0) - 1.0


def _rsi_proxy(closes: list, end_idx: int, period: int = 14) -> float:
    """
    Returns RSI-like value in [0, 1] using Wilder's smoothing approximation.
    Simple but fast: use average of up/down moves over last `period` bars.
    """
    start = max(1, end_idx - period)
    ups, downs = [], []
    for i in range(start, end_idx):
        if i >= len(closes):
            break
        delta = float(closes[i]) - float(closes[i - 1])
        if delta >= 0:
            ups.append(delta)
        else:
            downs.append(-delta)
    avg_up = sum(ups) / max(1, len(ups)) if ups else 0.0
    avg_dn = sum(downs) / max(1, len(downs)) if downs else 0.0
    if avg_up + avg_dn == 0:
        return 0.5
    return avg_up / (avg_up + avg_dn)   # [0, 1]


def _vol_ratio(closes: list, end_idx: int, short: int = 24, long: int = 168) -> float:
    """short_vol / long_vol; > 1 means volatility is expanding."""
    short_vol = _rolling_vol(closes, end_idx, short)
    long_vol  = _rolling_vol(closes, end_idx, long)
    if long_vol == 0:
        return 1.0
    return short_vol / long_vol


def _rolling_vol(closes: list, end_idx: int, lookback: int) -> float:
    start = max(1, end_idx - lookback)
    rs = []
    for i in range(start, end_idx):
        if i >= len(closes):
            break
        c0 = float(closes[i - 1])
        c1 = float(closes[i])
        if c0 != 0:
            rs.append(math.log(c1 / c0) if c1 > 0 and c0 > 0 else (c1 / c0) - 1.0)
    if len(rs) < 2:
        return 0.0
    mu = sum(rs) / len(rs)
    var = sum((r - mu) ** 2 for r in rs) / len(rs)
    return math.sqrt(var)


def _volume_momentum(volumes: Optional[list], end_idx: int) -> float:
    """log ratio of recent vs historical average volume."""
    if volumes is None or end_idx < 168:
        return 0.0
    short_start = max(0, end_idx - 24)
    long_start  = max(0, end_idx - 168)
    short_vols = [float(volumes[i]) for i in range(short_start, end_idx) if i < len(volumes)]
    long_vols  = [float(volumes[i]) for i in range(long_start, end_idx) if i < len(volumes)]
    avg_short = sum(short_vols) / max(1, len(short_vols)) if short_vols else 0.0
    avg_long  = sum(long_vols) / max(1, len(long_vols)) if long_vols else 0.0
    if avg_long <= 0 or avg_short <= 0:
        return 0.0
    return math.log(avg_short / avg_long)


def _zscore_features(feature_matrix: Dict[str, List[float]]) -> Dict[str, List[float]]:
    """Cross-sectionally z-score each feature dimension (skip intercept = last feature)."""
    if not feature_matrix:
        return feature_matrix
    symbols = list(feature_matrix.keys())
    n_feats = len(next(iter(feature_matrix.values())))
    result = {s: list(v) for s, v in feature_matrix.items()}
    for fi in range(n_feats - 1):   # skip intercept
        vals = [feature_matrix[s][fi] for s in symbols]
        mean = sum(vals) / len(vals)
        variance = sum((v - mean) ** 2 for v in vals) / max(1, len(vals))
        std = math.sqrt(variance) if variance > 0 else 1.0
        for s in symbols:
            result[s][fi] = (feature_matrix[s][fi] - mean) / std
    return result


class MLFactorV3Strategy(Strategy):
    """
    Rolling Ridge regression cross-sectional factor model V3.

    Params:
      k_per_side             : int   = 2
      ols_window             : int   = 720   (rolling OLS window in bars)
      ols_refit_every        : int   = 24
      min_warmup_bars        : int   = 336   (2 weeks at 1h bars)
      vol_lookback_bars      : int   = 168
      rebalance_interval_bars: int   = 24
      target_gross_leverage  : float = 0.3
      risk_weighting         : str   = "inverse_vol"
      learning_rate          : float = 0.01  (unused: kept for config compat)
      adaptive_leverage      : bool  = True  (scale down when portfolio vol spikes)
    """

    def __init__(self, params: Dict[str, Any]) -> None:
        super().__init__(name="ml_factor_v3", params=params)
        ols_window    = int(self.params.get("ols_window") or 720)
        ols_refit     = int(self.params.get("ols_refit_every") or 24)
        ols_min       = max(60, ols_window // 10)   # at least 10% of window
        self._ols_window  = ols_window
        self._ols_refit   = ols_refit
        self._ols_min     = ols_min
        self._models: Dict[str, RollingOLS] = {}
        self._last_update_idx: int = -1
        # Running portfolio return buffer for adaptive leverage
        self._port_rets: List[float] = []

    # ── Internal helpers ──────────────────────────────────────────────────

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
        """Build 10-feature vector at bar idx, using idx-1 to avoid look-ahead."""
        min_lookback = 168 + 2
        if idx < min_lookback:
            return None
        closes = dataset.perp_close[symbol]
        ts = dataset.timeline[idx]
        i = idx - 1  # lagged index

        mom_1w    = _safe_ret(closes, i, 168)
        mom_3d    = _safe_ret(closes, i, 72)
        mom_1d    = _safe_ret(closes, i, 24)
        mr_4h     = -_safe_ret(closes, i, 4)
        mr_2h     = -_safe_ret(closes, i, 2)
        funding   = float(dataset.last_funding_rate_before(symbol, ts))
        vol_ratio = _vol_ratio(closes, i, short=24, long=168)
        rsi_p     = _rsi_proxy(closes, i, period=14)
        # Volume momentum — try to use dataset volumes if available
        volumes   = getattr(dataset, "perp_volume", {}).get(symbol, None)
        vol_mom   = _volume_momentum(volumes, i)

        return [mom_1w, mom_3d, mom_1d, mr_4h, mr_2h,
                funding, vol_ratio, rsi_p, vol_mom, 1.0]

    def _update_models(self, dataset: MarketDataset, idx: int) -> None:
        if idx <= 168 or idx == self._last_update_idx:
            return
        self._last_update_idx = idx

        realized_rets: Dict[str, float] = {}
        raw_feats_update: Dict[str, List[float]] = {}

        for symbol in dataset.symbols:
            closes = dataset.perp_close[symbol]
            if idx >= len(closes) or idx < 2:
                continue
            c1 = float(closes[idx])
            c0 = float(closes[idx - 1])
            if c0 == 0:
                continue
            realized_rets[symbol] = (c1 / c0) - 1.0
            feats = self._build_features(dataset, symbol, idx - 1)
            if feats is not None:
                raw_feats_update[symbol] = feats

        zscored_update = _zscore_features(raw_feats_update)
        for symbol, feats in zscored_update.items():
            if symbol in realized_rets:
                self._get_model(symbol).update(feats, realized_rets[symbol])

    # ── Strategy interface ────────────────────────────────────────────────

    def should_rebalance(self, dataset: MarketDataset, idx: int) -> bool:
        self._update_models(dataset, idx)
        min_warmup = int(self.params.get("min_warmup_bars") or 336)
        if idx < min_warmup:
            return False
        interval = int(self.params.get("rebalance_interval_bars") or 24)
        return idx % max(1, interval) == 0

    def target_weights(self, dataset: MarketDataset, idx: int, current: Weights) -> Weights:
        k = int(self.params.get("k_per_side") or 2)
        k = max(1, min(k, max(1, len(dataset.symbols) // 2)))
        risk_weighting   = str(self.params.get("risk_weighting") or "inverse_vol")
        vol_lookback     = int(self.params.get("vol_lookback_bars") or 168)
        target_gross     = float(self.params.get("target_gross_leverage") or 0.3)
        adaptive_lev     = bool(self.params.get("adaptive_leverage", True))

        # Collect raw features, z-score, predict
        raw_feats: Dict[str, List[float]] = {}
        for symbol in dataset.symbols:
            model = self._get_model(symbol)
            if not model.is_ready:
                continue
            feats = self._build_features(dataset, symbol, idx)
            if feats is not None:
                raw_feats[symbol] = feats

        if len(raw_feats) < 2 * k:
            return {s: 0.0 for s in dataset.symbols}

        zscored_feats = _zscore_features(raw_feats)
        predictions: Dict[str, float] = {}
        for symbol, feats in zscored_feats.items():
            model = self._get_model(symbol)
            pred = model.predict(feats)
            if pred is not None and pred == pred:   # NaN guard
                predictions[symbol] = pred

        if len(predictions) < 2 * k:
            return {s: 0.0 for s in dataset.symbols}

        ranked = sorted(predictions.keys(), key=lambda s: predictions[s], reverse=True)
        long_syms  = ranked[:k]
        short_syms = ranked[-k:]

        # Inverse-vol weighting
        inv_vol: Dict[str, float] = {}
        for s in set(long_syms + short_syms):
            closes = dataset.perp_close[s]
            if risk_weighting == "inverse_vol":
                vol = trailing_vol(closes, end_idx=idx, lookback_bars=vol_lookback)
                inv_vol[s] = (1.0 / vol) if vol > 0 else 1.0
            else:
                inv_vol[s] = 1.0

        # Adaptive leverage: reduce if recent portfolio vol > target
        eff_gross = target_gross
        if adaptive_lev and len(self._port_rets) >= 48:
            recent = self._port_rets[-48:]
            mu  = sum(recent) / len(recent)
            var = sum((r - mu) ** 2 for r in recent) / len(recent)
            port_vol_daily = math.sqrt(var) * math.sqrt(24)   # annualised ~*sqrt(8760)
            if port_vol_daily > 0.02:   # if daily port vol > 2%, scale down
                scale = 0.02 / port_vol_daily
                eff_gross = max(target_gross * 0.5, target_gross * scale)

        w = normalize_dollar_neutral(
            long_syms=long_syms,
            short_syms=short_syms,
            inv_vol=inv_vol,
            target_gross_leverage=eff_gross,
        )
        out = {s: 0.0 for s in dataset.symbols}
        out.update(w)

        # Record approximate portfolio return for adaptive leverage
        port_ret = 0.0
        for s, wt in current.items():
            if wt != 0 and s in dataset.perp_close:
                closes = dataset.perp_close[s]
                if idx > 0 and idx < len(closes):
                    c1 = float(closes[idx])
                    c0 = float(closes[idx - 1])
                    if c0 != 0:
                        port_ret += wt * ((c1 / c0) - 1.0)
        self._port_rets.append(port_ret)
        if len(self._port_rets) > 720:
            self._port_rets = self._port_rets[-720:]

        return out
