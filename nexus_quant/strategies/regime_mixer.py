"""
Regime Mixer — Multi-strategy allocator with per-regime on/off switches.

Economic Foundation:
  Different strategies exploit different market microstructure regimes:
  - NexusAlpha V1 (carry+mom+MR): profits in TRENDING/BULL markets
    ("buy dips in uptrends" — momentum filter + MR timing)
  - Low-Vol Alpha (defensive): profits in BEAR/SIDEWAYS markets
    (flight-to-quality, low-vol anomaly dominates)
  - These are regime-ANTI-CORRELATED (ideal for mixing)

Regime Detection:
  Uses BTC as market proxy with TWO signals:
  1. Momentum signal: BTC trailing return over regime_lookback_bars (168h default)
  2. Volatility signal: BTC realized vol vs longer-term vol (vol_ratio)

  Regime classification:
  - BULL:     BTC return > +bull_threshold AND vol_ratio < vol_spike_threshold
  - BEAR:     BTC return < -bear_threshold OR vol_ratio > vol_spike_threshold
  - SIDEWAYS: neither bull nor bear (low conviction environment)

  The regime feeds into a weight matrix that determines how much
  capital each sub-strategy receives. Transitions use exponential
  smoothing to avoid whipsaw trades.

Weight Matrix (default):
  | Regime   | V1     | Low-Vol | Notes                    |
  |----------|--------|---------|--------------------------|
  | BULL     | 1.00   | 0.00    | V1 dominates uptrends    |
  | BEAR     | 0.00   | 1.00    | Low-Vol = quality flight |
  | SIDEWAYS | 0.30   | 0.70    | Low-Vol + small V1 edge  |
  | UNKNOWN  | 0.50   | 0.50    | Equal (hedge ignorance)  |

Smooth Blending:
  Instead of hard regime switches, the mixer uses exponential
  smoothing on the regime allocation weights:
    w_t = alpha * w_target + (1 - alpha) * w_{t-1}
  where alpha = smoothing_alpha (default 0.15 = ~7-bar half-life).
  This prevents whipsaw from regime oscillation.

Parameters:
  regime_lookback_bars    int   = 336    (14 days for BTC return)
  vol_lookback_bars       int   = 168    (for vol_ratio computation)
  bull_threshold          float = 0.02   (BTC 168h return > +2% = bull)
  bear_threshold          float = 0.02   (BTC 168h return < -2% = bear)
  vol_spike_threshold     float = 1.5    (vol_ratio > 1.5 = volatile → bear)
  smoothing_alpha         float = 0.15   (EMA smoothing for transitions)
  rebalance_interval_bars int   = 24     (daily)
  target_gross_leverage   float = 0.35
  weight_matrix           dict           (per-regime weights per strategy)
  v1_params               dict           (NexusAlpha V1 params)
  lv_params               dict           (Low-Vol params)
"""
from __future__ import annotations

import math
import statistics
from typing import Any, Dict, List, Optional

from .nexus_alpha import NexusAlphaV1Strategy
from .low_vol_alpha import LowVolAlphaStrategy
from .base import Strategy, Weights
from ..data.schema import MarketDataset


# Default weight matrix: { regime: { strategy_key: weight } }
_DEFAULT_WEIGHTS = {
    "bull":     {"v1": 1.00, "lv": 0.00},
    "bear":     {"v1": 0.00, "lv": 1.00},
    "sideways": {"v1": 0.30, "lv": 0.70},
    "unknown":  {"v1": 0.50, "lv": 0.50},
}


class RegimeMixerStrategy(Strategy):
    """Multi-strategy regime mixer with smooth blending and per-regime on/off."""

    def __init__(self, params: Dict[str, Any]) -> None:
        super().__init__(name="regime_mixer", params=params)

        # Sub-strategies
        v1_p = dict(params.get("v1_params") or {})
        lv_p = dict(params.get("lv_params") or {})
        self._v1 = NexusAlphaV1Strategy(v1_p)
        self._lv = LowVolAlphaStrategy(lv_p)

        # Regime detection params
        self._regime_lb = int(params.get("regime_lookback_bars") or 336)
        self._vol_lb = int(params.get("vol_lookback_bars") or 168)
        self._bull_thr = float(params.get("bull_threshold") or 0.02)
        self._bear_thr = float(params.get("bear_threshold") or 0.02)
        self._vol_spike = float(params.get("vol_spike_threshold") or 1.5)

        # Smoothing
        self._alpha = float(params.get("smoothing_alpha") or 0.15)

        # Rebalancing
        self._rebal_interval = int(params.get("rebalance_interval_bars") or 24)

        # Weight matrix
        wm = params.get("weight_matrix")
        if wm and isinstance(wm, dict):
            self._wm = {}
            for regime in ["bull", "bear", "sideways", "unknown"]:
                r = wm.get(regime, _DEFAULT_WEIGHTS.get(regime, {}))
                self._wm[regime] = {
                    "v1": float(r.get("v1", 0.5)),
                    "lv": float(r.get("lv", 0.5)),
                }
        else:
            self._wm = {k: dict(v) for k, v in _DEFAULT_WEIGHTS.items()}

        # State for smooth blending
        self._smooth_v1: float = 0.5
        self._smooth_lv: float = 0.5
        self._last_regime: str = "unknown"
        self._regime_history: List[str] = []

    def _p(self, key: str, default: Any) -> Any:
        v = self.params.get(key)
        if v is None:
            return default
        try:
            if isinstance(default, int):
                return int(v)
            if isinstance(default, float):
                return float(v)
        except (TypeError, ValueError):
            pass
        return v

    def _compute_btc_return(self, dataset: MarketDataset, idx: int) -> Optional[float]:
        """BTC trailing return over regime_lookback_bars."""
        btc_sym = dataset.symbols[0]  # BTC expected first
        closes = dataset.perp_close[btc_sym]
        if idx < self._regime_lb + 1 or idx >= len(closes):
            return None
        c_now = float(closes[min(idx - 1, len(closes) - 1)])
        c_back = float(closes[max(0, idx - 1 - self._regime_lb)])
        if c_back <= 0:
            return None
        return (c_now / c_back) - 1.0

    def _compute_vol_ratio(self, dataset: MarketDataset, idx: int) -> Optional[float]:
        """BTC recent vol / longer-term vol."""
        btc_sym = dataset.symbols[0]
        closes = dataset.perp_close[btc_sym]
        lb = self._vol_lb

        if idx < lb * 2 + 1:
            return None

        # Recent returns
        recent_rs = []
        for i in range(max(1, idx - lb), idx):
            c0 = float(closes[i - 1])
            c1 = float(closes[i])
            if c0 > 0:
                recent_rs.append((c1 / c0) - 1.0)

        # Longer-term returns (2x lookback)
        longer_rs = []
        for i in range(max(1, idx - lb * 2), idx):
            c0 = float(closes[i - 1])
            c1 = float(closes[i])
            if c0 > 0:
                longer_rs.append((c1 / c0) - 1.0)

        if len(recent_rs) < 10 or len(longer_rs) < 10:
            return None

        recent_vol = statistics.pstdev(recent_rs) if len(recent_rs) > 1 else 0.0
        longer_vol = statistics.pstdev(longer_rs) if len(longer_rs) > 1 else 0.0

        if longer_vol <= 0:
            return None
        return recent_vol / longer_vol

    def _detect_regime(self, dataset: MarketDataset, idx: int) -> str:
        """Multi-signal regime detection: bull / bear / sideways / unknown."""
        btc_ret = self._compute_btc_return(dataset, idx)
        vol_ratio = self._compute_vol_ratio(dataset, idx)

        if btc_ret is None:
            return "unknown"

        # Vol spike overrides to bear (crash/stress)
        if vol_ratio is not None and vol_ratio > self._vol_spike:
            return "bear"

        # Momentum-based classification
        if btc_ret > self._bull_thr:
            return "bull"
        elif btc_ret < -self._bear_thr:
            return "bear"
        else:
            return "sideways"

    def _update_smooth_weights(self, regime: str) -> None:
        """Exponential smoothing of strategy weights to prevent whipsaw."""
        target = self._wm.get(regime, self._wm.get("unknown", {"v1": 0.5, "lv": 0.5}))
        target_v1 = target.get("v1", 0.5)
        target_lv = target.get("lv", 0.5)

        a = self._alpha
        self._smooth_v1 = a * target_v1 + (1 - a) * self._smooth_v1
        self._smooth_lv = a * target_lv + (1 - a) * self._smooth_lv

        # Normalize to sum = 1
        total = self._smooth_v1 + self._smooth_lv
        if total > 0:
            self._smooth_v1 /= total
            self._smooth_lv /= total

    def should_rebalance(self, dataset: MarketDataset, idx: int) -> bool:
        warmup = max(self._regime_lb, self._vol_lb) * 2 + 10
        if idx <= warmup:
            return False
        # Let sub-strategies update their internal state
        self._v1.should_rebalance(dataset, idx)
        self._lv.should_rebalance(dataset, idx)
        return idx % max(1, self._rebal_interval) == 0

    def target_weights(self, dataset: MarketDataset, idx: int, current: Weights) -> Weights:
        regime = self._detect_regime(dataset, idx)
        self._last_regime = regime
        self._regime_history.append(regime)

        # Update smooth allocation weights
        self._update_smooth_weights(regime)

        # Get fresh weights from each sub-strategy every mixer rebalance.
        # Even though V1 rebalances every 168h, getting fresh signal estimates
        # at every mixer rebalance (24h) allows the regime blend to respond
        # to intra-cycle changes.
        v1_w = self._v1.target_weights(dataset, idx, current)
        lv_w = self._lv.target_weights(dataset, idx, current)

        # Blend according to smoothed regime weights
        out: Dict[str, float] = {}
        for s in dataset.symbols:
            w_v1 = float(v1_w.get(s, 0.0))
            w_lv = float(lv_w.get(s, 0.0))
            out[s] = self._smooth_v1 * w_v1 + self._smooth_lv * w_lv

        return out
