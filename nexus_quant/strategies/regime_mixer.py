"""
Regime Mixer — Multi-strategy allocator with adaptive performance blending.

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

Adaptive Performance Blending (NEW):
  Tracks rolling returns of each sub-strategy's recommended portfolio.
  When performance_blend_weight > 0, the final allocation is a weighted
  average of regime-based weights and performance-based weights.
  This solves the 2023 problem: in trending years where V1 outperforms,
  the adaptive blend automatically tilts toward V1 without changing
  regime detection thresholds.

Parameters:
  regime_lookback_bars    int   = 168    (7 days for BTC return)
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
  performance_blend_weight float = 0.0   (0=pure regime, 1=pure adaptive)
  performance_sensitivity  float = 20.0  (softmax temperature for perf blend)
  performance_min_observations int = 5   (min rebalances before adaptive kicks in)
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
    """Multi-strategy regime mixer with smooth blending and adaptive performance tracking."""

    def __init__(self, params: Dict[str, Any]) -> None:
        super().__init__(name="regime_mixer", params=params)

        # Sub-strategies
        v1_p = dict(params.get("v1_params") or {})
        lv_p = dict(params.get("lv_params") or {})
        self._v1 = NexusAlphaV1Strategy(v1_p)
        self._lv = LowVolAlphaStrategy(lv_p)

        # Regime detection params
        self._regime_lb = int(params.get("regime_lookback_bars") or 168)
        self._regime_lb_long = int(params.get("regime_lookback_long_bars") or 720)
        self._vol_lb = int(params.get("vol_lookback_bars") or 168)
        self._bull_thr = float(params.get("bull_threshold") or 0.02)
        self._bear_thr = float(params.get("bear_threshold") or 0.02)
        self._vol_spike = float(params.get("vol_spike_threshold") or 1.5)
        self._dual_horizon = bool(params.get("dual_horizon_regime", False))

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

        # Adaptive performance tracking (strategy momentum)
        self._perf_blend = float(params.get("performance_blend_weight") or 0.0)
        self._perf_sensitivity = float(params.get("performance_sensitivity") or 20.0)
        self._perf_min_obs = int(params.get("performance_min_observations") or 5)
        self._prev_v1_w: Optional[Weights] = None
        self._prev_lv_w: Optional[Weights] = None
        self._prev_rebal_idx: int = -1
        self._v1_rets: List[float] = []
        self._lv_rets: List[float] = []

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

    def _compute_btc_return(self, dataset: MarketDataset, idx: int, lookback: Optional[int] = None) -> Optional[float]:
        """BTC trailing return over given lookback (default: regime_lookback_bars)."""
        lb = lookback or self._regime_lb
        btc_sym = dataset.symbols[0]  # BTC expected first
        closes = dataset.perp_close[btc_sym]
        if idx < lb + 1 or idx >= len(closes):
            return None
        c_now = float(closes[min(idx - 1, len(closes) - 1)])
        c_back = float(closes[max(0, idx - 1 - lb)])
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
        """Multi-signal regime detection: bull / bear / sideways / unknown.

        When dual_horizon_regime=True, uses a long-term trend filter (720h)
        to prevent misclassifying dips in uptrends as 'bear'.
        Key insight: short-term bear + long-term bull = DIP → keep V1.
        """
        btc_ret = self._compute_btc_return(dataset, idx)
        vol_ratio = self._compute_vol_ratio(dataset, idx)

        if btc_ret is None:
            return "unknown"

        # Short-term classification
        if vol_ratio is not None and vol_ratio > self._vol_spike:
            short_regime = "bear"
        elif btc_ret > self._bull_thr:
            short_regime = "bull"
        elif btc_ret < -self._bear_thr:
            short_regime = "bear"
        else:
            short_regime = "sideways"

        if not self._dual_horizon:
            return short_regime

        # Dual-horizon: check long-term trend context
        btc_ret_long = self._compute_btc_return(dataset, idx, self._regime_lb_long)
        if btc_ret_long is None:
            return short_regime

        # If short-term says bear but long-term trend is bullish → it's a DIP
        # Reclassify as sideways (which gives V1 some allocation)
        if short_regime == "bear" and btc_ret_long > self._bull_thr:
            return "sideways"

        # If short-term says sideways but long-term is strongly bullish → treat as bull
        if short_regime == "sideways" and btc_ret_long > self._bull_thr * 2:
            return "bull"

        return short_regime

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

    def _compute_portfolio_return(
        self, dataset: MarketDataset, weights: Weights, start_idx: int, end_idx: int
    ) -> float:
        """Compute realized return of a portfolio held from start_idx to end_idx."""
        port_ret = 0.0
        for s, w in weights.items():
            if abs(w) < 1e-10:
                continue
            c = dataset.perp_close.get(s)
            if c is None:
                continue
            i0 = max(0, min(start_idx, len(c) - 1))
            i1 = max(0, min(end_idx - 1, len(c) - 1))
            c0 = float(c[i0])
            c1 = float(c[i1])
            if c0 > 0:
                port_ret += w * (c1 / c0 - 1.0)
        return port_ret

    def _adaptive_blend_weights(self) -> tuple:
        """Compute adaptive V1/LV weights from rolling performance history."""
        if len(self._v1_rets) < self._perf_min_obs:
            return self._smooth_v1, self._smooth_lv

        # Use last 30 observations (~30 rebalances = 30 days at 24h interval)
        n = min(30, len(self._v1_rets))
        v1_cum = sum(self._v1_rets[-n:])
        lv_cum = sum(self._lv_rets[-n:])

        # Softmax: tilt toward better-performing strategy
        sens = self._perf_sensitivity
        # Clamp to prevent overflow
        v1_exp = math.exp(min(50, max(-50, v1_cum * sens)))
        lv_exp = math.exp(min(50, max(-50, lv_cum * sens)))
        total = v1_exp + lv_exp
        adaptive_v1 = v1_exp / total
        adaptive_lv = lv_exp / total

        # Blend regime-based with adaptive
        blend = self._perf_blend
        final_v1 = (1 - blend) * self._smooth_v1 + blend * adaptive_v1
        final_lv = (1 - blend) * self._smooth_lv + blend * adaptive_lv

        # Normalize
        t = final_v1 + final_lv
        if t > 0:
            final_v1 /= t
            final_lv /= t

        return final_v1, final_lv

    def should_rebalance(self, dataset: MarketDataset, idx: int) -> bool:
        # Vol ratio needs 2x vol_lb; short-term regime needs regime_lb
        # Long-term regime only needs its lookback (no vol computation)
        warmup_short = max(self._regime_lb, self._vol_lb) * 2 + 10
        if self._dual_horizon:
            warmup = max(warmup_short, self._regime_lb_long + 10)
        else:
            warmup = warmup_short
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

        # Update smooth allocation weights from regime
        self._update_smooth_weights(regime)

        # Always get fresh signals — caching proven to degrade Sharpe ~50%
        # (V1's target_weights returns desired portfolio, not trade instructions;
        # fresh calls give current signal, not stale estimates)
        v1_w = self._v1.target_weights(dataset, idx, current)
        lv_w = self._lv.target_weights(dataset, idx, current)

        # Track performance of previous weights for adaptive blending
        if self._prev_v1_w is not None and self._prev_rebal_idx > 0:
            v1_r = self._compute_portfolio_return(
                dataset, self._prev_v1_w, self._prev_rebal_idx, idx
            )
            lv_r = self._compute_portfolio_return(
                dataset, self._prev_lv_w, self._prev_rebal_idx, idx
            )
            self._v1_rets.append(v1_r)
            self._lv_rets.append(lv_r)

        # Store for next rebalance's performance tracking
        self._prev_v1_w = dict(v1_w)
        self._prev_lv_w = dict(lv_w)
        self._prev_rebal_idx = idx

        # Determine final blend weights (regime + adaptive)
        if self._perf_blend > 0:
            final_v1, final_lv = self._adaptive_blend_weights()
        else:
            final_v1 = self._smooth_v1
            final_lv = self._smooth_lv

        # Blend sub-strategy portfolios
        out: Dict[str, float] = {}
        for s in dataset.symbols:
            w_v1 = float(v1_w.get(s, 0.0))
            w_lv = float(lv_w.get(s, 0.0))
            out[s] = final_v1 * w_v1 + final_lv * w_lv

        return out
