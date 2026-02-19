"""
Regime-Switching Ensemble — V1 (bull) + Low-Vol (bear/sideways).

Both GPT-5.2 and GLM-5 recommended regime-switching as the optimal
approach for combining pro-cyclical (V1) with defensive (Low-Vol) strategies.

Decision Rule:
  BTC 168h return > regime_threshold → BULL → use NexusAlpha V1 weights
  BTC 168h return <= regime_threshold → BEAR/SIDEWAYS → use Low-Vol weights

This avoids the "static blend" problem where Low-Vol kills V1 returns in
bull markets (2023: V1=1.07, LV=-1.46, 50/50=-0.20).

Expected result: positive Sharpe in ALL market regimes.

Parameters:
  regime_threshold     float = 0.0    (BTC 168h return threshold for regime)
  regime_lookback_bars int   = 168    (7 days)
  target_gross_leverage float = 0.35
  v1_params            dict           (NexusAlpha V1 params)
  lv_params            dict           (Low-Vol params)
"""
from __future__ import annotations

from typing import Any, Dict

from .nexus_alpha import NexusAlphaV1Strategy
from .low_vol_alpha import LowVolAlphaStrategy
from .base import Strategy, Weights
from ..data.schema import MarketDataset


class RegimeSwitchEnsembleStrategy(Strategy):
    """Regime-switching: V1 in bull, Low-Vol in bear/sideways."""

    def __init__(self, params: Dict[str, Any]) -> None:
        super().__init__(name="regime_switch_ensemble", params=params)
        v1_p = dict(params.get("v1_params") or {})
        lv_p = dict(params.get("lv_params") or {})
        self._v1 = NexusAlphaV1Strategy(v1_p)
        self._lv = LowVolAlphaStrategy(lv_p)
        self._regime_lb = int(params.get("regime_lookback_bars") or 168)
        self._regime_thr = float(params.get("regime_threshold") or 0.0)
        self._last_regime: str = "unknown"

    def _detect_regime(self, dataset: MarketDataset, idx: int) -> str:
        """Detect bull vs bear/sideways using BTC 168h return."""
        # Use first symbol as market proxy (BTC expected to be first)
        btc_sym = dataset.symbols[0]
        closes = dataset.perp_close[btc_sym]
        if idx < self._regime_lb + 1 or idx >= len(closes):
            return "unknown"
        c_now = float(closes[min(idx - 1, len(closes) - 1)])
        c_back = float(closes[max(0, idx - 1 - self._regime_lb)])
        if c_back <= 0:
            return "unknown"
        btc_ret = (c_now / c_back) - 1.0
        if btc_ret > self._regime_thr:
            return "bull"
        return "bear_sideways"

    def should_rebalance(self, dataset: MarketDataset, idx: int) -> bool:
        # Both sub-strategies need model updates, so call both
        v1_reb = self._v1.should_rebalance(dataset, idx)
        lv_reb = self._lv.should_rebalance(dataset, idx)
        # Rebalance if either wants to
        return v1_reb or lv_reb

    def target_weights(self, dataset: MarketDataset, idx: int, current: Weights) -> Weights:
        regime = self._detect_regime(dataset, idx)
        self._last_regime = regime

        if regime == "bull":
            return self._v1.target_weights(dataset, idx, current)
        elif regime == "bear_sideways":
            return self._lv.target_weights(dataset, idx, current)
        else:
            # Unknown regime (warmup period) — stay flat
            return {s: 0.0 for s in dataset.symbols}
