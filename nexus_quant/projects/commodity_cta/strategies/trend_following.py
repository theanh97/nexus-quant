"""
Strategy #1: Multi-Timeframe EMA Trend Following
=================================================
Uses EWMA crossovers as primary trend signal (more robust than raw returns).
Multiple timeframe signals (12/26 + 20/50 EMA) combined with 20d momentum.

Signal construction (per instrument):
  - s1 = sign(EMA_fast12 - EMA_slow26) : short-term trend
  - s2 = sign(EMA_fast20 - EMA_slow50) : medium-term trend
  - s3 = sign(mom_20d)                  : raw momentum confirmation
  combo = (1/3)*s1 + (1/3)*s2 + (1/3)*s3
  → if |combo| < 0.1: flat (signal disagreement)
  → direction = sign(combo)

Position sizing:
  - Vol-targeting: weight = direction * (vol_target / instrument_realized_vol)
  - Max gross leverage: 2.0x, max per-position: 25%

Rebalance: monthly (21 bars) — lower costs, signal is slow-moving

Benchmark context:
  - IS Sharpe ~0.34 (2007-2026, 13 commodities, 7bps RT cost)
  - Comparable to: SG CTA Index Sharpe ~0.5 (includes bonds+FX+equity futures)
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional

from nexus_quant.strategies.base import Strategy, Weights
from nexus_quant.data.schema import MarketDataset

_EPS = 1e-10


class TrendFollowingStrategy(Strategy):
    """
    Multi-timeframe EMA crossover + vol-targeting.
    Combines three signals: EMA(12,26), EMA(20,50), mom_20d.
    Monthly rebalancing to keep transaction costs manageable.
    """

    def __init__(self, name: str = "cta_trend", params: Optional[Dict[str, Any]] = None) -> None:
        p = params or {}
        super().__init__(name, p)

        # EMA spans for two signal timeframes
        self.fast1: int = int(p.get("fast1", 12))
        self.slow1: int = int(p.get("slow1", 26))
        self.fast2: int = int(p.get("fast2", 20))
        self.slow2: int = int(p.get("slow2", 50))

        # Signal weights (must sum to 1)
        self.w_ema1: float = float(p.get("w_ema1", 1.0 / 3.0))
        self.w_ema2: float = float(p.get("w_ema2", 1.0 / 3.0))
        self.w_mom: float = float(p.get("w_mom", 1.0 / 3.0))

        # Sizing
        self.max_gross_leverage: float = float(p.get("max_gross_leverage", 2.0))
        self.max_position: float = float(p.get("max_position", 0.25))
        # Per-instrument vol target → portfolio vol ~12-20% (depends on correlations)
        self.vol_target: float = float(p.get("vol_target", 0.12))

        # Warmup: need slow EMA history
        self.warmup: int = int(p.get("warmup", max(self.slow2 + 10, 60)))

        # Signal threshold: flat when |combo| < this
        self.signal_threshold: float = float(p.get("signal_threshold", 0.1))

        # Monthly rebalancing
        self.rebalance_freq: int = int(p.get("rebalance_freq", 21))
        self._last_rebalance: int = -1

        # EMA state (online computation)
        self._ema_state: Dict[str, Dict[str, float]] = {}
        self._prev_idx: int = -1

    # ── Rebalance logic ──────────────────────────────────────────────────────

    def should_rebalance(self, dataset: MarketDataset, idx: int) -> bool:
        # Update EMA state incrementally
        if idx > self._prev_idx:
            for i in range(self._prev_idx + 1, idx + 1):
                self._update_ema_state(dataset, i)
            self._prev_idx = idx

        if idx < self.warmup:
            return False
        if self._last_rebalance < 0 or (idx - self._last_rebalance) >= self.rebalance_freq:
            self._last_rebalance = idx
            return True
        return False

    # ── Weight computation ───────────────────────────────────────────────────

    def target_weights(
        self, dataset: MarketDataset, idx: int, current: Weights
    ) -> Weights:
        if idx < self.warmup:
            return {}

        symbols = dataset.symbols
        mom20 = dataset.features.get("mom_20d", {})
        rv20 = dataset.features.get("rv_20d", {})

        raw_weights: Dict[str, float] = {}

        for sym in symbols:
            state = self._ema_state.get(sym)
            if state is None or state.get("bars_seen", 0) < self.slow2:
                continue

            ema1_val = state.get("ema1", 0.0)
            ema2_val = state.get("ema2", 0.0)
            s1 = math.copysign(1.0, ema1_val) if abs(ema1_val) > 0.001 else 0.0
            s2 = math.copysign(1.0, ema2_val) if abs(ema2_val) > 0.001 else 0.0
            m20 = _get(mom20, sym, idx)
            s3 = math.copysign(1.0, m20) if (m20 is not None and abs(m20) > 0.01) else 0.0

            combo = self.w_ema1 * s1 + self.w_ema2 * s2 + self.w_mom * s3

            if abs(combo) < self.signal_threshold:
                continue

            direction = math.copysign(1.0, combo)

            vol = _get(rv20, sym, idx)
            if vol is None or vol < _EPS:
                vol = 0.20
            weight = direction * (self.vol_target / max(vol, _EPS))
            weight = math.copysign(min(abs(weight), self.max_position), weight)
            raw_weights[sym] = weight

        if not raw_weights:
            return {}

        return _normalise(raw_weights, self.max_gross_leverage, self.max_position)

    def _update_ema_state(self, dataset: MarketDataset, idx: int) -> None:
        """Incrementally update EMA for all symbols at bar idx."""
        k1f = 2.0 / (self.fast1 + 1)
        k1s = 2.0 / (self.slow1 + 1)
        k2f = 2.0 / (self.fast2 + 1)
        k2s = 2.0 / (self.slow2 + 1)

        for sym in dataset.symbols:
            price = dataset.close(sym, idx)
            if price != price or price <= 0:
                continue

            if sym not in self._ema_state:
                self._ema_state[sym] = {
                    "ef1": price, "es1": price,
                    "ef2": price, "es2": price,
                    "ema1": 0.0, "ema2": 0.0,
                    "bars_seen": 0,
                }

            state = self._ema_state[sym]
            state["ef1"] = state["ef1"] * (1 - k1f) + price * k1f
            state["es1"] = state["es1"] * (1 - k1s) + price * k1s
            state["ef2"] = state["ef2"] * (1 - k2f) + price * k2f
            state["es2"] = state["es2"] * (1 - k2s) + price * k2s
            state["bars_seen"] += 1

            if state["bars_seen"] >= self.slow2:
                state["ema1"] = (state["ef1"] - state["es1"]) / (state["es1"] + _EPS)
                state["ema2"] = (state["ef2"] - state["es2"]) / (state["es2"] + _EPS)


# ── Helpers ───────────────────────────────────────────────────────────────────


def _get(
    feat: Dict[str, List[float]],
    sym: str,
    idx: int,
) -> Optional[float]:
    arr = feat.get(sym)
    if arr is None or idx >= len(arr):
        return None
    v = arr[idx]
    return v if (v == v and math.isfinite(v)) else None


def _safe_price(dataset: MarketDataset, sym: str, idx: int) -> Optional[float]:
    prices = dataset.perp_close.get(sym)
    if prices is None or idx >= len(prices):
        return None
    v = prices[idx]
    return v if (v == v and v > 0) else None


def _normalise(
    weights: Dict[str, float],
    max_gross: float,
    max_pos: float,
) -> Dict[str, float]:
    gross = sum(abs(v) for v in weights.values())
    if gross < _EPS:
        return {}
    if gross > max_gross:
        scale = max_gross / gross
        out = {sym: w * scale for sym, w in weights.items()}
    else:
        out = dict(weights)
    for sym, w in out.items():
        if abs(w) > max_pos:
            out[sym] = math.copysign(max_pos, w)
    return out
