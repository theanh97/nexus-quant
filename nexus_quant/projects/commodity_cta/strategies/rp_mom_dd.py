"""
Strategy: Risk-Parity + Momentum Tilt + Drawdown Deleveraging
==============================================================
Phase 142 Champion — Best OOS1 Sharpe 0.824 on 14-div universe.

Architecture:
  1. Risk-Parity base weights (inverse-vol)
  2. TSMOM 6m/12m momentum tilt (80/20 base/tilt blend)
  3. Vol-targeting to desired portfolio volatility
  4. Drawdown deleveraging (linear ramp-down when DD > threshold)
  5. Vol regime ceiling (reduce exposure in high-vol regimes)

Best config (14-div/RP+Mom(12%)):
  FULL=+0.482  IS=+0.234  OOS1=+0.824  OOS2=+0.468
  MDD=35.5%  CAGR=6.0%
  Universe: 8 commodities + 4 FX + 2 bonds = 14 instruments
"""
from __future__ import annotations

import math
from typing import Any, Dict, Optional

from nexus_quant.strategies.base import Strategy, Weights
from nexus_quant.data.schema import MarketDataset

_EPS = 1e-10


class RPMomDDStrategy(Strategy):
    """Risk-Parity + Momentum Tilt + Drawdown Deleveraging."""

    def __init__(
        self,
        name: str = "rp_mom_dd",
        params: Optional[Dict[str, Any]] = None,
    ) -> None:
        p = params or {}
        super().__init__(name, p)

        # Momentum tilt blend
        self.base_pct: float = float(p.get("base_pct", 0.80))
        self.tilt_pct: float = float(p.get("tilt_pct", 0.20))
        self.signal_scale: float = float(p.get("signal_scale", 2.0))

        # Portfolio vol target
        self.vol_target: float = float(p.get("vol_target", 0.12))

        # Drawdown deleveraging
        self.dd_threshold: float = float(p.get("dd_threshold", 0.10))
        self.dd_max: float = float(p.get("dd_max", 0.25))
        self.dd_min_alloc: float = float(p.get("dd_min_alloc", 0.20))

        # Timing
        self.warmup: int = int(p.get("warmup", 270))
        self.rebalance_freq: int = int(p.get("rebalance_freq", 21))
        self._last_reb: int = -1

    def should_rebalance(self, dataset: MarketDataset, idx: int) -> bool:
        if idx < self.warmup:
            return False
        if self._last_reb < 0 or (idx - self._last_reb) >= self.rebalance_freq:
            self._last_reb = idx
            return True
        return False

    def target_weights(
        self, dataset: MarketDataset, idx: int, current: Weights
    ) -> Weights:
        if idx < self.warmup:
            return {}

        syms = dataset.symbols
        rv = dataset.features.get("rv_20d", {})
        tsmom_126 = dataset.features.get("tsmom_126d", {})
        tsmom_252 = dataset.features.get("tsmom_252d", {})
        vol_regime = dataset.features.get("vol_regime", {})

        # 1. Risk-parity weights (inverse-vol)
        inv_vols: Dict[str, float] = {}
        port_vols: Dict[str, float] = {}
        for sym in syms:
            arr = rv.get(sym, [])
            v = arr[idx] if idx < len(arr) else 0.2
            if not (v == v) or v <= 0.01:
                v = 0.2
            inv_vols[sym] = 1.0 / v
            port_vols[sym] = v
        total_iv = sum(inv_vols.values())
        if total_iv < _EPS:
            return {}
        rp_weights = {s: inv_vols[s] / total_iv for s in syms}

        # 2. Momentum tilt (6m/12m TSMOM blend)
        weights: Dict[str, float] = {}
        for sym in syms:
            arr126 = tsmom_126.get(sym, [])
            arr252 = tsmom_252.get(sym, [])
            s126 = arr126[idx] if idx < len(arr126) else 0
            s252 = arr252[idx] if idx < len(arr252) else 0
            if not (s126 == s126):
                s126 = 0
            if not (s252 == s252):
                s252 = 0
            combo = (
                0.5 * max(-1, min(1, s126 / self.signal_scale))
                + 0.5 * max(-1, min(1, s252 / self.signal_scale))
            )
            factor = max(0.0, self.base_pct + self.tilt_pct * combo)
            weights[sym] = rp_weights[sym] * factor

        # 3. Scale to vol target
        port_vol_sq = sum(
            (weights.get(s, 0) * port_vols.get(s, 0.2)) ** 2 for s in syms
        )
        port_vol = math.sqrt(port_vol_sq) if port_vol_sq > 0 else 0.01
        if port_vol > 0.01:
            scale = self.vol_target / port_vol
            weights = {s: w * min(scale, 3.0) for s, w in weights.items()}

        # 4. Drawdown deleveraging
        dd_factor = self._dd_factor(dataset, idx)
        if dd_factor < 1.0:
            weights = {s: w * dd_factor for s, w in weights.items()}

        # 5. Vol regime ceiling
        avg_vr = 0.0
        for sym in syms:
            arr = vol_regime.get(sym, [])
            if idx < len(arr):
                v = arr[idx]
                if v == v:
                    avg_vr += v
        avg_vr /= len(syms) if syms else 1
        if avg_vr > 0.8:
            weights = {
                s: w * max(0.5, 1.0 - (avg_vr - 0.8))
                for s, w in weights.items()
            }

        return {s: w for s, w in weights.items() if w > _EPS}

    def _dd_factor(self, dataset: MarketDataset, idx: int) -> float:
        """Compute drawdown deleveraging factor from recent cross-instrument returns."""
        lookback = min(60, idx)
        if lookback < 5:
            return 1.0

        cum_ret = 0.0
        peak_ret = 0.0
        for i in range(idx - lookback, idx + 1):
            bar_ret = 0.0
            cnt = 0
            for sym in dataset.symbols:
                p_prev = dataset.close(sym, max(0, i - 1))
                p_curr = dataset.close(sym, i)
                if p_prev > 0 and p_curr > 0:
                    bar_ret += (p_curr / p_prev - 1)
                    cnt += 1
            if cnt > 0:
                cum_ret += bar_ret / cnt
            if cum_ret > peak_ret:
                peak_ret = cum_ret

        dd = peak_ret - cum_ret
        if dd <= self.dd_threshold:
            return 1.0
        elif dd >= self.dd_max:
            return self.dd_min_alloc
        else:
            progress = (dd - self.dd_threshold) / (self.dd_max - self.dd_threshold)
            return 1.0 - progress * (1.0 - self.dd_min_alloc)


# ── Champion config (14-div universe) ────────────────────────────────────────

CHAMPION_PARAMS = {
    "vol_target": 0.12,
    "base_pct": 0.80,
    "tilt_pct": 0.20,
    "signal_scale": 2.0,
    "dd_threshold": 0.10,
    "dd_max": 0.25,
    "dd_min_alloc": 0.20,
    "warmup": 270,
    "rebalance_freq": 21,
}

CHAMPION_UNIVERSE = [
    # Commodities (8)
    "CL=F", "NG=F", "GC=F", "SI=F", "HG=F", "ZW=F", "ZC=F", "ZS=F",
    # FX (4)
    "EURUSD=X", "GBPUSD=X", "AUDUSD=X", "JPY=X",
    # Bonds (2)
    "TLT", "IEF",
]


__all__ = ["RPMomDDStrategy", "CHAMPION_PARAMS", "CHAMPION_UNIVERSE"]
