"""
Strategy: Sector-Balanced Risk-Parity + Momentum Tilt + DD Deleveraging
========================================================================
Phase 145 Champion — sector-balanced RP on 14-div universe.

Architecture:
  1. Sector-balanced risk parity (equal risk per sector, inv-vol within)
  2. TSMOM 6m/12m momentum tilt (90/10 base/tilt)
  3. Cross-sectional momentum rank tilt (5%)
  4. Vol-targeting (8% portfolio vol)
  5. Drawdown deleveraging (tight: 5%→15% ramp)
  6. Vol regime ceiling

Best config (Phase 145, c50/f20/b30):
  FULL=+0.614  IS=+0.195  OOS1=+0.803  OOS2=+0.749
  MDD=38.5%  CAGR=7.7%  OOS_MIN=+0.749
  Universe: 8 commodities + 4 FX + 2 bonds = 14 instruments
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional

from nexus_quant.strategies.base import Strategy, Weights
from nexus_quant.data.schema import MarketDataset

_EPS = 1e-10

# Sector classification for the 14-div universe
SECTOR_MAP = {
    "CL=F": "commodity", "NG=F": "commodity", "GC=F": "commodity",
    "SI=F": "commodity", "HG=F": "commodity", "ZW=F": "commodity",
    "ZC=F": "commodity", "ZS=F": "commodity", "PL=F": "commodity",
    "KC=F": "commodity", "SB=F": "commodity", "CT=F": "commodity",
    "BZ=F": "commodity",
    "EURUSD=X": "fx", "GBPUSD=X": "fx", "AUDUSD=X": "fx", "JPY=X": "fx",
    "TLT": "bond", "IEF": "bond",
}


class RPMomDDStrategy(Strategy):
    """Sector-Balanced Risk-Parity + Momentum Tilt + DD Deleveraging."""

    def __init__(
        self,
        name: str = "rp_mom_dd",
        params: Optional[Dict[str, Any]] = None,
    ) -> None:
        p = params or {}
        super().__init__(name, p)

        # Sector risk budgets
        self.w_commodity: float = float(p.get("w_commodity", 0.50))
        self.w_fx: float = float(p.get("w_fx", 0.20))
        self.w_bond: float = float(p.get("w_bond", 0.30))

        # Momentum tilt blend
        self.base_pct: float = float(p.get("base_pct", 0.90))
        self.tilt_pct: float = float(p.get("tilt_pct", 0.10))
        self.signal_scale: float = float(p.get("signal_scale", 1.0))

        # Cross-sectional momentum
        self.xsect_weight: float = float(p.get("xsect_weight", 0.05))

        # Portfolio vol target
        self.vol_target: float = float(p.get("vol_target", 0.08))

        # Drawdown deleveraging
        self.dd_threshold: float = float(p.get("dd_threshold", 0.05))
        self.dd_max: float = float(p.get("dd_max", 0.15))
        self.dd_min_alloc: float = float(p.get("dd_min_alloc", 0.10))

        # Timing
        self.warmup: int = int(p.get("warmup", 270))
        self.rebalance_freq: int = int(p.get("rebalance_freq", 42))
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

        # 1. Sector-balanced inverse-vol weights
        sector_budgets = {
            "commodity": self.w_commodity,
            "fx": self.w_fx,
            "bond": self.w_bond,
        }
        port_vols: Dict[str, float] = {}
        rp_weights: Dict[str, float] = {}

        for sector, budget in sector_budgets.items():
            sector_syms = [s for s in syms if SECTOR_MAP.get(s, "commodity") == sector]
            if not sector_syms:
                continue
            inv_vols: Dict[str, float] = {}
            for sym in sector_syms:
                arr = rv.get(sym, [])
                v = arr[idx] if idx < len(arr) else 0.2
                if not (v == v) or v <= 0.01:
                    v = 0.2
                inv_vols[sym] = 1.0 / v
                port_vols[sym] = v
            total_iv = sum(inv_vols.values())
            if total_iv < _EPS:
                continue
            for sym in sector_syms:
                rp_weights[sym] = budget * (inv_vols[sym] / total_iv)

        if not rp_weights:
            return {}

        # 2. TSMOM momentum tilt (6m/12m blend)
        tsmom_signals: Dict[str, float] = {}
        for sym in syms:
            s126 = _safe_get(tsmom_126, sym, idx)
            s252 = _safe_get(tsmom_252, sym, idx)
            combo = (
                0.5 * max(-1.0, min(1.0, s126 / self.signal_scale))
                + 0.5 * max(-1.0, min(1.0, s252 / self.signal_scale))
            )
            tsmom_signals[sym] = combo

        # 3. Cross-sectional momentum rank tilt
        xsect_tilt: Dict[str, float] = {}
        if self.xsect_weight > _EPS:
            ranked = sorted(syms, key=lambda s: tsmom_signals.get(s, 0))
            n = len(ranked)
            for rank_i, sym in enumerate(ranked):
                xsect_tilt[sym] = 2.0 * rank_i / max(n - 1, 1) - 1.0

        # 4. Apply tilts to RP weights
        weights: Dict[str, float] = {}
        for sym in syms:
            ts_factor = self.base_pct + self.tilt_pct * tsmom_signals.get(sym, 0)
            xs_factor = self.xsect_weight * xsect_tilt.get(sym, 0)
            factor = max(0.0, ts_factor + xs_factor)
            weights[sym] = rp_weights.get(sym, 0) * factor

        # 5. Scale to vol target
        port_vol_sq = sum(
            (weights.get(s, 0) * port_vols.get(s, 0.2)) ** 2 for s in syms
        )
        port_vol = math.sqrt(port_vol_sq) if port_vol_sq > 0 else 0.01
        if port_vol > 0.01:
            scale = self.vol_target / port_vol
            weights = {s: w * min(scale, 3.0) for s, w in weights.items()}

        # 6. Drawdown deleveraging
        dd_factor = self._dd_factor(dataset, idx)
        if dd_factor < 1.0:
            weights = {s: w * dd_factor for s, w in weights.items()}

        # 7. Vol regime ceiling
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


def _safe_get(feature_dict: dict, sym: str, idx: int) -> float:
    arr = feature_dict.get(sym, [])
    v = arr[idx] if idx < len(arr) else 0
    return v if (v == v) else 0


# ── Champion config (14-div universe) ────────────────────────────────────────

CHAMPION_PARAMS = {
    "w_commodity": 0.50,
    "w_fx": 0.20,
    "w_bond": 0.30,
    "vol_target": 0.08,
    "base_pct": 0.90,
    "tilt_pct": 0.10,
    "signal_scale": 1.0,
    "xsect_weight": 0.05,
    "dd_threshold": 0.05,
    "dd_max": 0.15,
    "dd_min_alloc": 0.10,
    "warmup": 270,
    "rebalance_freq": 42,
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
