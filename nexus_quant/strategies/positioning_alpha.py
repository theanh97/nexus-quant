"""
NEXUS Positioning Alpha V1 -- Futures Positioning / Sentiment Strategy.

Alpha Source: Positioning Signals (real or proxy)
-------------------------------------------------
When real Binance positioning data is available (recent ~30 days):
  - Uses openInterestHist, globalLongShortAccountRatio, topLongShortPositionRatio.

When real data is NOT available (historical backtests beyond ~30 days):
  - Computes PROXY signals from kline data that approximate the same economic signals:
    1. OI momentum proxy: volume-weighted price impact (high volume + directional move
       implies OI is increasing; volume spike alone implies position changes).
    2. Contrarian L/S proxy: taker buy ratio = taker_buy_vol / total_vol.
       High ratio = retail aggressively buying = contrarian short signal.
    3. Smart money divergence proxy: large-bar vs small-bar volume asymmetry.
       When big volume bars are bullish while small bars are mixed, smart money
       is accumulating.

Economic mechanism:
  - Contrarian retail sentiment: crowd is usually wrong at extremes.
  - OI expansion signal: new money entering market indicates conviction.
  - Smart money divergence: top traders position differently from retail.

Strategy:
  1. Compute OI momentum (or proxy) per symbol.
  2. Compute contrarian L/S z-score (or proxy) over rolling window.
  3. Compute smart money divergence (or proxy).
  4. Weighted composite score.
  5. Cross-sectional ranking: LONG top-k, SHORT bottom-k.
  6. Dollar-neutral, inverse-vol weighted, with configurable leverage.

Parameters
----------
oi_lookback_bars            : int   = 48     OI momentum lookback
ls_lookback_bars            : int   = 168    L/S z-score window
w_oi_momentum               : float = 0.3    weight for OI momentum signal
w_contrarian_ls             : float = 0.5    weight for contrarian L/S signal
w_smart_money               : float = 0.2    weight for smart money divergence
k_per_side                  : int   = 2      positions per side
target_gross_leverage       : float = 0.35   total gross leverage
vol_lookback_bars           : int   = 168    for inverse-vol weighting
rebalance_interval_bars     : int   = 24     rebalance frequency (every 24h)
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional

from ._math import normalize_dollar_neutral, trailing_vol, zscores
from .base import Strategy, Weights
from ..data.schema import MarketDataset


class PositioningAlphaV1Strategy(Strategy):
    """
    Positioning Alpha V1 -- Futures positioning / sentiment signals.

    Uses real Binance positioning data when available, falls back to
    kline-derived proxy signals for historical backtests.

    Ranks symbols by a composite score of OI momentum, contrarian retail
    sentiment, and smart money divergence.  Cross-sectional long/short,
    inverse-vol weighted, dollar-neutral.
    """

    def __init__(self, params: Dict[str, Any]) -> None:
        super().__init__(name="positioning_alpha_v1", params=params)
        self._logged_proxy_mode = False

    def _p(self, key: str, default: Any) -> Any:
        """Get parameter with type-coerced default."""
        v = self.params.get(key)
        if v is None:
            return default
        try:
            if isinstance(default, bool):
                return bool(v)
            if isinstance(default, int):
                return int(v)
            if isinstance(default, float):
                return float(v)
        except (TypeError, ValueError):
            pass
        return v

    def _has_real_positioning(self, dataset: MarketDataset) -> bool:
        """Check if the dataset has real positioning data (not all zeros/defaults)."""
        if dataset.open_interest is None and dataset.long_short_ratio_global is None:
            return False
        # Check if any symbol has non-trivial OI data
        if dataset.open_interest is not None:
            for sym in dataset.symbols:
                oi = dataset.open_interest.get(sym, [])
                # If there's any non-zero value, we have real data
                for v in oi:
                    if v > 0.0:
                        return True
        return False

    # ==================================================================
    # REAL SIGNALS (from Binance positioning data)
    # ==================================================================

    def _oi_momentum_real(
        self, dataset: MarketDataset, symbol: str, idx: int, lookback: int,
    ) -> Optional[float]:
        """OI momentum from real open interest data."""
        if dataset.open_interest is None:
            return None
        oi = dataset.open_interest.get(symbol)
        if oi is None:
            return None
        curr_idx = idx - 1
        past_idx = curr_idx - lookback
        if curr_idx < 0 or past_idx < 0 or curr_idx >= len(oi) or past_idx >= len(oi):
            return None
        oi_now = oi[curr_idx]
        oi_past = oi[past_idx]
        if oi_past <= 0.0 or oi_now <= 0.0:
            return None
        return (oi_now - oi_past) / oi_past

    def _contrarian_ls_real(
        self, dataset: MarketDataset, symbol: str, idx: int, lookback: int,
    ) -> Optional[float]:
        """Contrarian z-score from real global L/S ratio."""
        if dataset.long_short_ratio_global is None:
            return None
        ls = dataset.long_short_ratio_global.get(symbol)
        if ls is None:
            return None
        end = idx
        start = max(0, end - lookback)
        if start >= end or end > len(ls):
            return None
        window = ls[start:end]
        if len(window) < 10:
            return None
        unique_vals = set()
        for v in window:
            unique_vals.add(round(v, 6))
            if len(unique_vals) > 1:
                break
        if len(unique_vals) <= 1:
            return None
        mu = sum(window) / len(window)
        variance = sum((v - mu) ** 2 for v in window) / len(window)
        sd = math.sqrt(variance) if variance > 0 else 0.0
        if sd < 1e-10:
            return None
        z = (window[-1] - mu) / sd
        return -z  # contrarian

    def _smart_money_real(
        self, dataset: MarketDataset, symbol: str, idx: int,
    ) -> Optional[float]:
        """Smart money divergence from real top/global L/S ratios."""
        if dataset.long_short_ratio_top is None or dataset.long_short_ratio_global is None:
            return None
        top_ls = dataset.long_short_ratio_top.get(symbol)
        global_ls = dataset.long_short_ratio_global.get(symbol)
        if top_ls is None or global_ls is None:
            return None
        i = idx - 1
        if i < 0 or i >= len(top_ls) or i >= len(global_ls):
            return None
        top_val = top_ls[i]
        global_val = global_ls[i]
        if top_val == 1.0 and global_val == 1.0:
            return None
        return top_val - global_val

    # ==================================================================
    # PROXY SIGNALS (computed from kline data for historical backtests)
    # ==================================================================

    def _oi_momentum_proxy(
        self, dataset: MarketDataset, symbol: str, idx: int, lookback: int,
    ) -> float:
        """
        OI momentum proxy from volume data.

        Logic: Sum of (volume * |return|) over recent window vs previous window.
        When volume-weighted absolute returns are increasing, new positions are
        being opened (OI is expanding). The sign comes from volume-weighted
        signed returns -- positive means net long positioning is increasing.
        """
        if dataset.perp_volume is None:
            return 0.0

        vol = dataset.perp_volume.get(symbol)
        closes = dataset.perp_close.get(symbol)
        if vol is None or closes is None:
            return 0.0

        # Recent half vs older half within the lookback window
        half = lookback // 2
        recent_start = max(1, idx - half)
        older_start = max(1, recent_start - half)

        if older_start >= recent_start or recent_start >= idx:
            return 0.0

        # Compute volume-weighted signed returns for each half
        def _vw_signed_ret(start_i: int, end_i: int) -> float:
            total_vol = 0.0
            weighted_ret = 0.0
            for i in range(start_i, min(end_i, len(closes))):
                if i < 1 or i >= len(vol):
                    continue
                c0 = closes[i - 1]
                c1 = closes[i]
                if c0 <= 0:
                    continue
                ret = (c1 / c0) - 1.0
                v = vol[i]
                weighted_ret += v * ret
                total_vol += v
            if total_vol <= 0:
                return 0.0
            return weighted_ret / total_vol

        recent_vwr = _vw_signed_ret(recent_start, idx)
        older_vwr = _vw_signed_ret(older_start, recent_start)

        # OI momentum = change in volume-weighted positioning
        return recent_vwr - older_vwr

    def _contrarian_ls_proxy(
        self, dataset: MarketDataset, symbol: str, idx: int, lookback: int,
    ) -> float:
        """
        Contrarian L/S proxy from taker buy ratio.

        Logic: taker_buy_ratio = taker_buy_vol / total_vol.
        This approximates the retail L/S ratio -- when retail is aggressively
        buying (ratio > 0.5), they are net long. We compute z-score of this
        ratio over the lookback window and negate it (contrarian).
        """
        if dataset.taker_buy_volume is None or dataset.perp_volume is None:
            return 0.0

        tbv = dataset.taker_buy_volume.get(symbol)
        vol = dataset.perp_volume.get(symbol)
        if tbv is None or vol is None:
            return 0.0

        end = idx
        start = max(0, end - lookback)
        if start >= end:
            return 0.0

        # Compute taker buy ratio per bar, then z-score the current value
        ratios: List[float] = []
        for i in range(start, min(end, len(vol))):
            if i >= len(tbv):
                continue
            total_v = vol[i]
            taker_buy_v = tbv[i]
            if total_v <= 0.0:
                continue
            ratio = taker_buy_v / total_v
            # Clamp to [0, 1]
            ratio = max(0.0, min(1.0, ratio))
            ratios.append(ratio)

        if len(ratios) < 10:
            return 0.0

        mu = sum(ratios) / len(ratios)
        variance = sum((r - mu) ** 2 for r in ratios) / len(ratios)
        sd = math.sqrt(variance) if variance > 0 else 0.0

        if sd < 1e-10:
            return 0.0

        current_ratio = ratios[-1]
        z = (current_ratio - mu) / sd

        # Contrarian: negate (high taker buy ratio = crowd is long = go short)
        return -z

    def _smart_money_proxy(
        self, dataset: MarketDataset, symbol: str, idx: int, lookback: int,
    ) -> float:
        """
        Smart money divergence proxy from volume profile.

        Logic: Separate bars into high-volume (top 25%) and low-volume (bottom 75%).
        High-volume bars are more likely driven by institutional/smart money.
        Compute the average return of high-vol bars vs low-vol bars.
        If high-vol bars are bullish while low-vol bars are bearish (or vice versa),
        there is a divergence -- follow the smart money (high-vol) direction.
        """
        if dataset.perp_volume is None:
            return 0.0

        vol = dataset.perp_volume.get(symbol)
        closes = dataset.perp_close.get(symbol)
        if vol is None or closes is None:
            return 0.0

        start = max(1, idx - lookback)
        if start >= idx:
            return 0.0

        # Collect (volume, return) pairs
        bar_data: List[tuple] = []
        for i in range(start, min(idx, len(closes))):
            if i < 1 or i >= len(vol):
                continue
            c0 = closes[i - 1]
            c1 = closes[i]
            if c0 <= 0:
                continue
            ret = (c1 / c0) - 1.0
            v = vol[i]
            bar_data.append((v, ret))

        if len(bar_data) < 10:
            return 0.0

        # Sort by volume descending
        bar_data.sort(key=lambda x: x[0], reverse=True)

        # Top 25% = high volume (smart money), bottom 75% = low volume (retail)
        cutoff = max(1, len(bar_data) // 4)
        high_vol_bars = bar_data[:cutoff]
        low_vol_bars = bar_data[cutoff:]

        if not high_vol_bars or not low_vol_bars:
            return 0.0

        avg_ret_high = sum(r for _, r in high_vol_bars) / len(high_vol_bars)
        avg_ret_low = sum(r for _, r in low_vol_bars) / len(low_vol_bars)

        # Smart money divergence: high-vol direction minus low-vol direction
        return avg_ret_high - avg_ret_low

    # ==================================================================
    # Unified signal computation (auto-selects real vs proxy)
    # ==================================================================

    def _compute_signals(
        self, dataset: MarketDataset, symbol: str, idx: int,
        oi_lb: int, ls_lb: int,
    ) -> tuple:
        """
        Returns (oi_signal, contra_signal, smart_signal) for a symbol.
        Uses real positioning data if available, otherwise proxy signals.
        """
        use_real = self._has_real_positioning(dataset)

        if use_real:
            oi_val = self._oi_momentum_real(dataset, symbol, idx, oi_lb)
            contra_val = self._contrarian_ls_real(dataset, symbol, idx, ls_lb)
            smart_val = self._smart_money_real(dataset, symbol, idx)
            # Fall back to 0.0 for any missing signals
            return (
                oi_val if oi_val is not None else 0.0,
                contra_val if contra_val is not None else 0.0,
                smart_val if smart_val is not None else 0.0,
            )
        else:
            if not self._logged_proxy_mode:
                print("[PositioningAlphaV1] No real positioning data -- using kline proxy signals")
                self._logged_proxy_mode = True
            return (
                self._oi_momentum_proxy(dataset, symbol, idx, oi_lb),
                self._contrarian_ls_proxy(dataset, symbol, idx, ls_lb),
                self._smart_money_proxy(dataset, symbol, idx, ls_lb),
            )

    # ------------------------------------------------------------------
    # Rebalance timing
    # ------------------------------------------------------------------

    def should_rebalance(self, dataset: MarketDataset, idx: int) -> bool:
        oi_lb = self._p("oi_lookback_bars", 48)
        ls_lb = self._p("ls_lookback_bars", 168)
        interval = self._p("rebalance_interval_bars", 24)
        warmup = max(oi_lb, ls_lb) + 10  # extra buffer for z-score stability
        if idx <= warmup:
            return False
        return idx % max(1, interval) == 0

    # ------------------------------------------------------------------
    # Target weights
    # ------------------------------------------------------------------

    def target_weights(
        self,
        dataset: MarketDataset,
        idx: int,
        current: Weights,
    ) -> Weights:
        syms = dataset.symbols
        k = max(1, min(self._p("k_per_side", 2), len(syms) // 2))

        oi_lb       = self._p("oi_lookback_bars", 48)
        ls_lb       = self._p("ls_lookback_bars", 168)
        w_oi        = self._p("w_oi_momentum", 0.3)
        w_contra    = self._p("w_contrarian_ls", 0.5)
        w_smart     = self._p("w_smart_money", 0.2)
        vol_lb      = self._p("vol_lookback_bars", 168)
        target_gross = self._p("target_gross_leverage", 0.35)

        # ---- Compute raw signals per symbol ---------------------------------
        oi_raw: Dict[str, float] = {}
        contra_raw: Dict[str, float] = {}
        smart_raw: Dict[str, float] = {}

        for s in syms:
            oi_val, contra_val, smart_val = self._compute_signals(
                dataset, s, idx, oi_lb, ls_lb,
            )
            oi_raw[s] = oi_val
            contra_raw[s] = contra_val
            smart_raw[s] = smart_val

        # ---- Cross-sectional z-scores for each signal -----------------------
        oi_z = zscores(oi_raw)
        contra_z = zscores(contra_raw)
        smart_z = zscores(smart_raw)

        # ---- Composite score ------------------------------------------------
        score: Dict[str, float] = {}
        for s in syms:
            score[s] = (
                w_oi * oi_z.get(s, 0.0) +
                w_contra * contra_z.get(s, 0.0) +
                w_smart * smart_z.get(s, 0.0)
            )

        # ---- Cross-sectional ranking ----------------------------------------
        ranked = sorted(syms, key=lambda s: score[s], reverse=True)

        long_syms = ranked[:k]
        short_syms = ranked[-k:]

        # Avoid overlap (when len(syms) < 2*k)
        long_syms = [s for s in long_syms if s not in short_syms]
        short_syms = [s for s in short_syms if s not in long_syms]

        if not long_syms or not short_syms:
            return {s: 0.0 for s in syms}

        # ---- Portfolio construction (inverse-vol weighted) ------------------
        all_active = list(set(long_syms) | set(short_syms))
        inv_vol: Dict[str, float] = {}
        for s in all_active:
            v = trailing_vol(dataset.perp_close[s], end_idx=idx, lookback_bars=vol_lb)
            inv_vol[s] = (1.0 / v) if v > 0 else 1.0

        w = normalize_dollar_neutral(long_syms, short_syms, inv_vol, target_gross)

        out = {s: 0.0 for s in syms}
        out.update(w)
        return out
