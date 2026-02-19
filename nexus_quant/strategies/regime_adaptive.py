"""
NexusAlphaV1 with Regime-Adaptive Overlay.

Detects market regime (BULL / BEAR / SIDEWAYS / HIGH_VOL) in a rolling,
lookahead-free manner using three signals:

  (a) 200-bar SMA trend          -- price above/below long-run average
  (b) Realised vol ratio         -- 30-bar vol vs 90-bar average
  (c) Funding rate sign/magnitude -- positive/negative/large funding

Based on detected regime the strategy scales target_gross_leverage by a
multiplier, reducing exposure during drawdowns.

Also provides NexusAlphaV1VolScaledStrategy: a faster, continuous vol-scaling
approach that avoids regime-detection lag by computing a rolling realized-vol
leverage multiplier each bar.

stdlib only -- no numpy/pandas/sklearn.
"""
from __future__ import annotations

import math
import statistics
from typing import Any, Dict, List, Optional

from ._math import normalize_dollar_neutral, trailing_vol, zscores
from .base import Strategy, Weights
from ..data.schema import MarketDataset

# --------------------------------------------------------------------------- #
# Regime labels
# --------------------------------------------------------------------------- #

BULL     = "BULL"
BEAR     = "BEAR"
SIDEWAYS = "SIDEWAYS"
HIGH_VOL = "HIGH_VOL"


# --------------------------------------------------------------------------- #
# RegimeDetector
# --------------------------------------------------------------------------- #

class RegimeDetector:
    """
    Detects the current market regime from a rolling window of bar dicts.

    Three signals (all O(N), no lookahead):
      (a) 200-bar SMA trend  -- price relative to long-run moving average
      (b) Vol ratio          -- 30-bar realised vol / 90-bar realised vol
      (c) Funding sentiment  -- average sign/magnitude of recent funding rates

    Priority order for regime assignment:
      1. HIGH_VOL  -- vol_ratio > vol_ratio_thresh
      2. BEAR      -- below SMA AND negative funding
      3. BULL      -- above SMA AND vol_ratio <= thresh
      4. SIDEWAYS  -- everything else

    Parameters
    ----------
    sma_lookback     : int   bars for long-run SMA            (default 200)
    vol_short        : int   bars for short-window realised vol (default 30)
    vol_long         : int   bars for long-window realised vol  (default 90)
    vol_ratio_thresh : float HIGH_VOL threshold                 (default 1.5)
    """

    def __init__(
        self,
        sma_lookback: int = 200,
        vol_short: int = 30,
        vol_long: int = 90,
        vol_ratio_thresh: float = 1.5,
    ) -> None:
        self.sma_lookback     = sma_lookback
        self.vol_short        = vol_short
        self.vol_long         = vol_long
        self.vol_ratio_thresh = vol_ratio_thresh

    # -- helpers ------------------------------------------------------------- #

    @staticmethod
    def _sma(prices: List[float], end_idx: int, n: int) -> float:
        """Rolling SMA up to (but not including) end_idx -- O(n), no lookahead."""
        start = max(0, end_idx - n)
        window = prices[start:end_idx]
        if not window:
            return float("nan")
        return sum(window) / len(window)

    @staticmethod
    def _realised_vol(prices: List[float], end_idx: int, n: int) -> float:
        """Annualised realised vol over last n log-returns (assumes 8760 bars/year = 1h)."""
        start = max(1, end_idx - n)
        log_rets: List[float] = []
        for i in range(start, end_idx):
            if prices[i - 1] > 0 and prices[i] > 0:
                log_rets.append(math.log(prices[i] / prices[i - 1]))
        if len(log_rets) < 5:
            return 0.0
        return statistics.pstdev(log_rets) * math.sqrt(8760.0)

    # -- main interface ------------------------------------------------------ #

    def detect(
        self,
        bars: List[Dict],
        funding_rates: Optional[List[float]] = None,
    ) -> str:
        """
        Detect regime from a list of bar dicts (no lookahead).

        Parameters
        ----------
        bars          : list of dicts with 'close' key (most-recent last).
        funding_rates : optional list of recent funding rates (float).
                        Positive = longs pay shorts (bearish).

        Returns
        -------
        str -- one of "BULL", "BEAR", "SIDEWAYS", "HIGH_VOL"
        """
        if not bars:
            return SIDEWAYS

        prices  = [float(b.get("close", b.get("c", 0.0))) for b in bars]
        n       = len(prices)
        end_idx = n  # exclusive index -- prices[:end_idx] is all history, no lookahead

        # (a) SMA trend
        sma           = self._sma(prices, end_idx, self.sma_lookback)
        current_price = prices[-1]
        above_sma     = (not math.isnan(sma)) and (current_price > sma)

        # (b) Vol ratio
        vol_short = self._realised_vol(prices, end_idx, self.vol_short)
        vol_long  = self._realised_vol(prices, end_idx, self.vol_long)
        vol_ratio = (vol_short / vol_long) if vol_long > 0 else 1.0
        high_vol  = vol_ratio > self.vol_ratio_thresh

        # (c) Funding sentiment
        bearish_funding = False
        if funding_rates and len(funding_rates) >= 1:
            avg_funding     = sum(funding_rates) / len(funding_rates)
            bearish_funding = avg_funding < -0.0001  # < -0.01% per settlement

        # -- Priority cascade ------------------------------------------------ #
        if high_vol:
            return HIGH_VOL

        if not above_sma and bearish_funding:
            return BEAR

        if not above_sma and vol_ratio < 0.8:
            # below SMA but low vol -- sideways/accumulation
            return SIDEWAYS

        if above_sma:
            return BULL

        return SIDEWAYS  # default


# --------------------------------------------------------------------------- #
# Private helpers (mirror from nexus_alpha.py for self-containment)
# --------------------------------------------------------------------------- #

def _funding_trend(
    dataset: "MarketDataset",
    symbol: str,
    ts: int,
    fast_n: int = 3,
    slow_n: int = 10,
) -> float:
    """mean(last fast_n funding settlements) - mean(last slow_n settlements)."""
    fund_dict = dataset.funding.get(symbol)
    if not fund_dict:
        return 0.0
    all_times = sorted(t for t in fund_dict if t < ts)
    if len(all_times) < slow_n:
        return 0.0
    recent   = all_times[-slow_n:]
    rates    = [fund_dict[t] for t in recent]
    fast_avg = sum(rates[-fast_n:]) / fast_n
    slow_avg = sum(rates) / len(rates)
    return fast_avg - slow_avg


def _last_funding_rates(
    dataset: "MarketDataset",
    symbol: str,
    ts: int,
    n: int = 10,
) -> List[float]:
    """Return up to n most-recent funding rates before ts."""
    fund_dict = dataset.funding.get(symbol)
    if not fund_dict:
        return []
    all_times = sorted(t for t in fund_dict if t < ts)
    if not all_times:
        return []
    recent = all_times[-n:]
    return [fund_dict[t] for t in recent]


def _ex_ante_portfolio_vol(
    weights: Dict[str, float],
    closes: Dict[str, List[float]],
    end_idx: int,
    lookback: int,
    bars_per_year: float = 8760.0,
) -> float:
    port_rets: List[float] = []
    for bar in range(max(1, end_idx - lookback), end_idx):
        r = 0.0
        for sym, w in weights.items():
            if abs(w) < 1e-12:
                continue
            c = closes.get(sym)
            if c is None or bar >= len(c) or bar < 1:
                continue
            c0, c1 = c[bar - 1], c[bar]
            if c0 > 0:
                r += w * ((c1 / c0) - 1.0)
        port_rets.append(r)
    if len(port_rets) < 10:
        return 0.0
    return statistics.pstdev(port_rets) * math.sqrt(bars_per_year)


def _apply_vol_target(
    weights: Dict[str, float],
    closes: Dict[str, List[float]],
    end_idx: int,
    lookback: int,
    target_annual_vol: float,
    min_gross: float,
    max_gross: float,
) -> Dict[str, float]:
    if target_annual_vol <= 0 or not weights:
        return weights
    current_vol = _ex_ante_portfolio_vol(weights, closes, end_idx, lookback)
    if current_vol <= 1e-8:
        return weights
    scale     = target_annual_vol / current_vol
    gross     = sum(abs(v) for v in weights.values())
    new_gross = max(min_gross, min(max_gross, gross * scale))
    if gross < 1e-10:
        return weights
    return {s: w * (new_gross / gross) for s, w in weights.items()}


# --------------------------------------------------------------------------- #
# NexusAlphaV1RegimeStrategy
# --------------------------------------------------------------------------- #

class NexusAlphaV1RegimeStrategy(Strategy):
    """
    Regime-Adaptive wrapper around NexusAlphaV1.

    Copies exact signal generation from NexusAlphaV1Strategy and applies a
    RegimeDetector to scale target_gross_leverage:

      BULL     -> bull_leverage_mult     (default 1.0)
      SIDEWAYS -> average(bull, high_vol) (default 0.75)
      HIGH_VOL -> high_vol_leverage_mult (default 0.5)
      BEAR     -> bear_leverage_mult     (default 0.3)

    Additional params vs nexus_alpha_v1
    ------------------------------------
    regime_lookback_bars   : int   = 200
    bear_leverage_mult     : float = 0.3
    high_vol_leverage_mult : float = 0.5
    bull_leverage_mult     : float = 1.0
    use_regime_filter      : bool  = True
    """

    def __init__(self, params: Dict[str, Any]) -> None:
        super().__init__(name="nexus_alpha_v1_regime", params=params)
        self._detector = RegimeDetector(
            sma_lookback     = int(params.get("regime_lookback_bars", 200)),
            vol_short        = 30,
            vol_long         = 90,
            vol_ratio_thresh = 1.5,
        )

    # -- param helper -------------------------------------------------------- #

    def _p(self, key: str, default: Any) -> Any:
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

    # -- rebalance gate ------------------------------------------------------ #

    def should_rebalance(self, dataset: "MarketDataset", idx: int) -> bool:
        mom_lb    = self._p("momentum_lookback_bars", 168)
        vol_lb    = self._p("vol_lookback_bars", 168)
        interval  = self._p("rebalance_interval_bars", 168)
        regime_lb = self._p("regime_lookback_bars", 200)
        warmup    = max(mom_lb, vol_lb, regime_lb) + 2
        if idx <= warmup:
            return False
        return idx % max(1, interval) == 0

    # -- regime detection ---------------------------------------------------- #

    def _detect_regime(self, dataset: "MarketDataset", idx: int) -> str:
        """
        Aggregate regime across all symbols via majority vote.
        BEAR/HIGH_VOL win ties over SIDEWAYS/BULL (conservative bias).
        """
        regime_lb = self._p("regime_lookback_bars", 200)
        ts        = dataset.timeline[idx]
        votes: Dict[str, int] = {BULL: 0, BEAR: 0, SIDEWAYS: 0, HIGH_VOL: 0}

        for sym in dataset.symbols:
            closes    = dataset.perp_close.get(sym, [])
            start     = max(0, idx - regime_lb)
            bar_closes = closes[start:idx]          # no lookahead
            bars      = [{"close": c} for c in bar_closes]
            fund_rates = _last_funding_rates(dataset, sym, ts, n=10)
            regime    = self._detector.detect(bars, fund_rates)
            votes[regime] += 1

        max_votes = max(votes.values())
        # Conservative priority: prefer risk-reducing regimes on ties
        for r in (BEAR, HIGH_VOL, SIDEWAYS, BULL):
            if votes[r] == max_votes:
                return r
        return SIDEWAYS

    # -- leverage multiplier ------------------------------------------------- #

    def _regime_mult(self, regime: str) -> float:
        if not self._p("use_regime_filter", True):
            return 1.0
        if regime == BEAR:
            return self._p("bear_leverage_mult",     0.3)
        if regime == HIGH_VOL:
            return self._p("high_vol_leverage_mult", 0.5)
        if regime == BULL:
            return self._p("bull_leverage_mult",     1.0)
        # SIDEWAYS -- blend
        bull_m = self._p("bull_leverage_mult",     1.0)
        high_m = self._p("high_vol_leverage_mult", 0.5)
        return (bull_m + high_m) / 2.0

    # -- target_weights (main entry point) ----------------------------------- #

    def target_weights(
        self,
        dataset: "MarketDataset",
        idx: int,
        current: Weights,
    ) -> Weights:
        syms = dataset.symbols
        k    = max(1, min(self._p("k_per_side", 2), len(syms) // 2))

        mom_lb  = self._p("momentum_lookback_bars",    168)
        mr_lb   = self._p("mean_reversion_lookback_bars", 48)
        vol_lb  = self._p("vol_lookback_bars",         168)
        ft_fast = self._p("funding_trend_fast",          3)
        ft_slow = self._p("funding_trend_slow",         10)

        target_gross = self._p("target_gross_leverage", 0.35)
        risk_w       = self._p("risk_weighting",        "inverse_vol")
        target_vol   = self._p("target_portfolio_vol",   0.0)
        min_lev      = self._p("min_gross_leverage",    0.05)
        max_lev      = self._p("max_gross_leverage",    0.65)
        strict       = self._p("strict_agreement",      False)

        w_carry = self._p("w_carry",          0.35)
        w_mom   = self._p("w_mom",            0.35)
        w_conf  = self._p("w_confirm",        0.20)
        w_mr    = self._p("w_mean_reversion", 0.05)
        w_vm    = self._p("w_vol_momentum",   0.05)
        w_ft    = self._p("w_funding_trend",  0.00)

        ts = dataset.timeline[idx]

        # -- Signal 1: Funding carry ----------------------------------------- #
        f_raw = {s: float(dataset.last_funding_rate_before(s, ts)) for s in syms}
        fz    = zscores(f_raw)
        carry = {s: -float(fz.get(s, 0.0)) for s in syms}

        # -- Signal 2: Price momentum ---------------------------------------- #
        mom_raw: Dict[str, float] = {}
        for s in syms:
            c  = dataset.perp_close[s]
            i1 = min(idx - 1, len(c) - 1)
            i0 = max(0, idx - 1 - mom_lb)
            c1 = c[i1] if i1 >= 0 else 0.0
            c0 = c[i0] if i0 >= 0 and i0 < len(c) else 0.0
            mom_raw[s] = (c1 / c0 - 1.0) if c0 > 0 else 0.0
        mz  = zscores(mom_raw)
        mom = {s: float(mz.get(s, 0.0)) for s in syms}

        # -- Signal 3: Agreement (carry x momentum) -------------------------- #
        agreement = {s: carry[s] * mom[s] for s in syms}

        # -- Signal 4: Mean reversion (short-term contrarian) ---------------- #
        mr_raw: Dict[str, float] = {}
        for s in syms:
            c  = dataset.perp_close[s]
            i1 = min(idx - 1, len(c) - 1)
            i0 = max(0, idx - 1 - mr_lb)
            c1 = c[i1] if i1 >= 0 else 0.0
            c0 = c[i0] if i0 >= 0 and i0 < len(c) else 0.0
            mr_raw[s] = (c1 / c0 - 1.0) if c0 > 0 else 0.0
        mr_z = zscores(mr_raw)
        mr   = {s: -float(mr_z.get(s, 0.0)) for s in syms}

        # -- Signal 5: Vol-adjusted momentum --------------------------------- #
        vol_mom_raw: Dict[str, float] = {}
        for s in syms:
            v = trailing_vol(
                dataset.perp_close[s],
                end_idx=max(0, idx - 1),
                lookback_bars=vol_lb,
            )
            vol_mom_raw[s] = (mom_raw[s] / v) if v > 0 else mom_raw[s]
        vm_z    = zscores(vol_mom_raw)
        vol_mom = {s: float(vm_z.get(s, 0.0)) for s in syms}

        # -- Signal 6: Funding trend ----------------------------------------- #
        ft_raw = {s: _funding_trend(dataset, s, ts, ft_fast, ft_slow) for s in syms}
        ft_z   = zscores(ft_raw)
        ft     = {s: float(ft_z.get(s, 0.0)) for s in syms}

        # -- Composite score ------------------------------------------------- #
        score: Dict[str, float] = {}
        for s in syms:
            score[s] = (
                w_carry * carry[s]
                + w_mom  * mom[s]
                + w_conf * agreement[s]
                + w_mr   * mr[s]
                + w_vm   * vol_mom[s]
                + w_ft   * ft[s]
            )

        ranked      = sorted(syms, key=lambda s: score[s], reverse=True)
        long_cands  = ranked[:k]
        short_cands = ranked[-k:]

        if strict:
            long_cands  = [s for s in long_cands  if agreement[s] >= 0.0]
            short_cands = [s for s in short_cands if agreement[s] >= 0.0]

        long_syms  = [s for s in long_cands  if s not in short_cands]
        short_syms = [s for s in short_cands if s not in long_cands]

        if not long_syms or not short_syms:
            return {s: 0.0 for s in syms}

        # -- Regime detection + leverage scaling ----------------------------- #
        regime       = self._detect_regime(dataset, idx)
        regime_mult  = self._regime_mult(regime)
        effective_gross = max(min_lev, min(max_lev, target_gross * regime_mult))

        # -- Portfolio construction (inverse-vol, dollar-neutral) ------------ #
        all_active = list(set(long_syms) | set(short_syms))
        inv_vol: Dict[str, float] = {}
        if risk_w == "inverse_vol":
            for s in all_active:
                v = trailing_vol(dataset.perp_close[s], end_idx=idx, lookback_bars=vol_lb)
                inv_vol[s] = (1.0 / v) if v > 0 else 1.0
        else:
            inv_vol = {s: 1.0 for s in all_active}

        w = normalize_dollar_neutral(long_syms, short_syms, inv_vol, effective_gross)

        # -- Optional portfolio vol targeting -------------------------------- #
        if target_vol > 0:
            w = _apply_vol_target(
                weights           = w,
                closes            = dataset.perp_close,
                end_idx           = idx,
                lookback          = vol_lb,
                target_annual_vol = target_vol,
                min_gross         = min_lev,
                max_gross         = max_lev,
            )

        out = {s: 0.0 for s in syms}
        out.update(w)
        return out


# --------------------------------------------------------------------------- #
# NexusAlphaV1VolScaledStrategy
# --------------------------------------------------------------------------- #

class NexusAlphaV1VolScaledStrategy(Strategy):
    """
    Continuous Volatility-Scaled wrapper around NexusAlphaV1.

    Replaces binary regime detection with fast, continuous vol-scaling:

      1. Compute realized_vol_short = stdev of last vol_short_bars hourly returns
         (annualised: * sqrt(8760))
      2. Compute realized_vol_long  = stdev of last vol_long_bars hourly returns
         (annualised)
      3. vol_ratio = realized_vol_short / realized_vol_long  (regime signal, not
         used to gate, only informational)
      4. leverage_scale = vol_target / realized_vol_short
         -- clipped to [vol_scale_min, vol_scale_max]
         -- This sizes positions so expected portfolio vol = vol_target
      5. effective_leverage = target_gross_leverage * leverage_scale

    This is FASTER (48h short window vs 200-bar SMA) and CONTINUOUS (no
    binary flip), avoiding the whipsaw / lag that hurt the regime strategy.

    Parameters (in addition to all nexus_alpha_v1 params)
    -------------------------------------------------------
    vol_target      : float = 0.10   target annual portfolio vol (10%)
    vol_short_bars  : int   = 48     short vol window (hours)
    vol_long_bars   : int   = 336    long vol window (14 days)
    vol_scale_min   : float = 0.3    minimum leverage scale
    vol_scale_max   : float = 2.0    maximum leverage scale
    use_vol_scaling : bool  = True   set False to run as plain nexus_alpha_v1
    """

    def __init__(self, params: Dict[str, Any]) -> None:
        super().__init__(name="nexus_alpha_v1_vol_scaled", params=params)

    # -- param helper -------------------------------------------------------- #

    def _p(self, key: str, default: Any) -> Any:
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

    # -- rebalance gate ------------------------------------------------------ #

    def should_rebalance(self, dataset: "MarketDataset", idx: int) -> bool:
        mom_lb   = self._p("momentum_lookback_bars", 168)
        vol_lb   = self._p("vol_lookback_bars",      168)
        vol_long = self._p("vol_long_bars",          336)
        interval = self._p("rebalance_interval_bars", 168)
        warmup   = max(mom_lb, vol_lb, vol_long) + 2
        if idx <= warmup:
            return False
        return idx % max(1, interval) == 0

    # -- continuous vol scaling ---------------------------------------------- #

    def _vol_leverage_scale(
        self,
        dataset: "MarketDataset",
        idx: int,
    ) -> float:
        """
        Compute the continuous leverage scale factor based on realized vol.

        Returns a float in [vol_scale_min, vol_scale_max].
        """
        use_scaling  = self._p("use_vol_scaling",  True)
        if not use_scaling:
            return 1.0

        vol_target      = self._p("vol_target",      0.10)
        vol_short_bars  = self._p("vol_short_bars",   48)
        vol_long_bars   = self._p("vol_long_bars",   336)
        vol_scale_min   = self._p("vol_scale_min",   0.3)
        vol_scale_max   = self._p("vol_scale_max",   2.0)

        # Use cross-sectional average of per-symbol realized vols
        # (aggregate measure of market vol level; more robust than single symbol)
        short_vols: List[float] = []
        for sym in dataset.symbols:
            closes = dataset.perp_close.get(sym, [])
            # Compute log-returns over vol_short_bars window
            start = max(1, idx - vol_short_bars)
            log_rets: List[float] = []
            for i in range(start, idx):
                if i < len(closes) and closes[i - 1] > 0 and closes[i] > 0:
                    log_rets.append(math.log(closes[i] / closes[i - 1]))
            if len(log_rets) >= 5:
                short_vols.append(statistics.pstdev(log_rets) * math.sqrt(8760.0))

        if not short_vols:
            return 1.0

        realized_vol_short = sum(short_vols) / len(short_vols)

        if realized_vol_short <= 1e-8:
            return vol_scale_max

        # leverage_scale = vol_target / realized_vol_short
        # (so when vol is high, we scale down; when vol is low, we scale up)
        raw_scale = vol_target / realized_vol_short
        return max(vol_scale_min, min(vol_scale_max, raw_scale))

    # -- target_weights (main entry point) ----------------------------------- #

    def target_weights(
        self,
        dataset: "MarketDataset",
        idx: int,
        current: Weights,
    ) -> Weights:
        syms = dataset.symbols
        k    = max(1, min(self._p("k_per_side", 2), len(syms) // 2))

        mom_lb  = self._p("momentum_lookback_bars",       168)
        mr_lb   = self._p("mean_reversion_lookback_bars",  48)
        vol_lb  = self._p("vol_lookback_bars",            168)
        ft_fast = self._p("funding_trend_fast",             3)
        ft_slow = self._p("funding_trend_slow",            10)

        target_gross = self._p("target_gross_leverage", 0.35)
        risk_w       = self._p("risk_weighting",        "inverse_vol")
        target_vol   = self._p("target_portfolio_vol",   0.0)
        min_lev      = self._p("min_gross_leverage",    0.05)
        max_lev      = self._p("max_gross_leverage",    0.65)
        strict       = self._p("strict_agreement",      False)

        w_carry = self._p("w_carry",          0.35)
        w_mom   = self._p("w_mom",            0.35)
        w_conf  = self._p("w_confirm",        0.20)
        w_mr    = self._p("w_mean_reversion", 0.05)
        w_vm    = self._p("w_vol_momentum",   0.05)
        w_ft    = self._p("w_funding_trend",  0.00)

        ts = dataset.timeline[idx]

        # -- Signal 1: Funding carry ----------------------------------------- #
        f_raw = {s: float(dataset.last_funding_rate_before(s, ts)) for s in syms}
        fz    = zscores(f_raw)
        carry = {s: -float(fz.get(s, 0.0)) for s in syms}

        # -- Signal 2: Price momentum ---------------------------------------- #
        mom_raw: Dict[str, float] = {}
        for s in syms:
            c  = dataset.perp_close[s]
            i1 = min(idx - 1, len(c) - 1)
            i0 = max(0, idx - 1 - mom_lb)
            c1 = c[i1] if i1 >= 0 else 0.0
            c0 = c[i0] if i0 >= 0 and i0 < len(c) else 0.0
            mom_raw[s] = (c1 / c0 - 1.0) if c0 > 0 else 0.0
        mz  = zscores(mom_raw)
        mom = {s: float(mz.get(s, 0.0)) for s in syms}

        # -- Signal 3: Agreement (carry x momentum) -------------------------- #
        agreement = {s: carry[s] * mom[s] for s in syms}

        # -- Signal 4: Mean reversion (short-term contrarian) ---------------- #
        mr_raw: Dict[str, float] = {}
        for s in syms:
            c  = dataset.perp_close[s]
            i1 = min(idx - 1, len(c) - 1)
            i0 = max(0, idx - 1 - mr_lb)
            c1 = c[i1] if i1 >= 0 else 0.0
            c0 = c[i0] if i0 >= 0 and i0 < len(c) else 0.0
            mr_raw[s] = (c1 / c0 - 1.0) if c0 > 0 else 0.0
        mr_z = zscores(mr_raw)
        mr   = {s: -float(mr_z.get(s, 0.0)) for s in syms}

        # -- Signal 5: Vol-adjusted momentum --------------------------------- #
        vol_mom_raw: Dict[str, float] = {}
        for s in syms:
            v = trailing_vol(
                dataset.perp_close[s],
                end_idx=max(0, idx - 1),
                lookback_bars=vol_lb,
            )
            vol_mom_raw[s] = (mom_raw[s] / v) if v > 0 else mom_raw[s]
        vm_z    = zscores(vol_mom_raw)
        vol_mom = {s: float(vm_z.get(s, 0.0)) for s in syms}

        # -- Signal 6: Funding trend ----------------------------------------- #
        ft_raw = {s: _funding_trend(dataset, s, ts, ft_fast, ft_slow) for s in syms}
        ft_z   = zscores(ft_raw)
        ft     = {s: float(ft_z.get(s, 0.0)) for s in syms}

        # -- Composite score ------------------------------------------------- #
        score: Dict[str, float] = {}
        for s in syms:
            score[s] = (
                w_carry * carry[s]
                + w_mom  * mom[s]
                + w_conf * agreement[s]
                + w_mr   * mr[s]
                + w_vm   * vol_mom[s]
                + w_ft   * ft[s]
            )

        ranked      = sorted(syms, key=lambda s: score[s], reverse=True)
        long_cands  = ranked[:k]
        short_cands = ranked[-k:]

        if strict:
            long_cands  = [s for s in long_cands  if agreement[s] >= 0.0]
            short_cands = [s for s in short_cands if agreement[s] >= 0.0]

        long_syms  = [s for s in long_cands  if s not in short_cands]
        short_syms = [s for s in short_cands if s not in long_cands]

        if not long_syms or not short_syms:
            return {s: 0.0 for s in syms}

        # -- Continuous vol scaling ------------------------------------------ #
        vol_scale       = self._vol_leverage_scale(dataset, idx)
        effective_gross = max(min_lev, min(max_lev, target_gross * vol_scale))

        # -- Portfolio construction (inverse-vol, dollar-neutral) ------------ #
        all_active = list(set(long_syms) | set(short_syms))
        inv_vol: Dict[str, float] = {}
        if risk_w == "inverse_vol":
            for s in all_active:
                v = trailing_vol(dataset.perp_close[s], end_idx=idx, lookback_bars=vol_lb)
                inv_vol[s] = (1.0 / v) if v > 0 else 1.0
        else:
            inv_vol = {s: 1.0 for s in all_active}

        w = normalize_dollar_neutral(long_syms, short_syms, inv_vol, effective_gross)

        # -- Optional portfolio vol targeting (additional layer) -------------- #
        if target_vol > 0:
            w = _apply_vol_target(
                weights           = w,
                closes            = dataset.perp_close,
                end_idx           = idx,
                lookback          = vol_lb,
                target_annual_vol = target_vol,
                min_gross         = min_lev,
                max_gross         = max_lev,
            )

        out = {s: 0.0 for s in syms}
        out.update(w)
        return out
