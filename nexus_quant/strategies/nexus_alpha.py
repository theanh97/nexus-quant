"""
NEXUS Alpha V1 — Agreement-Enhanced Multi-Factor Strategy.

Core Innovation: Funding-Carry × Momentum Agreement Term
─────────────────────────────────────────────────────────
The key alpha source: symbols where BOTH funding-carry AND momentum agree
on direction produce higher-quality signals. Symbols where signals conflict
(e.g., high momentum but also high funding = crowded) get lower conviction.

  final_score[s] = w_carry * carry[s]          # low funding   → LONG
                 + w_mom   * mom[s]             # high momentum → LONG
                 + w_confirm * carry[s]*mom[s]  # agreement boost
                 + w_mr    * (-mr[s])           # mean-reversion: recent losers → LONG
                 + w_vm    * vol_mom[s]         # quality-adjusted momentum → LONG
                 + w_ft    * (-ft[s])           # funding trend: rising → LONG

HIGH final_score → LONG candidate
LOW  final_score → SHORT candidate

Additional enhancements vs combined_carry_mom_v1:
  1. Mean-reversion component (contrarian on short timeframe)
  2. Vol-adjusted momentum (quality signal: mom / trailing_vol)
  3. Funding trend (rising funding trend = bullish sentiment)
  4. Optional min-variance portfolio construction (scipy)
  5. Optional portfolio-level volatility targeting
"""
from __future__ import annotations

import math
import statistics
from typing import Any, Dict, List, Optional

from ._math import normalize_dollar_neutral, trailing_vol, zscores
from .base import Strategy, Weights
from ..data.schema import MarketDataset

try:
    import numpy as _np
    from scipy.optimize import minimize as _sp_minimize
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False
    _np = None  # type: ignore


# ── Portfolio volatility targeting ────────────────────────────────────────────

def _ex_ante_portfolio_vol(
    weights: Dict[str, float],
    closes: Dict[str, List[float]],
    end_idx: int,
    lookback: int,
    bars_per_year: float = 8760.0,
) -> float:
    """Estimate annualised portfolio vol from recent history using given weights."""
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
    """Scale weights so expected portfolio vol ≈ target_annual_vol."""
    if target_annual_vol <= 0 or not weights:
        return weights
    current_vol = _ex_ante_portfolio_vol(weights, closes, end_idx, lookback)
    if current_vol <= 1e-8:
        return weights
    scale = target_annual_vol / current_vol
    gross = sum(abs(v) for v in weights.values())
    new_gross = max(min_gross, min(max_gross, gross * scale))
    if gross < 1e-10:
        return weights
    return {s: w * (new_gross / gross) for s, w in weights.items()}


# ── Min-variance portfolio construction ───────────────────────────────────────

def _portfolio_weights_minvar(
    long_syms: List[str],
    short_syms: List[str],
    closes: Dict[str, List[float]],
    end_idx: int,
    cov_lookback: int,
    vol_lookback: int,
    target_gross: float,
) -> Dict[str, float]:
    """Build dollar-neutral weights using min-variance per leg (scipy) or inverse-vol."""
    all_syms = list(dict.fromkeys(long_syms + short_syms))

    if _HAS_SCIPY and end_idx >= cov_lookback + 5 and len(all_syms) >= 2:
        # Build returns matrix
        rets = _np.zeros((cov_lookback, len(all_syms)), dtype=_np.float64)
        for j, sym in enumerate(all_syms):
            c = closes.get(sym, [])
            for i in range(cov_lookback):
                bar = end_idx - cov_lookback + i
                if 0 < bar < len(c) and c[bar - 1] > 0:
                    rets[i, j] = (c[bar] / c[bar - 1]) - 1.0
        cov = _np.cov(rets.T) + _np.eye(len(all_syms)) * 1e-8
        if cov.ndim < 2:
            cov = cov.reshape(1, 1)
        sym_idx = {s: i for i, s in enumerate(all_syms)}

        def _min_var_leg(syms: List[str]) -> List[float]:
            n = len(syms)
            if n <= 1:
                return [1.0] * n
            leg_idx = [sym_idx[s] for s in syms]
            cov_sub = cov[_np.ix_(leg_idx, leg_idx)]
            w0 = _np.ones(n) / n
            try:
                res = _sp_minimize(
                    fun=lambda w: float(w @ cov_sub @ w),
                    x0=w0,
                    method="SLSQP",
                    constraints={"type": "eq", "fun": lambda w: w.sum() - 1.0},
                    bounds=[(0.0, 1.0)] * n,
                    options={"maxiter": 200, "ftol": 1e-10, "disp": False},
                )
                if res.success:
                    w = _np.maximum(0.0, res.x)
                    t = w.sum()
                    if t > 1e-10:
                        return (w / t).tolist()
            except Exception:
                pass
            return [1.0 / n] * n

        lw = _min_var_leg(long_syms)
        sw = _min_var_leg(short_syms)
        out: Dict[str, float] = {}
        budget = target_gross / 2.0
        for s, w in zip(long_syms, lw):
            out[s] = w * budget
        for s, w in zip(short_syms, sw):
            out[s] = -w * budget
        return out

    # Fallback: inverse-vol
    inv_vol: Dict[str, float] = {}
    for s in all_syms:
        v = trailing_vol(closes.get(s, []), end_idx=end_idx, lookback_bars=vol_lookback)
        inv_vol[s] = (1.0 / v) if v > 0 else 1.0
    return normalize_dollar_neutral(long_syms, short_syms, inv_vol, target_gross)


# ── Funding trend helper ───────────────────────────────────────────────────────

def _funding_trend(
    dataset: MarketDataset,
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
    recent = all_times[-slow_n:]
    rates = [fund_dict[t] for t in recent]
    fast_avg = sum(rates[-fast_n:]) / fast_n
    slow_avg = sum(rates) / len(rates)
    return fast_avg - slow_avg


# ── Main strategy class ────────────────────────────────────────────────────────

class NexusAlphaV1Strategy(Strategy):
    """
    NEXUS Alpha V1 — Agreement-Enhanced Multi-Factor Strategy.

    Extends combined_carry_mom_v1 with:
      - Mean-reversion component (24h contrarian)
      - Vol-adjusted momentum (quality factor)
      - Funding trend signal
      - Optional min-variance portfolio construction
      - Optional portfolio vol targeting

    Parameters
    ----------
    k_per_side                : int   = 2
    w_carry                   : float = 0.35   funding carry (low funding → LONG)
    w_mom                     : float = 0.35   price momentum (long winners)
    w_confirm                 : float = 0.20   carry × momentum agreement boost
    w_mean_reversion          : float = 0.05   short-term mean reversion (losers → LONG)
    w_vol_momentum            : float = 0.05   vol-adjusted momentum (quality)
    w_funding_trend           : float = 0.00   funding trend (rising → LONG)
    momentum_lookback_bars    : int   = 168
    mean_reversion_lookback_bars : int = 24
    vol_lookback_bars         : int   = 168
    funding_trend_fast        : int   = 3
    funding_trend_slow        : int   = 10
    target_portfolio_vol      : float = 0.0    (0 = disable; e.g. 0.12 for 12% annual)
    use_min_variance          : bool  = False
    cov_lookback_bars         : int   = 504
    target_gross_leverage     : float = 0.35
    min_gross_leverage        : float = 0.05
    max_gross_leverage        : float = 0.65
    risk_weighting            : str   = "inverse_vol"
    rebalance_interval_bars   : int   = 168
    strict_agreement          : bool  = False  if True, skip conflicting-signal positions
    """

    def __init__(self, params: Dict[str, Any]) -> None:
        super().__init__(name="nexus_alpha_v1", params=params)

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

    def should_rebalance(self, dataset: MarketDataset, idx: int) -> bool:
        mom_lb = self._p("momentum_lookback_bars", 168)
        vol_lb = self._p("vol_lookback_bars", 168)
        interval = self._p("rebalance_interval_bars", 168)
        warmup = max(mom_lb, vol_lb) + 2
        if idx <= warmup:
            return False

        # Adaptive rebalancing: also rebalance on high cross-sectional dispersion
        adaptive = self._p("adaptive_rebalance", False)
        if adaptive and idx % max(1, interval) != 0:
            # Check if dispersion spike warrants early rebalance (min interval = interval/3)
            min_gap = max(1, interval // 3)
            if hasattr(self, '_last_rebal_idx') and (idx - self._last_rebal_idx) < min_gap:
                return False
            # Compute cross-sectional return dispersion over last 24h
            disp_lb = 24
            if idx > disp_lb + 1:
                rets = []
                for s in dataset.symbols:
                    c = dataset.perp_close[s]
                    if idx - 1 < len(c) and idx - 1 - disp_lb >= 0:
                        c0 = c[idx - 1 - disp_lb]
                        c1 = c[idx - 1]
                        if c0 > 0:
                            rets.append(c1 / c0 - 1.0)
                if len(rets) >= 4:
                    disp = statistics.pstdev(rets)
                    # Trigger if dispersion > 2x recent average
                    avg_disp = getattr(self, '_avg_disp', disp)
                    self._avg_disp = 0.95 * avg_disp + 0.05 * disp  # EMA
                    if disp > 2.0 * avg_disp:
                        self._last_rebal_idx = idx
                        return True
            return False

        if idx % max(1, interval) == 0:
            self._last_rebal_idx = idx
            return True
        return False

    def target_weights(self, dataset: MarketDataset, idx: int, current: Weights) -> Weights:
        syms = dataset.symbols
        k = max(1, min(self._p("k_per_side", 2), len(syms) // 2))

        mom_lb    = self._p("momentum_lookback_bars", 168)
        mr_lb     = self._p("mean_reversion_lookback_bars", 24)
        vol_lb    = self._p("vol_lookback_bars", 168)
        ft_fast   = self._p("funding_trend_fast", 3)
        ft_slow   = self._p("funding_trend_slow", 10)

        target_gross = self._p("target_gross_leverage", 0.35)
        risk_w       = self._p("risk_weighting", "inverse_vol")
        use_min_var  = self._p("use_min_variance", False)
        cov_lb       = self._p("cov_lookback_bars", 504)
        target_vol   = self._p("target_portfolio_vol", 0.0)
        min_lev      = self._p("min_gross_leverage", 0.05)
        max_lev      = self._p("max_gross_leverage", 0.65)
        strict       = self._p("strict_agreement", False)

        # Signal weights
        w_carry  = self._p("w_carry",           0.35)
        w_mom    = self._p("w_mom",             0.35)
        w_conf   = self._p("w_confirm",         0.20)
        w_mr     = self._p("w_mean_reversion",  0.05)
        w_vm     = self._p("w_vol_momentum",    0.05)
        w_ft     = self._p("w_funding_trend",   0.00)

        # Phase 56-57 signal engineering enhancements (defaults preserve V1 behavior)
        conditional_mr   = self._p("conditional_mr", False)      # MR only when momentum agrees
        score_weight_mix = self._p("score_weight_mix", 0.0)      # 0=pure inv-vol, 1=pure score
        w_accel          = self._p("w_acceleration", 0.0)        # momentum acceleration weight
        accel_fast_bars  = self._p("accel_fast_bars", 168)       # short momentum for acceleration
        adaptive_rebal   = self._p("adaptive_rebalance", False)  # rebalance on high dispersion

        ts = dataset.timeline[idx]

        # ── Signal 1: Funding carry ────────────────────────────────
        # carry_signal: -fz so LOW funding → POSITIVE → LONG
        f_raw = {s: float(dataset.last_funding_rate_before(s, ts)) for s in syms}
        fz = zscores(f_raw)
        carry = {s: -float(fz.get(s, 0.0)) for s in syms}  # +ve = prefer LONG

        # ── Signal 2: Price momentum (long-term) ──────────────────
        # mom_signal: +mz so HIGH momentum → POSITIVE → LONG
        mom_raw: Dict[str, float] = {}
        for s in syms:
            c = dataset.perp_close[s]
            i1 = min(idx - 1, len(c) - 1)
            i0 = max(0, idx - 1 - mom_lb)
            c1 = c[i1] if i1 >= 0 else 0.0
            c0 = c[i0] if i0 >= 0 and i0 < len(c) else 0.0
            mom_raw[s] = (c1 / c0 - 1.0) if c0 > 0 else 0.0
        mz = zscores(mom_raw)
        mom = {s: float(mz.get(s, 0.0)) for s in syms}      # +ve = prefer LONG

        # ── Signal 3: Agreement (carry × momentum) ────────────────
        # Positive when both signals agree: low-funding AND high-momentum → LONG
        agreement = {s: carry[s] * mom[s] for s in syms}

        # ── Signal 4: Mean reversion (short-term contrarian) ──────
        # Recent LOSERS → LONG (invert short-term returns)
        mr_raw: Dict[str, float] = {}
        for s in syms:
            c = dataset.perp_close[s]
            i1 = min(idx - 1, len(c) - 1)
            i0 = max(0, idx - 1 - mr_lb)
            c1 = c[i1] if i1 >= 0 else 0.0
            c0 = c[i0] if i0 >= 0 and i0 < len(c) else 0.0
            mr_raw[s] = (c1 / c0 - 1.0) if c0 > 0 else 0.0
        mr_z = zscores(mr_raw)
        mr = {s: -float(mr_z.get(s, 0.0)) for s in syms}    # invert: losers → +ve → LONG

        # Conditional MR: only apply MR where momentum direction agrees
        # "Buy dips in uptrends" = MR positive only when momentum positive
        # "Short bounces in downtrends" = MR negative only when momentum negative
        if conditional_mr:
            for s in syms:
                if mom[s] > 0 and mr[s] < 0:
                    mr[s] = 0.0   # don't short MR in uptrend
                elif mom[s] < 0 and mr[s] > 0:
                    mr[s] = 0.0   # don't long MR in downtrend

        # ── Signal 5: Vol-adjusted momentum ──────────────────────
        # Momentum normalised by trailing vol (Sharpe-ratio-like quality signal)
        vol_mom_raw: Dict[str, float] = {}
        for s in syms:
            v = trailing_vol(dataset.perp_close[s], end_idx=max(0, idx - 1), lookback_bars=vol_lb)
            vol_mom_raw[s] = (mom_raw[s] / v) if v > 0 else mom_raw[s]
        vm_z = zscores(vol_mom_raw)
        vol_mom = {s: float(vm_z.get(s, 0.0)) for s in syms}  # +ve = prefer LONG

        # ── Signal 6: Funding trend ───────────────────────────────
        # Rising funding trend = increasing bullish sentiment → LONG
        ft_raw = {s: _funding_trend(dataset, s, ts, ft_fast, ft_slow) for s in syms}
        ft_z = zscores(ft_raw)
        ft = {s: float(ft_z.get(s, 0.0)) for s in syms}      # +ve = prefer LONG

        # ── Signal 7: Momentum acceleration (Phase 57) ──────────
        # accel = short_mom - long_mom: positive = accelerating upward
        accel: Dict[str, float] = {s: 0.0 for s in syms}
        if w_accel > 0:
            accel_raw: Dict[str, float] = {}
            for s in syms:
                c = dataset.perp_close[s]
                i1 = min(idx - 1, len(c) - 1)
                i0_fast = max(0, idx - 1 - accel_fast_bars)
                c1 = c[i1] if i1 >= 0 else 0.0
                c0_fast = c[i0_fast] if i0_fast >= 0 and i0_fast < len(c) else 0.0
                mom_fast = (c1 / c0_fast - 1.0) if c0_fast > 0 else 0.0
                # acceleration = fast_mom - slow_mom (both raw, before z-scoring)
                accel_raw[s] = mom_fast - mom_raw[s]
            az = zscores(accel_raw)
            accel = {s: float(az.get(s, 0.0)) for s in syms}

        # ── Composite score (HIGH = LONG, LOW = SHORT) ────────────
        score: Dict[str, float] = {}
        for s in syms:
            score[s] = (
                w_carry * carry[s]
                + w_mom   * mom[s]
                + w_conf  * agreement[s]
                + w_mr    * mr[s]
                + w_vm    * vol_mom[s]
                + w_ft    * ft[s]
                + w_accel * accel[s]
            )

        ranked = sorted(syms, key=lambda s: score[s], reverse=True)
        long_cands  = ranked[:k]    # highest score → LONG
        short_cands = ranked[-k:]   # lowest score  → SHORT

        # Optional: strict agreement filter
        if strict:
            long_cands  = [s for s in long_cands  if agreement[s] >= 0.0]
            short_cands = [s for s in short_cands if agreement[s] >= 0.0]

        # Avoid overlap
        long_syms  = [s for s in long_cands  if s not in short_cands]
        short_syms = [s for s in short_cands if s not in long_cands]

        if not long_syms or not short_syms:
            return {s: 0.0 for s in syms}

        # ── Portfolio construction ─────────────────────────────────
        if use_min_var:
            w = _portfolio_weights_minvar(
                long_syms=long_syms,
                short_syms=short_syms,
                closes=dataset.perp_close,
                end_idx=idx,
                cov_lookback=cov_lb,
                vol_lookback=vol_lb,
                target_gross=target_gross,
            )
        else:
            all_active = list(set(long_syms) | set(short_syms))
            inv_vol: Dict[str, float] = {}
            if risk_w == "inverse_vol":
                for s in all_active:
                    v = trailing_vol(dataset.perp_close[s], end_idx=idx, lookback_bars=vol_lb)
                    inv_vol[s] = (1.0 / v) if v > 0 else 1.0
            else:
                inv_vol = {s: 1.0 for s in all_active}

            # Score-weighted: blend inv_vol with score magnitude
            if score_weight_mix > 0.0:
                score_wt: Dict[str, float] = {}
                for s in all_active:
                    score_wt[s] = max(0.01, abs(score.get(s, 0.0)))
                # Blend: (1-mix)*inv_vol + mix*score_magnitude
                blend_wt: Dict[str, float] = {}
                for s in all_active:
                    iv = inv_vol.get(s, 1.0)
                    sw = score_wt.get(s, 1.0)
                    blend_wt[s] = (1.0 - score_weight_mix) * iv + score_weight_mix * sw
                inv_vol = blend_wt

            w = normalize_dollar_neutral(long_syms, short_syms, inv_vol, target_gross)

        # ── Portfolio vol targeting (optional) ────────────────────
        if target_vol > 0:
            w = _apply_vol_target(
                weights=w,
                closes=dataset.perp_close,
                end_idx=idx,
                lookback=vol_lb,
                target_annual_vol=target_vol,
                min_gross=min_lev,
                max_gross=max_lev,
            )

        out = {s: 0.0 for s in syms}
        out.update(w)
        return out
