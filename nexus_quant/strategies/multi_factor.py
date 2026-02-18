from __future__ import annotations

from typing import Any, Dict

from ._math import normalize_dollar_neutral, trailing_vol, zscores
from .base import Strategy, Weights
from ..data.schema import MarketDataset


class MultiFactorCrossSectionV1Strategy(Strategy):
    """
    Cross-sectional multi-factor ranker (perp-only friendly):
    - build factor z-scores cross-sectionally (funding, basis, momentum, mean-reversion)
    - combine into a single score
    - SHORT top score, LONG bottom score (dollar-neutral)

    Sign conventions:
    - funding/basis positive => often crowded long => prefer short (score positive)
    - momentum positive => prefer long => we invert momentum (score -= z(mom))
    - mean-reversion uses short-term return directly (winners short, losers long)
    """

    def __init__(self, params: Dict[str, Any]) -> None:
        super().__init__(name="multi_factor_xs_v1", params=params)

    def should_rebalance(self, dataset: MarketDataset, idx: int) -> bool:
        interval = int(self.params.get("rebalance_interval_bars") or 24)
        reb_on_funding = bool(self.params.get("rebalance_on_funding", False))

        mom_lb = int(self.params.get("momentum_lookback_bars") or 168)
        mr_lb = int(self.params.get("mean_reversion_lookback_bars") or 24)
        vol_lb = int(self.params.get("vol_lookback_bars") or 72)
        min_idx = max(mom_lb, mr_lb, vol_lb) + 2
        if idx <= min_idx:
            return False

        if reb_on_funding:
            ts = dataset.timeline[idx]
            for s in dataset.symbols:
                if ts in dataset.funding.get(s, {}):
                    return True
            return False

        return idx % max(1, interval) == 0

    def target_weights(self, dataset: MarketDataset, idx: int, current: Weights) -> Weights:
        k = int(self.params.get("k_per_side") or 2)
        k = max(1, min(k, max(1, len(dataset.symbols) // 2)))

        w_funding = float(self.params.get("w_funding") or 1.0)
        w_basis = float(self.params.get("w_basis") or 0.5)
        w_mom = float(self.params.get("w_momentum") or 0.5)
        w_mr = float(self.params.get("w_mean_reversion") or 0.5)

        use_basis = bool(self.params.get("use_basis_proxy", True))
        risk_weighting = str(self.params.get("risk_weighting") or "equal")
        vol_lookback = int(self.params.get("vol_lookback_bars") or 72)
        target_gross = float(self.params.get("target_gross_leverage") or 1.0)

        mom_lb = int(self.params.get("momentum_lookback_bars") or 168)
        mr_lb = int(self.params.get("mean_reversion_lookback_bars") or 24)

        ts = dataset.timeline[idx]

        # Funding (lagged): positive -> prefer short
        f_raw = {s: dataset.last_funding_rate_before(s, ts) for s in dataset.symbols}
        fz = zscores(f_raw)

        # Basis (lagged): positive -> perp rich -> prefer short
        if use_basis and dataset.spot_close is not None and idx > 0:
            b_raw = {s: dataset.basis(s, idx - 1) for s in dataset.symbols}
            bz = zscores(b_raw)
        else:
            bz = {s: 0.0 for s in dataset.symbols}

        # Momentum (lagged): positive -> prefer long => invert sign
        mom_raw = {}
        for s in dataset.symbols:
            c1 = dataset.perp_close[s][idx - 1]
            c0 = dataset.perp_close[s][max(0, idx - 1 - mom_lb)]
            mom_raw[s] = (c1 / c0) - 1.0 if c0 != 0 else 0.0
        mom_z = zscores(mom_raw)

        # Mean-reversion (lagged): short-term winners short, losers long => same sign
        mr_raw = {}
        for s in dataset.symbols:
            c1 = dataset.perp_close[s][idx - 1]
            c0 = dataset.perp_close[s][max(0, idx - 1 - mr_lb)]
            mr_raw[s] = (c1 / c0) - 1.0 if c0 != 0 else 0.0
        mr_z = zscores(mr_raw)

        score = {}
        for s in dataset.symbols:
            score[s] = (
                w_funding * float(fz.get(s, 0.0))
                + w_basis * float(bz.get(s, 0.0))
                + w_mom * (-float(mom_z.get(s, 0.0)))
                + w_mr * float(mr_z.get(s, 0.0))
            )

        ranked = sorted(dataset.symbols, key=lambda s: score[s], reverse=True)
        short_syms = ranked[:k]
        long_syms = ranked[-k:]

        inv_vol = {}
        if risk_weighting == "inverse_vol":
            for s in set(long_syms + short_syms):
                vol = trailing_vol(dataset.perp_close[s], end_idx=idx, lookback_bars=vol_lookback)
                inv_vol[s] = (1.0 / vol) if vol > 0 else 1.0
        else:
            for s in set(long_syms + short_syms):
                inv_vol[s] = 1.0

        w = normalize_dollar_neutral(long_syms=long_syms, short_syms=short_syms, inv_vol=inv_vol, target_gross_leverage=target_gross)
        out = {s: 0.0 for s in dataset.symbols}
        out.update(w)
        return out

