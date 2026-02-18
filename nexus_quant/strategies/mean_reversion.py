from __future__ import annotations

from typing import Any, Dict

from .base import Strategy, Weights
from ._math import normalize_dollar_neutral, trailing_vol
from ..data.schema import MarketDataset


class MeanReversionCrossSectionV1Strategy(Strategy):
    """
    Simple cross-sectional short-term reversal:
    - rank by recent return over lookback
    - long losers, short winners (dollar-neutral)
    """

    def __init__(self, params: Dict[str, Any]) -> None:
        super().__init__(name="mean_reversion_xs_v1", params=params)

    def should_rebalance(self, dataset: MarketDataset, idx: int) -> bool:
        interval = int(self.params.get("rebalance_interval_bars") or 24)
        lookback = int(self.params.get("lookback_bars") or 24)
        if idx <= lookback + 2:
            return False
        return idx % max(1, interval) == 0

    def target_weights(self, dataset: MarketDataset, idx: int, current: Weights) -> Weights:
        lookback = int(self.params.get("lookback_bars") or 24)
        k = int(self.params.get("k_per_side") or 2)
        k = max(1, min(k, max(1, len(dataset.symbols) // 2)))

        risk_weighting = str(self.params.get("risk_weighting") or "equal")
        vol_lookback = int(self.params.get("vol_lookback_bars") or 72)
        target_gross = float(self.params.get("target_gross_leverage") or 1.0)

        rets = {}
        for s in dataset.symbols:
            c1 = dataset.perp_close[s][idx - 1]
            c0 = dataset.perp_close[s][max(0, idx - 1 - lookback)]
            rets[s] = (c1 / c0) - 1.0 if c0 != 0 else 0.0

        ranked = sorted(dataset.symbols, key=lambda s: rets[s], reverse=True)
        # Mean-reversion: short winners, long losers
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

