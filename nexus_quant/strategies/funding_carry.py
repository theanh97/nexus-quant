from __future__ import annotations

from typing import Any, Dict, List

from .base import Strategy, Weights
from ._math import normalize_dollar_neutral, trailing_vol, zscores
from ..data.schema import MarketDataset


class FundingCarryPerpV1Strategy(Strategy):
    def __init__(self, params: Dict[str, Any]) -> None:
        super().__init__(name="funding_carry_perp_v1", params=params)

    def should_rebalance(self, dataset: MarketDataset, idx: int) -> bool:
        if idx <= 1:
            return False
        if bool(self.params.get("rebalance_on_funding", True)):
            ts = dataset.timeline[idx]
            # Rebalance when there's a funding event (union across symbols).
            for s in dataset.symbols:
                if ts in dataset.funding.get(s, {}):
                    return True
            return False
        interval = int(self.params.get("rebalance_interval_bars") or 8)
        return idx % max(1, interval) == 0

    def target_weights(self, dataset: MarketDataset, idx: int, current: Weights) -> Weights:
        k = int(self.params.get("k_per_side") or 2)
        k = max(1, min(k, max(1, len(dataset.symbols) // 2)))

        use_basis = bool(self.params.get("use_basis_proxy", True))
        basis_w = float(self.params.get("basis_weight") or 0.0)
        risk_weighting = str(self.params.get("risk_weighting") or "equal")
        vol_lookback = int(self.params.get("vol_lookback_bars") or 72)
        target_gross = float(self.params.get("target_gross_leverage") or 1.0)

        ts = dataset.timeline[idx]

        # Use lagged information only (no lookahead).
        f_raw = {s: dataset.last_funding_rate_before(s, ts) for s in dataset.symbols}
        fz = zscores(f_raw)

        if use_basis and dataset.spot_close is not None and idx > 0:
            b_raw = {s: dataset.basis(s, idx - 1) for s in dataset.symbols}
            bz = zscores(b_raw)
        else:
            bz = {s: 0.0 for s in dataset.symbols}

        score = {s: float(fz.get(s, 0.0)) + basis_w * float(bz.get(s, 0.0)) for s in dataset.symbols}
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

        w = normalize_dollar_neutral(
            long_syms=long_syms,
            short_syms=short_syms,
            inv_vol=inv_vol,
            target_gross_leverage=target_gross,
        )

        # Ensure every symbol exists in weights dict.
        out = {s: 0.0 for s in dataset.symbols}
        out.update(w)
        return out

