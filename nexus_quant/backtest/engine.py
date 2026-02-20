from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np

from ..data.schema import MarketDataset
from ..strategies.base import Strategy, Weights
from ..utils.time import iso_utc
from .costs import ExecutionCostModel

# Minimum equity to prevent NaN returns and negative-equity artifacts.
_EQUITY_FLOOR = 1e-10


@dataclass(frozen=True)
class BacktestConfig:
    costs: ExecutionCostModel


@dataclass
class BacktestResult:
    strategy: Dict[str, Any]
    timeline: List[int]
    equity_curve: List[float]
    returns: List[float]
    trades: List[Dict[str, Any]]
    breakdown: Dict[str, float]
    data_fingerprint: str
    code_fingerprint: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "strategy": self.strategy,
            "timeline": [iso_utc(t) for t in self.timeline],
            "equity_curve": self.equity_curve,
            "returns": self.returns,
            "trades": self.trades,
            "breakdown": self.breakdown,
            "data_fingerprint": self.data_fingerprint,
            "code_fingerprint": self.code_fingerprint,
        }


class BacktestEngine:
    def __init__(self, cfg: BacktestConfig) -> None:
        self.cfg = cfg

    def run(self, dataset: MarketDataset, strategy: Strategy, seed: int = 0) -> BacktestResult:
        symbols = list(dataset.symbols)
        n_sym = len(symbols)
        n = len(dataset.timeline)
        if n < 2:
            raise ValueError("Dataset too short")

        # ── Pre-compute close-price matrix & single-bar returns ────────
        # close_mat: shape (n_bars, n_symbols)
        close_mat = np.empty((n, n_sym), dtype=np.float64)
        for j, s in enumerate(symbols):
            close_mat[:, j] = dataset.perp_close[s][:n]

        # ret_mat[i, j] = close_mat[i,j] / close_mat[i-1,j] - 1  (0 when prev=0)
        ret_mat = np.zeros((n, n_sym), dtype=np.float64)
        prev_close = close_mat[:-1]
        safe_mask = prev_close != 0.0
        ret_mat[1:][safe_mask] = (close_mat[1:][safe_mask] / prev_close[safe_mask]) - 1.0

        # ── Pre-index funding events for O(1) lookup ──────────────────
        # Build a set per symbol for fast "does a funding event exist at ts?" check
        has_funding = dataset.has_funding
        funding_sets: List[Dict[int, float]] = []
        if has_funding:
            for s in symbols:
                funding_sets.append(dataset.funding.get(s, {}))

        # ── Main simulation loop ──────────────────────────────────────
        equity = 1.0
        equity_curve = np.empty(n, dtype=np.float64)
        equity_curve[0] = equity
        bar_returns = np.empty(n - 1, dtype=np.float64)

        weight_vec = np.zeros(n_sym, dtype=np.float64)
        weights: Weights = {s: 0.0 for s in symbols}
        trades: List[Dict[str, Any]] = []

        price_pnl = 0.0
        funding_pnl = 0.0
        cost_pnl = 0.0

        for idx in range(1, n):
            ts = dataset.timeline[idx]
            prev_equity = equity

            # Price PnL: vectorized dot product (weights · returns)
            port_ret = float(np.dot(weight_vec, ret_mat[idx]))
            dp = equity * port_ret
            equity += dp
            price_pnl += dp

            # Rebalance at timestamp ts (after the bar closed).
            if strategy.should_rebalance(dataset, idx):
                target = strategy.target_weights(dataset, idx, weights)
                for s in symbols:
                    target.setdefault(s, 0.0)

                turnover = 0.0
                for j, s in enumerate(symbols):
                    tw = float(target.get(s, 0.0))
                    turnover += abs(tw - weight_vec[j])
                    weight_vec[j] = tw

                bd = self.cfg.costs.cost(equity=equity, turnover=turnover)
                cost = float(bd.get("cost", 0.0))
                equity -= cost
                cost_pnl -= cost

                weights = {s: float(target.get(s, 0.0)) for s in symbols}
                trades.append(
                    {
                        "idx": idx,
                        "ts_epoch": int(ts),
                        "ts": iso_utc(ts),
                        "turnover": turnover,
                        **{k: bd[k] for k in sorted(bd.keys())},
                    }
                )

            # Funding PnL (event-based, crypto-specific).
            if has_funding:
                fp = 0.0
                eq_bf = equity
                for j in range(n_sym):
                    fr = funding_sets[j].get(ts, 0.0)
                    if fr == 0.0:
                        continue
                    fp -= eq_bf * weight_vec[j] * fr
                equity += fp
                funding_pnl += fp

            # Equity floor: prevent negative equity / NaN returns.
            if equity < _EQUITY_FLOOR:
                equity = _EQUITY_FLOOR

            equity_curve[idx] = equity
            bar_returns[idx - 1] = (equity / prev_equity) - 1.0 if prev_equity > _EQUITY_FLOOR else 0.0

        return BacktestResult(
            strategy=strategy.describe(),
            timeline=list(dataset.timeline),
            equity_curve=equity_curve.tolist(),
            returns=bar_returns.tolist(),
            trades=trades,
            breakdown={
                "price_pnl": price_pnl,
                "funding_pnl": funding_pnl,
                "cost_pnl": cost_pnl,
            },
            data_fingerprint=dataset.fingerprint,
            code_fingerprint="unknown",
        )
