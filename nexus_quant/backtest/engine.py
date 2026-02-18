from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from ..data.schema import MarketDataset
from ..strategies.base import Strategy, Weights
from ..utils.time import iso_utc
from .costs import ExecutionCostModel


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
        n = len(dataset.timeline)
        if n < 2:
            raise ValueError("Dataset too short")

        equity = 1.0
        equity_curve: List[float] = [equity]
        returns: List[float] = []

        weights: Weights = {s: 0.0 for s in symbols}
        trades: List[Dict[str, Any]] = []

        price_pnl = 0.0
        funding_pnl = 0.0
        cost_pnl = 0.0

        for idx in range(1, n):
            ts = dataset.timeline[idx]
            prev_equity = equity

            # Price PnL over (idx-1 -> idx) using weights held during that interval.
            port_ret = 0.0
            for s in symbols:
                c0 = dataset.close(s, idx - 1)
                c1 = dataset.close(s, idx)
                r = (c1 / c0) - 1.0 if c0 != 0 else 0.0
                port_ret += float(weights.get(s, 0.0)) * r
            dp = equity * port_ret
            equity += dp
            price_pnl += dp

            # Rebalance at timestamp ts (after the bar closed). Costs are charged on traded notional.
            if strategy.should_rebalance(dataset, idx):
                target = strategy.target_weights(dataset, idx, weights)
                for s in symbols:
                    target.setdefault(s, 0.0)

                turnover = 0.0
                for s in symbols:
                    d = float(target.get(s, 0.0)) - float(weights.get(s, 0.0))
                    turnover += abs(d)

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

            # Funding PnL (event-based). Apply on positions held at ts (after any rebalance).
            fp = 0.0
            equity_before_funding = equity
            for s in symbols:
                fr = dataset.funding_rate_at(s, ts)
                if fr == 0.0:
                    continue
                # Long pays when funding>0, short receives (and vice versa).
                fp += -(equity_before_funding * float(weights.get(s, 0.0)) * float(fr))
            equity += fp
            funding_pnl += fp

            equity_curve.append(equity)
            returns.append((equity / prev_equity) - 1.0 if prev_equity != 0 else 0.0)

        return BacktestResult(
            strategy=strategy.describe(),
            timeline=list(dataset.timeline),
            equity_curve=equity_curve,
            returns=returns,
            trades=trades,
            breakdown={
                "price_pnl": price_pnl,
                "funding_pnl": funding_pnl,
                "cost_pnl": cost_pnl,
            },
            data_fingerprint=dataset.fingerprint,
            code_fingerprint="unknown",
        )
