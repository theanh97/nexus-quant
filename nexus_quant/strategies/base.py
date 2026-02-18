from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional

from ..data.schema import MarketDataset


Weights = Dict[str, float]


@dataclass(frozen=True)
class StrategyContext:
    """Mutable state placeholder (future: indicators cache, etc.)."""

    pass


class Strategy(ABC):
    def __init__(self, name: str, params: Dict[str, Any]) -> None:
        self.name = name
        self.params = params
        self.ctx = StrategyContext()

    @abstractmethod
    def should_rebalance(self, dataset: MarketDataset, idx: int) -> bool:
        raise NotImplementedError

    @abstractmethod
    def target_weights(self, dataset: MarketDataset, idx: int, current: Weights) -> Weights:
        """
        Return target notional weights by symbol at timeline index `idx`.
        Convention:
        - weights are fractions of equity notionals (sum abs ~= gross leverage)
        - long > 0, short < 0
        """
        raise NotImplementedError

    def describe(self) -> Dict[str, Any]:
        return {"name": self.name, "params": self.params}

