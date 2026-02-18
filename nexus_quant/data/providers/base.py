from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict

from ..schema import MarketDataset


class DataProvider(ABC):
    def __init__(self, cfg: Dict[str, Any], seed: int) -> None:
        self.cfg = cfg
        self.seed = seed

    @abstractmethod
    def load(self) -> MarketDataset:
        raise NotImplementedError

