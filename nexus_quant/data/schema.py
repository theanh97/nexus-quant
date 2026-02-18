from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import bisect


@dataclass(frozen=True)
class MarketDataset:
    """
    Minimal aligned multi-symbol dataset for perp research.

    All series are aligned to `timeline` indices.
    Funding is event-based: per symbol mapping epoch->rate.
    """

    provider: str
    timeline: List[int]
    symbols: List[str]
    perp_close: Dict[str, List[float]]
    spot_close: Optional[Dict[str, List[float]]]
    funding: Dict[str, Dict[int, float]]
    fingerprint: str

    # Optional extended fields (for real venues / higher-fidelity sims)
    perp_volume: Optional[Dict[str, List[float]]] = None
    spot_volume: Optional[Dict[str, List[float]]] = None
    perp_mark_close: Optional[Dict[str, List[float]]] = None
    perp_index_close: Optional[Dict[str, List[float]]] = None
    bid_close: Optional[Dict[str, List[float]]] = None
    ask_close: Optional[Dict[str, List[float]]] = None
    meta: Dict[str, Any] = field(default_factory=dict)

    # Precomputed for fast "last funding before t" queries (optional).
    _funding_times: Dict[str, List[int]] = field(default_factory=dict)

    def close(self, symbol: str, idx: int) -> float:
        return self.perp_close[symbol][idx]

    def spot(self, symbol: str, idx: int) -> Optional[float]:
        if self.spot_close is None:
            return None
        return self.spot_close[symbol][idx]

    def basis(self, symbol: str, idx: int) -> float:
        sp = self.spot(symbol, idx)
        if sp is None or sp == 0:
            return 0.0
        return (self.close(symbol, idx) / sp) - 1.0

    def funding_rate_at(self, symbol: str, ts: int) -> float:
        return float(self.funding.get(symbol, {}).get(ts, 0.0))

    def last_funding_rate_before(self, symbol: str, ts: int) -> float:
        times = self._funding_times.get(symbol)
        if not times:
            times = sorted((self.funding.get(symbol) or {}).keys())
            if not times:
                return 0.0
        i = bisect.bisect_left(times, ts) - 1
        if i < 0:
            return 0.0
        t0 = times[i]
        return float(self.funding[symbol].get(t0, 0.0))
