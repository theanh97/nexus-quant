from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import bisect


@dataclass(frozen=True)
class MarketDataset:
    """
    Multi-symbol aligned dataset for quantitative research.

    Core fields (market-agnostic):
    - timeline, symbols, perp_close (or use 'close' alias)
    - meta: arbitrary metadata dict
    - features: extensible dict for market-specific data

    Crypto-specific fields (backward-compatible, optional):
    - funding, spot_close, perp_volume, taker_buy_volume, etc.

    For non-crypto markets (FX, equities, options):
    - Use perp_close for the primary price series
    - Use features dict for market-specific data (interest rates, greeks, etc.)
    """

    provider: str
    timeline: List[int]
    symbols: List[str]
    perp_close: Dict[str, List[float]]
    spot_close: Optional[Dict[str, List[float]]]
    funding: Dict[str, Dict[int, float]]
    fingerprint: str

    # Market type identifier (crypto, fx, equity, options)
    market_type: str = "crypto"

    # Extensible features dict — market-specific data goes here
    # Example: {"interest_rate_diff": {"EURUSD": [0.01, ...]}, "swap_points": {...}}
    features: Dict[str, Any] = field(default_factory=dict)

    # Optional extended fields (crypto-specific, backward-compatible)
    perp_volume: Optional[Dict[str, List[float]]] = None
    taker_buy_volume: Optional[Dict[str, List[float]]] = None
    spot_volume: Optional[Dict[str, List[float]]] = None
    perp_mark_close: Optional[Dict[str, List[float]]] = None
    perp_index_close: Optional[Dict[str, List[float]]] = None
    bid_close: Optional[Dict[str, List[float]]] = None
    ask_close: Optional[Dict[str, List[float]]] = None

    # Positioning data (from Binance futures analytics endpoints)
    open_interest: Optional[Dict[str, List[float]]] = None          # symbol -> [OI values per bar]
    long_short_ratio_global: Optional[Dict[str, List[float]]] = None  # symbol -> [global L/S ratio per bar]
    long_short_ratio_top: Optional[Dict[str, List[float]]] = None     # symbol -> [top trader L/S ratio per bar]
    taker_long_short_ratio: Optional[Dict[str, List[float]]] = None  # symbol -> [taker buy/sell vol ratio per bar]
    top_trader_ls_account: Optional[Dict[str, List[float]]] = None   # symbol -> [top trader account L/S ratio per bar]

    meta: Dict[str, Any] = field(default_factory=dict)

    # Precomputed for fast "last funding before t" queries (optional).
    _funding_times: Dict[str, List[int]] = field(default_factory=dict)

    @property
    def has_funding(self) -> bool:
        """Whether this dataset has funding rate data (crypto-specific)."""
        return bool(self.funding)

    def feature(self, name: str, symbol: str = "") -> Any:
        """Get a market-specific feature by name. Returns None if not available."""
        val = self.features.get(name)
        if val is None:
            return None
        if symbol and isinstance(val, dict):
            return val.get(symbol)
        return val

    def close(self, symbol: str, idx: int) -> float:
        series = self.perp_close.get(symbol)
        if series is None or idx < 0 or idx >= len(series):
            return 0.0
        return series[idx]

    def spot(self, symbol: str, idx: int) -> Optional[float]:
        if self.spot_close is None:
            return None
        series = self.spot_close.get(symbol)
        if series is None or idx < 0 or idx >= len(series):
            return None
        return series[idx]

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
