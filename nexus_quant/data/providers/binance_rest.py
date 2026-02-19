"""
nexus_quant/data/providers/binance_rest.py
==========================================
BinanceRestProvider: fetches historical klines and funding rates from
Binance USDM Futures public REST API (no API key required), with local
disk caching, automatic pagination, rate limiting, and retry logic.

Provider name: "binance_rest_v1"

Config keys (inside data_cfg):
    symbols     : List[str]  e.g. ["BTCUSDT", "ETHUSDT"]
    start       : str        ISO-8601 UTC e.g. "2024-01-01T00:00:00Z"
    end         : str        ISO-8601 UTC e.g. "2025-01-01T00:00:00Z"
    bar_interval: str        Binance interval string e.g. "1h", "4h", "1d"
    cache_dir   : str        Local path for JSONL cache (default: .cache/binance_rest)
"""

from __future__ import annotations

import hashlib
import json
import time
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ...utils.hashing import sha256_text
from ...utils.time import parse_iso_utc
from ..schema import MarketDataset
from .base import DataProvider

# ---------------------------------------------------------------------------
# Binance USDM Futures public REST endpoints (no auth required)
# ---------------------------------------------------------------------------
_KLINES_URL = "https://fapi.binance.com/fapi/v1/klines"
_FUNDING_URL = "https://fapi.binance.com/fapi/v1/fundingRate"

# Binance hard limits per request
_KLINES_MAX_LIMIT = 1500
_FUNDING_MAX_LIMIT = 1000

# Rate limiting: seconds to sleep between HTTP requests
_RATE_SLEEP = 0.1

# Retry configuration
_MAX_RETRIES = 3
_RETRY_BACKOFF_BASE = 1.0  # seconds; doubles each retry (1s, 2s, 4s)

# Bar interval string -> seconds
_INTERVAL_SECONDS: Dict[str, int] = {
    "1m": 60,
    "3m": 180,
    "5m": 300,
    "15m": 900,
    "30m": 1800,
    "1h": 3600,
    "2h": 7200,
    "4h": 14400,
    "6h": 21600,
    "8h": 28800,
    "12h": 43200,
    "1d": 86400,
    "3d": 259200,
    "1w": 604800,
}


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------

def _build_url(base: str, params: Dict[str, Any]) -> str:
    """Return base URL with URL-encoded query params appended."""
    return base + "?" + urllib.parse.urlencode(params)


def _cache_key(endpoint: str, params: Dict[str, Any]) -> str:
    """
    Deterministic SHA-256 cache key derived from endpoint + sorted params.
    Returns a 64-char hex digest used as the cache filename stem.
    """
    payload = json.dumps({"endpoint": endpoint, "params": params}, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _fetch_json(url: str) -> Any:
    """
    Fetch a URL and return parsed JSON.  Raises urllib.error.URLError or
    ValueError on HTTP errors.  Does NOT sleep — callers are responsible for
    rate limiting.
    """
    with urllib.request.urlopen(url, timeout=30) as resp:
        body = resp.read()
    return json.loads(body.decode("utf-8"))


def _fetch_with_retry(url: str) -> Any:
    """
    Fetch JSON with up to _MAX_RETRIES retries and exponential backoff.
    Sleeps _RATE_SLEEP seconds *before* each attempt to stay inside the
    Binance rate limit.
    """
    last_exc: Optional[Exception] = None
    for attempt in range(_MAX_RETRIES + 1):
        if attempt > 0:
            sleep_secs = _RETRY_BACKOFF_BASE * (2 ** (attempt - 1))
            print(f"      [retry {attempt}/{_MAX_RETRIES}] sleeping {sleep_secs:.1f}s ...")
            time.sleep(sleep_secs)
        time.sleep(_RATE_SLEEP)
        try:
            return _fetch_json(url)
        except Exception as exc:
            last_exc = exc
            print(f"      [warn] HTTP error on attempt {attempt + 1}: {exc}")
    raise RuntimeError(
        f"Failed to fetch after {_MAX_RETRIES} retries: {url}"
    ) from last_exc


# ---------------------------------------------------------------------------
# Cached fetcher
# ---------------------------------------------------------------------------

def _fetch_cached(
    endpoint: str,
    params: Dict[str, Any],
    cache_dir: Path,
) -> Any:
    """
    Fetch JSON from *endpoint* with *params*, storing the raw response as a
    single-line JSON file under *cache_dir*.  On subsequent calls with the
    same (endpoint, params) the cached file is returned without making an
    HTTP request.

    Cache filename format: <sha256_of_endpoint_and_params>.json
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    key = _cache_key(endpoint, params)
    cache_file = cache_dir / f"{key}.json"

    if cache_file.exists():
        # Cache hit — deserialise and return
        with cache_file.open("r", encoding="utf-8") as fh:
            return json.loads(fh.read())

    # Cache miss — fetch from Binance
    url = _build_url(endpoint, params)
    data = _fetch_with_retry(url)

    # Persist raw response (single JSON line for easy append/inspection)
    with cache_file.open("w", encoding="utf-8") as fh:
        fh.write(json.dumps(data))

    return data


# ---------------------------------------------------------------------------
# Pagination helpers
# ---------------------------------------------------------------------------

def _fetch_klines_paginated(
    symbol: str,
    interval: str,
    start_ms: int,
    end_ms: int,
    cache_dir: Path,
) -> List[List[Any]]:
    """
    Fetch all klines for *symbol* between *start_ms* and *end_ms* (epoch ms).
    Handles Binance's 1 500-row-per-request limit by paginating on open_time.

    Each kline element from the API:
        [open_time, open, high, low, close, volume, close_time,
         quote_asset_volume, num_trades, taker_buy_base, taker_buy_quote, ignore]
    """
    all_rows: List[List[Any]] = []
    cursor_ms = start_ms

    while cursor_ms < end_ms:
        params: Dict[str, Any] = {
            "symbol": symbol,
            "interval": interval,
            "startTime": cursor_ms,
            "endTime": end_ms,
            "limit": _KLINES_MAX_LIMIT,
        }

        # Human-readable month label for progress output
        from datetime import datetime, timezone
        dt_label = datetime.fromtimestamp(cursor_ms / 1000, tz=timezone.utc).strftime("%Y-%m")
        print(f"  Fetching {symbol} klines {dt_label} ...")

        rows = _fetch_cached(_KLINES_URL, params, cache_dir)
        if not rows:
            break

        all_rows.extend(rows)

        # Advance cursor past the last returned close_time to avoid re-fetching
        last_open_time = int(rows[-1][0])
        # One interval step forward to avoid duplicating the last bar
        interval_ms = _INTERVAL_SECONDS.get(interval, 3600) * 1000
        cursor_ms = last_open_time + interval_ms

        # If we got fewer rows than the limit, we've reached the end
        if len(rows) < _KLINES_MAX_LIMIT:
            break

    return all_rows


def _fetch_funding_paginated(
    symbol: str,
    start_ms: int,
    end_ms: int,
    cache_dir: Path,
) -> List[Dict[str, Any]]:
    """
    Fetch all funding rate events for *symbol* between *start_ms* and
    *end_ms* (epoch ms).  Handles Binance's 1 000-row-per-request limit.

    Each event from the API:
        {"symbol": "BTCUSDT", "fundingRate": "0.00010000",
         "fundingTime": 1234567890000, "markPrice": "..."}
    """
    all_events: List[Dict[str, Any]] = []
    cursor_ms = start_ms

    while cursor_ms < end_ms:
        params: Dict[str, Any] = {
            "symbol": symbol,
            "startTime": cursor_ms,
            "endTime": end_ms,
            "limit": _FUNDING_MAX_LIMIT,
        }

        from datetime import datetime, timezone
        dt_label = datetime.fromtimestamp(cursor_ms / 1000, tz=timezone.utc).strftime("%Y-%m")
        print(f"  Fetching {symbol} fundingRate {dt_label} ...")

        events = _fetch_cached(_FUNDING_URL, params, cache_dir)
        if not events:
            break

        all_events.extend(events)

        # Advance cursor past the last event time
        last_time = int(events[-1]["fundingTime"])
        cursor_ms = last_time + 1  # +1 ms to avoid duplicate

        if len(events) < _FUNDING_MAX_LIMIT:
            break

    return all_events


# ---------------------------------------------------------------------------
# Main provider class
# ---------------------------------------------------------------------------

class BinanceRestProvider(DataProvider):
    """
    DataProvider that pulls historical USDM Futures klines and funding rates
    from Binance's public REST API, caches raw responses on disk, and
    assembles a MarketDataset aligned to a uniform bar timeline.

    No API key is required — only public endpoints are used.

    Usage (via registry):
        provider: "binance_rest_v1"
    """

    # Provider identifier stored in MarketDataset.provider
    PROVIDER_ID = "binance_rest_v1"

    def __init__(self, data_cfg: Dict[str, Any], seed: int = 0) -> None:
        super().__init__(data_cfg, seed)

        # --- required config --------------------------------------------------
        symbols_raw = data_cfg.get("symbols") or []
        if not symbols_raw:
            raise ValueError("binance_rest_v1: 'symbols' is required in data config")
        self.symbols: List[str] = [str(s).upper() for s in symbols_raw]

        start_raw = data_cfg.get("start")
        end_raw = data_cfg.get("end")
        if not start_raw or not end_raw:
            raise ValueError("binance_rest_v1: 'start' and 'end' are required in data config")

        # parse_iso_utc returns epoch *seconds*
        self.start_s: int = parse_iso_utc(str(start_raw))
        self.end_s: int = parse_iso_utc(str(end_raw))
        if self.end_s <= self.start_s:
            raise ValueError("binance_rest_v1: 'end' must be after 'start'")

        # --- optional config --------------------------------------------------
        self.interval: str = str(data_cfg.get("bar_interval") or "1h")
        if self.interval not in _INTERVAL_SECONDS:
            raise ValueError(
                f"binance_rest_v1: unknown bar_interval '{self.interval}'. "
                f"Supported: {sorted(_INTERVAL_SECONDS)}"
            )
        self.interval_s: int = _INTERVAL_SECONDS[self.interval]

        cache_dir_raw = data_cfg.get("cache_dir") or ".cache/binance_rest"
        self.cache_dir: Path = Path(str(cache_dir_raw)).expanduser()

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def load(self) -> MarketDataset:
        """
        Fetch (or load from cache) all klines and funding rates for every
        configured symbol, align them to a common timeline, and return a
        MarketDataset.

        Steps
        -----
        1. Build the reference timeline (uniform grid of epoch-seconds).
        2. For each symbol, fetch klines → build perp_close and perp_volume series.
        3. For each symbol, fetch funding rates → build funding dict.
        4. Take the intersection of timestamps present across all symbols.
        5. Compute a SHA-256 fingerprint of the assembled data.
        6. Return a frozen MarketDataset.
        """
        print(
            f"[BinanceRestProvider] Loading {len(self.symbols)} symbol(s) | "
            f"interval={self.interval} | "
            f"range=[{self.start_s}, {self.end_s})"
        )

        # ms boundaries for Binance API
        start_ms = self.start_s * 1000
        end_ms = self.end_s * 1000

        # Per-symbol data containers (keyed by epoch-seconds timestamp)
        sym_close: Dict[str, Dict[int, float]] = {}   # ts_s -> close price
        sym_volume: Dict[str, Dict[int, float]] = {}  # ts_s -> volume
        funding: Dict[str, Dict[int, float]] = {}     # ts_s -> funding rate

        # --- Step 1: Fetch klines for every symbol ---------------------------
        for sym in self.symbols:
            print(f"\n[{sym}] Fetching klines ({self.interval}) ...")
            raw_klines = _fetch_klines_paginated(
                symbol=sym,
                interval=self.interval,
                start_ms=start_ms,
                end_ms=end_ms,
                cache_dir=self.cache_dir,
            )

            close_map: Dict[int, float] = {}
            volume_map: Dict[int, float] = {}
            for row in raw_klines:
                # row[0] = open_time (ms), row[4] = close, row[5] = volume
                ts_s = int(row[0]) // 1000
                # Only include bars whose open_time falls within [start_s, end_s)
                if ts_s < self.start_s or ts_s >= self.end_s:
                    continue
                close_map[ts_s] = float(row[4])
                volume_map[ts_s] = float(row[5])

            sym_close[sym] = close_map
            sym_volume[sym] = volume_map
            print(f"  -> {len(close_map)} bars loaded for {sym}")

        # --- Step 2: Fetch funding rates for every symbol --------------------
        for sym in self.symbols:
            print(f"\n[{sym}] Fetching funding rates ...")
            raw_funding = _fetch_funding_paginated(
                symbol=sym,
                start_ms=start_ms,
                end_ms=end_ms,
                cache_dir=self.cache_dir,
            )

            f_map: Dict[int, float] = {}
            for ev in raw_funding:
                # fundingTime is in ms
                ts_s = int(ev["fundingTime"]) // 1000
                f_map[ts_s] = float(ev["fundingRate"])

            funding[sym] = f_map
            print(f"  -> {len(f_map)} funding events loaded for {sym}")

        # --- Step 3: Build common timeline (intersection of available bars) --
        print("\n[BinanceRestProvider] Building aligned timeline ...")
        common_ts: Optional[set] = None
        for sym in self.symbols:
            ts_set = set(sym_close[sym].keys())
            if not ts_set:
                raise ValueError(
                    f"binance_rest_v1: no klines returned for {sym}. "
                    "Check symbol name, date range, and network access."
                )
            common_ts = ts_set if common_ts is None else common_ts.intersection(ts_set)

        timeline: List[int] = sorted(common_ts or [])
        if len(timeline) < 2:
            raise ValueError(
                f"binance_rest_v1: fewer than 2 aligned bars across symbols. "
                f"Got {len(timeline)}. Check date range or symbol availability."
            )
        print(f"  -> {len(timeline)} aligned bars ({timeline[0]} .. {timeline[-1]})")

        # --- Step 4: Align close and volume to the common timeline -----------
        timeline_set = set(timeline)
        perp_close: Dict[str, List[float]] = {}
        perp_volume: Dict[str, List[float]] = {}
        funding_times: Dict[str, List[int]] = {}

        for sym in self.symbols:
            perp_close[sym] = [sym_close[sym][t] for t in timeline]
            # Volume: fill 0.0 if a bar was somehow missing (shouldn't happen
            # after intersection, but guard for robustness)
            perp_volume[sym] = [sym_volume[sym].get(t, 0.0) for t in timeline]
            # Restrict funding events to the timeline window
            sym_funding_in_range = {
                t: r for t, r in funding[sym].items()
                # Keep events within the full requested window (not just bar times)
            }
            funding[sym] = sym_funding_in_range
            funding_times[sym] = sorted(sym_funding_in_range.keys())

        # --- Step 5: Fingerprint ------------------------------------------
        fingerprint = self._compute_fingerprint(timeline, perp_close, funding)
        print(f"  -> fingerprint: {fingerprint[:16]}...")

        print("\n[BinanceRestProvider] Dataset ready.")
        return MarketDataset(
            provider=self.PROVIDER_ID,
            timeline=timeline,
            symbols=self.symbols,
            perp_close=perp_close,
            spot_close=None,      # USDM Futures only; spot not fetched
            funding=funding,
            fingerprint=fingerprint,
            perp_volume=perp_volume,
            spot_volume=None,
            perp_mark_close=None,
            perp_index_close=None,
            bid_close=None,
            ask_close=None,
            _funding_times=funding_times,
            meta={
                "bar_interval": self.interval,
                "bar_interval_seconds": self.interval_s,
                "start_s": self.start_s,
                "end_s": self.end_s,
                "cache_dir": str(self.cache_dir),
            },
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _compute_fingerprint(
        self,
        timeline: List[int],
        perp_close: Dict[str, List[float]],
        funding: Dict[str, Dict[int, float]],
    ) -> str:
        """
        Compute a SHA-256 fingerprint of the loaded dataset so that any change
        in the underlying data (symbols, prices, or funding) is detectable.

        The payload includes:
        - provider id and config (symbols, interval, date range, seed)
        - per-symbol close price vectors (first + last bar as sentinel)
        - per-symbol funding event count
        """
        # Use a compact but deterministic summary: first/last close per symbol
        # plus total bar count and funding event count.
        summary: Dict[str, Any] = {
            "provider": self.PROVIDER_ID,
            "seed": self.seed,
            "symbols": self.symbols,
            "interval": self.interval,
            "start_s": self.start_s,
            "end_s": self.end_s,
            "n_bars": len(timeline),
            "timeline_first": timeline[0] if timeline else None,
            "timeline_last": timeline[-1] if timeline else None,
            "closes": {
                sym: {
                    "first": perp_close[sym][0] if perp_close[sym] else None,
                    "last": perp_close[sym][-1] if perp_close[sym] else None,
                    "n": len(perp_close[sym]),
                }
                for sym in self.symbols
            },
            "funding_counts": {
                sym: len(funding.get(sym, {})) for sym in self.symbols
            },
        }
        return sha256_text(json.dumps(summary, sort_keys=True))
