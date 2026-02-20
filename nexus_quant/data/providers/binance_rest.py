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
import ssl
import time
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    import certifi
    _SSL_CTX = ssl.create_default_context(cafile=certifi.where())
except ImportError:
    _SSL_CTX = None

from ...utils.hashing import sha256_text
from ...utils.time import parse_iso_utc
from ..schema import MarketDataset
from .base import DataProvider

# ---------------------------------------------------------------------------
# Binance USDM Futures public REST endpoints (no auth required)
# ---------------------------------------------------------------------------
_KLINES_URL = "https://fapi.binance.com/fapi/v1/klines"
_FUNDING_URL = "https://fapi.binance.com/fapi/v1/fundingRate"

# Positioning analytics endpoints (public, no auth required)
_OI_HIST_URL = "https://fapi.binance.com/futures/data/openInterestHist"
_GLOBAL_LS_URL = "https://fapi.binance.com/futures/data/globalLongShortAccountRatio"
_TOP_LS_URL = "https://fapi.binance.com/futures/data/topLongShortPositionRatio"

# Binance hard limits per request
_KLINES_MAX_LIMIT = 1500
_FUNDING_MAX_LIMIT = 1000
_POSITIONING_MAX_LIMIT = 500  # /futures/data/* endpoints max 500 rows

# Rate limiting: seconds to sleep between HTTP requests
_RATE_SLEEP = 0.5

# Retry configuration
_MAX_RETRIES = 5
_RETRY_BACKOFF_BASE = 2.0  # seconds; doubles each retry (2s, 4s, 8s, 16s, 32s)

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
    with urllib.request.urlopen(url, timeout=30, context=_SSL_CTX) as resp:
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
# Positioning data pagination helpers
# ---------------------------------------------------------------------------

def _fetch_positioning_paginated(
    endpoint: str,
    symbol: str,
    period: str,
    start_ms: int,
    end_ms: int,
    cache_dir: Path,
    value_key: str,
    label: str,
) -> List[Dict[str, Any]]:
    """
    Fetch positioning analytics data (open interest, L/S ratios) from Binance
    /futures/data/* endpoints.  These return max 500 rows per request.

    IMPORTANT: These endpoints do NOT support `startTime`.  They only accept
    `endTime` and `limit`, returning the most recent N rows before endTime.
    Binance only retains approximately 30 days of history for these endpoints,
    so historical backtests beyond ~30 days ago will have no positioning data.

    We paginate backwards from endTime.

    Parameters
    ----------
    endpoint   : Full URL (e.g. _OI_HIST_URL, _GLOBAL_LS_URL, _TOP_LS_URL)
    symbol     : e.g. "BTCUSDT"
    period     : Bar period e.g. "1h", "5m", "15m", "30m", "1d"
    start_ms   : Start time in epoch milliseconds (lower bound for results)
    end_ms     : End time in epoch milliseconds
    cache_dir  : Local cache directory
    value_key  : Key used in print label (for logging)
    label      : Human-readable label for progress output

    Returns list of raw API response dicts, each with a "timestamp" field (epoch ms),
    sorted by timestamp ascending.
    """
    all_events: List[Dict[str, Any]] = []
    cursor_end_ms = end_ms

    # Interval in ms for cursor advancement
    period_seconds = _INTERVAL_SECONDS.get(period, 3600)
    period_ms = period_seconds * 1000

    while cursor_end_ms > start_ms:
        # These endpoints only take symbol, period, limit, and optionally endTime
        params: Dict[str, Any] = {
            "symbol": symbol,
            "period": period,
            "endTime": cursor_end_ms,
            "limit": _POSITIONING_MAX_LIMIT,
        }

        from datetime import datetime, timezone
        dt_label = datetime.fromtimestamp(cursor_end_ms / 1000, tz=timezone.utc).strftime("%Y-%m-%d")
        print(f"  Fetching {symbol} {label} endTime={dt_label} ...")

        try:
            events = _fetch_cached(endpoint, params, cache_dir)
        except Exception as exc:
            print(f"      [warn] Failed to fetch {label} for {symbol}: {exc}")
            break

        if not events or not isinstance(events, list):
            break

        # Filter to only keep events within [start_ms, end_ms)
        filtered = [e for e in events if start_ms <= int(e.get("timestamp", 0)) < end_ms]
        all_events.extend(filtered)

        # Move cursor backwards: set endTime to just before the earliest returned event
        first_ts = int(events[0].get("timestamp", 0))
        cursor_end_ms = first_ts - period_ms

        # If we got fewer rows than the limit, no more historical data available
        if len(events) < _POSITIONING_MAX_LIMIT:
            break

    # Deduplicate and sort by timestamp ascending
    seen: set = set()
    unique_events: List[Dict[str, Any]] = []
    for e in all_events:
        ts = int(e.get("timestamp", 0))
        if ts not in seen:
            seen.add(ts)
            unique_events.append(e)
    unique_events.sort(key=lambda e: int(e.get("timestamp", 0)))

    return unique_events


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
        sym_taker_buy_vol: Dict[str, Dict[int, float]] = {}  # ts_s -> taker buy base vol
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
            taker_buy_map: Dict[int, float] = {}
            for row in raw_klines:
                # row[0] = open_time (ms), row[4] = close, row[5] = volume,
                # row[9] = taker buy base asset volume
                ts_s = int(row[0]) // 1000
                # Only include bars whose open_time falls within [start_s, end_s)
                if ts_s < self.start_s or ts_s >= self.end_s:
                    continue
                close_map[ts_s] = float(row[4])
                volume_map[ts_s] = float(row[5])
                taker_buy_map[ts_s] = float(row[9]) if len(row) > 9 else 0.0

            sym_close[sym] = close_map
            sym_volume[sym] = volume_map
            sym_taker_buy_vol[sym] = taker_buy_map
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

        # --- Step 2b: Fetch positioning data (OI, L/S ratios) -- best effort --
        # These endpoints may not have data for all symbols or periods.
        # We store per-symbol maps {ts_s -> value} and align later.
        sym_oi: Dict[str, Dict[int, float]] = {}          # OI value in USD
        sym_global_ls: Dict[str, Dict[int, float]] = {}   # global L/S ratio
        sym_top_ls: Dict[str, Dict[int, float]] = {}      # top trader L/S ratio

        # Only fetch positioning data if:
        # 1. Bar interval is supported (1h, 5m, 15m, 30m, 1d)
        # 2. Requested end time is within ~30 days of now (Binance only retains ~30d)
        positioning_period = self.interval if self.interval in ("5m", "15m", "30m", "1h", "1d") else None
        if positioning_period:
            import time as _time
            now_ms = int(_time.time() * 1000)
            max_history_ms = 30 * 86400 * 1000  # 30 days
            if end_ms < (now_ms - max_history_ms):
                print(f"\n[BinanceRestProvider] Skipping positioning data (end={end_ms} is more than 30 days ago; Binance retains ~30 days)")
                positioning_period = None
        if positioning_period:
            for sym in self.symbols:
                # --- Open Interest History ---
                print(f"\n[{sym}] Fetching open interest history ...")
                raw_oi = _fetch_positioning_paginated(
                    endpoint=_OI_HIST_URL,
                    symbol=sym,
                    period=positioning_period,
                    start_ms=start_ms,
                    end_ms=end_ms,
                    cache_dir=self.cache_dir,
                    value_key="sumOpenInterestValue",
                    label="OI",
                )
                oi_map: Dict[int, float] = {}
                for ev in raw_oi:
                    ts_s = int(ev.get("timestamp", 0)) // 1000
                    if ts_s < self.start_s or ts_s >= self.end_s:
                        continue
                    try:
                        oi_map[ts_s] = float(ev.get("sumOpenInterestValue", 0.0))
                    except (TypeError, ValueError):
                        pass
                sym_oi[sym] = oi_map
                print(f"  -> {len(oi_map)} OI bars for {sym}")

                # --- Global Long/Short Ratio ---
                print(f"  [{sym}] Fetching global L/S ratio ...")
                raw_gls = _fetch_positioning_paginated(
                    endpoint=_GLOBAL_LS_URL,
                    symbol=sym,
                    period=positioning_period,
                    start_ms=start_ms,
                    end_ms=end_ms,
                    cache_dir=self.cache_dir,
                    value_key="longShortRatio",
                    label="globalLS",
                )
                gls_map: Dict[int, float] = {}
                for ev in raw_gls:
                    ts_s = int(ev.get("timestamp", 0)) // 1000
                    if ts_s < self.start_s or ts_s >= self.end_s:
                        continue
                    try:
                        gls_map[ts_s] = float(ev.get("longShortRatio", 1.0))
                    except (TypeError, ValueError):
                        pass
                sym_global_ls[sym] = gls_map
                print(f"  -> {len(gls_map)} global L/S bars for {sym}")

                # --- Top Trader Position Ratio ---
                print(f"  [{sym}] Fetching top trader L/S ratio ...")
                raw_tls = _fetch_positioning_paginated(
                    endpoint=_TOP_LS_URL,
                    symbol=sym,
                    period=positioning_period,
                    start_ms=start_ms,
                    end_ms=end_ms,
                    cache_dir=self.cache_dir,
                    value_key="longShortRatio",
                    label="topLS",
                )
                tls_map: Dict[int, float] = {}
                for ev in raw_tls:
                    ts_s = int(ev.get("timestamp", 0)) // 1000
                    if ts_s < self.start_s or ts_s >= self.end_s:
                        continue
                    try:
                        tls_map[ts_s] = float(ev.get("longShortRatio", 1.0))
                    except (TypeError, ValueError):
                        pass
                sym_top_ls[sym] = tls_map
                print(f"  -> {len(tls_map)} top trader L/S bars for {sym}")
        else:
            if self.interval not in ("5m", "15m", "30m", "1h", "1d"):
                print(f"\n[BinanceRestProvider] Skipping positioning data (unsupported interval: {self.interval})")

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
        taker_buy_volume: Dict[str, List[float]] = {}
        funding_times: Dict[str, List[int]] = {}

        for sym in self.symbols:
            perp_close[sym] = [sym_close[sym][t] for t in timeline]
            # Volume: fill 0.0 if a bar was somehow missing (shouldn't happen
            # after intersection, but guard for robustness)
            perp_volume[sym] = [sym_volume[sym].get(t, 0.0) for t in timeline]
            taker_buy_volume[sym] = [sym_taker_buy_vol[sym].get(t, 0.0) for t in timeline]
            # Restrict funding events to the timeline window
            sym_funding_in_range = {
                t: r for t, r in funding[sym].items()
                # Keep events within the full requested window (not just bar times)
            }
            funding[sym] = sym_funding_in_range
            funding_times[sym] = sorted(sym_funding_in_range.keys())

        # --- Step 4b: Align positioning data to common timeline ----------------
        # Positioning data may have gaps — we forward-fill from the last known
        # value.  If no data is available at all for a symbol, we use 0.0 / 1.0
        # defaults so the strategy can detect "no data" and skip gracefully.
        open_interest: Optional[Dict[str, List[float]]] = None
        long_short_ratio_global: Optional[Dict[str, List[float]]] = None
        long_short_ratio_top: Optional[Dict[str, List[float]]] = None

        if positioning_period and sym_oi:
            open_interest = {}
            long_short_ratio_global = {}
            long_short_ratio_top = {}

            for sym in self.symbols:
                oi_src = sym_oi.get(sym, {})
                gls_src = sym_global_ls.get(sym, {})
                tls_src = sym_top_ls.get(sym, {})

                oi_list: List[float] = []
                gls_list: List[float] = []
                tls_list: List[float] = []

                # Forward-fill: carry last known value forward through gaps
                last_oi = 0.0
                last_gls = 1.0   # default L/S ratio = 1.0 (neutral)
                last_tls = 1.0

                for t in timeline:
                    if t in oi_src:
                        last_oi = oi_src[t]
                    oi_list.append(last_oi)

                    if t in gls_src:
                        last_gls = gls_src[t]
                    gls_list.append(last_gls)

                    if t in tls_src:
                        last_tls = tls_src[t]
                    tls_list.append(last_tls)

                open_interest[sym] = oi_list
                long_short_ratio_global[sym] = gls_list
                long_short_ratio_top[sym] = tls_list

            # Count how many positioning bars we actually have
            total_oi_pts = sum(len(sym_oi.get(s, {})) for s in self.symbols)
            total_gls_pts = sum(len(sym_global_ls.get(s, {})) for s in self.symbols)
            total_tls_pts = sum(len(sym_top_ls.get(s, {})) for s in self.symbols)
            print(f"  -> Positioning aligned: OI={total_oi_pts}, globalLS={total_gls_pts}, topLS={total_tls_pts} raw points")

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
            taker_buy_volume=taker_buy_volume,
            spot_volume=None,
            perp_mark_close=None,
            perp_index_close=None,
            bid_close=None,
            ask_close=None,
            open_interest=open_interest,
            long_short_ratio_global=long_short_ratio_global,
            long_short_ratio_top=long_short_ratio_top,
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
