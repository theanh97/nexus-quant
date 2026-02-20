"""
Yahoo Finance Futures Data Provider
====================================
Fetches daily OHLCV for commodity futures using urllib.request (stdlib only).
Caches locally to avoid repeated downloads.

Supported tickers (continuous front-month):
  Energy : CL=F (WTI Oil), NG=F (Natural Gas), BZ=F (Brent)
  Metals : GC=F (Gold), SI=F (Silver), HG=F (Copper), PL=F (Platinum)
  Grains : ZW=F (Wheat), ZC=F (Corn), ZS=F (Soybeans)
  Softs  : KC=F (Coffee), SB=F (Sugar), CT=F (Cotton)
"""
from __future__ import annotations

import csv
import hashlib
import json
import logging
import os
import time
import urllib.request
import urllib.error
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from nexus_quant.data.providers.base import DataProvider
from nexus_quant.data.schema import MarketDataset

logger = logging.getLogger("nexus.commodity.yahoo")

# ── Universe & metadata ─────────────────────────────────────────────────────

DEFAULT_SYMBOLS = [
    "CL=F",  # WTI Crude Oil
    "NG=F",  # Natural Gas
    "BZ=F",  # Brent Crude
    "GC=F",  # Gold
    "SI=F",  # Silver
    "HG=F",  # Copper
    "PL=F",  # Platinum
    "ZW=F",  # Wheat
    "ZC=F",  # Corn
    "ZS=F",  # Soybeans
    "KC=F",  # Coffee
    "SB=F",  # Sugar
    "CT=F",  # Cotton
]

SECTOR_MAP: Dict[str, str] = {
    "CL=F": "energy",
    "NG=F": "energy",
    "BZ=F": "energy",
    "GC=F": "metals",
    "SI=F": "metals",
    "HG=F": "metals",
    "PL=F": "metals",
    "ZW=F": "grains",
    "ZC=F": "grains",
    "ZS=F": "grains",
    "KC=F": "softs",
    "SB=F": "softs",
    "CT=F": "softs",
    "LE=F": "livestock",
    "HE=F": "livestock",
}

SYMBOL_NAMES: Dict[str, str] = {
    "CL=F": "WTI Crude Oil",
    "NG=F": "Natural Gas",
    "BZ=F": "Brent Crude",
    "GC=F": "Gold",
    "SI=F": "Silver",
    "HG=F": "Copper",
    "PL=F": "Platinum",
    "ZW=F": "Wheat",
    "ZC=F": "Corn",
    "ZS=F": "Soybeans",
    "KC=F": "Coffee",
    "SB=F": "Sugar",
    "CT=F": "Cotton",
    "LE=F": "Live Cattle",
    "HE=F": "Lean Hogs",
}

# ── Provider ─────────────────────────────────────────────────────────────────


class YahooFuturesProvider(DataProvider):
    """
    Downloads daily OHLCV for commodity futures from Yahoo Finance.
    Caches raw CSV data locally. Handles missing data via forward-fill.
    Outputs a MarketDataset with market_type="commodity".
    """

    def __init__(self, cfg: Dict[str, Any], seed: int = 42) -> None:
        super().__init__(cfg, seed)
        self.symbols: List[str] = cfg.get("symbols", DEFAULT_SYMBOLS)
        self.start: str = cfg.get("start", "2005-01-01")
        self.end: str = cfg.get("end", datetime.now(timezone.utc).strftime("%Y-%m-%d"))

        # Cache directory (relative to CWD if not absolute)
        raw_cache = Path(cfg.get("cache_dir", "data/cache/yahoo_futures"))
        self.cache_dir: Path = (
            raw_cache if raw_cache.is_absolute() else Path(os.getcwd()) / raw_cache
        )
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.min_valid_bars: int = cfg.get("min_valid_bars", 300)
        self.retry_count: int = cfg.get("retry_count", 3)
        self.request_delay: float = cfg.get("request_delay", 0.5)

    # ── Public ──────────────────────────────────────────────────────────────

    def load(self) -> MarketDataset:
        logger.info(
            f"[YahooFutures] Loading {len(self.symbols)} symbols "
            f"{self.start} → {self.end}"
        )

        raw_data: Dict[str, Dict[str, List]] = {}
        valid_symbols: List[str] = []

        for sym in self.symbols:
            data = self._fetch_symbol(sym)
            n = len(data["timestamps"]) if data else 0
            if data and n >= self.min_valid_bars:
                raw_data[sym] = data
                valid_symbols.append(sym)
                logger.info(f"  ✓ {sym:6s} ({SYMBOL_NAMES.get(sym, sym)}): {n} bars")
            else:
                logger.warning(
                    f"  ✗ {sym:6s}: only {n} bars — skipped (min={self.min_valid_bars})"
                )
            time.sleep(self.request_delay)

        if not valid_symbols:
            raise RuntimeError(
                "No valid commodity symbols loaded. "
                "Check network connectivity and Yahoo Finance API availability."
            )

        logger.info(f"[YahooFutures] Aligning {len(valid_symbols)} symbols...")
        timeline, aligned = self._align_timeline(raw_data, valid_symbols)

        logger.info(f"[YahooFutures] Computing features for {len(timeline)} bars...")
        features = self._build_features(aligned, valid_symbols, timeline)

        fp = hashlib.md5(
            f"{self.start}:{self.end}:{sorted(valid_symbols)}".encode()
        ).hexdigest()[:16]

        meta = {
            "start": self.start,
            "end": self.end,
            "bar_interval": "1d",
            "sector_map": {sym: SECTOR_MAP.get(sym, "other") for sym in valid_symbols},
            "symbol_names": {sym: SYMBOL_NAMES.get(sym, sym) for sym in valid_symbols},
            "n_bars": len(timeline),
            "n_symbols": len(valid_symbols),
        }

        logger.info(
            f"[YahooFutures] Dataset ready: {len(valid_symbols)} symbols × {len(timeline)} bars"
        )

        return MarketDataset(
            provider="yahoo_futures_v1",
            timeline=timeline,
            symbols=valid_symbols,
            perp_close={sym: aligned[sym]["close"] for sym in valid_symbols},
            spot_close=None,
            funding={},
            fingerprint=fp,
            market_type="commodity",
            features=features,
            meta=meta,
        )

    # ── Fetch & cache ────────────────────────────────────────────────────────

    def _fetch_symbol(self, symbol: str) -> Optional[Dict[str, List]]:
        cache_key = (
            f"{symbol.replace('=', '_').replace('/', '-')}"
            f"_{self.start}_{self.end}.json"
        )
        cache_path = self.cache_dir / cache_key

        if cache_path.exists():
            try:
                with open(cache_path) as f:
                    data = json.load(f)
                if data and data.get("timestamps"):
                    logger.debug(f"  {symbol}: loaded from cache ({cache_path.name})")
                    return data
            except Exception as e:
                logger.debug(f"  {symbol}: cache read failed ({e}), re-downloading")

        for attempt in range(self.retry_count):
            try:
                data = self._download_yahoo(symbol)
                if data and data.get("timestamps"):
                    with open(cache_path, "w") as f:
                        json.dump(data, f, separators=(",", ":"))
                    return data
                logger.warning(f"  {symbol}: empty response on attempt {attempt + 1}")
            except urllib.error.HTTPError as e:
                if e.code == 404:
                    logger.warning(f"  {symbol}: 404 Not Found — ticker may not exist")
                    return None
                logger.warning(f"  {symbol}: HTTP {e.code} on attempt {attempt + 1}")
            except Exception as e:
                logger.warning(f"  {symbol}: attempt {attempt + 1} failed: {e}")

            if attempt < self.retry_count - 1:
                time.sleep(2.0 * (attempt + 1))

        return None

    def _download_yahoo(self, symbol: str) -> Optional[Dict[str, List]]:
        start_ts = int(
            datetime.strptime(self.start, "%Y-%m-%d")
            .replace(tzinfo=timezone.utc)
            .timestamp()
        )
        end_ts = (
            int(
                datetime.strptime(self.end, "%Y-%m-%d")
                .replace(tzinfo=timezone.utc)
                .timestamp()
            )
            + 86400
        )

        url = (
            f"https://query1.finance.yahoo.com/v7/finance/download/{symbol}"
            f"?period1={start_ts}&period2={end_ts}"
            f"&interval=1d&events=history&includeAdjustedClose=true"
        )

        req = urllib.request.Request(url)
        req.add_header(
            "User-Agent",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36",
        )
        req.add_header("Accept", "text/csv,application/csv,*/*")
        req.add_header("Accept-Language", "en-US,en;q=0.9")

        with urllib.request.urlopen(req, timeout=30) as resp:
            raw = resp.read().decode("utf-8", errors="replace")

        return self._parse_csv(raw)

    def _parse_csv(self, raw: str) -> Optional[Dict[str, List]]:
        result: Dict[str, List] = {
            "dates": [],
            "timestamps": [],
            "open": [],
            "high": [],
            "low": [],
            "close": [],
            "volume": [],
        }

        lines = raw.strip().splitlines()
        if len(lines) < 2:
            return None

        reader = csv.DictReader(lines)
        for row in reader:
            try:
                date_str = (row.get("Date") or "").strip()
                if not date_str or date_str.lower() in ("null", "nan", ""):
                    continue

                # Use Adj Close if available (accounts for splits/adjustments)
                raw_close = row.get("Adj Close") or row.get("Close") or ""
                if not raw_close or raw_close.lower() in ("null", "nan", ""):
                    continue

                close = float(raw_close)
                if close != close or close <= 0:  # NaN or negative price
                    continue

                ts = int(
                    datetime.strptime(date_str, "%Y-%m-%d")
                    .replace(tzinfo=timezone.utc)
                    .timestamp()
                )

                def _safe_float(v: str, fallback: float = close) -> float:
                    try:
                        f = float(v or fallback)
                        return f if f == f and f > 0 else fallback
                    except (ValueError, TypeError):
                        return fallback

                result["dates"].append(date_str)
                result["timestamps"].append(ts)
                result["open"].append(_safe_float(row.get("Open", ""), close))
                result["high"].append(_safe_float(row.get("High", ""), close))
                result["low"].append(_safe_float(row.get("Low", ""), close))
                result["close"].append(close)
                result["volume"].append(
                    max(0.0, _safe_float(row.get("Volume", ""), 0.0))
                )

            except (ValueError, TypeError, KeyError):
                continue

        if not result["dates"]:
            return None
        return result

    # ── Timeline alignment (forward-fill) ───────────────────────────────────

    def _align_timeline(
        self,
        raw_data: Dict[str, Dict[str, List]],
        symbols: List[str],
    ) -> Tuple[List[int], Dict[str, Dict[str, List[float]]]]:
        """
        Align all symbols to a shared daily timeline (union of all trading days).
        Forward-fill prices on days when a symbol doesn't trade.
        """
        all_ts = sorted(
            {ts for sym in symbols for ts in raw_data[sym]["timestamps"]}
        )

        aligned: Dict[str, Dict[str, List[float]]] = {}

        for sym in symbols:
            src = raw_data[sym]
            ts_to_i = {ts: i for i, ts in enumerate(src["timestamps"])}

            last: Dict[str, float] = {
                "open": float("nan"),
                "high": float("nan"),
                "low": float("nan"),
                "close": float("nan"),
                "volume": 0.0,
            }
            sym_out: Dict[str, List[float]] = {k: [] for k in last}

            for ts in all_ts:
                if ts in ts_to_i:
                    i = ts_to_i[ts]
                    for k in last:
                        v = src[k][i]
                        if v == v:  # not NaN
                            last[k] = v
                for k in last:
                    sym_out[k].append(last[k])

            aligned[sym] = sym_out

        return all_ts, aligned

    # ── Feature pre-computation ──────────────────────────────────────────────

    def _build_features(
        self,
        aligned: Dict[str, Dict[str, List[float]]],
        symbols: List[str],
        timeline: List[int],
    ) -> Dict[str, Any]:
        """Delegate to features.py for pre-computation."""
        from nexus_quant.projects.commodity_cta.features import compute_features

        return compute_features(aligned, symbols, timeline)
