"""
Yahoo Finance Futures Data Provider
====================================
Fetches daily OHLCV for commodity futures using yfinance (primary).
Falls back to Stooq.com, then raw Yahoo v8/v7 API.
Caches locally to avoid repeated downloads.

Supported tickers (continuous front-month):
  Energy   : CL=F (WTI Oil), NG=F (Natural Gas), BZ=F (Brent)
  Metals   : GC=F (Gold), SI=F (Silver), HG=F (Copper), PL=F (Platinum)
  Grains   : ZW=F (Wheat), ZC=F (Corn), ZS=F (Soybeans)
  Softs    : KC=F (Coffee), SB=F (Sugar), CT=F (Cotton)
  FX       : EURUSD=X, GBPUSD=X, AUDUSD=X, JPY=X (USD/JPY)
  Bonds    : TLT (20yr Treasury ETF), IEF (10yr Treasury ETF)
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

try:
    import yfinance as _yf
    _HAS_YFINANCE = True
except ImportError:
    _HAS_YFINANCE = False

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

# Phase 139: Diversified multi-asset universe (Commodities + FX + Bonds)
# Key insight: FX and bonds provide crucial diversification during commodity bear markets
# (e.g. 2015-2019: commodity bear, bond bull, USD trends)
DIVERSIFIED_SYMBOLS = [
    # Commodities (top 8 by liquidity — reduced to avoid noise)
    "CL=F",      # WTI Crude Oil
    "NG=F",      # Natural Gas
    "GC=F",      # Gold
    "SI=F",      # Silver
    "HG=F",      # Copper
    "ZW=F",      # Wheat
    "ZC=F",      # Corn
    "ZS=F",      # Soybeans
    # FX majors (trend well due to interest rate differentials & global flows)
    "EURUSD=X",  # Euro / US Dollar
    "GBPUSD=X",  # British Pound / US Dollar
    "AUDUSD=X",  # Australian Dollar / US Dollar (commodity currency)
    "JPY=X",     # US Dollar / Japanese Yen
    # Government Bonds (key diversifier — opposite to commodity in many regimes)
    "TLT",       # iShares 20+ Year Treasury Bond ETF (proxy for 30yr bond futures)
    "IEF",       # iShares 7-10 Year Treasury Bond ETF (proxy for 10yr note futures)
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
    # FX
    "EURUSD=X": "fx",
    "GBPUSD=X": "fx",
    "AUDUSD=X": "fx",
    "JPY=X": "fx",
    # Bonds
    "TLT": "bonds",
    "IEF": "bonds",
    "ZB=F": "bonds",
    "ZN=F": "bonds",
}

# Stooq.com futures tickers (alternative free data source)
STOOQ_TICKERS: Dict[str, str] = {
    "CL=F": "cl.f",       # WTI Crude Oil
    "NG=F": "ng.f",       # Natural Gas
    "BZ=F": "co.f",       # Brent Crude (stooq uses co.f)
    "GC=F": "gc.f",       # Gold
    "SI=F": "si.f",       # Silver
    "HG=F": "hg.f",       # Copper
    "PL=F": "pl.f",       # Platinum
    "ZW=F": "w.f",        # Wheat
    "ZC=F": "c.f",        # Corn
    "ZS=F": "s.f",        # Soybeans
    "KC=F": "kc.f",       # Coffee
    "SB=F": "sb.f",       # Sugar #11
    "CT=F": "ct.f",       # Cotton
    "LE=F": "le.f",       # Live Cattle
    "HE=F": "he.f",       # Lean Hogs
    # FX (Stooq uses lowercase pair names)
    "EURUSD=X": "eurusd", # EUR/USD
    "GBPUSD=X": "gbpusd", # GBP/USD
    "AUDUSD=X": "audusd", # AUD/USD
    "JPY=X": "usdjpy",    # USD/JPY
    # Bond ETFs (Stooq uses .us suffix for US ETFs)
    "TLT": "tlt.us",
    "IEF": "ief.us",
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
    # FX
    "EURUSD=X": "Euro/USD",
    "GBPUSD=X": "GBP/USD",
    "AUDUSD=X": "AUD/USD",
    "JPY=X": "USD/JPY",
    # Bonds
    "TLT": "20yr Treasury Bond ETF",
    "IEF": "10yr Treasury Note ETF",
    "ZB=F": "30yr T-Bond Futures",
    "ZN=F": "10yr T-Note Futures",
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

        # Trim timeline to start where ALL symbols have valid (non-NaN) prices.
        # This prevents NaN × weight = NaN contaminating the backtest equity curve.
        trim_start = 0
        for sym in valid_symbols:
            close_arr = aligned[sym]["close"]
            first_valid = next(
                (i for i, v in enumerate(close_arr) if v == v and v > 0), len(close_arr)
            )
            trim_start = max(trim_start, first_valid)

        if trim_start > 0:
            logger.info(
                f"[YahooFutures] Trimming {trim_start} bars (waiting for all symbols to start)"
            )
            timeline = timeline[trim_start:]
            for sym in valid_symbols:
                for k in aligned[sym]:
                    aligned[sym][k] = aligned[sym][k][trim_start:]

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
        """
        Try in order:
        1. yfinance library (robust, handles sessions/cookies automatically)
        2. Stooq.com (free, no auth, no rate limit)
        3. Yahoo Finance v8 JSON API
        4. Yahoo Finance v7 CSV (last resort)
        """
        # 1. yfinance (primary: best rate-limit handling)
        if _HAS_YFINANCE:
            data = self._download_yfinance(symbol)
            if data and data.get("timestamps"):
                logger.debug(f"  {symbol}: yfinance OK ({len(data['timestamps'])} bars)")
                return data

        # 2. Stooq (secondary: free and reliable)
        stooq_ticker = STOOQ_TICKERS.get(symbol)
        if stooq_ticker:
            data = self._download_stooq(symbol, stooq_ticker)
            if data and data.get("timestamps"):
                logger.debug(f"  {symbol}: Stooq OK ({len(data['timestamps'])} bars)")
                return data

        # 3. Yahoo v8 JSON
        data = self._download_yahoo_v8(symbol)
        if data and data.get("timestamps"):
            return data

        # 4. Yahoo v7 CSV
        logger.debug(f"  {symbol}: trying Yahoo v7 CSV fallback")
        return self._download_yahoo_v7_csv(symbol)

    def _download_yfinance(self, symbol: str) -> Optional[Dict[str, List]]:
        """Download via yfinance library — handles sessions, cookies, and rate limits."""
        try:
            import yfinance as yf
            ticker = yf.Ticker(symbol)
            df = ticker.history(start=self.start, end=self.end, auto_adjust=True)

            if df is None or len(df) == 0:
                logger.debug(f"  {symbol}: yfinance returned empty DataFrame")
                return None

            # Normalize column names (yfinance may return multi-index or simple)
            if hasattr(df.columns, "get_level_values"):
                try:
                    df.columns = df.columns.get_level_values(0)
                except Exception:
                    pass

            col_map = {c.lower(): c for c in df.columns}
            close_col = col_map.get("close")
            if not close_col or close_col not in df.columns:
                logger.debug(f"  {symbol}: yfinance DataFrame missing Close column")
                return None

            result: Dict[str, List] = {
                "dates": [], "timestamps": [], "open": [],
                "high": [], "low": [], "close": [], "volume": [],
            }

            for idx, row in df.iterrows():
                try:
                    c = float(row[close_col])
                    if c != c or c <= 0:
                        continue
                    date_str = str(idx)[:10]  # YYYY-MM-DD
                    ts = int(datetime.strptime(date_str, "%Y-%m-%d")
                             .replace(tzinfo=timezone.utc).timestamp())

                    def _g(col_name: str, fb: float = c) -> float:
                        mapped = col_map.get(col_name)
                        if not mapped or mapped not in row.index:
                            return fb
                        v = row[mapped]
                        try:
                            f = float(v)
                            return f if f == f and f > 0 else fb
                        except (TypeError, ValueError):
                            return fb

                    result["dates"].append(date_str)
                    result["timestamps"].append(ts)
                    result["open"].append(_g("open"))
                    result["high"].append(_g("high"))
                    result["low"].append(_g("low"))
                    result["close"].append(c)
                    result["volume"].append(max(0.0, _g("volume", 0.0)))
                except Exception:
                    continue

            return result if result["timestamps"] else None

        except Exception as e:
            logger.debug(f"  {symbol}: yfinance error: {e}")
            return None

    def _download_stooq(self, symbol: str, stooq_ticker: str) -> Optional[Dict[str, List]]:
        """Download from Stooq.com — free commodity futures data, no auth required."""
        d1 = self.start.replace("-", "")
        d2 = self.end.replace("-", "")
        url = f"https://stooq.com/q/d/l/?s={stooq_ticker}&d1={d1}&d2={d2}&i=d"

        req = urllib.request.Request(url)
        req.add_header(
            "User-Agent",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/122.0.0.0 Safari/537.36",
        )
        req.add_header("Accept", "text/csv,*/*")
        req.add_header("Referer", "https://stooq.com/")

        with urllib.request.urlopen(req, timeout=30) as resp:
            raw = resp.read().decode("utf-8", errors="replace")

        # Stooq CSV: Date,Open,High,Low,Close,Volume
        return self._parse_stooq_csv(raw)

    def _parse_stooq_csv(self, raw: str) -> Optional[Dict[str, List]]:
        """Parse Stooq CSV. Format: Date,Open,High,Low,Close,Volume (newest first)."""
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

        # Stooq returns newest-first; we'll sort afterwards
        rows = []
        reader = csv.DictReader(lines)
        for row in reader:
            try:
                date_str = (row.get("Date") or "").strip()
                if not date_str:
                    continue
                c_raw = row.get("Close", "")
                if not c_raw:
                    continue
                c = float(c_raw)
                if c <= 0 or c != c:
                    continue
                ts = int(
                    datetime.strptime(date_str, "%Y-%m-%d")
                    .replace(tzinfo=timezone.utc)
                    .timestamp()
                )

                def _sf(k: str, fb: float = c) -> float:
                    try:
                        v = float(row.get(k, fb) or fb)
                        return v if v > 0 and v == v else fb
                    except (ValueError, TypeError):
                        return fb

                rows.append({
                    "date": date_str,
                    "ts": ts,
                    "o": _sf("Open"),
                    "h": _sf("High"),
                    "l": _sf("Low"),
                    "c": c,
                    "v": max(0.0, _sf("Volume", 0.0)),
                })
            except (ValueError, TypeError, KeyError):
                continue

        if not rows:
            return None

        # Sort chronologically (Stooq sends newest-first)
        rows.sort(key=lambda x: x["ts"])

        for r in rows:
            result["dates"].append(r["date"])
            result["timestamps"].append(r["ts"])
            result["open"].append(r["o"])
            result["high"].append(r["h"])
            result["low"].append(r["l"])
            result["close"].append(r["c"])
            result["volume"].append(r["v"])

        return result if result["timestamps"] else None

    def _download_yahoo_v8(self, symbol: str) -> Optional[Dict[str, List]]:
        """Yahoo Finance v8 chart API — returns JSON, often works without crumb."""
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

        for host in ("query2.finance.yahoo.com", "query1.finance.yahoo.com"):
            url = (
                f"https://{host}/v8/finance/chart/{symbol}"
                f"?period1={start_ts}&period2={end_ts}"
                f"&interval=1d&includePrePost=false"
            )
            try:
                req = urllib.request.Request(url)
                req.add_header(
                    "User-Agent",
                    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/122.0.0.0 Safari/537.36",
                )
                req.add_header("Accept", "application/json,*/*")
                req.add_header("Accept-Language", "en-US,en;q=0.9")
                req.add_header("Referer", "https://finance.yahoo.com/")

                with urllib.request.urlopen(req, timeout=30) as resp:
                    raw = resp.read().decode("utf-8", errors="replace")

                data = self._parse_v8_json(raw, symbol)
                if data and data.get("timestamps"):
                    return data

            except urllib.error.HTTPError as e:
                if e.code == 429:
                    raise  # propagate rate-limit to retry logic
                logger.debug(f"  {symbol}: v8/{host} HTTP {e.code}")
            except Exception as e:
                logger.debug(f"  {symbol}: v8/{host} error: {e}")

        return None

    def _parse_v8_json(self, raw: str, symbol: str) -> Optional[Dict[str, List]]:
        """Parse Yahoo Finance v8 chart JSON response."""
        import json as _json

        try:
            obj = _json.loads(raw)
            result_list = obj["chart"]["result"]
            if not result_list:
                return None
            chart = result_list[0]

            timestamps = chart.get("timestamp") or []
            if not timestamps:
                return None

            quotes = chart["indicators"]["quote"][0]
            adj_list = chart.get("indicators", {}).get("adjclose", [{}])
            adjclose_arr = adj_list[0].get("adjclose", []) if adj_list else []

            result: Dict[str, List] = {
                "dates": [],
                "timestamps": [],
                "open": [],
                "high": [],
                "low": [],
                "close": [],
                "volume": [],
            }

            raw_o = quotes.get("open", [])
            raw_h = quotes.get("high", [])
            raw_l = quotes.get("low", [])
            raw_c = quotes.get("close", [])
            raw_v = quotes.get("volume", [])

            for i, ts in enumerate(timestamps):
                if ts is None:
                    continue
                c = (adjclose_arr[i] if i < len(adjclose_arr) and adjclose_arr[i] else None
                     or (raw_c[i] if i < len(raw_c) else None))
                if c is None or c != c or c <= 0:
                    continue

                def _sf(arr: list, idx: int, fb: float = c) -> float:
                    v = arr[idx] if idx < len(arr) else None
                    return v if (v is not None and v == v and v > 0) else fb

                result["dates"].append(datetime.utcfromtimestamp(ts).strftime("%Y-%m-%d"))
                result["timestamps"].append(int(ts))
                result["open"].append(_sf(raw_o, i))
                result["high"].append(_sf(raw_h, i))
                result["low"].append(_sf(raw_l, i))
                result["close"].append(c)
                result["volume"].append(max(0.0, raw_v[i] if i < len(raw_v) and raw_v[i] else 0.0))

            return result if result["timestamps"] else None

        except (KeyError, IndexError, TypeError, _json.JSONDecodeError) as e:
            logger.debug(f"  {symbol}: v8 JSON parse error: {e}")
            return None

    def _download_yahoo_v7_csv(self, symbol: str) -> Optional[Dict[str, List]]:
        """Fallback: Yahoo Finance v7 download CSV endpoint."""
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
            "Chrome/122.0.0.0 Safari/537.36",
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
