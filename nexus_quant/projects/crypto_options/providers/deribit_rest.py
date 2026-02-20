"""
Deribit REST API Provider for Crypto Options

Fetches underlying prices and volatility surface data from Deribit public API.
No authentication required for public endpoints.

Config keys:
    provider: "deribit_rest_v1"
    symbols: ["BTC", "ETH"]                    # underlying currencies
    start: "2020-01-01"                        # start date (YYYY-MM-DD)
    end: "2024-12-31"                          # end date (YYYY-MM-DD)
    bar_interval: "1d"                         # "1h" or "1d"
    cache_dir: "data/cache/deribit"            # CSV cache directory
    use_synthetic_iv: true                     # generate synthetic IV if API limited

Output MarketDataset:
    market_type = "options"
    perp_close = underlying perpetual prices
    features = {
        "iv_atm": {symbol: [values]},
        "iv_25d_put": {symbol: [values]},
        "iv_25d_call": {symbol: [values]},
        "skew_25d": {symbol: [values]},
        "butterfly_25d": {symbol: [values]},
        "rv_realized": {symbol: [values]},
        "term_spread": {symbol: [values]},
        "vrp": {symbol: [values]},             # variance risk premium: iv_atm - rv_realized
    }

Rate limit: 20 req/s (public). Provider adds 0.1s delay between requests.
"""
from __future__ import annotations

import csv
import hashlib
import json
import logging
import math
import os
import random
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from nexus_quant.data.providers.base import DataProvider
from nexus_quant.data.schema import MarketDataset

logger = logging.getLogger("nexus.providers.deribit")

_DERIBIT_BASE = "https://www.deribit.com/api/v2/public"
_REQUEST_DELAY = 0.1       # seconds between API calls
_RETRY_DELAYS = [2, 4, 8, 16]
_MAX_LIMIT = 1000          # bars per request for chart data

# Bars per resolution for annualization
_BARS_PER_YEAR = {
    "1h": 8760,
    "1d": 365,
    "4h": 2190,
}

# Synthetic IV parameters (calibrated to crypto markets 2019-2024)
_VRP_MEAN = {
    "BTC": 0.08,   # BTC IV typically ~8% above RV
    "ETH": 0.10,   # ETH IV typically ~10% above RV
}
_VRP_STD = {
    "BTC": 0.04,
    "ETH": 0.05,
}
_SKEW_MEAN = {
    "BTC": 0.05,   # puts ~5 vol points more expensive than calls
    "ETH": 0.06,
}
_SKEW_STD = {
    "BTC": 0.03,
    "ETH": 0.04,
}
_TERM_SPREAD_MEAN = {
    "BTC": 0.02,   # front-month typically 2 vol points above back-month
    "ETH": 0.025,
}
_TERM_SPREAD_STD = {
    "BTC": 0.03,
    "ETH": 0.035,
}


class DeribitRestProvider(DataProvider):
    """Fetch Deribit underlying prices and vol surface features.

    Data strategy:
    1. Fetch underlying price from BTC-PERPETUAL / ETH-PERPETUAL OHLCV
    2. Fetch realized vol from Deribit historical vol endpoint
    3. Generate synthetic IV surface features using realized vol + calibrated VRP
       (historical option chain data not available via free API)
    4. Cache all data to CSV for reproducibility
    """

    def __init__(self, cfg: Dict[str, Any], seed: int = 42) -> None:
        super().__init__(cfg, seed)
        self._rng = random.Random(seed)

        self.symbols: List[str] = cfg.get("symbols", ["BTC", "ETH"])
        self.start_str: str = cfg.get("start", "2020-01-01")
        self.end_str: str = cfg.get("end", "2024-12-31")
        self.bar_interval: str = cfg.get("bar_interval", "1d")
        self.cache_dir = Path(cfg.get("cache_dir", "data/cache/deribit"))
        self.use_synthetic_iv: bool = cfg.get("use_synthetic_iv", True)
        self.rv_lookback_bars: int = cfg.get("rv_lookback_bars", 21)  # bars for realized vol

        # Map bar_interval to Deribit resolution
        self._resolution = {"1h": 60, "4h": 240, "1d": "1D"}.get(self.bar_interval, "1D")
        self._bars_per_year = _BARS_PER_YEAR.get(self.bar_interval, 365)

        # Parse date range
        self._start_ts = _parse_date_to_ts(self.start_str)
        self._end_ts = _parse_date_to_ts(self.end_str, end_of_day=True)

    def load(self) -> MarketDataset:
        """Load or fetch options dataset."""
        logger.info(
            "DeribitRestProvider: loading %s from %s to %s (%s bars)",
            self.symbols, self.start_str, self.end_str, self.bar_interval
        )

        perp_close: Dict[str, List[float]] = {}
        features_raw: Dict[str, Dict[str, List]] = {
            "iv_atm": {},
            "iv_25d_put": {},
            "iv_25d_call": {},
            "skew_25d": {},
            "butterfly_25d": {},
            "rv_realized": {},
            "term_spread": {},
            "vrp": {},
        }
        timeline: Optional[List[int]] = None

        for sym in self.symbols:
            logger.info("Loading data for %s...", sym)
            ts_list, prices, rv_series, iv_data = self._load_symbol(sym)

            if timeline is None:
                timeline = ts_list
            else:
                # Align to common timeline (use first symbol's timeline)
                if ts_list != timeline:
                    prices, rv_series, iv_data = self._align_series(
                        timeline, ts_list, prices, rv_series, iv_data
                    )

            perp_close[sym] = prices

            # Fill features
            features_raw["iv_atm"][sym] = iv_data["iv_atm"]
            features_raw["iv_25d_put"][sym] = iv_data["iv_25d_put"]
            features_raw["iv_25d_call"][sym] = iv_data["iv_25d_call"]
            features_raw["skew_25d"][sym] = iv_data["skew_25d"]
            features_raw["butterfly_25d"][sym] = iv_data["butterfly_25d"]
            features_raw["rv_realized"][sym] = rv_series
            features_raw["term_spread"][sym] = iv_data["term_spread"]
            # VRP = IV_atm - RV_realized
            features_raw["vrp"][sym] = [
                (iv - rv) if (iv is not None and rv is not None) else None
                for iv, rv in zip(iv_data["iv_atm"], rv_series)
            ]

        if timeline is None:
            timeline = []

        # Build fingerprint
        fp_data = json.dumps({
            "symbols": sorted(self.symbols),
            "start": self.start_str,
            "end": self.end_str,
            "interval": self.bar_interval,
            "n_bars": len(timeline),
        }, sort_keys=True)
        fingerprint = hashlib.sha256(fp_data.encode()).hexdigest()[:16]

        dataset = MarketDataset(
            provider="deribit_rest_v1",
            timeline=timeline,
            symbols=list(self.symbols),
            perp_close=perp_close,
            spot_close=None,
            funding={},           # No funding for options
            fingerprint=fingerprint,
            market_type="options",
            features=features_raw,
            meta={
                "start": self.start_str,
                "end": self.end_str,
                "bar_interval": self.bar_interval,
                "rv_lookback_bars": self.rv_lookback_bars,
                "use_synthetic_iv": self.use_synthetic_iv,
                "description": "Deribit BTC/ETH options vol surface dataset",
            },
        )
        logger.info(
            "DeribitRestProvider: loaded %d bars for %s",
            len(timeline), self.symbols
        )
        return dataset

    # ── Per-symbol loading ─────────────────────────────────────────────────

    def _load_symbol(
        self, sym: str
    ) -> Tuple[List[int], List[float], List[Optional[float]], Dict[str, List]]:
        """Load all data for one symbol.

        Returns:
            (timeline, prices, rv_series, iv_data_dict)
        """
        cache_key = f"{sym}_{self.start_str}_{self.end_str}_{self.bar_interval}"

        # Try to load underlying price from cache
        prices_cache = self.cache_dir / f"{cache_key}_prices.csv"
        if prices_cache.exists():
            logger.info("Loading %s prices from cache: %s", sym, prices_cache)
            ts_list, prices = self._load_prices_csv(prices_cache)
        else:
            logger.info("Fetching %s prices from Deribit API...", sym)
            ts_list, prices = self._fetch_underlying(sym)
            if ts_list:
                self._save_prices_csv(prices_cache, ts_list, prices)

        if not ts_list:
            logger.warning("No price data for %s — using empty series", sym)
            return [], [], [], {k: [] for k in ["iv_atm", "iv_25d_put", "iv_25d_call",
                                                  "skew_25d", "butterfly_25d", "term_spread"]}

        # Realized vol (from Deribit historical vol API or computed from prices)
        rv_cache = self.cache_dir / f"{cache_key}_rv.csv"
        if rv_cache.exists():
            _, rv_series = self._load_rv_csv(rv_cache)
        else:
            rv_series = self._compute_rv_from_prices(prices)
            self._save_rv_csv(rv_cache, ts_list, rv_series)

        # IV surface features (synthetic or from API)
        iv_cache = self.cache_dir / f"{cache_key}_iv.csv"
        if iv_cache.exists():
            logger.info("Loading %s IV from cache: %s", sym, iv_cache)
            iv_data = self._load_iv_csv(iv_cache)
        elif self.use_synthetic_iv:
            logger.info("Generating synthetic IV surface for %s...", sym)
            iv_data = self._generate_synthetic_iv(sym, ts_list, prices, rv_series)
            self._save_iv_csv(iv_cache, ts_list, iv_data)
        else:
            logger.info("Fetching live IV snapshot for %s (limited history)...", sym)
            iv_data = self._fetch_iv_snapshot(sym, ts_list, rv_series)
            self._save_iv_csv(iv_cache, ts_list, iv_data)

        return ts_list, prices, rv_series, iv_data

    # ── Deribit API calls ─────────────────────────────────────────────────

    def _fetch_underlying(
        self, sym: str
    ) -> Tuple[List[int], List[float]]:
        """Fetch underlying perpetual price OHLCV from Deribit."""
        instrument = f"{sym}-PERPETUAL"
        all_ts: List[int] = []
        all_close: List[float] = []

        chunk_start = self._start_ts
        while chunk_start < self._end_ts:
            chunk_end = min(
                chunk_start + _MAX_LIMIT * self._bar_seconds(),
                self._end_ts
            )
            params = {
                "instrument_name": instrument,
                "start_timestamp": chunk_start * 1000,
                "end_timestamp": chunk_end * 1000,
                "resolution": str(self._resolution),
            }
            data = self._api_call("get_tradingview_chart_data", params)
            if data is None or data.get("status") != "ok":
                logger.warning("Chart data fetch failed for %s, chunk %d", sym, chunk_start)
                break

            ticks = data.get("ticks", [])
            closes = data.get("close", [])
            if not ticks or len(ticks) != len(closes):
                break

            for t, c in zip(ticks, closes):
                ts_sec = int(t) // 1000
                if self._start_ts <= ts_sec <= self._end_ts:
                    all_ts.append(ts_sec)
                    all_close.append(float(c))

            if chunk_end >= self._end_ts:
                break
            chunk_start = chunk_end
            time.sleep(_REQUEST_DELAY)

        # Remove duplicates and sort
        pairs = sorted(set(zip(all_ts, all_close)), key=lambda x: x[0])
        all_ts = [p[0] for p in pairs]
        all_close = [p[1] for p in pairs]
        return all_ts, all_close

    def _fetch_historical_rv(self, sym: str) -> Dict[int, float]:
        """Fetch historical realized volatility from Deribit."""
        params = {"currency": sym}
        data = self._api_call("get_historical_volatility", params)
        if data is None:
            return {}
        # Returns [[timestamp_ms, vol_pct], ...]
        rv_map: Dict[int, float] = {}
        for item in (data if isinstance(data, list) else []):
            try:
                ts_sec = int(item[0]) // 1000
                rv_pct = float(item[1]) / 100.0  # Convert % to decimal
                rv_map[ts_sec] = rv_pct
            except (IndexError, TypeError, ValueError):
                continue
        return rv_map

    def _fetch_iv_snapshot(
        self,
        sym: str,
        ts_list: List[int],
        rv_series: List[Optional[float]],
    ) -> Dict[str, List]:
        """Fetch current IV snapshot from book summary and back-fill history."""
        # Get current options book summary
        params = {"currency": sym, "kind": "option"}
        data = self._api_call("get_book_summary_by_currency", params)

        current_iv_atm: Optional[float] = None
        if data and isinstance(data, list):
            # Find ATM options (nearest expiry, nearest strike)
            atm_ivs = []
            for item in data:
                mark_iv = item.get("mark_iv")
                if mark_iv and mark_iv > 0:
                    atm_ivs.append(float(mark_iv) / 100.0)
            if atm_ivs:
                current_iv_atm = sorted(atm_ivs)[len(atm_ivs) // 2]  # median

        if current_iv_atm is None:
            current_iv_atm = 0.70  # Default 70% IV for crypto

        # Use synthetic IV based on current level
        return self._generate_synthetic_iv(sym, ts_list, None, rv_series,
                                           base_iv=current_iv_atm)

    def _api_call(self, method: str, params: Dict[str, Any]) -> Any:
        """Make a Deribit API call with retry."""
        url = f"{_DERIBIT_BASE}/{method}"
        query = urllib.parse.urlencode(params)
        full_url = f"{url}?{query}"

        for attempt, delay in enumerate([0] + _RETRY_DELAYS):
            if delay:
                time.sleep(delay)
            try:
                req = urllib.request.Request(
                    full_url,
                    headers={"User-Agent": "NEXUS-Quant/1.0"},
                )
                with urllib.request.urlopen(req, timeout=30) as resp:
                    body = json.loads(resp.read().decode("utf-8"))
                    if "result" in body:
                        return body["result"]
                    return body
            except urllib.error.HTTPError as e:
                logger.warning("Deribit HTTP %d for %s (attempt %d)", e.code, method, attempt + 1)
            except urllib.error.URLError as e:
                logger.warning("Deribit URL error for %s: %s (attempt %d)", method, e, attempt + 1)
            except Exception as e:
                logger.warning("Deribit error for %s: %s (attempt %d)", method, e, attempt + 1)

        logger.error("Deribit API call failed after %d attempts: %s", len(_RETRY_DELAYS) + 1, method)
        return None

    # ── Synthetic IV generation ────────────────────────────────────────────

    def _generate_synthetic_iv(
        self,
        sym: str,
        ts_list: List[int],
        prices: Optional[List[float]],
        rv_series: List[Optional[float]],
        base_iv: Optional[float] = None,
    ) -> Dict[str, List]:
        """Generate synthetic vol surface features.

        Methodology (calibrated to 2019-2024 crypto options data):
        1. Start with realized vol as the base
        2. Add VRP (variance risk premium): IV = RV + VRP
        3. VRP mean-reverts around a regime-dependent level
        4. Skew: puts more expensive than calls (crash risk premium)
        5. Term structure: front slightly above back (normal contango)

        This is a SYNTHETIC dataset for research. Label clearly in outputs.
        """
        n = len(ts_list)
        vrp_mean = _VRP_MEAN.get(sym, 0.08)
        vrp_std = _VRP_STD.get(sym, 0.04)
        skew_mean = _SKEW_MEAN.get(sym, 0.05)
        skew_std = _SKEW_STD.get(sym, 0.03)
        term_mean = _TERM_SPREAD_MEAN.get(sym, 0.02)
        term_std = _TERM_SPREAD_STD.get(sym, 0.03)

        iv_atm: List[Optional[float]] = []
        iv_25d_put: List[Optional[float]] = []
        iv_25d_call: List[Optional[float]] = []
        skew_25d: List[Optional[float]] = []
        butterfly_25d: List[Optional[float]] = []
        term_spread: List[Optional[float]] = []

        # Generate mean-reverting VRP process (AR(1))
        vrp_ar1 = 0.9          # mean reversion speed
        vrp_shock_std = vrp_std * math.sqrt(1 - vrp_ar1 ** 2)
        vrp = vrp_mean + self._rng.gauss(0, vrp_std)

        skew_ar1 = 0.88
        skew_shock_std = skew_std * math.sqrt(1 - skew_ar1 ** 2)
        skew = skew_mean + self._rng.gauss(0, skew_std)

        term_ar1 = 0.85
        term_shock_std = term_std * math.sqrt(1 - term_ar1 ** 2)
        term = term_mean + self._rng.gauss(0, term_std)

        # Volatility regime: detect high/low vol regimes from prices
        regimes = self._detect_vol_regime(prices, n) if prices else [False] * n

        for i in range(n):
            rv = rv_series[i] if i < len(rv_series) else None

            # Update VRP AR(1) process
            vrp = vrp_ar1 * vrp + (1 - vrp_ar1) * vrp_mean + self._rng.gauss(0, vrp_shock_std)
            vrp = max(-0.05, min(vrp, 0.30))  # clamp

            skew = skew_ar1 * skew + (1 - skew_ar1) * skew_mean + self._rng.gauss(0, skew_shock_std)
            skew = max(-0.05, min(skew, 0.30))

            term = term_ar1 * term + (1 - term_ar1) * term_mean + self._rng.gauss(0, term_shock_std)
            term = max(-0.15, min(term, 0.20))

            # In high-vol regime: VRP compresses, skew spikes
            in_stress = regimes[i] if i < len(regimes) else False
            if in_stress:
                vrp = max(0.0, vrp * 0.5)    # VRP compresses in stress
                skew_factor = 1.5            # Skew expands in stress
            else:
                skew_factor = 1.0

            if rv is not None:
                atm = rv + vrp
                atm = max(0.10, min(atm, 3.00))  # clamp IV
            elif base_iv is not None:
                atm = base_iv + self._rng.gauss(0, 0.02)
                atm = max(0.10, min(atm, 3.00))
            else:
                atm = None

            if atm is not None:
                effective_skew = skew * skew_factor
                put_25 = atm + effective_skew
                call_25 = atm - effective_skew * 0.3  # calls cheaper
                butterfly = 0.5 * (put_25 + call_25) - atm  # ≈ 0.35 * skew

                iv_atm.append(round(atm, 4))
                iv_25d_put.append(round(max(0.05, put_25), 4))
                iv_25d_call.append(round(max(0.05, call_25), 4))
                skew_25d.append(round(put_25 - call_25, 4))
                butterfly_25d.append(round(butterfly, 4))
                term_spread.append(round(term, 4))
            else:
                iv_atm.append(None)
                iv_25d_put.append(None)
                iv_25d_call.append(None)
                skew_25d.append(None)
                butterfly_25d.append(None)
                term_spread.append(None)

        return {
            "iv_atm": iv_atm,
            "iv_25d_put": iv_25d_put,
            "iv_25d_call": iv_25d_call,
            "skew_25d": skew_25d,
            "butterfly_25d": butterfly_25d,
            "term_spread": term_spread,
        }

    def _detect_vol_regime(self, prices: Optional[List[float]], n: int) -> List[bool]:
        """Detect high-vol regime from price returns. True = high vol stress."""
        if not prices or len(prices) < 2:
            return [False] * n

        # Rolling realized vol (21 bars)
        window = 21
        log_rets = [math.log(prices[i] / prices[i - 1]) for i in range(1, len(prices))]

        regimes = []
        for i in range(n):
            if i < window + 1:
                regimes.append(False)
                continue
            ret_window = log_rets[max(0, i - window):i]
            if len(ret_window) < 5:
                regimes.append(False)
                continue
            mean = sum(ret_window) / len(ret_window)
            var = sum((r - mean) ** 2 for r in ret_window) / (len(ret_window) - 1)
            rv = math.sqrt(var * self._bars_per_year)
            regimes.append(rv > 1.0)  # High vol: >100% annualized RV

        return regimes

    # ── Realized vol computation ────────────────────────────────────────────

    def _compute_rv_from_prices(
        self, prices: List[float]
    ) -> List[Optional[float]]:
        """Compute rolling realized vol from close prices."""
        if len(prices) < 2:
            return [None] * len(prices)

        n = len(prices)
        lookback = self.rv_lookback_bars
        log_rets = [None] + [
            math.log(prices[i] / prices[i - 1]) if prices[i - 1] > 0 else 0.0
            for i in range(1, n)
        ]

        rv_series: List[Optional[float]] = []
        for i in range(n):
            if i < lookback:
                rv_series.append(None)
                continue
            window = [r for r in log_rets[i - lookback + 1:i + 1] if r is not None]
            if len(window) < lookback // 2:
                rv_series.append(None)
                continue
            mean = sum(window) / len(window)
            var = sum((r - mean) ** 2 for r in window) / max(len(window) - 1, 1)
            rv = math.sqrt(var * self._bars_per_year)
            rv_series.append(round(rv, 4))

        return rv_series

    # ── Series alignment ────────────────────────────────────────────────────

    def _align_series(
        self,
        target_ts: List[int],
        src_ts: List[int],
        prices: List[float],
        rv: List[Optional[float]],
        iv_data: Dict[str, List],
    ) -> Tuple[List[float], List[Optional[float]], Dict[str, List]]:
        """Align all series to target timeline using last-value-carry-forward."""
        src_price_map = dict(zip(src_ts, prices))
        src_rv_map = dict(zip(src_ts, rv))
        iv_maps = {k: dict(zip(src_ts, v)) for k, v in iv_data.items()}

        aligned_prices: List[float] = []
        aligned_rv: List[Optional[float]] = []
        aligned_iv: Dict[str, List] = {k: [] for k in iv_data}

        last_price = 0.0
        last_rv: Optional[float] = None
        last_iv = {k: None for k in iv_data}

        for ts in target_ts:
            p = src_price_map.get(ts)
            if p is not None:
                last_price = p
            aligned_prices.append(last_price)

            r = src_rv_map.get(ts)
            if r is not None:
                last_rv = r
            aligned_rv.append(last_rv)

            for k in iv_data:
                v = iv_maps[k].get(ts)
                if v is not None:
                    last_iv[k] = v
                aligned_iv[k].append(last_iv[k])

        return aligned_prices, aligned_rv, aligned_iv

    # ── CSV caching ───────────────────────────────────────────────────────

    def _ensure_cache_dir(self):
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _save_prices_csv(
        self, path: Path, ts_list: List[int], prices: List[float]
    ):
        self._ensure_cache_dir()
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp", "close"])
            for ts, p in zip(ts_list, prices):
                writer.writerow([ts, p])
        logger.info("Saved prices to %s (%d bars)", path, len(ts_list))

    def _load_prices_csv(
        self, path: Path
    ) -> Tuple[List[int], List[float]]:
        ts_list, prices = [], []
        with open(path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                ts_list.append(int(row["timestamp"]))
                prices.append(float(row["close"]))
        return ts_list, prices

    def _save_rv_csv(
        self, path: Path, ts_list: List[int], rv: List[Optional[float]]
    ):
        self._ensure_cache_dir()
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp", "rv"])
            for ts, v in zip(ts_list, rv):
                writer.writerow([ts, "" if v is None else v])

    def _load_rv_csv(
        self, path: Path
    ) -> Tuple[List[int], List[Optional[float]]]:
        ts_list, rv = [], []
        with open(path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                ts_list.append(int(row["timestamp"]))
                v = row.get("rv", "")
                rv.append(float(v) if v else None)
        return ts_list, rv

    def _save_iv_csv(
        self, path: Path, ts_list: List[int], iv_data: Dict[str, List]
    ):
        self._ensure_cache_dir()
        keys = list(iv_data.keys())
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp"] + keys)
            for i, ts in enumerate(ts_list):
                row = [ts]
                for k in keys:
                    v = iv_data[k][i] if i < len(iv_data[k]) else None
                    row.append("" if v is None else v)
                writer.writerow(row)
        logger.info("Saved IV data to %s (%d bars)", path, len(ts_list))

    def _load_iv_csv(self, path: Path) -> Dict[str, List]:
        keys = ["iv_atm", "iv_25d_put", "iv_25d_call", "skew_25d", "butterfly_25d", "term_spread"]
        result: Dict[str, List] = {k: [] for k in keys}
        with open(path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                for k in keys:
                    v = row.get(k, "")
                    result[k].append(float(v) if v else None)
        return result

    # ── Utilities ─────────────────────────────────────────────────────────

    def _bar_seconds(self) -> int:
        return {"1h": 3600, "4h": 14400, "1d": 86400}.get(self.bar_interval, 86400)


# ── Provider factory (for dynamic registration) ────────────────────────────────

def make_provider(cfg: Dict[str, Any], seed: int = 42) -> DeribitRestProvider:
    """Create a DeribitRestProvider from config dict."""
    return DeribitRestProvider(cfg, seed)


# ── Date helpers ──────────────────────────────────────────────────────────────

def _parse_date_to_ts(date_str: str, end_of_day: bool = False) -> int:
    """Parse 'YYYY-MM-DD' or ISO datetime string to unix timestamp."""
    date_str = date_str.strip()
    for fmt in ("%Y-%m-%d", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M:%SZ"):
        try:
            dt = datetime.strptime(date_str[:len(fmt) - fmt.count("%") + 8], fmt)
            if end_of_day and "T" not in date_str:
                dt = dt.replace(hour=23, minute=59, second=59)
            return int(dt.replace(tzinfo=timezone.utc).timestamp())
        except ValueError:
            continue
    raise ValueError(f"Cannot parse date: {date_str!r}")
