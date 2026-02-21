"""
Deribit Live Data Collector
============================

Collects real-time IV, skew, and price data from Deribit public API hourly.
Stores to CSV files for future backtest validation of Skew MR + Term Structure.

Data collected per hour:
- BTC/ETH perpetual price (from Deribit)
- ATM IV (inferred from nearest strike option)
- 25-delta put IV, call IV, skew (25d_put - 25d_call)
- Front-month / back-month IV spread (term structure)
- Realized vol (from hourly price moves)

Storage: data/cache/deribit/live/YYYY-MM/BTC_YYYY-MM-DD.csv

Why this matters:
- Skew MR and Term Structure strategies FAIL with synthetic data
- Real Deribit IV needed for proper backtest
- Collect 60+ days → re-validate strategies with real data

Usage:
    python3 -m nexus_quant.projects.crypto_options.collectors.deribit_live_collector
    # Runs once and exits (designed for cron / launchd scheduling)

    python3 -m nexus_quant.projects.crypto_options.collectors.deribit_live_collector --loop
    # Runs every COLLECT_INTERVAL seconds (default 3600 = 1 hour)
"""
from __future__ import annotations

import csv
import json
import logging
import math
import os
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("nexus.deribit_collector")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

# ── Constants ─────────────────────────────────────────────────────────────────
DERIBIT_BASE = "https://www.deribit.com/api/v2/public"
COLLECT_INTERVAL = 3600  # seconds (1 hour)
REQUEST_DELAY = 0.3       # seconds between API calls

SYMBOLS = {
    "BTC": {"perp": "BTC-PERPETUAL", "currency": "BTC"},
    "ETH": {"perp": "ETH-PERPETUAL", "currency": "ETH"},
}

BASE_CACHE = Path("data/cache/deribit/live")

# ── API helpers ────────────────────────────────────────────────────────────────

def _make_ssl_context():
    """Create SSL context that works on macOS (certifi or unverified fallback)."""
    import ssl
    try:
        import certifi
        return ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        pass
    # macOS fallback: use unverified context for public API
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    return ctx

_SSL_CTX = _make_ssl_context()


def _api_get(endpoint: str, params: Dict[str, Any]) -> Optional[Dict]:
    """Simple GET to Deribit public API with retry."""
    qs = "&".join(f"{k}={v}" for k, v in params.items())
    url = f"{DERIBIT_BASE}/{endpoint}?{qs}"
    for attempt in range(3):
        try:
            with urllib.request.urlopen(url, timeout=10, context=_SSL_CTX) as resp:
                data = json.loads(resp.read().decode())
                if "result" in data:
                    return data["result"]
                logger.warning("API returned no result: %s", data)
                return None
        except Exception as e:
            logger.warning("API error (attempt %d): %s — %s", attempt+1, url[:80], e)
            if attempt < 2:
                time.sleep(2 ** attempt)
    return None


def _get_perpetual_price(symbol: str) -> Optional[float]:
    """Get current perpetual price for BTC or ETH."""
    instrument = SYMBOLS[symbol]["perp"]
    result = _api_get("get_order_book", {"instrument_name": instrument, "depth": 1})
    if result:
        return float(result.get("mark_price", result.get("last_price", 0)) or 0)
    return None


def _get_options_chain(symbol: str) -> List[Dict]:
    """Get all active options for a currency."""
    currency = SYMBOLS[symbol]["currency"]
    result = _api_get("get_instruments", {"currency": currency, "kind": "option"})
    return result or []


def _get_option_summary(instrument_name: str) -> Optional[Dict]:
    """Get ticker data for a specific option."""
    result = _api_get("ticker", {"instrument_name": instrument_name})
    time.sleep(0.05)  # rate limit
    return result


def _parse_expiry_days(instrument_name: str, now: datetime) -> Optional[float]:
    """Parse days to expiry from instrument name like BTC-20MAR26-100000-C."""
    parts = instrument_name.split("-")
    if len(parts) < 4:
        return None
    try:
        expiry = datetime.strptime(parts[1], "%d%b%y").replace(tzinfo=timezone.utc)
        return (expiry - now).total_seconds() / 86400
    except Exception:
        return None


def _find_atm_option(
    options: List[Dict],
    spot: float,
    symbol: str,
    target_expiry_days: float = 30.0,
    tolerance_days: float = 7.0,
) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """
    Find ATM IV from the nearest-to-30d expiry option.
    Returns: (iv_atm, iv_25d_put, iv_25d_call) in decimal form (0.75 = 75%)
    """
    now = datetime.now(tz=timezone.utc)
    currency = SYMBOLS[symbol]["currency"]

    # Filter to our symbol's options
    relevant = [
        o for o in options
        if o.get("base_currency") == currency
        and o.get("kind") == "option"
        and not o.get("is_expired", False)
    ]

    # Find best expiry (closest to target_expiry_days)
    best_expiry_days = None
    best_expiry_name = None
    for opt in relevant:
        days = _parse_expiry_days(opt["instrument_name"], now)
        if days is not None and days > 1:
            if best_expiry_days is None or abs(days - target_expiry_days) < abs(best_expiry_days - target_expiry_days):
                best_expiry_days = days
                best_expiry_name = opt["instrument_name"].split("-")[1]  # e.g. "20MAR26"

    if best_expiry_name is None:
        logger.warning("%s: no suitable expiry found", symbol)
        return None, None, None

    # Get all options for this expiry
    expiry_opts = [
        o for o in relevant
        if f"-{best_expiry_name}-" in o["instrument_name"]
    ]

    # Separate calls and puts, find ATM (closest strike to spot)
    calls = [o for o in expiry_opts if o["instrument_name"].endswith("-C")]
    puts = [o for o in expiry_opts if o["instrument_name"].endswith("-P")]

    def get_strike(name: str) -> Optional[float]:
        parts = name.split("-")
        if len(parts) >= 4:
            try:
                return float(parts[2])
            except:
                return None
        return None

    # ATM call/put (strike closest to spot)
    calls_with_strikes = [(get_strike(o["instrument_name"]), o) for o in calls]
    calls_with_strikes = [(s, o) for s, o in calls_with_strikes if s is not None]
    calls_sorted = sorted(calls_with_strikes, key=lambda x: abs(x[0] - spot))

    iv_atm = None
    if calls_sorted:
        atm_strike, atm_opt = calls_sorted[0]
        ticker = _get_option_summary(atm_opt["instrument_name"])
        if ticker:
            iv_atm = ticker.get("mark_iv")
            if iv_atm:
                iv_atm = float(iv_atm) / 100.0  # convert % to decimal

    # 25-delta put (OTM put, strike ~20-25% below spot)
    target_put_strike = spot * 0.75  # rough 25-delta put
    puts_sorted = [(get_strike(o["instrument_name"]), o) for o in puts if get_strike(o["instrument_name"]) is not None]
    puts_sorted.sort(key=lambda x: abs(x[0] - target_put_strike))

    iv_25d_put = None
    if puts_sorted:
        _, put_25d = puts_sorted[0]
        ticker = _get_option_summary(put_25d["instrument_name"])
        if ticker:
            iv_raw = ticker.get("mark_iv")
            if iv_raw:
                iv_25d_put = float(iv_raw) / 100.0

    # 25-delta call (OTM call, strike ~20-25% above spot)
    target_call_strike = spot * 1.25  # rough 25-delta call
    calls_otm_sorted = sorted(
        [(get_strike(o["instrument_name"]), o) for o in calls if get_strike(o["instrument_name"]) is not None],
        key=lambda x: abs(x[0] - target_call_strike)
    )

    iv_25d_call = None
    if calls_otm_sorted:
        _, call_25d = calls_otm_sorted[0]
        ticker = _get_option_summary(call_25d["instrument_name"])
        if ticker:
            iv_raw = ticker.get("mark_iv")
            if iv_raw:
                iv_25d_call = float(iv_raw) / 100.0

    return iv_atm, iv_25d_put, iv_25d_call


def _get_term_structure(
    options: List[Dict],
    spot: float,
    symbol: str,
) -> Optional[float]:
    """
    Estimate front/back month IV spread.
    Returns: iv_front - iv_back (positive = contango normal, negative = backwardation)
    """
    now = datetime.now(tz=timezone.utc)
    currency = SYMBOLS[symbol]["currency"]

    relevant = [
        o for o in options
        if o.get("base_currency") == currency
        and o.get("kind") == "option"
        and not o.get("is_expired", False)
        and o["instrument_name"].endswith("-C")  # ATM calls for term structure
    ]

    # Group by expiry, find front (7-40d) and back (60-120d)
    expiry_groups: Dict[str, float] = {}  # expiry_name -> days
    for opt in relevant:
        days = _parse_expiry_days(opt["instrument_name"], now)
        expiry_name = opt["instrument_name"].split("-")[1]
        if days and days > 1 and expiry_name not in expiry_groups:
            expiry_groups[expiry_name] = days

    front_expiry = None
    back_expiry = None
    for exp, days in sorted(expiry_groups.items(), key=lambda x: x[1]):
        if 7 <= days <= 40 and front_expiry is None:
            front_expiry = exp
        elif 50 <= days <= 120 and back_expiry is None:
            back_expiry = exp

    if not front_expiry or not back_expiry:
        return None

    def get_atm_iv_for_expiry(expiry_name: str) -> Optional[float]:
        """Get ATM IV for a specific expiry."""
        expiry_calls = [
            o for o in relevant
            if f"-{expiry_name}-" in o["instrument_name"]
        ]
        if not expiry_calls:
            return None
        strikes = [(abs(float(o["instrument_name"].split("-")[2]) - spot), o) for o in expiry_calls
                   if o["instrument_name"].split("-")[2].isdigit()]
        strikes.sort(key=lambda x: x[0])
        if not strikes:
            return None
        _, atm_opt = strikes[0]
        ticker = _get_option_summary(atm_opt["instrument_name"])
        if ticker:
            iv_raw = ticker.get("mark_iv")
            return float(iv_raw) / 100.0 if iv_raw else None
        return None

    iv_front = get_atm_iv_for_expiry(front_expiry)
    iv_back = get_atm_iv_for_expiry(back_expiry)

    if iv_front and iv_back:
        return iv_front - iv_back
    return None


# ── Storage ────────────────────────────────────────────────────────────────────

def _get_csv_path(symbol: str, dt: datetime) -> Path:
    """Get CSV file path for given symbol and date."""
    month_dir = BASE_CACHE / dt.strftime("%Y-%m")
    month_dir.mkdir(parents=True, exist_ok=True)
    return month_dir / f"{symbol}_{dt.strftime('%Y-%m-%d')}.csv"


CSV_HEADER = [
    "ts_utc", "symbol", "price", "iv_atm", "iv_25d_put", "iv_25d_call",
    "skew_25d", "butterfly_25d", "term_spread", "funding_rate", "rv_1h", "rv_24h",
]


def _append_to_csv(path: Path, row: Dict[str, Any]) -> None:
    """Append a row to CSV, creating header if needed."""
    write_header = not path.exists() or path.stat().st_size == 0
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_HEADER)
        if write_header:
            writer.writeheader()
        writer.writerow({k: row.get(k, "") for k in CSV_HEADER})


def _get_funding_rate(symbol: str) -> Optional[float]:
    """Get current perpetual funding rate from Deribit.

    Returns the 8-hour funding rate as decimal (e.g. 0.00003 = 0.003% per 8h).
    Fetched from the ticker endpoint which includes funding_8h field.
    """
    instrument = SYMBOLS[symbol]["perp"]
    result = _api_get("ticker", {"instrument_name": instrument})
    if result:
        rate_8h = result.get("funding_8h")
        if rate_8h is not None:
            return float(rate_8h)
    return None


def _get_recent_rv(symbol: str, hours: int = 24) -> Optional[float]:
    """Estimate realized vol from recent price data in CSV files."""
    prices = []
    now = datetime.now(tz=timezone.utc)
    for i in range(2):  # check today and yesterday
        dt = now - timedelta(days=i)
        p = _get_csv_path(symbol, dt)
        if not p.exists():
            continue
        try:
            with open(p) as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if row.get("price"):
                        prices.append(float(row["price"]))
        except Exception:
            pass
    if len(prices) < 4:
        return None
    # Simple annualized vol from last `hours` data points
    rets = [math.log(prices[i] / prices[i-1]) for i in range(1, min(hours+1, len(prices)))]
    if len(rets) < 2:
        return None
    mean = sum(rets) / len(rets)
    var = sum((r - mean)**2 for r in rets) / (len(rets) - 1)
    hourly_vol = var ** 0.5
    return hourly_vol * math.sqrt(8760)  # annualize from hourly


# ── Main collection logic ─────────────────────────────────────────────────────

def collect_once() -> Dict[str, Any]:
    """
    Collect one snapshot of Deribit data for all symbols.
    Returns: summary of collected data
    """
    now = datetime.now(tz=timezone.utc)
    ts_utc = now.strftime("%Y-%m-%dT%H:%M:%SZ")
    summary = {}

    for symbol in SYMBOLS:
        logger.info("Collecting %s...", symbol)

        # 1. Spot price
        price = _get_perpetual_price(symbol)
        if not price:
            logger.warning("%s: failed to get price", symbol)
            summary[symbol] = {"error": "no_price"}
            continue
        logger.info("  %s price: %.2f", symbol, price)
        time.sleep(REQUEST_DELAY)

        # 2. Options chain (instruments list)
        options = _get_options_chain(symbol)
        logger.info("  %s options: %d instruments", symbol, len(options))
        time.sleep(REQUEST_DELAY)

        # 3. ATM IV + 25d skew
        iv_atm, iv_25d_put, iv_25d_call = _find_atm_option(options, price, symbol)
        skew_25d = None
        if iv_25d_put is not None and iv_25d_call is not None:
            skew_25d = iv_25d_put - iv_25d_call

        logger.info("  %s IV_atm=%.3f skew=%.3f",
                    symbol,
                    iv_atm or 0,
                    skew_25d or 0)
        time.sleep(REQUEST_DELAY)

        # 4. Term structure spread
        term_spread = _get_term_structure(options, price, symbol)
        logger.info("  %s term_spread=%.3f", symbol, term_spread or 0)

        # 5. Perpetual funding rate
        funding_rate = _get_funding_rate(symbol)
        if funding_rate is not None:
            logger.info("  %s funding_rate=%.6f (8h)", symbol, funding_rate)
        time.sleep(REQUEST_DELAY)

        # 6. Butterfly (vol smile convexity)
        butterfly_25d = None
        if iv_25d_put is not None and iv_25d_call is not None and iv_atm is not None:
            butterfly_25d = 0.5 * (iv_25d_put + iv_25d_call) - iv_atm

        # 7. Realized vol (from historical CSV data)
        rv_1h = _get_recent_rv(symbol, hours=1)
        rv_24h = _get_recent_rv(symbol, hours=24)

        # 8. Save to CSV
        row = {
            "ts_utc": ts_utc,
            "symbol": symbol,
            "price": round(price, 2),
            "iv_atm": round(iv_atm, 4) if iv_atm else "",
            "iv_25d_put": round(iv_25d_put, 4) if iv_25d_put else "",
            "iv_25d_call": round(iv_25d_call, 4) if iv_25d_call else "",
            "skew_25d": round(skew_25d, 4) if skew_25d else "",
            "butterfly_25d": round(butterfly_25d, 4) if butterfly_25d else "",
            "term_spread": round(term_spread, 4) if term_spread else "",
            "funding_rate": round(funding_rate, 8) if funding_rate is not None else "",
            "rv_1h": round(rv_1h, 4) if rv_1h else "",
            "rv_24h": round(rv_24h, 4) if rv_24h else "",
        }
        csv_path = _get_csv_path(symbol, now)
        _append_to_csv(csv_path, row)
        logger.info("  %s saved to %s", symbol, csv_path)

        summary[symbol] = {
            "price": price,
            "iv_atm": iv_atm,
            "skew_25d": skew_25d,
            "butterfly_25d": butterfly_25d,
            "term_spread": term_spread,
            "funding_rate": funding_rate,
        }

    return summary


def run_loop(interval: int = COLLECT_INTERVAL) -> None:
    """Run collection loop forever."""
    logger.info("Deribit Live Data Collector started. interval=%ds", interval)
    while True:
        start = time.time()
        try:
            summary = collect_once()
            logger.info("Collection complete: %s", {k: {kk: round(vv, 4) if isinstance(vv, float) else vv
                                                        for kk, vv in v.items() if vv} for k, v in summary.items()})
        except Exception as e:
            logger.error("Collection failed: %s", e, exc_info=True)

        elapsed = time.time() - start
        sleep_time = max(0, interval - elapsed)
        logger.info("Sleeping %.0fs until next collection...", sleep_time)
        time.sleep(sleep_time)


# ── Status reporter ────────────────────────────────────────────────────────────

def status_report() -> str:
    """Show how much live data has been collected so far."""
    lines = ["=== Deribit Live Data Collection Status ==="]
    total_rows = 0
    for symbol in SYMBOLS:
        csv_files = sorted(BASE_CACHE.glob(f"**/{symbol}_*.csv"))
        rows = 0
        for f in csv_files:
            try:
                with open(f) as fp:
                    rows += sum(1 for _ in fp) - 1  # subtract header
            except Exception:
                pass
        total_rows += rows
        if csv_files:
            oldest = csv_files[0].stem.split("_", 1)[-1]
            newest = csv_files[-1].stem.split("_", 1)[-1]
            lines.append(f"  {symbol}: {rows} rows, {oldest} → {newest} ({len(csv_files)} files)")
        else:
            lines.append(f"  {symbol}: no data yet")

    lines.append(f"  Total: {total_rows} rows")
    lines.append("")
    lines.append("Data path: " + str(BASE_CACHE))
    if total_rows >= 720:  # 30 days * 24 hours
        lines.append("STATUS: READY for Skew MR + Term Structure backtest (≥ 30 days)")
    else:
        lines.append(f"STATUS: Collecting... ({total_rows}/720 rows for 30-day threshold)")
    return "\n".join(lines)


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Deribit Live Data Collector")
    parser.add_argument("--loop", action="store_true", help="Run continuously (else run once)")
    parser.add_argument("--status", action="store_true", help="Show collection status")
    parser.add_argument("--interval", type=int, default=COLLECT_INTERVAL, help="Interval in seconds")
    args = parser.parse_args()

    if args.status:
        print(status_report())
        sys.exit(0)

    if args.loop:
        run_loop(interval=args.interval)
    else:
        collect_once()
        print(status_report())
