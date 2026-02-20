"""
Binance Vision Bulk Metrics Provider.

Downloads daily CSV files from data.binance.vision containing:
  - sum_open_interest / sum_open_interest_value
  - count_toptrader_long_short_ratio  (top trader account L/S)
  - sum_toptrader_long_short_ratio    (top trader position L/S)
  - count_long_short_ratio            (global account L/S)
  - sum_taker_long_short_vol_ratio    (taker buy/sell vol ratio)

Data is at 5-minute resolution from 2021-12-01. We resample to 1h by taking
the last value in each hour (point-in-time snapshot — no look-ahead).

All downloads are cached locally so subsequent loads are instant.

Usage:
    metrics = load_vision_metrics(
        symbols=["BTCUSDT", "ETHUSDT"],
        start_date="2022-01-01",
        end_date="2025-01-01",
    )
    # metrics = {
    #   "BTCUSDT": {
    #     "open_interest": {epoch_s: val, ...},
    #     "global_ls_ratio": {epoch_s: val, ...},
    #     "top_ls_position": {epoch_s: val, ...},
    #     "top_ls_account": {epoch_s: val, ...},
    #     "taker_ls_ratio": {epoch_s: val, ...},
    #   }, ...
    # }
"""
from __future__ import annotations

import csv
import io
import os
import time
import urllib.request
import zipfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

VISION_BASE = "https://data.binance.vision/data/futures/um/daily/metrics"
CACHE_DIR = Path(".cache/binance_vision_metrics")
EARLIEST_DATE = datetime(2021, 12, 1, tzinfo=timezone.utc)

# Column indices in the CSV (after header)
# create_time, symbol, sum_open_interest, sum_open_interest_value,
# count_toptrader_long_short_ratio, sum_toptrader_long_short_ratio,
# count_long_short_ratio, sum_taker_long_short_vol_ratio
COL_TIME = 0
COL_OI = 2          # sum_open_interest
COL_OI_VALUE = 3    # sum_open_interest_value
COL_TOP_LS_ACCT = 4 # count_toptrader_long_short_ratio
COL_TOP_LS_POS = 5  # sum_toptrader_long_short_ratio
COL_GLOBAL_LS = 6   # count_long_short_ratio
COL_TAKER_LS = 7    # sum_taker_long_short_vol_ratio


def _download_day(symbol: str, date: datetime, cache_dir: Path) -> Optional[str]:
    """Download one day's metrics CSV. Returns CSV text or None."""
    date_str = date.strftime("%Y-%m-%d")
    cache_file = cache_dir / symbol / f"{symbol}-metrics-{date_str}.csv"

    if cache_file.exists():
        return cache_file.read_text("utf-8")

    url = f"{VISION_BASE}/{symbol}/{symbol}-metrics-{date_str}.zip"
    try:
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, timeout=15) as resp:
            zip_data = resp.read()
    except Exception:
        return None

    try:
        with zipfile.ZipFile(io.BytesIO(zip_data)) as zf:
            names = zf.namelist()
            csv_name = [n for n in names if n.endswith(".csv")]
            if not csv_name:
                return None
            csv_text = zf.read(csv_name[0]).decode("utf-8")
    except Exception:
        return None

    cache_file.parent.mkdir(parents=True, exist_ok=True)
    cache_file.write_text(csv_text, "utf-8")
    return csv_text


def _parse_csv_to_5min(
    csv_text: str,
) -> List[Tuple[int, float, float, float, float, float]]:
    """
    Parse CSV text to list of (epoch_s, oi_value, top_ls_acct, top_ls_pos, global_ls, taker_ls).
    """
    rows = []
    reader = csv.reader(io.StringIO(csv_text))
    header = next(reader, None)
    if not header:
        return rows

    for line in reader:
        if len(line) < 8:
            continue
        try:
            # Parse time like "2024-01-15 00:05:00"
            dt = datetime.strptime(line[COL_TIME], "%Y-%m-%d %H:%M:%S")
            dt = dt.replace(tzinfo=timezone.utc)
            epoch_s = int(dt.timestamp())
            # Handle empty strings in early data (pre-2022-Q2)
            def _safe_float(s, default=0.0):
                s = s.strip().strip('"')
                return float(s) if s else default
            oi_val = _safe_float(line[COL_OI_VALUE])
            top_ls_acct = _safe_float(line[COL_TOP_LS_ACCT])
            top_ls_pos = _safe_float(line[COL_TOP_LS_POS])
            global_ls = _safe_float(line[COL_GLOBAL_LS])
            taker_ls = _safe_float(line[COL_TAKER_LS])
            rows.append((epoch_s, oi_val, top_ls_acct, top_ls_pos, global_ls, taker_ls))
        except (ValueError, IndexError):
            continue
    return rows


def _resample_to_hourly(
    rows_5m: List[Tuple[int, float, float, float, float, float]],
) -> Dict[int, Tuple[float, float, float, float, float]]:
    """
    Resample 5-min data to hourly by taking the last value in each hour.
    Returns {hour_epoch_s: (oi_val, top_ls_acct, top_ls_pos, global_ls, taker_ls)}.
    """
    hourly: Dict[int, Tuple[float, float, float, float, float]] = {}
    for epoch_s, oi_val, top_ls_acct, top_ls_pos, global_ls, taker_ls in rows_5m:
        # Round down to hour
        hour_s = (epoch_s // 3600) * 3600
        # Always overwrite — last 5-min bar in each hour wins (latest snapshot)
        hourly[hour_s] = (oi_val, top_ls_acct, top_ls_pos, global_ls, taker_ls)
    return hourly


def load_vision_metrics(
    symbols: List[str],
    start_date: str,
    end_date: str,
    cache_dir: Optional[Path] = None,
) -> Dict[str, Dict[str, Dict[int, float]]]:
    """
    Download and parse Binance Vision metrics for multiple symbols.

    Args:
        symbols: e.g. ["BTCUSDT", "ETHUSDT"]
        start_date: "YYYY-MM-DD"
        end_date: "YYYY-MM-DD"
        cache_dir: local cache path (default: .cache/binance_vision_metrics)

    Returns:
        {symbol: {metric_name: {epoch_s: value}}}
        metric names: "open_interest", "global_ls_ratio", "top_ls_position",
                      "top_ls_account", "taker_ls_ratio"
    """
    if cache_dir is None:
        cache_dir = CACHE_DIR
    cache_dir.mkdir(parents=True, exist_ok=True)

    start_dt = max(
        datetime.strptime(start_date, "%Y-%m-%d").replace(tzinfo=timezone.utc),
        EARLIEST_DATE,
    )
    end_dt = datetime.strptime(end_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)

    # Don't try to fetch tomorrow or later
    today = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
    end_dt = min(end_dt, today)

    total_days = (end_dt - start_dt).days
    if total_days <= 0:
        return {}

    result: Dict[str, Dict[str, Dict[int, float]]] = {}

    for sym in symbols:
        print(f"  [VisionMetrics] {sym}: downloading {total_days} days of metrics...")
        all_hourly: Dict[int, Tuple[float, float, float, float, float]] = {}

        downloaded = 0
        cached = 0
        failed = 0

        d = start_dt
        while d < end_dt:
            csv_cache = cache_dir / sym / f"{sym}-metrics-{d.strftime('%Y-%m-%d')}.csv"
            is_cached = csv_cache.exists()

            csv_text = _download_day(sym, d, cache_dir)
            if csv_text:
                rows_5m = _parse_csv_to_5min(csv_text)
                hourly = _resample_to_hourly(rows_5m)
                all_hourly.update(hourly)
                if is_cached:
                    cached += 1
                else:
                    downloaded += 1
                    # Rate limit: only sleep on actual downloads
                    if downloaded % 50 == 0:
                        time.sleep(1.0)
            else:
                failed += 1

            d += timedelta(days=1)

        print(f"    -> {len(all_hourly)} hourly bars ({downloaded} downloaded, {cached} cached, {failed} failed)")

        # Split into individual metric dicts
        oi_map: Dict[int, float] = {}
        global_ls_map: Dict[int, float] = {}
        top_ls_pos_map: Dict[int, float] = {}
        top_ls_acct_map: Dict[int, float] = {}
        taker_ls_map: Dict[int, float] = {}

        for ts, (oi_val, top_ls_acct, top_ls_pos, global_ls, taker_ls) in all_hourly.items():
            oi_map[ts] = oi_val
            global_ls_map[ts] = global_ls
            top_ls_pos_map[ts] = top_ls_pos
            top_ls_acct_map[ts] = top_ls_acct
            taker_ls_map[ts] = taker_ls

        result[sym] = {
            "open_interest": oi_map,
            "global_ls_ratio": global_ls_map,
            "top_ls_position": top_ls_pos_map,
            "top_ls_account": top_ls_acct_map,
            "taker_ls_ratio": taker_ls_map,
        }

    return result


def merge_metrics_into_dataset(
    dataset,
    metrics: Dict[str, Dict[str, Dict[int, float]]],
):
    """
    Create a new MarketDataset with metrics data merged in.

    Since MarketDataset is frozen, we create a new instance with the
    positioning fields populated from the vision metrics.
    """
    from ..schema import MarketDataset

    timeline = dataset.timeline
    symbols = dataset.symbols

    oi_dict: Dict[str, List[float]] = {}
    global_ls_dict: Dict[str, List[float]] = {}
    top_ls_pos_dict: Dict[str, List[float]] = {}
    top_ls_acct_dict: Dict[str, List[float]] = {}
    taker_ls_dict: Dict[str, List[float]] = {}

    for sym in symbols:
        sym_metrics = metrics.get(sym, {})
        oi_src = sym_metrics.get("open_interest", {})
        gls_src = sym_metrics.get("global_ls_ratio", {})
        tlp_src = sym_metrics.get("top_ls_position", {})
        tla_src = sym_metrics.get("top_ls_account", {})
        tkr_src = sym_metrics.get("taker_ls_ratio", {})

        # Forward-fill aligned to timeline
        oi_list, gls_list, tlp_list, tla_list, tkr_list = [], [], [], [], []
        last_oi, last_gls, last_tlp, last_tla, last_tkr = 0.0, 1.0, 1.0, 1.0, 1.0

        for t in timeline:
            if t in oi_src:
                last_oi = oi_src[t]
            oi_list.append(last_oi)

            if t in gls_src:
                last_gls = gls_src[t]
            gls_list.append(last_gls)

            if t in tlp_src:
                last_tlp = tlp_src[t]
            tlp_list.append(last_tlp)

            if t in tla_src:
                last_tla = tla_src[t]
            tla_list.append(last_tla)

            if t in tkr_src:
                last_tkr = tkr_src[t]
            tkr_list.append(last_tkr)

        oi_dict[sym] = oi_list
        global_ls_dict[sym] = gls_list
        top_ls_pos_dict[sym] = tlp_list
        top_ls_acct_dict[sym] = tla_list
        taker_ls_dict[sym] = tkr_list

    return MarketDataset(
        provider=dataset.provider,
        timeline=timeline,
        symbols=symbols,
        perp_close=dataset.perp_close,
        spot_close=dataset.spot_close,
        funding=dataset.funding,
        fingerprint=dataset.fingerprint,
        perp_volume=dataset.perp_volume,
        taker_buy_volume=dataset.taker_buy_volume,
        spot_volume=dataset.spot_volume,
        perp_mark_close=dataset.perp_mark_close,
        perp_index_close=dataset.perp_index_close,
        bid_close=dataset.bid_close,
        ask_close=dataset.ask_close,
        open_interest=oi_dict,
        long_short_ratio_global=global_ls_dict,
        long_short_ratio_top=top_ls_pos_dict,
        taker_long_short_ratio=taker_ls_dict,
        top_trader_ls_account=top_ls_acct_dict,
        _funding_times=dataset._funding_times,
        meta=dataset.meta,
    )
