"""
DefiLlama Stablecoin Supply Provider.

Fetches total stablecoin supply from DefiLlama API (free, no auth).
This is a daily macro signal — stablecoin supply expansion = capital inflows
into crypto ecosystem = bullish macro.

Data: daily from 2017-11-29, all stablecoins combined.

Usage:
    supply = load_stablecoin_supply()
    # supply = {epoch_s: total_usd_supply, ...}  (daily resolution)
"""
from __future__ import annotations

import json
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional

CACHE_DIR = Path(".cache/defillama")
STABLECOIN_CHART_URL = "https://stablecoins.llama.fi/stablecoincharts/all"
CACHE_TTL_HOURS = 24  # re-fetch at most once per day


def load_stablecoin_supply(
    cache_dir: Optional[Path] = None,
) -> Dict[int, float]:
    """
    Fetch total stablecoin circulating supply (USD) from DefiLlama.
    Returns {epoch_s: total_supply_usd} at daily resolution.
    Cached locally for 24h.
    """
    if cache_dir is None:
        cache_dir = CACHE_DIR
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / "stablecoin_total_supply.json"

    # Check cache freshness
    if cache_file.exists():
        age_hours = (time.time() - cache_file.stat().st_mtime) / 3600
        if age_hours < CACHE_TTL_HOURS:
            with open(cache_file) as f:
                return {int(k): v for k, v in json.load(f).items()}

    # Fetch from API
    print("  [DefiLlama] Fetching total stablecoin supply...")
    try:
        req = urllib.request.Request(STABLECOIN_CHART_URL)
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except Exception as e:
        print(f"  [DefiLlama] Fetch failed: {e}")
        # Try cache even if stale
        if cache_file.exists():
            with open(cache_file) as f:
                return {int(k): v for k, v in json.load(f).items()}
        return {}

    supply: Dict[int, float] = {}
    for entry in data:
        ts = int(entry.get("date", 0))
        usd = entry.get("totalCirculatingUSD", {}).get("peggedUSD", 0)
        if usd is None:
            usd = entry.get("totalCirculating", {}).get("peggedUSD", 0)
        if ts > 0 and usd and float(usd) > 0:
            supply[ts] = float(usd)

    # Cache
    with open(cache_file, "w") as f:
        json.dump(supply, f)

    print(f"    -> {len(supply)} daily entries ({min(supply.keys()) if supply else 0} .. {max(supply.keys()) if supply else 0})")
    return supply


def compute_stablecoin_momentum(
    supply: Dict[int, float],
    timeline: list,
    lookback_days: int = 7,
) -> "np.ndarray":
    """
    Compute stablecoin supply momentum z-score at each hourly bar.

    Logic:
    - For each bar, find the latest daily supply value
    - Compute 7-day supply change (momentum)
    - Z-score the momentum using a rolling window

    Returns numpy array of z-scores aligned to timeline.
    """
    import numpy as np

    n = len(timeline)
    z_scores = np.zeros(n)

    # Build sorted daily supply series
    sorted_ts = sorted(supply.keys())
    if len(sorted_ts) < lookback_days * 3:
        return z_scores

    supply_vals = [supply[t] for t in sorted_ts]

    # For each bar, find the most recent daily supply
    daily_at_bar = np.zeros(n)
    si = 0  # pointer into sorted_ts
    for i, bar_ts in enumerate(timeline):
        while si < len(sorted_ts) - 1 and sorted_ts[si + 1] <= bar_ts:
            si += 1
        if si < len(sorted_ts) and sorted_ts[si] <= bar_ts:
            daily_at_bar[i] = supply_vals[si]
        elif i > 0:
            daily_at_bar[i] = daily_at_bar[i - 1]

    # Compute log supply
    log_supply = np.log(np.maximum(daily_at_bar, 1.0))

    # Lookback in hours
    lb_hours = lookback_days * 24
    z_lb = lb_hours * 2  # z-score window

    for i in range(z_lb, n):
        if daily_at_bar[i] < 1e6:  # no data yet
            continue

        # Momentum: change over lookback period
        mom_i = log_supply[i] - log_supply[max(0, i - lb_hours)]

        # Rolling momentum stats
        moms = []
        for j in range(max(lb_hours, i - lb_hours), i + 1):
            if j >= lb_hours:
                moms.append(log_supply[j] - log_supply[j - lb_hours])

        if len(moms) < 10:
            continue

        mu = float(np.mean(moms))
        sigma = float(np.std(moms))
        if sigma > 0:
            z_scores[i] = (mom_i - mu) / sigma

    return z_scores
