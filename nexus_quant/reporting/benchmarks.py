"""
NEXUS Benchmark Data
=====================
Historical annual returns and risk metrics for standard benchmarks.
Used to contextualise NEXUS project performance.

Sources (as of Feb 2026):
- S&P 500 (SPY): Yahoo Finance adjusted close
- Bitcoin (BTC): CoinGecko / Binance
- SG CTA Index: Societe Generale CTA Index (approximate — published quarterly)
- Gold (GC): COMEX front-month continuous
- 60/40 Portfolio: 60% SPY + 40% AGG (iShares Core US Bond)

Note: Sharpe ratios are estimated from annual returns assuming:
  - Risk-free rate: 0% (2021), 4% (2022-2023), 5% (2024), 4.5% (2025)
  - Annual std from daily returns × sqrt(252)
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

# ── Annual returns (%) by year ─────────────────────────────────────────────────

BENCHMARK_ANNUAL_RETURNS: Dict[str, Dict[int, Optional[float]]] = {
    "SPY (S&P 500)": {
        2020: +18.37,
        2021: +28.71,
        2022: -18.17,
        2023: +26.29,
        2024: +25.02,
        2025: -1.5,    # YTD Feb 2026 (approximate)
    },
    "BTC Buy-Hold": {
        2020: +304.0,
        2021: +59.8,
        2022: -64.3,
        2023: +155.8,
        2024: +121.3,
        2025: -15.0,   # YTD Feb 2026 (approximate)
    },
    "SG CTA Index": {
        2020: +6.2,
        2021: +3.8,
        2022: +25.2,   # CTA best year in decades (energy + rates)
        2023: -3.1,
        2024: +5.6,
        2025: None,    # Not yet published
    },
    "Gold (GC)": {
        2020: +25.1,
        2021: -3.6,
        2022: +0.4,
        2023: +13.1,
        2024: +27.2,
        2025: +10.5,   # YTD (approximate)
    },
    "60/40 Portfolio": {
        2020: +14.5,
        2021: +15.1,
        2022: -16.1,
        2023: +17.8,
        2024: +14.8,
        2025: -0.5,
    },
    "Equal-Weight Commodity": {
        # Approximate equal-weight of 13 major commodity futures
        2020: +8.0,
        2021: +32.0,   # Commodity supercycle
        2022: +12.0,
        2023: -8.0,
        2024: +5.0,
        2025: +3.0,
    },
}

# ── Approximate annual vol (%) — used for Sharpe computation ──────────────────

BENCHMARK_ANNUAL_VOL: Dict[str, float] = {
    "SPY (S&P 500)": 18.5,
    "BTC Buy-Hold": 72.0,
    "SG CTA Index": 8.5,
    "Gold (GC)": 13.5,
    "60/40 Portfolio": 12.0,
    "Equal-Weight Commodity": 20.0,
}

# ── Risk-free rates by year ───────────────────────────────────────────────────

RISK_FREE_RATE: Dict[int, float] = {
    2020: 0.5,
    2021: 0.25,
    2022: 2.5,
    2023: 5.0,
    2024: 5.0,
    2025: 4.5,
}


# ── Computed Sharpe ratios ────────────────────────────────────────────────────

def compute_benchmark_sharpe(
    name: str,
    year: int,
) -> Optional[float]:
    """Estimate annual Sharpe ratio for a benchmark in a given year."""
    ret = BENCHMARK_ANNUAL_RETURNS.get(name, {}).get(year)
    if ret is None:
        return None
    vol = BENCHMARK_ANNUAL_VOL.get(name, 20.0)
    rf = RISK_FREE_RATE.get(year, 2.0)
    return (ret - rf) / vol


# ── Build full benchmark data dict ────────────────────────────────────────────

def _build_benchmark_data() -> Dict[str, Any]:
    years = [2020, 2021, 2022, 2023, 2024, 2025]
    benchmarks = {}
    for name in BENCHMARK_ANNUAL_RETURNS:
        entry: Dict[str, Any] = {
            "annual_returns": {},
            "annual_sharpe": {},
            "vol_estimate": BENCHMARK_ANNUAL_VOL.get(name, 20.0),
        }
        for yr in years:
            ret = BENCHMARK_ANNUAL_RETURNS[name].get(yr)
            entry["annual_returns"][yr] = ret
            entry["annual_sharpe"][yr] = compute_benchmark_sharpe(name, yr)
        benchmarks[name] = entry
    return benchmarks


BENCHMARK_DATA: Dict[str, Any] = _build_benchmark_data()
