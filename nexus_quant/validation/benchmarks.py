from __future__ import annotations

"""
NEXUS Benchmark Comparison Module.

Fetches real benchmark data and compares against strategy performance.
Data sources (all free, no API key):
- S&P 500 (SPY) via Yahoo Finance
- Bitcoin via Binance public REST
- Renaissance Medallion: hardcoded from public filings
- 60/40 Portfolio: constructed approximation
"""

import json
import math
import statistics
import urllib.request
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


# ── Hardcoded annual returns (source: public records / Wikipedia) ──────────

RENAISSANCE_RETURNS = {
    # Medallion fund gross returns (before 5%+44% fees)
    2000: 0.99, 2001: 0.33, 2002: 0.26, 2003: 0.26, 2004: 0.26,
    2005: 0.29, 2006: 0.44, 2007: 0.73, 2008: 0.82, 2009: 0.39,
    2010: 0.29, 2011: 0.37, 2012: 0.29, 2013: 0.47, 2014: 0.39,
    2015: 0.36, 2016: 0.35, 2017: 0.53, 2018: 0.40, 2019: 0.39,
    2020: 0.76, 2021: 0.26, 2022: 0.25, 2023: 0.32, 2024: 0.30,
}

SP500_RETURNS = {
    2018: -0.044, 2019: 0.315, 2020: 0.184, 2021: 0.287,
    2022: -0.181, 2023: 0.265, 2024: 0.230,
}

BTC_APPROX_RETURNS = {
    2018: -0.725, 2019: 0.921, 2020: 3.02, 2021: 0.594,
    2022: -0.648, 2023: 1.562, 2024: 1.247,
}


def _ann_to_daily(annual: float, trading_days: int = 252) -> float:
    return (1 + annual) ** (1 / trading_days) - 1


def _fetch_yahoo(symbol: str, start_iso: str, end_iso: str) -> List[float]:
    """Fetch daily closing prices from Yahoo Finance (no API key)."""
    try:
        p1 = int(datetime.fromisoformat(start_iso.replace("Z", "")).timestamp())
        p2 = int(datetime.fromisoformat(end_iso.replace("Z", "")).timestamp())
        url = (f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
               f"?period1={p1}&period2={p2}&interval=1d")
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 NEXUS/1.0"})
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read())
        closes = data["chart"]["result"][0]["indicators"]["quote"][0]["close"]
        return [float(c) for c in closes if c is not None]
    except Exception:
        return []


def _fetch_binance_daily(symbol: str, start_iso: str, end_iso: str) -> List[float]:
    """Fetch daily closes from Binance public REST."""
    try:
        def to_ms(s: str) -> int:
            return int(datetime.fromisoformat(s.replace("Z", "")).timestamp() * 1000)
        url = (f"https://api.binance.com/api/v3/klines"
               f"?symbol={symbol}&interval=1d"
               f"&startTime={to_ms(start_iso)}&endTime={to_ms(end_iso)}&limit=365")
        req = urllib.request.Request(url, headers={"User-Agent": "NEXUS/1.0"})
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read())
        return [float(k[4]) for k in data]  # close prices
    except Exception:
        return []


def _prices_to_returns(prices: List[float]) -> List[float]:
    if len(prices) < 2:
        return []
    return [(prices[i] - prices[i - 1]) / prices[i - 1] for i in range(1, len(prices))]


def _compute_metrics(returns: List[float], freq: float) -> Dict[str, float]:
    """Compute Sharpe, CAGR, MaxDD from a returns series."""
    if not returns:
        return {"sharpe": 0.0, "cagr": 0.0, "max_drawdown": 0.0, "total_return": 0.0, "n": 0}
    mu = statistics.mean(returns)
    try:
        sd = statistics.stdev(returns)
    except Exception:
        sd = 0.0
    sharpe = (mu / sd * math.sqrt(freq)) if sd > 0 else 0.0
    equity = 1.0
    peak = 1.0
    mdd = 0.0
    for r in returns:
        equity *= (1 + r)
        if equity > peak:
            peak = equity
        dd = (peak - equity) / peak
        if dd > mdd:
            mdd = dd
    years = len(returns) / freq
    cagr = (equity ** (1 / years) - 1) if years > 0 and equity > 0 else 0.0
    return {
        "sharpe": round(sharpe, 4),
        "cagr": round(cagr, 4),
        "max_drawdown": round(mdd, 4),
        "total_return": round(equity - 1, 4),
        "n": len(returns),
    }


def build_benchmark_comparison(
    strategy_returns: List[float],
    strategy_name: str = "NEXUS Strategy",
    strategy_freq: float = 8760,
    start: str = "2024-01-01",
    end: str = "2025-01-01",
) -> Dict[str, Any]:
    """
    Compare strategy against real benchmarks.
    Returns full comparison dict for dashboard display.
    """
    strategy_m = _compute_metrics(strategy_returns, strategy_freq)

    benchmarks = []

    # ── S&P 500 (Yahoo Finance) ───────────────────────────────────────────
    spy_prices = _fetch_yahoo("SPY", start, end)
    if spy_prices:
        spy_ret = _prices_to_returns(spy_prices)
        m = _compute_metrics(spy_ret, 252)
        benchmarks.append({
            "name": "S&P 500", "ticker": "SPY",
            "source": "Yahoo Finance (live)", **m,
            "beat_sharpe": strategy_m["sharpe"] > m["sharpe"],
            "beat_cagr": strategy_m["cagr"] > m["cagr"],
        })
    else:
        # Fallback hardcoded
        start_year = int(start[:4])
        ann = SP500_RETURNS.get(start_year, 0.23)
        daily = [_ann_to_daily(ann)] * 252
        m = _compute_metrics(daily, 252)
        benchmarks.append({
            "name": "S&P 500", "ticker": "SPY",
            "source": f"Hardcoded {start_year} annual", **m,
            "beat_sharpe": strategy_m["sharpe"] > m["sharpe"],
            "beat_cagr": strategy_m["cagr"] > m["cagr"],
        })

    # ── Bitcoin Buy & Hold (Binance) ─────────────────────────────────────
    btc_prices = _fetch_binance_daily("BTCUSDT", start, end)
    if btc_prices:
        btc_ret = _prices_to_returns(btc_prices)
        m = _compute_metrics(btc_ret, 365)
        benchmarks.append({
            "name": "Bitcoin Buy & Hold", "ticker": "BTCUSDT",
            "source": "Binance (live)", **m,
            "beat_sharpe": strategy_m["sharpe"] > m["sharpe"],
            "beat_cagr": strategy_m["cagr"] > m["cagr"],
        })
    else:
        start_year = int(start[:4])
        ann = BTC_APPROX_RETURNS.get(start_year, 1.25)
        daily = [_ann_to_daily(ann, 365)] * 365
        m = _compute_metrics(daily, 365)
        benchmarks.append({
            "name": "Bitcoin Buy & Hold", "ticker": "BTCUSDT",
            "source": f"Hardcoded {start_year} annual", **m,
            "beat_sharpe": strategy_m["sharpe"] > m["sharpe"],
            "beat_cagr": strategy_m["cagr"] > m["cagr"],
        })

    # ── Renaissance Medallion (hardcoded) ────────────────────────────────
    start_year = int(start[:4])
    ren_ann = RENAISSANCE_RETURNS.get(start_year, 0.30)
    ren_daily = [_ann_to_daily(ren_ann)] * 252
    m = _compute_metrics(ren_daily, 252)
    benchmarks.append({
        "name": "Renaissance Medallion*", "ticker": "REN",
        "source": "Public filings (gross, before 5%+44% fees)",
        "note": "*Gross returns before fees. Net ~39%/yr historically.",
        **m,
        "beat_sharpe": strategy_m["sharpe"] > m["sharpe"],
        "beat_cagr": strategy_m["cagr"] > m["cagr"],
    })

    # ── 60/40 Portfolio ───────────────────────────────────────────────────
    sp_ann = SP500_RETURNS.get(start_year, 0.23)
    bond_ann = 0.042  # ~US 10yr yield
    portfolio_ann = 0.60 * sp_ann + 0.40 * bond_ann
    port_daily = [_ann_to_daily(portfolio_ann)] * 252
    m = _compute_metrics(port_daily, 252)
    benchmarks.append({
        "name": "60/40 Portfolio", "ticker": "60/40",
        "source": "60% S&P + 40% US Bonds", **m,
        "beat_sharpe": strategy_m["sharpe"] > m["sharpe"],
        "beat_cagr": strategy_m["cagr"] > m["cagr"],
    })

    # ── Ethereum Buy & Hold ───────────────────────────────────────────────
    eth_prices = _fetch_binance_daily("ETHUSDT", start, end)
    if eth_prices:
        eth_ret = _prices_to_returns(eth_prices)
        m = _compute_metrics(eth_ret, 365)
        benchmarks.append({
            "name": "Ethereum Buy & Hold", "ticker": "ETHUSDT",
            "source": "Binance (live)", **m,
            "beat_sharpe": strategy_m["sharpe"] > m["sharpe"],
            "beat_cagr": strategy_m["cagr"] > m["cagr"],
        })

    n_beat_sharpe = sum(1 for b in benchmarks if b.get("beat_sharpe", False))
    n_beat_cagr = sum(1 for b in benchmarks if b.get("beat_cagr", False))

    if strategy_m["sharpe"] <= 0:
        verdict = "UNDERPERFORMS"
    elif n_beat_sharpe >= len(benchmarks) * 0.6:
        verdict = "OUTPERFORMS"
    elif n_beat_sharpe >= len(benchmarks) * 0.3:
        verdict = "MIXED"
    else:
        verdict = "UNDERPERFORMS"

    return {
        "ts": datetime.now(timezone.utc).isoformat(),
        "strategy": {"name": strategy_name, "freq": strategy_freq, **strategy_m},
        "benchmarks": benchmarks,
        "n_benchmarks": len(benchmarks),
        "n_beat_sharpe": n_beat_sharpe,
        "n_beat_cagr": n_beat_cagr,
        "overall_verdict": verdict,
        "start": start,
        "end": end,
    }
