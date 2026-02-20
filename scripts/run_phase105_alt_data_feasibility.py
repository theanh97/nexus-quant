#!/usr/bin/env python3
"""
Phase 105: Alternative Data Source Feasibility Study
=====================================================
Price-based signals are exhausted (Phase 93-96). Next alpha requires
fundamentally different data sources. This script:

  A) Tests what FREE APIs are actually accessible right now
  B) Checks data quality and history length for each source
  C) Estimates signal construction feasibility
  D) Prioritizes sources by effort/reward ratio

Candidate data sources:
  1. On-chain (Glassnode, CryptoQuant, Blockchain.com)
  2. Social sentiment (LunarCrush, Santiment, Fear&Greed)
  3. Exchange flows (CryptoQuant exchange netflow)
  4. Options/derivatives (Deribit via Coinalyze or similar)
  5. Google Trends (as contrarian indicator)
  6. Binance API extras (already have: funding, OI, L/S ratios)
"""

import json, os, sys, time
from pathlib import Path
from datetime import datetime

PROJ = "/Users/qtmobile/Desktop/Nexus - Quant Trading "
sys.path.insert(0, PROJ)

OUT_DIR = os.path.join(PROJ, "artifacts", "phase105")
os.makedirs(OUT_DIR, exist_ok=True)


def log(msg):
    print(f"[P105] {msg}", flush=True)


def test_api(name, url, headers=None):
    """Test if an API endpoint is accessible and returns data."""
    import urllib.request
    import urllib.error
    try:
        req = urllib.request.Request(url, headers=headers or {})
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = resp.read().decode("utf-8")
            return {"status": "OK", "bytes": len(data), "sample": data[:500]}
    except urllib.error.HTTPError as e:
        return {"status": f"HTTP_{e.code}", "error": str(e)}
    except Exception as e:
        return {"status": "ERROR", "error": str(e)}


if __name__ == "__main__":
    t0 = time.time()
    report = {"phase": 105, "title": "Alternative Data Feasibility Study"}

    # ════════════════════════════════════
    # 1. BINANCE EXTRAS (already integrated, check what we use vs what's available)
    # ════════════════════════════════════
    log("=" * 60)
    log("1. BINANCE API — Additional endpoints we don't use")
    log("=" * 60)

    binance_tests = {
        "long_short_ratio_top": {
            "url": "https://fapi.binance.com/futures/data/topLongShortAccountRatio?symbol=BTCUSDT&period=1h&limit=5",
            "signal_idea": "Contrarian: fade retail crowd",
            "history": "30 days on Binance API",
        },
        "global_long_short": {
            "url": "https://fapi.binance.com/futures/data/globalLongShortAccountRatio?symbol=BTCUSDT&period=1h&limit=5",
            "signal_idea": "Global L/S divergence from top traders",
            "history": "30 days",
        },
        "taker_buy_sell_volume": {
            "url": "https://fapi.binance.com/futures/data/takerlongshortRatio?symbol=BTCUSDT&period=1h&limit=5",
            "signal_idea": "Taker buy volume momentum as directional signal",
            "history": "30 days",
        },
        "open_interest_hist": {
            "url": "https://fapi.binance.com/futures/data/openInterestHist?symbol=BTCUSDT&period=1h&limit=5",
            "signal_idea": "OI changes predict liquidation cascades",
            "history": "30 days",
        },
        "klines_1m_depth": {
            "url": "https://fapi.binance.com/fapi/v1/depth?symbol=BTCUSDT&limit=5",
            "signal_idea": "Order book imbalance at bid/ask",
            "history": "Real-time only (no historical)",
        },
        "agg_trades": {
            "url": "https://fapi.binance.com/fapi/v1/aggTrades?symbol=BTCUSDT&limit=5",
            "signal_idea": "Large trade detection (whale watching)",
            "history": "Real-time stream + limited historical",
        },
    }

    binance_results = {}
    for name, cfg in binance_tests.items():
        result = test_api(name, cfg["url"])
        binance_results[name] = {**cfg, "test": result}
        status = result["status"]
        log(f"  {name}: {status} | Signal: {cfg['signal_idea']} | History: {cfg['history']}")

    report["binance_extras"] = binance_results

    # ════════════════════════════════════
    # 2. FREE ON-CHAIN APIs
    # ════════════════════════════════════
    log("\n" + "=" * 60)
    log("2. ON-CHAIN DATA — Free APIs")
    log("=" * 60)

    onchain_tests = {
        "blockchain_com_hashrate": {
            "url": "https://api.blockchain.info/charts/hash-rate?timespan=30days&format=json",
            "signal_idea": "Hashrate momentum as miner confidence proxy",
            "history": "Years of daily data",
            "coins": "BTC only",
        },
        "blockchain_com_mempool": {
            "url": "https://api.blockchain.info/charts/mempool-size?timespan=30days&format=json",
            "signal_idea": "Mempool congestion predicts volatility/activity",
            "history": "Years of daily data",
            "coins": "BTC only",
        },
        "blockchain_com_tx_count": {
            "url": "https://api.blockchain.info/charts/n-transactions?timespan=30days&format=json",
            "signal_idea": "Transaction count as network activity proxy",
            "history": "Years of daily data",
            "coins": "BTC only",
        },
        "coingecko_btc_market": {
            "url": "https://api.coingecko.com/api/v3/coins/bitcoin/market_chart?vs_currency=usd&days=30",
            "signal_idea": "Market cap + volume from alternative source (cross-validate)",
            "history": "Years (rate limited at 10-30 calls/min)",
            "coins": "All major coins",
        },
        "fear_greed_index": {
            "url": "https://api.alternative.me/ftp/?limit=30&format=json",
            "signal_idea": "Contrarian: buy at extreme fear, sell at extreme greed",
            "history": "Since 2018 (daily)",
            "coins": "BTC (crypto market proxy)",
        },
    }

    onchain_results = {}
    for name, cfg in onchain_tests.items():
        result = test_api(name, cfg["url"])
        onchain_results[name] = {**cfg, "test": result}
        status = result["status"]
        log(f"  {name}: {status} | Signal: {cfg['signal_idea']} | Coins: {cfg['coins']}")

    report["onchain"] = onchain_results

    # ════════════════════════════════════
    # 3. SOCIAL / SENTIMENT
    # ════════════════════════════════════
    log("\n" + "=" * 60)
    log("3. SOCIAL / SENTIMENT DATA")
    log("=" * 60)

    sentiment_tests = {
        "coingecko_trending": {
            "url": "https://api.coingecko.com/api/v3/search/trending",
            "signal_idea": "Trending coins = crowd momentum (contrarian or follow)",
            "history": "Real-time snapshot only",
            "coins": "All",
        },
    }

    sentiment_results = {}
    for name, cfg in sentiment_tests.items():
        result = test_api(name, cfg["url"])
        sentiment_results[name] = {**cfg, "test": result}
        status = result["status"]
        log(f"  {name}: {status}")

    report["sentiment"] = sentiment_results

    # ════════════════════════════════════
    # 4. DERIVATIVES DATA
    # ════════════════════════════════════
    log("\n" + "=" * 60)
    log("4. DERIVATIVES DATA (Options/Futures)")
    log("=" * 60)

    derivatives_tests = {
        "binance_futures_premium": {
            "url": "https://fapi.binance.com/fapi/v1/premiumIndex?symbol=BTCUSDT",
            "signal_idea": "Futures premium = leveraged long/short demand",
            "history": "Real-time",
            "coins": "All Binance perps",
        },
        "coinalyze_oi": {
            "url": "https://api.coinalyze.net/v1/open-interest?symbols=BTCUSD_PERP.6&api_key=free",
            "signal_idea": "Cross-exchange OI aggregation (more complete than single exchange)",
            "history": "Varies",
            "coins": "Major coins",
        },
    }

    derivatives_results = {}
    for name, cfg in derivatives_tests.items():
        result = test_api(name, cfg["url"])
        derivatives_results[name] = {**cfg, "test": result}
        status = result["status"]
        log(f"  {name}: {status} | Signal: {cfg['signal_idea']}")

    report["derivatives"] = derivatives_results

    # ════════════════════════════════════
    # 5. FEASIBILITY RANKING
    # ════════════════════════════════════
    log("\n" + "=" * 60)
    log("5. FEASIBILITY RANKING")
    log("=" * 60)

    # Score each data source on:
    # - Accessibility (is the API working? 0-3)
    # - History depth (can we backtest? 0-3)
    # - Signal novelty (is it orthogonal to our signals? 0-3)
    # - Implementation effort (hours to integrate, inverse scored 0-3)

    feasibility = [
        {
            "source": "Fear & Greed Index",
            "api": "alternative.me",
            "accessible": 3, "history": 3, "novelty": 2, "effort": 3,
            "total": 11,
            "notes": "Daily since 2018. Contrarian signal. Simple to integrate. But daily-only may miss intraday dynamics.",
        },
        {
            "source": "Binance Taker Buy/Sell Volume",
            "api": "fapi.binance.com",
            "accessible": 3, "history": 1, "novelty": 1, "effort": 3,
            "total": 8,
            "notes": "Already tested in Phase 94 — taker_buy_ratio signal was DEAD. 30-day API limit.",
        },
        {
            "source": "Binance Long/Short Ratio",
            "api": "fapi.binance.com",
            "accessible": 3, "history": 1, "novelty": 1, "effort": 3,
            "total": 8,
            "notes": "Already using in backtest engine (positioning data). 30-day API limit for historical.",
        },
        {
            "source": "Blockchain.com On-Chain (BTC)",
            "api": "blockchain.info",
            "accessible": 3, "history": 3, "novelty": 3, "effort": 2,
            "total": 11,
            "notes": "Hashrate + mempool + tx count. Years of data. BTC-only but truly orthogonal to price signals.",
        },
        {
            "source": "CoinGecko Market Data",
            "api": "coingecko.com",
            "accessible": 2, "history": 3, "novelty": 1, "effort": 2,
            "total": 8,
            "notes": "Rate limited (10-30 calls/min). Volume and market cap available but likely correlated with price signals.",
        },
        {
            "source": "Binance Futures Premium Index",
            "api": "fapi.binance.com",
            "accessible": 3, "history": 1, "novelty": 2, "effort": 3,
            "total": 9,
            "notes": "Real-time premium = basis. Related to funding but different signal. 30-day limit.",
        },
        {
            "source": "Deribit Options (via Coinalyze)",
            "api": "coinalyze.net",
            "accessible": 1, "history": 1, "novelty": 3, "effort": 1,
            "total": 6,
            "notes": "Truly orthogonal (implied vol, put/call ratio, skew). But hard to get free historical data.",
        },
        {
            "source": "Google Trends",
            "api": "pytrends library",
            "accessible": 2, "history": 3, "novelty": 2, "effort": 2,
            "total": 9,
            "notes": "Weekly data, years of history. Retail interest proxy. Rate limited. Needs pytrends install.",
        },
    ]

    feasibility.sort(key=lambda x: x["total"], reverse=True)
    report["feasibility_ranking"] = feasibility

    log(f"\n{'Source':>35} | Acc | Hist | Nov | Eff | TOTAL")
    log(f"{'-'*35}-+-----+------+-----+-----+------")
    for f in feasibility:
        log(f"{f['source']:>35} | {f['accessible']:>3} | {f['history']:>4} | {f['novelty']:>3} | {f['effort']:>3} | {f['total']:>5}")
        log(f"{'':>35}   → {f['notes'][:80]}")

    # ════════════════════════════════════
    # 6. RECOMMENDED NEXT STEPS
    # ════════════════════════════════════
    log("\n" + "=" * 60)
    log("6. RECOMMENDED NEXT STEPS")
    log("=" * 60)

    recommendations = {
        "tier_1_immediate": {
            "source": "Fear & Greed Index + Blockchain.com On-Chain",
            "why": "Highest feasibility score (11/12 each). Both truly orthogonal to price-based signals.",
            "action": "Build data provider for alternative.me Fear&Greed + blockchain.info on-chain metrics. Create contrarian signal (buy fear, sell greed) and on-chain momentum signal (rising hashrate + tx count = bullish).",
            "expected_time": "4-8 hours for data provider + signal + backtest",
            "caveat": "Daily resolution only — may not improve hourly Sharpe. Best as monthly/weekly filter.",
        },
        "tier_2_medium": {
            "source": "Binance Futures Premium Index",
            "why": "Accessible, real-time, different from funding rate. Basis trade is a known alpha source.",
            "action": "Extend Binance data provider to fetch premium index. Create basis momentum signal.",
            "expected_time": "2-4 hours",
            "caveat": "30-day API history limit. Need to build local cache for backtesting.",
        },
        "tier_3_long_term": {
            "source": "Options Data (Deribit/Coinalyze)",
            "why": "Most orthogonal source (implied vol, skew, put/call). But hardest to access.",
            "action": "Research paid API options. Consider Deribit direct API or Tardis.dev for historical.",
            "expected_time": "1-2 days + potential paid subscription",
            "caveat": "BTC/ETH only. Limited history. May need separate infrastructure.",
        },
        "not_recommended": {
            "sources": ["Taker volume ratio (already tested DEAD)", "L/S ratio (already using)", "CoinGecko (rate limited, not novel)"],
            "why": "Already tested or not orthogonal enough to move the needle.",
        },
    }
    report["recommendations"] = recommendations

    for tier, rec in recommendations.items():
        if tier == "not_recommended":
            log(f"\n  NOT RECOMMENDED: {rec['sources']}")
        else:
            log(f"\n  {tier}: {rec['source']}")
            log(f"    Why: {rec['why']}")
            log(f"    Time: {rec['expected_time']}")

    # ════════════════════════════════════
    # SAVE
    # ════════════════════════════════════
    elapsed = round(time.time() - t0, 1)
    report["elapsed_seconds"] = elapsed

    out_path = os.path.join(OUT_DIR, "phase105_report.json")
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)

    log("\n" + "=" * 60)
    log(f"Phase 105 COMPLETE in {elapsed}s → {out_path}")
    log("=" * 60)
    log("\nTOP PRIORITY: Fear & Greed + On-Chain → truly orthogonal, free, years of history")
