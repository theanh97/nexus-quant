"""
Commodity CTA Phase 1 Validation Script
==========================================
Tests the full pipeline end-to-end:
  1. Yahoo Finance data download for 13 commodities
  2. Feature engineering validation (check all features computed)
  3. Strategy instantiation and sanity check
  4. Quick backtest on IS period using BacktestEngine
  5. Report Sharpe vs benchmark and vs SG CTA Index target

Usage:
    cd "/Users/qtmobile/Desktop/Nexus - Quant Trading "
    python3 scripts/run_commodity_cta_phase1.py                  # Full Yahoo download
    python3 scripts/run_commodity_cta_phase1.py --synthetic      # Synthetic data (no download)
    python3 scripts/run_commodity_cta_phase1.py --quick          # 5 symbols, Yahoo
    python3 scripts/run_commodity_cta_phase1.py --no-download    # Cache only

Note: Yahoo Finance may rate-limit (HTTP 429). Use --synthetic for pipeline testing.
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

# ── Setup path ────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("commodity_cta.validation")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Commodity CTA Phase 1 Validation")
    p.add_argument("--start", default="2005-01-01", help="Start date (YYYY-MM-DD)")
    p.add_argument("--end", default="2026-02-20", help="End date (YYYY-MM-DD)")
    p.add_argument("--quick", action="store_true", help="Quick mode: 5 symbols only")
    p.add_argument("--no-download", action="store_true", help="Cache-only mode")
    p.add_argument("--synthetic", action="store_true",
                   help="Use synthetic data (no Yahoo download — for pipeline testing)")
    p.add_argument("--strategy", default="cta_ensemble",
                   choices=["cta_trend", "cta_carry", "cta_mom_value", "cta_ensemble"],
                   help="Strategy to backtest")
    return p.parse_args()


def make_synthetic_dataset(symbols: List[str], n_bars: int = 3000) -> Any:
    """
    Generate a synthetic MarketDataset mimicking commodity futures data.
    Used for pipeline validation when Yahoo Finance is rate-limited.
    Random walk prices with realistic volatility.
    """
    import random
    from nexus_quant.data.schema import MarketDataset
    from nexus_quant.projects.commodity_cta.features import compute_features

    random.seed(42)
    base_ts = 1104537600  # 2005-01-01 UTC
    timeline = [base_ts + i * 86400 for i in range(n_bars)]

    # Simulate OHLCV for each symbol
    aligned: Dict[str, Dict[str, List[float]]] = {}
    for sym in symbols:
        price = 100.0 + random.uniform(-20, 80)  # random starting price
        daily_vol = random.uniform(0.01, 0.025)  # 1-2.5% daily vol
        closes, opens, highs, lows, vols = [], [], [], [], []
        for _ in range(n_bars):
            ret = random.gauss(0.0002, daily_vol)  # slight positive drift
            price = max(price * (1 + ret), 1.0)
            intraday_range = price * daily_vol * 0.5
            o = price * (1 + random.gauss(0, 0.003))
            h = price + abs(random.gauss(0, intraday_range))
            lo = price - abs(random.gauss(0, intraday_range))
            v = abs(random.gauss(1e6, 5e5))
            closes.append(price)
            opens.append(o)
            highs.append(h)
            lows.append(lo)
            vols.append(v)
        aligned[sym] = {"close": closes, "open": opens, "high": highs, "low": lows, "volume": vols}

    features = compute_features(aligned, symbols, timeline)

    return MarketDataset(
        provider="synthetic_commodity_v1",
        timeline=timeline,
        symbols=symbols,
        perp_close={sym: aligned[sym]["close"] for sym in symbols},
        spot_close=None,
        funding={},
        fingerprint="synthetic",
        market_type="commodity",
        features=features,
        meta={"synthetic": True, "n_bars": n_bars},
    )


def main() -> None:
    args = parse_args()
    logger.info("=" * 60)
    logger.info("NEXUS Commodity CTA — Phase 1 Validation")
    logger.info("=" * 60)

    symbols = (
        ["CL=F", "GC=F", "ZC=F", "KC=F", "HG=F"]
        if args.quick
        else ["CL=F", "NG=F", "BZ=F", "GC=F", "SI=F", "HG=F", "PL=F",
              "ZW=F", "ZC=F", "ZS=F", "KC=F", "SB=F", "CT=F"]
    )
    delay = 0.1 if args.no_download else 0.5

    # ── Step 1: Load data ─────────────────────────────────────────────────────
    t0 = time.time()
    if args.synthetic:
        logger.info(f"\nStep 1: Generating synthetic data ({len(symbols)} symbols, 3000 bars)")
        logger.info("  [SYNTHETIC MODE — no Yahoo download]")
        dataset = make_synthetic_dataset(symbols, n_bars=3000)
    else:
        logger.info(f"\nStep 1: Loading data ({len(symbols)} symbols, {args.start} → {args.end})")
        logger.info("  Note: Yahoo Finance may rate-limit. Use --synthetic for fast pipeline test.")
        from nexus_quant.projects.commodity_cta.providers.yahoo_futures import YahooFuturesProvider
        cfg = {
            "provider": "yahoo_futures_v1",
            "symbols": symbols,
            "start": args.start,
            "end": args.end,
            "cache_dir": "data/cache/yahoo_futures",
            "min_valid_bars": 300,
            "request_delay": delay,
        }
        provider = YahooFuturesProvider(cfg, seed=42)
        dataset = provider.load()

    elapsed = time.time() - t0
    logger.info(f"  ✓ Dataset loaded in {elapsed:.1f}s")
    logger.info(f"  ✓ Symbols: {dataset.symbols}")
    logger.info(f"  ✓ Bars: {len(dataset.timeline)}")
    logger.info(f"  ✓ Market type: {dataset.market_type}")
    logger.info(f"  ✓ Has funding: {dataset.has_funding}")
    logger.info(f"  ✓ Features: {sorted(dataset.features.keys())}")

    # ── Step 2: Feature validation ─────────────────────────────────────────────
    logger.info("\nStep 2: Validating features")
    required_features = [
        "mom_20d", "mom_60d", "mom_120d", "atr_14", "rv_20d",
        "rsi_14", "zscore_60d", "ewma_signal", "vol_mom_z",
    ]
    missing = [f for f in required_features if f not in dataset.features]
    if missing:
        logger.error(f"  ✗ Missing features: {missing}")
    else:
        logger.info(f"  ✓ All {len(required_features)} required features present")

    # Spot check: last bar values for first symbol
    sym0 = dataset.symbols[0]
    idx_last = len(dataset.timeline) - 1
    logger.info(f"\n  Spot check: {sym0} at bar {idx_last}")
    for feat_name in ["mom_120d", "atr_14", "rv_20d", "rsi_14", "vol_mom_z"]:
        feat = dataset.features.get(feat_name, {}).get(sym0, [])
        val = feat[idx_last] if feat and idx_last < len(feat) else None
        logger.info(f"    {feat_name:15s}: {val:.4f}" if val is not None else f"    {feat_name:15s}: N/A")

    # ── Step 3: Strategy instantiation ────────────────────────────────────────
    logger.info(f"\nStep 3: Instantiating strategy '{args.strategy}'")
    from nexus_quant.projects.commodity_cta.strategies import STRATEGIES

    StratClass = STRATEGIES.get(args.strategy)
    if StratClass is None:
        logger.error(f"  ✗ Unknown strategy: {args.strategy}")
        sys.exit(1)

    strategy = StratClass()
    logger.info(f"  ✓ Strategy created: {strategy.name}")

    # Test should_rebalance and target_weights on last bar
    idx_test = len(dataset.timeline) - 1
    should_reb = strategy.should_rebalance(dataset, idx_test)
    logger.info(f"  ✓ should_rebalance({idx_test}): {should_reb}")

    if should_reb:
        t0 = time.time()
        weights = strategy.target_weights(dataset, idx_test, {})
        elapsed = time.time() - t0
        logger.info(f"  ✓ target_weights computed in {elapsed*1000:.1f}ms")
        logger.info(f"  ✓ Positions: {len(weights)} symbols")
        if weights:
            gross = sum(abs(v) for v in weights.values())
            net = sum(weights.values())
            logger.info(f"  ✓ Gross leverage: {gross:.3f}")
            logger.info(f"  ✓ Net leverage: {net:.3f}")
            top5 = sorted(weights.items(), key=lambda x: abs(x[1]), reverse=True)[:5]
            for sym, w in top5:
                side = "LONG" if w > 0 else "SHORT"
                logger.info(f"     {sym:8s} {side:5s}: {w:+.4f}")

    # ── Step 4: Quick backtest ─────────────────────────────────────────────────
    logger.info("\nStep 4: Running backtest (full history)")
    try:
        from nexus_quant.backtest.engine import BacktestEngine, BacktestConfig
        from nexus_quant.projects.commodity_cta.costs.futures_fees import commodity_futures_cost_model

        cost_model = commodity_futures_cost_model(
            slippage_bps=5.0,
            commission_bps=1.0,
            spread_bps=1.0,
        )
        cfg_bt = BacktestConfig(costs=cost_model)
        engine = BacktestEngine(cfg_bt)

        strategy2 = StratClass()  # fresh instance for backtest
        t0 = time.time()
        result = engine.run(dataset, strategy2, seed=42)
        elapsed = time.time() - t0
        logger.info(f"  ✓ Backtest completed in {elapsed:.1f}s")

        # Compute metrics from equity curve
        equity = result.equity_curve
        rets = [equity[i] / equity[i - 1] - 1 for i in range(1, len(equity)) if equity[i - 1] > 0]

        if rets:
            n = len(rets)
            mean_ret = sum(rets) / n
            std_ret = math.sqrt(sum((r - mean_ret) ** 2 for r in rets) / n) if n > 1 else 0.0
            sharpe = (mean_ret / (std_ret + 1e-10)) * math.sqrt(252) if std_ret > 0 else 0.0
            cagr = (equity[-1] ** (252 / n)) - 1 if n > 0 else 0.0
            peak = 1.0
            max_dd = 0.0
            for e in equity:
                if e > peak:
                    peak = e
                dd = (peak - e) / peak
                if dd > max_dd:
                    max_dd = dd

            logger.info(f"\n  {'─' * 40}")
            logger.info(f"  Strategy : {args.strategy}")
            logger.info(f"  Period   : {args.start} → {args.end}")
            logger.info(f"  Bars     : {len(dataset.timeline)}")
            logger.info(f"  {'─' * 40}")
            logger.info(f"  Sharpe   : {sharpe:.3f}  (target > 0.8)")
            logger.info(f"  CAGR     : {cagr*100:.2f}%")
            logger.info(f"  Max DD   : {max_dd*100:.2f}%")
            logger.info(f"  Final eq : {equity[-1]:.4f}")
            logger.info(f"  {'─' * 40}")

            # Benchmark comparison
            logger.info(f"\n  Benchmarks:")
            logger.info(f"    Buy-and-hold basket: Sharpe ~0.2-0.3")
            logger.info(f"    Simple 200-day MA  : Sharpe ~0.4-0.5")
            logger.info(f"    SG CTA Index       : Sharpe ~0.5-0.7")
            logger.info(f"    NEXUS target       : Sharpe > 0.8")
            logger.info(f"    >>> THIS RESULT    : Sharpe = {sharpe:.3f}")

            # Pass/fail
            if sharpe > 0.8:
                logger.info(f"\n  ✅ PASS: Sharpe {sharpe:.3f} > 0.8 target!")
            elif sharpe > 0.5:
                logger.info(f"\n  ⚠️  PARTIAL: Sharpe {sharpe:.3f} beats SG CTA but below 0.8 target")
            elif sharpe > 0.3:
                logger.info(f"\n  ⚠️  PARTIAL: Sharpe {sharpe:.3f} above minimum 0.3 threshold")
            else:
                logger.info(f"\n  ❌ FAIL: Sharpe {sharpe:.3f} below 0.3 minimum threshold")

    except Exception as e:
        logger.error(f"  ✗ Backtest failed: {e}")
        import traceback
        traceback.print_exc()

    # ── Step 5: Correlation check (if multiple strategies) ────────────────────
    logger.info("\nStep 5: Vol momentum overlay test")
    vol_mom_z = dataset.features.get("vol_mom_z", {})
    if vol_mom_z:
        # Count how often avg vol_mom_z > threshold
        threshold = 1.5
        n_bars = len(dataset.timeline)
        n_crowded = 0
        for i in range(n_bars):
            vals = [vol_mom_z.get(sym, [0.0])[i] for sym in dataset.symbols
                    if i < len(vol_mom_z.get(sym, []))]
            if vals:
                avg_z = sum(vals) / len(vals)
                if avg_z > threshold:
                    n_crowded += 1
        pct_crowded = 100 * n_crowded / n_bars if n_bars > 0 else 0
        logger.info(f"  Vol tilt active: {n_crowded}/{n_bars} bars ({pct_crowded:.1f}%)")
        logger.info(f"  (Crypto: ~5% of time; commodity target: similar)")

    logger.info("\n" + "=" * 60)
    logger.info("Phase 1 validation complete!")
    logger.info("Next: Walk-forward validation (Phase 5)")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
