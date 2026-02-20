"""
Commodity CTA Walk-Forward Validation
======================================
Validates the CTA strategy across different time periods to check for robustness.

Walk-forward design:
  - Train: fit nothing (strategy is parameter-free EMA crossover)
  - IS  : 2007-2015 (includes 2008 oil boom/bust, 2011-2014 cycles)
  - OOS1: 2016-2020 (oil bear 2015-2016, COVID 2020)
  - OOS2: 2021-2026 (commodity bull 2021-2022, recent chop)

Benchmark: SG CTA Index (Sharpe ~0.5 over this period)
Target:    MIN Sharpe > 0.3 across all periods

Usage:
    cd "/Users/qtmobile/Desktop/Nexus - Quant Trading "
    python3 scripts/run_commodity_cta_wf.py
"""
from __future__ import annotations

import json
import logging
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("cta_wf")


def compute_metrics(equity_curve, n_per_year=252):
    """Compute Sharpe, CAGR, MaxDD from equity curve."""
    if len(equity_curve) < 10:
        return {"sharpe": 0.0, "cagr": 0.0, "max_drawdown": 1.0}

    rets = []
    for i in range(1, len(equity_curve)):
        e0 = equity_curve[i - 1]
        e1 = equity_curve[i]
        if e0 > 0 and e1 == e1:
            rets.append(e1 / e0 - 1.0)

    if not rets:
        return {"sharpe": 0.0, "cagr": 0.0, "max_drawdown": 1.0}

    n = len(rets)
    mn = sum(rets) / n
    var = sum((r - mn) ** 2 for r in rets) / n
    std = math.sqrt(var) if var > 0 else 1e-10
    sharpe = (mn / std) * math.sqrt(n_per_year)

    final_eq = equity_curve[-1]
    cagr = final_eq ** (n_per_year / n) - 1 if n > 0 and final_eq > 0 else 0.0

    peak = 1.0
    max_dd = 0.0
    for e in equity_curve:
        if e > peak:
            peak = e
        dd = (peak - e) / peak if peak > 0 else 0.0
        if dd > max_dd:
            max_dd = dd

    return {"sharpe": sharpe, "cagr": cagr, "max_drawdown": max_dd}


def slice_dataset(dataset, start_date: str, end_date: str):
    """Return a dataset sliced to [start_date, end_date]."""
    from nexus_quant.data.schema import MarketDataset
    import hashlib

    start_ts = int(
        datetime.strptime(start_date, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp()
    )
    end_ts = int(
        datetime.strptime(end_date, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp()
    )

    mask = [start_ts <= ts <= end_ts for ts in dataset.timeline]
    indices = [i for i, m in enumerate(mask) if m]

    if not indices:
        raise ValueError(f"No bars in [{start_date}, {end_date}]")

    i0, i1 = indices[0], indices[-1] + 1
    timeline = dataset.timeline[i0:i1]
    perp_close = {sym: dataset.perp_close[sym][i0:i1] for sym in dataset.symbols}
    features = {
        feat_name: {
            sym: arr[i0:i1] for sym, arr in feat_data.items()
        }
        for feat_name, feat_data in dataset.features.items()
    }

    fp = hashlib.md5(f"{start_date}:{end_date}:{dataset.symbols}".encode()).hexdigest()[:8]

    return MarketDataset(
        provider=dataset.provider,
        timeline=timeline,
        symbols=dataset.symbols,
        perp_close=perp_close,
        spot_close=None,
        funding={},
        fingerprint=fp,
        market_type=dataset.market_type,
        features=features,
        meta={**dataset.meta, "wf_start": start_date, "wf_end": end_date},
    )


def main():
    logger.info("=" * 60)
    logger.info("NEXUS Commodity CTA — Walk-Forward Validation")
    logger.info("=" * 60)

    # ── Load data ─────────────────────────────────────────────────────────────
    from nexus_quant.projects.commodity_cta.providers.yahoo_futures import (
        YahooFuturesProvider, DEFAULT_SYMBOLS,
    )
    from nexus_quant.projects.commodity_cta.strategies import STRATEGIES
    from nexus_quant.backtest.engine import BacktestEngine, BacktestConfig
    from nexus_quant.projects.commodity_cta.costs.futures_fees import commodity_futures_cost_model

    logger.info("\nLoading commodity data (from cache)...")
    cfg = {
        "symbols": DEFAULT_SYMBOLS,
        "start": "2005-01-01",
        "end": "2026-02-20",
        "cache_dir": "data/cache/yahoo_futures",
        "min_valid_bars": 300,
        "request_delay": 0.1,
    }
    provider = YahooFuturesProvider(cfg, seed=42)
    full_dataset = provider.load()
    logger.info(
        f"  Full dataset: {len(full_dataset.symbols)} symbols × {len(full_dataset.timeline)} bars"
    )

    cost_model = commodity_futures_cost_model(slippage_bps=5.0, commission_bps=1.0, spread_bps=1.0)

    # ── Walk-forward windows ───────────────────────────────────────────────────
    periods = [
        ("IS",   "2007-07-01", "2015-12-31"),  # In-sample (2008 oil boom/bust)
        ("OOS1", "2016-01-01", "2020-12-31"),  # OOS1 (oil bear + COVID)
        ("OOS2", "2021-01-01", "2026-02-20"),  # OOS2 (commodity bull + chop)
        ("FULL", "2007-07-01", "2026-02-20"),  # Full period
    ]

    results = {}
    logger.info("\nRunning walk-forward validation:")
    logger.info(f"{'Period':6s}  {'Start':12s}  {'End':12s}  {'Bars':5s}  {'Sharpe':7s}  {'CAGR':8s}  {'MaxDD':7s}")
    logger.info("-" * 65)

    for label, start, end in periods:
        try:
            ds_slice = slice_dataset(full_dataset, start, end)
        except ValueError as e:
            logger.warning(f"  {label}: {e}")
            continue

        strategy = STRATEGIES["cta_trend"]()
        engine = BacktestEngine(BacktestConfig(costs=cost_model))

        try:
            result = engine.run(ds_slice, strategy, seed=42)
        except Exception as e:
            logger.warning(f"  {label}: backtest failed: {e}")
            continue

        metrics = compute_metrics(result.equity_curve)
        results[label] = metrics

        logger.info(
            f"{label:6s}  {start:12s}  {end:12s}  {len(ds_slice.timeline):5d}  "
            f"{metrics['sharpe']:+7.3f}  {metrics['cagr']*100:+7.2f}%  "
            f"{metrics['max_drawdown']*100:6.1f}%"
        )

    # ── Verdict ────────────────────────────────────────────────────────────────
    logger.info("\n" + "─" * 65)
    logger.info("Walk-Forward Verdict:")

    if "OOS1" not in results or "OOS2" not in results:
        logger.error("  Missing OOS periods — cannot validate")
        return

    sharpes = {k: results[k]["sharpe"] for k in results}
    oos_sharpes = [sharpes.get("OOS1", -99), sharpes.get("OOS2", -99)]
    oos_min = min(oos_sharpes)
    oos_avg = sum(oos_sharpes) / len(oos_sharpes)

    logger.info(f"  IS  Sharpe : {sharpes.get('IS', 0):+.3f}")
    logger.info(f"  OOS1 Sharpe: {sharpes.get('OOS1', 0):+.3f}")
    logger.info(f"  OOS2 Sharpe: {sharpes.get('OOS2', 0):+.3f}")
    logger.info(f"  FULL Sharpe: {sharpes.get('FULL', 0):+.3f}")
    logger.info(f"  OOS_MIN    : {oos_min:+.3f}  (target > 0.3)")
    logger.info(f"  OOS_AVG    : {oos_avg:+.3f}  (target > 0.5)")

    # Benchmarks
    logger.info("\n  Benchmarks:")
    logger.info("    SG CTA Index  : ~0.5  (diversified, 50+ markets)")
    logger.info("    Buy-and-hold  : ~0.2  (commodity basket)")
    logger.info("    200-day MA    : ~0.4  (simple trend)")
    logger.info(f"    NEXUS Comm CTA: {sharpes.get('FULL', 0):+.3f} (13 commodities only)")

    if oos_min > 0.5:
        verdict = "STRONG PASS — OOS MIN > 0.5"
    elif oos_min > 0.3:
        verdict = "PASS — OOS MIN > 0.3"
    elif oos_min > 0.0:
        verdict = "WEAK PASS — OOS MIN positive but < 0.3"
    else:
        verdict = "FAIL — OOS MIN negative"

    logger.info(f"\n  Verdict: {verdict}")

    # Save results
    output = {
        "strategy": "cta_trend",
        "run_date": datetime.now().isoformat(),
        "periods": results,
        "oos_min": oos_min,
        "oos_avg": oos_avg,
        "verdict": verdict,
        "notes": "EMA(12/26 + 20/50) + mom_20d, monthly rebalance, vol-targeting, 7bps RT",
    }
    out_path = ROOT / "artifacts" / "commodity_cta_wf.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    logger.info(f"\n  Results saved to {out_path}")


if __name__ == "__main__":
    main()
