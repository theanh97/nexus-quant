"""
Phase 149: Cost Sensitivity Analysis
======================================
Test P148 champion at different cost levels to ensure robustness.
Default: 7 bps RT (1bp commission + 5bp slippage + 1bp spread)
"""
from __future__ import annotations

import json, logging, math, sys, hashlib
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
import os; os.chdir(ROOT)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("cta_p149")

from nexus_quant.projects.commodity_cta.providers.yahoo_futures import YahooFuturesProvider
from nexus_quant.projects.commodity_cta.costs.futures_fees import commodity_futures_cost_model
from nexus_quant.projects.commodity_cta.strategies.rp_mom_dd import RPMomDDStrategy, CHAMPION_PARAMS
from nexus_quant.data.schema import MarketDataset
from nexus_quant.backtest.engine import BacktestEngine, BacktestConfig


def compute_metrics(equity_curve, n_per_year=252):
    rets = [equity_curve[i] / equity_curve[i - 1] - 1 for i in range(1, len(equity_curve)) if equity_curve[i - 1] > 0]
    if not rets: return {"sharpe": 0, "cagr": 0, "max_drawdown": 1, "n_bars": 0}
    n = len(rets); mn = sum(rets) / n; var = sum((r - mn) ** 2 for r in rets) / n
    std = math.sqrt(var) if var > 0 else 1e-10
    sharpe = (mn / std) * math.sqrt(n_per_year)
    final = equity_curve[-1]; cagr = final ** (n_per_year / n) - 1 if final > 0 else 0
    peak = 1; mdd = 0
    for e in equity_curve:
        if e > peak: peak = e
        dd = (peak - e) / peak if peak > 0 else 0
        if dd > mdd: mdd = dd
    return {"sharpe": sharpe, "cagr": cagr, "max_drawdown": mdd, "n_bars": n}


def slice_dataset(dataset, start_date, end_date):
    start_ts = int(datetime.strptime(start_date, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp())
    end_ts = int(datetime.strptime(end_date, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp())
    indices = [i for i, ts in enumerate(dataset.timeline) if start_ts <= ts <= end_ts]
    if not indices: raise ValueError(f"No bars in [{start_date}, {end_date}]")
    i0, i1 = indices[0], indices[-1] + 1
    return MarketDataset(
        provider=dataset.provider, timeline=dataset.timeline[i0:i1], symbols=dataset.symbols,
        perp_close={s: dataset.perp_close[s][i0:i1] for s in dataset.symbols},
        spot_close=None, funding={}, fingerprint=hashlib.md5(f"{start_date}:{end_date}".encode()).hexdigest()[:8],
        market_type=dataset.market_type,
        features={fn: {s: arr[i0:i1] for s, arr in fd.items()} for fn, fd in dataset.features.items()},
        meta={**dataset.meta, "wf_start": start_date, "wf_end": end_date},
    )


def main():
    logger.info("=" * 80)
    logger.info("Phase 149: Cost Sensitivity Analysis")
    logger.info("=" * 80)

    DIVERSIFIED_14 = ["CL=F", "NG=F", "GC=F", "SI=F", "HG=F", "ZW=F", "ZC=F", "ZS=F",
                       "EURUSD=X", "GBPUSD=X", "AUDUSD=X", "JPY=X", "TLT", "IEF"]

    cfg = {"symbols": DIVERSIFIED_14, "start": "2005-01-01", "end": "2026-02-20",
           "cache_dir": "data/cache/yahoo_futures", "min_valid_bars": 300, "request_delay": 0.3}
    dataset = YahooFuturesProvider(cfg, seed=42).load()

    periods = [
        ("FULL", "2007-01-01", "2026-02-20"),
        ("OOS1", "2016-01-01", "2020-12-31"),
        ("OOS2", "2021-01-01", "2026-02-20"),
    ]

    # Cost levels to test
    cost_configs = [
        ("Zero cost", 0, 0, 0),
        ("Low (3.5bp)", 0.5, 2.5, 0.5),
        ("Default (7bp)", 1.0, 5.0, 1.0),
        ("Medium (14bp)", 2.0, 10.0, 2.0),
        ("High (21bp)", 3.0, 15.0, 3.0),
        ("Extreme (35bp)", 5.0, 25.0, 5.0),
    ]

    logger.info(f"\n  {'Cost Level':20s}  {'Tot bps':>8s}  {'FULL':>8s}  {'OOS1':>8s}  {'OOS2':>8s}  {'CAGR':>7s}  {'MDD':>6s}")
    logger.info("  " + "-" * 80)

    all_results = {}

    for label, comm, slip, spread in cost_configs:
        total_bps = comm + slip + spread
        cost_model = commodity_futures_cost_model(
            slippage_bps=slip, commission_bps=comm, spread_bps=spread
        )

        results = {}
        for plabel, start, end in periods:
            ds_s = slice_dataset(dataset, start, end)
            strat = RPMomDDStrategy(params=CHAMPION_PARAMS)
            result = BacktestEngine(BacktestConfig(costs=cost_model)).run(ds_s, strat, seed=42)
            results[plabel] = compute_metrics(result.equity_curve)

        all_results[label] = results
        s_full = results.get("FULL", {}).get("sharpe", 0)
        s_oos1 = results.get("OOS1", {}).get("sharpe", 0)
        s_oos2 = results.get("OOS2", {}).get("sharpe", 0)
        cagr = results.get("FULL", {}).get("cagr", 0)
        mdd = results.get("FULL", {}).get("max_drawdown", 1)
        logger.info(f"  {label:20s}  {total_bps:7.1f}  {s_full:+8.3f}  {s_oos1:+8.3f}  {s_oos2:+8.3f}  {cagr*100:+6.1f}%  {mdd*100:5.1f}%")

    logger.info(f"\n  Strategy rebalances every 42 days → ~6 RT/year")
    logger.info(f"  At 7 bps default: 6 × 0.07% = 0.42%/year in costs")
    logger.info(f"  At 35 bps extreme: 6 × 0.35% = 2.1%/year in costs")

    # Save
    output = {
        "phase": "149", "title": "Cost Sensitivity",
        "results": {k: {p: dict(m) for p, m in r.items()} for k, r in all_results.items()},
    }
    out_dir = ROOT / "artifacts" / "phase149"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "cost_sensitivity_report.json", "w") as f:
        json.dump(output, f, indent=2)
    logger.info(f"\n  Saved: artifacts/phase149/cost_sensitivity_report.json")


if __name__ == "__main__":
    main()
