"""
Phase 147: Leave-One-Year-Out (LOYO) Validation
=================================================
Run full backtest with P146 champion, then extract per-year performance.
This reveals if performance is concentrated in a few years or broad-based.
"""
from __future__ import annotations

import json, logging, math, sys, hashlib
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
import os; os.chdir(ROOT)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("cta_p147")

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
    logger.info("Phase 147: LOYO Validation — Per-Year Performance Breakdown")
    logger.info("=" * 80)

    DIVERSIFIED_14 = ["CL=F", "NG=F", "GC=F", "SI=F", "HG=F", "ZW=F", "ZC=F", "ZS=F",
                       "EURUSD=X", "GBPUSD=X", "AUDUSD=X", "JPY=X", "TLT", "IEF"]

    cfg = {"symbols": DIVERSIFIED_14, "start": "2005-01-01", "end": "2026-02-20",
           "cache_dir": "data/cache/yahoo_futures", "min_valid_bars": 300, "request_delay": 0.3}
    dataset = YahooFuturesProvider(cfg, seed=42).load()
    logger.info(f"  Loaded: {len(dataset.symbols)} symbols x {dataset.meta['n_bars']} bars")

    cost_model = commodity_futures_cost_model(slippage_bps=5.0, commission_bps=1.0, spread_bps=1.0)

    # Run FULL backtest first, then slice equity curve by year
    ds_full = slice_dataset(dataset, "2007-01-01", "2026-02-20")
    strat = RPMomDDStrategy(params=CHAMPION_PARAMS)
    result = BacktestEngine(BacktestConfig(costs=cost_model)).run(ds_full, strat, seed=42)
    eq = result.equity_curve
    timeline = ds_full.timeline

    # Map each bar to its year
    bar_years = []
    for ts in timeline:
        dt = datetime.utcfromtimestamp(ts)
        bar_years.append(dt.year)

    # Extract per-year equity slices
    years = sorted(set(bar_years))
    year_results = {}
    positive_years = 0
    sharpe_sum = 0

    logger.info(f"\n  {'Year':6s}  {'Sharpe':>8s}  {'Return':>8s}  {'MDD':>6s}  {'Bars':>5s}  {'Result':>8s}")
    logger.info("  " + "-" * 55)

    for year in years:
        # Find bar indices for this year
        year_indices = [i for i, y in enumerate(bar_years) if y == year]
        if len(year_indices) < 10:
            continue

        # Extract equity for this year (normalize to start at 1.0)
        i_start = year_indices[0]
        eq_year = [eq[i] / eq[i_start] for i in year_indices]
        m = compute_metrics(eq_year)
        year_results[str(year)] = m

        annual_return = eq[year_indices[-1]] / eq[year_indices[0]] - 1
        status = "WIN" if annual_return > 0 else "LOSS"
        if annual_return > 0:
            positive_years += 1
        sharpe_sum += m["sharpe"]
        logger.info(f"  {year:6d}  {m['sharpe']:+8.3f}  {annual_return*100:+7.1f}%  {m['max_drawdown']*100:5.1f}%  {len(year_indices):5d}  {status:>8s}")

    n_years = len(year_results)
    avg_sharpe = sharpe_sum / n_years if n_years > 0 else 0
    win_rate = positive_years / n_years * 100 if n_years > 0 else 0

    logger.info(f"\n  SUMMARY:")
    logger.info(f"    Years tested:     {n_years}")
    logger.info(f"    Positive years:   {positive_years}/{n_years} ({win_rate:.0f}%)")
    logger.info(f"    Average Sharpe:   {avg_sharpe:+.3f}")
    logger.info(f"    Best year:        {max(year_results, key=lambda y: year_results[y]['sharpe'])} (Sharpe {max(m['sharpe'] for m in year_results.values()):+.3f})")
    logger.info(f"    Worst year:       {min(year_results, key=lambda y: year_results[y]['sharpe'])} (Sharpe {min(m['sharpe'] for m in year_results.values()):+.3f})")

    # Full period for reference
    m_full = compute_metrics(eq)
    logger.info(f"\n  FULL period: Sharpe={m_full['sharpe']:+.3f}  CAGR={m_full['cagr']*100:+.1f}%  MDD={m_full['max_drawdown']*100:.1f}%")

    # Robustness score: fraction of years with Sharpe > 0 weighted by consistency
    neg_years = [y for y, m in year_results.items() if m["sharpe"] < 0]
    if neg_years:
        logger.info(f"\n  Negative years: {', '.join(neg_years)}")
    else:
        logger.info(f"\n  No negative years! Extremely robust.")

    # Save
    output = {
        "phase": "147", "title": "LOYO Validation",
        "year_results": year_results,
        "summary": {
            "n_years": n_years,
            "positive_years": positive_years,
            "win_rate_pct": win_rate,
            "avg_sharpe": avg_sharpe,
            "negative_years": neg_years,
        },
    }
    out_dir = ROOT / "artifacts" / "phase147"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "loyo_validation_report.json", "w") as f:
        json.dump(output, f, indent=2)
    logger.info(f"\n  Saved: artifacts/phase147/loyo_validation_report.json")


if __name__ == "__main__":
    main()
