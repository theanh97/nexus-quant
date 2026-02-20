"""
Phase 139: Diversified Multi-Asset CTA
========================================
Hypothesis: Adding FX + Government Bonds to the commodity CTA universe
provides crucial diversification during commodity bear markets (2015-2019).

Strategy: CTAEnsembleStrategy (Trend 40% + Carry 30% + MomValue 30%)
Universe A: Commodity-only (8 symbols)  ← baseline
Universe B: Diversified (8 comm + 4 FX + 2 bonds = 14 symbols) ← Phase 139

Walk-forward design:
  IS  : 2007-2015 (8 years, commodity bull+bust)
  OOS1: 2016-2020 (5 years, commodity bear + COVID)
  OOS2: 2021-2026 (5 years, commodity bull + inflation)

Expected improvement:
  - OOS1 Sharpe: commodity-only ~-0.2 → diversified ~0.4+ (bonds trend UP 2016-2019)
  - OOS2 Sharpe: both good (commodities + bond SHORT + FX trends in 2022)

Cost model: 7 bps RT (commodities), same for FX/bonds (conservative)

Usage:
    cd "/Users/qtmobile/Desktop/Nexus - Quant Trading "
    python3 scripts/run_commodity_cta_diversified.py
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
logger = logging.getLogger("cta_diversified")


# ── Metrics helpers ───────────────────────────────────────────────────────────

def compute_metrics(equity_curve, n_per_year=252):
    """Compute Sharpe, CAGR, MaxDD from equity curve (list of floats)."""
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
    """Slice a MarketDataset to [start_date, end_date]."""
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


def run_wf(label: str, dataset, strategy_cls, cost_model, periods):
    """Run walk-forward validation for a given dataset + strategy."""
    from nexus_quant.backtest.engine import BacktestEngine, BacktestConfig

    results = {}
    for period_label, start, end in periods:
        try:
            ds_slice = slice_dataset(dataset, start, end)
        except ValueError as e:
            logger.warning(f"  [{label}] {period_label}: {e}")
            continue

        strategy = strategy_cls()
        engine = BacktestEngine(BacktestConfig(costs=cost_model))

        try:
            result = engine.run(ds_slice, strategy, seed=42)
            metrics = compute_metrics(result.equity_curve)
        except Exception as e:
            logger.warning(f"  [{label}] {period_label}: backtest failed: {e}")
            import traceback
            traceback.print_exc()
            continue

        results[period_label] = metrics
        logger.info(
            f"  [{label:12s}] {period_label:6s}  {start}→{end}  "
            f"Sharpe={metrics['sharpe']:+6.3f}  "
            f"CAGR={metrics['cagr']*100:+6.2f}%  "
            f"MDD={metrics['max_drawdown']*100:5.1f}%"
        )

    return results


def main():
    logger.info("=" * 70)
    logger.info("NEXUS Phase 139 — Diversified Multi-Asset CTA")
    logger.info("Hypothesis: FX + Bonds fix commodity-only failure in 2015-2019")
    logger.info("=" * 70)

    from nexus_quant.projects.commodity_cta.providers.yahoo_futures import (
        YahooFuturesProvider, DIVERSIFIED_SYMBOLS,
    )
    from nexus_quant.projects.commodity_cta.strategies.cta_ensemble import CTAEnsembleStrategy
    from nexus_quant.projects.commodity_cta.costs.futures_fees import commodity_futures_cost_model

    COMMODITY_ONLY = [
        "CL=F", "NG=F", "GC=F", "SI=F", "HG=F", "ZW=F", "ZC=F", "ZS=F",
    ]

    cost_model = commodity_futures_cost_model(
        slippage_bps=5.0, commission_bps=1.0, spread_bps=1.0
    )

    periods = [
        ("IS",   "2007-01-01", "2015-12-31"),
        ("OOS1", "2016-01-01", "2020-12-31"),
        ("OOS2", "2021-01-01", "2026-02-20"),
        ("FULL", "2007-01-01", "2026-02-20"),
    ]

    # ── Load datasets ──────────────────────────────────────────────────────────

    logger.info("\n[1/2] Loading commodity-only dataset (8 symbols)...")
    cfg_comm = {
        "symbols": COMMODITY_ONLY,
        "start": "2005-01-01",
        "end": "2026-02-20",
        "cache_dir": "data/cache/yahoo_futures",
        "min_valid_bars": 300,
        "request_delay": 0.3,
    }
    provider_comm = YahooFuturesProvider(cfg_comm, seed=42)
    ds_comm = provider_comm.load()
    logger.info(f"  Loaded: {ds_comm.symbols}")

    logger.info(f"\n[2/2] Loading diversified dataset ({len(DIVERSIFIED_SYMBOLS)} symbols)...")
    cfg_div = {
        "symbols": DIVERSIFIED_SYMBOLS,
        "start": "2005-01-01",
        "end": "2026-02-20",
        "cache_dir": "data/cache/yahoo_futures",
        "min_valid_bars": 300,
        "request_delay": 0.3,
    }
    provider_div = YahooFuturesProvider(cfg_div, seed=42)
    ds_div = provider_div.load()
    logger.info(f"  Loaded: {ds_div.symbols}")

    # ── Walk-forward validation ────────────────────────────────────────────────

    logger.info("\n" + "─" * 70)
    logger.info("Walk-Forward Results:")
    logger.info("─" * 70)

    logger.info("\n=== A. Commodity-Only (8 symbols, baseline) ===")
    res_comm = run_wf("commodity", ds_comm, CTAEnsembleStrategy, cost_model, periods)

    logger.info("\n=== B. Diversified CTA (14 symbols: comm + FX + bonds) ===")
    res_div = run_wf("diversified", ds_div, CTAEnsembleStrategy, cost_model, periods)

    # ── Comparison table ───────────────────────────────────────────────────────

    logger.info("\n" + "=" * 70)
    logger.info("COMPARISON TABLE: Commodity-Only vs Diversified CTA")
    logger.info("=" * 70)
    logger.info(f"{'Period':6s}  {'Commodity':>10s}  {'Diversified':>12s}  {'Delta':>8s}")
    logger.info("-" * 50)

    for label, _, _ in periods:
        c = res_comm.get(label, {}).get("sharpe", float("nan"))
        d = res_div.get(label, {}).get("sharpe", float("nan"))
        delta = d - c if (c == c and d == d) else float("nan")
        sign = "✓" if delta > 0.05 else ("~" if abs(delta) <= 0.05 else "✗")
        delta_str = f"{delta:+.3f} {sign}" if delta == delta else "N/A"
        logger.info(
            f"{label:6s}  {c:>+10.3f}  {d:>+12.3f}  {delta_str:>10s}"
        )

    # ── Verdict ────────────────────────────────────────────────────────────────

    logger.info("\n" + "─" * 70)

    comm_oos_min = min(
        res_comm.get("OOS1", {}).get("sharpe", -99),
        res_comm.get("OOS2", {}).get("sharpe", -99),
    )
    div_oos_min = min(
        res_div.get("OOS1", {}).get("sharpe", -99),
        res_div.get("OOS2", {}).get("sharpe", -99),
    )
    div_oos_avg = (
        res_div.get("OOS1", {}).get("sharpe", 0) +
        res_div.get("OOS2", {}).get("sharpe", 0)
    ) / 2.0

    logger.info(f"  Commodity-only OOS_MIN : {comm_oos_min:+.3f}")
    logger.info(f"  Diversified    OOS_MIN : {div_oos_min:+.3f}  (delta: {div_oos_min - comm_oos_min:+.3f})")
    logger.info(f"  Diversified    OOS_AVG : {div_oos_avg:+.3f}")

    hypothesis_validated = div_oos_min > comm_oos_min + 0.1

    if div_oos_min > 0.5:
        verdict = "STRONG PASS — Diversification validated (OOS_MIN > 0.5)"
    elif div_oos_min > 0.3:
        verdict = "PASS — Diversification validated (OOS_MIN > 0.3)"
    elif div_oos_min > comm_oos_min + 0.1:
        verdict = "PARTIAL — Diversification helps but OOS still weak"
    elif div_oos_min > 0.0:
        verdict = "WEAK — OOS positive but minimal improvement"
    else:
        verdict = "FAIL — FX+Bonds do not fix commodity OOS failure"

    logger.info(f"\n  VERDICT: {verdict}")
    logger.info(f"  Hypothesis validated: {hypothesis_validated}")

    logger.info("\n  Benchmark comparison:")
    logger.info(f"    SG CTA Index   : Sharpe ~0.5 (diversified, 50+ markets)")
    logger.info(f"    Commodity-only : Full Sharpe {res_comm.get('FULL', {}).get('sharpe', 0):+.3f}")
    logger.info(f"    Diversified    : Full Sharpe {res_div.get('FULL', {}).get('sharpe', 0):+.3f}")

    # ── Save results ───────────────────────────────────────────────────────────

    output = {
        "phase": 139,
        "title": "Diversified Multi-Asset CTA (Commodity + FX + Bonds)",
        "run_date": datetime.now().isoformat(),
        "commodity_only": {
            "symbols": COMMODITY_ONLY,
            "n_symbols": len(COMMODITY_ONLY),
            "periods": res_comm,
            "oos_min": comm_oos_min,
        },
        "diversified": {
            "symbols": list(ds_div.symbols),
            "n_symbols": len(ds_div.symbols),
            "periods": res_div,
            "oos_min": div_oos_min,
            "oos_avg": div_oos_avg,
        },
        "hypothesis_validated": hypothesis_validated,
        "verdict": verdict,
        "notes": (
            "CTAEnsemble (Trend40%+Carry30%+MomValue30%) on commodity vs "
            "commodity+FX+bonds universe. 7bps RT cost model."
        ),
    }

    out_path = ROOT / "artifacts" / "phase139_diversified_cta.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    logger.info(f"\n  Results saved to {out_path}")

    return output


if __name__ == "__main__":
    main()
