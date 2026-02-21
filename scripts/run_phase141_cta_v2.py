"""
Phase 141: CTA v2 — TSMOM + Donchian + MomValue (no carry)
=============================================================
Complete rebuild of commodity CTA based on Phase 140 diagnostic.

Phase 140 findings:
  - EMA crossover: Sharpe 0.18 IS, negative OOS → insufficient
  - Carry: Sharpe -0.50 → confirmed drag, KILLED
  - MomValue: Sharpe +0.03 → neutral, keep as diversifier
  - Ensemble: worse than components → conflicting signals

Phase 141 approach:
  1. TSMOM (Moskowitz 2012) — multi-scale risk-adjusted momentum
  2. Donchian breakout — channel breakout (Turtle trading)
  3. CTA v2 ensemble — TSMOM(50%) + Donchian(30%) + MomValue(20%)

Tests:
  Step 1: TSMOM standalone on 8-comm (IS, OOS1, OOS2)
  Step 2: Donchian standalone on 8-comm
  Step 3: CTA v2 ensemble on 8-comm
  Step 4: CTA v2 ensemble on 14-sym (if 8-comm passes)

Pass criteria: OOS_MIN > 0.3 (any strategy)
Target: Sharpe > 0.8

Usage:
    cd /Users/truonglys/projects/quant
    python3 scripts/run_phase141_cta_v2.py
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

import os
os.chdir(ROOT)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("cta_p141")


def compute_metrics(equity_curve, n_per_year=252):
    if len(equity_curve) < 10:
        return {"sharpe": 0.0, "cagr": 0.0, "max_drawdown": 1.0, "n_bars": len(equity_curve)}
    rets = []
    for i in range(1, len(equity_curve)):
        e0, e1 = equity_curve[i - 1], equity_curve[i]
        if e0 > 0 and e1 == e1:
            rets.append(e1 / e0 - 1.0)
    if not rets:
        return {"sharpe": 0.0, "cagr": 0.0, "max_drawdown": 1.0, "n_bars": 0}
    n = len(rets)
    mn = sum(rets) / n
    var = sum((r - mn) ** 2 for r in rets) / n
    std = math.sqrt(var) if var > 0 else 1e-10
    sharpe = (mn / std) * math.sqrt(n_per_year)
    final_eq = equity_curve[-1]
    cagr = final_eq ** (n_per_year / n) - 1 if n > 0 and final_eq > 0 else 0.0
    peak, max_dd = 1.0, 0.0
    for e in equity_curve:
        if e > peak:
            peak = e
        dd = (peak - e) / peak if peak > 0 else 0.0
        if dd > max_dd:
            max_dd = dd
    return {"sharpe": sharpe, "cagr": cagr, "max_drawdown": max_dd, "n_bars": n}


def slice_dataset(dataset, start_date: str, end_date: str):
    from nexus_quant.data.schema import MarketDataset
    import hashlib
    start_ts = int(datetime.strptime(start_date, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp())
    end_ts = int(datetime.strptime(end_date, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp())
    mask = [start_ts <= ts <= end_ts for ts in dataset.timeline]
    indices = [i for i, m in enumerate(mask) if m]
    if not indices:
        raise ValueError(f"No bars in [{start_date}, {end_date}]")
    i0, i1 = indices[0], indices[-1] + 1
    timeline = dataset.timeline[i0:i1]
    perp_close = {sym: dataset.perp_close[sym][i0:i1] for sym in dataset.symbols}
    features = {
        feat_name: {sym: arr[i0:i1] for sym, arr in feat_data.items()}
        for feat_name, feat_data in dataset.features.items()
    }
    fp = hashlib.md5(f"{start_date}:{end_date}:{dataset.symbols}".encode()).hexdigest()[:8]
    return MarketDataset(
        provider=dataset.provider, timeline=timeline, symbols=dataset.symbols,
        perp_close=perp_close, spot_close=None, funding={}, fingerprint=fp,
        market_type=dataset.market_type, features=features,
        meta={**dataset.meta, "wf_start": start_date, "wf_end": end_date},
    )


def run_period(label, period_label, start, end, dataset, strategy_cls, strategy_params, cost_model):
    from nexus_quant.backtest.engine import BacktestEngine, BacktestConfig
    try:
        ds_slice = slice_dataset(dataset, start, end)
    except ValueError as e:
        logger.warning(f"  [{label}] {period_label}: {e}")
        return None
    strategy = strategy_cls(params=strategy_params)
    engine = BacktestEngine(BacktestConfig(costs=cost_model))
    try:
        result = engine.run(ds_slice, strategy, seed=42)
        metrics = compute_metrics(result.equity_curve)
    except Exception as e:
        logger.warning(f"  [{label}] {period_label}: backtest failed: {e}")
        import traceback; traceback.print_exc()
        return None
    logger.info(
        f"  [{label:18s}] {period_label:6s}  {start}→{end}  "
        f"Sharpe={metrics['sharpe']:+6.3f}  "
        f"CAGR={metrics['cagr']*100:+6.2f}%  "
        f"MDD={metrics['max_drawdown']*100:5.1f}%  "
        f"bars={metrics['n_bars']}"
    )
    return metrics


def run_strategy(label, dataset, strategy_cls, strategy_params, cost_model, periods):
    results = {}
    for period_label, start, end in periods:
        m = run_period(label, period_label, start, end, dataset, strategy_cls, strategy_params, cost_model)
        if m is not None:
            results[period_label] = m
    return results


def sharpe_of(results, period):
    return results.get(period, {}).get("sharpe", float("nan"))


def main():
    logger.info("=" * 70)
    logger.info("NEXUS Phase 141 — CTA v2: TSMOM + Donchian + MomValue")
    logger.info("Goal: Sharpe > 0.8 (beat SG CTA Index ~0.5-0.7)")
    logger.info("=" * 70)

    from nexus_quant.projects.commodity_cta.providers.yahoo_futures import (
        YahooFuturesProvider, DIVERSIFIED_SYMBOLS,
    )
    from nexus_quant.projects.commodity_cta.strategies.tsmom import TSMOMStrategy
    from nexus_quant.projects.commodity_cta.strategies.donchian_breakout import DonchianBreakoutStrategy
    from nexus_quant.projects.commodity_cta.strategies.momentum_value import MomentumValueStrategy
    from nexus_quant.projects.commodity_cta.strategies.cta_ensemble_v2 import CTAEnsembleV2Strategy
    from nexus_quant.projects.commodity_cta.strategies.trend_following import TrendFollowingStrategy
    from nexus_quant.projects.commodity_cta.costs.futures_fees import commodity_futures_cost_model

    COMMODITY_ONLY = [
        "CL=F", "NG=F", "GC=F", "SI=F", "HG=F", "ZW=F", "ZC=F", "ZS=F",
    ]

    cost_model = commodity_futures_cost_model(slippage_bps=5.0, commission_bps=1.0, spread_bps=1.0)

    periods = [
        ("FULL", "2007-01-01", "2026-02-20"),
        ("IS",   "2007-01-01", "2015-12-31"),
        ("OOS1", "2016-01-01", "2020-12-31"),
        ("OOS2", "2021-01-01", "2026-02-20"),
    ]

    # ── Load dataset ─────────────────────────────────────────────────────────
    logger.info("\n[1/1] Loading 8-commodity dataset (from cache)...")
    cfg = {
        "symbols": COMMODITY_ONLY,
        "start": "2005-01-01",
        "end": "2026-02-20",
        "cache_dir": "data/cache/yahoo_futures",
        "min_valid_bars": 300,
        "request_delay": 0.3,
    }
    ds = YahooFuturesProvider(cfg, seed=42).load()
    logger.info(f"  Symbols ({len(ds.symbols)}): {ds.symbols}")
    logger.info(f"  Timeline: {ds.meta['n_bars']} bars")
    logger.info(f"  Features: {list(ds.features.keys())}")

    # Verify new features exist
    for feat in ["tsmom_21d", "tsmom_63d", "tsmom_126d", "tsmom_252d",
                 "donchian_20_high", "donchian_55_high", "donchian_120_high"]:
        assert feat in ds.features, f"Missing feature: {feat}"
    logger.info("  ✓ All Phase 141 features present")

    # ─────────────────────────────────────────────────────────────────────────
    # STEP 1: TSMOM standalone
    # ─────────────────────────────────────────────────────────────────────────
    logger.info("\n" + "=" * 70)
    logger.info("  STEP 1: TSMOM Standalone (Moskowitz 2012)")
    logger.info("=" * 70)

    # Conservative base params for 8-commodity universe
    tsmom_params = {
        "max_gross_leverage": 1.5,
        "max_position": 0.20,
        "max_net_leverage": 0.8,
        "vol_target": 0.08,
        "signal_scale": 2.0,
        "rebalance_freq": 21,
        "w_21": 0.10, "w_63": 0.20, "w_126": 0.35, "w_252": 0.35,
    }
    res_tsmom = run_strategy("TSMOM", ds, TSMOMStrategy, tsmom_params, cost_model, periods)

    # Higher vol target variant
    tsmom_hi = {**tsmom_params, "vol_target": 0.12, "max_gross_leverage": 2.0, "max_net_leverage": 1.0}
    logger.info("\n  --- TSMOM variant: higher vol target (0.12) ---")
    res_tsmom_hi = run_strategy("TSMOM-hiVol", ds, TSMOMStrategy, tsmom_hi, cost_model, periods)

    # 6m/12m only (Baltas & Kosowski best for commodities)
    tsmom_slow = {**tsmom_params, "w_21": 0.0, "w_63": 0.0, "w_126": 0.50, "w_252": 0.50}
    logger.info("\n  --- TSMOM variant: 6m+12m only (Baltas-Kosowski) ---")
    res_tsmom_slow = run_strategy("TSMOM-6m12m", ds, TSMOMStrategy, tsmom_slow, cost_model, periods)

    # Signal scale sensitivity: sharper signal
    tsmom_sharp = {**tsmom_params, "signal_scale": 1.0}
    logger.info("\n  --- TSMOM variant: signal_scale=1.0 (sharper) ---")
    res_tsmom_sharp = run_strategy("TSMOM-sharp", ds, TSMOMStrategy, tsmom_sharp, cost_model, periods)

    # ─────────────────────────────────────────────────────────────────────────
    # STEP 2: Donchian standalone
    # ─────────────────────────────────────────────────────────────────────────
    logger.info("\n" + "=" * 70)
    logger.info("  STEP 2: Donchian Breakout Standalone")
    logger.info("=" * 70)

    donchian_params = {
        "max_gross_leverage": 1.5,
        "max_position": 0.20,
        "max_net_leverage": 0.8,
        "vol_target": 0.08,
        "rebalance_freq": 10,
        "w_fast": 0.20, "w_medium": 0.40, "w_slow": 0.40,
    }
    res_donchian = run_strategy("Donchian", ds, DonchianBreakoutStrategy, donchian_params, cost_model, periods)

    # Slow channels only (55d + 120d)
    donchian_slow = {**donchian_params, "w_fast": 0.0, "w_medium": 0.50, "w_slow": 0.50}
    logger.info("\n  --- Donchian variant: 55d+120d only ---")
    res_donchian_slow = run_strategy("Donchian-slow", ds, DonchianBreakoutStrategy, donchian_slow, cost_model, periods)

    # ─────────────────────────────────────────────────────────────────────────
    # STEP 3: Old EMA Trend (Phase 140 baseline for comparison)
    # ─────────────────────────────────────────────────────────────────────────
    logger.info("\n" + "=" * 70)
    logger.info("  STEP 3: Old EMA Trend (Phase 140 baseline)")
    logger.info("=" * 70)

    ema_params = {
        "max_gross_leverage": 2.0, "max_position": 0.25, "vol_target": 0.12,
        "signal_threshold": 0.1, "rebalance_freq": 21,
    }
    res_ema = run_strategy("EMA-Trend(old)", ds, TrendFollowingStrategy, ema_params, cost_model, periods)

    # ─────────────────────────────────────────────────────────────────────────
    # STEP 4: CTA v2 Ensemble
    # ─────────────────────────────────────────────────────────────────────────
    logger.info("\n" + "=" * 70)
    logger.info("  STEP 4: CTA v2 Ensemble (TSMOM 50% + Donchian 30% + MomValue 20%)")
    logger.info("=" * 70)

    ensemble_params = {
        "w_tsmom": 0.50, "w_donchian": 0.30, "w_mom_value": 0.20,
        "max_gross_leverage": 1.5, "max_position": 0.20, "max_net_leverage": 0.8,
        "rebalance_freq": 10,
    }
    res_ensemble = run_strategy("CTA-v2-Ensemble", ds, CTAEnsembleV2Strategy, ensemble_params, cost_model, periods)

    # TSMOM + Donchian only (no MomValue)
    ensemble_td = {**ensemble_params, "w_tsmom": 0.60, "w_donchian": 0.40, "w_mom_value": 0.0}
    logger.info("\n  --- CTA v2 variant: TSMOM(60%) + Donchian(40%) only ---")
    res_ensemble_td = run_strategy("CTA-v2-T60D40", ds, CTAEnsembleV2Strategy, ensemble_td, cost_model, periods)

    # TSMOM only through ensemble (for comparison)
    ensemble_tonly = {**ensemble_params, "w_tsmom": 1.0, "w_donchian": 0.0, "w_mom_value": 0.0}
    logger.info("\n  --- CTA v2 variant: TSMOM only (100%) ---")
    res_ensemble_tonly = run_strategy("CTA-v2-TSMOM100", ds, CTAEnsembleV2Strategy, ensemble_tonly, cost_model, periods)

    # ─────────────────────────────────────────────────────────────────────────
    # FINAL COMPARISON TABLE
    # ─────────────────────────────────────────────────────────────────────────
    logger.info("\n" + "=" * 70)
    logger.info("  FINAL COMPARISON TABLE — Phase 141 CTA v2")
    logger.info("=" * 70)

    all_results = [
        ("EMA-Trend(P140)",   res_ema),
        ("TSMOM",             res_tsmom),
        ("TSMOM-hiVol",       res_tsmom_hi),
        ("TSMOM-6m12m",       res_tsmom_slow),
        ("TSMOM-sharp",       res_tsmom_sharp),
        ("Donchian",          res_donchian),
        ("Donchian-slow",     res_donchian_slow),
        ("CTA-v2-Ensemble",   res_ensemble),
        ("CTA-v2-T60D40",    res_ensemble_td),
        ("CTA-v2-TSMOM100",  res_ensemble_tonly),
    ]

    logger.info(f"\n  {'Strategy':20s}  {'FULL':>8s}  {'IS':>8s}  {'OOS1':>8s}  {'OOS2':>8s}  {'OOS_MIN':>8s}  {'MDD':>7s}")
    logger.info("  " + "-" * 80)

    best_oos_min = -99.0
    best_strategy = "none"
    best_full_sharpe = -99.0

    for name, r in all_results:
        full = sharpe_of(r, "FULL")
        is_ = sharpe_of(r, "IS")
        oos1 = sharpe_of(r, "OOS1")
        oos2 = sharpe_of(r, "OOS2")
        mdd_full = r.get("FULL", {}).get("max_drawdown", float("nan"))
        oos_min = min(oos1, oos2) if (oos1 == oos1 and oos2 == oos2) else float("nan")

        def fmt(v, w=8):
            return f"{v:+{w}.3f}" if v == v else " " * (w - 3) + "N/A"

        mdd_str = f"{mdd_full*100:5.1f}%" if mdd_full == mdd_full else "   N/A"

        star = ""
        if oos_min == oos_min and oos_min > best_oos_min:
            best_oos_min = oos_min
            best_strategy = name
            star = " ← BEST OOS"
        if full == full and full > best_full_sharpe:
            best_full_sharpe = full

        logger.info(f"  {name:20s}  {fmt(full)}  {fmt(is_)}  {fmt(oos1)}  {fmt(oos2)}  {fmt(oos_min)}  {mdd_str}{star}")

    # ─────────────────────────────────────────────────────────────────────────
    # VERDICT
    # ─────────────────────────────────────────────────────────────────────────
    logger.info("\n" + "=" * 70)
    logger.info("  VERDICT")
    logger.info("=" * 70)

    logger.info(f"  Best strategy:    {best_strategy}")
    logger.info(f"  Best OOS_MIN:     {best_oos_min:+.3f}")
    logger.info(f"  Best FULL Sharpe: {best_full_sharpe:+.3f}")

    # Compare to Phase 140 baseline
    ema_oos_min = min(sharpe_of(res_ema, "OOS1"), sharpe_of(res_ema, "OOS2"))
    improvement = best_oos_min - ema_oos_min if (best_oos_min == best_oos_min and ema_oos_min == ema_oos_min) else 0.0

    logger.info(f"\n  vs Phase 140 EMA baseline:")
    logger.info(f"    EMA OOS_MIN:     {ema_oos_min:+.3f}")
    logger.info(f"    Best v2 OOS_MIN: {best_oos_min:+.3f}")
    logger.info(f"    Improvement:     {improvement:+.3f}")

    if best_oos_min > 0.8:
        verdict = "TARGET MET — Sharpe > 0.8 OOS!"
        next_step = "Commit champion. Walk-forward validation. Production prep."
    elif best_oos_min > 0.5:
        verdict = "STRONG — Sharpe > 0.5 OOS (beats SG CTA Index)"
        next_step = "Optimize params, test 14-sym universe, consider vol regime overlay"
    elif best_oos_min > 0.3:
        verdict = "PASS — Sharpe > 0.3 OOS, viable with further tuning"
        next_step = "Parameter sweep on TSMOM lookback weights and signal thresholds"
    elif best_oos_min > 0.0:
        verdict = "WEAK PASS — positive OOS but below target"
        next_step = "Add more signals (breakout persistence, vol breakout, sector rotation)"
    else:
        verdict = "FAIL — still negative OOS"
        next_step = "Commodity CTA with free Yahoo data may be fundamentally limited"

    logger.info(f"\n  VERDICT: {verdict}")
    logger.info(f"  NEXT STEP: {next_step}")

    # ─────────────────────────────────────────────────────────────────────────
    # SAVE RESULTS
    # ─────────────────────────────────────────────────────────────────────────
    output = {
        "phase": "141",
        "title": "CTA v2 — TSMOM + Donchian + MomValue",
        "run_date": datetime.now().isoformat(),
        "phase140_baseline": {
            "ema_trend_oos_min": ema_oos_min,
        },
        "results": {
            name: {p: {m: v for m, v in met.items()} for p, met in r.items()}
            for name, r in all_results
        },
        "best_strategy": best_strategy,
        "best_oos_min": best_oos_min,
        "best_full_sharpe": best_full_sharpe,
        "improvement_vs_p140": improvement,
        "verdict": verdict,
        "next_step": next_step,
    }

    out_dir = ROOT / "artifacts" / "phase141"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "cta_v2_report.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    logger.info(f"\n  Results saved: {out_path}")
    logger.info("\n" + "=" * 70)
    logger.info("Phase 141 complete.")
    logger.info("=" * 70)

    return output


if __name__ == "__main__":
    main()
