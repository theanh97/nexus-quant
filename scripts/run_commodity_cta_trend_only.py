"""
Phase 140: CTA Diagnostic + Trend-Only Universe Expansion
===========================================================
Continuation of Phase 139 (diversified CTA failed — FX+Bonds did not fix OOS).

PHASE 139 RESULTS (confirmed):
  TrendFollowing standalone : FULL=+0.338, IS=+0.524, OOS1=-0.206, OOS2=+0.252
  CTAEnsemble (8 comm)      : FULL=-0.298  ← WORSE than Trend-only!
  CTAEnsemble (14 sym)      : FULL=-0.451  ← Even worse

ROOT CAUSE HYPOTHESIS:
  CTAEnsemble = Trend(40%) + Carry(30%) + MomValue(30%)
  → Carry and MomValue use cross-sectional ranking → opposing positions in bear market
  → These components DRAG on Trend alpha, especially in 2015-2020 commodity bear

THIS PHASE:
  Step 1: DIAGNOSTIC — run each sub-strategy standalone on 8-comm (2007-2026)
           Goal: confirm Carry/MomValue are drags vs Trend
  Step 2: TREND-ONLY test on 8-comm (rebalance_freq=21, max_gross=2.0)
           Baseline confirmation
  Step 3: TREND-ONLY test on 14-sym (comm + FX + bonds)
           True test of universe expansion hypothesis without carry/value drag

  Kill criteria: if Trend-only 14-sym OOS_MIN < 0.0 → EMA signals insufficient
  → Move to: crypto_perps new data sources OR Deribit Skew MR with real IV data

Usage:
    cd "/Users/truonglys/Desktop/nexus-quant"
    python3 scripts/run_commodity_cta_trend_only.py
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
logger = logging.getLogger("cta_p140")


# ── Metrics ─────────────────────────────────────────────────────────────────

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
    """Slice a MarketDataset to [start_date, end_date]."""
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


def run_period(label: str, period_label: str, start: str, end: str,
               dataset, strategy_cls, strategy_params, cost_model, *, verbose=True):
    """Run a single period backtest. Returns metrics dict."""
    from nexus_quant.backtest.engine import BacktestEngine, BacktestConfig

    try:
        ds_slice = slice_dataset(dataset, start, end)
    except ValueError as e:
        if verbose:
            logger.warning(f"  [{label}] {period_label}: {e}")
        return None

    strategy = strategy_cls(params=strategy_params)
    engine = BacktestEngine(BacktestConfig(costs=cost_model))

    try:
        result = engine.run(ds_slice, strategy, seed=42)
        metrics = compute_metrics(result.equity_curve)
    except Exception as e:
        if verbose:
            logger.warning(f"  [{label}] {period_label}: backtest failed: {e}")
            import traceback; traceback.print_exc()
        return None

    if verbose:
        logger.info(
            f"  [{label:14s}] {period_label:6s}  {start}→{end}  "
            f"Sharpe={metrics['sharpe']:+6.3f}  "
            f"CAGR={metrics['cagr']*100:+6.2f}%  "
            f"MDD={metrics['max_drawdown']*100:5.1f}%  "
            f"bars={metrics['n_bars']}"
        )
    return metrics


def run_strategy(label: str, dataset, strategy_cls, strategy_params, cost_model, periods):
    """Run walk-forward for all periods."""
    results = {}
    for period_label, start, end in periods:
        m = run_period(label, period_label, start, end, dataset, strategy_cls, strategy_params, cost_model)
        if m is not None:
            results[period_label] = m
    return results


def print_section(title: str):
    logger.info("\n" + "=" * 70)
    logger.info(f"  {title}")
    logger.info("=" * 70)


def sharpe_of(results, period):
    return results.get(period, {}).get("sharpe", float("nan"))


def main():
    logger.info("=" * 70)
    logger.info("NEXUS Phase 140 — CTA Diagnostic + Trend-Only Universe Expansion")
    logger.info("Hypothesis: Carry+MomValue are alpha destroyers. Trend-only + 14-sym works.")
    logger.info("=" * 70)

    from nexus_quant.projects.commodity_cta.providers.yahoo_futures import (
        YahooFuturesProvider, DIVERSIFIED_SYMBOLS,
    )
    from nexus_quant.projects.commodity_cta.strategies.trend_following import TrendFollowingStrategy
    from nexus_quant.projects.commodity_cta.strategies.carry_roll import CarryRollStrategy
    from nexus_quant.projects.commodity_cta.strategies.momentum_value import MomentumValueStrategy
    from nexus_quant.projects.commodity_cta.strategies.cta_ensemble import CTAEnsembleStrategy
    from nexus_quant.projects.commodity_cta.costs.futures_fees import commodity_futures_cost_model

    COMMODITY_ONLY = [
        "CL=F", "NG=F", "GC=F", "SI=F", "HG=F", "ZW=F", "ZC=F", "ZS=F",
    ]

    cost_model = commodity_futures_cost_model(slippage_bps=5.0, commission_bps=1.0, spread_bps=1.0)

    # Walk-forward periods — same as Phase 139
    periods = [
        ("FULL", "2007-01-01", "2026-02-20"),
        ("IS",   "2007-01-01", "2015-12-31"),
        ("OOS1", "2016-01-01", "2020-12-31"),
        ("OOS2", "2021-01-01", "2026-02-20"),
    ]

    # ── Load datasets ─────────────────────────────────────────────────────────
    logger.info("\n[1/2] Loading 8-commodity dataset (from cache)...")
    cfg_comm = {
        "symbols": COMMODITY_ONLY,
        "start": "2005-01-01",
        "end": "2026-02-20",
        "cache_dir": "data/cache/yahoo_futures",
        "min_valid_bars": 300,
        "request_delay": 0.3,
    }
    ds_comm = YahooFuturesProvider(cfg_comm, seed=42).load()
    logger.info(f"  Symbols ({len(ds_comm.symbols)}): {ds_comm.symbols}")
    logger.info(f"  Timeline: {ds_comm.meta['n_bars']} bars")

    logger.info(f"\n[2/2] Loading diversified dataset (14-sym, from cache)...")
    cfg_div = {
        "symbols": DIVERSIFIED_SYMBOLS,
        "start": "2005-01-01",
        "end": "2026-02-20",
        "cache_dir": "data/cache/yahoo_futures",
        "min_valid_bars": 300,
        "request_delay": 0.3,
    }
    ds_div = YahooFuturesProvider(cfg_div, seed=42).load()
    logger.info(f"  Symbols ({len(ds_div.symbols)}): {ds_div.symbols}")
    logger.info(f"  Timeline: {ds_div.meta['n_bars']} bars")

    # ─────────────────────────────────────────────────────────────────────────
    # STEP 1: DIAGNOSTIC — individual sub-strategies on 8-comm
    # ─────────────────────────────────────────────────────────────────────────
    print_section("STEP 1: DIAGNOSTIC — Individual Sub-Strategy Sharpe (8-comm, 2007-2026)")
    logger.info("  Goal: confirm which components drag on CTAEnsemble alpha")

    diag_configs = [
        ("TrendOnly",  TrendFollowingStrategy, {"max_gross_leverage": 2.0, "max_position": 0.25, "vol_target": 0.12, "rebalance_freq": 21}),
        ("CarryOnly",  CarryRollStrategy,      {"max_gross_leverage": 2.0, "max_position": 0.25, "rebalance_freq": 5}),
        ("MomValOnly", MomentumValueStrategy,  {"max_gross_leverage": 2.0, "max_position": 0.25, "n_long": 4, "n_short": 4, "rebalance_freq": 21}),
        ("CTAEnsemble",CTAEnsembleStrategy,    {"rebalance_freq": 5, "w_trend": 0.40, "w_carry": 0.30, "w_mom_value": 0.30}),
    ]

    diag_results = {}
    for name, cls, params in diag_configs:
        logger.info(f"\n  --- {name} ---")
        diag_results[name] = run_strategy(name, ds_comm, cls, params, cost_model, periods)

    # Diagnostic summary table
    logger.info("\n  DIAGNOSTIC SUMMARY (8-comm universe):")
    logger.info(f"  {'Strategy':14s}  {'FULL':>8s}  {'IS':>8s}  {'OOS1':>8s}  {'OOS2':>8s}")
    logger.info("  " + "-" * 56)
    for name, _, _ in diag_configs:
        r = diag_results.get(name, {})
        full = sharpe_of(r, "FULL")
        is_  = sharpe_of(r, "IS")
        oos1 = sharpe_of(r, "OOS1")
        oos2 = sharpe_of(r, "OOS2")
        def fmt(v):
            return f"{v:+8.3f}" if v == v else "     N/A"
        logger.info(f"  {name:14s}  {fmt(full)}  {fmt(is_)}  {fmt(oos1)}  {fmt(oos2)}")

    # Determine if Carry/MomValue are drags
    trend_oos_min = min(
        sharpe_of(diag_results.get("TrendOnly", {}), "OOS1"),
        sharpe_of(diag_results.get("TrendOnly", {}), "OOS2"),
    )
    ensemble_oos_min = min(
        sharpe_of(diag_results.get("CTAEnsemble", {}), "OOS1"),
        sharpe_of(diag_results.get("CTAEnsemble", {}), "OOS2"),
    )
    carry_full = sharpe_of(diag_results.get("CarryOnly", {}), "FULL")
    mv_full = sharpe_of(diag_results.get("MomValOnly", {}), "FULL")
    trend_full = sharpe_of(diag_results.get("TrendOnly", {}), "FULL")

    carry_is_drag = carry_full < trend_full
    mv_is_drag = mv_full < trend_full
    ensemble_hurts = ensemble_oos_min < trend_oos_min

    logger.info(f"\n  DIAGNOSTIC VERDICT:")
    logger.info(f"    Carry FULL Sharpe: {carry_full:+.3f} vs Trend: {trend_full:+.3f} → {'DRAG ✗' if carry_is_drag else 'HELPS ✓'}")
    logger.info(f"    MomVal FULL Sharpe: {mv_full:+.3f} vs Trend: {trend_full:+.3f} → {'DRAG ✗' if mv_is_drag else 'HELPS ✓'}")
    logger.info(f"    Ensemble OOS_MIN ({ensemble_oos_min:+.3f}) < Trend OOS_MIN ({trend_oos_min:+.3f}): {'YES — ensemble WORSE ✗' if ensemble_hurts else 'NO — ensemble OK'}")

    # ─────────────────────────────────────────────────────────────────────────
    # STEP 2: TREND-ONLY on 8-comm (baseline confirmation)
    # ─────────────────────────────────────────────────────────────────────────
    print_section("STEP 2: Trend-Only on 8-Commodity (baseline, rebalance=21)")

    trend_params_monthly = {
        "max_gross_leverage": 2.0,
        "max_position": 0.25,
        "vol_target": 0.12,
        "signal_threshold": 0.1,
        "rebalance_freq": 21,
        "fast1": 12, "slow1": 26,
        "fast2": 20, "slow2": 50,
    }

    res_trend_8 = run_strategy("trend_8comm", ds_comm, TrendFollowingStrategy, trend_params_monthly, cost_model, periods)

    trend_8_oos_min = min(sharpe_of(res_trend_8, "OOS1"), sharpe_of(res_trend_8, "OOS2"))
    logger.info(f"\n  Trend-8comm OOS_MIN: {trend_8_oos_min:+.3f}")

    # ─────────────────────────────────────────────────────────────────────────
    # STEP 3: TREND-ONLY on 14-sym (main hypothesis — FX+Bonds help Trend?)
    # ─────────────────────────────────────────────────────────────────────────
    print_section("STEP 3: Trend-Only on 14-sym (Commodity + FX + Bonds)")
    logger.info("  Hypothesis: FX trends (rate differentials) + Bond trends (2016-2019 bull)")
    logger.info("  fix OOS1 performance without carry/value signal noise")

    # Also test a variant with tighter signal threshold to reduce noise from FX/bonds
    res_trend_14 = run_strategy("trend_14sym", ds_div, TrendFollowingStrategy, trend_params_monthly, cost_model, periods)

    # Tighter threshold variant (only trade when all 3 signals agree)
    trend_params_tight = {**trend_params_monthly, "signal_threshold": 0.33}  # all 3 signals must agree
    logger.info("\n  --- Variant: tight threshold (signal_threshold=0.33, all signals agree) ---")
    res_trend_14_tight = run_strategy("trend_14_tight", ds_div, TrendFollowingStrategy, trend_params_tight, cost_model, periods)

    trend_14_oos_min = min(sharpe_of(res_trend_14, "OOS1"), sharpe_of(res_trend_14, "OOS2"))
    trend_14_tight_oos_min = min(sharpe_of(res_trend_14_tight, "OOS1"), sharpe_of(res_trend_14_tight, "OOS2"))

    logger.info(f"\n  Trend-14sym OOS_MIN: {trend_14_oos_min:+.3f}")
    logger.info(f"  Trend-14-tight OOS_MIN: {trend_14_tight_oos_min:+.3f}")

    # ─────────────────────────────────────────────────────────────────────────
    # FINAL COMPARISON TABLE
    # ─────────────────────────────────────────────────────────────────────────
    print_section("FINAL COMPARISON TABLE")

    all_results = [
        ("TrendOnly-8comm",   diag_results.get("TrendOnly", {})),
        ("Carry-8comm",       diag_results.get("CarryOnly", {})),
        ("MomVal-8comm",      diag_results.get("MomValOnly", {})),
        ("CTAEnsemble-8comm", diag_results.get("CTAEnsemble", {})),
        ("Trend-8comm-m21",   res_trend_8),
        ("Trend-14sym-m21",   res_trend_14),
        ("Trend-14tight-m21", res_trend_14_tight),
    ]

    logger.info(f"\n  {'Strategy':20s}  {'FULL':>8s}  {'IS':>8s}  {'OOS1':>8s}  {'OOS2':>8s}  {'OOS_MIN':>8s}")
    logger.info("  " + "-" * 72)

    best_oos_min = -99.0
    best_strategy = "none"

    for name, r in all_results:
        full = sharpe_of(r, "FULL")
        is_  = sharpe_of(r, "IS")
        oos1 = sharpe_of(r, "OOS1")
        oos2 = sharpe_of(r, "OOS2")
        oos_min = min(oos1, oos2) if (oos1 == oos1 and oos2 == oos2) else float("nan")

        def fmt(v, w=8):
            return f"{v:+{w}.3f}" if v == v else " " * (w - 3) + "N/A"

        star = " ← BEST" if (oos_min == oos_min and oos_min > best_oos_min) else ""
        if oos_min == oos_min and oos_min > best_oos_min:
            best_oos_min = oos_min
            best_strategy = name

        logger.info(f"  {name:20s}  {fmt(full)}  {fmt(is_)}  {fmt(oos1)}  {fmt(oos2)}  {fmt(oos_min)}{star}")

    # ─────────────────────────────────────────────────────────────────────────
    # VERDICT
    # ─────────────────────────────────────────────────────────────────────────
    print_section("VERDICT")

    logger.info(f"  Best strategy: {best_strategy}")
    logger.info(f"  Best OOS_MIN:  {best_oos_min:+.3f}")

    # Validate hypothesis
    validated = best_oos_min > 0.0
    trend_helps = trend_14_oos_min > trend_8_oos_min + 0.05

    if trend_14_oos_min > 0.5:
        verdict = "STRONG PASS — Trend-only + 14-sym validated (OOS_MIN > 0.5)"
        next_step = "Commit champion, build position sizing refinement"
    elif trend_14_oos_min > 0.3:
        verdict = "PASS — Trend-only + 14-sym validated (OOS_MIN > 0.3)"
        next_step = "Commit champion, test longer EMA variants"
    elif trend_14_oos_min > 0.0:
        verdict = "WEAK PASS — Trend-only 14-sym positive OOS but fragile"
        next_step = "Test EMA parameter variants, different signal timeframes"
    elif trend_14_oos_min > trend_8_oos_min:
        verdict = "PARTIAL — 14-sym helps Trend but OOS still negative"
        next_step = "EMA signals insufficient for commodity CTA → consider Deribit Skew MR"
    else:
        verdict = "FAIL — Trend-only 14-sym does not improve on 8-comm"
        next_step = "EMA signals confirmed insufficient → move to Deribit Skew MR / new data sources"

    logger.info(f"\n  VERDICT: {verdict}")
    logger.info(f"  NEXT STEP: {next_step}")

    # Kill criteria
    if not validated:
        logger.info("\n  ⚠️  KILL CRITERIA MET: OOS_MIN < 0.0")
        logger.info("  → Commodity CTA with EMA signals is NOT viable standalone")
        logger.info("  → Adding to Failed_Strategies knowledge base")
        logger.info("  → Recommended pivot:")
        logger.info("    Option A: Deribit Skew Mean-Reversion (real IV data, crypto)")
        logger.info("    Option B: crypto_perps with new data sources (order flow, on-chain)")
        logger.info("    Option C: Return to P91b champion (Sharpe ~1.5) — incremental improvements")

    # ─────────────────────────────────────────────────────────────────────────
    # SAVE RESULTS
    # ─────────────────────────────────────────────────────────────────────────
    output = {
        "phase": "140",
        "title": "CTA Diagnostic + Trend-Only Universe Expansion",
        "run_date": datetime.now().isoformat(),
        "context": {
            "phase139_results": {
                "trend_standalone": {"FULL": 0.338, "IS": 0.524, "OOS1": -0.206, "OOS2": 0.252},
                "cta_ensemble_8comm": {"FULL": -0.298},
                "cta_ensemble_14sym": {"FULL": -0.451},
                "root_cause": "Carry+MomValue cross-sectional ranking creates opposing positions in bear markets",
            }
        },
        "diagnostic_8comm": {
            k: {p: {m: v for m, v in met.items()} for p, met in r.items()}
            for k, r in [
                ("trend_only", diag_results.get("TrendOnly", {})),
                ("carry_only", diag_results.get("CarryOnly", {})),
                ("mom_val_only", diag_results.get("MomValOnly", {})),
                ("cta_ensemble", diag_results.get("CTAEnsemble", {})),
            ]
        },
        "trend_only_8comm_m21": {p: m for p, m in res_trend_8.items()},
        "trend_only_14sym_m21": {p: m for p, m in res_trend_14.items()},
        "trend_only_14sym_tight": {p: m for p, m in res_trend_14_tight.items()},
        "best_strategy": best_strategy,
        "best_oos_min": best_oos_min,
        "hypothesis_validated": validated,
        "universe_expansion_helps": trend_helps,
        "verdict": verdict,
        "next_step": next_step,
        "diagnostic_verdicts": {
            "carry_is_drag": carry_is_drag,
            "mv_is_drag": mv_is_drag,
            "ensemble_hurts": ensemble_hurts,
        },
    }

    out_dir = ROOT / "artifacts" / "phase140"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "cta_trend_only_report.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    logger.info(f"\n  Results saved: {out_path}")

    # ─────────────────────────────────────────────────────────────────────────
    # PHASE 141 PLAN (printed based on outcome)
    # ─────────────────────────────────────────────────────────────────────────
    print_section("PHASE 141 PLAN")

    if validated and trend_14_oos_min > 0.3:
        logger.info("  DIRECTION: Refine Trend-only 14-sym champion")
        logger.info("    → Test EMA spans: (8,21), (20,60), (40,120)")
        logger.info("    → Test signal weights: equal vs momentum-heavy")
        logger.info("    → Test vol-targeting: instrument vol vs portfolio vol")
        logger.info("    → Walk-forward with expanding IS window")
    elif validated:
        logger.info("  DIRECTION: Investigate signal weakness")
        logger.info("    → EMA signals weak for cross-asset universe")
        logger.info("    → Add Donchian channel breakout signal (classic CTA)")
        logger.info("    → Test ATR-based position sizing (TURTLE system)")
    else:
        logger.info("  DIRECTION: PIVOT — commodity CTA EMA approach exhausted")
        logger.info("    Option A: Deribit Skew Mean-Reversion")
        logger.info("      → Crypto IV skew as leading indicator")
        logger.info("      → Mean-revert: 25D call - 25D put skew (DVOL)")
        logger.info("      → Source: Deribit API (real-time IV available)")
        logger.info("    Option B: crypto_perps P91b incremental improvements")
        logger.info("      → Current champion: Sharpe ~1.5 (LOYO)")
        logger.info("      → Phase 141: Donchian/momentum signals for crypto")
        logger.info("      → Build on KNOWN working system vs new hypothesis")

    logger.info("\n" + "=" * 70)
    logger.info("Phase 140 complete.")
    logger.info("=" * 70)

    return output


if __name__ == "__main__":
    main()
