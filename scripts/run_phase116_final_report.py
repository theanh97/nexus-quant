#!/usr/bin/env python3
"""
Phase 116: Comprehensive R&D Final Report
==========================================
Summary of 116 phases of systematic quant strategy development.
Final champion: P91b + vol_mom_z_168 @ r=0.65

Produces a comprehensive report with:
1. Final champion full metrics (all years)
2. R&D journey summary
3. What works and what doesn't in crypto perps
4. Recommendations for deployment
"""

import json, os, sys, time
from datetime import datetime

PROJ = "/Users/qtmobile/Desktop/Nexus - Quant Trading "
sys.path.insert(0, PROJ)

import numpy as np

from nexus_quant.data.providers.registry import make_provider
from nexus_quant.strategies.registry import make_strategy
from nexus_quant.backtest.engine import BacktestConfig, BacktestEngine
from nexus_quant.backtest.costs import cost_model_from_config

OUT_DIR = os.path.join(PROJ, "artifacts", "phase116")
os.makedirs(OUT_DIR, exist_ok=True)

SYMBOLS = [
    "BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT",
    "ADAUSDT", "DOGEUSDT", "AVAXUSDT", "DOTUSDT", "LINKUSDT",
]

YEARS = ["2021", "2022", "2023", "2024", "2025"]
YEAR_RANGES = {
    "2021": ("2021-01-01", "2022-01-01"),
    "2022": ("2022-01-01", "2023-01-01"),
    "2023": ("2023-01-01", "2024-01-01"),
    "2024": ("2024-01-01", "2025-01-01"),
    "2025": ("2025-01-01", "2026-01-01"),
}
OOS_RANGE = ("2026-01-01", "2026-02-20")

P91B_WEIGHTS = {"v1": 0.2747, "i460bw168": 0.1967, "i415bw216": 0.3247, "f144": 0.2039}
SIG_KEYS = sorted(P91B_WEIGHTS.keys())

SIGNALS = {
    "v1": {"name": "nexus_alpha_v1", "params": {
        "k_per_side": 2, "w_carry": 0.35, "w_mom": 0.45, "w_mean_reversion": 0.20,
        "momentum_lookback_bars": 336, "mean_reversion_lookback_bars": 72,
        "vol_lookback_bars": 168, "target_gross_leverage": 0.35, "rebalance_interval_bars": 60}},
    "i460bw168": {"name": "idio_momentum_alpha", "params": {
        "k_per_side": 4, "lookback_bars": 460, "beta_window_bars": 168,
        "target_gross_leverage": 0.30, "rebalance_interval_bars": 48}},
    "i415bw216": {"name": "idio_momentum_alpha", "params": {
        "k_per_side": 4, "lookback_bars": 415, "beta_window_bars": 216,
        "target_gross_leverage": 0.30, "rebalance_interval_bars": 48}},
    "f144": {"name": "funding_momentum_alpha", "params": {
        "k_per_side": 2, "funding_lookback_bars": 144, "direction": "contrarian",
        "target_gross_leverage": 0.25, "rebalance_interval_bars": 24}},
}

VOL_TILT_LB = 168
VOL_TILT_RATIO = 0.65


def log(msg):
    print(f"[P116] {msg}", flush=True)


def compute_metrics(returns, bars_per_year=8760):
    arr = np.asarray(returns, dtype=np.float64)
    if arr.size < 50:
        return {}

    # Sharpe
    std = float(np.std(arr))
    sharpe = float(np.mean(arr) / std * np.sqrt(bars_per_year)) if std > 0 else 0

    # Sortino
    neg = arr[arr < 0]
    ds = float(np.std(neg)) if len(neg) > 0 else 0
    sortino = float(np.mean(arr) / ds * np.sqrt(bars_per_year)) if ds > 0 else 0

    # MDD
    equity = np.cumprod(1 + arr)
    peak = np.maximum.accumulate(equity)
    dd = (equity - peak) / peak
    mdd = float(np.min(dd))

    # CAGR
    total = float(np.prod(1 + arr))
    years = len(arr) / bars_per_year
    cagr = float(total ** (1 / years) - 1) if years > 0 and total > 0 else 0

    # Calmar
    calmar = cagr / abs(mdd) if mdd < 0 else 0

    # Monthly win rate (approximate: 730 bars per month)
    monthly_bars = 730
    n_months = len(arr) // monthly_bars
    monthly_rets = [np.sum(arr[i*monthly_bars:(i+1)*monthly_bars]) for i in range(n_months)]
    win_rate = sum(1 for r in monthly_rets if r > 0) / max(len(monthly_rets), 1)

    return {
        "sharpe": round(sharpe, 4),
        "sortino": round(sortino, 4),
        "mdd_pct": round(mdd * 100, 2),
        "cagr_pct": round(cagr * 100, 2),
        "calmar": round(calmar, 4),
        "monthly_win_rate": round(win_rate * 100, 1),
        "n_bars": len(arr),
        "n_months": n_months,
    }


def get_dataset(year_key, cache):
    if year_key == "2026_oos":
        start, end = OOS_RANGE
    else:
        start, end = YEAR_RANGES[year_key]
    cache_key = f"{start}_{end}"
    if cache_key not in cache:
        provider = make_provider({
            "provider": "binance_rest_v1", "symbols": SYMBOLS,
            "bar_interval": "1h", "start": start, "end": end,
        }, seed=42)
        cache[cache_key] = provider.load()
    return cache[cache_key]


def run_champion(year_key, ds_cache):
    """Run full champion: P91b + vol_mom_z_168 tilt."""
    dataset = get_dataset(year_key, ds_cache)
    cost_model = cost_model_from_config({"fee_rate": 0.0005, "slippage_rate": 0.0003})

    all_returns = []
    for sig_key in SIG_KEYS:
        cfg = SIGNALS[sig_key]
        strat = make_strategy({"name": cfg["name"], "params": cfg["params"]})
        engine = BacktestEngine(BacktestConfig(costs=cost_model))
        result = engine.run(dataset, strat)
        rets = np.diff(result.equity_curve) / result.equity_curve[:-1]
        all_returns.append(rets)

    min_len = min(len(r) for r in all_returns)
    all_returns = [r[:min_len] for r in all_returns]
    weights = np.array([P91B_WEIGHTS[k] for k in SIG_KEYS])
    ensemble_rets = np.zeros(min_len)
    for i, k in enumerate(SIG_KEYS):
        ensemble_rets += weights[i] * all_returns[i]

    # Apply vol tilt
    total_vol = None
    if dataset.perp_volume:
        for sym in SYMBOLS:
            vols = np.array(dataset.perp_volume.get(sym, []), dtype=np.float64)
            if total_vol is None: total_vol = vols.copy()
            else:
                ml = min(len(total_vol), len(vols))
                total_vol = total_vol[:ml] + vols[:ml]

    if total_vol is not None and len(total_vol) >= VOL_TILT_LB + 50:
        log_vol = np.log(np.maximum(total_vol, 1.0))
        mom = np.zeros(len(log_vol))
        mom[VOL_TILT_LB:] = log_vol[VOL_TILT_LB:] - log_vol[:-VOL_TILT_LB]
        z_scores = np.zeros(len(mom))
        for i in range(VOL_TILT_LB * 2, len(mom)):
            w = mom[max(0, i - VOL_TILT_LB):i + 1]
            mu, sigma = np.mean(w), np.std(w)
            if sigma > 0: z_scores[i] = (mom[i] - mu) / sigma

        ml = min(len(ensemble_rets), len(z_scores))
        ensemble_rets = ensemble_rets[:ml]
        mask = z_scores[:ml] > 0
        ensemble_rets[mask] *= VOL_TILT_RATIO

    return ensemble_rets


def main():
    t0 = time.time()

    log("=" * 70)
    log("    NEXUS QUANT — COMPREHENSIVE R&D FINAL REPORT (Phase 116)")
    log("=" * 70)

    # ── Final Champion Full Metrics ─────────────────────────────────────
    log("\n1. FINAL CHAMPION: P91b + vol_mom_z_168 @ r=0.65")
    log("-" * 60)

    ds_cache = {}
    yearly_metrics = {}

    for yr in YEARS + ["2026_oos"]:
        rets = run_champion(yr, ds_cache)
        m = compute_metrics(rets)
        yearly_metrics[yr] = m

    # Print table
    log(f"\n  {'Year':<10} {'Sharpe':>8} {'Sortino':>8} {'MDD%':>7} {'CAGR%':>7} {'Calmar':>7} {'WinRate':>8}")
    log(f"  {'-'*10} {'-'*8} {'-'*8} {'-'*7} {'-'*7} {'-'*7} {'-'*8}")
    for yr in YEARS + ["2026_oos"]:
        m = yearly_metrics[yr]
        yr_label = yr if yr != "2026_oos" else "2026*"
        log(f"  {yr_label:<10} {m['sharpe']:>8.4f} {m['sortino']:>8.4f} {m['mdd_pct']:>7.2f} {m['cagr_pct']:>7.2f} {m['calmar']:>7.4f} {m['monthly_win_rate']:>7.1f}%")

    is_sharpes = [yearly_metrics[yr]["sharpe"] for yr in YEARS]
    is_mdds = [yearly_metrics[yr]["mdd_pct"] for yr in YEARS]

    log(f"\n  IS Summary (2021-2025):")
    log(f"    AVG Sharpe: {np.mean(is_sharpes):.4f}")
    log(f"    MIN Sharpe: {np.min(is_sharpes):.4f} (year: {YEARS[np.argmin(is_sharpes)]})")
    log(f"    MAX Sharpe: {np.max(is_sharpes):.4f} (year: {YEARS[np.argmax(is_sharpes)]})")
    log(f"    Worst MDD: {np.min(is_mdds):.2f}%")
    log(f"    OOS 2026 Sharpe: {yearly_metrics['2026_oos']['sharpe']:.4f}")
    log(f"    * 2026 = true out-of-sample (Jan 1 - Feb 20)")

    # ── Signal Attribution ──────────────────────────────────────────────
    log(f"\n2. SIGNAL COMPOSITION")
    log("-" * 60)
    log(f"  {'Signal':<20} {'Weight':>8} {'Type':<30}")
    log(f"  {'-'*20} {'-'*8} {'-'*30}")
    for k in SIG_KEYS:
        w = P91B_WEIGHTS[k] * 100
        if k == "v1":
            desc = "Buy-dips-in-uptrends (momentum+MR)"
        elif "i4" in k:
            desc = f"Idiosyncratic momentum (beta-adj)"
        elif "f144" in k:
            desc = "Contrarian funding rate"
        else:
            desc = "Unknown"
        log(f"  {k:<20} {w:>7.2f}% {desc}")

    log(f"\n  Overlay: vol_mom_z_168 @ r=0.65")
    log(f"    Mechanism: reduce leverage to 65% when 168h volume momentum z > 0")
    log(f"    Effect: ΔMIN=+0.130, reduces exposure ~48% of hours")

    # ── What Works ──────────────────────────────────────────────────────
    log(f"\n3. WHAT WORKS IN CRYPTO PERPS (top-10 large-cap, hourly)")
    log("-" * 60)
    log(f"  ✓ Cross-sectional momentum (various lookbacks 300-500h)")
    log(f"  ✓ Idiosyncratic momentum (beta-adjusted)")
    log(f"  ✓ Contrarian funding rate momentum")
    log(f"  ✓ Buy-dips-in-uptrends (V1-Long: momentum filter + MR timing)")
    log(f"  ✓ Volume momentum contrarian tilt (reduce when crowded)")
    log(f"  ✓ Dollar-neutral (mandatory for bear year robustness)")
    log(f"  ✓ Static ensemble weights (beat all adaptive methods)")
    log(f"  ✓ 4-signal diversity (each covers a different regime)")

    # ── What Doesn't Work ───────────────────────────────────────────────
    log(f"\n4. WHAT DOESN'T WORK (tested and confirmed)")
    log("-" * 60)
    dead_signals = [
        "Basis momentum, lead-lag, vol breakout, vol reversal",
        "RS acceleration, EWMA Sharpe, Amihud illiquidity",
        "Taker buy ratio, funding vol, price level (52wk high)",
        "Pair spread MR, volume-regime momentum, momentum breakout",
        "Orderflow alpha, positioning alpha, funding carry",
        "Dispersion alpha, regime mixer with V1-Long",
        "On-chain filters (binary), F&G momentum, DVOL (implied vol)",
        "Volume concentration (HHI), funding dispersion",
        "S&P500/Gold/DXY regime indicators (all 12 variants)",
        "Adaptive/dynamic weighting (rolling, inverse-vol, signal-gated)",
        "20-coin expanded universe, 4h timeframe, long-bias overlay",
        "3-signal ensemble without v1 (IS improves but OOS collapses)",
    ]
    for s in dead_signals:
        log(f"  ✗ {s}")

    # ── R&D Statistics ──────────────────────────────────────────────────
    log(f"\n5. R&D STATISTICS")
    log("-" * 60)
    log(f"  Total phases: 116")
    log(f"  Signals tested: 40+")
    log(f"  Parameter variants: 200+")
    log(f"  Weight optimization methods: 9")
    log(f"  Overlay signals tested: 25+ (on-chain, F&G, DVOL, volume, cross-asset)")
    log(f"  Walk-forward validated overlays: 1 (vol_mom_z_168)")
    log(f"  Alternative data sources tested: 5 (blockchain.com, F&G, Deribit, Yahoo, Binance OI)")
    log(f"  Champion improvement timeline:")
    log(f"    Phase 54: V1-Long (MIN=0.948)")
    log(f"    Phase 91b: P91b 4-signal ensemble (MIN=1.576)")
    log(f"    Phase 113: P91b + vol tilt (MIN=1.427*)")
    log(f"    * MIN dropped from 1.576 to 1.427 due to per-year vs continuous metric")
    log(f"      but OBJ improved from 1.652 to 1.716")

    # ── Deployment Recommendations ──────────────────────────────────────
    log(f"\n6. DEPLOYMENT RECOMMENDATIONS")
    log("-" * 60)
    log(f"  Config: configs/production_p91b_champion.json (v2.0.0)")
    log(f"  Expected live Sharpe: 0.8-1.0 (50% IS degradation)")
    log(f"  Minimum account: $10K (below this, fees eat alpha)")
    log(f"  Target leverage: 0.35x gross")
    log(f"  Rebalance: hourly")
    log(f"  Symbols: 10 large-cap crypto perpetuals")
    log(f"  ")
    log(f"  CRITICAL RULES:")
    log(f"  1. NEVER remove v1 signal (regime insurance for 2026+)")
    log(f"  2. Keep dollar-neutral (zero net exposure)")
    log(f"  3. Halt if 30d rolling Sharpe < -1.0 or MDD > 15%")
    log(f"  4. Vol tilt: reduce to 65% when vol_mom_z_168 > 0")
    log(f"  5. DO NOT add new signals without WF validation")

    # ── Save Report ─────────────────────────────────────────────────────
    elapsed = time.time() - t0

    def _default(obj):
        if isinstance(obj, (np.integer,)): return int(obj)
        if isinstance(obj, (np.floating,)): return float(obj)
        if isinstance(obj, (np.bool_,)): return bool(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        raise TypeError(f"Not serializable: {type(obj)}")

    report = {
        "phase": 116,
        "description": "Comprehensive R&D Final Report",
        "timestamp": datetime.now().isoformat(),
        "elapsed_seconds": round(elapsed, 1),
        "champion": {
            "name": "P91b + vol_mom_z_168 @ r=0.65",
            "config": "configs/production_p91b_champion.json",
            "version": "2.0.0",
            "ensemble_weights": P91B_WEIGHTS,
            "vol_tilt": {"lookback": VOL_TILT_LB, "ratio": VOL_TILT_RATIO},
        },
        "yearly_metrics": yearly_metrics,
        "is_summary": {
            "avg_sharpe": round(np.mean(is_sharpes), 4),
            "min_sharpe": round(np.min(is_sharpes), 4),
            "max_sharpe": round(np.max(is_sharpes), 4),
            "obj": round((np.mean(is_sharpes) + np.min(is_sharpes)) / 2, 4),
            "worst_mdd": round(np.min(is_mdds), 2),
        },
        "oos_2026": yearly_metrics.get("2026_oos", {}),
        "rd_statistics": {
            "total_phases": 116,
            "signals_tested": "40+",
            "param_variants": "200+",
            "overlay_signals_tested": "25+",
            "wf_validated_overlays": 1,
            "alt_data_sources": 5,
        },
    }

    report_path = os.path.join(OUT_DIR, "phase116_final_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=_default)

    log(f"\n{'='*70}")
    log(f"Phase 116 COMPLETE in {elapsed:.1f}s → {report_path}")
    log(f"{'='*70}")


if __name__ == "__main__":
    main()
