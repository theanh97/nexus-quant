#!/usr/bin/env python3
"""
Phase 93: New Signal Profiling + 5-Signal Ensemble + 2026 OOS Validation
=========================================================================
R&D directions (post-validation):
  A) Profile I410_bw216 (5th signal candidate for balanced ensemble)
  B) Profile 5 new alpha signals: basis_momentum, lead_lag, vol_breakout,
     volume_reversal, rs_acceleration
  C) Correlation analysis: find signals orthogonal to existing 4
  D) If promising: test 5-signal ensemble (V1 + I460bw168 + I415bw216 + I410bw216 + F144)
  E) If new signals work: test expanded ensemble
  F) Validate ALL improvements with 2026 OOS (zero tuning from P91b)
"""

import copy, json, math, os, random, statistics, sys, time
from pathlib import Path

PROJ = "/Users/qtmobile/Desktop/Nexus - Quant Trading "
sys.path.insert(0, PROJ)

import numpy as np

from nexus_quant.data.providers.registry import make_provider
from nexus_quant.strategies.registry import make_strategy
from nexus_quant.backtest.engine import BacktestConfig, BacktestEngine
from nexus_quant.backtest.costs import cost_model_from_config

OUT_DIR = os.path.join(PROJ, "artifacts", "phase93")
os.makedirs(OUT_DIR, exist_ok=True)

SYMBOLS = [
    "BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT",
    "ADAUSDT", "DOGEUSDT", "AVAXUSDT", "DOTUSDT", "LINKUSDT",
]
YEAR_RANGES = {
    "2021": ("2021-01-01", "2022-01-01"),
    "2022": ("2022-01-01", "2023-01-01"),
    "2023": ("2023-01-01", "2024-01-01"),
    "2024": ("2024-01-01", "2025-01-01"),
    "2025": ("2025-01-01", "2026-01-01"),
}
YEAR_RANGES_OOS = {"2026": ("2026-01-01", "2026-02-20")}

# ── Existing champion signals ──
EXISTING_SIGNALS = {
    "v1": {"name": "nexus_alpha_v1", "params": {
        "k_per_side": 2, "w_carry": 0.35, "w_mom": 0.45, "w_confirm": 0.0,
        "w_mean_reversion": 0.20, "w_vol_momentum": 0.0, "w_funding_trend": 0.0,
        "momentum_lookback_bars": 336, "mean_reversion_lookback_bars": 72,
        "vol_lookback_bars": 168, "target_portfolio_vol": 0.0, "use_min_variance": False,
        "target_gross_leverage": 0.35, "min_gross_leverage": 0.05,
        "max_gross_leverage": 0.65, "rebalance_interval_bars": 60, "strict_agreement": False,
    }},
    "i460bw168": {"name": "idio_momentum_alpha", "params": {
        "k_per_side": 4, "lookback_bars": 460, "beta_window_bars": 168,
        "vol_lookback_bars": 168, "target_gross_leverage": 0.30, "rebalance_interval_bars": 48,
    }},
    "i415bw216": {"name": "idio_momentum_alpha", "params": {
        "k_per_side": 4, "lookback_bars": 415, "beta_window_bars": 216,
        "vol_lookback_bars": 168, "target_gross_leverage": 0.30, "rebalance_interval_bars": 48,
    }},
    "f144": {"name": "funding_momentum_alpha", "params": {
        "k_per_side": 2, "funding_lookback_bars": 144, "direction": "contrarian",
        "vol_lookback_bars": 168, "target_gross_leverage": 0.25, "rebalance_interval_bars": 24,
    }},
}

# ── New signals to profile ──
NEW_SIGNALS = {
    "i410bw216": {"name": "idio_momentum_alpha", "params": {
        "k_per_side": 4, "lookback_bars": 410, "beta_window_bars": 216,
        "vol_lookback_bars": 168, "target_gross_leverage": 0.30, "rebalance_interval_bars": 48,
    }},
    "basis_mom": {"name": "basis_momentum_alpha", "params": {
        "k_per_side": 2, "lookback_bars": 168, "vol_lookback_bars": 168,
        "target_gross_leverage": 0.25, "rebalance_interval_bars": 24,
    }},
    "lead_lag": {"name": "lead_lag_alpha", "params": {
        "k_per_side": 2, "lookback_bars": 168, "vol_lookback_bars": 168,
        "target_gross_leverage": 0.25, "rebalance_interval_bars": 48,
    }},
    "vol_breakout": {"name": "vol_breakout_alpha", "params": {
        "k_per_side": 2, "lookback_bars": 168, "vol_lookback_bars": 168,
        "target_gross_leverage": 0.25, "rebalance_interval_bars": 48,
    }},
    "vol_reversal": {"name": "volume_reversal_alpha", "params": {
        "k_per_side": 2, "lookback_bars": 168, "vol_lookback_bars": 168,
        "target_gross_leverage": 0.25, "rebalance_interval_bars": 48,
    }},
    "rs_accel": {"name": "rs_acceleration_alpha", "params": {
        "k_per_side": 2, "lookback_bars": 168, "vol_lookback_bars": 168,
        "target_gross_leverage": 0.25, "rebalance_interval_bars": 48,
    }},
}

P91B_WEIGHTS = {"v1": 0.2747, "i460bw168": 0.1967, "i415bw216": 0.3247, "f144": 0.2039}


def log(msg, end="\n"):
    print(f"[P93] {msg}", end=end, flush=True)


def run_signal(sig_cfg, year_ranges_dict, year):
    start, end = year_ranges_dict[year]
    data_cfg = {
        "provider": "binance_rest_v1", "symbols": SYMBOLS,
        "start": start, "end": end, "bar_interval": "1h",
        "cache_dir": ".cache/binance_rest",
    }
    costs_cfg = {"fee_rate": 0.0005, "slippage_rate": 0.0003, "cost_multiplier": 1.0}
    exec_cfg = {"style": "taker", "slippage_bps": 3.0}
    provider = make_provider(data_cfg, seed=42)
    dataset = provider.load()
    strategy = make_strategy({"name": sig_cfg["name"], "params": copy.deepcopy(sig_cfg["params"])})
    cost_model = cost_model_from_config(costs_cfg, execution_cfg=exec_cfg)
    engine = BacktestEngine(BacktestConfig(costs=cost_model))
    result = engine.run(dataset=dataset, strategy=strategy, seed=42)
    return result.returns


def compute_sharpe(returns):
    if not returns or len(returns) < 50:
        return 0.0
    arr = np.array(returns, dtype=np.float64)
    std = float(np.std(arr))
    return float(np.mean(arr) / std * np.sqrt(8760)) if std > 0 else 0.0


def correlation(r1, r2):
    n = min(len(r1), len(r2))
    if n < 100:
        return 0.0
    a1 = np.array(r1[:n], dtype=np.float64)
    a2 = np.array(r2[:n], dtype=np.float64)
    c = np.corrcoef(a1, a2)[0, 1]
    return float(c) if np.isfinite(c) else 0.0


def blend_returns(sig_rets, weights):
    keys = sorted(weights.keys())
    n = min(len(sig_rets.get(k, [])) for k in keys)
    if n == 0:
        return []
    R = np.zeros((len(keys), n), dtype=np.float64)
    W = np.array([weights[k] for k in keys], dtype=np.float64)
    for i, k in enumerate(keys):
        R[i, :] = sig_rets[k][:n]
    return (W @ R).tolist()


def numpy_weight_sweep(returns_dict, signal_keys, weight_ranges, step=0.025):
    """Sweep weights and find best AVG/MIN Sharpe configs."""
    import itertools
    # Build weight grid
    grids = []
    for k in signal_keys:
        lo, hi = weight_ranges[k]
        grids.append(np.arange(lo, hi + step/2, step))

    best_balanced = {"avg": 0, "min_s": -999, "weights": {}}
    best_avgmax = {"avg": 0, "min_s": -999, "weights": {}}
    n_tested = 0

    # Vectorized: load all year returns
    years = sorted(returns_dict.keys())
    year_R = {}
    for yr in years:
        n_bars = min(len(returns_dict[yr].get(k, [])) for k in signal_keys)
        if n_bars == 0:
            continue
        R = np.zeros((len(signal_keys), n_bars), dtype=np.float64)
        for i, k in enumerate(signal_keys):
            R[i, :] = returns_dict[yr][k][:n_bars]
        year_R[yr] = R

    if not year_R:
        return best_balanced, best_avgmax, 0

    for combo in itertools.product(*grids):
        w_arr = np.array(combo, dtype=np.float64)
        total = w_arr.sum()
        if total < 0.9 or total > 1.1:
            continue
        # Normalize
        w_arr = w_arr / total

        sharpes = []
        for yr in years:
            if yr not in year_R:
                continue
            blended = w_arr @ year_R[yr]
            std = float(np.std(blended))
            if std > 0:
                s = float(np.mean(blended) / std * np.sqrt(8760))
            else:
                s = 0.0
            sharpes.append(s)

        if not sharpes:
            continue

        avg_s = statistics.mean(sharpes)
        min_s = min(sharpes)
        n_tested += 1

        # Balanced: maximize MIN
        if min_s > best_balanced["min_s"] or (min_s == best_balanced["min_s"] and avg_s > best_balanced["avg"]):
            best_balanced = {"avg": round(avg_s, 4), "min_s": round(min_s, 4),
                             "weights": {k: round(float(w_arr[i]), 4) for i, k in enumerate(signal_keys)},
                             "yby": [round(s, 3) for s in sharpes]}

        # AVG-max: maximize AVG
        if avg_s > best_avgmax["avg"] or (avg_s == best_avgmax["avg"] and min_s > best_avgmax["min_s"]):
            best_avgmax = {"avg": round(avg_s, 4), "min_s": round(min_s, 4),
                           "weights": {k: round(float(w_arr[i]), 4) for i, k in enumerate(signal_keys)},
                           "yby": [round(s, 3) for s in sharpes]}

    return best_balanced, best_avgmax, n_tested


def main():
    t0 = time.time()
    report = {"phase": 93}

    # ══════════════════════════════════════════════════════════════════
    # SECTION A: Profile existing + I410_bw216 + new signals
    # ══════════════════════════════════════════════════════════════════
    log("=" * 60)
    log("SECTION A: Profile all signals (existing + new)")
    log("=" * 60)

    ALL_SIGNALS = {**EXISTING_SIGNALS, **NEW_SIGNALS}
    yr_rets = {}  # yr_rets[year][signal] = [returns]
    all_rets = {} # signal -> concat 2021-2025 returns
    profiles = {} # signal -> {year: sharpe}

    for sig_key in sorted(ALL_SIGNALS):
        profiles[sig_key] = {}
        all_rets[sig_key] = []
        for year in sorted(YEAR_RANGES):
            if year not in yr_rets:
                yr_rets[year] = {}
            log(f"  {sig_key}/{year}...")
            try:
                rets = run_signal(ALL_SIGNALS[sig_key], YEAR_RANGES, year)
            except Exception as e:
                log(f"    ERROR: {e}")
                rets = []
            yr_rets[year][sig_key] = rets
            all_rets[sig_key].extend(rets)
            s = compute_sharpe(rets)
            profiles[sig_key][year] = round(s, 3)
            log(f"    Sharpe={s:.3f}, n={len(rets)}")

    report["profiles"] = profiles

    log("\n  Signal Profiles (2021-2025):")
    log(f"  {'Signal':<15} {'2021':>7} {'2022':>7} {'2023':>7} {'2024':>7} {'2025':>7} {'AVG':>7} {'MIN':>7}")
    log(f"  {'-'*15} {'-'*7} {'-'*7} {'-'*7} {'-'*7} {'-'*7} {'-'*7} {'-'*7}")
    for sig in sorted(profiles):
        vals = [profiles[sig].get(y, 0) for y in ["2021", "2022", "2023", "2024", "2025"]]
        avg = statistics.mean(vals) if vals else 0
        mn = min(vals) if vals else 0
        marker = " ***" if sig in NEW_SIGNALS and avg > 0.5 and mn > 0 else ""
        log(f"  {sig:<15} {vals[0]:>7.3f} {vals[1]:>7.3f} {vals[2]:>7.3f} {vals[3]:>7.3f} {vals[4]:>7.3f} {avg:>7.3f} {mn:>7.3f}{marker}")

    # ══════════════════════════════════════════════════════════════════
    # SECTION B: Correlation analysis
    # ══════════════════════════════════════════════════════════════════
    log("\n" + "=" * 60)
    log("SECTION B: Correlation matrix (all signals)")
    log("=" * 60)

    sig_keys = sorted(ALL_SIGNALS.keys())
    corr_matrix = {}
    log(f"\n  {'':>15}", end="")
    for k2 in sig_keys:
        log(f" {k2[:8]:>8}", end="")
    log("")

    for k1 in sig_keys:
        corr_matrix[k1] = {}
        log(f"  {k1:<15}", end="")
        for k2 in sig_keys:
            c = correlation(all_rets.get(k1, []), all_rets.get(k2, []))
            corr_matrix[k1][k2] = round(c, 3)
            log(f" {c:>8.3f}", end="")
        log("")

    report["correlations"] = corr_matrix

    # Find promising new signals: AVG > 0.5, MIN > 0, low corr with existing
    log("\n  New signal assessment:")
    promising = []
    for sig in sorted(NEW_SIGNALS):
        vals = [profiles[sig].get(y, 0) for y in ["2021", "2022", "2023", "2024", "2025"]]
        avg = statistics.mean(vals)
        mn = min(vals)
        # Check correlation with existing signals
        max_corr_existing = max(
            abs(corr_matrix.get(sig, {}).get(ex, 0))
            for ex in EXISTING_SIGNALS
        )
        pass_filter = avg > 0.3 and mn > -0.5 and max_corr_existing < 0.5
        status = "PROMISING" if pass_filter else "SKIP"
        log(f"  {sig:<15}: AVG={avg:.3f}, MIN={mn:.3f}, max_corr_existing={max_corr_existing:.3f} -> {status}")
        if pass_filter:
            promising.append(sig)

    report["promising_new"] = promising

    # ══════════════════════════════════════════════════════════════════
    # SECTION C: 5-signal balanced sweep (V1 + I460 + I415 + I410 + F144)
    # ══════════════════════════════════════════════════════════════════
    log("\n" + "=" * 60)
    log("SECTION C: 5-signal balanced sweep (add I410_bw216)")
    log("=" * 60)

    signals_5 = ["v1", "i460bw168", "i410bw216", "i415bw216", "f144"]
    # Check I410 profile
    i410_vals = [profiles.get("i410bw216", {}).get(y, 0) for y in ["2021", "2022", "2023", "2024", "2025"]]
    log(f"  I410_bw216 profile: {i410_vals} AVG={statistics.mean(i410_vals):.3f}")

    if all(len(yr_rets.get(y, {}).get(k, [])) > 0 for y in YEAR_RANGES for k in signals_5):
        # Coarse sweep: 5% step
        weight_ranges_5 = {
            "v1": (0.15, 0.35),
            "i460bw168": (0.10, 0.25),
            "i410bw216": (0.05, 0.20),
            "i415bw216": (0.15, 0.35),
            "f144": (0.10, 0.25),
        }
        log("  Running coarse sweep (5% step)...")
        bal5, avg5, n5 = numpy_weight_sweep(yr_rets, signals_5, weight_ranges_5, step=0.05)
        log(f"  Tested {n5:,} configs")
        log(f"  5-signal BALANCED: AVG={bal5['avg']}, MIN={bal5['min_s']}")
        log(f"    Weights: {bal5['weights']}")
        log(f"    YbY: {bal5.get('yby', [])}")
        log(f"  5-signal AVG-MAX: AVG={avg5['avg']}, MIN={avg5['min_s']}")
        log(f"    Weights: {avg5['weights']}")

        report["five_signal_coarse"] = {"balanced": bal5, "avgmax": avg5, "n_tested": n5}

        # Fine sweep around balanced winner
        if bal5["min_s"] > 1.5:
            log("\n  Running fine sweep (2.5% step) around balanced winner...")
            fine_ranges = {}
            for k in signals_5:
                center = bal5["weights"].get(k, 0.2)
                fine_ranges[k] = (max(0.0, center - 0.075), min(1.0, center + 0.075))
            bal5f, avg5f, n5f = numpy_weight_sweep(yr_rets, signals_5, fine_ranges, step=0.025)
            log(f"  Tested {n5f:,} configs")
            log(f"  5-signal BALANCED (fine): AVG={bal5f['avg']}, MIN={bal5f['min_s']}")
            log(f"    Weights: {bal5f['weights']}")
            log(f"    YbY: {bal5f.get('yby', [])}")
            report["five_signal_fine"] = {"balanced": bal5f, "n_tested": n5f}

            # Compare with P91b 4-signal
            improvement = bal5f["min_s"] - 1.5761
            log(f"\n  vs P91b 4-signal (MIN=1.5761): delta={improvement:+.4f}")
    else:
        log("  Missing data for some signals, skipping 5-signal sweep")

    # ══════════════════════════════════════════════════════════════════
    # SECTION D: Test promising new signals in ensemble
    # ══════════════════════════════════════════════════════════════════
    log("\n" + "=" * 60)
    log("SECTION D: Test promising new signals in ensemble")
    log("=" * 60)

    if promising:
        for new_sig in promising:
            log(f"\n  Testing ensemble with {new_sig}:")
            signals_new = ["v1", "i460bw168", "i415bw216", "f144", new_sig]

            if all(len(yr_rets.get(y, {}).get(k, [])) > 0 for y in YEAR_RANGES for k in signals_new):
                wr = {
                    "v1": (0.15, 0.35),
                    "i460bw168": (0.10, 0.25),
                    "i415bw216": (0.20, 0.40),
                    "f144": (0.10, 0.25),
                    new_sig: (0.05, 0.20),
                }
                bal_new, avg_new, n_new = numpy_weight_sweep(yr_rets, signals_new, wr, step=0.05)
                log(f"    Tested {n_new:,} configs")
                log(f"    BALANCED: AVG={bal_new['avg']}, MIN={bal_new['min_s']}")
                log(f"    Weights: {bal_new['weights']}")
                log(f"    YbY: {bal_new.get('yby', [])}")
                report[f"ensemble_with_{new_sig}"] = {"balanced": bal_new, "avgmax": avg_new}
            else:
                log(f"    Missing data, skipping")
    else:
        log("  No promising new signals found")

    # ══════════════════════════════════════════════════════════════════
    # SECTION E: 2026 OOS validation for any improvements
    # ══════════════════════════════════════════════════════════════════
    log("\n" + "=" * 60)
    log("SECTION E: 2026 OOS validation")
    log("=" * 60)

    # Run all signals on 2026
    oos_rets = {}
    for sig_key in sorted(ALL_SIGNALS):
        log(f"  {sig_key}/2026...")
        try:
            rets = run_signal(ALL_SIGNALS[sig_key], YEAR_RANGES_OOS, "2026")
            oos_rets[sig_key] = rets
            s = compute_sharpe(rets)
            log(f"    Sharpe={s:.3f}, n={len(rets)}")
        except Exception as e:
            log(f"    ERROR: {e}")
            oos_rets[sig_key] = []

    # P91b baseline on 2026
    if all(len(oos_rets.get(k, [])) > 0 for k in P91B_WEIGHTS):
        p91b_oos = compute_sharpe(blend_returns(oos_rets, P91B_WEIGHTS))
        log(f"\n  P91b 4-signal 2026 OOS: Sharpe={p91b_oos:.4f}")
        report["oos_p91b"] = round(p91b_oos, 4)

    # 5-signal on 2026 (if found improvement)
    five_sig_best = report.get("five_signal_fine", report.get("five_signal_coarse", {})).get("balanced", {})
    if five_sig_best.get("weights") and all(len(oos_rets.get(k, [])) > 0 for k in signals_5):
        five_oos = compute_sharpe(blend_returns(oos_rets, five_sig_best["weights"]))
        log(f"  5-signal 2026 OOS: Sharpe={five_oos:.4f}")
        report["oos_5signal"] = round(five_oos, 4)

    # New signal ensembles on 2026
    for new_sig in promising:
        ens_key = f"ensemble_with_{new_sig}"
        ens_best = report.get(ens_key, {}).get("balanced", {})
        if ens_best.get("weights"):
            keys = list(ens_best["weights"].keys())
            if all(len(oos_rets.get(k, [])) > 0 for k in keys):
                ns_oos = compute_sharpe(blend_returns(oos_rets, ens_best["weights"]))
                log(f"  {new_sig} ensemble 2026 OOS: Sharpe={ns_oos:.4f}")
                report[f"oos_{new_sig}_ensemble"] = round(ns_oos, 4)

    # ══════════════════════════════════════════════════════════════════
    # SUMMARY
    # ══════════════════════════════════════════════════════════════════
    elapsed = time.time() - t0
    log("\n" + "=" * 60)
    log("PHASE 93 SUMMARY")
    log("=" * 60)

    log(f"\n  P91b baseline: AVG=2.010, MIN=1.576, 2026 OOS={report.get('oos_p91b', 'N/A')}")

    if "five_signal_fine" in report or "five_signal_coarse" in report:
        best5 = report.get("five_signal_fine", report.get("five_signal_coarse", {})).get("balanced", {})
        log(f"  5-signal:      AVG={best5.get('avg')}, MIN={best5.get('min_s')}, 2026 OOS={report.get('oos_5signal', 'N/A')}")
        if best5.get("min_s", 0) > 1.5761:
            log(f"  >>> 5-SIGNAL BEATS P91b! MIN improved by {best5['min_s'] - 1.5761:+.4f}")
        else:
            log(f"  >>> 5-signal does NOT beat P91b")

    for new_sig in promising:
        ens = report.get(f"ensemble_with_{new_sig}", {}).get("balanced", {})
        oos = report.get(f"oos_{new_sig}_ensemble", "N/A")
        log(f"  +{new_sig}: AVG={ens.get('avg')}, MIN={ens.get('min_s')}, 2026 OOS={oos}")

    log(f"\n  Elapsed: {elapsed:.0f}s")

    path = os.path.join(OUT_DIR, "phase93_report.json")
    with open(path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    log(f"  Saved: {path}")


if __name__ == "__main__":
    main()
