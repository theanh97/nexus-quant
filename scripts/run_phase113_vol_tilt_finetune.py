#!/usr/bin/env python3
"""
Phase 113: Fine-Tune Vol Tilt Ratio + Production Config Update
================================================================
Phase 112 confirmed vol_mom_z_168 @ r=0.7 as new champion.
This phase:
1. Fine-grid search around r=0.7 (0.55-0.85, step 0.05)
2. Test sensitivity to lookback (144, 168, 192, 216)
3. Update production config with best parameters
4. Final scorecard for the new champion
"""

import copy, json, os, sys, time
from datetime import datetime

PROJ = "/Users/qtmobile/Desktop/Nexus - Quant Trading "
sys.path.insert(0, PROJ)

import numpy as np

from nexus_quant.data.providers.registry import make_provider
from nexus_quant.strategies.registry import make_strategy
from nexus_quant.backtest.engine import BacktestConfig, BacktestEngine
from nexus_quant.backtest.costs import cost_model_from_config

OUT_DIR = os.path.join(PROJ, "artifacts", "phase113")
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


def log(msg):
    print(f"[P113] {msg}", flush=True)


def compute_sharpe(returns, bars_per_year=8760):
    arr = np.asarray(returns, dtype=np.float64)
    if arr.size < 50: return 0.0
    std = float(np.std(arr))
    if std <= 0: return 0.0
    return float(np.mean(arr) / std * np.sqrt(bars_per_year))


def compute_mdd(returns):
    arr = np.asarray(returns, dtype=np.float64)
    equity = np.cumprod(1 + arr)
    peak = np.maximum.accumulate(equity)
    dd = (equity - peak) / peak
    return float(np.min(dd))


def compute_cagr(returns, bars_per_year=8760):
    arr = np.asarray(returns, dtype=np.float64)
    total = float(np.prod(1 + arr))
    years = len(arr) / bars_per_year
    if years <= 0 or total <= 0: return 0.0
    return float(total ** (1 / years) - 1)


def get_dataset(year_key, cache):
    if year_key == "2026_oos":
        start, end = OOS_RANGE
    else:
        start, end = YEAR_RANGES[year_key]
    cache_key = f"{start}_{end}"
    if cache_key not in cache:
        provider = make_provider({
            "provider": "binance_rest_v1",
            "symbols": SYMBOLS,
            "bar_interval": "1h",
            "start": start,
            "end": end,
        }, seed=42)
        cache[cache_key] = provider.load()
    return cache[cache_key]


def run_p91b_returns(dataset):
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
    return ensemble_rets


def compute_volume_z(dataset, lookback):
    total_vol = None
    if dataset.perp_volume:
        for sym in SYMBOLS:
            vols = np.array(dataset.perp_volume.get(sym, []), dtype=np.float64)
            if total_vol is None:
                total_vol = vols.copy()
            else:
                min_l = min(len(total_vol), len(vols))
                total_vol = total_vol[:min_l] + vols[:min_l]
    if total_vol is None or len(total_vol) < lookback + 50:
        return None
    log_vol = np.log(np.maximum(total_vol, 1.0))
    mom = np.zeros(len(log_vol))
    mom[lookback:] = log_vol[lookback:] - log_vol[:-lookback]
    z_scores = np.zeros(len(mom))
    for i in range(lookback * 2, len(mom)):
        window = mom[max(0, i - lookback):i + 1]
        mu = np.mean(window)
        sigma = np.std(window)
        if sigma > 0:
            z_scores[i] = (mom[i] - mu) / sigma
    return z_scores


def apply_tilt(rets, z_scores, ratio):
    min_len = min(len(rets), len(z_scores))
    tilted = rets[:min_len].copy()
    mask = z_scores[:min_len] > 0
    tilted[mask] *= ratio
    return tilted


def main():
    t0 = time.time()
    log("Phase 113: Fine-Tune Vol Tilt")
    log("=" * 60)

    # Fetch all data
    ds_cache = {}
    yearly_rets = {}
    log("Fetching data and computing P91b returns...")
    for yr in YEARS + ["2026_oos"]:
        dataset = get_dataset(yr, ds_cache)
        yearly_rets[yr] = run_p91b_returns(dataset)

    # Baseline
    baseline = {}
    for yr in YEARS + ["2026_oos"]:
        baseline[yr] = round(compute_sharpe(yearly_rets[yr]), 4)
    base_is = [baseline[yr] for yr in YEARS]
    base_avg = np.mean(base_is)
    base_min = np.min(base_is)
    base_obj = (base_avg + base_min) / 2
    log(f"Baseline: AVG={base_avg:.4f}, MIN={base_min:.4f}, OBJ={base_obj:.4f}, OOS={baseline['2026_oos']}")

    # ── Test 1: Fine-grid ratio around 0.7 ──────────────────────────────
    log("\n" + "=" * 60)
    log("Test 1: Fine-grid ratio (lookback=168)")
    log("=" * 60)

    ratios = [round(0.55 + i * 0.05, 2) for i in range(7)]  # 0.55 to 0.85
    ratio_results = {}

    for ratio in ratios:
        yr_sharpes = []
        for yr in YEARS:
            ds = get_dataset(yr, ds_cache)
            z = compute_volume_z(ds, 168)
            tilted = apply_tilt(yearly_rets[yr], z, ratio)
            yr_sharpes.append(compute_sharpe(tilted))

        avg_s = np.mean(yr_sharpes)
        min_s = np.min(yr_sharpes)
        obj = (avg_s + min_s) / 2

        # OOS
        ds_oos = get_dataset("2026_oos", ds_cache)
        z_oos = compute_volume_z(ds_oos, 168)
        tilted_oos = apply_tilt(yearly_rets["2026_oos"], z_oos, ratio)
        oos_s = compute_sharpe(tilted_oos)

        ratio_results[ratio] = {
            "avg": round(avg_s, 4), "min": round(min_s, 4),
            "obj": round(obj, 4), "oos": round(oos_s, 4)
        }
        log(f"  r={ratio}: AVG={avg_s:.4f}, MIN={min_s:.4f}, OBJ={obj:.4f}, OOS={oos_s:.4f}")

    best_ratio = max(ratio_results, key=lambda r: ratio_results[r]["obj"])
    log(f"  BEST ratio: {best_ratio} (OBJ={ratio_results[best_ratio]['obj']})")

    # ── Test 2: Lookback sensitivity ────────────────────────────────────
    log("\n" + "=" * 60)
    log("Test 2: Lookback sensitivity (ratio=0.7)")
    log("=" * 60)

    lookbacks = [120, 144, 168, 192, 216, 240]
    lb_results = {}

    for lb in lookbacks:
        yr_sharpes = []
        for yr in YEARS:
            ds = get_dataset(yr, ds_cache)
            z = compute_volume_z(ds, lb)
            if z is None:
                yr_sharpes.append(compute_sharpe(yearly_rets[yr]))
                continue
            tilted = apply_tilt(yearly_rets[yr], z, 0.7)
            yr_sharpes.append(compute_sharpe(tilted))

        avg_s = np.mean(yr_sharpes)
        min_s = np.min(yr_sharpes)
        obj = (avg_s + min_s) / 2

        ds_oos = get_dataset("2026_oos", ds_cache)
        z_oos = compute_volume_z(ds_oos, lb)
        if z_oos is not None:
            tilted_oos = apply_tilt(yearly_rets["2026_oos"], z_oos, 0.7)
            oos_s = compute_sharpe(tilted_oos)
        else:
            oos_s = baseline["2026_oos"]

        lb_results[lb] = {
            "avg": round(avg_s, 4), "min": round(min_s, 4),
            "obj": round(obj, 4), "oos": round(oos_s, 4)
        }
        log(f"  lb={lb}: AVG={avg_s:.4f}, MIN={min_s:.4f}, OBJ={obj:.4f}, OOS={oos_s:.4f}")

    best_lb = max(lb_results, key=lambda lb: lb_results[lb]["obj"])
    log(f"  BEST lookback: {best_lb} (OBJ={lb_results[best_lb]['obj']})")

    # ── Test 3: Best combo (best_ratio × best_lb) ──────────────────────
    log("\n" + "=" * 60)
    log(f"Test 3: Best combo — lb={best_lb}, r={best_ratio}")
    log("=" * 60)

    final_yearly = {}
    for yr in YEARS + ["2026_oos"]:
        ds = get_dataset(yr, ds_cache)
        z = compute_volume_z(ds, best_lb)
        if z is not None:
            tilted = apply_tilt(yearly_rets[yr], z, best_ratio)
        else:
            tilted = yearly_rets[yr]

        s = compute_sharpe(tilted)
        mdd = compute_mdd(tilted) * 100
        cagr = compute_cagr(tilted) * 100
        final_yearly[yr] = {
            "sharpe": round(s, 4),
            "mdd": round(mdd, 2),
            "cagr": round(cagr, 2),
        }
        delta = s - baseline[yr]
        log(f"  {yr}: Sharpe={s:.4f} (Δ{delta:+.4f}), MDD={mdd:.2f}%, CAGR={cagr:.2f}%")

    final_is = [final_yearly[yr]["sharpe"] for yr in YEARS]
    final_avg = np.mean(final_is)
    final_min = np.min(final_is)
    final_obj = (final_avg + final_min) / 2

    log(f"\n  FINAL: AVG={final_avg:.4f}, MIN={final_min:.4f}, OBJ={final_obj:.4f}")
    log(f"  vs baseline: ΔAVG={final_avg - base_avg:+.4f}, ΔMIN={final_min - base_min:+.4f}, ΔOBJ={final_obj - base_obj:+.4f}")

    # ── Summary ─────────────────────────────────────────────────────────
    log("\n" + "=" * 60)
    log("PHASE 113 SUMMARY")
    log("=" * 60)
    log(f"  Fine-grid ratio: best={best_ratio}")
    log(f"  Lookback sensitivity: best={best_lb}")
    log(f"  Best combo: lb={best_lb}, r={best_ratio}")
    log(f"  Final OBJ: {final_obj:.4f} (baseline {base_obj:.4f}, Δ{final_obj - base_obj:+.4f})")

    if final_obj > base_obj:
        log(f"\n  NEW CHAMPION CONFIG:")
        log(f"    P91b + vol_mom_z_{best_lb} @ r={best_ratio}")
        log(f"    AVG={final_avg:.4f}, MIN={final_min:.4f}")
        log(f"    OOS 2026: {final_yearly['2026_oos']['sharpe']}")
    else:
        log(f"\n  No improvement over baseline. P91b unchanged.")

    # ── Save report ─────────────────────────────────────────────────────
    elapsed = time.time() - t0

    def _default(obj):
        if isinstance(obj, (np.integer,)): return int(obj)
        if isinstance(obj, (np.floating,)): return float(obj)
        if isinstance(obj, (np.bool_,)): return bool(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

    report = {
        "phase": 113,
        "description": "Fine-tune vol tilt + production config",
        "timestamp": datetime.now().isoformat(),
        "elapsed_seconds": round(elapsed, 1),
        "baseline_obj": round(base_obj, 4),
        "ratio_grid": ratio_results,
        "lookback_grid": {str(k): v for k, v in lb_results.items()},
        "best_ratio": best_ratio,
        "best_lookback": best_lb,
        "final_yearly": final_yearly,
        "final_obj": round(final_obj, 4),
        "improvement": round(final_obj - base_obj, 4),
        "is_champion": final_obj > base_obj,
    }

    report_path = os.path.join(OUT_DIR, "phase113_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=_default)

    log(f"\nPhase 113 COMPLETE in {elapsed:.1f}s → {report_path}")


if __name__ == "__main__":
    main()
