#!/usr/bin/env python3
"""
Phase 119: Cost Sensitivity + Parameter Perturbation + Monthly Stability
========================================================================
Completes the Phase 118 robustness tests that timed out.
Optimized: caches data aggressively, signal returns computed once.
"""

import copy, json, os, signal, sys, time
from datetime import datetime

PROJ = "/Users/qtmobile/Desktop/Nexus - Quant Trading "
sys.path.insert(0, PROJ)

import numpy as np

from nexus_quant.data.providers.registry import make_provider
from nexus_quant.strategies.registry import make_strategy
from nexus_quant.backtest.engine import BacktestConfig, BacktestEngine
from nexus_quant.backtest.costs import cost_model_from_config

OUT_DIR = os.path.join(PROJ, "artifacts", "phase119")
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

VOL_TILT_LOOKBACK = 168
VOL_TILT_RATIO = 0.65
RNG = np.random.RandomState(42)

# Partial results saved on timeout
_partial_results = {}


def log(msg):
    print(f"[P119] {msg}", flush=True)


def compute_sharpe(returns, bars_per_year=8760):
    arr = np.asarray(returns, dtype=np.float64)
    if arr.size < 50:
        return 0.0
    std = float(np.std(arr))
    if std <= 0:
        return 0.0
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
    if years <= 0 or total <= 0:
        return 0.0
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


def run_signal_returns(dataset, sig_key, cost_cfg=None):
    if cost_cfg is None:
        cost_cfg = {"fee_rate": 0.0005, "slippage_rate": 0.0003}
    cost_model = cost_model_from_config(cost_cfg)
    cfg = SIGNALS[sig_key]
    strat = make_strategy({"name": cfg["name"], "params": cfg["params"]})
    engine = BacktestEngine(BacktestConfig(costs=cost_model))
    result = engine.run(dataset, strat)
    return np.diff(result.equity_curve) / result.equity_curve[:-1]


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


def save_partial(report):
    """Save whatever we have so far (timeout safety)."""
    report_path = os.path.join(OUT_DIR, "phase119_report.json")
    def _default(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=_default)
    log(f"  Saved partial → {report_path}")


def _timeout_handler(signum, frame):
    log("TIMEOUT — saving partial results")
    save_partial(_partial_results)
    sys.exit(0)


# ─── Test A: Parameter Perturbation (FAST — cached signal returns) ────

def parameter_perturbation(sig_returns_cache, z_cache, n_trials=500):
    """
    Perturb ensemble weights ±20% and vol tilt ratio/lookback.
    Uses CACHED signal returns — only array math, no backtest re-runs.
    """
    WEIGHT_RANGES = {k: (v * 0.8, v * 1.2) for k, v in P91B_WEIGHTS.items()}
    LOOKBACKS = sorted(set(lb for (_, lb) in z_cache.keys()))

    trial_results = []
    for trial in range(n_trials):
        # Random weights (normalized)
        raw_w = {k: RNG.uniform(lo, hi) for k, (lo, hi) in WEIGHT_RANGES.items()}
        wt_sum = sum(raw_w.values())
        norm_w = {k: v / wt_sum for k, v in raw_w.items()}
        w_arr = np.array([norm_w[k] for k in SIG_KEYS])

        # Random vol tilt params
        vt_lb = RNG.choice(LOOKBACKS)
        vt_ratio = RNG.uniform(0.52, 0.78)

        yr_sharpes = []
        for yr in YEARS:
            rets_list = [sig_returns_cache[(sk, yr)] for sk in SIG_KEYS]
            min_len = min(len(r) for r in rets_list)
            ens = np.zeros(min_len)
            for i in range(len(SIG_KEYS)):
                ens += w_arr[i] * rets_list[i][:min_len]
            z = z_cache.get((yr, vt_lb))
            if z is not None:
                ens = apply_tilt(ens, z, vt_ratio)
            yr_sharpes.append(compute_sharpe(ens))

        avg_s = np.mean(yr_sharpes)
        min_s = np.min(yr_sharpes)
        obj = (avg_s + min_s) / 2
        trial_results.append({
            "obj": obj, "avg": avg_s, "min": min_s,
            "weights": norm_w, "vt_lb": int(vt_lb), "vt_ratio": float(vt_ratio),
        })

        if (trial + 1) % 100 == 0:
            objs = [t["obj"] for t in trial_results]
            log(f"  Perturbation {trial+1}/{n_trials}: median OBJ={np.median(objs):.4f}")

    objs = np.array([t["obj"] for t in trial_results])
    avgs = np.array([t["avg"] for t in trial_results])
    mins = np.array([t["min"] for t in trial_results])

    # Find best trial
    best_idx = np.argmax(objs)
    best = trial_results[best_idx]

    return {
        "n_trials": n_trials,
        "obj": {
            "mean": round(float(np.mean(objs)), 4),
            "median": round(float(np.median(objs)), 4),
            "std": round(float(np.std(objs)), 4),
            "min": round(float(np.min(objs)), 4),
            "max": round(float(np.max(objs)), 4),
            "p5": round(float(np.percentile(objs, 5)), 4),
            "p25": round(float(np.percentile(objs, 25)), 4),
            "p75": round(float(np.percentile(objs, 75)), 4),
            "p95": round(float(np.percentile(objs, 95)), 4),
        },
        "avg_sharpe": {
            "mean": round(float(np.mean(avgs)), 4),
            "std": round(float(np.std(avgs)), 4),
        },
        "min_sharpe": {
            "mean": round(float(np.mean(mins)), 4),
            "std": round(float(np.std(mins)), 4),
        },
        "pct_above_1_0": round(float(np.mean(objs > 1.0) * 100), 1),
        "pct_above_1_5": round(float(np.mean(objs > 1.5) * 100), 1),
        "pct_above_champion": round(float(np.mean(objs > 1.7159) * 100), 1),
        "best_trial": {
            "obj": round(best["obj"], 4),
            "avg": round(best["avg"], 4),
            "min": round(best["min"], 4),
            "weights": {k: round(v, 4) for k, v in best["weights"].items()},
            "vt_lb": best["vt_lb"],
            "vt_ratio": round(best["vt_ratio"], 4),
        },
        "interpretation": "",  # Filled in main()
    }


# ─── Test B: Transaction Cost Sensitivity ─────────────────────────────

def cost_sensitivity(ds_cache):
    """Scale transaction costs from 0x to 3x and measure degradation."""
    multipliers = [0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 2.5, 3.0]
    base_fee = 0.0005
    base_slip = 0.0003
    results = {}

    for mult in multipliers:
        cost_cfg = {
            "fee_rate": base_fee * mult,
            "slippage_rate": base_slip * mult,
        }
        yr_sharpes = {}
        for yr in YEARS:
            ds = get_dataset(yr, ds_cache)
            base_rets = run_p91b_returns(ds, cost_cfg)
            z = compute_volume_z(ds, VOL_TILT_LOOKBACK)
            if z is not None:
                rets = apply_tilt(base_rets, z, VOL_TILT_RATIO)
            else:
                rets = base_rets
            yr_sharpes[yr] = round(compute_sharpe(rets), 4)

        vals = list(yr_sharpes.values())
        avg_s = np.mean(vals)
        min_s = np.min(vals)
        obj = (avg_s + min_s) / 2
        results[str(mult)] = {
            "cost_mult": mult,
            "fee_bps": round(base_fee * mult * 10000, 1),
            "slip_bps": round(base_slip * mult * 10000, 1),
            "total_bps": round((base_fee + base_slip) * mult * 10000, 1),
            "avg_sharpe": round(float(avg_s), 4),
            "min_sharpe": round(float(min_s), 4),
            "obj": round(float(obj), 4),
            "yearly": yr_sharpes,
        }
        log(f"  Cost {mult:.2f}x ({results[str(mult)]['total_bps']:.0f}bps): AVG={avg_s:.4f}, MIN={min_s:.4f}, OBJ={obj:.4f}")

    # Find breakeven (where MIN Sharpe drops below 0)
    sorted_m = sorted(float(m) for m in results.keys())
    breakeven_obj = None
    breakeven_min = None
    for i in range(len(sorted_m) - 1):
        m1, m2 = sorted_m[i], sorted_m[i + 1]
        o1 = results[str(m1)]["obj"]
        o2 = results[str(m2)]["obj"]
        if o1 > 0 and o2 <= 0:
            breakeven_obj = m1 + (m2 - m1) * o1 / (o1 - o2)
        mn1 = results[str(m1)]["min_sharpe"]
        mn2 = results[str(m2)]["min_sharpe"]
        if mn1 > 0 and mn2 <= 0:
            breakeven_min = m1 + (m2 - m1) * mn1 / (mn1 - mn2)

    if breakeven_obj is None and results[str(sorted_m[-1])]["obj"] > 0:
        breakeven_obj = ">3.0x"
    if breakeven_min is None and results[str(sorted_m[-1])]["min_sharpe"] > 0:
        breakeven_min = ">3.0x"

    return {
        "base_cost": f"{(base_fee + base_slip) * 10000:.0f} bps ({base_fee * 10000:.0f} fee + {base_slip * 10000:.0f} slip)",
        "results": results,
        "breakeven_obj": breakeven_obj,
        "breakeven_min_sharpe": breakeven_min,
    }


def run_p91b_returns(dataset, cost_cfg=None):
    all_returns = []
    for sig_key in SIG_KEYS:
        rets = run_signal_returns(dataset, sig_key, cost_cfg)
        all_returns.append(rets)
    min_len = min(len(r) for r in all_returns)
    all_returns = [r[:min_len] for r in all_returns]
    weights = np.array([P91B_WEIGHTS[k] for k in SIG_KEYS])
    ensemble_rets = np.zeros(min_len)
    for i, k in enumerate(SIG_KEYS):
        ensemble_rets += weights[i] * all_returns[i]
    return ensemble_rets


# ─── Test C: Monthly Stability ────────────────────────────────────────

def monthly_stability(sig_returns_cache, z_cache):
    """Monthly Sharpe distribution for each year."""
    results = {}
    for yr in YEARS + ["2026_oos"]:
        # Reconstruct champion returns from cache
        if yr == "2026_oos":
            # Skip if not in cache (OOS not pre-loaded for perturbation)
            results[yr] = {"note": "skipped — OOS not pre-cached for this test"}
            continue

        rets_list = [sig_returns_cache[(sk, yr)] for sk in SIG_KEYS]
        min_len = min(len(r) for r in rets_list)
        weights = np.array([P91B_WEIGHTS[k] for k in SIG_KEYS])
        ens = np.zeros(min_len)
        for i in range(len(SIG_KEYS)):
            ens += weights[i] * rets_list[i][:min_len]
        z = z_cache.get((yr, VOL_TILT_LOOKBACK))
        if z is not None:
            ens = apply_tilt(ens, z, VOL_TILT_RATIO)

        # Split into ~monthly blocks (730 bars ≈ 1 month)
        block_size = 730
        n_blocks = len(ens) // block_size
        month_sharpes = []
        for b in range(n_blocks):
            chunk = ens[b * block_size:(b + 1) * block_size]
            month_sharpes.append(compute_sharpe(chunk))

        if not month_sharpes:
            results[yr] = {"note": "insufficient data"}
            continue

        ms = np.array(month_sharpes)
        results[yr] = {
            "n_months": len(month_sharpes),
            "mean": round(float(np.mean(ms)), 4),
            "median": round(float(np.median(ms)), 4),
            "std": round(float(np.std(ms)), 4),
            "min": round(float(np.min(ms)), 4),
            "max": round(float(np.max(ms)), 4),
            "pct_positive": round(float(np.mean(ms > 0) * 100), 1),
            "pct_above_1": round(float(np.mean(ms > 1.0) * 100), 1),
            "worst_month": round(float(np.min(ms)), 4),
            "best_month": round(float(np.max(ms)), 4),
        }
    return results


def main():
    global _partial_results
    signal.signal(signal.SIGTERM, _timeout_handler)
    signal.signal(signal.SIGALRM, _timeout_handler)
    # 8 minute hard timeout
    signal.alarm(480)

    t0 = time.time()
    log("Phase 119: Cost Sensitivity + Parameter Perturbation + Monthly Stability")
    log("=" * 70)

    ds_cache = {}
    _partial_results = {
        "phase": 119,
        "description": "Cost sensitivity, parameter perturbation, monthly stability",
        "timestamp": datetime.now().isoformat(),
    }

    # ── Step 1: Load all data + cache signal returns ──────────────────
    log("\nStep 1: Loading data & caching signal returns (4 signals × 5 years)...")
    t_load = time.time()

    sig_returns_cache = {}  # (sig_key, year) -> np.array
    for yr in YEARS:
        ds = get_dataset(yr, ds_cache)
        for sig_key in SIG_KEYS:
            rets = run_signal_returns(ds, sig_key)
            sig_returns_cache[(sig_key, yr)] = rets
            log(f"  {sig_key}/{yr}: {len(rets)} bars")

    # Cache z-scores for multiple lookbacks (perturbation test)
    z_cache = {}
    for yr in YEARS:
        ds = get_dataset(yr, ds_cache)
        for lb in range(120, 250, 12):
            z_cache[(yr, lb)] = compute_volume_z(ds, lb)

    log(f"  Data loaded in {time.time() - t_load:.1f}s")

    # Baseline champion
    baseline_sharpes = {}
    for yr in YEARS:
        rets_list = [sig_returns_cache[(sk, yr)] for sk in SIG_KEYS]
        min_len = min(len(r) for r in rets_list)
        weights = np.array([P91B_WEIGHTS[k] for k in SIG_KEYS])
        ens = np.zeros(min_len)
        for i in range(len(SIG_KEYS)):
            ens += weights[i] * rets_list[i][:min_len]
        z = z_cache.get((yr, VOL_TILT_LOOKBACK))
        if z is not None:
            ens = apply_tilt(ens, z, VOL_TILT_RATIO)
        baseline_sharpes[yr] = round(compute_sharpe(ens), 4)
    champ_avg = np.mean(list(baseline_sharpes.values()))
    champ_min = np.min(list(baseline_sharpes.values()))
    champ_obj = (champ_avg + champ_min) / 2
    log(f"  Champion baseline: AVG={champ_avg:.4f}, MIN={champ_min:.4f}, OBJ={champ_obj:.4f}")
    _partial_results["champion_baseline"] = baseline_sharpes
    _partial_results["champion_obj"] = round(float(champ_obj), 4)

    # ── Test A: Parameter Perturbation ────────────────────────────────
    log("\n" + "=" * 70)
    log("Test A: Parameter Perturbation (±20% weights, ±20% vol tilt, 500 trials)")
    log("=" * 70)
    t_a = time.time()

    perturb = parameter_perturbation(sig_returns_cache, z_cache, 500)

    # Add interpretation
    if perturb["pct_above_1_0"] > 95:
        interp = "VERY ROBUST — 95%+ perturbations maintain positive OBJ. Champion is on a broad plateau."
    elif perturb["pct_above_1_0"] > 80:
        interp = "ROBUST — 80%+ perturbations maintain Sharpe > 1. Strategy survives parameter mis-specification."
    else:
        interp = "FRAGILE — significant fraction of perturbations fall below 1.0. Precise tuning matters."
    perturb["interpretation"] = interp

    log(f"\n  Perturbation Summary:")
    log(f"    OBJ: median={perturb['obj']['median']:.4f}, [P5={perturb['obj']['p5']:.4f}, P95={perturb['obj']['p95']:.4f}]")
    log(f"    % OBJ > 1.0: {perturb['pct_above_1_0']:.1f}%")
    log(f"    % OBJ > 1.5: {perturb['pct_above_1_5']:.1f}%")
    log(f"    % OBJ > champion ({champ_obj:.4f}): {perturb['pct_above_champion']:.1f}%")
    log(f"    Best trial OBJ: {perturb['best_trial']['obj']:.4f}")
    log(f"    Verdict: {interp}")
    log(f"    Time: {time.time() - t_a:.1f}s")

    _partial_results["parameter_perturbation"] = perturb
    save_partial(_partial_results)

    # ── Test B: Cost Sensitivity ──────────────────────────────────────
    log("\n" + "=" * 70)
    log("Test B: Cost Sensitivity (0x - 3.0x, 9 levels)")
    log("=" * 70)
    t_b = time.time()

    cost = cost_sensitivity(ds_cache)
    log(f"\n  Cost Sensitivity Summary:")
    log(f"    Base cost: {cost['base_cost']}")
    log(f"    Breakeven (OBJ<0): {cost['breakeven_obj']}x")
    log(f"    Breakeven (MIN Sharpe<0): {cost['breakeven_min_sharpe']}x")
    log(f"    At 2x costs: OBJ={cost['results']['2.0']['obj']}")
    log(f"    At 3x costs: OBJ={cost['results']['3.0']['obj']}")
    log(f"    Time: {time.time() - t_b:.1f}s")

    _partial_results["cost_sensitivity"] = cost
    save_partial(_partial_results)

    # ── Test C: Monthly Stability ─────────────────────────────────────
    log("\n" + "=" * 70)
    log("Test C: Monthly Sharpe Stability")
    log("=" * 70)

    stability = monthly_stability(sig_returns_cache, z_cache)
    for yr, stats in stability.items():
        if "note" in stats:
            log(f"  {yr}: {stats['note']}")
        else:
            log(f"  {yr}: mean={stats['mean']:.2f}, std={stats['std']:.2f}, "
                f"min={stats['min']:.2f}, max={stats['max']:.2f}, "
                f"positive={stats['pct_positive']:.0f}%, above_1={stats['pct_above_1']:.0f}%")

    _partial_results["monthly_stability"] = stability

    # ── FINAL VERDICTS ────────────────────────────────────────────────
    elapsed = time.time() - t0
    log("\n" + "=" * 70)
    log("PHASE 119 SUMMARY")
    log("=" * 70)

    robust_perturb = perturb["pct_above_1_0"] > 90
    cost_safe_2x = cost["results"]["2.0"]["obj"] > 0
    cost_safe_3x = cost["results"]["3.0"]["obj"] > 0

    verdicts = {
        "perturbation_robust": robust_perturb,
        "perturbation_pct_above_1": perturb["pct_above_1_0"],
        "cost_profitable_at_2x": cost_safe_2x,
        "cost_profitable_at_3x": cost_safe_3x,
        "breakeven_cost_multiplier": cost["breakeven_obj"],
        "monthly_worst": {yr: stats.get("worst_month", "N/A") for yr, stats in stability.items()},
        "deployment_confidence": "HIGH" if (robust_perturb and cost_safe_2x) else "MODERATE",
    }

    log(f"  Perturbation: {'ROBUST' if robust_perturb else 'FRAGILE'} ({perturb['pct_above_1_0']:.0f}% > 1.0)")
    log(f"  Cost safety: profitable at 2x? {'YES' if cost_safe_2x else 'NO'}, at 3x? {'YES' if cost_safe_3x else 'NO'}")
    log(f"  Breakeven: {cost['breakeven_obj']}x costs")
    log(f"  Deployment confidence: {verdicts['deployment_confidence']}")
    log(f"\n  Total time: {elapsed:.1f}s")

    _partial_results["verdicts"] = verdicts
    _partial_results["elapsed_seconds"] = round(elapsed, 1)
    save_partial(_partial_results)

    # Also merge with Phase 118 results for complete picture
    p118_path = os.path.join(PROJ, "artifacts", "phase118", "phase118_robustness_report.json")
    if os.path.exists(p118_path):
        with open(p118_path) as f:
            p118 = json.load(f)
        combined = {
            "phase": "118+119",
            "description": "Complete robustness analysis (Bootstrap CI + Permutation + Perturbation + Cost + Stability)",
            "timestamp": datetime.now().isoformat(),
            "champion_baseline": p118.get("champion_baseline", {}),
            "champion_obj": p118.get("champion_obj", champ_obj),
            "bootstrap_ci": p118.get("bootstrap_ci", {}),
            "permutation_test": p118.get("permutation_test", {}),
            "parameter_perturbation": perturb,
            "cost_sensitivity": cost,
            "monthly_stability": stability,
            "verdicts": {
                **p118.get("verdicts", {}),
                **verdicts,
            },
        }
        combined_path = os.path.join(OUT_DIR, "combined_robustness_report.json")
        def _default(obj):
            if isinstance(obj, (np.integer,)):
                return int(obj)
            if isinstance(obj, (np.floating,)):
                return float(obj)
            if isinstance(obj, (np.bool_,)):
                return bool(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
        with open(combined_path, "w") as f:
            json.dump(combined, f, indent=2, default=_default)
        log(f"\n  Combined report → {combined_path}")

    log(f"\nPhase 119 COMPLETE in {elapsed:.1f}s")


if __name__ == "__main__":
    main()
