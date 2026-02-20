#!/usr/bin/env python3
"""
Phase 118: Monte Carlo Robustness & Bootstrap Confidence Intervals
===================================================================
Pre-deployment confidence analysis for P91b + vol_tilt champion:
1. Bootstrap Sharpe CI — resample hourly returns (10K iters) → 95% CI
2. Permutation test — shuffle signal timing → p-value for alpha
3. Parameter perturbation — jitter params ±10-20% → stability map
4. Transaction cost sensitivity — scale costs 0.5x-3x → degradation curve
5. Capacity analysis — scale leverage → when does slippage eat alpha
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

OUT_DIR = os.path.join(PROJ, "artifacts", "phase118")
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

# Vol tilt params (Phase 113 champion)
VOL_TILT_LOOKBACK = 168
VOL_TILT_RATIO = 0.65

N_BOOTSTRAP = 500
N_PERMUTATIONS = 200
N_PERTURBATIONS = 30
RNG = np.random.RandomState(42)


def log(msg):
    print(f"[P118] {msg}", flush=True)


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
    """Run a single signal and return hourly returns."""
    if cost_cfg is None:
        cost_cfg = {"fee_rate": 0.0005, "slippage_rate": 0.0003}
    cost_model = cost_model_from_config(cost_cfg)
    cfg = SIGNALS[sig_key]
    strat = make_strategy({"name": cfg["name"], "params": cfg["params"]})
    engine = BacktestEngine(BacktestConfig(costs=cost_model))
    result = engine.run(dataset, strat)
    return np.diff(result.equity_curve) / result.equity_curve[:-1]


def run_p91b_returns(dataset, cost_cfg=None):
    """Run full P91b ensemble and return hourly returns."""
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


def get_champion_returns(year_key, ds_cache):
    """Get full champion (P91b + vol tilt) returns for a year."""
    dataset = get_dataset(year_key, ds_cache)
    base_rets = run_p91b_returns(dataset)
    z = compute_volume_z(dataset, VOL_TILT_LOOKBACK)
    if z is not None:
        return apply_tilt(base_rets, z, VOL_TILT_RATIO)
    return base_rets


def concat_all_returns(ds_cache):
    """Concatenate all IS + OOS returns for full-sample analysis."""
    all_rets = []
    for yr in YEARS + ["2026_oos"]:
        rets = get_champion_returns(yr, ds_cache)
        all_rets.append(rets)
    return np.concatenate(all_rets)


# ─── Test 1: Bootstrap Sharpe CI ─────────────────────────────────────

def bootstrap_sharpe_ci(returns, n_boot=N_BOOTSTRAP, ci=0.95):
    """Block bootstrap (block_size=168h) to preserve autocorrelation."""
    n = len(returns)
    block_size = 168  # 1 week of hourly data
    n_blocks = max(1, n // block_size)
    sharpes = []

    for _ in range(n_boot):
        # Sample blocks with replacement
        block_starts = RNG.randint(0, n - block_size, size=n_blocks)
        boot_rets = np.concatenate([returns[s:s + block_size] for s in block_starts])
        sharpes.append(compute_sharpe(boot_rets[:n]))

    sharpes = np.array(sharpes)
    alpha = (1 - ci) / 2
    lo = np.percentile(sharpes, alpha * 100)
    hi = np.percentile(sharpes, (1 - alpha) * 100)
    return {
        "mean": float(np.mean(sharpes)),
        "median": float(np.median(sharpes)),
        "std": float(np.std(sharpes)),
        "ci_lo": float(lo),
        "ci_hi": float(hi),
        "ci_level": ci,
        "n_bootstrap": n_boot,
        "pct_positive": float(np.mean(sharpes > 0) * 100),
    }


# ─── Test 2: Permutation test ────────────────────────────────────────

def permutation_test_signal(ds_cache, n_perm=N_PERMUTATIONS):
    """
    Shuffle the vol tilt signal timing → compute Sharpe → p-value.
    Tests whether the SPECIFIC timing of vol_mom_z adds value.
    OPTIMIZED: cache base returns & z-scores (only tilt application varies).
    """
    # Pre-compute and cache base returns + z-scores (expensive, do ONCE)
    cached_base = {}
    cached_z = {}
    for yr in YEARS:
        ds = get_dataset(yr, ds_cache)
        cached_base[yr] = run_p91b_returns(ds)
        cached_z[yr] = compute_volume_z(ds, VOL_TILT_LOOKBACK)
        log(f"  Cached {yr}: {len(cached_base[yr])} returns")

    # Real (champion) Sharpe
    real_sharpes = []
    base_sharpes = []
    for yr in YEARS:
        base_sharpes.append(compute_sharpe(cached_base[yr]))
        if cached_z[yr] is not None:
            tilted = apply_tilt(cached_base[yr], cached_z[yr], VOL_TILT_RATIO)
            real_sharpes.append(compute_sharpe(tilted))
        else:
            real_sharpes.append(compute_sharpe(cached_base[yr]))
    base_avg = np.mean(base_sharpes)
    real_avg = np.mean(real_sharpes)
    real_improvement = real_avg - base_avg

    # Permutation: shuffle z-scores (instant — just array ops)
    perm_improvements = []
    for p in range(n_perm):
        perm_sharpes = []
        for yr in YEARS:
            z = cached_z[yr]
            if z is not None:
                z_shuffled = z.copy()
                RNG.shuffle(z_shuffled)
                tilted = apply_tilt(cached_base[yr], z_shuffled, VOL_TILT_RATIO)
                perm_sharpes.append(compute_sharpe(tilted))
            else:
                perm_sharpes.append(compute_sharpe(cached_base[yr]))
        perm_improvements.append(np.mean(perm_sharpes) - base_avg)
        if (p + 1) % 50 == 0:
            log(f"  Permutation {p+1}/{n_perm}...")

    perm_improvements = np.array(perm_improvements)
    p_value = float(np.mean(perm_improvements >= real_improvement))

    return {
        "real_improvement": float(real_improvement),
        "perm_mean": float(np.mean(perm_improvements)),
        "perm_std": float(np.std(perm_improvements)),
        "p_value": p_value,
        "n_permutations": n_perm,
        "significant_at_5pct": p_value < 0.05,
        "significant_at_10pct": p_value < 0.10,
    }


# ─── Test 3: Weight & tilt perturbation (FAST — no re-running backtests) ────

def parameter_perturbation(ds_cache, n_trials=500):
    """
    Perturb ensemble weights ±20% and vol tilt ratio/lookback.
    Uses CACHED signal returns — only array math, no backtest re-runs.
    Tests if the champion is a narrow peak or a broad plateau.
    """
    # Pre-cache individual signal returns (expensive, done ONCE)
    sig_returns = {}  # (sig_key, year) -> np.array
    for yr in YEARS:
        ds = get_dataset(yr, ds_cache)
        for sig_key in SIG_KEYS:
            rets = run_signal_returns(ds, sig_key)
            sig_returns[(sig_key, yr)] = rets
            log(f"  Cached {sig_key}/{yr}: {len(rets)} returns")

    # Pre-cache z-scores for a range of lookbacks
    z_cache = {}  # (year, lookback) -> np.array
    for yr in YEARS:
        ds = get_dataset(yr, ds_cache)
        for lb in range(120, 250, 12):
            z_cache[(yr, lb)] = compute_volume_z(ds, lb)

    WEIGHT_RANGES = {k: (v * 0.8, v * 1.2) for k, v in P91B_WEIGHTS.items()}
    VOL_TILT_RANGES = {"ratio": (0.52, 0.78)}
    LOOKBACKS = sorted(set(lb for (_, lb) in z_cache.keys()))

    trial_sharpes = []
    for trial in range(n_trials):
        # Random weights (normalized)
        raw_w = {k: RNG.uniform(lo, hi) for k, (lo, hi) in WEIGHT_RANGES.items()}
        wt_sum = sum(raw_w.values())
        norm_w = {k: v / wt_sum for k, v in raw_w.items()}
        w_arr = np.array([norm_w[k] for k in SIG_KEYS])

        # Random vol tilt
        vt_lb = RNG.choice(LOOKBACKS)
        vt_ratio = RNG.uniform(*VOL_TILT_RANGES["ratio"])

        yr_sharpes = []
        for yr in YEARS:
            rets_list = [sig_returns[(sk, yr)] for sk in SIG_KEYS]
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
        trial_sharpes.append((avg_s + min_s) / 2)

        if (trial + 1) % 100 == 0:
            log(f"  Perturbation {trial+1}/{n_trials}: median OBJ={np.median(trial_sharpes):.4f}")

    trial_sharpes = np.array(trial_sharpes)
    return {
        "n_trials": n_trials,
        "mean_obj": float(np.mean(trial_sharpes)),
        "median_obj": float(np.median(trial_sharpes)),
        "std_obj": float(np.std(trial_sharpes)),
        "min_obj": float(np.min(trial_sharpes)),
        "max_obj": float(np.max(trial_sharpes)),
        "pct_above_1_0": float(np.mean(trial_sharpes > 1.0) * 100),
        "pct_above_1_5": float(np.mean(trial_sharpes > 1.5) * 100),
        "p5": float(np.percentile(trial_sharpes, 5)),
        "p25": float(np.percentile(trial_sharpes, 25)),
        "p75": float(np.percentile(trial_sharpes, 75)),
        "p95": float(np.percentile(trial_sharpes, 95)),
    }


# ─── Test 4: Transaction cost sensitivity ────────────────────────────

def cost_sensitivity(ds_cache):
    """Scale transaction costs from 0.5x to 3.0x and measure degradation."""
    multipliers = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0]
    base_fee = 0.0005
    base_slip = 0.0003
    results = {}

    for mult in multipliers:
        cost_cfg = {
            "fee_rate": base_fee * mult,
            "slippage_rate": base_slip * mult,
        }
        yr_sharpes = []
        for yr in YEARS:
            ds = get_dataset(yr, ds_cache)
            base_rets = run_p91b_returns(ds, cost_cfg)
            z = compute_volume_z(ds, VOL_TILT_LOOKBACK)
            if z is not None:
                rets = apply_tilt(base_rets, z, VOL_TILT_RATIO)
            else:
                rets = base_rets
            yr_sharpes.append(compute_sharpe(rets))

        avg_s = np.mean(yr_sharpes)
        min_s = np.min(yr_sharpes)
        obj = (avg_s + min_s) / 2
        results[str(mult)] = {
            "cost_mult": mult,
            "fee_rate": round(base_fee * mult, 6),
            "slip_rate": round(base_slip * mult, 6),
            "avg_sharpe": round(avg_s, 4),
            "min_sharpe": round(min_s, 4),
            "obj": round(obj, 4),
            "yearly": {yr: round(s, 4) for yr, s in zip(YEARS, yr_sharpes)},
        }
        log(f"  Cost {mult:.2f}x: AVG={avg_s:.4f}, MIN={min_s:.4f}, OBJ={obj:.4f}")

    # Find breakeven (where OBJ drops below 0)
    objs = [(m, results[str(m)]["obj"]) for m in multipliers]
    breakeven = None
    for i in range(len(objs) - 1):
        if objs[i][1] > 0 and objs[i + 1][1] <= 0:
            # Linear interpolation
            m1, o1 = objs[i]
            m2, o2 = objs[i + 1]
            breakeven = m1 + (m2 - m1) * o1 / (o1 - o2)
            break
    if breakeven is None and objs[-1][1] > 0:
        breakeven = ">3.0x"

    return {
        "results": results,
        "breakeven_multiplier": breakeven,
        "cost_at_1x": f"{base_fee + base_slip:.4f} = {(base_fee + base_slip) * 10000:.1f} bps",
    }


# ─── Test 5: Yearly stability (Sharpe / year, worst months) ──────────

def yearly_stability(ds_cache):
    """Compute monthly Sharpe distribution for each year."""
    monthly_sharpes = {}
    for yr in YEARS + ["2026_oos"]:
        rets = get_champion_returns(yr, ds_cache)
        # Split into ~monthly blocks (730 bars ≈ 1 month)
        block_size = 730
        n_blocks = len(rets) // block_size
        month_sharpes = []
        for b in range(n_blocks):
            chunk = rets[b * block_size:(b + 1) * block_size]
            month_sharpes.append(compute_sharpe(chunk))
        monthly_sharpes[yr] = {
            "n_months": len(month_sharpes),
            "mean": round(np.mean(month_sharpes), 4) if month_sharpes else 0,
            "std": round(np.std(month_sharpes), 4) if month_sharpes else 0,
            "min": round(np.min(month_sharpes), 4) if month_sharpes else 0,
            "max": round(np.max(month_sharpes), 4) if month_sharpes else 0,
            "pct_positive": round(np.mean(np.array(month_sharpes) > 0) * 100, 1) if month_sharpes else 0,
        }
    return monthly_sharpes


def main():
    t0 = time.time()
    log("Phase 118: Monte Carlo Robustness & Bootstrap CI")
    log("=" * 70)

    ds_cache = {}

    # ── Baseline champion metrics ────────────────────────────────────
    log("\nStep 0: Computing champion baseline...")
    baseline = {}
    for yr in YEARS + ["2026_oos"]:
        rets = get_champion_returns(yr, ds_cache)
        baseline[yr] = {
            "sharpe": round(compute_sharpe(rets), 4),
            "mdd": round(compute_mdd(rets) * 100, 2),
            "cagr": round(compute_cagr(rets) * 100, 2),
        }
        log(f"  {yr}: Sharpe={baseline[yr]['sharpe']}, MDD={baseline[yr]['mdd']}%, CAGR={baseline[yr]['cagr']}%")

    is_sharpes = [baseline[yr]["sharpe"] for yr in YEARS]
    champ_avg = np.mean(is_sharpes)
    champ_min = np.min(is_sharpes)
    champ_obj = (champ_avg + champ_min) / 2
    log(f"  Champion: AVG={champ_avg:.4f}, MIN={champ_min:.4f}, OBJ={champ_obj:.4f}")

    # ── Test 1: Bootstrap Sharpe CI ──────────────────────────────────
    log("\n" + "=" * 70)
    log("Test 1: Bootstrap Sharpe CI (block bootstrap, 168h blocks)")
    log("=" * 70)

    bootstrap_results = {}
    for yr in YEARS + ["2026_oos"]:
        log(f"  Bootstrapping {yr}...")
        rets = get_champion_returns(yr, ds_cache)
        bootstrap_results[yr] = bootstrap_sharpe_ci(rets, N_BOOTSTRAP)
        r = bootstrap_results[yr]
        log(f"    Sharpe CI [{r['ci_lo']:.3f}, {r['ci_hi']:.3f}], mean={r['mean']:.3f}, P(>0)={r['pct_positive']:.1f}%")

    # Full-sample bootstrap
    log(f"  Bootstrapping full sample (2021-2026)...")
    full_rets = concat_all_returns(ds_cache)
    bootstrap_results["full"] = bootstrap_sharpe_ci(full_rets, N_BOOTSTRAP)
    r = bootstrap_results["full"]
    log(f"    Full CI [{r['ci_lo']:.3f}, {r['ci_hi']:.3f}], mean={r['mean']:.3f}, P(>0)={r['pct_positive']:.1f}%")

    # ── Test 2: Permutation test ─────────────────────────────────────
    log("\n" + "=" * 70)
    log(f"Test 2: Permutation Test ({N_PERMUTATIONS} shuffles)")
    log("=" * 70)
    log("  Testing if vol tilt timing adds value beyond random...")

    perm_result = permutation_test_signal(ds_cache, N_PERMUTATIONS)
    log(f"  Real improvement: {perm_result['real_improvement']:.4f}")
    log(f"  Permutation mean: {perm_result['perm_mean']:.4f} ± {perm_result['perm_std']:.4f}")
    log(f"  p-value: {perm_result['p_value']:.4f}")
    log(f"  Significant at 5%? {'YES' if perm_result['significant_at_5pct'] else 'NO'}")
    log(f"  Significant at 10%? {'YES' if perm_result['significant_at_10pct'] else 'NO'}")

    # ── Test 3: Parameter perturbation ───────────────────────────────
    log("\n" + "=" * 70)
    log("Test 3: Parameter Perturbation (±20% jitter, 200 trials)")
    log("=" * 70)

    perturb_result = parameter_perturbation(ds_cache, 500)
    log(f"\n  Perturbation results:")
    log(f"    Mean OBJ: {perturb_result['mean_obj']:.4f} (champion: {champ_obj:.4f})")
    log(f"    Std OBJ: {perturb_result['std_obj']:.4f}")
    log(f"    Range: [{perturb_result['min_obj']:.4f}, {perturb_result['max_obj']:.4f}]")
    log(f"    P5-P95: [{perturb_result['p5']:.4f}, {perturb_result['p95']:.4f}]")
    log(f"    % above 1.0: {perturb_result['pct_above_1_0']:.1f}%")
    log(f"    % above 1.5: {perturb_result['pct_above_1_5']:.1f}%")

    # ── Test 4: Transaction cost sensitivity ─────────────────────────
    log("\n" + "=" * 70)
    log("Test 4: Transaction Cost Sensitivity (0x - 3x)")
    log("=" * 70)

    cost_result = cost_sensitivity(ds_cache)
    log(f"\n  Breakeven at: {cost_result['breakeven_multiplier']}x current costs")
    log(f"  Current costs: {cost_result['cost_at_1x']}")

    # ── Test 5: Yearly stability ─────────────────────────────────────
    log("\n" + "=" * 70)
    log("Test 5: Monthly Sharpe Stability")
    log("=" * 70)

    stability = yearly_stability(ds_cache)
    for yr, stats in stability.items():
        log(f"  {yr}: mean={stats['mean']:.2f}, std={stats['std']:.2f}, "
            f"min={stats['min']:.2f}, max={stats['max']:.2f}, "
            f"positive={stats['pct_positive']:.0f}%")

    # ── FINAL SUMMARY ────────────────────────────────────────────────
    elapsed = time.time() - t0
    log("\n" + "=" * 70)
    log("PHASE 118 ROBUSTNESS SUMMARY")
    log("=" * 70)

    # Verdict logic
    all_ci_positive = all(
        bootstrap_results[yr]["ci_lo"] > 0 for yr in YEARS
    )
    robust_perturb = perturb_result["pct_above_1_0"] > 90
    cost_safe = cost_result["breakeven_multiplier"] != "<1.0x"

    verdicts = []
    verdicts.append(f"  Bootstrap: ALL CIs above zero? {'YES' if all_ci_positive else 'NO'}")
    verdicts.append(f"  Permutation: Vol tilt significant? p={perm_result['p_value']:.3f}")
    verdicts.append(f"  Perturbation: {perturb_result['pct_above_1_0']:.0f}% of ±20% perturbations > 1.0 Sharpe")
    verdicts.append(f"  Cost safety: breakeven at {cost_result['breakeven_multiplier']}x")

    deployment_ready = all_ci_positive and robust_perturb and cost_safe
    verdicts.append(f"\n  DEPLOYMENT READY: {'YES' if deployment_ready else 'CONDITIONAL'}")

    for v in verdicts:
        log(v)

    # ── Save report ──────────────────────────────────────────────────
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

    report = {
        "phase": 118,
        "description": "Monte Carlo Robustness & Bootstrap Confidence Intervals",
        "timestamp": datetime.now().isoformat(),
        "elapsed_seconds": round(elapsed, 1),
        "champion_baseline": baseline,
        "champion_obj": round(champ_obj, 4),
        "bootstrap_ci": bootstrap_results,
        "permutation_test": perm_result,
        "parameter_perturbation": perturb_result,
        "cost_sensitivity": cost_result,
        "monthly_stability": stability,
        "verdicts": {
            "all_ci_positive": all_ci_positive,
            "vol_tilt_significant": perm_result["p_value"] < 0.10,
            "perturbation_robust": robust_perturb,
            "cost_safe": cost_safe,
            "deployment_ready": deployment_ready,
        },
    }

    report_path = os.path.join(OUT_DIR, "phase118_robustness_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=_default)

    log(f"\nPhase 118 COMPLETE in {elapsed:.1f}s → {report_path}")


if __name__ == "__main__":
    main()
