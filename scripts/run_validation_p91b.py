#!/usr/bin/env python3
"""
P91b Champion — Comprehensive Validation Suite
================================================
Tests: Sharpe significance, Deflated Sharpe, Block Bootstrap CI,
       Permutation test, Parameter sensitivity, LOYO CV, Full suite,
       2026 true OOS, Survivorship bias, Multiple testing correction.

Output: artifacts/validation/p91b_validation_report.json
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

OUT_DIR = os.path.join(PROJ, "artifacts", "validation")
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
    "2026": ("2026-01-01", "2026-02-20"),
}

SIGNAL_CONFIGS = {
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

CHAMPION_WEIGHTS = {"v1": 0.2747, "i460bw168": 0.1967, "i415bw216": 0.3247, "f144": 0.2039}
N_PARAMS_SEARCHED = 1_000_000


def log(msg):
    print(f"[VAL] {msg}", flush=True)


def run_signal(signal_key, year):
    sig = SIGNAL_CONFIGS[signal_key]
    start, end = YEAR_RANGES[year]
    data_cfg = {
        "provider": "binance_rest_v1", "symbols": SYMBOLS,
        "start": start, "end": end, "bar_interval": "1h",
        "cache_dir": ".cache/binance_rest",
    }
    costs_cfg = {"fee_rate": 0.0005, "slippage_rate": 0.0003, "cost_multiplier": 1.0}
    exec_cfg = {"style": "taker", "slippage_bps": 3.0}
    provider = make_provider(data_cfg, seed=42)
    dataset = provider.load()
    strategy = make_strategy({"name": sig["name"], "params": copy.deepcopy(sig["params"])})
    cost_model = cost_model_from_config(costs_cfg, execution_cfg=exec_cfg)
    engine = BacktestEngine(BacktestConfig(costs=cost_model))
    result = engine.run(dataset=dataset, strategy=strategy, seed=42)
    return result.returns


def compute_sharpe(returns):
    if not returns or len(returns) < 100:
        return 0.0
    arr = np.array(returns, dtype=np.float64)
    std = float(np.std(arr))
    return float(np.mean(arr) / std * np.sqrt(8760)) if std > 0 else 0.0


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


def norm_cdf(x):
    t = 1.0 / (1.0 + 0.2316419 * abs(x))
    poly = t * (0.319381530 + t * (-0.356563782 + t * (1.781477937 + t * (-1.821255978 + t * 1.330274429))))
    cdf = 1.0 - (1.0 / math.sqrt(2 * math.pi)) * math.exp(-x * x / 2) * poly
    return cdf if x >= 0 else 1.0 - cdf


def sharpe_significance(returns):
    n = len(returns)
    if n < 30:
        return {"t_stat": 0, "p_value": 1.0, "significant_95": False, "significant_99": False, "sharpe": 0}
    arr = np.array(returns, dtype=np.float64)
    mu, sd = float(arr.mean()), float(arr.std())
    if sd <= 0:
        return {"t_stat": 0, "p_value": 1.0, "significant_95": False, "significant_99": False, "sharpe": 0}
    sharpe = mu / sd * np.sqrt(8760)
    t_stat = float(sharpe * np.sqrt(n) / np.sqrt(8760))
    p_value = float(2.0 * (1.0 - norm_cdf(abs(t_stat))))
    return {"sharpe": round(float(sharpe), 4), "t_stat": round(t_stat, 4),
            "p_value": round(max(0, min(1, p_value)), 6), "n": n,
            "significant_95": p_value < 0.05, "significant_99": p_value < 0.01}


def deflated_sharpe(returns, n_params):
    n = len(returns)
    if n < 30:
        return {"dsr": 0.0, "sr_benchmark": 0.0, "likely_overfit": True, "mtrl_years": float("inf")}
    arr = np.array(returns, dtype=np.float64)
    mu, sd = float(arr.mean()), float(arr.std())
    if sd <= 0:
        return {"dsr": 0.0, "sr_benchmark": 0.0, "likely_overfit": True, "mtrl_years": float("inf")}
    sr = mu / sd
    skew = float(((arr - mu) ** 3).mean() / sd ** 3)
    kurt = float(((arr - mu) ** 4).mean() / sd ** 4)
    v = max(1e-12, 1.0 - skew * sr + ((kurt - 1) / 4) * sr * sr)
    sigma_sr = math.sqrt(v)
    sr_std = sigma_sr / math.sqrt(max(1, n - 1))
    N = max(2, n_params)
    a = math.sqrt(2.0 * math.log(N))
    mu_max = a - (math.log(math.log(N)) + math.log(4 * math.pi)) / (2 * a)
    sr_bench = mu_max * sr_std
    z = (sr - sr_bench) / sr_std if sr_std > 0 else 0
    dsr = norm_cdf(z)
    mtrl = (1.0 + v * (1.96 / sr) ** 2) if sr > 0 else float("inf")
    return {"dsr": round(dsr, 4), "sr_observed": round(sr * math.sqrt(8760), 4),
            "sr_benchmark": round(sr_bench * math.sqrt(8760), 4),
            "mtrl_years": round(mtrl / 8760, 2), "skewness": round(skew, 4),
            "kurtosis": round(kurt, 4), "likely_overfit": dsr < 0.95}


def block_bootstrap_ci(returns, n_boot=1000, block_size=168, seed=42):
    rng = random.Random(seed)
    n = len(returns)
    if n < block_size * 2:
        return {"ci_lower": 0.0, "ci_upper": 0.0, "median": 0.0, "pct_positive": 0.0}
    sharpes = []
    for _ in range(n_boot):
        resampled = []
        while len(resampled) < n:
            start = rng.randint(0, n - block_size)
            resampled.extend(returns[start:start + block_size])
        sharpes.append(compute_sharpe(resampled[:n]))
    sharpes.sort()
    return {"ci_lower": round(sharpes[int(n_boot * 0.025)], 4),
            "ci_upper": round(sharpes[int(n_boot * 0.975)], 4),
            "median": round(sharpes[n_boot // 2], 4),
            "pct_positive": round(sum(1 for s in sharpes if s > 0) / len(sharpes), 4)}


def permutation_test(sig_rets, weights, n_perms=1000, seed=42):
    rng = random.Random(seed)
    keys = sorted(weights.keys())
    n = min(len(sig_rets[k]) for k in keys)
    observed = compute_sharpe(blend_returns(sig_rets, weights))
    perm_sharpes = []
    for _ in range(n_perms):
        perm = {}
        for k in keys:
            arr = list(sig_rets[k][:n])
            rng.shuffle(arr)
            perm[k] = arr
        perm_sharpes.append(compute_sharpe(blend_returns(perm, weights)))
    perm_sharpes.sort()
    n_gte = sum(1 for s in perm_sharpes if s >= observed)
    p = n_gte / n_perms
    return {"observed": round(observed, 4), "perm_mean": round(statistics.mean(perm_sharpes), 4),
            "perm_p95": round(perm_sharpes[int(n_perms * 0.95)], 4),
            "perm_max": round(max(perm_sharpes), 4), "p_value": round(p, 4),
            "significant_95": p < 0.05, "significant_99": p < 0.01}


def parameter_sensitivity(sig_rets, base_w, delta=0.05):
    keys = sorted(base_w.keys())
    base_s = compute_sharpe(blend_returns(sig_rets, base_w))
    results = {}
    for k in keys:
        for d_name, d_val in [("up", delta), ("down", -delta)]:
            w = dict(base_w)
            w[k] = max(0.0, w[k] + d_val)
            tot = sum(w.values())
            if tot > 0:
                w = {kk: vv / tot for kk, vv in w.items()}
            s = compute_sharpe(blend_returns(sig_rets, w))
            elast = (s - base_s) / base_s * 100 if base_s != 0 else 0
            results[f"{k}_{d_name}"] = {"sharpe": round(s, 4), "change_pct": round(elast, 2)}
    max_e = max(abs(v["change_pct"]) for v in results.values())
    return {"base_sharpe": round(base_s, 4), "details": results,
            "max_elasticity_pct": round(max_e, 2), "robust": max_e < 10}


def leave_one_year_out(yr_rets, weights):
    years = [y for y in sorted(yr_rets.keys()) if y != "2026"]
    results = []
    for test_y in years:
        train_r = {k: [] for k in weights}
        for y in years:
            if y != test_y:
                for k in weights:
                    train_r[k].extend(yr_rets[y].get(k, []))
        test_r = {k: yr_rets[test_y].get(k, []) for k in weights}
        if all(len(test_r[k]) > 0 for k in weights) and all(len(train_r[k]) > 0 for k in weights):
            ts = compute_sharpe(blend_returns(test_r, weights))
            trs = compute_sharpe(blend_returns(train_r, weights))
            results.append({"held_out": test_y, "train": round(trs, 4), "test": round(ts, 4)})
    test_sharpes = [r["test"] for r in results]
    return {"results": results,
            "mean_test": round(statistics.mean(test_sharpes), 4) if test_sharpes else 0,
            "min_test": round(min(test_sharpes), 4) if test_sharpes else 0,
            "stable": min(test_sharpes) > 0.5 if test_sharpes else False}


def main():
    t0 = time.time()
    report = {"champion": "P91b Balanced", "weights": CHAMPION_WEIGHTS}

    log("=" * 60)
    log("STEP 1: Running 4 signals x 6 years = 24 backtests")
    log("=" * 60)

    yr_rets = {}
    all_rets = {}
    yr_sharpes = {}

    for sig in sorted(SIGNAL_CONFIGS):
        yr_sharpes[sig] = {}
        all_rets[sig] = []
        for year in sorted(YEAR_RANGES):
            if year not in yr_rets:
                yr_rets[year] = {}
            log(f"  {sig}/{year}...")
            try:
                rets = run_signal(sig, year)
            except Exception as e:
                log(f"    ERROR: {e}")
                rets = []
            yr_rets[year][sig] = rets
            s = compute_sharpe(rets)
            yr_sharpes[sig][year] = round(s, 3)
            if year != "2026":
                all_rets[sig].extend(rets)
            log(f"    Sharpe={s:.3f}, n={len(rets)}")

    report["signal_profiles"] = yr_sharpes
    log("\nSignal Year-by-Year:")
    for sig in sorted(yr_sharpes):
        vals = [yr_sharpes[sig].get(y, 0) for y in ["2021", "2022", "2023", "2024", "2025"]]
        log(f"  {sig}: {vals} AVG={statistics.mean(vals):.3f}")

    log("\n" + "=" * 60)
    log("STEP 2: Blend with champion weights")
    log("=" * 60)

    yr_blended = {}
    yr_blend_sharpes = {}
    for year in sorted(YEAR_RANGES):
        sr = yr_rets.get(year, {})
        if all(len(sr.get(k, [])) > 0 for k in CHAMPION_WEIGHTS):
            bl = blend_returns(sr, CHAMPION_WEIGHTS)
            yr_blended[year] = bl
            yr_blend_sharpes[year] = round(compute_sharpe(bl), 4)
            log(f"  {year}: Sharpe={yr_blend_sharpes[year]}, n={len(bl)}")
        else:
            yr_blended[year] = []
            yr_blend_sharpes[year] = 0.0
            missing = [k for k in CHAMPION_WEIGHTS if len(sr.get(k, [])) == 0]
            log(f"  {year}: SKIPPED (missing: {missing})")

    all_is = []
    for y in ["2021", "2022", "2023", "2024", "2025"]:
        all_is.extend(yr_blended.get(y, []))

    is_years = [y for y in ["2021", "2022", "2023", "2024", "2025"] if yr_blend_sharpes.get(y, 0) != 0]
    is_sharpes = [yr_blend_sharpes[y] for y in is_years]
    report["year_sharpes"] = yr_blend_sharpes
    report["avg_sharpe"] = round(statistics.mean(is_sharpes), 4) if is_sharpes else 0
    report["min_sharpe"] = round(min(is_sharpes), 4) if is_sharpes else 0
    log(f"\n  AVG={report['avg_sharpe']}, MIN={report['min_sharpe']}")
    if yr_blend_sharpes.get("2026", 0) != 0:
        log(f"  2026 OOS={yr_blend_sharpes['2026']}")

    log("\n" + "=" * 60)
    log("STEP 3: Sharpe Significance (t-test)")
    log("=" * 60)
    sig_tests = {}
    for y in is_years:
        sig_tests[y] = sharpe_significance(yr_blended[y])
        log(f"  {y}: t={sig_tests[y]['t_stat']}, p={sig_tests[y]['p_value']}, sig99={sig_tests[y]['significant_99']}")
    sig_tests["all"] = sharpe_significance(all_is)
    log(f"  ALL: t={sig_tests['all']['t_stat']}, p={sig_tests['all']['p_value']}")
    report["sharpe_significance"] = sig_tests

    log("\n" + "=" * 60)
    log(f"STEP 4: Deflated Sharpe (n_params={N_PARAMS_SEARCHED:,})")
    log("=" * 60)
    dsr = {}
    for y in is_years:
        dsr[y] = deflated_sharpe(yr_blended[y], N_PARAMS_SEARCHED)
        log(f"  {y}: DSR={dsr[y]['dsr']}, bench={dsr[y]['sr_benchmark']}, overfit={dsr[y]['likely_overfit']}")
    dsr["all"] = deflated_sharpe(all_is, N_PARAMS_SEARCHED)
    log(f"  ALL: DSR={dsr['all']['dsr']}, MTRL={dsr['all']['mtrl_years']}yr")
    report["deflated_sharpe"] = dsr

    log("\n" + "=" * 60)
    log("STEP 5: Block Bootstrap CI (1000 iter, block=168h)")
    log("=" * 60)
    boot = {}
    for y in is_years:
        if len(yr_blended[y]) > 500:
            boot[y] = block_bootstrap_ci(yr_blended[y])
            log(f"  {y}: [{boot[y]['ci_lower']}, {boot[y]['ci_upper']}], pct+={boot[y]['pct_positive']}")
    if len(all_is) > 500:
        boot["all"] = block_bootstrap_ci(all_is)
        log(f"  ALL: [{boot['all']['ci_lower']}, {boot['all']['ci_upper']}]")
    report["bootstrap_ci"] = boot

    log("\n" + "=" * 60)
    log("STEP 6: Permutation Test (1000 shuffles)")
    log("=" * 60)
    perm = {}
    for y in is_years:
        sr = yr_rets.get(y, {})
        if all(len(sr.get(k, [])) > 100 for k in CHAMPION_WEIGHTS):
            perm[y] = permutation_test(sr, CHAMPION_WEIGHTS)
            log(f"  {y}: obs={perm[y]['observed']}, p={perm[y]['p_value']}, sig99={perm[y]['significant_99']}")
    if all(len(all_rets.get(k, [])) > 100 for k in CHAMPION_WEIGHTS):
        perm["all"] = permutation_test(all_rets, CHAMPION_WEIGHTS)
        log(f"  ALL: obs={perm['all']['observed']}, p={perm['all']['p_value']}")
    report["permutation_test"] = perm

    log("\n" + "=" * 60)
    log("STEP 7: Parameter Sensitivity (+-5%)")
    log("=" * 60)
    sens = {}
    for y in is_years:
        sr = yr_rets.get(y, {})
        if all(len(sr.get(k, [])) > 100 for k in CHAMPION_WEIGHTS):
            sens[y] = parameter_sensitivity(sr, CHAMPION_WEIGHTS)
            log(f"  {y}: max_elast={sens[y]['max_elasticity_pct']}%, robust={sens[y]['robust']}")
    if all(len(all_rets.get(k, [])) > 100 for k in CHAMPION_WEIGHTS):
        sens["all"] = parameter_sensitivity(all_rets, CHAMPION_WEIGHTS)
        log(f"  ALL: max_elast={sens['all']['max_elasticity_pct']}%")
    report["sensitivity"] = sens

    log("\n" + "=" * 60)
    log("STEP 8: Leave-One-Year-Out CV")
    log("=" * 60)
    loyo = leave_one_year_out(yr_rets, CHAMPION_WEIGHTS)
    for r in loyo["results"]:
        log(f"  Hold-out {r['held_out']}: train={r['train']}, test={r['test']}")
    log(f"  Mean={loyo['mean_test']}, Min={loyo['min_test']}, Stable={loyo['stable']}")
    report["loyo"] = loyo

    log("\n" + "=" * 60)
    log("STEP 9: Full Validation Suite")
    log("=" * 60)
    if len(all_is) > 500:
        try:
            from nexus_quant.validation.full_suite import run_full_validation_suite
            fvs = run_full_validation_suite(all_is, run_name="p91b_champion", n_monte_carlo=500)
            log(f"  Verdict: {fvs['overall_verdict']}, Fails={fvs['n_fails']}, Warns={fvs['n_warns']}")
            for tn, tr in fvs.get("tests", {}).items():
                log(f"    {tn}: {tr.get('verdict', 'N/A')}")
            report["full_suite"] = fvs
        except Exception as e:
            log(f"  Error: {e}")

    log("\n" + "=" * 60)
    log("STEP 10: 2026 True OOS")
    log("=" * 60)
    oos26 = yr_blend_sharpes.get("2026", 0)
    if oos26 != 0 and len(yr_blended.get("2026", [])) > 50:
        od = {"sharpe": oos26, "n": len(yr_blended["2026"])}
        od["significance"] = sharpe_significance(yr_blended["2026"])
        if len(yr_blended["2026"]) > 200:
            od["bootstrap_ci"] = block_bootstrap_ci(yr_blended["2026"], n_boot=500, block_size=24)
        log(f"  Sharpe={oos26}, p={od['significance']['p_value']}")
        if "bootstrap_ci" in od:
            log(f"  CI=[{od['bootstrap_ci']['ci_lower']}, {od['bootstrap_ci']['ci_upper']}]")
        report["oos_2026"] = od
    else:
        log("  Not available or failed")
        report["oos_2026"] = {"status": "NOT_AVAILABLE"}

    # ── FINAL ──
    elapsed = time.time() - t0
    log("\n" + "=" * 60)
    log("FINAL HONEST ASSESSMENT")
    log("=" * 60)

    dsr_all = dsr.get("all", {}).get("dsr", 0)
    perm_p = perm.get("all", {}).get("p_value", 1.0)
    boot_lo = boot.get("all", {}).get("ci_lower", 0)
    loyo_min = loyo.get("min_test", 0)

    checks = []
    issues = []

    all_sig = all(sig_tests.get(y, {}).get("significant_99", False) for y in is_years)
    checks.append(("All years sig99", all_sig))
    if not all_sig:
        issues.append("Not all years pass 99% significance")

    checks.append(("DSR >= 0.95", dsr_all >= 0.95))
    if dsr_all < 0.95:
        issues.append(f"Deflated Sharpe {dsr_all:.3f} < 0.95 ({N_PARAMS_SEARCHED:,} params)")

    checks.append(("Permutation p < 0.05", perm_p < 0.05))
    if perm_p >= 0.05:
        issues.append(f"Permutation p={perm_p:.3f}")

    checks.append(("Bootstrap CI lower > 0.5", boot_lo > 0.5))
    if boot_lo <= 0.5:
        issues.append(f"Bootstrap 95% CI lower = {boot_lo:.2f}")

    checks.append(("LOYO min > 0.5", loyo_min > 0.5))
    if loyo_min <= 0.5:
        issues.append(f"LOYO worst = {loyo_min:.2f}")

    if isinstance(oos26, (int, float)) and oos26 != 0:
        checks.append(("2026 OOS > 0.5", oos26 > 0.5))
        if oos26 <= 0.5:
            issues.append(f"2026 OOS = {oos26:.2f}")
    else:
        issues.append("2026 OOS not available")

    passed = sum(1 for _, v in checks if v)
    total = len(checks)
    verdict = "CREDIBLE" if passed >= total - 1 else ("MARGINAL" if passed >= total - 2 else "NOT_CREDIBLE")

    summary = {
        "backtest_avg": report["avg_sharpe"], "backtest_min": report["min_sharpe"],
        "oos_2026": oos26 if isinstance(oos26, (int, float)) else "N/A",
        "deflated_sharpe": dsr_all, "permutation_p": perm_p,
        "bootstrap_ci_lower": boot_lo, "loyo_min": loyo_min,
        "checks": f"{passed}/{total}", "verdict": verdict,
        "issues": issues, "elapsed_sec": round(elapsed, 1),
    }
    report["summary"] = summary

    log(f"\n  Backtest AVG:    {report['avg_sharpe']}")
    log(f"  Backtest MIN:    {report['min_sharpe']}")
    log(f"  2026 OOS:        {oos26}")
    log(f"  Deflated Sharpe: {dsr_all}")
    log(f"  Permutation p:   {perm_p}")
    log(f"  Bootstrap CI lo: {boot_lo}")
    log(f"  LOYO min:        {loyo_min}")
    log(f"\n  Checks: {passed}/{total}")
    for name, val in checks:
        log(f"    {'PASS' if val else 'FAIL'}: {name}")
    log(f"\n  VERDICT: {verdict}")
    for i in issues:
        log(f"    - {i}")
    log(f"\n  Elapsed: {elapsed:.0f}s")

    path = os.path.join(OUT_DIR, "p91b_validation_report.json")
    with open(path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    log(f"  Saved: {path}")


if __name__ == "__main__":
    main()
