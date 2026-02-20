#!/usr/bin/env python3
"""
Phase 62: Robustness & Anti-Overfitting Validation

Three tests to validate champions:
  A. Parameter Sensitivity (plateau vs spike) — sweep ±10/20/30% around optimal
  B. Deflated Sharpe Audit — correct n_params_searched for multiple comparison
  C. Walk-Forward Stability — fraction of windows profitable

Addresses:
  - Multiple comparison problem (195 tests in Phase 60B, uncorrected)
  - Parameter sensitivity (are optimal params on a plateau or a lucky spike?)
  - Correct Deflated Sharpe via Bailey et al. 2014
"""

import json
import math
import statistics
import subprocess
import sys
import copy
from pathlib import Path

OUT_DIR = "artifacts/phase62"
YEARS = ["2021", "2022", "2023", "2024", "2025"]
SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT", "ADAUSDT", "DOGEUSDT", "AVAXUSDT", "DOTUSDT", "LINKUSDT"]

YEAR_RANGES = {
    "2021": ("2021-01-01", "2022-01-01"),
    "2022": ("2022-01-01", "2023-01-01"),
    "2023": ("2023-01-01", "2024-01-01"),
    "2024": ("2024-01-01", "2025-01-01"),
    "2025": ("2025-01-01", "2026-01-01"),
}

BASE_CONFIG = {
    "seed": 42,
    "venue": {"name": "binance_usdm", "kind": "crypto_perp", "vip_tier": 0},
    "data": {"provider": "binance_rest_v1", "symbols": SYMBOLS, "bar_interval": "1h", "cache_dir": ".cache/binance_rest"},
    "execution": {"style": "taker", "slippage_bps": 3.0},
    "costs": {"fee_rate": 0.0005, "slippage_rate": 0.0003, "cost_multiplier": 1.0},
    "benchmark": {"version": "v1", "walk_forward": {"enabled": True}},
    "risk": {"max_drawdown": 0.30, "max_turnover_per_rebalance": 0.8, "max_gross_leverage": 0.7, "max_position_per_symbol": 0.3},
    "self_learn": {"enabled": False},
}

# ============================================================
# CHAMPION PARAMETER SETS (optimal from Phase 60-61)
# ============================================================

# V1-Long champion params (Phase 54)
V1_PARAMS = {
    "k_per_side": 2,
    "w_carry": 0.35, "w_mom": 0.45, "w_confirm": 0.0,
    "w_mean_reversion": 0.20, "w_vol_momentum": 0.0, "w_funding_trend": 0.0,
    "momentum_lookback_bars": 336, "mean_reversion_lookback_bars": 72,
    "vol_lookback_bars": 168,
    "target_portfolio_vol": 0.0, "use_min_variance": False,
    "target_gross_leverage": 0.35, "min_gross_leverage": 0.05, "max_gross_leverage": 0.65,
    "rebalance_interval_bars": 60, "strict_agreement": False,
}

# Sharpe Ratio Alpha champion params (Phase 60B: sr_336h)
SR_PARAMS = {
    "k_per_side": 2,
    "lookback_bars": 336,
    "vol_lookback_bars": 168,
    "target_gross_leverage": 0.35,
    "rebalance_interval_bars": 48,
}

# Idio Momentum champion params (Phase 61B: idio_336_72)
IDIO_PARAMS = {
    "k_per_side": 2,
    "lookback_bars": 336,
    "beta_window_bars": 72,
    "vol_lookback_bars": 168,
    "target_gross_leverage": 0.30,
    "rebalance_interval_bars": 48,
}

# Phase 60B had ~39 variants × 5 years = 195 tests
N_PARAMS_PHASE60B = 195
# Phase 61B had ~6 variants × 5 years = 30 tests
N_PARAMS_PHASE61B = 30
# Phase 60D ensemble sweep: ~50 weight combinations × 5 years = 250
N_PARAMS_PHASE60D = 250
# Total across all phases:
N_PARAMS_TOTAL = N_PARAMS_PHASE60B + N_PARAMS_PHASE61B + N_PARAMS_PHASE60D  # 475


# ============================================================
# HELPERS
# ============================================================

def _norm_cdf(x: float) -> float:
    t = 1.0 / (1.0 + 0.2316419 * abs(x))
    poly = t * (0.319381530 + t * (-0.356563782 + t * (1.781477937 + t * (-1.821255978 + t * 1.330274429))))
    cdf = 1.0 - (1.0 / math.sqrt(2 * math.pi)) * math.exp(-x * x / 2) * poly
    return cdf if x >= 0 else 1.0 - cdf


def compute_metrics(result_path: str) -> dict:
    d = json.load(open(result_path))
    eq = d.get("equity_curve", [])
    rets = d.get("returns", [])
    if not eq or not rets or len(rets) < 100:
        return {"sharpe": 0, "cagr": 0, "max_dd": 0, "error": "insufficient data"}
    n_years = len(rets) / 8760.0
    cagr = (eq[-1] ** (1.0 / n_years) - 1.0) if n_years > 0 and eq[-1] > 0 else 0
    peak = eq[0]
    max_dd = 0
    for v in eq:
        if v > peak:
            peak = v
        dd = 1 - v / peak if peak > 0 else 0
        if dd > max_dd:
            max_dd = dd
    mu = statistics.mean(rets)
    sd = statistics.pstdev(rets)
    sharpe = (mu / sd) * math.sqrt(8760) if sd > 0 else 0
    return {"sharpe": round(sharpe, 3), "cagr": round(cagr * 100, 2), "max_dd": round(max_dd * 100, 2), "returns": rets}


def deflated_sharpe_ratio(returns: list, n_params_searched: int) -> dict:
    """
    Bailey et al. (2014) Deflated Sharpe Ratio.
    Corrects for non-normality AND multiple comparison bias.
    """
    n = len(returns)
    if n < 10:
        return {"deflated_sharpe": 0.0, "verdict": "INSUFFICIENT_DATA"}

    mu = sum(returns) / n
    std = math.sqrt(sum((r - mu)**2 for r in returns) / (n - 1))
    if std <= 0:
        return {"deflated_sharpe": 0.0, "verdict": "NO_VARIANCE"}

    sr = mu / std  # per-bar Sharpe

    # Higher moments
    m3 = sum((r - mu)**3 for r in returns) / n
    m4 = sum((r - mu)**4 for r in returns) / n
    skew = m3 / (std**3) if std > 0 else 0
    kurt = m4 / (std**4) if std > 0 else 3

    # Non-normality adjustment
    v = 1.0 - skew * sr + ((kurt - 1.0) / 4.0) * sr**2
    if not math.isfinite(v) or v <= 0:
        v = 1e-12
    sigma_sr = math.sqrt(v)
    sr_std = sigma_sr / math.sqrt(max(n - 1, 1))

    # Expected maximum Sharpe under null (Gumbel distribution of maxima)
    n_eff = max(int(n_params_searched), 2)
    a = math.sqrt(2.0 * math.log(n_eff))
    mu_max = a - (math.log(math.log(n_eff)) + math.log(4.0 * math.pi)) / (2.0 * a)
    sr_benchmark = mu_max * sr_std

    # Deflated Sharpe
    if sr_std > 0 and math.isfinite(sr):
        z = (sr - sr_benchmark) / sr_std
        dsr = _norm_cdf(z)
    else:
        dsr = 0.0

    # Annualized Sharpe for reporting
    ann_sharpe = sr * math.sqrt(8760)

    return {
        "n_observations": n,
        "annualized_sharpe": round(ann_sharpe, 3),
        "skewness": round(skew, 3),
        "excess_kurtosis": round(kurt - 3, 3),
        "n_params_searched": n_params_searched,
        "sr_benchmark": round(sr_benchmark * math.sqrt(8760), 3),  # annualized
        "deflated_sharpe": round(dsr, 4),
        "verdict": "NOT_OVERFIT" if dsr >= 0.95 else "LIKELY_OVERFIT",
    }


def make_config(run_name: str, year: str, strategy_name: str, params: dict) -> str:
    cfg = copy.deepcopy(BASE_CONFIG)
    start, end = YEAR_RANGES[year]
    cfg["run_name"] = run_name
    cfg["data"]["start"] = start
    cfg["data"]["end"] = end
    cfg["strategy"] = {"name": strategy_name, "params": params}
    path = f"/tmp/phase62_{run_name}_{year}.json"
    with open(path, "w") as f:
        json.dump(cfg, f, indent=2)
    return path


def run_variant(name: str, strategy_name: str, params: dict, years: list = None) -> dict:
    """Run a strategy across years and return year metrics."""
    if years is None:
        years = YEARS
    year_results = {}
    for year in years:
        run_name = f"{name}_{year}"
        config_path = make_config(run_name, year, strategy_name, params)
        try:
            result = subprocess.run(
                [sys.executable, "-m", "nexus_quant", "run", "--config", config_path, "--out", OUT_DIR],
                capture_output=True, text=True, timeout=600,
            )
            if result.returncode != 0:
                year_results[year] = {"error": result.stderr[-200:]}
                print(f"    {year}: ERROR", flush=True)
                continue
        except subprocess.TimeoutExpired:
            year_results[year] = {"error": "timeout"}
            continue

        runs_dir = Path(OUT_DIR) / "runs"
        if not runs_dir.exists():
            year_results[year] = {"error": "no runs dir"}
            continue
        matching = sorted(
            [d for d in runs_dir.iterdir() if d.is_dir() and d.name.startswith(run_name)],
            key=lambda d: d.stat().st_mtime,
        )
        if matching:
            rp = matching[-1] / "result.json"
            if rp.exists():
                m = compute_metrics(str(rp))
                year_results[year] = m
                print(f"    {year}: Sharpe={m.get('sharpe', '?')}", flush=True)
                continue
        year_results[year] = {"error": "no result"}
        print(f"    {year}: no result", flush=True)

    sharpes = [y.get("sharpe", 0) for y in year_results.values() if isinstance(y.get("sharpe"), (int, float))]
    avg = round(sum(sharpes) / len(sharpes), 3) if sharpes else 0
    mn = round(min(sharpes), 3) if sharpes else 0
    pos = sum(1 for s in sharpes if s > 0)
    year_results["_avg_sharpe"] = avg
    year_results["_min_sharpe"] = mn
    year_results["_pos_years"] = pos
    return year_results


def collect_all_returns(results: dict, years: list = None) -> list:
    """Collect all per-bar returns from year results for DSR computation."""
    if years is None:
        years = YEARS
    all_rets = []
    for year in years:
        yr = results.get(year, {})
        all_rets.extend(yr.get("returns", []))
    return all_rets


# ============================================================
# PHASE 62A: PARAMETER SENSITIVITY SWEEP
# ============================================================

def phase_62a_sensitivity():
    """
    Test: are champion params on a PLATEAU (robust) or SPIKE (overfit)?
    For each key parameter: sweep -30%, -20%, -10%, 0% (optimal), +10%, +20%, +30%.
    A robust parameter should show ≤ 20% Sharpe degradation across the range.
    """
    print("\n" + "=" * 80)
    print("  PHASE 62A: PARAMETER SENSITIVITY (PLATEAU vs SPIKE)")
    print("=" * 80, flush=True)
    print("  Methodology: sweep key params ±10/20/30%, track AVG Sharpe degradation")
    print("  If Sharpe drops >30% on ±10% perturbation → SPIKE (overfitting indicator)")
    print("=" * 80, flush=True)

    all_results = {}

    # ─── Sharpe Ratio Alpha: lookback_bars sensitivity ───
    print("\n" + "─" * 60)
    print("  SR Alpha: lookback_bars sensitivity (optimal=336)")
    print("─" * 60, flush=True)

    sr_lookbacks = {
        "sr_lb235": 235,   # -30%
        "sr_lb269": 269,   # -20%
        "sr_lb302": 302,   # -10%
        "sr_lb336": 336,   # OPTIMAL
        "sr_lb370": 370,   # +10%
        "sr_lb403": 403,   # +20%
        "sr_lb437": 437,   # +30%
    }
    sr_lb_results = {}
    for name, lb in sr_lookbacks.items():
        params = {**SR_PARAMS, "lookback_bars": lb}
        print(f"\n  >> {name} (lookback={lb})", flush=True)
        r = run_variant(name, "sharpe_ratio_alpha", params)
        sr_lb_results[name] = r
        print(f"     AVG={r['_avg_sharpe']}, MIN={r['_min_sharpe']}, pos={r['_pos_years']}/5", flush=True)
    all_results["sr_lookback"] = sr_lb_results

    # ─── Sharpe Ratio Alpha: rebalance_interval_bars sensitivity ───
    print("\n" + "─" * 60)
    print("  SR Alpha: rebalance_interval_bars sensitivity (optimal=48)")
    print("─" * 60, flush=True)

    sr_rebals = {
        "sr_rb24": 24,   # -50%
        "sr_rb36": 36,   # -25%
        "sr_rb48": 48,   # OPTIMAL
        "sr_rb60": 60,   # +25%
        "sr_rb72": 72,   # +50%
        "sr_rb96": 96,   # +100%
    }
    sr_rebal_results = {}
    for name, rb in sr_rebals.items():
        params = {**SR_PARAMS, "rebalance_interval_bars": rb}
        print(f"\n  >> {name} (rebal={rb}h)", flush=True)
        r = run_variant(name, "sharpe_ratio_alpha", params)
        sr_rebal_results[name] = r
        print(f"     AVG={r['_avg_sharpe']}, MIN={r['_min_sharpe']}, pos={r['_pos_years']}/5", flush=True)
    all_results["sr_rebalance"] = sr_rebal_results

    # ─── Idio Momentum: lookback_bars sensitivity ───
    print("\n" + "─" * 60)
    print("  Idio Momentum: lookback_bars sensitivity (optimal=336)")
    print("─" * 60, flush=True)

    idio_lookbacks = {
        "idio_lb235": 235,   # -30%
        "idio_lb269": 269,   # -20%
        "idio_lb302": 302,   # -10%
        "idio_lb336": 336,   # OPTIMAL
        "idio_lb370": 370,   # +10%
        "idio_lb403": 403,   # +20%
        "idio_lb437": 437,   # +30%
    }
    idio_lb_results = {}
    for name, lb in idio_lookbacks.items():
        params = {**IDIO_PARAMS, "lookback_bars": lb}
        print(f"\n  >> {name} (lookback={lb})", flush=True)
        r = run_variant(name, "idio_momentum_alpha", params)
        idio_lb_results[name] = r
        print(f"     AVG={r['_avg_sharpe']}, MIN={r['_min_sharpe']}, pos={r['_pos_years']}/5", flush=True)
    all_results["idio_lookback"] = idio_lb_results

    # ─── Idio Momentum: beta_window_bars sensitivity ───
    print("\n" + "─" * 60)
    print("  Idio Momentum: beta_window_bars sensitivity (optimal=72)")
    print("─" * 60, flush=True)

    idio_betas = {
        "idio_bw24": 24,    # -67%
        "idio_bw48": 48,    # -33%
        "idio_bw72": 72,    # OPTIMAL
        "idio_bw96": 96,    # +33%
        "idio_bw120": 120,  # +67%
        "idio_bw168": 168,  # +133%
    }
    idio_beta_results = {}
    for name, bw in idio_betas.items():
        params = {**IDIO_PARAMS, "beta_window_bars": bw}
        print(f"\n  >> {name} (beta_window={bw}h)", flush=True)
        r = run_variant(name, "idio_momentum_alpha", params)
        idio_beta_results[name] = r
        print(f"     AVG={r['_avg_sharpe']}, MIN={r['_min_sharpe']}, pos={r['_pos_years']}/5", flush=True)
    all_results["idio_beta_window"] = idio_beta_results

    # ─── Idio Momentum: target_gross_leverage sensitivity ───
    print("\n" + "─" * 60)
    print("  Idio Momentum: leverage sensitivity (optimal=0.30)")
    print("─" * 60, flush=True)

    idio_levs = {
        "idio_lev20": 0.20,
        "idio_lev25": 0.25,
        "idio_lev30": 0.30,   # OPTIMAL
        "idio_lev35": 0.35,
        "idio_lev40": 0.40,
    }
    idio_lev_results = {}
    for name, lev in idio_levs.items():
        params = {**IDIO_PARAMS, "target_gross_leverage": lev}
        print(f"\n  >> {name} (lev={lev})", flush=True)
        r = run_variant(name, "idio_momentum_alpha", params)
        idio_lev_results[name] = r
        print(f"     AVG={r['_avg_sharpe']}, MIN={r['_min_sharpe']}, pos={r['_pos_years']}/5", flush=True)
    all_results["idio_leverage"] = idio_lev_results

    return all_results


# ============================================================
# PHASE 62B: DEFLATED SHARPE AUDIT
# ============================================================

def phase_62b_deflated_sharpe(sensitivity_results: dict):
    """
    Correct the Deflated Sharpe Ratio computation using the actual
    number of parameter trials tested in Phase 60-61.
    """
    print("\n" + "=" * 80)
    print("  PHASE 62B: DEFLATED SHARPE RATIO AUDIT")
    print("=" * 80)
    print("  Methodology: Bailey et al. (2014) DSR with correct n_params_searched")
    print(f"  Phase 60B ran {N_PARAMS_PHASE60B} variants × 5 years = {N_PARAMS_PHASE60B*5} backtests")
    print(f"  Phase 61B ran {N_PARAMS_PHASE61B} variants × 5 years = {N_PARAMS_PHASE61B*5} backtests")
    print(f"  Total parameter trials: ~{N_PARAMS_TOTAL}")
    print("=" * 80, flush=True)

    dsr_results = {}

    # Collect returns for each champion across all 5 years from sensitivity results
    # Use the optimal (0% perturbation) variant

    champion_keys = {
        "sr_336h": ("sr_lookback", "sr_lb336", "sharpe_ratio_alpha"),
        "idio_336_72": ("idio_lookback", "idio_lb336", "idio_momentum_alpha"),
    }

    for champion_name, (group, key, strat) in champion_keys.items():
        group_results = sensitivity_results.get(group, {})
        yr_data = group_results.get(key, {})
        all_rets = []
        for year in YEARS:
            all_rets.extend(yr_data.get(year, {}).get("returns", []))

        if not all_rets:
            print(f"\n  {champion_name}: no returns data collected")
            continue

        print(f"\n  {'─'*60}")
        print(f"  CHAMPION: {champion_name} ({len(all_rets)} hourly obs)")
        print(f"  {'─'*60}")

        for n_trials, label in [
            (1, "n=1 (INCORRECT — how it was computed)"),
            (N_PARAMS_PHASE60B, f"n={N_PARAMS_PHASE60B} (Phase 60B: correct)"),
            (N_PARAMS_TOTAL, f"n={N_PARAMS_TOTAL} (ALL phases: conservative)"),
        ]:
            dsr = deflated_sharpe_ratio(all_rets, n_trials)
            verdict_marker = "✓" if dsr["verdict"] == "NOT_OVERFIT" else "⚠"
            print(f"  {verdict_marker} {label}")
            print(f"    Annualized Sharpe: {dsr['annualized_sharpe']}")
            print(f"    Expected max Sharpe (null): {dsr.get('sr_benchmark', '?')}")
            print(f"    Deflated Sharpe: {dsr['deflated_sharpe']}")
            print(f"    Verdict: {dsr['verdict']}")

        dsr_results[champion_name] = {
            "n_obs": len(all_rets),
            "dsr_n1": deflated_sharpe_ratio(all_rets, 1),
            f"dsr_n{N_PARAMS_PHASE60B}": deflated_sharpe_ratio(all_rets, N_PARAMS_PHASE60B),
            f"dsr_n{N_PARAMS_TOTAL}": deflated_sharpe_ratio(all_rets, N_PARAMS_TOTAL),
        }

    return dsr_results


# ============================================================
# PHASE 62C: WALK-FORWARD STABILITY
# ============================================================

def phase_62c_wf_stability():
    """
    For each champion, run and report walk-forward window statistics.
    The benchmark.py computes walk_forward metrics — we extract them from result.json.
    A strategy is walk-forward stable if fraction_profitable > 0.70.
    """
    print("\n" + "=" * 80)
    print("  PHASE 62C: WALK-FORWARD WINDOW STABILITY")
    print("=" * 80)
    print("  Methodology: rolling 720-bar windows, report fraction profitable")
    print("  Threshold: fraction_profitable >= 0.70 = STABLE")
    print("=" * 80, flush=True)

    strategies = [
        ("wf_v1long", "nexus_alpha_v1", V1_PARAMS),
        ("wf_sr336", "sharpe_ratio_alpha", SR_PARAMS),
        ("wf_idio336", "idio_momentum_alpha", IDIO_PARAMS),
    ]

    wf_results = {}

    for name, strat, params in strategies:
        print(f"\n  {'─'*60}")
        print(f"  Running: {name} ({strat})", flush=True)
        year_wf = {}
        for year in YEARS:
            run_name = f"{name}_{year}"
            config_path = make_config(run_name, year, strat, params)
            try:
                result = subprocess.run(
                    [sys.executable, "-m", "nexus_quant", "run", "--config", config_path, "--out", OUT_DIR],
                    capture_output=True, text=True, timeout=600,
                )
                if result.returncode != 0:
                    year_wf[year] = {"error": "run failed"}
                    print(f"    {year}: ERROR", flush=True)
                    continue
            except subprocess.TimeoutExpired:
                year_wf[year] = {"error": "timeout"}
                continue

            runs_dir = Path(OUT_DIR) / "runs"
            matching = sorted(
                [d for d in runs_dir.iterdir() if d.is_dir() and d.name.startswith(run_name)],
                key=lambda d: d.stat().st_mtime,
            ) if runs_dir.exists() else []
            if not matching:
                year_wf[year] = {"error": "no result"}
                continue

            rp = matching[-1] / "result.json"
            if not rp.exists():
                year_wf[year] = {"error": "no result.json"}
                continue

            d = json.load(open(rp))
            sharpe = 0
            rets = d.get("returns", [])
            if rets:
                mu = statistics.mean(rets)
                sd = statistics.pstdev(rets)
                sharpe = round((mu / sd) * math.sqrt(8760), 3) if sd > 0 else 0

            # Extract walk-forward stats from benchmark
            bench = d.get("benchmark", {})
            wf = bench.get("walk_forward", {})
            frac_prof = wf.get("fraction_profitable", None)
            frac_calmar = wf.get("fraction_calmar_positive", None)
            n_windows = wf.get("n_windows", None)

            year_wf[year] = {
                "sharpe": sharpe,
                "wf_fraction_profitable": frac_prof,
                "wf_fraction_calmar_positive": frac_calmar,
                "wf_n_windows": n_windows,
            }

            prof_str = f"{frac_prof:.2f}" if frac_prof is not None else "?"
            print(f"    {year}: Sharpe={sharpe}, WF_frac_prof={prof_str}", flush=True)

        # Compute aggregate WF stability
        frac_profs = [v["wf_fraction_profitable"] for v in year_wf.values()
                      if isinstance(v.get("wf_fraction_profitable"), (int, float))]
        avg_wf = round(sum(frac_profs) / len(frac_profs), 3) if frac_profs else None
        wf_stable = avg_wf is not None and avg_wf >= 0.70

        wf_results[name] = {
            "years": year_wf,
            "avg_wf_fraction_profitable": avg_wf,
            "wf_stable": wf_stable,
        }
        print(f"\n  {name}: avg WF frac_prof={avg_wf}, STABLE={wf_stable}", flush=True)

    return wf_results


# ============================================================
# PHASE 62D: SENSITIVITY PLATEAU ANALYSIS
# ============================================================

def analyze_sensitivity_plateau(sensitivity_results: dict):
    """
    For each parameter sweep group, compute:
    - Coefficient of Variation (CV) of Sharpe across perturbations
    - Max degradation from optimal
    - PLATEAU verdict: CV < 0.15 and max_degradation < 30%
    """
    print("\n" + "=" * 80)
    print("  PHASE 62D: PLATEAU ANALYSIS SUMMARY")
    print("=" * 80, flush=True)
    print("  Plateau = CV of Sharpe < 0.15 AND max degradation < 30%")
    print("  Spike   = Sharpe collapses sharply away from optimal → overfitting signal")
    print("=" * 80)

    plateau_results = {}

    for group_name, group_data in sensitivity_results.items():
        sharpes = {}
        for variant, vdata in group_data.items():
            avg = vdata.get("_avg_sharpe")
            if avg is not None and isinstance(avg, (int, float)):
                sharpes[variant] = avg

        if not sharpes:
            continue

        vals = list(sharpes.values())
        optimal = max(vals)
        mu_s = sum(vals) / len(vals)
        if mu_s > 0:
            cv = math.sqrt(sum((s - mu_s)**2 for s in vals) / len(vals)) / mu_s
        else:
            cv = float("inf")

        min_s = min(vals)
        max_deg = (optimal - min_s) / abs(optimal) if optimal != 0 else 0

        is_plateau = cv < 0.15 and max_deg < 0.30
        verdict = "PLATEAU ✓" if is_plateau else "SPIKE ⚠ (overfitting risk)"

        print(f"\n  [{group_name}]")
        print(f"    Optimal AVG: {optimal:.3f}")
        print(f"    Range: [{min_s:.3f}, {optimal:.3f}]")
        print(f"    Max degradation: {max_deg*100:.1f}%")
        print(f"    CV (variation): {cv:.3f}")
        print(f"    Verdict: {verdict}")
        for k, v in sorted(sharpes.items()):
            mark = "★" if v == optimal else "  "
            pct = (v - optimal) / abs(optimal) * 100 if optimal != 0 else 0
            print(f"      {mark} {k:<20}: {v:.3f}  ({pct:+.1f}% from optimal)")

        plateau_results[group_name] = {
            "optimal": optimal,
            "cv": round(cv, 4),
            "max_degradation": round(max_deg, 4),
            "is_plateau": is_plateau,
            "verdict": verdict,
        }

    return plateau_results


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 80)
    print("  PHASE 62: ROBUSTNESS & ANTI-OVERFITTING VALIDATION")
    print("=" * 80)
    print("  Tests: Parameter Sensitivity | Deflated Sharpe | Walk-Forward Stability")
    print("=" * 80, flush=True)

    Path(OUT_DIR).mkdir(parents=True, exist_ok=True)

    # A: Parameter Sensitivity
    sensitivity_results = phase_62a_sensitivity()

    # D: Plateau Analysis (uses results from A)
    plateau_results = analyze_sensitivity_plateau(sensitivity_results)

    # B: Deflated Sharpe (uses returns collected in A)
    dsr_results = phase_62b_deflated_sharpe(sensitivity_results)

    # C: Walk-Forward Stability
    wf_results = phase_62c_wf_stability()

    # ─── Final Summary ───
    print("\n" + "=" * 80)
    print("  PHASE 62 FINAL SUMMARY")
    print("=" * 80)

    print("\n  PARAMETER SENSITIVITY (PLATEAU vs SPIKE):")
    for group, r in plateau_results.items():
        print(f"    {group:<25}: {r['verdict']}")
        print(f"      CV={r['cv']:.3f}, MaxDeg={r['max_degradation']*100:.1f}%, Optimal={r['optimal']:.3f}")

    print("\n  DEFLATED SHARPE RATIO (corrected for n_trials):")
    for champion, r in dsr_results.items():
        n1 = r.get("dsr_n1", {})
        nc = r.get(f"dsr_n{N_PARAMS_PHASE60B}", {})
        nt = r.get(f"dsr_n{N_PARAMS_TOTAL}", {})
        print(f"    {champion}:")
        print(f"      Raw (n=1):              Sharpe={n1.get('annualized_sharpe','?')}, DSR={n1.get('deflated_sharpe','?')}, {n1.get('verdict','?')}")
        print(f"      Corrected (n={N_PARAMS_PHASE60B}): DSR={nc.get('deflated_sharpe','?')}, {nc.get('verdict','?')}")
        print(f"      Conservative (n={N_PARAMS_TOTAL}): DSR={nt.get('deflated_sharpe','?')}, {nt.get('verdict','?')}")

    print("\n  WALK-FORWARD STABILITY:")
    for strat, r in wf_results.items():
        frac = r.get("avg_wf_fraction_profitable")
        stable = r.get("wf_stable")
        stable_str = "STABLE ✓" if stable else "UNSTABLE ⚠"
        print(f"    {strat:<25}: avg_wf_frac_prof={frac}, {stable_str}")

    # Save all
    out = {
        "sensitivity": sensitivity_results,
        "plateau_analysis": plateau_results,
        "deflated_sharpe": dsr_results,
        "walk_forward": wf_results,
    }
    out_path = Path(OUT_DIR) / "phase62_results.json"
    # Strip large returns arrays before saving
    def strip_returns(d):
        if isinstance(d, dict):
            return {k: strip_returns(v) for k, v in d.items() if k != "returns"}
        return d
    with open(out_path, "w") as f:
        json.dump(strip_returns(out), f, indent=2)
    print(f"\nResults saved to: {out_path}")

    print("\n" + "=" * 80)
    print("  INTERPRETATION GUIDE")
    print("=" * 80)
    print("  PLATEAU ✓ = Parameter is robust; Sharpe stable across perturbations")
    print("  SPIKE ⚠  = Sharpe collapses away from optimal → potential overfitting")
    print("  DSR ≥ 0.95 = NOT_OVERFIT (after correcting for multiple comparisons)")
    print("  DSR < 0.95 = LIKELY_OVERFIT (expected max Sharpe under null exceeds observed)")
    print("  WF frac_prof ≥ 0.70 = STABLE walk-forward consistency")
    print("=" * 80)


if __name__ == "__main__":
    main()
