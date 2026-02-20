#!/usr/bin/env python3
"""
Phase 102: 3-Signal Ensemble — Drop V1, Optimize Weights
==========================================================
Phase 101 revealed:
  - v1 gets 0-5% weight across ALL walk-forward windows
  - Dropping v1 IMPROVES AVG by +0.17 and MIN by +0.01
  - f144 is perfectly stable at 40%
  - i415bw216 is most critical for MIN

This phase tests:
  A) 3-signal ensemble (i415 + i460 + f144) with multiple weight schemes:
     - P91b renormalized (remove v1, renorm remaining)
     - Walk-forward consensus (avg of WF window weights)
     - Grid-optimized on full IS (5yr balanced objective)
     - f144-heavy (40/30/30) — guided by WF stability

  B) OOS validation on 2026

  C) Compare all variants vs 4-signal P91b baseline
"""

import copy, json, os, sys, time
from pathlib import Path
from itertools import product

PROJ = "/Users/qtmobile/Desktop/Nexus - Quant Trading "
sys.path.insert(0, PROJ)

import numpy as np

from nexus_quant.data.providers.registry import make_provider
from nexus_quant.strategies.registry import make_strategy
from nexus_quant.backtest.engine import BacktestConfig, BacktestEngine
from nexus_quant.backtest.costs import cost_model_from_config

OUT_DIR = os.path.join(PROJ, "artifacts", "phase102")
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

# 4-signal P91b baseline (for comparison)
P91B_WEIGHTS_4SIG = {"v1": 0.2747, "i460bw168": 0.1967, "i415bw216": 0.3247, "f144": 0.2039}

# 3-signal candidates (no v1)
THREE_SIG_KEYS = ["f144", "i415bw216", "i460bw168"]

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
    print(f"[P102] {msg}", flush=True)


def compute_sharpe(returns, bars_per_year=8760):
    arr = np.asarray(returns, dtype=np.float64)
    if arr.size < 50:
        return 0.0
    std = float(np.std(arr))
    if std <= 0:
        return 0.0
    return float(np.mean(arr) / std * np.sqrt(bars_per_year))


def run_signal(sig_cfg, start, end):
    data_cfg = {
        "provider": "binance_rest_v1", "symbols": SYMBOLS,
        "start": start, "end": end, "bar_interval": "1h",
        "cache_dir": ".cache/binance_rest",
    }
    costs_cfg = {"fee_rate": 0.0005, "slippage_rate": 0.0003}
    exec_cfg = {"style": "taker", "slippage_bps": 3.0}
    provider = make_provider(data_cfg, seed=42)
    dataset = provider.load()
    strategy = make_strategy({"name": sig_cfg["name"], "params": copy.deepcopy(sig_cfg["params"])})
    cost_model = cost_model_from_config(costs_cfg, execution_cfg=exec_cfg)
    engine = BacktestEngine(BacktestConfig(costs=cost_model))
    result = engine.run(dataset=dataset, strategy=strategy, seed=42)
    return result.returns


def blend(sig_rets, weights):
    keys = sorted(weights.keys())
    n = min(len(sig_rets.get(k, [])) for k in keys)
    if n == 0:
        return []
    R = np.zeros((len(keys), n), dtype=np.float64)
    W = np.array([weights[k] for k in keys], dtype=np.float64)
    for i, k in enumerate(keys):
        R[i, :] = sig_rets[k][:n]
    return (W @ R).tolist()


def grid_optimize_3sig(sig_rets_by_year, keys, step=0.025):
    """Fine-grid search for optimal 3-signal weights using balanced (AVG+MIN)/2."""
    n_sig = len(keys)
    n_steps = int(round(1.0 / step))
    best_obj = -999
    best_w = None
    tested = 0

    for combo in product(range(n_steps + 1), repeat=n_sig - 1):
        s = sum(combo)
        if s > n_steps:
            continue
        last = n_steps - s
        raw = list(combo) + [last]
        w = np.array(raw, dtype=np.float64) / n_steps

        # Skip if any weight is 0 (we already tested LOO)
        if any(w_i < 0.001 for w_i in w):
            continue

        year_sharpes = []
        for year in YEARS:
            yr_rets = sig_rets_by_year[year]
            n = min(len(yr_rets[k]) for k in keys)
            if n < 50:
                continue
            R = np.zeros((n_sig, n), dtype=np.float64)
            for i, k in enumerate(keys):
                R[i, :] = yr_rets[k][:n]
            blended = w @ R
            std = float(np.std(blended))
            if std > 0:
                year_sharpes.append(float(np.mean(blended) / std * np.sqrt(8760)))
            else:
                year_sharpes.append(0.0)

        if len(year_sharpes) < 3:
            continue
        avg_s = sum(year_sharpes) / len(year_sharpes)
        min_s = min(year_sharpes)
        obj = (avg_s + min_s) / 2
        tested += 1
        if obj > best_obj:
            best_obj = obj
            best_w = {k: round(float(w[i]), 4) for i, k in enumerate(keys)}

    return best_w, best_obj, tested


if __name__ == "__main__":
    t0 = time.time()
    report = {"phase": 102}

    # ════════════════════════════════════
    # Precompute all signal returns per year + OOS
    # ════════════════════════════════════
    log("Precomputing signal returns per year...")
    all_sig_keys = sorted(set(list(P91B_WEIGHTS_4SIG.keys()) + THREE_SIG_KEYS))
    sig_rets_by_year = {}
    for year in YEARS:
        start, end = YEAR_RANGES[year]
        log(f"  {year}")
        sig_rets_by_year[year] = {}
        for sig_key in all_sig_keys:
            try:
                rets = run_signal(SIGNALS[sig_key], start, end)
            except Exception as exc:
                log(f"    {sig_key} ERROR: {exc}")
                rets = []
            sig_rets_by_year[year][sig_key] = rets

    # OOS 2026
    log("  2026 OOS")
    sig_rets_oos = {}
    for sig_key in all_sig_keys:
        try:
            rets = run_signal(SIGNALS[sig_key], OOS_RANGE[0], OOS_RANGE[1])
        except Exception as exc:
            log(f"    {sig_key} OOS ERROR: {exc}")
            rets = []
        sig_rets_oos[sig_key] = rets

    # ════════════════════════════════════
    # SECTION A: Weight variants for 3-signal ensemble
    # ════════════════════════════════════
    log("\n" + "=" * 60)
    log("SECTION A: 3-signal weight variants")
    log("=" * 60)

    # Variant 1: P91b renormalized (drop v1, renorm)
    p91b_3sig = {k: v for k, v in P91B_WEIGHTS_4SIG.items() if k != "v1"}
    total = sum(p91b_3sig.values())
    p91b_renorm = {k: round(v / total, 4) for k, v in p91b_3sig.items()}
    log(f"  p91b_renorm: {p91b_renorm}")

    # Variant 2: Walk-forward consensus (avg of WF windows from Phase 101)
    # WF windows gave: [f144=0.40, i415=0.55, i460=0.00], [f144=0.40, i415=0.60, i460=0.00],
    #                  [f144=0.40, i415=0.35, i460=0.20], [f144=0.40, i415=0.25, i460=0.30]
    # Average (3sig only, renormalized):
    wf_avg_raw = {"f144": 0.40, "i415bw216": 0.4375, "i460bw168": 0.125}
    wf_total = sum(wf_avg_raw.values())
    wf_consensus = {k: round(v / wf_total, 4) for k, v in wf_avg_raw.items()}
    log(f"  wf_consensus: {wf_consensus}")

    # Variant 3: Grid-optimized (full IS, 2.5% steps)
    log("  Grid optimizing 3-signal weights (2.5% steps)...")
    three_sig_rets = {y: {k: sig_rets_by_year[y][k] for k in THREE_SIG_KEYS} for y in YEARS}
    grid_opt, grid_obj, n_tested = grid_optimize_3sig(three_sig_rets, THREE_SIG_KEYS, step=0.025)
    log(f"  grid_optimal: {grid_opt} (obj={grid_obj:.4f}, tested {n_tested} combos)")

    # Variant 4: f144-heavy (guided by WF stability)
    f144_heavy = {"f144": 0.40, "i415bw216": 0.35, "i460bw168": 0.25}
    log(f"  f144_heavy: {f144_heavy}")

    # Variant 5: Equal weight
    equal_w = {"f144": 0.3333, "i415bw216": 0.3334, "i460bw168": 0.3333}
    log(f"  equal: {equal_w}")

    weight_variants = {
        "p91b_4sig_baseline": P91B_WEIGHTS_4SIG,
        "p91b_renorm_3sig": p91b_renorm,
        "wf_consensus_3sig": wf_consensus,
        "grid_optimal_3sig": grid_opt,
        "f144_heavy_3sig": f144_heavy,
        "equal_3sig": equal_w,
    }

    # ════════════════════════════════════
    # SECTION B: Evaluate all variants
    # ════════════════════════════════════
    log("\n" + "=" * 60)
    log("SECTION B: Evaluate all weight variants")
    log("=" * 60)

    results = {}
    for variant_name, weights in weight_variants.items():
        keys_used = sorted(weights.keys())
        is_3sig = "v1" not in keys_used

        yearly_sharpes = {}
        for year in YEARS:
            rets = {k: sig_rets_by_year[year][k] for k in keys_used}
            blended = blend(rets, weights)
            yearly_sharpes[year] = round(compute_sharpe(blended), 4)

        # OOS
        rets_oos = {k: sig_rets_oos[k] for k in keys_used}
        blended_oos = blend(rets_oos, weights)
        oos_sharpe = round(compute_sharpe(blended_oos), 4)

        vals = list(yearly_sharpes.values())
        avg_s = round(sum(vals) / len(vals), 4)
        min_s = round(min(vals), 4)
        obj = round((avg_s + min_s) / 2, 4)

        results[variant_name] = {
            "weights": weights,
            "n_signals": len(keys_used),
            "yearly": yearly_sharpes,
            "avg": avg_s,
            "min": min_s,
            "objective": obj,
            "oos_2026": oos_sharpe,
        }
        log(f"\n  {variant_name}:")
        log(f"    Weights: {weights}")
        log(f"    Yearly:  {yearly_sharpes}")
        log(f"    AVG={avg_s}, MIN={min_s}, OBJ={obj}, OOS={oos_sharpe}")

    report["variants"] = results

    # ════════════════════════════════════
    # SECTION C: Fine-tune around grid optimum (1% steps)
    # ════════════════════════════════════
    log("\n" + "=" * 60)
    log("SECTION C: Ultra-fine grid around optimum (1% steps)")
    log("=" * 60)

    fine_opt, fine_obj, fine_n = grid_optimize_3sig(three_sig_rets, THREE_SIG_KEYS, step=0.01)
    log(f"  fine_optimal: {fine_opt} (obj={fine_obj:.4f}, tested {fine_n} combos)")

    # Evaluate fine optimum
    fine_yearly = {}
    for year in YEARS:
        rets = {k: sig_rets_by_year[year][k] for k in THREE_SIG_KEYS}
        blended = blend(rets, fine_opt)
        fine_yearly[year] = round(compute_sharpe(blended), 4)
    fine_oos_rets = {k: sig_rets_oos[k] for k in THREE_SIG_KEYS}
    fine_blended_oos = blend(fine_oos_rets, fine_opt)
    fine_oos = round(compute_sharpe(fine_blended_oos), 4)
    fine_vals = list(fine_yearly.values())
    fine_avg = round(sum(fine_vals) / len(fine_vals), 4)
    fine_min = round(min(fine_vals), 4)

    report["fine_optimal"] = {
        "weights": fine_opt,
        "yearly": fine_yearly,
        "avg": fine_avg,
        "min": fine_min,
        "objective": round((fine_avg + fine_min) / 2, 4),
        "oos_2026": fine_oos,
        "n_tested": fine_n,
    }
    log(f"  Yearly: {fine_yearly}")
    log(f"  AVG={fine_avg}, MIN={fine_min}, OOS={fine_oos}")

    # ════════════════════════════════════
    # SECTION D: Comparison table
    # ════════════════════════════════════
    log("\n" + "=" * 60)
    log("SECTION D: Comparison")
    log("=" * 60)

    baseline = results["p91b_4sig_baseline"]
    comparisons = {}
    for name, data in results.items():
        if name == "p91b_4sig_baseline":
            continue
        comparisons[name] = {
            "delta_avg": round(data["avg"] - baseline["avg"], 4),
            "delta_min": round(data["min"] - baseline["min"], 4),
            "delta_obj": round(data["objective"] - baseline["objective"], 4),
            "delta_oos": round(data["oos_2026"] - baseline["oos_2026"], 4),
        }
    # Add fine optimal comparison
    comparisons["fine_optimal_3sig"] = {
        "delta_avg": round(fine_avg - baseline["avg"], 4),
        "delta_min": round(fine_min - baseline["min"], 4),
        "delta_obj": round((fine_avg + fine_min) / 2 - baseline["objective"], 4),
        "delta_oos": round(fine_oos - baseline["oos_2026"], 4),
    }
    report["comparisons"] = comparisons

    log(f"\n{'Variant':>25} | {'AVG':>7} | {'MIN':>7} | {'OBJ':>7} | {'OOS':>7} | ΔAVG | ΔMIN | ΔOOS")
    log(f"{'-'*25}-+-{'-'*7}-+-{'-'*7}-+-{'-'*7}-+-{'-'*7}-+------+------+------")
    log(f"{'p91b_4sig_baseline':>25} | {baseline['avg']:>7.3f} | {baseline['min']:>7.3f} | {baseline['objective']:>7.3f} | {baseline['oos_2026']:>7.3f} |  ref |  ref |  ref")
    for name, data in results.items():
        if name == "p91b_4sig_baseline":
            continue
        c = comparisons[name]
        log(f"{name:>25} | {data['avg']:>7.3f} | {data['min']:>7.3f} | {data['objective']:>7.3f} | {data['oos_2026']:>7.3f} | {c['delta_avg']:+.3f} | {c['delta_min']:+.3f} | {c['delta_oos']:+.3f}")
    c = comparisons["fine_optimal_3sig"]
    log(f"{'fine_optimal_3sig':>25} | {fine_avg:>7.3f} | {fine_min:>7.3f} | {(fine_avg+fine_min)/2:>7.3f} | {fine_oos:>7.3f} | {c['delta_avg']:+.3f} | {c['delta_min']:+.3f} | {c['delta_oos']:+.3f}")

    # ════════════════════════════════════
    # SUMMARY & RECOMMENDATION
    # ════════════════════════════════════
    elapsed = round(time.time() - t0, 1)
    report["elapsed_seconds"] = elapsed

    # Find best 3-signal variant by objective
    all_3sig = {k: v for k, v in results.items() if v["n_signals"] == 3}
    all_3sig["fine_optimal_3sig"] = {
        "weights": fine_opt, "n_signals": 3,
        "yearly": fine_yearly, "avg": fine_avg, "min": fine_min,
        "objective": round((fine_avg + fine_min) / 2, 4), "oos_2026": fine_oos,
    }
    best_name = max(all_3sig, key=lambda k: all_3sig[k]["objective"])
    best_data = all_3sig[best_name]

    report["recommendation"] = {
        "best_3sig_variant": best_name,
        "best_weights": best_data["weights"],
        "best_avg": best_data["avg"],
        "best_min": best_data["min"],
        "best_objective": best_data["objective"],
        "best_oos": best_data["oos_2026"],
        "vs_p91b_4sig": {
            "delta_avg": round(best_data["avg"] - baseline["avg"], 4),
            "delta_min": round(best_data["min"] - baseline["min"], 4),
            "delta_oos": round(best_data["oos_2026"] - baseline["oos_2026"], 4),
        },
        "verdict": "UPGRADE" if best_data["min"] > baseline["min"] else "KEEP_P91B",
    }

    out_path = os.path.join(OUT_DIR, "phase102_report.json")
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)

    log("\n" + "=" * 60)
    log(f"Phase 102 COMPLETE in {elapsed}s → {out_path}")
    log("=" * 60)

    log(f"\nRECOMMENDATION: {report['recommendation']['verdict']}")
    log(f"  Best 3-sig: {best_name}")
    log(f"  Weights: {best_data['weights']}")
    log(f"  AVG={best_data['avg']}, MIN={best_data['min']}, OOS={best_data['oos_2026']}")
    log(f"  vs P91b 4-sig: ΔAVG={report['recommendation']['vs_p91b_4sig']['delta_avg']:+.4f}, "
        f"ΔMIN={report['recommendation']['vs_p91b_4sig']['delta_min']:+.4f}, "
        f"ΔOOS={report['recommendation']['vs_p91b_4sig']['delta_oos']:+.4f}")
