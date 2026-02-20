#!/usr/bin/env python3
"""
Phase 101: Walk-Forward Weight Stability + Signal Contribution
===============================================================
Two critical deployment questions:

  A) WEIGHT STABILITY: If we re-optimize P91b weights on rolling 3-year
     windows, do we get similar weights each time? Stable → safe to deploy
     with fixed weights. Unstable → need adaptive system.

  B) SIGNAL CONTRIBUTION (Leave-One-Out): If we drop any single signal,
     how much does performance degrade? Tells us which signals are load-bearing.

  C) TURNOVER ANALYSIS: How much does the strategy trade per month/year?
"""

import copy, json, math, os, sys, time
from pathlib import Path
from itertools import product

PROJ = "/Users/qtmobile/Desktop/Nexus - Quant Trading "
sys.path.insert(0, PROJ)

import numpy as np

from nexus_quant.data.providers.registry import make_provider
from nexus_quant.strategies.registry import make_strategy
from nexus_quant.backtest.engine import BacktestConfig, BacktestEngine
from nexus_quant.backtest.costs import cost_model_from_config

OUT_DIR = os.path.join(PROJ, "artifacts", "phase101")
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
    print(f"[P101] {msg}", flush=True)


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


def optimize_weights(return_matrix, keys, step=0.05):
    """Grid search for balanced (AVG+MIN)/2 optimal weights."""
    n_sig = len(keys)
    n_steps = int(round(1.0 / step))
    best_obj = -999
    best_w = None

    # Generate weight combinations that sum to 1
    for combo in product(range(n_steps + 1), repeat=n_sig - 1):
        s = sum(combo)
        if s > n_steps:
            continue
        last = n_steps - s
        raw = list(combo) + [last]
        w = np.array(raw, dtype=np.float64) / n_steps

        # Compute blended returns per year
        year_sharpes = []
        for yr_rets in return_matrix:
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

        if len(year_sharpes) < 2:
            continue
        avg_s = sum(year_sharpes) / len(year_sharpes)
        min_s = min(year_sharpes)
        obj = (avg_s + min_s) / 2
        if obj > best_obj:
            best_obj = obj
            best_w = {k: round(float(w[i]), 4) for i, k in enumerate(keys)}

    return best_w, best_obj


if __name__ == "__main__":
    t0 = time.time()
    report = {"phase": 101}

    # ════════════════════════════════════
    # Precompute all signal returns per year
    # ════════════════════════════════════
    log("Precomputing signal returns per year...")
    sig_rets_by_year = {}
    for year in YEARS:
        start, end = YEAR_RANGES[year]
        log(f"  {year}")
        sig_rets_by_year[year] = {}
        for sig_key in SIG_KEYS:
            try:
                rets = run_signal(SIGNALS[sig_key], start, end)
            except Exception as exc:
                log(f"    {sig_key} ERROR: {exc}")
                rets = []
            sig_rets_by_year[year][sig_key] = rets

    # ════════════════════════════════════
    # SECTION A: Walk-Forward Weight Optimization
    # ════════════════════════════════════
    log("\n" + "=" * 60)
    log("SECTION A: Walk-forward weight stability")
    log("=" * 60)

    WF_WINDOWS = [
        {"train": ["2021", "2022", "2023"], "test": ["2024"]},
        {"train": ["2022", "2023", "2024"], "test": ["2025"]},
        {"train": ["2021", "2022", "2023", "2024"], "test": ["2025"]},
        {"train": YEARS, "test": []},  # full IS
    ]

    wf_results = []
    for wf in WF_WINDOWS:
        train_label = "+".join(wf["train"])
        test_label = "+".join(wf["test"]) if wf["test"] else "none"
        log(f"\n  Train={train_label} → Test={test_label}")

        # Gather train returns
        train_rets = [sig_rets_by_year[y] for y in wf["train"]]
        opt_w, opt_obj = optimize_weights(train_rets, SIG_KEYS, step=0.05)
        log(f"    Optimal weights: {opt_w} (obj={opt_obj:.4f})")

        # Compute train performance
        train_sharpes = {}
        for year in wf["train"]:
            blended = blend(sig_rets_by_year[year], opt_w)
            train_sharpes[year] = round(compute_sharpe(blended), 4)

        # Compute test performance (if any)
        test_sharpes = {}
        for year in wf["test"]:
            blended = blend(sig_rets_by_year[year], opt_w)
            test_sharpes[year] = round(compute_sharpe(blended), 4)

        result = {
            "train_years": wf["train"],
            "test_years": wf["test"],
            "optimal_weights": opt_w,
            "objective": round(opt_obj, 4),
            "train_sharpes": train_sharpes,
            "test_sharpes": test_sharpes,
        }
        log(f"    Train: {train_sharpes}")
        if test_sharpes:
            log(f"    Test:  {test_sharpes}")
        wf_results.append(result)

    report["walk_forward"] = wf_results

    # Weight stability: how much do weights vary across windows?
    all_weights = [r["optimal_weights"] for r in wf_results]
    weight_stability = {}
    for sig_key in SIG_KEYS:
        weights = [w[sig_key] for w in all_weights]
        weight_stability[sig_key] = {
            "mean": round(np.mean(weights), 4),
            "std": round(np.std(weights), 4),
            "min": round(min(weights), 4),
            "max": round(max(weights), 4),
        }
    report["weight_stability"] = weight_stability
    log(f"\n  Weight stability:")
    for k, v in weight_stability.items():
        log(f"    {k}: mean={v['mean']:.3f} ± {v['std']:.3f} [{v['min']:.2f}, {v['max']:.2f}]")

    # ════════════════════════════════════
    # SECTION B: Leave-One-Out Signal Analysis
    # ════════════════════════════════════
    log("\n" + "=" * 60)
    log("SECTION B: Leave-one-out signal contribution")
    log("=" * 60)

    # Full P91b baseline
    baseline_sharpes = {}
    for year in YEARS:
        blended = blend(sig_rets_by_year[year], P91B_WEIGHTS)
        baseline_sharpes[year] = round(compute_sharpe(blended), 4)
    baseline_avg = round(sum(baseline_sharpes.values()) / len(baseline_sharpes), 4)
    baseline_min = round(min(baseline_sharpes.values()), 4)
    log(f"  Full P91b: AVG={baseline_avg}, MIN={baseline_min}")

    loo_results = {}
    for drop_sig in SIG_KEYS:
        # Re-normalize weights without dropped signal
        remaining = {k: v for k, v in P91B_WEIGHTS.items() if k != drop_sig}
        total = sum(remaining.values())
        renorm = {k: round(v / total, 4) for k, v in remaining.items()}

        yearly = {}
        for year in YEARS:
            rets_subset = {k: sig_rets_by_year[year][k] for k in remaining}
            blended = blend(rets_subset, renorm)
            yearly[year] = round(compute_sharpe(blended), 4)

        vals = list(yearly.values())
        avg_s = round(sum(vals) / len(vals), 4)
        min_s = round(min(vals), 4)
        delta_avg = round(avg_s - baseline_avg, 4)
        delta_min = round(min_s - baseline_min, 4)

        loo_results[drop_sig] = {
            "remaining_weights": renorm,
            "yearly": yearly,
            "avg": avg_s, "min": min_s,
            "delta_avg": delta_avg, "delta_min": delta_min,
        }
        log(f"  Drop {drop_sig}: AVG={avg_s} (Δ{delta_avg:+.4f}), MIN={min_s} (Δ{delta_min:+.4f})")

    report["leave_one_out"] = loo_results
    report["baseline"] = {"yearly": baseline_sharpes, "avg": baseline_avg, "min": baseline_min}

    # ════════════════════════════════════
    # SECTION C: Turnover Analysis
    # ════════════════════════════════════
    log("\n" + "=" * 60)
    log("SECTION C: Turnover analysis")
    log("=" * 60)

    turnover_by_year = {}
    for year in YEARS:
        start, end = YEAR_RANGES[year]
        log(f"  {year}")
        # Run a single signal (v1) to get position changes
        data_cfg = {
            "provider": "binance_rest_v1", "symbols": SYMBOLS,
            "start": start, "end": end, "bar_interval": "1h",
            "cache_dir": ".cache/binance_rest",
        }
        costs_cfg = {"fee_rate": 0.0005, "slippage_rate": 0.0003}
        exec_cfg = {"style": "taker", "slippage_bps": 3.0}
        provider = make_provider(data_cfg, seed=42)
        dataset = provider.load()

        total_turnover = {}
        for sig_key in SIG_KEYS:
            strategy = make_strategy({"name": SIGNALS[sig_key]["name"],
                                     "params": copy.deepcopy(SIGNALS[sig_key]["params"])})
            cost_model = cost_model_from_config(costs_cfg, execution_cfg=exec_cfg)
            engine = BacktestEngine(BacktestConfig(costs=cost_model))
            result = engine.run(dataset=dataset, strategy=strategy, seed=42)

            # Count rebalances and estimate turnover from position changes
            n_trades = getattr(result, 'n_trades', 0)
            total_costs = getattr(result, 'total_costs', 0.0)
            total_turnover[sig_key] = {
                "n_trades": n_trades,
                "total_costs": round(float(total_costs), 6) if total_costs else 0.0,
            }

        turnover_by_year[year] = total_turnover
        log(f"    {total_turnover}")

    report["turnover"] = turnover_by_year

    # ════════════════════════════════════
    # SUMMARY
    # ════════════════════════════════════
    elapsed = round(time.time() - t0, 1)
    report["elapsed_seconds"] = elapsed

    out_path = os.path.join(OUT_DIR, "phase101_report.json")
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)

    log("\n" + "=" * 60)
    log(f"Phase 101 COMPLETE in {elapsed}s → {out_path}")
    log("=" * 60)

    log(f"\nWalk-forward: do weights stay stable?")
    for r in wf_results:
        log(f"  {'+'.join(r['train_years'])} → {r['optimal_weights']}")

    log(f"\nLeave-one-out: which signals are critical?")
    for sig, data in loo_results.items():
        marker = "!!!" if data['delta_min'] < -0.3 else ""
        log(f"  Drop {sig}: ΔAVG={data['delta_avg']:+.4f}, ΔMIN={data['delta_min']:+.4f} {marker}")
