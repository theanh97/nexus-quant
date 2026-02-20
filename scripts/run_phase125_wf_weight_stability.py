#!/usr/bin/env python3
"""
Phase 125: Walk-Forward Weight Stability + LOYO Validation
============================================================
Leave-One-Year-Out (LOYO) cross-validation of ensemble weights:
1. For each year Y: optimize weights on remaining 4 years, test on Y
2. Compare LOYO test Sharpe vs champion weights on same test year
3. Measure weight stability across folds (do optimal weights change a lot?)
4. Final verdict: are champion weights robust or overfit?
"""
import json
import os
import signal as _signal
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from nexus_quant.backtest.engine import BacktestConfig, BacktestEngine
from nexus_quant.backtest.costs import cost_model_from_config
from nexus_quant.data.providers.registry import make_provider
from nexus_quant.strategies.registry import make_strategy

# Timeout
_partial = {}
def _on_timeout(signum, frame):
    print("\n⏰ TIMEOUT — saving partial...")
    _save(_partial, partial=True)
    sys.exit(0)
_signal.signal(_signal.SIGALRM, _on_timeout)
_signal.alarm(600)

PROD_CFG = json.loads((ROOT / "configs" / "production_p91b_champion.json").read_text())
SYMBOLS = PROD_CFG["data"]["symbols"]
ENSEMBLE_WEIGHTS = PROD_CFG["ensemble"]["weights"]
SIGNAL_DEFS = PROD_CFG["ensemble"]["signals"]
SIG_KEYS = list(SIGNAL_DEFS.keys())

YEAR_RANGES = {
    "2021": ("2021-02-01", "2021-12-31"),
    "2022": ("2022-01-01", "2022-12-31"),
    "2023": ("2023-01-01", "2023-12-31"),
    "2024": ("2024-01-01", "2024-12-31"),
    "2025": ("2025-01-01", "2025-12-31"),
}
YEARS = sorted(YEAR_RANGES.keys())

OUT_DIR = ROOT / "artifacts" / "phase125"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def sharpe(rets: np.ndarray) -> float:
    if len(rets) < 100:
        return 0.0
    mu = float(np.mean(rets)) * 8760
    sd = float(np.std(rets)) * np.sqrt(8760)
    return round(mu / sd, 4) if sd > 1e-12 else 0.0


def _save(data: dict, partial: bool = False) -> None:
    data["partial"] = partial
    data["timestamp"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    path = OUT_DIR / "wf_weight_stability_report.json"
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"  → Saved: {path}")


def run_signal_backtest(sig_key: str, start: str, end: str) -> np.ndarray:
    """Run one sub-signal, return hourly returns."""
    sig_def = SIGNAL_DEFS[sig_key]
    cfg = {
        "data": {
            "provider": "binance_rest_v1", "symbols": SYMBOLS,
            "start": start, "end": end, "bar_interval": "1h",
            "cache_dir": ".cache/binance_rest",
        },
        "strategy": {"name": sig_def["strategy"], "params": sig_def["params"]},
        "costs": {"fee_rate": 0.0005, "slippage_rate": 0.0003, "cost_multiplier": 1.0},
    }
    provider = make_provider(cfg["data"], seed=42)
    dataset = provider.load()
    cost_model = cost_model_from_config(cfg["costs"])
    bt_cfg = BacktestConfig(costs=cost_model)
    strat = make_strategy(cfg["strategy"])
    engine = BacktestEngine(bt_cfg)
    result = engine.run(dataset, strat)
    return np.array(result.returns, dtype=np.float64)


def blend_returns(weights: dict, sig_returns: dict, year: str) -> np.ndarray:
    """Blend signal returns with given weights for a specific year."""
    min_len = min(len(sig_returns[sk][year]) for sk in SIG_KEYS
                  if len(sig_returns.get(sk, {}).get(year, [])) > 0)
    ens = np.zeros(int(min_len))
    for sk in SIG_KEYS:
        ens += weights[sk] * sig_returns[sk][year][:int(min_len)]
    return ens


def optimize_weights_on_years(sig_returns: dict, train_years: list) -> dict:
    """Grid search for best weights on training years (5% steps)."""
    best_obj = -999
    best_w = None

    for w1 in range(5, 56, 5):
        for w2 in range(5, 56 - w1, 5):
            for w3 in range(5, 56 - w1 - w2, 5):
                w4 = 100 - w1 - w2 - w3
                if w4 < 5 or w4 > 55:
                    continue
                weights = {
                    SIG_KEYS[0]: w1 / 100,
                    SIG_KEYS[1]: w2 / 100,
                    SIG_KEYS[2]: w3 / 100,
                    SIG_KEYS[3]: w4 / 100,
                }
                year_sharpes = []
                for y in train_years:
                    try:
                        ens = blend_returns(weights, sig_returns, y)
                        year_sharpes.append(sharpe(ens))
                    except Exception:
                        year_sharpes.append(0.0)

                arr = np.array(year_sharpes)
                obj = float(np.mean(arr) - 0.5 * np.std(arr))
                if obj > best_obj:
                    best_obj = obj
                    best_w = weights

    return best_w or {sk: 0.25 for sk in SIG_KEYS}


def main():
    global _partial
    t0 = time.time()
    print("=" * 70)
    print("Phase 125: Walk-Forward Weight Stability + LOYO Validation")
    print("=" * 70)

    # 1. Cache all signal returns
    print("\n[1/3] Computing per-signal returns...")
    sig_returns = {}
    for sk in SIG_KEYS:
        sig_returns[sk] = {}
        print(f"  {sk}:", end="", flush=True)
        for year, (start, end) in YEAR_RANGES.items():
            try:
                rets = run_signal_backtest(sk, start, end)
                sig_returns[sk][year] = rets
                print(f" {year}={sharpe(rets):.2f}", end="", flush=True)
            except Exception as e:
                sig_returns[sk][year] = np.array([])
                print(f" {year}=ERR", end="", flush=True)
        print()

    _partial = {"phase": 125, "description": "WF Weight Stability"}

    # 2. Leave-One-Year-Out
    print("\n[2/3] Leave-One-Year-Out cross-validation...")
    loyo_results = []
    optimal_weights_per_fold = {}

    for test_year in YEARS:
        train_years = [y for y in YEARS if y != test_year]
        print(f"\n  Test={test_year}, Train={train_years}")

        # Optimize on training years
        opt_w = optimize_weights_on_years(sig_returns, train_years)
        optimal_weights_per_fold[test_year] = {k: round(v, 4) for k, v in opt_w.items()}

        # Evaluate on test year
        try:
            # Champion weights on test year
            champ_ens = blend_returns(ENSEMBLE_WEIGHTS, sig_returns, test_year)
            champ_sharpe = sharpe(champ_ens)

            # Optimized weights on test year
            opt_ens = blend_returns(opt_w, sig_returns, test_year)
            opt_sharpe = sharpe(opt_ens)

            delta = opt_sharpe - champ_sharpe
            loyo_results.append({
                "test_year": test_year,
                "train_years": train_years,
                "champion_sharpe": champ_sharpe,
                "optimized_sharpe": opt_sharpe,
                "delta": round(delta, 4),
                "optimal_weights": optimal_weights_per_fold[test_year],
            })
            win = "✓" if delta >= 0 else "✗"
            print(f"    Champion={champ_sharpe:.3f}  Optimized={opt_sharpe:.3f}  Δ={delta:+.3f}  {win}")
            print(f"    Optimal: {optimal_weights_per_fold[test_year]}")
        except Exception as e:
            print(f"    ERROR: {e}")
            loyo_results.append({
                "test_year": test_year,
                "error": str(e),
            })

    # 3. Weight stability analysis
    print("\n[3/3] Weight stability analysis...")

    # How stable are the optimal weights across folds?
    weight_vectors = {sk: [] for sk in SIG_KEYS}
    for fold_w in optimal_weights_per_fold.values():
        for sk in SIG_KEYS:
            weight_vectors[sk].append(fold_w.get(sk, 0.25))

    stability = {}
    for sk in SIG_KEYS:
        vals = np.array(weight_vectors[sk])
        stability[sk] = {
            "mean": round(float(np.mean(vals)), 4),
            "std": round(float(np.std(vals)), 4),
            "min": round(float(np.min(vals)), 4),
            "max": round(float(np.max(vals)), 4),
            "cv": round(float(np.std(vals) / np.mean(vals) * 100), 1) if np.mean(vals) > 0 else 0,
            "current": ENSEMBLE_WEIGHTS[sk],
        }
        s = stability[sk]
        stable = "STABLE" if s["cv"] < 30 else "MODERATE" if s["cv"] < 50 else "UNSTABLE"
        print(f"  {sk:12s}: mean={s['mean']:.3f} ± {s['std']:.3f}  range=[{s['min']:.2f}, {s['max']:.2f}]  CV={s['cv']:.0f}%  {stable}")

    # Summary
    champ_sharpes = [r["champion_sharpe"] for r in loyo_results if "champion_sharpe" in r]
    opt_sharpes = [r["optimized_sharpe"] for r in loyo_results if "optimized_sharpe" in r]
    deltas = [r["delta"] for r in loyo_results if "delta" in r]

    wins = sum(1 for d in deltas if d >= 0)
    avg_delta = float(np.mean(deltas)) if deltas else 0
    avg_champ = float(np.mean(champ_sharpes)) if champ_sharpes else 0
    avg_opt = float(np.mean(opt_sharpes)) if opt_sharpes else 0

    # Verdict
    if avg_delta < 0.05 and wins <= len(deltas) // 2:
        verdict = "CHAMPION ROBUST — optimized weights don't consistently beat OOS"
    elif avg_delta < 0.1:
        verdict = "MARGINAL — small OOS improvement, not worth the complexity"
    else:
        verdict = "REWEIGHT SUGGESTED — consistent OOS improvement"

    avg_cv = float(np.mean([s["cv"] for s in stability.values()]))
    if avg_cv > 40:
        verdict += " (BUT weights unstable — high CV)"

    elapsed = time.time() - t0
    _partial = {
        "phase": 125,
        "description": "Walk-Forward Weight Stability + LOYO Validation",
        "elapsed_seconds": round(elapsed, 1),
        "loyo_results": loyo_results,
        "optimal_weights_per_fold": optimal_weights_per_fold,
        "weight_stability": stability,
        "summary": {
            "avg_champion_sharpe": round(avg_champ, 4),
            "avg_optimized_sharpe": round(avg_opt, 4),
            "avg_delta": round(avg_delta, 4),
            "wins_out_of_folds": f"{wins}/{len(deltas)}",
            "avg_weight_cv": round(avg_cv, 1),
            "verdict": verdict,
        },
    }
    _save(_partial, partial=False)

    print(f"\n{'='*70}")
    print(f"  LOYO Summary:")
    print(f"    Champion avg Sharpe (OOS): {avg_champ:.3f}")
    print(f"    Optimized avg Sharpe (OOS): {avg_opt:.3f}")
    print(f"    Avg delta: {avg_delta:+.3f}")
    print(f"    Wins: {wins}/{len(deltas)} folds")
    print(f"    Avg weight CV: {avg_cv:.0f}%")
    print(f"\n  VERDICT: {verdict}")
    print(f"\nPhase 125 complete in {elapsed:.0f}s")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
