#!/usr/bin/env python3
"""
Phase 124: Adaptive Ensemble Weight Optimizer
================================================
Tests whether reweighting the P91b ensemble based on rolling signal
performance improves OBJ = AVG - 0.5*std (conservative risk-adjusted metric).

Approach:
1. Load cached signal returns from Phase 123 (or recompute)
2. Test N weight combinations: original, suggested, inverse-var, equal, grid
3. For each: compute blended returns → yearly Sharpe → OBJ
4. Report best weights + improvement vs champion

Key constraint: weights must sum to 1.0, each >= 5%
"""
import json
import math
import os
import signal as _signal
import sys
import time
from itertools import product
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

# Config
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

OUT_DIR = ROOT / "artifacts" / "phase124"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def sharpe(rets: np.ndarray) -> float:
    if len(rets) < 100:
        return 0.0
    mu = float(np.mean(rets)) * 8760
    sd = float(np.std(rets)) * np.sqrt(8760)
    return round(mu / sd, 4) if sd > 1e-12 else 0.0


def obj_func(yearly_sharpes: list) -> float:
    """OBJ = AVG - 0.5*std (penalize variance across years)."""
    if not yearly_sharpes:
        return 0.0
    arr = np.array(yearly_sharpes)
    return round(float(np.mean(arr) - 0.5 * np.std(arr)), 4)


def _save(data: dict, partial: bool = False) -> None:
    data["partial"] = partial
    data["timestamp"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    path = OUT_DIR / "adaptive_weights_report.json"
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"  → Saved: {path}")


def run_signal_backtest(sig_key: str, start: str, end: str) -> dict:
    """Run one sub-signal, return result with returns."""
    sig_def = SIGNAL_DEFS[sig_key]
    cfg = {
        "data": {
            "provider": "binance_rest_v1",
            "symbols": SYMBOLS,
            "start": start, "end": end,
            "bar_interval": "1h",
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
    return {
        "returns": np.array(result.returns, dtype=np.float64),
        "sharpe": sharpe(np.array(result.returns)),
    }


def evaluate_weights(weights: dict, all_returns: dict) -> dict:
    """Evaluate a weight combination across all years."""
    yearly_sharpes = []
    for year in YEARS:
        # Get minimum length across signals for this year
        min_len = float("inf")
        for sk in SIG_KEYS:
            r = all_returns.get(sk, {}).get(year)
            if r is not None and len(r) > 0:
                min_len = min(min_len, len(r))

        if min_len < 100 or min_len == float("inf"):
            yearly_sharpes.append(0.0)
            continue

        # Blend
        ens = np.zeros(int(min_len))
        for sk in SIG_KEYS:
            r = all_returns.get(sk, {}).get(year)
            if r is not None and len(r) >= min_len:
                ens += weights[sk] * r[:int(min_len)]

        yearly_sharpes.append(sharpe(ens))

    return {
        "weights": {k: round(v, 4) for k, v in weights.items()},
        "yearly_sharpes": {y: s for y, s in zip(YEARS, yearly_sharpes)},
        "avg_sharpe": round(float(np.mean(yearly_sharpes)), 4),
        "min_sharpe": round(float(np.min(yearly_sharpes)), 4),
        "std_sharpe": round(float(np.std(yearly_sharpes)), 4),
        "obj": obj_func(yearly_sharpes),
    }


def main():
    global _partial
    t0 = time.time()
    print("=" * 70)
    print("Phase 124: Adaptive Ensemble Weight Optimizer")
    print("=" * 70)

    # 1. Cache all signal returns
    print("\n[1/3] Computing per-signal returns across all years...")
    all_returns = {}  # sig_key -> year -> np.ndarray

    for sk in SIG_KEYS:
        all_returns[sk] = {}
        print(f"  {sk}:", end="", flush=True)
        for year, (start, end) in YEAR_RANGES.items():
            try:
                result = run_signal_backtest(sk, start, end)
                all_returns[sk][year] = result["returns"]
                print(f"  {year}={result['sharpe']:.2f}", end="", flush=True)
            except Exception as e:
                print(f"  {year}=ERR", end="", flush=True)
                all_returns[sk][year] = np.array([])
        print()

    _partial = {"phase": 124, "description": "Adaptive Weight Optimizer"}

    # 2. Evaluate weight combinations
    print("\n[2/3] Testing weight combinations...")

    # Champion (current)
    champion = evaluate_weights(ENSEMBLE_WEIGHTS, all_returns)
    print(f"  CHAMPION: OBJ={champion['obj']:.4f}  AVG={champion['avg_sharpe']:.3f}  MIN={champion['min_sharpe']:.3f}")

    results = [{"label": "CHAMPION (current)", **champion}]

    # Equal weights
    eq_w = {sk: 0.25 for sk in SIG_KEYS}
    eq = evaluate_weights(eq_w, all_returns)
    results.append({"label": "EQUAL (25/25/25/25)", **eq})
    print(f"  EQUAL:    OBJ={eq['obj']:.4f}  AVG={eq['avg_sharpe']:.3f}  MIN={eq['min_sharpe']:.3f}")

    # Phase 123 suggestion (recent-performance-weighted)
    health_path = ROOT / "artifacts" / "phase123" / "signal_health_report.json"
    if health_path.exists():
        hd = json.loads(health_path.read_text())
        if "optimal_weights" in hd:
            p123_w = hd["optimal_weights"]
            p123 = evaluate_weights(p123_w, all_returns)
            results.append({"label": "P123_SUGGESTED", **p123})
            print(f"  P123_SUG: OBJ={p123['obj']:.4f}  AVG={p123['avg_sharpe']:.3f}  MIN={p123['min_sharpe']:.3f}")

    # Inverse-variance weighted (more weight to lower-variance signals)
    print("\n  Grid search (5% steps, 4 signals, sum=100%)...")
    grid_step = 5  # percentage points
    min_w = 5      # minimum 5% per signal
    max_w = 60     # maximum 60% per signal

    best_grid = None
    best_grid_obj = -999
    n_tested = 0

    # 4D grid with constraint sum=100
    for w1 in range(min_w, max_w + 1, grid_step):
        for w2 in range(min_w, max_w + 1 - w1, grid_step):
            for w3 in range(min_w, max_w + 1 - w1 - w2, grid_step):
                w4 = 100 - w1 - w2 - w3
                if w4 < min_w or w4 > max_w:
                    continue
                n_tested += 1
                weights = {
                    SIG_KEYS[0]: w1 / 100,
                    SIG_KEYS[1]: w2 / 100,
                    SIG_KEYS[2]: w3 / 100,
                    SIG_KEYS[3]: w4 / 100,
                }
                ev = evaluate_weights(weights, all_returns)
                if ev["obj"] > best_grid_obj:
                    best_grid_obj = ev["obj"]
                    best_grid = ev

    if best_grid:
        results.append({"label": "GRID_BEST", **best_grid})
        print(f"  GRID_BEST: OBJ={best_grid['obj']:.4f}  AVG={best_grid['avg_sharpe']:.3f}  MIN={best_grid['min_sharpe']:.3f}")
        print(f"    Weights: {best_grid['weights']}")
    print(f"    ({n_tested} combinations tested)")

    # Fine-tune around grid best (1% steps)
    if best_grid:
        print("\n  Fine-tuning around grid best (1% steps)...")
        base = best_grid["weights"]
        best_fine = best_grid
        best_fine_obj = best_grid_obj

        for dw1 in range(-4, 5):
            for dw2 in range(-4, 5):
                for dw3 in range(-4, 5):
                    dw4 = -(dw1 + dw2 + dw3)
                    ws = {
                        SIG_KEYS[0]: base[SIG_KEYS[0]] + dw1 / 100,
                        SIG_KEYS[1]: base[SIG_KEYS[1]] + dw2 / 100,
                        SIG_KEYS[2]: base[SIG_KEYS[2]] + dw3 / 100,
                        SIG_KEYS[3]: base[SIG_KEYS[3]] + dw4 / 100,
                    }
                    if any(v < 0.05 or v > 0.60 for v in ws.values()):
                        continue
                    if abs(sum(ws.values()) - 1.0) > 0.001:
                        continue
                    ev = evaluate_weights(ws, all_returns)
                    if ev["obj"] > best_fine_obj:
                        best_fine_obj = ev["obj"]
                        best_fine = ev

        results.append({"label": "FINE_TUNED", **best_fine})
        print(f"  FINE_TUNED: OBJ={best_fine['obj']:.4f}  AVG={best_fine['avg_sharpe']:.3f}  MIN={best_fine['min_sharpe']:.3f}")
        print(f"    Weights: {best_fine['weights']}")

    # 3. Summary
    print("\n[3/3] Summary...")
    results.sort(key=lambda r: r.get("obj", 0), reverse=True)

    print(f"\n  {'Label':20s} {'OBJ':>8s} {'AVG':>8s} {'MIN':>8s} {'STD':>8s}")
    print(f"  {'-'*20} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")
    for r in results:
        label = r.get("label", "?")[:20]
        print(f"  {label:20s} {r['obj']:8.4f} {r['avg_sharpe']:8.3f} {r['min_sharpe']:8.3f} {r['std_sharpe']:8.3f}")

    improvement = results[0]["obj"] - champion["obj"]
    best = results[0]

    elapsed = time.time() - t0
    _partial = {
        "phase": 124,
        "description": "Adaptive Ensemble Weight Optimizer",
        "elapsed_seconds": round(elapsed, 1),
        "champion": champion,
        "results": results,
        "best": best,
        "improvement_vs_champion": round(improvement, 4),
        "n_grid_tested": n_tested,
        "recommendation": "KEEP CHAMPION" if improvement < 0.05 else f"ADOPT {best.get('label', '?')} (+{improvement:.3f} OBJ)",
    }
    _save(_partial, partial=False)

    print(f"\n  Champion OBJ: {champion['obj']:.4f}")
    print(f"  Best OBJ:     {best['obj']:.4f} ({best.get('label', '?')})")
    print(f"  Improvement:  {improvement:+.4f}")

    if improvement < 0.05:
        print(f"\n  RECOMMENDATION: KEEP current champion weights (improvement < 0.05)")
        print(f"  Champion is already near the optimal plateau.")
    else:
        print(f"\n  RECOMMENDATION: Consider adopting {best.get('label', '?')}")
        print(f"  Weights: {best['weights']}")
        print(f"  ⚠ Caveat: optimized on IS data. Validate on 2026 OOS before production.")

    print(f"\n{'='*70}")
    print(f"Phase 124 complete in {elapsed:.0f}s")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
