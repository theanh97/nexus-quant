#!/usr/bin/env python3
"""
Phase 140: Universe Expansion — 10 → 15 → 20 Symbols
======================================================
The P91b ensemble is optimized for 10 symbols. More symbols = more
cross-sectional dispersion = potentially better ranking-based alpha.

Tests:
1. Can we fetch data for 15/20 symbols across 2021-2025?
2. Run the 4-signal ensemble on 15 and 20 symbols
3. Compare Sharpe per year vs 10-symbol baseline
4. Test whether more symbols improve or hurt each sub-strategy

Candidate additional symbols (all USDM perps listed before 2021):
  Tier 2 (5 more): LTCUSDT, UNIUSDT, MATICUSDT, FILUSDT, AAVEUSDT
  Tier 3 (5 more): ATOMUSDT, NEARUSDT, ICPUSDT, ALGOUSDT, FTMUSDT
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

_partial = {}
def _on_timeout(signum, frame):
    print("\n⏰ TIMEOUT — saving partial...")
    _save(_partial, partial=True)
    sys.exit(0)
_signal.signal(_signal.SIGALRM, _on_timeout)
_signal.alarm(900)

PROD_CFG = json.loads((ROOT / "configs" / "production_p91b_champion.json").read_text())
SYMBOLS_10 = PROD_CFG["data"]["symbols"]
ENSEMBLE_WEIGHTS = PROD_CFG["ensemble"]["weights"]
SIGNAL_DEFS = PROD_CFG["ensemble"]["signals"]
SIG_KEYS = list(SIGNAL_DEFS.keys())

# Extended universes
TIER2_ADD = ["LTCUSDT", "UNIUSDT", "MATICUSDT", "FILUSDT", "AAVEUSDT"]
TIER3_ADD = ["ATOMUSDT", "NEARUSDT", "ICPUSDT", "ALGOUSDT", "FTMUSDT"]

SYMBOLS_15 = SYMBOLS_10 + TIER2_ADD
SYMBOLS_20 = SYMBOLS_15 + TIER3_ADD

YEAR_RANGES = {
    "2021": ("2021-02-01", "2021-12-31"),
    "2022": ("2022-01-01", "2022-12-31"),
    "2023": ("2023-01-01", "2023-12-31"),
    "2024": ("2024-01-01", "2024-12-31"),
    "2025": ("2025-01-01", "2025-12-31"),
}
YEARS = sorted(YEAR_RANGES.keys())

OUT_DIR = ROOT / "artifacts" / "phase140"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def sharpe(rets: np.ndarray) -> float:
    if len(rets) < 100:
        return 0.0
    mu = float(np.mean(rets)) * 8760
    sd = float(np.std(rets)) * np.sqrt(8760)
    return round(mu / sd, 4) if sd > 1e-12 else 0.0


def obj_func(yearly_sharpes: list) -> float:
    arr = np.array(yearly_sharpes)
    return round(float(np.mean(arr) - 0.5 * np.std(arr)), 4)


def _save(data: dict, partial: bool = False) -> None:
    data["partial"] = partial
    data["timestamp"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    path = OUT_DIR / "universe_expansion_report.json"
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"  → Saved: {path}")


def run_ensemble_for_universe(symbols: list, universe_name: str) -> dict:
    """Run full P91b 4-signal ensemble on the given symbol universe."""
    results = {}
    data_issues = []

    for year, (start, end) in YEAR_RANGES.items():
        print(f"  {universe_name} {year}:", end="", flush=True)

        try:
            cfg_data = {
                "provider": "binance_rest_v1", "symbols": symbols,
                "start": start, "end": end, "bar_interval": "1h",
                "cache_dir": ".cache/binance_rest",
            }
            provider = make_provider(cfg_data, seed=42)
            dataset = provider.load()
        except Exception as e:
            print(f" DATA ERROR: {e}", flush=True)
            data_issues.append({"year": year, "error": str(e)})
            continue

        # Run each sub-strategy
        sig_rets = {}
        for sk in SIG_KEYS:
            sig_def = SIGNAL_DEFS[sk]
            cost_model = cost_model_from_config({"fee_rate": 0.0005, "slippage_rate": 0.0003, "cost_multiplier": 1.0})
            bt_cfg = BacktestConfig(costs=cost_model)
            strat = make_strategy({"name": sig_def["strategy"], "params": sig_def["params"]})
            engine = BacktestEngine(bt_cfg)
            result = engine.run(dataset, strat)
            sig_rets[sk] = np.array(result.returns, dtype=np.float64)

        # Ensemble
        min_len = min(len(sig_rets[sk]) for sk in SIG_KEYS)
        ens = np.zeros(min_len)
        for sk in SIG_KEYS:
            ens += ENSEMBLE_WEIGHTS[sk] * sig_rets[sk][:min_len]

        s = sharpe(ens)
        results[year] = {
            "ensemble_sharpe": s,
            "sub_sharpes": {sk: sharpe(sig_rets[sk]) for sk in SIG_KEYS},
            "n_bars": min_len,
            "n_symbols": len(symbols),
        }
        print(f" Sharpe={s:.3f} (bars={min_len})", flush=True)

    yearly_sharpes = [results[y]["ensemble_sharpe"] for y in YEARS if y in results]
    avg = round(float(np.mean(yearly_sharpes)), 4) if yearly_sharpes else 0.0
    mn = round(float(np.min(yearly_sharpes)), 4) if yearly_sharpes else 0.0
    ob = obj_func(yearly_sharpes) if len(yearly_sharpes) >= 3 else 0.0

    return {
        "universe": universe_name,
        "symbols": symbols,
        "n_symbols": len(symbols),
        "yearly": results,
        "avg_sharpe": avg,
        "min_sharpe": mn,
        "obj": ob,
        "data_issues": data_issues,
    }


def main():
    global _partial
    t0 = time.time()
    print("=" * 70)
    print("Phase 140: Universe Expansion — 10 → 15 → 20 Symbols")
    print("=" * 70)

    # Test each universe size
    all_results = {}

    print("\n[1/3] Testing 10-symbol baseline...")
    all_results["10sym"] = run_ensemble_for_universe(SYMBOLS_10, "10sym")

    print("\n[2/3] Testing 15-symbol universe (+LTC, UNI, MATIC, FIL, AAVE)...")
    all_results["15sym"] = run_ensemble_for_universe(SYMBOLS_15, "15sym")

    print("\n[3/3] Testing 20-symbol universe (+ATOM, NEAR, ICP, ALGO, FTM)...")
    all_results["20sym"] = run_ensemble_for_universe(SYMBOLS_20, "20sym")

    # Summary comparison
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    base_obj = all_results["10sym"]["obj"]

    print(f"\n  {'Universe':10s} {'OBJ':>8s} {'AVG':>8s} {'MIN':>8s} {'ΔOBJ':>8s}")
    print(f"  {'-'*10} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")
    for uname in ["10sym", "15sym", "20sym"]:
        r = all_results[uname]
        delta = r["obj"] - base_obj
        tag = " ✓" if delta > 0.03 else ""
        print(f"  {uname:10s} {r['obj']:8.4f} {r['avg_sharpe']:8.3f} "
              f"{r['min_sharpe']:8.3f} {delta:+8.4f}{tag}")

    # Per-year comparison
    print(f"\n  {'Year':6s} {'10sym':>8s} {'15sym':>8s} {'20sym':>8s} {'Δ15':>8s} {'Δ20':>8s}")
    print(f"  {'-'*6} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")
    for year in YEARS:
        s10 = all_results["10sym"]["yearly"].get(year, {}).get("ensemble_sharpe", 0)
        s15 = all_results["15sym"]["yearly"].get(year, {}).get("ensemble_sharpe", 0)
        s20 = all_results["20sym"]["yearly"].get(year, {}).get("ensemble_sharpe", 0)
        print(f"  {year:6s} {s10:8.3f} {s15:8.3f} {s20:8.3f} {s15-s10:+8.3f} {s20-s10:+8.3f}")

    # Per sub-strategy comparison
    print("\n  Sub-strategy Sharpe comparison (AVG across years):")
    for sk in SIG_KEYS:
        for uname in ["10sym", "15sym", "20sym"]:
            ys = []
            for year in YEARS:
                yr_data = all_results[uname]["yearly"].get(year, {})
                sub_s = yr_data.get("sub_sharpes", {}).get(sk, 0)
                ys.append(sub_s)
            avg = float(np.mean(ys))
            print(f"    {sk:12s} {uname:6s}: AVG={avg:+.3f}")

    # Determine verdict
    best_univ = max(all_results, key=lambda k: all_results[k]["obj"])
    best_obj = all_results[best_univ]["obj"]
    delta = best_obj - base_obj

    if delta > 0.05:
        verdict = f"PROMISING — {best_univ} adds +{delta:.3f} OBJ, needs LOYO validation"
    elif delta > 0.02:
        verdict = f"MARGINAL — {best_univ} adds +{delta:.3f} OBJ"
    elif delta > -0.02:
        verdict = "NEUTRAL — universe expansion has negligible effect"
    else:
        verdict = f"NEGATIVE — expansion HURTS by {abs(delta):.3f} OBJ"

    elapsed = time.time() - t0
    _partial = {
        "phase": 140,
        "description": "Universe Expansion — 10 → 15 → 20 Symbols",
        "elapsed_seconds": round(elapsed, 1),
        "results": all_results,
        "verdict": verdict,
    }
    _save(_partial, partial=False)

    print(f"\n  VERDICT: {verdict}")
    print(f"\nPhase 140 complete in {elapsed:.0f}s")
    print("=" * 70)


if __name__ == "__main__":
    main()
