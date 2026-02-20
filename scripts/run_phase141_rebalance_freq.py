#!/usr/bin/env python3
"""
Phase 141: Rebalance Frequency Optimization
=============================================
Current sub-strategy rebalance intervals:
  V1:   60h (rebalance_interval_bars=60)
  I460: 48h
  I415: 48h
  F144: 24h

Hypothesis: Trading less frequently reduces costs (fewer rebalances)
while preserving most of the alpha (slow signals don't need hourly updating).

Tests:
1. Scale ALL intervals by 1.5x, 2x, 3x
2. Scale ONLY F144 (fastest mover) to 48h, 72h
3. Scale V1 down (to 48h) — V1 is already slow, maybe it benefits from faster
4. Compute turnover reduction + cost savings estimate
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

OUT_DIR = ROOT / "artifacts" / "phase141"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Current rebalance intervals
BASE_INTERVALS = {
    "v1": 60, "i460bw168": 48, "i415bw216": 48, "f144": 24,
}

# Variants: name → {signal_key: new_interval}
VARIANTS = {
    "baseline": {},  # no changes
    "all_1.5x": {sk: int(BASE_INTERVALS[sk] * 1.5) for sk in SIG_KEYS},
    "all_2x": {sk: BASE_INTERVALS[sk] * 2 for sk in SIG_KEYS},
    "all_3x": {sk: BASE_INTERVALS[sk] * 3 for sk in SIG_KEYS},
    "f144_48h": {"f144": 48},
    "f144_72h": {"f144": 72},
    "f144_96h": {"f144": 96},
    "v1_48h": {"v1": 48},
    "v1_36h": {"v1": 36},
    "idio_72h": {"i460bw168": 72, "i415bw216": 72},
    "idio_96h": {"i460bw168": 96, "i415bw216": 96},
    "slow_mix": {"v1": 72, "i460bw168": 72, "i415bw216": 72, "f144": 48},
    "fast_mix": {"v1": 48, "i460bw168": 36, "i415bw216": 36, "f144": 12},
}


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
    path = OUT_DIR / "rebalance_freq_report.json"
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"  → Saved: {path}")


def estimate_turnover(result) -> float:
    """Estimate annualized turnover from backtest result."""
    # Use the trades/rebalances from the result
    if hasattr(result, 'weight_history') and result.weight_history:
        total_turnover = 0.0
        prev_w = None
        for w in result.weight_history:
            if prev_w is not None:
                for s in w:
                    total_turnover += abs(w.get(s, 0) - prev_w.get(s, 0))
            prev_w = w
        n_bars = len(result.returns)
        if n_bars > 0:
            ann_factor = 8760 / n_bars
            return total_turnover * ann_factor
    # Fallback: count number of rebalances
    return 0.0


def main():
    global _partial
    t0 = time.time()
    print("=" * 70)
    print("Phase 141: Rebalance Frequency Optimization")
    print("=" * 70)

    # Load data once
    print("\n[1/2] Loading data...")
    datasets = {}
    for year, (start, end) in YEAR_RANGES.items():
        print(f"  {year}:", end="", flush=True)
        cfg_data = {
            "provider": "binance_rest_v1", "symbols": SYMBOLS,
            "start": start, "end": end, "bar_interval": "1h",
            "cache_dir": ".cache/binance_rest",
        }
        provider = make_provider(cfg_data, seed=42)
        datasets[year] = provider.load()
        print(f" OK", flush=True)

    # Test each variant
    print("\n[2/2] Testing rebalance frequency variants...")
    variant_results = {}

    for vname, overrides in VARIANTS.items():
        yearly_sharpes = []
        yearly_detail = {}
        total_rebal_count = 0

        for year in YEARS:
            dataset = datasets[year]
            sig_rets = {}

            for sk in SIG_KEYS:
                sig_def = SIGNAL_DEFS[sk]
                params = dict(sig_def["params"])

                # Apply interval override
                if sk in overrides:
                    params["rebalance_interval_bars"] = overrides[sk]

                cost_model = cost_model_from_config({"fee_rate": 0.0005, "slippage_rate": 0.0003, "cost_multiplier": 1.0})
                bt_cfg = BacktestConfig(costs=cost_model)
                strat = make_strategy({"name": sig_def["strategy"], "params": params})
                engine = BacktestEngine(bt_cfg)
                result = engine.run(dataset, strat)
                sig_rets[sk] = np.array(result.returns, dtype=np.float64)

                # Estimate rebalance count
                n = len(dataset.timeline)
                interval = params.get("rebalance_interval_bars", 48)
                total_rebal_count += n // interval

            # Ensemble
            min_len = min(len(sig_rets[sk]) for sk in SIG_KEYS)
            ens = np.zeros(min_len)
            for sk in SIG_KEYS:
                ens += ENSEMBLE_WEIGHTS[sk] * sig_rets[sk][:min_len]

            s = sharpe(ens)
            yearly_sharpes.append(s)
            yearly_detail[year] = s

        avg = round(float(np.mean(yearly_sharpes)), 4)
        mn = round(float(np.min(yearly_sharpes)), 4)
        ob = obj_func(yearly_sharpes)

        # Compute effective intervals
        eff_intervals = {}
        for sk in SIG_KEYS:
            if sk in overrides:
                eff_intervals[sk] = overrides[sk]
            else:
                eff_intervals[sk] = BASE_INTERVALS[sk]

        variant_results[vname] = {
            "yearly": yearly_detail,
            "avg_sharpe": avg,
            "min_sharpe": mn,
            "obj": ob,
            "intervals": eff_intervals,
            "est_rebalances_per_year": total_rebal_count // len(YEARS),
        }

    baseline_obj = variant_results["baseline"]["obj"]
    baseline_rebal = variant_results["baseline"]["est_rebalances_per_year"]

    # Display results
    print(f"\n  {'Variant':16s} {'OBJ':>8s} {'AVG':>8s} {'MIN':>8s} {'ΔOBJ':>8s} {'Rebal/yr':>10s} {'ΔRebal':>8s}")
    print(f"  {'-'*16} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*10} {'-'*8}")
    for vname in sorted(variant_results, key=lambda k: variant_results[k]["obj"], reverse=True):
        r = variant_results[vname]
        delta = r["obj"] - baseline_obj
        rebal_delta = r["est_rebalances_per_year"] - baseline_rebal
        tag = " ✓" if delta > 0.02 else (" ★" if vname == "baseline" else "")
        print(f"  {vname:16s} {r['obj']:8.4f} {r['avg_sharpe']:8.3f} "
              f"{r['min_sharpe']:8.3f} {delta:+8.4f} {r['est_rebalances_per_year']:10d} {rebal_delta:+8d}{tag}")

    # Per-year detail for top variants
    print("\n  Per-year Sharpe for top variants:")
    top_variants = sorted(variant_results, key=lambda k: variant_results[k]["obj"], reverse=True)[:5]
    header = f"  {'Year':6s}" + "".join(f" {v:>12s}" for v in top_variants)
    print(header)
    for year in YEARS:
        row = f"  {year:6s}"
        for v in top_variants:
            s = variant_results[v]["yearly"][year]
            row += f" {s:12.3f}"
        print(row)

    # Cost saving analysis
    print("\n  Cost impact analysis:")
    base_fee = 0.0005 + 0.0003  # fee + slippage
    for vname in top_variants:
        r = variant_results[vname]
        rebal_ratio = r["est_rebalances_per_year"] / baseline_rebal if baseline_rebal > 0 else 1.0
        cost_saving = (1 - rebal_ratio) * 100
        print(f"    {vname:16s}: {rebal_ratio:.2f}x rebalances → ~{cost_saving:.0f}% cost reduction")

    # Find best that improves or matches baseline with fewer trades
    best_efficient = None
    for vname in sorted(variant_results, key=lambda k: variant_results[k]["obj"], reverse=True):
        r = variant_results[vname]
        if r["obj"] >= baseline_obj - 0.02 and r["est_rebalances_per_year"] < baseline_rebal:
            best_efficient = vname
            break

    if best_efficient:
        be = variant_results[best_efficient]
        verdict = (f"EFFICIENT — {best_efficient} saves "
                   f"{baseline_rebal - be['est_rebalances_per_year']} rebalances/yr "
                   f"with only {be['obj'] - baseline_obj:+.3f} OBJ change")
    else:
        best_overall = max(variant_results, key=lambda k: variant_results[k]["obj"])
        bo = variant_results[best_overall]
        if bo["obj"] > baseline_obj + 0.02:
            verdict = f"IMPROVED — {best_overall} adds +{bo['obj'] - baseline_obj:.3f} OBJ"
        else:
            verdict = "OPTIMAL — current rebalance frequencies are already optimal"

    elapsed = time.time() - t0
    _partial = {
        "phase": 141,
        "description": "Rebalance Frequency Optimization",
        "elapsed_seconds": round(elapsed, 1),
        "baseline_intervals": BASE_INTERVALS,
        "variant_results": variant_results,
        "best_efficient": best_efficient,
        "verdict": verdict,
    }
    _save(_partial, partial=False)

    print(f"\n  VERDICT: {verdict}")
    print(f"\nPhase 141 complete in {elapsed:.0f}s")
    print("=" * 70)


if __name__ == "__main__":
    main()
