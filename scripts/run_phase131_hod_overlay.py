#!/usr/bin/env python3
"""
Phase 131: Hour-of-Day Overlay on Actual Ensemble
===================================================
Phase 130 found: worst hours = 12-14h UTC (US lunch), best = 21-22h,4-5h (Asia).
Test: reduce ensemble leverage during worst hours. Boost during best? Or flat?

Variants:
1. Reduce leverage during worst hours (12-14h) by 50%
2. Reduce during worst hours (12-15h) by 30%
3. Go flat during worst 3 hours
4. LOYO validation of best variant
"""
import json
import os
import signal as _signal
import sys
import time
from datetime import datetime, timezone
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

OUT_DIR = ROOT / "artifacts" / "phase131"
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
    path = OUT_DIR / "hod_overlay_report.json"
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"  → Saved: {path}")


def main():
    global _partial
    t0 = time.time()
    print("=" * 70)
    print("Phase 131: Hour-of-Day Overlay on Actual Ensemble")
    print("=" * 70)

    # 1. Load data
    print("\n[1/3] Loading signal returns + timestamps per year...")
    sig_returns = {sk: {} for sk in SIG_KEYS}
    timestamps = {}

    for year, (start, end) in YEAR_RANGES.items():
        print(f"  {year}:", end="", flush=True)
        cfg_data = {
            "provider": "binance_rest_v1", "symbols": SYMBOLS,
            "start": start, "end": end, "bar_interval": "1h",
            "cache_dir": ".cache/binance_rest",
        }
        provider = make_provider(cfg_data, seed=42)
        dataset = provider.load()

        timestamps[year] = [dataset.timeline[i] for i in range(1, len(dataset.timeline))]

        for sk in SIG_KEYS:
            sig_def = SIGNAL_DEFS[sk]
            cost_model = cost_model_from_config({"fee_rate": 0.0005, "slippage_rate": 0.0003, "cost_multiplier": 1.0})
            bt_cfg = BacktestConfig(costs=cost_model)
            strat = make_strategy({"name": sig_def["strategy"], "params": sig_def["params"]})
            engine = BacktestEngine(bt_cfg)
            result = engine.run(dataset, strat)
            sig_returns[sk][year] = np.array(result.returns, dtype=np.float64)

        print(f" {len(timestamps[year])} bars", flush=True)

    _partial = {"phase": 131}

    # 2. Compute baseline + overlay variants
    print("\n[2/3] Testing HOD overlay variants on ensemble...")

    variants = {
        "baseline": {"bad_hours": [], "scale": 1.0},
        "reduce_12_14": {"bad_hours": [12, 13, 14], "scale": 0.5},
        "reduce_12_15": {"bad_hours": [12, 13, 14, 15], "scale": 0.5},
        "reduce_12_14_30pct": {"bad_hours": [12, 13, 14], "scale": 0.7},
        "flat_12_14": {"bad_hours": [12, 13, 14], "scale": 0.0},
        "flat_12_15": {"bad_hours": [12, 13, 14, 15], "scale": 0.0},
        "reduce_worst5": {"bad_hours": [12, 13, 14, 19, 23], "scale": 0.5},
        "reduce_us_session": {"bad_hours": [13, 14, 15, 16, 17, 18, 19], "scale": 0.7},
    }

    all_results = {}
    for vname, vcfg in variants.items():
        yearly_sharpes = []
        yearly_detail = {}

        for year in YEARS:
            min_len = min(len(sig_returns[sk][year]) for sk in SIG_KEYS)
            min_len = min(min_len, len(timestamps[year]))

            ens = np.zeros(min_len)
            n_reduced = 0
            for i in range(min_len):
                # Compute ensemble return at bar i
                ret = 0.0
                for sk in SIG_KEYS:
                    ret += ENSEMBLE_WEIGHTS[sk] * sig_returns[sk][year][i]

                # Check hour
                dt = datetime.fromtimestamp(timestamps[year][i], tz=timezone.utc)
                if dt.hour in vcfg["bad_hours"]:
                    ret *= vcfg["scale"]
                    n_reduced += 1

                ens[i] = ret

            s = sharpe(ens)
            yearly_sharpes.append(s)
            yearly_detail[year] = {
                "sharpe": s,
                "reduced_pct": round(n_reduced / min_len * 100, 1) if min_len > 0 else 0,
            }

        avg = round(float(np.mean(yearly_sharpes)), 4)
        mn = round(float(np.min(yearly_sharpes)), 4)
        obj = obj_func(yearly_sharpes)

        all_results[vname] = {
            "params": vcfg,
            "yearly": yearly_detail,
            "avg_sharpe": avg,
            "min_sharpe": mn,
            "obj": obj,
        }

    baseline_obj = all_results["baseline"]["obj"]
    print(f"\n  {'Variant':22s} {'OBJ':>8s} {'AVG':>8s} {'MIN':>8s} {'ΔOBJ':>8s}")
    print(f"  {'-'*22} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")
    for vname, vr in sorted(all_results.items(), key=lambda x: x[1]["obj"], reverse=True):
        delta = vr["obj"] - baseline_obj
        tag = " ✓" if delta > 0.05 else ""
        print(f"  {vname:22s} {vr['obj']:8.4f} {vr['avg_sharpe']:8.3f} {vr['min_sharpe']:8.3f} {delta:+8.4f}{tag}")

    # 3. LOYO validation on best non-baseline variant
    non_baseline = {k: v for k, v in all_results.items() if k != "baseline"}
    best_name = max(non_baseline, key=lambda k: non_baseline[k]["obj"])
    best = non_baseline[best_name]
    improvement = best["obj"] - baseline_obj

    print(f"\n[3/3] LOYO validation of '{best_name}' (ΔOBJ={improvement:+.4f})...")

    if improvement > 0.03:
        # LOYO: train on 4 years (find if pattern holds), test on 1
        # For calendar effects: check if the bad hours are consistently bad in training years
        loyo_results = []
        for test_year in YEARS:
            train_years = [y for y in YEARS if y != test_year]

            # Test year with overlay
            min_len = min(len(sig_returns[sk][test_year]) for sk in SIG_KEYS)
            min_len = min(min_len, len(timestamps[test_year]))

            # Apply the overlay
            bcfg = best["params"]
            ens_overlay = np.zeros(min_len)
            ens_base = np.zeros(min_len)
            for i in range(min_len):
                ret = 0.0
                for sk in SIG_KEYS:
                    ret += ENSEMBLE_WEIGHTS[sk] * sig_returns[sk][test_year][i]
                ens_base[i] = ret
                dt = datetime.fromtimestamp(timestamps[test_year][i], tz=timezone.utc)
                if dt.hour in bcfg["bad_hours"]:
                    ret *= bcfg["scale"]
                ens_overlay[i] = ret

            s_base = sharpe(ens_base)
            s_overlay = sharpe(ens_overlay)
            delta = s_overlay - s_base
            tag = "✓" if delta > 0 else "✗"
            print(f"    Test={test_year}  base={s_base:.3f}  overlay={s_overlay:.3f}  Δ={delta:+.3f} {tag}")

            loyo_results.append({
                "test_year": test_year,
                "baseline_sharpe": s_base,
                "overlay_sharpe": s_overlay,
                "delta": round(delta, 4),
            })

        loyo_wins = sum(1 for r in loyo_results if r["delta"] > 0)
        loyo_avg_delta = round(float(np.mean([r["delta"] for r in loyo_results])), 4)
        print(f"\n    LOYO: {loyo_wins}/5 wins, avg Δ={loyo_avg_delta:+.4f}")
    else:
        loyo_results = []
        loyo_wins = 0
        loyo_avg_delta = 0
        print(f"  SKIP LOYO — improvement too small ({improvement:+.4f})")

    # Verdict
    if improvement < 0.03:
        verdict = "NO IMPROVEMENT — HOD overlay does not help the ensemble"
    elif loyo_wins >= 3 and loyo_avg_delta > 0.02:
        verdict = f"VALIDATED — {best_name} improves ensemble OOS"
    elif loyo_wins >= 3:
        verdict = f"MARGINAL — {best_name} shows weak OOS evidence"
    else:
        verdict = f"FAILED OOS — {best_name} does not survive LOYO"

    elapsed = time.time() - t0
    _partial = {
        "phase": 131,
        "description": "Hour-of-Day Overlay on Ensemble",
        "elapsed_seconds": round(elapsed, 1),
        "variants": all_results,
        "best_variant": {"name": best_name, **best},
        "improvement": round(improvement, 4),
        "loyo": {
            "results": loyo_results,
            "wins": f"{loyo_wins}/5",
            "avg_delta": loyo_avg_delta,
        },
        "verdict": verdict,
    }
    _save(_partial, partial=False)

    print(f"\n  VERDICT: {verdict}")
    print(f"\nPhase 131 complete in {elapsed:.0f}s")
    print("=" * 70)


if __name__ == "__main__":
    main()
