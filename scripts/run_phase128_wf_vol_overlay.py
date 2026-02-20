#!/usr/bin/env python3
"""
Phase 128: Walk-Forward Validation of Combined Vol Overlay
===========================================================
Phase 127 found: combined_85pct overlay (reduce leverage 50% + tilt F144 +20%
when rolling vol > 85th percentile) improves OBJ from 1.57 → 1.85.

BUT: the 85th percentile threshold was computed on ALL years simultaneously.
This is look-ahead bias for threshold calibration.

Walk-forward tests:
1. LOYO (Leave-One-Year-Out): calibrate threshold on 4 years, test on 1
2. Expanding window: calibrate on years before, test on next year
3. Fixed threshold: use absolute vol levels instead of percentiles (no calibration needed)

If LOYO and expanding window both show improvement, the overlay is real.
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
_signal.alarm(900)  # 15 min

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

# Phase 127 best params
VOL_WINDOW = 168
THRESHOLD_PCT = 85
SCALE_FACTOR = 0.5
F144_BOOST = 0.20

OUT_DIR = ROOT / "artifacts" / "phase128"
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
    path = OUT_DIR / "wf_vol_overlay_report.json"
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"  → Saved: {path}")


def compute_rolling_vol(rets: np.ndarray, window: int) -> np.ndarray:
    """Compute rolling annualized volatility (CAUSAL — look-back only)."""
    n = len(rets)
    vol = np.full(n, np.nan)
    for i in range(window, n):
        vol[i] = float(np.std(rets[i - window:i])) * np.sqrt(8760)
    vol[:window] = vol[window] if window < n else 0.0
    return vol


def apply_combined_overlay(sig_rets: dict, rvol: np.ndarray,
                           vol_threshold: float, year: str) -> float:
    """Apply combined overlay on a single year using a pre-computed vol threshold.
    Returns Sharpe for that year."""
    min_len = min(len(sig_rets[sk]) for sk in SIG_KEYS if len(sig_rets[sk]) > 0)
    if min_len < 200:
        return 0.0

    rvol_trimmed = rvol[:int(min_len)]
    ens = np.zeros(int(min_len))

    n_high = 0
    for i in range(int(min_len)):
        if np.isnan(rvol_trimmed[i]) or rvol_trimmed[i] <= vol_threshold:
            # Normal regime — static weights
            for sk in SIG_KEYS:
                ens[i] += ENSEMBLE_WEIGHTS[sk] * sig_rets[sk][i]
        else:
            # HIGH VOL — reduce leverage + tilt F144
            n_high += 1
            boosted = dict(ENSEMBLE_WEIGHTS)
            boost_from_each = F144_BOOST / 3
            for sk in SIG_KEYS:
                if sk == "f144":
                    boosted[sk] = min(0.60, boosted[sk] + F144_BOOST)
                else:
                    boosted[sk] = max(0.05, boosted[sk] - boost_from_each)
            total = sum(boosted.values())
            for sk in SIG_KEYS:
                ens[i] += (boosted[sk] / total) * sig_rets[sk][i] * SCALE_FACTOR

    return sharpe(ens), n_high, min_len


def main():
    global _partial
    t0 = time.time()
    print("=" * 70)
    print("Phase 128: Walk-Forward Validation of Combined Vol Overlay")
    print("=" * 70)

    # 1. Load all data
    print("\n[1/4] Loading signal & BTC returns per year...")
    sig_returns = {sk: {} for sk in SIG_KEYS}
    btc_returns = {}

    for year, (start, end) in YEAR_RANGES.items():
        print(f"  {year}:", end="", flush=True)
        cfg_data = {
            "provider": "binance_rest_v1", "symbols": SYMBOLS,
            "start": start, "end": end, "bar_interval": "1h",
            "cache_dir": ".cache/binance_rest",
        }
        provider = make_provider(cfg_data, seed=42)
        dataset = provider.load()

        # BTC returns
        btc = []
        for i in range(1, len(dataset.timeline)):
            c0 = dataset.close("BTCUSDT", i - 1)
            c1 = dataset.close("BTCUSDT", i)
            btc.append((c1 / c0 - 1.0) if c0 > 0 else 0.0)
        btc_returns[year] = np.array(btc, dtype=np.float64)

        # Signal returns
        for sk in SIG_KEYS:
            sig_def = SIGNAL_DEFS[sk]
            cost_model = cost_model_from_config({"fee_rate": 0.0005, "slippage_rate": 0.0003, "cost_multiplier": 1.0})
            bt_cfg = BacktestConfig(costs=cost_model)
            strat = make_strategy({"name": sig_def["strategy"], "params": sig_def["params"]})
            engine = BacktestEngine(bt_cfg)
            result = engine.run(dataset, strat)
            sig_returns[sk][year] = np.array(result.returns, dtype=np.float64)

        print(f" {len(btc)} bars", flush=True)

    # Compute rolling vol per year
    rvol_per_year = {}
    for year in YEARS:
        rvol_per_year[year] = compute_rolling_vol(btc_returns[year], VOL_WINDOW)

    _partial = {"phase": 128}

    # 2. Baseline (no overlay)
    print("\n[2/4] Computing baseline (no overlay)...")
    baseline_sharpes = {}
    for year in YEARS:
        min_len = min(len(sig_returns[sk][year]) for sk in SIG_KEYS)
        ens = np.zeros(int(min_len))
        for sk in SIG_KEYS:
            ens += ENSEMBLE_WEIGHTS[sk] * sig_returns[sk][year][:int(min_len)]
        baseline_sharpes[year] = sharpe(ens)
    baseline_avg = round(float(np.mean(list(baseline_sharpes.values()))), 4)
    baseline_min = round(float(np.min(list(baseline_sharpes.values()))), 4)
    baseline_obj = obj_func(list(baseline_sharpes.values()))
    print(f"  BASELINE: AVG={baseline_avg:.3f}  MIN={baseline_min:.3f}  OBJ={baseline_obj:.4f}")
    for y in YEARS:
        print(f"    {y}: {baseline_sharpes[y]:.4f}")

    # 3. LOYO validation
    print("\n[3/4] LOYO Walk-Forward (leave-one-year-out)...")
    loyo_results = []

    for test_year in YEARS:
        train_years = [y for y in YEARS if y != test_year]

        # Calibrate: compute vol threshold from TRAINING years only
        all_train_vol = []
        for ty in train_years:
            rv = rvol_per_year[ty]
            valid = rv[~np.isnan(rv)]
            all_train_vol.extend(valid.tolist())
        all_train_vol = np.array(all_train_vol)
        vol_threshold = float(np.percentile(all_train_vol, THRESHOLD_PCT))

        # Test: apply overlay on test year using calibrated threshold
        test_sig_rets = {sk: sig_returns[sk][test_year] for sk in SIG_KEYS}
        test_rvol = rvol_per_year[test_year]

        overlay_sharpe, n_high, n_total = apply_combined_overlay(
            test_sig_rets, test_rvol, vol_threshold, test_year
        )

        delta = overlay_sharpe - baseline_sharpes[test_year]
        tag = "✓" if delta > 0 else "✗"
        print(f"  Test={test_year}  Train={','.join(train_years)}  "
              f"thresh={vol_threshold:.2f}  "
              f"baseline={baseline_sharpes[test_year]:.3f}  "
              f"overlay={overlay_sharpe:.3f}  "
              f"Δ={delta:+.3f} {tag}  "
              f"high_vol={n_high}/{n_total} ({n_high/n_total*100:.1f}%)")

        loyo_results.append({
            "test_year": test_year,
            "train_years": train_years,
            "vol_threshold_calibrated": round(vol_threshold, 4),
            "baseline_sharpe": baseline_sharpes[test_year],
            "overlay_sharpe": overlay_sharpe,
            "delta": round(delta, 4),
            "high_vol_bars": n_high,
            "total_bars": n_total,
            "high_vol_pct": round(n_high / n_total * 100, 1),
        })

    loyo_deltas = [r["delta"] for r in loyo_results]
    loyo_wins = sum(1 for d in loyo_deltas if d > 0)
    loyo_avg_delta = round(float(np.mean(loyo_deltas)), 4)
    loyo_overlay_sharpes = [r["overlay_sharpe"] for r in loyo_results]
    loyo_avg = round(float(np.mean(loyo_overlay_sharpes)), 4)
    loyo_min = round(float(np.min(loyo_overlay_sharpes)), 4)
    loyo_obj = obj_func(loyo_overlay_sharpes)

    print(f"\n  LOYO Summary: {loyo_wins}/5 wins, avg Δ={loyo_avg_delta:+.4f}")
    print(f"  LOYO OBJ={loyo_obj:.4f} vs Baseline OBJ={baseline_obj:.4f} (Δ={loyo_obj - baseline_obj:+.4f})")

    # 4. Expanding window validation (chronological)
    print("\n[4/4] Expanding Window (chronological)...")
    expanding_results = []

    for i, test_year in enumerate(YEARS):
        if i == 0:
            # No prior years to calibrate from — skip
            expanding_results.append({
                "test_year": test_year,
                "train_years": [],
                "note": "SKIP — no prior years for calibration",
                "baseline_sharpe": baseline_sharpes[test_year],
                "overlay_sharpe": None,
                "delta": None,
            })
            print(f"  Test={test_year}  SKIP (no prior years)")
            continue

        train_years = YEARS[:i]

        # Calibrate from prior years only
        all_train_vol = []
        for ty in train_years:
            rv = rvol_per_year[ty]
            valid = rv[~np.isnan(rv)]
            all_train_vol.extend(valid.tolist())
        all_train_vol = np.array(all_train_vol)
        vol_threshold = float(np.percentile(all_train_vol, THRESHOLD_PCT))

        test_sig_rets = {sk: sig_returns[sk][test_year] for sk in SIG_KEYS}
        test_rvol = rvol_per_year[test_year]

        overlay_sharpe, n_high, n_total = apply_combined_overlay(
            test_sig_rets, test_rvol, vol_threshold, test_year
        )

        delta = overlay_sharpe - baseline_sharpes[test_year]
        tag = "✓" if delta > 0 else "✗"
        print(f"  Test={test_year}  Train={','.join(train_years)}  "
              f"thresh={vol_threshold:.2f}  "
              f"baseline={baseline_sharpes[test_year]:.3f}  "
              f"overlay={overlay_sharpe:.3f}  "
              f"Δ={delta:+.3f} {tag}  "
              f"high_vol={n_high}/{n_total} ({n_high/n_total*100:.1f}%)")

        expanding_results.append({
            "test_year": test_year,
            "train_years": train_years,
            "vol_threshold_calibrated": round(vol_threshold, 4),
            "baseline_sharpe": baseline_sharpes[test_year],
            "overlay_sharpe": overlay_sharpe,
            "delta": round(delta, 4),
            "high_vol_bars": n_high,
            "total_bars": n_total,
            "high_vol_pct": round(n_high / n_total * 100, 1),
        })

    # Expanding window summary (exclude first year which was skipped)
    exp_valid = [r for r in expanding_results if r["delta"] is not None]
    exp_deltas = [r["delta"] for r in exp_valid]
    exp_wins = sum(1 for d in exp_deltas if d > 0)
    exp_avg_delta = round(float(np.mean(exp_deltas)), 4) if exp_deltas else 0
    exp_overlay_sharpes = [r["overlay_sharpe"] for r in exp_valid]
    exp_avg = round(float(np.mean(exp_overlay_sharpes)), 4) if exp_overlay_sharpes else 0
    exp_obj = obj_func(exp_overlay_sharpes) if exp_overlay_sharpes else 0

    print(f"\n  Expanding Summary: {exp_wins}/{len(exp_valid)} wins, avg Δ={exp_avg_delta:+.4f}")

    # 5. Fixed absolute threshold test (no calibration needed = truly OOS)
    print("\n[BONUS] Fixed absolute vol thresholds (no calibration)...")
    abs_thresholds = [0.40, 0.50, 0.60, 0.70, 0.80, 1.00]
    abs_results = []

    for abs_thresh in abs_thresholds:
        yearly_sharpes = []
        yearly_detail = {}
        for year in YEARS:
            test_sig_rets = {sk: sig_returns[sk][year] for sk in SIG_KEYS}
            test_rvol = rvol_per_year[year]
            s, n_high, n_total = apply_combined_overlay(
                test_sig_rets, test_rvol, abs_thresh, year
            )
            yearly_sharpes.append(s)
            yearly_detail[year] = {
                "sharpe": s,
                "high_vol_pct": round(n_high / n_total * 100, 1) if n_total > 0 else 0,
            }

        avg_s = round(float(np.mean(yearly_sharpes)), 4)
        min_s = round(float(np.min(yearly_sharpes)), 4)
        obj = obj_func(yearly_sharpes)
        delta_obj = obj - baseline_obj
        tag = "✓" if delta_obj > 0 else "✗"
        print(f"  thresh={abs_thresh:.2f}  AVG={avg_s:.3f}  MIN={min_s:.3f}  OBJ={obj:.4f}  ΔOBJ={delta_obj:+.4f} {tag}")

        abs_results.append({
            "threshold": abs_thresh,
            "yearly": yearly_detail,
            "avg_sharpe": avg_s,
            "min_sharpe": min_s,
            "obj": obj,
            "delta_obj": round(delta_obj, 4),
        })

    # Final verdict
    print("\n" + "=" * 70)
    print("FINAL VERDICT")
    print("=" * 70)

    loyo_pass = loyo_wins >= 3 and loyo_avg_delta > 0.05
    exp_pass = exp_wins >= 2 and exp_avg_delta > 0.05

    print(f"\n  LOYO: {loyo_wins}/5 wins, avg Δ={loyo_avg_delta:+.4f}, OBJ={loyo_obj:.4f} {'PASS' if loyo_pass else 'FAIL'}")
    print(f"  Expanding: {exp_wins}/{len(exp_valid)} wins, avg Δ={exp_avg_delta:+.4f} {'PASS' if exp_pass else 'FAIL'}")

    if loyo_pass and exp_pass:
        verdict = "VALIDATED — overlay survives walk-forward, consider for production"
    elif loyo_pass or exp_pass:
        verdict = "MARGINAL — partial walk-forward evidence, monitor before production"
    else:
        verdict = "FAILED — overlay does NOT survive walk-forward, IS-overfit"

    print(f"\n  >>> {verdict}")

    elapsed = time.time() - t0
    _partial = {
        "phase": 128,
        "description": "Walk-Forward Validation of Combined Vol Overlay",
        "elapsed_seconds": round(elapsed, 1),
        "overlay_params": {
            "vol_window": VOL_WINDOW,
            "threshold_pct": THRESHOLD_PCT,
            "scale_factor": SCALE_FACTOR,
            "f144_boost": F144_BOOST,
        },
        "baseline": {
            "yearly_sharpes": baseline_sharpes,
            "avg_sharpe": baseline_avg,
            "min_sharpe": baseline_min,
            "obj": baseline_obj,
        },
        "loyo": {
            "results": loyo_results,
            "wins": f"{loyo_wins}/5",
            "avg_delta": loyo_avg_delta,
            "avg_sharpe": loyo_avg,
            "min_sharpe": loyo_min,
            "obj": loyo_obj,
            "pass": loyo_pass,
        },
        "expanding_window": {
            "results": expanding_results,
            "wins": f"{exp_wins}/{len(exp_valid)}",
            "avg_delta": exp_avg_delta,
            "pass": exp_pass,
        },
        "fixed_threshold": abs_results,
        "verdict": verdict,
    }
    _save(_partial, partial=False)

    print(f"\nPhase 128 complete in {elapsed:.0f}s")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
