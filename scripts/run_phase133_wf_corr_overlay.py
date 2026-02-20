#!/usr/bin/env python3
"""
Phase 133: Walk-Forward Validation of Correlation Overlay
==========================================================
Phase 132 found: reduce_high_corr_50pct (reduce leverage 50% when rolling
avg pairwise correlation > 75th percentile) improves OBJ from 1.57 → 1.67.

BUT: the 75th percentile threshold was computed per-year = look-ahead bias.

Walk-forward tests:
1. LOYO (Leave-One-Year-Out): calibrate corr threshold on 4 years, test on 1
2. Expanding window: calibrate on years before, test on next year
3. Fixed absolute threshold: use absolute corr levels (no calibration)
4. Interaction test: correlation overlay + vol overlay together vs each alone
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

# Phase 132 best params
CORR_WINDOW = 168
THRESHOLD_PCT = 75  # percentile of rolling avg correlation
SCALE_FACTOR = 0.5  # reduce leverage by 50% during high corr

# Vol overlay params (from Phase 127-128, for interaction test)
VOL_WINDOW = 168
VOL_THRESHOLD = 0.50  # fixed absolute threshold from Phase 128
VOL_SCALE = 0.5
VOL_F144_BOOST = 0.20

OUT_DIR = ROOT / "artifacts" / "phase133"
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
    path = OUT_DIR / "wf_corr_overlay_report.json"
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"  → Saved: {path}")


def compute_rolling_avg_corr(sym_rets: dict, window: int = 168) -> np.ndarray:
    """Compute rolling average pairwise correlation across all symbols.
    Returns array of avg correlation values (causal / look-back only)."""
    syms = list(sym_rets.keys())
    n = min(len(sym_rets[s]) for s in syms)
    avg_corr = np.full(n, np.nan)

    for i in range(window, n):
        rets_matrix = []
        for s in syms:
            rets_matrix.append(sym_rets[s][i - window:i])
        rets_matrix = np.array(rets_matrix)
        corr_mat = np.corrcoef(rets_matrix)

        n_syms = len(syms)
        off_diag = []
        for r in range(n_syms):
            for c in range(r + 1, n_syms):
                if not np.isnan(corr_mat[r, c]):
                    off_diag.append(corr_mat[r, c])

        avg_corr[i] = float(np.mean(off_diag)) if off_diag else 0.0

    first_valid = window
    if first_valid < n:
        avg_corr[:first_valid] = avg_corr[first_valid]

    return avg_corr


def compute_rolling_vol(rets: np.ndarray, window: int) -> np.ndarray:
    """Compute rolling annualized volatility (causal)."""
    n = len(rets)
    vol = np.full(n, np.nan)
    for i in range(window, n):
        vol[i] = float(np.std(rets[i - window:i])) * np.sqrt(8760)
    vol[:window] = vol[window] if window < n else 0.0
    return vol


def apply_corr_overlay(sig_rets: dict, avg_corr: np.ndarray,
                       corr_threshold: float) -> tuple:
    """Apply correlation overlay: reduce leverage when avg corr > threshold.
    Returns (sharpe, n_high_corr, total)."""
    min_len = min(len(sig_rets[sk]) for sk in SIG_KEYS)
    min_len = min(min_len, len(avg_corr))

    ens = np.zeros(min_len)
    n_high = 0
    for i in range(min_len):
        ret = 0.0
        for sk in SIG_KEYS:
            ret += ENSEMBLE_WEIGHTS[sk] * sig_rets[sk][i]
        if not np.isnan(avg_corr[i]) and avg_corr[i] >= corr_threshold:
            ret *= SCALE_FACTOR
            n_high += 1
        ens[i] = ret

    return sharpe(ens), n_high, min_len


def apply_vol_overlay(sig_rets: dict, rvol: np.ndarray,
                      vol_threshold: float) -> np.ndarray:
    """Apply vol overlay: reduce + tilt F144 when BTC vol > threshold.
    Returns modified ensemble returns array."""
    min_len = min(len(sig_rets[sk]) for sk in SIG_KEYS)
    min_len = min(min_len, len(rvol))

    ens = np.zeros(min_len)
    for i in range(min_len):
        if not np.isnan(rvol[i]) and rvol[i] > vol_threshold:
            boosted = dict(ENSEMBLE_WEIGHTS)
            boost_from_each = VOL_F144_BOOST / 3
            for sk in SIG_KEYS:
                if sk == "f144":
                    boosted[sk] = min(0.60, boosted[sk] + VOL_F144_BOOST)
                else:
                    boosted[sk] = max(0.05, boosted[sk] - boost_from_each)
            total = sum(boosted.values())
            for sk in SIG_KEYS:
                ens[i] += (boosted[sk] / total) * sig_rets[sk][i] * VOL_SCALE
        else:
            for sk in SIG_KEYS:
                ens[i] += ENSEMBLE_WEIGHTS[sk] * sig_rets[sk][i]
    return ens


def apply_both_overlays(sig_rets: dict, avg_corr: np.ndarray,
                        corr_threshold: float, rvol: np.ndarray,
                        vol_threshold: float) -> tuple:
    """Apply BOTH correlation and vol overlays.
    If both trigger: apply both reductions (multiplicative).
    Returns (sharpe, n_corr_only, n_vol_only, n_both, total)."""
    min_len = min(len(sig_rets[sk]) for sk in SIG_KEYS)
    min_len = min(min_len, len(avg_corr), len(rvol))

    ens = np.zeros(min_len)
    n_corr = 0
    n_vol = 0
    n_both = 0

    for i in range(min_len):
        high_corr = (not np.isnan(avg_corr[i]) and avg_corr[i] >= corr_threshold)
        high_vol = (not np.isnan(rvol[i]) and rvol[i] > vol_threshold)

        if high_corr:
            n_corr += 1
        if high_vol:
            n_vol += 1
        if high_corr and high_vol:
            n_both += 1

        if high_vol:
            # Vol overlay takes precedence (tilt + reduce)
            boosted = dict(ENSEMBLE_WEIGHTS)
            boost_from_each = VOL_F144_BOOST / 3
            for sk in SIG_KEYS:
                if sk == "f144":
                    boosted[sk] = min(0.60, boosted[sk] + VOL_F144_BOOST)
                else:
                    boosted[sk] = max(0.05, boosted[sk] - boost_from_each)
            total = sum(boosted.values())
            ret = 0.0
            for sk in SIG_KEYS:
                ret += (boosted[sk] / total) * sig_rets[sk][i] * VOL_SCALE
            if high_corr:
                ret *= SCALE_FACTOR  # additional corr reduction
            ens[i] = ret
        elif high_corr:
            # Corr overlay only
            ret = 0.0
            for sk in SIG_KEYS:
                ret += ENSEMBLE_WEIGHTS[sk] * sig_rets[sk][i]
            ens[i] = ret * SCALE_FACTOR
        else:
            # Normal regime
            for sk in SIG_KEYS:
                ens[i] += ENSEMBLE_WEIGHTS[sk] * sig_rets[sk][i]

    return sharpe(ens), n_corr, n_vol, n_both, min_len


def main():
    global _partial
    t0 = time.time()
    print("=" * 70)
    print("Phase 133: Walk-Forward Validation of Correlation Overlay")
    print("=" * 70)

    # 1. Load all data
    print("\n[1/5] Loading signal returns + per-symbol returns...")
    sig_returns = {sk: {} for sk in SIG_KEYS}
    sym_returns = {}
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

        # Per-symbol returns (for correlation computation)
        sym_returns[year] = {}
        for sym in SYMBOLS:
            rets = []
            for i in range(1, len(dataset.timeline)):
                c0 = dataset.close(sym, i - 1)
                c1 = dataset.close(sym, i)
                rets.append((c1 / c0 - 1.0) if c0 > 0 else 0.0)
            sym_returns[year][sym] = np.array(rets, dtype=np.float64)

        # BTC returns (for vol overlay)
        btc_returns[year] = sym_returns[year].get("BTCUSDT", np.zeros(1))

        # Signal returns
        for sk in SIG_KEYS:
            sig_def = SIGNAL_DEFS[sk]
            cost_model = cost_model_from_config({"fee_rate": 0.0005, "slippage_rate": 0.0003, "cost_multiplier": 1.0})
            bt_cfg = BacktestConfig(costs=cost_model)
            strat = make_strategy({"name": sig_def["strategy"], "params": sig_def["params"]})
            engine = BacktestEngine(bt_cfg)
            result = engine.run(dataset, strat)
            sig_returns[sk][year] = np.array(result.returns, dtype=np.float64)

        print(f" {len(sym_returns[year][SYMBOLS[0]])} bars", flush=True)

    # Compute rolling correlation per year
    print("\n  Computing rolling correlations...")
    corr_per_year = {}
    for year in YEARS:
        corr_per_year[year] = compute_rolling_avg_corr(sym_returns[year], CORR_WINDOW)
        valid = corr_per_year[year][~np.isnan(corr_per_year[year])]
        print(f"    {year}: mean_corr={float(np.mean(valid)):.3f}, "
              f"p75={float(np.percentile(valid, 75)):.3f}")

    # Compute rolling vol per year (for interaction test)
    rvol_per_year = {}
    for year in YEARS:
        rvol_per_year[year] = compute_rolling_vol(btc_returns[year], VOL_WINDOW)

    _partial = {"phase": 133}

    # 2. Baseline (no overlay)
    print("\n[2/5] Computing baseline (no overlay)...")
    baseline_sharpes = {}
    for year in YEARS:
        min_len = min(len(sig_returns[sk][year]) for sk in SIG_KEYS)
        ens = np.zeros(int(min_len))
        for sk in SIG_KEYS:
            ens += ENSEMBLE_WEIGHTS[sk] * sig_returns[sk][year][:int(min_len)]
        baseline_sharpes[year] = sharpe(ens)
    baseline_obj = obj_func(list(baseline_sharpes.values()))
    baseline_avg = round(float(np.mean(list(baseline_sharpes.values()))), 4)
    baseline_min = round(float(np.min(list(baseline_sharpes.values()))), 4)
    print(f"  BASELINE: AVG={baseline_avg:.3f}  MIN={baseline_min:.3f}  OBJ={baseline_obj:.4f}")
    for y in YEARS:
        print(f"    {y}: {baseline_sharpes[y]:.4f}")

    # 3. LOYO validation
    print("\n[3/5] LOYO Walk-Forward (leave-one-year-out)...")
    loyo_results = []

    for test_year in YEARS:
        train_years = [y for y in YEARS if y != test_year]

        # Calibrate: compute correlation threshold from TRAINING years only
        all_train_corr = []
        for ty in train_years:
            corr = corr_per_year[ty]
            valid = corr[~np.isnan(corr)]
            all_train_corr.extend(valid.tolist())
        all_train_corr = np.array(all_train_corr)
        corr_threshold = float(np.percentile(all_train_corr, THRESHOLD_PCT))

        # Test: apply overlay on test year using calibrated threshold
        test_sig_rets = {sk: sig_returns[sk][test_year] for sk in SIG_KEYS}
        test_corr = corr_per_year[test_year]

        overlay_sharpe, n_high, n_total = apply_corr_overlay(
            test_sig_rets, test_corr, corr_threshold
        )

        delta = overlay_sharpe - baseline_sharpes[test_year]
        tag = "✓" if delta > 0 else "✗"
        print(f"  Test={test_year}  Train={','.join(train_years)}  "
              f"thresh={corr_threshold:.3f}  "
              f"base={baseline_sharpes[test_year]:.3f}  "
              f"overlay={overlay_sharpe:.3f}  "
              f"Δ={delta:+.3f} {tag}  "
              f"high_corr={n_high}/{n_total} ({n_high/n_total*100:.1f}%)")

        loyo_results.append({
            "test_year": test_year,
            "train_years": train_years,
            "corr_threshold_calibrated": round(corr_threshold, 4),
            "baseline_sharpe": baseline_sharpes[test_year],
            "overlay_sharpe": overlay_sharpe,
            "delta": round(delta, 4),
            "high_corr_bars": n_high,
            "total_bars": n_total,
            "high_corr_pct": round(n_high / n_total * 100, 1),
        })

    loyo_deltas = [r["delta"] for r in loyo_results]
    loyo_wins = sum(1 for d in loyo_deltas if d > 0)
    loyo_avg_delta = round(float(np.mean(loyo_deltas)), 4)
    loyo_overlay_sharpes = [r["overlay_sharpe"] for r in loyo_results]
    loyo_obj = obj_func(loyo_overlay_sharpes)

    print(f"\n  LOYO Summary: {loyo_wins}/5 wins, avg Δ={loyo_avg_delta:+.4f}")
    print(f"  LOYO OBJ={loyo_obj:.4f} vs Baseline OBJ={baseline_obj:.4f} "
          f"(Δ={loyo_obj - baseline_obj:+.4f})")

    # 4. Expanding window validation
    print("\n[4/5] Expanding Window (chronological)...")
    expanding_results = []

    for i, test_year in enumerate(YEARS):
        if i == 0:
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

        all_train_corr = []
        for ty in train_years:
            corr = corr_per_year[ty]
            valid = corr[~np.isnan(corr)]
            all_train_corr.extend(valid.tolist())
        all_train_corr = np.array(all_train_corr)
        corr_threshold = float(np.percentile(all_train_corr, THRESHOLD_PCT))

        test_sig_rets = {sk: sig_returns[sk][test_year] for sk in SIG_KEYS}
        test_corr = corr_per_year[test_year]

        overlay_sharpe, n_high, n_total = apply_corr_overlay(
            test_sig_rets, test_corr, corr_threshold
        )

        delta = overlay_sharpe - baseline_sharpes[test_year]
        tag = "✓" if delta > 0 else "✗"
        print(f"  Test={test_year}  Train={','.join(train_years)}  "
              f"thresh={corr_threshold:.3f}  "
              f"base={baseline_sharpes[test_year]:.3f}  "
              f"overlay={overlay_sharpe:.3f}  "
              f"Δ={delta:+.3f} {tag}  "
              f"high_corr={n_high}/{n_total} ({n_high/n_total*100:.1f}%)")

        expanding_results.append({
            "test_year": test_year,
            "train_years": train_years,
            "corr_threshold_calibrated": round(corr_threshold, 4),
            "baseline_sharpe": baseline_sharpes[test_year],
            "overlay_sharpe": overlay_sharpe,
            "delta": round(delta, 4),
            "high_corr_bars": n_high,
            "total_bars": n_total,
            "high_corr_pct": round(n_high / n_total * 100, 1),
        })

    exp_valid = [r for r in expanding_results if r["delta"] is not None]
    exp_deltas = [r["delta"] for r in exp_valid]
    exp_wins = sum(1 for d in exp_deltas if d > 0)
    exp_avg_delta = round(float(np.mean(exp_deltas)), 4) if exp_deltas else 0
    exp_overlay_sharpes = [r["overlay_sharpe"] for r in exp_valid]
    exp_obj = obj_func(exp_overlay_sharpes) if exp_overlay_sharpes else 0

    print(f"\n  Expanding Summary: {exp_wins}/{len(exp_valid)} wins, "
          f"avg Δ={exp_avg_delta:+.4f}")

    # 5. Fixed absolute thresholds (no calibration = truly OOS)
    print("\n[5a/5] Fixed absolute correlation thresholds...")
    abs_thresholds = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]
    abs_results = []

    for abs_thresh in abs_thresholds:
        yearly_sharpes = []
        yearly_detail = {}
        for year in YEARS:
            test_sig_rets = {sk: sig_returns[sk][year] for sk in SIG_KEYS}
            test_corr = corr_per_year[year]
            s, n_high, n_total = apply_corr_overlay(
                test_sig_rets, test_corr, abs_thresh
            )
            yearly_sharpes.append(s)
            yearly_detail[year] = {
                "sharpe": s,
                "high_corr_pct": round(n_high / n_total * 100, 1) if n_total > 0 else 0,
            }

        avg_s = round(float(np.mean(yearly_sharpes)), 4)
        min_s = round(float(np.min(yearly_sharpes)), 4)
        obj = obj_func(yearly_sharpes)
        delta_obj = obj - baseline_obj
        tag = "✓" if delta_obj > 0.03 else ""
        print(f"  corr>{abs_thresh:.2f}  AVG={avg_s:.3f}  MIN={min_s:.3f}  "
              f"OBJ={obj:.4f}  ΔOBJ={delta_obj:+.4f} {tag}")

        abs_results.append({
            "threshold": abs_thresh,
            "yearly": yearly_detail,
            "avg_sharpe": avg_s,
            "min_sharpe": min_s,
            "obj": obj,
            "delta_obj": round(delta_obj, 4),
        })

    # 5b. Interaction test: corr overlay + vol overlay together
    print("\n[5b/5] Interaction test: corr overlay + vol overlay...")

    # Use best fixed corr threshold
    best_abs = max(abs_results, key=lambda x: x["obj"])
    best_corr_thresh = best_abs["threshold"]
    print(f"  Using corr threshold={best_corr_thresh:.2f} (best from fixed)")

    interaction_variants = {
        "baseline": {"corr": False, "vol": False},
        "corr_only": {"corr": True, "vol": False},
        "vol_only": {"corr": False, "vol": True},
        "both": {"corr": True, "vol": True},
    }

    interaction_results = {}
    for vname, vcfg in interaction_variants.items():
        yearly_sharpes = []
        yearly_detail = {}

        for year in YEARS:
            test_sig_rets = {sk: sig_returns[sk][year] for sk in SIG_KEYS}
            test_corr = corr_per_year[year]
            test_rvol = rvol_per_year[year]

            if vcfg["corr"] and vcfg["vol"]:
                s, n_corr, n_vol, n_both, n_total = apply_both_overlays(
                    test_sig_rets, test_corr, best_corr_thresh,
                    test_rvol, VOL_THRESHOLD
                )
                yearly_detail[year] = {
                    "sharpe": s,
                    "corr_active_pct": round(n_corr / n_total * 100, 1),
                    "vol_active_pct": round(n_vol / n_total * 100, 1),
                    "both_active_pct": round(n_both / n_total * 100, 1),
                }
            elif vcfg["corr"]:
                s, n_high, n_total = apply_corr_overlay(
                    test_sig_rets, test_corr, best_corr_thresh
                )
                yearly_detail[year] = {"sharpe": s}
            elif vcfg["vol"]:
                ens = apply_vol_overlay(test_sig_rets, test_rvol, VOL_THRESHOLD)
                s = sharpe(ens)
                yearly_detail[year] = {"sharpe": s}
            else:
                min_len = min(len(sig_returns[sk][year]) for sk in SIG_KEYS)
                ens = np.zeros(int(min_len))
                for sk in SIG_KEYS:
                    ens += ENSEMBLE_WEIGHTS[sk] * sig_returns[sk][year][:int(min_len)]
                s = sharpe(ens)
                yearly_detail[year] = {"sharpe": s}

            yearly_sharpes.append(s)

        avg_s = round(float(np.mean(yearly_sharpes)), 4)
        min_s = round(float(np.min(yearly_sharpes)), 4)
        obj = obj_func(yearly_sharpes)
        interaction_results[vname] = {
            "yearly": yearly_detail,
            "avg_sharpe": avg_s,
            "min_sharpe": min_s,
            "obj": obj,
        }

    print(f"\n  {'Variant':14s} {'OBJ':>8s} {'AVG':>8s} {'MIN':>8s} {'ΔOBJ':>8s}")
    print(f"  {'-'*14} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")
    for vname in ["baseline", "corr_only", "vol_only", "both"]:
        r = interaction_results[vname]
        delta = r["obj"] - interaction_results["baseline"]["obj"]
        tag = " ✓" if delta > 0.05 else ""
        print(f"  {vname:14s} {r['obj']:8.4f} {r['avg_sharpe']:8.3f} "
              f"{r['min_sharpe']:8.3f} {delta:+8.4f}{tag}")

    # Check if overlays are additive or redundant
    corr_delta = interaction_results["corr_only"]["obj"] - interaction_results["baseline"]["obj"]
    vol_delta = interaction_results["vol_only"]["obj"] - interaction_results["baseline"]["obj"]
    both_delta = interaction_results["both"]["obj"] - interaction_results["baseline"]["obj"]
    expected_additive = corr_delta + vol_delta
    synergy = both_delta - expected_additive
    print(f"\n  Corr ΔOBJ={corr_delta:+.4f}, Vol ΔOBJ={vol_delta:+.4f}")
    print(f"  Expected if additive: {expected_additive:+.4f}")
    print(f"  Actual both: {both_delta:+.4f}")
    print(f"  Synergy: {synergy:+.4f} ({'additive' if abs(synergy) < 0.03 else 'redundant' if synergy < -0.03 else 'synergistic'})")

    # Final verdict
    print("\n" + "=" * 70)
    print("FINAL VERDICT")
    print("=" * 70)

    loyo_pass = loyo_wins >= 3 and loyo_avg_delta > 0.02
    exp_pass = exp_wins >= 2 and exp_avg_delta > 0.02

    print(f"\n  LOYO: {loyo_wins}/5 wins, avg Δ={loyo_avg_delta:+.4f}, "
          f"OBJ={loyo_obj:.4f} {'PASS' if loyo_pass else 'FAIL'}")
    print(f"  Expanding: {exp_wins}/{len(exp_valid)} wins, "
          f"avg Δ={exp_avg_delta:+.4f} {'PASS' if exp_pass else 'FAIL'}")
    print(f"  Best fixed threshold: corr>{best_corr_thresh:.2f} "
          f"OBJ={best_abs['obj']:.4f} (ΔOBJ={best_abs['delta_obj']:+.4f})")

    # IS→OOS decay
    is_improvement = 0.1072  # Phase 132 IS result
    oos_improvement = loyo_obj - baseline_obj
    decay = round(1.0 - (oos_improvement / is_improvement), 3) if is_improvement > 0 else 0
    print(f"  IS→OOS decay: {decay*100:.0f}% (IS={is_improvement:+.4f} → LOYO={oos_improvement:+.4f})")

    if loyo_pass and exp_pass:
        verdict = "VALIDATED — corr overlay survives walk-forward"
    elif loyo_pass or exp_pass:
        verdict = "MARGINAL — partial walk-forward evidence"
    else:
        verdict = "FAILED — corr overlay does NOT survive walk-forward, IS-overfit"

    # Check interaction
    if both_delta > max(corr_delta, vol_delta) + 0.02:
        interaction_verdict = "ADDITIVE — both overlays contribute independently"
    elif both_delta < max(corr_delta, vol_delta) - 0.02:
        interaction_verdict = "REDUNDANT — overlays overlap, use only the stronger one"
    else:
        interaction_verdict = "NEUTRAL — both overlays are similar in effect"

    elapsed = time.time() - t0
    _partial = {
        "phase": 133,
        "description": "Walk-Forward Validation of Correlation Overlay",
        "elapsed_seconds": round(elapsed, 1),
        "overlay_params": {
            "corr_window": CORR_WINDOW,
            "threshold_pct": THRESHOLD_PCT,
            "scale_factor": SCALE_FACTOR,
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
            "obj": loyo_obj,
            "pass": loyo_pass,
        },
        "expanding_window": {
            "results": expanding_results,
            "wins": f"{exp_wins}/{len(exp_valid)}",
            "avg_delta": exp_avg_delta,
            "obj": exp_obj,
            "pass": exp_pass,
        },
        "fixed_threshold": abs_results,
        "best_fixed": {
            "threshold": best_corr_thresh,
            "obj": best_abs["obj"],
            "delta_obj": best_abs["delta_obj"],
        },
        "interaction": {
            "results": interaction_results,
            "corr_delta": round(corr_delta, 4),
            "vol_delta": round(vol_delta, 4),
            "both_delta": round(both_delta, 4),
            "synergy": round(synergy, 4),
            "verdict": interaction_verdict,
        },
        "is_oos_decay": f"{decay*100:.0f}%",
        "verdict": verdict,
    }
    _save(_partial, partial=False)

    print(f"\n  VERDICT: {verdict}")
    print(f"  INTERACTION: {interaction_verdict}")
    print(f"\nPhase 133 complete in {elapsed:.0f}s")
    print("=" * 70)


if __name__ == "__main__":
    main()
