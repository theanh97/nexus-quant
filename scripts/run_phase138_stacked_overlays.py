#!/usr/bin/env python3
"""
Phase 138: Stacked Overlays — Vol + Breadth Combined
======================================================
Phase 137 showed breadth overlay validated (OBJ=1.762, 4/5 LOYO).
The interaction test showed vol+breadth combined OBJ=1.874 (+0.307).

This phase does proper walk-forward validation of the STACKED overlay:
1. LOYO: calibrate breadth params on 4 years (with vol overlay active),
   test on held-out year
2. Parameter sensitivity: test breadth params grid WITH vol overlay active
3. Robustness: compare best single (lb=168) vs safer (lb=504, 5/5 wins)
4. Production readiness: recommend final params for integration
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

# Vol overlay params from production config
VOL_WINDOW = PROD_CFG["vol_regime_overlay"]["window_bars"]  # 168
VOL_THRESHOLD = PROD_CFG["vol_regime_overlay"]["threshold"]  # 0.50
VOL_SCALE = PROD_CFG["vol_regime_overlay"]["scale_factor"]  # 0.50
VOL_F144_BOOST = PROD_CFG["vol_regime_overlay"]["f144_boost"]  # 0.20

YEAR_RANGES = {
    "2021": ("2021-02-01", "2021-12-31"),
    "2022": ("2022-01-01", "2022-12-31"),
    "2023": ("2023-01-01", "2023-12-31"),
    "2024": ("2024-01-01", "2024-12-31"),
    "2025": ("2025-01-01", "2025-12-31"),
}
YEARS = sorted(YEAR_RANGES.keys())

OUT_DIR = ROOT / "artifacts" / "phase138"
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
    path = OUT_DIR / "stacked_overlays_report.json"
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"  → Saved: {path}")


def compute_btc_price_vol(dataset, window: int = 168) -> np.ndarray:
    """Rolling annualized BTC price vol (causal)."""
    n = len(dataset.timeline)
    rets = np.zeros(n)
    for i in range(1, n):
        c0 = dataset.close("BTCUSDT", i - 1)
        c1 = dataset.close("BTCUSDT", i)
        rets[i] = (c1 / c0 - 1.0) if c0 > 0 else 0.0

    vol = np.full(n, np.nan)
    for i in range(window, n):
        vol[i] = float(np.std(rets[i - window:i])) * np.sqrt(8760)
    if window < n:
        vol[:window] = vol[window]
    return vol


def compute_momentum_breadth(sym_rets: dict, lookback: int = 168) -> np.ndarray:
    """Fraction of symbols with positive lookback return. Range: 0 to 1."""
    syms = list(sym_rets.keys())
    n = min(len(sym_rets[s]) for s in syms)
    breadth = np.full(n, np.nan)
    for i in range(lookback, n):
        n_positive = 0
        for s in syms:
            cum_ret = float(np.sum(sym_rets[s][i - lookback:i]))
            if cum_ret > 0:
                n_positive += 1
        breadth[i] = n_positive / len(syms)
    if lookback < n:
        breadth[:lookback] = 0.5
    return breadth


def apply_vol_overlay(sig_returns: dict, btc_vol: np.ndarray,
                      threshold: float, scale: float, f144_boost: float) -> np.ndarray:
    """Apply vol overlay to ensemble returns: reduce leverage + tilt F144 when vol > threshold."""
    min_len = min(len(sig_returns[sk]) for sk in SIG_KEYS)
    min_len = min(min_len, len(btc_vol))

    ens = np.zeros(min_len)
    for i in range(min_len):
        if not np.isnan(btc_vol[i]) and btc_vol[i] > threshold:
            # Tilt: boost F144, reduce others
            boost_per_other = f144_boost / max(1, len(SIG_KEYS) - 1)
            for sk in SIG_KEYS:
                if sk == "f144":
                    adj_w = min(0.60, ENSEMBLE_WEIGHTS[sk] + f144_boost)
                else:
                    adj_w = max(0.05, ENSEMBLE_WEIGHTS[sk] - boost_per_other)
                ens[i] += adj_w * sig_returns[sk][i]
            ens[i] *= scale
        else:
            for sk in SIG_KEYS:
                ens[i] += ENSEMBLE_WEIGHTS[sk] * sig_returns[sk][i]
    return ens


def apply_breadth_overlay(ens: np.ndarray, breadth: np.ndarray,
                          low: float, high: float, scale: float) -> np.ndarray:
    """Boost ensemble when breadth is in [low, high] range (mixed = more dispersion)."""
    n = min(len(ens), len(breadth))
    out = ens[:n].copy()
    for i in range(n):
        if not np.isnan(breadth[i]) and low <= breadth[i] <= high:
            out[i] *= scale
    return out


def main():
    global _partial
    t0 = time.time()
    print("=" * 70)
    print("Phase 138: Stacked Overlays — Vol + Breadth Combined")
    print("=" * 70)

    # 1. Load data
    print("\n[1/5] Loading data...")
    sig_returns = {sk: {} for sk in SIG_KEYS}
    sym_returns = {}
    datasets = {}

    for year, (start, end) in YEAR_RANGES.items():
        print(f"  {year}:", end="", flush=True)
        cfg_data = {
            "provider": "binance_rest_v1", "symbols": SYMBOLS,
            "start": start, "end": end, "bar_interval": "1h",
            "cache_dir": ".cache/binance_rest",
        }
        provider = make_provider(cfg_data, seed=42)
        dataset = provider.load()
        datasets[year] = dataset

        sym_returns[year] = {}
        for sym in SYMBOLS:
            rets = []
            for i in range(1, len(dataset.timeline)):
                c0 = dataset.close(sym, i - 1)
                c1 = dataset.close(sym, i)
                rets.append((c1 / c0 - 1.0) if c0 > 0 else 0.0)
            sym_returns[year][sym] = np.array(rets, dtype=np.float64)

        for sk in SIG_KEYS:
            sig_def = SIGNAL_DEFS[sk]
            cost_model = cost_model_from_config({"fee_rate": 0.0005, "slippage_rate": 0.0003, "cost_multiplier": 1.0})
            bt_cfg = BacktestConfig(costs=cost_model)
            strat = make_strategy({"name": sig_def["strategy"], "params": sig_def["params"]})
            engine = BacktestEngine(bt_cfg)
            result = engine.run(dataset, strat)
            sig_returns[sk][year] = np.array(result.returns, dtype=np.float64)

        print(f" OK", flush=True)

    _partial = {"phase": 138}

    # 2. Compute vol overlay returns + breadth for each year
    print("\n[2/5] Computing vol-overlaid returns + breadth...")
    vol_ens = {}  # ensemble returns WITH vol overlay
    raw_ens = {}  # ensemble returns WITHOUT overlays
    breadth_data = {}
    btc_vol_data = {}

    for year in YEARS:
        # Raw ensemble
        min_len = min(len(sig_returns[sk][year]) for sk in SIG_KEYS)
        raw = np.zeros(min_len)
        for sk in SIG_KEYS:
            raw += ENSEMBLE_WEIGHTS[sk] * sig_returns[sk][year][:min_len]
        raw_ens[year] = raw

        # BTC vol
        btc_vol = compute_btc_price_vol(datasets[year], window=VOL_WINDOW)[:min_len]
        btc_vol_data[year] = btc_vol

        # Vol-overlaid ensemble
        vol_overlaid = apply_vol_overlay(
            {sk: sig_returns[sk][year][:min_len] for sk in SIG_KEYS},
            btc_vol, VOL_THRESHOLD, VOL_SCALE, VOL_F144_BOOST
        )
        vol_ens[year] = vol_overlaid

        # Breadth
        br = compute_momentum_breadth(sym_returns[year])[:min_len]
        breadth_data[year] = br

        s_raw = sharpe(raw)
        s_vol = sharpe(vol_overlaid)
        print(f"  {year}: raw={s_raw:.3f}  vol_overlay={s_vol:.3f}  Δ={s_vol-s_raw:+.3f}")

    # 2b. Pre-compute breadth for all lookback values (avoid redundant computation)
    print("\n[2b/5] Pre-computing breadth for all lookback values...")
    LOOKBACKS = [168, 336, 504]
    precomputed_breadth = {}  # {(year, lb): np.ndarray}
    for lb in LOOKBACKS:
        for year in YEARS:
            min_len = len(vol_ens[year])
            br = compute_momentum_breadth(sym_returns[year], lookback=lb)[:min_len]
            precomputed_breadth[(year, lb)] = br
        print(f"  lb={lb}: precomputed for all years", flush=True)

    # 3. Sensitivity scan: breadth params WITH vol overlay active
    print("\n[3/5] Sensitivity scan — breadth on top of vol overlay...")
    breadth_grid = []
    for lb in LOOKBACKS:
        for lo in [0.2, 0.3, 0.4]:
            for hi in [0.6, 0.7, 0.8]:
                if lo >= hi:
                    continue
                for sc in [1.3, 1.5, 1.7]:
                    breadth_grid.append({"lookback": lb, "low": lo, "high": hi, "scale": sc})

    # Baseline: vol overlay only
    baseline_yearly = [sharpe(vol_ens[y]) for y in YEARS]
    baseline_obj = obj_func(baseline_yearly)
    baseline_avg = round(float(np.mean(baseline_yearly)), 4)
    baseline_min = round(float(np.min(baseline_yearly)), 4)

    print(f"  Baseline (vol only): AVG={baseline_avg:.3f} MIN={baseline_min:.3f} OBJ={baseline_obj:.4f}")

    sensitivity_results = []
    for cfg in breadth_grid:
        yearly_sharpes = []
        for year in YEARS:
            br = precomputed_breadth[(year, cfg["lookback"])]
            min_len = min(len(vol_ens[year]), len(br))
            stacked = apply_breadth_overlay(vol_ens[year][:min_len], br[:min_len],
                                            cfg["low"], cfg["high"], cfg["scale"])
            yearly_sharpes.append(sharpe(stacked))

        avg = round(float(np.mean(yearly_sharpes)), 4)
        mn = round(float(np.min(yearly_sharpes)), 4)
        ob = obj_func(yearly_sharpes)
        delta = round(ob - baseline_obj, 4)
        sensitivity_results.append({
            **cfg, "avg_sharpe": avg, "min_sharpe": mn,
            "obj": ob, "delta_obj": delta,
        })

    sensitivity_results.sort(key=lambda x: x["obj"], reverse=True)
    n_positive = sum(1 for r in sensitivity_results if r["delta_obj"] > 0)

    print(f"  Tested {len(sensitivity_results)} configs, {n_positive}/{len(sensitivity_results)} beat vol-only baseline")
    print(f"\n  {'LB':>4s} {'LO':>4s} {'HI':>4s} {'SC':>4s} {'OBJ':>8s} {'AVG':>8s} {'MIN':>8s} {'ΔOBJ':>8s}")
    print(f"  {'----':>4s} {'----':>4s} {'----':>4s} {'----':>4s} {'--------':>8s} {'--------':>8s} {'--------':>8s} {'--------':>8s}")
    for r in sensitivity_results[:10]:
        tag = " ✓" if r["delta_obj"] > 0.03 else ""
        print(f"  {r['lookback']:4d} {r['low']:4.1f} {r['high']:4.1f} {r['scale']:4.1f} "
              f"{r['obj']:8.4f} {r['avg_sharpe']:8.3f} {r['min_sharpe']:8.3f} {r['delta_obj']:+8.4f}{tag}")

    best_params = sensitivity_results[0]

    # 4. LOYO validation of stacked overlay
    print("\n[4/5] Leave-One-Year-Out validation of stacked overlay...")
    loyo_results = []

    for test_year in YEARS:
        train_years = [y for y in YEARS if y != test_year]

        # Find best breadth params on training years (with vol overlay active)
        best_train_obj = -999
        best_train_params = None

        for cfg in breadth_grid:
            train_sharpes = []
            for y in train_years:
                br = precomputed_breadth[(y, cfg["lookback"])]
                min_len = min(len(vol_ens[y]), len(br))
                stacked = apply_breadth_overlay(vol_ens[y][:min_len], br[:min_len],
                                                cfg["low"], cfg["high"], cfg["scale"])
                train_sharpes.append(sharpe(stacked))

            train_obj = obj_func(train_sharpes)
            if train_obj > best_train_obj:
                best_train_obj = train_obj
                best_train_params = cfg.copy()

        # Apply best params to test year
        br_test = precomputed_breadth[(test_year, best_train_params["lookback"])]
        min_len = min(len(vol_ens[test_year]), len(br_test))
        stacked_test = apply_breadth_overlay(
            vol_ens[test_year][:min_len], br_test[:min_len],
            best_train_params["low"], best_train_params["high"], best_train_params["scale"]
        )

        baseline_s = sharpe(vol_ens[test_year])
        stacked_s = sharpe(stacked_test)
        delta = round(stacked_s - baseline_s, 4)

        loyo_results.append({
            "test_year": test_year,
            "baseline_sharpe": baseline_s,
            "stacked_sharpe": stacked_s,
            "delta": delta,
            "train_params": best_train_params,
            "train_obj": round(best_train_obj, 4),
        })

        tag = "✓" if delta > 0 else "✗"
        print(f"  {test_year}: vol_only={baseline_s:.3f}  stacked={stacked_s:.3f}  "
              f"Δ={delta:+.4f} {tag}  (trained: lb={best_train_params['lookback']} "
              f"lo={best_train_params['low']} hi={best_train_params['high']} "
              f"sc={best_train_params['scale']})")

    loyo_wins = sum(1 for r in loyo_results if r["delta"] > 0)
    loyo_avg_delta = round(float(np.mean([r["delta"] for r in loyo_results])), 4)
    stacked_sharpes = [r["stacked_sharpe"] for r in loyo_results]
    loyo_obj = obj_func(stacked_sharpes)

    print(f"\n  LOYO: {loyo_wins}/{len(loyo_results)} wins, avg Δ={loyo_avg_delta:+.4f}, OBJ={loyo_obj:.4f}")

    # 5. Compare specific candidate configs for production
    print("\n[5/5] Production candidate comparison...")
    candidates = {
        "vol_only": {"breadth": False},
        "best_sensitivity": {
            "breadth": True,
            "lookback": best_params["lookback"],
            "low": best_params["low"],
            "high": best_params["high"],
            "scale": best_params["scale"],
        },
        "conservative_168": {
            "breadth": True, "lookback": 168, "low": 0.2, "high": 0.7, "scale": 1.3,
        },
        "robust_504": {
            "breadth": True, "lookback": 504, "low": 0.4, "high": 0.6, "scale": 1.5,
        },
        "moderate_168": {
            "breadth": True, "lookback": 168, "low": 0.2, "high": 0.7, "scale": 1.5,
        },
    }

    candidate_results = {}
    for cname, ccfg in candidates.items():
        yearly = {}
        for year in YEARS:
            if not ccfg.get("breadth"):
                s = sharpe(vol_ens[year])
            else:
                br = precomputed_breadth[(year, ccfg["lookback"])]
                min_len = min(len(vol_ens[year]), len(br))
                stacked = apply_breadth_overlay(vol_ens[year][:min_len], br[:min_len],
                                                ccfg["low"], ccfg["high"], ccfg["scale"])
                s = sharpe(stacked)
            yearly[year] = s

        ys = list(yearly.values())
        avg = round(float(np.mean(ys)), 4)
        mn = round(float(np.min(ys)), 4)
        ob = obj_func(ys)
        candidate_results[cname] = {
            "config": ccfg, "yearly": yearly,
            "avg_sharpe": avg, "min_sharpe": mn, "obj": ob,
        }

    print(f"\n  {'Candidate':22s} {'OBJ':>8s} {'AVG':>8s} {'MIN':>8s} {'ΔOBJ':>8s}")
    print(f"  {'-'*22} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")
    vol_only_obj = candidate_results["vol_only"]["obj"]
    for cname in sorted(candidate_results, key=lambda k: candidate_results[k]["obj"], reverse=True):
        r = candidate_results[cname]
        delta = r["obj"] - vol_only_obj
        tag = " ★" if cname == "vol_only" else (" ✓" if delta > 0.03 else "")
        print(f"  {cname:22s} {r['obj']:8.4f} {r['avg_sharpe']:8.3f} "
              f"{r['min_sharpe']:8.3f} {delta:+8.4f}{tag}")

    # Full comparison: raw baseline vs vol only vs stacked
    print("\n  Full progression:")
    raw_yearly = {y: sharpe(raw_ens[y]) for y in YEARS}
    raw_obj = obj_func(list(raw_yearly.values()))
    print(f"    raw baseline:  OBJ={raw_obj:.4f}")
    print(f"    + vol overlay: OBJ={vol_only_obj:.4f} (+{vol_only_obj - raw_obj:.4f})")
    best_cand = max(candidate_results, key=lambda k: candidate_results[k]["obj"])
    best_obj = candidate_results[best_cand]["obj"]
    print(f"    + breadth:     OBJ={best_obj:.4f} (+{best_obj - vol_only_obj:.4f}) [{best_cand}]")
    print(f"    TOTAL GAIN:    +{best_obj - raw_obj:.4f} OBJ vs raw baseline")

    # Determine production recommendation
    # Use the conservative_168 if it's close to best (prefer stability)
    # Use robust_504 if it has better MIN sharpe
    prod_recommendation = best_cand
    cons_obj = candidate_results.get("conservative_168", {}).get("obj", 0)
    rob_obj = candidate_results.get("robust_504", {}).get("obj", 0)

    # Prefer conservative if within 0.03 of best (more stable)
    if best_obj - cons_obj < 0.03 and cons_obj > vol_only_obj:
        prod_recommendation = "conservative_168"
    # Prefer robust_504 if its MIN is significantly higher
    if (candidate_results.get("robust_504", {}).get("min_sharpe", 0) >
            candidate_results.get(best_cand, {}).get("min_sharpe", 0) + 0.1):
        if rob_obj > vol_only_obj:
            prod_recommendation = "robust_504"

    # Check LOYO pass
    loyo_pass = loyo_wins >= 3 and loyo_avg_delta > 0

    if loyo_pass and best_obj > vol_only_obj + 0.03:
        verdict = f"VALIDATED — stacked overlay adds +{best_obj - vol_only_obj:.3f} OBJ (recommend: {prod_recommendation})"
    elif best_obj > vol_only_obj:
        verdict = f"MARGINAL — stacked adds +{best_obj - vol_only_obj:.3f} OBJ but LOYO {loyo_wins}/5"
    else:
        verdict = "NO IMPROVEMENT — breadth does not help on top of vol overlay"

    elapsed = time.time() - t0
    _partial = {
        "phase": 138,
        "description": "Stacked Overlays — Vol + Breadth Combined",
        "elapsed_seconds": round(elapsed, 1),
        "baseline_vol_only": {
            "yearly": {y: sharpe(vol_ens[y]) for y in YEARS},
            "avg_sharpe": baseline_avg,
            "min_sharpe": baseline_min,
            "obj": baseline_obj,
        },
        "raw_baseline": {
            "yearly": raw_yearly,
            "obj": raw_obj,
        },
        "sensitivity": {
            "n_positive": n_positive,
            "n_total": len(sensitivity_results),
            "pct_positive": round(100 * n_positive / len(sensitivity_results), 1),
            "top5": sensitivity_results[:5],
        },
        "loyo": {
            "results": loyo_results,
            "wins": f"{loyo_wins}/{len(loyo_results)}",
            "avg_delta": loyo_avg_delta,
            "obj": loyo_obj,
            "pass": loyo_pass,
        },
        "candidates": candidate_results,
        "production_recommendation": {
            "name": prod_recommendation,
            **candidate_results[prod_recommendation],
        },
        "total_gain_over_raw": round(best_obj - raw_obj, 4),
        "verdict": verdict,
    }
    _save(_partial, partial=False)

    print(f"\n  VERDICT: {verdict}")
    print(f"\nPhase 138 complete in {elapsed:.0f}s")
    print("=" * 70)


if __name__ == "__main__":
    main()
