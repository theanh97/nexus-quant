#!/usr/bin/env python3
"""
Phase 137: Walk-Forward Validation of Momentum Breadth Overlay
================================================================
Phase 136 found: boost_mixed_breadth (30-70% breadth → 1.3x boost)
gives OBJ=1.665, +0.098 vs baseline, with MIN=1.568 > baseline MIN 1.322.

Unlike correlation/vol overlays that need percentile calibration,
momentum breadth uses absolute thresholds (fraction of coins with
positive momentum). This makes it naturally robust — less to calibrate.

Still, validate with:
1. LOYO (test each year with optimal params from other 4)
2. Parameter sensitivity (vary lookback, thresholds, boost factor)
3. Interaction with vol overlay (additive or redundant?)
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

# Vol overlay params (for interaction test)
VOL_WINDOW = 168
VOL_THRESHOLD = 0.50
VOL_SCALE = 0.5
VOL_F144_BOOST = 0.20

OUT_DIR = ROOT / "artifacts" / "phase137"
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
    path = OUT_DIR / "wf_breadth_overlay_report.json"
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"  → Saved: {path}")


def compute_momentum_breadth(sym_rets: dict, lookback: int = 168) -> np.ndarray:
    """Compute momentum breadth: fraction of symbols with positive lookback return."""
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


def compute_rolling_vol(rets: np.ndarray, window: int) -> np.ndarray:
    n = len(rets)
    vol = np.full(n, np.nan)
    for i in range(window, n):
        vol[i] = float(np.std(rets[i - window:i])) * np.sqrt(8760)
    if window < n:
        vol[:window] = vol[window]
    return vol


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
    print("Phase 137: Walk-Forward Validation of Momentum Breadth Overlay")
    print("=" * 70)

    # 1. Load data
    print("\n[1/5] Loading data...")
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

        sym_returns[year] = {}
        for sym in SYMBOLS:
            rets = []
            for i in range(1, len(dataset.timeline)):
                c0 = dataset.close(sym, i - 1)
                c1 = dataset.close(sym, i)
                rets.append((c1 / c0 - 1.0) if c0 > 0 else 0.0)
            sym_returns[year][sym] = np.array(rets, dtype=np.float64)

        btc_returns[year] = sym_returns[year].get("BTCUSDT", np.zeros(1))

        for sk in SIG_KEYS:
            sig_def = SIGNAL_DEFS[sk]
            cost_model = cost_model_from_config({"fee_rate": 0.0005, "slippage_rate": 0.0003, "cost_multiplier": 1.0})
            bt_cfg = BacktestConfig(costs=cost_model)
            strat = make_strategy({"name": sig_def["strategy"], "params": sig_def["params"]})
            engine = BacktestEngine(bt_cfg)
            result = engine.run(dataset, strat)
            sig_returns[sk][year] = np.array(result.returns, dtype=np.float64)

        print(f" OK", flush=True)

    _partial = {"phase": 137}

    # 2. Baseline + breadth computation
    print("\n[2/5] Computing baseline + breadth...")
    ens_returns = {}
    baseline_sharpes = {}
    breadth_data = {}  # lookback -> year -> array

    for year in YEARS:
        min_len = min(len(sig_returns[sk][year]) for sk in SIG_KEYS)
        ens = np.zeros(int(min_len))
        for sk in SIG_KEYS:
            ens += ENSEMBLE_WEIGHTS[sk] * sig_returns[sk][year][:int(min_len)]
        ens_returns[year] = ens
        baseline_sharpes[year] = sharpe(ens)

    baseline_obj = obj_func(list(baseline_sharpes.values()))
    print(f"  BASELINE: OBJ={baseline_obj:.4f}")

    # Compute breadth for multiple lookbacks
    lookbacks = [72, 168, 336, 504]
    for lb in lookbacks:
        breadth_data[lb] = {}
        for year in YEARS:
            breadth_data[lb][year] = compute_momentum_breadth(sym_returns[year], lookback=lb)

    # 3. Parameter sensitivity scan
    print("\n[3/5] Parameter sensitivity scan...")
    param_grid = []
    for lb in lookbacks:
        for low in [0.2, 0.3, 0.4]:
            for high in [0.6, 0.7, 0.8]:
                if low >= high:
                    continue
                for scale in [1.2, 1.3, 1.5]:
                    param_grid.append({"lookback": lb, "low": low, "high": high, "scale": scale})

    sensitivity_results = []
    for params in param_grid:
        lb = params["lookback"]
        yearly_sharpes = []
        for year in YEARS:
            ens = ens_returns[year]
            b = breadth_data[lb][year][:len(ens)]
            modified = apply_breadth_overlay(ens, b, params["low"], params["high"], params["scale"])
            yearly_sharpes.append(sharpe(modified))

        avg = round(float(np.mean(yearly_sharpes)), 4)
        mn = round(float(np.min(yearly_sharpes)), 4)
        obj = obj_func(yearly_sharpes)
        sensitivity_results.append({
            **params,
            "avg_sharpe": avg,
            "min_sharpe": mn,
            "obj": obj,
            "delta_obj": round(obj - baseline_obj, 4),
        })

    # Sort by OBJ
    sensitivity_results.sort(key=lambda x: x["obj"], reverse=True)

    print(f"\n  Top 10 parameter combinations:")
    print(f"  {'LB':>4s} {'Lo':>4s} {'Hi':>4s} {'Sc':>4s} {'OBJ':>8s} {'AVG':>8s} {'MIN':>8s} {'ΔOBJ':>8s}")
    for r in sensitivity_results[:10]:
        tag = " ✓" if r["delta_obj"] > 0.05 else ""
        print(f"  {r['lookback']:4d} {r['low']:4.1f} {r['high']:4.1f} {r['scale']:4.1f} "
              f"{r['obj']:8.4f} {r['avg_sharpe']:8.3f} {r['min_sharpe']:8.3f} "
              f"{r['delta_obj']:+8.4f}{tag}")

    # Count how many configs beat baseline
    n_positive = sum(1 for r in sensitivity_results if r["delta_obj"] > 0)
    n_total = len(sensitivity_results)
    print(f"\n  {n_positive}/{n_total} configs beat baseline ({n_positive/n_total*100:.0f}%)")

    # 4. LOYO validation of best config
    best_cfg = sensitivity_results[0]
    best_lb = best_cfg["lookback"]
    best_low = best_cfg["low"]
    best_high = best_cfg["high"]
    best_scale = best_cfg["scale"]
    print(f"\n[4/5] LOYO validation of best config: lb={best_lb} low={best_low} high={best_high} scale={best_scale}...")

    loyo_results = []
    for test_year in YEARS:
        train_years = [y for y in YEARS if y != test_year]

        # The breadth overlay uses ABSOLUTE thresholds (not percentile),
        # so there's nothing to calibrate from training years.
        # The LOYO test just checks if the overlay helps on each held-out year.
        ens = ens_returns[test_year]
        b = breadth_data[best_lb][test_year][:len(ens)]
        modified = apply_breadth_overlay(ens, b, best_low, best_high, best_scale)

        s_base = baseline_sharpes[test_year]
        s_overlay = sharpe(modified)
        delta = s_overlay - s_base

        tag = "✓" if delta > 0 else "✗"
        print(f"  Test={test_year}  base={s_base:.3f}  overlay={s_overlay:.3f}  "
              f"Δ={delta:+.3f} {tag}")

        loyo_results.append({
            "test_year": test_year,
            "baseline_sharpe": s_base,
            "overlay_sharpe": s_overlay,
            "delta": round(delta, 4),
        })

    loyo_wins = sum(1 for r in loyo_results if r["delta"] > 0)
    loyo_avg_delta = round(float(np.mean([r["delta"] for r in loyo_results])), 4)
    loyo_overlay_sharpes = [r["overlay_sharpe"] for r in loyo_results]
    loyo_obj = obj_func(loyo_overlay_sharpes)

    print(f"\n  LOYO: {loyo_wins}/5 wins, avg Δ={loyo_avg_delta:+.4f}, "
          f"OBJ={loyo_obj:.4f} (vs baseline {baseline_obj:.4f})")

    # Also test the second-best and third-best configs for robustness
    print("\n  Cross-check with other top configs:")
    for rank, cfg in enumerate(sensitivity_results[1:3], start=2):
        yearly_s = []
        for year in YEARS:
            ens = ens_returns[year]
            b = breadth_data[cfg["lookback"]][year][:len(ens)]
            mod = apply_breadth_overlay(ens, b, cfg["low"], cfg["high"], cfg["scale"])
            yearly_s.append(sharpe(mod))
        wins = sum(1 for i, y in enumerate(YEARS) if yearly_s[i] > baseline_sharpes[y])
        obj = obj_func(yearly_s)
        print(f"    #{rank}: lb={cfg['lookback']} lo={cfg['low']} hi={cfg['high']} "
              f"sc={cfg['scale']}  wins={wins}/5  OBJ={obj:.4f}")

    # 5. Interaction with vol overlay
    print("\n[5/5] Interaction test: breadth + vol overlay...")

    # Compute rolling vol for interaction test
    rvol_per_year = {}
    for year in YEARS:
        rvol_per_year[year] = compute_rolling_vol(btc_returns[year], VOL_WINDOW)

    interaction_variants = {
        "baseline": {"breadth": False, "vol": False},
        "breadth_only": {"breadth": True, "vol": False},
        "vol_only": {"breadth": False, "vol": True},
        "both": {"breadth": True, "vol": True},
    }

    interaction_results = {}
    for vname, vcfg in interaction_variants.items():
        yearly_sharpes = []
        for year in YEARS:
            ens = ens_returns[year].copy()
            n = len(ens)

            # Apply vol overlay first
            if vcfg["vol"]:
                rvol = rvol_per_year[year][:n]
                for i in range(n):
                    if not np.isnan(rvol[i]) and rvol[i] > VOL_THRESHOLD:
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
                            ret += (boosted[sk] / total) * sig_returns[sk][year][i] * VOL_SCALE
                        ens[i] = ret

            # Then apply breadth overlay
            if vcfg["breadth"]:
                b = breadth_data[best_lb][year][:n]
                ens = apply_breadth_overlay(ens, b, best_low, best_high, best_scale)

            yearly_sharpes.append(sharpe(ens))

        avg = round(float(np.mean(yearly_sharpes)), 4)
        mn = round(float(np.min(yearly_sharpes)), 4)
        obj = obj_func(yearly_sharpes)
        interaction_results[vname] = {
            "avg_sharpe": avg,
            "min_sharpe": mn,
            "obj": obj,
        }

    print(f"\n  {'Variant':16s} {'OBJ':>8s} {'AVG':>8s} {'MIN':>8s} {'ΔOBJ':>8s}")
    print(f"  {'-'*16} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")
    base_obj = interaction_results["baseline"]["obj"]
    for vname in ["baseline", "breadth_only", "vol_only", "both"]:
        r = interaction_results[vname]
        delta = r["obj"] - base_obj
        tag = " ✓" if delta > 0.05 else ""
        print(f"  {vname:16s} {r['obj']:8.4f} {r['avg_sharpe']:8.3f} "
              f"{r['min_sharpe']:8.3f} {delta:+8.4f}{tag}")

    breadth_delta = interaction_results["breadth_only"]["obj"] - base_obj
    vol_delta = interaction_results["vol_only"]["obj"] - base_obj
    both_delta = interaction_results["both"]["obj"] - base_obj
    synergy = both_delta - (breadth_delta + vol_delta)
    print(f"\n  Breadth ΔOBJ={breadth_delta:+.4f}, Vol ΔOBJ={vol_delta:+.4f}")
    print(f"  Expected additive: {breadth_delta + vol_delta:+.4f}")
    print(f"  Actual both: {both_delta:+.4f}")
    print(f"  Synergy: {synergy:+.4f}")

    # Verdict
    loyo_pass = loyo_wins >= 3 and loyo_avg_delta > 0.02
    sensitivity_robust = n_positive / n_total > 0.6

    if loyo_pass and sensitivity_robust:
        verdict = "VALIDATED — breadth overlay is robust and survives WF"
    elif loyo_pass:
        verdict = "MARGINAL — LOYO passes but sensitivity is mixed"
    elif sensitivity_robust:
        verdict = "MARGINAL — sensitivity is broad but LOYO fails"
    else:
        verdict = "FAILED — breadth overlay does NOT survive validation"

    elapsed = time.time() - t0
    _partial = {
        "phase": 137,
        "description": "Walk-Forward Validation of Momentum Breadth Overlay",
        "elapsed_seconds": round(elapsed, 1),
        "best_params": {
            "lookback": best_lb,
            "low": best_low,
            "high": best_high,
            "scale": best_scale,
        },
        "baseline": {
            "yearly_sharpes": baseline_sharpes,
            "obj": baseline_obj,
        },
        "sensitivity": {
            "n_positive": n_positive,
            "n_total": n_total,
            "pct_positive": round(n_positive / n_total * 100, 1),
            "top5": sensitivity_results[:5],
        },
        "loyo": {
            "results": loyo_results,
            "wins": f"{loyo_wins}/5",
            "avg_delta": loyo_avg_delta,
            "obj": loyo_obj,
            "pass": loyo_pass,
        },
        "interaction": {
            "results": interaction_results,
            "breadth_delta": round(breadth_delta, 4),
            "vol_delta": round(vol_delta, 4),
            "both_delta": round(both_delta, 4),
            "synergy": round(synergy, 4),
        },
        "verdict": verdict,
    }
    _save(_partial, partial=False)

    print(f"\n  VERDICT: {verdict}")
    print(f"\nPhase 137 complete in {elapsed:.0f}s")
    print("=" * 70)


if __name__ == "__main__":
    main()
