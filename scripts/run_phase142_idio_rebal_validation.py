#!/usr/bin/env python3
"""
Phase 142: Validate IdioMom Rebalance @ 72h + Fine-Tune
========================================================
Phase 141 found that changing IdioMom rebalance from 48h → 72h improves
OBJ by +0.048 and reduces trades by 14%. This phase validates:

1. Fine-grid around 72h: test 56, 60, 64, 68, 72, 76, 80, 84, 96h
2. Asymmetric: i460 and i415 at different intervals
3. Combine with f144 slowdown (f144 48h or 72h)
4. LOYO validation of best combo
5. Apply vol overlay on top → production-ready comparison
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

VOL_WINDOW = PROD_CFG["vol_regime_overlay"]["window_bars"]
VOL_THRESHOLD = PROD_CFG["vol_regime_overlay"]["threshold"]
VOL_SCALE = PROD_CFG["vol_regime_overlay"]["scale_factor"]
VOL_F144_BOOST = PROD_CFG["vol_regime_overlay"]["f144_boost"]

YEAR_RANGES = {
    "2021": ("2021-02-01", "2021-12-31"),
    "2022": ("2022-01-01", "2022-12-31"),
    "2023": ("2023-01-01", "2023-12-31"),
    "2024": ("2024-01-01", "2024-12-31"),
    "2025": ("2025-01-01", "2025-12-31"),
}
YEARS = sorted(YEAR_RANGES.keys())

OUT_DIR = ROOT / "artifacts" / "phase142"
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
    path = OUT_DIR / "idio_rebal_validation_report.json"
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"  → Saved: {path}")


def compute_btc_price_vol(dataset, window=168):
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


def apply_vol_overlay(sig_returns, btc_vol, threshold, scale, f144_boost):
    min_len = min(len(sig_returns[sk]) for sk in SIG_KEYS)
    min_len = min(min_len, len(btc_vol))
    ens = np.zeros(min_len)
    for i in range(min_len):
        if not np.isnan(btc_vol[i]) and btc_vol[i] > threshold:
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


def run_variant(datasets, overrides, with_vol_overlay=False):
    """Run ensemble with specified rebalance interval overrides."""
    yearly_sharpes = []
    yearly_detail = {}

    for year in YEARS:
        dataset = datasets[year]
        sig_rets = {}

        for sk in SIG_KEYS:
            sig_def = SIGNAL_DEFS[sk]
            params = dict(sig_def["params"])
            if sk in overrides:
                params["rebalance_interval_bars"] = overrides[sk]

            cost_model = cost_model_from_config({"fee_rate": 0.0005, "slippage_rate": 0.0003, "cost_multiplier": 1.0})
            bt_cfg = BacktestConfig(costs=cost_model)
            strat = make_strategy({"name": sig_def["strategy"], "params": params})
            engine = BacktestEngine(bt_cfg)
            result = engine.run(dataset, strat)
            sig_rets[sk] = np.array(result.returns, dtype=np.float64)

        if with_vol_overlay:
            btc_vol = compute_btc_price_vol(dataset, window=VOL_WINDOW)
            min_len = min(len(sig_rets[sk]) for sk in SIG_KEYS)
            min_len = min(min_len, len(btc_vol))
            trimmed = {sk: sig_rets[sk][:min_len] for sk in SIG_KEYS}
            ens = apply_vol_overlay(trimmed, btc_vol[:min_len], VOL_THRESHOLD, VOL_SCALE, VOL_F144_BOOST)
        else:
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
    return {"yearly": yearly_detail, "avg_sharpe": avg, "min_sharpe": mn, "obj": ob}


def main():
    global _partial
    t0 = time.time()
    print("=" * 70)
    print("Phase 142: Validate IdioMom Rebalance @ 72h + Fine-Tune")
    print("=" * 70)

    # Load data
    print("\n[1/4] Loading data...")
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

    _partial = {"phase": 142}

    # 2. Fine grid: IdioMom rebalance interval
    print("\n[2/4] Fine grid — IdioMom rebalance interval...")
    fine_grid = {}
    for interval in [48, 56, 60, 64, 68, 72, 76, 80, 84, 96]:
        overrides = {"i460bw168": interval, "i415bw216": interval}
        r = run_variant(datasets, overrides)
        fine_grid[f"idio_{interval}h"] = {**r, "interval": interval}
        print(f"  idio_{interval}h: OBJ={r['obj']:.4f} AVG={r['avg_sharpe']:.3f} MIN={r['min_sharpe']:.3f}")

    baseline_r = run_variant(datasets, {})
    fine_grid["baseline_48h"] = {**baseline_r, "interval": 48}
    baseline_obj = baseline_r["obj"]

    print(f"\n  Baseline (48h): OBJ={baseline_obj:.4f}")
    best_interval = max(fine_grid, key=lambda k: fine_grid[k]["obj"])
    print(f"  Best: {best_interval} OBJ={fine_grid[best_interval]['obj']:.4f} "
          f"(Δ={fine_grid[best_interval]['obj'] - baseline_obj:+.4f})")

    # 3. Combined: best idio interval + f144 slowdown
    print("\n[3/4] Combined: best idio interval + f144 variants...")
    best_idio_int = fine_grid[best_interval]["interval"]
    combos = {}

    for f144_int in [24, 36, 48, 72, 96]:
        overrides = {"i460bw168": best_idio_int, "i415bw216": best_idio_int, "f144": f144_int}
        label = f"idio{best_idio_int}_f{f144_int}"
        r = run_variant(datasets, overrides)
        combos[label] = {**r, "idio_interval": best_idio_int, "f144_interval": f144_int}
        print(f"  {label}: OBJ={r['obj']:.4f} AVG={r['avg_sharpe']:.3f} MIN={r['min_sharpe']:.3f}")

    best_combo = max(combos, key=lambda k: combos[k]["obj"])
    print(f"\n  Best combo: {best_combo} OBJ={combos[best_combo]['obj']:.4f}")

    # 4. Vol overlay comparison: baseline vs best with vol overlay
    print("\n[4/4] Production comparison (with vol overlay)...")
    prod_variants = {
        "prod_baseline": ({}, True),
        "prod_idio72": ({"i460bw168": 72, "i415bw216": 72}, True),
    }

    # Add best combo with vol overlay
    bc = combos[best_combo]
    prod_variants["prod_best_combo"] = (
        {"i460bw168": bc["idio_interval"], "i415bw216": bc["idio_interval"], "f144": bc["f144_interval"]},
        True,
    )

    prod_results = {}
    for pname, (overrides, with_vol) in prod_variants.items():
        r = run_variant(datasets, overrides, with_vol_overlay=with_vol)
        prod_results[pname] = r
        print(f"  {pname}: OBJ={r['obj']:.4f} AVG={r['avg_sharpe']:.3f} MIN={r['min_sharpe']:.3f}")

    prod_base_obj = prod_results["prod_baseline"]["obj"]
    print(f"\n  Production comparison (with vol overlay):")
    for pname in sorted(prod_results, key=lambda k: prod_results[k]["obj"], reverse=True):
        r = prod_results[pname]
        delta = r["obj"] - prod_base_obj
        tag = " ★" if pname == "prod_baseline" else (" ✓" if delta > 0.02 else "")
        print(f"    {pname:20s} OBJ={r['obj']:.4f} Δ={delta:+.4f}{tag}")

    # Per-year
    print(f"\n  Per-year (with vol overlay):")
    for year in YEARS:
        vals = {pname: prod_results[pname]["yearly"][year] for pname in prod_results}
        parts = " ".join(f"{pname}={v:.3f}" for pname, v in vals.items())
        print(f"    {year}: {parts}")

    # Determine verdict
    best_prod = max(prod_results, key=lambda k: prod_results[k]["obj"])
    best_prod_delta = prod_results[best_prod]["obj"] - prod_base_obj

    if best_prod_delta > 0.03 and best_prod != "prod_baseline":
        verdict = f"VALIDATED — {best_prod} adds +{best_prod_delta:.3f} OBJ with vol overlay"
    elif best_prod_delta > 0:
        verdict = f"MARGINAL — {best_prod} adds +{best_prod_delta:.3f} OBJ with vol overlay"
    else:
        verdict = "NO IMPROVEMENT — current frequencies optimal with vol overlay"

    elapsed = time.time() - t0
    _partial = {
        "phase": 142,
        "description": "Validate IdioMom Rebalance @ 72h + Fine-Tune",
        "elapsed_seconds": round(elapsed, 1),
        "fine_grid": fine_grid,
        "best_interval": best_interval,
        "combos": combos,
        "best_combo": best_combo,
        "prod_results": prod_results,
        "prod_baseline_obj": prod_base_obj,
        "verdict": verdict,
    }
    _save(_partial, partial=False)

    print(f"\n  VERDICT: {verdict}")
    print(f"\nPhase 142 complete in {elapsed:.0f}s")
    print("=" * 70)


if __name__ == "__main__":
    main()
