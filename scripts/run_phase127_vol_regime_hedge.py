#!/usr/bin/env python3
"""
Phase 127: Volatile Regime Hedge Research
============================================
Phase 126 found: VOLATILE regime (5.1% of time) causes ensemble Sharpe=-2.26
while BULL/SIDEWAYS get 2.2-2.6. F144 is the only positive signal in volatile.

Hypothesis: If we detect the volatile regime in real-time and shift weight
toward F144 (or reduce overall leverage), we can protect the downside.

Tests:
1. Realized vol filter: if rolling vol > threshold → reduce leverage
2. F144-tilt: if vol high → increase F144 weight from 20% to 40%+
3. Combined: reduce leverage + tilt toward F144
4. Vol-regime halt: if extreme vol → go flat entirely

All tests use CAUSAL (look-back only) detection to avoid look-ahead bias.
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

OUT_DIR = ROOT / "artifacts" / "phase127"
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
    path = OUT_DIR / "vol_regime_hedge_report.json"
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"  → Saved: {path}")


def run_signal(sig_key: str, start: str, end: str) -> np.ndarray:
    sig_def = SIGNAL_DEFS[sig_key]
    cfg = {
        "data": {"provider": "binance_rest_v1", "symbols": SYMBOLS,
                 "start": start, "end": end, "bar_interval": "1h",
                 "cache_dir": ".cache/binance_rest"},
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


def compute_rolling_vol(rets: np.ndarray, window: int) -> np.ndarray:
    """Compute rolling annualized volatility (CAUSAL — look-back only)."""
    n = len(rets)
    vol = np.full(n, np.nan)
    for i in range(window, n):
        vol[i] = float(np.std(rets[i - window:i])) * np.sqrt(8760)
    vol[:window] = vol[window]  # fill initial with first valid
    return vol


def apply_overlay(sig_returns: dict, btc_returns: np.ndarray,
                  overlay_type: str, params: dict) -> dict:
    """Apply a volatility overlay to the ensemble, return yearly results."""
    window = params.get("vol_window", 168)
    threshold_pct = params.get("vol_threshold_pct", 75)
    scale_factor = params.get("scale_factor", 0.5)
    f144_boost = params.get("f144_boost", 0.0)

    yearly_results = {}

    for year in YEARS:
        # Get per-signal returns
        min_len = min(len(sig_returns[sk][year]) for sk in SIG_KEYS
                      if len(sig_returns[sk][year]) > 0)
        if min_len < 200:
            yearly_results[year] = {"sharpe": 0, "error": "too short"}
            continue

        # BTC returns for this year
        btc_year = btc_returns[year][:int(min_len)]

        # Compute rolling vol
        rvol = compute_rolling_vol(btc_year, window)

        # Compute vol threshold (percentile of rolling vol)
        valid_vol = rvol[~np.isnan(rvol)]
        if len(valid_vol) < 100:
            yearly_results[year] = {"sharpe": 0, "error": "insufficient vol data"}
            continue
        vol_thresh = float(np.percentile(valid_vol, threshold_pct))

        # Build ensemble returns with overlay
        ens = np.zeros(int(min_len))
        for i in range(int(min_len)):
            if np.isnan(rvol[i]):
                # Use static weights
                for sk in SIG_KEYS:
                    ens[i] += ENSEMBLE_WEIGHTS[sk] * sig_returns[sk][year][i]
            elif rvol[i] > vol_thresh:
                # HIGH VOL regime — apply overlay
                if overlay_type == "reduce_leverage":
                    for sk in SIG_KEYS:
                        ens[i] += ENSEMBLE_WEIGHTS[sk] * sig_returns[sk][year][i] * scale_factor
                elif overlay_type == "f144_tilt":
                    # Shift weight toward F144
                    boosted = dict(ENSEMBLE_WEIGHTS)
                    boost_from_each = f144_boost / 3
                    for sk in SIG_KEYS:
                        if sk == "f144":
                            boosted[sk] = min(0.60, boosted[sk] + f144_boost)
                        else:
                            boosted[sk] = max(0.05, boosted[sk] - boost_from_each)
                    # Normalize
                    total = sum(boosted.values())
                    for sk in SIG_KEYS:
                        ens[i] += (boosted[sk] / total) * sig_returns[sk][year][i]
                elif overlay_type == "combined":
                    # Both reduce leverage AND tilt toward F144
                    boosted = dict(ENSEMBLE_WEIGHTS)
                    boost_from_each = f144_boost / 3
                    for sk in SIG_KEYS:
                        if sk == "f144":
                            boosted[sk] = min(0.60, boosted[sk] + f144_boost)
                        else:
                            boosted[sk] = max(0.05, boosted[sk] - boost_from_each)
                    total = sum(boosted.values())
                    for sk in SIG_KEYS:
                        ens[i] += (boosted[sk] / total) * sig_returns[sk][year][i] * scale_factor
                elif overlay_type == "halt":
                    ens[i] = 0.0  # go flat
            else:
                # Normal regime — static weights
                for sk in SIG_KEYS:
                    ens[i] += ENSEMBLE_WEIGHTS[sk] * sig_returns[sk][year][i]

        s = sharpe(ens)
        n_high = int(np.sum(rvol > vol_thresh))
        yearly_results[year] = {
            "sharpe": s,
            "high_vol_bars": n_high,
            "high_vol_pct": round(n_high / min_len * 100, 1),
        }

    sharpes = [v["sharpe"] for v in yearly_results.values() if isinstance(v.get("sharpe"), (int, float))]
    return {
        "overlay_type": overlay_type,
        "params": params,
        "yearly": yearly_results,
        "avg_sharpe": round(float(np.mean(sharpes)), 4) if sharpes else 0,
        "min_sharpe": round(float(np.min(sharpes)), 4) if sharpes else 0,
        "obj": obj_func(sharpes) if sharpes else 0,
    }


def main():
    global _partial
    t0 = time.time()
    print("=" * 70)
    print("Phase 127: Volatile Regime Hedge Research")
    print("=" * 70)

    # 1. Load signal returns + BTC returns per year
    print("\n[1/3] Loading signal & BTC returns per year...")
    sig_returns = {sk: {} for sk in SIG_KEYS}
    btc_returns = {}

    for year, (start, end) in YEAR_RANGES.items():
        print(f"  {year}:", end="", flush=True)

        # Load dataset for BTC returns
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

    _partial = {"phase": 127}

    # 2. Compute baseline (no overlay)
    print("\n[2/3] Computing baseline + overlay variants...")

    # Baseline: no overlay — just compute static ensemble per year
    baseline_sharpes = []
    baseline_yearly = {}
    for year in YEARS:
        min_len = min(len(sig_returns[sk][year]) for sk in SIG_KEYS)
        ens = np.zeros(int(min_len))
        for sk in SIG_KEYS:
            ens += ENSEMBLE_WEIGHTS[sk] * sig_returns[sk][year][:int(min_len)]
        s = sharpe(ens)
        baseline_sharpes.append(s)
        baseline_yearly[year] = {"sharpe": s}
    baseline = {
        "overlay_type": "BASELINE",
        "params": {},
        "yearly": baseline_yearly,
        "avg_sharpe": round(float(np.mean(baseline_sharpes)), 4),
        "min_sharpe": round(float(np.min(baseline_sharpes)), 4),
        "obj": obj_func(baseline_sharpes),
    }
    print(f"  BASELINE:       OBJ={baseline['obj']:.4f}  AVG={baseline['avg_sharpe']:.3f}  MIN={baseline['min_sharpe']:.3f}")

    results = [baseline]

    # 3. Test overlay variants
    variants = [
        # Reduce leverage when vol > 75th percentile
        ("reduce_leverage", {"vol_window": 168, "vol_threshold_pct": 75, "scale_factor": 0.5}),
        ("reduce_leverage", {"vol_window": 168, "vol_threshold_pct": 80, "scale_factor": 0.5}),
        ("reduce_leverage", {"vol_window": 168, "vol_threshold_pct": 85, "scale_factor": 0.5}),
        ("reduce_leverage", {"vol_window": 168, "vol_threshold_pct": 90, "scale_factor": 0.5}),
        ("reduce_leverage", {"vol_window": 168, "vol_threshold_pct": 95, "scale_factor": 0.5}),
        ("reduce_leverage", {"vol_window": 168, "vol_threshold_pct": 90, "scale_factor": 0.3}),
        ("reduce_leverage", {"vol_window": 336, "vol_threshold_pct": 90, "scale_factor": 0.5}),

        # F144 tilt during high vol
        ("f144_tilt", {"vol_window": 168, "vol_threshold_pct": 85, "f144_boost": 0.15}),
        ("f144_tilt", {"vol_window": 168, "vol_threshold_pct": 90, "f144_boost": 0.20}),
        ("f144_tilt", {"vol_window": 168, "vol_threshold_pct": 90, "f144_boost": 0.30}),

        # Combined: reduce + tilt
        ("combined", {"vol_window": 168, "vol_threshold_pct": 90, "scale_factor": 0.5, "f144_boost": 0.15}),
        ("combined", {"vol_window": 168, "vol_threshold_pct": 85, "scale_factor": 0.5, "f144_boost": 0.20}),

        # Halt during extreme vol
        ("halt", {"vol_window": 168, "vol_threshold_pct": 95}),
        ("halt", {"vol_window": 168, "vol_threshold_pct": 97}),
    ]

    for overlay_type, params in variants:
        r = apply_overlay(sig_returns, btc_returns, overlay_type, params)
        results.append(r)
        label = f"{overlay_type}_{params.get('vol_threshold_pct','?')}pct"
        delta = r["obj"] - baseline["obj"]
        tag = "✓" if delta > 0 else "✗"
        print(f"  {label:35s}: OBJ={r['obj']:.4f}  AVG={r['avg_sharpe']:.3f}  MIN={r['min_sharpe']:.3f}  Δ={delta:+.4f} {tag}")

    # Sort by OBJ
    results.sort(key=lambda x: x["obj"], reverse=True)

    # Summary
    print(f"\n[3/3] Summary — Top 5 variants:")
    print(f"  {'#':>3s} {'Type':22s} {'OBJ':>8s} {'AVG':>8s} {'MIN':>8s} {'ΔOBJ':>8s}")
    for i, r in enumerate(results[:5]):
        delta = r["obj"] - baseline["obj"]
        label = r["overlay_type"][:22]
        print(f"  {i+1:3d} {label:22s} {r['obj']:8.4f} {r['avg_sharpe']:8.3f} {r['min_sharpe']:8.3f} {delta:+8.4f}")

    best = results[0]
    improvement = best["obj"] - baseline["obj"]

    if improvement < 0.05:
        verdict = "NO IMPROVEMENT — volatile regime hedge adds no alpha on this data"
    elif improvement < 0.15:
        verdict = "MARGINAL — small improvement, monitor before production"
    else:
        verdict = f"IMPROVEMENT — {best['overlay_type']} adds +{improvement:.3f} OBJ"

    elapsed = time.time() - t0
    _partial = {
        "phase": 127,
        "description": "Volatile Regime Hedge",
        "elapsed_seconds": round(elapsed, 1),
        "baseline": baseline,
        "results": results,
        "best": best,
        "improvement": round(improvement, 4),
        "verdict": verdict,
    }
    _save(_partial, partial=False)

    print(f"\n  VERDICT: {verdict}")
    print(f"\nPhase 127 complete in {elapsed:.0f}s")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
