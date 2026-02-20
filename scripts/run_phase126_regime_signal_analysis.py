#!/usr/bin/env python3
"""
Phase 126: Regime-Conditional Signal Analysis
================================================
Analyzes how each signal performs under different market regimes:
1. Detect regimes from BTC returns (bull/bear/sideways/volatile)
2. Compute per-signal Sharpe within each regime
3. Identify which signals are regime-dependent vs all-weather
4. Test whether a regime-aware ensemble outperforms static weights

Uses 2021-2025 data. No new configs needed — reuses cached signal returns.
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

# Timeout
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

# Full 5-year backtest
FULL_RANGE = ("2021-02-01", "2025-12-31")

OUT_DIR = ROOT / "artifacts" / "phase126"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def sharpe(rets: np.ndarray) -> float:
    if len(rets) < 50:
        return 0.0
    mu = float(np.mean(rets)) * 8760
    sd = float(np.std(rets)) * np.sqrt(8760)
    return round(mu / sd, 4) if sd > 1e-12 else 0.0


def _save(data: dict, partial: bool = False) -> None:
    data["partial"] = partial
    data["timestamp"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    path = OUT_DIR / "regime_signal_analysis.json"
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"  → Saved: {path}")


def detect_regimes(btc_returns: np.ndarray, lookback: int = 720) -> np.ndarray:
    """
    Classify each bar into a regime based on rolling BTC momentum + vol.

    Regimes:
    0 = BULL (positive rolling return, low vol)
    1 = BEAR (negative rolling return, any vol)
    2 = SIDEWAYS (small absolute return, low vol)
    3 = VOLATILE (any direction, high vol)
    """
    n = len(btc_returns)
    regimes = np.zeros(n, dtype=np.int32)

    for i in range(lookback, n):
        window = btc_returns[i - lookback:i]
        cum_ret = float(np.sum(window))  # approx total return
        vol = float(np.std(window)) * np.sqrt(8760)

        vol_median = 0.6  # ~60% annual vol is median for crypto

        if vol > vol_median * 1.5:
            regimes[i] = 3  # VOLATILE
        elif cum_ret > 0.1:
            regimes[i] = 0  # BULL
        elif cum_ret < -0.1:
            regimes[i] = 1  # BEAR
        else:
            regimes[i] = 2  # SIDEWAYS

    # First lookback bars = SIDEWAYS (no regime info)
    regimes[:lookback] = 2
    return regimes


REGIME_NAMES = {0: "BULL", 1: "BEAR", 2: "SIDEWAYS", 3: "VOLATILE"}


def main():
    global _partial
    t0 = time.time()
    print("=" * 70)
    print("Phase 126: Regime-Conditional Signal Analysis")
    print("=" * 70)

    # 1. Load full 5-year data for all signals
    print("\n[1/4] Running full-period backtests for each signal...")
    start, end = FULL_RANGE
    sig_returns_full = {}
    min_len = float("inf")

    for sk in SIG_KEYS:
        sig_def = SIGNAL_DEFS[sk]
        cfg = {
            "data": {
                "provider": "binance_rest_v1", "symbols": SYMBOLS,
                "start": start, "end": end, "bar_interval": "1h",
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
        rets = np.array(result.returns, dtype=np.float64)
        sig_returns_full[sk] = rets
        min_len = min(min_len, len(rets))
        print(f"  {sk}: {len(rets)} bars, Sharpe={sharpe(rets):.3f}")

    # Also get BTC returns for regime detection
    # Use dataset from last backtest (same timeline)
    btc_rets = []
    for i in range(1, len(dataset.timeline)):
        c0 = dataset.close("BTCUSDT", i - 1)
        c1 = dataset.close("BTCUSDT", i)
        btc_rets.append((c1 / c0 - 1.0) if c0 > 0 else 0.0)
    btc_rets = np.array(btc_rets, dtype=np.float64)

    # Trim all to same length
    min_len = min(min_len, len(btc_rets))
    for sk in SIG_KEYS:
        sig_returns_full[sk] = sig_returns_full[sk][:int(min_len)]
    btc_rets = btc_rets[:int(min_len)]

    _partial = {"phase": 126, "n_bars": int(min_len)}

    # 2. Detect regimes
    print(f"\n[2/4] Detecting market regimes (720h lookback)...")
    regimes = detect_regimes(btc_rets, lookback=720)

    regime_counts = {}
    for r in range(4):
        count = int(np.sum(regimes == r))
        pct = count / len(regimes) * 100
        regime_counts[REGIME_NAMES[r]] = {"count": count, "pct": round(pct, 1)}
        print(f"  {REGIME_NAMES[r]:10s}: {count:5d} bars ({pct:.1f}%)")

    _partial["regime_distribution"] = regime_counts

    # 3. Per-signal per-regime Sharpe
    print(f"\n[3/4] Per-signal per-regime Sharpe...")
    regime_sharpes = {}  # sig_key -> regime_name -> sharpe

    print(f"\n  {'Signal':12s}", end="")
    for r in range(4):
        print(f"  {REGIME_NAMES[r]:>10s}", end="")
    print(f"  {'FULL':>10s}")
    print(f"  {'-'*12}", end="")
    for _ in range(5):
        print(f"  {'-'*10}", end="")
    print()

    for sk in SIG_KEYS:
        regime_sharpes[sk] = {}
        rets = sig_returns_full[sk]

        print(f"  {sk:12s}", end="")
        for r in range(4):
            mask = regimes == r
            regime_rets = rets[mask]
            s = sharpe(regime_rets)
            regime_sharpes[sk][REGIME_NAMES[r]] = s
            color = "✓" if s > 0.5 else ("⚠" if s > 0 else "✗")
            print(f"  {s:9.3f}{color}", end="")

        full_s = sharpe(rets)
        regime_sharpes[sk]["FULL"] = full_s
        print(f"  {full_s:9.3f}")

    # Ensemble
    ens_rets = np.zeros(int(min_len))
    for sk in SIG_KEYS:
        ens_rets += ENSEMBLE_WEIGHTS[sk] * sig_returns_full[sk]

    regime_sharpes["ENSEMBLE"] = {}
    print(f"  {'ENSEMBLE':12s}", end="")
    for r in range(4):
        mask = regimes == r
        s = sharpe(ens_rets[mask])
        regime_sharpes["ENSEMBLE"][REGIME_NAMES[r]] = s
        print(f"  {s:9.3f}{'✓' if s > 0.5 else '⚠'}", end="")
    print(f"  {sharpe(ens_rets):9.3f}")
    regime_sharpes["ENSEMBLE"]["FULL"] = sharpe(ens_rets)

    _partial["regime_sharpes"] = regime_sharpes

    # 4. Signal regime-dependence analysis
    print(f"\n[4/4] Signal regime-dependence analysis...")
    dependence = {}

    for sk in SIG_KEYS:
        sharpes = [regime_sharpes[sk][REGIME_NAMES[r]] for r in range(4)]
        avg = float(np.mean(sharpes))
        std = float(np.std(sharpes))
        min_s = float(np.min(sharpes))
        max_s = float(np.max(sharpes))
        cv = std / abs(avg) * 100 if abs(avg) > 0.01 else 0

        # Best and worst regime
        best_regime = REGIME_NAMES[int(np.argmax(sharpes))]
        worst_regime = REGIME_NAMES[int(np.argmin(sharpes))]

        # Classification
        if cv < 30 and min_s > 0:
            classification = "ALL-WEATHER"
        elif cv < 50:
            classification = "MODERATE_DEPENDENCE"
        else:
            classification = "REGIME_DEPENDENT"

        dependence[sk] = {
            "classification": classification,
            "regime_cv": round(cv, 1),
            "best_regime": best_regime,
            "worst_regime": worst_regime,
            "avg_across_regimes": round(avg, 4),
            "std_across_regimes": round(std, 4),
        }
        print(f"  {sk:12s}: {classification:22s}  CV={cv:5.1f}%  best={best_regime:10s}  worst={worst_regime:10s}")

    _partial["dependence"] = dependence

    # Test: regime-aware ensemble (use best-performing weights per regime)
    print(f"\n  Testing regime-aware dynamic ensemble...")

    # For each regime, find best weight combination
    regime_optimal = {}
    for r in range(4):
        mask = regimes == r
        regime_total = int(np.sum(mask))
        if regime_total < 100:
            regime_optimal[REGIME_NAMES[r]] = dict(ENSEMBLE_WEIGHTS)
            continue

        best_obj = -999
        best_w = None
        for w1 in range(5, 56, 10):
            for w2 in range(5, 56 - w1, 10):
                for w3 in range(5, 56 - w1 - w2, 10):
                    w4 = 100 - w1 - w2 - w3
                    if w4 < 5 or w4 > 55:
                        continue
                    ws = {
                        SIG_KEYS[0]: w1/100, SIG_KEYS[1]: w2/100,
                        SIG_KEYS[2]: w3/100, SIG_KEYS[3]: w4/100,
                    }
                    ens_r = np.zeros(regime_total)
                    idx = 0
                    for i in range(len(mask)):
                        if mask[i]:
                            for sk in SIG_KEYS:
                                ens_r[idx] += ws[sk] * sig_returns_full[sk][i]
                            idx += 1
                    s = sharpe(ens_r)
                    if s > best_obj:
                        best_obj = s
                        best_w = ws

        regime_optimal[REGIME_NAMES[r]] = {k: round(v, 4) for k, v in (best_w or ENSEMBLE_WEIGHTS).items()}
        print(f"    {REGIME_NAMES[r]:10s}: Sharpe={best_obj:.3f}  weights={regime_optimal[REGIME_NAMES[r]]}")

    # Simulate: apply regime-optimal weights
    dynamic_rets = np.zeros(int(min_len))
    for i in range(int(min_len)):
        r = int(regimes[i])
        w = regime_optimal[REGIME_NAMES[r]]
        for sk in SIG_KEYS:
            dynamic_rets[i] += w[sk] * sig_returns_full[sk][i]

    dynamic_sharpe = sharpe(dynamic_rets)
    static_sharpe = sharpe(ens_rets)
    lift = dynamic_sharpe - static_sharpe

    print(f"\n  Static ensemble Sharpe:  {static_sharpe:.3f}")
    print(f"  Dynamic ensemble Sharpe: {dynamic_sharpe:.3f}")
    print(f"  Lift: {lift:+.3f}")

    if lift < 0.1:
        dynamic_verdict = "NEGLIGIBLE — regime-aware switching adds no meaningful alpha"
    elif lift < 0.3:
        dynamic_verdict = "MODEST — small lift but adds complexity and look-ahead bias risk"
    else:
        dynamic_verdict = "SIGNIFICANT — worth investigating further (but beware look-ahead)"

    _partial["regime_optimal_weights"] = regime_optimal
    _partial["dynamic_vs_static"] = {
        "static_sharpe": static_sharpe,
        "dynamic_sharpe": dynamic_sharpe,
        "lift": round(lift, 4),
        "verdict": dynamic_verdict,
    }

    # Final verdict
    elapsed = time.time() - t0
    _partial["elapsed_seconds"] = round(elapsed, 1)

    # Overall assessment
    all_weather_count = sum(1 for d in dependence.values() if d["classification"] == "ALL-WEATHER")
    _partial["overall_assessment"] = {
        "all_weather_signals": all_weather_count,
        "regime_dependent_signals": len(SIG_KEYS) - all_weather_count,
        "diversification_benefit": "HIGH" if all_weather_count >= 2 else "MODERATE",
        "recommendation": "KEEP STATIC ENSEMBLE" if lift < 0.2 else "INVESTIGATE DYNAMIC SWITCHING",
    }

    _save(_partial, partial=False)

    print(f"\n{'='*70}")
    print(f"Phase 126 complete in {elapsed:.0f}s")
    print(f"  All-weather signals: {all_weather_count}/{len(SIG_KEYS)}")
    print(f"  Dynamic verdict: {dynamic_verdict}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
