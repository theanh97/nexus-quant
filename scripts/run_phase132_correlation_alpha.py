#!/usr/bin/env python3
"""
Phase 132: Cross-Asset Correlation Alpha
==========================================
Hypothesis: When all coins are highly correlated (correlation regime),
cross-sectional signals (idio momentum) lose alpha because there's less
dispersion to exploit. When correlations are low, cross-sectional alpha
is more available.

Tests:
1. Measure rolling pairwise correlation across our 10 coins
2. Split returns by high-corr vs low-corr regimes
3. Check if ensemble Sharpe differs across regimes
4. Test overlay: reduce idio-momentum weight when corr is high, tilt toward V1/F144
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

OUT_DIR = ROOT / "artifacts" / "phase132"
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
    path = OUT_DIR / "correlation_alpha_report.json"
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
        # Build return matrix for this window
        rets_matrix = []
        for s in syms:
            rets_matrix.append(sym_rets[s][i - window:i])
        rets_matrix = np.array(rets_matrix)  # (n_syms, window)

        # Compute correlation matrix
        corr_mat = np.corrcoef(rets_matrix)

        # Average of off-diagonal elements
        n_syms = len(syms)
        off_diag = []
        for r in range(n_syms):
            for c in range(r + 1, n_syms):
                if not np.isnan(corr_mat[r, c]):
                    off_diag.append(corr_mat[r, c])

        avg_corr[i] = float(np.mean(off_diag)) if off_diag else 0.0

    # Fill initial values
    first_valid = window
    if first_valid < n:
        avg_corr[:first_valid] = avg_corr[first_valid]

    return avg_corr


def main():
    global _partial
    t0 = time.time()
    print("=" * 70)
    print("Phase 132: Cross-Asset Correlation Alpha")
    print("=" * 70)

    # 1. Load data
    print("\n[1/4] Loading data...")
    sym_returns = {}  # year -> sym -> returns
    sig_returns = {sk: {} for sk in SIG_KEYS}

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

        for sk in SIG_KEYS:
            sig_def = SIGNAL_DEFS[sk]
            cost_model = cost_model_from_config({"fee_rate": 0.0005, "slippage_rate": 0.0003, "cost_multiplier": 1.0})
            bt_cfg = BacktestConfig(costs=cost_model)
            strat = make_strategy({"name": sig_def["strategy"], "params": sig_def["params"]})
            engine = BacktestEngine(bt_cfg)
            result = engine.run(dataset, strat)
            sig_returns[sk][year] = np.array(result.returns, dtype=np.float64)

        print(f" OK", flush=True)

    _partial = {"phase": 132}

    # 2. Compute rolling correlation per year
    print("\n[2/4] Computing rolling correlations...")
    corr_per_year = {}
    for year in YEARS:
        corr_per_year[year] = compute_rolling_avg_corr(sym_returns[year], window=168)
        valid = corr_per_year[year][~np.isnan(corr_per_year[year])]
        print(f"  {year}: mean_corr={float(np.mean(valid)):.3f}  "
              f"p25={float(np.percentile(valid, 25)):.3f}  "
              f"p50={float(np.percentile(valid, 50)):.3f}  "
              f"p75={float(np.percentile(valid, 75)):.3f}")

    # 3. Regime analysis: split ensemble returns by high/low corr
    print("\n[3/4] Regime analysis: ensemble Sharpe in high vs low correlation...")
    corr_regime_analysis = {}

    for year in YEARS:
        min_len = min(len(sig_returns[sk][year]) for sk in SIG_KEYS)
        min_len = min(min_len, len(corr_per_year[year]))

        # Compute ensemble returns
        ens = np.zeros(min_len)
        for sk in SIG_KEYS:
            ens += ENSEMBLE_WEIGHTS[sk] * sig_returns[sk][year][:min_len]

        corr = corr_per_year[year][:min_len]
        valid_mask = ~np.isnan(corr)

        # Split by median
        median_corr = float(np.median(corr[valid_mask]))

        high_corr_mask = valid_mask & (corr >= median_corr)
        low_corr_mask = valid_mask & (corr < median_corr)

        ens_high = ens[high_corr_mask]
        ens_low = ens[low_corr_mask]

        s_high = sharpe(ens_high)
        s_low = sharpe(ens_low)
        s_full = sharpe(ens)

        # Also check per-signal
        sig_regime = {}
        for sk in SIG_KEYS:
            sig_r = sig_returns[sk][year][:min_len]
            sig_regime[sk] = {
                "high_corr_sharpe": sharpe(sig_r[high_corr_mask]),
                "low_corr_sharpe": sharpe(sig_r[low_corr_mask]),
            }

        corr_regime_analysis[year] = {
            "median_corr": round(median_corr, 3),
            "ensemble_full": s_full,
            "ensemble_high_corr": s_high,
            "ensemble_low_corr": s_low,
            "delta_low_minus_high": round(s_low - s_high, 4),
            "per_signal": sig_regime,
        }
        tag = "✓" if s_low > s_high else "✗"
        print(f"  {year}: med_corr={median_corr:.3f}  "
              f"high_corr={s_high:.2f}  low_corr={s_low:.2f}  "
              f"Δ={s_low - s_high:+.2f} {tag}")

    # Check consistency
    delta_signs = [1 if corr_regime_analysis[y]["delta_low_minus_high"] > 0 else -1 for y in YEARS]
    n_positive = sum(1 for s in delta_signs if s > 0)
    avg_delta = float(np.mean([corr_regime_analysis[y]["delta_low_minus_high"] for y in YEARS]))
    print(f"\n  Low corr > High corr: {n_positive}/5 years, avg Δ={avg_delta:+.3f}")

    # 4. Test correlation overlay
    print("\n[4/4] Testing correlation-based overlays on ensemble...")

    overlay_configs = {
        "baseline": {"mode": "none"},
        "reduce_high_corr_50pct": {"mode": "reduce", "threshold_pct": 75, "scale": 0.5},
        "reduce_high_corr_70pct": {"mode": "reduce", "threshold_pct": 75, "scale": 0.7},
        "tilt_v1f144_high_corr": {"mode": "tilt", "threshold_pct": 75, "boost_v1": 0.15, "boost_f144": 0.15},
        "reduce_high_corr_90th": {"mode": "reduce", "threshold_pct": 90, "scale": 0.5},
        "flat_extreme_corr": {"mode": "reduce", "threshold_pct": 95, "scale": 0.0},
    }

    overlay_results = {}
    for oname, ocfg in overlay_configs.items():
        yearly_sharpes = []
        yearly_detail = {}

        for year in YEARS:
            min_len = min(len(sig_returns[sk][year]) for sk in SIG_KEYS)
            min_len = min(min_len, len(corr_per_year[year]))
            corr = corr_per_year[year][:min_len]
            valid = corr[~np.isnan(corr)]

            if ocfg["mode"] == "none":
                ens = np.zeros(min_len)
                for sk in SIG_KEYS:
                    ens += ENSEMBLE_WEIGHTS[sk] * sig_returns[sk][year][:min_len]
            elif ocfg["mode"] == "reduce":
                threshold = float(np.percentile(valid, ocfg["threshold_pct"]))
                ens = np.zeros(min_len)
                for i in range(min_len):
                    ret = 0.0
                    for sk in SIG_KEYS:
                        ret += ENSEMBLE_WEIGHTS[sk] * sig_returns[sk][year][i]
                    if not np.isnan(corr[i]) and corr[i] >= threshold:
                        ret *= ocfg["scale"]
                    ens[i] = ret
            elif ocfg["mode"] == "tilt":
                threshold = float(np.percentile(valid, ocfg["threshold_pct"]))
                ens = np.zeros(min_len)
                for i in range(min_len):
                    if not np.isnan(corr[i]) and corr[i] >= threshold:
                        # Tilt toward V1 and F144, away from idio-momentum
                        w = dict(ENSEMBLE_WEIGHTS)
                        idio_reduce = (ocfg["boost_v1"] + ocfg["boost_f144"]) / 2
                        w["v1"] = min(0.60, w["v1"] + ocfg["boost_v1"])
                        w["f144"] = min(0.60, w["f144"] + ocfg["boost_f144"])
                        w["i460bw168"] = max(0.05, w["i460bw168"] - idio_reduce)
                        w["i415bw216"] = max(0.05, w["i415bw216"] - idio_reduce)
                        total = sum(w.values())
                        for sk in SIG_KEYS:
                            ens[i] += (w[sk] / total) * sig_returns[sk][year][i]
                    else:
                        for sk in SIG_KEYS:
                            ens[i] += ENSEMBLE_WEIGHTS[sk] * sig_returns[sk][year][i]

            s = sharpe(ens)
            yearly_sharpes.append(s)
            yearly_detail[year] = s

        avg = round(float(np.mean(yearly_sharpes)), 4)
        mn = round(float(np.min(yearly_sharpes)), 4)
        obj = obj_func(yearly_sharpes)
        overlay_results[oname] = {
            "yearly": yearly_detail,
            "avg_sharpe": avg,
            "min_sharpe": mn,
            "obj": obj,
        }

    baseline_obj = overlay_results["baseline"]["obj"]
    print(f"\n  {'Variant':28s} {'OBJ':>8s} {'AVG':>8s} {'MIN':>8s} {'ΔOBJ':>8s}")
    print(f"  {'-'*28} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")
    for oname in sorted(overlay_results, key=lambda k: overlay_results[k]["obj"], reverse=True):
        r = overlay_results[oname]
        delta = r["obj"] - baseline_obj
        tag = " ✓" if delta > 0.05 else ""
        print(f"  {oname:28s} {r['obj']:8.4f} {r['avg_sharpe']:8.3f} {r['min_sharpe']:8.3f} {delta:+8.4f}{tag}")

    best_name = max(overlay_results, key=lambda k: overlay_results[k]["obj"])
    best = overlay_results[best_name]
    improvement = best["obj"] - baseline_obj

    if improvement < 0.03:
        verdict = "NO IMPROVEMENT — correlation overlay does not help the ensemble"
    elif improvement < 0.10:
        verdict = f"MARGINAL — {best_name} shows +{improvement:.3f} OBJ, needs WF validation"
    else:
        verdict = f"POTENTIAL — {best_name} adds +{improvement:.3f} OBJ"

    elapsed = time.time() - t0
    _partial = {
        "phase": 132,
        "description": "Cross-Asset Correlation Alpha",
        "elapsed_seconds": round(elapsed, 1),
        "regime_analysis": corr_regime_analysis,
        "consistency": f"{n_positive}/5 low>high, avg Δ={avg_delta:+.3f}",
        "overlays": overlay_results,
        "best": {"name": best_name, **best},
        "improvement": round(improvement, 4),
        "verdict": verdict,
    }
    _save(_partial, partial=False)

    print(f"\n  VERDICT: {verdict}")
    print(f"\nPhase 132 complete in {elapsed:.0f}s")
    print("=" * 70)


if __name__ == "__main__":
    main()
