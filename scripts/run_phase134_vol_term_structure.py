#!/usr/bin/env python3
"""
Phase 134: Realized Vol Term Structure Alpha
==============================================
Hypothesis: The RATIO of short-term to long-term realized vol captures
regime transitions orthogonal to vol level:
- Inverted (short > long): panic/crash, margin calls → reduce exposure
- Normal (short < long): calm trending → full exposure
- Steep normal (short << long): complacency → maybe boost exposure

This is DIFFERENT from the vol overlay (Phase 127-128) which uses absolute
vol level. Vol term structure captures the SHAPE / inflection of vol.

Tests:
1. Compute vol ratio (24h/168h, 48h/168h, 72h/168h) across years
2. Split ensemble returns by term structure regime
3. Test overlay: reduce exposure when term structure is inverted
4. Compare to vol overlay — check orthogonality
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

OUT_DIR = ROOT / "artifacts" / "phase134"
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
    path = OUT_DIR / "vol_term_structure_report.json"
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"  → Saved: {path}")


def compute_rolling_vol(rets: np.ndarray, window: int) -> np.ndarray:
    """Compute rolling annualized vol (causal)."""
    n = len(rets)
    vol = np.full(n, np.nan)
    for i in range(window, n):
        vol[i] = float(np.std(rets[i - window:i])) * np.sqrt(8760)
    if window < n:
        vol[:window] = vol[window]
    return vol


def compute_vol_ratio(rets: np.ndarray, short_w: int, long_w: int) -> np.ndarray:
    """Compute vol ratio = short-term vol / long-term vol.
    > 1 means inverted (short vol > long vol = panic).
    < 1 means normal (calm, trending)."""
    short_vol = compute_rolling_vol(rets, short_w)
    long_vol = compute_rolling_vol(rets, long_w)
    n = len(rets)
    ratio = np.full(n, np.nan)
    for i in range(n):
        if not np.isnan(short_vol[i]) and not np.isnan(long_vol[i]) and long_vol[i] > 1e-8:
            ratio[i] = short_vol[i] / long_vol[i]
    return ratio


def main():
    global _partial
    t0 = time.time()
    print("=" * 70)
    print("Phase 134: Realized Vol Term Structure Alpha")
    print("=" * 70)

    # 1. Load data
    print("\n[1/4] Loading data...")
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

        print(f" OK", flush=True)

    _partial = {"phase": 134}

    # 2. Compute vol ratios for different short/long pairs
    print("\n[2/4] Computing vol term structure ratios...")
    ratio_configs = {
        "24h_168h": (24, 168),
        "48h_168h": (48, 168),
        "72h_168h": (72, 168),
        "24h_336h": (24, 336),
        "48h_336h": (48, 336),
    }

    vol_ratios = {}  # config_name -> year -> ratio_array
    for rname, (sw, lw) in ratio_configs.items():
        vol_ratios[rname] = {}
        print(f"  {rname}:", end="")
        for year in YEARS:
            ratio = compute_vol_ratio(btc_returns[year], sw, lw)
            vol_ratios[rname][year] = ratio
            valid = ratio[~np.isnan(ratio)]
            mean_r = float(np.mean(valid))
            pct_inverted = float(np.sum(valid > 1.0) / len(valid) * 100)
            print(f"  {year}={mean_r:.2f}({pct_inverted:.0f}%inv)", end="")
        print()

    # Also compute absolute vol level for comparison
    abs_vol = {}
    for year in YEARS:
        abs_vol[year] = compute_rolling_vol(btc_returns[year], 168)

    # 3. Regime analysis: split ensemble by vol term structure
    print("\n[3/4] Regime analysis: ensemble Sharpe in inverted vs normal term structure...")

    regime_analysis = {}
    for rname in ratio_configs:
        regime_analysis[rname] = {}
        print(f"\n  --- {rname} ---")
        for year in YEARS:
            min_len = min(len(sig_returns[sk][year]) for sk in SIG_KEYS)
            ratio = vol_ratios[rname][year][:min_len]
            valid_mask = ~np.isnan(ratio)

            # Compute ensemble returns
            ens = np.zeros(min_len)
            for sk in SIG_KEYS:
                ens += ENSEMBLE_WEIGHTS[sk] * sig_returns[sk][year][:min_len]

            # Split by ratio > 1 (inverted) vs < 1 (normal)
            inverted_mask = valid_mask & (ratio > 1.0)
            normal_mask = valid_mask & (ratio <= 1.0)

            s_inv = sharpe(ens[inverted_mask])
            s_norm = sharpe(ens[normal_mask])
            s_full = sharpe(ens)
            pct_inv = float(np.sum(inverted_mask) / np.sum(valid_mask) * 100) if np.sum(valid_mask) > 0 else 0

            regime_analysis[rname][year] = {
                "full_sharpe": s_full,
                "inverted_sharpe": s_inv,
                "normal_sharpe": s_norm,
                "delta_normal_minus_inv": round(s_norm - s_inv, 4),
                "pct_inverted": round(pct_inv, 1),
            }

            tag = "✓" if s_norm > s_inv else "✗"
            print(f"    {year}: inv={s_inv:+.2f} norm={s_norm:+.2f} "
                  f"Δ={s_norm - s_inv:+.2f} {tag} ({pct_inv:.0f}% inverted)")

        # Consistency
        deltas = [regime_analysis[rname][y]["delta_normal_minus_inv"] for y in YEARS]
        n_positive = sum(1 for d in deltas if d > 0)
        avg_delta = float(np.mean(deltas))
        print(f"    → {n_positive}/5 years normal>inverted, avg Δ={avg_delta:+.2f}")

    # 4. Test overlays
    print("\n[4/4] Testing vol term structure overlays...")

    # Use the ratio configuration with best regime separation
    best_ratio_name = None
    best_consistency = -1
    for rname in ratio_configs:
        deltas = [regime_analysis[rname][y]["delta_normal_minus_inv"] for y in YEARS]
        n_pos = sum(1 for d in deltas if d > 0)
        avg_d = float(np.mean(deltas))
        score = n_pos + avg_d * 0.1  # prefer consistency, break ties with magnitude
        if score > best_consistency:
            best_consistency = score
            best_ratio_name = rname
    print(f"  Using best ratio: {best_ratio_name}")

    overlay_configs = {
        "baseline": {"mode": "none"},
        "reduce_inverted_50pct": {"mode": "reduce", "threshold": 1.0, "scale": 0.5},
        "reduce_inverted_30pct": {"mode": "reduce", "threshold": 1.0, "scale": 0.7},
        "reduce_inverted_p75": {"mode": "reduce_pct", "pct": 75, "scale": 0.5},
        "flat_inverted": {"mode": "reduce", "threshold": 1.0, "scale": 0.0},
        "reduce_steep_inv_50pct": {"mode": "reduce", "threshold": 1.3, "scale": 0.5},
        "reduce_steep_inv_flat": {"mode": "reduce", "threshold": 1.5, "scale": 0.0},
        "boost_normal_20pct": {"mode": "boost", "threshold": 0.7, "scale": 1.2},
    }

    overlay_results = {}
    for oname, ocfg in overlay_configs.items():
        yearly_sharpes = []
        yearly_detail = {}

        for year in YEARS:
            min_len = min(len(sig_returns[sk][year]) for sk in SIG_KEYS)
            ratio = vol_ratios[best_ratio_name][year][:min_len]

            if ocfg["mode"] == "none":
                ens = np.zeros(min_len)
                for sk in SIG_KEYS:
                    ens += ENSEMBLE_WEIGHTS[sk] * sig_returns[sk][year][:min_len]
            elif ocfg["mode"] == "reduce":
                ens = np.zeros(min_len)
                for i in range(min_len):
                    ret = 0.0
                    for sk in SIG_KEYS:
                        ret += ENSEMBLE_WEIGHTS[sk] * sig_returns[sk][year][i]
                    if not np.isnan(ratio[i]) and ratio[i] >= ocfg["threshold"]:
                        ret *= ocfg["scale"]
                    ens[i] = ret
            elif ocfg["mode"] == "reduce_pct":
                # Use percentile of ratio distribution for threshold
                valid = ratio[~np.isnan(ratio)]
                thresh = float(np.percentile(valid, ocfg["pct"]))
                ens = np.zeros(min_len)
                for i in range(min_len):
                    ret = 0.0
                    for sk in SIG_KEYS:
                        ret += ENSEMBLE_WEIGHTS[sk] * sig_returns[sk][year][i]
                    if not np.isnan(ratio[i]) and ratio[i] >= thresh:
                        ret *= ocfg["scale"]
                    ens[i] = ret
            elif ocfg["mode"] == "boost":
                ens = np.zeros(min_len)
                for i in range(min_len):
                    ret = 0.0
                    for sk in SIG_KEYS:
                        ret += ENSEMBLE_WEIGHTS[sk] * sig_returns[sk][year][i]
                    if not np.isnan(ratio[i]) and ratio[i] <= ocfg["threshold"]:
                        ret *= ocfg["scale"]
                    ens[i] = ret

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
        print(f"  {oname:28s} {r['obj']:8.4f} {r['avg_sharpe']:8.3f} "
              f"{r['min_sharpe']:8.3f} {delta:+8.4f}{tag}")

    # Check orthogonality with vol overlay
    print("\n  Orthogonality check: vol term structure vs absolute vol level...")
    for year in YEARS:
        ratio = vol_ratios[best_ratio_name][year]
        avol = abs_vol[year]
        min_n = min(len(ratio), len(avol))
        r = ratio[:min_n]
        v = avol[:min_n]
        valid = ~np.isnan(r) & ~np.isnan(v)
        if np.sum(valid) > 100:
            corr = float(np.corrcoef(r[valid], v[valid])[0, 1])
            print(f"    {year}: corr(vol_ratio, abs_vol) = {corr:+.3f}")

    # Verdict
    best_name = max(overlay_results, key=lambda k: overlay_results[k]["obj"])
    best = overlay_results[best_name]
    improvement = best["obj"] - baseline_obj

    if improvement < 0.03:
        verdict = "NO IMPROVEMENT — vol term structure overlay does not help"
    elif improvement < 0.10:
        verdict = f"MARGINAL — {best_name} +{improvement:.3f} OBJ, needs WF validation"
    else:
        verdict = f"POTENTIAL — {best_name} adds +{improvement:.3f} OBJ"

    elapsed = time.time() - t0
    _partial = {
        "phase": 134,
        "description": "Realized Vol Term Structure Alpha",
        "elapsed_seconds": round(elapsed, 1),
        "ratio_configs": {k: list(v) for k, v in ratio_configs.items()},
        "best_ratio": best_ratio_name,
        "regime_analysis": regime_analysis,
        "overlays": overlay_results,
        "best": {"name": best_name, **best},
        "improvement": round(improvement, 4),
        "verdict": verdict,
    }
    _save(_partial, partial=False)

    print(f"\n  VERDICT: {verdict}")
    print(f"\nPhase 134 complete in {elapsed:.0f}s")
    print("=" * 70)


if __name__ == "__main__":
    main()
