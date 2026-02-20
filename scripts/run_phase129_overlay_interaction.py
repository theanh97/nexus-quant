#!/usr/bin/env python3
"""
Phase 129: Overlay Interaction Test + Production Integration
=============================================================
Production has volume_tilt (vol momentum z>0 → 0.65x leverage).
Phase 128 validated a price-vol overlay (high BTC vol → 0.5x + F144 tilt).

Test combinations:
1. Baseline (no overlay)
2. Volume tilt only (current production)
3. Price-vol overlay only (Phase 128)
4. Both combined

If stacking helps, integrate into production.
Uses fixed threshold=0.50 for price-vol (no calibration needed).
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
VOL_TILT_RATIO = PROD_CFG.get("volume_tilt", {}).get("tilt_ratio", 0.65)
VOL_TILT_LOOKBACK = PROD_CFG.get("volume_tilt", {}).get("lookback_bars", 168)

YEAR_RANGES = {
    "2021": ("2021-02-01", "2021-12-31"),
    "2022": ("2022-01-01", "2022-12-31"),
    "2023": ("2023-01-01", "2023-12-31"),
    "2024": ("2024-01-01", "2024-12-31"),
    "2025": ("2025-01-01", "2025-12-31"),
}
YEARS = sorted(YEAR_RANGES.keys())

# Price-vol overlay params (Phase 128 validated)
PRICE_VOL_WINDOW = 168
PRICE_VOL_THRESHOLD = 0.50  # fixed absolute threshold
SCALE_FACTOR = 0.5
F144_BOOST = 0.20

OUT_DIR = ROOT / "artifacts" / "phase129"
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
    path = OUT_DIR / "overlay_interaction_report.json"
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"  → Saved: {path}")


def compute_rolling_vol(rets: np.ndarray, window: int) -> np.ndarray:
    n = len(rets)
    vol = np.full(n, np.nan)
    for i in range(window, n):
        vol[i] = float(np.std(rets[i - window:i])) * np.sqrt(8760)
    if window < n:
        vol[:window] = vol[window]
    return vol


def compute_vol_tilt_signal(volumes: np.ndarray, lookback: int) -> np.ndarray:
    """Compute volume momentum z-score (matching signal_generator.py logic).
    Returns array of booleans: True when tilt should activate (z > 0)."""
    n = len(volumes)
    tilt_active = np.zeros(n, dtype=bool)

    if n < lookback + 50:
        return tilt_active

    # Log volume
    log_vol = np.log1p(np.maximum(volumes, 0))

    for i in range(lookback + 50, n):
        # Momentum = current - lookback
        mom_now = log_vol[i] - log_vol[i - lookback]
        # Rolling window of momenta for z-score
        moms = []
        for j in range(max(lookback, i - 500), i + 1):
            moms.append(log_vol[j] - log_vol[j - lookback])
        moms = np.array(moms)
        mu = float(np.mean(moms))
        sd = float(np.std(moms))
        if sd > 1e-12:
            z = (mom_now - mu) / sd
        else:
            z = 0.0
        tilt_active[i] = z > 0.0

    return tilt_active


def build_ensemble(sig_rets: dict, min_len: int,
                   vol_tilt: np.ndarray = None,
                   price_vol: np.ndarray = None,
                   price_vol_thresh: float = None) -> np.ndarray:
    """Build ensemble returns with optional overlays."""
    ens = np.zeros(min_len)

    for i in range(min_len):
        # Start with base weights
        weights = dict(ENSEMBLE_WEIGHTS)

        # Apply price-vol overlay (F144 tilt + scale)
        price_vol_active = False
        if price_vol is not None and price_vol_thresh is not None:
            if not np.isnan(price_vol[i]) and price_vol[i] > price_vol_thresh:
                price_vol_active = True
                boost_from_each = F144_BOOST / 3
                for sk in SIG_KEYS:
                    if sk == "f144":
                        weights[sk] = min(0.60, weights[sk] + F144_BOOST)
                    else:
                        weights[sk] = max(0.05, weights[sk] - boost_from_each)
                # Normalize
                total = sum(weights.values())
                weights = {k: v / total for k, v in weights.items()}

        # Compute weighted return
        ret = 0.0
        for sk in SIG_KEYS:
            ret += weights[sk] * sig_rets[sk][i]

        # Apply leverage scales
        if price_vol_active:
            ret *= SCALE_FACTOR
        if vol_tilt is not None and vol_tilt[i]:
            ret *= VOL_TILT_RATIO

        ens[i] = ret

    return ens


def main():
    global _partial
    t0 = time.time()
    print("=" * 70)
    print("Phase 129: Overlay Interaction Test")
    print("=" * 70)

    # 1. Load data
    print("\n[1/3] Loading data...")
    sig_returns = {sk: {} for sk in SIG_KEYS}
    btc_returns = {}
    agg_volumes = {}  # For vol tilt computation

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

        # Aggregate perp volume (sum across all symbols)
        vols = []
        for i in range(1, len(dataset.timeline)):
            v = 0.0
            for sym in SYMBOLS:
                try:
                    v += dataset.volume(sym, i)
                except Exception:
                    pass
            vols.append(v)
        agg_volumes[year] = np.array(vols, dtype=np.float64)

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

    _partial = {"phase": 129}

    # Compute overlays
    print("\n[2/3] Computing overlays per year...")
    price_vol_per_year = {}
    vol_tilt_per_year = {}

    for year in YEARS:
        price_vol_per_year[year] = compute_rolling_vol(btc_returns[year], PRICE_VOL_WINDOW)
        vol_tilt_per_year[year] = compute_vol_tilt_signal(agg_volumes[year], VOL_TILT_LOOKBACK)

        n_pvol = int(np.sum(price_vol_per_year[year] > PRICE_VOL_THRESHOLD))
        n_vtilt = int(np.sum(vol_tilt_per_year[year]))
        n_both = int(np.sum((price_vol_per_year[year] > PRICE_VOL_THRESHOLD) & vol_tilt_per_year[year]))
        n = len(btc_returns[year])
        print(f"  {year}: price_vol_active={n_pvol}/{n} ({n_pvol/n*100:.1f}%)  "
              f"vol_tilt_active={n_vtilt}/{n} ({n_vtilt/n*100:.1f}%)  "
              f"both={n_both}/{n} ({n_both/n*100:.1f}%)")

    # 3. Test 4 combinations
    print("\n[3/3] Testing overlay combinations...")
    variants = {
        "baseline": {"vol_tilt": False, "price_vol": False},
        "vol_tilt_only": {"vol_tilt": True, "price_vol": False},
        "price_vol_only": {"vol_tilt": False, "price_vol": True},
        "both_combined": {"vol_tilt": True, "price_vol": True},
    }

    all_results = {}
    for label, cfg in variants.items():
        yearly_sharpes = []
        yearly_detail = {}

        for year in YEARS:
            min_len = min(len(sig_returns[sk][year]) for sk in SIG_KEYS)
            # Trim signals
            sig_trimmed = {sk: sig_returns[sk][year][:min_len] for sk in SIG_KEYS}

            vt = vol_tilt_per_year[year][:min_len] if cfg["vol_tilt"] else None
            pv = price_vol_per_year[year][:min_len] if cfg["price_vol"] else None
            pvt = PRICE_VOL_THRESHOLD if cfg["price_vol"] else None

            ens = build_ensemble(sig_trimmed, min_len, vol_tilt=vt,
                                 price_vol=pv, price_vol_thresh=pvt)
            s = sharpe(ens)
            yearly_sharpes.append(s)
            yearly_detail[year] = s

        avg = round(float(np.mean(yearly_sharpes)), 4)
        mn = round(float(np.min(yearly_sharpes)), 4)
        obj = obj_func(yearly_sharpes)

        all_results[label] = {
            "yearly": yearly_detail,
            "avg_sharpe": avg,
            "min_sharpe": mn,
            "obj": obj,
        }

    # Print summary
    baseline_obj = all_results["baseline"]["obj"]
    print(f"\n  {'Variant':22s} {'OBJ':>8s} {'AVG':>8s} {'MIN':>8s} {'ΔOBJ':>8s}")
    print(f"  {'-'*22} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")
    for label, r in all_results.items():
        delta = r["obj"] - baseline_obj
        tag = " ✓" if delta > 0.05 else ""
        print(f"  {label:22s} {r['obj']:8.4f} {r['avg_sharpe']:8.3f} {r['min_sharpe']:8.3f} {delta:+8.4f}{tag}")

    # Per-year comparison
    print(f"\n  Per-year Sharpe comparison:")
    print(f"  {'Year':6s}", end="")
    for label in variants:
        print(f" {label[:12]:>12s}", end="")
    print()
    for year in YEARS:
        print(f"  {year:6s}", end="")
        for label in variants:
            print(f" {all_results[label]['yearly'][year]:12.3f}", end="")
        print()

    # Determine best
    best_label = max(all_results, key=lambda k: all_results[k]["obj"])
    best = all_results[best_label]

    # Check interaction: is combined > either alone?
    combined_obj = all_results["both_combined"]["obj"]
    vtilt_obj = all_results["vol_tilt_only"]["obj"]
    pvol_obj = all_results["price_vol_only"]["obj"]
    best_single = max(vtilt_obj, pvol_obj)

    interaction = "SYNERGY" if combined_obj > best_single + 0.02 else \
                  "REDUNDANT" if combined_obj < best_single - 0.02 else \
                  "ADDITIVE"

    print(f"\n  Interaction: {interaction}")
    print(f"    Vol tilt alone:  OBJ={vtilt_obj:.4f}")
    print(f"    Price vol alone: OBJ={pvol_obj:.4f}")
    print(f"    Both combined:   OBJ={combined_obj:.4f}")
    print(f"    Best single:     OBJ={best_single:.4f}")

    if best_label == "baseline":
        recommendation = "KEEP BASELINE — no overlay helps"
    elif best_label == "vol_tilt_only":
        recommendation = "KEEP CURRENT — vol tilt alone is best"
    elif best_label == "price_vol_only":
        recommendation = "REPLACE — price vol overlay beats vol tilt"
    else:
        recommendation = "STACK BOTH — combined overlay is best"

    elapsed = time.time() - t0
    _partial = {
        "phase": 129,
        "description": "Overlay Interaction Test",
        "elapsed_seconds": round(elapsed, 1),
        "variants": all_results,
        "baseline_obj": baseline_obj,
        "best": {"label": best_label, **best},
        "interaction": interaction,
        "recommendation": recommendation,
    }
    _save(_partial, partial=False)

    print(f"\n  RECOMMENDATION: {recommendation}")
    print(f"\nPhase 129 complete in {elapsed:.0f}s")
    print("=" * 70)


if __name__ == "__main__":
    main()
