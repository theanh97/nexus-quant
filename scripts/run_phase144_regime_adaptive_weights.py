#!/usr/bin/env python3
"""
Phase 144: Regime-Adaptive Weight Switching
===========================================
Key insight from Phase 143:
- Production weights (V1=27.47%) protect 2025 (Sharpe=2.053) — volatile year
- Optimized weights (V1=5%, i415bw216=40%) win 2021/2024 but fail 2025 (Sharpe=1.706)
- LOYO: only 3/5 wins — 2025 OOS failure killed it

Hypothesis: A vol-regime classifier that SWITCHES weight sets per year
can capture BOTH: 2021/2024 trending alpha AND 2025 volatile floor.

Method:
1. Run all 4 signals for all years → precompute per-signal return arrays
2. Blend returns with different weight sets to get per-year Sharpes (very fast)
3. Exhaustive grid: for each year, assign HIGH_VOL or LOW_VOL weight set
   (2^5 = 32 combos — all tractable)
4. Also test mid-vol interpolation
5. LOYO validate best adaptive combo
6. Also test CONDITIONAL regime: use yearly realized vol percentile to classify

All weight sets tested:
  "prod"  = P91b production: V1=27.47%, I460=19.67%, I415=32.47%, F144=20.39%
  "p143a" = P143 top1: V1=5%, I460=25%, I415=40%, F144=30%
  "p143b" = P143 top4: V1=5%, I460=25%, I415=45%, F144=25%
  "p143c" = P143 top3: V1=5%, I460=30%, I415=40%, F144=25%
  "mid"   = interpolated: V1=16%, I460=22%, I415=37%, F144=25%
"""
import json
import os
import signal as _signal
import sys
import time
from itertools import product as iproduct
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from nexus_quant.backtest.costs import cost_model_from_config
from nexus_quant.backtest.engine import BacktestConfig, BacktestEngine
from nexus_quant.data.providers.registry import make_provider
from nexus_quant.strategies.registry import make_strategy

_partial: dict = {}
_start = time.time()


def _on_timeout(signum, frame):
    print("\n⏰ TIMEOUT — saving partial...")
    _save(_partial, partial=True)
    sys.exit(0)


_signal.signal(_signal.SIGALRM, _on_timeout)
_signal.alarm(1200)  # 20 min

PROD_CFG = json.loads((ROOT / "configs" / "production_p91b_champion.json").read_text())
SYMBOLS = PROD_CFG["data"]["symbols"]
SIGNAL_DEFS = PROD_CFG["ensemble"]["signals"]
SIG_KEYS = list(SIGNAL_DEFS.keys())

VOL_OVERLAY = PROD_CFG.get("vol_regime_overlay", {})
VOL_WINDOW = VOL_OVERLAY.get("window_bars", 168)
VOL_THRESHOLD = VOL_OVERLAY.get("threshold", 0.04)
VOL_SCALE = VOL_OVERLAY.get("scale_factor", 0.6)
VOL_F144_BOOST = VOL_OVERLAY.get("f144_boost", 0.15)

YEAR_RANGES = {
    "2021": ("2021-02-01", "2021-12-31"),
    "2022": ("2022-01-01", "2022-12-31"),
    "2023": ("2023-01-01", "2023-12-31"),
    "2024": ("2024-01-01", "2024-12-31"),
    "2025": ("2025-01-01", "2025-12-31"),
}
YEARS = sorted(YEAR_RANGES.keys())

OUT_DIR = ROOT / "artifacts" / "phase144"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Weight Sets ─────────────────────────────────────────────────────────────
WEIGHT_SETS = {
    "prod": {
        "v1": 0.2747, "i460bw168": 0.1967, "i415bw216": 0.3247, "f144": 0.2039,
    },
    "p143a": {  # top1 from Phase 143 grid
        "v1": 0.05, "i460bw168": 0.25, "i415bw216": 0.40, "f144": 0.30,
    },
    "p143b": {  # top4: slightly more i415
        "v1": 0.05, "i460bw168": 0.25, "i415bw216": 0.45, "f144": 0.25,
    },
    "p143c": {  # top3: more i460
        "v1": 0.05, "i460bw168": 0.30, "i415bw216": 0.40, "f144": 0.25,
    },
    "mid": {    # halfway interpolation
        "v1": 0.16, "i460bw168": 0.22, "i415bw216": 0.37, "f144": 0.25,
    },
}

# Vol regime classification: rank years by difficulty (higher = more volatile)
YEAR_VOL_RANK = {
    "2022": 5,  # bear/crash — HARDEST
    "2025": 4,  # choppy corrections
    "2023": 3,  # mid (recovery)
    "2021": 2,  # strong bull
    "2024": 1,  # EASIEST bull
}

COST_MODEL = cost_model_from_config({
    "fee_rate": 0.0005,
    "slippage_rate": 0.0003,
    "cost_multiplier": 1.0,
})


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
    out = OUT_DIR / "phase144_report.json"
    out.write_text(json.dumps(data, indent=2))
    print(f"✅ Saved → {out}")


def compute_btc_price_vol(dataset, window: int = 168) -> np.ndarray:
    """Compute BTC realized vol using dataset.close() + dataset.timeline."""
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


def blend_ensemble(weights: dict, sig_rets: dict, btc_vol: np.ndarray,
                   with_vol_overlay: bool = False) -> np.ndarray:
    """Blend per-signal returns with given weights into ensemble return series."""
    min_len = min(len(sig_rets[sk]) for sk in SIG_KEYS)
    bv = btc_vol[:min_len]
    ens = np.zeros(min_len)
    for i in range(min_len):
        if with_vol_overlay and not np.isnan(bv[i]) and bv[i] > VOL_THRESHOLD:
            boost_per_other = VOL_F144_BOOST / max(1, len(SIG_KEYS) - 1)
            for sk in SIG_KEYS:
                if sk == "f144":
                    adj_w = min(0.60, weights[sk] + VOL_F144_BOOST)
                else:
                    adj_w = max(0.05, weights[sk] - boost_per_other)
                ens[i] += adj_w * sig_rets[sk][i]
            ens[i] *= VOL_SCALE
        else:
            for sk in SIG_KEYS:
                ens[i] += weights[sk] * sig_rets[sk][i]
    return ens


def main():
    global _partial

    print("=" * 65)
    print("PHASE 144: Regime-Adaptive Weight Switching")
    print("=" * 65)

    # ── Step 1: Load data + run all signals for all years ─────────────
    print("\n[1/4] Loading data + pre-computing per-signal returns...")
    sig_returns: dict[str, dict[str, np.ndarray]] = {sk: {} for sk in SIG_KEYS}
    btc_vol_data: dict[str, np.ndarray] = {}

    for year, (start, end) in YEAR_RANGES.items():
        print(f"  {year}: ", end="", flush=True)
        cfg_data = {
            "provider": "binance_rest_v1", "symbols": SYMBOLS,
            "start": start, "end": end, "bar_interval": "1h",
            "cache_dir": ".cache/binance_rest",
        }
        provider = make_provider(cfg_data, seed=42)
        dataset = provider.load()
        btc_vol_data[year] = compute_btc_price_vol(dataset, window=VOL_WINDOW)

        for sk in SIG_KEYS:
            sig_def = SIGNAL_DEFS[sk]
            bt_cfg = BacktestConfig(costs=COST_MODEL)
            strat = make_strategy({"name": sig_def["strategy"], "params": sig_def["params"]})
            engine = BacktestEngine(bt_cfg)
            result = engine.run(dataset, strat)
            sig_returns[sk][year] = np.array(result.returns, dtype=np.float64)
            print(".", end="", flush=True)
        print(f" ✓")

    # ── Step 2: Compute Sharpe for all weight sets across all years ────
    print("\n[2/4] Computing per-year Sharpes for all weight sets...")
    ws_yearly: dict[str, dict[str, float]] = {}
    ws_summary: dict[str, dict] = {}

    for label, weights in WEIGHT_SETS.items():
        yearly_sharpes = {}
        for year in YEARS:
            ens = blend_ensemble(weights, {sk: sig_returns[sk][year] for sk in SIG_KEYS},
                                 btc_vol_data[year], with_vol_overlay=True)
            yearly_sharpes[year] = sharpe(ens)
        avg = round(float(np.mean(list(yearly_sharpes.values()))), 4)
        mn = round(float(np.min(list(yearly_sharpes.values()))), 4)
        obj = obj_func(list(yearly_sharpes.values()))
        ws_yearly[label] = yearly_sharpes
        ws_summary[label] = {"weights": weights, "yearly": yearly_sharpes,
                              "avg_sharpe": avg, "min_sharpe": mn, "obj": obj}
        print(f"  {label:10s}: AVG={avg:.4f} MIN={mn:.4f} OBJ={obj:.4f}")
        print(f"            {yearly_sharpes}")

    _partial.update({"phase": 144, "weight_set_results": ws_summary})
    _save(_partial, partial=True)

    # ── Step 3: Exhaustive 2-set switching: 2^5 = 32 combos ──────────
    print("\n[3/4] Exhaustive regime-switching grid (HIGH_VOL vs LOW_VOL)...")
    top_results = []

    for candidate_high, candidate_low in [
        ("prod", "p143a"),
        ("prod", "p143b"),
        ("prod", "p143c"),
        ("prod", "mid"),
        ("mid", "p143a"),
    ]:
        for combo in iproduct([candidate_high, candidate_low], repeat=len(YEARS)):
            assignment = dict(zip(YEARS, combo))
            adaptive_yearly = {yr: ws_yearly[assignment[yr]][yr] for yr in YEARS}
            obj = obj_func(list(adaptive_yearly.values()))
            avg = round(float(np.mean(list(adaptive_yearly.values()))), 4)
            mn = round(float(np.min(list(adaptive_yearly.values()))), 4)
            top_results.append({
                "high_set": candidate_high,
                "low_set": candidate_low,
                "assignment": assignment,
                "yearly": adaptive_yearly,
                "avg_sharpe": avg,
                "min_sharpe": mn,
                "obj": obj,
            })

    # deduplicate by assignment
    seen = set()
    unique_results = []
    for r in top_results:
        key = "|".join(f"{y}:{r['assignment'][y]}" for y in YEARS)
        if key not in seen:
            seen.add(key)
            unique_results.append(r)

    unique_results.sort(key=lambda x: x["obj"], reverse=True)
    top10 = unique_results[:10]
    best = top10[0]
    baseline_obj = ws_summary["prod"]["obj"]

    print(f"\n  Top 5 regime combos:")
    for r in top10[:5]:
        assign_str = " | ".join(f"{y}:{r['assignment'][y]}" for y in YEARS)
        print(f"    OBJ={r['obj']:.4f} AVG={r['avg_sharpe']:.4f} MIN={r['min_sharpe']:.4f} | {assign_str}")
    print(f"\n  Baseline (prod all years): OBJ={baseline_obj:.4f}")
    print(f"  Best adaptive:             OBJ={best['obj']:.4f} (Δ={best['obj']-baseline_obj:+.4f})")

    # ── Vol-rank hypothesis (fixed assignments) ────────────────────────
    hypothesis_results = {}
    for thresh in [3, 4]:
        assignment = {yr: "prod" if YEAR_VOL_RANK[yr] >= thresh else "p143a"
                      for yr in YEARS}
        adaptive_yearly = {yr: ws_yearly[assignment[yr]][yr] for yr in YEARS}
        obj = obj_func(list(adaptive_yearly.values()))
        avg = round(float(np.mean(list(adaptive_yearly.values()))), 4)
        mn = round(float(np.min(list(adaptive_yearly.values()))), 4)
        label = f"vol_thresh_{thresh}"
        hypothesis_results[label] = {
            "assignment": assignment, "yearly": adaptive_yearly,
            "avg_sharpe": avg, "min_sharpe": mn, "obj": obj,
        }
        print(f"  Vol-thresh≥{thresh}: OBJ={obj:.4f} AVG={avg:.4f} MIN={mn:.4f} | {assignment}")

    _partial.update({
        "top10_adaptive": top10,
        "hypothesis_results": hypothesis_results,
    })
    _save(_partial, partial=True)

    # ── Step 4: LOYO validation ────────────────────────────────────────
    print("\n[4/4] LOYO validation of best adaptive combo...")
    loyo_results = []
    for test_yr in YEARS:
        train_yrs = [y for y in YEARS if y != test_yr]
        # For each candidate assignment pattern, compute train OBJ
        best_loyo_label = None
        best_train_obj = -999.0
        for label, weights in WEIGHT_SETS.items():
            train_sharpes = [ws_yearly[label][y] for y in train_yrs]
            train_obj = obj_func(train_sharpes)
            if train_obj > best_train_obj:
                best_train_obj = train_obj
                best_loyo_label = label

        # Also test best adaptive assignment on train years
        best_adaptive_train = None
        best_adaptive_train_obj = -999.0
        for r in unique_results[:20]:
            train_sharpes = [r["yearly"][y] for y in train_yrs]
            train_obj = obj_func(train_sharpes)
            if train_obj > best_adaptive_train_obj:
                best_adaptive_train_obj = train_obj
                best_adaptive_train = r

        # Use the weight set that had best train performance for test year
        test_sharpe_static = ws_yearly[best_loyo_label][test_yr]
        baseline_sharpe = ws_yearly["prod"][test_yr]

        # Use best adaptive test assignment for test year
        if best_adaptive_train:
            test_year_set = best_adaptive_train["assignment"][test_yr]
            test_sharpe_adaptive = ws_yearly[test_year_set][test_yr]
        else:
            test_sharpe_adaptive = test_sharpe_static

        loyo_results.append({
            "test_year": test_yr,
            "chosen_static": best_loyo_label,
            "chosen_adaptive": best_adaptive_train["assignment"][test_yr] if best_adaptive_train else best_loyo_label,
            "test_sharpe_adaptive": round(test_sharpe_adaptive, 4),
            "baseline_sharpe": round(baseline_sharpe, 4),
            "delta": round(test_sharpe_adaptive - baseline_sharpe, 4),
        })
        print(f"  LOYO {test_yr}: set={loyo_results[-1]['chosen_adaptive']:10s} "
              f"Sharpe={test_sharpe_adaptive:.4f} vs baseline={baseline_sharpe:.4f} "
              f"(Δ={test_sharpe_adaptive-baseline_sharpe:+.4f})")

    loyo_wins = sum(1 for r in loyo_results if r["delta"] > 0)
    loyo_avg_delta = round(float(np.mean([r["delta"] for r in loyo_results])), 4)
    loyo_summary = {
        "results": loyo_results,
        "wins": f"{loyo_wins}/5",
        "avg_delta": loyo_avg_delta,
    }
    print(f"  LOYO: {loyo_wins}/5 wins | avg_delta={loyo_avg_delta:+.4f}")

    # ── Verdict ────────────────────────────────────────────────────────
    delta_obj = round(best["obj"] - baseline_obj, 4)
    loyo_passes = loyo_wins >= 3

    if delta_obj > 0.05 and loyo_passes:
        verdict = f"VALIDATED — adaptive switching +{delta_obj:.4f} OBJ, LOYO {loyo_wins}/5"
    elif delta_obj > 0 and loyo_passes:
        verdict = f"MARGINAL — small gain +{delta_obj:.4f} OBJ, LOYO {loyo_wins}/5"
    elif delta_obj > 0:
        verdict = f"OVERFIT — in-sample gain but LOYO fails ({loyo_wins}/5) — KEEP prod"
    else:
        verdict = f"FAIL — no improvement (Δ={delta_obj:+.4f}), LOYO {loyo_wins}/5 — KEEP prod"

    print(f"\n{'='*65}")
    print(f"VERDICT: {verdict}")
    print(f"  Baseline (prod all years)  OBJ={baseline_obj:.4f}")
    print(f"  Best adaptive              OBJ={best['obj']:.4f}  Δ={delta_obj:+.4f}")
    print(f"  LOYO {loyo_wins}/5 | avg_delta={loyo_avg_delta:+.4f}")
    print(f"{'='*65}")

    if "VALIDATED" in verdict:
        next_note = "Update production config to include vol_regime_weight_switching."
        next_phase = "Phase 145: Production integration of adaptive weights + live monitoring."
    else:
        next_note = "Regime switching does not generalize. Try new signal source."
        next_phase = (
            "Phase 145 options:\n"
            "  A) On-chain whale flows (Glassnode/Nansen) — new alpha source\n"
            "  B) Deribit IV surface skew — options-implied regime signal\n"
            "  C) Exchange net flow signal (CEX inflow/outflow)\n"
            "  D) Intra-day time-of-day seasonality (HOD alpha from Phase 131)"
        )

    report = {
        "phase": 144,
        "description": "Regime-Adaptive Weight Switching",
        "hypothesis": (
            "Vol-regime classifier switches between V1-heavy (2022/2025) "
            "and i415bw216-heavy (2021/2024) weights to capture trending + volatile alpha"
        ),
        "elapsed_seconds": round(time.time() - _start, 1),
        "baseline_obj": baseline_obj,
        "weight_set_results": ws_summary,
        "top10_adaptive": top10,
        "hypothesis_results": hypothesis_results,
        "best_adaptive": best,
        "delta_obj": delta_obj,
        "loyo": loyo_summary,
        "verdict": verdict,
        "next_phase_notes": next_phase,
        "next_phase_options": next_note,
    }
    _save(report, partial=False)
    return report


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback
        print(f"\n❌ ERROR: {e}")
        traceback.print_exc()
        _partial["error"] = str(e)
        _partial["traceback"] = traceback.format_exc()
        _save(_partial, partial=True)
        sys.exit(1)
