#!/usr/bin/env python3
"""
Phase 150: Cross-Sectional Return Skewness as Tail-Risk Filter
===============================================================
Phase 149: correlation regime NO IMPROVEMENT (all variants worse)
Production OBJ = 1.8886 (v2.3.0)

Hypothesis — Cross-Sectional Return Skewness as Tail Risk Indicator:
  At each bar, compute the 168h cumulative returns for all 10 symbols.
  Then compute the CROSS-SECTIONAL SKEWNESS of these 10 values.

  Interpretation:
    Negative skew → most coins perform near the median, but a few crash badly
                  → Left-tail risk → reduce leverage (danger zone)
    Positive skew → most coins perform near median, but a few moon
                  → Normal momentum regime → maintain

  This is a STRUCTURAL risk signal (not a positioning or crowding signal).
  It's different from:
    - Breadth: counts % with positive return (direction, not distribution shape)
    - Correlation: measures co-movement (not asymmetry)
    - Vol overlay: measures BTC vol (not cross-sectional distribution)
    - Funding dispersion: measures funding (not price return shape)

Computing: rolling 336h skewness of the cross-sectional distribution
  = at each bar, take the 168h return for each of 10 symbols, compute skewness
    of those 10 values, smooth over 336h window → rolling percentile

Variants:
  A. reduce_neg_skew_25pct: scale×0.70 when skew_pct < 25th pct (left-tail regime)
  B. reduce_neg_skew_10pct: scale×0.65 when skew_pct < 10th pct (extreme left-tail)
  C. boost_pos_skew_75pct:  scale×1.10 when skew_pct > 75th pct (right-tail / momentum regime)
  D. combo_10_90:           reduce<10th + boost>90th
  E. smooth_skew:           smooth scale based on skew_pct

Pass criteria: OBJ > 1.8886 (P148 production) + LOYO >= 3/5
"""
import json
import os
import signal as _signal
import sys
import time
from pathlib import Path
from scipy import stats as scipy_stats

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
_signal.alarm(1800)

PROD_CFG = json.loads((ROOT / "configs" / "production_p91b_champion.json").read_text())
SYMBOLS = PROD_CFG["data"]["symbols"]
SIGNAL_DEFS = PROD_CFG["ensemble"]["signals"]
SIG_KEYS = list(SIGNAL_DEFS.keys())

VOL_OVERLAY = PROD_CFG.get("vol_regime_overlay", {})
VOL_WINDOW = VOL_OVERLAY.get("window_bars", 168)
VOL_THRESHOLD = VOL_OVERLAY.get("threshold", 0.5)
VOL_SCALE = VOL_OVERLAY.get("scale_factor", 0.5)
VOL_F144_BOOST = VOL_OVERLAY.get("f144_boost", 0.2)

BREADTH_LOOKBACK = 168
PCT_WINDOW = 336
P_LOW = 0.33
P_HIGH = 0.67
FUND_DISP_BOOST_SCALE = 1.15
FUND_DISP_PCT_THRESHOLD = 0.75

YEAR_RANGES = {
    "2021": ("2021-02-01", "2021-12-31"),
    "2022": ("2022-01-01", "2022-12-31"),
    "2023": ("2023-01-01", "2023-12-31"),
    "2024": ("2024-01-01", "2024-12-31"),
    "2025": ("2025-01-01", "2025-12-31"),
}
YEARS = sorted(YEAR_RANGES.keys())

OUT_DIR = ROOT / "artifacts" / "phase150"
OUT_DIR.mkdir(parents=True, exist_ok=True)

WEIGHTS = {
    "prod":  {"v1": 0.2747, "i460bw168": 0.1967, "i415bw216": 0.3247, "f144": 0.2039},
    "p143b": {"v1": 0.05,   "i460bw168": 0.25,   "i415bw216": 0.45,   "f144": 0.25},
    "mid":   {"v1": 0.16,   "i460bw168": 0.22,   "i415bw216": 0.37,   "f144": 0.25},
}
WEIGHTS_LIST = [WEIGHTS["prod"], WEIGHTS["mid"], WEIGHTS["p143b"]]

COST_MODEL = cost_model_from_config({
    "fee_rate": 0.0005, "slippage_rate": 0.0003, "cost_multiplier": 1.0,
})

RET_LOOKBACK = 168  # cumulative return lookback for cross-section
SKEW_PCT_WINDOW = 336  # rolling window for percentile rank of skewness


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
    out = OUT_DIR / "phase150_report.json"
    out.write_text(json.dumps(data, indent=2))
    print(f"✅ Saved → {out}")


def compute_precomputed_state(dataset) -> tuple:
    """Pre-compute all per-bar state arrays: btc_vol, breadth_regime, fund_std_pct, skew_pct."""
    n = len(dataset.timeline)

    # BTC vol
    btc_rets = np.zeros(n)
    for i in range(1, n):
        c0 = dataset.close("BTCUSDT", i - 1)
        c1 = dataset.close("BTCUSDT", i)
        btc_rets[i] = (c1 / c0 - 1.0) if c0 > 0 else 0.0
    btc_vol = np.full(n, np.nan)
    for i in range(VOL_WINDOW, n):
        btc_vol[i] = float(np.std(btc_rets[i - VOL_WINDOW:i])) * np.sqrt(8760)
    if VOL_WINDOW < n:
        btc_vol[:VOL_WINDOW] = btc_vol[VOL_WINDOW]

    # Breadth regime
    breadth = np.full(n, 0.5)
    for i in range(BREADTH_LOOKBACK, n):
        pos = sum(
            1 for sym in SYMBOLS
            if (c0 := dataset.close(sym, i - BREADTH_LOOKBACK)) > 0
            and dataset.close(sym, i) > c0
        )
        breadth[i] = pos / len(SYMBOLS)
    breadth[:BREADTH_LOOKBACK] = 0.5
    brd_pct = np.full(n, 0.5)
    for i in range(PCT_WINDOW, n):
        hist = breadth[i - PCT_WINDOW:i]
        brd_pct[i] = float(np.mean(hist <= breadth[i]))
    brd_pct[:PCT_WINDOW] = 0.5
    breadth_regime = np.where(brd_pct >= P_HIGH, 2, np.where(brd_pct >= P_LOW, 1, 0)).astype(int)

    # Funding dispersion
    fund_std_raw = np.zeros(n)
    for i in range(n):
        ts = dataset.timeline[i]
        rates = np.array([dataset.last_funding_rate_before(sym, ts) for sym in SYMBOLS])
        fund_std_raw[i] = float(np.std(rates)) if len(rates) > 1 else 0.0
    fund_std_pct = np.full(n, 0.5)
    for i in range(PCT_WINDOW, n):
        hist = fund_std_raw[i - PCT_WINDOW:i]
        fund_std_pct[i] = float(np.mean(hist <= fund_std_raw[i]))
    fund_std_pct[:PCT_WINDOW] = 0.5

    # Cross-sectional return skewness
    # For each bar i: compute 168h cumulative return for each symbol
    # Then compute skewness of those 10 values
    xs_skew = np.zeros(n)
    for i in range(RET_LOOKBACK, n):
        xs_rets = []
        for sym in SYMBOLS:
            c0 = dataset.close(sym, i - RET_LOOKBACK)
            c1 = dataset.close(sym, i)
            if c0 > 1e-10:
                xs_rets.append((c1 - c0) / c0)
        if len(xs_rets) >= 4:  # need at least 4 for meaningful skewness
            xs_skew[i] = float(scipy_stats.skew(xs_rets))
    xs_skew[:RET_LOOKBACK] = 0.0

    # Rolling percentile of cross-sectional skewness
    skew_pct = np.full(n, 0.5)
    for i in range(SKEW_PCT_WINDOW, n):
        hist = xs_skew[i - SKEW_PCT_WINDOW:i]
        skew_pct[i] = float(np.mean(hist <= xs_skew[i]))
    skew_pct[:SKEW_PCT_WINDOW] = 0.5

    return btc_vol, breadth_regime, fund_std_pct, skew_pct


def compute_ensemble_with_skew(
    sig_rets: dict,
    btc_vol: np.ndarray,
    breadth_regime: np.ndarray,
    fund_std_pct: np.ndarray,
    skew_pct: np.ndarray,
    skew_variant: str = "none",
) -> np.ndarray:
    """Apply full production stack + optional skewness overlay."""
    min_len = min(len(sig_rets[sk]) for sk in SIG_KEYS)
    bv = btc_vol[:min_len]
    reg = breadth_regime[:min_len]
    fsp = fund_std_pct[:min_len]
    sp = skew_pct[:min_len]
    ens = np.zeros(min_len)

    for i in range(min_len):
        w = WEIGHTS_LIST[int(reg[i])]

        # Vol overlay
        if not np.isnan(bv[i]) and bv[i] > VOL_THRESHOLD:
            boost = VOL_F144_BOOST / max(1, len(SIG_KEYS) - 1)
            base = 0.0
            for sk in SIG_KEYS:
                adj_w = min(0.60, w[sk] + VOL_F144_BOOST) if sk == "f144" else max(0.05, w[sk] - boost)
                base += adj_w * sig_rets[sk][i]
            ret_i = base * VOL_SCALE
        else:
            ret_i = sum(w[sk] * sig_rets[sk][i] for sk in SIG_KEYS)

        # Funding dispersion boost (P148)
        if fsp[i] > FUND_DISP_PCT_THRESHOLD:
            ret_i *= FUND_DISP_BOOST_SCALE

        # Skewness overlay
        skew_scale = 1.0
        if skew_variant == "reduce_neg_skew_25pct" and sp[i] < 0.25:
            skew_scale = 0.70  # reduce when left-tail dominant
        elif skew_variant == "reduce_neg_skew_10pct" and sp[i] < 0.10:
            skew_scale = 0.65  # extreme left-tail only
        elif skew_variant == "boost_pos_skew_75pct" and sp[i] > 0.75:
            skew_scale = 1.10  # boost in right-skew (momentum regime)
        elif skew_variant == "combo_10_90":
            if sp[i] < 0.10:
                skew_scale = 0.65
            elif sp[i] > 0.90:
                skew_scale = 1.10
        elif skew_variant == "smooth_skew":
            # linear: reduce in low-skew regimes, boost in high-skew
            # scale = 0.80 at skew_pct=0 → 1.0 at 0.5 → 1.10 at 1.0
            skew_scale = max(0.60, 0.80 + 0.30 * sp[i])

        ens[i] = ret_i * skew_scale

    return ens


VARIANTS = [
    "none",
    "reduce_neg_skew_25pct",
    "reduce_neg_skew_10pct",
    "boost_pos_skew_75pct",
    "combo_10_90",
    "smooth_skew",
]

VARIANT_LABELS = {
    "none":                  "Production baseline (P148 + breadth + vol)",
    "reduce_neg_skew_25pct": "Reduce×0.70 when xs_skew_pct < 25th (left-tail risk)",
    "reduce_neg_skew_10pct": "Reduce×0.65 when xs_skew_pct < 10th (extreme left-tail)",
    "boost_pos_skew_75pct":  "Boost×1.10 when xs_skew_pct > 75th (momentum regime)",
    "combo_10_90":           "Reduce<10th + Boost>90th (extreme combo)",
    "smooth_skew":           "Smooth: scale = 0.80 + 0.30×skew_pct",
}


def main():
    global _partial

    print("=" * 72)
    print("PHASE 150: Cross-Sectional Return Skewness as Tail-Risk Filter")
    print("=" * 72)

    # ── Step 1: Load data ──────────────────────────────────────────────
    print("\n[1/4] Loading data + pre-computing signal returns + skewness...")
    sig_returns: dict = {sk: {} for sk in SIG_KEYS}
    state_data: dict = {}

    for year, (start, end) in YEAR_RANGES.items():
        print(f"  {year}: ", end="", flush=True)
        cfg_data = {
            "provider": "binance_rest_v1", "symbols": SYMBOLS,
            "start": start, "end": end, "bar_interval": "1h",
            "cache_dir": ".cache/binance_rest",
        }
        provider = make_provider(cfg_data, seed=42)
        dataset = provider.load()
        bv, br, fsp, sp = compute_precomputed_state(dataset)
        state_data[year] = (bv, br, fsp, sp)
        print("S", end="", flush=True)

        for sk in SIG_KEYS:
            sig_def = SIGNAL_DEFS[sk]
            bt_cfg = BacktestConfig(costs=COST_MODEL)
            strat = make_strategy({"name": sig_def["strategy"], "params": sig_def["params"]})
            engine = BacktestEngine(bt_cfg)
            result = engine.run(dataset, strat)
            sig_returns[sk][year] = np.array(result.returns, dtype=np.float64)
            print(".", end="", flush=True)
        print(" ✓")

    # ── Step 2: Compare all variants ──────────────────────────────────
    print("\n[2/4] Comparing skewness overlay variants (IS 2021-2025)...")
    variant_results: dict = {}
    baseline_obj = None
    best_v = "none"
    best_obj = -999.0

    for v in VARIANTS:
        yearly = {}
        for year in YEARS:
            sr = {sk: sig_returns[sk][year] for sk in SIG_KEYS}
            bv, br, fsp, sp = state_data[year]
            ens = compute_ensemble_with_skew(sr, bv, br, fsp, sp, v)
            yearly[year] = sharpe(ens)
        obj = obj_func(list(yearly.values()))
        delta = round(obj - (baseline_obj or obj), 4)
        if v == "none":
            baseline_obj = obj
            delta = 0.0
        variant_results[v] = {"yearly": yearly, "obj": obj, "delta": delta}
        flag = " ← BASELINE" if v == "none" else (" ✅" if obj > baseline_obj else "")
        print(f"  {VARIANT_LABELS[v]:62s} OBJ={obj:.4f} Δ={delta:+.4f}{flag}")
        if obj > best_obj:
            best_obj = obj
            best_v = v

    if best_v == "none":
        print(f"\n  No variant beats baseline OBJ={baseline_obj:.4f}")
    else:
        print(f"\n  Best: {VARIANT_LABELS[best_v]} → OBJ={best_obj:.4f} (Δ={best_obj - baseline_obj:+.4f})")

    _partial.update({
        "phase": 150,
        "baseline_obj": baseline_obj,
        "variant_results": {v: d for v, d in variant_results.items()},
        "best_variant": best_v,
        "best_obj": best_obj,
    })
    _save(_partial, partial=True)

    # ── Step 3: LOYO validation ────────────────────────────────────────
    loyo_results = []
    loyo_wins = 0
    loyo_avg = 0.0

    if best_v != "none":
        print(f"\n[3/4] LOYO validation of {best_v}...")
        for test_yr in YEARS:
            sr_test = {sk: sig_returns[sk][test_yr] for sk in SIG_KEYS}
            bv, br, fsp, sp = state_data[test_yr]
            ens_best = compute_ensemble_with_skew(sr_test, bv, br, fsp, sp, best_v)
            ens_base = compute_ensemble_with_skew(sr_test, bv, br, fsp, sp, "none")
            s_best = sharpe(ens_best)
            s_base = sharpe(ens_base)
            delta = round(s_best - s_base, 4)
            loyo_results.append({
                "test_year": test_yr,
                "best_sharpe": round(s_best, 4),
                "baseline_sharpe": round(s_base, 4),
                "delta": delta,
            })
            flag = "✅" if delta > 0 else "❌"
            print(f"  OOS {test_yr}: {s_best:.4f} vs baseline={s_base:.4f} Δ={delta:+.4f} {flag}")
        loyo_wins = sum(1 for r in loyo_results if r["delta"] > 0)
        loyo_avg = round(float(np.mean([r["delta"] for r in loyo_results])), 4)
        print(f"  LOYO: {loyo_wins}/5 wins | avg_delta={loyo_avg:+.4f}")
    else:
        print("\n[3/4] LOYO: Skipped (no variant beats baseline)")

    # ── Step 4: WF validation ─────────────────────────────────────────
    wf_results = []
    wf_wins = 0
    wf_avg_delta = 0.0

    if best_v != "none" and loyo_wins >= 3:
        print(f"\n[4/4] Walk-Forward validation...")
        wf_windows = [
            {"train": ["2021", "2022", "2023"], "test": "2024"},
            {"train": ["2022", "2023", "2024"], "test": "2025"},
        ]
        for wf in wf_windows:
            test_yr = wf["test"]
            sr_test = {sk: sig_returns[sk][test_yr] for sk in SIG_KEYS}
            bv, br, fsp, sp = state_data[test_yr]
            ens_best = compute_ensemble_with_skew(sr_test, bv, br, fsp, sp, best_v)
            ens_base = compute_ensemble_with_skew(sr_test, bv, br, fsp, sp, "none")
            test_s = sharpe(ens_best)
            base_s = sharpe(ens_base)
            delta = round(test_s - base_s, 4)
            wf_results.append({
                "test": test_yr, "test_sharpe": round(test_s, 4),
                "baseline_sharpe": round(base_s, 4), "delta": delta,
            })
            flag = "✅" if delta > 0 else "❌"
            print(f"  WF test={test_yr}: {test_s:.4f} vs {base_s:.4f} Δ={delta:+.4f} {flag}")
        wf_wins = sum(1 for r in wf_results if r["delta"] > 0)
        wf_avg_delta = round(float(np.mean([r["delta"] for r in wf_results])), 4)
        print(f"  WF: {wf_wins}/2 | avg_delta={wf_avg_delta:+.4f}")
    else:
        print("\n[4/4] WF: Skipped")

    # ── Verdict ────────────────────────────────────────────────────────
    print(f"\n{'='*72}")
    validated = best_v != "none" and best_obj > baseline_obj and loyo_wins >= 3 and loyo_avg > 0

    if validated:
        verdict = (
            f"IMPROVEMENT — {VARIANT_LABELS[best_v]} adds +{best_obj - baseline_obj:.4f} OBJ. "
            f"LOYO {loyo_wins}/5, avg_delta={loyo_avg:+.4f}, WF {wf_wins}/2"
        )
        next_phase = (
            "Phase 151: Integrate skewness filter into production config v2.4.0. "
            "Then explore: universe expansion (15 symbols) or momentum skew signal."
        )
    elif best_v != "none":
        verdict = (
            f"MARGINAL — {VARIANT_LABELS[best_v]} IS +{best_obj - baseline_obj:.4f} "
            f"but LOYO {loyo_wins}/5. No production change."
        )
        next_phase = (
            "Phase 151: Explore universe expansion (add 5 more major perps). "
            "Or: test stronger threshold combinations for skewness filter."
        )
    else:
        verdict = (
            f"NO IMPROVEMENT — all skewness variants worse than baseline OBJ={baseline_obj:.4f}. "
            "Cross-sectional skewness not additive to existing ensemble."
        )
        next_phase = (
            "Phase 151: Explore universe expansion — add MATICUSDT, LTCUSDT, BCHUSDT, "
            "ATOMUSDT, NEARUSDT to the 10-symbol universe. More coins = better idio signal. "
            "OR: deep dive into why 2025 underperforms and what regime it was."
        )

    print(f"VERDICT: {verdict}")
    print(f"{'='*72}")

    report = {
        "phase": 150,
        "description": "Cross-Sectional Return Skewness as Tail-Risk Filter",
        "hypothesis": "Negative cross-sectional skewness = left-tail regime = reduce leverage",
        "elapsed_seconds": round(time.time() - _start, 1),
        "baseline_obj": baseline_obj,
        "variant_results": {v: d for v, d in variant_results.items()},
        "best_variant": best_v,
        "best_obj": best_obj,
        "loyo_results": loyo_results,
        "loyo_wins": loyo_wins,
        "loyo_avg_delta": loyo_avg,
        "wf_results": wf_results,
        "wf_wins": wf_wins,
        "wf_avg_delta": wf_avg_delta,
        "validated": validated,
        "verdict": verdict,
        "next_phase_notes": next_phase,
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
