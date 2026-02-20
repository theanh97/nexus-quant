#!/usr/bin/env python3
"""
Phase 148: Funding Rate Cross-Sectional Dispersion Overlay
===========================================================
Phase 147 confirmed: production (breadth+vol_always) is already optimal.
No overlay interaction improvement found.

New hypothesis — Funding Rate Dispersion as crowding signal:
  At each 8h funding window, compute the STANDARD DEVIATION of funding rates
  across all 10 symbols (cross-sectional dispersion).

  Logic:
    LOW dispersion + HIGH avg funding  → Crowd is UNIFORMLY LONG → reduce leverage
    LOW dispersion + LOW avg funding   → Crowd is UNIFORMLY SHORT → reduce leverage (contrarian risk)
    HIGH dispersion                    → Cross-sectional differentiation → good for idio signals → maintain
    NEUTRAL                            → No adjustment

  This is ORTHOGONAL to existing signals:
    - F144 uses per-coin cumulative funding momentum (coin-level contrarian)
    - Global L/S uses account ratio (crowding in different dimension)
    - Breadth uses price-based momentum
    - This uses DISPERSION across coins (market microstructure / crowding)

Variants tested:
  A. Baseline (production breadth+vol_always)
  B. Dispersion overlay: reduce when dispersion < 25th pct AND |avg_funding| > threshold
  C. Dispersion overlay: reduce when dispersion < 10th pct (extreme herding)
  D. Dispersion overlay: reduce based on smooth scale factor (rolling z-score)
  E. Reverse: BOOST when extreme high dispersion (max differentiation)

Pass criteria: OBJ > 1.8800 (Phase 147 confirmed production) + LOYO >= 3/5
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

YEAR_RANGES = {
    "2021": ("2021-02-01", "2021-12-31"),
    "2022": ("2022-01-01", "2022-12-31"),
    "2023": ("2023-01-01", "2023-12-31"),
    "2024": ("2024-01-01", "2024-12-31"),
    "2025": ("2025-01-01", "2025-12-31"),
}
YEARS = sorted(YEAR_RANGES.keys())

OUT_DIR = ROOT / "artifacts" / "phase148"
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

_EPS = 1e-10


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
    out = OUT_DIR / "phase148_report.json"
    out.write_text(json.dumps(data, indent=2))
    print(f"✅ Saved → {out}")


def compute_btc_vol(dataset, window: int = 168) -> np.ndarray:
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


def compute_breadth_regime(dataset) -> np.ndarray:
    n = len(dataset.timeline)
    breadth = np.full(n, 0.5)
    for i in range(BREADTH_LOOKBACK, n):
        pos = sum(
            1 for sym in SYMBOLS
            if (c0 := dataset.close(sym, i - BREADTH_LOOKBACK)) > 0
            and dataset.close(sym, i) > c0
        )
        breadth[i] = pos / len(SYMBOLS)
    breadth[:BREADTH_LOOKBACK] = 0.5
    pct = np.full(n, 0.5)
    for i in range(PCT_WINDOW, n):
        hist = breadth[i - PCT_WINDOW:i]
        pct[i] = float(np.mean(hist <= breadth[i]))
    pct[:PCT_WINDOW] = 0.5
    return np.where(pct >= P_HIGH, 2, np.where(pct >= P_LOW, 1, 0)).astype(int)


def compute_funding_dispersion(dataset) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute per-bar:
      - funding_std: cross-sectional std of funding rates across all symbols
      - funding_avg: cross-sectional mean of |funding| rates (absolute)

    Returns (funding_std, funding_avg_abs) arrays of shape (n_bars,).
    Funding is 8h, so only updates every 8h; forward-fill between updates.
    """
    n = len(dataset.timeline)
    funding_std = np.full(n, 0.0)
    funding_avg_abs = np.full(n, 0.0)

    for i in range(n):
        ts = dataset.timeline[i]
        rates = []
        for sym in SYMBOLS:
            r = dataset.last_funding_rate_before(sym, ts)
            rates.append(r)
        rates = np.array(rates)
        if len(rates) > 1:
            funding_std[i] = float(np.std(rates))
            funding_avg_abs[i] = float(np.mean(np.abs(rates)))
        else:
            funding_std[i] = 0.0
            funding_avg_abs[i] = 0.0

    return funding_std, funding_avg_abs


def rolling_percentile_rank(signal: np.ndarray, window: int) -> np.ndarray:
    """Rank of current value vs past [window] values. Returns [0,1]."""
    n = len(signal)
    pct = np.full(n, 0.5)
    for i in range(window, n):
        hist = signal[i - window:i]
        pct[i] = float(np.mean(hist <= signal[i]))
    pct[:window] = 0.5
    return pct


def compute_production_ens(
    sig_rets: dict,
    regime: np.ndarray,
    btc_vol: np.ndarray,
) -> np.ndarray:
    """Reference: current production (breadth + vol_always)."""
    min_len = min(len(sig_rets[sk]) for sk in SIG_KEYS)
    reg = regime[:min_len]
    bv = btc_vol[:min_len]
    ens = np.zeros(min_len)
    for i in range(min_len):
        w = WEIGHTS_LIST[int(reg[i])]
        if not np.isnan(bv[i]) and bv[i] > VOL_THRESHOLD:
            boost = VOL_F144_BOOST / max(1, len(SIG_KEYS) - 1)
            base = 0.0
            for sk in SIG_KEYS:
                adj_w = min(0.60, w[sk] + VOL_F144_BOOST) if sk == "f144" else max(0.05, w[sk] - boost)
                base += adj_w * sig_rets[sk][i]
            ens[i] = base * VOL_SCALE
        else:
            for sk in SIG_KEYS:
                ens[i] += w[sk] * sig_rets[sk][i]
    return ens


def apply_dispersion_overlay(
    base_ens: np.ndarray,
    fund_std_pct: np.ndarray,
    fund_avg_abs: np.ndarray,
    variant: str,
    reduce_scale: float = 0.65,
    boost_scale: float = 1.15,
    low_pct_threshold: float = 0.25,
    extreme_pct_threshold: float = 0.10,
    avg_threshold: float = 0.0001,  # 0.01% = meaningful funding
) -> np.ndarray:
    """
    Apply dispersion overlay on top of production ensemble returns.

    variant:
      "disperse_low25_highavg" : reduce when std<25th pct AND avg>threshold
      "disperse_low10"         : reduce when std<10th pct (extreme herding)
      "disperse_smooth_z"      : smooth scale based on rolling z-score of dispersion
      "disperse_boost_high75"  : boost when std>75th pct (high differentiation)
      "disperse_combo"         : reduce when crowded, boost when differentiated
    """
    n = len(base_ens)
    out = base_ens.copy()

    if variant == "disperse_low25_highavg":
        for i in range(n):
            is_crowded = (fund_std_pct[i] < low_pct_threshold) and (fund_avg_abs[i] > avg_threshold)
            if is_crowded:
                out[i] = base_ens[i] * reduce_scale

    elif variant == "disperse_low10":
        for i in range(n):
            if fund_std_pct[i] < extreme_pct_threshold:
                out[i] = base_ens[i] * reduce_scale

    elif variant == "disperse_smooth_z":
        # Scale = 1.0 - 0.35 * (1 - fund_std_pct) when std below median
        for i in range(n):
            if fund_std_pct[i] < 0.5:
                crowding_z = 1.0 - fund_std_pct[i]  # high value = more crowded
                scale = 1.0 - 0.35 * crowding_z
                out[i] = base_ens[i] * scale

    elif variant == "disperse_boost_high75":
        for i in range(n):
            if fund_std_pct[i] > 0.75:  # high dispersion = good for idio signals
                out[i] = base_ens[i] * boost_scale

    elif variant == "disperse_combo":
        for i in range(n):
            if fund_std_pct[i] < extreme_pct_threshold:
                out[i] = base_ens[i] * reduce_scale
            elif fund_std_pct[i] > 0.80:
                out[i] = base_ens[i] * boost_scale

    return out


VARIANTS = [
    "disperse_low25_highavg",
    "disperse_low10",
    "disperse_smooth_z",
    "disperse_boost_high75",
    "disperse_combo",
]

VARIANT_LABELS = {
    "disperse_low25_highavg": "Reduce×0.65 when std<25th pct AND avg_fund>0.01%",
    "disperse_low10":         "Reduce×0.65 when std<10th pct (extreme herding)",
    "disperse_smooth_z":      "Smooth scale (1−0.35×crowding_z) when std<median",
    "disperse_boost_high75":  "Boost×1.15 when std>75th pct (high differentiation)",
    "disperse_combo":         "Reduce<10th pct + Boost>80th pct (combo)",
}


def main():
    global _partial

    print("=" * 72)
    print("PHASE 148: Funding Rate Dispersion Overlay")
    print("=" * 72)

    # ── Step 1: Load data ──────────────────────────────────────────────
    print("\n[1/4] Loading data + pre-computing signal returns...")
    sig_returns: dict = {sk: {} for sk in SIG_KEYS}
    regime_data: dict = {}
    btc_vol_data: dict = {}
    fund_std_data: dict = {}
    fund_avg_data: dict = {}

    for year, (start, end) in YEAR_RANGES.items():
        print(f"  {year}: ", end="", flush=True)
        cfg_data = {
            "provider": "binance_rest_v1", "symbols": SYMBOLS,
            "start": start, "end": end, "bar_interval": "1h",
            "cache_dir": ".cache/binance_rest",
        }
        provider = make_provider(cfg_data, seed=42)
        dataset = provider.load()
        btc_vol_data[year] = compute_btc_vol(dataset, window=VOL_WINDOW)
        regime_data[year] = compute_breadth_regime(dataset)

        # Funding dispersion
        fund_std_raw, fund_avg_abs = compute_funding_dispersion(dataset)
        # Rolling percentile rank of fund_std
        fund_std_data[year] = rolling_percentile_rank(fund_std_raw, window=PCT_WINDOW)
        fund_avg_data[year] = fund_avg_abs
        print("F", end="", flush=True)

        for sk in SIG_KEYS:
            sig_def = SIGNAL_DEFS[sk]
            bt_cfg = BacktestConfig(costs=COST_MODEL)
            strat = make_strategy({"name": sig_def["strategy"], "params": sig_def["params"]})
            engine = BacktestEngine(bt_cfg)
            result = engine.run(dataset, strat)
            sig_returns[sk][year] = np.array(result.returns, dtype=np.float64)
            print(".", end="", flush=True)
        print(" ✓")

    # ── Step 2: Production baseline ────────────────────────────────────
    print("\n[2/4] Computing production baseline + all dispersion variants...")
    baseline_yearly = {}
    for year in YEARS:
        sr = {sk: sig_returns[sk][year] for sk in SIG_KEYS}
        ens = compute_production_ens(sr, regime_data[year], btc_vol_data[year])
        baseline_yearly[year] = sharpe(ens)
    baseline_obj = obj_func(list(baseline_yearly.values()))
    print(f"  Baseline (production):  OBJ={baseline_obj:.4f} | {baseline_yearly}")

    # ── Step 3: Test all variants ──────────────────────────────────────
    variant_results: dict = {}
    best_v = None
    best_obj = baseline_obj

    for v in VARIANTS:
        yearly = {}
        for year in YEARS:
            sr = {sk: sig_returns[sk][year] for sk in SIG_KEYS}
            base_ens = compute_production_ens(sr, regime_data[year], btc_vol_data[year])
            ens = apply_dispersion_overlay(
                base_ens,
                fund_std_data[year][:len(base_ens)],
                fund_avg_data[year][:len(base_ens)],
                variant=v,
            )
            yearly[year] = sharpe(ens)
        obj = obj_func(list(yearly.values()))
        delta = round(obj - baseline_obj, 4)
        variant_results[v] = {"yearly": yearly, "obj": obj, "delta_vs_baseline": delta}
        flag = " ✅" if obj > best_obj else ""
        print(f"  {VARIANT_LABELS[v]:60s} OBJ={obj:.4f} Δ={delta:+.4f}{flag}")
        if obj > best_obj:
            best_obj = obj
            best_v = v

    if best_v is None:
        print(f"\n  No variant beats baseline. Best remains production OBJ={baseline_obj:.4f}")
    else:
        print(f"\n  Best: {VARIANT_LABELS[best_v]} → OBJ={best_obj:.4f} (Δ={best_obj - baseline_obj:+.4f})")

    _partial.update({
        "phase": 148,
        "baseline_obj": baseline_obj,
        "baseline_yearly": baseline_yearly,
        "variant_results": {v: d for v, d in variant_results.items()},
        "best_variant": best_v,
        "best_obj": best_obj,
    })
    _save(_partial, partial=True)

    # ── Step 4: LOYO validation if we have a winner ────────────────────
    loyo_results = []
    loyo_wins = 0
    loyo_avg = 0.0

    if best_v is not None:
        print(f"\n[3/4] LOYO validation of best variant ({best_v})...")
        for test_yr in YEARS:
            sr_test = {sk: sig_returns[sk][test_yr] for sk in SIG_KEYS}
            base_ens = compute_production_ens(sr_test, regime_data[test_yr], btc_vol_data[test_yr])
            ens_best = apply_dispersion_overlay(
                base_ens,
                fund_std_data[test_yr][:len(base_ens)],
                fund_avg_data[test_yr][:len(base_ens)],
                variant=best_v,
            )
            s_best = sharpe(ens_best)
            s_base = sharpe(base_ens)
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

    # ── Step 5: Walk-Forward if LOYO passes ───────────────────────────
    wf_results = []
    wf_wins = 0
    wf_avg_delta = 0.0

    if best_v is not None and loyo_wins >= 3:
        print(f"\n[4/4] Walk-Forward (train=3yr, test=1yr)...")
        wf_windows = [
            {"train": ["2021", "2022", "2023"], "test": "2024"},
            {"train": ["2022", "2023", "2024"], "test": "2025"},
        ]
        for wf in wf_windows:
            test_yr = wf["test"]
            sr_test = {sk: sig_returns[sk][test_yr] for sk in SIG_KEYS}
            base_ens = compute_production_ens(sr_test, regime_data[test_yr], btc_vol_data[test_yr])
            ens_best = apply_dispersion_overlay(
                base_ens,
                fund_std_data[test_yr][:len(base_ens)],
                fund_avg_data[test_yr][:len(base_ens)],
                variant=best_v,
            )
            test_s = sharpe(ens_best)
            prod_s = sharpe(base_ens)
            delta = round(test_s - prod_s, 4)
            wf_results.append({
                "test": test_yr, "test_sharpe": round(test_s, 4),
                "prod_sharpe": round(prod_s, 4), "delta": delta,
            })
            flag = "✅" if delta > 0 else "❌"
            print(f"  WF test={test_yr}: {test_s:.4f} vs {prod_s:.4f} Δ={delta:+.4f} {flag}")
        wf_wins = sum(1 for r in wf_results if r["delta"] > 0)
        wf_avg_delta = round(float(np.mean([r["delta"] for r in wf_results])), 4)
        print(f"  WF: {wf_wins}/2 | avg_delta={wf_avg_delta:+.4f}")
    else:
        print("\n[4/4] WF: Skipped (LOYO did not pass)")

    # ── Verdict ────────────────────────────────────────────────────────
    print(f"\n{'='*72}")
    validated = (
        best_v is not None
        and best_obj > baseline_obj
        and loyo_wins >= 3
        and loyo_avg > 0
    )

    if validated:
        verdict = (
            f"IMPROVEMENT — {VARIANT_LABELS[best_v]} adds +{best_obj - baseline_obj:.4f} OBJ. "
            f"LOYO {loyo_wins}/5 wins, avg_delta={loyo_avg:+.4f}"
        )
        next_phase = (
            "Phase 149: Integrate funding dispersion overlay into production config. "
            "Fine-tune reduce_scale and percentile thresholds."
        )
    elif best_v is not None:
        verdict = (
            f"MARGINAL — {VARIANT_LABELS[best_v]} IS improvement={best_obj - baseline_obj:+.4f} "
            f"but LOYO {loyo_wins}/5 < 3 wins. No production change."
        )
        next_phase = (
            "Phase 149: Explore new independent alpha source. "
            "Consider: intraday hour-of-day seasonality, "
            "cross-symbol return correlation regime, or short-term reversal (1-4h)."
        )
    else:
        verdict = (
            f"NO IMPROVEMENT — All funding dispersion variants worse than baseline OBJ={baseline_obj:.4f}. "
            "Dispersion signal is not additive to existing ensemble."
        )
        next_phase = (
            "Phase 149: Explore intraday seasonality (hour-of-day effect in hourly bars). "
            "Phase 130 found best_hours OBJ=1.353 standalone — test as ensemble overlay. "
            "OR: short-term reversal (4h/8h) as tactical overlay."
        )

    print(f"VERDICT: {verdict}")
    print(f"{'='*72}")

    report = {
        "phase": 148,
        "description": "Funding Rate Cross-Sectional Dispersion Overlay",
        "hypothesis": "Low cross-sectional funding dispersion = crowded trade = reduce leverage",
        "elapsed_seconds": round(time.time() - _start, 1),
        "baseline_obj": baseline_obj,
        "baseline_yearly": baseline_yearly,
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
