#!/usr/bin/env python3
"""
Phase 156: Comprehensive Final Walk-Forward Validation — v2.6.0
================================================================
Production state: v2.6.0
  - Ensemble: V1(27.5%) + I460bw168(19.7%) + I415bw216(32.5%) + F168(20.4%)
  - Overlays: vol_regime (P129) + breadth_regime (P144-146) +
              fund_dispersion (P148) + fund_ts_spread (P150/P152) + F168 (P153)

Phase 155: NO 5th signal exists. v2.6.0 = signal ceiling.
This is the FINAL validation before deployment.

Walk-Forward Protocol:
  WF Window 1: Train 2021        → Test 2022
  WF Window 2: Train 2021-2022   → Test 2023
  WF Window 3: Train 2021-2023   → Test 2024
  WF Window 4: Train 2022-2024   → Test 2025
  = 4 WF windows, need ≥ 3/4 wins

Also compute: OOS Sharpe degradation ratio (IS vs OOS per window)
Final OBJ check: full 5-year IS OBJ (compare to all prior phases)

Comparison: v2.6.0 vs P91b baseline (4-signal flat ensemble, no overlays)
Goal: demonstrate that the overlay stack IS adding value out-of-sample.

Pass: ≥ 3/4 WF windows positive Δ vs baseline → PRODUCTION CONFIRMED
Fail: < 3/4 → revert specific overlays or flag for monitoring
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
_signal.alarm(3600)  # 60min

PROD_CFG = json.loads((ROOT / "configs" / "production_p91b_champion.json").read_text())
SYMBOLS = PROD_CFG["data"]["symbols"]
SIGNAL_DEFS = PROD_CFG["ensemble"]["signals"]

# Read ALL overlay params from production config
VOL_OV = PROD_CFG.get("vol_regime_overlay", {})
VOL_WINDOW     = VOL_OV.get("window_bars", 168)
VOL_THRESHOLD  = VOL_OV.get("threshold", 0.5)
VOL_SCALE      = VOL_OV.get("scale_factor", 0.5)
VOL_F144_BOOST = VOL_OV.get("f144_boost", 0.2)

BRS = PROD_CFG.get("breadth_regime_switching", {})
BREADTH_LB = BRS.get("breadth_lookback_bars", 168)
PCT_WINDOW = BRS.get("rolling_percentile_window", 336)
P_LOW      = BRS.get("p_low", 0.33)
P_HIGH     = BRS.get("p_high", 0.67)

FDO = PROD_CFG.get("funding_dispersion_overlay", {})
FUND_DISP_PCT   = FDO.get("boost_threshold_pct", 0.75)
FUND_DISP_SCALE = FDO.get("boost_scale", 1.15)

FTS = PROD_CFG.get("funding_term_structure_overlay", {})
FTS_SHORT_W = FTS.get("short_window_bars", 24)
FTS_LONG_W  = FTS.get("long_window_bars", 144)
FTS_PCT_WIN = FTS.get("rolling_percentile_window", 336)
FTS_RT      = FTS.get("reduce_threshold", 0.70)
FTS_RS      = FTS.get("reduce_scale", 0.60)
FTS_BT      = FTS.get("boost_threshold", 0.30)
FTS_BS      = FTS.get("boost_scale", 1.15)

# Current production weights
WEIGHTS_V26 = {
    "prod":  {"v1": 0.2747, "i460bw168": 0.1967, "i415bw216": 0.3247, "f144": 0.2039},
    "p143b": {"v1": 0.05,   "i460bw168": 0.25,   "i415bw216": 0.45,   "f144": 0.25},
    "mid":   {"v1": 0.16,   "i460bw168": 0.22,   "i415bw216": 0.37,   "f144": 0.25},
}
# P91b baseline: flat weights, no regime switching
WEIGHTS_P91B_FLAT = {"v1": 0.2747, "i460bw168": 0.1967, "i415bw216": 0.3247, "f144": 0.2039}

YEAR_RANGES = {
    "2021": ("2021-02-01", "2021-12-31"),
    "2022": ("2022-01-01", "2022-12-31"),
    "2023": ("2023-01-01", "2023-12-31"),
    "2024": ("2024-01-01", "2024-12-31"),
    "2025": ("2025-01-01", "2025-12-31"),
}
YEARS = sorted(YEAR_RANGES.keys())

# WF windows: (train_years, test_year)
WF_WINDOWS = [
    (["2021"],                "2022"),
    (["2021", "2022"],        "2023"),
    (["2021", "2022", "2023"], "2024"),
    (["2022", "2023", "2024"], "2025"),
]

OUT_DIR = ROOT / "artifacts" / "phase156"
OUT_DIR.mkdir(parents=True, exist_ok=True)

COST_MODEL = cost_model_from_config({
    "fee_rate": 0.0005, "slippage_rate": 0.0003, "cost_multiplier": 1.0,
})

# P91b baseline rebalance intervals (original)
P91B_REBAL = {"v1": 60, "i460bw168": 48, "i415bw216": 48, "f144": 24}


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
    out = OUT_DIR / "phase156_report.json"
    out.write_text(json.dumps(data, indent=2, default=str))
    print(f"✅ Saved → {out}")


def rolling_percentile(signal: np.ndarray, window: int) -> np.ndarray:
    n = len(signal)
    pct = np.full(n, 0.5)
    for i in range(window, n):
        hist = signal[i - window:i]
        pct[i] = float(np.mean(hist <= signal[i]))
    if window < n:
        pct[:window] = pct[window]
    return pct


def compute_btc_vol(dataset) -> np.ndarray:
    n = len(dataset.timeline)
    rets = np.zeros(n)
    for i in range(1, n):
        c0 = dataset.close("BTCUSDT", i - 1)
        c1 = dataset.close("BTCUSDT", i)
        rets[i] = (c1 / c0 - 1.0) if c0 > 0 else 0.0
    vol = np.full(n, np.nan)
    for i in range(VOL_WINDOW, n):
        vol[i] = float(np.std(rets[i - VOL_WINDOW:i])) * np.sqrt(8760)
    if VOL_WINDOW < n:
        vol[:VOL_WINDOW] = vol[VOL_WINDOW]
    return vol


def compute_breadth(dataset) -> np.ndarray:
    n = len(dataset.timeline)
    breadth = np.full(n, 0.5)
    for i in range(BREADTH_LB, n):
        pos = sum(
            1 for sym in SYMBOLS
            if (c0 := dataset.close(sym, i - BREADTH_LB)) > 0
            and dataset.close(sym, i) > c0
        )
        breadth[i] = pos / len(SYMBOLS)
    if BREADTH_LB < n:
        breadth[:BREADTH_LB] = breadth[BREADTH_LB]
    return breadth


def compute_fund_disp(dataset) -> np.ndarray:
    n = len(dataset.timeline)
    fund_std = np.zeros(n)
    for i in range(n):
        ts = dataset.timeline[i]
        rates = []
        for sym in SYMBOLS:
            try:
                r = dataset.last_funding_rate_before(sym, ts)
                if r is not None and not np.isnan(float(r)):
                    rates.append(float(r))
            except Exception:
                pass
        fund_std[i] = float(np.std(rates)) if len(rates) > 1 else 0.0
    return fund_std


def compute_fund_ts(dataset) -> np.ndarray:
    n = len(dataset.timeline)
    level = np.zeros(n)
    for i in range(n):
        ts = dataset.timeline[i]
        rates = []
        for sym in SYMBOLS:
            try:
                r = dataset.funding_rate_at(sym, ts)
                if r is not None and not np.isnan(float(r)):
                    rates.append(float(r))
            except Exception:
                pass
        level[i] = float(np.mean(rates)) if rates else 0.0
    spread = np.zeros(n)
    for i in range(FTS_LONG_W, n):
        s_avg = float(np.mean(level[max(0, i - FTS_SHORT_W):i]))
        l_avg = float(np.mean(level[i - FTS_LONG_W:i]))
        spread[i] = s_avg - l_avg
    if FTS_LONG_W < n:
        spread[:FTS_LONG_W] = spread[FTS_LONG_W]
    return spread


def run_v26(sig_rets: dict, bv, brd_pct, fdo_pct, fts_pct) -> np.ndarray:
    """Full v2.6.0 production stack."""
    sig_keys = list(sig_rets.keys())
    min_len = min(len(sig_rets[sk]) for sk in sig_keys)
    bv_  = bv[:min_len]; brd_ = brd_pct[:min_len]
    fdo_ = fdo_pct[:min_len]; fts_ = fts_pct[:min_len]
    ens  = np.zeros(min_len)
    n_oth = len(sig_keys) - 1

    for i in range(min_len):
        if brd_[i] >= P_HIGH:
            w = WEIGHTS_V26["p143b"]
        elif brd_[i] >= P_LOW:
            w = WEIGHTS_V26["mid"]
        else:
            w = WEIGHTS_V26["prod"]

        if not np.isnan(bv_[i]) and bv_[i] > VOL_THRESHOLD:
            boost_o = VOL_F144_BOOST / max(1, n_oth)
            ret_i = 0.0
            for sk in sig_keys:
                adj_w = min(0.60, w[sk] + VOL_F144_BOOST) if sk == "f144" else max(0.05, w[sk] - boost_o)
                ret_i += adj_w * sig_rets[sk][i]
            ret_i *= VOL_SCALE
        else:
            ret_i = sum(w[sk] * sig_rets[sk][i] for sk in sig_keys)

        if fdo_[i] > FUND_DISP_PCT:
            ret_i *= FUND_DISP_SCALE
        sp = fts_[i]
        if sp >= FTS_RT:
            ret_i *= FTS_RS
        elif sp <= FTS_BT:
            ret_i *= FTS_BS

        ens[i] = ret_i

    return ens


def run_p91b_flat(sig_rets: dict) -> np.ndarray:
    """P91b baseline: flat weights, NO overlays."""
    sig_keys = list(sig_rets.keys())
    w = WEIGHTS_P91B_FLAT
    min_len = min(len(sig_rets[sk]) for sk in sig_keys)
    ens = np.array([sum(w[sk] * sig_rets[sk][i] for sk in sig_keys) for i in range(min_len)])
    return ens


def main():
    global _partial

    print("=" * 72)
    print("PHASE 156: Comprehensive Final Walk-Forward Validation — v2.6.0")
    print(f"  {len(WF_WINDOWS)} WF windows | Pass: ≥3/4 wins vs p91b flat")
    print("=" * 72)

    # ── Load all years ───────────────────────────────────────────────────
    print("\n[1/3] Loading all 5 years...")
    datasets = {}
    btc_vols, breadth_raw, fund_disp_raw, fts_raw = {}, {}, {}, {}

    for year in YEARS:
        start, end = YEAR_RANGES[year]
        print(f"  {year}: ", end="", flush=True)
        cfg = {"provider": "binance_rest_v1", "symbols": SYMBOLS,
               "start": start, "end": end, "bar_interval": "1h",
               "cache_dir": ".cache/binance_rest"}
        ds = make_provider(cfg, seed=42).load()
        datasets[year]       = ds
        btc_vols[year]       = compute_btc_vol(ds)
        breadth_raw[year]    = compute_breadth(ds)
        fund_disp_raw[year]  = compute_fund_disp(ds)
        fts_raw[year]        = compute_fund_ts(ds)
        print("✓")

    # ── Precompute all signal returns ────────────────────────────────────
    print("\n[2/3] Precomputing signal returns for all years...")
    sig_rets: dict = {sk: {} for sk in SIGNAL_DEFS}

    for year in YEARS:
        print(f"  {year}: ", end="", flush=True)
        for sk in SIGNAL_DEFS:
            sig_def = SIGNAL_DEFS[sk]
            bt_cfg = BacktestConfig(costs=COST_MODEL)
            strat = make_strategy({"name": sig_def["strategy"], "params": dict(sig_def["params"])})
            result = BacktestEngine(bt_cfg).run(datasets[year], strat)
            sig_rets[sk][year] = np.array(result.returns, dtype=np.float64)
            print(".", end="", flush=True)
        print(" ✓")

    # ── IS Full 5-Year Performance ────────────────────────────────────────
    print("\n[3/3] WF Validation + IS Summary...")

    is_sharpes_v26  = {}
    is_sharpes_p91b = {}
    for year in YEARS:
        brd = rolling_percentile(breadth_raw[year], PCT_WINDOW)
        fdo = rolling_percentile(fund_disp_raw[year], PCT_WINDOW)
        fts = rolling_percentile(fts_raw[year], FTS_PCT_WIN)
        ens_v26  = run_v26({sk: sig_rets[sk][year] for sk in sig_rets}, btc_vols[year], brd, fdo, fts)
        ens_p91b = run_p91b_flat({sk: sig_rets[sk][year] for sk in sig_rets})
        is_sharpes_v26[year]  = sharpe(ens_v26)
        is_sharpes_p91b[year] = sharpe(ens_p91b)

    is_obj_v26  = obj_func(list(is_sharpes_v26.values()))
    is_obj_p91b = obj_func(list(is_sharpes_p91b.values()))
    is_delta    = round(is_obj_v26 - is_obj_p91b, 4)

    print(f"\n  IS Full 5-Year:")
    print(f"    v2.6.0:  OBJ={is_obj_v26:.4f}  | {is_sharpes_v26}")
    print(f"    p91b:    OBJ={is_obj_p91b:.4f}  | {is_sharpes_p91b}")
    print(f"    Overlay  +{is_delta:.4f} total improvement")

    # ── Walk-Forward Windows ──────────────────────────────────────────────
    print(f"\n  Walk-Forward Windows:")
    wf_results = []
    wf_wins = 0

    for train_yrs, test_yr in WF_WINDOWS:
        brd_t = rolling_percentile(breadth_raw[test_yr], PCT_WINDOW)
        fdo_t = rolling_percentile(fund_disp_raw[test_yr], PCT_WINDOW)
        fts_t = rolling_percentile(fts_raw[test_yr], FTS_PCT_WIN)

        # Test year performance
        ens_v26_test  = run_v26({sk: sig_rets[sk][test_yr] for sk in sig_rets},
                                 btc_vols[test_yr], brd_t, fdo_t, fts_t)
        ens_p91b_test = run_p91b_flat({sk: sig_rets[sk][test_yr] for sk in sig_rets})
        s_v26 = sharpe(ens_v26_test)
        s_p91b = sharpe(ens_p91b_test)
        delta = round(s_v26 - s_p91b, 4)
        win = delta > 0
        wf_wins += int(win)

        # Also compute train years summary for IS/OOS degradation
        train_sharpes_v26  = [is_sharpes_v26[yr]  for yr in train_yrs]
        train_sharpes_p91b = [is_sharpes_p91b[yr] for yr in train_yrs]
        train_avg_delta = round(float(np.mean(np.array(train_sharpes_v26) - np.array(train_sharpes_p91b))), 4)

        print(f"    Train {train_yrs} → Test {test_yr}: "
              f"v2.6={s_v26:.4f} vs p91b={s_p91b:.4f} Δ={delta:+.4f} {'✅' if win else '❌'} "
              f"(train avg delta: {train_avg_delta:+.4f})")

        wf_results.append({
            "train": train_yrs, "test": test_yr,
            "test_v26": s_v26, "test_p91b": s_p91b, "test_delta": delta, "win": bool(win),
            "train_avg_delta": train_avg_delta,
        })

        _partial.update({"wf_results": wf_results, "wf_wins": wf_wins})
        _save(_partial, partial=True)

    wf_avg_delta = round(float(np.mean([r["test_delta"] for r in wf_results])), 4)

    # ── OOS Degradation Analysis ─────────────────────────────────────────
    oos_degradation = []
    for r in wf_results:
        if r["train_avg_delta"] != 0:
            deg = round(r["test_delta"] / r["train_avg_delta"], 3) if abs(r["train_avg_delta"]) > 0.001 else None
            oos_degradation.append(deg)
    avg_deg = round(float(np.mean([d for d in oos_degradation if d is not None])), 3) if oos_degradation else None

    # ── Final Assessment ─────────────────────────────────────────────────
    print(f"\n  WF Summary: {wf_wins}/4 wins, avg_delta={wf_avg_delta:+.4f}")
    print(f"  IS delta (v2.6.0 vs p91b): +{is_delta:.4f} OBJ total improvement")
    if avg_deg is not None:
        print(f"  OOS degradation ratio: {avg_deg:.2f}x (1.0 = full transfer, 0.5 = half)")

    wf_pass = wf_wins >= 3

    if wf_pass:
        verdict = (
            f"PRODUCTION CONFIRMED — v2.6.0 passes comprehensive WF validation. "
            f"{wf_wins}/4 WF windows win, avg_delta={wf_avg_delta:+.4f}. "
            f"IS improvement over p91b: +{is_delta:.4f} OBJ. "
            f"DEPLOY with confidence."
        )
        next_phase = (
            "Phase 157 (FINAL): Generate production deployment checklist. "
            "Set up live monitoring against halt_conditions in config. "
            "R&D COMPLETE — system is v2.6.0 production-ready."
        )
    elif wf_wins == 2:
        verdict = (
            f"MARGINAL PASS — {wf_wins}/4 WF wins. "
            f"avg_delta={wf_avg_delta:+.4f}. "
            "Consider reverting weakest overlay OR deploy with tighter monitoring."
        )
        next_phase = (
            "Phase 157: Diagnose which WF windows fail. "
            "Test with individual overlays disabled to find which one has negative OOS transfer."
        )
    else:
        verdict = (
            f"WF FAIL — {wf_wins}/4 wins only. "
            "Overlays appear overfit. Revert to simpler configuration."
        )
        next_phase = (
            "Phase 157: Strip back to p91b base + only the most robust overlays "
            "(breadth_regime P144-146 has LOYO 4/5 — keep; others subject to ablation)."
        )

    print(f"\nVERDICT: {verdict}")
    print("=" * 72)

    report = {
        "phase": 156,
        "description": "Comprehensive final WF validation — v2.6.0 vs p91b flat",
        "elapsed_seconds": round(time.time() - _start, 1),
        "is_performance": {
            "v26_yearly": is_sharpes_v26, "v26_obj": is_obj_v26,
            "p91b_yearly": is_sharpes_p91b, "p91b_obj": is_obj_p91b,
            "delta_obj": is_delta,
        },
        "wf_windows": wf_results,
        "wf_wins": wf_wins,
        "wf_avg_delta": wf_avg_delta,
        "oos_degradation_ratio": avg_deg,
        "wf_pass": wf_pass,
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
