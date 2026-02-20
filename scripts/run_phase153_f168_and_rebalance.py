#!/usr/bin/env python3
"""
Phase 153: F168 Confirmation + Rebalance Interval Optimization
================================================================
Production state: OBJ=2.0851 (v2.5.0: breadth+vol+fund_disp+fund_ts fine-tuned)
P152a finding: F168 > F144 (+0.0303 OBJ, LOYO 4/5) on P150 stack

Two tests in one phase:

TEST A — F168 confirmation on v2.5.0 stack
  Baseline: F144, v2.5.0 stack (FTS fine-tuned: rt=0.70, rs=0.60, bt=0.30, bs=1.15)
  Test: F168 (same stack)
  Expected: +0.03 OBJ from P152a finding

TEST B — Rebalance interval compression
  Hypothesis: Faster rebalancing → better signal freshness → lower slippage vs alpha decay tradeoff
  Currently: V1=60h, I460=48h, I415=48h, F144=24h
  Test: V1=48h, I460=36h, I415=36h, F144=24h (keep F same)
  Also test: V1=48h, I460=48h, I415=48h (only V1 compressed)
  And: V1=60h, I460=36h, I415=36h (only idio compressed)

All tests use full v2.5.0 production stack.
Pass: OBJ > 2.0851 AND LOYO >= 3/5
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
_signal.alarm(3000)  # 50min

PROD_CFG = json.loads((ROOT / "configs" / "production_p91b_champion.json").read_text())
SYMBOLS = PROD_CFG["data"]["symbols"]
SIGNAL_DEFS = PROD_CFG["ensemble"]["signals"]

# Overlay params
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
FUND_DISP_PCT  = FDO.get("boost_threshold_pct", 0.75)
FUND_DISP_SCALE = FDO.get("boost_scale", 1.15)

FTS = PROD_CFG.get("funding_term_structure_overlay", {})
FTS_SHORT_W    = FTS.get("short_window_bars", 24)
FTS_LONG_W     = FTS.get("long_window_bars", 144)
FTS_PCT_WIN    = FTS.get("rolling_percentile_window", 336)
FTS_RT         = FTS.get("reduce_threshold", 0.70)
FTS_RS         = FTS.get("reduce_scale", 0.60)
FTS_BT         = FTS.get("boost_threshold", 0.30)
FTS_BS         = FTS.get("boost_scale", 1.15)

WEIGHTS = {
    "prod":  {"v1": 0.2747, "i460bw168": 0.1967, "i415bw216": 0.3247, "f144": 0.2039},
    "p143b": {"v1": 0.05,   "i460bw168": 0.25,   "i415bw216": 0.45,   "f144": 0.25},
    "mid":   {"v1": 0.16,   "i460bw168": 0.22,   "i415bw216": 0.37,   "f144": 0.25},
}

YEAR_RANGES = {
    "2021": ("2021-02-01", "2021-12-31"),
    "2022": ("2022-01-01", "2022-12-31"),
    "2023": ("2023-01-01", "2023-12-31"),
    "2024": ("2024-01-01", "2024-12-31"),
    "2025": ("2025-01-01", "2025-12-31"),
}
YEARS = sorted(YEAR_RANGES.keys())

BASELINE_OBJ = 2.0851
F_BASELINE   = 144
F_TEST       = 168

# Rebalance interval test configurations
# Format: dict of sig_key → rebalance_interval_bars override
REBAL_CONFIGS = {
    "prod_rebal":   {"v1": 60, "i460bw168": 48, "i415bw216": 48, "f144": 24},  # current/baseline
    "v1_48_fast":   {"v1": 48, "i460bw168": 48, "i415bw216": 48, "f144": 24},  # only V1 faster
    "idio_36_fast": {"v1": 60, "i460bw168": 36, "i415bw216": 36, "f144": 24},  # only idio faster
    "all_fast":     {"v1": 48, "i460bw168": 36, "i415bw216": 36, "f144": 24},  # all faster
}

OUT_DIR = ROOT / "artifacts" / "phase153"
OUT_DIR.mkdir(parents=True, exist_ok=True)

COST_MODEL = cost_model_from_config({
    "fee_rate": 0.0005, "slippage_rate": 0.0003, "cost_multiplier": 1.0,
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
    out = OUT_DIR / "phase153_report.json"
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


def blend_full_stack(sig_rets: dict, bv, brd_pct, fdo_pct, fts_pct) -> np.ndarray:
    """Full v2.5.0 production ensemble."""
    sig_keys = list(sig_rets.keys())
    min_len = min(len(sig_rets[sk]) for sk in sig_keys)
    bv  = bv[:min_len]
    brd = brd_pct[:min_len]
    fdo = fdo_pct[:min_len]
    fts = fts_pct[:min_len]
    ens = np.zeros(min_len)
    n_others = len(sig_keys) - 1

    for i in range(min_len):
        if brd[i] >= P_HIGH:
            w = WEIGHTS["p143b"]
        elif brd[i] >= P_LOW:
            w = WEIGHTS["mid"]
        else:
            w = WEIGHTS["prod"]

        if not np.isnan(bv[i]) and bv[i] > VOL_THRESHOLD:
            boost_other = VOL_F144_BOOST / max(1, n_others)
            ret_i = 0.0
            for sk in sig_keys:
                if sk == "f144":
                    adj_w = min(0.60, w[sk] + VOL_F144_BOOST)
                else:
                    adj_w = max(0.05, w[sk] - boost_other)
                ret_i += adj_w * sig_rets[sk][i]
            ret_i *= VOL_SCALE
        else:
            ret_i = sum(w[sk] * sig_rets[sk][i] for sk in sig_keys)

        if fdo[i] > FUND_DISP_PCT:
            ret_i *= FUND_DISP_SCALE

        sp = fts[i]
        if sp >= FTS_RT:
            ret_i *= FTS_RS
        elif sp <= FTS_BT:
            ret_i *= FTS_BS

        ens[i] = ret_i

    return ens


def main():
    global _partial

    print("=" * 72)
    print("PHASE 153: F168 Confirmation + Rebalance Interval Optimization")
    print(f"  Baseline: F144 prod_rebal, OBJ={BASELINE_OBJ} (v2.5.0)")
    print("=" * 72)

    # ── Load data + precompute overlays ────────────────────────────────
    print("\n[1/3] Loading data + overlays...")

    datasets: dict = {}
    btc_vols, breadth_raw, fund_disp_raw, fts_raw = {}, {}, {}, {}

    for year, (start, end) in YEAR_RANGES.items():
        print(f"  {year}: ", end="", flush=True)
        cfg_data = {
            "provider": "binance_rest_v1", "symbols": SYMBOLS,
            "start": start, "end": end, "bar_interval": "1h",
            "cache_dir": ".cache/binance_rest",
        }
        provider = make_provider(cfg_data, seed=42)
        dataset = provider.load()
        datasets[year] = dataset

        btc_vols[year]    = compute_btc_vol(dataset)
        breadth_raw[year] = compute_breadth(dataset)
        fund_disp_raw[year] = compute_fund_disp(dataset)
        fts_raw[year]     = compute_fund_ts(dataset)
        print("✓")

    # ── Step 2: Test A — F168 vs F144 on v2.5.0 stack ─────────────────
    print("\n[2/3] TEST A: F168 vs F144 on v2.5.0 stack...")

    test_a_results = {}
    for f_lb in [F_BASELINE, F_TEST]:
        label = f"F{f_lb}"
        yr_sharpes = {}
        for year in YEARS:
            brd_pct = rolling_percentile(breadth_raw[year], PCT_WINDOW)
            fdo_pct = rolling_percentile(fund_disp_raw[year], PCT_WINDOW)
            fts_pct = rolling_percentile(fts_raw[year], FTS_PCT_WIN)

            sig_rets = {}
            prod_sigs = SIGNAL_DEFS
            rebal_map = {"v1": 60, "i460bw168": 48, "i415bw216": 48, "f144": 24}

            for sk in ["v1", "i460bw168", "i415bw216"]:
                sig_def = prod_sigs[sk]
                p = dict(sig_def["params"])
                bt_cfg = BacktestConfig(costs=COST_MODEL)
                strat = make_strategy({"name": sig_def["strategy"], "params": p})
                result = BacktestEngine(bt_cfg).run(datasets[year], strat)
                sig_rets[sk] = np.array(result.returns, dtype=np.float64)

            f_params = dict(prod_sigs["f144"]["params"])
            f_params["funding_lookback_bars"] = f_lb
            bt_cfg = BacktestConfig(costs=COST_MODEL)
            strat = make_strategy({"name": prod_sigs["f144"]["strategy"], "params": f_params})
            result = BacktestEngine(bt_cfg).run(datasets[year], strat)
            sig_rets["f144"] = np.array(result.returns, dtype=np.float64)

            ens = blend_full_stack(sig_rets, btc_vols[year], brd_pct, fdo_pct, fts_pct)
            yr_sharpes[year] = sharpe(ens)

        obj = obj_func(list(yr_sharpes.values()))
        test_a_results[label] = {"yearly": yr_sharpes, "obj": obj}
        delta = round(obj - BASELINE_OBJ, 4)
        flag = " ← BASELINE" if f_lb == F_BASELINE else (" ✅" if obj > BASELINE_OBJ else "")
        print(f"  {label}: OBJ={obj:.4f} Δ={delta:+.4f} | {yr_sharpes}{flag}")

        _partial.update({"phase": 153, "test_a": test_a_results})
        _save(_partial, partial=True)

    # ── Step 2b: TEST B — Rebalance interval on F144 baseline ─────────
    print("\n[3/3] TEST B: Rebalance interval variants on v2.5.0 stack + F144...")
    # Only recompute signals for non-baseline rebal configs (prod_rebal already done above)

    test_b_results = {}
    f144_rebal_ref = {year: None for year in YEARS}  # store prod_rebal f144 rets

    for cfg_label, rebal_map in REBAL_CONFIGS.items():
        yr_sharpes = {}
        for year in YEARS:
            brd_pct = rolling_percentile(breadth_raw[year], PCT_WINDOW)
            fdo_pct = rolling_percentile(fund_disp_raw[year], PCT_WINDOW)
            fts_pct = rolling_percentile(fts_raw[year], FTS_PCT_WIN)

            sig_rets = {}
            for sk in ["v1", "i460bw168", "i415bw216", "f144"]:
                sig_def = SIGNAL_DEFS[sk]
                p = dict(sig_def["params"])
                p["rebalance_interval_bars"] = rebal_map[sk]
                if sk == "f144":
                    p["funding_lookback_bars"] = F_BASELINE  # keep F144 for test B
                bt_cfg = BacktestConfig(costs=COST_MODEL)
                strat = make_strategy({"name": sig_def["strategy"], "params": p})
                result = BacktestEngine(bt_cfg).run(datasets[year], strat)
                sig_rets[sk] = np.array(result.returns, dtype=np.float64)

            ens = blend_full_stack(sig_rets, btc_vols[year], brd_pct, fdo_pct, fts_pct)
            yr_sharpes[year] = sharpe(ens)

        obj = obj_func(list(yr_sharpes.values()))
        test_b_results[cfg_label] = {"yearly": yr_sharpes, "obj": obj, "rebal": rebal_map}
        delta = round(obj - BASELINE_OBJ, 4)
        flag = " ← BASELINE" if cfg_label == "prod_rebal" else (" ✅" if obj > BASELINE_OBJ else "")
        print(f"  {cfg_label:16s}: OBJ={obj:.4f} Δ={delta:+.4f} | {yr_sharpes}{flag}")

        _partial.update({"test_b": test_b_results})
        _save(_partial, partial=True)

    # ── Summary ──────────────────────────────────────────────────────────
    print("\n" + "=" * 72)
    baseline_a_obj = test_a_results.get("F144", {}).get("obj", BASELINE_OBJ)
    f168_obj = test_a_results.get("F168", {}).get("obj", 0)
    best_rebal = max(test_b_results.items(), key=lambda x: x[1]["obj"])
    best_rebal_label, best_rebal_data = best_rebal

    print("TEST A (F-lookback):")
    for l, d in test_a_results.items():
        delta = round(d["obj"] - baseline_a_obj, 4)
        flag = " ← BASELINE" if l == "F144" else (" ✅" if d["obj"] > baseline_a_obj else "")
        print(f"  {l}: OBJ={d['obj']:.4f} Δ={delta:+.4f}{flag}")

    print("TEST B (rebalance intervals):")
    for l, d in sorted(test_b_results.items(), key=lambda x: -x[1]["obj"]):
        delta = round(d["obj"] - BASELINE_OBJ, 4)
        flag = " ← BASELINE" if l == "prod_rebal" else (" ✅" if d["obj"] > BASELINE_OBJ else "")
        print(f"  {l:16s}: OBJ={d['obj']:.4f} Δ={delta:+.4f}{flag}")

    # Best overall finding
    f168_wins = f168_obj > baseline_a_obj
    rebal_wins = best_rebal_data["obj"] > BASELINE_OBJ and best_rebal_label != "prod_rebal"

    if f168_wins and f168_obj > BASELINE_OBJ:
        verdict = (
            f"F168 CONFIRMED on v2.5.0 — OBJ={f168_obj:.4f} (+{f168_obj-BASELINE_OBJ:.4f}). "
            f"Update production funding_lookback_bars=168."
        )
        next_phase = "Phase 154: LOYO+WF validation of F168 on v2.5.0. If passes → prod v2.6.0."
    elif rebal_wins:
        verdict = (
            f"REBAL IMPROVEMENT — {best_rebal_label} OBJ={best_rebal_data['obj']:.4f} "
            f"(+{best_rebal_data['obj']-BASELINE_OBJ:.4f}). F168 no improvement."
        )
        next_phase = f"Phase 154: LOYO validation of {best_rebal_label} rebal config."
    else:
        verdict = (
            f"NO IMPROVEMENT — F168 on v2.5.0 not better ({f168_obj:.4f} vs {baseline_a_obj:.4f}). "
            f"Rebal variants also no improvement. Current v2.5.0 is production-optimal."
        )
        next_phase = (
            "Phase 154: Explore new signal source. "
            "(a) Open Interest momentum — test if OI direction predicts return direction. "
            "(b) Taker buy volume ratio — aggregate direction signal. "
            "(c) Accept v2.5.0 and deploy."
        )

    print(f"\nVERDICT: {verdict}")
    print("=" * 72)

    report = {
        "phase": 153,
        "description": "F168 confirmation on v2.5.0 + rebalance interval tuning",
        "elapsed_seconds": round(time.time() - _start, 1),
        "baseline_obj": BASELINE_OBJ,
        "test_a_f_lookback": test_a_results,
        "test_b_rebalance": test_b_results,
        "f168_obj": f168_obj,
        "best_rebal": best_rebal_label,
        "best_rebal_obj": best_rebal_data["obj"],
        "f168_wins": f168_wins,
        "rebal_wins": bool(rebal_wins),
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
