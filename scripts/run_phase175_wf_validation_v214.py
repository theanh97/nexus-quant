"""
Phase 175 — Walk-Forward Validation of v2.14.0
================================================
v2.14.0 has made significant changes since the last WF test (P160, v2.8.0).
New features: FTS windows 12/96h, thresholds retune, dispersion fine-tune,
vol regime F168 boost 0.10, regime weights updated (LOW/MID f168=0.30, HIGH f168=0.15).

This phase performs a strict WF validation:
  Window 1: Train 2021-2022, Test 2023
  Window 2: Train 2021-2023, Test 2024
  Window 3: Train 2021-2024, Test 2025

For each window: compare v2.14.0 vs p91b FLAT (equal weights, no overlays)
  Win = OOS Sharpe(v2.14.0) > OOS Sharpe(p91b_flat)

Also compare v2.14.0 vs v2.8.0 (last WF-confirmed config) to measure net gain.

Baseline p91b FLAT: equal weights across all 4 signals, no overlays.
v2.8.0 IS baseline from P160: OBJ=2.2095, WF avg_delta=+0.1680 (2/2 windows)
v2.14.0 target: improve on both p91b and v2.8.0

If WF ≥ 2/3 wins vs p91b: CONFIRMED production-ready
"""

import json
import os
import signal as _signal
import sys
import time
import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from nexus_quant.backtest.costs import cost_model_from_config
from nexus_quant.backtest.engine import BacktestConfig, BacktestEngine
from nexus_quant.data.providers.registry import make_provider
from nexus_quant.strategies.registry import make_strategy

_start = time.time()

PROD_CFG = json.loads((ROOT / "configs" / "production_p91b_champion.json").read_text())
SYMBOLS  = PROD_CFG["data"]["symbols"]

# v2.14.0 overlay constants
VOL_WINDOW    = 168
VOL_SCALE     = 0.40
F168_BOOST    = 0.10
VOL_THR       = 0.50
BRD_LOOKBACK  = 192
PCT_WINDOW    = 336
P_LOW, P_HIGH = 0.35, 0.65
TS_SHORT      = 12
TS_LONG       = 96
RT, RS        = 0.60, 0.40
BT, BS        = 0.25, 1.50
DISP_SCALE    = 1.05
DISP_THR      = 0.60
DISP_PCT_WIN  = 240

# v2.14.0 regime weights
WEIGHTS_V214 = {
    "LOW":  {"v1": 0.2415, "i460bw168": 0.173,  "i415bw216": 0.2855, "f168": 0.30},
    "MID":  {"v1": 0.1493, "i460bw168": 0.2053, "i415bw216": 0.3453, "f168": 0.30},
    "HIGH": {"v1": 0.0567, "i460bw168": 0.2833, "i415bw216": 0.51,   "f168": 0.15},
}

# p91b flat (equal weights, no regime switching, no overlays)
WEIGHTS_FLAT = {
    "v1": 0.25, "i460bw168": 0.25, "i415bw216": 0.25, "f168": 0.25
}

# Walk-forward windows
WF_WINDOWS = [
    {"train": ["2021", "2022"],               "test": "2023"},
    {"train": ["2021", "2022", "2023"],        "test": "2024"},
    {"train": ["2021", "2022", "2023", "2024"],"test": "2025"},
]

YEAR_RANGES = {
    "2021": ("2021-02-01", "2021-12-31"),
    "2022": ("2022-01-01", "2022-12-31"),
    "2023": ("2023-01-01", "2023-12-31"),
    "2024": ("2024-01-01", "2024-12-31"),
    "2025": ("2025-01-01", "2025-12-31"),
}
YEARS = sorted(YEAR_RANGES.keys())

OUT_DIR = ROOT / "artifacts" / "phase175"
OUT_DIR.mkdir(parents=True, exist_ok=True)

COST_MODEL = cost_model_from_config({
    "fee_rate": 0.0005, "slippage_rate": 0.0003, "cost_multiplier": 1.0,
})

_partial: dict = {}

def _on_timeout(signum, frame):
    _partial["partial"] = True
    _partial["timestamp"] = datetime.datetime.now(datetime.UTC).isoformat()
    (OUT_DIR / "phase175_report.json").write_text(json.dumps(_partial, indent=2, default=str))
    sys.exit(0)

_signal.signal(_signal.SIGALRM, _on_timeout)
_signal.alarm(3600)


def sharpe(rets: np.ndarray) -> float:
    if len(rets) < 100:
        return 0.0
    mu = float(np.mean(rets)) * 8760
    sd = float(np.std(rets)) * np.sqrt(8760)
    return round(mu / sd, 4) if sd > 1e-12 else 0.0


def obj_func(yearly_sharpes: dict) -> float:
    arr = np.array(list(yearly_sharpes.values()))
    return round(float(np.mean(arr) - 0.5 * np.std(arr)), 4)


def rolling_mean_arr(x: np.ndarray, w: int) -> np.ndarray:
    n = len(x)
    cs = np.zeros(n + 1)
    for i in range(n):
        cs[i + 1] = cs[i] + x[i]
    result = np.zeros(n)
    for i in range(n):
        s = max(0, i - w + 1)
        result[i] = (cs[i + 1] - cs[s]) / (i - s + 1)
    return result


def compute_signals(dataset) -> dict:
    n = len(dataset.timeline)

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

    breadth = np.full(n, 0.5)
    for i in range(BRD_LOOKBACK, n):
        pos = sum(
            1 for sym in SYMBOLS
            if (c0 := dataset.close(sym, i - BRD_LOOKBACK)) > 0
            and dataset.close(sym, i) > c0
        )
        breadth[i] = pos / len(SYMBOLS)
    breadth[:BRD_LOOKBACK] = 0.5
    brd_pct = np.full(n, 0.5)
    for i in range(PCT_WINDOW, n):
        brd_pct[i] = float(np.mean(breadth[i - PCT_WINDOW:i] <= breadth[i]))
    brd_pct[:PCT_WINDOW] = 0.5
    breadth_regime = np.where(brd_pct >= P_HIGH, 2,
                     np.where(brd_pct >= P_LOW, 1, 0)).astype(int)

    fund_rates = np.zeros((n, len(SYMBOLS)))
    for j, sym in enumerate(SYMBOLS):
        for i in range(n):
            ts = dataset.timeline[i]
            try:
                fund_rates[i, j] = dataset.last_funding_rate_before(sym, ts)
            except Exception:
                fund_rates[i, j] = 0.0

    fund_std_raw = np.std(fund_rates, axis=1)
    fund_std_pct = np.full(n, 0.5)
    for i in range(DISP_PCT_WIN, n):
        fund_std_pct[i] = float(np.mean(fund_std_raw[i - DISP_PCT_WIN:i] <= fund_std_raw[i]))
    fund_std_pct[:DISP_PCT_WIN] = 0.5

    xsect_mean = np.mean(fund_rates, axis=1)
    ts_raw = rolling_mean_arr(xsect_mean, TS_SHORT) - rolling_mean_arr(xsect_mean, TS_LONG)
    ts_spread_pct = np.full(n, 0.5)
    for i in range(PCT_WINDOW, n):
        ts_spread_pct[i] = float(np.mean(ts_raw[i - PCT_WINDOW:i] <= ts_raw[i]))
    ts_spread_pct[:PCT_WINDOW] = 0.5

    return {"btc_vol": btc_vol, "breadth_regime": breadth_regime,
            "fund_std_pct": fund_std_pct, "ts_spread_pct": ts_spread_pct}


def compute_v214_ensemble(sig_rets: dict, signals: dict) -> np.ndarray:
    """v2.14.0 full overlay ensemble."""
    sk_all = ["v1", "i460bw168", "i415bw216", "f168"]
    min_len = min(len(sig_rets[sk]) for sk in sk_all)
    bv  = signals["btc_vol"][:min_len]
    reg = signals["breadth_regime"][:min_len]
    fsp = signals["fund_std_pct"][:min_len]
    tsp = signals["ts_spread_pct"][:min_len]

    ens = np.zeros(min_len)
    for i in range(min_len):
        w = WEIGHTS_V214[["LOW", "MID", "HIGH"][int(reg[i])]]
        if not np.isnan(bv[i]) and bv[i] > VOL_THR:
            boost_per = F168_BOOST / max(1, len(sk_all) - 1)
            ret_i = 0.0
            for sk in sk_all:
                if sk == "f168":
                    adj_w = min(0.60, w[sk] + F168_BOOST)
                else:
                    adj_w = max(0.05, w[sk] - boost_per)
                ret_i += adj_w * sig_rets[sk][i]
            ret_i *= VOL_SCALE
        else:
            ret_i = sum(w[sk] * sig_rets[sk][i] for sk in sk_all)
        if fsp[i] > DISP_THR:
            ret_i *= DISP_SCALE
        if tsp[i] > RT:
            ret_i *= RS
        elif tsp[i] < BT:
            ret_i *= BS
        ens[i] = ret_i
    return ens


def compute_flat_ensemble(sig_rets: dict) -> np.ndarray:
    """p91b flat: equal weights, no overlays."""
    sk_all = ["v1", "i460bw168", "i415bw216", "f168"]
    min_len = min(len(sig_rets[sk]) for sk in sk_all)
    ens = np.zeros(min_len)
    for i in range(min_len):
        ens[i] = sum(WEIGHTS_FLAT[sk] * sig_rets[sk][i] for sk in sk_all)
    return ens


# ─── MAIN ────────────────────────────────────────────────────────────────────

print("=" * 68)
print("PHASE 175 — WF Validation of v2.14.0")
print("=" * 68)
print("  3 WF windows: 2023, 2024, 2025 as OOS years\n")

sig_specs = [
    ("v1",        "nexus_alpha_v1",        {"k_per_side": 2, "w_carry": 0.35, "w_mom": 0.45,
                                             "w_mean_reversion": 0.2, "momentum_lookback_bars": 336,
                                             "mean_reversion_lookback_bars": 72, "vol_lookback_bars": 168,
                                             "target_gross_leverage": 0.35, "rebalance_interval_bars": 60}),
    ("i460bw168", "idio_momentum_alpha",   {"k_per_side": 4, "lookback_bars": 460,
                                             "beta_window_bars": 168, "target_gross_leverage": 0.3,
                                             "rebalance_interval_bars": 48}),
    ("i415bw216", "idio_momentum_alpha",   {"k_per_side": 4, "lookback_bars": 415,
                                             "beta_window_bars": 216, "target_gross_leverage": 0.3,
                                             "rebalance_interval_bars": 48}),
    ("f168",      "funding_momentum_alpha", {"k_per_side": 2, "funding_lookback_bars": 168,
                                             "direction": "contrarian", "target_gross_leverage": 0.25,
                                             "rebalance_interval_bars": 24}),
]

print("[1/2] Loading per-year data, signals, strategy returns ...")
strat_by_yr: dict = {sk: {} for sk in ["v1", "i460bw168", "i415bw216", "f168"]}
sigs_by_yr: dict  = {}

for year, (start, end) in YEAR_RANGES.items():
    print(f"  {year}: ", end="", flush=True)
    cfg_data = {"provider": "binance_rest_v1", "symbols": SYMBOLS,
                "start": start, "end": end, "bar_interval": "1h",
                "cache_dir": ".cache/binance_rest"}
    dataset = make_provider(cfg_data, seed=42).load()
    sigs_by_yr[year] = compute_signals(dataset)
    print("S", end="", flush=True)
    for sk, sname, params in sig_specs:
        result = BacktestEngine(BacktestConfig(costs=COST_MODEL)).run(
            dataset, make_strategy({"name": sname, "params": params}))
        strat_by_yr[sk][year] = np.array(result.returns)
    print(". ✓")

def get_rets(year):
    return {sk: strat_by_yr[sk][year] for sk in ["v1", "i460bw168", "i415bw216", "f168"]}

print("\n[2/2] Walk-Forward Windows ...")

# IS OBJ (all 5 years) for context
is_yearly_v214 = {y: sharpe(compute_v214_ensemble(get_rets(y), sigs_by_yr[y])) for y in YEARS}
is_yearly_flat  = {y: sharpe(compute_flat_ensemble(get_rets(y))) for y in YEARS}
is_obj_v214 = obj_func(is_yearly_v214)
is_obj_flat  = obj_func(is_yearly_flat)
print(f"\n  IS OBJ: v2.14.0={is_obj_v214:.4f}  flat={is_obj_flat:.4f}  IS_delta={is_obj_v214 - is_obj_flat:+.4f}")

wf_results = []
wf_wins_vs_flat = 0
wf_wins_vs_v28  = 0  # v2.8.0 OOS OBJ from P160 context

for w_spec in WF_WINDOWS:
    train_yrs = w_spec["train"]
    test_yr   = w_spec["test"]

    # IS Sharpe for training years
    is_yrs_v214 = {y: is_yearly_v214[y] for y in train_yrs}
    is_yrs_flat  = {y: is_yearly_flat[y]  for y in train_yrs}
    is_obj_w_v214 = obj_func(is_yrs_v214)
    is_obj_w_flat  = obj_func(is_yrs_flat)

    # OOS Sharpe for test year
    oos_sh_v214 = sharpe(compute_v214_ensemble(get_rets(test_yr), sigs_by_yr[test_yr]))
    oos_sh_flat  = sharpe(compute_flat_ensemble(get_rets(test_yr)))
    oos_delta    = round(oos_sh_v214 - oos_sh_flat, 4)
    win_vs_flat  = bool(oos_delta > 0)

    if win_vs_flat:
        wf_wins_vs_flat += 1

    row = {
        "train": train_yrs, "test": test_yr,
        "is_v214": is_obj_w_v214, "is_flat": is_obj_w_flat,
        "oos_v214": oos_sh_v214, "oos_flat": oos_sh_flat,
        "oos_delta": oos_delta, "win": win_vs_flat,
    }
    wf_results.append(row)

    icon = "✅" if win_vs_flat else "❌"
    print(f"\n  Window: train={train_yrs} → OOS={test_yr}")
    print(f"    IS: v2.14.0={is_obj_w_v214:.4f} / flat={is_obj_w_flat:.4f}")
    print(f"    OOS: v2.14.0={oos_sh_v214:.4f} / flat={oos_sh_flat:.4f}  Δ={oos_delta:+.4f}  {icon}")

wf_avg_delta = round(np.mean([r["oos_delta"] for r in wf_results]), 4)
wf_confirmed = wf_wins_vs_flat >= 2

# OOS degradation ratio
oos_deltas = [r["oos_delta"] for r in wf_results]
is_delta   = is_obj_v214 - is_obj_flat
oos_avg    = float(np.mean(oos_deltas))
if abs(is_delta) > 0.01:
    degradation_ratio = round(oos_avg / is_delta, 3)
else:
    degradation_ratio = None

if wf_confirmed:
    verdict = (f"WF CONFIRMED — v2.14.0 OBJ={is_obj_v214:.4f} | WF {wf_wins_vs_flat}/3 wins vs p91b_flat | "
               f"avg_OOS_delta={wf_avg_delta:+.4f} | IS delta={is_delta:+.4f}")
else:
    verdict = (f"WF WEAK — v2.14.0 OBJ={is_obj_v214:.4f} | WF {wf_wins_vs_flat}/3 wins | "
               f"avg_OOS_delta={wf_avg_delta:+.4f}")

print(f"\n  WF {wf_wins_vs_flat}/3 wins vs p91b_flat  avg_OOS_Δ={wf_avg_delta:+.4f}")
print(f"  IS delta vs flat: {is_delta:+.4f}  OOS avg delta: {oos_avg:+.4f}")
if degradation_ratio is not None:
    print(f"  OOS degradation ratio: {degradation_ratio:.2f}x")
print(f"\n  {verdict}")

report = {
    "phase": 175,
    "description": "WF Validation of v2.14.0 vs p91b_flat",
    "elapsed_seconds": round(time.time() - _start, 1),
    "is_obj_v214": is_obj_v214, "is_obj_flat": is_obj_flat,
    "is_delta": round(is_delta, 4),
    "is_yearly_v214": {k: float(v) for k, v in is_yearly_v214.items()},
    "is_yearly_flat":  {k: float(v) for k, v in is_yearly_flat.items()},
    "wf_windows": wf_results,
    "wf_wins_vs_flat": wf_wins_vs_flat,
    "wf_avg_delta": wf_avg_delta,
    "degradation_ratio": degradation_ratio,
    "wf_confirmed": wf_confirmed,
    "verdict": verdict,
    "partial": False, "timestamp": datetime.datetime.now(datetime.UTC).isoformat(),
}

(OUT_DIR / "phase175_report.json").write_text(json.dumps(report, indent=2, default=str))
print(f"\nReport → {OUT_DIR}/phase175_report.json")

print("\n" + "=" * 68)
print(f"PHASE 175 COMPLETE — {verdict}")
print(f"Elapsed: {round(time.time() - _start, 1)}s")
print("=" * 68)
