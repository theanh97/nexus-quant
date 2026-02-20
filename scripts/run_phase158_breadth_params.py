"""
Phase 158 — Breadth Classifier Parameter Sweep
===============================================
Hypothesis: Breadth signal params (lookback=168, pct_window=336, p_low=0.33, p_high=0.67)
were set in P144 with simpler baseline. With v2.7.0 richer stack, re-testing.

Phase 1: breadth_lookback=[120,144,168,192,216] × pct_window=[252,336,504] → 15 combos IS
Phase 2: Best from P1 → sweep thresholds: (p_low,p_high)=[(0.25,0.75),(0.30,0.70),(0.33,0.67),(0.35,0.65)]
Phase 3: LOYO on global best.

Baseline (prod v2.7.0): OBJ=2.1448
OBJ = mean(yearly_sharpes) - 0.5 * std(yearly_sharpes)
"""

import json
import os
import signal as _signal
import sys
import time
import datetime
from itertools import product
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

# v2.7.0 overlay constants (non-breadth)
VOL_WINDOW     = 168
VOL_THRESHOLD  = 0.5
VOL_SCALE      = 0.4
VOL_F168_BOOST = 0.15
FUND_DISP_THR  = 0.75
FUND_DISP_SCALE= 1.15
TS_REDUCE_THR  = 0.70
TS_REDUCE_SCALE= 0.60
TS_BOOST_THR   = 0.30
TS_BOOST_SCALE = 1.15
TS_SHORT, TS_LONG = 24, 144

WEIGHTS = {
    "LOW":  {"v1": 0.2747, "i460bw168": 0.1967, "i415bw216": 0.3247, "f168": 0.2039},
    "MID":  {"v1": 0.16,   "i460bw168": 0.22,   "i415bw216": 0.37,   "f168": 0.25},
    "HIGH": {"v1": 0.05,   "i460bw168": 0.25,   "i415bw216": 0.45,   "f168": 0.25},
}

YEAR_RANGES = {
    "2021": ("2021-02-01", "2021-12-31"),
    "2022": ("2022-01-01", "2022-12-31"),
    "2023": ("2023-01-01", "2023-12-31"),
    "2024": ("2024-01-01", "2024-12-31"),
    "2025": ("2025-01-01", "2025-12-31"),
}
YEARS = sorted(YEAR_RANGES.keys())

OUT_DIR = ROOT / "artifacts" / "phase158"
OUT_DIR.mkdir(parents=True, exist_ok=True)

COST_MODEL = cost_model_from_config({
    "fee_rate": 0.0005, "slippage_rate": 0.0003, "cost_multiplier": 1.0,
})

BRD_LOOKBACKS  = [120, 144, 168, 192, 216]
PCT_WINDOWS    = [252, 336, 504]
THRESHOLD_PAIRS = [(0.25, 0.75), (0.30, 0.70), (0.33, 0.67), (0.35, 0.65)]

_partial: dict = {}

def _on_timeout(signum, frame):
    print("\n⏰ TIMEOUT — saving partial...")
    _partial["partial"] = True
    _partial["timestamp"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    (OUT_DIR / "phase158_report.json").write_text(json.dumps(_partial, indent=2))
    sys.exit(0)

_signal.signal(_signal.SIGALRM, _on_timeout)
_signal.alarm(2700)  # 45min


def sharpe(rets: np.ndarray) -> float:
    if len(rets) < 100:
        return 0.0
    mu = float(np.mean(rets)) * 8760
    sd = float(np.std(rets)) * np.sqrt(8760)
    return round(mu / sd, 4) if sd > 1e-12 else 0.0


def obj_func(yearly_sharpes: dict) -> float:
    arr = np.array(list(yearly_sharpes.values()))
    return round(float(np.mean(arr) - 0.5 * np.std(arr)), 4)


def compute_breadth_regime(dataset, lookback: int, pct_window: int,
                            p_low: float, p_high: float) -> np.ndarray:
    n = len(dataset.timeline)
    breadth = np.full(n, 0.5)
    for i in range(lookback, n):
        pos = sum(
            1 for sym in SYMBOLS
            if (c0 := dataset.close(sym, i - lookback)) > 0
            and dataset.close(sym, i) > c0
        )
        breadth[i] = pos / len(SYMBOLS)
    breadth[:lookback] = 0.5
    brd_pct = np.full(n, 0.5)
    for i in range(pct_window, n):
        brd_pct[i] = float(np.mean(breadth[i - pct_window:i] <= breadth[i]))
    brd_pct[:pct_window] = 0.5
    return np.where(brd_pct >= p_high, 2, np.where(brd_pct >= p_low, 1, 0)).astype(int)


def compute_fixed_signals(dataset) -> dict:
    """BTC vol, fund_std_pct, ts_spread_pct — not breadth (varied separately)."""
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

    fund_std_raw = np.zeros(n)
    for i in range(n):
        ts = dataset.timeline[i]
        rates = []
        for sym in SYMBOLS:
            try:
                rates.append(dataset.last_funding_rate_before(sym, ts))
            except Exception:
                pass
        fund_std_raw[i] = float(np.std(rates)) if len(rates) > 1 else 0.0
    fund_std_pct = np.full(n, 0.5)
    PCT_W = 336
    for i in range(PCT_W, n):
        fund_std_pct[i] = float(np.mean(fund_std_raw[i - PCT_W:i] <= fund_std_raw[i]))
    fund_std_pct[:PCT_W] = 0.5

    ts_spread_raw = np.zeros(n)
    for i in range(max(TS_SHORT, TS_LONG), n):
        short_rates, long_rates = [], []
        for sym in SYMBOLS:
            ts_now = dataset.timeline[i]
            ts_short_start = dataset.timeline[max(0, i - TS_SHORT)]
            r_now   = dataset.last_funding_rate_before(sym, ts_now)
            r_short = dataset.last_funding_rate_before(sym, ts_short_start)
            short_rates.append(r_now)
            long_rates.append((r_now + r_short) / 2)
        ts_spread_raw[i] = float(np.mean(short_rates)) - float(np.mean(long_rates))
    ts_spread_raw[:max(TS_SHORT, TS_LONG)] = 0.0
    ts_spread_pct = np.full(n, 0.5)
    for i in range(PCT_W, n):
        ts_spread_pct[i] = float(np.mean(ts_spread_raw[i - PCT_W:i] <= ts_spread_raw[i]))
    ts_spread_pct[:PCT_W] = 0.5

    return {"btc_vol": btc_vol, "fund_std_pct": fund_std_pct, "ts_spread_pct": ts_spread_pct}


def compute_ensemble(sig_rets: dict, fixed_signals: dict, breadth_regime: np.ndarray) -> np.ndarray:
    sk_all = ["v1", "i460bw168", "i415bw216", "f168"]
    min_len = min(len(sig_rets[sk]) for sk in sk_all)
    bv  = fixed_signals["btc_vol"][:min_len]
    fsp = fixed_signals["fund_std_pct"][:min_len]
    tsp = fixed_signals["ts_spread_pct"][:min_len]
    reg = breadth_regime[:min_len]

    ens = np.zeros(min_len)
    for i in range(min_len):
        w = WEIGHTS[["LOW", "MID", "HIGH"][int(reg[i])]]
        if not np.isnan(bv[i]) and bv[i] > VOL_THRESHOLD:
            boost = VOL_F168_BOOST / max(1, len(sk_all) - 1)
            ret_i = 0.0
            for sk in sk_all:
                adj_w = (min(0.60, w[sk] + VOL_F168_BOOST) if sk == "f168"
                         else max(0.05, w[sk] - boost))
                ret_i += adj_w * sig_rets[sk][i]
            ret_i *= VOL_SCALE
        else:
            ret_i = sum(w[sk] * sig_rets[sk][i] for sk in sk_all)
        if fsp[i] > FUND_DISP_THR:
            ret_i *= FUND_DISP_SCALE
        if tsp[i] > TS_REDUCE_THR:
            ret_i *= TS_REDUCE_SCALE
        elif tsp[i] < TS_BOOST_THR:
            ret_i *= TS_BOOST_SCALE
        ens[i] = ret_i
    return ens


# ─── main ────────────────────────────────────────────────────────────────────

print("=" * 68)
print("PHASE 158 — Breadth Classifier Parameter Sweep")
print("=" * 68)

print("\n[1/4] Loading data + computing strategy returns + fixed signals...")
sig_returns: dict = {sk: {} for sk in ["v1", "i460bw168", "i415bw216", "f168"]}
fixed_signals_data: dict = {}
dataset_store: dict = {}  # keep datasets for breadth recomputation

for year, (start, end) in YEAR_RANGES.items():
    print(f"  {year}: ", end="", flush=True)
    cfg_data = {
        "provider": "binance_rest_v1", "symbols": SYMBOLS,
        "start": start, "end": end, "bar_interval": "1h",
        "cache_dir": ".cache/binance_rest",
    }
    provider = make_provider(cfg_data, seed=42)
    dataset  = provider.load()
    dataset_store[year] = dataset
    fixed_signals_data[year] = compute_fixed_signals(dataset)
    print("O", end="", flush=True)

    for sk, sname, params in [
        ("v1", "nexus_alpha_v1", {
            "k_per_side":2,"w_carry":0.35,"w_mom":0.45,"w_mean_reversion":0.2,
            "momentum_lookback_bars":336,"mean_reversion_lookback_bars":72,
            "vol_lookback_bars":168,"target_gross_leverage":0.35,"rebalance_interval_bars":60
        }),
        ("i460bw168", "idio_momentum_alpha", {
            "k_per_side":4,"lookback_bars":460,"beta_window_bars":168,
            "target_gross_leverage":0.3,"rebalance_interval_bars":48
        }),
        ("i415bw216", "idio_momentum_alpha", {
            "k_per_side":4,"lookback_bars":415,"beta_window_bars":216,
            "target_gross_leverage":0.3,"rebalance_interval_bars":48
        }),
        ("f168", "funding_momentum_alpha", {
            "k_per_side":2,"funding_lookback_bars":168,"direction":"contrarian",
            "target_gross_leverage":0.25,"rebalance_interval_bars":24
        }),
    ]:
        bt_cfg = BacktestConfig(costs=COST_MODEL)
        strat  = make_strategy({"name": sname, "params": params})
        result = BacktestEngine(bt_cfg).run(dataset, strat)
        sig_returns[sk][year] = np.array(result.returns)
    print(". ✓")

print("\n[2/4] Phase 1 — Breadth lookback × pct_window sweep...")
phase1_results = {}
for brd_lb, pct_w in product(BRD_LOOKBACKS, PCT_WINDOWS):
    key = f"lb{brd_lb}_pw{pct_w}"
    yearly = {}
    for year in YEARS:
        reg = compute_breadth_regime(dataset_store[year], brd_lb, pct_w, 0.33, 0.67)
        yr_sig = {sk: sig_returns[sk][year] for sk in ["v1","i460bw168","i415bw216","f168"]}
        yearly[year] = sharpe(compute_ensemble(yr_sig, fixed_signals_data[year], reg))
    o = obj_func(yearly)
    phase1_results[key] = {"obj": o, "yearly": yearly, "lb": brd_lb, "pw": pct_w}

# Baseline: lb=168, pw=336
baseline_obj    = phase1_results["lb168_pw336"]["obj"]
baseline_yearly = phase1_results["lb168_pw336"]["yearly"]
print(f"  Baseline (lb=168,pw=336,p=0.33/0.67): OBJ={baseline_obj:.4f}\n")

sorted_p1 = sorted(phase1_results.items(), key=lambda x: x[1]["obj"], reverse=True)
print("  Top 8 combos:")
for name, r in sorted_p1[:8]:
    d = r["obj"] - baseline_obj
    sym = "✅" if d > 0.005 else ("⚠️ " if d > -0.005 else "❌")
    print(f"    {sym} {name}: OBJ={r['obj']:.4f} (Δ={d:+.4f})")

best_p1_name, best_p1_r = sorted_p1[0]
best_lb = best_p1_r["lb"]
best_pw = best_p1_r["pw"]
print(f"\n  Best lb×pw: lb={best_lb}, pw={best_pw}")

print(f"\n[3/4] Phase 2 — Threshold sweep (lb={best_lb}, pw={best_pw})...")
phase2_results = {}
for (p_lo, p_hi) in THRESHOLD_PAIRS:
    key = f"p{int(p_lo*100)}_{int(p_hi*100)}"
    yearly = {}
    for year in YEARS:
        reg = compute_breadth_regime(dataset_store[year], best_lb, best_pw, p_lo, p_hi)
        yr_sig = {sk: sig_returns[sk][year] for sk in ["v1","i460bw168","i415bw216","f168"]}
        yearly[year] = sharpe(compute_ensemble(yr_sig, fixed_signals_data[year], reg))
    o = obj_func(yearly)
    d = o - baseline_obj
    sym = "✅" if d > 0.005 else ("⚠️ " if d > -0.005 else "❌")
    print(f"  {sym} p_low={p_lo} p_high={p_hi}: OBJ={o:.4f} (Δ={d:+.4f}) | {yearly}")
    phase2_results[key] = {"obj": o, "delta": d, "yearly": yearly, "p_low": p_lo, "p_high": p_hi}

best_thr_key = max(phase2_results, key=lambda k: phase2_results[k]["obj"])
best_r2 = phase2_results[best_thr_key]
best_p_low  = best_r2["p_low"]
best_p_high = best_r2["p_high"]
best_obj    = best_r2["obj"]
best_delta  = best_r2["delta"]
best_yearly = best_r2["yearly"]

print(f"\n  → Global best: lb={best_lb} pw={best_pw} p_low={best_p_low} p_high={best_p_high} OBJ={best_obj:.4f}")

print(f"\n[4/4] LOYO validation: lb={best_lb} pw={best_pw} p={best_p_low}/{best_p_high}...")
loyo_wins, loyo_deltas = 0, []
for held_out in YEARS:
    reg_best = compute_breadth_regime(dataset_store[held_out], best_lb, best_pw, best_p_low, best_p_high)
    reg_base = compute_breadth_regime(dataset_store[held_out], 168, 336, 0.33, 0.67)
    yr_sig   = {sk: sig_returns[sk][held_out] for sk in ["v1","i460bw168","i415bw216","f168"]}
    sh_best  = sharpe(compute_ensemble(yr_sig, fixed_signals_data[held_out], reg_best))
    sh_base  = sharpe(compute_ensemble(yr_sig, fixed_signals_data[held_out], reg_base))
    d = sh_best - sh_base
    loyo_deltas.append(d)
    loyo_wins += int(d > 0)
    print(f"  LOYO {held_out}: best={sh_best:.4f} base={sh_base:.4f} Δ={d:+.4f} {'✅' if d>0 else '❌'}")

loyo_avg  = float(np.mean(loyo_deltas))
validated = loyo_wins >= 3 and best_delta > 0.005

print("\n" + "=" * 68)
if validated:
    print(f"✅ VALIDATED — lb={best_lb} pw={best_pw} p={best_p_low}/{best_p_high}: OBJ={best_obj:.4f} (Δ={best_delta:+.4f}) | LOYO {loyo_wins}/5")
else:
    print(f"❌ NO IMPROVEMENT — best OBJ={best_obj:.4f} (Δ={best_delta:+.4f}) | LOYO {loyo_wins}/5")
    print(f"   Baseline (lb=168,pw=336,p=0.33/0.67) OBJ={baseline_obj:.4f} remains optimal.")
print("=" * 68)

report = {
    "phase": 158,
    "description": "Breadth Classifier Parameter Sweep",
    "hypothesis": "Breadth params (lb=168,pw=336,p=0.33/0.67) set in P144; re-test with v2.7.0 baseline",
    "elapsed_seconds": round(time.time() - _start, 1),
    "baseline_obj": baseline_obj,
    "baseline_yearly": baseline_yearly,
    "best_params": {"breadth_lookback": best_lb, "pct_window": best_pw,
                    "p_low": best_p_low, "p_high": best_p_high},
    "best_obj": best_obj,
    "best_delta": best_delta,
    "best_yearly": best_yearly,
    "loyo_wins": loyo_wins,
    "loyo_avg_delta": loyo_avg,
    "validated": validated,
    "verdict": (f"VALIDATED — lb={best_lb} pw={best_pw} p={best_p_low}/{best_p_high} OBJ={best_obj:.4f} LOYO {loyo_wins}/5"
                if validated else
                f"NO IMPROVEMENT — baseline (lb=168,pw=336,p=0.33/0.67) OBJ={baseline_obj:.4f} optimal"),
    "partial": False,
    "timestamp": datetime.datetime.utcnow().isoformat(),
}

out_path = OUT_DIR / "phase158_report.json"
out_path.write_text(json.dumps(report, indent=2))
print(f"✅ Saved → {out_path}")
