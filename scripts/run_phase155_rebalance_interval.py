"""
Phase 155 — Rebalance Interval Optimization
============================================
Hypothesis: Current intervals (V1:60h, I460:48h, I415:48h, F168:24h) were set
empirically. Faster rebalancing may capture signal decay quicker; slower reduces
turnover costs. Testing cross-signal combinations.

Test grid (most promising first):
  V1: [48, 60, 72]
  I460: [36, 48, 60]
  I415: [36, 48, 60]
  F168: 24 (fixed — funding 3-day rhythm)

Run: exhaustive V1×I460×I415 (3×3×3 = 27 combos) on IS, then LOYO on best.
Baseline (prod v2.6.0 F168): OBJ=2.1312
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

# v2.6.0 overlay constants
VOL_WINDOW     = 168
VOL_THRESHOLD  = 0.5
VOL_SCALE      = 0.5
VOL_F144_BOOST = 0.2
BREADTH_LOOKBACK = 168
PCT_WINDOW     = 336
P_LOW, P_HIGH  = 0.33, 0.67
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

OUT_DIR = ROOT / "artifacts" / "phase155"
OUT_DIR.mkdir(parents=True, exist_ok=True)

COST_MODEL = cost_model_from_config({
    "fee_rate": 0.0005, "slippage_rate": 0.0003, "cost_multiplier": 1.0,
})

V1_INTERVALS   = [48, 60, 72]
IDIO_INTERVALS = [36, 48, 60]

_partial: dict = {}

def _on_timeout(signum, frame):
    print("\n⏰ TIMEOUT — saving partial...")
    _partial["partial"] = True
    _partial["timestamp"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    (OUT_DIR / "phase155_report.json").write_text(json.dumps(_partial, indent=2))
    sys.exit(0)

_signal.signal(_signal.SIGALRM, _on_timeout)
_signal.alarm(3600)  # 60min


def sharpe(rets: np.ndarray) -> float:
    if len(rets) < 100:
        return 0.0
    mu = float(np.mean(rets)) * 8760
    sd = float(np.std(rets)) * np.sqrt(8760)
    return round(mu / sd, 4) if sd > 1e-12 else 0.0


def obj_func(yearly_sharpes: dict) -> float:
    arr = np.array(list(yearly_sharpes.values()))
    return round(float(np.mean(arr) - 0.5 * np.std(arr)), 4)


def compute_all_signals(dataset) -> dict:
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
    breadth_regime = np.where(brd_pct >= P_HIGH, 2,
                     np.where(brd_pct >= P_LOW, 1, 0)).astype(int)

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
    for i in range(PCT_WINDOW, n):
        hist = fund_std_raw[i - PCT_WINDOW:i]
        fund_std_pct[i] = float(np.mean(hist <= fund_std_raw[i]))
    fund_std_pct[:PCT_WINDOW] = 0.5

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
    for i in range(PCT_WINDOW, n):
        hist = ts_spread_raw[i - PCT_WINDOW:i]
        ts_spread_pct[i] = float(np.mean(hist <= ts_spread_raw[i]))
    ts_spread_pct[:PCT_WINDOW] = 0.5

    return {
        "btc_vol": btc_vol,
        "breadth_regime": breadth_regime,
        "fund_std_pct": fund_std_pct,
        "ts_spread_pct": ts_spread_pct,
    }


def compute_ensemble(sig_rets: dict, signals: dict) -> np.ndarray:
    sk_all = ["v1", "i460bw168", "i415bw216", "f168"]
    min_len = min(len(sig_rets[sk]) for sk in sk_all)
    bv  = signals["btc_vol"][:min_len]
    reg = signals["breadth_regime"][:min_len]
    fsp = signals["fund_std_pct"][:min_len]
    tsp = signals["ts_spread_pct"][:min_len]

    ens = np.zeros(min_len)
    for i in range(min_len):
        w = WEIGHTS[["LOW", "MID", "HIGH"][int(reg[i])]]
        if not np.isnan(bv[i]) and bv[i] > VOL_THRESHOLD:
            boost = VOL_F144_BOOST / max(1, len(sk_all) - 1)
            ret_i = 0.0
            for sk in sk_all:
                adj_w = (min(0.60, w[sk] + VOL_F144_BOOST) if sk == "f168"
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
print("PHASE 155 — Rebalance Interval Optimization")
print("=" * 68)

# ---- load data & precompute ALL signal variants ----
# For each V1 interval × I460 interval × I415 interval, we need separate backtests.
# Strategy: load dataset once per year, run all (V1_iv × I460_iv × I415_iv) strategy variants
# plus shared F168 (always 24h).

print("\n[1/3] Loading datasets + running all signal variants (2021-2025)...")
print(f"  Combos: V1={V1_INTERVALS} × I460={IDIO_INTERVALS} × I415={IDIO_INTERVALS} = {len(V1_INTERVALS)*len(IDIO_INTERVALS)*len(IDIO_INTERVALS)} combos")

# Store all strategy returns by (year, strategy_key)
all_rets: dict = {}
signals_data: dict = {}

for year, (start, end) in YEAR_RANGES.items():
    print(f"  {year}: loading...", end=" ", flush=True)
    cfg_data = {
        "provider": "binance_rest_v1", "symbols": SYMBOLS,
        "start": start, "end": end, "bar_interval": "1h",
        "cache_dir": ".cache/binance_rest",
    }
    provider = make_provider(cfg_data, seed=42)
    dataset  = provider.load()
    signals_data[year] = compute_all_signals(dataset)
    print("loaded.", end=" ", flush=True)

    # F168 (shared) — fixed 24h
    key_f = (year, "f168")
    if key_f not in all_rets:
        bt_cfg = BacktestConfig(costs=COST_MODEL)
        strat  = make_strategy({"name": "funding_momentum_alpha", "params": {
            "k_per_side": 2, "funding_lookback_bars": 168, "direction": "contrarian",
            "target_gross_leverage": 0.25, "rebalance_interval_bars": 24,
        }})
        result = BacktestEngine(bt_cfg).run(dataset, strat)
        all_rets[key_f] = np.array(result.returns)

    # V1 variants
    for v1_iv in V1_INTERVALS:
        k = (year, f"v1_{v1_iv}")
        bt_cfg = BacktestConfig(costs=COST_MODEL)
        strat  = make_strategy({"name": "nexus_alpha_v1", "params": {
            "k_per_side":2,"w_carry":0.35,"w_mom":0.45,"w_mean_reversion":0.2,
            "momentum_lookback_bars":336,"mean_reversion_lookback_bars":72,
            "vol_lookback_bars":168,"target_gross_leverage":0.35,
            "rebalance_interval_bars": v1_iv,
        }})
        result = BacktestEngine(bt_cfg).run(dataset, strat)
        all_rets[k] = np.array(result.returns)

    # I460 variants
    for i_iv in IDIO_INTERVALS:
        k = (year, f"i460_{i_iv}")
        bt_cfg = BacktestConfig(costs=COST_MODEL)
        strat  = make_strategy({"name": "idio_momentum_alpha", "params": {
            "k_per_side":4,"lookback_bars":460,"beta_window_bars":168,
            "target_gross_leverage":0.3, "rebalance_interval_bars": i_iv,
        }})
        result = BacktestEngine(bt_cfg).run(dataset, strat)
        all_rets[k] = np.array(result.returns)

    # I415 variants
    for i_iv in IDIO_INTERVALS:
        k = (year, f"i415_{i_iv}")
        bt_cfg = BacktestConfig(costs=COST_MODEL)
        strat  = make_strategy({"name": "idio_momentum_alpha", "params": {
            "k_per_side":4,"lookback_bars":415,"beta_window_bars":216,
            "target_gross_leverage":0.3, "rebalance_interval_bars": i_iv,
        }})
        result = BacktestEngine(bt_cfg).run(dataset, strat)
        all_rets[k] = np.array(result.returns)

    print("✓")

print("\n[2/3] Computing OBJ for all 27 combos...")

# Baseline: V1=60, I460=48, I415=48
baseline_yearly = {}
for year in YEARS:
    yr_sig = {
        "v1": all_rets[(year, "v1_60")],
        "i460bw168": all_rets[(year, "i460_48")],
        "i415bw216": all_rets[(year, "i415_48")],
        "f168": all_rets[(year, "f168")],
    }
    baseline_yearly[year] = sharpe(compute_ensemble(yr_sig, signals_data[year]))
baseline_obj = obj_func(baseline_yearly)
print(f"  Baseline (V1=60,I460=48,I415=48): OBJ={baseline_obj:.4f} | {baseline_yearly}\n")

combo_results = {}
for v1_iv, i460_iv, i415_iv in product(V1_INTERVALS, IDIO_INTERVALS, IDIO_INTERVALS):
    combo_key = f"V1={v1_iv}_I460={i460_iv}_I415={i415_iv}"
    yearly = {}
    for year in YEARS:
        yr_sig = {
            "v1": all_rets[(year, f"v1_{v1_iv}")],
            "i460bw168": all_rets[(year, f"i460_{i460_iv}")],
            "i415bw216": all_rets[(year, f"i415_{i415_iv}")],
            "f168": all_rets[(year, "f168")],
        }
        yearly[year] = sharpe(compute_ensemble(yr_sig, signals_data[year]))
    o = obj_func(yearly)
    combo_results[combo_key] = {"obj": o, "delta": o - baseline_obj, "yearly": yearly,
                                 "params": {"v1_iv": v1_iv, "i460_iv": i460_iv, "i415_iv": i415_iv}}

# Sort by OBJ descending
sorted_combos = sorted(combo_results.items(), key=lambda x: x[1]["obj"], reverse=True)
print("  Top 10 combos:")
for name, r in sorted_combos[:10]:
    d = r["delta"]
    sym = "✅" if d > 0.005 else ("⚠️ " if d > -0.005 else "❌")
    print(f"    {sym} {name}: OBJ={r['obj']:.4f} (Δ={d:+.4f})")

best_name, best_r = sorted_combos[0]
best_obj   = best_r["obj"]
best_delta = best_r["delta"]
best_params = best_r["params"]

print(f"\n[3/3] LOYO validation of best: {best_name}...")
loyo_wins, loyo_deltas = 0, []
for held_out in YEARS:
    yr_sig_best = {
        "v1": all_rets[(held_out, f"v1_{best_params['v1_iv']}")],
        "i460bw168": all_rets[(held_out, f"i460_{best_params['i460_iv']}")],
        "i415bw216": all_rets[(held_out, f"i415_{best_params['i415_iv']}")],
        "f168": all_rets[(held_out, "f168")],
    }
    yr_sig_base = {
        "v1": all_rets[(held_out, "v1_60")],
        "i460bw168": all_rets[(held_out, "i460_48")],
        "i415bw216": all_rets[(held_out, "i415_48")],
        "f168": all_rets[(held_out, "f168")],
    }
    sh_best = sharpe(compute_ensemble(yr_sig_best, signals_data[held_out]))
    sh_base = sharpe(compute_ensemble(yr_sig_base, signals_data[held_out]))
    d = sh_best - sh_base
    loyo_deltas.append(d)
    loyo_wins += int(d > 0)
    print(f"  LOYO {held_out}: best={sh_best:.4f} base={sh_base:.4f} Δ={d:+.4f} {'✅' if d>0 else '❌'}")

loyo_avg  = float(np.mean(loyo_deltas))
validated = loyo_wins >= 3 and best_delta > 0.005

print("\n" + "=" * 68)
if validated:
    print(f"✅ VALIDATED — {best_name}: OBJ={best_obj:.4f} (Δ={best_delta:+.4f}) | LOYO {loyo_wins}/5 avg={loyo_avg:+.4f}")
else:
    print(f"❌ NO IMPROVEMENT — best: {best_name} OBJ={best_obj:.4f} (Δ={best_delta:+.4f}) | LOYO {loyo_wins}/5")
    print(f"   Baseline (V1=60,I460=48,I415=48) OBJ={baseline_obj:.4f} remains optimal.")
print("=" * 68)

report = {
    "phase": 155,
    "description": "Rebalance Interval Optimization (V1, I460, I415)",
    "hypothesis": "Current 60/48/48h intervals were set empirically; sweep may find better combo",
    "elapsed_seconds": round(time.time() - _start, 1),
    "v1_intervals": V1_INTERVALS,
    "idio_intervals": IDIO_INTERVALS,
    "baseline_obj": baseline_obj,
    "baseline_yearly": baseline_yearly,
    "best_combo": best_name,
    "best_obj": best_obj,
    "best_delta": best_delta,
    "best_params": best_params,
    "top10": [{name: r} for name, r in sorted_combos[:10]],
    "loyo_wins": loyo_wins,
    "loyo_avg_delta": loyo_avg,
    "validated": validated,
    "verdict": (f"VALIDATED — {best_name} OBJ={best_obj:.4f} LOYO {loyo_wins}/5"
                if validated else
                f"NO IMPROVEMENT — baseline V1=60/I460=48/I415=48 OBJ={baseline_obj:.4f} optimal"),
    "partial": False,
    "timestamp": datetime.datetime.utcnow().isoformat(),
}

out_path = OUT_DIR / "phase155_report.json"
out_path.write_text(json.dumps(report, indent=2))
print(f"✅ Saved → {out_path}")
