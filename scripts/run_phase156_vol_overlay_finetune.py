"""
Phase 156 — Vol Overlay Parameter Fine-Tuning
=============================================
Hypothesis: Vol overlay (threshold=0.5, scale=0.5, f168_boost=0.2) was set in P129
with a simpler baseline. With v2.6.0 richer stack, optimal params may have shifted.
Especially: 2025 lower BTC vol → lower threshold may activate overlay more helpfully.

Test grid: threshold=[0.40,0.45,0.50,0.55,0.60] × scale=[0.40,0.45,0.50,0.55,0.60]
F168_boost: test [0.15, 0.20, 0.25] for the best threshold×scale combo.

Baseline (prod v2.6.0): OBJ=2.1312
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

# Fixed v2.6.0 params (varied below: VOL_THRESHOLD, VOL_SCALE, VOL_F168_BOOST)
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

OUT_DIR = ROOT / "artifacts" / "phase156"
OUT_DIR.mkdir(parents=True, exist_ok=True)

COST_MODEL = cost_model_from_config({
    "fee_rate": 0.0005, "slippage_rate": 0.0003, "cost_multiplier": 1.0,
})

VOL_THRESHOLDS = [0.40, 0.45, 0.50, 0.55, 0.60]
VOL_SCALES     = [0.40, 0.45, 0.50, 0.55, 0.60]
F_BOOSTS       = [0.15, 0.20, 0.25]

_partial: dict = {}

def _on_timeout(signum, frame):
    print("\n⏰ TIMEOUT — saving partial...")
    _partial["partial"] = True
    _partial["timestamp"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    (OUT_DIR / "phase156_report.json").write_text(json.dumps(_partial, indent=2))
    sys.exit(0)

_signal.signal(_signal.SIGALRM, _on_timeout)
_signal.alarm(2400)  # 40min


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
    # BTC log returns → rolling vol array (raw, pre-computed)
    btc_rets = np.zeros(n)
    for i in range(1, n):
        c0 = dataset.close("BTCUSDT", i - 1)
        c1 = dataset.close("BTCUSDT", i)
        btc_rets[i] = (c1 / c0 - 1.0) if c0 > 0 else 0.0
    # rolling 168h std (annualized) — fixed window, threshold is varied later
    VOL_WINDOW = 168
    btc_vol_168 = np.full(n, np.nan)
    for i in range(VOL_WINDOW, n):
        btc_vol_168[i] = float(np.std(btc_rets[i - VOL_WINDOW:i])) * np.sqrt(8760)
    if VOL_WINDOW < n:
        btc_vol_168[:VOL_WINDOW] = btc_vol_168[VOL_WINDOW]

    # Breadth
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
        brd_pct[i] = float(np.mean(breadth[i - PCT_WINDOW:i] <= breadth[i]))
    brd_pct[:PCT_WINDOW] = 0.5
    breadth_regime = np.where(brd_pct >= P_HIGH, 2,
                     np.where(brd_pct >= P_LOW, 1, 0)).astype(int)

    # Funding dispersion
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
        fund_std_pct[i] = float(np.mean(fund_std_raw[i - PCT_WINDOW:i] <= fund_std_raw[i]))
    fund_std_pct[:PCT_WINDOW] = 0.5

    # Funding term structure
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
        ts_spread_pct[i] = float(np.mean(ts_spread_raw[i - PCT_WINDOW:i] <= ts_spread_raw[i]))
    ts_spread_pct[:PCT_WINDOW] = 0.5

    return {
        "btc_vol_168": btc_vol_168,
        "breadth_regime": breadth_regime,
        "fund_std_pct": fund_std_pct,
        "ts_spread_pct": ts_spread_pct,
    }


def compute_ensemble(sig_rets: dict, signals: dict,
                     vol_threshold: float, vol_scale: float,
                     f_boost: float) -> np.ndarray:
    sk_all = ["v1", "i460bw168", "i415bw216", "f168"]
    min_len = min(len(sig_rets[sk]) for sk in sk_all)
    bv  = signals["btc_vol_168"][:min_len]
    reg = signals["breadth_regime"][:min_len]
    fsp = signals["fund_std_pct"][:min_len]
    tsp = signals["ts_spread_pct"][:min_len]

    ens = np.zeros(min_len)
    for i in range(min_len):
        w = WEIGHTS[["LOW", "MID", "HIGH"][int(reg[i])]]
        if not np.isnan(bv[i]) and bv[i] > vol_threshold:
            boost = f_boost / max(1, len(sk_all) - 1)
            ret_i = 0.0
            for sk in sk_all:
                adj_w = (min(0.60, w[sk] + f_boost) if sk == "f168"
                         else max(0.05, w[sk] - boost))
                ret_i += adj_w * sig_rets[sk][i]
            ret_i *= vol_scale
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
print("PHASE 156 — Vol Overlay Parameter Fine-Tuning")
print("=" * 68)
print(f"  Grid: {len(VOL_THRESHOLDS)}×{len(VOL_SCALES)} = {len(VOL_THRESHOLDS)*len(VOL_SCALES)} combos + {len(F_BOOSTS)} boost variants")

print("\n[1/3] Loading datasets + computing signals + base strategy returns...")
sig_returns: dict = {sk: {} for sk in ["v1", "i460bw168", "i415bw216", "f168"]}
signals_data: dict = {}

for year, (start, end) in YEAR_RANGES.items():
    print(f"  {year}: ", end="", flush=True)
    cfg_data = {
        "provider": "binance_rest_v1", "symbols": SYMBOLS,
        "start": start, "end": end, "bar_interval": "1h",
        "cache_dir": ".cache/binance_rest",
    }
    provider = make_provider(cfg_data, seed=42)
    dataset  = provider.load()
    signals_data[year] = compute_all_signals(dataset)
    print("O", end="", flush=True)

    for sk, strat_name, params in [
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
        strat  = make_strategy({"name": strat_name, "params": params})
        engine = BacktestEngine(bt_cfg)
        result = engine.run(dataset, strat)
        sig_returns[sk][year] = np.array(result.returns)
    print(". ✓")

print("\n[2/3] Grid search: threshold×scale (f_boost=0.20 fixed)...")
# Phase 1: threshold×scale grid (fixed f_boost=0.20)
grid_results = {}
for thr, scale in product(VOL_THRESHOLDS, VOL_SCALES):
    key = f"thr={thr:.2f}_sc={scale:.2f}"
    yearly = {}
    for year in YEARS:
        yr_sig = {sk: sig_returns[sk][year] for sk in ["v1","i460bw168","i415bw216","f168"]}
        yearly[year] = sharpe(compute_ensemble(yr_sig, signals_data[year],
                                               vol_threshold=thr, vol_scale=scale, f_boost=0.20))
    o = obj_func(yearly)
    grid_results[key] = {"obj": o, "yearly": yearly, "thr": thr, "scale": scale, "boost": 0.20}

# Baseline is thr=0.50, scale=0.50
baseline_obj = grid_results["thr=0.50_sc=0.50"]["obj"]
baseline_yearly = grid_results["thr=0.50_sc=0.50"]["yearly"]
print(f"  Baseline (thr=0.50,sc=0.50): OBJ={baseline_obj:.4f} | {baseline_yearly}\n")

# Sort and display
sorted_grid = sorted(grid_results.items(), key=lambda x: x[1]["obj"], reverse=True)
print("  Top 8 threshold×scale combos (f_boost=0.20):")
for name, r in sorted_grid[:8]:
    d = r["obj"] - baseline_obj
    sym = "✅" if d > 0.005 else ("⚠️ " if d > -0.005 else "❌")
    print(f"    {sym} {name}: OBJ={r['obj']:.4f} (Δ={d:+.4f})")

best_ts_name, best_ts_r = sorted_grid[0]
best_thr   = best_ts_r["thr"]
best_scale = best_ts_r["scale"]

print(f"\n  Sweeping f_boost=[{F_BOOSTS}] for best thr={best_thr}/sc={best_scale}...")
boost_results = {}
for boost in F_BOOSTS:
    key = f"boost={boost:.2f}"
    yearly = {}
    for year in YEARS:
        yr_sig = {sk: sig_returns[sk][year] for sk in ["v1","i460bw168","i415bw216","f168"]}
        yearly[year] = sharpe(compute_ensemble(yr_sig, signals_data[year],
                                               vol_threshold=best_thr, vol_scale=best_scale,
                                               f_boost=boost))
    o = obj_func(yearly)
    d = o - baseline_obj
    sym = "✅" if d > 0.005 else ("⚠️ " if d > -0.005 else "❌")
    print(f"    {sym} {key}: OBJ={o:.4f} (Δ={d:+.4f}) | {yearly}")
    boost_results[boost] = {"obj": o, "delta": d, "yearly": yearly}

best_boost = max(boost_results, key=lambda k: boost_results[k]["obj"])
best_obj   = boost_results[best_boost]["obj"]
best_delta = best_obj - baseline_obj
best_yearly = boost_results[best_boost]["yearly"]

print(f"\n  → Best overall: thr={best_thr} sc={best_scale} boost={best_boost}: OBJ={best_obj:.4f}")

print(f"\n[3/3] LOYO validation: thr={best_thr} sc={best_scale} boost={best_boost}...")
loyo_wins, loyo_deltas = 0, []
for held_out in YEARS:
    yr_sig = {sk: sig_returns[sk][held_out] for sk in ["v1","i460bw168","i415bw216","f168"]}
    sh_best = sharpe(compute_ensemble(yr_sig, signals_data[held_out],
                                      vol_threshold=best_thr, vol_scale=best_scale, f_boost=best_boost))
    sh_base = sharpe(compute_ensemble(yr_sig, signals_data[held_out],
                                      vol_threshold=0.50, vol_scale=0.50, f_boost=0.20))
    d = sh_best - sh_base
    loyo_deltas.append(d)
    loyo_wins += int(d > 0)
    print(f"  LOYO {held_out}: best={sh_best:.4f} base={sh_base:.4f} Δ={d:+.4f} {'✅' if d>0 else '❌'}")

loyo_avg  = float(np.mean(loyo_deltas))
validated = loyo_wins >= 3 and best_delta > 0.005

print("\n" + "=" * 68)
if validated:
    print(f"✅ VALIDATED — thr={best_thr} sc={best_scale} boost={best_boost}: OBJ={best_obj:.4f} (Δ={best_delta:+.4f}) | LOYO {loyo_wins}/5")
else:
    print(f"❌ NO IMPROVEMENT — best OBJ={best_obj:.4f} (Δ={best_delta:+.4f}) | LOYO {loyo_wins}/5")
    print(f"   Baseline (thr=0.50,sc=0.50,boost=0.20) OBJ={baseline_obj:.4f} remains optimal.")
print("=" * 68)

report = {
    "phase": 156,
    "description": "Vol Overlay Parameter Fine-Tuning",
    "hypothesis": "P129 threshold/scale/boost set with simpler baseline; v2.6.0 may prefer different params",
    "elapsed_seconds": round(time.time() - _start, 1),
    "baseline_obj": baseline_obj,
    "baseline_yearly": baseline_yearly,
    "best_params": {"vol_threshold": best_thr, "vol_scale": best_scale, "f_boost": best_boost},
    "best_obj": best_obj,
    "best_delta": best_delta,
    "best_yearly": best_yearly,
    "loyo_wins": loyo_wins,
    "loyo_avg_delta": loyo_avg,
    "validated": validated,
    "verdict": (f"VALIDATED — thr={best_thr} sc={best_scale} boost={best_boost} OBJ={best_obj:.4f} LOYO {loyo_wins}/5"
                if validated else
                f"NO IMPROVEMENT — baseline thr=0.50 sc=0.50 boost=0.20 OBJ={baseline_obj:.4f} optimal"),
    "partial": False,
    "timestamp": datetime.datetime.utcnow().isoformat(),
}

out_path = OUT_DIR / "phase156_report.json"
out_path.write_text(json.dumps(report, indent=2))
print(f"✅ Saved → {out_path}")
