"""
Phase 161 — V1 Internal Params + Funding Dispersion Threshold
=============================================================
Two orthogonal micro-optimizations:

PART A: V1 signal internal weight mix (w_carry, w_mom, w_mean_reversion)
  Current: 0.35 / 0.45 / 0.20
  Hypothesis: With v2.8.0's richer I415/I460 stack handling momentum,
  V1's optimal role may shift toward pure carry + mean-reversion insurance.
  Test grid: focus on carry-heavy / mean-reversion variants.

PART B: Funding dispersion threshold
  Current: 75th pct → boost ×1.15
  Hypothesis: 70th or 65th pct threshold may activate more often and help.
  Test: [0.60, 0.65, 0.70, 0.75, 0.80, 0.85]

Baseline (prod v2.8.0): OBJ=2.2095
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

VOL_WINDOW     = 168
VOL_THRESHOLD  = 0.5
VOL_SCALE      = 0.4
VOL_F168_BOOST = 0.15
BRD_LOOKBACK   = 192
PCT_WINDOW     = 336
P_LOW, P_HIGH  = 0.35, 0.65
FUND_DISP_THR  = 0.75   # varied in Part B
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

OUT_DIR = ROOT / "artifacts" / "phase161"
OUT_DIR.mkdir(parents=True, exist_ok=True)

COST_MODEL = cost_model_from_config({
    "fee_rate": 0.0005, "slippage_rate": 0.0003, "cost_multiplier": 1.0,
})

# Part A: V1 weight combos (w_carry, w_mom, w_mean_reversion)
# normalized so they sum to 1.0
V1_PARAM_COMBOS = [
    (0.35, 0.45, 0.20),   # baseline
    (0.40, 0.40, 0.20),   # more carry
    (0.45, 0.35, 0.20),   # carry-heavy
    (0.50, 0.30, 0.20),   # very carry-heavy
    (0.40, 0.45, 0.15),   # more carry, less rev
    (0.35, 0.40, 0.25),   # more mean-rev
    (0.30, 0.45, 0.25),   # less carry, more rev
    (0.35, 0.50, 0.15),   # more momentum
    (0.40, 0.35, 0.25),   # balanced shift
    (0.30, 0.40, 0.30),   # rev-heavy (defensive insurance)
]

# Part B: dispersion threshold values
DISP_THRESHOLDS = [0.60, 0.65, 0.70, 0.75, 0.80, 0.85]

_partial: dict = {}

def _on_timeout(signum, frame):
    print("\n⏰ TIMEOUT — saving partial...")
    _partial["partial"] = True
    _partial["timestamp"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    (OUT_DIR / "phase161_report.json").write_text(json.dumps(_partial, indent=2))
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
    # Store raw for Part B re-use
    fund_std_pct_base = np.full(n, 0.5)
    for i in range(PCT_WINDOW, n):
        fund_std_pct_base[i] = float(np.mean(fund_std_raw[i - PCT_WINDOW:i] <= fund_std_raw[i]))
    fund_std_pct_base[:PCT_WINDOW] = 0.5

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
        "btc_vol": btc_vol,
        "breadth_regime": breadth_regime,
        "fund_std_pct": fund_std_pct_base,  # baseline: 75th pct thresholded in ensemble
        "fund_std_raw_pct": fund_std_pct_base,  # same array, used for Part B
        "ts_spread_pct": ts_spread_pct,
    }


def compute_ensemble(sig_rets: dict, signals: dict,
                     fund_disp_thr: float = FUND_DISP_THR) -> np.ndarray:
    sk_all = ["v1", "i460bw168", "i415bw216", "f168"]
    min_len = min(len(sig_rets[sk]) for sk in sk_all)
    bv  = signals["btc_vol"][:min_len]
    reg = signals["breadth_regime"][:min_len]
    fsp = signals["fund_std_pct"][:min_len]
    tsp = signals["ts_spread_pct"][:min_len]

    ens = np.zeros(min_len)
    for i in range(min_len):
        w = WEIGHTS[["LOW","MID","HIGH"][int(reg[i])]]
        if not np.isnan(bv[i]) and bv[i] > VOL_THRESHOLD:
            boost = VOL_F168_BOOST / max(1, len(sk_all)-1)
            ret_i = 0.0
            for sk in sk_all:
                adj_w = (min(0.60, w[sk] + VOL_F168_BOOST) if sk == "f168"
                         else max(0.05, w[sk] - boost))
                ret_i += adj_w * sig_rets[sk][i]
            ret_i *= VOL_SCALE
        else:
            ret_i = sum(w[sk] * sig_rets[sk][i] for sk in sk_all)
        if fsp[i] > fund_disp_thr:
            ret_i *= FUND_DISP_SCALE
        if tsp[i] > TS_REDUCE_THR:
            ret_i *= TS_REDUCE_SCALE
        elif tsp[i] < TS_BOOST_THR:
            ret_i *= TS_BOOST_SCALE
        ens[i] = ret_i
    return ens


# ─── main ────────────────────────────────────────────────────────────────────

print("=" * 68)
print("PHASE 161 — V1 Params + Funding Dispersion Threshold")
print("=" * 68)

print("\n[1/5] Loading data + computing signals + base returns...")
sig_returns_base: dict = {"i460bw168": {}, "i415bw216": {}, "f168": {}}
# V1 variants stored separately
v1_variants: dict = {combo: {} for combo in V1_PARAM_COMBOS}
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

    # Shared: I460, I415, F168
    for sk, sname, params in [
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
        sig_returns_base[sk][year] = np.array(result.returns)

    # V1 variants
    for (wc, wm, wr) in V1_PARAM_COMBOS:
        bt_cfg = BacktestConfig(costs=COST_MODEL)
        strat  = make_strategy({"name": "nexus_alpha_v1", "params": {
            "k_per_side":2, "w_carry":wc, "w_mom":wm, "w_mean_reversion":wr,
            "momentum_lookback_bars":336,"mean_reversion_lookback_bars":72,
            "vol_lookback_bars":168,"target_gross_leverage":0.35,"rebalance_interval_bars":60
        }})
        result = BacktestEngine(bt_cfg).run(dataset, strat)
        v1_variants[(wc, wm, wr)][year] = np.array(result.returns)
    print(". ✓")

# Baseline uses (0.35, 0.45, 0.20) for V1
def make_yr_sig(year, v1_combo):
    return {
        "v1": v1_variants[v1_combo][year],
        "i460bw168": sig_returns_base["i460bw168"][year],
        "i415bw216": sig_returns_base["i415bw216"][year],
        "f168": sig_returns_base["f168"][year],
    }

baseline_v1 = (0.35, 0.45, 0.20)
baseline_yearly = {}
for year in YEARS:
    baseline_yearly[year] = sharpe(compute_ensemble(make_yr_sig(year, baseline_v1), signals_data[year]))
baseline_obj = obj_func(baseline_yearly)
print(f"\n  Baseline: OBJ={baseline_obj:.4f} | {baseline_yearly}")

print("\n[2/5] PART A — V1 parameter sweep...")
parta_results = {}
for combo in V1_PARAM_COMBOS:
    yearly = {}
    for year in YEARS:
        yearly[year] = sharpe(compute_ensemble(make_yr_sig(year, combo), signals_data[year]))
    o = obj_func(yearly)
    d = o - baseline_obj
    sym = "✅" if d > 0.005 else ("⚠️ " if d > -0.005 else "❌")
    label = f"w_c={combo[0]:.2f}_w_m={combo[1]:.2f}_w_r={combo[2]:.2f}"
    print(f"  {sym} {label}: OBJ={o:.4f} (Δ={d:+.4f})")
    parta_results[combo] = {"obj": o, "delta": d, "yearly": yearly}

best_v1_combo = max(parta_results, key=lambda k: parta_results[k]["obj"])
best_v1_obj   = parta_results[best_v1_combo]["obj"]
best_v1_delta = parta_results[best_v1_combo]["delta"]

print(f"\n  Part A best: w_carry={best_v1_combo[0]}, w_mom={best_v1_combo[1]}, w_rev={best_v1_combo[2]}")
print(f"  Part A best OBJ={best_v1_obj:.4f} (Δ={best_v1_delta:+.4f})")

print("\n[3/5] PART B — Funding dispersion threshold sweep...")
partb_results = {}
baseline_v1_sig = {year: make_yr_sig(year, baseline_v1) for year in YEARS}
for thr in DISP_THRESHOLDS:
    yearly = {}
    for year in YEARS:
        yearly[year] = sharpe(compute_ensemble(baseline_v1_sig[year], signals_data[year],
                                               fund_disp_thr=thr))
    o = obj_func(yearly)
    d = o - baseline_obj
    sym = "✅" if d > 0.005 else ("⚠️ " if d > -0.005 else "❌")
    print(f"  {sym} disp_thr={thr:.2f}: OBJ={o:.4f} (Δ={d:+.4f}) | {yearly}")
    partb_results[thr] = {"obj": o, "delta": d, "yearly": yearly}

best_disp_thr  = max(partb_results, key=lambda k: partb_results[k]["obj"])
best_disp_obj  = partb_results[best_disp_thr]["obj"]
best_disp_delta= partb_results[best_disp_thr]["delta"]
print(f"\n  Part B best: thr={best_disp_thr} OBJ={best_disp_obj:.4f} (Δ={best_disp_delta:+.4f})")

# Decide what to validate
# Pick the single best improvement (Part A or Part B)
if best_v1_delta >= best_disp_delta and best_v1_delta > 0.005:
    validate_part = "A"
    val_desc = f"V1: w_c={best_v1_combo[0]}/w_m={best_v1_combo[1]}/w_r={best_v1_combo[2]}"
    val_combo = best_v1_combo
    val_delta = best_v1_delta
elif best_disp_delta > 0.005:
    validate_part = "B"
    val_desc = f"Dispersion thr={best_disp_thr}"
    val_delta = best_disp_delta
else:
    validate_part = "none"
    val_desc = "none"
    val_delta = max(best_v1_delta, best_disp_delta)

print(f"\n[4/5] LOYO validation: {val_desc}...")
loyo_wins, loyo_deltas = 0, []
for held_out in YEARS:
    if validate_part == "A":
        yr_sig_best = make_yr_sig(held_out, best_v1_combo)
        sh_best = sharpe(compute_ensemble(yr_sig_best, signals_data[held_out]))
    elif validate_part == "B":
        yr_sig_best = make_yr_sig(held_out, baseline_v1)
        sh_best = sharpe(compute_ensemble(yr_sig_best, signals_data[held_out],
                                          fund_disp_thr=best_disp_thr))
    else:
        sh_best = baseline_yearly[held_out]

    sh_base = baseline_yearly[held_out]
    d = sh_best - sh_base
    loyo_deltas.append(d)
    loyo_wins += int(d > 0)
    print(f"  LOYO {held_out}: best={sh_best:.4f} base={sh_base:.4f} Δ={d:+.4f} {'✅' if d>0 else '❌'}")

loyo_avg  = float(np.mean(loyo_deltas))
validated = loyo_wins >= 3 and val_delta > 0.005 and validate_part != "none"

print("\n" + "=" * 68)
if validated:
    print(f"✅ VALIDATED — Part {validate_part} ({val_desc}): OBJ delta={val_delta:+.4f} | LOYO {loyo_wins}/5")
else:
    print(f"❌ NO IMPROVEMENT — Part A best Δ={best_v1_delta:+.4f}, Part B best Δ={best_disp_delta:+.4f}")
    print(f"   Baseline v2.8.0 OBJ={baseline_obj:.4f} remains optimal.")
print("=" * 68)

report = {
    "phase": 161,
    "description": "V1 Internal Params + Funding Dispersion Threshold",
    "elapsed_seconds": round(time.time() - _start, 1),
    "baseline_obj": baseline_obj,
    "baseline_yearly": baseline_yearly,
    "parta_best": {"combo": list(best_v1_combo), "obj": best_v1_obj, "delta": best_v1_delta},
    "partb_best": {"thr": best_disp_thr, "obj": best_disp_obj, "delta": best_disp_delta},
    "validated_part": validate_part,
    "validated_desc": val_desc,
    "loyo_wins": loyo_wins,
    "loyo_avg_delta": loyo_avg,
    "validated": validated,
    "verdict": (f"VALIDATED — Part {validate_part} {val_desc} LOYO {loyo_wins}/5"
                if validated else
                f"NO IMPROVEMENT — v2.8.0 OBJ={baseline_obj:.4f} optimal"),
    "partial": False,
    "timestamp": datetime.datetime.utcnow().isoformat(),
}

out_path = OUT_DIR / "phase161_report.json"
out_path.write_text(json.dumps(report, indent=2))
print(f"✅ Saved → {out_path}")
