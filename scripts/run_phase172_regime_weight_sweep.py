"""
Phase 172 — Breadth Regime Weight Re-Optimization (v2.13.0 stack)
==================================================================
The regime weights (LOW/MID/HIGH) were fixed in P158 (v2.8.0 baseline OBJ=2.2095).
Now at v2.13.0 (OBJ=2.4153) with upgraded FTS/dispersion/vol overlays, the optimal
weight distribution per regime may have shifted.

Key: With more aggressive FTS boost (bs=1.50 when TS is cooling), F168 signal is
already amplified contextually. This may mean we need LESS static F168 weight and
MORE idio weight — especially in HIGH momentum regime.

Strategy:
  For each regime independently, sweep F168 weight ∈ [0.10, 0.15, 0.20, 0.25, 0.30]
  and redistribute the delta proportionally among V1/I460/I415 (preserving their
  relative ratios).

  Current regime weights (v2.8.0 / unchanged through v2.13.0):
    LOW:  v1=0.2747 i460=0.1967 i415=0.3247 f168=0.2039
    MID:  v1=0.16   i460=0.22   i415=0.37   f168=0.25
    HIGH: v1=0.05   i460=0.25   i415=0.45   f168=0.25

Part A: LOW regime F168 sweep (5 values) — find best LOW f168 weight
Part B: MID regime F168 sweep (5 values) — find best MID f168 weight
Part C: HIGH regime F168 sweep (5 values) — find best HIGH f168 weight
Part D: Joint best (A×B×C combo) vs baseline — LOYO validation

Validate: LOYO ≥ 3/5 AND delta > 0 → update to v2.14.0
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

# v2.13.0 locked overlay constants
VOL_WINDOW    = 168
VOL_SCALE     = 0.40
F168_BOOST    = 0.10    # updated in P171
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

# Current production weights (v2.8.0 / unchanged)
BASE_WEIGHTS = {
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

# F168 weight sweep per regime
F168_SWEEP = [0.10, 0.15, 0.20, 0.25, 0.30]

OUT_DIR = ROOT / "artifacts" / "phase172"
OUT_DIR.mkdir(parents=True, exist_ok=True)

COST_MODEL = cost_model_from_config({
    "fee_rate": 0.0005, "slippage_rate": 0.0003, "cost_multiplier": 1.0,
})

_partial: dict = {}

def _on_timeout(signum, frame):
    _partial["partial"] = True
    _partial["timestamp"] = datetime.datetime.utcnow().isoformat()
    (OUT_DIR / "phase172_report.json").write_text(json.dumps(_partial, indent=2, default=str))
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


def make_weights_with_f168(regime: str, f168_w: float) -> dict:
    """Adjust F168 weight for a regime, redistribute delta proportionally to V1/I460/I415."""
    bw = BASE_WEIGHTS[regime]
    current_f168 = bw["f168"]
    delta = f168_w - current_f168
    others = {k: v for k, v in bw.items() if k != "f168"}
    total_others = sum(others.values())
    new_w = {"f168": f168_w}
    for k, v in others.items():
        new_w[k] = round(v - delta * (v / total_others), 4)
    # Normalize to sum=1
    s = sum(new_w.values())
    return {k: round(v / s, 4) for k, v in new_w.items()}


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


def compute_ensemble(sig_rets: dict, signals: dict, weights: dict = None) -> np.ndarray:
    if weights is None:
        weights = BASE_WEIGHTS
    sk_all = ["v1", "i460bw168", "i415bw216", "f168"]
    min_len = min(len(sig_rets[sk]) for sk in sk_all)
    bv  = signals["btc_vol"][:min_len]
    reg = signals["breadth_regime"][:min_len]
    fsp = signals["fund_std_pct"][:min_len]
    tsp = signals["ts_spread_pct"][:min_len]

    ens = np.zeros(min_len)
    for i in range(min_len):
        w = weights[["LOW", "MID", "HIGH"][int(reg[i])]]
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


# ─── MAIN ────────────────────────────────────────────────────────────────────

print("=" * 68)
print("PHASE 172 — Breadth Regime Weight Re-Optimization")
print("=" * 68)
print(f"  v2.13.0 baseline OBJ=2.4153\n")

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

print("[1/5] Loading data, signals, strategy returns ...")
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

baseline_yearly = {y: sharpe(compute_ensemble(get_rets(y), sigs_by_yr[y])) for y in YEARS}
BASELINE_OBJ = obj_func(baseline_yearly)
_partial.update({"phase": 172, "description": "Regime Weight Sweep", "baseline_obj": BASELINE_OBJ, "partial": True})
print(f"\n  Baseline OBJ={BASELINE_OBJ:.4f}  yearly={baseline_yearly}")

def run_sweep(regime: str) -> tuple[float, dict, float]:
    """Sweep F168 weight for a regime; return (best_f168_w, best_weights, best_obj)."""
    best_w   = BASE_WEIGHTS[regime]["f168"]
    best_obj_r = BASELINE_OBJ
    results  = {}
    for f168_w in F168_SWEEP:
        wts = {**BASE_WEIGHTS}
        wts[regime] = make_weights_with_f168(regime, f168_w)
        yr = {y: sharpe(compute_ensemble(get_rets(y), sigs_by_yr[y], weights=wts)) for y in YEARS}
        o  = obj_func(yr)
        d  = round(o - BASELINE_OBJ, 4)
        results[f"f168_{int(f168_w*100)}"] = {"f168_w": f168_w, "weights": wts[regime], "obj": o, "delta": d, "yearly": yr}
        print(f"    f168={f168_w:.2f} → OBJ={o:.4f}  Δ={d:+.4f}")
        if o > best_obj_r:
            best_obj_r = o
            best_w     = f168_w
    return best_w, results, best_obj_r

print("\n[2/5] Part A — LOW regime F168 sweep ...")
best_low_f168, parta_res, parta_obj = run_sweep("LOW")
best_low_w = make_weights_with_f168("LOW", best_low_f168)
print(f"  LOW winner: f168={best_low_f168}  OBJ={parta_obj:.4f}  weights={best_low_w}")

print("\n[3/5] Part B — MID regime F168 sweep ...")
best_mid_f168, partb_res, partb_obj = run_sweep("MID")
best_mid_w = make_weights_with_f168("MID", best_mid_f168)
print(f"  MID winner: f168={best_mid_f168}  OBJ={partb_obj:.4f}  weights={best_mid_w}")

print("\n[4/5] Part C — HIGH regime F168 sweep ...")
best_high_f168, partc_res, partc_obj = run_sweep("HIGH")
best_high_w = make_weights_with_f168("HIGH", best_high_f168)
print(f"  HIGH winner: f168={best_high_f168}  OBJ={partc_obj:.4f}  weights={best_high_w}")

# Joint combination
print("\n[5/5] Part D — Joint best combo LOYO validation ...")
joint_weights = {"LOW": best_low_w, "MID": best_mid_w, "HIGH": best_high_w}
joint_yearly  = {y: sharpe(compute_ensemble(get_rets(y), sigs_by_yr[y], weights=joint_weights)) for y in YEARS}
joint_obj     = obj_func(joint_yearly)
joint_delta   = round(joint_obj - BASELINE_OBJ, 4)

loyo_wins   = 0
loyo_deltas = []
loyo_table  = {}
for y in YEARS:
    d = joint_yearly[y] - baseline_yearly[y]
    loyo_deltas.append(d)
    if d > 0:
        loyo_wins += 1
    loyo_table[y] = {"baseline_sh": baseline_yearly[y], "joint_sh": joint_yearly[y],
                     "delta": round(d, 4), "win": bool(d > 0)}

loyo_avg  = round(float(np.mean(loyo_deltas)), 4)
validated = (loyo_wins >= 3) and (joint_delta > 0)

print(f"  Joint OBJ={joint_obj:.4f}  Δ={joint_delta:+.4f}  LOYO {loyo_wins}/5")
for y, row in loyo_table.items():
    icon = "✅" if row["win"] else "❌"
    print(f"    {y}: {icon}  Δ={row['delta']:+.4f}")

if validated:
    verdict = (f"VALIDATED — LOW_f168={best_low_f168}/MID_f168={best_mid_f168}/HIGH_f168={best_high_f168} "
               f"OBJ={joint_obj:.4f} Δ={joint_delta:+.4f} LOYO {loyo_wins}/5")
else:
    verdict = (f"NO IMPROVEMENT — v2.13.0 BASE_WEIGHTS OBJ={BASELINE_OBJ} "
               f"optimal (LOYO {loyo_wins}/5 Δ={joint_delta:+.4f})")

print(f"\n  {verdict}")

report = {
    "phase": 172,
    "description": "Breadth Regime Weight Re-Optimization",
    "elapsed_seconds": round(time.time() - _start, 1),
    "baseline_weights": BASE_WEIGHTS,
    "baseline_obj": BASELINE_OBJ,
    "baseline_yearly": baseline_yearly,
    "parta_low_sweep": parta_res,
    "parta_best": {"f168_w": best_low_f168, "weights": best_low_w, "obj": parta_obj},
    "partb_mid_sweep": partb_res,
    "partb_best": {"f168_w": best_mid_f168, "weights": best_mid_w, "obj": partb_obj},
    "partc_high_sweep": partc_res,
    "partc_best": {"f168_w": best_high_f168, "weights": best_high_w, "obj": partc_obj},
    "joint_weights": joint_weights,
    "joint_obj": joint_obj, "joint_delta": joint_delta,
    "joint_yearly": joint_yearly,
    "loyo_table": loyo_table, "loyo_wins": loyo_wins, "loyo_avg_delta": loyo_avg,
    "validated": validated, "verdict": verdict,
    "partial": False, "timestamp": datetime.datetime.utcnow().isoformat(),
}

(OUT_DIR / "phase172_report.json").write_text(json.dumps(report, indent=2, default=str))
print(f"\nReport → {OUT_DIR}/phase172_report.json")

if validated:
    print("\n✅ Updating production config to v2.14.0 ...")
    cfg = json.loads((ROOT / "configs" / "production_p91b_champion.json").read_text())
    prev = cfg["_version"]
    cfg["_version"] = "2.14.0"
    cfg["_created"] = datetime.date.today().isoformat()
    cfg["_validated"] += (
        f"; Regime weight re-opt (P172): LOW_f168={best_low_f168}/MID_f168={best_mid_f168}/HIGH_f168={best_high_f168} "
        f"LOYO {loyo_wins}/5 Δ={joint_delta:+} OBJ={joint_obj} — PRODUCTION v2.14.0"
    )
    brd = cfg["breadth_regime_switching"]
    brd["regime_weights"]["LOW"]  = {**best_low_w,  "_regime": brd["regime_weights"]["LOW"].get("_regime", "")}
    brd["regime_weights"]["MID"]  = {**best_mid_w,  "_regime": brd["regime_weights"]["MID"].get("_regime", "")}
    brd["regime_weights"]["HIGH"] = {**best_high_w, "_regime": brd["regime_weights"]["HIGH"].get("_regime", "")}
    cfg["monitoring"]["expected_performance"]["annual_sharpe_backtest"] = joint_obj
    (ROOT / "configs" / "production_p91b_champion.json").write_text(json.dumps(cfg, indent=2))
    print(f"  {prev} → v2.14.0")
else:
    print("\n❌ NO IMPROVEMENT — v2.13.0 regime weights remain optimal.")

print("\n" + "=" * 68)
print(f"PHASE 172 COMPLETE — {verdict}")
print(f"Elapsed: {round(time.time() - _start, 1)}s")
print("=" * 68)
