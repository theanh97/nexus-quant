"""
Phase 176 — FTS Boost Params Re-Tune (v2.14.0 stack)
=====================================================
P169 set bt=0.25/bs=1.50 at v2.10.0 baseline (OBJ=2.3510).
Since then: regime weights updated (P172 — F168 now 0.30 in LOW/MID),
vol boost reduced (P171 — f168_boost 0.15→0.10).

With F168 getting MORE weight in LOW/MID regimes (0.30 vs 0.2039/0.25),
the FTS boost (which amplifies ALL signals including the heavier F168) might
need to be moderated to avoid over-amplification.

Alternatively: F168 is now stronger → FTS boost can be more targeted.

Part A: boost_threshold bt ∈ [0.15, 0.20, 0.25, 0.30, 0.35]  (bs=1.50 fixed)
Part B: boost_scale bs ∈ [1.25, 1.35, 1.45, 1.50, 1.60, 1.75]  (best bt)

Baseline (v2.14.0): bt=0.25, bs=1.50 → OBJ=2.4817
Validate: LOYO ≥ 3/5 AND delta > 0 → update to v2.15.0
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

# v2.14.0 locked constants (except bt/bs being tested)
VOL_WINDOW    = 168
VOL_SCALE     = 0.40
F168_BOOST    = 0.10
VOL_THR       = 0.50
BRD_LOOKBACK  = 192
PCT_WINDOW    = 336
P_LOW, P_HIGH = 0.35, 0.65
TS_SHORT      = 12
TS_LONG       = 96
RT, RS        = 0.60, 0.40  # FTS reduce — locked
DISP_SCALE    = 1.05
DISP_THR      = 0.60
DISP_PCT_WIN  = 240

# Baseline FTS boost (from v2.14.0)
BASELINE_BT = 0.25
BASELINE_BS = 1.50

# v2.14.0 regime weights
WEIGHTS = {
    "LOW":  {"v1": 0.2415, "i460bw168": 0.173,  "i415bw216": 0.2855, "f168": 0.30},
    "MID":  {"v1": 0.1493, "i460bw168": 0.2053, "i415bw216": 0.3453, "f168": 0.30},
    "HIGH": {"v1": 0.0567, "i460bw168": 0.2833, "i415bw216": 0.51,   "f168": 0.15},
}

YEAR_RANGES = {
    "2021": ("2021-02-01", "2021-12-31"),
    "2022": ("2022-01-01", "2022-12-31"),
    "2023": ("2023-01-01", "2023-12-31"),
    "2024": ("2024-01-01", "2024-12-31"),
    "2025": ("2025-01-01", "2025-12-31"),
}
YEARS = sorted(YEAR_RANGES.keys())

BT_SWEEP = [0.15, 0.20, 0.25, 0.30, 0.35]
BS_SWEEP = [1.25, 1.35, 1.45, 1.50, 1.60, 1.75]

OUT_DIR = ROOT / "artifacts" / "phase176"
OUT_DIR.mkdir(parents=True, exist_ok=True)

COST_MODEL = cost_model_from_config({
    "fee_rate": 0.0005, "slippage_rate": 0.0003, "cost_multiplier": 1.0,
})

_partial: dict = {}

def _on_timeout(signum, frame):
    _partial["partial"] = True
    _partial["timestamp"] = datetime.datetime.now(datetime.UTC).isoformat()
    (OUT_DIR / "phase176_report.json").write_text(json.dumps(_partial, indent=2, default=str))
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


def compute_ensemble(sig_rets: dict, signals: dict, bt: float, bs: float) -> np.ndarray:
    sk_all = ["v1", "i460bw168", "i415bw216", "f168"]
    min_len = min(len(sig_rets[sk]) for sk in sk_all)
    bv  = signals["btc_vol"][:min_len]
    reg = signals["breadth_regime"][:min_len]
    fsp = signals["fund_std_pct"][:min_len]
    tsp = signals["ts_spread_pct"][:min_len]

    ens = np.zeros(min_len)
    for i in range(min_len):
        w = WEIGHTS[["LOW", "MID", "HIGH"][int(reg[i])]]
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
        elif tsp[i] < bt:
            ret_i *= bs
        ens[i] = ret_i
    return ens


# ─── MAIN ────────────────────────────────────────────────────────────────────

print("=" * 68)
print("PHASE 176 — FTS Boost Params Re-Tune (v2.14.0)")
print("=" * 68)
print(f"  Baseline: bt={BASELINE_BT} bs={BASELINE_BS}  v2.14.0 OBJ=2.4817\n")

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

print("[1/4] Loading data + signals + strategy returns ...")
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

baseline_yearly = {y: sharpe(compute_ensemble(get_rets(y), sigs_by_yr[y], BASELINE_BT, BASELINE_BS)) for y in YEARS}
BASELINE_OBJ    = obj_func(baseline_yearly)
_partial.update({"phase": 176, "description": "FTS Boost Re-tune", "baseline_obj": BASELINE_OBJ, "partial": True})
print(f"\n  Baseline OBJ={BASELINE_OBJ:.4f}  yearly={baseline_yearly}")

print("\n[2/4] Part A — boost_threshold sweep ...")
parta_results  = {}
parta_best_obj = BASELINE_OBJ
parta_best_bt  = BASELINE_BT

for bt in BT_SWEEP:
    yr = {y: sharpe(compute_ensemble(get_rets(y), sigs_by_yr[y], bt, BASELINE_BS)) for y in YEARS}
    o  = obj_func(yr)
    d  = round(o - BASELINE_OBJ, 4)
    parta_results[f"bt{int(bt*100)}"] = {"bt": bt, "obj": o, "delta": d, "yearly": yr}
    if o > parta_best_obj:
        parta_best_obj = o
        parta_best_bt  = bt
    print(f"  bt={bt:.2f} → OBJ={o:.4f}  Δ={d:+.4f}")
print(f"\n  Part A winner: bt={parta_best_bt}  OBJ={parta_best_obj:.4f}")

print("\n[3/4] Part B — boost_scale sweep ...")
partb_results  = {}
partb_best_obj = parta_best_obj
partb_best_bs  = BASELINE_BS

for bs in BS_SWEEP:
    yr = {y: sharpe(compute_ensemble(get_rets(y), sigs_by_yr[y], parta_best_bt, bs)) for y in YEARS}
    o  = obj_func(yr)
    d  = round(o - BASELINE_OBJ, 4)
    partb_results[f"bs{int(bs*100)}"] = {"bs": bs, "obj": o, "delta": d, "yearly": yr}
    if o > partb_best_obj:
        partb_best_obj = o
        partb_best_bs  = bs
    print(f"  bs={bs:.2f} → OBJ={o:.4f}  Δ={d:+.4f}")

best_bt     = parta_best_bt
best_bs     = partb_best_bs
best_obj    = partb_best_obj
best_delta  = round(best_obj - BASELINE_OBJ, 4)
print(f"\n  Best: bt={best_bt}/bs={best_bs}  OBJ={best_obj:.4f}  Δ={best_delta:+.4f}")

print("\n[4/4] LOYO validation ...")
best_yearly = {y: sharpe(compute_ensemble(get_rets(y), sigs_by_yr[y], best_bt, best_bs)) for y in YEARS}

loyo_wins   = 0
loyo_deltas = []
loyo_table  = {}
for y in YEARS:
    d = best_yearly[y] - baseline_yearly[y]
    loyo_deltas.append(d)
    if d > 0:
        loyo_wins += 1
    loyo_table[y] = {"baseline_sh": baseline_yearly[y], "best_sh": best_yearly[y],
                     "delta": round(d, 4), "win": bool(d > 0)}

loyo_avg  = round(float(np.mean(loyo_deltas)), 4)
validated = (loyo_wins >= 3) and (best_delta > 0)

print(f"  LOYO {loyo_wins}/5  avg_Δ={loyo_avg:+.4f}")
for y, row in loyo_table.items():
    icon = "✅" if row["win"] else "❌"
    print(f"    {y}: {icon}  Δ={row['delta']:+.4f}")

if validated:
    verdict = f"VALIDATED — bt={best_bt}/bs={best_bs} OBJ={best_obj:.4f} Δ={best_delta:+.4f} LOYO {loyo_wins}/5"
else:
    verdict = f"NO IMPROVEMENT — bt=0.25/bs=1.50 OBJ={BASELINE_OBJ} optimal (LOYO {loyo_wins}/5)"

print(f"\n  {verdict}")

report = {
    "phase": 176,
    "description": "FTS Boost Params Re-Tune (v2.14.0)",
    "elapsed_seconds": round(time.time() - _start, 1),
    "baseline_bt": BASELINE_BT, "baseline_bs": BASELINE_BS, "baseline_obj": BASELINE_OBJ,
    "baseline_yearly": baseline_yearly,
    "parta_results": parta_results,
    "parta_best": {"bt": parta_best_bt, "obj": parta_best_obj},
    "partb_results": partb_results,
    "partb_best": {"bs": partb_best_bs, "obj": partb_best_obj},
    "best_bt": best_bt, "best_bs": best_bs, "best_obj": best_obj, "best_delta": best_delta,
    "best_yearly": best_yearly,
    "loyo_table": loyo_table, "loyo_wins": loyo_wins, "loyo_avg_delta": loyo_avg,
    "validated": validated, "verdict": verdict,
    "partial": False, "timestamp": datetime.datetime.now(datetime.UTC).isoformat(),
}

(OUT_DIR / "phase176_report.json").write_text(json.dumps(report, indent=2, default=str))
print(f"\nReport → {OUT_DIR}/phase176_report.json")

if validated:
    print("\n✅ Updating production config to v2.15.0 ...")
    cfg = json.loads((ROOT / "configs" / "production_p91b_champion.json").read_text())
    prev = cfg["_version"]
    cfg["_version"] = "2.15.0"
    cfg["_created"] = datetime.date.today().isoformat()
    cfg["_validated"] += (
        f"; FTS boost re-tune (P176): bt={best_bt}/bs={best_bs} "
        f"LOYO {loyo_wins}/5 Δ={best_delta:+} OBJ={best_obj} — PRODUCTION v2.15.0"
    )
    fts = cfg["funding_term_structure_overlay"]
    fts["boost_threshold"] = best_bt
    fts["boost_scale"]     = best_bs
    fts["_validated"] = fts.get("_validated", "") + f"; P176 boost retune LOYO {loyo_wins}/5 Δ={best_delta:+}"
    cfg["monitoring"]["expected_performance"]["annual_sharpe_backtest"] = best_obj
    (ROOT / "configs" / "production_p91b_champion.json").write_text(json.dumps(cfg, indent=2))
    print(f"  {prev} → v2.15.0  bt: {BASELINE_BT} → {best_bt}  bs: {BASELINE_BS} → {best_bs}")
else:
    print("\n❌ NO IMPROVEMENT — bt=0.25/bs=1.50 remain optimal.")

print("\n" + "=" * 68)
print(f"PHASE 176 COMPLETE — {verdict}")
print(f"Elapsed: {round(time.time() - _start, 1)}s")
print("=" * 68)
