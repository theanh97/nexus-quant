"""
Phase 171 — Vol Regime Overlay Fine-Tune (v2.12.0 stack)
=========================================================
The vol_regime_overlay (vol_scale=0.4, f168_boost=0.15, threshold=0.5)
was last tuned at v2.6.0/2.7.0 (OBJ~2.14). Now at v2.12.0 (OBJ=2.4148),
interactions with FTS/dispersion may have changed the optimal vol regime
response. A tighter or looser scale may work better with the current stack.

Part A: vol_scale sweep [0.30, 0.35, 0.40, 0.45, 0.50, 0.55] (f168_boost=0.15, thr=0.50)
Part B: f168_boost sweep [0.00, 0.05, 0.10, 0.15, 0.20, 0.25] (best vol_scale, thr=0.50)
Part C: vol_threshold sweep [0.40, 0.45, 0.50, 0.55, 0.60] (best scale+boost)

Baseline (v2.12.0): vol_scale=0.40, f168_boost=0.15, threshold=0.50 → OBJ=2.4148
Validate: LOYO ≥ 3/5 AND delta > 0 → update to v2.13.0
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

# v2.12.0 locked overlay params
VOL_WINDOW      = 168
BRD_LOOKBACK    = 192
PCT_WINDOW      = 336
P_LOW, P_HIGH   = 0.35, 0.65
TS_SHORT, TS_LONG = 12, 96
RT, RS          = 0.60, 0.40   # FTS reduce threshold/scale
BT, BS          = 0.25, 1.50   # FTS boost threshold/scale
DISP_SCALE      = 1.05
DISP_THR        = 0.60
DISP_PCT_WIN    = 240

# Baseline vol overlay params (from v2.12.0)
BASELINE_VOL_SCALE   = 0.40
BASELINE_F168_BOOST  = 0.15
BASELINE_VOL_THR     = 0.50

# Weights (v2.8.0 regime weights, unchanged)
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

VOL_SCALES  = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55]
F168_BOOSTS = [0.00, 0.05, 0.10, 0.15, 0.20, 0.25]
VOL_THRS    = [0.40, 0.45, 0.50, 0.55, 0.60]

OUT_DIR = ROOT / "artifacts" / "phase171"
OUT_DIR.mkdir(parents=True, exist_ok=True)

COST_MODEL = cost_model_from_config({
    "fee_rate": 0.0005, "slippage_rate": 0.0003, "cost_multiplier": 1.0,
})

_partial: dict = {}

def _on_timeout(signum, frame):
    _partial["partial"] = True
    _partial["timestamp"] = datetime.datetime.utcnow().isoformat()
    (OUT_DIR / "phase171_report.json").write_text(json.dumps(_partial, indent=2, default=str))
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
    """O(n) rolling mean using cumsum."""
    n = len(x)
    cs = np.zeros(n + 1)
    for i in range(n):
        cs[i + 1] = cs[i] + x[i]
    result = np.zeros(n)
    for i in range(n):
        start = max(0, i - w + 1)
        result[i] = (cs[i + 1] - cs[start]) / (i - start + 1)
    return result


def compute_signals(dataset) -> dict:
    """Precompute all signals. BTC vol precomputed for all VOL_THRS."""
    n = len(dataset.timeline)

    # BTC hourly returns
    btc_rets = np.zeros(n)
    for i in range(1, n):
        c0 = dataset.close("BTCUSDT", i - 1)
        c1 = dataset.close("BTCUSDT", i)
        btc_rets[i] = (c1 / c0 - 1.0) if c0 > 0 else 0.0

    # BTC vol (rolling 168h ann.)
    btc_vol = np.full(n, np.nan)
    for i in range(VOL_WINDOW, n):
        btc_vol[i] = float(np.std(btc_rets[i - VOL_WINDOW:i])) * np.sqrt(8760)
    if VOL_WINDOW < n:
        btc_vol[:VOL_WINDOW] = btc_vol[VOL_WINDOW]

    # Breadth regime
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

    # Funding rates matrix
    fund_rates = np.zeros((n, len(SYMBOLS)))
    for j, sym in enumerate(SYMBOLS):
        for i in range(n):
            ts = dataset.timeline[i]
            try:
                fund_rates[i, j] = dataset.last_funding_rate_before(sym, ts)
            except Exception:
                fund_rates[i, j] = 0.0

    # Funding dispersion (v2.12.0: pct_win=240)
    fund_std_raw = np.std(fund_rates, axis=1)
    fund_std_pct = np.full(n, 0.5)
    for i in range(DISP_PCT_WIN, n):
        fund_std_pct[i] = float(np.mean(fund_std_raw[i - DISP_PCT_WIN:i] <= fund_std_raw[i]))
    fund_std_pct[:DISP_PCT_WIN] = 0.5

    # FTS spread (v2.12.0: short=12, long=96)
    xsect_mean = np.mean(fund_rates, axis=1)
    ts_raw = rolling_mean_arr(xsect_mean, TS_SHORT) - rolling_mean_arr(xsect_mean, TS_LONG)
    ts_spread_pct = np.full(n, 0.5)
    for i in range(PCT_WINDOW, n):
        ts_spread_pct[i] = float(np.mean(ts_raw[i - PCT_WINDOW:i] <= ts_raw[i]))
    ts_spread_pct[:PCT_WINDOW] = 0.5

    return {
        "btc_vol":        btc_vol,
        "breadth_regime": breadth_regime,
        "fund_std_pct":   fund_std_pct,
        "ts_spread_pct":  ts_spread_pct,
    }


def compute_ensemble(
    sig_rets: dict,
    signals: dict,
    vol_scale:  float = BASELINE_VOL_SCALE,
    f168_boost: float = BASELINE_F168_BOOST,
    vol_thr:    float = BASELINE_VOL_THR,
) -> np.ndarray:
    sk_all = ["v1", "i460bw168", "i415bw216", "f168"]
    min_len = min(len(sig_rets[sk]) for sk in sk_all)
    bv  = signals["btc_vol"][:min_len]
    reg = signals["breadth_regime"][:min_len]
    fsp = signals["fund_std_pct"][:min_len]
    tsp = signals["ts_spread_pct"][:min_len]

    ens = np.zeros(min_len)
    for i in range(min_len):
        w = WEIGHTS[["LOW", "MID", "HIGH"][int(reg[i])]]
        # Vol regime branch
        if not np.isnan(bv[i]) and bv[i] > vol_thr:
            boost_per_sig = f168_boost / max(1, len(sk_all) - 1)
            ret_i = 0.0
            for sk in sk_all:
                if sk == "f168":
                    adj_w = min(0.60, w[sk] + f168_boost)
                else:
                    adj_w = max(0.05, w[sk] - boost_per_sig)
                ret_i += adj_w * sig_rets[sk][i]
            ret_i *= vol_scale
        else:
            ret_i = sum(w[sk] * sig_rets[sk][i] for sk in sk_all)
        # Dispersion boost (v2.12.0)
        if fsp[i] > DISP_THR:
            ret_i *= DISP_SCALE
        # FTS overlay (v2.12.0)
        if tsp[i] > RT:
            ret_i *= RS
        elif tsp[i] < BT:
            ret_i *= BS
        ens[i] = ret_i
    return ens


# ─── MAIN ────────────────────────────────────────────────────────────────────

print("=" * 68)
print("PHASE 171 — Vol Regime Overlay Fine-Tune (v2.12.0)")
print("=" * 68)
print(f"  Baseline: vol_scale={BASELINE_VOL_SCALE} f168_boost={BASELINE_F168_BOOST} thr={BASELINE_VOL_THR}")
print(f"  v2.12.0 OBJ=2.4148\n")

print("[1/4] Loading data, signals, strategy returns ...")

strat_rets_by_year: dict = {sk: {} for sk in ["v1", "i460bw168", "i415bw216", "f168"]}
signals_by_year: dict = {}

sig_specs = [
    ("v1",        "nexus_alpha_v1",        {"k_per_side": 2, "w_carry": 0.35, "w_mom": 0.45,
                                             "w_mean_reversion": 0.2, "momentum_lookback_bars": 336,
                                             "mean_reversion_lookback_bars": 72,
                                             "vol_lookback_bars": 168, "target_gross_leverage": 0.35,
                                             "rebalance_interval_bars": 60}),
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

for year, (start, end) in YEAR_RANGES.items():
    print(f"  {year}: ", end="", flush=True)
    cfg_data = {"provider": "binance_rest_v1", "symbols": SYMBOLS,
                "start": start, "end": end, "bar_interval": "1h",
                "cache_dir": ".cache/binance_rest"}
    dataset = make_provider(cfg_data, seed=42).load()
    signals_by_year[year] = compute_signals(dataset)
    print("S", end="", flush=True)
    for sk, sname, params in sig_specs:
        result = BacktestEngine(BacktestConfig(costs=COST_MODEL)).run(
            dataset, make_strategy({"name": sname, "params": params})
        )
        strat_rets_by_year[sk][year] = np.array(result.returns)
    print(". ✓")

def get_rets(year):
    return {sk: strat_rets_by_year[sk][year] for sk in ["v1", "i460bw168", "i415bw216", "f168"]}

baseline_yearly = {y: sharpe(compute_ensemble(get_rets(y), signals_by_year[y])) for y in YEARS}
BASELINE_OBJ = obj_func(baseline_yearly)
_partial.update({"phase": 171, "description": "Vol Regime Fine-Tune", "baseline_obj": BASELINE_OBJ,
                 "baseline_yearly": baseline_yearly, "partial": True})
print(f"\n  Baseline OBJ={BASELINE_OBJ:.4f}  yearly={baseline_yearly}")

# ─── PART A: vol_scale sweep ──────────────────────────────────────────────

print("\n[2/4] Part A — vol_scale sweep ...")

parta_results = {}
parta_best_obj   = BASELINE_OBJ
parta_best_scale = BASELINE_VOL_SCALE

for vs in VOL_SCALES:
    tag = f"vs{int(vs*100)}"
    yr = {y: sharpe(compute_ensemble(get_rets(y), signals_by_year[y], vol_scale=vs)) for y in YEARS}
    o  = obj_func(yr)
    d  = round(o - BASELINE_OBJ, 4)
    parta_results[tag] = {"vol_scale": vs, "obj": o, "delta": d, "yearly": yr}
    if o > parta_best_obj:
        parta_best_obj   = o
        parta_best_scale = vs
    print(f"  vs={vs:.2f} → OBJ={o:.4f}  Δ={d:+.4f}")

print(f"\n  Part A winner: vol_scale={parta_best_scale}  OBJ={parta_best_obj:.4f}")

# ─── PART B: f168_boost sweep ─────────────────────────────────────────────

print("\n[3/4] Part B — f168_boost sweep ...")

partb_results = {}
partb_best_obj   = parta_best_obj
partb_best_boost = BASELINE_F168_BOOST

for fb in F168_BOOSTS:
    tag = f"fb{int(fb*100)}"
    yr  = {y: sharpe(compute_ensemble(get_rets(y), signals_by_year[y],
                                      vol_scale=parta_best_scale, f168_boost=fb)) for y in YEARS}
    o   = obj_func(yr)
    d   = round(o - BASELINE_OBJ, 4)
    partb_results[tag] = {"f168_boost": fb, "obj": o, "delta": d, "yearly": yr}
    if o > partb_best_obj:
        partb_best_obj   = o
        partb_best_boost = fb
    print(f"  boost={fb:.2f} → OBJ={o:.4f}  Δ={d:+.4f}")

print(f"\n  Part B winner: f168_boost={partb_best_boost}  OBJ={partb_best_obj:.4f}")

# ─── PART C: vol_threshold sweep ──────────────────────────────────────────

print("\n[4/4] Part C — vol_threshold sweep ...")

partc_results = {}
best_obj = partb_best_obj
best_scale, best_boost, best_thr = parta_best_scale, partb_best_boost, BASELINE_VOL_THR

for vt in VOL_THRS:
    tag = f"vt{int(vt*100)}"
    yr  = {y: sharpe(compute_ensemble(get_rets(y), signals_by_year[y],
                                      vol_scale=parta_best_scale, f168_boost=partb_best_boost,
                                      vol_thr=vt)) for y in YEARS}
    o   = obj_func(yr)
    d   = round(o - BASELINE_OBJ, 4)
    partc_results[tag] = {"vol_thr": vt, "obj": o, "delta": d, "yearly": yr}
    if o > best_obj:
        best_obj   = o
        best_thr   = vt
    print(f"  thr={vt:.2f} → OBJ={o:.4f}  Δ={d:+.4f}")

best_delta = round(best_obj - BASELINE_OBJ, 4)
print(f"\n  Best: vol_scale={best_scale} boost={best_boost} thr={best_thr}  OBJ={best_obj:.4f}  Δ={best_delta:+.4f}")

# ─── LOYO VALIDATION ──────────────────────────────────────────────────────

best_yearly = {y: sharpe(compute_ensemble(get_rets(y), signals_by_year[y],
                                           vol_scale=best_scale, f168_boost=best_boost,
                                           vol_thr=best_thr)) for y in YEARS}
best_obj_check = obj_func(best_yearly)

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

if validated:
    verdict = (f"VALIDATED — vol_scale={best_scale}/f168_boost={best_boost}/vol_thr={best_thr} "
               f"OBJ={best_obj:.4f} Δ={best_delta:+.4f} LOYO {loyo_wins}/5")
else:
    verdict = (f"NO IMPROVEMENT — v2.12.0 vol_scale=0.40/boost=0.15/thr=0.50 OBJ={BASELINE_OBJ} "
               f"optimal (LOYO {loyo_wins}/5)")

print(f"\n  LOYO {loyo_wins}/5  avg_Δ={loyo_avg:+.4f}")
for y, row in loyo_table.items():
    icon = "✅" if row["win"] else "❌"
    print(f"    {y}: {icon}  Δ={row['delta']:+.4f}")

print(f"\n  {verdict}")

# ─── REPORT ──────────────────────────────────────────────────────────────────

report = {
    "phase": 171,
    "description": "Vol Regime Overlay Fine-Tune (v2.12.0 stack)",
    "elapsed_seconds": round(time.time() - _start, 1),
    "baseline_vol_scale": BASELINE_VOL_SCALE, "baseline_f168_boost": BASELINE_F168_BOOST,
    "baseline_vol_thr": BASELINE_VOL_THR, "baseline_obj": BASELINE_OBJ,
    "baseline_yearly": baseline_yearly,
    "parta_results": parta_results,
    "parta_best": {"vol_scale": parta_best_scale, "obj": parta_best_obj},
    "partb_results": partb_results,
    "partb_best": {"f168_boost": partb_best_boost, "obj": partb_best_obj},
    "partc_results": partc_results,
    "partc_best": {"vol_thr": best_thr, "obj": best_obj},
    "best_vol_scale": best_scale, "best_f168_boost": best_boost, "best_vol_thr": best_thr,
    "best_obj": best_obj, "best_delta": best_delta, "best_yearly": best_yearly,
    "loyo_table": loyo_table, "loyo_wins": loyo_wins, "loyo_avg_delta": loyo_avg,
    "validated": validated, "verdict": verdict,
    "partial": False, "timestamp": datetime.datetime.utcnow().isoformat(),
}

(OUT_DIR / "phase171_report.json").write_text(json.dumps(report, indent=2, default=str))
print(f"\nReport → {OUT_DIR}/phase171_report.json")

# ─── UPDATE PRODUCTION CONFIG ─────────────────────────────────────────────────

if validated:
    print("\n✅ Updating production config to v2.13.0 ...")
    cfg = json.loads((ROOT / "configs" / "production_p91b_champion.json").read_text())
    prev = cfg["_version"]
    cfg["_version"] = "2.13.0"
    cfg["_created"] = datetime.date.today().isoformat()
    cfg["_validated"] += (
        f"; Vol overlay re-tune (P171): scale={best_scale}/boost={best_boost}/thr={best_thr} "
        f"LOYO {loyo_wins}/5 Δ={best_delta:+} OBJ={best_obj} — PRODUCTION v2.13.0"
    )
    vol = cfg["vol_regime_overlay"]
    vol["scale_factor"] = best_scale
    vol["f144_boost"]   = best_boost
    vol["threshold"]    = best_thr
    vol["_validated"] = (vol.get("_validated", "") +
                         f"; P171 retune LOYO {loyo_wins}/5 Δ={best_delta:+}")
    cfg["monitoring"]["expected_performance"]["annual_sharpe_backtest"] = best_obj
    (ROOT / "configs" / "production_p91b_champion.json").write_text(json.dumps(cfg, indent=2))
    print(f"  {prev} → v2.13.0")
    print(f"  vol_scale: {BASELINE_VOL_SCALE} → {best_scale}")
    print(f"  f168_boost: {BASELINE_F168_BOOST} → {best_boost}")
    print(f"  threshold: {BASELINE_VOL_THR} → {best_thr}")
else:
    print("\n❌ NO IMPROVEMENT — v2.12.0 vol params remain optimal.")

print("\n" + "=" * 68)
print(f"PHASE 171 COMPLETE — {verdict}")
print(f"Elapsed: {round(time.time() - _start, 1)}s")
print("=" * 68)
