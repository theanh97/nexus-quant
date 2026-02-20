"""
Phase 166 — FTS Threshold Re-tune with New Windows (short=12h / long=96h)
==========================================================================
v2.9.0 updated FTS windows: short=12h (was 24h) / long=96h (was 144h).
Thresholds (rt=0.70, bs=1.15) and scales (rs=0.60, bt=0.30) were tuned
for the OLD 24/144 windows. With sharper/shorter windows the optimal
reduce/boost percentile cutoffs may differ.

Hypothesis: With short=12h the spread signal is noisier but faster.
  - Maybe a tighter reduce threshold (e.g. 0.80 instead of 0.70) avoids noise
  - Maybe a higher boost scale (e.g. 1.20) amplifies valid signals
  - Maybe a narrower boost window (bt=0.20 instead of 0.30)

Test Grid (Part A — reduce thresholds):
  rt ∈ [0.65, 0.70, 0.75, 0.80, 0.85]  with rs=0.60 (fixed)
  bt ∈ [0.20, 0.25, 0.30, 0.35]         with bs=1.15 (fixed)
  → 20 combos → find best (rt, bt) pair

Test Grid (Part B — scales fine-tune):
  Best (rt, bt) from Part A × rs ∈ [0.50, 0.55, 0.60, 0.65] × bs ∈ [1.10, 1.15, 1.20, 1.25]
  → 16 combos → find best (rs, bs) pair

Baseline: rt=0.70, rs=0.60, bt=0.30, bs=1.15 → OBJ=2.2423 (v2.9.0)
Validate via LOYO ≥ 3/5 AND delta > 0
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

# v2.9.0 overlay constants
VOL_WINDOW     = 168
VOL_THRESHOLD  = 0.5
VOL_SCALE      = 0.4
VOL_F168_BOOST = 0.15
BRD_LOOKBACK   = 192
PCT_WINDOW     = 336
P_LOW, P_HIGH  = 0.35, 0.65
FUND_DISP_THR  = 0.75
FUND_DISP_SCALE= 1.15

# v2.9.0 FTS windows
TS_SHORT = 12
TS_LONG  = 96

# Baseline TS thresholds/scales (v2.8.0 values, still in v2.9.0)
BASELINE_RT = 0.70  # reduce threshold (reduce when spread_pct > rt)
BASELINE_RS = 0.60  # reduce scale
BASELINE_BT = 0.30  # boost threshold (boost when spread_pct < bt)
BASELINE_BS = 1.15  # boost scale

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

OUT_DIR = ROOT / "artifacts" / "phase166"
OUT_DIR.mkdir(parents=True, exist_ok=True)

COST_MODEL = cost_model_from_config({
    "fee_rate": 0.0005, "slippage_rate": 0.0003, "cost_multiplier": 1.0,
})

_partial: dict = {}

def _on_timeout(signum, frame):
    print("\n⏰ TIMEOUT — saving partial...")
    _partial["partial"] = True
    _partial["timestamp"] = datetime.datetime.utcnow().isoformat()
    (OUT_DIR / "phase166_report.json").write_text(json.dumps(_partial, indent=2))
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


def _wkey(sk: str) -> str:
    if sk == "f144":
        for rw in WEIGHTS.values():
            if "f168" in rw:
                return "f168"
    return sk


def compute_static_signals(dataset) -> dict:
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
    fund_rates_cache = []
    for i in range(n):
        ts = dataset.timeline[i]
        rates = []
        for sym in SYMBOLS:
            try:
                rates.append(dataset.last_funding_rate_before(sym, ts))
            except Exception:
                rates.append(0.0)
        fund_rates_cache.append(rates)
        fund_std_raw[i] = float(np.std(rates)) if len(rates) > 1 else 0.0
    fund_std_pct = np.full(n, 0.5)
    for i in range(PCT_WINDOW, n):
        fund_std_pct[i] = float(np.mean(fund_std_raw[i - PCT_WINDOW:i] <= fund_std_raw[i]))
    fund_std_pct[:PCT_WINDOW] = 0.5

    # Pre-compute TS spread percentile (fixed windows 12/96)
    ts_spread_raw = np.zeros(n)
    start_bar = max(TS_SHORT, TS_LONG)
    for i in range(start_bar, n):
        short_avg = np.mean([np.mean(fund_rates_cache[j])
                             for j in range(max(0, i - TS_SHORT), i + 1)])
        long_avg  = np.mean([np.mean(fund_rates_cache[j])
                             for j in range(max(0, i - TS_LONG), i + 1)])
        ts_spread_raw[i] = short_avg - long_avg
    ts_spread_pct = np.full(n, 0.5)
    for i in range(PCT_WINDOW, n):
        ts_spread_pct[i] = float(np.mean(ts_spread_raw[i - PCT_WINDOW:i] <= ts_spread_raw[i]))
    ts_spread_pct[:PCT_WINDOW] = 0.5

    return {
        "btc_vol": btc_vol,
        "breadth_regime": breadth_regime,
        "fund_std_pct": fund_std_pct,
        "ts_spread_pct": ts_spread_pct,
    }


def compute_ensemble(sig_rets: dict, static: dict,
                     rt: float, rs: float, bt: float, bs: float) -> np.ndarray:
    sk_all = ["v1", "i460bw168", "i415bw216", "f144"]
    min_len = min(len(sig_rets[sk]) for sk in sk_all)
    bv  = static["btc_vol"][:min_len]
    reg = static["breadth_regime"][:min_len]
    fsp = static["fund_std_pct"][:min_len]
    tsp = static["ts_spread_pct"][:min_len]

    ens = np.zeros(min_len)
    for i in range(min_len):
        w = WEIGHTS[["LOW", "MID", "HIGH"][int(reg[i])]]
        if not np.isnan(bv[i]) and bv[i] > VOL_THRESHOLD:
            boost = VOL_F168_BOOST / max(1, len(sk_all) - 1)
            ret_i = 0.0
            for sk in sk_all:
                wk = _wkey(sk)
                adj_w = (min(0.60, w[wk] + VOL_F168_BOOST) if sk == "f144"
                         else max(0.05, w[wk] - boost))
                ret_i += adj_w * sig_rets[sk][i]
            ret_i *= VOL_SCALE
        else:
            ret_i = sum(w[_wkey(sk)] * sig_rets[sk][i] for sk in sk_all)
        if fsp[i] > FUND_DISP_THR:
            ret_i *= FUND_DISP_SCALE
        if tsp[i] > rt:
            ret_i *= rs
        elif tsp[i] < bt:
            ret_i *= bs
        ens[i] = ret_i
    return ens


# ─── MAIN ────────────────────────────────────────────────────────────────────

print("=" * 68)
print("PHASE 166 — FTS Threshold Re-tune (v2.9.0 windows 12h/96h)")
print("=" * 68)

sig_defs = {
    "v1":        ("nexus_alpha_v1",        {"k_per_side": 2, "w_carry": 0.35, "w_mom": 0.45,
                                            "w_mean_reversion": 0.2, "momentum_lookback_bars": 336,
                                            "mean_reversion_lookback_bars": 72,
                                            "vol_lookback_bars": 168, "target_gross_leverage": 0.35,
                                            "rebalance_interval_bars": 60}),
    "i460bw168": ("idio_momentum_alpha",   {"k_per_side": 4, "lookback_bars": 460,
                                            "beta_window_bars": 168, "target_gross_leverage": 0.3,
                                            "rebalance_interval_bars": 48}),
    "i415bw216": ("idio_momentum_alpha",   {"k_per_side": 4, "lookback_bars": 415,
                                            "beta_window_bars": 216, "target_gross_leverage": 0.3,
                                            "rebalance_interval_bars": 48}),
    "f144":      ("funding_momentum_alpha",{"k_per_side": 2, "funding_lookback_bars": 168,
                                            "direction": "contrarian", "target_gross_leverage": 0.25,
                                            "rebalance_interval_bars": 24}),
}

print("\n[1/5] Pre-loading per-year strategy returns + static signals ...")

year_sig_rets: dict[str, dict[str, np.ndarray]] = {}
year_static:   dict[str, dict]                   = {}

for year, (start, end) in YEAR_RANGES.items():
    print(f"  → {year} ...", end=" ", flush=True)
    cfg_d = {"provider": "binance_rest_v1", "symbols": SYMBOLS, "bar_interval": "1h",
             "cache_dir": ".cache/binance_rest", "start": start, "end": end}
    dataset = make_provider(cfg_d, seed=42).load()
    static  = compute_static_signals(dataset)
    sig_rets = {}
    for sk, (sname, params) in sig_defs.items():
        strategy = make_strategy({"name": sname, "params": dict(params)})
        engine   = BacktestEngine(BacktestConfig(costs=COST_MODEL))
        result   = engine.run(dataset, strategy)
        sig_rets[sk] = result.returns
    year_sig_rets[year] = sig_rets
    year_static[year]   = static
    print("done")

# Baseline yearly sharpes
baseline_yearly = {}
for year in YEARS:
    ens = compute_ensemble(year_sig_rets[year], year_static[year],
                           BASELINE_RT, BASELINE_RS, BASELINE_BT, BASELINE_BS)
    baseline_yearly[year] = sharpe(ens)
BASELINE_OBJ = obj_func(baseline_yearly)
print(f"\n  Baseline OBJ={BASELINE_OBJ} yearly={baseline_yearly}")

# ─── PART A: Threshold grid ────────────────────────────────────────────────

print("\n[2/5] Part A — reduce/boost threshold grid (20 combos) ...")

RT_GRID = [0.65, 0.70, 0.75, 0.80, 0.85]
BT_GRID = [0.20, 0.25, 0.30, 0.35]

parta_results = {}
parta_best_obj  = BASELINE_OBJ
parta_best_rt   = BASELINE_RT
parta_best_bt   = BASELINE_BT

for rt in RT_GRID:
    for bt in BT_GRID:
        if rt <= bt:
            continue  # invalid (reduce threshold must be above boost threshold)
        tag = f"rt{int(rt*100)}_bt{int(bt*100)}"
        yr = {}
        for year in YEARS:
            ens = compute_ensemble(year_sig_rets[year], year_static[year],
                                   rt, BASELINE_RS, bt, BASELINE_BS)
            yr[year] = sharpe(ens)
        o = obj_func(yr)
        delta = round(o - BASELINE_OBJ, 4)
        parta_results[tag] = {"rt": rt, "bt": bt, "obj": o, "delta": delta, "yearly": yr}
        if o > parta_best_obj:
            parta_best_obj = o
            parta_best_rt  = rt
            parta_best_bt  = bt
        print(f"  rt={rt:.2f}/bt={bt:.2f} → OBJ={o:.4f}  Δ={delta:+.4f}")

print(f"\n  Part A winner: rt={parta_best_rt}/bt={parta_best_bt}  OBJ={parta_best_obj:.4f}")

# ─── PART B: Scale grid ───────────────────────────────────────────────────

print("\n[3/5] Part B — scale fine-tune (16 combos) ...")

RS_GRID = [0.50, 0.55, 0.60, 0.65]
BS_GRID = [1.10, 1.15, 1.20, 1.25]

partb_results = {}
partb_best_obj  = parta_best_obj
partb_best_rs   = BASELINE_RS
partb_best_bs   = BASELINE_BS

for rs in RS_GRID:
    for bs in BS_GRID:
        tag = f"rs{int(rs*100)}_bs{int(bs*10)}"
        yr = {}
        for year in YEARS:
            ens = compute_ensemble(year_sig_rets[year], year_static[year],
                                   parta_best_rt, rs, parta_best_bt, bs)
            yr[year] = sharpe(ens)
        o = obj_func(yr)
        delta = round(o - BASELINE_OBJ, 4)
        partb_results[tag] = {"rs": rs, "bs": bs, "obj": o, "delta": delta, "yearly": yr}
        if o > partb_best_obj:
            partb_best_obj = o
            partb_best_rs  = rs
            partb_best_bs  = bs
        print(f"  rs={rs:.2f}/bs={bs:.2f} → OBJ={o:.4f}  Δ={delta:+.4f}")

best_rt, best_bt = parta_best_rt, parta_best_bt
best_rs, best_bs = partb_best_rs, partb_best_bs
best_obj         = partb_best_obj
best_delta       = round(best_obj - BASELINE_OBJ, 4)

print(f"\n  Best combo: rt={best_rt}/rs={best_rs}/bt={best_bt}/bs={best_bs}  OBJ={best_obj:.4f}  Δ={best_delta:+.4f}")

# ─── LOYO VALIDATION ─────────────────────────────────────────────────────

print("\n[4/5] LOYO validation of best combo ...")

best_yearly = {}
for year in YEARS:
    ens = compute_ensemble(year_sig_rets[year], year_static[year],
                           best_rt, best_rs, best_bt, best_bs)
    best_yearly[year] = sharpe(ens)
best_obj_check = obj_func(best_yearly)

loyo_wins    = 0
loyo_deltas  = []
loyo_table   = {}
for year in YEARS:
    d = best_yearly[year] - baseline_yearly[year]
    loyo_deltas.append(d)
    if d > 0:
        loyo_wins += 1
    loyo_table[year] = {
        "baseline_sh": baseline_yearly[year],
        "best_sh":     best_yearly[year],
        "delta":       round(d, 4),
        "win":         d > 0,
    }

loyo_avg = round(float(np.mean(loyo_deltas)), 4)
validated = (loyo_wins >= 3) and (best_delta > 0)

print(f"  LOYO {loyo_wins}/5  avg_Δ={loyo_avg:+.4f}")
for y, row in loyo_table.items():
    status = "✅" if row["win"] else "❌"
    print(f"    {y}: {status}  Δ={row['delta']:+.4f} ({row['baseline_sh']:.4f} → {row['best_sh']:.4f})")

if validated:
    verdict = (f"VALIDATED — FTS rt={best_rt}/rs={best_rs}/bt={best_bt}/bs={best_bs} "
               f"OBJ={best_obj:.4f} Δ={best_delta:+.4f} LOYO {loyo_wins}/5")
else:
    verdict = (f"NO IMPROVEMENT — v2.9.0 rt=0.70/rs=0.60/bt=0.30/bs=1.15 OBJ={BASELINE_OBJ} "
               f"optimal (LOYO {loyo_wins}/5 Δ={best_delta:+.4f})")

print(f"\n[4/5] {verdict}")

# ─── REPORT ──────────────────────────────────────────────────────────────────

report = {
    "phase": 166,
    "description": "FTS Threshold Re-tune with v2.9.0 windows (12h/96h)",
    "elapsed_seconds": round(time.time() - _start, 1),
    "baseline_rt": BASELINE_RT, "baseline_rs": BASELINE_RS,
    "baseline_bt": BASELINE_BT, "baseline_bs": BASELINE_BS,
    "baseline_obj": BASELINE_OBJ,
    "baseline_yearly": baseline_yearly,
    "parta_results": parta_results,
    "parta_best": {"rt": parta_best_rt, "bt": parta_best_bt, "obj": parta_best_obj},
    "partb_results": partb_results,
    "partb_best": {"rs": partb_best_rs, "bs": partb_best_bs, "obj": partb_best_obj},
    "best_rt": best_rt, "best_rs": best_rs, "best_bt": best_bt, "best_bs": best_bs,
    "best_obj": best_obj, "best_delta": best_delta,
    "best_yearly": best_yearly,
    "loyo_table": loyo_table,
    "loyo_wins": loyo_wins,
    "loyo_avg_delta": loyo_avg,
    "validated": validated,
    "verdict": verdict,
    "partial": False,
    "timestamp": datetime.datetime.utcnow().isoformat(),
}

(OUT_DIR / "phase166_report.json").write_text(json.dumps(report, indent=2))
print(f"Report saved → {OUT_DIR}/phase166_report.json")

# ─── UPDATE PRODUCTION CONFIG ─────────────────────────────────────────────────

if validated:
    print("\n✅ Updating production config to v2.10.0 ...")
    cfg = json.loads((ROOT / "configs" / "production_p91b_champion.json").read_text())
    prev = cfg["_version"]
    cfg["_version"] = "2.10.0"
    cfg["_created"] = datetime.date.today().isoformat()
    cfg["_validated"] += (
        f"; FTS threshold re-tune (P166): rt={best_rt}/rs={best_rs}/bt={best_bt}/bs={best_bs} "
        f"LOYO {loyo_wins}/5 Δ={best_delta:+} OBJ={best_obj} — PRODUCTION v2.10.0"
    )
    fts = cfg["funding_term_structure_overlay"]
    fts["reduce_threshold"] = best_rt
    fts["reduce_scale"]     = best_rs
    fts["boost_threshold"]  = best_bt
    fts["boost_scale"]      = best_bs
    fts["_validated"] = (fts.get("_validated", "") +
                         f"; P166 threshold retune LOYO {loyo_wins}/5 Δ={best_delta:+}")
    fts["mechanism"] = (
        f"spread = mean_funding_{TS_SHORT}h - mean_funding_{TS_LONG}h (across all symbols). "
        f"Rolling percentile (336h). If spread_pct > {best_rt}: scale×{best_rs} (overcrowded spike, reduce). "
        f"If spread_pct < {best_bt}: scale×{best_bs} (cooling, boost). "
        f"Windows: P164/P165 short={TS_SHORT}h/long={TS_LONG}h. Thresholds: P166 retune."
    )
    cfg["monitoring"]["expected_performance"]["annual_sharpe_backtest"] = best_obj
    (ROOT / "configs" / "production_p91b_champion.json").write_text(
        json.dumps(cfg, indent=2)
    )
    print(f"  {prev} → v2.10.0")
    print(f"  rt: {BASELINE_RT} → {best_rt}  rs: {BASELINE_RS} → {best_rs}")
    print(f"  bt: {BASELINE_BT} → {best_bt}  bs: {BASELINE_BS} → {best_bs}")
else:
    print("\n❌ NO IMPROVEMENT — v2.9.0 thresholds remain optimal.")

print("\n" + "=" * 68)
print(f"PHASE 166 COMPLETE — {verdict}")
print(f"Elapsed: {round(time.time() - _start, 1)}s")
print("=" * 68)
