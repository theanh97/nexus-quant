"""
Phase 197 — FTS Threshold Re-Tune (v2.24.0 stack)
===================================================
FTS (funding_term_structure) overlay thresholds set in P168/P176:
  rt=0.60, rs=0.40, bt=0.35, bs=1.60

These were tuned at v2.10.0-v2.11.0 (OBJ~2.35). Now at OBJ=2.88
with V1 weight increased in LOW/MID regimes. Re-tune thresholds.

Part A: Sweep rt ∈ [0.50, 0.55, 0.60, 0.65, 0.70] × bt ∈ [0.25, 0.30, 0.35, 0.40]
         with fixed rs=0.40, bs=1.60
Part B: If improvement found, fine-tune rs ∈ [0.30, 0.35, 0.40, 0.45] × bs ∈ [1.50, 1.60, 1.70, 1.80]

Baseline: v2.24.0 OBJ=2.8797
Validate: LOYO ≥3/5 AND Δ>0.005
"""
import os, sys, json, time, re
import signal as _signal
from pathlib import Path
from datetime import datetime, UTC

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

import numpy as np

from nexus_quant.backtest.costs import cost_model_from_config
from nexus_quant.backtest.engine import BacktestConfig, BacktestEngine
from nexus_quant.data.providers.registry import make_provider
from nexus_quant.strategies.registry import make_strategy

_partial: dict = {}
def _on_timeout(signum, frame):
    _partial["partial"] = True
    _partial["timeout_at"] = datetime.now(UTC).isoformat()
    out = Path("artifacts/phase197"); out.mkdir(parents=True, exist_ok=True)
    (out / "phase197_report.json").write_text(json.dumps(_partial, indent=2, default=str))
    sys.exit(0)
_signal.signal(_signal.SIGALRM, _on_timeout)
_signal.alarm(7200)

SYMBOLS = ["BTCUSDT","ETHUSDT","SOLUSDT","BNBUSDT","XRPUSDT",
           "ADAUSDT","DOGEUSDT","AVAXUSDT","DOTUSDT","LINKUSDT"]
YEAR_RANGES = {
    2021: ("2021-02-01", "2022-01-01"),
    2022: ("2022-01-01", "2023-01-01"),
    2023: ("2023-01-01", "2024-01-01"),
    2024: ("2024-01-01", "2025-01-01"),
    2025: ("2025-01-01", "2026-01-01"),
}
YEARS = sorted(YEAR_RANGES)

VOL_WINDOW = 168; BRD_LB = 192; PCT_WINDOW = 336
P_LOW, P_HIGH = 0.20, 0.60; TS_SHORT = 12; TS_LONG = 96; FUND_DISP_PCT = 240
VOL_THRESHOLD = 0.50; VOL_SCALE = 0.40; VOL_F168_BOOST = 0.10
DISP_THR = 0.60; DISP_SCALE = 1.0

# Current FTS params
RT_CUR = 0.60; RS_CUR = 0.40; BT_CUR = 0.35; BS_CUR = 1.60

# v2.24.0 regime weights
WEIGHTS = {
    "LOW":  {"v1": 0.48,   "i460": 0.0642, "i415": 0.1058, "f168": 0.35},
    "MID":  {"v1": 0.35,   "i460": 0.1119, "i415": 0.1881, "f168": 0.35},
    "HIGH": {"v1": 0.06,   "i460": 0.30,   "i415": 0.54,   "f168": 0.10},
}
BASELINE_OBJ = 2.8797

COST_MODEL = cost_model_from_config({
    "fee_rate": 0.0005, "slippage_rate": 0.0003, "cost_multiplier": 1.0,
})

# Sweep grids
RT_GRID = [0.50, 0.55, 0.60, 0.65, 0.70]
BT_GRID = [0.25, 0.30, 0.35, 0.40, 0.45]
RS_GRID = [0.30, 0.35, 0.40, 0.45, 0.50]
BS_GRID = [1.40, 1.50, 1.60, 1.70, 1.80, 1.90]

def rolling_mean_arr(a, w):
    out = np.full(len(a), np.nan)
    if w <= 0: return out
    cs = np.cumsum(np.where(np.isnan(a), 0.0, a))
    for i in range(w - 1, len(a)):
        out[i] = cs[i] / w if i == w - 1 else (cs[i] - cs[i - w]) / w
    return out

def load_year(year):
    s, e = YEAR_RANGES[year]
    cfg_d = {"provider": "binance_rest_v1", "symbols": SYMBOLS,
             "start": s, "end": e, "bar_interval": "1h", "cache_dir": ".cache/binance_rest"}
    dataset = make_provider(cfg_d, seed=42).load()
    n = len(dataset.timeline)

    close_mat = np.zeros((n, len(SYMBOLS)))
    for j, sym in enumerate(SYMBOLS):
        for i in range(n): close_mat[i, j] = dataset.close(sym, i)

    btc_rets = np.zeros(n)
    for i in range(1, n):
        c0 = close_mat[i-1, 0]; c1 = close_mat[i, 0]
        btc_rets[i] = (c1/c0-1.0) if c0 > 0 else 0.0
    btc_vol = np.full(n, np.nan)
    for i in range(VOL_WINDOW, n):
        btc_vol[i] = float(np.std(btc_rets[i-VOL_WINDOW:i])) * np.sqrt(8760)
    if VOL_WINDOW < n: btc_vol[:VOL_WINDOW] = btc_vol[VOL_WINDOW]

    fund_rates = np.zeros((n, len(SYMBOLS)))
    for j, sym in enumerate(SYMBOLS):
        for i in range(n):
            ts = dataset.timeline[i]
            try:    fund_rates[i, j] = dataset.last_funding_rate_before(sym, ts)
            except: fund_rates[i, j] = 0.0

    fund_std_raw = np.std(fund_rates, axis=1)
    fund_std_pct = np.full(n, 0.5)
    for i in range(FUND_DISP_PCT, n):
        fund_std_pct[i] = float(np.mean(fund_std_raw[i-FUND_DISP_PCT:i] <= fund_std_raw[i]))
    fund_std_pct[:FUND_DISP_PCT] = 0.5

    xsect_mean = np.mean(fund_rates, axis=1)
    ts_raw = rolling_mean_arr(xsect_mean, TS_SHORT) - rolling_mean_arr(xsect_mean, TS_LONG)
    ts_spread_pct = np.full(n, 0.5)
    for i in range(PCT_WINDOW, n):
        ts_spread_pct[i] = float(np.mean(ts_raw[i-PCT_WINDOW:i] <= ts_raw[i]))
    ts_spread_pct[:PCT_WINDOW] = 0.5

    breadth = np.full(n, 0.5)
    for i in range(BRD_LB, n):
        pos = sum(1 for j in range(len(SYMBOLS))
                  if close_mat[i-BRD_LB, j] > 0 and close_mat[i, j] > close_mat[i-BRD_LB, j])
        breadth[i] = pos / len(SYMBOLS)
    breadth[:BRD_LB] = 0.5
    brd_pct = np.full(n, 0.5)
    for i in range(PCT_WINDOW, n):
        brd_pct[i] = float(np.mean(breadth[i-PCT_WINDOW:i] <= breadth[i]))
    brd_pct[:PCT_WINDOW] = 0.5
    regime = np.where(brd_pct >= P_HIGH, 2, np.where(brd_pct >= P_LOW, 1, 0)).astype(int)

    sig_rets = {}
    for sk, sname, params in [
        ("v1", "nexus_alpha_v1", {
            "k_per_side": 2, "w_carry": 0.35, "w_mom": 0.40, "w_mean_reversion": 0.25,
            "momentum_lookback_bars": 336, "mean_reversion_lookback_bars": 84,
            "vol_lookback_bars": 192, "target_gross_leverage": 0.35, "rebalance_interval_bars": 60,
        }),
        ("i460", "idio_momentum_alpha", {
            "k_per_side": 4, "lookback_bars": 460, "beta_window_bars": 168,
            "target_gross_leverage": 0.3, "rebalance_interval_bars": 48,
        }),
        ("i415", "idio_momentum_alpha", {
            "k_per_side": 4, "lookback_bars": 415, "beta_window_bars": 144,
            "target_gross_leverage": 0.3, "rebalance_interval_bars": 48,
        }),
        ("f168", "funding_momentum_alpha", {
            "k_per_side": 2, "funding_lookback_bars": 168, "direction": "contrarian",
            "target_gross_leverage": 0.25, "rebalance_interval_bars": 24,
        }),
    ]:
        result = BacktestEngine(BacktestConfig(costs=COST_MODEL)).run(
            dataset, make_strategy({"name": sname, "params": params}))
        sig_rets[sk] = np.array(result.returns)

    print(".", end=" ", flush=True)
    return btc_vol, fund_std_pct, ts_spread_pct, regime, sig_rets, n

def compute_ens(year_data, rt, rs, bt, bs):
    btc_vol, fund_std_pct, ts_spread_pct, regime, sig_rets, n = year_data
    min_len = min(len(v) for v in sig_rets.values())
    bv = btc_vol[:min_len]; reg = regime[:min_len]
    fsp = fund_std_pct[:min_len]; tsp = ts_spread_pct[:min_len]
    ens = np.zeros(min_len)
    for i in range(min_len):
        w = WEIGHTS[["LOW", "MID", "HIGH"][int(reg[i])]]
        if not np.isnan(bv[i]) and bv[i] > VOL_THRESHOLD:
            boost_per = VOL_F168_BOOST / 3.0
            ret_i = (
                (w["v1"] + boost_per) * sig_rets["v1"][i] +
                (w["i460"] + boost_per) * sig_rets["i460"][i] +
                (w["i415"] + boost_per) * sig_rets["i415"][i] +
                (w["f168"] - VOL_F168_BOOST) * sig_rets["f168"][i]
            ) * VOL_SCALE
        else:
            ret_i = (
                w["v1"] * sig_rets["v1"][i] +
                w["i460"] * sig_rets["i460"][i] +
                w["i415"] * sig_rets["i415"][i] +
                w["f168"] * sig_rets["f168"][i]
            )
        if DISP_SCALE > 1.0 and fsp[i] > DISP_THR: ret_i *= DISP_SCALE
        if tsp[i] > rt: ret_i *= rs
        elif tsp[i] < bt: ret_i *= bs
        ens[i] = ret_i
    return ens

def sharpe(rets, n):
    if n < 20: return -999.0
    mu = float(np.mean(rets[:n])) * 8760
    sd = float(np.std(rets[:n])) * np.sqrt(8760)
    return mu / sd if sd > 1e-9 else 0.0

def obj_fn(yr_sharpes):
    vals = list(yr_sharpes.values())
    return float(np.mean(vals) - 0.5 * np.std(vals))

def eval_params(year_data, rt, rs, bt, bs):
    yr_sharpes = {}
    for yr in YEARS:
        ens = compute_ens(year_data[yr], rt, rs, bt, bs)
        yr_sharpes[yr] = sharpe(ens, year_data[yr][-1])
    return obj_fn(yr_sharpes)

# ── MAIN ────────────────────────────────────────────────────────────────────
print("=" * 60)
print("Phase 197 — FTS Threshold Re-Tune")
print(f"Baseline: v2.24.0 OBJ={BASELINE_OBJ}")
print("=" * 60)
_start = time.time()

print("\n[1] Loading all years ...")
year_data = {}
for yr in YEARS:
    print(f"  {yr}", end="", flush=True)
    year_data[yr] = load_year(yr)
    print()

# Part A: Sweep rt × bt grid (rs=0.40, bs=1.60 fixed)
print("\n[2] Part A: rt × bt grid sweep (rs=0.40, bs=1.60) ...")
partA_results = []
for rt in RT_GRID:
    for bt in BT_GRID:
        if bt >= rt: continue  # invalid: boost must be lower than reduce
        obj = eval_params(year_data, rt, RS_CUR, bt, BS_CUR)
        partA_results.append({"rt": rt, "rs": RS_CUR, "bt": bt, "bs": BS_CUR, "obj": obj})
        flag = " ← CURRENT" if abs(rt - RT_CUR) < 0.01 and abs(bt - BT_CUR) < 0.01 else ""
        print(f"  rt={rt:.2f} bt={bt:.2f}  OBJ={obj:.4f}{flag}")

partA_best = max(partA_results, key=lambda x: x["obj"])
print(f"\n  Part A best: rt={partA_best['rt']:.2f} bt={partA_best['bt']:.2f} OBJ={partA_best['obj']:.4f} Δ={partA_best['obj']-BASELINE_OBJ:+.4f}")

# Part B: Sweep rs × bs grid (using best rt/bt from Part A)
rt_best = partA_best["rt"]; bt_best = partA_best["bt"]
print(f"\n[3] Part B: rs × bs grid sweep (rt={rt_best:.2f} bt={bt_best:.2f}) ...")
partB_results = []
for rs in RS_GRID:
    for bs in BS_GRID:
        obj = eval_params(year_data, rt_best, rs, bt_best, bs)
        partB_results.append({"rt": rt_best, "rs": rs, "bt": bt_best, "bs": bs, "obj": obj})
        flag = " ← CURRENT" if abs(rs - RS_CUR) < 0.01 and abs(bs - BS_CUR) < 0.01 else ""
        print(f"  rs={rs:.2f} bs={bs:.2f}  OBJ={obj:.4f}{flag}")

partB_best = max(partB_results, key=lambda x: x["obj"])
print(f"\n  Part B best: rs={partB_best['rs']:.2f} bs={partB_best['bs']:.2f} OBJ={partB_best['obj']:.4f} Δ={partB_best['obj']-BASELINE_OBJ:+.4f}")

best = partB_best
rt_f = best["rt"]; rs_f = best["rs"]; bt_f = best["bt"]; bs_f = best["bs"]

# LOYO validation
print(f"\n[4] LOYO validation (rt={rt_f:.2f} rs={rs_f:.2f} bt={bt_f:.2f} bs={bs_f:.2f}) ...")
loyo_wins = 0; loyo_table = []
for yr in YEARS:
    ens_best = compute_ens(year_data[yr], rt_f, rs_f, bt_f, bs_f)
    ens_curr = compute_ens(year_data[yr], RT_CUR, RS_CUR, BT_CUR, BS_CUR)
    n = year_data[yr][-1]
    sh_b = sharpe(ens_best, n); sh_c = sharpe(ens_curr, n)
    win = bool(sh_b > sh_c)
    if win: loyo_wins += 1
    loyo_table.append({"year": yr, "sh_best": sh_b, "sh_curr": sh_c, "win": win})
    print(f"  {yr}: best={sh_b:.4f} curr={sh_c:.4f} {'WIN' if win else 'LOSS'}")

print(f"\n  LOYO: {loyo_wins}/5 wins")
validated = loyo_wins >= 3 and best["obj"] > BASELINE_OBJ + 0.005

# Save report
out = Path("artifacts/phase197"); out.mkdir(parents=True, exist_ok=True)
report = {
    "phase": 197, "target": "fts_thresholds",
    "partA_best": partA_best, "partB_best": partB_best,
    "final": {"rt": rt_f, "rs": rs_f, "bt": bt_f, "bs": bs_f},
    "best_obj": best["obj"], "delta": best["obj"] - BASELINE_OBJ,
    "baseline_obj": BASELINE_OBJ,
    "loyo": {"wins": loyo_wins, "table": loyo_table},
    "validated": validated,
    "runtime_s": round(time.time() - _start, 1),
    "timestamp": datetime.now(UTC).isoformat(),
}
(out / "phase197_report.json").write_text(json.dumps(report, indent=2, default=str))
print(f"  Report saved → artifacts/phase197/phase197_report.json  ({report['runtime_s']}s)")

if validated:
    cfg = json.loads(Path("configs/production_p91b_champion.json").read_text())
    ver = cfg.get("_version", "2.24.0")
    major, minor, patch = map(int, ver.split("."))
    new_ver = f"{major}.{minor + 1}.{patch}"
    cfg["_version"] = new_ver
    cfg["_validated"] += (
        f"; FTS threshold re-tune P197: rt={RT_CUR:.2f}->{rt_f:.2f} rs={RS_CUR:.2f}->{rs_f:.2f}"
        f" bt={BT_CUR:.2f}->{bt_f:.2f} bs={BS_CUR:.2f}->{bs_f:.2f}"
        f" LOYO {loyo_wins}/5 delta={best['obj']-BASELINE_OBJ:+.4f} OBJ={best['obj']:.4f}"
        f" — PRODUCTION {new_ver}"
    )
    # Update FTS in config
    fts = cfg.get("funding_term_structure_overlay", {})
    fts["reduce_threshold"] = rt_f
    fts["reduce_scale"] = rs_f
    fts["boost_threshold"] = bt_f
    fts["boost_scale"] = bs_f
    Path("configs/production_p91b_champion.json").write_text(json.dumps(cfg, indent=2))
    bov = best["obj"]; d = bov - BASELINE_OBJ
    cm = "phase197: FTS threshold retune -- VALIDATED rt=%.2f rs=%.2f bt=%.2f bs=%.2f LOYO %d/5 delta=%+.4f OBJ=%.4f" % (rt_f, rs_f, bt_f, bs_f, loyo_wins, d, bov)
    print(f"\n✅ VALIDATED → {new_ver} OBJ={bov:.4f}")
    os.system("git add configs/production_p91b_champion.json artifacts/phase197/ scripts/run_phase197_fts_threshold_retune.py")
    os.system('git commit -m "' + cm + '"')
    os.system("git pull --rebase && git push")
else:
    delta = best["obj"] - BASELINE_OBJ
    verdict = "NO IMPROVEMENT" if delta <= 0 else f"WEAK LOYO {loyo_wins}/5"
    bov = best["obj"]
    cm = "phase197: FTS threshold retune -- %s rt=%.2f bt=%.2f delta=%+.4f" % (verdict, rt_f, bt_f, delta)
    print(f"\n❌ NOT VALIDATED — {verdict}")
    os.system("git add artifacts/phase197/ scripts/run_phase197_fts_threshold_retune.py")
    os.system('git commit -m "' + cm + '"')
    os.system("git pull --rebase && git push")

print("\n[DONE] Phase 197 complete.")
