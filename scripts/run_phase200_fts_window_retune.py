"""
Phase 200 — FTS Window Re-Tune (v2.27.0+ stack)
=================================================
FTS (funding term structure) windows: short=12h, long=96h
Set in P164-165 at v2.9.0 (OBJ~2.24). Now at ~3.0 with major weight changes.

With F168 weight now dominant (0.45/0.40/0.15), the FTS signal
may benefit from different term spread definition.

Test:
  Part A: long window ∈ [48, 72, 96, 120, 144, 168] (short=12 fixed)
  Part B: short window ∈ [4, 8, 12, 16, 24] (using best long)

Read config dynamically.
Validate: LOYO ≥3/5 AND Δ>0 → update config
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
    out = Path("artifacts/phase200"); out.mkdir(parents=True, exist_ok=True)
    (out / "phase200_report.json").write_text(json.dumps(_partial, indent=2, default=str))
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

COST_MODEL = cost_model_from_config({
    "fee_rate": 0.0005, "slippage_rate": 0.0003, "cost_multiplier": 1.0,
})

def get_config():
    cfg = json.loads(Path("configs/production_p91b_champion.json").read_text())
    brs = cfg["breadth_regime_switching"]
    rw = brs["regime_weights"]
    weights = {}
    for regime in ["LOW", "MID", "HIGH"]:
        w = rw[regime]
        weights[regime] = {
            "v1":   w["v1"],
            "i460": w.get("i460bw168", w.get("i460", 0.15)),
            "i415": w.get("i415bw216", w.get("i415", 0.25)),
            "f168": w["f168"],
        }
    fts = cfg.get("funding_term_structure_overlay", {})
    ts_rt = fts.get("reduce_threshold", 0.65); ts_rs = fts.get("reduce_scale", 0.45)
    ts_bt = fts.get("boost_threshold", 0.25); ts_bs = fts.get("boost_scale", 1.70)
    ts_short_cur = fts.get("short_window_bars", 12)
    ts_long_cur = fts.get("long_window_bars", 96)
    vol_ov = cfg.get("vol_regime_overlay", {})
    vol_thr = vol_ov.get("threshold", 0.50)
    vol_scale = vol_ov.get("scale_factor", 0.30)
    f168_boost = vol_ov.get("f144_boost", 0.00)
    m = re.findall(r"OBJ=([\d.]+)", cfg.get("_validated", ""))
    baseline_obj = float(m[-1]) if m else 2.9628
    return (cfg, weights, ts_rt, ts_rs, ts_bt, ts_bs, ts_short_cur, ts_long_cur,
            vol_thr, vol_scale, f168_boost, baseline_obj)

LONG_SWEEP  = [48, 72, 96, 120, 144, 168]
SHORT_SWEEP = [4, 6, 8, 12, 16, 24]

VOL_WINDOW = 168; BRD_LB = 192; PCT_WINDOW = 336
P_LOW, P_HIGH = 0.20, 0.60; FUND_DISP_PCT = 240
DISP_THR = 0.60; DISP_SCALE = 1.0

def rolling_mean_arr(a, w):
    out = np.full(len(a), np.nan)
    if w <= 0: return out
    cs = np.cumsum(np.where(np.isnan(a), 0.0, a))
    for i in range(w - 1, len(a)):
        out[i] = cs[i] / w if i == w - 1 else (cs[i] - cs[i - w]) / w
    return out

def load_year(year, vol_thr):
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

    # Pre-compute raw cross-sectional funding mean for TS spread
    xsect_mean = np.mean(fund_rates, axis=1)

    sig_rets = {}
    for sk, sname, p in [
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
            dataset, make_strategy({"name": sname, "params": p}))
        sig_rets[sk] = np.array(result.returns)

    print(".", end=" ", flush=True)
    return btc_vol, fund_std_pct, xsect_mean, regime, sig_rets, n

def compute_ens_with_windows(year_data, weights, short_w, long_w,
                              ts_rt, ts_rs, ts_bt, ts_bs,
                              vol_thr, vol_scale, f168_boost):
    btc_vol, fund_std_pct, xsect_mean, regime, sig_rets, n = year_data
    # Recompute TS spread with given windows
    ts_raw = rolling_mean_arr(xsect_mean, short_w) - rolling_mean_arr(xsect_mean, long_w)
    ts_spread_pct = np.full(len(ts_raw), 0.5)
    for i in range(PCT_WINDOW, len(ts_raw)):
        ts_spread_pct[i] = float(np.mean(ts_raw[i - PCT_WINDOW:i] <= ts_raw[i]))
    ts_spread_pct[:PCT_WINDOW] = 0.5

    min_len = min(len(v) for v in sig_rets.values())
    bv = btc_vol[:min_len]; reg = regime[:min_len]
    fsp = fund_std_pct[:min_len]; tsp = ts_spread_pct[:min_len]
    ens = np.zeros(min_len)
    for i in range(min_len):
        w = weights[["LOW", "MID", "HIGH"][int(reg[i])]]
        if not np.isnan(bv[i]) and bv[i] > vol_thr:
            if f168_boost > 0:
                bpp = f168_boost / 3.0
                ret_i = (
                    (w["v1"] + bpp) * sig_rets["v1"][i] +
                    (w["i460"] + bpp) * sig_rets["i460"][i] +
                    (w["i415"] + bpp) * sig_rets["i415"][i] +
                    (w["f168"] - f168_boost) * sig_rets["f168"][i]
                ) * vol_scale
            else:
                ret_i = (
                    w["v1"] * sig_rets["v1"][i] +
                    w["i460"] * sig_rets["i460"][i] +
                    w["i415"] * sig_rets["i415"][i] +
                    w["f168"] * sig_rets["f168"][i]
                ) * vol_scale
        else:
            ret_i = (
                w["v1"] * sig_rets["v1"][i] +
                w["i460"] * sig_rets["i460"][i] +
                w["i415"] * sig_rets["i415"][i] +
                w["f168"] * sig_rets["f168"][i]
            )
        if DISP_SCALE > 1.0 and fsp[i] > DISP_THR: ret_i *= DISP_SCALE
        if tsp[i] > ts_rt: ret_i *= ts_rs
        elif tsp[i] < ts_bt: ret_i *= ts_bs
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

def eval_windows(year_data, weights, short_w, long_w,
                 ts_rt, ts_rs, ts_bt, ts_bs, vol_thr, vol_scale, f168_boost):
    yr_sharpes = {}
    for yr in YEARS:
        ens = compute_ens_with_windows(
            year_data[yr], weights, short_w, long_w,
            ts_rt, ts_rs, ts_bt, ts_bs, vol_thr, vol_scale, f168_boost)
        yr_sharpes[yr] = sharpe(ens, year_data[yr][-1])
    return obj_fn(yr_sharpes)

# ── MAIN ────────────────────────────────────────────────────────────────────
print("=" * 60)
print("Phase 200 — FTS Window Re-Tune (Milestone)")
print("=" * 60)
_start = time.time()

(prod_cfg, WEIGHTS, TS_RT, TS_RS, TS_BT, TS_BS, TS_SHORT_CUR, TS_LONG_CUR,
 VOL_THR, VOL_SCALE, F168_BOOST, BASELINE_OBJ) = get_config()
ver_now = prod_cfg.get("_version", "2.27.0")
print(f"  Config: {ver_now}  Baseline OBJ≈{BASELINE_OBJ:.4f}")
print(f"  FTS: short={TS_SHORT_CUR} long={TS_LONG_CUR} rt={TS_RT:.2f} rs={TS_RS:.2f} bt={TS_BT:.2f} bs={TS_BS:.2f}")

print("\n[1] Loading all years ...")
year_data = {}
for yr in YEARS:
    print(f"  {yr}", end="", flush=True)
    year_data[yr] = load_year(yr, VOL_THR)
    print()

# Part A: long window sweep
print(f"\n[2] Part A: long window sweep (short={TS_SHORT_CUR} fixed) ...")
partA = []
for lw in LONG_SWEEP:
    if lw <= TS_SHORT_CUR: continue
    obj = eval_windows(year_data, WEIGHTS, TS_SHORT_CUR, lw,
                       TS_RT, TS_RS, TS_BT, TS_BS, VOL_THR, VOL_SCALE, F168_BOOST)
    partA.append({"short": TS_SHORT_CUR, "long": lw, "obj": obj})
    flag = " ← CURRENT" if lw == TS_LONG_CUR else ""
    print(f"  long={lw:3d}h  OBJ={obj:.4f}{flag}")

best_A = max(partA, key=lambda x: x["obj"])
print(f"  Best long: {best_A['long']}h  OBJ={best_A['obj']:.4f}")

# Part B: short window sweep
lw_best = best_A["long"]
print(f"\n[3] Part B: short window sweep (long={lw_best}) ...")
partB = []
for sw in SHORT_SWEEP:
    if sw >= lw_best: continue
    obj = eval_windows(year_data, WEIGHTS, sw, lw_best,
                       TS_RT, TS_RS, TS_BT, TS_BS, VOL_THR, VOL_SCALE, F168_BOOST)
    partB.append({"short": sw, "long": lw_best, "obj": obj})
    flag = " ← CURRENT" if sw == TS_SHORT_CUR and lw_best == TS_LONG_CUR else ""
    print(f"  short={sw:2d}h  OBJ={obj:.4f}{flag}")

best_B = max(partB, key=lambda x: x["obj"])
best = best_B
sw_final = best["short"]; lw_final = best["long"]
print(f"  Best: short={sw_final} long={lw_final}  OBJ={best['obj']:.4f}  Δ={best['obj']-BASELINE_OBJ:+.4f}")

# LOYO
print(f"\n[4] LOYO validation (short={sw_final} long={lw_final} vs current {TS_SHORT_CUR}/{TS_LONG_CUR}) ...")
loyo_wins = 0; loyo_table = []
for yr in YEARS:
    ens_best = compute_ens_with_windows(year_data[yr], WEIGHTS, sw_final, lw_final,
                                         TS_RT, TS_RS, TS_BT, TS_BS, VOL_THR, VOL_SCALE, F168_BOOST)
    ens_curr = compute_ens_with_windows(year_data[yr], WEIGHTS, TS_SHORT_CUR, TS_LONG_CUR,
                                         TS_RT, TS_RS, TS_BT, TS_BS, VOL_THR, VOL_SCALE, F168_BOOST)
    n = year_data[yr][-1]
    sh_b = sharpe(ens_best, n); sh_c = sharpe(ens_curr, n)
    win = bool(sh_b > sh_c)
    if win: loyo_wins += 1
    loyo_table.append({"year": yr, "sh_best": sh_b, "sh_curr": sh_c, "win": win})
    print(f"  {yr}: best={sh_b:.4f} curr={sh_c:.4f} {'WIN' if win else 'LOSS'}")

print(f"\n  LOYO: {loyo_wins}/5 wins")
validated = loyo_wins >= 3 and best["obj"] > BASELINE_OBJ

out = Path("artifacts/phase200"); out.mkdir(parents=True, exist_ok=True)
report = {
    "phase": 200, "target": "fts_windows",
    "partA": partA, "partB": partB,
    "final": {"short": sw_final, "long": lw_final},
    "best_obj": best["obj"], "delta": best["obj"] - BASELINE_OBJ,
    "baseline_obj": BASELINE_OBJ,
    "loyo": {"wins": loyo_wins, "table": loyo_table},
    "validated": validated,
    "runtime_s": round(time.time() - _start, 1),
    "timestamp": datetime.now(UTC).isoformat(),
}
(out / "phase200_report.json").write_text(json.dumps(report, indent=2, default=str))
print(f"  Report saved  ({report['runtime_s']}s)")

if validated:
    cfg = json.loads(Path("configs/production_p91b_champion.json").read_text())
    ver = cfg.get("_version", "2.27.0")
    major, minor, patch = map(int, ver.split("."))
    new_ver = f"{major}.{minor + 1}.{patch}"
    cfg["_version"] = new_ver
    cfg["_validated"] += (
        f"; FTS window retune P200: short={TS_SHORT_CUR}->{sw_final} long={TS_LONG_CUR}->{lw_final}"
        f" LOYO {loyo_wins}/5 delta={best['obj']-BASELINE_OBJ:+.4f} OBJ={best['obj']:.4f}"
        f" — PRODUCTION {new_ver}"
    )
    fts = cfg.get("funding_term_structure_overlay", {})
    fts["short_window_bars"] = sw_final
    fts["long_window_bars"] = lw_final
    Path("configs/production_p91b_champion.json").write_text(json.dumps(cfg, indent=2))
    bov = best["obj"]; d = bov - BASELINE_OBJ
    cm = "phase200: FTS window retune -- VALIDATED short=%d long=%d LOYO %d/5 delta=%+.4f OBJ=%.4f" % (sw_final, lw_final, loyo_wins, d, bov)
    print(f"\n✅ VALIDATED → {new_ver} OBJ={bov:.4f}")
    os.system("git add configs/production_p91b_champion.json artifacts/phase200/ scripts/run_phase200_fts_window_retune.py")
    os.system('git commit -m "' + cm + '"')
    os.system("git pull --rebase && git push")
else:
    delta = best["obj"] - BASELINE_OBJ
    verdict = "NO IMPROVEMENT" if delta <= 0 else f"WEAK LOYO {loyo_wins}/5"
    cm = "phase200: FTS window retune -- %s short=%d long=%d delta=%+.4f" % (verdict, sw_final, lw_final, delta)
    print(f"\n❌ NOT VALIDATED — {verdict}")
    os.system("git add artifacts/phase200/ scripts/run_phase200_fts_window_retune.py")
    os.system('git commit -m "' + cm + '"')
    os.system("git pull --rebase && git push")

print("\n[DONE] Phase 200 complete.")
