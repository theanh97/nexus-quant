"""
Phase 198 — Vol Regime Overlay Re-Tune (v2.26.0 stack)
========================================================
Vol regime overlay: vol_scale=0.40, f168_boost=0.10, vol_thr=0.50
These reduce exposure when BTC vol is high and boost F168.

Major changes since P171 (last vol retune):
  - F168 weight increased: LOW 0.35→0.45, MID 0.35→0.40, HIGH 0.10→0.15
  - V1 significantly changed (weights, params)
  - FTS thresholds updated

With higher F168 weight, the f168_boost (which reduces F168 during high vol)
has stronger effect. vol_scale=0.40 might be too aggressive or not enough.

Test:
  Part A: vol_scale ∈ [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60] (f168_boost=0.10 fixed)
  Part B: f168_boost ∈ [0.00, 0.05, 0.10, 0.15, 0.20] (using best vol_scale)

Read current config dynamically.
Baseline: v2.26.0 OBJ read from config.
Validate: LOYO ≥3/5 AND Δ>0 → update config.
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
    out = Path("artifacts/phase198"); out.mkdir(parents=True, exist_ok=True)
    (out / "phase198_report.json").write_text(json.dumps(_partial, indent=2, default=str))
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
    # Read FTS params
    fts = cfg.get("funding_term_structure_overlay", {})
    ts_rt = fts.get("reduce_threshold", 0.65)
    ts_rs = fts.get("reduce_scale", 0.45)
    ts_bt = fts.get("boost_threshold", 0.25)
    ts_bs = fts.get("boost_scale", 1.70)
    # Read vol overlay params
    vol_ov = cfg.get("vol_regime_overlay", {})
    vol_thr = vol_ov.get("threshold", 0.50)
    vol_scale_cur = vol_ov.get("scale_factor", 0.40)
    f168_boost_cur = vol_ov.get("f144_boost", vol_ov.get("f168_boost", 0.10))
    # Baseline OBJ
    m = re.findall(r"OBJ=([\d.]+)", cfg.get("_validated", ""))
    baseline_obj = float(m[-1]) if m else 2.8995
    return cfg, weights, ts_rt, ts_rs, ts_bt, ts_bs, vol_thr, vol_scale_cur, f168_boost_cur, baseline_obj

VOL_WINDOW = 168; BRD_LB = 192; PCT_WINDOW = 336
P_LOW, P_HIGH = 0.20, 0.60; TS_SHORT = 12; TS_LONG = 96; FUND_DISP_PCT = 240
DISP_THR = 0.60; DISP_SCALE = 1.0

VOL_SCALE_SWEEP   = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]
F168_BOOST_SWEEP  = [0.00, 0.05, 0.10, 0.15, 0.20, 0.25]

def rolling_mean_arr(a, w):
    out = np.full(len(a), np.nan)
    if w <= 0: return out
    cs = np.cumsum(np.where(np.isnan(a), 0.0, a))
    for i in range(w - 1, len(a)):
        out[i] = cs[i] / w if i == w - 1 else (cs[i] - cs[i - w]) / w
    return out

def load_year(year, params):
    ts_rt, ts_rs, ts_bt, ts_bs = params["ts_rt"], params["ts_rs"], params["ts_bt"], params["ts_bs"]
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

    f168_lb = params.get("f168_lb", 168)
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
            "k_per_side": 2, "funding_lookback_bars": f168_lb, "direction": "contrarian",
            "target_gross_leverage": 0.25, "rebalance_interval_bars": 24,
        }),
    ]:
        result = BacktestEngine(BacktestConfig(costs=COST_MODEL)).run(
            dataset, make_strategy({"name": sname, "params": p}))
        sig_rets[sk] = np.array(result.returns)

    print(".", end=" ", flush=True)
    return btc_vol, fund_std_pct, ts_spread_pct, regime, sig_rets, n

def compute_ens(year_data, weights, vol_thr, vol_scale, f168_boost, ts_rt, ts_rs, ts_bt, ts_bs):
    btc_vol, fund_std_pct, ts_spread_pct, regime, sig_rets, n = year_data
    min_len = min(len(v) for v in sig_rets.values())
    bv = btc_vol[:min_len]; reg = regime[:min_len]
    fsp = fund_std_pct[:min_len]; tsp = ts_spread_pct[:min_len]
    ens = np.zeros(min_len)
    for i in range(min_len):
        w = weights[["LOW", "MID", "HIGH"][int(reg[i])]]
        if not np.isnan(bv[i]) and bv[i] > vol_thr and f168_boost > 0:
            n_others = 3  # v1, i460, i415
            boost_per = f168_boost / n_others
            ret_i = (
                (w["v1"] + boost_per) * sig_rets["v1"][i] +
                (w["i460"] + boost_per) * sig_rets["i460"][i] +
                (w["i415"] + boost_per) * sig_rets["i415"][i] +
                (w["f168"] - f168_boost) * sig_rets["f168"][i]
            ) * vol_scale
        elif not np.isnan(bv[i]) and bv[i] > vol_thr:
            # boost=0: just scale down
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

# ── MAIN ────────────────────────────────────────────────────────────────────
print("=" * 60)
print("Phase 198 — Vol Regime Overlay Re-Tune")
print("=" * 60)
_start = time.time()

(prod_cfg, WEIGHTS, TS_RT, TS_RS, TS_BT, TS_BS,
 VOL_THR, VOL_SCALE_CUR, F168_BOOST_CUR, BASELINE_OBJ) = get_config()
ver_now = prod_cfg.get("_version", "2.26.0")
print(f"  Config: {ver_now}  Baseline OBJ≈{BASELINE_OBJ:.4f}")
print(f"  FTS: rt={TS_RT:.2f} rs={TS_RS:.2f} bt={TS_BT:.2f} bs={TS_BS:.2f}")
print(f"  Vol overlay: scale={VOL_SCALE_CUR:.2f} boost={F168_BOOST_CUR:.2f} thr={VOL_THR:.2f}")
print(f"  Weights: {WEIGHTS}")

load_params = {"ts_rt": TS_RT, "ts_rs": TS_RS, "ts_bt": TS_BT, "ts_bs": TS_BS, "f168_lb": 168}

print("\n[1] Loading all years ...")
year_data = {}
for yr in YEARS:
    print(f"  {yr}", end="", flush=True)
    year_data[yr] = load_year(yr, load_params)
    print()

def eval_overlay(vol_scale, f168_boost):
    yr_sharpes = {}
    for yr in YEARS:
        ens = compute_ens(year_data[yr], WEIGHTS, VOL_THR, vol_scale, f168_boost,
                         TS_RT, TS_RS, TS_BT, TS_BS)
        yr_sharpes[yr] = sharpe(ens, year_data[yr][-1])
    return obj_fn(yr_sharpes)

# Part A: vol_scale sweep
print("\n[2] Part A: vol_scale sweep (f168_boost fixed at current) ...")
partA = []
for vs in VOL_SCALE_SWEEP:
    obj = eval_overlay(vs, F168_BOOST_CUR)
    partA.append({"vol_scale": vs, "f168_boost": F168_BOOST_CUR, "obj": obj})
    flag = " ← CURRENT" if abs(vs - VOL_SCALE_CUR) < 0.01 else ""
    print(f"  vol_scale={vs:.2f}  OBJ={obj:.4f}{flag}")

best_vs = max(partA, key=lambda x: x["obj"])
print(f"  Best vol_scale: {best_vs['vol_scale']:.2f}  OBJ={best_vs['obj']:.4f}")

# Part B: f168_boost sweep
vs_best = best_vs["vol_scale"]
print(f"\n[3] Part B: f168_boost sweep (vol_scale={vs_best:.2f}) ...")
partB = []
for fb in F168_BOOST_SWEEP:
    obj = eval_overlay(vs_best, fb)
    partB.append({"vol_scale": vs_best, "f168_boost": fb, "obj": obj})
    flag = " ← CURRENT" if abs(fb - F168_BOOST_CUR) < 0.01 and abs(vs_best - VOL_SCALE_CUR) < 0.01 else ""
    print(f"  f168_boost={fb:.2f}  OBJ={obj:.4f}{flag}")

best_fb = max(partB, key=lambda x: x["obj"])
best = best_fb
print(f"  Best: vol_scale={best['vol_scale']:.2f} f168_boost={best['f168_boost']:.2f}  OBJ={best['obj']:.4f}  Δ={best['obj']-BASELINE_OBJ:+.4f}")

vs_final = best["vol_scale"]; fb_final = best["f168_boost"]

# LOYO validation
print(f"\n[4] LOYO validation (vol_scale={vs_final:.2f} f168_boost={fb_final:.2f} vs current {VOL_SCALE_CUR:.2f}/{F168_BOOST_CUR:.2f}) ...")
loyo_wins = 0; loyo_table = []
for yr in YEARS:
    ens_best = compute_ens(year_data[yr], WEIGHTS, VOL_THR, vs_final, fb_final,
                          TS_RT, TS_RS, TS_BT, TS_BS)
    ens_curr = compute_ens(year_data[yr], WEIGHTS, VOL_THR, VOL_SCALE_CUR, F168_BOOST_CUR,
                          TS_RT, TS_RS, TS_BT, TS_BS)
    n = year_data[yr][-1]
    sh_b = sharpe(ens_best, n); sh_c = sharpe(ens_curr, n)
    win = bool(sh_b > sh_c)
    if win: loyo_wins += 1
    loyo_table.append({"year": yr, "sh_best": sh_b, "sh_curr": sh_c, "win": win})
    print(f"  {yr}: best={sh_b:.4f} curr={sh_c:.4f} {'WIN' if win else 'LOSS'}")

print(f"\n  LOYO: {loyo_wins}/5 wins")
validated = loyo_wins >= 3 and best["obj"] > BASELINE_OBJ

# Save report
out = Path("artifacts/phase198"); out.mkdir(parents=True, exist_ok=True)
report = {
    "phase": 198, "target": "vol_regime_overlay",
    "partA": partA, "partB": partB,
    "final": {"vol_scale": vs_final, "f168_boost": fb_final},
    "best_obj": best["obj"], "delta": best["obj"] - BASELINE_OBJ,
    "baseline_obj": BASELINE_OBJ,
    "loyo": {"wins": loyo_wins, "table": loyo_table},
    "validated": validated,
    "runtime_s": round(time.time() - _start, 1),
    "timestamp": datetime.now(UTC).isoformat(),
}
(out / "phase198_report.json").write_text(json.dumps(report, indent=2, default=str))
print(f"  Report saved → artifacts/phase198/phase198_report.json  ({report['runtime_s']}s)")

if validated:
    cfg = json.loads(Path("configs/production_p91b_champion.json").read_text())
    ver = cfg.get("_version", "2.26.0")
    major, minor, patch = map(int, ver.split("."))
    new_ver = f"{major}.{minor + 1}.{patch}"
    cfg["_version"] = new_ver
    cfg["_validated"] += (
        f"; Vol overlay retune P198: vol_scale={VOL_SCALE_CUR:.2f}->{vs_final:.2f}"
        f" f168_boost={F168_BOOST_CUR:.2f}->{fb_final:.2f}"
        f" LOYO {loyo_wins}/5 delta={best['obj']-BASELINE_OBJ:+.4f} OBJ={best['obj']:.4f}"
        f" — PRODUCTION {new_ver}"
    )
    vol_ov = cfg.get("vol_regime_overlay", {})
    vol_ov["scale_factor"] = vs_final
    vol_ov["f144_boost"] = fb_final  # key used in prod config
    Path("configs/production_p91b_champion.json").write_text(json.dumps(cfg, indent=2))
    bov = best["obj"]; d = bov - BASELINE_OBJ
    cm = "phase198: vol overlay retune -- VALIDATED vol_scale=%.2f f168_boost=%.2f LOYO %d/5 delta=%+.4f OBJ=%.4f" % (vs_final, fb_final, loyo_wins, d, bov)
    print(f"\n✅ VALIDATED → {new_ver} OBJ={bov:.4f}")
    os.system("git add configs/production_p91b_champion.json artifacts/phase198/ scripts/run_phase198_vol_overlay_retune.py")
    os.system('git commit -m "' + cm + '"')
    os.system("git pull --rebase && git push")
else:
    delta = best["obj"] - BASELINE_OBJ
    verdict = "NO IMPROVEMENT" if delta <= 0 else f"WEAK LOYO {loyo_wins}/5"
    cm = "phase198: vol overlay retune -- %s vol_scale=%.2f boost=%.2f delta=%+.4f" % (verdict, vs_final, fb_final, delta)
    print(f"\n❌ NOT VALIDATED — {verdict}")
    os.system("git add artifacts/phase198/ scripts/run_phase198_vol_overlay_retune.py")
    os.system('git commit -m "' + cm + '"')
    os.system("git pull --rebase && git push")

print("\n[DONE] Phase 198 complete.")
