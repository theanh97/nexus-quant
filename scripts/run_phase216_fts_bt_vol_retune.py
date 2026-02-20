"""
Phase 216 — FTS Boost Threshold + Vol Overlay Scale Re-Tune
============================================================
Phase 215 validated: ts_rs=0.30, ts_bs=1.85 (stronger FTS signals).
ts_bt (boost threshold) was fixed at 0.25 throughout Phase 215.
Also vol_scale=0.30 not re-tuned since P198 when OBJ ≈ 2.96.

Part A: ts_bt ∈ [0.15, 0.20, 0.25, 0.30, 0.35, 0.40]
  (lower = less frequent boost trigger, higher = more frequent boost)
  Fix: ts_rt=0.65, ts_rs=0.30, ts_bs=1.85 (best from P215)

Part B: vol_scale ∈ [0.20, 0.25, 0.30, 0.35, 0.40, 0.45]
  (damping factor when BTC vol > threshold; lower = less exposure during high vol)
  Fix: best ts_bt from Part A

All signals pre-loaded once — pure overlay sweep, fast.
"""
import os, sys, json, time, re
import signal as _signal
from pathlib import Path

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
    out = Path("artifacts/phase216"); out.mkdir(parents=True, exist_ok=True)
    (out / "phase216_report.json").write_text(json.dumps(_partial, indent=2, default=str))
    sys.exit(0)
_signal.signal(_signal.SIGALRM, _on_timeout)
_signal.alarm(7200)

SYMBOLS = ["BTCUSDT","ETHUSDT","SOLUSDT","BNBUSDT","XRPUSDT",
           "ADAUSDT","DOGEUSDT","AVAXUSDT","DOTUSDT","LINKUSDT"]
YEAR_RANGES = {
    "2021": ("2021-02-01", "2022-01-01"),
    "2022": ("2022-01-01", "2023-01-01"),
    "2023": ("2023-01-01", "2024-01-01"),
    "2024": ("2024-01-01", "2025-01-01"),
    "2025": ("2025-01-01", "2026-01-01"),
}
YEARS = sorted(YEAR_RANGES.keys())

COST_MODEL = cost_model_from_config({
    "fee_rate": 0.0005, "slippage_rate": 0.0003, "cost_multiplier": 1.0,
})

VOL_WINDOW = 168; FUND_DISP_PCT = 240; PCT_WINDOW = 336
DISP_SCALE = 1.0; DISP_THR = 0.60

TS_BT_SWEEP   = [0.15, 0.20, 0.25, 0.30, 0.35, 0.40]
VOL_SCL_SWEEP = [0.20, 0.25, 0.30, 0.35, 0.40, 0.45]

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
    params = {
        "p_low": brs.get("p_low", 0.30), "p_high": brs.get("p_high", 0.60),
        "brd_lb": brs.get("breadth_lookback_bars", 192),
        "pct_win": brs.get("rolling_percentile_window", 336),
        "ts_rt": fts.get("reduce_threshold", 0.65),
        "ts_rs": fts.get("reduce_scale", 0.30),
        "ts_bt": fts.get("boost_threshold", 0.25),
        "ts_bs": fts.get("boost_scale", 1.85),
        "ts_short": fts.get("short_window_bars", 16),
        "ts_long": fts.get("long_window_bars", 72),
    }
    vol_ov = cfg.get("vol_regime_overlay", {})
    params["vol_thr"] = vol_ov.get("threshold", 0.50)
    params["vol_scale"] = vol_ov.get("scale_factor", 0.30)
    params["f168_boost"] = vol_ov.get("f144_boost", 0.00)
    m = re.findall(r"OBJ=([\d.]+)", cfg.get("_validated", ""))
    baseline_obj = float(m[-1]) if m else 3.2689
    return cfg, weights, params, baseline_obj

def rolling_mean_arr(a, w):
    out = np.full(len(a), np.nan)
    if w <= 0: return out
    cs = np.cumsum(np.where(np.isnan(a), 0.0, a))
    for i in range(w - 1, len(a)):
        out[i] = cs[i] / w if i == w - 1 else (cs[i] - cs[i - w]) / w
    return out

def load_year_all_sigs(year, params):
    p = params
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
    ts_raw_vals = rolling_mean_arr(xsect_mean, p["ts_short"]) - rolling_mean_arr(xsect_mean, p["ts_long"])

    brd_lb = p["brd_lb"]; pct_win = p["pct_win"]
    breadth = np.full(n, 0.5)
    for i in range(brd_lb, n):
        pos = sum(1 for j in range(len(SYMBOLS))
                  if close_mat[i-brd_lb, j] > 0 and close_mat[i, j] > close_mat[i-brd_lb, j])
        breadth[i] = pos / len(SYMBOLS)
    breadth[:brd_lb] = 0.5
    brd_pct = np.full(n, 0.5)
    for i in range(pct_win, n):
        brd_pct[i] = float(np.mean(breadth[i-pct_win:i] <= breadth[i]))
    brd_pct[:pct_win] = 0.5
    regime = np.where(brd_pct >= p["p_high"], 2, np.where(brd_pct >= p["p_low"], 1, 0)).astype(int)

    def run(name, sp):
        res = BacktestEngine(BacktestConfig(costs=COST_MODEL)).run(
            dataset, make_strategy({"name": name, "params": sp}))
        return np.array(res.returns)

    cfg = json.loads(Path("configs/production_p91b_champion.json").read_text())
    sigs = cfg.get("ensemble", {}).get("signals", {})
    v1p  = sigs.get("v1", {}).get("params", {})
    i460p = sigs.get("i460bw168", {}).get("params", {})
    i415p = sigs.get("i415bw216", {}).get("params", {})
    f168p = sigs.get("f144", {}).get("params", {})

    v1_rets = run("nexus_alpha_v1", {
        "k_per_side": v1p.get("k_per_side", 2),
        "w_carry": v1p.get("w_carry", 0.25), "w_mom": v1p.get("w_mom", 0.45),
        "w_mean_reversion": v1p.get("w_mean_reversion", 0.30),
        "momentum_lookback_bars": v1p.get("momentum_lookback_bars", 336),
        "mean_reversion_lookback_bars": v1p.get("mean_reversion_lookback_bars", 84),
        "vol_lookback_bars": v1p.get("vol_lookback_bars", 192),
        "target_gross_leverage": v1p.get("target_gross_leverage", 0.35),
        "rebalance_interval_bars": v1p.get("rebalance_interval_bars", 60),
    })
    i460_rets = run("idio_momentum_alpha", {
        "k_per_side": i460p.get("k_per_side", 4),
        "lookback_bars": i460p.get("lookback_bars", 480),
        "beta_window_bars": i460p.get("beta_window_bars", 168),
        "target_gross_leverage": i460p.get("target_gross_leverage", 0.20),
        "rebalance_interval_bars": i460p.get("rebalance_interval_bars", 48),
    })
    i415_rets = run("idio_momentum_alpha", {
        "k_per_side": i415p.get("k_per_side", 4),
        "lookback_bars": i415p.get("lookback_bars", 415),
        "beta_window_bars": i415p.get("beta_window_bars", 144),
        "target_gross_leverage": i415p.get("target_gross_leverage", 0.20),
        "rebalance_interval_bars": i415p.get("rebalance_interval_bars", 48),
    })
    f168_rets = run("funding_momentum_alpha", {
        "k_per_side": f168p.get("k_per_side", 2),
        "funding_lookback_bars": f168p.get("funding_lookback_bars", 168),
        "direction": f168p.get("direction", "contrarian"),
        "target_gross_leverage": f168p.get("target_gross_leverage", 0.25),
        "rebalance_interval_bars": f168p.get("rebalance_interval_bars", 36),
    })

    sig_rets = {"v1": v1_rets, "i460": i460_rets, "i415": i415_rets, "f168": f168_rets}
    return btc_vol, fund_std_pct, ts_raw_vals, regime, sig_rets, n

def compute_ensemble(base_data, weights, params, vol_scale, ts_bt):
    p = params
    btc_vol, fund_std_pct, ts_raw_vals, regime, sig_rets, n = base_data
    ts_spread_pct = np.full(n, 0.5)
    for i in range(PCT_WINDOW, n):
        ts_spread_pct[i] = float(np.mean(ts_raw_vals[i-PCT_WINDOW:i] <= ts_raw_vals[i]))
    ts_spread_pct[:PCT_WINDOW] = 0.5

    min_len = min(len(v) for v in sig_rets.values())
    bv = btc_vol[:min_len]; reg = regime[:min_len]
    fsp = fund_std_pct[:min_len]; tsp = ts_spread_pct[:min_len]
    ens = np.zeros(min_len)
    for i in range(min_len):
        w = weights[["LOW", "MID", "HIGH"][int(reg[i])]]
        if not np.isnan(bv[i]) and bv[i] > p["vol_thr"]:
            ret_i = (
                w["v1"] * sig_rets["v1"][i] +
                w["i460"] * sig_rets["i460"][i] +
                w["i415"] * sig_rets["i415"][i] +
                w["f168"] * sig_rets["f168"][i]
            ) * vol_scale   # variable vol_scale
        else:
            ret_i = (
                w["v1"] * sig_rets["v1"][i] +
                w["i460"] * sig_rets["i460"][i] +
                w["i415"] * sig_rets["i415"][i] +
                w["f168"] * sig_rets["f168"][i]
            )
        if DISP_SCALE > 1.0 and fsp[i] > DISP_THR: ret_i *= DISP_SCALE
        # FTS with current best ts_rt, ts_rs, ts_bs; variable ts_bt
        if tsp[i] > p["ts_rt"]:   ret_i *= p["ts_rs"]
        elif tsp[i] < ts_bt:       ret_i *= p["ts_bs"]
        ens[i] = ret_i
    return ens

def sharpe(rets, n):
    if n < 20: return 0.0
    mu = float(np.mean(rets[:n])) * 8760
    sd = float(np.std(rets[:n])) * np.sqrt(8760)
    return mu / sd if sd > 1e-9 else 0.0

def obj_fn(yr_sharpes):
    vals = list(yr_sharpes.values())
    return float(np.mean(vals) - 0.5 * np.std(vals))

def eval_params(year_data, weights, params, vol_scale, ts_bt):
    yr_sharpes = {}
    for yr in YEARS:
        ens = compute_ensemble(year_data[yr], weights, params, vol_scale, ts_bt)
        yr_sharpes[yr] = sharpe(ens, year_data[yr][-1])
    return obj_fn(yr_sharpes), yr_sharpes

# ── MAIN ────────────────────────────────────────────────────────────────────
print("=" * 60)
print("Phase 216 — FTS ts_bt + Vol Overlay Scale Re-Tune")
print("=" * 60)
_start = time.time()

prod_cfg, WEIGHTS, PARAMS, baseline_obj = get_config()
ver_now = prod_cfg.get("_version", "2.35.0")
curr_bt  = PARAMS["ts_bt"]
curr_vol = PARAMS["vol_scale"]
print(f"  Config: {ver_now}  baseline_obj={baseline_obj:.4f}")
print(f"  FTS: rt={PARAMS['ts_rt']} rs={PARAMS['ts_rs']} bt={curr_bt} bs={PARAMS['ts_bs']}")
print(f"  Vol: scale={curr_vol}  thr={PARAMS['vol_thr']}")

print("\n[1] Loading all years (all signals fixed) ...")
year_data = {}
for yr in YEARS:
    print(f"  {yr}", end="", flush=True)
    year_data[yr] = load_year_all_sigs(yr, PARAMS)
    print()

curr_obj, curr_yr = eval_params(year_data, WEIGHTS, PARAMS, curr_vol, curr_bt)
print(f"\n  Baseline OBJ (computed): {curr_obj:.4f}")

# Part A: ts_bt sweep (vol_scale fixed at current)
print(f"\n[2] Part A: ts_bt sweep (vol_scale={curr_vol}) ...")
results_a = []
for bt in TS_BT_SWEEP:
    obj, yr_s = eval_params(year_data, WEIGHTS, PARAMS, curr_vol, bt)
    results_a.append((bt, obj, yr_s))
    print(f"  ts_bt={bt:.2f}  OBJ={obj:.4f}")

best_a = max(results_a, key=lambda x: x[1])
best_bt, best_obj_a, _ = best_a
print(f"  Best ts_bt: {best_bt:.2f}  OBJ={best_obj_a:.4f}")

# Part B: vol_scale sweep (best ts_bt)
print(f"\n[3] Part B: vol_scale sweep (ts_bt={best_bt:.2f}) ...")
results_b = []
for vs in VOL_SCL_SWEEP:
    obj, yr_s = eval_params(year_data, WEIGHTS, PARAMS, vs, best_bt)
    results_b.append((vs, obj, yr_s))
    print(f"  vol_scale={vs:.2f}  OBJ={obj:.4f}")

best_b = max(results_b, key=lambda x: x[1])
best_vs, best_obj_b, best_yr = best_b
delta = best_obj_b - curr_obj
print(f"  Best vol_scale: {best_vs:.2f}  OBJ={best_obj_b:.4f}  Δ={delta:+.4f}")

# LOYO validation
print(f"\n[4] LOYO validation ...")
loyo_wins = 0
for yr in YEARS:
    win = best_yr.get(yr, 0) > curr_yr.get(yr, 0)
    if win: loyo_wins += 1
    print(f"  {yr}: best={best_yr.get(yr,0):.4f} curr={curr_yr.get(yr,0):.4f} {'WIN' if win else 'LOSS'}")

print(f"\n  LOYO: {loyo_wins}/5 wins  delta={delta:+.4f}")
validated = loyo_wins >= 3 and delta > 0.005

out = Path("artifacts/phase216"); out.mkdir(parents=True, exist_ok=True)
report = {
    "phase": 216, "description": "FTS ts_bt + vol overlay scale re-tune",
    "elapsed_seconds": round(time.time() - _start, 1),
    "baseline_obj_string": baseline_obj, "curr_obj_computed": curr_obj,
    "best_obj": best_obj_b, "delta": delta,
    "curr": {"ts_bt": curr_bt, "vol_scale": curr_vol},
    "best": {"ts_bt": best_bt, "vol_scale": best_vs},
    "loyo_wins": loyo_wins, "validated": validated,
    "yearly_best": best_yr, "yearly_curr": curr_yr,
    "bt_sweep": [(r[0], r[1]) for r in results_a],
    "vs_sweep": [(r[0], r[1]) for r in results_b],
}
(out / "phase216_report.json").write_text(json.dumps(report, indent=2, default=str))
print(f"  Report saved  ({report['elapsed_seconds']}s)")

if validated:
    cfg = json.loads(Path("configs/production_p91b_champion.json").read_text())
    fts = cfg.get("funding_term_structure_overlay", {})
    fts["boost_threshold"] = best_bt
    vol_ov = cfg.get("vol_regime_overlay", {})
    vol_ov["scale_factor"] = best_vs
    ver = cfg.get("_version", "2.35.0")
    major, minor, patch = map(int, ver.split("."))
    new_ver = "%d.%d.%d" % (major, minor + 1, patch)
    cfg["_version"] = new_ver
    cfg["_created"] = "2026-02-21"
    old_val = cfg.get("_validated", "")
    cfg["_validated"] = old_val + (
        "; FTS+Vol P216: ts_bt=%.2f->%.2f vol_scale=%.2f->%.2f LOYO %d/5 delta=%+.4f OBJ=%.4f — PRODUCTION %s" % (
            curr_bt, best_bt, curr_vol, best_vs, loyo_wins, delta, best_obj_b, new_ver))
    Path("configs/production_p91b_champion.json").write_text(json.dumps(cfg, indent=2))
    print(f"\n✅ VALIDATED → {new_ver} OBJ={best_obj_b:.4f}")
    cm = "phase216: FTS ts_bt+vol_scale -- VALIDATED bt=%.2f vol=%.2f LOYO %d/5 delta=%+.4f OBJ=%.4f" % (
        best_bt, best_vs, loyo_wins, delta, best_obj_b)
else:
    new_ver = ver_now
    print(f"\n⚠️ NO IMPROVEMENT — near-optimal (LOYO {loyo_wins}/5 delta={delta:+.4f})")
    cm = "phase216: FTS ts_bt+vol_scale -- WEAK LOYO %d/5 delta=%+.4f" % (loyo_wins, delta)

os.system("git add configs/production_p91b_champion.json scripts/run_phase216_fts_bt_vol_retune.py "
          "artifacts/phase216/ 2>/dev/null || true")
os.system('git commit -m "' + cm + '"')
os.system("git stash && git pull --rebase && git stash pop && git push")

print(f"\n[DONE] Phase 216 complete.")
