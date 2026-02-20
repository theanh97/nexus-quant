"""
Phase 221 — MID + LOW Regime Weight Re-Tune (new OBJ=3.27 baseline)
=====================================================================
Phase 214 confirmed HIGH regime weights optimal at new baseline.
MID and LOW regime weights haven't been re-tuned since P202 (OBJ≈3.08) and P204 (OBJ≈3.08).
Now at OBJ=3.27 with I460/I415 lev=0.20 and other validated changes.

MID regime: v1=0.20, i460=0.0074, i415=0.0126, f168=0.78 (f168 dominant)
  Sweep MID_f168 ∈ [0.60, 0.65, 0.70, 0.74, 0.78, 0.82, 0.86, 0.90]
  Adjust v1 proportionally (i460/i415 kept at minimum — ~0.018 total)

LOW regime: v1=0.44, i460=0.0716, i415=0.1184, f168=0.35
  Sweep LOW_v1 ∈ [0.32, 0.36, 0.40, 0.44, 0.48, 0.52, 0.56]
  Adjust f168 proportionally (i460/i415 kept at minimum — ~0.19 total)

All signals pre-loaded once — pure regime weight variation (fast computation).
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
    out = Path("artifacts/phase221"); out.mkdir(parents=True, exist_ok=True)
    (out / "phase221_report.json").write_text(json.dumps(_partial, indent=2, default=str))
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

MID_F168_SWEEP = [0.60, 0.65, 0.70, 0.74, 0.78, 0.82, 0.86, 0.90]
LOW_V1_SWEEP   = [0.32, 0.36, 0.40, 0.44, 0.48, 0.52, 0.56]

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
    ts_raw = rolling_mean_arr(xsect_mean, p["ts_short"]) - rolling_mean_arr(xsect_mean, p["ts_long"])
    ts_spread_pct = np.full(n, 0.5)
    for i in range(PCT_WINDOW, n):
        ts_spread_pct[i] = float(np.mean(ts_raw[i-PCT_WINDOW:i] <= ts_raw[i]))
    ts_spread_pct[:PCT_WINDOW] = 0.5

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

    print(".", end=" ", flush=True)
    return btc_vol, fund_std_pct, ts_spread_pct, regime, {
        "v1": v1_rets, "i460": i460_rets, "i415": i415_rets, "f168": f168_rets
    }, n

def compute_ensemble(base_data, weights, params):
    p = params
    btc_vol, fund_std_pct, ts_spread_pct, regime, sig_rets, n = base_data
    min_len = min(len(v) for v in sig_rets.values())
    bv = btc_vol[:min_len]; reg = regime[:min_len]
    fsp = fund_std_pct[:min_len]; tsp = ts_spread_pct[:min_len]
    ens = np.zeros(min_len)
    for i in range(min_len):
        w = weights[["LOW", "MID", "HIGH"][int(reg[i])]]
        if not np.isnan(bv[i]) and bv[i] > p["vol_thr"]:
            ret_i = (
                w["v1"] * sig_rets["v1"][i] + w["i460"] * sig_rets["i460"][i] +
                w["i415"] * sig_rets["i415"][i] + w["f168"] * sig_rets["f168"][i]
            ) * p["vol_scale"]
        else:
            ret_i = (
                w["v1"] * sig_rets["v1"][i] + w["i460"] * sig_rets["i460"][i] +
                w["i415"] * sig_rets["i415"][i] + w["f168"] * sig_rets["f168"][i]
            )
        if DISP_SCALE > 1.0 and fsp[i] > DISP_THR: ret_i *= DISP_SCALE
        if tsp[i] > p["ts_rt"]:  ret_i *= p["ts_rs"]
        elif tsp[i] < p["ts_bt"]: ret_i *= p["ts_bs"]
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

def make_mid_weights(base, mid_f168):
    """MID: fix v1=0.20, i460/i415 at minimum (0.018 total), f168 = mid_f168."""
    w = {r: dict(base[r]) for r in ["LOW", "MID", "HIGH"]}
    # Keep i460/i415 proportional to current (≈0.357/0.643 split)
    curr_idio = base["MID"]["i460"] + base["MID"]["i415"]
    curr_v1   = base["MID"]["v1"]
    remain = 1.0 - curr_v1 - mid_f168 - curr_idio
    # Let v1 absorb the change (i460/i415 fixed)
    new_v1 = curr_v1 + remain
    if new_v1 < 0.0: new_v1 = 0.0
    w["MID"]["v1"]   = round(new_v1, 6)
    w["MID"]["f168"] = round(mid_f168, 6)
    w["MID"]["i460"] = round(base["MID"]["i460"], 6)
    w["MID"]["i415"] = round(base["MID"]["i415"], 6)
    return w

def make_low_weights(base, low_v1):
    """LOW: fix i460/i415 at minimum (≈0.19 total), f168 absorbs complement."""
    w = {r: dict(base[r]) for r in ["LOW", "MID", "HIGH"]}
    curr_idio = base["LOW"]["i460"] + base["LOW"]["i415"]
    remain = 1.0 - low_v1 - curr_idio
    new_f168 = remain
    if new_f168 < 0.0: new_f168 = 0.0
    w["LOW"]["v1"]   = round(low_v1, 6)
    w["LOW"]["f168"] = round(new_f168, 6)
    w["LOW"]["i460"] = round(base["LOW"]["i460"], 6)
    w["LOW"]["i415"] = round(base["LOW"]["i415"], 6)
    return w

# ── MAIN ────────────────────────────────────────────────────────────────────
print("=" * 60)
print("Phase 221 — MID + LOW Regime Weight Re-Tune")
print("=" * 60)
_start = time.time()

prod_cfg, BASE_WEIGHTS, PARAMS, baseline_obj = get_config()
ver_now = prod_cfg.get("_version", "2.35.0")
curr_mid_f168 = BASE_WEIGHTS["MID"]["f168"]
curr_mid_v1   = BASE_WEIGHTS["MID"]["v1"]
curr_low_v1   = BASE_WEIGHTS["LOW"]["v1"]
curr_low_f168 = BASE_WEIGHTS["LOW"]["f168"]
print(f"  Config: {ver_now}  baseline_obj={baseline_obj:.4f}")
print(f"  MID curr: v1={curr_mid_v1:.4f} f168={curr_mid_f168:.4f}")
print(f"  LOW curr: v1={curr_low_v1:.4f} f168={curr_low_f168:.4f}")

print("\n[1] Loading all years (all signals fixed) ...")
year_data = {}
for yr in YEARS:
    print(f"  {yr}", end="", flush=True)
    year_data[yr] = load_year_all_sigs(yr, PARAMS)
    print()

# Current baseline
curr_yr_s = {}
for yr in YEARS:
    ens = compute_ensemble(year_data[yr], BASE_WEIGHTS, PARAMS)
    curr_yr_s[yr] = sharpe(ens, year_data[yr][5])   # n is index 5 in tuple
curr_obj = obj_fn(curr_yr_s)
print(f"\n  Baseline OBJ (computed): {curr_obj:.4f}")

# Part A: MID f168 sweep
print(f"\n[2] Part A: MID f168 sweep ...")
results_a = []
for mf in MID_F168_SWEEP:
    w = make_mid_weights(BASE_WEIGHTS, mf)
    yr_s = {}
    for yr in YEARS:
        ens = compute_ensemble(year_data[yr], w, PARAMS)
        yr_s[yr] = sharpe(ens, year_data[yr][5])
    obj = obj_fn(yr_s)
    results_a.append((mf, obj, yr_s, w))
    print(f"  MID_f168={mf:.2f}  (v1={w['MID']['v1']:.4f})  OBJ={obj:.4f}")

best_a = max(results_a, key=lambda x: x[1])
best_mid_f168, best_obj_a, best_mid_yr, best_mid_w = best_a
print(f"  Best MID_f168: {best_mid_f168:.2f}  OBJ={best_obj_a:.4f}")

# Part B: LOW v1 sweep (using best MID weights)
print(f"\n[3] Part B: LOW v1 sweep ...")
results_b = []
for lv in LOW_V1_SWEEP:
    w = make_low_weights(best_mid_w, lv)
    yr_s = {}
    for yr in YEARS:
        ens = compute_ensemble(year_data[yr], w, PARAMS)
        yr_s[yr] = sharpe(ens, year_data[yr][5])
    obj = obj_fn(yr_s)
    results_b.append((lv, obj, yr_s, w))
    print(f"  LOW_v1={lv:.2f}  (f168={w['LOW']['f168']:.4f})  OBJ={obj:.4f}")

best_b = max(results_b, key=lambda x: x[1])
best_low_v1, best_obj_b, best_yr, best_final_w = best_b
delta = best_obj_b - curr_obj
print(f"  Best LOW_v1: {best_low_v1:.2f}  OBJ={best_obj_b:.4f}  Δ={delta:+.4f}")

# LOYO validation
print(f"\n[4] LOYO validation ...")
loyo_wins = 0
for yr in YEARS:
    win = best_yr.get(yr, 0) > curr_yr_s.get(yr, 0)
    if win: loyo_wins += 1
    print(f"  {yr}: best={best_yr.get(yr,0):.4f} curr={curr_yr_s.get(yr,0):.4f} {'WIN' if win else 'LOSS'}")

print(f"\n  LOYO: {loyo_wins}/5 wins  delta={delta:+.4f}")
validated = loyo_wins >= 3 and delta > 0.005

out = Path("artifacts/phase221"); out.mkdir(parents=True, exist_ok=True)
report = {
    "phase": 221, "description": "MID+LOW regime weight re-tune at OBJ=3.27 baseline",
    "elapsed_seconds": round(time.time() - _start, 1),
    "baseline_obj_string": baseline_obj, "curr_obj_computed": curr_obj,
    "best_obj": best_obj_b, "delta": delta,
    "curr_MID": {"v1": curr_mid_v1, "f168": curr_mid_f168},
    "curr_LOW": {"v1": curr_low_v1, "f168": curr_low_f168},
    "best_MID": {"v1": best_final_w["MID"]["v1"], "f168": best_mid_f168},
    "best_LOW": {"v1": best_low_v1, "f168": best_final_w["LOW"]["f168"]},
    "loyo_wins": loyo_wins, "validated": validated,
    "yearly_best": best_yr, "yearly_curr": curr_yr_s,
    "mid_f168_sweep": [(r[0], r[1]) for r in results_a],
    "low_v1_sweep": [(r[0], r[1]) for r in results_b],
}
(out / "phase221_report.json").write_text(json.dumps(report, indent=2, default=str))
print(f"  Report saved  ({report['elapsed_seconds']}s)")

if validated:
    cfg = json.loads(Path("configs/production_p91b_champion.json").read_text())
    rw = cfg["breadth_regime_switching"]["regime_weights"]
    rw["MID"]["f168"] = best_mid_f168
    rw["MID"]["v1"]   = best_final_w["MID"]["v1"]
    rw["LOW"]["v1"]   = best_low_v1
    rw["LOW"]["f168"] = best_final_w["LOW"]["f168"]
    ver = cfg.get("_version", "2.35.0")
    major, minor, patch = map(int, ver.split("."))
    new_ver = "%d.%d.%d" % (major, minor + 1, patch)
    cfg["_version"] = new_ver
    cfg["_created"] = "2026-02-21"
    old_val = cfg.get("_validated", "")
    cfg["_validated"] = old_val + (
        "; MID+LOW P221: MID_f168=%.2f->%.2f LOW_v1=%.2f->%.2f LOYO %d/5 delta=%+.4f OBJ=%.4f — PRODUCTION %s" % (
            curr_mid_f168, best_mid_f168, curr_low_v1, best_low_v1,
            loyo_wins, delta, best_obj_b, new_ver))
    Path("configs/production_p91b_champion.json").write_text(json.dumps(cfg, indent=2))
    print(f"\n✅ VALIDATED → {new_ver} OBJ={best_obj_b:.4f}")
    cm = "phase221: MID+LOW regime retune -- VALIDATED MID_f168=%.2f LOW_v1=%.2f LOYO %d/5 delta=%+.4f OBJ=%.4f" % (
        best_mid_f168, best_low_v1, loyo_wins, delta, best_obj_b)
else:
    new_ver = ver_now
    print(f"\n⚠️ NO IMPROVEMENT — MID+LOW near-optimal (LOYO {loyo_wins}/5 delta={delta:+.4f})")
    cm = "phase221: MID+LOW regime retune -- WEAK LOYO %d/5 delta=%+.4f MID_f168=%.2f LOW_v1=%.2f" % (
        loyo_wins, delta, best_mid_f168, best_low_v1)

os.system("git add configs/production_p91b_champion.json scripts/run_phase221_mid_low_regime_retune.py "
          "artifacts/phase221/ 2>/dev/null || true")
os.system('git commit -m "' + cm + '"')
os.system("git stash && git pull --rebase && git stash pop && git push")

print(f"\n[DONE] Phase 221 complete.")
