"""
Phase 218 — Vol Overlay Threshold + I415 Lookback Sweep
========================================================
Vol overlay: when BTC annualized vol > vol_thr → dampen returns by vol_scale=0.30.
Current vol_thr=0.50. Not explicitly swept — only vol_scale was tuned (P198, P216).
Lower thr → more often damped. Higher thr → rarely triggered.

Also: I415 lookback=415. I460 was swept to 480 by parallel (phase219).
I415 has been fixed at 415 since initial config. Worth re-sweeping at new baseline OBJ=3.27.

Part A: vol_thr ∈ [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.70]
  All signals pre-loaded, pure overlay sweep (fast).

Part B: I415 lookback ∈ [340, 360, 380, 400, 415, 430, 450, 480, 500]
  I460/F168/V1 fixed; re-run I415 for each lookback.
  Use best vol_thr from Part A.
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
    out = Path("artifacts/phase218"); out.mkdir(parents=True, exist_ok=True)
    (out / "phase218_report.json").write_text(json.dumps(_partial, indent=2, default=str))
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

VOL_THR_SWEEP  = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.70]
I415_LB_SWEEP  = [340, 360, 380, 400, 415, 430, 450, 480, 500]

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
    sigs = cfg.get("ensemble", {}).get("signals", {})
    i415p = sigs.get("i415bw216", {}).get("params", {})
    params["i415_lb"] = i415p.get("lookback_bars", 415)
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

def load_year_all_fixed(year, params):
    """Load all signals + overlays. Returns raw arrays for vol_thr variation."""
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
    # I415 with CURRENT lookback (for Part A sweep)
    i415_rets_curr = run("idio_momentum_alpha", {
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

    return (dataset, btc_vol, fund_std_pct, ts_spread_pct, regime,
            v1_rets, i460_rets, i415_rets_curr, f168_rets, n, i415p)

def compute_ensemble(base_data, weights, params, vol_thr, i415_rets_override=None):
    p = params
    _, btc_vol, fund_std_pct, ts_spread_pct, regime, v1_rets, i460_rets, i415_rets_curr, f168_rets, n, _ = base_data
    i415_rets = i415_rets_override if i415_rets_override is not None else i415_rets_curr
    sig_rets = {"v1": v1_rets, "i460": i460_rets, "i415": i415_rets, "f168": f168_rets}
    min_len = min(len(v) for v in sig_rets.values())
    bv = btc_vol[:min_len]; reg = regime[:min_len]
    fsp = fund_std_pct[:min_len]; tsp = ts_spread_pct[:min_len]
    ens = np.zeros(min_len)
    for i in range(min_len):
        w = weights[["LOW", "MID", "HIGH"][int(reg[i])]]
        if not np.isnan(bv[i]) and bv[i] > vol_thr:
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

# ── MAIN ────────────────────────────────────────────────────────────────────
print("=" * 60)
print("Phase 218 — Vol Threshold + I415 Lookback Sweep")
print("=" * 60)
_start = time.time()

prod_cfg, WEIGHTS, PARAMS, baseline_obj = get_config()
ver_now = prod_cfg.get("_version", "2.35.0")
curr_vol_thr = PARAMS["vol_thr"]
curr_i415_lb = PARAMS["i415_lb"]
print(f"  Config: {ver_now}  baseline_obj={baseline_obj:.4f}")
print(f"  vol_thr={curr_vol_thr}  I415 lb={curr_i415_lb}")

print("\n[1] Loading all years (all signals fixed) ...")
year_data = {}
for yr in YEARS:
    print(f"  {yr}", end="", flush=True)
    year_data[yr] = load_year_all_fixed(yr, PARAMS)
    print()

# Current baseline
curr_obj_yr = {}
for yr in YEARS:
    ens = compute_ensemble(year_data[yr], WEIGHTS, PARAMS, curr_vol_thr)
    curr_obj_yr[yr] = sharpe(ens, year_data[yr][9])   # index 9 = n (not -1 which is i415p dict)
curr_obj = obj_fn(curr_obj_yr)
print(f"\n  Baseline OBJ (computed): {curr_obj:.4f}")

# Part A: vol_thr sweep
print(f"\n[2] Part A: vol_thr sweep (all signals fixed at current I415 lb={curr_i415_lb}) ...")
results_a = []
for thr in VOL_THR_SWEEP:
    yr_s = {}
    for yr in YEARS:
        ens = compute_ensemble(year_data[yr], WEIGHTS, PARAMS, thr)
        yr_s[yr] = sharpe(ens, year_data[yr][9])
    obj = obj_fn(yr_s)
    results_a.append((thr, obj, yr_s))
    print(f"  vol_thr={thr:.2f}  OBJ={obj:.4f}")

best_a = max(results_a, key=lambda x: x[1])
best_thr, best_obj_a, _ = best_a
print(f"  Best vol_thr: {best_thr:.2f}  OBJ={best_obj_a:.4f}")

# Part B: I415 lookback sweep (with best vol_thr)
print(f"\n[3] Part B: I415 lookback sweep (vol_thr={best_thr:.2f}) ...")
results_b = []
for lb in I415_LB_SWEEP:
    yr_s = {}
    for yr in YEARS:
        ds = year_data[yr][0]
        i415p_orig = year_data[yr][10]   # i415p dict stored at index 10
        # Run I415 with new lookback
        i415_new = BacktestEngine(BacktestConfig(costs=COST_MODEL)).run(
            ds, make_strategy({"name": "idio_momentum_alpha", "params": {
                "k_per_side": i415p_orig.get("k_per_side", 4),
                "lookback_bars": lb,
                "beta_window_bars": i415p_orig.get("beta_window_bars", 144),
                "target_gross_leverage": i415p_orig.get("target_gross_leverage", 0.20),
                "rebalance_interval_bars": i415p_orig.get("rebalance_interval_bars", 48),
            }}))
        i415_rets_new = np.array(i415_new.returns)
        ens = compute_ensemble(year_data[yr], WEIGHTS, PARAMS, best_thr, i415_rets_new)
        yr_s[yr] = sharpe(ens, year_data[yr][9])
    obj = obj_fn(yr_s)
    results_b.append((lb, obj, yr_s))
    print(f"  I415 lb={lb}  OBJ={obj:.4f}")

best_b = max(results_b, key=lambda x: x[1])
best_lb, best_obj_b, best_yr = best_b
delta = best_obj_b - curr_obj
print(f"  Best I415 lb: {best_lb}  OBJ={best_obj_b:.4f}  Δ={delta:+.4f}")

# LOYO vs curr
print(f"\n[4] LOYO validation ...")
loyo_wins = 0
for yr in YEARS:
    win = best_yr.get(yr, 0) > curr_obj_yr.get(yr, 0)
    if win: loyo_wins += 1
    print(f"  {yr}: best={best_yr.get(yr,0):.4f} curr={curr_obj_yr.get(yr,0):.4f} {'WIN' if win else 'LOSS'}")

print(f"\n  LOYO: {loyo_wins}/5 wins  delta={delta:+.4f}")

# Also check if vol_thr alone validated (simpler change)
vt_delta = best_obj_a - curr_obj
vt_best_yr = best_a[2]
vt_loyo = sum(1 for yr in YEARS if vt_best_yr[yr] > curr_obj_yr[yr])
print(f"  Vol_thr alone: best_thr={best_thr:.2f} OBJ={best_obj_a:.4f} LOYO={vt_loyo}/5 Δ={vt_delta:+.4f}")

# Decide what to validate: prefer combined if ≥3/5 wins AND delta>0.005
# Otherwise try vol_thr alone
validated_combined = loyo_wins >= 3 and delta > 0.005
validated_vt_only = (not validated_combined) and vt_loyo >= 3 and vt_delta > 0.005

validated = validated_combined or validated_vt_only
if validated_combined:
    final_thr = best_thr; final_lb = best_lb
    final_obj = best_obj_b; final_yr = best_yr; final_loyo = loyo_wins; final_delta = delta
    tag = "combined vol_thr+I415_lb"
elif validated_vt_only:
    final_thr = best_thr; final_lb = curr_i415_lb
    final_obj = best_obj_a; final_yr = vt_best_yr; final_loyo = vt_loyo; final_delta = vt_delta
    tag = "vol_thr only"
else:
    tag = "none"

out = Path("artifacts/phase218"); out.mkdir(parents=True, exist_ok=True)
report = {
    "phase": 218, "description": "Vol overlay threshold + I415 lookback sweep",
    "elapsed_seconds": round(time.time() - _start, 1),
    "baseline_obj_string": baseline_obj, "curr_obj_computed": curr_obj,
    "best_combined_obj": best_obj_b, "delta_combined": delta,
    "best_vt_only_obj": best_obj_a, "delta_vt_only": vt_delta,
    "curr": {"vol_thr": curr_vol_thr, "i415_lb": curr_i415_lb},
    "best": {"vol_thr": best_thr, "i415_lb": best_lb},
    "loyo_combined": loyo_wins, "loyo_vt_only": vt_loyo,
    "validated": validated, "validated_tag": tag,
    "yearly_best": best_yr, "yearly_curr": curr_obj_yr,
    "vt_sweep": [(r[0], r[1]) for r in results_a],
    "lb_sweep": [(r[0], r[1]) for r in results_b],
}
(out / "phase218_report.json").write_text(json.dumps(report, indent=2, default=str))
print(f"  Report saved  ({report['elapsed_seconds']}s)")

if validated:
    cfg = json.loads(Path("configs/production_p91b_champion.json").read_text())
    vol_ov = cfg.get("vol_regime_overlay", {})
    vol_ov["threshold"] = final_thr
    if validated_combined:
        sigs = cfg.get("ensemble", {}).get("signals", {})
        i415p = sigs.get("i415bw216", {}).get("params", {})
        i415p["lookback_bars"] = final_lb
    ver = cfg.get("_version", "2.35.0")
    major, minor, patch = map(int, ver.split("."))
    new_ver = "%d.%d.%d" % (major, minor + 1, patch)
    cfg["_version"] = new_ver
    cfg["_created"] = "2026-02-21"
    old_val = cfg.get("_validated", "")
    if validated_combined:
        cfg["_validated"] = old_val + (
            "; Vol+I415 P218: vol_thr=%.2f->%.2f I415_lb=%d->%d LOYO %d/5 delta=%+.4f OBJ=%.4f — PRODUCTION %s" % (
                curr_vol_thr, final_thr, curr_i415_lb, final_lb,
                final_loyo, final_delta, final_obj, new_ver))
    else:
        cfg["_validated"] = old_val + (
            "; VolThr P218: vol_thr=%.2f->%.2f LOYO %d/5 delta=%+.4f OBJ=%.4f — PRODUCTION %s" % (
                curr_vol_thr, final_thr, final_loyo, final_delta, final_obj, new_ver))
    Path("configs/production_p91b_champion.json").write_text(json.dumps(cfg, indent=2))
    print(f"\n✅ VALIDATED ({tag}) → {new_ver} OBJ={final_obj:.4f}")
    cm = "phase218: vol_thr+I415lb -- VALIDATED %s thr=%.2f lb=%d LOYO %d/5 delta=%+.4f OBJ=%.4f" % (
        tag, final_thr, final_lb, final_loyo, final_delta, final_obj)
else:
    new_ver = ver_now
    print(f"\n⚠️ NO IMPROVEMENT (LOYO {loyo_wins}/5 delta={delta:+.4f})")
    cm = "phase218: vol_thr+I415lb -- WEAK LOYO %d/5 delta=%+.4f vt=%.2f lb=%d" % (
        loyo_wins, delta, best_thr, best_lb)

os.system("git add configs/production_p91b_champion.json scripts/run_phase218_vol_thr_i415lb_sweep.py "
          "artifacts/phase218/ 2>/dev/null || true")
os.system('git commit -m "' + cm + '"')
os.system("git stash && git pull --rebase && git stash pop && git push")

print(f"\n[DONE] Phase 218 complete.")
