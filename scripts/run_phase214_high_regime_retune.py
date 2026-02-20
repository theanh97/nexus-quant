"""
Phase 214 — HIGH Regime Weight Re-Tune (post I460/I415 lev=0.20)
==================================================================
After Phase 213b validated I460/I415 lev=0.30->0.20 (OBJ 3.0849->3.1603, LOYO 5/5),
the effective net contribution of idio momentum in HIGH regime changed.
HIGH regime: v1=0.04, i460=0.2785, i415=0.5015, f168=0.18 (idio=0.78 combined)

With lower lev (0.20 vs 0.30), idio signals generate less return per bar.
The optimal regime weight balance may shift — possibly needing higher idio weight
to compensate, or different v1/f168 balance.

Part A: HIGH idio_total ∈ [0.55, 0.60, 0.65, 0.70, 0.75, 0.78, 0.82, 0.86, 0.90]
  v1=0.04 fixed, f168 = 1 - v1 - idio_total, i460/i415 split proportionally (0.357/0.643)

Part B: HIGH f168 weight sweep ∈ [0.08, 0.12, 0.15, 0.18, 0.22, 0.26, 0.30]
  (using best idio_total from Part A, adjust v1 complementarily)
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
    out = Path("artifacts/phase214"); out.mkdir(parents=True, exist_ok=True)
    (out / "phase214_report.json").write_text(json.dumps(_partial, indent=2, default=str))
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

HIGH_IDIO_SWEEP  = [0.55, 0.60, 0.65, 0.70, 0.75, 0.78, 0.82, 0.86, 0.90]
HIGH_F168_SWEEP  = [0.08, 0.12, 0.15, 0.18, 0.22, 0.26, 0.30]
HIGH_IDIO_RATIO  = (0.278529 / 0.78, 0.501471 / 0.78)   # i460 : i415 split

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
        "ts_rt": fts.get("reduce_threshold", 0.65), "ts_rs": fts.get("reduce_scale", 0.45),
        "ts_bt": fts.get("boost_threshold", 0.25), "ts_bs": fts.get("boost_scale", 1.70),
        "ts_short": fts.get("short_window_bars", 16), "ts_long": fts.get("long_window_bars", 72),
    }
    vol_ov = cfg.get("vol_regime_overlay", {})
    params["vol_thr"] = vol_ov.get("threshold", 0.50)
    params["vol_scale"] = vol_ov.get("scale_factor", 0.30)
    params["f168_boost"] = vol_ov.get("f144_boost", 0.00)
    m = re.findall(r"OBJ=([\d.]+)", cfg.get("_validated", ""))
    baseline_obj = float(m[-1]) if m else 3.1603
    return cfg, weights, params, baseline_obj

def rolling_mean_arr(a, w):
    out = np.full(len(a), np.nan)
    if w <= 0: return out
    cs = np.cumsum(np.where(np.isnan(a), 0.0, a))
    for i in range(w - 1, len(a)):
        out[i] = cs[i] / w if i == w - 1 else (cs[i] - cs[i - w]) / w
    return out

def load_year_all_fixed(year, params):
    """Load all 4 signals fixed + overlays for year."""
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

    # Read signal params from config
    cfg = json.loads(Path("configs/production_p91b_champion.json").read_text())
    sigs = cfg.get("ensemble", {}).get("signals", {})
    v1p = sigs.get("v1", {}).get("params", {})
    i460p = sigs.get("i460bw168", {}).get("params", {})
    i415p = sigs.get("i415bw216", {}).get("params", {})
    f168p = sigs.get("f144", {}).get("params", {})

    v1_rets   = run("nexus_alpha_v1", {
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
    sig_rets = {"v1": v1_rets, "i460": i460_rets, "i415": i415_rets, "f168": f168_rets}
    return btc_vol, fund_std_pct, ts_spread_pct, regime, sig_rets, n

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
            fb = p["f168_boost"]
            if fb > 0:
                bpp = fb / 3.0
                ret_i = (
                    (w["v1"] + bpp) * sig_rets["v1"][i] +
                    (w["i460"] + bpp) * sig_rets["i460"][i] +
                    (w["i415"] + bpp) * sig_rets["i415"][i] +
                    (w["f168"] - fb) * sig_rets["f168"][i]
                ) * p["vol_scale"]
            else:
                ret_i = (
                    w["v1"] * sig_rets["v1"][i] +
                    w["i460"] * sig_rets["i460"][i] +
                    w["i415"] * sig_rets["i415"][i] +
                    w["f168"] * sig_rets["f168"][i]
                ) * p["vol_scale"]
        else:
            ret_i = (
                w["v1"] * sig_rets["v1"][i] +
                w["i460"] * sig_rets["i460"][i] +
                w["i415"] * sig_rets["i415"][i] +
                w["f168"] * sig_rets["f168"][i]
            )
        if DISP_SCALE > 1.0 and fsp[i] > DISP_THR: ret_i *= DISP_SCALE
        if tsp[i] > p["ts_rt"]: ret_i *= p["ts_rs"]
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

def make_high_weights(base_weights, high_idio_total, high_f168):
    """Construct weights with new HIGH idio total and f168."""
    w = {r: dict(base_weights[r]) for r in ["LOW", "MID", "HIGH"]}
    high_v1 = max(0.0, 1.0 - high_idio_total - high_f168)
    i460_frac, i415_frac = HIGH_IDIO_RATIO
    w["HIGH"] = {
        "v1":   round(high_v1, 6),
        "i460": round(high_idio_total * i460_frac, 6),
        "i415": round(high_idio_total * i415_frac, 6),
        "f168": round(high_f168, 6),
    }
    return w

# ── MAIN ────────────────────────────────────────────────────────────────────
print("=" * 60)
print("Phase 214 — HIGH Regime Weight Re-Tune (post lev=0.20)")
print("=" * 60)
_start = time.time()

prod_cfg, BASE_WEIGHTS, PARAMS, baseline_obj = get_config()
ver_now = prod_cfg.get("_version", "2.34.0")
curr_high = BASE_WEIGHTS["HIGH"]
curr_idio_total = curr_high["i460"] + curr_high["i415"]
curr_f168 = curr_high["f168"]
curr_v1   = curr_high["v1"]
print(f"  Config: {ver_now}  baseline_obj={baseline_obj:.4f}")
print(f"  HIGH curr: v1={curr_v1:.4f} idio={curr_idio_total:.4f} f168={curr_f168:.4f}")

print("\n[1] Loading all years (all signals fixed) ...")
year_data = {}
for yr in YEARS:
    print(f"  {yr}", end="", flush=True)
    year_data[yr] = load_year_all_fixed(yr, PARAMS)
    print()

# Part A: HIGH idio_total sweep (f168 fixed at current)
print(f"\n[2] Part A: HIGH idio_total sweep (f168={curr_f168:.2f} fixed) ...")
results_a = []
for idio_t in HIGH_IDIO_SWEEP:
    if idio_t + curr_f168 > 0.98: continue  # leave at least 2% for v1
    w = make_high_weights(BASE_WEIGHTS, idio_t, curr_f168)
    yr_sharpes = {}
    for yr in YEARS:
        ens = compute_ensemble(year_data[yr], w, PARAMS)
        yr_sharpes[yr] = sharpe(ens, year_data[yr][-1])
    obj = obj_fn(yr_sharpes)
    results_a.append((idio_t, obj, yr_sharpes))
    print(f"  HIGH_idio={idio_t:.2f}  (v1={w['HIGH']['v1']:.3f} i460={w['HIGH']['i460']:.4f} i415={w['HIGH']['i415']:.4f} f168={curr_f168:.2f})  OBJ={obj:.4f}")

best_a = max(results_a, key=lambda x: x[1])
best_idio_total, best_obj_a, _ = best_a
print(f"  Best idio_total: {best_idio_total:.2f}  OBJ={best_obj_a:.4f}")

# Part B: HIGH f168 sweep (best idio_total)
print(f"\n[3] Part B: HIGH f168 sweep (idio_total={best_idio_total:.2f}) ...")
results_b = []
for f168v in HIGH_F168_SWEEP:
    if best_idio_total + f168v > 0.98: continue
    w = make_high_weights(BASE_WEIGHTS, best_idio_total, f168v)
    yr_sharpes = {}
    for yr in YEARS:
        ens = compute_ensemble(year_data[yr], w, PARAMS)
        yr_sharpes[yr] = sharpe(ens, year_data[yr][-1])
    obj = obj_fn(yr_sharpes)
    results_b.append((f168v, obj, yr_sharpes))
    print(f"  HIGH_f168={f168v:.2f}  (v1={w['HIGH']['v1']:.3f})  OBJ={obj:.4f}")

best_b = max(results_b, key=lambda x: x[1])
best_f168, best_obj_b, best_yr = best_b
delta = best_obj_b - baseline_obj
print(f"  Best f168: {best_f168:.2f}  OBJ={best_obj_b:.4f}  Δ={delta:+.4f}")

# LOYO: compare best vs current weights (pure weight change, no signal rerun)
curr_yr = {}
for yr in YEARS:
    ens = compute_ensemble(year_data[yr], BASE_WEIGHTS, PARAMS)
    curr_yr[yr] = sharpe(ens, year_data[yr][-1])

print(f"\n[4] LOYO validation ...")
loyo_wins = 0
for yr in YEARS:
    win = best_yr.get(yr, 0) > curr_yr.get(yr, 0)
    if win: loyo_wins += 1
    print(f"  {yr}: best={best_yr.get(yr,0):.4f} curr={curr_yr.get(yr,0):.4f} {'WIN' if win else 'LOSS'}")

print(f"\n  LOYO: {loyo_wins}/5 wins  delta={delta:+.4f}")
validated = loyo_wins >= 3 and delta > 0.005

out = Path("artifacts/phase214"); out.mkdir(parents=True, exist_ok=True)
best_w = make_high_weights(BASE_WEIGHTS, best_idio_total, best_f168)
report = {
    "phase": 214, "description": "HIGH regime weight re-tune post lev=0.20",
    "elapsed_seconds": round(time.time() - _start, 1),
    "baseline_obj": baseline_obj, "best_obj": best_obj_b, "delta": delta,
    "curr_HIGH": {"v1": curr_v1, "idio_total": curr_idio_total, "f168": curr_f168},
    "best_HIGH": {"v1": best_w["HIGH"]["v1"], "i460": best_w["HIGH"]["i460"],
                  "i415": best_w["HIGH"]["i415"], "f168": best_f168},
    "loyo_wins": loyo_wins, "validated": validated,
    "yearly_best": best_yr, "yearly_curr": curr_yr,
    "idio_sweep": [(r[0], r[1]) for r in results_a],
    "f168_sweep": [(r[0], r[1]) for r in results_b],
}
(out / "phase214_report.json").write_text(json.dumps(report, indent=2, default=str))
print(f"  Report saved  ({report['elapsed_seconds']}s)")

if validated:
    cfg = json.loads(Path("configs/production_p91b_champion.json").read_text())
    rw = cfg["breadth_regime_switching"]["regime_weights"]
    best_hi_w = best_w["HIGH"]
    rw["HIGH"]["v1"]          = best_hi_w["v1"]
    rw["HIGH"]["i460bw168"]   = best_hi_w["i460"]
    rw["HIGH"]["i415bw216"]   = best_hi_w["i415"]
    rw["HIGH"]["f168"]        = best_f168
    ver = cfg.get("_version", "2.34.0")
    major, minor, patch = map(int, ver.split("."))
    new_ver = "%d.%d.%d" % (major, minor + 1, patch)
    cfg["_version"] = new_ver
    cfg["_created"] = "2026-02-21"
    old_val = cfg.get("_validated", "")
    cfg["_validated"] = old_val + (
        "; HIGH regime re-tune P214 lev-0.20: idio=%.4f->%.4f f168=%.2f->%.2f LOYO %d/5 delta=%+.4f OBJ=%.4f — PRODUCTION %s" % (
            curr_idio_total, best_idio_total, curr_f168, best_f168,
            loyo_wins, delta, best_obj_b, new_ver))
    Path("configs/production_p91b_champion.json").write_text(json.dumps(cfg, indent=2))
    print(f"\n✅ VALIDATED → {new_ver} OBJ={best_obj_b:.4f}")
    cm = "phase214: HIGH regime weight re-tune -- VALIDATED idio=%.2f f168=%.2f LOYO %d/5 delta=%+.4f OBJ=%.4f" % (
        best_idio_total, best_f168, loyo_wins, delta, best_obj_b)
else:
    new_ver = ver_now
    print(f"\n⚠️ NO IMPROVEMENT — HIGH regime near-optimal (LOYO {loyo_wins}/5 delta={delta:+.4f})")
    cm = "phase214: HIGH regime weight re-tune -- WEAK LOYO %d/5 delta=%+.4f" % (loyo_wins, delta)

os.system("git add configs/production_p91b_champion.json scripts/run_phase214_high_regime_retune.py "
          "artifacts/phase214/ 2>/dev/null || true")
os.system('git commit -m "' + cm + '"')
os.system("git stash && git pull --rebase && git stash pop && git push")

print(f"\n[DONE] Phase 214 complete.")
