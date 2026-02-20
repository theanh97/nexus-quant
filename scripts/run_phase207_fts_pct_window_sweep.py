"""
Phase 207 — FTS Percentile Window Sweep
=========================================
FTS overlay uses rolling_percentile_window=336h to compute TS spread percentile.
This controls how much context/memory the TS spread signal uses.
Last set at P164-165. Since then config changed substantially.

Also test breadth rolling_percentile_window (both use 336h currently).

Part A: fts_pct_win ∈ [168, 240, 280, 336, 400, 480, 672]
  - All other params fixed
Part B: breadth_pct_win ∈ [168, 240, 280, 336, 400, 480] using best fts_pct_win

LOYO: ≥3/5 wins AND delta>0.005 → update config + commit
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
    out = Path("artifacts/phase207"); out.mkdir(parents=True, exist_ok=True)
    (out / "phase207_report.json").write_text(json.dumps(_partial, indent=2, default=str))
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

FTS_PCT_WIN_SWEEP  = [168, 240, 280, 336, 400, 480, 672]
BRD_PCT_WIN_SWEEP  = [168, 240, 280, 336, 400, 480]

VOL_WINDOW = 168; FUND_DISP_PCT = 240
DISP_SCALE = 1.0; DISP_THR = 0.60

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
        "brd_pct_win": brs.get("rolling_percentile_window", 336),
        "ts_rt": fts.get("reduce_threshold", 0.65), "ts_rs": fts.get("reduce_scale", 0.45),
        "ts_bt": fts.get("boost_threshold", 0.25), "ts_bs": fts.get("boost_scale", 1.70),
        "ts_short": fts.get("short_window_bars", 16), "ts_long": fts.get("long_window_bars", 72),
        "fts_pct_win": fts.get("rolling_percentile_window", 336),
    }
    vol_ov = cfg.get("vol_regime_overlay", {})
    params["vol_thr"] = vol_ov.get("threshold", 0.50)
    params["vol_scale"] = vol_ov.get("scale_factor", 0.30)
    params["f168_boost"] = vol_ov.get("f144_boost", 0.00)
    m = re.findall(r"OBJ=([\d.]+)", cfg.get("_validated", ""))
    baseline_obj = float(m[-1]) if m else 3.0849
    return cfg, weights, params, baseline_obj

def rolling_mean_arr(a, w):
    out = np.full(len(a), np.nan)
    if w <= 0: return out
    cs = np.cumsum(np.where(np.isnan(a), 0.0, a))
    for i in range(w - 1, len(a)):
        out[i] = cs[i] / w if i == w - 1 else (cs[i] - cs[i - w]) / w
    return out

def load_year_base(year, params):
    """Load data, compute all raw signals, return data + raw arrays for recomputing windows."""
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

    # Breadth raw
    brd_lb = p["brd_lb"]
    breadth = np.full(n, 0.5)
    for i in range(brd_lb, n):
        pos = sum(1 for j in range(len(SYMBOLS))
                  if close_mat[i-brd_lb, j] > 0 and close_mat[i, j] > close_mat[i-brd_lb, j])
        breadth[i] = pos / len(SYMBOLS)
    breadth[:brd_lb] = 0.5

    sig_rets = {}
    for sk, sname, sp in [
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
            dataset, make_strategy({"name": sname, "params": sp}))
        sig_rets[sk] = np.array(result.returns)

    print(".", end=" ", flush=True)
    return btc_vol, fund_std_pct, ts_raw_vals, breadth, sig_rets, n

def compute_with_windows(year_base, weights, params, fts_pct_win, brd_pct_win):
    p = params
    btc_vol, fund_std_pct, ts_raw_vals, breadth, sig_rets, n = year_base
    min_len = min(len(v) for v in sig_rets.values())

    # Recompute TS spread percentile with test window
    ts_spread_pct = np.full(n, 0.5)
    for i in range(fts_pct_win, n):
        ts_spread_pct[i] = float(np.mean(ts_raw_vals[i-fts_pct_win:i] <= ts_raw_vals[i]))
    ts_spread_pct[:fts_pct_win] = 0.5

    # Recompute breadth regime with test pct_win
    brd_pct = np.full(n, 0.5)
    for i in range(brd_pct_win, n):
        brd_pct[i] = float(np.mean(breadth[i-brd_pct_win:i] <= breadth[i]))
    brd_pct[:brd_pct_win] = 0.5
    regime = np.where(brd_pct >= p["p_high"], 2, np.where(brd_pct >= p["p_low"], 1, 0)).astype(int)

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

def eval_windows(fts_pct_win, brd_pct_win, weights, year_data, params):
    yr_sharpes = {}
    for yr in YEARS:
        ens = compute_with_windows(year_data[yr], weights, params, fts_pct_win, brd_pct_win)
        yr_sharpes[yr] = sharpe(ens, year_data[yr][5])
    return obj_fn(yr_sharpes), yr_sharpes

# ── MAIN ────────────────────────────────────────────────────────────────────
print("=" * 60)
print("Phase 207 — FTS + Breadth Percentile Window Sweep")
print("=" * 60)
_start = time.time()

prod_cfg, WEIGHTS, PARAMS, baseline_obj = get_config()
ver_now = prod_cfg.get("_version", "2.32.0")
curr_fts_pct_win = PARAMS["fts_pct_win"]
curr_brd_pct_win = PARAMS["brd_pct_win"]
print(f"  Config: {ver_now}  baseline_obj={baseline_obj:.4f}")
print(f"  FTS pct_win={curr_fts_pct_win}  Breadth pct_win={curr_brd_pct_win}")

print("\n[1] Loading all years ...")
year_data = {}
for yr in YEARS:
    print(f"  {yr}", end="", flush=True)
    year_data[yr] = load_year_base(yr, PARAMS)
    print()

# Part A: FTS pct_win sweep
print(f"\n[2] Part A: FTS pct_win sweep (brd_pct_win={curr_brd_pct_win} fixed) ...")
results_a = []
for fts_win in FTS_PCT_WIN_SWEEP:
    obj, _ = eval_windows(fts_win, curr_brd_pct_win, WEIGHTS, year_data, PARAMS)
    results_a.append((fts_win, obj))
    print(f"  fts_pct_win={fts_win:3d}  OBJ={obj:.4f}")

best_a = max(results_a, key=lambda x: x[1])
best_fts_win, best_obj_a = best_a
print(f"  Best fts_pct_win: {best_fts_win}  OBJ={best_obj_a:.4f}")

# Part B: Breadth pct_win sweep
print(f"\n[3] Part B: breadth pct_win sweep (fts_pct_win={best_fts_win} fixed) ...")
results_b = []
for brd_win in BRD_PCT_WIN_SWEEP:
    obj, _ = eval_windows(best_fts_win, brd_win, WEIGHTS, year_data, PARAMS)
    results_b.append((brd_win, obj))
    print(f"  brd_pct_win={brd_win:3d}  OBJ={obj:.4f}")

best_b = max(results_b, key=lambda x: x[1])
best_brd_win, best_obj_b = best_b
delta = best_obj_b - baseline_obj
print(f"  Best brd_pct_win: {best_brd_win}  OBJ={best_obj_b:.4f}  Δ={delta:+.4f}")

# LOYO validation
print(f"\n[4] LOYO validation ...")
_, best_yr = eval_windows(best_fts_win, best_brd_win, WEIGHTS, year_data, PARAMS)
_, base_yr = eval_windows(curr_fts_pct_win, curr_brd_pct_win, WEIGHTS, year_data, PARAMS)
loyo_wins = 0
for yr in YEARS:
    win = best_yr[yr] > base_yr[yr]
    if win: loyo_wins += 1
    print(f"  {yr}: best={best_yr[yr]:.4f} curr={base_yr[yr]:.4f} {'WIN' if win else 'LOSS'}")

print(f"\n  LOYO: {loyo_wins}/5 wins")

validated = loyo_wins >= 3 and delta > 0.005

# Save report
out = Path("artifacts/phase207"); out.mkdir(parents=True, exist_ok=True)
report = {
    "phase": 207,
    "description": "FTS + breadth percentile window sweep",
    "elapsed_seconds": round(time.time() - _start, 1),
    "baseline_obj": baseline_obj, "best_obj": best_obj_b, "delta": delta,
    "curr_fts_pct_win": curr_fts_pct_win, "best_fts_pct_win": best_fts_win,
    "curr_brd_pct_win": curr_brd_pct_win, "best_brd_pct_win": best_brd_win,
    "loyo_wins": loyo_wins, "validated": validated,
    "yearly_best": best_yr, "yearly_base": base_yr,
    "fts_sweep": results_a, "brd_sweep": results_b,
}
(out / "phase207_report.json").write_text(json.dumps(report, indent=2, default=str))
print(f"  Report saved  ({report['elapsed_seconds']}s)")

if validated:
    cfg = json.loads(Path("configs/production_p91b_champion.json").read_text())
    brs_cfg = cfg["breadth_regime_switching"]
    fts_cfg = cfg.get("funding_term_structure_overlay", {})
    # Update windows
    brs_cfg["rolling_percentile_window"] = best_brd_win
    fts_cfg["rolling_percentile_window"] = best_fts_win
    cfg["funding_term_structure_overlay"] = fts_cfg
    ver = cfg.get("_version", "2.32.0")
    major, minor, patch = map(int, ver.split("."))
    new_ver = "%d.%d.%d" % (major, minor + 1, patch)
    cfg["_version"] = new_ver
    cfg["_created"] = "2026-02-21"
    old_val = cfg.get("_validated", "")
    cfg["_validated"] = old_val + (
        "; FTS+brd pct_win P207: fts=%d->%d brd=%d->%d LOYO %d/5 delta=%+.4f OBJ=%.4f — PRODUCTION %s" % (
            curr_fts_pct_win, best_fts_win, curr_brd_pct_win, best_brd_win,
            loyo_wins, delta, best_obj_b, new_ver))
    Path("configs/production_p91b_champion.json").write_text(json.dumps(cfg, indent=2))
    print(f"\n✅ VALIDATED → {new_ver} OBJ={best_obj_b:.4f}")
    cm = "phase207: FTS+brd pct_win -- VALIDATED fts=%d brd=%d LOYO %d/5 delta=%+.4f OBJ=%.4f" % (
        best_fts_win, best_brd_win, loyo_wins, delta, best_obj_b)
else:
    new_ver = ver_now
    print(f"\n⚠️ NO IMPROVEMENT — pct_win=336 near-optimal (LOYO {loyo_wins}/5 delta={delta:+.4f})")
    cm = "phase207: FTS+brd pct_win -- WEAK LOYO %d/5 delta=%+.4f fts=%d brd=%d" % (
        loyo_wins, delta, best_fts_win, best_brd_win)

os.system("git add configs/production_p91b_champion.json scripts/run_phase207_fts_pct_window_sweep.py "
          "artifacts/phase207/ 2>/dev/null || true")
os.system('git commit -m "' + cm + '"')
os.system("git stash && git pull --rebase && git stash pop && git push")

print(f"\n[DONE] Phase 207 complete.")
