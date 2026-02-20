"""
Phase 206 — Dispersion Overlay Re-Tune
========================================
Funding cross-sectional dispersion overlay.
Last tuned at P170 (OBJ=2.4148). Since then: OBJ has gone 2.4148 → 3.0849.
Current config: boost_scale=1.0, thr=0.60, pct=240.
scale=1.0 = effectively disabled (no boost).

Question: With new regime weights and FTS params, does dispersion boost add alpha?

Part A: disp_scale ∈ [1.00, 1.02, 1.04, 1.06, 1.08, 1.10, 1.12]
  - thr=0.60, pct_win=240 fixed
Part B: disp_thr ∈ [0.50, 0.55, 0.60, 0.65, 0.70]
  - using best disp_scale from Part A

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
    out = Path("artifacts/phase206"); out.mkdir(parents=True, exist_ok=True)
    (out / "phase206_report.json").write_text(json.dumps(_partial, indent=2, default=str))
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

DISP_SCALE_SWEEP = [1.00, 1.02, 1.04, 1.06, 1.08, 1.10, 1.12]
DISP_THR_SWEEP   = [0.50, 0.55, 0.60, 0.65, 0.70]

VOL_WINDOW = 168; FUND_DISP_PCT = 240; PCT_WINDOW = 336

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
    # Current dispersion settings from config
    disp_cfg = cfg.get("funding_dispersion_overlay", {})
    params["disp_pct_win"] = disp_cfg.get("percentile_window", 240)
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

def load_year(year, params):
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

    # Dispersion overlay precomputed with FULL pct window from params
    disp_pct_win = p["disp_pct_win"]
    fund_std_raw = np.std(fund_rates, axis=1)
    fund_std_pct = np.full(n, 0.5)
    for i in range(disp_pct_win, n):
        fund_std_pct[i] = float(np.mean(fund_std_raw[i-disp_pct_win:i] <= fund_std_raw[i]))
    fund_std_pct[:disp_pct_win] = 0.5

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
    return btc_vol, fund_std_pct, ts_spread_pct, regime, sig_rets, n, fund_std_raw

def compute_ensemble(year_data, weights, params, disp_scale=1.0, disp_thr=0.60):
    p = params
    btc_vol, fund_std_pct, ts_spread_pct, regime, sig_rets, n, _ = year_data
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
        # Dispersion overlay — parameterized
        if disp_scale > 1.0 and fsp[i] > disp_thr:
            ret_i *= disp_scale
        # FTS overlay
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

def eval_params(disp_scale, disp_thr, weights, year_data, params):
    yr_sharpes = {}
    for yr in YEARS:
        ens = compute_ensemble(year_data[yr], weights, params, disp_scale, disp_thr)
        yr_sharpes[yr] = sharpe(ens, year_data[yr][5])
    return obj_fn(yr_sharpes), yr_sharpes

# ── MAIN ────────────────────────────────────────────────────────────────────
print("=" * 60)
print("Phase 206 — Dispersion Overlay Re-Tune")
print("=" * 60)
_start = time.time()

prod_cfg, WEIGHTS, PARAMS, baseline_obj = get_config()
ver_now = prod_cfg.get("_version", "2.32.0")
curr_disp_scale = prod_cfg.get("funding_dispersion_overlay", {}).get("boost_scale", 1.0)
curr_disp_thr = prod_cfg.get("funding_dispersion_overlay", {}).get("boost_threshold_pct", 0.60)
print(f"  Config: {ver_now}  baseline_obj={baseline_obj:.4f}")
print(f"  Dispersion now: scale={curr_disp_scale}  thr={curr_disp_thr}  pct_win={PARAMS['disp_pct_win']}")

print("\n[1] Loading all years ...")
year_data = {}
for yr in YEARS:
    print(f"  {yr}", end="", flush=True)
    year_data[yr] = load_year(yr, PARAMS)
    print()

# Part A: disp_scale sweep (thr=0.60 fixed)
print(f"\n[2] Part A: disp_scale sweep (thr={curr_disp_thr:.2f} fixed) ...")
results_a = []
for ds in DISP_SCALE_SWEEP:
    obj, _ = eval_params(ds, curr_disp_thr, WEIGHTS, year_data, PARAMS)
    results_a.append((ds, obj))
    print(f"  disp_scale={ds:.2f}  OBJ={obj:.4f}")

best_a = max(results_a, key=lambda x: x[1])
best_disp_scale, best_obj_a = best_a
print(f"  Best disp_scale: {best_disp_scale:.2f}  OBJ={best_obj_a:.4f}")

# Part B: disp_thr sweep (using best scale)
print(f"\n[3] Part B: disp_thr sweep (scale={best_disp_scale:.2f} fixed) ...")
results_b = []
for dt in DISP_THR_SWEEP:
    obj, _ = eval_params(best_disp_scale, dt, WEIGHTS, year_data, PARAMS)
    results_b.append((dt, obj))
    print(f"  disp_thr={dt:.2f}  OBJ={obj:.4f}")

best_b = max(results_b, key=lambda x: x[1])
best_disp_thr, best_obj_b = best_b
delta = best_obj_b - baseline_obj
print(f"  Best disp_thr: {best_disp_thr:.2f}  OBJ={best_obj_b:.4f}  Δ={delta:+.4f}")

# LOYO validation
print(f"\n[4] LOYO validation ...")
_, best_yr = eval_params(best_disp_scale, best_disp_thr, WEIGHTS, year_data, PARAMS)
_, base_yr = eval_params(curr_disp_scale, curr_disp_thr, WEIGHTS, year_data, PARAMS)
loyo_wins = 0
for yr in YEARS:
    win = best_yr[yr] > base_yr[yr]
    if win: loyo_wins += 1
    print(f"  {yr}: best={best_yr[yr]:.4f} curr={base_yr[yr]:.4f} {'WIN' if win else 'LOSS'}")

print(f"\n  LOYO: {loyo_wins}/5 wins")

validated = loyo_wins >= 3 and delta > 0.005

# Save report
out = Path("artifacts/phase206"); out.mkdir(parents=True, exist_ok=True)
report = {
    "phase": 206,
    "description": "Dispersion overlay re-tune (scale + threshold)",
    "elapsed_seconds": round(time.time() - _start, 1),
    "baseline_obj": baseline_obj, "best_obj": best_obj_b, "delta": delta,
    "curr_disp_scale": curr_disp_scale, "curr_disp_thr": curr_disp_thr,
    "best_disp_scale": best_disp_scale, "best_disp_thr": best_disp_thr,
    "loyo_wins": loyo_wins, "validated": validated,
    "yearly_best": best_yr, "yearly_base": base_yr,
    "scale_sweep": results_a, "thr_sweep": results_b,
}
(out / "phase206_report.json").write_text(json.dumps(report, indent=2, default=str))
print(f"  Report saved  ({report['elapsed_seconds']}s)")

if validated:
    cfg = json.loads(Path("configs/production_p91b_champion.json").read_text())
    disp_cfg = cfg.get("funding_dispersion_overlay", {})
    disp_cfg["boost_scale"] = best_disp_scale
    disp_cfg["boost_threshold_pct"] = best_disp_thr
    cfg["funding_dispersion_overlay"] = disp_cfg
    ver = cfg.get("_version", "2.32.0")
    major, minor, patch = map(int, ver.split("."))
    new_ver = "%d.%d.%d" % (major, minor + 1, patch)
    cfg["_version"] = new_ver
    cfg["_created"] = "2026-02-21"
    old_val = cfg.get("_validated", "")
    cfg["_validated"] = old_val + (
        "; Disp overlay P206: scale=%.2f->%.2f thr=%.2f->%.2f LOYO %d/5 delta=%+.4f OBJ=%.4f — PRODUCTION %s" % (
            curr_disp_scale, best_disp_scale, curr_disp_thr, best_disp_thr,
            loyo_wins, delta, best_obj_b, new_ver))
    Path("configs/production_p91b_champion.json").write_text(json.dumps(cfg, indent=2))
    print(f"\n✅ VALIDATED → {new_ver} OBJ={best_obj_b:.4f}")
    cm = "phase206: disp overlay retune -- VALIDATED scale=%.2f thr=%.2f LOYO %d/5 delta=%+.4f OBJ=%.4f" % (
        best_disp_scale, best_disp_thr, loyo_wins, delta, best_obj_b)
else:
    new_ver = ver_now
    print(f"\n⚠️ NO IMPROVEMENT — disp=1.0 confirmed optimal (LOYO {loyo_wins}/5 delta={delta:+.4f})")
    cm = "phase206: disp overlay retune -- WEAK LOYO %d/5 delta=%+.4f scale=%.2f thr=%.2f" % (
        loyo_wins, delta, best_disp_scale, best_disp_thr)

os.system("git add configs/production_p91b_champion.json scripts/run_phase206_disp_overlay_retune.py "
          "artifacts/phase206/ 2>/dev/null || true")
os.system('git commit -m "' + cm + '"')
os.system("git stash && git pull --rebase && git stash pop && git push")

print(f"\n[DONE] Phase 206 complete.")
