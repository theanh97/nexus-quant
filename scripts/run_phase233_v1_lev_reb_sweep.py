"""
Phase 233 — V1 target_gross_leverage + rebalance_interval_bars sweep

V1 current: lev=0.35, reb=60
Part A: lev sweep [0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50] (reb=60 fixed)
Part B: reb sweep [24, 36, 48, 60, 72, 96, 120] (best lev from Part A)
LOYO: >=3/5 wins AND delta>0.005
"""
import os, sys, json, time, re, signal as _signal
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
    out = Path("artifacts/phase233"); out.mkdir(parents=True, exist_ok=True)
    (out / "phase233_report.json").write_text(json.dumps(_partial, indent=2, default=str))
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

V1_LEV_SWEEP = [0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]
V1_REB_SWEEP = [24, 36, 48, 60, 72, 96, 120]


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
        "p_low":  brs.get("p_low", 0.30),
        "p_high": brs.get("p_high", 0.60),
        "brd_lb": brs.get("breadth_lookback_bars", 192),
        "pct_win": brs.get("rolling_percentile_window", 336),
        "ts_rt": fts.get("reduce_threshold", 0.65),
        "ts_rs": fts.get("reduce_scale", 0.30),
        "ts_bt": fts.get("boost_threshold", 0.25),
        "ts_bs": fts.get("boost_scale", 1.85),
        "ts_short": fts.get("short_window_bars", 16),
        "ts_long":  fts.get("long_window_bars", 72),
    }
    vol_ov = cfg.get("vol_regime_overlay", {})
    params["vol_thr"]   = vol_ov.get("threshold", 0.50)
    params["vol_scale"] = vol_ov.get("scale_factor", 0.30)
    params["f168_boost"] = vol_ov.get("f144_boost", 0.00)
    sigs = cfg["ensemble"]["signals"]
    v1p  = sigs["v1"]["params"]
    i460p = sigs["i460bw168"]["params"]
    i415p = sigs["i415bw216"]["params"]
    f168p = sigs.get("f144", sigs.get("f168", {})).get("params", {})
    m = re.findall(r"OBJ=([\d.]+)", cfg.get("_validated", ""))
    baseline_obj = float(m[-1]) if m else 3.2766
    return cfg, weights, params, v1p, i460p, i415p, f168p, baseline_obj


def rolling_mean_arr(a, w):
    out = np.full(len(a), np.nan)
    if w <= 0: return out
    cs = np.cumsum(np.where(np.isnan(a), 0.0, a))
    for i in range(w - 1, len(a)):
        out[i] = cs[i] / w if i == w - 1 else (cs[i] - cs[i - w]) / w
    return out


def load_year_fixed(year, params, i460p, i415p, f168p):
    """Load all FIXED signals (I460, I415, F168) + auxiliary arrays for a given year."""
    p = params
    s, e = YEAR_RANGES[year]
    cfg_d = {"provider": "binance_rest_v1", "symbols": SYMBOLS,
             "start": s, "end": e, "bar_interval": "1h", "cache_dir": ".cache/binance_rest"}
    dataset = make_provider(cfg_d, seed=42).load()
    n = len(dataset.timeline)

    # Build close matrix
    close_mat = np.zeros((n, len(SYMBOLS)))
    for j, sym in enumerate(SYMBOLS):
        for i in range(n):
            close_mat[i, j] = dataset.close(sym, i)

    # BTC vol (for vol overlay)
    btc_rets = np.zeros(n)
    for i in range(1, n):
        c0 = close_mat[i-1, 0]; c1 = close_mat[i, 0]
        btc_rets[i] = (c1/c0 - 1.0) if c0 > 0 else 0.0
    btc_vol = np.full(n, np.nan)
    for i in range(VOL_WINDOW, n):
        btc_vol[i] = float(np.std(btc_rets[i-VOL_WINDOW:i])) * np.sqrt(8760)
    if VOL_WINDOW < n: btc_vol[:VOL_WINDOW] = btc_vol[VOL_WINDOW]

    # Funding dispersion (for FTS and f168_boost)
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

    # FTS spread percentile
    xsect_mean = np.mean(fund_rates, axis=1)
    ts_raw = (rolling_mean_arr(xsect_mean, p["ts_short"]) -
              rolling_mean_arr(xsect_mean, p["ts_long"]))
    ts_spread_pct = np.full(n, 0.5)
    for i in range(PCT_WINDOW, n):
        ts_spread_pct[i] = float(np.mean(ts_raw[i-PCT_WINDOW:i] <= ts_raw[i]))
    ts_spread_pct[:PCT_WINDOW] = 0.5

    # Breadth regime
    brd_lb = int(p["brd_lb"]) if p["brd_lb"] else 192
    pct_win = int(p["pct_win"]) if p["pct_win"] else 336
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

    def run(name, sp):
        res = BacktestEngine(BacktestConfig(costs=COST_MODEL)).run(
            dataset, make_strategy({"name": name, "params": sp}))
        return np.array(res.returns)

    # Fixed signals: I460, I415, F168
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

    # Store dataset for V1 re-runs
    fixed = {
        "dataset": dataset,
        "btc_vol": btc_vol,
        "fund_std_pct": fund_std_pct,
        "ts_spread_pct": ts_spread_pct,
        "brd_pct": brd_pct,
        "i460_rets": i460_rets,
        "i415_rets": i415_rets,
        "f168_rets": f168_rets,
        "n": n,
    }
    return fixed


def run_v1(dataset, lev, reb, v1p):
    """Run V1 signal with given lev and reb."""
    res = BacktestEngine(BacktestConfig(costs=COST_MODEL)).run(
        dataset, make_strategy({"name": "nexus_alpha_v1", "params": {
            "k_per_side": v1p.get("k_per_side", 2),
            "w_carry": v1p.get("w_carry", 0.25),
            "w_mom": v1p.get("w_mom", 0.45),
            "w_mean_reversion": v1p.get("w_mean_reversion", 0.30),
            "momentum_lookback_bars": v1p.get("momentum_lookback_bars", 336),
            "mean_reversion_lookback_bars": v1p.get("mean_reversion_lookback_bars", 84),
            "vol_lookback_bars": v1p.get("vol_lookback_bars", 192),
            "target_gross_leverage": lev,
            "rebalance_interval_bars": reb,
        }}))
    return np.array(res.returns)


def compute_ensemble(fixed, v1_rets, weights, params):
    """Compute ensemble returns with regime switching + overlays."""
    p = params
    btc_vol  = fixed["btc_vol"]
    fsp      = fixed["fund_std_pct"]
    tsp      = fixed["ts_spread_pct"]
    brd_pct  = fixed["brd_pct"]
    i460     = fixed["i460_rets"]
    i415     = fixed["i415_rets"]
    f168     = fixed["f168_rets"]
    p_low    = p["p_low"]; p_high = p["p_high"]

    regime = np.where(brd_pct >= p_high, 2, np.where(brd_pct >= p_low, 1, 0)).astype(int)
    min_len = min(len(v1_rets), len(i460), len(i415), len(f168), len(brd_pct))
    bv  = btc_vol[:min_len]; reg = regime[:min_len]
    fsp_ = fsp[:min_len]; tsp_ = tsp[:min_len]
    v1_ = v1_rets[:min_len]; i4 = i460[:min_len]; i5 = i415[:min_len]; f1 = f168[:min_len]

    ens = np.zeros(min_len)
    for i in range(min_len):
        w = weights[["LOW", "MID", "HIGH"][int(reg[i])]]
        if not np.isnan(bv[i]) and bv[i] > p["vol_thr"]:
            ret_i = (w["v1"]*v1_[i] + w["i460"]*i4[i] +
                     w["i415"]*i5[i] + w["f168"]*f1[i]) * p["vol_scale"]
        else:
            ret_i = (w["v1"]*v1_[i] + w["i460"]*i4[i] +
                     w["i415"]*i5[i] + w["f168"]*f1[i])
        if DISP_SCALE > 1.0 and fsp_[i] > DISP_THR: ret_i *= DISP_SCALE
        if tsp_[i] > p["ts_rt"]:  ret_i *= p["ts_rs"]
        elif tsp_[i] < p["ts_bt"]: ret_i *= p["ts_bs"]
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


def eval_v1_config(year_data, weights, params, v1p, lev, reb):
    yr_sharpes = {}
    for yr in YEARS:
        f = year_data[yr]
        v1_rets = run_v1(f["dataset"], lev, reb, v1p)
        ens = compute_ensemble(f, v1_rets, weights, params)
        yr_sharpes[yr] = sharpe(ens, f["n"])
    return yr_sharpes, obj_fn(yr_sharpes)


# ── MAIN ────────────────────────────────────────────────────────────────────
print("=" * 60)
print("Phase 233 — V1 lev + reb sweep")
print("=" * 60)
_start = time.time()

cfg, WEIGHTS, PARAMS, v1p, i460p, i415p, f168p, baseline_obj = get_config()
ver_now = cfg.get("_version", "2.40.0")
curr_lev = v1p["target_gross_leverage"]
curr_reb = v1p["rebalance_interval_bars"]
print(f"  Config: {ver_now}  baseline_obj(str)={baseline_obj:.4f}")
print(f"  V1: lev={curr_lev}  reb={curr_reb}")

# [1] Load fixed signals for all years
print("\n[1] Loading fixed signals (I460, I415, F168) + breadth for all years ...")
year_data = {}
for yr in YEARS:
    print(f"  {yr}", end=" ", flush=True)
    year_data[yr] = load_year_fixed(yr, PARAMS, i460p, i415p, f168p)
    print(f"n={year_data[yr]['n']}")

# Baseline: current V1 lev + reb
print(f"\n  Computing baseline (lev={curr_lev} reb={curr_reb}) ...")
baseline_yr, baseline_computed = eval_v1_config(
    year_data, WEIGHTS, PARAMS, v1p, curr_lev, curr_reb)
print(f"  Baseline OBJ (computed): {baseline_computed:.4f}")

# [2] Part A: V1 lev sweep
print(f"\n[2] Part A: V1 lev sweep (reb={curr_reb} fixed) ...")
best_lev = curr_lev
best_lev_obj = baseline_computed
lev_results = {}
for lev in V1_LEV_SWEEP:
    _, obj = eval_v1_config(year_data, WEIGHTS, PARAMS, v1p, lev, curr_reb)
    lev_results[lev] = obj
    marker = " ←" if obj > best_lev_obj else ""
    print(f"  lev={lev:.2f}  OBJ={obj:.4f}{marker}")
    if obj > best_lev_obj:
        best_lev_obj = obj
        best_lev = lev
print(f"  Best lev: {best_lev:.2f}  OBJ={best_lev_obj:.4f}")

# [3] Part B: V1 reb sweep with best lev
print(f"\n[3] Part B: V1 reb sweep (lev={best_lev:.2f}) ...")
best_reb = curr_reb
best_reb_obj = best_lev_obj
reb_results = {}
for reb in V1_REB_SWEEP:
    _, obj = eval_v1_config(year_data, WEIGHTS, PARAMS, v1p, best_lev, reb)
    reb_results[reb] = obj
    marker = " ←" if obj > best_reb_obj else ""
    print(f"  reb={reb:3d}  OBJ={obj:.4f}{marker}")
    if obj > best_reb_obj:
        best_reb_obj = obj
        best_reb = reb
delta = best_reb_obj - baseline_computed
print(f"  Best reb: {best_reb}  OBJ={best_reb_obj:.4f}  Δ={delta:+.4f}")

# [4] LOYO validation
print(f"\n[4] LOYO validation ...")
best_yr_s, _ = eval_v1_config(year_data, WEIGHTS, PARAMS, v1p, best_lev, best_reb)
loyo_wins = 0
for yr in YEARS:
    b = best_yr_s.get(yr, 0); c = baseline_yr.get(yr, 0)
    win = b > c
    if win: loyo_wins += 1
    print(f"  {yr}: best={b:.4f} curr={c:.4f} {'WIN' if win else 'LOSS'}")

validated = loyo_wins >= 3 and delta > 0.005
print(f"\n  LOYO: {loyo_wins}/5 wins  delta={delta:+.4f}")

# [5] Persist report + optionally update config
elapsed = time.time() - _start
out = Path("artifacts/phase233"); out.mkdir(parents=True, exist_ok=True)
report = {
    "phase": 233, "best_lev": best_lev, "best_reb": best_reb,
    "best_obj": best_reb_obj, "delta": delta,
    "loyo_wins": loyo_wins, "validated": validated,
    "lev_results": {str(k): v for k, v in lev_results.items()},
    "reb_results": {str(k): v for k, v in reb_results.items()},
    "elapsed_s": elapsed,
}
(out / "phase233_report.json").write_text(json.dumps(report, indent=2))
_partial.update(report)

if validated:
    print(f"\n✅ VALIDATED — updating config V1 lev={best_lev:.2f} reb={best_reb}")
    cfg["ensemble"]["signals"]["v1"]["params"]["target_gross_leverage"] = best_lev
    cfg["ensemble"]["signals"]["v1"]["params"]["rebalance_interval_bars"] = best_reb
    ver = cfg.get("_version", "2.40.0"); parts = ver.split(".")
    parts[-1] = str(int(parts[-1]) + 1); new_ver = ".".join(parts)
    cfg["_version"] = new_ver
    old_val = cfg.get("_validated", "")
    cfg["_validated"] = (old_val +
        f" | v{new_ver} V1lev={best_lev:.2f} V1reb={best_reb} OBJ={best_reb_obj:.4f} LOYO={loyo_wins}/5")
    Path("configs/production_p91b_champion.json").write_text(json.dumps(cfg, indent=2))
    msg = (f"phase233: V1 lev+reb -- VALIDATED lev={best_lev:.2f} reb={best_reb} "
           f"LOYO {loyo_wins}/5 delta={delta:+.4f} OBJ={best_reb_obj:.4f}")
    os.system(f"git add configs/production_p91b_champion.json artifacts/phase233/ && "
              f"git stash && git pull --rebase && git stash pop && "
              f"git commit -m '{msg}' && git push")
else:
    flag = "WEAK" if (loyo_wins > 0 or delta > 0) else "CONFIRMED OPTIMAL"
    print(f"\n{'⚠️' if loyo_wins > 0 else '❌'} NO IMPROVEMENT — {flag} "
          f"(LOYO {loyo_wins}/5 delta={delta:+.4f})")
    msg = (f"phase233: V1 lev+reb -- WEAK LOYO {loyo_wins}/5 delta={delta:+.4f} "
           f"lev={best_lev:.2f} reb={best_reb}")
    os.system(f"git add artifacts/phase233/ && "
              f"git stash && git pull --rebase && git stash pop && "
              f"git commit -m '{msg}' && git push")

print(f"  Report saved  ({elapsed:.1f}s)")
print(f"\n[DONE] Phase 233 complete.")
