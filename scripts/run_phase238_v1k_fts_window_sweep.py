"""
Phase 238 — V1 k_per_side + FTS short/long window sweep

V1 k_per_side: not explicitly swept in recent phases. Current k=2.
FTS windows: short=16, long=72. Last tuned in P200.

Part A: V1 k ∈ [1, 2, 3, 4] (all other params fixed)
Part B: FTS short ∈ [8, 12, 16, 20, 24, 32] (best V1 k)
Part C: FTS long ∈ [48, 60, 72, 84, 96, 120] (best V1 k + best short)

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
    _partial["partial"] = True; sys.exit(0)
_signal.signal(_signal.SIGALRM, _on_timeout)
_signal.alarm(10800)

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

V1_K_SWEEP      = [1, 2, 3, 4]
FTS_SHORT_SWEEP = [8, 12, 16, 20, 24, 32]
FTS_LONG_SWEEP  = [48, 60, 72, 84, 96, 120]


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


def load_year_all(year, params, v1p, i460p, i415p, f168p):
    """Load ALL signals (including varying FTS params) + auxiliary data."""
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
        btc_rets[i] = (c1/c0 - 1.0) if c0 > 0 else 0.0
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

    # Store raw FTS data for recomputing ts_spread_pct with different windows
    xsect_mean = np.mean(fund_rates, axis=1)

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

    # Fixed signals: I460, I415, F168 (FTS overlay recomputed separately)
    i460_r = run("idio_momentum_alpha", {
        "k_per_side": i460p.get("k_per_side", 4),
        "lookback_bars": i460p.get("lookback_bars", 480),
        "beta_window_bars": i460p.get("beta_window_bars", 168),
        "target_gross_leverage": i460p.get("target_gross_leverage", 0.20),
        "rebalance_interval_bars": i460p.get("rebalance_interval_bars", 48),
    })
    i415_r = run("idio_momentum_alpha", {
        "k_per_side": i415p.get("k_per_side", 4),
        "lookback_bars": i415p.get("lookback_bars", 415),
        "beta_window_bars": i415p.get("beta_window_bars", 144),
        "target_gross_leverage": i415p.get("target_gross_leverage", 0.20),
        "rebalance_interval_bars": i415p.get("rebalance_interval_bars", 48),
    })
    f168_r = run("funding_momentum_alpha", {
        "k_per_side": f168p.get("k_per_side", 2),
        "funding_lookback_bars": f168p.get("funding_lookback_bars", 168),
        "direction": f168p.get("direction", "contrarian"),
        "target_gross_leverage": f168p.get("target_gross_leverage", 0.25),
        "rebalance_interval_bars": f168p.get("rebalance_interval_bars", 36),
    })
    # Baseline V1 (k=current)
    v1_curr = run("nexus_alpha_v1", {
        "k_per_side": v1p.get("k_per_side", 2),
        "w_carry": v1p.get("w_carry", 0.25), "w_mom": v1p.get("w_mom", 0.45),
        "w_mean_reversion": v1p.get("w_mean_reversion", 0.30),
        "momentum_lookback_bars": v1p.get("momentum_lookback_bars", 336),
        "mean_reversion_lookback_bars": v1p.get("mean_reversion_lookback_bars", 84),
        "vol_lookback_bars": v1p.get("vol_lookback_bars", 192),
        "target_gross_leverage": v1p.get("target_gross_leverage", 0.35),
        "rebalance_interval_bars": v1p.get("rebalance_interval_bars", 60),
    })

    return {
        "dataset": dataset, "n": n,
        "btc_vol": btc_vol, "fund_std_pct": fund_std_pct,
        "brd_pct": brd_pct, "xsect_mean": xsect_mean,
        "i460": i460_r, "i415": i415_r, "f168": f168_r, "v1_curr": v1_curr,
    }


def compute_ts_spread_pct(xsect_mean, ts_short, ts_long, n):
    ts_raw = rolling_mean_arr(xsect_mean, ts_short) - rolling_mean_arr(xsect_mean, ts_long)
    ts_spread_pct = np.full(n, 0.5)
    for i in range(PCT_WINDOW, n):
        ts_spread_pct[i] = float(np.mean(ts_raw[i-PCT_WINDOW:i] <= ts_raw[i]))
    ts_spread_pct[:PCT_WINDOW] = 0.5
    return ts_spread_pct


def compute_ensemble(base, v1_rets, i460_rets, i415_rets, f168_rets, weights, params, ts_spread_pct=None):
    p = params
    brd_pct = base["brd_pct"]
    p_low = p["p_low"]; p_high = p["p_high"]
    regime = np.where(brd_pct >= p_high, 2, np.where(brd_pct >= p_low, 1, 0)).astype(int)
    if ts_spread_pct is None:
        ts_spread_pct = compute_ts_spread_pct(
            base["xsect_mean"], p["ts_short"], p["ts_long"], base["n"])
    min_len = min(len(v1_rets), len(i460_rets), len(i415_rets), len(f168_rets), len(brd_pct))
    bv = base["btc_vol"][:min_len]; reg = regime[:min_len]
    fsp = base["fund_std_pct"][:min_len]; tsp = ts_spread_pct[:min_len]
    v1 = v1_rets[:min_len]; i4 = i460_rets[:min_len]
    i5 = i415_rets[:min_len]; f1 = f168_rets[:min_len]
    ens = np.zeros(min_len)
    for i in range(min_len):
        w = weights[["LOW", "MID", "HIGH"][int(reg[i])]]
        if not np.isnan(bv[i]) and bv[i] > p["vol_thr"]:
            ret_i = (w["v1"]*v1[i] + w["i460"]*i4[i] +
                     w["i415"]*i5[i] + w["f168"]*f1[i]) * p["vol_scale"]
        else:
            ret_i = (w["v1"]*v1[i] + w["i460"]*i4[i] +
                     w["i415"]*i5[i] + w["f168"]*f1[i])
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
print("Phase 238 — V1 k_per_side + FTS window sweep")
print("=" * 60)
_start = time.time()

cfg, WEIGHTS, PARAMS, v1p, i460p, i415p, f168p, baseline_obj = get_config()
ver_now = cfg.get("_version", "2.41.0")
curr_v1_k     = v1p.get("k_per_side", 2)
curr_ts_short = PARAMS["ts_short"]
curr_ts_long  = PARAMS["ts_long"]
print(f"  Config: {ver_now}  baseline_obj(str)={baseline_obj:.4f}")
print(f"  V1 k={curr_v1_k}  ts_short={curr_ts_short}  ts_long={curr_ts_long}")

# [1] Load all data
print("\n[1] Loading all data ...")
year_data = {}
for yr in YEARS:
    print(f"  {yr}", end=" ", flush=True)
    year_data[yr] = load_year_all(yr, PARAMS, v1p, i460p, i415p, f168p)
    print(f"n={year_data[yr]['n']}")

# Pre-compute baseline ts_spread_pct (current short/long)
baseline_tsp = {yr: compute_ts_spread_pct(
    year_data[yr]["xsect_mean"], curr_ts_short, curr_ts_long, year_data[yr]["n"])
    for yr in YEARS}

# Baseline OBJ
baseline_yr_s = {}
for yr in YEARS:
    d = year_data[yr]
    ens = compute_ensemble(
        d, d["v1_curr"], d["i460"], d["i415"], d["f168"], WEIGHTS, PARAMS, baseline_tsp[yr])
    baseline_yr_s[yr] = sharpe(ens, d["n"])
baseline_computed = obj_fn(baseline_yr_s)
print(f"\n  Baseline OBJ (computed): {baseline_computed:.4f}")

def run_v1_k(dataset, k):
    from nexus_quant.backtest.engine import BacktestConfig, BacktestEngine
    from nexus_quant.strategies.registry import make_strategy
    res = BacktestEngine(BacktestConfig(costs=COST_MODEL)).run(
        dataset, make_strategy({"name": "nexus_alpha_v1", "params": {
            "k_per_side": k,
            "w_carry": v1p.get("w_carry", 0.25), "w_mom": v1p.get("w_mom", 0.45),
            "w_mean_reversion": v1p.get("w_mean_reversion", 0.30),
            "momentum_lookback_bars": v1p.get("momentum_lookback_bars", 336),
            "mean_reversion_lookback_bars": v1p.get("mean_reversion_lookback_bars", 84),
            "vol_lookback_bars": v1p.get("vol_lookback_bars", 192),
            "target_gross_leverage": v1p.get("target_gross_leverage", 0.35),
            "rebalance_interval_bars": v1p.get("rebalance_interval_bars", 60),
        }}))
    return np.array(res.returns)

# ── Part A: V1 k sweep ────────────────────────────────────────────────────
print(f"\n[2] Part A: V1 k_per_side sweep ...")
best_v1_k = curr_v1_k
best_v1_obj = baseline_computed
v1_k_results = {}
for k in V1_K_SWEEP:
    yr_s = {}
    for yr in YEARS:
        d = year_data[yr]
        v1_new = run_v1_k(d["dataset"], k)
        ens = compute_ensemble(
            d, v1_new, d["i460"], d["i415"], d["f168"], WEIGHTS, PARAMS, baseline_tsp[yr])
        yr_s[yr] = sharpe(ens, d["n"])
    obj = obj_fn(yr_s)
    v1_k_results[k] = obj
    marker = " ←" if obj > best_v1_obj else ""
    print(f"  V1 k={k}  OBJ={obj:.4f}{marker}")
    if obj > best_v1_obj:
        best_v1_obj = obj; best_v1_k = k
print(f"  Best V1 k: {best_v1_k}  OBJ={best_v1_obj:.4f}")

# Re-run V1 with best k for all years
best_v1_rets = {}
if best_v1_k != curr_v1_k:
    for yr in YEARS:
        best_v1_rets[yr] = run_v1_k(year_data[yr]["dataset"], best_v1_k)
else:
    for yr in YEARS:
        best_v1_rets[yr] = year_data[yr]["v1_curr"]

# ── Part B: FTS short window sweep ────────────────────────────────────────
print(f"\n[3] Part B: FTS short window sweep (V1 k={best_v1_k}) ...")
best_short = curr_ts_short
best_short_obj = best_v1_obj
short_results = {}
for sh in FTS_SHORT_SWEEP:
    tsp_sh = {yr: compute_ts_spread_pct(
        year_data[yr]["xsect_mean"], sh, curr_ts_long, year_data[yr]["n"])
        for yr in YEARS}
    yr_s = {}
    for yr in YEARS:
        d = year_data[yr]
        ens = compute_ensemble(
            d, best_v1_rets[yr], d["i460"], d["i415"], d["f168"], WEIGHTS, PARAMS, tsp_sh[yr])
        yr_s[yr] = sharpe(ens, d["n"])
    obj = obj_fn(yr_s)
    short_results[sh] = obj
    marker = " ←" if obj > best_short_obj else ""
    print(f"  ts_short={sh:3d}  OBJ={obj:.4f}{marker}")
    if obj > best_short_obj:
        best_short_obj = obj; best_short = sh
print(f"  Best ts_short: {best_short}  OBJ={best_short_obj:.4f}")

# ── Part C: FTS long window sweep ────────────────────────────────────────
print(f"\n[4] Part C: FTS long window sweep (ts_short={best_short}) ...")
best_long = curr_ts_long
best_long_obj = best_short_obj
long_results = {}
for lg in FTS_LONG_SWEEP:
    if lg <= best_short: continue
    tsp_lg = {yr: compute_ts_spread_pct(
        year_data[yr]["xsect_mean"], best_short, lg, year_data[yr]["n"])
        for yr in YEARS}
    yr_s = {}
    for yr in YEARS:
        d = year_data[yr]
        ens = compute_ensemble(
            d, best_v1_rets[yr], d["i460"], d["i415"], d["f168"], WEIGHTS, PARAMS, tsp_lg[yr])
        yr_s[yr] = sharpe(ens, d["n"])
    obj = obj_fn(yr_s)
    long_results[lg] = obj
    marker = " ←" if obj > best_long_obj else ""
    print(f"  ts_long={lg:4d}  OBJ={obj:.4f}{marker}")
    if obj > best_long_obj:
        best_long_obj = obj; best_long = lg
delta = best_long_obj - baseline_computed
print(f"  Best ts_long: {best_long}  OBJ={best_long_obj:.4f}  Δ={delta:+.4f}")

# ── LOYO validation ──────────────────────────────────────────────────────
print(f"\n[5] LOYO validation ...")
best_tsp = {yr: compute_ts_spread_pct(
    year_data[yr]["xsect_mean"], best_short, best_long, year_data[yr]["n"])
    for yr in YEARS}
best_yr_s = {}
for yr in YEARS:
    d = year_data[yr]
    ens = compute_ensemble(
        d, best_v1_rets[yr], d["i460"], d["i415"], d["f168"], WEIGHTS, PARAMS, best_tsp[yr])
    best_yr_s[yr] = sharpe(ens, d["n"])

loyo_wins = 0
for yr in YEARS:
    b_s = best_yr_s[yr]; c_s = baseline_yr_s[yr]
    win = b_s > c_s
    if win: loyo_wins += 1
    print(f"  {yr}: best={b_s:.4f} curr={c_s:.4f} {'WIN' if win else 'LOSS'}")

validated = loyo_wins >= 3 and delta > 0.005
print(f"\n  LOYO: {loyo_wins}/5 wins  delta={delta:+.4f}")

# ── Save + commit ────────────────────────────────────────────────────────
elapsed = time.time() - _start

if validated:
    print(f"\n✅ VALIDATED — V1k={best_v1_k} ts_short={best_short} ts_long={best_long}")
    cfg["ensemble"]["signals"]["v1"]["params"]["k_per_side"] = best_v1_k
    fts = cfg.get("funding_term_structure_overlay", {})
    fts["short_window_bars"] = best_short
    fts["long_window_bars"] = best_long
    cfg["funding_term_structure_overlay"] = fts
    ver = cfg.get("_version", "2.41.0"); parts = ver.split(".")
    parts[-1] = str(int(parts[-1]) + 1); new_ver = ".".join(parts)
    cfg["_version"] = new_ver
    old_val = cfg.get("_validated", "")
    cfg["_validated"] = (old_val +
        f" | v{new_ver} V1k={best_v1_k} ts_short={best_short} ts_long={best_long} "
        f"OBJ={best_long_obj:.4f} LOYO={loyo_wins}/5")
    Path("configs/production_p91b_champion.json").write_text(json.dumps(cfg, indent=2))
    msg = (f"phase238: V1k+FTSwin -- VALIDATED V1k={best_v1_k} short={best_short} long={best_long} "
           f"LOYO {loyo_wins}/5 delta={delta:+.4f} OBJ={best_long_obj:.4f}")
    os.system(f"git add configs/production_p91b_champion.json scripts/run_phase238_v1k_fts_window_sweep.py && "
              f"git pull --rebase && git commit -m '{msg}' && git push")
else:
    flag = "WEAK" if (loyo_wins > 0 or delta > 0) else "CONFIRMED OPTIMAL"
    print(f"\n{'⚠️' if loyo_wins > 0 else '❌'} NO IMPROVEMENT — {flag} "
          f"(LOYO {loyo_wins}/5 delta={delta:+.4f})")
    msg = (f"phase238: V1k+FTSwin -- WEAK LOYO {loyo_wins}/5 delta={delta:+.4f} "
           f"V1k={best_v1_k} short={best_short} long={best_long}")
    os.system(f"git add scripts/run_phase238_v1k_fts_window_sweep.py && "
              f"git pull --rebase && git commit -m '{msg}' && git push")

print(f"  Done  ({elapsed:.1f}s)")
print(f"\n[DONE] Phase 238 complete.")
