"""
Phase 234 — k_per_side sweep for I460, I415, F168

Current: I460 k=4, I415 k=4, F168 k=2
Part A: I460 k ∈ [2, 3, 4, 5, 6]  (I415/F168/V1 fixed)
Part B: I415 k ∈ [2, 3, 4, 5, 6]  (best I460 k, F168/V1 fixed)
Part C: F168 k ∈ [1, 2, 3, 4]      (best I460+I415 k, V1 fixed)

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
    out = Path("artifacts/phase234"); out.mkdir(parents=True, exist_ok=True)
    (out / "phase234_report.json").write_text(json.dumps(_partial, indent=2, default=str))
    sys.exit(0)
_signal.signal(_signal.SIGALRM, _on_timeout)
_signal.alarm(10800)  # 3hr timeout

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

I460_K_SWEEP = [2, 3, 4, 5, 6]
I415_K_SWEEP = [2, 3, 4, 5, 6]
F168_K_SWEEP = [1, 2, 3, 4]


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


def load_year_base(year, params):
    """Load all signals + auxiliary data for a year. Returns dict."""
    p = params
    s, e = YEAR_RANGES[year]
    cfg_d = {"provider": "binance_rest_v1", "symbols": SYMBOLS,
             "start": s, "end": e, "bar_interval": "1h", "cache_dir": ".cache/binance_rest"}
    dataset = make_provider(cfg_d, seed=42).load()
    n = len(dataset.timeline)

    close_mat = np.zeros((n, len(SYMBOLS)))
    for j, sym in enumerate(SYMBOLS):
        for i in range(n):
            close_mat[i, j] = dataset.close(sym, i)

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

    xsect_mean = np.mean(fund_rates, axis=1)
    ts_raw = (rolling_mean_arr(xsect_mean, p["ts_short"]) -
              rolling_mean_arr(xsect_mean, p["ts_long"]))
    ts_spread_pct = np.full(n, 0.5)
    for i in range(PCT_WINDOW, n):
        ts_spread_pct[i] = float(np.mean(ts_raw[i-PCT_WINDOW:i] <= ts_raw[i]))
    ts_spread_pct[:PCT_WINDOW] = 0.5

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

    return {
        "dataset": dataset,
        "btc_vol": btc_vol,
        "fund_std_pct": fund_std_pct,
        "ts_spread_pct": ts_spread_pct,
        "brd_pct": brd_pct,
        "n": n,
    }


def run_signal(dataset, name, sp):
    res = BacktestEngine(BacktestConfig(costs=COST_MODEL)).run(
        dataset, make_strategy({"name": name, "params": sp}))
    return np.array(res.returns)


def compute_ensemble(base, v1_rets, i460_rets, i415_rets, f168_rets, weights, params):
    p = params
    brd_pct = base["brd_pct"]
    p_low = p["p_low"]; p_high = p["p_high"]
    regime = np.where(brd_pct >= p_high, 2, np.where(brd_pct >= p_low, 1, 0)).astype(int)
    min_len = min(len(v1_rets), len(i460_rets), len(i415_rets), len(f168_rets), len(brd_pct))
    bv = base["btc_vol"][:min_len]
    fsp = base["fund_std_pct"][:min_len]
    tsp = base["ts_spread_pct"][:min_len]
    reg = regime[:min_len]
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
print("Phase 234 — k_per_side sweep (I460, I415, F168)")
print("=" * 60)
_start = time.time()

cfg, WEIGHTS, PARAMS, v1p, i460p, i415p, f168p, baseline_obj = get_config()
ver_now = cfg.get("_version", "2.40.0")
curr_i460_k = i460p.get("k_per_side", 4)
curr_i415_k = i415p.get("k_per_side", 4)
curr_f168_k = f168p.get("k_per_side", 2)
print(f"  Config: {ver_now}  baseline_obj(str)={baseline_obj:.4f}")
print(f"  I460 k={curr_i460_k}  I415 k={curr_i415_k}  F168 k={curr_f168_k}")

# [1] Load base data (dataset + aux) for all years
print("\n[1] Loading dataset + auxiliary data for all years ...")
base_data = {}
for yr in YEARS:
    print(f"  {yr}", end=" ", flush=True)
    base_data[yr] = load_year_base(yr, PARAMS)
    print(f"n={base_data[yr]['n']}")

# [2] Pre-load baseline signals (current k values) for all years
print("\n[2] Loading baseline signals (current k values) ...")
baseline_sigs = {}
for yr in YEARS:
    ds = base_data[yr]["dataset"]
    v1_r  = run_signal(ds, "nexus_alpha_v1", {
        "k_per_side": v1p.get("k_per_side", 2),
        "w_carry": v1p.get("w_carry", 0.25), "w_mom": v1p.get("w_mom", 0.45),
        "w_mean_reversion": v1p.get("w_mean_reversion", 0.30),
        "momentum_lookback_bars": v1p.get("momentum_lookback_bars", 336),
        "mean_reversion_lookback_bars": v1p.get("mean_reversion_lookback_bars", 84),
        "vol_lookback_bars": v1p.get("vol_lookback_bars", 192),
        "target_gross_leverage": v1p.get("target_gross_leverage", 0.35),
        "rebalance_interval_bars": v1p.get("rebalance_interval_bars", 60),
    })
    i460_r = run_signal(ds, "idio_momentum_alpha", {
        "k_per_side": curr_i460_k,
        "lookback_bars": i460p.get("lookback_bars", 480),
        "beta_window_bars": i460p.get("beta_window_bars", 168),
        "target_gross_leverage": i460p.get("target_gross_leverage", 0.20),
        "rebalance_interval_bars": i460p.get("rebalance_interval_bars", 48),
    })
    i415_r = run_signal(ds, "idio_momentum_alpha", {
        "k_per_side": curr_i415_k,
        "lookback_bars": i415p.get("lookback_bars", 415),
        "beta_window_bars": i415p.get("beta_window_bars", 144),
        "target_gross_leverage": i415p.get("target_gross_leverage", 0.20),
        "rebalance_interval_bars": i415p.get("rebalance_interval_bars", 48),
    })
    f168_r = run_signal(ds, "funding_momentum_alpha", {
        "k_per_side": curr_f168_k,
        "funding_lookback_bars": f168p.get("funding_lookback_bars", 168),
        "direction": f168p.get("direction", "contrarian"),
        "target_gross_leverage": f168p.get("target_gross_leverage", 0.25),
        "rebalance_interval_bars": f168p.get("rebalance_interval_bars", 36),
    })
    baseline_sigs[yr] = {"v1": v1_r, "i460": i460_r, "i415": i415_r, "f168": f168_r}
    print(f"  {yr} baseline loaded")

# Baseline OBJ
baseline_yr_s = {}
for yr in YEARS:
    b = base_data[yr]; s = baseline_sigs[yr]
    ens = compute_ensemble(b, s["v1"], s["i460"], s["i415"], s["f168"], WEIGHTS, PARAMS)
    baseline_yr_s[yr] = sharpe(ens, b["n"])
baseline_computed = obj_fn(baseline_yr_s)
print(f"\n  Baseline OBJ (computed): {baseline_computed:.4f}")

# ── Part A: I460 k sweep ─────────────────────────────────────────────────
print(f"\n[3] Part A: I460 k sweep (I415 k={curr_i415_k}, F168 k={curr_f168_k}) ...")
best_i460_k = curr_i460_k
best_i460_obj = baseline_computed
i460_k_results = {}
for k in I460_K_SWEEP:
    yr_s = {}
    for yr in YEARS:
        ds = base_data[yr]["dataset"]; b = base_data[yr]; s = baseline_sigs[yr]
        i460_new = run_signal(ds, "idio_momentum_alpha", {
            "k_per_side": k,
            "lookback_bars": i460p.get("lookback_bars", 480),
            "beta_window_bars": i460p.get("beta_window_bars", 168),
            "target_gross_leverage": i460p.get("target_gross_leverage", 0.20),
            "rebalance_interval_bars": i460p.get("rebalance_interval_bars", 48),
        })
        ens = compute_ensemble(b, s["v1"], i460_new, s["i415"], s["f168"], WEIGHTS, PARAMS)
        yr_s[yr] = sharpe(ens, b["n"])
    obj = obj_fn(yr_s)
    i460_k_results[k] = obj
    marker = " ←" if obj > best_i460_obj else ""
    print(f"  I460 k={k}  OBJ={obj:.4f}{marker}")
    if obj > best_i460_obj:
        best_i460_obj = obj
        best_i460_k = k
print(f"  Best I460 k: {best_i460_k}  OBJ={best_i460_obj:.4f}")

# Re-run I460 with best k for all years
best_i460_rets = {}
for yr in YEARS:
    ds = base_data[yr]["dataset"]
    best_i460_rets[yr] = run_signal(ds, "idio_momentum_alpha", {
        "k_per_side": best_i460_k,
        "lookback_bars": i460p.get("lookback_bars", 480),
        "beta_window_bars": i460p.get("beta_window_bars", 168),
        "target_gross_leverage": i460p.get("target_gross_leverage", 0.20),
        "rebalance_interval_bars": i460p.get("rebalance_interval_bars", 48),
    })

# ── Part B: I415 k sweep ─────────────────────────────────────────────────
print(f"\n[4] Part B: I415 k sweep (I460 k={best_i460_k}, F168 k={curr_f168_k}) ...")
best_i415_k = curr_i415_k
best_i415_obj = best_i460_obj
i415_k_results = {}
for k in I415_K_SWEEP:
    yr_s = {}
    for yr in YEARS:
        ds = base_data[yr]["dataset"]; b = base_data[yr]; s = baseline_sigs[yr]
        i415_new = run_signal(ds, "idio_momentum_alpha", {
            "k_per_side": k,
            "lookback_bars": i415p.get("lookback_bars", 415),
            "beta_window_bars": i415p.get("beta_window_bars", 144),
            "target_gross_leverage": i415p.get("target_gross_leverage", 0.20),
            "rebalance_interval_bars": i415p.get("rebalance_interval_bars", 48),
        })
        ens = compute_ensemble(
            b, s["v1"], best_i460_rets[yr], i415_new, s["f168"], WEIGHTS, PARAMS)
        yr_s[yr] = sharpe(ens, b["n"])
    obj = obj_fn(yr_s)
    i415_k_results[k] = obj
    marker = " ←" if obj > best_i415_obj else ""
    print(f"  I415 k={k}  OBJ={obj:.4f}{marker}")
    if obj > best_i415_obj:
        best_i415_obj = obj
        best_i415_k = k
print(f"  Best I415 k: {best_i415_k}  OBJ={best_i415_obj:.4f}")

# Re-run I415 with best k for all years
best_i415_rets = {}
for yr in YEARS:
    ds = base_data[yr]["dataset"]
    best_i415_rets[yr] = run_signal(ds, "idio_momentum_alpha", {
        "k_per_side": best_i415_k,
        "lookback_bars": i415p.get("lookback_bars", 415),
        "beta_window_bars": i415p.get("beta_window_bars", 144),
        "target_gross_leverage": i415p.get("target_gross_leverage", 0.20),
        "rebalance_interval_bars": i415p.get("rebalance_interval_bars", 48),
    })

# ── Part C: F168 k sweep ─────────────────────────────────────────────────
print(f"\n[5] Part C: F168 k sweep (I460 k={best_i460_k}, I415 k={best_i415_k}) ...")
best_f168_k = curr_f168_k
best_f168_obj = best_i415_obj
f168_k_results = {}
for k in F168_K_SWEEP:
    yr_s = {}
    for yr in YEARS:
        ds = base_data[yr]["dataset"]; b = base_data[yr]; s = baseline_sigs[yr]
        f168_new = run_signal(ds, "funding_momentum_alpha", {
            "k_per_side": k,
            "funding_lookback_bars": f168p.get("funding_lookback_bars", 168),
            "direction": f168p.get("direction", "contrarian"),
            "target_gross_leverage": f168p.get("target_gross_leverage", 0.25),
            "rebalance_interval_bars": f168p.get("rebalance_interval_bars", 36),
        })
        ens = compute_ensemble(
            b, s["v1"], best_i460_rets[yr], best_i415_rets[yr], f168_new, WEIGHTS, PARAMS)
        yr_s[yr] = sharpe(ens, b["n"])
    obj = obj_fn(yr_s)
    f168_k_results[k] = obj
    marker = " ←" if obj > best_f168_obj else ""
    print(f"  F168 k={k}  OBJ={obj:.4f}{marker}")
    if obj > best_f168_obj:
        best_f168_obj = obj
        best_f168_k = k
delta = best_f168_obj - baseline_computed
print(f"  Best F168 k: {best_f168_k}  OBJ={best_f168_obj:.4f}  Δ={delta:+.4f}")

# ── LOYO validation ──────────────────────────────────────────────────────
print(f"\n[6] LOYO validation ...")
# Re-run F168 with best k
best_f168_rets = {}
for yr in YEARS:
    ds = base_data[yr]["dataset"]
    best_f168_rets[yr] = run_signal(ds, "funding_momentum_alpha", {
        "k_per_side": best_f168_k,
        "funding_lookback_bars": f168p.get("funding_lookback_bars", 168),
        "direction": f168p.get("direction", "contrarian"),
        "target_gross_leverage": f168p.get("target_gross_leverage", 0.25),
        "rebalance_interval_bars": f168p.get("rebalance_interval_bars", 36),
    })
best_yr_s = {}
for yr in YEARS:
    b = base_data[yr]; s = baseline_sigs[yr]
    ens = compute_ensemble(
        b, s["v1"], best_i460_rets[yr], best_i415_rets[yr], best_f168_rets[yr], WEIGHTS, PARAMS)
    best_yr_s[yr] = sharpe(ens, b["n"])

loyo_wins = 0
for yr in YEARS:
    b = best_yr_s[yr]; c = baseline_yr_s[yr]
    win = b > c
    if win: loyo_wins += 1
    print(f"  {yr}: best={b:.4f} curr={c:.4f} {'WIN' if win else 'LOSS'}")

validated = loyo_wins >= 3 and delta > 0.005
print(f"\n  LOYO: {loyo_wins}/5 wins  delta={delta:+.4f}")

# ── Save + commit ────────────────────────────────────────────────────────
elapsed = time.time() - _start
out = Path("artifacts/phase234"); out.mkdir(parents=True, exist_ok=True)
report = {
    "phase": 234, "elapsed_s": elapsed,
    "best_i460_k": best_i460_k, "best_i415_k": best_i415_k, "best_f168_k": best_f168_k,
    "best_obj": best_f168_obj, "delta": delta,
    "loyo_wins": loyo_wins, "validated": validated,
    "i460_k_results": {str(k): v for k, v in i460_k_results.items()},
    "i415_k_results": {str(k): v for k, v in i415_k_results.items()},
    "f168_k_results": {str(k): v for k, v in f168_k_results.items()},
}
(out / "phase234_report.json").write_text(json.dumps(report, indent=2))
_partial.update(report)

if validated:
    print(f"\n✅ VALIDATED — updating config k: I460={best_i460_k} I415={best_i415_k} F168={best_f168_k}")
    sigs = cfg["ensemble"]["signals"]
    sigs["i460bw168"]["params"]["k_per_side"] = best_i460_k
    sigs["i415bw216"]["params"]["k_per_side"] = best_i415_k
    f168_key = "f144" if "f144" in sigs else "f168"
    sigs[f168_key]["params"]["k_per_side"] = best_f168_k
    ver = cfg.get("_version", "2.40.0"); parts = ver.split(".")
    parts[-1] = str(int(parts[-1]) + 1); new_ver = ".".join(parts)
    cfg["_version"] = new_ver
    old_val = cfg.get("_validated", "")
    cfg["_validated"] = (old_val +
        f" | v{new_ver} k_sweep I460k={best_i460_k} I415k={best_i415_k} F168k={best_f168_k} "
        f"OBJ={best_f168_obj:.4f} LOYO={loyo_wins}/5")
    Path("configs/production_p91b_champion.json").write_text(json.dumps(cfg, indent=2))
    msg = (f"phase234: k_per_side sweep -- VALIDATED I460k={best_i460_k} I415k={best_i415_k} "
           f"F168k={best_f168_k} LOYO {loyo_wins}/5 delta={delta:+.4f} OBJ={best_f168_obj:.4f}")
    os.system(f"git add configs/production_p91b_champion.json artifacts/phase234/ && "
              f"git stash && git pull --rebase && git stash pop && "
              f"git commit -m '{msg}' && git push")
else:
    flag = "WEAK" if (loyo_wins > 0 or delta > 0) else "CONFIRMED OPTIMAL"
    print(f"\n{'⚠️' if loyo_wins > 0 else '❌'} NO IMPROVEMENT — {flag} "
          f"(LOYO {loyo_wins}/5 delta={delta:+.4f})")
    msg = (f"phase234: k_per_side sweep -- WEAK LOYO {loyo_wins}/5 delta={delta:+.4f} "
           f"I460k={best_i460_k} I415k={best_i415_k} F168k={best_f168_k}")
    os.system(f"git add artifacts/phase234/ && "
              f"git stash && git pull --rebase && git stash pop && "
              f"git commit -m '{msg}' && git push")

print(f"  Report saved  ({elapsed:.1f}s)")
print(f"\n[DONE] Phase 234 complete.")
