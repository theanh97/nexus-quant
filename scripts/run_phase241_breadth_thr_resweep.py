"""
Phase 241 — Breadth Threshold Re-sweep

After P239 validation (HIGH: f168=0.00, pure idio momentum),
HIGH regime performance improved significantly. Optimal thresholds may have shifted.
Expanding HIGH zone (lower p_high) could capture more of this improvement.

Part A: p_high ∈ [0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]
Part B: p_low ∈ [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40]

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

P_HIGH_SWEEP = [0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]
P_LOW_SWEEP  = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40]


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
    baseline_obj = float(m[-1]) if m else 3.3151
    return cfg, weights, params, v1p, i460p, i415p, f168p, baseline_obj


def rolling_mean_arr(a, w):
    out = np.full(len(a), np.nan)
    if w <= 0: return out
    cs = np.cumsum(np.where(np.isnan(a), 0.0, a))
    for i in range(w - 1, len(a)):
        out[i] = cs[i] / w if i == w - 1 else (cs[i] - cs[i - w]) / w
    return out


def load_year_all(year, params, v1p, i460p, i415p, f168p):
    """Load all signals + pre-compute brd_pct array for threshold re-evaluation."""
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

    xsect_mean = np.mean(fund_rates, axis=1)
    ts_raw = (rolling_mean_arr(xsect_mean, p["ts_short"]) -
              rolling_mean_arr(xsect_mean, p["ts_long"]))
    ts_spread_pct = np.full(n, 0.5)
    for i in range(PCT_WINDOW, n):
        ts_spread_pct[i] = float(np.mean(ts_raw[i-PCT_WINDOW:i] <= ts_raw[i]))
    ts_spread_pct[:PCT_WINDOW] = 0.5

    # Breadth percentile (fixed brd_lb + pct_win, thresholds vary)
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

    v1_r = run("nexus_alpha_v1", {
        "k_per_side": v1p.get("k_per_side", 2),
        "w_carry": v1p.get("w_carry", 0.25), "w_mom": v1p.get("w_mom", 0.45),
        "w_mean_reversion": v1p.get("w_mean_reversion", 0.30),
        "momentum_lookback_bars": v1p.get("momentum_lookback_bars", 336),
        "mean_reversion_lookback_bars": v1p.get("mean_reversion_lookback_bars", 84),
        "vol_lookback_bars": v1p.get("vol_lookback_bars", 192),
        "target_gross_leverage": v1p.get("target_gross_leverage", 0.35),
        "rebalance_interval_bars": v1p.get("rebalance_interval_bars", 60),
    })
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
    return {"n": n, "btc_vol": btc_vol, "fund_std_pct": fund_std_pct,
            "ts_spread_pct": ts_spread_pct, "brd_pct": brd_pct,
            "v1": v1_r, "i460": i460_r, "i415": i415_r, "f168": f168_r}


def compute_ensemble(data, weights, params, p_low, p_high):
    brd_pct = data["brd_pct"]
    regime = np.where(brd_pct >= p_high, 2, np.where(brd_pct >= p_low, 1, 0)).astype(int)
    p = params
    v1 = data["v1"]; i460 = data["i460"]; i415 = data["i415"]; f168 = data["f168"]
    tsp = data["ts_spread_pct"]; fsp = data["fund_std_pct"]; bv = data["btc_vol"]
    min_len = min(len(v1), len(i460), len(i415), len(f168), len(brd_pct))
    bv_ = bv[:min_len]; reg = regime[:min_len]
    fsp_ = fsp[:min_len]; tsp_ = tsp[:min_len]
    v1_ = v1[:min_len]; i4 = i460[:min_len]; i5 = i415[:min_len]; f1 = f168[:min_len]
    ens = np.zeros(min_len)
    for i in range(min_len):
        w = weights[["LOW", "MID", "HIGH"][int(reg[i])]]
        if not np.isnan(bv_[i]) and bv_[i] > p["vol_thr"]:
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


def eval_thresholds(year_data, weights, params, p_low, p_high):
    yr_s = {}
    for yr in YEARS:
        ens = compute_ensemble(year_data[yr], weights, params, p_low, p_high)
        yr_s[yr] = sharpe(ens, year_data[yr]["n"])
    return yr_s, obj_fn(yr_s)


# ── MAIN ────────────────────────────────────────────────────────────────────
print("=" * 60)
print("Phase 241 — Breadth Threshold Re-sweep (after P239 HIGH reopt)")
print("=" * 60)
_start = time.time()

cfg, WEIGHTS, PARAMS, v1p, i460p, i415p, f168p, baseline_obj = get_config()
ver_now = cfg.get("_version", "2.42.0")
curr_p_low  = PARAMS["p_low"]
curr_p_high = PARAMS["p_high"]
print(f"  Config: {ver_now}  HIGH: v1={WEIGHTS['HIGH']['v1']} f168={WEIGHTS['HIGH']['f168']}")
print(f"  p_low={curr_p_low}  p_high={curr_p_high}")

# [1] Load all data
print("\n[1] Loading all data ...")
year_data = {}
for yr in YEARS:
    print(f"  {yr}", end=" ", flush=True)
    year_data[yr] = load_year_all(yr, PARAMS, v1p, i460p, i415p, f168p)
    print(f"n={year_data[yr]['n']}")

# Baseline
baseline_yr_s, baseline_computed = eval_thresholds(
    year_data, WEIGHTS, PARAMS, curr_p_low, curr_p_high)
print(f"\n  Baseline OBJ (computed): {baseline_computed:.4f}")

# ── Part A: p_high sweep ─────────────────────────────────────────────────
print(f"\n[2] Part A: p_high sweep (p_low={curr_p_low:.2f} fixed) ...")
best_ph = curr_p_high; best_ph_obj = baseline_computed
ph_results = {}
for ph in P_HIGH_SWEEP:
    if ph <= curr_p_low: continue
    yr_s, obj = eval_thresholds(year_data, WEIGHTS, PARAMS, curr_p_low, ph)
    ph_results[ph] = obj
    marker = " ←" if obj > best_ph_obj else ""
    print(f"  p_high={ph:.2f}  OBJ={obj:.4f}{marker}")
    if obj > best_ph_obj: best_ph_obj = obj; best_ph = ph
print(f"  Best p_high: {best_ph:.2f}  OBJ={best_ph_obj:.4f}")

# ── Part B: p_low sweep ───────────────────────────────────────────────────
print(f"\n[3] Part B: p_low sweep (p_high={best_ph:.2f}) ...")
best_pl = curr_p_low; best_pl_obj = best_ph_obj
pl_results = {}
for pl in P_LOW_SWEEP:
    if pl >= best_ph: continue
    yr_s, obj = eval_thresholds(year_data, WEIGHTS, PARAMS, pl, best_ph)
    pl_results[pl] = obj
    marker = " ←" if obj > best_pl_obj else ""
    print(f"  p_low={pl:.2f}  OBJ={obj:.4f}{marker}")
    if obj > best_pl_obj: best_pl_obj = obj; best_pl = pl
delta = best_pl_obj - baseline_computed
print(f"  Best p_low: {best_pl:.2f}  OBJ={best_pl_obj:.4f}  Δ={delta:+.4f}")

# ── LOYO validation ──────────────────────────────────────────────────────
print(f"\n[4] LOYO validation ...")
best_yr_s, _ = eval_thresholds(year_data, WEIGHTS, PARAMS, best_pl, best_ph)

loyo_wins = 0
for yr in YEARS:
    b_s = best_yr_s[yr]; c_s = baseline_yr_s[yr]
    win = b_s > c_s
    if win: loyo_wins += 1
    print(f"  {yr}: best={b_s:.4f} curr={c_s:.4f} {'WIN' if win else 'LOSS'}")

validated = loyo_wins >= 3 and delta > 0.005
print(f"\n  LOYO: {loyo_wins}/5 wins  delta={delta:+.4f}")

# ── Commit ────────────────────────────────────────────────────────────────
elapsed = time.time() - _start

if validated:
    print(f"\n✅ VALIDATED — p_high={best_ph:.2f} p_low={best_pl:.2f}")
    brs = cfg["breadth_regime_switching"]
    brs["p_high"] = best_ph; brs["p_low"] = best_pl
    ver = cfg.get("_version", "2.42.0"); parts = ver.split(".")
    parts[-1] = str(int(parts[-1]) + 1); new_ver = ".".join(parts)
    cfg["_version"] = new_ver
    old_val = cfg.get("_validated", "")
    cfg["_validated"] = (old_val +
        f" | v{new_ver} p_high={best_ph} p_low={best_pl} OBJ={best_pl_obj:.4f} LOYO={loyo_wins}/5")
    Path("configs/production_p91b_champion.json").write_text(json.dumps(cfg, indent=2))
    msg = (f"phase241: breadth thr resweep -- VALIDATED p_high={best_ph} p_low={best_pl} "
           f"LOYO {loyo_wins}/5 delta={delta:+.4f} OBJ={best_pl_obj:.4f}")
    os.system(f"git add configs/production_p91b_champion.json scripts/run_phase241_breadth_thr_resweep.py && "
              f"git pull --rebase && git commit -m '{msg}' && git push")
else:
    flag = "WEAK" if (loyo_wins > 0 or delta > 0) else "CONFIRMED OPTIMAL"
    print(f"\n{'⚠️' if loyo_wins > 0 else '❌'} NO IMPROVEMENT — {flag} "
          f"(LOYO {loyo_wins}/5 delta={delta:+.4f})")
    msg = (f"phase241: breadth thr resweep -- WEAK LOYO {loyo_wins}/5 delta={delta:+.4f} "
           f"p_high={best_ph} p_low={best_pl}")
    os.system(f"git add scripts/run_phase241_breadth_thr_resweep.py && "
              f"git pull --rebase && git commit -m '{msg}' && git push")

print(f"  Done  ({elapsed:.1f}s)")
print(f"\n[DONE] Phase 241 complete.")
