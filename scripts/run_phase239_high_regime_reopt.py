"""
Phase 239 — HIGH regime weight re-optimization

After v2.41.0 weight changes (parallel session), HIGH regime now has:
  v1=0.18, f168=0.03, i460+i415=0.79
Previous was: v1=0.04, f168=0.18, i460+i415=0.78

Re-sweep HIGH weights with current config to confirm or improve.
Keep i460:i415 ratio fixed (0.357:0.643 = I460/(I460+I415)).

Part A: idio_total ∈ [0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]
  f168 = max(0, 1 - v1 - idio_total)  [v1 from Part B best]
  [First pass: v1=current, idio_total varies]

Part B: v1 ∈ [0.02, 0.04, 0.06, 0.10, 0.15, 0.18, 0.22]
  idio_total = best from Part A

Part C: f168 ∈ [0.00, 0.03, 0.06, 0.09, 0.12, 0.15, 0.18, 0.21]
  v1 = best, idio_total = best (renormalize)

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

# I460:I415 ratio from current config (will be read dynamically)
IDIO_RATIO = 0.357  # i460 / (i460 + i415)

HIGH_IDIO_SWEEP = [0.55, 0.60, 0.65, 0.70, 0.75, 0.78, 0.80, 0.85, 0.90]
HIGH_V1_SWEEP   = [0.02, 0.04, 0.06, 0.10, 0.14, 0.18, 0.22, 0.26]
HIGH_F168_SWEEP = [0.00, 0.03, 0.06, 0.09, 0.12, 0.15, 0.18, 0.21]


def get_config():
    cfg = json.loads(Path("configs/production_p91b_champion.json").read_text())
    brs = cfg["breadth_regime_switching"]
    rw = brs["regime_weights"]
    base_weights = {}
    for regime in ["LOW", "MID", "HIGH"]:
        w = rw[regime]
        base_weights[regime] = {
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

    # Compute current idio ratio from HIGH weights
    hi = base_weights["HIGH"]
    idio_sum = hi["i460"] + hi["i415"]
    idio_ratio = hi["i460"] / idio_sum if idio_sum > 0 else IDIO_RATIO

    m = re.findall(r"OBJ=([\d.]+)", cfg.get("_validated", ""))
    baseline_obj = float(m[-1]) if m else 3.2766
    return cfg, base_weights, params, v1p, i460p, i415p, f168p, idio_ratio, baseline_obj


def rolling_mean_arr(a, w):
    out = np.full(len(a), np.nan)
    if w <= 0: return out
    cs = np.cumsum(np.where(np.isnan(a), 0.0, a))
    for i in range(w - 1, len(a)):
        out[i] = cs[i] / w if i == w - 1 else (cs[i] - cs[i - w]) / w
    return out


def load_year_all(year, params, v1p, i460p, i415p, f168p):
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
    return {"dataset": dataset, "n": n,
            "btc_vol": btc_vol, "fund_std_pct": fund_std_pct,
            "ts_spread_pct": ts_spread_pct, "brd_pct": brd_pct,
            "v1": v1_r, "i460": i460_r, "i415": i415_r, "f168": f168_r}


def make_high_weights(base, v1_w, idio_total, f168_w, idio_ratio):
    """Build weights dict with custom HIGH regime params."""
    i460_w = idio_total * idio_ratio
    i415_w = idio_total * (1 - idio_ratio)
    w = {
        "LOW":  dict(base["LOW"]),
        "MID":  dict(base["MID"]),
        "HIGH": {"v1": v1_w, "i460": i460_w, "i415": i415_w, "f168": f168_w},
    }
    return w


def compute_ensemble(data, weights, params):
    p = params
    brd_pct = data["brd_pct"]
    p_low = p["p_low"]; p_high = p["p_high"]
    regime = np.where(brd_pct >= p_high, 2, np.where(brd_pct >= p_low, 1, 0)).astype(int)
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


def eval_high(year_data, base_weights, params, v1_w, idio_total, f168_w, idio_ratio):
    w = make_high_weights(base_weights, v1_w, idio_total, f168_w, idio_ratio)
    yr_s = {}
    for yr in YEARS:
        ens = compute_ensemble(year_data[yr], w, params)
        yr_s[yr] = sharpe(ens, year_data[yr]["n"])
    return yr_s, obj_fn(yr_s)


# ── MAIN ────────────────────────────────────────────────────────────────────
print("=" * 60)
print("Phase 239 — HIGH regime weight re-optimization")
print("=" * 60)
_start = time.time()

cfg, base_weights, PARAMS, v1p, i460p, i415p, f168p, IDIO_RATIO, baseline_obj = get_config()
ver_now = cfg.get("_version", "2.41.0")
hi = base_weights["HIGH"]
curr_v1_w    = hi["v1"]
curr_i460_w  = hi["i460"]
curr_i415_w  = hi["i415"]
curr_f168_w  = hi["f168"]
curr_idio    = curr_i460_w + curr_i415_w
print(f"  Config: {ver_now}  idio_ratio={IDIO_RATIO:.3f}")
print(f"  HIGH: v1={curr_v1_w} idio={curr_idio:.4f} f168={curr_f168_w}")

# [1] Load all year data (all 4 signals)
print("\n[1] Loading all data ...")
year_data = {}
for yr in YEARS:
    print(f"  {yr}", end=" ", flush=True)
    year_data[yr] = load_year_all(yr, PARAMS, v1p, i460p, i415p, f168p)
    print(f"n={year_data[yr]['n']}")

# Baseline
baseline_yr_s, baseline_computed = eval_high(
    year_data, base_weights, PARAMS, curr_v1_w, curr_idio, curr_f168_w, IDIO_RATIO)
print(f"\n  Baseline OBJ (computed): {baseline_computed:.4f}")

# ── Part A: idio_total sweep (v1=curr, f168 = 1 - v1 - idio) ─────────────
print(f"\n[2] Part A: HIGH idio_total sweep (v1={curr_v1_w}) ...")
best_idio = curr_idio
best_idio_obj = baseline_computed
idio_results = {}
for it in HIGH_IDIO_SWEEP:
    f_w = max(0.0, 1.0 - curr_v1_w - it)
    yr_s, obj = eval_high(year_data, base_weights, PARAMS, curr_v1_w, it, f_w, IDIO_RATIO)
    idio_results[it] = obj
    marker = " ←" if obj > best_idio_obj else ""
    print(f"  idio={it:.2f}  f168={f_w:.3f}  OBJ={obj:.4f}{marker}")
    if obj > best_idio_obj:
        best_idio_obj = obj; best_idio = it
best_f168_a = max(0.0, 1.0 - curr_v1_w - best_idio)
print(f"  Best idio: {best_idio:.2f}  f168={best_f168_a:.3f}  OBJ={best_idio_obj:.4f}")

# ── Part B: v1 sweep (idio=best, f168 = 1 - v1 - idio) ──────────────────
print(f"\n[3] Part B: HIGH v1 sweep (idio={best_idio:.2f}) ...")
best_v1 = curr_v1_w
best_v1_obj = best_idio_obj
v1_results = {}
for v1w in HIGH_V1_SWEEP:
    f_w = max(0.0, 1.0 - v1w - best_idio)
    yr_s, obj = eval_high(year_data, base_weights, PARAMS, v1w, best_idio, f_w, IDIO_RATIO)
    v1_results[v1w] = obj
    marker = " ←" if obj > best_v1_obj else ""
    print(f"  v1={v1w:.2f}  f168={f_w:.3f}  OBJ={obj:.4f}{marker}")
    if obj > best_v1_obj:
        best_v1_obj = obj; best_v1 = v1w
best_f168_b = max(0.0, 1.0 - best_v1 - best_idio)
print(f"  Best v1: {best_v1:.2f}  f168={best_f168_b:.3f}  OBJ={best_v1_obj:.4f}")

# ── Part C: f168 sweep (v1=best, idio adjusted) ──────────────────────────
print(f"\n[4] Part C: HIGH f168 sweep (v1={best_v1:.2f}, idio adjusted) ...")
best_f168 = best_f168_b
best_f168_obj = best_v1_obj
f168_results = {}
for fw in HIGH_F168_SWEEP:
    # Renormalize: keep idio_total = best_idio, adjust nothing; just set f168 and rescale
    # v1 + idio + f168 = 1 → idio = 1 - v1 - f168 (clipped to 0)
    idio_adj = max(0.0, 1.0 - best_v1 - fw)
    yr_s, obj = eval_high(year_data, base_weights, PARAMS, best_v1, idio_adj, fw, IDIO_RATIO)
    f168_results[fw] = obj
    marker = " ←" if obj > best_f168_obj else ""
    print(f"  f168={fw:.2f}  idio={idio_adj:.3f}  OBJ={obj:.4f}{marker}")
    if obj > best_f168_obj:
        best_f168_obj = obj; best_f168 = fw
best_idio_final = max(0.0, 1.0 - best_v1 - best_f168)
delta = best_f168_obj - baseline_computed
print(f"  Best f168: {best_f168:.2f}  idio={best_idio_final:.3f}  OBJ={best_f168_obj:.4f}  Δ={delta:+.4f}")

# ── LOYO validation ──────────────────────────────────────────────────────
print(f"\n[5] LOYO validation ...")
best_yr_s, _ = eval_high(
    year_data, base_weights, PARAMS, best_v1, best_idio_final, best_f168, IDIO_RATIO)

loyo_wins = 0
for yr in YEARS:
    b_s = best_yr_s[yr]; c_s = baseline_yr_s[yr]
    win = b_s > c_s
    if win: loyo_wins += 1
    print(f"  {yr}: best={b_s:.4f} curr={c_s:.4f} {'WIN' if win else 'LOSS'}")

validated = loyo_wins >= 3 and delta > 0.005
print(f"\n  LOYO: {loyo_wins}/5 wins  delta={delta:+.4f}")
print(f"  Best HIGH: v1={best_v1} idio={best_idio_final:.4f} f168={best_f168}")

# ── Commit ────────────────────────────────────────────────────────────────
elapsed = time.time() - _start

if validated:
    print(f"\n✅ VALIDATED — updating HIGH weights v1={best_v1} idio={best_idio_final:.4f} f168={best_f168}")
    brs = cfg["breadth_regime_switching"]
    rw = brs["regime_weights"]
    i460_new = best_idio_final * IDIO_RATIO
    i415_new = best_idio_final * (1 - IDIO_RATIO)
    rw["HIGH"]["v1"] = best_v1
    rw["HIGH"]["f168"] = best_f168
    rw["HIGH"]["i460bw168"] = round(i460_new, 6)
    rw["HIGH"]["i415bw216"] = round(i415_new, 6)
    ver = cfg.get("_version", "2.41.0"); parts = ver.split(".")
    parts[-1] = str(int(parts[-1]) + 1); new_ver = ".".join(parts)
    cfg["_version"] = new_ver
    old_val = cfg.get("_validated", "")
    cfg["_validated"] = (old_val +
        f" | v{new_ver} HIGH v1={best_v1} idio={best_idio_final:.3f} f168={best_f168} "
        f"OBJ={best_f168_obj:.4f} LOYO={loyo_wins}/5")
    Path("configs/production_p91b_champion.json").write_text(json.dumps(cfg, indent=2))
    msg = (f"phase239: HIGH reopt -- VALIDATED v1={best_v1} idio={best_idio_final:.3f} "
           f"f168={best_f168} LOYO {loyo_wins}/5 delta={delta:+.4f} OBJ={best_f168_obj:.4f}")
    os.system(f"git add configs/production_p91b_champion.json scripts/run_phase239_high_regime_reopt.py && "
              f"git pull --rebase && git commit -m '{msg}' && git push")
else:
    flag = "WEAK" if (loyo_wins > 0 or delta > 0) else "CONFIRMED OPTIMAL"
    print(f"\n{'⚠️' if loyo_wins > 0 else '❌'} NO IMPROVEMENT — {flag} "
          f"(LOYO {loyo_wins}/5 delta={delta:+.4f})")
    msg = (f"phase239: HIGH reopt -- WEAK LOYO {loyo_wins}/5 delta={delta:+.4f} "
           f"v1={best_v1} idio={best_idio_final:.3f} f168={best_f168}")
    os.system(f"git add scripts/run_phase239_high_regime_reopt.py && "
              f"git pull --rebase && git commit -m '{msg}' && git push")

print(f"  Done  ({elapsed:.1f}s)")
print(f"\n[DONE] Phase 239 complete.")
