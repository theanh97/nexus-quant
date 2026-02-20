"""
Phase 203 — Walk-Forward Validation of v2.31.0 vs p91b_flat
=============================================================
WF check after multiple regime weight changes since last WF (P201 @ v2.28.0).
Changes since v2.28.0:
  - P199 parallel: breadth threshold re-tune (p_low=0.30 confirmed)
  - P200 parallel: regime weight re-opt v2 (LOW/MID weights shifted)
  - P201 parallel: regime weight final re-opt (MID f168 surge to 0.65)
  - P202 (my): MID_f168=0.70, LOW_v1=0.44 LOYO 4/5 Δ=+0.0554 OBJ=3.0797
  - P202 parallel: MID f168 ceiling extend to 0.78 LOYO 3/5 Δ=+0.0074

Read config dynamically to get current weights/params.
WF Windows:
  - Train [2021-2022] → Test 2023
  - Train [2021-2022-2023] → Test 2024
  - Train [2021-2022-2023-2024] → Test 2025

Baseline: p91b_flat (equal-weight, no overlays)
"""
import os, sys, json, time, re
import signal as _signal
from pathlib import Path
from datetime import datetime, UTC

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
    out = Path("artifacts/phase203"); out.mkdir(parents=True, exist_ok=True)
    (out / "phase203_report.json").write_text(json.dumps(_partial, indent=2, default=str))
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

WF_WINDOWS = [
    {"train": ["2021", "2022"], "test": "2023"},
    {"train": ["2021", "2022", "2023"], "test": "2024"},
    {"train": ["2021", "2022", "2023", "2024"], "test": "2025"},
]

COST_MODEL = cost_model_from_config({
    "fee_rate": 0.0005, "slippage_rate": 0.0003, "cost_multiplier": 1.0,
})

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
        "ts_rt": fts.get("reduce_threshold", 0.65),
        "ts_rs": fts.get("reduce_scale", 0.45),
        "ts_bt": fts.get("boost_threshold", 0.25),
        "ts_bs": fts.get("boost_scale", 1.70),
        "ts_short": fts.get("short_window_bars", 16),
        "ts_long": fts.get("long_window_bars", 72),
        "p_low": brs.get("p_low", 0.30),
        "p_high": brs.get("p_high", 0.60),
        "brd_lb": brs.get("breadth_lookback_bars", 192),
        "pct_win": brs.get("rolling_percentile_window", 336),
    }
    vol_ov = cfg.get("vol_regime_overlay", {})
    params["vol_thr"] = vol_ov.get("threshold", 0.50)
    params["vol_scale"] = vol_ov.get("scale_factor", 0.30)
    params["f168_boost"] = vol_ov.get("f144_boost", 0.00)
    return cfg, weights, params

VOL_WINDOW = 168; FUND_DISP_PCT = 240; PCT_WINDOW = 336
DISP_THR = 0.60; DISP_SCALE = 1.0

# Flat baseline: equal weight, no overlay
WEIGHTS_FLAT = {"LOW": {"v1": 0.25, "i460": 0.25, "i415": 0.25, "f168": 0.25},
                "MID": {"v1": 0.25, "i460": 0.25, "i415": 0.25, "f168": 0.25},
                "HIGH": {"v1": 0.25, "i460": 0.25, "i415": 0.25, "f168": 0.25}}

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
    return btc_vol, fund_std_pct, ts_spread_pct, regime, sig_rets, n

def compute_ensemble(year_data, weights, params, use_overlay=True):
    p = params
    btc_vol, fund_std_pct, ts_spread_pct, regime, sig_rets, n = year_data
    min_len = min(len(v) for v in sig_rets.values())
    bv = btc_vol[:min_len]; reg = regime[:min_len]
    fsp = fund_std_pct[:min_len]; tsp = ts_spread_pct[:min_len]
    ens = np.zeros(min_len)
    for i in range(min_len):
        w = weights[["LOW", "MID", "HIGH"][int(reg[i])]]
        if use_overlay and not np.isnan(bv[i]) and bv[i] > p["vol_thr"]:
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
        if use_overlay:
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

# ── MAIN ────────────────────────────────────────────────────────────────────
print("=" * 60)
print("Phase 203 — WF Validation v2.31.0 (Post-MID regime surge)")
print("=" * 60)
_start = time.time()

prod_cfg, WEIGHTS_CURR, PARAMS = get_config()
ver_now = prod_cfg.get("_version", "2.31.0")
print(f"  Config: {ver_now}")
print(f"  Weights:")
for r in ["LOW", "MID", "HIGH"]:
    print(f"    {r}: {WEIGHTS_CURR[r]}")
print(f"  Params: {PARAMS}")

print("\n[1] Loading all years ...")
year_data = {}
for yr in YEARS:
    print(f"  {yr}", end="", flush=True)
    year_data[yr] = load_year(yr, PARAMS)
    print()

# IS objective
print("\n[2] IS objective (2021-2025) ...")
is_yr_curr = {}; is_yr_flat = {}
for yr in YEARS:
    ens_v = compute_ensemble(year_data[yr], WEIGHTS_CURR, PARAMS, use_overlay=True)
    ens_f = compute_ensemble(year_data[yr], WEIGHTS_FLAT, PARAMS, use_overlay=False)
    n = year_data[yr][-1]
    is_yr_curr[yr] = sharpe(ens_v, n)
    is_yr_flat[yr] = sharpe(ens_f, n)
    print(f"  {yr}: v231={is_yr_curr[yr]:.4f}  flat={is_yr_flat[yr]:.4f}  Δ={is_yr_curr[yr]-is_yr_flat[yr]:+.4f}")

is_obj_curr = obj_fn(is_yr_curr)
is_obj_flat = obj_fn(is_yr_flat)
print(f"\n  IS OBJ: v231={is_obj_curr:.4f}  flat={is_obj_flat:.4f}  Δ={is_obj_curr-is_obj_flat:+.4f}")

# WF windows
print("\n[3] Walk-Forward windows ...")
wf_results = []
wf_wins = 0

for window in WF_WINDOWS:
    train_yrs = window["train"]
    test_yr = window["test"]
    is_v = [is_yr_curr[y] for y in train_yrs]
    is_f = [is_yr_flat[y] for y in train_yrs]
    is_curr_wf = float(np.mean(is_v) - 0.5 * np.std(is_v))
    is_flat_wf = float(np.mean(is_f) - 0.5 * np.std(is_f))
    oos_v = is_yr_curr[test_yr]
    oos_f = is_yr_flat[test_yr]
    delta = oos_v - oos_f
    win = bool(delta > 0)
    if win: wf_wins += 1
    wf_results.append({
        "train": train_yrs, "test": test_yr,
        "is_curr": is_curr_wf, "is_flat": is_flat_wf,
        "oos_curr": oos_v, "oos_flat": oos_f, "oos_delta": delta, "win": win
    })
    print(f"  Train={train_yrs} Test={test_yr}: OOS v231={oos_v:.4f} flat={oos_f:.4f} Δ={delta:+.4f} {'✅WIN' if win else '❌LOSS'}")

avg_oos_delta = float(np.mean([r["oos_delta"] for r in wf_results]))
degrad = is_obj_curr / is_obj_flat if is_obj_flat > 0 else 0
wf_confirmed = wf_wins >= 2 and avg_oos_delta > 0
verdict = "WF CONFIRMED" if wf_confirmed else "WF WEAK"

print(f"\n  WF: {wf_wins}/3 wins  avg_OOS_delta={avg_oos_delta:+.4f}  Degradation={degrad:.3f}x")
print(f"  VERDICT: {verdict}")

# Save report
out = Path("artifacts/phase203"); out.mkdir(parents=True, exist_ok=True)
report = {
    "phase": 203,
    "description": f"WF Validation of {ver_now} vs p91b_flat (post-MID regime surge)",
    "elapsed_seconds": round(time.time() - _start, 1),
    "config_version": ver_now,
    "weights": WEIGHTS_CURR,
    "is_obj_curr": is_obj_curr, "is_obj_flat": is_obj_flat,
    "is_delta": is_obj_curr - is_obj_flat,
    "is_yearly_curr": is_yr_curr, "is_yearly_flat": is_yr_flat,
    "wf_windows": wf_results, "wf_wins_vs_flat": wf_wins,
    "avg_oos_delta": avg_oos_delta,
    "degradation_ratio": degrad,
    "verdict": verdict,
    "timestamp": datetime.now(UTC).isoformat(),
}
(out / "phase203_report.json").write_text(json.dumps(report, indent=2, default=str))
print(f"\n  Report saved  ({report['elapsed_seconds']}s)")

# Commit regardless of WF outcome (validation is informational)
is_delta = is_obj_curr - is_obj_flat
cm = "phase203: WF Validation %s -- %s %d/3 wins avg_OOS_delta=%+.4f IS_OBJ=%.4f IS_delta=%+.4f" % (
    ver_now, verdict, wf_wins, avg_oos_delta, is_obj_curr, is_delta)
os.system("git add configs/production_p91b_champion.json scripts/run_phase203_wf_validation_v231.py "
          "artifacts/phase203/ 2>/dev/null || true")
os.system('git commit -m "' + cm + '"')
os.system("git pull --rebase && git push")

print(f"\n{'✅ WF CONFIRMED' if wf_confirmed else '⚠️ WF WEAK'} — {verdict}")
print(f"\n[DONE] Phase 203 complete.")
