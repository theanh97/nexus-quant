"""
Phase 195 — Walk-Forward Validation of v2.24.0 vs p91b_flat
=============================================================
Major improvements since last WF (P175 @ v2.14.0):
  - P185: V1 blend weights (w_carry=0.35, w_mom=0.40, w_mr=0.25)
  - P187: V1 mr_lb 72→84
  - P189+P190: Regime V1 weights (LOW=0.48, MID=0.35)
  - P190: V1 vol_lb 168→192
  - P194: I460 bw 120→168

Current OBJ: 2.8797 (IS, 2021-2025)
Last WF at P175: v2.14.0 OBJ=2.4817, WF 3/3 wins, avg_OOS_delta=+0.7051

WF Windows:
  - Train [2021-2022] → Test 2023
  - Train [2021-2023] → Test 2024
  - Train [2021-2024] → Test 2025

Baseline: p91b_flat (equal-weight ensemble, no overlays)
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
    out = Path("artifacts/phase195"); out.mkdir(parents=True, exist_ok=True)
    (out / "phase195_report.json").write_text(json.dumps(_partial, indent=2, default=str))
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

# v2.24.0 config
VOL_WINDOW = 168; BRD_LB = 192; PCT_WINDOW = 336
P_LOW, P_HIGH = 0.20, 0.60; TS_SHORT = 12; TS_LONG = 96; FUND_DISP_PCT = 240
VOL_THRESHOLD = 0.50; VOL_SCALE = 0.40; VOL_F168_BOOST = 0.10
TS_RT = 0.60; TS_RS = 0.40; TS_BT = 0.35; TS_BS = 1.60
DISP_THR = 0.60; DISP_SCALE = 1.0

# v2.24.0 regime weights
WEIGHTS_V224 = {
    "LOW":  {"v1": 0.48,   "i460": 0.0642, "i415": 0.1058, "f168": 0.35},
    "MID":  {"v1": 0.35,   "i460": 0.1119, "i415": 0.1881, "f168": 0.35},
    "HIGH": {"v1": 0.06,   "i460": 0.30,   "i415": 0.54,   "f168": 0.10},
}

# Read latest from config (override hardcoded if needed)
try:
    _cfg = json.loads(Path("configs/production_p91b_champion.json").read_text())
    _brs = _cfg["breadth_regime_switching"]["regime_weights"]
    for _r in ["LOW", "MID", "HIGH"]:
        _w = _brs[_r]
        WEIGHTS_V224[_r] = {
            "v1":   _w["v1"],
            "i460": _w.get("i460bw168", _w.get("i460", 0.15)),
            "i415": _w.get("i415bw216", _w.get("i415", 0.25)),
            "f168": _w["f168"],
        }
except Exception as exc:
    print(f"Warning: could not read weights from config: {exc}")

V1_PARAMS = {
    "k_per_side": 2, "w_carry": 0.35, "w_mom": 0.40, "w_mean_reversion": 0.25,
    "momentum_lookback_bars": 336, "mean_reversion_lookback_bars": 84,
    "vol_lookback_bars": 192, "target_gross_leverage": 0.35, "rebalance_interval_bars": 60,
}
I460_PARAMS = {
    "k_per_side": 4, "lookback_bars": 460, "beta_window_bars": 168,
    "target_gross_leverage": 0.3, "rebalance_interval_bars": 48,
}
I415_PARAMS = {
    "k_per_side": 4, "lookback_bars": 415, "beta_window_bars": 144,
    "target_gross_leverage": 0.3, "rebalance_interval_bars": 48,
}
F168_PARAMS = {
    "k_per_side": 2, "funding_lookback_bars": 168, "direction": "contrarian",
    "target_gross_leverage": 0.25, "rebalance_interval_bars": 24,
}
# Flat baseline: equal weight, no overlay
WEIGHTS_FLAT = {
    "LOW":  {"v1": 0.25, "i460": 0.25, "i415": 0.25, "f168": 0.25},
    "MID":  {"v1": 0.25, "i460": 0.25, "i415": 0.25, "f168": 0.25},
    "HIGH": {"v1": 0.25, "i460": 0.25, "i415": 0.25, "f168": 0.25},
}

COST_MODEL = cost_model_from_config({
    "fee_rate": 0.0005, "slippage_rate": 0.0003, "cost_multiplier": 1.0,
})

def rolling_mean_arr(a, w):
    out = np.full(len(a), np.nan)
    if w <= 0: return out
    cs = np.cumsum(np.where(np.isnan(a), 0.0, a))
    for i in range(w - 1, len(a)):
        out[i] = cs[i] / w if i == w - 1 else (cs[i] - cs[i - w]) / w
    return out

def load_year(year: str):
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
    ts_raw = rolling_mean_arr(xsect_mean, TS_SHORT) - rolling_mean_arr(xsect_mean, TS_LONG)
    ts_spread_pct = np.full(n, 0.5)
    for i in range(PCT_WINDOW, n):
        ts_spread_pct[i] = float(np.mean(ts_raw[i-PCT_WINDOW:i] <= ts_raw[i]))
    ts_spread_pct[:PCT_WINDOW] = 0.5

    breadth = np.full(n, 0.5)
    for i in range(BRD_LB, n):
        pos = sum(1 for j in range(len(SYMBOLS))
                  if close_mat[i-BRD_LB, j] > 0 and close_mat[i, j] > close_mat[i-BRD_LB, j])
        breadth[i] = pos / len(SYMBOLS)
    breadth[:BRD_LB] = 0.5
    brd_pct = np.full(n, 0.5)
    for i in range(PCT_WINDOW, n):
        brd_pct[i] = float(np.mean(breadth[i-PCT_WINDOW:i] <= breadth[i]))
    brd_pct[:PCT_WINDOW] = 0.5
    regime = np.where(brd_pct >= P_HIGH, 2, np.where(brd_pct >= P_LOW, 1, 0)).astype(int)

    sig_rets = {}
    for sk, sname, params in [
        ("v1", "nexus_alpha_v1", V1_PARAMS),
        ("i460", "idio_momentum_alpha", I460_PARAMS),
        ("i415", "idio_momentum_alpha", I415_PARAMS),
        ("f168", "funding_momentum_alpha", F168_PARAMS),
    ]:
        result = BacktestEngine(BacktestConfig(costs=COST_MODEL)).run(
            dataset, make_strategy({"name": sname, "params": params}))
        sig_rets[sk] = np.array(result.returns)

    print(".", end=" ", flush=True)
    return btc_vol, fund_std_pct, ts_spread_pct, regime, sig_rets, n

def compute_ensemble(year_data, weights, use_overlay=True):
    btc_vol, fund_std_pct, ts_spread_pct, regime, sig_rets, n = year_data
    min_len = min(len(v) for v in sig_rets.values())
    bv = btc_vol[:min_len]; reg = regime[:min_len]
    fsp = fund_std_pct[:min_len]; tsp = ts_spread_pct[:min_len]

    ens = np.zeros(min_len)
    for i in range(min_len):
        w = weights[["LOW", "MID", "HIGH"][int(reg[i])]]
        if use_overlay and not np.isnan(bv[i]) and bv[i] > VOL_THRESHOLD:
            boost_per = VOL_F168_BOOST / 3.0
            ret_i = (
                (w["v1"] + boost_per) * sig_rets["v1"][i] +
                (w["i460"] + boost_per) * sig_rets["i460"][i] +
                (w["i415"] + boost_per) * sig_rets["i415"][i] +
                (w["f168"] - VOL_F168_BOOST) * sig_rets["f168"][i]
            ) * VOL_SCALE
        else:
            ret_i = (
                w["v1"] * sig_rets["v1"][i] +
                w["i460"] * sig_rets["i460"][i] +
                w["i415"] * sig_rets["i415"][i] +
                w["f168"] * sig_rets["f168"][i]
            )
        if use_overlay:
            if DISP_SCALE > 1.0 and fsp[i] > DISP_THR: ret_i *= DISP_SCALE
            if tsp[i] > TS_RT: ret_i *= TS_RS
            elif tsp[i] < TS_BT: ret_i *= TS_BS
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
print("Phase 195 — WF Validation v2.24.0 vs p91b_flat")
print("Weights:", WEIGHTS_V224)
print("=" * 60)
_start = time.time()

# Load all years
print("\n[1] Loading all years ...")
year_data = {}
for yr in YEARS:
    print(f"  {yr}", end="", flush=True)
    year_data[yr] = load_year(yr)
    print()

# IS objective (full 2021-2025)
print("\n[2] IS objective (2021-2025) ...")
is_yr_v224 = {}; is_yr_flat = {}
for yr in YEARS:
    ens_v224 = compute_ensemble(year_data[yr], WEIGHTS_V224, use_overlay=True)
    ens_flat = compute_ensemble(year_data[yr], WEIGHTS_FLAT, use_overlay=False)
    n = year_data[yr][-1]
    is_yr_v224[yr] = sharpe(ens_v224, n)
    is_yr_flat[yr] = sharpe(ens_flat, n)
    print(f"  {yr}: v2.24={is_yr_v224[yr]:.4f}  flat={is_yr_flat[yr]:.4f}  Δ={is_yr_v224[yr]-is_yr_flat[yr]:+.4f}")

is_obj_v224 = obj_fn(is_yr_v224)
is_obj_flat = obj_fn(is_yr_flat)
print(f"\n  IS OBJ: v2.24={is_obj_v224:.4f}  flat={is_obj_flat:.4f}  Δ={is_obj_v224-is_obj_flat:+.4f}")

# WF windows
print("\n[3] Walk-Forward windows ...")
wf_results = []
wf_wins = 0

for window in WF_WINDOWS:
    train_yrs = window["train"]
    test_yr = window["test"]

    # IS Sharpes
    is_v = [is_yr_v224[y] for y in train_yrs]
    is_f = [is_yr_flat[y] for y in train_yrs]
    is_v224_wf = float(np.mean(is_v) - 0.5 * np.std(is_v))
    is_flat_wf = float(np.mean(is_f) - 0.5 * np.std(is_f))

    # OOS Sharpes
    oos_v = is_yr_v224[test_yr]
    oos_f = is_yr_flat[test_yr]
    delta = oos_v - oos_f
    win = bool(delta > 0)
    if win: wf_wins += 1
    wf_results.append({
        "train": train_yrs, "test": test_yr,
        "is_v224": is_v224_wf, "is_flat": is_flat_wf,
        "oos_v224": oos_v, "oos_flat": oos_f, "oos_delta": delta, "win": win
    })
    print(f"  Train={train_yrs} Test={test_yr}: OOS v224={oos_v:.4f} flat={oos_f:.4f} Δ={delta:+.4f} {'✅WIN' if win else '❌LOSS'}")

avg_oos_delta = float(np.mean([r["oos_delta"] for r in wf_results]))
degrad = is_obj_v224 / is_obj_flat if is_obj_flat > 0 else 0
wf_confirmed = wf_wins >= 2 and avg_oos_delta > 0

print(f"\n  WF: {wf_wins}/3 wins  avg_OOS_delta={avg_oos_delta:+.4f}  IS_delta={is_obj_v224-is_obj_flat:+.4f}")
print(f"  Degradation (IS→OOS): {degrad:.3f}x")
verdict = "WF CONFIRMED" if wf_confirmed else "WF WEAK"

# Save report
out = Path("artifacts/phase195"); out.mkdir(parents=True, exist_ok=True)
report = {
    "phase": 195,
    "description": "WF Validation of v2.24.0 vs p91b_flat",
    "elapsed_seconds": round(time.time() - _start, 1),
    "is_obj_v224": is_obj_v224, "is_obj_flat": is_obj_flat,
    "is_delta": is_obj_v224 - is_obj_flat,
    "is_yearly_v224": is_yr_v224, "is_yearly_flat": is_yr_flat,
    "wf_windows": wf_results, "wf_wins_vs_flat": wf_wins,
    "wf_avg_delta": avg_oos_delta, "degradation_ratio": round(degrad, 3),
    "wf_confirmed": wf_confirmed,
    "verdict": f"{verdict} — v2.24.0 OBJ={is_obj_v224:.4f} | WF {wf_wins}/3 wins vs p91b_flat | avg_OOS_delta={avg_oos_delta:+.4f} | IS delta={is_obj_v224-is_obj_flat:+.4f}",
    "partial": False,
    "timestamp": datetime.now(UTC).isoformat(),
}
(out / "phase195_report.json").write_text(json.dumps(report, indent=2, default=str))
print(f"\n  Saved: artifacts/phase195/phase195_report.json")
print(f"  VERDICT: {report['verdict']}")

# Update config
cfg = json.loads(Path("configs/production_p91b_champion.json").read_text())
cfg["_validated"] += f"; WF Validation P195: {verdict} v2.24.0 OBJ={is_obj_v224:.4f} WF {wf_wins}/3 avg_OOS_delta={avg_oos_delta:+.4f}"
Path("configs/production_p91b_champion.json").write_text(json.dumps(cfg, indent=2))

os.system("git add configs/production_p91b_champion.json artifacts/phase195/ scripts/run_phase195_wf_validation_v224.py")
wf_str = "CONFIRMED" if wf_confirmed else "WEAK"
cm = "phase195: WF Validation v2.24.0 -- %s %d/3 wins avg_OOS_delta=%+.4f IS_delta=%+.4f" % (wf_str, wf_wins, avg_oos_delta, is_obj_v224-is_obj_flat)
os.system('git commit -m "' + cm + '"')
os.system("git pull --rebase && git push")
print("\n[DONE] Phase 195 complete.")
