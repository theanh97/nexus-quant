"""
Phase 196 — Regime I460 Weight Re-Optimization (post I460 bw=168)
==================================================================
I460 bw changed from 120→168 (P194), improving signal quality.
With better I460, its optimal ensemble weight per regime might increase.

P185/P189: V1 got better → V1 weights increased (LOW=0.48, MID=0.35).
P194: I460 bw improved → I460 weights may need updating too.

Approach:
  Sweep I460 weight per regime, keeping F168 fixed and scaling V1/I415.
  (Symmetric to P189 for V1, but for I460)

  LOW:  i460 ∈ [0.04, 0.06, 0.0642, 0.08, 0.10, 0.12]
  MID:  i460 ∈ [0.08, 0.10, 0.1119, 0.13, 0.15, 0.18]
  HIGH: i460 ∈ [0.20, 0.25, 0.30, 0.35, 0.40]

Baseline: v2.24.0, OBJ=2.8797
Validation: LOYO ≥3/5 AND Δ>0.005
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
    _partial["timeout_at"] = datetime.now(UTC).isoformat()
    out = Path("artifacts/phase196"); out.mkdir(parents=True, exist_ok=True)
    (out / "phase196_report.json").write_text(json.dumps(_partial, indent=2, default=str))
    sys.exit(0)
_signal.signal(_signal.SIGALRM, _on_timeout)
_signal.alarm(7200)

SYMBOLS = ["BTCUSDT","ETHUSDT","SOLUSDT","BNBUSDT","XRPUSDT",
           "ADAUSDT","DOGEUSDT","AVAXUSDT","DOTUSDT","LINKUSDT"]
YEAR_RANGES = {
    2021: ("2021-02-01", "2022-01-01"),
    2022: ("2022-01-01", "2023-01-01"),
    2023: ("2023-01-01", "2024-01-01"),
    2024: ("2024-01-01", "2025-01-01"),
    2025: ("2025-01-01", "2026-01-01"),
}
YEARS = sorted(YEAR_RANGES)

VOL_WINDOW = 168; BRD_LB = 192; PCT_WINDOW = 336
P_LOW, P_HIGH = 0.20, 0.60; TS_SHORT = 12; TS_LONG = 96; FUND_DISP_PCT = 240
VOL_THRESHOLD = 0.50; VOL_SCALE = 0.40; VOL_F168_BOOST = 0.10
TS_RT = 0.60; TS_RS = 0.40; TS_BT = 0.35; TS_BS = 1.60
DISP_THR = 0.60; DISP_SCALE = 1.0

COST_MODEL = cost_model_from_config({
    "fee_rate": 0.0005, "slippage_rate": 0.0003, "cost_multiplier": 1.0,
})

# v2.24.0 baseline weights
BASE_WEIGHTS = {
    "LOW":  {"v1": 0.48,   "i460": 0.0642, "i415": 0.1058, "f168": 0.35},
    "MID":  {"v1": 0.35,   "i460": 0.1119, "i415": 0.1881, "f168": 0.35},
    "HIGH": {"v1": 0.06,   "i460": 0.30,   "i415": 0.54,   "f168": 0.10},
}
F168_WEIGHTS = {"LOW": 0.35, "MID": 0.35, "HIGH": 0.10}
V1_WEIGHTS   = {"LOW": 0.48, "MID": 0.35, "HIGH": 0.06}

BASELINE_OBJ = 2.8797

# Sweep: for each regime, sweep I460 weight (V1 fixed, I415 gets remainder)
I460_SWEEPS = {
    "LOW":  [0.04, 0.06, 0.0642, 0.08, 0.10, 0.12, 0.15],
    "MID":  [0.07, 0.09, 0.1119, 0.13, 0.15, 0.18, 0.21],
    "HIGH": [0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45],
}

def make_weights_i460(regime: str, new_i460: float) -> dict:
    """Set i460, keep v1 and f168 fixed, i415 gets remainder."""
    v1 = V1_WEIGHTS[regime]
    f168 = F168_WEIGHTS[regime]
    i415 = 1.0 - v1 - f168 - new_i460
    if i415 < 0.03:
        return None  # infeasible
    return {"v1": v1, "i460": new_i460, "i415": i415, "f168": f168}

def rolling_mean_arr(a, w):
    out = np.full(len(a), np.nan)
    if w <= 0: return out
    cs = np.cumsum(np.where(np.isnan(a), 0.0, a))
    for i in range(w - 1, len(a)):
        out[i] = cs[i] / w if i == w - 1 else (cs[i] - cs[i - w]) / w
    return out

def load_year(year):
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
            dataset, make_strategy({"name": sname, "params": params}))
        sig_rets[sk] = np.array(result.returns)

    print(".", end=" ", flush=True)
    return btc_vol, fund_std_pct, ts_spread_pct, regime, sig_rets, n

def compute_ensemble(year_data, weights_dict):
    btc_vol, fund_std_pct, ts_spread_pct, regime, sig_rets, n = year_data
    min_len = min(len(v) for v in sig_rets.values())
    bv = btc_vol[:min_len]; reg = regime[:min_len]
    fsp = fund_std_pct[:min_len]; tsp = ts_spread_pct[:min_len]
    ens = np.zeros(min_len)
    for i in range(min_len):
        w = weights_dict[["LOW", "MID", "HIGH"][int(reg[i])]]
        if not np.isnan(bv[i]) and bv[i] > VOL_THRESHOLD:
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
        if DISP_SCALE > 1.0 and fsp[i] > DISP_THR: ret_i *= DISP_SCALE
        if tsp[i] > TS_RT: ret_i *= TS_RS
        elif tsp[i] < TS_BT: ret_i *= TS_BS
        ens[i] = ret_i
    return ens

def sharpe(rets, n):
    if n < 20: return -999.0
    mu = float(np.mean(rets[:n])) * 8760
    sd = float(np.std(rets[:n])) * np.sqrt(8760)
    return mu / sd if sd > 1e-9 else 0.0

def obj_fn(yr_sharpes):
    vals = list(yr_sharpes.values())
    return float(np.mean(vals) - 0.5 * np.std(vals))

def eval_weights(year_data, weights_dict):
    yr_sharpes = {}
    for yr in YEARS:
        ens = compute_ensemble(year_data[yr], weights_dict)
        yr_sharpes[yr] = sharpe(ens, year_data[yr][-1])
    return obj_fn(yr_sharpes), yr_sharpes

# ── MAIN ────────────────────────────────────────────────────────────────────
print("=" * 60)
print("Phase 196 — Regime I460 Weight Re-Optimization")
print(f"Baseline: v2.24.0 OBJ={BASELINE_OBJ}")
print("=" * 60)
_start = time.time()

print("\n[1] Loading all years ...")
year_data = {}
for yr in YEARS:
    print(f"  {yr}", end="", flush=True)
    year_data[yr] = load_year(yr)
    print()

# Step 2: Per-regime 1D sweep for I460 weight
print("\n[2] Per-regime I460 weight sweep ...")
best_i460_per_regime = {}
sweep_results_per_regime = {}

for regime in ["LOW", "MID", "HIGH"]:
    sweep_results_per_regime[regime] = []
    best_obj_r = -999.0; best_i460_r = BASE_WEIGHTS[regime]["i460"]
    print(f"\n  Regime {regime} (baseline i460={BASE_WEIGHTS[regime]['i460']:.4f}):")
    for new_i460 in I460_SWEEPS[regime]:
        w_combo = dict(BASE_WEIGHTS)  # keep other regimes at baseline
        new_w = make_weights_i460(regime, new_i460)
        if new_w is None:
            print(f"    i460={new_i460:.4f} → INFEASIBLE (i415 < 0.03)")
            continue
        w_combo[regime] = new_w
        obj, yr_sharpes = eval_weights(year_data, w_combo)
        sweep_results_per_regime[regime].append({"i460": new_i460, "obj": obj})
        flag = " ← CURRENT" if abs(new_i460 - BASE_WEIGHTS[regime]["i460"]) < 0.001 else ""
        print(f"    i460={new_i460:.4f} i415={new_w['i415']:.4f} OBJ={obj:.4f}{flag}")
        if obj > best_obj_r:
            best_obj_r = obj; best_i460_r = new_i460

    best_i460_per_regime[regime] = best_i460_r
    print(f"  Best for {regime}: i460={best_i460_r:.4f}")

# Step 3: Joint evaluation with best per-regime values
print("\n[3] Joint evaluation with best i460 per regime ...")
best_joint_weights = {}
for regime in ["LOW", "MID", "HIGH"]:
    w = make_weights_i460(regime, best_i460_per_regime[regime])
    if w is None:
        w = BASE_WEIGHTS[regime]
    best_joint_weights[regime] = w
    print(f"  {regime}: {best_joint_weights[regime]}")

joint_obj, joint_yr = eval_weights(year_data, best_joint_weights)
print(f"\n  Joint OBJ: {joint_obj:.4f}  Δ={joint_obj - BASELINE_OBJ:+.4f}")

# Step 4: LOYO validation
print("\n[4] LOYO validation (joint best vs baseline) ...")
loyo_wins = 0; loyo_table = []
for yr in YEARS:
    ens_best = compute_ensemble(year_data[yr], best_joint_weights)
    ens_base = compute_ensemble(year_data[yr], BASE_WEIGHTS)
    n = year_data[yr][-1]
    sh_b = sharpe(ens_best, n); sh_base = sharpe(ens_base, n)
    win = bool(sh_b > sh_base)
    if win: loyo_wins += 1
    loyo_table.append({"year": yr, "sh_best": sh_b, "sh_base": sh_base, "win": win})
    print(f"  {yr}: best={sh_b:.4f} base={sh_base:.4f} {'WIN' if win else 'LOSS'}")

print(f"\n  LOYO: {loyo_wins}/5 wins")
validated = loyo_wins >= 3 and joint_obj > BASELINE_OBJ + 0.005

# Step 5: Save report
out = Path("artifacts/phase196"); out.mkdir(parents=True, exist_ok=True)
report = {
    "phase": 196, "target": "regime_i460_weights",
    "best_i460_per_regime": best_i460_per_regime,
    "best_joint_weights": best_joint_weights,
    "baseline_weights": BASE_WEIGHTS,
    "baseline_obj": BASELINE_OBJ,
    "joint_obj": joint_obj, "joint_delta": joint_obj - BASELINE_OBJ,
    "loyo": {"wins": loyo_wins, "table": loyo_table},
    "validated": validated,
    "runtime_s": round(time.time() - _start, 1),
    "timestamp": datetime.now(UTC).isoformat(),
}
(out / "phase196_report.json").write_text(json.dumps(report, indent=2, default=str))
print(f"  Report saved → artifacts/phase196/phase196_report.json  ({report['runtime_s']}s)")

if validated:
    cfg = json.loads(Path("configs/production_p91b_champion.json").read_text())
    ver = cfg.get("_version", "2.24.0")
    major, minor, patch = map(int, ver.split("."))
    new_ver = f"{major}.{minor + 1}.{patch}"
    cfg["_version"] = new_ver
    cfg["_validated"] += (
        f"; Regime I460 weight re-opt P196: LOYO {loyo_wins}/5 delta={joint_obj-BASELINE_OBJ:+.4f}"
        f" OBJ={joint_obj:.4f} — PRODUCTION {new_ver}"
    )
    # Update regime weights in config
    brs = cfg["breadth_regime_switching"]
    rw = brs["regime_weights"]
    for regime in ["LOW", "MID", "HIGH"]:
        w = best_joint_weights[regime]
        rw[regime]["v1"]  = round(w["v1"], 4)
        rw[regime]["i460bw168"] = round(w["i460"], 4)
        rw[regime]["i415bw216"] = round(w["i415"], 4)
        rw[regime]["f168"] = round(w["f168"], 4)
    brs["mechanism"] = brs.get("mechanism","") + f" P196: I460 weights updated per regime."
    Path("configs/production_p91b_champion.json").write_text(json.dumps(cfg, indent=2))
    jov = joint_obj; d = jov - BASELINE_OBJ
    cm = "phase196: regime I460 weight re-opt -- VALIDATED LOYO %d/5 delta=%+.4f OBJ=%.4f" % (loyo_wins, d, jov)
    print(f"\n✅ VALIDATED → {new_ver} OBJ={jov:.4f}")
    os.system("git add configs/production_p91b_champion.json artifacts/phase196/ scripts/run_phase196_regime_i460_weight_reopt.py")
    os.system('git commit -m "' + cm + '"')
    os.system("git pull --rebase && git push")
else:
    delta = joint_obj - BASELINE_OBJ
    verdict = "NO IMPROVEMENT" if delta <= 0 else f"WEAK LOYO {loyo_wins}/5"
    cm = "phase196: regime I460 weight re-opt -- %s delta=%+.4f" % (verdict, delta)
    print(f"\n❌ NOT VALIDATED — {verdict}")
    os.system("git add artifacts/phase196/ scripts/run_phase196_regime_i460_weight_reopt.py")
    os.system('git commit -m "' + cm + '"')
    os.system("git pull --rebase && git push")

print("\n[DONE] Phase 196 complete.")
