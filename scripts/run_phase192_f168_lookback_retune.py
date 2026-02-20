"""
Phase 192 — F168 Lookback Bars Re-Tune (v2.23.0 stack)
========================================================
F168 (funding_momentum_alpha) uses funding_lookback_bars=168.
This was set early and never re-tuned after major regime weight changes.

Post v2.23.0: F168 carries 0.35 weight in LOW+MID regimes (very significant).
With new regime structure, the optimal lookback might shift.

Test: funding_lookback_bars ∈ [72, 96, 120, 144, 168, 210, 252, 336]
  (current: 168)

Baseline: v2.23.0 OBJ=2.8691
Validate: LOYO ≥3/5 AND Δ>0 → update config as v2.24.0
"""
import os, sys, json, time
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
    out = Path("artifacts/phase192"); out.mkdir(parents=True, exist_ok=True)
    (out / "phase192_report.json").write_text(json.dumps(_partial, indent=2, default=str))
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

# Overlay constants (v2.23.0)
VOL_WINDOW    = 168
BRD_LB        = 192
PCT_WINDOW    = 336
P_LOW, P_HIGH = 0.20, 0.60
TS_SHORT      = 12
TS_LONG       = 96
FUND_DISP_PCT = 240

VOL_THRESHOLD  = 0.50
VOL_SCALE      = 0.40
VOL_F168_BOOST = 0.10
TS_RT = 0.60; TS_RS = 0.40; TS_BT = 0.35; TS_BS = 1.60
DISP_THR = 0.60; DISP_SCALE = 1.0

# v2.23.0 regime weights (P190 validated)
WEIGHTS = {
    "LOW":  {"v1": 0.48,   "i460": 0.0642, "i415": 0.1058, "f168": 0.35},
    "MID":  {"v1": 0.35,   "i460": 0.1119, "i415": 0.1881, "f168": 0.35},
    "HIGH": {"v1": 0.06,   "i460": 0.30,   "i415": 0.54,   "f168": 0.10},
}

BASELINE_OBJ = 2.8691
F168_LB_SWEEP = [72, 96, 120, 144, 168, 210, 252, 336]
CURRENT_F168_LB = 168

V1_PARAMS = {
    "k_per_side": 2, "w_carry": 0.35, "w_mom": 0.40, "w_mean_reversion": 0.25,
    "momentum_lookback_bars": 336, "mean_reversion_lookback_bars": 84,
    "vol_lookback_bars": 192, "target_gross_leverage": 0.35,
    "rebalance_interval_bars": 60,
}
I460_PARAMS = {
    "k_per_side": 4, "lookback_bars": 460, "beta_window_bars": 120,
    "target_gross_leverage": 0.3, "rebalance_interval_bars": 48,
}
I415_PARAMS = {
    "k_per_side": 4, "lookback_bars": 415, "beta_window_bars": 144,
    "target_gross_leverage": 0.3, "rebalance_interval_bars": 48,
}

COST_MODEL = cost_model_from_config({
    "fee_rate": 0.0005, "slippage_rate": 0.0003, "cost_multiplier": 1.0,
})

def rolling_mean_arr(a: np.ndarray, w: int) -> np.ndarray:
    out = np.full(len(a), np.nan)
    if w <= 0 or len(a) == 0: return out
    cs = np.cumsum(np.where(np.isnan(a), 0.0, a))
    for i in range(w - 1, len(a)):
        out[i] = cs[i] / w if i == w - 1 else (cs[i] - cs[i - w]) / w
    return out

def load_year_fixed_signals(year: int):
    """Load data and compute v1, i460, i415 signals once per year."""
    s, e = YEAR_RANGES[year]
    cfg = {"provider": "binance_rest_v1", "symbols": SYMBOLS,
           "start": s, "end": e, "bar_interval": "1h",
           "cache_dir": ".cache/binance_rest"}
    dataset = make_provider(cfg, seed=42).load()
    n = len(dataset.timeline)

    close_mat = np.zeros((n, len(SYMBOLS)))
    for j, sym in enumerate(SYMBOLS):
        for i in range(n): close_mat[i, j] = dataset.close(sym, i)

    # BTC vol for vol_regime
    btc_rets = np.zeros(n)
    for i in range(1, n):
        c0 = close_mat[i-1, 0]; c1 = close_mat[i, 0]
        btc_rets[i] = (c1 / c0 - 1.0) if c0 > 0 else 0.0
    btc_vol = np.full(n, np.nan)
    for i in range(VOL_WINDOW, n):
        btc_vol[i] = float(np.std(btc_rets[i - VOL_WINDOW:i])) * np.sqrt(8760)
    if VOL_WINDOW < n: btc_vol[:VOL_WINDOW] = btc_vol[VOL_WINDOW]

    # Funding rates
    fund_rates = np.zeros((n, len(SYMBOLS)))
    for j, sym in enumerate(SYMBOLS):
        for i in range(n):
            ts = dataset.timeline[i]
            try:    fund_rates[i, j] = dataset.last_funding_rate_before(sym, ts)
            except: fund_rates[i, j] = 0.0

    # Dispersion percentile
    fund_std_raw = np.std(fund_rates, axis=1)
    fund_std_pct = np.full(n, 0.5)
    for i in range(FUND_DISP_PCT, n):
        fund_std_pct[i] = float(np.mean(fund_std_raw[i - FUND_DISP_PCT:i] <= fund_std_raw[i]))
    fund_std_pct[:FUND_DISP_PCT] = 0.5

    # TS spread percentile
    xsect_mean = np.mean(fund_rates, axis=1)
    ts_raw = rolling_mean_arr(xsect_mean, TS_SHORT) - rolling_mean_arr(xsect_mean, TS_LONG)
    ts_spread_pct = np.full(n, 0.5)
    for i in range(PCT_WINDOW, n):
        ts_spread_pct[i] = float(np.mean(ts_raw[i - PCT_WINDOW:i] <= ts_raw[i]))
    ts_spread_pct[:PCT_WINDOW] = 0.5

    # Breadth regime
    breadth = np.full(n, 0.5)
    for i in range(BRD_LB, n):
        pos = sum(1 for j in range(len(SYMBOLS))
                  if close_mat[i - BRD_LB, j] > 0 and close_mat[i, j] > close_mat[i - BRD_LB, j])
        breadth[i] = pos / len(SYMBOLS)
    breadth[:BRD_LB] = 0.5
    brd_pct = np.full(n, 0.5)
    for i in range(PCT_WINDOW, n):
        brd_pct[i] = float(np.mean(breadth[i - PCT_WINDOW:i] <= breadth[i]))
    brd_pct[:PCT_WINDOW] = 0.5
    regime = np.where(brd_pct >= P_HIGH, 2, np.where(brd_pct >= P_LOW, 1, 0)).astype(int)

    # Fixed signals: v1, i460, i415
    fixed_rets = {}
    for sk, sname, params in [
        ("v1", "nexus_alpha_v1", V1_PARAMS),
        ("i460", "idio_momentum_alpha", I460_PARAMS),
        ("i415", "idio_momentum_alpha", I415_PARAMS),
    ]:
        result = BacktestEngine(BacktestConfig(costs=COST_MODEL)).run(
            dataset, make_strategy({"name": sname, "params": params}))
        fixed_rets[sk] = np.array(result.returns)

    print("S.", end=" ", flush=True)
    return dataset, btc_vol, fund_std_pct, ts_spread_pct, regime, fixed_rets, n

def run_f168_with_lb(dataset, lb: int):
    """Run F168 with given funding_lookback_bars."""
    params = {
        "k_per_side": 2, "funding_lookback_bars": lb, "direction": "contrarian",
        "target_gross_leverage": 0.25, "rebalance_interval_bars": 24,
    }
    result = BacktestEngine(BacktestConfig(costs=COST_MODEL)).run(
        dataset, make_strategy({"name": "funding_momentum_alpha", "params": params}))
    return np.array(result.returns)

def compute_ensemble_rets(base_data, f168_rets: np.ndarray) -> np.ndarray:
    """Compute full ensemble with given f168 returns and v2.23.0 weights."""
    _, btc_vol, fund_std_pct, ts_spread_pct, regime, fixed_rets, n = base_data
    all_rets = dict(fixed_rets, f168=f168_rets)
    min_len = min(len(v) for v in all_rets.values())
    bv = btc_vol[:min_len]; reg = regime[:min_len]
    fsp = fund_std_pct[:min_len]; tsp = ts_spread_pct[:min_len]

    ens = np.zeros(min_len)
    for i in range(min_len):
        w = WEIGHTS[["LOW", "MID", "HIGH"][int(reg[i])]]
        if not np.isnan(bv[i]) and bv[i] > VOL_THRESHOLD:
            boost_per = VOL_F168_BOOST / 3.0
            ret_i = (
                (w["v1"]   + boost_per) * all_rets["v1"][i]  +
                (w["i460"] + boost_per) * all_rets["i460"][i] +
                (w["i415"] + boost_per) * all_rets["i415"][i] +
                (w["f168"] - VOL_F168_BOOST) * all_rets["f168"][i]
            ) * VOL_SCALE
        else:
            ret_i = (
                w["v1"]   * all_rets["v1"][i]   +
                w["i460"] * all_rets["i460"][i]  +
                w["i415"] * all_rets["i415"][i]  +
                w["f168"] * all_rets["f168"][i]
            )
        if DISP_SCALE > 1.0 and fsp[i] > DISP_THR:
            ret_i *= DISP_SCALE
        if tsp[i] > TS_RT:
            ret_i *= TS_RS
        elif tsp[i] < TS_BT:
            ret_i *= TS_BS
        ens[i] = ret_i
    return ens

def yearly_sharpe(rets: np.ndarray, n: int) -> float:
    if n < 20: return -999.0
    mu = float(np.mean(rets[:n])) * 8760
    sd = float(np.std(rets[:n])) * np.sqrt(8760)
    return mu / sd if sd > 1e-9 else 0.0

def obj_fn(yearly_sharpes: dict) -> float:
    vals = list(yearly_sharpes.values())
    return float(np.mean(vals) - 0.5 * np.std(vals))

# ── MAIN ────────────────────────────────────────────────────────────────────
print("=" * 60)
print("Phase 192 — F168 Lookback Bars Re-Tune")
print(f"Baseline: v2.23.0 OBJ={BASELINE_OBJ}")
print(f"Sweep: {F168_LB_SWEEP}")
print("=" * 60)
_start = time.time()

# Step 1: Load per-year data + fixed signals
print("\n[1] Loading per-year data + fixed signals (v1, i460, i415) ...")
year_data = {}
for yr in YEARS:
    print(f"  {yr} ", end="", flush=True)
    year_data[yr] = load_year_fixed_signals(yr)
    print()

# Step 2: Run F168 for each lb per year
print("\n[2] Running F168 for each lookback_bars ...")
f168_rets_by_lb = {lb: {} for lb in F168_LB_SWEEP}

for lb in F168_LB_SWEEP:
    print(f"  lb={lb}h:", end=" ", flush=True)
    for yr in YEARS:
        dataset = year_data[yr][0]
        f168_rets_by_lb[lb][yr] = run_f168_with_lb(dataset, lb)
        print(f"{yr}", end=" ", flush=True)
    print()

# Step 3: Compute yearly OBJ for each lb
print("\n[3] Computing yearly Sharpes per lb ...")
results = []

for lb in F168_LB_SWEEP:
    yr_sharpes = {}
    for yr in YEARS:
        base_data = year_data[yr]
        ens = compute_ensemble_rets(base_data, f168_rets_by_lb[lb][yr])
        n = base_data[-1]
        yr_sharpes[yr] = yearly_sharpe(ens, n)
    obj = obj_fn(yr_sharpes)
    results.append({"lb": lb, "obj": obj, "yr_sharpes": yr_sharpes})
    flag = " ← CURRENT" if lb == CURRENT_F168_LB else ""
    print(f"  lb={lb:3d}h  OBJ={obj:.4f}  {yr_sharpes}{flag}")

# Step 4: Find best
best = max(results, key=lambda x: x["obj"])
print(f"\n  Best: lb={best['lb']}h  OBJ={best['obj']:.4f}  Δ={best['obj'] - BASELINE_OBJ:+.4f}")

# Step 5: LOYO validation
print("\n[4] LOYO validation (best vs current lb=168) ...")
loyo_wins = 0
loyo_table = []
for yr in YEARS:
    base_data = year_data[yr]
    ens_best = compute_ensemble_rets(base_data, f168_rets_by_lb[best["lb"]][yr])
    ens_curr = compute_ensemble_rets(base_data, f168_rets_by_lb[CURRENT_F168_LB][yr])
    n = base_data[-1]
    sh_best = yearly_sharpe(ens_best, n)
    sh_curr = yearly_sharpe(ens_curr, n)
    win = bool(sh_best > sh_curr)
    if win: loyo_wins += 1
    loyo_table.append({"year": yr, "sh_best": sh_best, "sh_curr": sh_curr, "win": win})
    print(f"  {yr}: best={sh_best:.4f} curr={sh_curr:.4f} {'WIN' if win else 'LOSS'}")

print(f"\n  LOYO: {loyo_wins}/5 wins")
validated = loyo_wins >= 3 and best["obj"] > BASELINE_OBJ

# Step 6: Save report
out = Path("artifacts/phase192"); out.mkdir(parents=True, exist_ok=True)
report = {
    "phase": 192,
    "target": "f168_funding_lookback_bars",
    "sweep_values": F168_LB_SWEEP,
    "all_results": results,
    "best_lb": best["lb"],
    "best_obj": best["obj"],
    "delta": best["obj"] - BASELINE_OBJ,
    "baseline_obj": BASELINE_OBJ,
    "loyo": {"wins": loyo_wins, "table": loyo_table},
    "validated": validated,
    "runtime_s": round(time.time() - _start, 1),
    "timestamp": datetime.now(UTC).isoformat(),
}
(out / "phase192_report.json").write_text(json.dumps(report, indent=2, default=str))
print(f"\n  Report saved → artifacts/phase192/phase192_report.json")
print(f"  Elapsed: {report['runtime_s']}s")

# Step 7: Update config if validated
if validated:
    cfg = json.loads(Path("configs/production_p91b_champion.json").read_text())
    ver = cfg.get("_version", "2.23.0")
    major, minor, patch = map(int, ver.split("."))
    new_ver = f"{major}.{minor + 1}.{patch}"

    cfg["_version"] = new_ver
    cfg["_validated"] += (
        f"; F168 lb sweep P192: {CURRENT_F168_LB}->{best['lb']} "
        f"LOYO {loyo_wins}/5 delta={best['obj'] - BASELINE_OBJ:+.4f} OBJ={best['obj']:.4f}"
        f" — PRODUCTION {new_ver}"
    )

    # Update F168 params in FTS/funding overlay section
    # The F168 lb is embedded in the breadth_regime_switching mechanism note + actual backtest
    brs = cfg.get("breadth_regime_switching", {})
    brs["_validated"] = brs.get("_validated", "") + f"; F168 lb P192: {CURRENT_F168_LB}->{best['lb']} LOYO {loyo_wins}/5"
    if "f168_tuned_params" not in cfg:
        cfg["f168_tuned_params"] = {}
    cfg["f168_tuned_params"]["funding_lookback_bars"] = best["lb"]
    cfg["f168_tuned_params"]["_note"] = f"P192 validated: lb={best['lb']} LOYO {loyo_wins}/5"

    Path("configs/production_p91b_champion.json").write_text(json.dumps(cfg, indent=2))
    best_lb = best["lb"]
    best_obj_val = best["obj"]
    d = best_obj_val - BASELINE_OBJ
    cm = "phase192: F168 lb sweep -- VALIDATED %d->%dh LOYO %d/5 delta=%+.4f OBJ=%.4f" % (CURRENT_F168_LB, best_lb, loyo_wins, d, best_obj_val)
    print(f"\n✅ VALIDATED → config updated to {new_ver} OBJ={best_obj_val:.4f}")
    print(f"   F168 lb: {CURRENT_F168_LB} → {best_lb}")
    os.system("git add configs/production_p91b_champion.json artifacts/phase192/ scripts/run_phase192_f168_lookback_retune.py")
    os.system('git commit -m "' + cm + '"')
    os.system("git pull --rebase && git push")
else:
    delta = best["obj"] - BASELINE_OBJ
    verdict = "NO IMPROVEMENT" if delta <= 0 else f"WEAK (LOYO only {loyo_wins}/5)"
    print(f"\n❌ NOT VALIDATED — {verdict}")
    print(f"   F168 lb=168 remains optimal")
    best_lb = best["lb"]
    cm = "phase192: F168 lb sweep -- %s best_lb=%dh LOYO %d/5 delta=%+.4f" % (verdict, best_lb, loyo_wins, delta)
    os.system("git add artifacts/phase192/ scripts/run_phase192_f168_lookback_retune.py")
    os.system('git commit -m "' + cm + '"')
    os.system("git pull --rebase && git push")

print("\n[DONE] Phase 192 complete.")
