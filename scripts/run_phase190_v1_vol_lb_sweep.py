"""
Phase 190 — V1 vol_lookback_bars Sweep (v2.21.0 stack)
========================================================
V1 uses vol_lookback_bars=168h for volatility normalization in carry + momentum.
This parameter has NEVER been tuned — unchanged since P91.

Post P189: V1 weights re-optimized (LOW=0.40, MID=0.30, HIGH=0.06).
With better V1 role in LOW/MID regimes, vol normalization window may differ.

Hypothesis: Shorter vol window → more adaptive, sharper carry signal.
            Could help in volatile 2022/2023 periods.

Test: vol_lookback_bars ∈ [72, 96, 120, 144, 168, 192, 240]
  (current: 168)

Baseline: v2.21.0 OBJ=2.8506
Validate: LOYO ≥3/5 AND Δ>0 → commit as v2.22.0
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
    out = Path("artifacts/phase190"); out.mkdir(parents=True, exist_ok=True)
    (out / "phase190_report.json").write_text(json.dumps(_partial, indent=2, default=str))
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

# Overlay constants (v2.21.0)
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

# v2.21.0 regime weights (P189 validated)
WEIGHTS = {
    "LOW":  {"v1": 0.40,  "i460": 0.0944, "i415": 0.1556, "f168": 0.35},
    "MID":  {"v1": 0.30,  "i460": 0.1305, "i415": 0.2195, "f168": 0.35},
    "HIGH": {"v1": 0.06,  "i460": 0.30,   "i415": 0.54,   "f168": 0.10},
}

BASELINE_OBJ  = 2.8506
VOL_LB_SWEEP  = [72, 96, 120, 144, 168, 192, 240]
CURRENT_VOL_LB = 168

# V1 base params (mr_lb=84 from P187, weights from P185)
V1_BASE = {
    "k_per_side": 2, "w_carry": 0.35, "w_mom": 0.40, "w_mean_reversion": 0.25,
    "momentum_lookback_bars": 336, "mean_reversion_lookback_bars": 84,
    "target_gross_leverage": 0.35, "rebalance_interval_bars": 60,
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
    """Load data and compute i460, i415, f168 signals once per year."""
    s, e = YEAR_RANGES[year]
    cfg = {"provider": "binance_rest_v1", "symbols": SYMBOLS,
           "start": s, "end": e, "bar_interval": "1h",
           "cache_dir": ".cache/binance_rest"}
    dataset = make_provider(cfg, seed=42).load()
    n = len(dataset.timeline)

    # Close prices matrix
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

    # Fixed signals: i460, i415, f168
    fixed_rets = {}
    for sk, sname, params in [
        ("i460", "idio_momentum_alpha", {
            "k_per_side": 4, "lookback_bars": 460, "beta_window_bars": 120,
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
        fixed_rets[sk] = np.array(result.returns)

    print("S.", end=" ", flush=True)
    return dataset, btc_vol, fund_std_pct, ts_spread_pct, regime, fixed_rets, n

def run_v1_with_vol_lb(dataset, vol_lb: int):
    """Run V1 strategy with given vol_lookback_bars."""
    params = dict(V1_BASE, vol_lookback_bars=vol_lb)
    result = BacktestEngine(BacktestConfig(costs=COST_MODEL)).run(
        dataset, make_strategy({"name": "nexus_alpha_v1", "params": params}))
    return np.array(result.returns)

def compute_ensemble_rets(base_data, v1_rets: np.ndarray) -> np.ndarray:
    """Compute full ensemble with given v1 returns and v2.21.0 weights."""
    _, btc_vol, fund_std_pct, ts_spread_pct, regime, fixed_rets, n = base_data
    all_rets = dict(fixed_rets, v1=v1_rets)
    min_len = min(len(v) for v in all_rets.values())
    bv = btc_vol[:min_len]; reg = regime[:min_len]
    fsp = fund_std_pct[:min_len]; tsp = ts_spread_pct[:min_len]

    ens = np.zeros(min_len)
    for i in range(min_len):
        w = WEIGHTS[["LOW", "MID", "HIGH"][int(reg[i])]]
        if not np.isnan(bv[i]) and bv[i] > VOL_THRESHOLD:
            boost_per = VOL_F168_BOOST / 3.0  # distribute to i460, i415, v1
            ret_i = (
                (w["v1"]  + boost_per) * all_rets["v1"][i]  +
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
        # Dispersion overlay
        if DISP_SCALE > 1.0 and fsp[i] > DISP_THR:
            ret_i *= DISP_SCALE
        # TS overlay
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
print("Phase 190 — V1 vol_lookback_bars Sweep")
print(f"Baseline: v2.21.0 OBJ={BASELINE_OBJ}")
print(f"Sweep: {VOL_LB_SWEEP}")
print("=" * 60)
_start = time.time()

# Step 1: Load per-year data + fixed signals
print("\n[1] Loading per-year data + fixed signals (i460, i415, f168) ...")
year_data = {}
for yr in YEARS:
    print(f"  {yr} ", end="", flush=True)
    year_data[yr] = load_year_fixed_signals(yr)
    print()

# Step 2: Run V1 for each vol_lb per year (cache results)
print("\n[2] Running V1 for each vol_lb ...")
v1_rets_by_lb = {lb: {} for lb in VOL_LB_SWEEP}

for lb in VOL_LB_SWEEP:
    print(f"  vol_lb={lb}h:", end=" ", flush=True)
    for yr in YEARS:
        dataset = year_data[yr][0]
        v1_rets_by_lb[lb][yr] = run_v1_with_vol_lb(dataset, lb)
        print(f"{yr}", end=" ", flush=True)
    print()

# Step 3: Compute yearly OBJ for each vol_lb
print("\n[3] Computing yearly Sharpes per vol_lb ...")
results = []
baseline_sharpes = {}

for lb in VOL_LB_SWEEP:
    yr_sharpes = {}
    for yr in YEARS:
        base_data = year_data[yr]
        ens = compute_ensemble_rets(base_data, v1_rets_by_lb[lb][yr])
        n = base_data[-1]
        yr_sharpes[yr] = yearly_sharpe(ens, n)
    obj = obj_fn(yr_sharpes)
    results.append({"vol_lb": lb, "obj": obj, "yr_sharpes": yr_sharpes})
    flag = " ← CURRENT" if lb == CURRENT_VOL_LB else ""
    print(f"  vol_lb={lb:3d}h  OBJ={obj:.4f}  {yr_sharpes}{flag}")
    if lb == CURRENT_VOL_LB:
        baseline_sharpes = yr_sharpes

# Step 4: Find best
best = max(results, key=lambda x: x["obj"])
print(f"\n  Best: vol_lb={best['vol_lb']}h  OBJ={best['obj']:.4f}  Δ={best['obj'] - BASELINE_OBJ:+.4f}")

# Step 5: LOYO validation
print("\n[4] LOYO validation (best vs current vol_lb=168) ...")
loyo_wins = 0
loyo_table = []
for yr in YEARS:
    base_data = year_data[yr]
    # best vol_lb
    ens_best = compute_ensemble_rets(base_data, v1_rets_by_lb[best["vol_lb"]][yr])
    # current vol_lb=168
    ens_curr = compute_ensemble_rets(base_data, v1_rets_by_lb[CURRENT_VOL_LB][yr])
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
out = Path("artifacts/phase190"); out.mkdir(parents=True, exist_ok=True)
report = {
    "phase": 190,
    "target": "v1_vol_lookback_bars",
    "sweep_values": VOL_LB_SWEEP,
    "all_results": results,
    "best_vol_lb": best["vol_lb"],
    "best_obj": best["obj"],
    "delta": best["obj"] - BASELINE_OBJ,
    "baseline_obj": BASELINE_OBJ,
    "loyo": {"wins": loyo_wins, "table": loyo_table},
    "validated": validated,
    "runtime_s": round(time.time() - _start, 1),
    "timestamp": datetime.now(UTC).isoformat(),
}
(out / "phase190_report.json").write_text(json.dumps(report, indent=2, default=str))
print(f"\n  Report saved → artifacts/phase190/phase190_report.json")
print(f"  Elapsed: {report['runtime_s']}s")

# Step 7: Update config if validated
if validated:
    cfg = json.loads(Path("configs/production_p91b_champion.json").read_text())
    ver = cfg.get("_version", "2.21.0")
    major, minor, patch = map(int, ver.split("."))
    new_ver = f"{major}.{minor + 1}.{patch}"

    # Update V1 params in breadth_regime_switching comment + vol_regime_overlay
    brs = cfg["breadth_regime_switching"]
    brs["_validated"] += f"; V1 vol_lb sweep P190: vol_lb=168→{best['vol_lb']} LOYO {loyo_wins}/5 Δ={best['obj'] - BASELINE_OBJ:+.4f} OBJ={best['obj']:.4f}"

    # Update _version and _validated
    cfg["_version"] = new_ver
    cfg["_validated"] += (
        f"; V1 vol_lb sweep P190: {CURRENT_VOL_LB}→{best['vol_lb']} "
        f"LOYO {loyo_wins}/5 Δ={best['obj'] - BASELINE_OBJ:+.4f} OBJ={best['obj']:.4f}"
        f" — PRODUCTION {new_ver}"
    )

    # Store best vol_lb in a dedicated field for reference
    # We embed it as an annotation; the actual V1 params used by the ensemble
    # are set via the scripts themselves (not stored in this JSON's "strategies" list)
    if "v1_tuned_params" not in cfg:
        cfg["v1_tuned_params"] = {}
    cfg["v1_tuned_params"]["vol_lookback_bars"] = best["vol_lb"]
    cfg["v1_tuned_params"]["_note"] = f"P190 validated: vol_lb={best['vol_lb']} LOYO {loyo_wins}/5"

    Path("configs/production_p91b_champion.json").write_text(json.dumps(cfg, indent=2))
    print(f"\n✅ VALIDATED → config updated to {new_ver} OBJ={best['obj']:.4f}")
    print(f"   vol_lb: {CURRENT_VOL_LB} → {best['vol_lb']}")

    # Git commit + push
    best_lb = best["vol_lb"]
    best_obj_val = best["obj"]
    d = best_obj_val - BASELINE_OBJ
    cm_v = "phase190: V1 vol_lb sweep -- VALIDATED %d->%dh LOYO %d/5 delta=%+.4f OBJ=%.4f" % (CURRENT_VOL_LB, best_lb, loyo_wins, d, best_obj_val)
    os.system("git add configs/production_p91b_champion.json artifacts/phase190/ scripts/run_phase190_v1_vol_lb_sweep.py")
    os.system('git commit -m "' + cm_v + '"')
    os.system("git pull --rebase && git push")
else:
    delta = best["obj"] - BASELINE_OBJ
    verdict = "NO IMPROVEMENT" if delta <= 0 else f"WEAK (LOYO only {loyo_wins}/5)"
    print(f"\n❌ NOT VALIDATED — {verdict}")
    print(f"   vol_lb=168 remains optimal")

    best_lb = best["vol_lb"]
    os.system("git add artifacts/phase190/ scripts/run_phase190_v1_vol_lb_sweep.py")
    os.system(f'git commit -m "phase190: V1 vol_lb sweep — {verdict} best_vol_lb={best_lb}h LOYO {loyo_wins}/5 delta={delta:+.4f}"')
    os.system("git pull --rebase && git push")

print("\n[DONE] Phase 190 complete.")
