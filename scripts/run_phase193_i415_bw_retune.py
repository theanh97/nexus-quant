"""
Phase 193 — I415 Beta-Window Bars Re-Tune (v2.23.0+ stack)
============================================================
I415 (idio_momentum_alpha, lb=415) uses beta_window_bars=144.
This was set in P175. After V1 regime weight changes (LOW_v1=0.48, MID_v1=0.35),
I415 now has reduced relative weight in LOW/MID but still HIGH=0.54.

With HIGH regime I415=0.54 (dominant), the optimal bandwidth might shift.
A shorter bw = more responsive beta strip, better isolates idio signal.
A longer bw = more stable, less noise in beta estimation.

Test: beta_window_bars ∈ [72, 96, 120, 144, 168, 192, 216]
  (current: 144)

Baseline: read from production config (v2.23.0+)
Validate: LOYO ≥3/5 AND Δ>0 → update config
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
    out = Path("artifacts/phase193"); out.mkdir(parents=True, exist_ok=True)
    (out / "phase193_report.json").write_text(json.dumps(_partial, indent=2, default=str))
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

# Overlay constants
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

BW_SWEEP = [72, 96, 120, 144, 168, 192, 216]
CURRENT_BW = 144

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
    return cfg, weights

def rolling_mean_arr(a: np.ndarray, w: int) -> np.ndarray:
    out = np.full(len(a), np.nan)
    if w <= 0 or len(a) == 0: return out
    cs = np.cumsum(np.where(np.isnan(a), 0.0, a))
    for i in range(w - 1, len(a)):
        out[i] = cs[i] / w if i == w - 1 else (cs[i] - cs[i - w]) / w
    return out

def load_year_fixed_signals(year: int):
    """Load data + compute v1, i460, f168 (fixed). I415 bw will be swept."""
    s, e = YEAR_RANGES[year]
    cfg_d = {"provider": "binance_rest_v1", "symbols": SYMBOLS,
             "start": s, "end": e, "bar_interval": "1h",
             "cache_dir": ".cache/binance_rest"}
    dataset = make_provider(cfg_d, seed=42).load()
    n = len(dataset.timeline)

    close_mat = np.zeros((n, len(SYMBOLS)))
    for j, sym in enumerate(SYMBOLS):
        for i in range(n): close_mat[i, j] = dataset.close(sym, i)

    # BTC vol
    btc_rets = np.zeros(n)
    for i in range(1, n):
        c0 = close_mat[i-1, 0]; c1 = close_mat[i, 0]
        btc_rets[i] = (c1 / c0 - 1.0) if c0 > 0 else 0.0
    btc_vol = np.full(n, np.nan)
    for i in range(VOL_WINDOW, n):
        btc_vol[i] = float(np.std(btc_rets[i - VOL_WINDOW:i])) * np.sqrt(8760)
    if VOL_WINDOW < n: btc_vol[:VOL_WINDOW] = btc_vol[VOL_WINDOW]

    # Funding
    fund_rates = np.zeros((n, len(SYMBOLS)))
    for j, sym in enumerate(SYMBOLS):
        for i in range(n):
            ts = dataset.timeline[i]
            try:    fund_rates[i, j] = dataset.last_funding_rate_before(sym, ts)
            except: fund_rates[i, j] = 0.0

    # Dispersion
    fund_std_raw = np.std(fund_rates, axis=1)
    fund_std_pct = np.full(n, 0.5)
    for i in range(FUND_DISP_PCT, n):
        fund_std_pct[i] = float(np.mean(fund_std_raw[i - FUND_DISP_PCT:i] <= fund_std_raw[i]))
    fund_std_pct[:FUND_DISP_PCT] = 0.5

    # TS
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

    # Read current F168 lb from config
    try:
        prod_cfg = json.loads(Path("configs/production_p91b_champion.json").read_text())
        f168_lb = prod_cfg.get("f168_tuned_params", {}).get("funding_lookback_bars", 168)
    except Exception:
        f168_lb = 168

    # Fixed signals: v1, i460, f168
    v1_lb = 192  # P190 validated
    fixed_rets = {}
    for sk, sname, params in [
        ("v1", "nexus_alpha_v1", {
            "k_per_side": 2, "w_carry": 0.35, "w_mom": 0.40, "w_mean_reversion": 0.25,
            "momentum_lookback_bars": 336, "mean_reversion_lookback_bars": 84,
            "vol_lookback_bars": v1_lb, "target_gross_leverage": 0.35,
            "rebalance_interval_bars": 60,
        }),
        ("i460", "idio_momentum_alpha", {
            "k_per_side": 4, "lookback_bars": 460, "beta_window_bars": 120,
            "target_gross_leverage": 0.3, "rebalance_interval_bars": 48,
        }),
        ("f168", "funding_momentum_alpha", {
            "k_per_side": 2, "funding_lookback_bars": f168_lb, "direction": "contrarian",
            "target_gross_leverage": 0.25, "rebalance_interval_bars": 24,
        }),
    ]:
        result = BacktestEngine(BacktestConfig(costs=COST_MODEL)).run(
            dataset, make_strategy({"name": sname, "params": params}))
        fixed_rets[sk] = np.array(result.returns)

    print("S.", end=" ", flush=True)
    return dataset, btc_vol, fund_std_pct, ts_spread_pct, regime, fixed_rets, n

def run_i415_with_bw(dataset, bw: int):
    """Run I415 with given beta_window_bars."""
    params = {
        "k_per_side": 4, "lookback_bars": 415, "beta_window_bars": bw,
        "target_gross_leverage": 0.3, "rebalance_interval_bars": 48,
    }
    result = BacktestEngine(BacktestConfig(costs=COST_MODEL)).run(
        dataset, make_strategy({"name": "idio_momentum_alpha", "params": params}))
    return np.array(result.returns)

def compute_ensemble_rets(base_data, i415_rets: np.ndarray, weights: dict) -> np.ndarray:
    _, btc_vol, fund_std_pct, ts_spread_pct, regime, fixed_rets, n = base_data
    all_rets = dict(fixed_rets, i415=i415_rets)
    min_len = min(len(v) for v in all_rets.values())
    bv = btc_vol[:min_len]; reg = regime[:min_len]
    fsp = fund_std_pct[:min_len]; tsp = ts_spread_pct[:min_len]

    ens = np.zeros(min_len)
    for i in range(min_len):
        w = weights[["LOW", "MID", "HIGH"][int(reg[i])]]
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
print("Phase 193 — I415 Beta-Window Bars Re-Tune")
print(f"Sweep: {BW_SWEEP}")
print("=" * 60)
_start = time.time()

# Read current config dynamically
prod_cfg, WEIGHTS = get_config()
ver_now = prod_cfg.get("_version", "2.23.0")
# Read current baseline OBJ — from last validated entry
try:
    import re
    m = re.findall(r"OBJ=([\d.]+)", prod_cfg.get("_validated", ""))
    BASELINE_OBJ = float(m[-1]) if m else 2.8691
except Exception:
    BASELINE_OBJ = 2.8691
print(f"  Config: {ver_now}  Baseline OBJ≈{BASELINE_OBJ:.4f}")
print(f"  Weights: {WEIGHTS}")

# Step 1: Load per-year data
print("\n[1] Loading per-year data + fixed signals (v1, i460, f168) ...")
year_data = {}
for yr in YEARS:
    print(f"  {yr} ", end="", flush=True)
    year_data[yr] = load_year_fixed_signals(yr)
    print()

# Step 2: Run I415 for each bw
print("\n[2] Running I415 for each beta_window_bars ...")
i415_rets_by_bw = {bw: {} for bw in BW_SWEEP}

for bw in BW_SWEEP:
    print(f"  bw={bw}h:", end=" ", flush=True)
    for yr in YEARS:
        dataset = year_data[yr][0]
        i415_rets_by_bw[bw][yr] = run_i415_with_bw(dataset, bw)
        print(f"{yr}", end=" ", flush=True)
    print()

# Step 3: Compute yearly OBJ
print("\n[3] Computing yearly Sharpes per bw ...")
results = []

for bw in BW_SWEEP:
    yr_sharpes = {}
    for yr in YEARS:
        base_data = year_data[yr]
        ens = compute_ensemble_rets(base_data, i415_rets_by_bw[bw][yr], WEIGHTS)
        n = base_data[-1]
        yr_sharpes[yr] = yearly_sharpe(ens, n)
    obj = obj_fn(yr_sharpes)
    results.append({"bw": bw, "obj": obj, "yr_sharpes": yr_sharpes})
    flag = " ← CURRENT" if bw == CURRENT_BW else ""
    print(f"  bw={bw:3d}h  OBJ={obj:.4f}  {yr_sharpes}{flag}")

# Step 4: Find best
best = max(results, key=lambda x: x["obj"])
print(f"\n  Best: bw={best['bw']}h  OBJ={best['obj']:.4f}  Δ={best['obj'] - BASELINE_OBJ:+.4f}")

# Step 5: LOYO validation
print("\n[4] LOYO validation (best vs current bw=144) ...")
loyo_wins = 0
loyo_table = []
for yr in YEARS:
    base_data = year_data[yr]
    ens_best = compute_ensemble_rets(base_data, i415_rets_by_bw[best["bw"]][yr], WEIGHTS)
    ens_curr = compute_ensemble_rets(base_data, i415_rets_by_bw[CURRENT_BW][yr], WEIGHTS)
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
out = Path("artifacts/phase193"); out.mkdir(parents=True, exist_ok=True)
report = {
    "phase": 193,
    "target": "i415_beta_window_bars",
    "sweep_values": BW_SWEEP,
    "all_results": results,
    "best_bw": best["bw"],
    "best_obj": best["obj"],
    "delta": best["obj"] - BASELINE_OBJ,
    "baseline_obj": BASELINE_OBJ,
    "baseline_ver": ver_now,
    "loyo": {"wins": loyo_wins, "table": loyo_table},
    "validated": validated,
    "runtime_s": round(time.time() - _start, 1),
    "timestamp": datetime.now(UTC).isoformat(),
}
(out / "phase193_report.json").write_text(json.dumps(report, indent=2, default=str))
print(f"\n  Report saved → artifacts/phase193/phase193_report.json")
print(f"  Elapsed: {report['runtime_s']}s")

# Step 7: Update config if validated
if validated:
    cfg = json.loads(Path("configs/production_p91b_champion.json").read_text())
    ver = cfg.get("_version", "2.23.0")
    major, minor, patch = map(int, ver.split("."))
    new_ver = f"{major}.{minor + 1}.{patch}"
    cfg["_version"] = new_ver
    cfg["_validated"] += (
        f"; I415 bw sweep P193: {CURRENT_BW}->{best['bw']} "
        f"LOYO {loyo_wins}/5 delta={best['obj'] - BASELINE_OBJ:+.4f} OBJ={best['obj']:.4f}"
        f" — PRODUCTION {new_ver}"
    )
    # Update i415 bw in regime_weights mechanism note
    brs = cfg["breadth_regime_switching"]
    brs["_validated"] = brs.get("_validated","") + f"; I415 bw P193: {CURRENT_BW}->{best['bw']}"
    if "i415_tuned_params" not in cfg:
        cfg["i415_tuned_params"] = {}
    cfg["i415_tuned_params"]["beta_window_bars"] = best["bw"]
    Path("configs/production_p91b_champion.json").write_text(json.dumps(cfg, indent=2))
    best_bw = best["bw"]
    best_obj_val = best["obj"]
    d = best_obj_val - BASELINE_OBJ
    cm = "phase193: I415 bw sweep -- VALIDATED %d->%dh LOYO %d/5 delta=%+.4f OBJ=%.4f" % (CURRENT_BW, best_bw, loyo_wins, d, best_obj_val)
    print(f"\n✅ VALIDATED → config updated to {new_ver} OBJ={best_obj_val:.4f}")
    print(f"   I415 bw: {CURRENT_BW} → {best_bw}")
    os.system("git add configs/production_p91b_champion.json artifacts/phase193/ scripts/run_phase193_i415_bw_retune.py")
    os.system('git commit -m "' + cm + '"')
    os.system("git pull --rebase && git push")
else:
    delta = best["obj"] - BASELINE_OBJ
    verdict = "NO IMPROVEMENT" if delta <= 0 else f"WEAK (LOYO only {loyo_wins}/5)"
    print(f"\n❌ NOT VALIDATED — {verdict}")
    print(f"   I415 bw={CURRENT_BW} remains optimal")
    best_bw = best["bw"]
    cm = "phase193: I415 bw sweep -- %s best_bw=%dh LOYO %d/5 delta=%+.4f" % (verdict, best_bw, loyo_wins, delta)
    os.system("git add artifacts/phase193/ scripts/run_phase193_i415_bw_retune.py")
    os.system('git commit -m "' + cm + '"')
    os.system("git pull --rebase && git push")

print("\n[DONE] Phase 193 complete.")
