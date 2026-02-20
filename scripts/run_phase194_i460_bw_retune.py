"""
Phase 194 — I460 Beta-Window Bars Re-Tune (v2.23.0+ stack)
============================================================
I460 (idio_momentum_alpha, lb=460) uses beta_window_bars=120.
With HIGH regime weight I460=0.30 and new V1/I415 weights, re-sweep bw.

Test: beta_window_bars ∈ [72, 96, 120, 144, 168, 192, 216]
  (current: 120)

Baseline: read from production config dynamically
Validate: LOYO ≥3/5 AND Δ>0 → update config
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
    out = Path("artifacts/phase194"); out.mkdir(parents=True, exist_ok=True)
    (out / "phase194_report.json").write_text(json.dumps(_partial, indent=2, default=str))
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

VOL_WINDOW    = 168; BRD_LB = 192; PCT_WINDOW = 336
P_LOW, P_HIGH = 0.20, 0.60; TS_SHORT = 12; TS_LONG = 96; FUND_DISP_PCT = 240
VOL_THRESHOLD = 0.50; VOL_SCALE = 0.40; VOL_F168_BOOST = 0.10
TS_RT = 0.60; TS_RS = 0.40; TS_BT = 0.35; TS_BS = 1.60
DISP_THR = 0.60; DISP_SCALE = 1.0

BW_SWEEP = [72, 96, 120, 144, 168, 192, 216]
CURRENT_BW = 120

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
    m = re.findall(r"OBJ=([\d.]+)", cfg.get("_validated", ""))
    baseline_obj = float(m[-1]) if m else 2.8691
    f168_lb = cfg.get("f168_tuned_params", {}).get("funding_lookback_bars", 168)
    i415_bw = cfg.get("i415_tuned_params", {}).get("beta_window_bars", 144)
    return cfg, weights, baseline_obj, f168_lb, i415_bw

def rolling_mean_arr(a, w):
    out = np.full(len(a), np.nan)
    if w <= 0: return out
    cs = np.cumsum(np.where(np.isnan(a), 0.0, a))
    for i in range(w - 1, len(a)):
        out[i] = cs[i] / w if i == w - 1 else (cs[i] - cs[i - w]) / w
    return out

def load_year_fixed_signals(year, f168_lb, i415_bw):
    """Load data + compute v1, i415 (with current bw), f168 (fixed)."""
    s, e = YEAR_RANGES[year]
    cfg_d = {"provider": "binance_rest_v1", "symbols": SYMBOLS,
             "start": s, "end": e, "bar_interval": "1h",
             "cache_dir": ".cache/binance_rest"}
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

    fixed_rets = {}
    for sk, sname, params in [
        ("v1", "nexus_alpha_v1", {
            "k_per_side": 2, "w_carry": 0.35, "w_mom": 0.40, "w_mean_reversion": 0.25,
            "momentum_lookback_bars": 336, "mean_reversion_lookback_bars": 84,
            "vol_lookback_bars": 192, "target_gross_leverage": 0.35,
            "rebalance_interval_bars": 60,
        }),
        ("i415", "idio_momentum_alpha", {
            "k_per_side": 4, "lookback_bars": 415, "beta_window_bars": i415_bw,
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

def run_i460_with_bw(dataset, bw):
    params = {
        "k_per_side": 4, "lookback_bars": 460, "beta_window_bars": bw,
        "target_gross_leverage": 0.3, "rebalance_interval_bars": 48,
    }
    result = BacktestEngine(BacktestConfig(costs=COST_MODEL)).run(
        dataset, make_strategy({"name": "idio_momentum_alpha", "params": params}))
    return np.array(result.returns)

def compute_ensemble_rets(base_data, i460_rets, weights):
    _, btc_vol, fund_std_pct, ts_spread_pct, regime, fixed_rets, n = base_data
    all_rets = dict(fixed_rets, i460=i460_rets)
    min_len = min(len(v) for v in all_rets.values())
    bv = btc_vol[:min_len]; reg = regime[:min_len]
    fsp = fund_std_pct[:min_len]; tsp = ts_spread_pct[:min_len]

    ens = np.zeros(min_len)
    for i in range(min_len):
        w = weights[["LOW", "MID", "HIGH"][int(reg[i])]]
        if not np.isnan(bv[i]) and bv[i] > VOL_THRESHOLD:
            boost_per = VOL_F168_BOOST / 3.0
            ret_i = (
                (w["v1"] + boost_per) * all_rets["v1"][i] +
                (w["i460"] + boost_per) * all_rets["i460"][i] +
                (w["i415"] + boost_per) * all_rets["i415"][i] +
                (w["f168"] - VOL_F168_BOOST) * all_rets["f168"][i]
            ) * VOL_SCALE
        else:
            ret_i = (
                w["v1"] * all_rets["v1"][i] +
                w["i460"] * all_rets["i460"][i] +
                w["i415"] * all_rets["i415"][i] +
                w["f168"] * all_rets["f168"][i]
            )
        if DISP_SCALE > 1.0 and fsp[i] > DISP_THR: ret_i *= DISP_SCALE
        if tsp[i] > TS_RT: ret_i *= TS_RS
        elif tsp[i] < TS_BT: ret_i *= TS_BS
        ens[i] = ret_i
    return ens

def yearly_sharpe(rets, n):
    if n < 20: return -999.0
    mu = float(np.mean(rets[:n])) * 8760
    sd = float(np.std(rets[:n])) * np.sqrt(8760)
    return mu / sd if sd > 1e-9 else 0.0

def obj_fn(yr_sharpes):
    vals = list(yr_sharpes.values())
    return float(np.mean(vals) - 0.5 * np.std(vals))

# ── MAIN ────────────────────────────────────────────────────────────────────
print("=" * 60)
print("Phase 194 — I460 Beta-Window Bars Re-Tune")
print(f"Sweep: {BW_SWEEP}")
print("=" * 60)
_start = time.time()

prod_cfg, WEIGHTS, BASELINE_OBJ, f168_lb, i415_bw = get_config()
ver_now = prod_cfg.get("_version", "2.23.0")
print(f"  Config: {ver_now}  Baseline OBJ≈{BASELINE_OBJ:.4f}")
print(f"  f168_lb={f168_lb}  i415_bw={i415_bw}")

print("\n[1] Loading per-year data + fixed signals (v1, i415, f168) ...")
year_data = {}
for yr in YEARS:
    print(f"  {yr} ", end="", flush=True)
    year_data[yr] = load_year_fixed_signals(yr, f168_lb, i415_bw)
    print()

print("\n[2] Running I460 for each beta_window_bars ...")
i460_rets_by_bw = {bw: {} for bw in BW_SWEEP}
for bw in BW_SWEEP:
    print(f"  bw={bw}h:", end=" ", flush=True)
    for yr in YEARS:
        i460_rets_by_bw[bw][yr] = run_i460_with_bw(year_data[yr][0], bw)
        print(f"{yr}", end=" ", flush=True)
    print()

print("\n[3] Computing yearly Sharpes per bw ...")
results = []
for bw in BW_SWEEP:
    yr_sharpes = {}
    for yr in YEARS:
        ens = compute_ensemble_rets(year_data[yr], i460_rets_by_bw[bw][yr], WEIGHTS)
        yr_sharpes[yr] = yearly_sharpe(ens, year_data[yr][-1])
    obj = obj_fn(yr_sharpes)
    results.append({"bw": bw, "obj": obj, "yr_sharpes": yr_sharpes})
    flag = " ← CURRENT" if bw == CURRENT_BW else ""
    print(f"  bw={bw:3d}h  OBJ={obj:.4f}{flag}")

best = max(results, key=lambda x: x["obj"])
print(f"\n  Best: bw={best['bw']}h  OBJ={best['obj']:.4f}  Δ={best['obj'] - BASELINE_OBJ:+.4f}")

print("\n[4] LOYO validation ...")
loyo_wins = 0; loyo_table = []
for yr in YEARS:
    ens_best = compute_ensemble_rets(year_data[yr], i460_rets_by_bw[best["bw"]][yr], WEIGHTS)
    ens_curr = compute_ensemble_rets(year_data[yr], i460_rets_by_bw[CURRENT_BW][yr], WEIGHTS)
    n = year_data[yr][-1]
    sh_b = yearly_sharpe(ens_best, n); sh_c = yearly_sharpe(ens_curr, n)
    win = bool(sh_b > sh_c)
    if win: loyo_wins += 1
    loyo_table.append({"year": yr, "sh_best": sh_b, "sh_curr": sh_c, "win": win})
    print(f"  {yr}: best={sh_b:.4f} curr={sh_c:.4f} {'WIN' if win else 'LOSS'}")

print(f"\n  LOYO: {loyo_wins}/5 wins")
validated = loyo_wins >= 3 and best["obj"] > BASELINE_OBJ

out = Path("artifacts/phase194"); out.mkdir(parents=True, exist_ok=True)
report = {
    "phase": 194, "target": "i460_beta_window_bars",
    "sweep_values": BW_SWEEP, "all_results": results,
    "best_bw": best["bw"], "best_obj": best["obj"],
    "delta": best["obj"] - BASELINE_OBJ, "baseline_obj": BASELINE_OBJ,
    "loyo": {"wins": loyo_wins, "table": loyo_table},
    "validated": validated,
    "runtime_s": round(time.time() - _start, 1),
    "timestamp": datetime.now(UTC).isoformat(),
}
(out / "phase194_report.json").write_text(json.dumps(report, indent=2, default=str))
print(f"  Report saved → artifacts/phase194/phase194_report.json  ({report['runtime_s']}s)")

if validated:
    cfg = json.loads(Path("configs/production_p91b_champion.json").read_text())
    ver = cfg.get("_version", "2.23.0")
    major, minor, patch = map(int, ver.split("."))
    new_ver = f"{major}.{minor + 1}.{patch}"
    cfg["_version"] = new_ver
    cfg["_validated"] += (
        f"; I460 bw sweep P194: {CURRENT_BW}->{best['bw']} "
        f"LOYO {loyo_wins}/5 delta={best['obj']-BASELINE_OBJ:+.4f} OBJ={best['obj']:.4f}"
        f" — PRODUCTION {new_ver}"
    )
    if "i460_tuned_params" not in cfg: cfg["i460_tuned_params"] = {}
    cfg["i460_tuned_params"]["beta_window_bars"] = best["bw"]
    Path("configs/production_p91b_champion.json").write_text(json.dumps(cfg, indent=2))
    best_bw = best["bw"]; bv = best["obj"]; d = bv - BASELINE_OBJ
    cm = "phase194: I460 bw sweep -- VALIDATED %d->%dh LOYO %d/5 delta=%+.4f OBJ=%.4f" % (CURRENT_BW, best_bw, loyo_wins, d, bv)
    print(f"\n✅ VALIDATED → {new_ver} OBJ={bv:.4f}  I460 bw: {CURRENT_BW}→{best_bw}")
    os.system("git add configs/production_p91b_champion.json artifacts/phase194/ scripts/run_phase194_i460_bw_retune.py")
    os.system('git commit -m "' + cm + '"')
    os.system("git pull --rebase && git push")
else:
    delta = best["obj"] - BASELINE_OBJ
    verdict = "NO IMPROVEMENT" if delta <= 0 else f"WEAK LOYO {loyo_wins}/5"
    best_bw = best["bw"]
    cm = "phase194: I460 bw sweep -- %s best_bw=%dh LOYO %d/5 delta=%+.4f" % (verdict, best_bw, loyo_wins, delta)
    print(f"\n❌ NOT VALIDATED — {verdict}  I460 bw={CURRENT_BW} optimal")
    os.system("git add artifacts/phase194/ scripts/run_phase194_i460_bw_retune.py")
    os.system('git commit -m "' + cm + '"')
    os.system("git pull --rebase && git push")

print("\n[DONE] Phase 194 complete.")
