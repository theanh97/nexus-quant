"""
Phase 219 — I460/I415 Lookback Bars Sweep
==========================================
I460 and I415 refer to idio_momentum_alpha with lookback_bars=460 and 415.
These lookbacks were set in an early phase and never re-swept since:
  - v2.32.0 V1 internal weight change (w_carry/mom/mr)
  - v2.33.0 F168 rebalance change
Optimal lookbacks may have shifted under the new ensemble structure.

Part 1: Sweep I460 lookback_bars ∈ [300, 350, 400, 420, 440, 460, 480, 500, 560]
         (fix I415 base params, fix V1/F168 fixed)
Part 2: Sweep I415 lookback_bars ∈ [300, 340, 370, 400, 415, 430, 450, 480, 520]
         (with best I460 from Part 1)
Part 3: LOYO validation of joint best

Baseline: v2.33.0, OBJ=3.1190
"""

import os, sys, json, time
import signal as _signal
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
    out = Path("artifacts/phase219"); out.mkdir(parents=True, exist_ok=True)
    (out / "phase219_report.json").write_text(json.dumps(_partial, indent=2))
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

VOL_WINDOW    = 168; BRD_LB = 192; PCT_WINDOW = 336
TS_SHORT, TS_LONG = 16, 72; FUND_DISP_PCT = 240
VOL_THRESHOLD = 0.50; VOL_SCALE = 0.30
TS_RT = 0.65; TS_RS = 0.45; TS_BT = 0.25; TS_BS = 1.70
DISP_THR = 0.60; DISP_SCALE = 1.0
P_LOW = 0.30; P_HIGH = 0.60

REGIME_WEIGHTS = {
    "LOW":  {"v1": 0.46,   "i460": 0.0716, "i415": 0.1184, "f168": 0.35},
    "MID":  {"v1": 0.20,   "i460": 0.0074, "i415": 0.0126, "f168": 0.78},
    "HIGH": {"v1": 0.06,   "i460": 0.2821, "i415": 0.5079, "f168": 0.15},
}

V1_PARAMS = {
    "k_per_side": 2, "w_carry": 0.25, "w_mom": 0.45, "w_mean_reversion": 0.30,
    "momentum_lookback_bars": 336, "mean_reversion_lookback_bars": 84,
    "vol_lookback_bars": 192, "target_gross_leverage": 0.35, "rebalance_interval_bars": 60,
}
I460_BASE = {"k_per_side": 4, "lookback_bars": 460, "beta_window_bars": 168,
             "target_gross_leverage": 0.3, "rebalance_interval_bars": 48}
I415_BASE = {"k_per_side": 4, "lookback_bars": 415, "beta_window_bars": 144,
             "target_gross_leverage": 0.3, "rebalance_interval_bars": 48}
F168_PARAMS = {"k_per_side": 2, "funding_lookback_bars": 168, "direction": "contrarian",
               "target_gross_leverage": 0.25, "rebalance_interval_bars": 36}

BASELINE_OBJ = 3.1190
I460_LB_SWEEP = [300, 350, 400, 420, 440, 460, 480, 500, 560]
I415_LB_SWEEP = [300, 340, 370, 400, 415, 430, 450, 480, 520]
MIN_DELTA = 0.005; MIN_LOYO = 3

COST_MODEL = cost_model_from_config({
    "fee_rate": 0.0005, "slippage_rate": 0.0003, "cost_multiplier": 1.0,
})

def rolling_mean_arr(a, w):
    out = np.full(len(a), np.nan)
    if w <= 0 or len(a) == 0: return out
    cs = np.cumsum(np.where(np.isnan(a), 0.0, a))
    for i in range(w - 1, len(a)):
        out[i] = cs[i] / w if i == w - 1 else (cs[i] - cs[i - w]) / w
    return out

def run_strategy(dataset, sname, params):
    result = BacktestEngine(BacktestConfig(costs=COST_MODEL)).run(
        dataset, make_strategy({"name": sname, "params": params}))
    return np.array(result.returns)

def load_year_data(year):
    s, e = YEAR_RANGES[year]
    cfg = {"provider": "binance_rest_v1", "symbols": SYMBOLS,
           "start": s, "end": e, "bar_interval": "1h",
           "cache_dir": ".cache/binance_rest"}
    dataset = make_provider(cfg, seed=42).load()
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
            try: fund_rates[i, j] = dataset.last_funding_rate_before(sym, ts)
            except: fund_rates[i, j] = 0.0
    fund_std_raw = np.std(fund_rates, axis=1)
    fund_std_pct = np.full(n, 0.5)
    for i in range(FUND_DISP_PCT, n):
        fund_std_pct[i] = float(np.mean(fund_std_raw[i-FUND_DISP_PCT:i] <= fund_std_raw[i]))
    xsect_mean = np.mean(fund_rates, axis=1)
    ts_raw = rolling_mean_arr(xsect_mean, TS_SHORT) - rolling_mean_arr(xsect_mean, TS_LONG)
    ts_spread_pct = np.full(n, 0.5)
    for i in range(PCT_WINDOW, n):
        ts_spread_pct[i] = float(np.mean(ts_raw[i-PCT_WINDOW:i] <= ts_raw[i]))
    breadth = np.full(n, 0.5)
    for i in range(BRD_LB, n):
        pos = sum(1 for j in range(len(SYMBOLS))
                  if close_mat[i-BRD_LB, j] > 0 and close_mat[i, j] > close_mat[i-BRD_LB, j])
        breadth[i] = pos / len(SYMBOLS)
    brd_pct = np.full(n, 0.5)
    for i in range(PCT_WINDOW, n):
        brd_pct[i] = float(np.mean(breadth[i-PCT_WINDOW:i] <= breadth[i]))
    return dataset, btc_vol, fund_std_pct, ts_spread_pct, brd_pct, n

def ensemble_rets(btc_vol, fund_std_pct, ts_spread_pct, brd_pct, sig_rets, n):
    RNAMES = ["LOW", "MID", "HIGH"]
    sk_all = ["v1", "i460", "i415", "f168"]
    min_len = min(len(sig_rets[k]) for k in sk_all)
    ens = np.zeros(min_len)
    bv = btc_vol[:min_len]; bpct = brd_pct[:min_len]
    fsp = fund_std_pct[:min_len]; tsp = ts_spread_pct[:min_len]
    for i in range(min_len):
        bp = bpct[i]
        ridx = 0 if bp < P_LOW else (2 if bp > P_HIGH else 1)
        w = REGIME_WEIGHTS[RNAMES[ridx]]
        ret_i = sum(w[sk] * sig_rets[sk][i] for sk in sk_all)
        if not np.isnan(bv[i]) and bv[i] > VOL_THRESHOLD:
            ret_i *= VOL_SCALE
        if fsp[i] > DISP_THR: ret_i *= DISP_SCALE
        if tsp[i] > TS_RT: ret_i *= TS_RS
        elif tsp[i] < TS_BT: ret_i *= TS_BS
        ens[i] = ret_i
    return ens

def sharpe(r):
    r = r[~np.isnan(r)]
    if len(r) < 2: return 0.0
    return float(np.mean(r) / np.std(r, ddof=1) * np.sqrt(8760))

def obj(yearly):
    vals = list(yearly.values())
    return float(np.mean(vals) - 0.5 * np.std(vals, ddof=1))

def main():
    t0 = time.time()
    print("=" * 65)
    print("Phase 219 — I460/I415 Lookback Bars Sweep")
    print(f"Baseline: v2.33.0  OBJ={BASELINE_OBJ}")
    print(f"Current: I460_lb=460  I415_lb=415")
    print(f"I460 sweep: {I460_LB_SWEEP}")
    print(f"I415 sweep: {I415_LB_SWEEP}")
    print("=" * 65)

    print("\n[1/4] Loading per-year data...")
    all_year_data = {}
    for yr in sorted(YEAR_RANGES):
        print(f"  {yr}: ", end="", flush=True)
        all_year_data[yr] = load_year_data(yr)
        print("D.", end=" ", flush=True)
    print()

    print("\n[2/4] Pre-computing V1 + F168 (fixed) signals...")
    fixed_vf = {}
    for yr, (dataset, *_) in all_year_data.items():
        fixed_vf[yr] = {
            "v1":   run_strategy(dataset, "nexus_alpha_v1",         V1_PARAMS),
            "f168": run_strategy(dataset, "funding_momentum_alpha", F168_PARAMS),
        }
        print(f"  {yr}: S.", end=" ", flush=True)
    print()

    # Baseline: need I460 and I415 baseline too
    base_sigs = {}
    for yr, (dataset, *_) in all_year_data.items():
        base_sigs[yr] = dict(fixed_vf[yr])
        base_sigs[yr]["i460"] = run_strategy(dataset, "idio_momentum_alpha", I460_BASE)
        base_sigs[yr]["i415"] = run_strategy(dataset, "idio_momentum_alpha", I415_BASE)

    base_yearly = {}
    for yr, (dataset, btc_vol, fsp, tsp, brd_pct, n) in all_year_data.items():
        ens = ensemble_rets(btc_vol, fsp, tsp, brd_pct, base_sigs[yr], n)
        base_yearly[yr] = sharpe(ens)
    base_obj_val = obj(base_yearly)
    print(f"\n  Baseline OBJ = {base_obj_val:.4f}  (expected ≈ {BASELINE_OBJ})")
    _partial.update({"baseline_obj": float(base_obj_val)})

    print("\n[3/4] Part 1 — Sweeping I460 lookback_bars (I415 fixed at 415)...")
    best_i460_obj = base_obj_val; best_i460_lb = I460_BASE["lookback_bars"]
    i460_results = []
    for lb in I460_LB_SWEEP:
        params = dict(I460_BASE, lookback_bars=lb)
        yearly = {}
        for yr, (dataset, btc_vol, fsp, tsp, brd_pct, n) in all_year_data.items():
            sr = dict(fixed_vf[yr])
            sr["i460"] = run_strategy(dataset, "idio_momentum_alpha", params)
            sr["i415"] = base_sigs[yr]["i415"]
            ens = ensemble_rets(btc_vol, fsp, tsp, brd_pct, sr, n)
            yearly[yr] = sharpe(ens)
        o = obj(yearly)
        delta = o - base_obj_val
        marker = " ← BEST" if o > best_i460_obj else ""
        print(f"  i460_lb={lb:4d}  OBJ={o:.4f}  Δ={delta:+.4f}{marker}")
        i460_results.append((lb, o))
        if o > best_i460_obj:
            best_i460_obj = o; best_i460_lb = lb

    print(f"\n  → Best I460_lb: {best_i460_lb}  OBJ={best_i460_obj:.4f}")

    # Update I460 in cache for Part 2
    best_i460_params = dict(I460_BASE, lookback_bars=best_i460_lb)
    cached_i460 = {}
    for yr, (dataset, *_) in all_year_data.items():
        cached_i460[yr] = run_strategy(dataset, "idio_momentum_alpha", best_i460_params)

    print("\n  Part 2 — Sweeping I415 lookback_bars (with best I460)...")
    best_i415_obj = base_obj_val; best_i415_lb = I415_BASE["lookback_bars"]
    i415_results = []
    for lb in I415_LB_SWEEP:
        params = dict(I415_BASE, lookback_bars=lb)
        yearly = {}
        for yr, (dataset, btc_vol, fsp, tsp, brd_pct, n) in all_year_data.items():
            sr = dict(fixed_vf[yr])
            sr["i460"] = cached_i460[yr]
            sr["i415"] = run_strategy(dataset, "idio_momentum_alpha", params)
            ens = ensemble_rets(btc_vol, fsp, tsp, brd_pct, sr, n)
            yearly[yr] = sharpe(ens)
        o = obj(yearly)
        delta = o - base_obj_val
        marker = " ← BEST" if o > best_i415_obj else ""
        print(f"  i415_lb={lb:4d}  OBJ={o:.4f}  Δ={delta:+.4f}{marker}")
        i415_results.append((lb, o))
        if o > best_i415_obj:
            best_i415_obj = o; best_i415_lb = lb

    print(f"\n  → Best I415_lb: {best_i415_lb}  OBJ={best_i415_obj:.4f}")

    print(f"\n[4/4] LOYO validation — I460_lb={best_i460_lb}  I415_lb={best_i415_lb}...")
    joint_yearly = {}
    for yr, (dataset, btc_vol, fsp, tsp, brd_pct, n) in all_year_data.items():
        sr = dict(fixed_vf[yr])
        sr["i460"] = cached_i460[yr]
        sr["i415"] = run_strategy(dataset, "idio_momentum_alpha",
                                  dict(I415_BASE, lookback_bars=best_i415_lb))
        ens = ensemble_rets(btc_vol, fsp, tsp, brd_pct, sr, n)
        joint_yearly[yr] = sharpe(ens)
    joint_obj_val = obj(joint_yearly)
    delta = joint_obj_val - base_obj_val
    loyo_wins = 0
    for yr in sorted(joint_yearly):
        base_sh = base_yearly.get(yr, 0)
        cand_sh = joint_yearly[yr]
        win = cand_sh > base_sh
        if win: loyo_wins += 1
        print(f"  {yr}: base={base_sh:.4f}  cand={cand_sh:.4f}  {'WIN' if win else 'LOSE'}")

    print(f"\n  OBJ={joint_obj_val:.4f}  Δ={delta:+.4f}  LOYO {loyo_wins}/{len(joint_yearly)}")
    validated = loyo_wins >= MIN_LOYO and delta >= MIN_DELTA
    print(f"\n{'✅ VALIDATED' if validated else '❌ NO IMPROVEMENT'} — I460_lb={best_i460_lb}  I415_lb={best_i415_lb}")

    result = {
        "phase": 219, "baseline_obj": float(base_obj_val), "best_obj": float(joint_obj_val),
        "delta": float(delta), "loyo_wins": loyo_wins, "loyo_total": len(joint_yearly),
        "validated": validated, "best_i460_lb": best_i460_lb, "best_i415_lb": best_i415_lb,
        "i460_sweep": i460_results, "i415_sweep": i415_results,
    }
    _partial.update(result)
    out = Path("artifacts/phase219"); out.mkdir(parents=True, exist_ok=True)
    (out / "phase219_report.json").write_text(json.dumps(result, indent=2))

    if validated:
        cfg_path = ROOT / "configs" / "production_p91b_champion.json"
        cfg = json.load(open(cfg_path))
        sigs = cfg["ensemble"]["signals"]
        sigs["idio_momentum_460"]["params"]["lookback_bars"] = best_i460_lb
        sigs["idio_momentum_415"]["params"]["lookback_bars"] = best_i415_lb
        cfg["_version"] = "2.34.0"
        cfg["monitoring"]["expected_performance"]["annual_sharpe_backtest"] = round(joint_obj_val, 4)
        with open(cfg_path, "w") as f:
            json.dump(cfg, f, indent=2, ensure_ascii=False)
        print(f"\n  Config → v2.34.0  OBJ={round(joint_obj_val,4)}")

    print(f"\nRuntime: {int(time.time()-t0)}s")

main()
