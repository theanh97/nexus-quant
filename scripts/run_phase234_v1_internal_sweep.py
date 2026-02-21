"""
Phase 234 — V1 Internal Weights + Lookback Sweep
==================================================
P233 validated regime weights (Δ=+0.0684, LOYO 3/5, OBJ=3.7718).
V1 internal params haven't been tuned with the current overlay stack:
  - w_carry=0.25, w_mom=0.45, w_mean_reversion=0.30 (fixed for many phases)
  - momentum_lookback_bars=336
  - mean_reversion_lookback_bars=84

In HIGH regime, v1=0.18 (significant); in LOW v1=0.44 (dominant).
Getting V1's internal allocation right is material.

Approach:
  Part 1: Sweep w_carry × w_mom (w_mr = 1 - w_carry - w_mom)
    w_carry: [0.0, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40]
    w_mom:   [0.20, 0.30, 0.40, 0.45, 0.50, 0.55, 0.60, 0.70]

  Part 2: Sweep momentum_lookback × mean_reversion_lookback (best weights from P1)
    mom_lb:  [168, 240, 288, 336, 400, 480]
    mr_lb:   [48, 60, 72, 84, 96, 120, 168]

Pre-compute I460/I415/F168 signals (fixed), vary only V1.

Baseline: v2.41.0, OBJ=3.7718
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
    out = Path("artifacts/phase234"); out.mkdir(parents=True, exist_ok=True)
    (out / "phase234_report.json").write_text(json.dumps(_partial, indent=2))
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

BRD_LB = 192; PCT_WINDOW = 336; FUND_DISP_PCT = 240
TS_PCT_WIN = 288
VOL_WINDOW = 168
TS_SHORT = 16; TS_LONG = 72
P_LOW = 0.30; P_HIGH = 0.60

FTS_RS = {"LOW": 0.50, "MID": 0.20, "HIGH": 0.40}
FTS_BS = {"LOW": 3.00, "MID": 3.00, "HIGH": 2.00}
FTS_RT = {"LOW": 0.80, "MID": 0.65, "HIGH": 0.55}
FTS_BT = {"LOW": 0.30, "MID": 0.25, "HIGH": 0.25}
VOL_THR   = {"LOW": 0.50, "MID": 0.50, "HIGH": 0.50}
VOL_SCALE = {"LOW": 0.40, "MID": 0.15, "HIGH": 0.10}
DISP_THR   = {"LOW": 0.70, "MID": 0.70, "HIGH": 0.40}
DISP_SCALE = {"LOW": 0.50, "MID": 1.50, "HIGH": 0.50}

REGIME_WEIGHTS = {
    "LOW":  {"v1": 0.44,   "i460": 0.0864, "i415": 0.1035, "f168": 0.37},
    "MID":  {"v1": 0.20,   "i460": 0.0000, "i415": 0.0200, "f168": 0.78},
    "HIGH": {"v1": 0.18,   "i460": 0.7900, "i415": 0.0000, "f168": 0.03},
}

# Base V1 params
V1_BASE = {
    "k_per_side": 2, "w_carry": 0.25, "w_mom": 0.45, "w_mean_reversion": 0.30,
    "momentum_lookback_bars": 336, "mean_reversion_lookback_bars": 84,
    "vol_lookback_bars": 192, "target_gross_leverage": 0.35, "rebalance_interval_bars": 60,
}
I460_PARAMS = {"k_per_side": 4, "lookback_bars": 480, "beta_window_bars": 168,
               "target_gross_leverage": 0.3, "rebalance_interval_bars": 48}
I415_PARAMS = {"k_per_side": 4, "lookback_bars": 415, "beta_window_bars": 144,
               "target_gross_leverage": 0.3, "rebalance_interval_bars": 48}
F168_PARAMS = {"k_per_side": 2, "funding_lookback_bars": 168, "direction": "contrarian",
               "target_gross_leverage": 0.25, "rebalance_interval_bars": 36}

BASELINE_OBJ = 3.7718
W_CARRY_SWEEP = [0.0, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40]
W_MOM_SWEEP   = [0.20, 0.30, 0.40, 0.45, 0.50, 0.55, 0.60, 0.70]
MOM_LB_SWEEP  = [168, 240, 288, 336, 400, 480]
MR_LB_SWEEP   = [48, 60, 72, 84, 96, 120, 168]
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

def load_year_data_base(year):
    """Load fixed signals (I460/I415/F168) and market data."""
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
    for i in range(TS_PCT_WIN, n):
        ts_spread_pct[i] = float(np.mean(ts_raw[i-TS_PCT_WIN:i] <= ts_raw[i]))
    breadth = np.full(n, 0.5)
    for i in range(BRD_LB, n):
        pos = sum(1 for j in range(len(SYMBOLS))
                  if close_mat[i-BRD_LB, j] > 0 and close_mat[i, j] > close_mat[i-BRD_LB, j])
        breadth[i] = pos / len(SYMBOLS)
    brd_pct = np.full(n, 0.5)
    for i in range(PCT_WINDOW, n):
        brd_pct[i] = float(np.mean(breadth[i-PCT_WINDOW:i] <= breadth[i]))
    # Fixed signals
    fixed_rets = {}
    for sk, sname, params in [
        ("i460", "idio_momentum_alpha",    I460_PARAMS),
        ("i415", "idio_momentum_alpha",    I415_PARAMS),
        ("f168", "funding_momentum_alpha", F168_PARAMS),
    ]:
        result = BacktestEngine(BacktestConfig(costs=COST_MODEL)).run(
            dataset, make_strategy({"name": sname, "params": params}))
        fixed_rets[sk] = np.array(result.returns)
    return btc_vol, fund_std_pct, ts_spread_pct, brd_pct, fixed_rets, n, dataset

def compute_v1_rets(dataset, v1_params):
    result = BacktestEngine(BacktestConfig(costs=COST_MODEL)).run(
        dataset, make_strategy({"name": "nexus_alpha_v1", "params": v1_params}))
    return np.array(result.returns)

def ensemble_rets(v1_rets, base_data):
    btc_vol, fund_std_pct, ts_spread_pct, brd_pct, fixed_rets, n, _ = base_data
    RNAMES = ["LOW", "MID", "HIGH"]
    sk_all = ["v1", "i460", "i415", "f168"]
    all_rets = {"v1": v1_rets, **fixed_rets}
    min_len = min(len(all_rets[k]) for k in sk_all)
    ens = np.zeros(min_len)
    bv = btc_vol[:min_len]; bpct = brd_pct[:min_len]
    fsp = fund_std_pct[:min_len]; tsp = ts_spread_pct[:min_len]
    for i in range(min_len):
        bp = bpct[i]
        ridx = 0 if bp < P_LOW else (2 if bp > P_HIGH else 1)
        rname = RNAMES[ridx]
        w = REGIME_WEIGHTS[rname]
        ret_i = sum(w[sk] * all_rets[sk][i] for sk in sk_all)
        if not np.isnan(bv[i]) and bv[i] > VOL_THR[rname]:
            ret_i *= VOL_SCALE[rname]
        if fsp[i] > DISP_THR[rname]: ret_i *= DISP_SCALE[rname]
        if tsp[i] > FTS_RT[rname]: ret_i *= FTS_RS[rname]
        elif tsp[i] < FTS_BT[rname]: ret_i *= FTS_BS[rname]
        ens[i] = ret_i
    return ens

def sharpe(r):
    r = r[~np.isnan(r)]
    if len(r) < 2: return 0.0
    return float(np.mean(r) / np.std(r, ddof=1) * np.sqrt(8760))

def obj(yearly):
    vals = list(yearly.values())
    return float(np.mean(vals) - 0.5 * np.std(vals, ddof=1))

def evaluate_v1(v1_params, all_base_data):
    all_v1_rets = {}
    for yr, bd in all_base_data.items():
        all_v1_rets[yr] = compute_v1_rets(bd[6], v1_params)
    yearly = {yr: sharpe(ensemble_rets(all_v1_rets[yr], all_base_data[yr]))
              for yr in all_base_data}
    return obj(yearly), yearly

def main():
    t0 = time.time()
    print("=" * 65)
    print("Phase 234 — V1 Internal Weights + Lookback Sweep")
    print(f"Baseline: v2.41.0  OBJ={BASELINE_OBJ}")
    print(f"Current: w_carry={V1_BASE['w_carry']} w_mom={V1_BASE['w_mom']} w_mr={V1_BASE['w_mean_reversion']}")
    print(f"mom_lb={V1_BASE['momentum_lookback_bars']} mr_lb={V1_BASE['mean_reversion_lookback_bars']}")
    print("=" * 65)

    print("\n[1] Loading per-year data & pre-computing fixed signals (I460/I415/F168)...")
    all_base_data = {}
    for yr in sorted(YEAR_RANGES):
        print(f"  {yr}: ", end="", flush=True)
        all_base_data[yr] = load_year_data_base(yr)
        print("fixed done.", flush=True)

    # Baseline V1
    base_v1_rets = {yr: compute_v1_rets(all_base_data[yr][6], V1_BASE) for yr in all_base_data}
    base_yearly = {yr: sharpe(ensemble_rets(base_v1_rets[yr], all_base_data[yr])) for yr in all_base_data}
    base_obj_val = obj(base_yearly)
    print(f"\n  Baseline OBJ = {base_obj_val:.4f}  (expected ≈ {BASELINE_OBJ})")
    _partial.update({"baseline_obj": float(base_obj_val)})

    print("\n[2a] Part 1: w_carry × w_mom sweep...")
    best_wc = V1_BASE["w_carry"]; best_wm = V1_BASE["w_mom"]
    best_obj_val = base_obj_val
    for wc in W_CARRY_SWEEP:
        for wm in W_MOM_SWEEP:
            wmr = round(1.0 - wc - wm, 4)
            if wmr < 0 or wmr > 1: continue
            vp = dict(V1_BASE); vp["w_carry"] = wc; vp["w_mom"] = wm; vp["w_mean_reversion"] = wmr
            o, _ = evaluate_v1(vp, all_base_data)
            if o > best_obj_val:
                best_obj_val = o; best_wc = wc; best_wm = wm
    best_wmr = round(1.0 - best_wc - best_wm, 4)
    print(f"  → Best weights: w_carry={best_wc} w_mom={best_wm} w_mr={best_wmr}  OBJ={best_obj_val:.4f}  Δ={best_obj_val-base_obj_val:+.4f}")
    _partial.update({"best_v1_weights": {"w_carry": best_wc, "w_mom": best_wm, "w_mean_reversion": best_wmr}})

    print("\n[2b] Part 2: momentum_lb × mean_reversion_lb sweep...")
    best_mom_lb = V1_BASE["momentum_lookback_bars"]
    best_mr_lb  = V1_BASE["mean_reversion_lookback_bars"]
    for mom_lb in MOM_LB_SWEEP:
        for mr_lb in MR_LB_SWEEP:
            if mr_lb >= mom_lb: continue  # mr_lb should be < mom_lb
            vp = dict(V1_BASE)
            vp["w_carry"] = best_wc; vp["w_mom"] = best_wm; vp["w_mean_reversion"] = best_wmr
            vp["momentum_lookback_bars"] = mom_lb; vp["mean_reversion_lookback_bars"] = mr_lb
            o, _ = evaluate_v1(vp, all_base_data)
            if o > best_obj_val:
                best_obj_val = o; best_mom_lb = mom_lb; best_mr_lb = mr_lb
    print(f"  → Best lookbacks: mom_lb={best_mom_lb} mr_lb={best_mr_lb}  OBJ={best_obj_val:.4f}  Δ={best_obj_val-base_obj_val:+.4f}")

    # Final best V1 params
    best_v1 = dict(V1_BASE)
    best_v1["w_carry"] = best_wc; best_v1["w_mom"] = best_wm; best_v1["w_mean_reversion"] = best_wmr
    best_v1["momentum_lookback_bars"] = best_mom_lb; best_v1["mean_reversion_lookback_bars"] = best_mr_lb

    print(f"\n[3] LOYO validation with best V1 params...")
    joint_obj_val, joint_yearly = evaluate_v1(best_v1, all_base_data)
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
    print(f"\n{'✅ VALIDATED' if validated else '❌ NO IMPROVEMENT'}")

    result = {
        "phase": 234, "baseline_obj": float(base_obj_val), "best_obj": float(joint_obj_val),
        "delta": float(delta), "loyo_wins": loyo_wins, "loyo_total": len(joint_yearly),
        "validated": validated, "best_v1_params": best_v1,
    }
    _partial.update(result)
    out = Path("artifacts/phase234"); out.mkdir(parents=True, exist_ok=True)
    (out / "phase234_report.json").write_text(json.dumps(result, indent=2))

    if validated:
        cfg_path = ROOT / "configs" / "production_p91b_champion.json"
        cfg = json.load(open(cfg_path))
        # Find V1 signal in ensemble signals
        for sk, sv in cfg["ensemble"]["signals"].items():
            if sv.get("name") == "nexus_alpha_v1":
                sv["params"]["w_carry"] = best_v1["w_carry"]
                sv["params"]["w_mom"] = best_v1["w_mom"]
                sv["params"]["w_mean_reversion"] = best_v1["w_mean_reversion"]
                sv["params"]["momentum_lookback_bars"] = best_v1["momentum_lookback_bars"]
                sv["params"]["mean_reversion_lookback_bars"] = best_v1["mean_reversion_lookback_bars"]
                break
        cfg["_version"] = "2.42.0"
        cfg["monitoring"]["expected_performance"]["annual_sharpe_backtest"] = round(joint_obj_val, 4)
        with open(cfg_path, "w") as f:
            json.dump(cfg, f, indent=2, ensure_ascii=False)
        print(f"\n  Config → v2.42.0  OBJ={round(joint_obj_val,4)}")

    print(f"\nRuntime: {int(time.time()-t0)}s")

main()
