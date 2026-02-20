"""
Phase 203 — V1 Internal Weight Re-Sweep
=========================================
V1 is 46% LOW, 20% MID. Last tuned at P185 under different regime weights.
Re-sweep w_carry × w_mom (w_mr = 1 - carry - mom) with correct overlay constants.

Baseline: v2.31.0, OBJ=3.0690
Current V1: w_carry=0.35, w_mom=0.40, w_mr=0.25
Sweep: carry ∈ [0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45]
       mom   ∈ [0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55]
       constraint: w_mr = 1 - carry - mom ≥ 0.05
"""

import os, sys, json, time
import signal as _signal
from pathlib import Path
from itertools import product

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
    (out / "phase203_report.json").write_text(json.dumps(_partial, indent=2))
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

# Overlay constants (v2.31.0 correct)
VOL_WINDOW    = 168; BRD_LB = 192; PCT_WINDOW = 336
TS_SHORT, TS_LONG = 16, 72; FUND_DISP_PCT = 240
VOL_THRESHOLD = 0.50; VOL_SCALE = 0.30; VOL_F168_BOOST = 0.00
TS_RT = 0.65; TS_RS = 0.45; TS_BT = 0.25; TS_BS = 1.70
DISP_THR = 0.60; DISP_SCALE = 1.0
P_LOW = 0.30; P_HIGH = 0.60

# Current regime weights (v2.31.0)
REGIME_WEIGHTS = {
    "LOW":  {"v1": 0.46,   "i460": 0.0716, "i415": 0.1184, "f168": 0.35},
    "MID":  {"v1": 0.20,   "i460": 0.0074, "i415": 0.0126, "f168": 0.78},
    "HIGH": {"v1": 0.06,   "i460": 0.2821, "i415": 0.5079, "f168": 0.15},
}

# Fixed strategy params (non-V1)
I460_PARAMS = {"k_per_side": 4, "lookback_bars": 460, "beta_window_bars": 168,
               "target_gross_leverage": 0.3, "rebalance_interval_bars": 48}
I415_PARAMS = {"k_per_side": 4, "lookback_bars": 415, "beta_window_bars": 144,
               "target_gross_leverage": 0.3, "rebalance_interval_bars": 48}
F168_PARAMS = {"k_per_side": 2, "funding_lookback_bars": 168, "direction": "contrarian",
               "target_gross_leverage": 0.25, "rebalance_interval_bars": 24}

# V1 base params (non-weight components fixed)
V1_BASE = {
    "k_per_side": 2, "momentum_lookback_bars": 336, "mean_reversion_lookback_bars": 84,
    "vol_lookback_bars": 192, "target_gross_leverage": 0.35, "rebalance_interval_bars": 60,
}

BASELINE_OBJ = 3.0690
MIN_DELTA = 0.005; MIN_LOYO = 3

# Sweep grid
W_CARRY_SWEEP = [0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45]
W_MOM_SWEEP   = [0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55]
W_MR_MIN      = 0.05

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

    # Pre-compute fixed signals (i460, i415, f168)
    fixed_rets = {}
    for sk, sname, params in [
        ("i460", "idio_momentum_alpha", I460_PARAMS),
        ("i415", "idio_momentum_alpha", I415_PARAMS),
        ("f168", "funding_momentum_alpha", F168_PARAMS),
    ]:
        result = BacktestEngine(BacktestConfig(costs=COST_MODEL)).run(
            dataset, make_strategy({"name": sname, "params": params}))
        fixed_rets[sk] = np.array(result.returns)
    print("F.", end=" ", flush=True)
    return dataset, btc_vol, fund_std_pct, ts_spread_pct, brd_pct, fixed_rets, n

def run_v1(dataset, w_carry, w_mom):
    w_mr = 1.0 - w_carry - w_mom
    params = dict(V1_BASE, w_carry=w_carry, w_mom=w_mom, w_mean_reversion=w_mr)
    result = BacktestEngine(BacktestConfig(costs=COST_MODEL)).run(
        dataset, make_strategy({"name": "nexus_alpha_v1", "params": params}))
    return np.array(result.returns)

def ensemble_rets(v1_rets, fixed_rets, btc_vol, fund_std_pct, ts_spread_pct, brd_pct, n):
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
        w = REGIME_WEIGHTS[RNAMES[ridx]]
        ret_i = sum(w[sk] * all_rets[sk][i] for sk in sk_all)
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
    print("Phase 203 — V1 Internal Weight Re-Sweep")
    print(f"Baseline: v2.31.0  OBJ={BASELINE_OBJ}")
    print(f"Current: w_carry=0.35 w_mom=0.40 w_mr=0.25")
    print(f"Grid: carry={W_CARRY_SWEEP}")
    print(f"      mom={W_MOM_SWEEP}")
    print("=" * 65)

    print("\n[1/3] Loading per-year data & fixed signals...")
    all_data = {}
    for yr in sorted(YEAR_RANGES):
        print(f"  {yr}: ", end="", flush=True)
        all_data[yr] = load_year_data(yr)
    print()

    # Baseline (current w_carry=0.35, w_mom=0.40)
    def evaluate_combo(w_carry, w_mom):
        if w_carry + w_mom > 1.0 - W_MR_MIN:
            return None, None
        yearly = {}
        for yr, (dataset, btc_vol, fsp, tsp, brd_pct, fixed_rets, n) in all_data.items():
            v1r = run_v1(dataset, w_carry, w_mom)
            ens = ensemble_rets(v1r, fixed_rets, btc_vol, fsp, tsp, brd_pct, n)
            yearly[yr] = sharpe(ens)
        return obj(yearly), yearly

    base_obj, base_yearly = evaluate_combo(0.35, 0.40)
    print(f"\n  Baseline OBJ = {base_obj:.4f}  (expected ≈ {BASELINE_OBJ})")
    _partial.update({"baseline_obj": float(base_obj)})

    print("\n[2/3] Sweep...")
    best_obj = base_obj; best_carry = 0.35; best_mom = 0.40
    results = []
    for w_carry in W_CARRY_SWEEP:
        for w_mom in W_MOM_SWEEP:
            w_mr = 1.0 - w_carry - w_mom
            if w_mr < W_MR_MIN: continue
            o, _ = evaluate_combo(w_carry, w_mom)
            if o is None: continue
            delta = o - base_obj
            marker = " ← BEST" if o > best_obj else ""
            print(f"  carry={w_carry:.2f} mom={w_mom:.2f} mr={w_mr:.2f}  OBJ={o:.4f}  Δ={delta:+.4f}{marker}")
            results.append((o, w_carry, w_mom, w_mr))
            if o > best_obj:
                best_obj = o; best_carry = w_carry; best_mom = w_mom

    best_mr = 1.0 - best_carry - best_mom
    print(f"\n  → Best: w_carry={best_carry:.2f} w_mom={best_mom:.2f} w_mr={best_mr:.2f}  OBJ={best_obj:.4f}")

    print("\n[3/3] LOYO validation...")
    joint_obj, joint_yearly = evaluate_combo(best_carry, best_mom)
    delta = joint_obj - base_obj
    loyo_wins = 0
    for yr in sorted(joint_yearly):
        base_sh = base_yearly.get(yr, 0)
        cand_sh = joint_yearly[yr]
        win = cand_sh > base_sh
        if win: loyo_wins += 1
        print(f"  {yr}: base={base_sh:.4f}  cand={cand_sh:.4f}  {'WIN' if win else 'LOSE'}")

    print(f"\n  OBJ={joint_obj:.4f}  Δ={delta:+.4f}  LOYO {loyo_wins}/{len(joint_yearly)}")
    validated = loyo_wins >= MIN_LOYO and delta >= MIN_DELTA
    print(f"\n{'✅ VALIDATED' if validated else '❌ NO IMPROVEMENT'} — carry={best_carry} mom={best_mom} mr={best_mr:.2f}")

    result = {
        "phase": 203, "baseline_obj": float(base_obj), "best_obj": float(joint_obj),
        "delta": float(delta), "loyo_wins": loyo_wins, "loyo_total": len(joint_yearly),
        "validated": validated,
        "best_w_carry": best_carry, "best_w_mom": best_mom, "best_w_mr": float(best_mr),
        "top5": sorted(results, key=lambda x: -x[0])[:5],
    }
    _partial.update(result)
    out = Path("artifacts/phase203"); out.mkdir(parents=True, exist_ok=True)
    (out / "phase203_report.json").write_text(json.dumps(result, indent=2))

    if validated:
        cfg_path = ROOT / "configs" / "production_p91b_champion.json"
        cfg = json.load(open(cfg_path))
        sigs = cfg["ensemble"]["signals"]
        sigs["v1"]["params"]["w_carry"]           = best_carry
        sigs["v1"]["params"]["w_mom"]             = best_mom
        sigs["v1"]["params"]["w_mean_reversion"]  = float(best_mr)
        cfg["_version"] = "2.32.0"
        cfg["monitoring"]["expected_performance"]["annual_sharpe_backtest"] = round(joint_obj, 4)
        with open(cfg_path, "w") as f:
            json.dump(cfg, f, indent=2, ensure_ascii=False)
        print(f"\n  Config → v2.32.0  OBJ={round(joint_obj,4)}")

    print(f"\nRuntime: {int(time.time()-t0)}s")

main()
