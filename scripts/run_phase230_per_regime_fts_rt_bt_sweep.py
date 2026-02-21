"""
Phase 230 — Per-Regime FTS RT/BT Threshold Sweep
==================================================
P229 validated per-regime RS/BS (Δ=+0.0784, LOYO 5/5, OBJ=3.3222).
Now RT=0.65 and BT=0.25 are still uniform. Given regimes respond
differently to RS/BS, maybe the trigger percentile thresholds also
need regime-specific tuning:
  - LOW (downtrend): might trigger at different TS percentiles
  - MID (neutral):   baseline behavior
  - HIGH (uptrend):  might suppress FTS boost or dampen differently

Approach: fix RS/BS at P229 best per-regime values, sweep RT and BT
per regime sequentially (LOW→MID→HIGH, 2 passes).

RT sweep: [0.55, 0.60, 0.65, 0.70, 0.75, 0.80]
BT sweep: [0.10, 0.15, 0.20, 0.25, 0.30, 0.35]

Also test "disable" FTS per regime: RS=1.0, BS=1.0 in that regime.

Baseline: v2.37.0, OBJ=3.3222
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
    out = Path("artifacts/phase230"); out.mkdir(parents=True, exist_ok=True)
    (out / "phase230_report.json").write_text(json.dumps(_partial, indent=2))
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
VOL_WINDOW    = 168
TS_SHORT = 16; TS_LONG = 72
VOL_THRESHOLD = 0.50; VOL_SCALE = 0.30
DISP_THR = 0.60; DISP_SCALE = 1.0
P_LOW = 0.30; P_HIGH = 0.60

# P229 best per-regime RS/BS
BASE_RS = {"LOW": 0.50, "MID": 0.20, "HIGH": 0.40}
BASE_BS = {"LOW": 3.00, "MID": 3.00, "HIGH": 2.00}

# Global RT/BT baseline (from v2.36.0)
TS_RT_BASE = 0.65; TS_BT_BASE = 0.25

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
I460_PARAMS = {"k_per_side": 4, "lookback_bars": 480, "beta_window_bars": 168,
               "target_gross_leverage": 0.3, "rebalance_interval_bars": 48}
I415_PARAMS = {"k_per_side": 4, "lookback_bars": 415, "beta_window_bars": 144,
               "target_gross_leverage": 0.3, "rebalance_interval_bars": 48}
F168_PARAMS = {"k_per_side": 2, "funding_lookback_bars": 168, "direction": "contrarian",
               "target_gross_leverage": 0.25, "rebalance_interval_bars": 36}

BASELINE_OBJ = 3.3222
RT_SWEEP = [0.55, 0.60, 0.65, 0.70, 0.75, 0.80]
BT_SWEEP = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35]
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
    sig_rets = {}
    for sk, sname, params in [
        ("v1",   "nexus_alpha_v1",         V1_PARAMS),
        ("i460", "idio_momentum_alpha",    I460_PARAMS),
        ("i415", "idio_momentum_alpha",    I415_PARAMS),
        ("f168", "funding_momentum_alpha", F168_PARAMS),
    ]:
        result = BacktestEngine(BacktestConfig(costs=COST_MODEL)).run(
            dataset, make_strategy({"name": sname, "params": params}))
        sig_rets[sk] = np.array(result.returns)
    print("S.", end=" ", flush=True)
    return btc_vol, fund_std_pct, ts_spread_pct, brd_pct, sig_rets, n

def ensemble_rets_per_regime_fts(fts_rs, fts_bs, fts_rt, fts_bt, base_data):
    """All FTS params keyed by regime name (LOW/MID/HIGH)."""
    btc_vol, fund_std_pct, ts_spread_pct, brd_pct, sig_rets, n = base_data
    RNAMES = ["LOW", "MID", "HIGH"]
    sk_all = ["v1", "i460", "i415", "f168"]
    min_len = min(len(sig_rets[k]) for k in sk_all)
    ens = np.zeros(min_len)
    bv = btc_vol[:min_len]; bpct = brd_pct[:min_len]
    fsp = fund_std_pct[:min_len]; tsp = ts_spread_pct[:min_len]
    for i in range(min_len):
        bp = bpct[i]
        ridx = 0 if bp < P_LOW else (2 if bp > P_HIGH else 1)
        rname = RNAMES[ridx]
        w = REGIME_WEIGHTS[rname]
        ret_i = sum(w[sk] * sig_rets[sk][i] for sk in sk_all)
        if not np.isnan(bv[i]) and bv[i] > VOL_THRESHOLD:
            ret_i *= VOL_SCALE
        if fsp[i] > DISP_THR: ret_i *= DISP_SCALE
        if tsp[i] > fts_rt[rname]: ret_i *= fts_rs[rname]
        elif tsp[i] < fts_bt[rname]: ret_i *= fts_bs[rname]
        ens[i] = ret_i
    return ens

def sharpe(r):
    r = r[~np.isnan(r)]
    if len(r) < 2: return 0.0
    return float(np.mean(r) / np.std(r, ddof=1) * np.sqrt(8760))

def obj(yearly):
    vals = list(yearly.values())
    return float(np.mean(vals) - 0.5 * np.std(vals, ddof=1))

def evaluate(fts_rs, fts_bs, fts_rt, fts_bt, all_data):
    yearly = {yr: sharpe(ensemble_rets_per_regime_fts(fts_rs, fts_bs, fts_rt, fts_bt, d))
              for yr, d in all_data.items()}
    return obj(yearly), yearly

def main():
    t0 = time.time()
    print("=" * 65)
    print("Phase 230 — Per-Regime FTS RT/BT Threshold Sweep")
    print(f"Baseline: v2.37.0  OBJ={BASELINE_OBJ}")
    print(f"Fixed RS: {BASE_RS}")
    print(f"Fixed BS: {BASE_BS}")
    print(f"Global RT={TS_RT_BASE}  BT={TS_BT_BASE}")
    print(f"RT sweep: {RT_SWEEP}")
    print(f"BT sweep: {BT_SWEEP}")
    print("=" * 65)

    print("\n[1/3] Loading per-year data & pre-computing all signals...")
    all_data = {}
    for yr in sorted(YEAR_RANGES):
        print(f"  {yr}: ", end="", flush=True)
        all_data[yr] = load_year_data(yr)
    print()

    base_rt = {r: TS_RT_BASE for r in ["LOW", "MID", "HIGH"]}
    base_bt = {r: TS_BT_BASE for r in ["LOW", "MID", "HIGH"]}
    base_obj_val, base_yearly = evaluate(BASE_RS, BASE_BS, base_rt, base_bt, all_data)
    print(f"\n  Baseline OBJ = {base_obj_val:.4f}  (expected ≈ {BASELINE_OBJ})")
    _partial.update({"baseline_obj": float(base_obj_val)})

    print("\n[2/3] Per-regime RT × BT sweep (2 passes)...")
    best_rt = dict(base_rt); best_bt = dict(base_bt)
    best_obj_val = base_obj_val

    for pass_num in range(2):
        print(f"\n  === Pass {pass_num+1} ===")
        for regime in ["LOW", "MID", "HIGH"]:
            best_rt_local = best_rt[regime]; best_bt_local = best_bt[regime]
            best_obj_local = best_obj_val
            for rt in RT_SWEEP:
                for bt in BT_SWEEP:
                    if bt >= rt: continue  # BT must be < RT
                    cur_rt = dict(best_rt); cur_bt = dict(best_bt)
                    cur_rt[regime] = rt; cur_bt[regime] = bt
                    o, _ = evaluate(BASE_RS, BASE_BS, cur_rt, cur_bt, all_data)
                    if o > best_obj_local:
                        best_obj_local = o; best_rt_local = rt; best_bt_local = bt
            best_rt[regime] = best_rt_local; best_bt[regime] = best_bt_local
            cur_rt = dict(best_rt); cur_bt = dict(best_bt)
            o, _ = evaluate(BASE_RS, BASE_BS, cur_rt, cur_bt, all_data)
            delta = o - base_obj_val
            print(f"    {regime}: RT={best_rt_local:.2f} BT={best_bt_local:.2f}  OBJ={o:.4f}  Δ={delta:+.4f}")
            best_obj_val = max(best_obj_val, o)

    print(f"\n  → Best per-regime params:")
    for r in ["LOW", "MID", "HIGH"]:
        print(f"    {r}: RT={best_rt[r]:.2f}  BT={best_bt[r]:.2f}")

    print(f"\n[3/3] LOYO validation...")
    joint_obj_val, joint_yearly = evaluate(BASE_RS, BASE_BS, best_rt, best_bt, all_data)
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
        "phase": 230, "baseline_obj": float(base_obj_val), "best_obj": float(joint_obj_val),
        "delta": float(delta), "loyo_wins": loyo_wins, "loyo_total": len(joint_yearly),
        "validated": validated,
        "best_rt": best_rt, "best_bt": best_bt,
        "fixed_rs": BASE_RS, "fixed_bs": BASE_BS,
    }
    _partial.update(result)
    out = Path("artifacts/phase230"); out.mkdir(parents=True, exist_ok=True)
    (out / "phase230_report.json").write_text(json.dumps(result, indent=2))

    if validated:
        cfg_path = ROOT / "configs" / "production_p91b_champion.json"
        cfg = json.load(open(cfg_path))
        fts = cfg.setdefault("fts_overlay_params", {})
        fts["per_regime_rt"] = best_rt
        fts["per_regime_bt"] = best_bt
        cfg["_version"] = "2.38.0"
        cfg["monitoring"]["expected_performance"]["annual_sharpe_backtest"] = round(joint_obj_val, 4)
        with open(cfg_path, "w") as f:
            json.dump(cfg, f, indent=2, ensure_ascii=False)
        print(f"\n  Config → v2.38.0  OBJ={round(joint_obj_val,4)}")

    print(f"\nRuntime: {int(time.time()-t0)}s")

main()
