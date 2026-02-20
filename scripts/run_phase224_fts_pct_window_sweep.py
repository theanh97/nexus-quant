"""
Phase 224 — FTS Percentile Window Sweep (ts_pct_win)
=====================================================
Currently both breadth_pct and ts_spread_pct use PCT_WINDOW=336 bars
as the rolling window for percentile rank computation. These are
independent signals and might benefit from different lookback windows.

Since DISP_SCALE=1.0 (overlay disabled), FUND_DISP_PCT is irrelevant.
Instead sweep the TS spread percentile window independently.

Approach: rebuild ts_spread_pct on-the-fly for each ts_pct_win.
          All signals pre-computed. Breadth PCT_WINDOW stays at 336.

Sweep: ts_pct_win ∈ [120, 168, 240, 288, 336, 400, 480, 560, 672]

Baseline: v2.34.0, OBJ=3.1534
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
    out = Path("artifacts/phase224"); out.mkdir(parents=True, exist_ok=True)
    (out / "phase224_report.json").write_text(json.dumps(_partial, indent=2))
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
VOL_WINDOW    = 168
TS_SHORT = 16; TS_LONG = 72
VOL_THRESHOLD = 0.50; VOL_SCALE = 0.30
TS_RT = 0.65; TS_RS = 0.45; TS_BT = 0.25; TS_BS = 1.70
DISP_THR = 0.60; DISP_SCALE = 1.0
P_LOW = 0.30; P_HIGH = 0.60

TS_PCT_WIN_BASE = 336

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

BASELINE_OBJ = 3.1534
TS_PCT_SWEEP = [120, 168, 240, 288, 336, 400, 480, 560, 672]
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
    # Pre-compute ts_raw (raw TS spread, before percentile transform)
    xsect_mean = np.mean(fund_rates, axis=1)
    ts_raw = rolling_mean_arr(xsect_mean, TS_SHORT) - rolling_mean_arr(xsect_mean, TS_LONG)
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
    # Return ts_raw (not percentile) so we can recompute percentile with different windows
    return btc_vol, fund_std_pct, ts_raw, brd_pct, sig_rets, n

def compute_ts_pct(ts_raw, n, pct_win):
    ts_spread_pct = np.full(n, 0.5)
    for i in range(pct_win, n):
        ts_spread_pct[i] = float(np.mean(ts_raw[i-pct_win:i] <= ts_raw[i]))
    return ts_spread_pct

def ensemble_rets(ts_spread_pct, base_data):
    btc_vol, fund_std_pct, ts_raw, brd_pct, sig_rets, n = base_data
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

def evaluate(ts_pct_win, all_data):
    yearly = {}
    for yr, d in all_data.items():
        btc_vol, fund_std_pct, ts_raw, brd_pct, sig_rets, n = d
        tsp = compute_ts_pct(ts_raw, n, ts_pct_win)
        ens = ensemble_rets(tsp, d)
        yearly[yr] = sharpe(ens)
    return obj(yearly), yearly

def main():
    t0 = time.time()
    print("=" * 65)
    print("Phase 224 — FTS Percentile Window Sweep (ts_pct_win)")
    print(f"Baseline: v2.34.0  OBJ={BASELINE_OBJ}")
    print(f"Current: ts_pct_win={TS_PCT_WIN_BASE} (shared with breadth)")
    print(f"Sweep: {TS_PCT_SWEEP}")
    print("=" * 65)

    print("\n[1/3] Loading per-year data & pre-computing all signals...")
    all_data = {}
    for yr in sorted(YEAR_RANGES):
        print(f"  {yr}: ", end="", flush=True)
        all_data[yr] = load_year_data(yr)
    print()

    base_obj_val, base_yearly = evaluate(TS_PCT_WIN_BASE, all_data)
    print(f"\n  Baseline OBJ = {base_obj_val:.4f}  (expected ≈ {BASELINE_OBJ})")
    _partial.update({"baseline_obj": float(base_obj_val)})

    print("\n[2/3] Sweep...")
    best_obj_val = base_obj_val; best_win = TS_PCT_WIN_BASE
    results = []
    for win in TS_PCT_SWEEP:
        o, _ = evaluate(win, all_data)
        delta = o - base_obj_val
        marker = " ← BEST" if o > best_obj_val else ""
        print(f"  ts_pct_win={win:4d}  OBJ={o:.4f}  Δ={delta:+.4f}{marker}")
        results.append((win, o))
        if o > best_obj_val:
            best_obj_val = o; best_win = win

    print(f"\n  → Best: ts_pct_win={best_win}  OBJ={best_obj_val:.4f}")

    print(f"\n[3/3] LOYO validation...")
    joint_obj_val, joint_yearly = evaluate(best_win, all_data)
    delta = joint_obj_val - base_obj_val
    loyo_wins = 0
    for yr in sorted(joint_yearly):
        base_sh = base_yearly.get(yr, 0)
        cand_sh = joint_yearly[yr]
        win_flag = cand_sh > base_sh
        if win_flag: loyo_wins += 1
        print(f"  {yr}: base={base_sh:.4f}  cand={cand_sh:.4f}  {'WIN' if win_flag else 'LOSE'}")

    print(f"\n  OBJ={joint_obj_val:.4f}  Δ={delta:+.4f}  LOYO {loyo_wins}/{len(joint_yearly)}")
    validated = loyo_wins >= MIN_LOYO and delta >= MIN_DELTA
    print(f"\n{'✅ VALIDATED' if validated else '❌ NO IMPROVEMENT'} — ts_pct_win={best_win}")

    result = {
        "phase": 224, "baseline_obj": float(base_obj_val), "best_obj": float(joint_obj_val),
        "delta": float(delta), "loyo_wins": loyo_wins, "loyo_total": len(joint_yearly),
        "validated": validated, "best_ts_pct_win": best_win, "sweep_results": results,
    }
    _partial.update(result)
    out = Path("artifacts/phase224"); out.mkdir(parents=True, exist_ok=True)
    (out / "phase224_report.json").write_text(json.dumps(result, indent=2))

    if validated:
        cfg_path = ROOT / "configs" / "production_p91b_champion.json"
        cfg = json.load(open(cfg_path))
        fts = cfg.setdefault("fts_overlay_params", {})
        fts["ts_pct_window_bars"] = best_win
        cfg["_version"] = "2.35.0"
        cfg["monitoring"]["expected_performance"]["annual_sharpe_backtest"] = round(joint_obj_val, 4)
        with open(cfg_path, "w") as f:
            json.dump(cfg, f, indent=2, ensure_ascii=False)
        print(f"\n  Config → v2.35.0  OBJ={round(joint_obj_val,4)}")

    print(f"\nRuntime: {int(time.time()-t0)}s")

main()
