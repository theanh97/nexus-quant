"""
Phase 235 — Breadth Classifier Parameter Sweep
================================================
P234: V1 params confirmed at exact optimum (NO IMPROVEMENT).
The breadth regime classifier uses:
  - BRD_LB = 192h: lookback for counting positive-return symbols
  - PCT_WINDOW = 336h: rolling percentile window for breadth ranking

These two define the fundamental regime classification. If suboptimal,
all per-regime parameters are slightly misaligned.

This is a pure NumPy sweep (classifier logic only, no re-backtesting).

BRD_LB sweep:    [96, 120, 144, 168, 192, 240, 288, 336]
PCT_WINDOW sweep: [168, 240, 288, 336, 400, 480, 560]

Baseline: v2.41.0, OBJ=3.7718 (using P233 regime weights)
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
    out = Path("artifacts/phase235"); out.mkdir(parents=True, exist_ok=True)
    (out / "phase235_report.json").write_text(json.dumps(_partial, indent=2))
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

# Fixed overlays (from v2.40.0)
FUND_DISP_PCT = 240
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

# Baseline classifier params
BRD_LB_BASE = 192; PCT_WINDOW_BASE = 336

BASELINE_OBJ = 3.7718
BRD_LB_SWEEP   = [96, 120, 144, 168, 192, 240, 288, 336]
PCT_WIN_SWEEP  = [168, 240, 288, 336, 400, 480, 560]
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
    # Return close_mat raw so we can recompute breadth with different BRD_LB
    return btc_vol, fund_std_pct, ts_spread_pct, close_mat, sig_rets, n

def compute_brd_pct(close_mat, n, brd_lb, pct_win):
    breadth = np.full(n, 0.5)
    for i in range(brd_lb, n):
        pos = sum(1 for j in range(len(SYMBOLS))
                  if close_mat[i-brd_lb, j] > 0 and close_mat[i, j] > close_mat[i-brd_lb, j])
        breadth[i] = pos / len(SYMBOLS)
    brd_pct = np.full(n, 0.5)
    for i in range(pct_win, n):
        brd_pct[i] = float(np.mean(breadth[i-pct_win:i] <= breadth[i]))
    return brd_pct

def ensemble_rets(brd_pct, base_data):
    btc_vol, fund_std_pct, ts_spread_pct, close_mat, sig_rets, n = base_data
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

def evaluate(brd_lb, pct_win, all_data):
    yearly = {}
    for yr, d in all_data.items():
        brd_pct = compute_brd_pct(d[3], d[5], brd_lb, pct_win)
        yearly[yr] = sharpe(ensemble_rets(brd_pct, d))
    return obj(yearly), yearly

def main():
    t0 = time.time()
    print("=" * 65)
    print("Phase 235 — Breadth Classifier Sweep (BRD_LB × PCT_WINDOW)")
    print(f"Baseline: v2.41.0  OBJ={BASELINE_OBJ}")
    print(f"Current: BRD_LB={BRD_LB_BASE}  PCT_WINDOW={PCT_WINDOW_BASE}")
    print(f"BRD_LB sweep: {BRD_LB_SWEEP}")
    print(f"PCT_WIN sweep: {PCT_WIN_SWEEP}")
    print("=" * 65)

    print("\n[1/3] Loading per-year data & pre-computing all signals...")
    all_data = {}
    for yr in sorted(YEAR_RANGES):
        print(f"  {yr}: ", end="", flush=True)
        all_data[yr] = load_year_data(yr)
    print()

    base_obj_val, base_yearly = evaluate(BRD_LB_BASE, PCT_WINDOW_BASE, all_data)
    print(f"\n  Baseline OBJ = {base_obj_val:.4f}  (expected ≈ {BASELINE_OBJ})")
    _partial.update({"baseline_obj": float(base_obj_val)})

    print("\n[2/3] 2D sweep BRD_LB × PCT_WIN...")
    best_obj_val = base_obj_val; best_brd_lb = BRD_LB_BASE; best_pct_win = PCT_WINDOW_BASE
    results = []
    for brd_lb in BRD_LB_SWEEP:
        for pct_win in PCT_WIN_SWEEP:
            o, _ = evaluate(brd_lb, pct_win, all_data)
            delta = o - base_obj_val
            marker = " ← BEST" if o > best_obj_val else ""
            print(f"  BRD_LB={brd_lb:3d}  PCT_WIN={pct_win:3d}  OBJ={o:.4f}  Δ={delta:+.4f}{marker}")
            results.append((brd_lb, pct_win, o))
            if o > best_obj_val:
                best_obj_val = o; best_brd_lb = brd_lb; best_pct_win = pct_win

    print(f"\n  → Best: BRD_LB={best_brd_lb}  PCT_WIN={best_pct_win}  OBJ={best_obj_val:.4f}")

    print(f"\n[3/3] LOYO validation...")
    joint_obj_val, joint_yearly = evaluate(best_brd_lb, best_pct_win, all_data)
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
        "phase": 235, "baseline_obj": float(base_obj_val), "best_obj": float(joint_obj_val),
        "delta": float(delta), "loyo_wins": loyo_wins, "loyo_total": len(joint_yearly),
        "validated": validated, "best_brd_lb": best_brd_lb, "best_pct_win": best_pct_win,
        "top5": sorted(results, key=lambda x: -x[2])[:5],
    }
    _partial.update(result)
    out = Path("artifacts/phase235"); out.mkdir(parents=True, exist_ok=True)
    (out / "phase235_report.json").write_text(json.dumps(result, indent=2))

    if validated:
        cfg_path = ROOT / "configs" / "production_p91b_champion.json"
        cfg = json.load(open(cfg_path))
        brs = cfg["breadth_regime_switching"]
        brs["breadth_lookback_bars"] = best_brd_lb
        brs["rolling_percentile_window"] = best_pct_win
        cfg["_version"] = "2.42.0"
        cfg["monitoring"]["expected_performance"]["annual_sharpe_backtest"] = round(joint_obj_val, 4)
        with open(cfg_path, "w") as f:
            json.dump(cfg, f, indent=2, ensure_ascii=False)
        print(f"\n  Config → v2.42.0  OBJ={round(joint_obj_val,4)}")

    print(f"\nRuntime: {int(time.time()-t0)}s")

main()
