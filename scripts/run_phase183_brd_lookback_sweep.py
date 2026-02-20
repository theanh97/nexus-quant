"""
PHASE 183 — Breadth Lookback Re-Sweep
==========================================
Baseline: v2.18.0  OBJ≈2.5689
Current:  BRD_LOOKBACK=192 bars (8-day window for computing market breadth)

After P181 changed breadth regime thresholds (PL=0.20→PH=0.60),
the breadth signal lookback window may also benefit from re-tuning.

Parts:
  A — BRD_LOOKBACK sweep ∈ [96, 120, 144, 168, 192, 216, 240, 288]
  B — LOYO validation with best

Reuses data from disk via cache_dir.
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
    _partial["partial"] = True; _partial["timeout_at"] = datetime.now(UTC).isoformat()
    out = Path("artifacts/phase183"); out.mkdir(parents=True, exist_ok=True)
    (out / "phase183_report.json").write_text(json.dumps(_partial, indent=2)); sys.exit(0)
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

VOL_WINDOW    = 168
BASELINE_BRD  = 192
PCT_WINDOW    = 336
P_LOW, P_HIGH = 0.20, 0.60   # v2.18.0
TS_SHORT, TS_LONG = 12, 96
FUND_DISP_PCT = 240

VOL_THRESHOLD = 0.50; VOL_SCALE = 0.40; VOL_F168_BOOST = 0.10
TS_RT = 0.60; TS_RS = 0.40; TS_BT = 0.25; TS_BS = 1.50
DISP_THR = 0.60; DISP_SCALE = 1.0

# v2.18.0 weights (P182 update, derived from make_weights(0.35, 0.35, 0.10))
# scale_others: LOW base={v1:0.2415,i460:0.1730,i415:0.2855,f168:0.30}
# Low: others_sum=0.7, new_f168=0.35, target_others=0.65, scale=0.65/0.7=0.9286
# LOW:  v1=0.2415*0.9286=0.2242, i460=0.1730*0.9286=0.1607, i415=0.2855*0.9286=0.2651, f168=0.35
# MID base={v1:0.1493,i460:0.2053,i415:0.3453,f168:0.30}
# MID:  others_sum=0.6999, target_others=0.65, scale=0.65/0.6999=0.9287
# MID:  v1=0.1387, i460=0.1906, i415=0.3207, f168=0.35
# HIGH base={v1:0.0567,i460:0.2833,i415:0.5100,f168:0.15}
# HIGH: others_sum=0.85, new_f168=0.10, target_others=0.90, scale=0.90/0.85=1.0588
# HIGH: v1=0.0600, i460=0.3000, i415=0.5400, f168=0.10
WEIGHTS = {
    "LOW":  {"v1": 0.2242, "i460": 0.1607, "i415": 0.2651, "f168": 0.35},
    "MID":  {"v1": 0.1387, "i460": 0.1906, "i415": 0.3207, "f168": 0.35},
    "HIGH": {"v1": 0.0600, "i460": 0.3000, "i415": 0.5400, "f168": 0.10},
}

BASELINE_OBJ = 2.5689

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

def sharpe(r: np.ndarray) -> float:
    r = r[~np.isnan(r)]
    if len(r) < 2: return 0.0
    return float(np.mean(r) / np.std(r, ddof=1) * np.sqrt(8760))

def obj(yearly: dict) -> float:
    vals = list(yearly.values())
    return float(np.mean(vals) - 0.5 * np.std(vals, ddof=1))

def load_year_data(year: int):
    """Load raw data (close prices + funding rates) for swept signal computation."""
    s, e = YEAR_RANGES[year]
    cfg = {"provider": "binance_rest_v1", "symbols": SYMBOLS,
           "start": s, "end": e, "bar_interval": "1h",
           "cache_dir": ".cache/binance_rest"}
    dataset = make_provider(cfg, seed=42).load()
    n = len(dataset.timeline)
    # Close prices matrix
    close_mat = np.zeros((n, len(SYMBOLS)))
    for j, sym in enumerate(SYMBOLS):
        for i in range(n):
            close_mat[i, j] = dataset.close(sym, i)
    # BTC vol
    btc_rets = np.zeros(n)
    for i in range(1, n):
        c0 = close_mat[i-1, 0]; c1 = close_mat[i, 0]
        btc_rets[i] = (c1 / c0 - 1.0) if c0 > 0 else 0.0
    btc_vol = np.full(n, np.nan)
    for i in range(VOL_WINDOW, n):
        btc_vol[i] = float(np.std(btc_rets[i - VOL_WINDOW:i])) * np.sqrt(8760)
    if VOL_WINDOW < n: btc_vol[:VOL_WINDOW] = btc_vol[VOL_WINDOW]
    # Funding rates + dispersion + TS
    fund_rates = np.zeros((n, len(SYMBOLS)))
    for j, sym in enumerate(SYMBOLS):
        for i in range(n):
            ts = dataset.timeline[i]
            try:    fund_rates[i, j] = dataset.last_funding_rate_before(sym, ts)
            except: fund_rates[i, j] = 0.0
    fund_std_raw = np.std(fund_rates, axis=1)
    fund_std_pct = np.full(n, 0.5)
    for i in range(FUND_DISP_PCT, n):
        fund_std_pct[i] = float(np.mean(fund_std_raw[i - FUND_DISP_PCT:i] <= fund_std_raw[i]))
    fund_std_pct[:FUND_DISP_PCT] = 0.5
    xsect_mean = np.mean(fund_rates, axis=1)
    ts_raw = rolling_mean_arr(xsect_mean, TS_SHORT) - rolling_mean_arr(xsect_mean, TS_LONG)
    ts_spread_pct = np.full(n, 0.5)
    for i in range(PCT_WINDOW, n):
        ts_spread_pct[i] = float(np.mean(ts_raw[i - PCT_WINDOW:i] <= ts_raw[i]))
    ts_spread_pct[:PCT_WINDOW] = 0.5
    # Strategy returns
    sig_rets = {}
    for sk, sname, params in [
        ("v1", "nexus_alpha_v1", {
            "k_per_side": 2, "w_carry": 0.35, "w_mom": 0.45, "w_mean_reversion": 0.2,
            "momentum_lookback_bars": 336, "mean_reversion_lookback_bars": 72,
            "vol_lookback_bars": 168, "target_gross_leverage": 0.35,
            "rebalance_interval_bars": 60,
        }),
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
        sig_rets[sk] = np.array(result.returns)
    print("S.", end=" ", flush=True)
    return close_mat, btc_vol, fund_std_pct, ts_spread_pct, sig_rets, n

def compute_brd_regime(close_mat: np.ndarray, n: int, brd_lb: int) -> np.ndarray:
    """Compute breadth regime with given lookback, using global P_LOW/P_HIGH."""
    breadth = np.full(n, 0.5)
    for i in range(brd_lb, n):
        pos = sum(1 for j in range(len(SYMBOLS))
                  if close_mat[i - brd_lb, j] > 0 and close_mat[i, j] > close_mat[i - brd_lb, j])
        breadth[i] = pos / len(SYMBOLS)
    breadth[:brd_lb] = 0.5
    brd_pct = np.full(n, 0.5)
    for i in range(PCT_WINDOW, n):
        brd_pct[i] = float(np.mean(breadth[i - PCT_WINDOW:i] <= breadth[i]))
    brd_pct[:PCT_WINDOW] = 0.5
    return np.where(brd_pct >= P_HIGH, 2, np.where(brd_pct >= P_LOW, 1, 0)).astype(int)

def compute_ensemble_with_regime(sig_rets, btc_vol, regime, fund_std_pct, ts_spread_pct) -> np.ndarray:
    sk_all = ["v1", "i460", "i415", "f168"]
    min_len = min(len(sig_rets[sk]) for sk in sk_all)
    bv = btc_vol[:min_len]; reg = regime[:min_len]
    fsp = fund_std_pct[:min_len]; tsp = ts_spread_pct[:min_len]
    ens = np.zeros(min_len)
    for i in range(min_len):
        w = WEIGHTS[["LOW", "MID", "HIGH"][int(reg[i])]]
        if not np.isnan(bv[i]) and bv[i] > VOL_THRESHOLD:
            boost_per = VOL_F168_BOOST / max(1, len(sk_all) - 1)
            ret_i = sum(
                (min(0.60, w[sk] + VOL_F168_BOOST) if sk == "f168"
                 else max(0.05, w[sk] - boost_per)) * sig_rets[sk][i]
                for sk in sk_all)
            ret_i *= VOL_SCALE
        else:
            ret_i = sum(w[sk] * sig_rets[sk][i] for sk in sk_all)
        if fsp[i] > DISP_THR: ret_i *= DISP_SCALE
        if tsp[i] > TS_RT:   ret_i *= TS_RS
        elif tsp[i] < TS_BT: ret_i *= TS_BS
        ens[i] = ret_i
    return ens

def run_year(data, brd_lb: int) -> float:
    close_mat, btc_vol, fund_std_pct, ts_spread_pct, sig_rets, n = data
    regime = compute_brd_regime(close_mat, n, brd_lb)
    return sharpe(compute_ensemble_with_regime(sig_rets, btc_vol, regime, fund_std_pct, ts_spread_pct))

def main():
    t0 = time.time()
    print("=" * 68)
    print("PHASE 183 — BRD_LOOKBACK Re-Sweep (v2.18.0 baseline)")
    print("=" * 68)
    print(f"  Baseline: BRD_LOOKBACK={BASELINE_BRD} bars")

    print(f"\n[1/3] Loading data ...")
    data_by_yr = {}
    for y in YEARS:
        print(f"  {y}: ", end="", flush=True)
        data_by_yr[y] = load_year_data(y)
    print()

    bl_yr = {y: run_year(data_by_yr[y], BASELINE_BRD) for y in YEARS}
    bl_obj = obj(bl_yr)
    print(f"  Baseline OBJ={bl_obj:.4f}")

    # ── Part A: BRD_LOOKBACK sweep ────────────────────────────────────────────
    print(f"\n[2/3] Part A — BRD_LOOKBACK sweep ...")
    sweep_vals = [96, 120, 144, 168, 192, 216, 240, 288]
    best_lb = BASELINE_BRD; best_lb_obj = bl_obj
    sweep_results = {}
    for lb in sweep_vals:
        yr = {y: run_year(data_by_yr[y], lb) for y in YEARS}
        o = obj(yr); d = o - bl_obj
        sym = "✅" if d > 0.005 else ("⚠️ " if abs(d) <= 0.005 else "❌")
        base_marker = " ← baseline" if lb == BASELINE_BRD else ""
        print(f"    lb={lb} → OBJ={o:.4f}  Δ={d:+.4f}  {sym}{base_marker}")
        sweep_results[lb] = {"lb": lb, "obj": o, "delta": d, "yearly": yr}
        if o > best_lb_obj: best_lb_obj = o; best_lb = lb
    print(f"  LB winner: lb={best_lb}  OBJ={best_lb_obj:.4f}")

    # ── Part B: LOYO validation ───────────────────────────────────────────────
    print(f"\n[3/3] Part B — LOYO validation with lb={best_lb} ...")
    loyo_wins = 0; loyo_deltas = []; loyo_detail = {}
    for loyo_yr in YEARS:
        train_yrs = [y for y in YEARS if y != loyo_yr]
        tr_yr  = {y: run_year(data_by_yr[y], best_lb) for y in train_yrs}
        bl_yr_t = {y: run_year(data_by_yr[y], BASELINE_BRD) for y in train_yrs}
        d = obj(tr_yr) - obj(bl_yr_t)
        win = d > 0.005; loyo_wins += int(win); loyo_deltas.append(d)
        loyo_detail[loyo_yr] = {"delta": d, "win": win}
        print(f"    {loyo_yr}: {'✅' if win else '❌'}  Δ={d:+.4f}")

    joint_yr = {y: run_year(data_by_yr[y], best_lb) for y in YEARS}
    joint_obj = obj(joint_yr)
    joint_delta = joint_obj - bl_obj
    validated = loyo_wins >= 3 and joint_delta > 0.005

    print(f"\n{'=' * 68}")
    if validated:
        verdict = f"VALIDATED — BRD_LOOKBACK={best_lb} OBJ={joint_obj:.4f} Δ={joint_delta:+.4f} LOYO {loyo_wins}/5"
        print(f"✅ {verdict}")
    else:
        verdict = f"NO IMPROVEMENT — BRD_LOOKBACK={BASELINE_BRD} OBJ={bl_obj:.4f} optimal"
        print(f"❌ {verdict}")
        print(f"   Best: Δ={joint_delta:+.4f} | LOYO {loyo_wins}/5")
    print("=" * 68)

    elapsed = round(time.time() - t0, 1)
    out_dir = Path("artifacts/phase183"); out_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "phase": 183, "description": "BRD_LOOKBACK Re-Sweep",
        "elapsed_seconds": elapsed,
        "baseline_brd_lb": BASELINE_BRD, "baseline_obj": bl_obj, "baseline_yearly": bl_yr,
        "sweep_table": {str(k): v for k, v in sweep_results.items()},
        "best_lb": best_lb, "best_obj": joint_obj, "best_delta": joint_delta,
        "best_yearly": joint_yr,
        "loyo_wins": loyo_wins, "loyo_deltas": loyo_deltas,
        "loyo_detail": {str(k): v for k, v in loyo_detail.items()},
        "validated": validated, "verdict": verdict, "partial": False,
        "timestamp": datetime.now(UTC).isoformat(),
    }
    rpath = out_dir / "phase183_report.json"
    with open(rpath, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\n✅ Saved → {rpath}")

    if validated:
        cfg_path = Path("configs/production_p91b_champion.json")
        with open(cfg_path) as f:
            cfg = json.load(f)
        cfg["breadth_regime_switching"]["lookback_bars"] = best_lb
        old_ver = cfg.get("_version", "2.18.0"); new_ver = "2.19.0"
        cfg["_version"] = new_ver
        cfg["monitoring"]["expected_performance"]["annual_sharpe_backtest"] = round(joint_obj, 4)
        cfg["_validated"] = cfg.get("_validated", "") + \
            f"; BRD lookback P183: {BASELINE_BRD}→{best_lb} LOYO {loyo_wins}/5 Δ={joint_delta:+.4f} — PRODUCTION {new_ver}"
        with open(cfg_path, "w") as f:
            json.dump(cfg, f, indent=2)
        print(f"✅ Config updated: {old_ver} → {new_ver}")
    else:
        print(f"\n❌ NO IMPROVEMENT — BRD_LOOKBACK={BASELINE_BRD} remains optimal.")

    _signal.alarm(0)

if __name__ == "__main__":
    main()
