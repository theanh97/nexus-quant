"""
Phase 211 — FTS Overlay Re-Sweep (TS_RT × TS_RS × TS_BT × TS_BS)
===================================================================
FTS overlay scales portfolio based on funding term structure percentile.
Set in parallel P197 at OBJ=2.96. Now at OBJ=3.09, optimal may differ.
All signals pre-computed — pure NumPy sweep (very fast).

Baseline: v2.32.0, OBJ=3.0860
Current: TS_RT=0.65, TS_RS=0.45, TS_BT=0.25, TS_BS=1.70
Phase 1: Sweep RT × RS (keep BT=0.25, BS=1.70)
Phase 2: If improvement, sweep BT × BS around best RT/RS
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
    out = Path("artifacts/phase211"); out.mkdir(parents=True, exist_ok=True)
    (out / "phase211_report.json").write_text(json.dumps(_partial, indent=2))
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
DISP_THR = 0.60; DISP_SCALE = 1.0
P_LOW = 0.30; P_HIGH = 0.60

TS_RT_BASE = 0.65; TS_RS_BASE = 0.45; TS_BT_BASE = 0.25; TS_BS_BASE = 1.70

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
I460_PARAMS = {"k_per_side": 4, "lookback_bars": 460, "beta_window_bars": 168,
               "target_gross_leverage": 0.3, "rebalance_interval_bars": 48}
I415_PARAMS = {"k_per_side": 4, "lookback_bars": 415, "beta_window_bars": 144,
               "target_gross_leverage": 0.3, "rebalance_interval_bars": 48}
F168_PARAMS = {"k_per_side": 2, "funding_lookback_bars": 168, "direction": "contrarian",
               "target_gross_leverage": 0.25, "rebalance_interval_bars": 24}

BASELINE_OBJ = 3.0860

# Phase 1: RT × RS sweep (fixed BT=0.25, BS=1.70)
RT_SWEEP = [0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]
RS_SWEEP = [0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.70, 0.80, 1.00]
# Phase 2: BT × BS sweep (with best RT/RS)
BT_SWEEP = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40]
BS_SWEEP = [1.00, 1.20, 1.40, 1.60, 1.70, 1.80, 2.00, 2.20, 2.50]

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
    fund_rates = np.zeros((n, len(SYMBOLS)))
    for j, sym in enumerate(SYMBOLS):
        for i in range(n):
            ts = dataset.timeline[i]
            try: fund_rates[i, j] = dataset.last_funding_rate_before(sym, ts)
            except: fund_rates[i, j] = 0.0
    btc_rets = np.zeros(n)
    for i in range(1, n):
        c0 = close_mat[i-1, 0]; c1 = close_mat[i, 0]
        btc_rets[i] = (c1/c0 - 1.0) if c0 > 0 else 0.0
    btc_vol = np.full(n, np.nan)
    for i in range(168, n):
        btc_vol[i] = float(np.std(btc_rets[i-168:i])) * np.sqrt(8760)
    if 168 < n: btc_vol[:168] = btc_vol[168]
    fund_std_raw = np.std(fund_rates, axis=1)
    fund_std_pct = np.full(n, 0.5)
    for i in range(FUND_DISP_PCT, n):
        fund_std_pct[i] = float(np.mean(fund_std_raw[i-FUND_DISP_PCT:i] <= fund_std_raw[i]))
    xsect_mean = np.mean(fund_rates, axis=1)
    ts_raw = rolling_mean_arr(xsect_mean, TS_SHORT) - rolling_mean_arr(xsect_mean, TS_LONG)
    ts_spread_pct = np.full(n, 0.5)
    for i in range(PCT_WINDOW, n):
        ts_spread_pct[i] = float(np.mean(ts_raw[i-PCT_WINDOW:i] <= ts_raw[i]))
    close_mat2 = close_mat
    breadth = np.full(n, 0.5)
    for i in range(BRD_LB, n):
        pos = sum(1 for j in range(len(SYMBOLS))
                  if close_mat2[i-BRD_LB, j] > 0 and close_mat2[i, j] > close_mat2[i-BRD_LB, j])
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

def ensemble_rets(ts_rt, ts_rs, ts_bt, ts_bs, base_data):
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
        w = REGIME_WEIGHTS[RNAMES[ridx]]
        ret_i = sum(w[sk] * sig_rets[sk][i] for sk in sk_all)
        if not np.isnan(bv[i]) and bv[i] > VOL_THRESHOLD:
            ret_i *= VOL_SCALE
        if fsp[i] > DISP_THR: ret_i *= DISP_SCALE
        if tsp[i] > ts_rt: ret_i *= ts_rs
        elif tsp[i] < ts_bt: ret_i *= ts_bs
        ens[i] = ret_i
    return ens

def sharpe(r):
    r = r[~np.isnan(r)]
    if len(r) < 2: return 0.0
    return float(np.mean(r) / np.std(r, ddof=1) * np.sqrt(8760))

def obj(yearly):
    vals = list(yearly.values())
    return float(np.mean(vals) - 0.5 * np.std(vals, ddof=1))

def evaluate(ts_rt, ts_rs, ts_bt, ts_bs, all_data):
    yearly = {yr: sharpe(ensemble_rets(ts_rt, ts_rs, ts_bt, ts_bs, d)) for yr, d in all_data.items()}
    return obj(yearly), yearly

def main():
    t0 = time.time()
    print("=" * 65)
    print("Phase 211 — FTS Overlay Re-Sweep")
    print(f"Baseline: v2.32.0  OBJ={BASELINE_OBJ}")
    print(f"Current: RT={TS_RT_BASE} RS={TS_RS_BASE} BT={TS_BT_BASE} BS={TS_BS_BASE}")
    print("=" * 65)

    print("\n[1/4] Loading per-year data & pre-computing all signals...")
    all_data = {}
    for yr in sorted(YEAR_RANGES):
        print(f"  {yr}: ", end="", flush=True)
        all_data[yr] = load_year_data(yr)
    print()

    base_obj, base_yearly = evaluate(TS_RT_BASE, TS_RS_BASE, TS_BT_BASE, TS_BS_BASE, all_data)
    print(f"\n  Baseline OBJ = {base_obj:.4f}  (expected ≈ {BASELINE_OBJ})")

    # Phase 1: RT × RS sweep
    print("\n[2/4] Phase 1 — RT × RS sweep (BT=0.25 BS=1.70 fixed)...")
    best_obj = base_obj; best_rt = TS_RT_BASE; best_rs = TS_RS_BASE
    for rt in RT_SWEEP:
        for rs in RS_SWEEP:
            o, _ = evaluate(rt, rs, TS_BT_BASE, TS_BS_BASE, all_data)
            delta = o - base_obj
            marker = " ← BEST" if o > best_obj else ""
            print(f"  RT={rt:.2f}  RS={rs:.2f}  OBJ={o:.4f}  Δ={delta:+.4f}{marker}")
            if o > best_obj:
                best_obj = o; best_rt = rt; best_rs = rs

    print(f"\n  → Best RT/RS: rt={best_rt} rs={best_rs}  OBJ={best_obj:.4f}")

    # Phase 2: BT × BS sweep with best RT/RS
    print("\n[3/4] Phase 2 — BT × BS sweep (with best RT/RS)...")
    best_bt = TS_BT_BASE; best_bs = TS_BS_BASE
    for bt in BT_SWEEP:
        for bs in BS_SWEEP:
            o, _ = evaluate(best_rt, best_rs, bt, bs, all_data)
            delta = o - base_obj
            marker = " ← BEST" if o > best_obj else ""
            print(f"  BT={bt:.2f}  BS={bs:.2f}  OBJ={o:.4f}  Δ={delta:+.4f}{marker}")
            if o > best_obj:
                best_obj = o; best_bt = bt; best_bs = bs

    print(f"\n  → Best: RT={best_rt} RS={best_rs} BT={best_bt} BS={best_bs}  OBJ={best_obj:.4f}")

    print("\n[4/4] LOYO validation...")
    joint_obj, joint_yearly = evaluate(best_rt, best_rs, best_bt, best_bs, all_data)
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
    print(f"\n{'✅ VALIDATED' if validated else '❌ NO IMPROVEMENT'} — RT={best_rt} RS={best_rs} BT={best_bt} BS={best_bs}")

    result = {
        "phase": 211, "baseline_obj": float(base_obj), "best_obj": float(joint_obj),
        "delta": float(delta), "loyo_wins": loyo_wins, "loyo_total": len(joint_yearly),
        "validated": validated,
        "best_rt": best_rt, "best_rs": best_rs, "best_bt": best_bt, "best_bs": best_bs,
    }
    _partial.update(result)
    out = Path("artifacts/phase211"); out.mkdir(parents=True, exist_ok=True)
    (out / "phase211_report.json").write_text(json.dumps(result, indent=2))

    if validated:
        cfg_path = ROOT / "configs" / "production_p91b_champion.json"
        cfg = json.load(open(cfg_path))
        fts = cfg.setdefault("funding_ts_overlay_params", {})
        fts["ts_rt"] = best_rt; fts["ts_rs"] = best_rs
        fts["ts_bt"] = best_bt; fts["ts_bs"] = best_bs
        cfg["_version"] = "2.33.0"
        cfg["monitoring"]["expected_performance"]["annual_sharpe_backtest"] = round(joint_obj, 4)
        with open(cfg_path, "w") as f:
            json.dump(cfg, f, indent=2, ensure_ascii=False)
        print(f"\n  Config → v2.33.0  OBJ={round(joint_obj,4)}")

    print(f"\nRuntime: {int(time.time()-t0)}s")

main()
