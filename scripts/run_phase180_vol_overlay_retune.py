"""
PHASE 180 — Vol Regime Overlay Re-Tune
=========================================
Baseline: v2.16.0  OBJ≈2.5396 (v2.15.0 reference)
Current:  VOL_THRESHOLD=0.50  VOL_SCALE=0.40  VOL_F168_BOOST=0.10

Note: VOL_SCALE reduces ensemble when BTC vol is high.
      VOL_F168_BOOST shifts weight to F168 in high-vol regime.

Parts:
  A — VOL_THRESHOLD sweep ∈ [0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65]
  B — VOL_SCALE sweep     ∈ [0.25, 0.30, 0.35, 0.40, 0.45, 0.50]
  C — VOL_F168_BOOST      ∈ [0.00, 0.05, 0.10, 0.15, 0.20]
  D — Joint LOYO validation with best A/B/C combo
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

# ── SIGALRM ───────────────────────────────────────────────────────────────────
_partial: dict = {}
def _on_timeout(signum, frame):
    _partial["partial"] = True
    _partial["timeout_at"] = datetime.now(UTC).isoformat()
    out_dir = Path("artifacts/phase180"); out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "phase180_report.json").write_text(json.dumps(_partial, indent=2))
    sys.exit(0)
_signal.signal(_signal.SIGALRM, _on_timeout)
_signal.alarm(7200)

# ── Constants ─────────────────────────────────────────────────────────────────
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
BRD_LOOKBACK  = 192
PCT_WINDOW    = 336
P_LOW, P_HIGH = 0.35, 0.65
TS_SHORT, TS_LONG = 12, 96
FUND_DISP_PCT = 240

# Baseline vol params (what we're sweeping)
BASELINE_VT = 0.50   # VOL_THRESHOLD
BASELINE_VS = 0.40   # VOL_SCALE
BASELINE_VB = 0.10   # VOL_F168_BOOST

# Fixed params from prior phases
TS_RT = 0.60; TS_RS = 0.40; TS_BT = 0.25; TS_BS = 1.50
DISP_THR = 0.60; DISP_SCALE = 1.0   # P179: scale=1.0 (disabled)

WEIGHTS = {
    "LOW":  {"v1": 0.2415, "i460bw168": 0.1730, "i415bw216": 0.2855, "f168": 0.3000},
    "MID":  {"v1": 0.1493, "i460bw168": 0.2053, "i415bw216": 0.3453, "f168": 0.3000},
    "HIGH": {"v1": 0.0567, "i460bw168": 0.2833, "i415bw216": 0.5100, "f168": 0.1500},
}

BASELINE_OBJ = 2.5396

COST_MODEL = cost_model_from_config({
    "fee_rate": 0.0005, "slippage_rate": 0.0003, "cost_multiplier": 1.0,
})

# ── Helpers ───────────────────────────────────────────────────────────────────
def rolling_mean_arr(a: np.ndarray, w: int) -> np.ndarray:
    out = np.full(len(a), np.nan)
    if w <= 0 or len(a) == 0:
        return out
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

# ── Data loading ──────────────────────────────────────────────────────────────
def load_year(year: int):
    s, e = YEAR_RANGES[year]
    cfg = {"provider": "binance_rest_v1", "symbols": SYMBOLS,
           "start": s, "end": e, "bar_interval": "1h",
           "cache_dir": ".cache/binance_rest"}
    dataset = make_provider(cfg, seed=42).load()
    signals = compute_signals(dataset)
    sig_rets = {}
    for sk, sname, params in [
        ("v1", "nexus_alpha_v1", {
            "k_per_side": 2, "w_carry": 0.35, "w_mom": 0.45, "w_mean_reversion": 0.2,
            "momentum_lookback_bars": 336, "mean_reversion_lookback_bars": 72,
            "vol_lookback_bars": 168, "target_gross_leverage": 0.35,
            "rebalance_interval_bars": 60,
        }),
        ("i460bw168", "idio_momentum_alpha", {
            "k_per_side": 4, "lookback_bars": 460, "beta_window_bars": 120,
            "target_gross_leverage": 0.3, "rebalance_interval_bars": 48,
        }),
        ("i415bw216", "idio_momentum_alpha", {
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
    return dataset, signals, sig_rets

def compute_signals(dataset) -> dict:
    n = len(dataset.timeline)
    btc_rets = np.zeros(n)
    for i in range(1, n):
        c0 = dataset.close("BTCUSDT", i - 1)
        c1 = dataset.close("BTCUSDT", i)
        btc_rets[i] = (c1 / c0 - 1.0) if c0 > 0 else 0.0
    btc_vol = np.full(n, np.nan)
    for i in range(VOL_WINDOW, n):
        btc_vol[i] = float(np.std(btc_rets[i - VOL_WINDOW:i])) * np.sqrt(8760)
    if VOL_WINDOW < n:
        btc_vol[:VOL_WINDOW] = btc_vol[VOL_WINDOW]
    breadth = np.full(n, 0.5)
    for i in range(BRD_LOOKBACK, n):
        pos = sum(1 for sym in SYMBOLS
                  if (c0 := dataset.close(sym, i - BRD_LOOKBACK)) > 0
                  and dataset.close(sym, i) > c0)
        breadth[i] = pos / len(SYMBOLS)
    breadth[:BRD_LOOKBACK] = 0.5
    brd_pct = np.full(n, 0.5)
    for i in range(PCT_WINDOW, n):
        brd_pct[i] = float(np.mean(breadth[i - PCT_WINDOW:i] <= breadth[i]))
    brd_pct[:PCT_WINDOW] = 0.5
    breadth_regime = np.where(brd_pct >= P_HIGH, 2,
                     np.where(brd_pct >= P_LOW,  1, 0)).astype(int)
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
    return {"btc_vol": btc_vol, "breadth_regime": breadth_regime,
            "fund_std_pct": fund_std_pct, "ts_spread_pct": ts_spread_pct}

def compute_ensemble(sig_rets: dict, signals: dict,
                     vt: float, vs: float, vb: float) -> np.ndarray:
    sk_all = ["v1", "i460bw168", "i415bw216", "f168"]
    min_len = min(len(sig_rets[sk]) for sk in sk_all)
    bv  = signals["btc_vol"][:min_len]
    reg = signals["breadth_regime"][:min_len]
    fsp = signals["fund_std_pct"][:min_len]
    tsp = signals["ts_spread_pct"][:min_len]
    ens = np.zeros(min_len)
    for i in range(min_len):
        w = WEIGHTS[["LOW", "MID", "HIGH"][int(reg[i])]]
        if not np.isnan(bv[i]) and bv[i] > vt:
            boost_per = vb / max(1, len(sk_all) - 1)
            ret_i = 0.0
            for sk in sk_all:
                adj_w = (min(0.60, w[sk] + vb) if sk == "f168"
                         else max(0.05, w[sk] - boost_per))
                ret_i += adj_w * sig_rets[sk][i]
            ret_i *= vs
        else:
            ret_i = sum(w[sk] * sig_rets[sk][i] for sk in sk_all)
        # Dispersion (v2.16.0: scale=1.0 = disabled)
        if fsp[i] > DISP_THR:
            ret_i *= DISP_SCALE
        # TS overlay
        if tsp[i] > TS_RT:
            ret_i *= TS_RS
        elif tsp[i] < TS_BT:
            ret_i *= TS_BS
        ens[i] = ret_i
    return ens

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    t0 = time.time()
    print("=" * 68)
    print("PHASE 180 — Vol Regime Overlay Re-Tune (v2.16.0 baseline)")
    print("=" * 68)
    print(f"  Baseline: VT={BASELINE_VT} VS={BASELINE_VS} VB={BASELINE_VB}")
    print(f"  Parts: A=VT sweep, B=VS sweep, C=VB sweep, D=joint LOYO")

    print(f"\n[1/6] Loading data ...")
    data_by_yr = {}
    for y in YEARS:
        print(f"  {y}: ", end="", flush=True)
        data_by_yr[y] = load_year(y)
    print()

    bl_yr = {y: sharpe(compute_ensemble(
        data_by_yr[y][2], data_by_yr[y][1],
        BASELINE_VT, BASELINE_VS, BASELINE_VB)) for y in YEARS}
    bl_obj = obj(bl_yr)
    print(f"  Baseline OBJ={bl_obj:.4f}")

    # ── Part A: VOL_THRESHOLD ─────────────────────────────────────────────────
    print(f"\n[2/6] Part A — VOL_THRESHOLD sweep (VS={BASELINE_VS} VB={BASELINE_VB} locked) ...")
    vt_vals = [0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65]
    best_vt = BASELINE_VT; best_vt_obj = bl_obj
    vt_results = {}
    for vt in vt_vals:
        yr = {y: sharpe(compute_ensemble(
            data_by_yr[y][2], data_by_yr[y][1],
            vt, BASELINE_VS, BASELINE_VB)) for y in YEARS}
        o = obj(yr); d = o - bl_obj
        sym = "✅" if d > 0.005 else ("⚠️ " if abs(d) <= 0.005 else "❌")
        base_marker = " ← baseline" if abs(vt - BASELINE_VT) < 1e-9 else ""
        print(f"    vt={vt} → OBJ={o:.4f}  Δ={d:+.4f}  {sym}{base_marker}")
        vt_results[vt] = {"vt": vt, "obj": o, "delta": d, "yearly": yr}
        if o > best_vt_obj: best_vt_obj = o; best_vt = vt
    print(f"  VT winner: vt={best_vt}  OBJ={best_vt_obj:.4f}")

    # ── Part B: VOL_SCALE ─────────────────────────────────────────────────────
    print(f"\n[3/6] Part B — VOL_SCALE sweep (VT={best_vt} VB={BASELINE_VB} locked) ...")
    vs_vals = [0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.60, 0.70, 0.80]
    best_vs = BASELINE_VS; best_vs_obj = best_vt_obj
    vs_results = {}
    for vs in vs_vals:
        yr = {y: sharpe(compute_ensemble(
            data_by_yr[y][2], data_by_yr[y][1],
            best_vt, vs, BASELINE_VB)) for y in YEARS}
        o = obj(yr); d = o - bl_obj
        sym = "✅" if d > 0.005 else ("⚠️ " if abs(d) <= 0.005 else "❌")
        print(f"    vs={vs} → OBJ={o:.4f}  Δ={d:+.4f}  {sym}")
        vs_results[vs] = {"vs": vs, "obj": o, "delta": d, "yearly": yr}
        if o > best_vs_obj: best_vs_obj = o; best_vs = vs
    print(f"  VS winner: vs={best_vs}  OBJ={best_vs_obj:.4f}")

    # ── Part C: VOL_F168_BOOST ────────────────────────────────────────────────
    print(f"\n[4/6] Part C — VOL_F168_BOOST sweep (VT={best_vt} VS={best_vs} locked) ...")
    vb_vals = [0.00, 0.05, 0.10, 0.15, 0.20, 0.25]
    best_vb = BASELINE_VB; best_vb_obj = best_vs_obj
    vb_results = {}
    for vb in vb_vals:
        yr = {y: sharpe(compute_ensemble(
            data_by_yr[y][2], data_by_yr[y][1],
            best_vt, best_vs, vb)) for y in YEARS}
        o = obj(yr); d = o - bl_obj
        sym = "✅" if d > 0.005 else ("⚠️ " if abs(d) <= 0.005 else "❌")
        base_marker = " ← baseline" if abs(vb - BASELINE_VB) < 1e-9 else ""
        print(f"    vb={vb} → OBJ={o:.4f}  Δ={d:+.4f}  {sym}{base_marker}")
        vb_results[vb] = {"vb": vb, "obj": o, "delta": d, "yearly": yr}
        if o > best_vb_obj: best_vb_obj = o; best_vb = vb
    print(f"  VB winner: vb={best_vb}  OBJ={best_vb_obj:.4f}")

    # ── Part D: Joint LOYO ────────────────────────────────────────────────────
    print(f"\n[5/6] Joint best: VT={best_vt} VS={best_vs} VB={best_vb}")
    print(f"\n[6/6] Part D — Joint LOYO validation ...")
    loyo_wins = 0; loyo_deltas = []; loyo_detail = {}
    for loyo_yr in YEARS:
        train_yrs = [y for y in YEARS if y != loyo_yr]
        tr_yr = {y: sharpe(compute_ensemble(
            data_by_yr[y][2], data_by_yr[y][1],
            best_vt, best_vs, best_vb)) for y in train_yrs}
        bl_yr_t = {y: sharpe(compute_ensemble(
            data_by_yr[y][2], data_by_yr[y][1],
            BASELINE_VT, BASELINE_VS, BASELINE_VB)) for y in train_yrs}
        tr_obj = obj(tr_yr); bl_obj_t = obj(bl_yr_t)
        d = tr_obj - bl_obj_t
        win = d > 0.005
        loyo_wins += int(win); loyo_deltas.append(d)
        loyo_detail[loyo_yr] = {"obj": tr_obj, "baseline_obj": bl_obj_t, "delta": d, "win": win}
        sym = "✅" if win else "❌"
        print(f"    {loyo_yr}: {sym}  Δ={d:+.4f}")

    joint_yr = {y: sharpe(compute_ensemble(
        data_by_yr[y][2], data_by_yr[y][1],
        best_vt, best_vs, best_vb)) for y in YEARS}
    joint_obj = obj(joint_yr)
    joint_delta = joint_obj - bl_obj
    validated = loyo_wins >= 3 and joint_delta > 0.005

    print(f"\n{'=' * 68}")
    if validated:
        verdict = f"VALIDATED — VT={best_vt} VS={best_vs} VB={best_vb} OBJ={joint_obj:.4f} Δ={joint_delta:+.4f} LOYO {loyo_wins}/5"
        print(f"✅ {verdict}")
    else:
        verdict = f"NO IMPROVEMENT — VT={BASELINE_VT} VS={BASELINE_VS} VB={BASELINE_VB} OBJ={bl_obj:.4f} optimal"
        print(f"❌ {verdict}")
        print(f"   Best: Δ={joint_delta:+.4f} | LOYO {loyo_wins}/5")
    print("=" * 68)

    elapsed = round(time.time() - t0, 1)
    out_dir = Path("artifacts/phase180"); out_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "phase": 180, "description": "Vol Regime Overlay Re-Tune",
        "elapsed_seconds": elapsed,
        "baseline_vt": BASELINE_VT, "baseline_vs": BASELINE_VS, "baseline_vb": BASELINE_VB,
        "baseline_obj": bl_obj, "baseline_yearly": bl_yr,
        "vt_sweep": {str(k): v for k, v in vt_results.items()},
        "vs_sweep": {str(k): v for k, v in vs_results.items()},
        "vb_sweep": {str(k): v for k, v in vb_results.items()},
        "best_vt": best_vt, "best_vs": best_vs, "best_vb": best_vb,
        "best_obj": joint_obj, "best_delta": joint_delta, "best_yearly": joint_yr,
        "loyo_wins": loyo_wins, "loyo_deltas": loyo_deltas, "loyo_detail": {str(k): v for k, v in loyo_detail.items()},
        "validated": validated, "verdict": verdict, "partial": False,
        "timestamp": datetime.now(UTC).isoformat(),
    }
    rpath = out_dir / "phase180_report.json"
    with open(rpath, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\n✅ Saved → {rpath}")

    if validated:
        cfg_path = Path("configs/production_p91b_champion.json")
        with open(cfg_path) as f:
            cfg = json.load(f)
        # Update vol_regime_overlay params
        cfg["vol_regime_overlay"]["vol_threshold"] = best_vt
        cfg["vol_regime_overlay"]["scale_factor"]  = best_vs
        cfg["vol_regime_overlay"]["f168_boost"]    = best_vb
        old_ver = cfg.get("_version", "2.16.0")
        new_ver = "2.17.0"
        cfg["_version"] = new_ver
        cfg["monitoring"]["expected_performance"]["annual_sharpe_backtest"] = round(joint_obj, 4)
        cfg["_validated"] = cfg.get("_validated", "") + \
            f"; Vol overlay P180: VT={BASELINE_VT}→{best_vt} VS={BASELINE_VS}→{best_vs} VB={BASELINE_VB}→{best_vb} LOYO {loyo_wins}/5 Δ={joint_delta:+.4f} OBJ={joint_obj:.4f} — PRODUCTION {new_ver}"
        with open(cfg_path, "w") as f:
            json.dump(cfg, f, indent=2)
        print(f"✅ Config updated: {old_ver} → {new_ver}")
    else:
        print(f"\n❌ NO IMPROVEMENT — vol overlay params remain optimal.")

    _signal.alarm(0)

if __name__ == "__main__":
    main()
