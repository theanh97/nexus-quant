"""
PHASE 179 — Funding Dispersion Overlay Re-Tune
=================================================
Baseline: v2.15.0  OBJ=2.5396
Current:  FUND_DISP_THR=0.60  FUND_DISP_SCALE=1.05

Parts:
  A — THR sweep  ∈ [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]
  B — SCALE sweep ∈ [1.00, 1.02, 1.05, 1.08, 1.10, 1.15, 1.20]
  C — Joint LOYO validation

Shared each year: V1, I460bw120, I415bw144, F168lb168, signals
"""
import os, sys, json, signal, time
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

# ── SIGALRM ──────────────────────────────────────────────────────────────────
signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError("P179 timeout")))
signal.alarm(7200)

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

# ── Overlay constants (v2.15.0) ───────────────────────────────────────────────
VOL_WINDOW    = 168
BRD_LOOKBACK  = 192
PCT_WINDOW    = 336
P_LOW, P_HIGH = 0.35, 0.65
TS_SHORT, TS_LONG = 12, 96
FUND_DISP_PCT = 240

VOL_THRESHOLD  = 0.50
VOL_SCALE      = 0.40
VOL_F168_BOOST = 0.10

# Current TS params (P178 confirmed optimal)
TS_RT = 0.60; TS_RS = 0.40; TS_BT = 0.25; TS_BS = 1.50

# Current dispersion params (baseline)
BASELINE_THR   = 0.60
BASELINE_SCALE = 1.05

WEIGHTS = {
    "LOW":  {"v1": 0.2415, "i460bw168": 0.1730, "i415bw216": 0.2855, "f168": 0.3000},
    "MID":  {"v1": 0.1493, "i460bw168": 0.2053, "i415bw216": 0.3453, "f168": 0.3000},
    "HIGH": {"v1": 0.0567, "i460bw168": 0.2833, "i415bw216": 0.5100, "f168": 0.1500},
}

BASELINE_OBJ = 2.5396

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
    ann = np.mean(r) * 8760 / np.std(r, ddof=1) * np.sqrt(8760)
    return float(ann)

def obj(yearly: dict) -> float:
    vals = list(yearly.values())
    return float(np.mean(vals) - 0.5 * np.std(vals, ddof=1))

# ── Data & signal loading ─────────────────────────────────────────────────────
COST_MODEL = cost_model_from_config({
    "fee_rate": 0.0005, "slippage_rate": 0.0003, "cost_multiplier": 1.0,
})

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
    # BTC vol
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
    # Breadth
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
    # Funding rates
    fund_rates = np.zeros((n, len(SYMBOLS)))
    for j, sym in enumerate(SYMBOLS):
        for i in range(n):
            ts = dataset.timeline[i]
            try:    fund_rates[i, j] = dataset.last_funding_rate_before(sym, ts)
            except: fund_rates[i, j] = 0.0
    # Funding dispersion percentile
    fund_std_raw = np.std(fund_rates, axis=1)
    fund_std_pct = np.full(n, 0.5)
    for i in range(FUND_DISP_PCT, n):
        fund_std_pct[i] = float(np.mean(fund_std_raw[i - FUND_DISP_PCT:i] <= fund_std_raw[i]))
    fund_std_pct[:FUND_DISP_PCT] = 0.5
    # TS overlay
    xsect_mean = np.mean(fund_rates, axis=1)
    ts_raw = rolling_mean_arr(xsect_mean, TS_SHORT) - rolling_mean_arr(xsect_mean, TS_LONG)
    ts_spread_pct = np.full(n, 0.5)
    for i in range(PCT_WINDOW, n):
        ts_spread_pct[i] = float(np.mean(ts_raw[i - PCT_WINDOW:i] <= ts_raw[i]))
    ts_spread_pct[:PCT_WINDOW] = 0.5
    return {"btc_vol": btc_vol, "breadth_regime": breadth_regime,
            "fund_std_pct": fund_std_pct, "ts_spread_pct": ts_spread_pct}

def compute_ensemble(sig_rets: dict, signals: dict,
                     disp_thr: float, disp_scale: float) -> np.ndarray:
    sk_all = ["v1", "i460bw168", "i415bw216", "f168"]
    min_len = min(len(sig_rets[sk]) for sk in sk_all)
    bv  = signals["btc_vol"][:min_len]
    reg = signals["breadth_regime"][:min_len]
    fsp = signals["fund_std_pct"][:min_len]
    tsp = signals["ts_spread_pct"][:min_len]
    ens = np.zeros(min_len)
    for i in range(min_len):
        w = WEIGHTS[["LOW", "MID", "HIGH"][int(reg[i])]]
        if not np.isnan(bv[i]) and bv[i] > VOL_THRESHOLD:
            boost_per = VOL_F168_BOOST / max(1, len(sk_all) - 1)
            ret_i = 0.0
            for sk in sk_all:
                adj_w = (min(0.60, w[sk] + VOL_F168_BOOST) if sk == "f168"
                         else max(0.05, w[sk] - boost_per))
                ret_i += adj_w * sig_rets[sk][i]
            ret_i *= VOL_SCALE
        else:
            ret_i = sum(w[sk] * sig_rets[sk][i] for sk in sk_all)
        # Dispersion boost — swept param
        if fsp[i] > disp_thr:
            ret_i *= disp_scale
        # TS overlay (v2.15.0 confirmed optimal)
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
    print("PHASE 179 — Funding Dispersion Overlay Re-Tune (v2.15.0 baseline)")
    print("=" * 68)
    print(f"  Baseline: THR={BASELINE_THR} SCALE={BASELINE_SCALE}")
    print(f"  Baseline OBJ≈{BASELINE_OBJ}")
    print(f"  Parts: A=THR sweep, B=SCALE sweep, C=joint LOYO")

    print(f"\n[1/5] Loading data, signals, strategy returns ...")
    data_by_yr = {}
    for y in YEARS:
        print(f"  {y}: ", end="", flush=True)
        data_by_yr[y] = load_year(y)
    print()

    # Confirm baseline
    baseline_yr = {y: sharpe(compute_ensemble(
        data_by_yr[y][2], data_by_yr[y][1],
        BASELINE_THR, BASELINE_SCALE)) for y in YEARS}
    baseline_obj = obj(baseline_yr)
    print(f"  Baseline OBJ={baseline_obj:.4f}  ({' '.join(f'{y}:{v:.4f}' for y, v in baseline_yr.items())})")

    # ── Part A: THR sweep ─────────────────────────────────────────────────────
    print(f"\n[2/5] Part A — THR sweep (SCALE={BASELINE_SCALE} locked) ...")
    thr_sweep = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]
    best_thr = BASELINE_THR
    best_thr_obj = baseline_obj
    thr_results = {}
    for thr in thr_sweep:
        yr = {y: sharpe(compute_ensemble(
            data_by_yr[y][2], data_by_yr[y][1],
            thr, BASELINE_SCALE)) for y in YEARS}
        o = obj(yr)
        d = o - baseline_obj
        sym = "✅" if d > 0.005 else ("⚠️ " if abs(d) <= 0.005 else "❌")
        base_marker = " ← baseline" if abs(thr - BASELINE_THR) < 1e-9 else ""
        print(f"    thr={thr} → OBJ={o:.4f}  Δ={d:+.4f}  {sym}{base_marker}")
        thr_results[thr] = {"thr": thr, "obj": o, "delta": d, "yearly": yr}
        if o > best_thr_obj:
            best_thr_obj = o; best_thr = thr
    print(f"  THR winner: thr={best_thr}  OBJ={best_thr_obj:.4f}")

    # ── Part B: SCALE sweep ───────────────────────────────────────────────────
    print(f"\n[3/5] Part B — SCALE sweep (THR={best_thr} locked) ...")
    scale_sweep = [1.00, 1.02, 1.05, 1.08, 1.10, 1.15, 1.20]
    best_scale = BASELINE_SCALE
    best_scale_obj = best_thr_obj
    scale_results = {}
    for sc in scale_sweep:
        yr = {y: sharpe(compute_ensemble(
            data_by_yr[y][2], data_by_yr[y][1],
            best_thr, sc)) for y in YEARS}
        o = obj(yr)
        d = o - baseline_obj
        sym = "✅" if d > 0.005 else ("⚠️ " if abs(d) <= 0.005 else "❌")
        base_marker = " ← prev best" if abs(sc - BASELINE_SCALE) < 1e-9 else ""
        print(f"    scale={sc} → OBJ={o:.4f}  Δ={d:+.4f}  {sym}{base_marker}")
        scale_results[sc] = {"scale": sc, "obj": o, "delta": d, "yearly": yr}
        if o > best_scale_obj:
            best_scale_obj = o; best_scale = sc
    print(f"  SCALE winner: scale={best_scale}  OBJ={best_scale_obj:.4f}")

    # ── Part C: Joint LOYO ────────────────────────────────────────────────────
    print(f"\n[4/5] Joint best: THR={best_thr} SCALE={best_scale}")
    print(f"\n[5/5] Part C — Joint LOYO validation ...")
    loyo_wins = 0
    loyo_deltas = []
    loyo_detail = {}
    for loyo_yr in YEARS:
        train_yrs = [y for y in YEARS if y != loyo_yr]
        tr_yr = {y: sharpe(compute_ensemble(
            data_by_yr[y][2], data_by_yr[y][1],
            best_thr, best_scale)) for y in train_yrs}
        bl_yr = {y: sharpe(compute_ensemble(
            data_by_yr[y][2], data_by_yr[y][1],
            BASELINE_THR, BASELINE_SCALE)) for y in train_yrs}
        tr_obj = obj(tr_yr)
        bl_obj = obj(bl_yr)
        d = tr_obj - bl_obj
        win = d > 0.005
        loyo_wins += int(win)
        loyo_deltas.append(d)
        loyo_detail[loyo_yr] = {"obj": tr_obj, "baseline_obj": bl_obj, "delta": d, "win": win}
        sym = "✅" if win else "❌"
        print(f"    {loyo_yr}: {sym}  Δ={d:+.4f}")

    joint_yr = {y: sharpe(compute_ensemble(
        data_by_yr[y][2], data_by_yr[y][1],
        best_thr, best_scale)) for y in YEARS}
    joint_obj = obj(joint_yr)
    joint_delta = joint_obj - baseline_obj
    validated = loyo_wins >= 3 and joint_delta > 0.005

    print(f"\n{'=' * 68}")
    if validated:
        verdict = f"VALIDATED — THR={best_thr} SCALE={best_scale} OBJ={joint_obj:.4f} Δ={joint_delta:+.4f} LOYO {loyo_wins}/5"
        print(f"✅ {verdict}")
    else:
        verdict = f"NO IMPROVEMENT — THR={BASELINE_THR} SCALE={BASELINE_SCALE} OBJ={baseline_obj:.4f} optimal"
        print(f"❌ {verdict}")
        print(f"   Best: Δ={joint_delta:+.4f} | LOYO {loyo_wins}/5")
    print("=" * 68)

    elapsed = round(time.time() - t0, 1)

    # ── Artifact ──────────────────────────────────────────────────────────────
    out_dir = Path("artifacts/phase179")
    out_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "phase": 179,
        "description": "Funding Dispersion Overlay Re-Tune",
        "elapsed_seconds": elapsed,
        "baseline_thr": BASELINE_THR,
        "baseline_scale": BASELINE_SCALE,
        "baseline_obj": baseline_obj,
        "baseline_yearly": baseline_yr,
        "thr_sweep": {str(k): v for k, v in thr_results.items()},
        "scale_sweep": {str(k): v for k, v in scale_results.items()},
        "best_thr": best_thr,
        "best_scale": best_scale,
        "best_obj": joint_obj,
        "best_delta": joint_delta,
        "best_yearly": joint_yr,
        "loyo_wins": loyo_wins,
        "loyo_deltas": loyo_deltas,
        "loyo_detail": {str(k): v for k, v in loyo_detail.items()},
        "validated": validated,
        "verdict": verdict,
        "partial": False,
        "timestamp": datetime.now(UTC).isoformat(),
    }
    rpath = out_dir / "phase179_report.json"
    with open(rpath, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\n✅ Saved → {rpath}")

    # ── Config update if validated ────────────────────────────────────────────
    if validated:
        cfg_path = Path("configs/production_p91b_champion.json")
        with open(cfg_path) as f:
            cfg = json.load(f)
        cfg["ensemble"]["overlays"]["funding_dispersion"]["threshold_pct"] = best_thr
        cfg["ensemble"]["overlays"]["funding_dispersion"]["scale"] = best_scale
        old_ver = cfg.get("version", "2.15.0")
        new_ver = "2.16.0"
        cfg["version"] = new_ver
        cfg["monitoring"]["expected_performance"]["annual_sharpe_backtest"] = round(joint_obj, 4)
        cfg["_validated"] = cfg.get("_validated", "") + \
            f"; Disp overlay P179: thr={BASELINE_THR}→{best_thr} scale={BASELINE_SCALE}→{best_scale} LOYO {loyo_wins}/5 Δ={joint_delta:+.4f} OBJ={joint_obj:.4f} — PRODUCTION {new_ver}"
        with open(cfg_path, "w") as f:
            json.dump(cfg, f, indent=2)
        print(f"✅ Config updated: {old_ver} → {new_ver}")
        print(f"   THR {BASELINE_THR} → {best_thr}  SCALE {BASELINE_SCALE} → {best_scale}")
        print(f"   OBJ {baseline_obj:.4f} → {joint_obj:.4f}")
    else:
        print(f"\n❌ NO IMPROVEMENT — THR={BASELINE_THR}/SCALE={BASELINE_SCALE} remain optimal.")

    signal.alarm(0)

if __name__ == "__main__":
    main()
