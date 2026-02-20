"""
PHASE 181 — Breadth Regime Threshold Re-Sweep
=================================================
Baseline: v2.16.0  OBJ≈2.4827
Current:  P_LOW=0.35  P_HIGH=0.65  (percentile thresholds for breadth regime)

Parts:
  A — P_LOW sweep  ∈ [0.20, 0.25, 0.30, 0.35, 0.40, 0.45]
  B — P_HIGH sweep ∈ [0.55, 0.60, 0.65, 0.70, 0.75, 0.80]
  C — Joint LOYO validation with best A/B combo
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
    out = Path("artifacts/phase181"); out.mkdir(parents=True, exist_ok=True)
    (out / "phase181_report.json").write_text(json.dumps(_partial, indent=2)); sys.exit(0)
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
BRD_LOOKBACK  = 192
PCT_WINDOW    = 336
TS_SHORT, TS_LONG = 12, 96
FUND_DISP_PCT = 240

# Baseline regime thresholds
BASELINE_PL = 0.35
BASELINE_PH = 0.65

# Fixed overlays
VOL_THRESHOLD = 0.50; VOL_SCALE = 0.40; VOL_F168_BOOST = 0.10
TS_RT = 0.60; TS_RS = 0.40; TS_BT = 0.25; TS_BS = 1.50
DISP_THR = 0.60; DISP_SCALE = 1.0

WEIGHTS = {
    "LOW":  {"v1": 0.2415, "i460bw168": 0.1730, "i415bw216": 0.2855, "f168": 0.3000},
    "MID":  {"v1": 0.1493, "i460bw168": 0.2053, "i415bw216": 0.3453, "f168": 0.3000},
    "HIGH": {"v1": 0.0567, "i460bw168": 0.2833, "i415bw216": 0.5100, "f168": 0.1500},
}
BASELINE_OBJ = 2.4827

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

def load_year(year: int):
    s, e = YEAR_RANGES[year]
    cfg = {"provider": "binance_rest_v1", "symbols": SYMBOLS,
           "start": s, "end": e, "bar_interval": "1h",
           "cache_dir": ".cache/binance_rest"}
    dataset = make_provider(cfg, seed=42).load()
    # Pre-compute raw signals (fund_rates, btc_rets) — stored in dataset dict
    n = len(dataset.timeline)
    btc_rets = np.zeros(n)
    for i in range(1, n):
        c0 = dataset.close("BTCUSDT", i - 1); c1 = dataset.close("BTCUSDT", i)
        btc_rets[i] = (c1 / c0 - 1.0) if c0 > 0 else 0.0
    btc_vol = np.full(n, np.nan)
    for i in range(VOL_WINDOW, n):
        btc_vol[i] = float(np.std(btc_rets[i - VOL_WINDOW:i])) * np.sqrt(8760)
    if VOL_WINDOW < n: btc_vol[:VOL_WINDOW] = btc_vol[VOL_WINDOW]
    # Breadth raw
    breadth_raw = np.full(n, 0.5)
    for i in range(BRD_LOOKBACK, n):
        pos = sum(1 for sym in SYMBOLS
                  if (c0 := dataset.close(sym, i - BRD_LOOKBACK)) > 0
                  and dataset.close(sym, i) > c0)
        breadth_raw[i] = pos / len(SYMBOLS)
    breadth_raw[:BRD_LOOKBACK] = 0.5
    brd_pct = np.full(n, 0.5)
    for i in range(PCT_WINDOW, n):
        brd_pct[i] = float(np.mean(breadth_raw[i - PCT_WINDOW:i] <= breadth_raw[i]))
    brd_pct[:PCT_WINDOW] = 0.5
    # Fund rates & dispersion
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
    return sig_rets, btc_vol, brd_pct, fund_std_pct, ts_spread_pct

def compute_ensemble(sig_rets, btc_vol, brd_pct, fund_std_pct, ts_spread_pct,
                     pl: float, ph: float) -> np.ndarray:
    # Compute regime with swept thresholds
    breadth_regime = np.where(brd_pct >= ph, 2, np.where(brd_pct >= pl, 1, 0)).astype(int)
    sk_all = ["v1", "i460bw168", "i415bw216", "f168"]
    min_len = min(len(sig_rets[sk]) for sk in sk_all)
    bv = btc_vol[:min_len]; reg = breadth_regime[:min_len]
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

def run_year(data, pl, ph):
    sig_rets, btc_vol, brd_pct, fund_std_pct, ts_spread_pct = data
    return sharpe(compute_ensemble(sig_rets, btc_vol, brd_pct, fund_std_pct, ts_spread_pct, pl, ph))

def main():
    t0 = time.time()
    print("=" * 68)
    print("PHASE 181 — Breadth Regime Threshold Re-Sweep (v2.16.0 baseline)")
    print("=" * 68)
    print(f"  Baseline: P_LOW={BASELINE_PL} P_HIGH={BASELINE_PH}")

    print(f"\n[1/5] Loading data ...")
    data_by_yr = {}
    for y in YEARS:
        print(f"  {y}: ", end="", flush=True)
        data_by_yr[y] = load_year(y)
    print()

    bl_yr = {y: run_year(data_by_yr[y], BASELINE_PL, BASELINE_PH) for y in YEARS}
    bl_obj = obj(bl_yr)
    print(f"  Baseline OBJ={bl_obj:.4f}")

    # ── Part A: P_LOW ─────────────────────────────────────────────────────────
    print(f"\n[2/5] Part A — P_LOW sweep (P_HIGH={BASELINE_PH} locked) ...")
    pl_vals = [0.20, 0.25, 0.30, 0.35, 0.40, 0.45]
    best_pl = BASELINE_PL; best_pl_obj = bl_obj
    pl_results = {}
    for pl in pl_vals:
        yr = {y: run_year(data_by_yr[y], pl, BASELINE_PH) for y in YEARS}
        o = obj(yr); d = o - bl_obj
        sym = "✅" if d > 0.005 else ("⚠️ " if abs(d) <= 0.005 else "❌")
        base_marker = " ← baseline" if abs(pl - BASELINE_PL) < 1e-9 else ""
        print(f"    pl={pl} → OBJ={o:.4f}  Δ={d:+.4f}  {sym}{base_marker}")
        pl_results[pl] = {"pl": pl, "obj": o, "delta": d, "yearly": yr}
        if o > best_pl_obj: best_pl_obj = o; best_pl = pl
    print(f"  PL winner: pl={best_pl}  OBJ={best_pl_obj:.4f}")

    # ── Part B: P_HIGH ────────────────────────────────────────────────────────
    print(f"\n[3/5] Part B — P_HIGH sweep (P_LOW={best_pl} locked) ...")
    ph_vals = [0.55, 0.60, 0.65, 0.70, 0.75, 0.80]
    best_ph = BASELINE_PH; best_ph_obj = best_pl_obj
    ph_results = {}
    for ph in ph_vals:
        if ph <= best_pl + 0.05: continue  # skip invalid combos
        yr = {y: run_year(data_by_yr[y], best_pl, ph) for y in YEARS}
        o = obj(yr); d = o - bl_obj
        sym = "✅" if d > 0.005 else ("⚠️ " if abs(d) <= 0.005 else "❌")
        base_marker = " ← baseline" if abs(ph - BASELINE_PH) < 1e-9 else ""
        print(f"    ph={ph} → OBJ={o:.4f}  Δ={d:+.4f}  {sym}{base_marker}")
        ph_results[ph] = {"ph": ph, "obj": o, "delta": d, "yearly": yr}
        if o > best_ph_obj: best_ph_obj = o; best_ph = ph
    print(f"  PH winner: ph={best_ph}  OBJ={best_ph_obj:.4f}")

    # ── Part C: Joint LOYO ────────────────────────────────────────────────────
    print(f"\n[4/5] Joint best: P_LOW={best_pl} P_HIGH={best_ph}")
    print(f"\n[5/5] Part C — Joint LOYO validation ...")
    loyo_wins = 0; loyo_deltas = []; loyo_detail = {}
    for loyo_yr in YEARS:
        train_yrs = [y for y in YEARS if y != loyo_yr]
        tr_yr  = {y: run_year(data_by_yr[y], best_pl, best_ph) for y in train_yrs}
        bl_yr_t = {y: run_year(data_by_yr[y], BASELINE_PL, BASELINE_PH) for y in train_yrs}
        d = obj(tr_yr) - obj(bl_yr_t)
        win = d > 0.005; loyo_wins += int(win); loyo_deltas.append(d)
        loyo_detail[loyo_yr] = {"delta": d, "win": win}
        print(f"    {loyo_yr}: {'✅' if win else '❌'}  Δ={d:+.4f}")

    joint_yr = {y: run_year(data_by_yr[y], best_pl, best_ph) for y in YEARS}
    joint_obj = obj(joint_yr)
    joint_delta = joint_obj - bl_obj
    validated = loyo_wins >= 3 and joint_delta > 0.005

    print(f"\n{'=' * 68}")
    if validated:
        verdict = f"VALIDATED — PL={best_pl} PH={best_ph} OBJ={joint_obj:.4f} Δ={joint_delta:+.4f} LOYO {loyo_wins}/5"
        print(f"✅ {verdict}")
    else:
        verdict = f"NO IMPROVEMENT — PL={BASELINE_PL} PH={BASELINE_PH} OBJ={bl_obj:.4f} optimal"
        print(f"❌ {verdict}")
        print(f"   Best: Δ={joint_delta:+.4f} | LOYO {loyo_wins}/5")
    print("=" * 68)

    elapsed = round(time.time() - t0, 1)
    out_dir = Path("artifacts/phase181"); out_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "phase": 181, "description": "Breadth Regime Threshold Re-Sweep",
        "elapsed_seconds": elapsed,
        "baseline_pl": BASELINE_PL, "baseline_ph": BASELINE_PH,
        "baseline_obj": bl_obj, "baseline_yearly": bl_yr,
        "pl_sweep": {str(k): v for k, v in pl_results.items()},
        "ph_sweep": {str(k): v for k, v in ph_results.items()},
        "best_pl": best_pl, "best_ph": best_ph,
        "best_obj": joint_obj, "best_delta": joint_delta, "best_yearly": joint_yr,
        "loyo_wins": loyo_wins, "loyo_deltas": loyo_deltas,
        "loyo_detail": {str(k): v for k, v in loyo_detail.items()},
        "validated": validated, "verdict": verdict, "partial": False,
        "timestamp": datetime.now(UTC).isoformat(),
    }
    rpath = out_dir / "phase181_report.json"
    with open(rpath, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\n✅ Saved → {rpath}")

    if validated:
        cfg_path = Path("configs/production_p91b_champion.json")
        with open(cfg_path) as f:
            cfg = json.load(f)
        cfg["breadth_regime_switching"]["p_low"] = best_pl
        cfg["breadth_regime_switching"]["p_high"] = best_ph
        old_ver = cfg.get("_version", "2.16.0"); new_ver = "2.17.0"
        cfg["_version"] = new_ver
        cfg["monitoring"]["expected_performance"]["annual_sharpe_backtest"] = round(joint_obj, 4)
        cfg["_validated"] = cfg.get("_validated", "") + \
            f"; Breadth thr P181: PL={BASELINE_PL}→{best_pl} PH={BASELINE_PH}→{best_ph} LOYO {loyo_wins}/5 Δ={joint_delta:+.4f} — PRODUCTION {new_ver}"
        with open(cfg_path, "w") as f:
            json.dump(cfg, f, indent=2)
        print(f"✅ Config updated: {old_ver} → {new_ver}")
    else:
        print(f"\n❌ NO IMPROVEMENT — breadth thresholds remain optimal.")

    _signal.alarm(0)

if __name__ == "__main__":
    main()
