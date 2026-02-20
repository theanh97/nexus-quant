"""
PHASE 182 — Regime Weight Re-Optimization (v2.17.0 baseline)
===============================================================
After P181 changed breadth thresholds (PL=0.20, PH=0.60 from 0.35/0.65),
regime distribution shifted. Re-optimize F168 weight per regime (LOW/MID/HIGH).
Other signal weights scaled proportionally.

Baseline: v2.17.0  OBJ≈2.5228  P_LOW=0.20  P_HIGH=0.60
Current regime weights (from P172b):
  LOW:  v1=0.2415 i460=0.1730 i415=0.2855 f168=0.3000
  MID:  v1=0.1493 i460=0.2053 i415=0.3453 f168=0.3000
  HIGH: v1=0.0567 i460=0.2833 i415=0.5100 f168=0.1500

Sweep: F168 weight per regime ∈ [0.10, 0.15, 0.20, 0.25, 0.30, 0.35]
       Other signals scaled proportionally to sum=1.0
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
    out = Path("artifacts/phase182"); out.mkdir(parents=True, exist_ok=True)
    (out / "phase182_report.json").write_text(json.dumps(_partial, indent=2)); sys.exit(0)
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
P_LOW, P_HIGH = 0.20, 0.60   # P181 update
TS_SHORT, TS_LONG = 12, 96
FUND_DISP_PCT = 240

VOL_THRESHOLD = 0.50; VOL_SCALE = 0.40; VOL_F168_BOOST = 0.10
TS_RT = 0.60; TS_RS = 0.40; TS_BT = 0.25; TS_BS = 1.50
DISP_THR = 0.60; DISP_SCALE = 1.0

# Baseline weights (P172b)
BASE_W = {
    "LOW":  {"v1": 0.2415, "i460": 0.1730, "i415": 0.2855, "f168": 0.3000},
    "MID":  {"v1": 0.1493, "i460": 0.2053, "i415": 0.3453, "f168": 0.3000},
    "HIGH": {"v1": 0.0567, "i460": 0.2833, "i415": 0.5100, "f168": 0.1500},
}
BASELINE_OBJ = 2.5228

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

def make_weights(low_f168: float, mid_f168: float, high_f168: float) -> dict:
    """Build weights with given F168 per regime, scale others proportionally."""
    def scale_others(base_w: dict, new_f168: float) -> dict:
        others_sum = sum(v for k, v in base_w.items() if k != "f168")
        target_others = 1.0 - new_f168
        scale = target_others / others_sum if others_sum > 0 else 1.0
        return {k: (new_f168 if k == "f168" else max(0.01, v * scale))
                for k, v in base_w.items()}
    return {
        "LOW":  scale_others(BASE_W["LOW"],  low_f168),
        "MID":  scale_others(BASE_W["MID"],  mid_f168),
        "HIGH": scale_others(BASE_W["HIGH"], high_f168),
    }

def load_year(year: int):
    s, e = YEAR_RANGES[year]
    cfg = {"provider": "binance_rest_v1", "symbols": SYMBOLS,
           "start": s, "end": e, "bar_interval": "1h",
           "cache_dir": ".cache/binance_rest"}
    dataset = make_provider(cfg, seed=42).load()
    n = len(dataset.timeline)
    btc_rets = np.zeros(n)
    for i in range(1, n):
        c0 = dataset.close("BTCUSDT", i - 1); c1 = dataset.close("BTCUSDT", i)
        btc_rets[i] = (c1 / c0 - 1.0) if c0 > 0 else 0.0
    btc_vol = np.full(n, np.nan)
    for i in range(VOL_WINDOW, n):
        btc_vol[i] = float(np.std(btc_rets[i - VOL_WINDOW:i])) * np.sqrt(8760)
    if VOL_WINDOW < n: btc_vol[:VOL_WINDOW] = btc_vol[VOL_WINDOW]
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
    breadth_regime = np.where(brd_pct >= P_HIGH, 2, np.where(brd_pct >= P_LOW, 1, 0)).astype(int)
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
    return sig_rets, btc_vol, breadth_regime, fund_std_pct, ts_spread_pct

def compute_ensemble(sig_rets, btc_vol, breadth_regime, fund_std_pct, ts_spread_pct,
                     weights: dict) -> np.ndarray:
    sk_all = ["v1", "i460", "i415", "f168"]
    min_len = min(len(sig_rets[sk]) for sk in sk_all)
    bv = btc_vol[:min_len]; reg = breadth_regime[:min_len]
    fsp = fund_std_pct[:min_len]; tsp = ts_spread_pct[:min_len]
    ens = np.zeros(min_len)
    for i in range(min_len):
        w = weights[["LOW", "MID", "HIGH"][int(reg[i])]]
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

F168_SWEEP = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35]

def main():
    t0 = time.time()
    print("=" * 68)
    print("PHASE 182 — Regime Weight Re-Opt (v2.17.0, PL=0.20 PH=0.60)")
    print("=" * 68)
    print(f"  Baseline F168: LOW=0.30 MID=0.30 HIGH=0.15")
    print(f"  Sweep F168 per regime ∈ {F168_SWEEP}")

    print(f"\n[1/6] Loading data ...")
    data_by_yr = {}
    for y in YEARS:
        print(f"  {y}: ", end="", flush=True)
        data_by_yr[y] = load_year(y)
    print()

    bl_weights = make_weights(0.30, 0.30, 0.15)
    bl_yr = {y: sharpe(compute_ensemble(*data_by_yr[y], bl_weights)) for y in YEARS}
    bl_obj = obj(bl_yr)
    print(f"  Baseline OBJ={bl_obj:.4f}")

    # ── Part A: LOW regime F168 ───────────────────────────────────────────────
    print(f"\n[2/6] Part A — LOW regime F168 sweep (MID=0.30 HIGH=0.15 locked) ...")
    best_low_f168 = 0.30; best_low_obj = bl_obj
    low_results = {}
    for f in F168_SWEEP:
        w = make_weights(f, 0.30, 0.15)
        yr = {y: sharpe(compute_ensemble(*data_by_yr[y], w)) for y in YEARS}
        o = obj(yr); d = o - bl_obj
        sym = "✅" if d > 0.005 else ("⚠️ " if abs(d) <= 0.005 else "❌")
        base_marker = " ← baseline" if abs(f - 0.30) < 1e-9 else ""
        print(f"    low_f168={f} → OBJ={o:.4f}  Δ={d:+.4f}  {sym}{base_marker}")
        low_results[f] = {"f168": f, "obj": o, "delta": d, "yearly": yr}
        if o > best_low_obj: best_low_obj = o; best_low_f168 = f
    print(f"  LOW winner: f168={best_low_f168}  OBJ={best_low_obj:.4f}")

    # ── Part B: MID regime F168 ───────────────────────────────────────────────
    print(f"\n[3/6] Part B — MID regime F168 sweep (LOW={best_low_f168} HIGH=0.15 locked) ...")
    best_mid_f168 = 0.30; best_mid_obj = best_low_obj
    mid_results = {}
    for f in F168_SWEEP:
        w = make_weights(best_low_f168, f, 0.15)
        yr = {y: sharpe(compute_ensemble(*data_by_yr[y], w)) for y in YEARS}
        o = obj(yr); d = o - bl_obj
        sym = "✅" if d > 0.005 else ("⚠️ " if abs(d) <= 0.005 else "❌")
        base_marker = " ← baseline" if abs(f - 0.30) < 1e-9 else ""
        print(f"    mid_f168={f} → OBJ={o:.4f}  Δ={d:+.4f}  {sym}{base_marker}")
        mid_results[f] = {"f168": f, "obj": o, "delta": d, "yearly": yr}
        if o > best_mid_obj: best_mid_obj = o; best_mid_f168 = f
    print(f"  MID winner: f168={best_mid_f168}  OBJ={best_mid_obj:.4f}")

    # ── Part C: HIGH regime F168 ──────────────────────────────────────────────
    print(f"\n[4/6] Part C — HIGH regime F168 sweep (LOW={best_low_f168} MID={best_mid_f168} locked) ...")
    best_high_f168 = 0.15; best_high_obj = best_mid_obj
    high_results = {}
    for f in F168_SWEEP:
        w = make_weights(best_low_f168, best_mid_f168, f)
        yr = {y: sharpe(compute_ensemble(*data_by_yr[y], w)) for y in YEARS}
        o = obj(yr); d = o - bl_obj
        sym = "✅" if d > 0.005 else ("⚠️ " if abs(d) <= 0.005 else "❌")
        base_marker = " ← baseline" if abs(f - 0.15) < 1e-9 else ""
        print(f"    high_f168={f} → OBJ={o:.4f}  Δ={d:+.4f}  {sym}{base_marker}")
        high_results[f] = {"f168": f, "obj": o, "delta": d, "yearly": yr}
        if o > best_high_obj: best_high_obj = o; best_high_f168 = f
    print(f"  HIGH winner: f168={best_high_f168}  OBJ={best_high_obj:.4f}")

    # ── Joint LOYO ────────────────────────────────────────────────────────────
    best_w = make_weights(best_low_f168, best_mid_f168, best_high_f168)
    print(f"\n[5/6] Joint best: LOW={best_low_f168} MID={best_mid_f168} HIGH={best_high_f168}")
    print(f"\n[6/6] Joint LOYO validation ...")
    loyo_wins = 0; loyo_deltas = []; loyo_detail = {}
    for loyo_yr in YEARS:
        train_yrs = [y for y in YEARS if y != loyo_yr]
        tr_yr   = {y: sharpe(compute_ensemble(*data_by_yr[y], best_w)) for y in train_yrs}
        bl_yr_t = {y: sharpe(compute_ensemble(*data_by_yr[y], bl_weights)) for y in train_yrs}
        d = obj(tr_yr) - obj(bl_yr_t)
        win = d > 0.005; loyo_wins += int(win); loyo_deltas.append(d)
        loyo_detail[loyo_yr] = {"delta": d, "win": win}
        print(f"    {loyo_yr}: {'✅' if win else '❌'}  Δ={d:+.4f}")

    joint_yr = {y: sharpe(compute_ensemble(*data_by_yr[y], best_w)) for y in YEARS}
    joint_obj = obj(joint_yr)
    joint_delta = joint_obj - bl_obj
    validated = loyo_wins >= 3 and joint_delta > 0.005

    print(f"\n{'=' * 68}")
    if validated:
        verdict = (f"VALIDATED — LOW_f168={best_low_f168} MID_f168={best_mid_f168} "
                   f"HIGH_f168={best_high_f168} OBJ={joint_obj:.4f} Δ={joint_delta:+.4f} LOYO {loyo_wins}/5")
        print(f"✅ {verdict}")
    else:
        verdict = f"NO IMPROVEMENT — P172b weights remain optimal OBJ={bl_obj:.4f}"
        print(f"❌ {verdict}")
        print(f"   Best: Δ={joint_delta:+.4f} | LOYO {loyo_wins}/5")
    print("=" * 68)

    elapsed = round(time.time() - t0, 1)
    out_dir = Path("artifacts/phase182"); out_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "phase": 182, "description": "Regime Weight Re-Opt v2.17.0",
        "elapsed_seconds": elapsed,
        "baseline_weights": {r: dict(BASE_W[r]) for r in BASE_W},
        "baseline_obj": bl_obj, "baseline_yearly": bl_yr,
        "low_sweep": {str(k): v for k, v in low_results.items()},
        "mid_sweep": {str(k): v for k, v in mid_results.items()},
        "high_sweep": {str(k): v for k, v in high_results.items()},
        "best_low_f168": best_low_f168,
        "best_mid_f168": best_mid_f168,
        "best_high_f168": best_high_f168,
        "best_weights": best_w,
        "best_obj": joint_obj, "best_delta": joint_delta, "best_yearly": joint_yr,
        "loyo_wins": loyo_wins, "loyo_deltas": loyo_deltas,
        "loyo_detail": {str(k): v for k, v in loyo_detail.items()},
        "validated": validated, "verdict": verdict, "partial": False,
        "timestamp": datetime.now(UTC).isoformat(),
    }
    rpath = out_dir / "phase182_report.json"
    with open(rpath, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\n✅ Saved → {rpath}")

    if validated:
        cfg_path = Path("configs/production_p91b_champion.json")
        with open(cfg_path) as f:
            cfg = json.load(f)
        # Update ensemble signal weights per regime
        ens_sigs = cfg.get("ensemble", {}).get("signals", {})
        # Map: LOW/MID/HIGH regime index
        for regime_name, w in best_w.items():
            for sig_key_map, cfg_sig_key in [
                ("v1", "nexus_alpha_v1"), ("i460", "idio_momentum_alpha_i460bw120"),
                ("i415", "idio_momentum_alpha_i415bw144"), ("f168", "funding_momentum_alpha_f168"),
            ]:
                if cfg_sig_key in ens_sigs:
                    if "weights" not in ens_sigs[cfg_sig_key]:
                        ens_sigs[cfg_sig_key]["weights"] = {}
                    ens_sigs[cfg_sig_key]["weights"][regime_name] = round(w[sig_key_map], 6)
        old_ver = cfg.get("_version", "2.17.0"); new_ver = "2.18.0"
        cfg["_version"] = new_ver
        cfg["monitoring"]["expected_performance"]["annual_sharpe_backtest"] = round(joint_obj, 4)
        cfg["_validated"] = cfg.get("_validated", "") + \
            (f"; Regime weight re-opt P182: LOW_f168={best_low_f168} MID_f168={best_mid_f168} "
             f"HIGH_f168={best_high_f168} LOYO {loyo_wins}/5 Δ={joint_delta:+.4f} — PRODUCTION {new_ver}")
        with open(cfg_path, "w") as f:
            json.dump(cfg, f, indent=2)
        print(f"✅ Config updated: {old_ver} → {new_ver}")
    else:
        print(f"\n❌ NO IMPROVEMENT — P172b weights remain optimal with new thresholds.")

    _signal.alarm(0)

if __name__ == "__main__":
    main()
