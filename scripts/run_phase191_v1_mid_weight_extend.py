"""
Phase 191 — V1 MID Regime Weight Extension
==========================================
Motivation: P190 showed MID v1 weight still increasing at 0.35 (ceiling).
Extend MID sweep beyond 0.35 to find true peak.

Sweep:
  MID v1 weight ∈ [0.35, 0.37, 0.40, 0.42, 0.45, 0.48, 0.50]
  LOW fixed at 0.48 (P190 validated), HIGH fixed at 0.06

Baseline: v2.22.0, OBJ=2.8660 (LOW=0.48, MID=0.35, HIGH=0.06)
Validation: LOYO ≥3/5 AND Δ>0.005
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
    _partial["partial"] = True
    _partial["timeout_at"] = datetime.now(UTC).isoformat()
    out = Path("artifacts/phase191"); out.mkdir(parents=True, exist_ok=True)
    (out / "phase191_report.json").write_text(json.dumps(_partial, indent=2))
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
YEARS = sorted(YEAR_RANGES)

VOL_WINDOW    = 168
BRD_LB        = 192
PCT_WINDOW    = 336
P_LOW, P_HIGH = 0.20, 0.60
TS_SHORT, TS_LONG = 12, 96
FUND_DISP_PCT = 240

VOL_THRESHOLD = 0.50; VOL_SCALE = 0.40; VOL_F168_BOOST = 0.10
TS_RT = 0.60; TS_RS = 0.40; TS_BT = 0.25; TS_BS = 1.50
DISP_THR = 0.60; DISP_SCALE = 1.0

# v2.22.0 baseline (P190 validated)
BASE_WEIGHTS = {
    "LOW":  {"v1": 0.48,   "i460": 0.0642, "i415": 0.1058, "f168": 0.35},
    "MID":  {"v1": 0.35,   "i460": 0.1119, "i415": 0.1881, "f168": 0.35},
    "HIGH": {"v1": 0.0600, "i460": 0.3000, "i415": 0.5400, "f168": 0.10},
}
F168_WEIGHTS = {"LOW": 0.35, "MID": 0.35, "HIGH": 0.10}

BASELINE_OBJ = 2.8660
MID_SWEEP    = [0.35, 0.37, 0.40, 0.42, 0.45, 0.48, 0.50]

COST_MODEL = cost_model_from_config({
    "fee_rate": 0.0005, "slippage_rate": 0.0003, "cost_multiplier": 1.0,
})

def make_mid_weights(new_v1: float) -> dict | None:
    f168 = F168_WEIGHTS["MID"]
    base = BASE_WEIGHTS["MID"]
    others_sum = base["i460"] + base["i415"]
    target_others = 1.0 - new_v1 - f168
    if target_others < 0.08 or others_sum <= 0:
        return None
    scale = target_others / others_sum
    return {
        "v1":   round(new_v1, 4),
        "i460": round(max(0.04, base["i460"] * scale), 4),
        "i415": round(max(0.04, base["i415"] * scale), 4),
        "f168": f168,
    }

def rolling_mean_arr(a: np.ndarray, w: int) -> np.ndarray:
    out = np.full(len(a), np.nan)
    if w <= 0 or len(a) == 0: return out
    cs = np.cumsum(np.where(np.isnan(a), 0.0, a))
    for i in range(w - 1, len(a)):
        out[i] = cs[i] / w if i == w - 1 else (cs[i] - cs[i - w]) / w
    return out

def load_year_data(year: int):
    s, e = YEAR_RANGES[year]
    cfg = {"provider": "binance_rest_v1", "symbols": SYMBOLS,
           "start": s, "end": e, "bar_interval": "1h",
           "cache_dir": ".cache/binance_rest"}
    dataset = make_provider(cfg, seed=42).load()
    n = len(dataset.timeline)

    close_mat = np.zeros((n, len(SYMBOLS)))
    for j, sym in enumerate(SYMBOLS):
        for i in range(n):
            close_mat[i, j] = dataset.close(sym, i)

    btc_rets = np.zeros(n)
    for i in range(1, n):
        c0 = close_mat[i-1, 0]; c1 = close_mat[i, 0]
        btc_rets[i] = (c1 / c0 - 1.0) if c0 > 0 else 0.0
    btc_vol = np.full(n, np.nan)
    for i in range(VOL_WINDOW, n):
        btc_vol[i] = float(np.std(btc_rets[i - VOL_WINDOW:i])) * np.sqrt(8760)
    if VOL_WINDOW < n: btc_vol[:VOL_WINDOW] = btc_vol[VOL_WINDOW]

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

    breadth = np.full(n, 0.5)
    for i in range(BRD_LB, n):
        pos = sum(1 for j in range(len(SYMBOLS))
                  if close_mat[i - BRD_LB, j] > 0 and close_mat[i, j] > close_mat[i - BRD_LB, j])
        breadth[i] = pos / len(SYMBOLS)
    breadth[:BRD_LB] = 0.5
    brd_pct = np.full(n, 0.5)
    for i in range(PCT_WINDOW, n):
        brd_pct[i] = float(np.mean(breadth[i - PCT_WINDOW:i] <= breadth[i]))
    brd_pct[:PCT_WINDOW] = 0.5
    regime = np.where(brd_pct >= P_HIGH, 2, np.where(brd_pct >= P_LOW, 1, 0)).astype(int)

    sig_rets = {}
    for sk, sname, params in [
        ("v1", "nexus_alpha_v1", {
            "k_per_side": 2, "w_carry": 0.35, "w_mom": 0.40, "w_mean_reversion": 0.25,
            "momentum_lookback_bars": 336, "mean_reversion_lookback_bars": 84,
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
    return btc_vol, fund_std_pct, ts_spread_pct, regime, sig_rets, n

def compute_ensemble(base_data, weights: dict) -> np.ndarray:
    btc_vol, fund_std_pct, ts_spread_pct, regime, sig_rets, n = base_data
    min_len = min(len(v) for v in sig_rets.values())
    bv = btc_vol[:min_len]; reg = regime[:min_len]
    fsp = fund_std_pct[:min_len]; tsp = ts_spread_pct[:min_len]
    sk_all = ["v1", "i460", "i415", "f168"]

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
        if tsp[i] > TS_RT:    ret_i *= TS_RS
        elif tsp[i] < TS_BT:  ret_i *= TS_BS
        ens[i] = ret_i
    return ens

def sharpe(r: np.ndarray) -> float:
    r = r[~np.isnan(r)]
    if len(r) < 2: return 0.0
    return float(np.mean(r) / np.std(r, ddof=1) * np.sqrt(8760))

def obj(yearly: dict) -> float:
    vals = list(yearly.values())
    return float(np.mean(vals) - 0.5 * np.std(vals, ddof=1))

def main():
    t0 = time.time()
    print("=" * 65)
    print("Phase 191 — V1 MID Regime Weight Extension")
    print(f"Baseline: v2.22.0  OBJ={BASELINE_OBJ}")
    print(f"  LOW_v1=0.48 (fixed), HIGH_v1=0.06 (fixed)")
    print(f"  MID_v1=0.35 (was still increasing at P190 ceiling)")
    print(f"Sweep:    MID v1 ∈ {MID_SWEEP}")
    print("=" * 65)

    print("\n[1/3] Loading year data ...")
    data_by_yr: dict = {}
    for yr in YEARS:
        print(f"  {yr}: ", end="", flush=True)
        data_by_yr[yr] = load_year_data(yr)
    print()

    bl_yr = {yr: sharpe(compute_ensemble(data_by_yr[yr], BASE_WEIGHTS)) for yr in YEARS}
    bl_obj = obj(bl_yr)
    print(f"  Baseline OBJ = {bl_obj:.4f}  (expected ≈ {BASELINE_OBJ})")

    print(f"\n[2/3] MID v1 weight extension ...")
    sweep_results: dict = {}
    best_mid_v1, best_mid_obj = 0.35, bl_obj
    best_mid_yr = bl_yr

    for v1w in MID_SWEEP:
        new_mid_w = make_mid_weights(v1w)
        if new_mid_w is None:
            print(f"  MID v1={v1w:.2f} → INFEASIBLE"); continue
        weights = {"LOW": BASE_WEIGHTS["LOW"], "MID": new_mid_w, "HIGH": BASE_WEIGHTS["HIGH"]}
        yr_s = {yr: sharpe(compute_ensemble(data_by_yr[yr], weights)) for yr in YEARS}
        o = obj(yr_s); d = o - bl_obj
        loyo_wins = sum(1 for yr in YEARS if (yr_s[yr] - bl_yr[yr]) > 0)
        is_base = abs(v1w - 0.35) < 1e-4
        flag = ("⭐" if (o > best_mid_obj and loyo_wins >= 2) else
                ("⚠️ " if is_base else ("+" if d > 0 else "")))
        print(f"  MID v1={v1w:.2f} i460={new_mid_w['i460']:.4f} i415={new_mid_w['i415']:.4f} "
              f"→ OBJ={o:.4f}  Δ={d:+.4f}  {flag}")
        sweep_results[v1w] = {"obj": o, "delta": d, "loyo_wins": loyo_wins,
                               "weights": new_mid_w, "yr_sharpes": yr_s}
        if o > best_mid_obj and loyo_wins >= 2:
            best_mid_v1, best_mid_obj, best_mid_yr = v1w, o, yr_s
    print(f"  → Best MID v1: {best_mid_v1:.4f}  OBJ={best_mid_obj:.4f}")

    print(f"\n[3/3] LOYO validation — best_mid_v1={best_mid_v1}  OBJ={best_mid_obj:.4f}")
    if best_mid_v1 == 0.35:
        print("  ✔  CONFIRMED — MID v1=0.35 already optimal")
        loyo_wins_final, loyo_avg_d, validated = 0, 0.0, False
    else:
        loyo_wins_final = 0; loyo_deltas = []
        for loyo_yr in YEARS:
            train_yrs = [y for y in YEARS if y != loyo_yr]
            d = (obj({y: best_mid_yr[y] for y in train_yrs}) -
                 obj({y: bl_yr[y] for y in train_yrs}))
            win = d > 0.005
            loyo_wins_final += int(win); loyo_deltas.append(d)
            print(f"  {loyo_yr} left out: {'✅' if win else '❌'}  Δ={d:+.4f}")
        loyo_avg_d = float(np.mean(loyo_deltas))
        validated = loyo_wins_final >= 3 and (best_mid_obj - bl_obj) > 0.005
        print(f"  LOYO {loyo_wins_final}/5  avg_Δ={loyo_avg_d:+.4f}  Δ_OBJ={best_mid_obj - bl_obj:+.4f}")

    if validated:
        print(f"\n✅ VALIDATED — MID_v1={best_mid_v1}  OBJ={best_mid_obj:.4f}")
    else:
        print(f"\n ✔  CONFIRMED — MID_v1=0.35 remains optimal (v2.22.0)")

    report = {
        "phase": 191, "best_mid_v1": best_mid_v1,
        "sweep": {str(k): {kk: vv for kk, vv in v.items() if kk != "yr_sharpes"}
                  for k, v in sweep_results.items()},
        "baseline": {"obj": bl_obj, "mid_v1": 0.35},
        "best": {"obj": best_mid_obj, "delta": best_mid_obj - bl_obj},
        "loyo": {"wins": loyo_wins_final if best_mid_v1 != 0.35 else 0,
                 "avg_delta": loyo_avg_d if best_mid_v1 != 0.35 else 0.0},
        "validated": validated, "runtime_s": round(time.time() - t0, 1),
    }
    out_dir = ROOT / "artifacts" / "phase191"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "phase191_report.json").write_text(json.dumps(report, indent=2))
    print(f"\nReport → artifacts/phase191/phase191_report.json")
    print(f"Runtime: {report['runtime_s']:.0f}s")

    if validated:
        best_mid_weights = sweep_results[best_mid_v1]["weights"]
        cfg_path = ROOT / "configs" / "production_p91b_champion.json"
        cfg = json.loads(cfg_path.read_text())
        rw = cfg["breadth_regime_switching"]["regime_weights"]
        rw["MID"]["v1"]        = round(best_mid_weights["v1"],   4)
        rw["MID"]["i460bw168"] = round(best_mid_weights["i460"], 4)
        rw["MID"]["i415bw216"] = round(best_mid_weights["i415"], 4)
        rw["MID"]["f168"]      = round(best_mid_weights["f168"],  4)
        old_v = cfg["_version"]; parts = old_v.split(".")
        new_v = f"{parts[0]}.{int(parts[1]) + 1}.0" if len(parts) >= 2 else old_v
        cfg["_version"] = new_v
        cfg["_validated"] += (
            f"; V1 MID weight extend P191: MID_v1={best_mid_v1:.4f} "
            f"LOYO {loyo_wins_final}/5 Δ=+{best_mid_obj - bl_obj:.4f} OBJ={best_mid_obj:.4f} — PRODUCTION {new_v}"
        )
        cfg["monitoring"]["expected_performance"]["annual_sharpe_backtest"] = round(best_mid_obj, 4)
        cfg_path.write_text(json.dumps(cfg, indent=2, ensure_ascii=False))
        print(f"\nConfig updated: v{old_v} → v{new_v}  OBJ={best_mid_obj:.4f}")

if __name__ == "__main__":
    main()
