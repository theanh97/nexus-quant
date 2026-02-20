"""
Phase 187 — V1 Mean-Reversion Lookback Re-Sweep
================================================
Motivation: P185 increased V1 w_mean_reversion from 0.20→0.25.
Higher MR weight may benefit from a different lookback than 72h.

Sweep:
  mean_reversion_lookback_bars ∈ [24, 36, 48, 60, 72, 84, 96, 120, 144, 168]

Baseline: v2.19.0, OBJ=2.5945 (V1: wc=0.35, wm=0.40, wr=0.25, mr_lb=72)
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
    out = Path("artifacts/phase187"); out.mkdir(parents=True, exist_ok=True)
    (out / "phase187_report.json").write_text(json.dumps(_partial, indent=2))
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

WEIGHTS = {
    "LOW":  {"v1": 0.2242, "i460": 0.1607, "i415": 0.2651, "f168": 0.35},
    "MID":  {"v1": 0.1387, "i460": 0.1906, "i415": 0.3207, "f168": 0.35},
    "HIGH": {"v1": 0.0600, "i460": 0.3000, "i415": 0.5400, "f168": 0.10},
}

V1_W_CARRY = 0.35; V1_W_MOM = 0.40; V1_W_MR = 0.25
V1_MOM_LB  = 336
V1_MR_LB_BASE = 72

BASELINE_OBJ = 2.5945
SWEEP_LBS    = [24, 36, 48, 60, 72, 84, 96, 120, 144, 168]

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
        ("v1_base", "nexus_alpha_v1", {
            "k_per_side": 2, "w_carry": V1_W_CARRY, "w_mom": V1_W_MOM,
            "w_mean_reversion": V1_W_MR, "momentum_lookback_bars": V1_MOM_LB,
            "mean_reversion_lookback_bars": V1_MR_LB_BASE, "vol_lookback_bars": 168,
            "target_gross_leverage": 0.35, "rebalance_interval_bars": 60,
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
    return dataset, btc_vol, fund_std_pct, ts_spread_pct, regime, sig_rets, n

def run_v1_mr_lb(base_data, mr_lb: int) -> np.ndarray:
    dataset, btc_vol, fund_std_pct, ts_spread_pct, regime, sig_rets, n = base_data
    if mr_lb == V1_MR_LB_BASE:
        v1_rets = sig_rets["v1_base"]
    else:
        result = BacktestEngine(BacktestConfig(costs=COST_MODEL)).run(
            dataset,
            make_strategy({"name": "nexus_alpha_v1", "params": {
                "k_per_side": 2, "w_carry": V1_W_CARRY, "w_mom": V1_W_MOM,
                "w_mean_reversion": V1_W_MR,
                "momentum_lookback_bars": V1_MOM_LB,
                "mean_reversion_lookback_bars": mr_lb,
                "vol_lookback_bars": 168, "target_gross_leverage": 0.35,
                "rebalance_interval_bars": 60,
            }}))
        v1_rets = np.array(result.returns)

    all_rets = {"v1": v1_rets, "i460": sig_rets["i460"],
                "i415": sig_rets["i415"], "f168": sig_rets["f168"]}
    min_len = min(len(v) for v in all_rets.values())
    bv = btc_vol[:min_len]; reg = regime[:min_len]
    fsp = fund_std_pct[:min_len]; tsp = ts_spread_pct[:min_len]
    sk_all = ["v1", "i460", "i415", "f168"]

    ens = np.zeros(min_len)
    for i in range(min_len):
        w = WEIGHTS[["LOW", "MID", "HIGH"][int(reg[i])]]
        if not np.isnan(bv[i]) and bv[i] > VOL_THRESHOLD:
            boost_per = VOL_F168_BOOST / max(1, len(sk_all) - 1)
            ret_i = sum(
                (min(0.60, w[sk] + VOL_F168_BOOST) if sk == "f168"
                 else max(0.05, w[sk] - boost_per)) * all_rets[sk][i]
                for sk in sk_all)
            ret_i *= VOL_SCALE
        else:
            ret_i = sum(w[sk] * all_rets[sk][i] for sk in sk_all)
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
    print("Phase 187 — V1 Mean-Reversion Lookback Re-Sweep")
    print(f"Baseline: v2.19.0  OBJ={BASELINE_OBJ}  mr_lb={V1_MR_LB_BASE}")
    print(f"Sweep:    mean_reversion_lookback_bars ∈ {SWEEP_LBS}")
    print("=" * 65)

    print("\n[1/3] Loading year data ...")
    data_by_yr: dict = {}
    for yr in YEARS:
        print(f"  {yr}: ", end="", flush=True)
        data_by_yr[yr] = load_year_data(yr)
    print()

    bl_yr = {yr: sharpe(run_v1_mr_lb(data_by_yr[yr], V1_MR_LB_BASE)) for yr in YEARS}
    bl_obj = obj(bl_yr)
    print(f"  Baseline OBJ = {bl_obj:.4f}  (expected ≈ {BASELINE_OBJ})")

    print(f"\n[2/3] Sweeping mean_reversion_lookback_bars ...")
    sweep_results: dict = {}
    best_lb, best_obj_val = V1_MR_LB_BASE, bl_obj

    for lb in SWEEP_LBS:
        yr_sharpes = {yr: sharpe(run_v1_mr_lb(data_by_yr[yr], lb)) for yr in YEARS}
        o = obj(yr_sharpes); d = o - bl_obj
        loyo_wins = sum(1 for yr in YEARS if (yr_sharpes[yr] - bl_yr[yr]) > 0)
        flag = ("⭐" if (o > best_obj_val and loyo_wins >= 3 and d > 0.005) else
                ("✅" if (loyo_wins >= 3 and d > 0.005) else
                ("⚠️ " if lb == V1_MR_LB_BASE else "❌")))
        base_marker = " ← baseline" if lb == V1_MR_LB_BASE else ""
        print(f"  lb={lb:3d} → OBJ={o:.4f}  Δ={d:+.4f}  LOYO {loyo_wins}/5  {flag}{base_marker}")
        sweep_results[lb] = {"obj": o, "delta": d, "loyo_wins": loyo_wins}
        if o > best_obj_val and loyo_wins >= 3 and d > 0.005:
            best_lb, best_obj_val = lb, o

    print(f"\n[3/3] LOYO validation — best_lb={best_lb}  OBJ={best_obj_val:.4f}")
    if best_lb == V1_MR_LB_BASE:
        print("  ❌ NO IMPROVEMENT — mr_lb=72 already optimal")
        loyo_wins_final, loyo_avg_d, validated = 0, 0.0, False
    else:
        loyo_wins_final = 0; loyo_deltas = []
        # Need to store yr_sharpes for LOYO — re-compute best
        best_yr_sharpes = {yr: sharpe(run_v1_mr_lb(data_by_yr[yr], best_lb)) for yr in YEARS}
        for loyo_yr in YEARS:
            train_yrs = [y for y in YEARS if y != loyo_yr]
            best_train = {y: best_yr_sharpes[y] for y in train_yrs}
            base_train = {y: bl_yr[y] for y in train_yrs}
            d = obj(best_train) - obj(base_train)
            win = d > 0.005
            loyo_wins_final += int(win); loyo_deltas.append(d)
            print(f"  {loyo_yr} left out: {'✅' if win else '❌'}  Δ={d:+.4f}")
        loyo_avg_d = float(np.mean(loyo_deltas))
        validated = loyo_wins_final >= 3 and (best_obj_val - bl_obj) > 0.005
        print(f"  LOYO {loyo_wins_final}/5  avg_Δ={loyo_avg_d:+.4f}  Δ_OBJ={best_obj_val - bl_obj:+.4f}")

    if validated:
        print(f"\n✅ VALIDATED — mean_reversion_lookback_bars={best_lb}  OBJ={best_obj_val:.4f}")
    else:
        print(f"\n❌ NO IMPROVEMENT — mean_reversion_lookback_bars={V1_MR_LB_BASE}  OBJ={bl_obj:.4f} optimal")

    report = {
        "phase": 187, "target": "v1_mean_reversion_lookback_bars",
        "sweep": sweep_results,
        "baseline": {"obj": bl_obj, "lb": V1_MR_LB_BASE},
        "best":     {"obj": best_obj_val, "lb": best_lb},
        "loyo":     {"wins": loyo_wins_final if best_lb != V1_MR_LB_BASE else 0,
                     "avg_delta": loyo_avg_d if best_lb != V1_MR_LB_BASE else 0.0},
        "validated": validated, "runtime_s": round(time.time() - t0, 1),
    }
    out_dir = ROOT / "artifacts" / "phase187"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "phase187_report.json").write_text(json.dumps(report, indent=2))
    print(f"\nReport → artifacts/phase187/phase187_report.json")
    print(f"Runtime: {report['runtime_s']:.0f}s")

    if validated:
        cfg_path = ROOT / "configs" / "production_p91b_champion.json"
        cfg = json.loads(cfg_path.read_text())
        cfg["ensemble"]["signals"]["v1"]["params"]["mean_reversion_lookback_bars"] = best_lb
        old_v = cfg["_version"]; parts = old_v.split(".")
        new_v = f"{parts[0]}.{int(parts[1]) + 1}.0" if len(parts) >= 2 else old_v
        cfg["_version"] = new_v
        cfg["_validated"] += (
            f"; V1 mr_lb re-sweep P187: {V1_MR_LB_BASE}→{best_lb} LOYO {loyo_wins_final}/5 "
            f"Δ=+{best_obj_val - bl_obj:.4f} OBJ={best_obj_val:.4f} — PRODUCTION {new_v}"
        )
        cfg["monitoring"]["expected_performance"]["annual_sharpe_backtest"] = round(best_obj_val, 4)
        cfg_path.write_text(json.dumps(cfg, indent=2, ensure_ascii=False))
        print(f"\nConfig updated: v{old_v} → v{new_v}  OBJ={best_obj_val:.4f}")

if __name__ == "__main__":
    main()
