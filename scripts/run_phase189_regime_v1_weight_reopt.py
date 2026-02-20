"""
Phase 189 — Regime V1 Weight Re-Optimization (post P185+P187)
==============================================================
Motivation: P185 (V1 blend) + P187 (V1 mr_lb) significantly improved V1 quality.
With a better V1 signal, its optimal ensemble weight may have increased.
Re-sweep V1 weight per regime, keeping F168 at 0.35/0.35/0.10 and scaling others.

Approach:
  For each regime (LOW/MID/HIGH), sweep V1 weight ∈ [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40]
  Scale i460/i415 proportionally to fill remaining non-F168 space.

  Independent 1D sweeps per regime (then joint LOYO on best combo).

Baseline: v2.20.0, OBJ=2.7103
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
    out = Path("artifacts/phase189"); out.mkdir(parents=True, exist_ok=True)
    (out / "phase189_report.json").write_text(json.dumps(_partial, indent=2))
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

# v2.20.0 baseline weights (P182 + P187 improvements)
BASE_WEIGHTS = {
    "LOW":  {"v1": 0.2242, "i460": 0.1607, "i415": 0.2651, "f168": 0.35},
    "MID":  {"v1": 0.1387, "i460": 0.1906, "i415": 0.3207, "f168": 0.35},
    "HIGH": {"v1": 0.0600, "i460": 0.3000, "i415": 0.5400, "f168": 0.10},
}
F168_WEIGHTS = {"LOW": 0.35, "MID": 0.35, "HIGH": 0.10}  # fixed (P182 validated)

BASELINE_OBJ = 2.7103
V1_SWEEP     = [0.10, 0.15, 0.20, 0.2242, 0.25, 0.30, 0.35, 0.40]

COST_MODEL = cost_model_from_config({
    "fee_rate": 0.0005, "slippage_rate": 0.0003, "cost_multiplier": 1.0,
})

def make_weights_v1(regime: str, new_v1: float) -> dict:
    """Scale i460/i415 proportionally when V1 changes, keeping F168 fixed."""
    f168 = F168_WEIGHTS[regime]
    base = BASE_WEIGHTS[regime]
    others_sum = base["i460"] + base["i415"]   # sum of i460+i415 at baseline
    target_others = 1.0 - new_v1 - f168        # remaining for i460+i415
    if target_others <= 0.05 or others_sum <= 0:
        return None  # infeasible
    scale = target_others / others_sum
    return {
        "v1":   new_v1,
        "i460": max(0.05, base["i460"] * scale),
        "i415": max(0.05, base["i415"] * scale),
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

    # Run all signals (v1 with P187 params)
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
    all_rets = sig_rets
    min_len = min(len(v) for v in all_rets.values())
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
    print("Phase 189 — Regime V1 Weight Re-Optimization")
    print(f"Baseline: v2.20.0  OBJ={BASELINE_OBJ}")
    print(f"Sweep:    V1 weight per regime ∈ {V1_SWEEP}")
    print(f"Fixed:    F168={F168_WEIGHTS}")
    print("=" * 65)

    print("\n[1/4] Loading year data ...")
    data_by_yr: dict = {}
    for yr in YEARS:
        print(f"  {yr}: ", end="", flush=True)
        data_by_yr[yr] = load_year_data(yr)
    print()

    bl_yr = {yr: sharpe(compute_ensemble(data_by_yr[yr], BASE_WEIGHTS)) for yr in YEARS}
    bl_obj = obj(bl_yr)
    print(f"  Baseline OBJ = {bl_obj:.4f}  (expected ≈ {BASELINE_OBJ})")

    # ── Per-regime independent sweeps ─────────────────────────────────────────
    best_v1_per_regime = {r: BASE_WEIGHTS[r]["v1"] for r in ["LOW", "MID", "HIGH"]}
    regime_sweep_results: dict = {"LOW": {}, "MID": {}, "HIGH": {}}

    for regime_name in ["LOW", "MID", "HIGH"]:
        print(f"\n[2/4] Sweeping V1 weight in {regime_name} regime ...")
        best_v1, best_v1_obj = BASE_WEIGHTS[regime_name]["v1"], bl_obj
        base_v1 = BASE_WEIGHTS[regime_name]["v1"]

        for v1w in V1_SWEEP:
            new_w = make_weights_v1(regime_name, v1w)
            if new_w is None:
                print(f"  {regime_name} v1={v1w:.4f} → INFEASIBLE (skip)")
                continue
            # Use new weight for this regime, baseline for others
            weights = {
                r: (new_w if r == regime_name else BASE_WEIGHTS[r])
                for r in ["LOW", "MID", "HIGH"]
            }
            yr_sharpes = {yr: sharpe(compute_ensemble(data_by_yr[yr], weights)) for yr in YEARS}
            o = obj(yr_sharpes); d = o - bl_obj
            loyo_wins = sum(1 for yr in YEARS if (yr_sharpes[yr] - bl_yr[yr]) > 0)
            is_base = abs(v1w - base_v1) < 1e-4
            flag = ("⭐" if (o > best_v1_obj and loyo_wins >= 2 and d > 0.002) else
                    ("⚠️ " if is_base else ("+" if d > 0 else "")))
            print(f"  {regime_name} v1={v1w:.4f} i460={new_w['i460']:.4f} i415={new_w['i415']:.4f} "
                  f"→ OBJ={o:.4f}  Δ={d:+.4f}  {flag}")
            regime_sweep_results[regime_name][v1w] = {
                "obj": o, "delta": d, "loyo_wins": loyo_wins, "weights": new_w,
                "yr_sharpes": yr_sharpes,
            }
            if o > best_v1_obj and loyo_wins >= 2 and d > 0.002:
                best_v1, best_v1_obj = v1w, o

        best_v1_per_regime[regime_name] = best_v1
        print(f"  → Best V1 for {regime_name}: {best_v1:.4f}  OBJ={best_v1_obj:.4f}")

    # ── Build joint best weights ───────────────────────────────────────────────
    print(f"\n[3/4] Joint LOYO validation ...")
    best_weights = {}
    for r in ["LOW", "MID", "HIGH"]:
        w = make_weights_v1(r, best_v1_per_regime[r])
        best_weights[r] = w if w else BASE_WEIGHTS[r]

    joint_yr = {yr: sharpe(compute_ensemble(data_by_yr[yr], best_weights)) for yr in YEARS}
    joint_obj = obj(joint_yr); joint_delta = joint_obj - bl_obj
    print(f"  Joint OBJ={joint_obj:.4f}  Δ={joint_delta:+.4f}")
    print(f"  Best V1 per regime: LOW={best_v1_per_regime['LOW']:.4f}  "
          f"MID={best_v1_per_regime['MID']:.4f}  HIGH={best_v1_per_regime['HIGH']:.4f}")

    loyo_wins_final = 0; loyo_deltas = []
    for loyo_yr in YEARS:
        train_yrs = [y for y in YEARS if y != loyo_yr]
        best_train = {y: joint_yr[y] for y in train_yrs}
        base_train = {y: bl_yr[y] for y in train_yrs}
        d = obj(best_train) - obj(base_train)
        win = d > 0.005
        loyo_wins_final += int(win); loyo_deltas.append(d)
        print(f"  {loyo_yr} left out: {'✅' if win else '❌'}  Δ={d:+.4f}")
    loyo_avg_d = float(np.mean(loyo_deltas))
    validated = loyo_wins_final >= 3 and joint_delta > 0.005
    print(f"  LOYO {loyo_wins_final}/5  avg_Δ={loyo_avg_d:+.4f}")

    if validated:
        print(f"\n✅ VALIDATED — OBJ={joint_obj:.4f}  Δ={joint_delta:+.4f}")
    else:
        print(f"\n❌ NO IMPROVEMENT — baseline v1 weights optimal")

    report = {
        "phase": 189, "target": "regime_v1_weights",
        "per_regime_best": best_v1_per_regime,
        "best_weights": best_weights,
        "baseline": {"obj": bl_obj, "weights": BASE_WEIGHTS},
        "best": {"obj": joint_obj, "delta": joint_delta},
        "loyo": {"wins": loyo_wins_final, "avg_delta": loyo_avg_d},
        "validated": validated, "runtime_s": round(time.time() - t0, 1),
    }
    out_dir = ROOT / "artifacts" / "phase189"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "phase189_report.json").write_text(json.dumps(report, indent=2))
    print(f"\nReport → artifacts/phase189/phase189_report.json")
    print(f"Runtime: {report['runtime_s']:.0f}s")

    if validated:
        cfg_path = ROOT / "configs" / "production_p91b_champion.json"
        cfg = json.loads(cfg_path.read_text())
        rw = cfg["breadth_regime_switching"]["regime_weights"]
        for r in ["LOW", "MID", "HIGH"]:
            w = best_weights[r]
            rw[r]["v1"]       = round(w["v1"],   4)
            rw[r]["i460bw168"] = round(w["i460"],  4)
            rw[r]["i415bw216"] = round(w["i415"],  4)
            rw[r]["f168"]     = round(w["f168"],  4)
        old_v = cfg["_version"]; parts = old_v.split(".")
        new_v = f"{parts[0]}.{int(parts[1]) + 1}.0" if len(parts) >= 2 else old_v
        cfg["_version"] = new_v
        cfg["_validated"] += (
            f"; Regime V1 weight re-opt P189: LOW_v1={best_v1_per_regime['LOW']:.4f} "
            f"MID_v1={best_v1_per_regime['MID']:.4f} HIGH_v1={best_v1_per_regime['HIGH']:.4f} "
            f"LOYO {loyo_wins_final}/5 Δ=+{joint_delta:.4f} OBJ={joint_obj:.4f} — PRODUCTION {new_v}"
        )
        cfg["monitoring"]["expected_performance"]["annual_sharpe_backtest"] = round(joint_obj, 4)
        cfg_path.write_text(json.dumps(cfg, indent=2, ensure_ascii=False))
        print(f"\nConfig updated: v{old_v} → v{new_v}  OBJ={joint_obj:.4f}")

if __name__ == "__main__":
    main()
