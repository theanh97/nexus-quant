"""
Phase 190 — V1 LOW Regime Weight Extension
==========================================
Motivation: P189 showed monotonic improvement in LOW regime V1 weight
up to 0.40 (the max swept). Need to extend to find true peak.
Also: MID peaked at 0.30, fine-tune 0.25-0.35 range.

Sweep:
  Part A: LOW v1 weight ∈ [0.40, 0.42, 0.44, 0.46, 0.48, 0.50, 0.52, 0.55]
  Part B: MID v1 weight fine-tune ∈ [0.25, 0.27, 0.30, 0.32, 0.35] (keeping best LOW)

Baseline: v2.21.0, OBJ=2.8506 (LOW_v1=0.40, MID_v1=0.30, HIGH_v1=0.06)
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
    out = Path("artifacts/phase190"); out.mkdir(parents=True, exist_ok=True)
    (out / "phase190_report.json").write_text(json.dumps(_partial, indent=2))
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

# v2.21.0 baseline (P189 result)
BASE_WEIGHTS = {
    "LOW":  {"v1": 0.40,   "i460": 0.0944, "i415": 0.1556, "f168": 0.35},
    "MID":  {"v1": 0.30,   "i460": 0.1305, "i415": 0.2195, "f168": 0.35},
    "HIGH": {"v1": 0.0600, "i460": 0.3000, "i415": 0.5400, "f168": 0.10},
}
F168_WEIGHTS = {"LOW": 0.35, "MID": 0.35, "HIGH": 0.10}

BASELINE_OBJ = 2.8506

LOW_SWEEP  = [0.40, 0.42, 0.44, 0.46, 0.48, 0.50, 0.52, 0.55]
MID_SWEEP  = [0.25, 0.27, 0.30, 0.32, 0.35]

COST_MODEL = cost_model_from_config({
    "fee_rate": 0.0005, "slippage_rate": 0.0003, "cost_multiplier": 1.0,
})

def make_weights_v1(regime: str, new_v1: float, ref_weights: dict) -> dict | None:
    f168 = F168_WEIGHTS[regime]
    base = ref_weights[regime]
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
    print("Phase 190 — V1 LOW Regime Weight Extension + MID Fine-Tune")
    print(f"Baseline: v2.21.0  OBJ={BASELINE_OBJ}")
    print(f"  LOW v1=0.40 (was ceiling of P189 sweep)")
    print(f"  MID v1=0.30 (peaked in P189)")
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

    # ── Part A: LOW regime extension ──────────────────────────────────────────
    print(f"\n[2/4] Part A — LOW regime V1 weight extension ...")
    best_low_v1, best_low_obj = 0.40, bl_obj
    low_results = {}
    for v1w in LOW_SWEEP:
        new_low_w = make_weights_v1("LOW", v1w, BASE_WEIGHTS)
        if new_low_w is None:
            print(f"  LOW v1={v1w:.2f} → INFEASIBLE"); continue
        weights = {"LOW": new_low_w, "MID": BASE_WEIGHTS["MID"], "HIGH": BASE_WEIGHTS["HIGH"]}
        yr_s = {yr: sharpe(compute_ensemble(data_by_yr[yr], weights)) for yr in YEARS}
        o = obj(yr_s); d = o - bl_obj
        loyo_wins = sum(1 for yr in YEARS if (yr_s[yr] - bl_yr[yr]) > 0)
        is_base = abs(v1w - 0.40) < 1e-4
        flag = ("⭐" if (o > best_low_obj and loyo_wins >= 2) else
                ("⚠️ " if is_base else ("+" if d > 0 else "")))
        print(f"  LOW v1={v1w:.2f} i460={new_low_w['i460']:.4f} i415={new_low_w['i415']:.4f} "
              f"→ OBJ={o:.4f}  Δ={d:+.4f}  {flag}")
        low_results[v1w] = {"obj": o, "delta": d, "loyo_wins": loyo_wins,
                             "weights": new_low_w, "yr_sharpes": yr_s}
        if o > best_low_obj and loyo_wins >= 2:
            best_low_v1, best_low_obj = v1w, o
    print(f"  → Best LOW v1: {best_low_v1:.4f}  OBJ={best_low_obj:.4f}")
    best_low_weights = (make_weights_v1("LOW", best_low_v1, BASE_WEIGHTS)
                        if best_low_v1 != 0.40 else BASE_WEIGHTS["LOW"])

    # ── Part B: MID fine-tune with best LOW ───────────────────────────────────
    print(f"\n[3/4] Part B — MID regime fine-tune (with LOW={best_low_v1:.4f}) ...")
    best_mid_v1, best_mid_obj = 0.30, best_low_obj
    mid_results = {}
    for v1w in MID_SWEEP:
        new_mid_w = make_weights_v1("MID", v1w, BASE_WEIGHTS)
        if new_mid_w is None:
            print(f"  MID v1={v1w:.2f} → INFEASIBLE"); continue
        weights = {"LOW": best_low_weights, "MID": new_mid_w, "HIGH": BASE_WEIGHTS["HIGH"]}
        yr_s = {yr: sharpe(compute_ensemble(data_by_yr[yr], weights)) for yr in YEARS}
        o = obj(yr_s); d = o - bl_obj
        loyo_wins = sum(1 for yr in YEARS if (yr_s[yr] - bl_yr[yr]) > 0)
        is_base = abs(v1w - 0.30) < 1e-4
        flag = ("⭐" if (o > best_mid_obj and loyo_wins >= 2) else
                ("⚠️ " if is_base else ("+" if d > 0 else "")))
        print(f"  MID v1={v1w:.2f} i460={new_mid_w['i460']:.4f} i415={new_mid_w['i415']:.4f} "
              f"→ OBJ={o:.4f}  Δ={d:+.4f}  {flag}")
        mid_results[v1w] = {"obj": o, "delta": d, "loyo_wins": loyo_wins,
                             "weights": new_mid_w, "yr_sharpes": yr_s}
        if o > best_mid_obj and loyo_wins >= 2:
            best_mid_v1, best_mid_obj = v1w, o
    print(f"  → Best MID v1: {best_mid_v1:.4f}  OBJ={best_mid_obj:.4f}")
    best_mid_weights = (make_weights_v1("MID", best_mid_v1, BASE_WEIGHTS)
                        if best_mid_v1 != 0.30 else BASE_WEIGHTS["MID"])

    # ── Part C: Joint LOYO validation ─────────────────────────────────────────
    print(f"\n[4/4] Joint LOYO validation ...")
    best_weights = {"LOW": best_low_weights, "MID": best_mid_weights, "HIGH": BASE_WEIGHTS["HIGH"]}
    joint_yr = {yr: sharpe(compute_ensemble(data_by_yr[yr], best_weights)) for yr in YEARS}
    joint_obj = obj(joint_yr); joint_delta = joint_obj - bl_obj
    print(f"  Joint OBJ={joint_obj:.4f}  Δ={joint_delta:+.4f}")

    loyo_wins_final = 0; loyo_deltas = []
    for loyo_yr in YEARS:
        train_yrs = [y for y in YEARS if y != loyo_yr]
        d = obj({y: joint_yr[y] for y in train_yrs}) - obj({y: bl_yr[y] for y in train_yrs})
        win = d > 0.005
        loyo_wins_final += int(win); loyo_deltas.append(d)
        print(f"  {loyo_yr} left out: {'✅' if win else '❌'}  Δ={d:+.4f}")
    loyo_avg_d = float(np.mean(loyo_deltas))
    validated = loyo_wins_final >= 3 and joint_delta > 0.005
    print(f"  LOYO {loyo_wins_final}/5  avg_Δ={loyo_avg_d:+.4f}")

    if validated:
        print(f"\n✅ VALIDATED — LOW_v1={best_low_v1} MID_v1={best_mid_v1}  OBJ={joint_obj:.4f}")
    elif joint_delta > 0:
        print(f"\n ✔  PARTIAL IMPROVEMENT — Δ={joint_delta:+.4f} but LOYO {loyo_wins_final}/5")
    else:
        print(f"\n ✔  CONFIRMED — P189 values (LOW=0.40, MID=0.30) already optimal")

    report = {
        "phase": 190,
        "best_low_v1": best_low_v1, "best_mid_v1": best_mid_v1,
        "best_weights": best_weights,
        "baseline": {"obj": bl_obj},
        "best": {"obj": joint_obj, "delta": joint_delta},
        "loyo": {"wins": loyo_wins_final, "avg_delta": loyo_avg_d},
        "validated": validated,
        "runtime_s": round(time.time() - t0, 1),
    }
    out_dir = ROOT / "artifacts" / "phase190"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "phase190_report.json").write_text(json.dumps(report, indent=2))
    print(f"\nReport → artifacts/phase190/phase190_report.json")
    print(f"Runtime: {report['runtime_s']:.0f}s")

    if validated:
        cfg_path = ROOT / "configs" / "production_p91b_champion.json"
        cfg = json.loads(cfg_path.read_text())
        rw = cfg["breadth_regime_switching"]["regime_weights"]
        for r, w in [("LOW", best_low_weights), ("MID", best_mid_weights)]:
            rw[r]["v1"]        = round(w["v1"], 4)
            rw[r]["i460bw168"] = round(w["i460"], 4)
            rw[r]["i415bw216"] = round(w["i415"], 4)
            rw[r]["f168"]      = round(w["f168"], 4)
        old_v = cfg["_version"]; parts = old_v.split(".")
        new_v = f"{parts[0]}.{int(parts[1]) + 1}.0" if len(parts) >= 2 else old_v
        cfg["_version"] = new_v
        cfg["_validated"] += (
            f"; V1 weight extend P190: LOW_v1={best_low_v1:.4f} MID_v1={best_mid_v1:.4f} "
            f"LOYO {loyo_wins_final}/5 Δ=+{joint_delta:.4f} OBJ={joint_obj:.4f} — PRODUCTION {new_v}"
        )
        cfg["monitoring"]["expected_performance"]["annual_sharpe_backtest"] = round(joint_obj, 4)
        cfg_path.write_text(json.dumps(cfg, indent=2, ensure_ascii=False))
        print(f"\nConfig updated: v{old_v} → v{new_v}  OBJ={joint_obj:.4f}")

if __name__ == "__main__":
    main()
