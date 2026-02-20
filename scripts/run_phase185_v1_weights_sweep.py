"""
Phase 185 — V1 Internal Signal Weight Sweep
=============================================
Motivation: V1's internal blend (w_carry=0.35, w_mom=0.45, w_mean_reversion=0.20)
was set at strategy design time and never optimized. V1 is now the primary
contributor in LOW regime (22%) and secondary in MID (14%).

Approach:
  - 2D grid: w_mom × w_carry (w_mr = 1 - w_mom - w_carry, if > 0)
  - w_mom ∈ [0.20, 0.30, 0.40, 0.50, 0.60]
  - w_carry ∈ [0.15, 0.25, 0.35, 0.45]
  - w_mean_reversion = 1 - w_mom - w_carry (skip if < 0.05)
  → 20 combinations (minus infeasible)

Pre-compute all non-V1 signals once, re-run V1 only for each combination.

Baseline: v2.18.0, OBJ=2.5689
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

# ── timeout guard ─────────────────────────────────────────────────────────────
_partial: dict = {}
def _on_timeout(signum, frame):
    _partial["partial"] = True
    _partial["timeout_at"] = datetime.now(UTC).isoformat()
    out = Path("artifacts/phase185"); out.mkdir(parents=True, exist_ok=True)
    (out / "phase185_report.json").write_text(json.dumps(_partial, indent=2))
    sys.exit(0)
_signal.signal(_signal.SIGALRM, _on_timeout)
_signal.alarm(7200)

# ── constants ─────────────────────────────────────────────────────────────────
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
BASELINE_OBJ = 2.5689

# Baseline V1 weights
BASE_W_CARRY = 0.35
BASE_W_MOM   = 0.45
BASE_W_MR    = 0.20

# Sweep grid
W_MOM_VALS   = [0.20, 0.30, 0.40, 0.45, 0.50, 0.55, 0.60]
W_CARRY_VALS = [0.15, 0.25, 0.30, 0.35, 0.40, 0.45]

COST_MODEL = cost_model_from_config({
    "fee_rate": 0.0005, "slippage_rate": 0.0003, "cost_multiplier": 1.0,
})

# ── rolling helpers ───────────────────────────────────────────────────────────
def rolling_mean_arr(a: np.ndarray, w: int) -> np.ndarray:
    out = np.full(len(a), np.nan)
    if w <= 0 or len(a) == 0: return out
    cs = np.cumsum(np.where(np.isnan(a), 0.0, a))
    for i in range(w - 1, len(a)):
        out[i] = cs[i] / w if i == w - 1 else (cs[i] - cs[i - w]) / w
    return out

# ── data loading ──────────────────────────────────────────────────────────────
def load_year_data(year: int):
    """Load all overlay signals + base non-v1 signals once."""
    s, e = YEAR_RANGES[year]
    cfg = {"provider": "binance_rest_v1", "symbols": SYMBOLS,
           "start": s, "end": e, "bar_interval": "1h",
           "cache_dir": ".cache/binance_rest"}
    dataset = make_provider(cfg, seed=42).load()
    n = len(dataset.timeline)

    # Close prices
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

    # Funding rates
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

    # Breadth regime
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

    # Non-V1 signals (fixed)
    sig_rets = {}
    for sk, sname, params in [
        ("v1_base", "nexus_alpha_v1", {
            "k_per_side": 2, "w_carry": BASE_W_CARRY, "w_mom": BASE_W_MOM,
            "w_mean_reversion": BASE_W_MR, "momentum_lookback_bars": 336,
            "mean_reversion_lookback_bars": 72, "vol_lookback_bars": 168,
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

def run_v1_weights(base_data, w_carry: float, w_mom: float):
    """Run ensemble with V1 using given w_carry/w_mom/w_mr combination."""
    dataset, btc_vol, fund_std_pct, ts_spread_pct, regime, sig_rets, n = base_data
    w_mr = round(1.0 - w_carry - w_mom, 4)
    assert w_mr >= 0.05, f"w_mr={w_mr} too small"

    # Only re-run V1 with new weights
    if abs(w_carry - BASE_W_CARRY) < 1e-6 and abs(w_mom - BASE_W_MOM) < 1e-6:
        v1_rets = sig_rets["v1_base"]
    else:
        result = BacktestEngine(BacktestConfig(costs=COST_MODEL)).run(
            dataset,
            make_strategy({"name": "nexus_alpha_v1", "params": {
                "k_per_side": 2, "w_carry": w_carry, "w_mom": w_mom,
                "w_mean_reversion": w_mr, "momentum_lookback_bars": 336,
                "mean_reversion_lookback_bars": 72, "vol_lookback_bars": 168,
                "target_gross_leverage": 0.35, "rebalance_interval_bars": 60,
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

# ── metrics ───────────────────────────────────────────────────────────────────
def sharpe(r: np.ndarray) -> float:
    r = r[~np.isnan(r)]
    if len(r) < 2: return 0.0
    return float(np.mean(r) / np.std(r, ddof=1) * np.sqrt(8760))

def obj(yearly: dict) -> float:
    vals = list(yearly.values())
    return float(np.mean(vals) - 0.5 * np.std(vals, ddof=1))

# ── main ──────────────────────────────────────────────────────────────────────
def main():
    t0 = time.time()
    # Build sweep combos
    combos = []
    for wm in W_MOM_VALS:
        for wc in W_CARRY_VALS:
            wr = round(1.0 - wm - wc, 4)
            if 0.05 <= wr <= 0.55:
                combos.append((wc, wm, wr))
    combos.sort(key=lambda x: x[1])  # sort by w_mom

    print("=" * 65)
    print("Phase 185 — V1 Internal Signal Weight Sweep")
    print(f"Baseline: v2.18.0  OBJ={BASELINE_OBJ}")
    print(f"Baseline: w_carry={BASE_W_CARRY} w_mom={BASE_W_MOM} w_mr={BASE_W_MR}")
    print(f"Sweep:    {len(combos)} (w_carry, w_mom, w_mr) combinations")
    print("=" * 65)

    # ── [1/3] Load year data ──────────────────────────────────────────────────
    print("\n[1/3] Loading year data ...")
    data_by_yr: dict = {}
    for yr in YEARS:
        print(f"  {yr}: ", end="", flush=True)
        data_by_yr[yr] = load_year_data(yr)
    print()

    # Baseline
    bl_yr = {yr: sharpe(run_v1_weights(data_by_yr[yr], BASE_W_CARRY, BASE_W_MOM))
             for yr in YEARS}
    bl_obj = obj(bl_yr)
    print(f"  Baseline OBJ = {bl_obj:.4f}  (expected ≈ {BASELINE_OBJ})")

    # ── [2/3] Grid sweep ──────────────────────────────────────────────────────
    print(f"\n[2/3] Grid sweep — {len(combos)} combos ...")
    sweep_results: dict = {}
    best_combo = (BASE_W_CARRY, BASE_W_MOM, BASE_W_MR)
    best_obj_val = bl_obj

    for wc, wm, wr in combos:
        yr_sharpes = {yr: sharpe(run_v1_weights(data_by_yr[yr], wc, wm)) for yr in YEARS}
        o = obj(yr_sharpes); d = o - bl_obj
        loyo_wins = sum(1 for yr in YEARS if (yr_sharpes[yr] - bl_yr[yr]) > 0)
        tag = f"wc={wc:.2f} wm={wm:.2f} wr={wr:.2f}"
        is_base = (abs(wc - BASE_W_CARRY) < 1e-4 and abs(wm - BASE_W_MOM) < 1e-4)
        flag = ("⭐" if (o > best_obj_val and loyo_wins >= 3 and d > 0.005) else
                ("✅" if (loyo_wins >= 3 and d > 0.005) else
                ("⚠️ " if is_base else "❌")))
        base_tag = " ← baseline" if is_base else ""
        print(f"  [{tag}] OBJ={o:.4f}  Δ={d:+.4f}  LOYO {loyo_wins}/5  {flag}{base_tag}")
        key = f"{wc:.2f}_{wm:.2f}_{wr:.2f}"
        sweep_results[key] = {"wc": wc, "wm": wm, "wr": wr, "obj": o, "delta": d,
                               "loyo_wins": loyo_wins, "yr_sharpes": yr_sharpes}
        if o > best_obj_val and loyo_wins >= 3 and d > 0.005:
            best_combo = (wc, wm, wr)
            best_obj_val = o

    # ── [3/3] LOYO validation of winner ───────────────────────────────────────
    bwc, bwm, bwr = best_combo
    print(f"\n[3/3] LOYO validation — best wc={bwc} wm={bwm} wr={bwr}  OBJ={best_obj_val:.4f}")

    if abs(bwc - BASE_W_CARRY) < 1e-4 and abs(bwm - BASE_W_MOM) < 1e-4:
        print("  ❌ NO IMPROVEMENT — baseline weights already optimal")
        loyo_wins_final, loyo_avg_d = 0, 0.0
        validated = False
    else:
        loyo_wins_final = 0; loyo_deltas = []
        for loyo_yr in YEARS:
            train_yrs = [y for y in YEARS if y != loyo_yr]
            best_train = {y: sharpe(run_v1_weights(data_by_yr[y], bwc, bwm)) for y in train_yrs}
            base_train = {y: bl_yr[y] for y in train_yrs}
            d = obj(best_train) - obj(base_train)
            win = d > 0.005
            loyo_wins_final += int(win); loyo_deltas.append(d)
            print(f"  {loyo_yr} left out: {'✅' if win else '❌'}  Δ={d:+.4f}")
        loyo_avg_d = float(np.mean(loyo_deltas))
        validated = loyo_wins_final >= 3 and (best_obj_val - bl_obj) > 0.005
        print(f"  LOYO {loyo_wins_final}/5  avg_Δ={loyo_avg_d:+.4f}  Δ_OBJ={best_obj_val - bl_obj:+.4f}")

    if validated:
        print(f"\n✅ VALIDATED — V1 weights: wc={bwc} wm={bwm} wr={bwr}  OBJ={best_obj_val:.4f}")
    else:
        print(f"\n❌ NO IMPROVEMENT — V1 weights wc={BASE_W_CARRY}/wm={BASE_W_MOM}/wr={BASE_W_MR} optimal")

    # ── report ────────────────────────────────────────────────────────────────
    report = {
        "phase": 185,
        "target": "v1_internal_weights",
        "sweep": {k: {kk: vv for kk, vv in v.items() if kk != "yr_sharpes"}
                  for k, v in sweep_results.items()},
        "baseline": {"obj": bl_obj, "w_carry": BASE_W_CARRY,
                     "w_mom": BASE_W_MOM, "w_mr": BASE_W_MR},
        "best": {"obj": best_obj_val, "w_carry": bwc, "w_mom": bwm, "w_mr": bwr},
        "loyo": {"wins": loyo_wins_final, "avg_delta": loyo_avg_d},
        "validated": validated,
        "runtime_s": round(time.time() - t0, 1),
    }
    out_dir = ROOT / "artifacts" / "phase185"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "phase185_report.json").write_text(json.dumps(report, indent=2))
    print(f"\nReport → artifacts/phase185/phase185_report.json")
    print(f"Runtime: {report['runtime_s']:.0f}s")

    # ── update config if validated ────────────────────────────────────────────
    if validated:
        cfg_path = ROOT / "configs" / "production_p91b_champion.json"
        cfg = json.loads(cfg_path.read_text())
        sig = cfg["ensemble"]["signals"]["v1"]["params"]
        sig["w_carry"] = bwc
        sig["w_mom"] = bwm
        sig["w_mean_reversion"] = bwr
        old_v = cfg["_version"]
        parts = old_v.split(".")
        new_v = f"{parts[0]}.{parts[1]}.{int(parts[2]) + 1}" if len(parts) == 3 else old_v
        cfg["_version"] = new_v
        cfg["_validated"] += (
            f"; V1 weights re-tune P185: wc={bwc} wm={bwm} wr={bwr} LOYO {loyo_wins_final}/5 "
            f"Δ=+{best_obj_val - bl_obj:.4f} OBJ={best_obj_val:.4f} — PRODUCTION {new_v}"
        )
        cfg["monitoring"]["expected_performance"]["annual_sharpe_backtest"] = round(best_obj_val, 4)
        cfg_path.write_text(json.dumps(cfg, indent=2, ensure_ascii=False))
        print(f"\nConfig updated: v{old_v} → v{new_v}  OBJ={best_obj_val:.4f}")

if __name__ == "__main__":
    main()
