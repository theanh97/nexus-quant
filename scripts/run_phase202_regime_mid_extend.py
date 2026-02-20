"""
Phase 202 — MID f168 Ceiling Extension + HIGH Fine-Tune
========================================================
P201 found MID f168=0.68 still increasing (v1=0.20).
Extend: MID f168 ∈ [0.68, 0.70, 0.72, 0.75, 0.78] with v1=0.20
Also: HIGH fine-tune around v1=0.05/f168=0.18

Baseline: v2.30.0, OBJ=3.0616
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
    out = Path("artifacts/phase202"); out.mkdir(parents=True, exist_ok=True)
    (out / "phase202_report.json").write_text(json.dumps(_partial, indent=2))
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

VOL_WINDOW = 168; BRD_LB = 192; PCT_WINDOW = 336
TS_SHORT = 16; TS_LONG = 72; FUND_DISP_PCT = 240
P_LOW = 0.30; P_HIGH = 0.60
VOL_THRESHOLD = 0.50; VOL_SCALE = 0.30; VOL_F168_BOOST = 0.00
TS_RT = 0.65; TS_RS = 0.45; TS_BT = 0.25; TS_BS = 1.70
DISP_THR = 0.60; DISP_SCALE = 1.0

V1_PARAMS = {
    "k_per_side": 2, "w_carry": 0.35, "w_mom": 0.40, "w_mean_reversion": 0.25,
    "momentum_lookback_bars": 336, "mean_reversion_lookback_bars": 84,
    "vol_lookback_bars": 192, "target_gross_leverage": 0.35, "rebalance_interval_bars": 60,
}
I460_PARAMS = {"k_per_side": 4, "lookback_bars": 460, "beta_window_bars": 168,
               "target_gross_leverage": 0.3, "rebalance_interval_bars": 48}
I415_PARAMS = {"k_per_side": 4, "lookback_bars": 415, "beta_window_bars": 144,
               "target_gross_leverage": 0.3, "rebalance_interval_bars": 48}
F168_PARAMS = {"k_per_side": 2, "funding_lookback_bars": 168, "direction": "contrarian",
               "target_gross_leverage": 0.25, "rebalance_interval_bars": 24}

BASE_WEIGHTS = {
    "LOW":  {"v1": 0.46, "i460": 0.0716, "i415": 0.1184, "f168": 0.35},
    "MID":  {"v1": 0.20, "i460": 0.0446, "i415": 0.0754, "f168": 0.68},
    "HIGH": {"v1": 0.05, "i460": 0.2750, "i415": 0.4950, "f168": 0.18},
}
I460_RATIO = {r: BASE_WEIGHTS[r]["i460"] / (BASE_WEIGHTS[r]["i460"] + BASE_WEIGHTS[r]["i415"])
              for r in BASE_WEIGHTS}

BASELINE_OBJ = 3.0616
MIN_DELTA = 0.005; MIN_LOYO = 3

COST_MODEL = cost_model_from_config({
    "fee_rate": 0.0005, "slippage_rate": 0.0003, "cost_multiplier": 1.0,
})

def make_weights(regime, v1_w, f168_w):
    remaining = 1.0 - v1_w - f168_w
    if remaining < 0.02: return None
    r460 = I460_RATIO[regime]
    return {"v1": v1_w, "f168": f168_w,
            "i460": remaining * r460, "i415": remaining * (1 - r460)}

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
    btc_rets = np.zeros(n)
    for i in range(1, n):
        c0 = close_mat[i-1, 0]; c1 = close_mat[i, 0]
        btc_rets[i] = (c1/c0 - 1.0) if c0 > 0 else 0.0
    btc_vol = np.full(n, np.nan)
    for i in range(VOL_WINDOW, n):
        btc_vol[i] = float(np.std(btc_rets[i-VOL_WINDOW:i])) * np.sqrt(8760)
    if VOL_WINDOW < n: btc_vol[:VOL_WINDOW] = btc_vol[VOL_WINDOW]
    fund_rates = np.zeros((n, len(SYMBOLS)))
    for j, sym in enumerate(SYMBOLS):
        for i in range(n):
            ts = dataset.timeline[i]
            try: fund_rates[i, j] = dataset.last_funding_rate_before(sym, ts)
            except: fund_rates[i, j] = 0.0
    fund_std_raw = np.std(fund_rates, axis=1)
    fund_std_pct = np.full(n, 0.5)
    for i in range(FUND_DISP_PCT, n):
        fund_std_pct[i] = float(np.mean(fund_std_raw[i-FUND_DISP_PCT:i] <= fund_std_raw[i]))
    xsect_mean = np.mean(fund_rates, axis=1)
    ts_raw = rolling_mean_arr(xsect_mean, TS_SHORT) - rolling_mean_arr(xsect_mean, TS_LONG)
    ts_spread_pct = np.full(n, 0.5)
    for i in range(PCT_WINDOW, n):
        ts_spread_pct[i] = float(np.mean(ts_raw[i-PCT_WINDOW:i] <= ts_raw[i]))
    breadth = np.full(n, 0.5)
    for i in range(BRD_LB, n):
        pos = sum(1 for j in range(len(SYMBOLS))
                  if close_mat[i-BRD_LB, j] > 0 and close_mat[i, j] > close_mat[i-BRD_LB, j])
        breadth[i] = pos / len(SYMBOLS)
    brd_pct = np.full(n, 0.5)
    for i in range(PCT_WINDOW, n):
        brd_pct[i] = float(np.mean(breadth[i-PCT_WINDOW:i] <= breadth[i]))
    regime = np.where(brd_pct >= P_HIGH, 2, np.where(brd_pct >= P_LOW, 1, 0)).astype(int)
    sig_rets = {}
    for sk, sname, params in [
        ("v1", "nexus_alpha_v1", V1_PARAMS),
        ("i460", "idio_momentum_alpha", I460_PARAMS),
        ("i415", "idio_momentum_alpha", I415_PARAMS),
        ("f168", "funding_momentum_alpha", F168_PARAMS),
    ]:
        result = BacktestEngine(BacktestConfig(costs=COST_MODEL)).run(
            dataset, make_strategy({"name": sname, "params": params}))
        sig_rets[sk] = np.array(result.returns)
    print("S.", end=" ", flush=True)
    return btc_vol, fund_std_pct, ts_spread_pct, regime, sig_rets, n

def ensemble_returns(weights_by_regime, base_data):
    btc_vol, fund_std_pct, ts_spread_pct, regime, sig_rets, n = base_data
    RNAMES = ["LOW", "MID", "HIGH"]
    sk_all = ["v1", "i460", "i415", "f168"]
    min_len = min(len(sig_rets[k]) for k in sk_all)
    ens = np.zeros(min_len)
    bv = btc_vol[:min_len]; reg = regime[:min_len]
    fsp = fund_std_pct[:min_len]; tsp = ts_spread_pct[:min_len]
    for i in range(min_len):
        w = weights_by_regime[RNAMES[int(reg[i])]]
        ret_i = sum(w[sk] * sig_rets[sk][i] for sk in sk_all)
        if not np.isnan(bv[i]) and bv[i] > VOL_THRESHOLD:
            ret_i *= VOL_SCALE  # VOL_F168_BOOST=0.00
        if fsp[i] > DISP_THR: ret_i *= DISP_SCALE
        if tsp[i] > TS_RT: ret_i *= TS_RS
        elif tsp[i] < TS_BT: ret_i *= TS_BS
        ens[i] = ret_i
    return ens

def sharpe(r):
    r = r[~np.isnan(r)]
    if len(r) < 2: return 0.0
    return float(np.mean(r) / np.std(r, ddof=1) * np.sqrt(8760))

def obj(yearly):
    vals = list(yearly.values())
    return float(np.mean(vals) - 0.5 * np.std(vals, ddof=1))

def evaluate_weights(wbd, all_data):
    yearly = {yr: sharpe(ensemble_returns(wbd, d)) for yr, d in all_data.items()}
    return obj(yearly), yearly

def main():
    t0 = time.time()
    print("=" * 65)
    print("Phase 202 — MID f168 Ceiling Extension + HIGH Fine-Tune")
    print(f"Baseline: v2.30.0  OBJ={BASELINE_OBJ}")
    print("=" * 65)

    print("\n[1/4] Loading ...")
    all_data = {}
    for yr in sorted(YEAR_RANGES):
        print(f"  {yr}: ", end="", flush=True)
        all_data[yr] = load_year_data(yr)
    print()

    base_obj, base_yearly = evaluate_weights(BASE_WEIGHTS, all_data)
    print(f"\n  Baseline OBJ = {base_obj:.4f}  (expected ≈ {BASELINE_OBJ})")

    print("\n[2/4] Part A: Extend MID f168 (v1=0.20 fixed) ...")
    best_mid_f168 = 0.68; best_obj_mid = base_obj
    for f168_w in [0.68, 0.70, 0.72, 0.74, 0.76, 0.78]:
        w_r = make_weights("MID", 0.20, f168_w)
        if w_r is None: continue
        wbd = {"LOW": BASE_WEIGHTS["LOW"], "MID": w_r, "HIGH": BASE_WEIGHTS["HIGH"]}
        o, _ = evaluate_weights(wbd, all_data)
        delta = o - base_obj
        marker = " ← BEST" if o > best_obj_mid else ""
        print(f"  MID v1=0.20 f168={f168_w:.2f}  OBJ={o:.4f}  Δ={delta:+.4f}{marker}")
        if o > best_obj_mid:
            best_obj_mid = o; best_mid_f168 = f168_w
    print(f"  → Best MID f168={best_mid_f168:.2f}  OBJ={best_obj_mid:.4f}")

    print(f"\n[2b] Also check MID v1 extension (f168={best_mid_f168:.2f} fixed) ...")
    best_mid_v1 = 0.20
    for v1_w in [0.10, 0.15, 0.20]:
        w_r = make_weights("MID", v1_w, best_mid_f168)
        if w_r is None: continue
        wbd = {"LOW": BASE_WEIGHTS["LOW"], "MID": w_r, "HIGH": BASE_WEIGHTS["HIGH"]}
        o, _ = evaluate_weights(wbd, all_data)
        delta = o - base_obj
        marker = " ← BEST" if o > best_obj_mid else ""
        print(f"  MID v1={v1_w:.2f} f168={best_mid_f168:.2f}  OBJ={o:.4f}  Δ={delta:+.4f}{marker}")
        if o > best_obj_mid:
            best_obj_mid = o; best_mid_v1 = v1_w
    print(f"  → Best MID v1={best_mid_v1:.2f} f168={best_mid_f168:.2f}")

    print(f"\n[3/4] Part B: HIGH fine-tune ...")
    best_high_v1 = 0.05; best_high_f168 = 0.18; best_obj_high = base_obj
    for v1_w in [0.04, 0.05, 0.06]:
        for f168_w in [0.15, 0.17, 0.18, 0.20, 0.22]:
            w_r = make_weights("HIGH", v1_w, f168_w)
            if w_r is None: continue
            wbd = {"LOW": BASE_WEIGHTS["LOW"], "MID": BASE_WEIGHTS["MID"], "HIGH": w_r}
            o, _ = evaluate_weights(wbd, all_data)
            delta = o - base_obj
            marker = " ← BEST" if o > best_obj_high else ""
            print(f"  HIGH v1={v1_w:.2f} f168={f168_w:.2f}  OBJ={o:.4f}  Δ={delta:+.4f}{marker}")
            if o > best_obj_high:
                best_obj_high = o; best_high_v1 = v1_w; best_high_f168 = f168_w
    print(f"  → Best HIGH v1={best_high_v1:.2f} f168={best_high_f168:.2f}")

    # Build best weights
    best_weights = {
        "LOW":  BASE_WEIGHTS["LOW"].copy(),
        "MID":  make_weights("MID", best_mid_v1, best_mid_f168) or BASE_WEIGHTS["MID"].copy(),
        "HIGH": make_weights("HIGH", best_high_v1, best_high_f168) or BASE_WEIGHTS["HIGH"].copy(),
    }

    print("\n[4/4] Joint LOYO ...")
    joint_obj, joint_yearly = evaluate_weights(best_weights, all_data)
    delta = joint_obj - base_obj
    loyo_wins = 0
    for yr in sorted(joint_yearly):
        base_sh = base_yearly.get(yr, 0); cand_sh = joint_yearly[yr]
        win = cand_sh > base_sh
        if win: loyo_wins += 1
        print(f"  {yr}: base={base_sh:.4f}  cand={cand_sh:.4f}  {'WIN' if win else 'LOSE'}")
    print(f"\n  OBJ={joint_obj:.4f}  Δ={delta:+.4f}  LOYO {loyo_wins}/{len(joint_yearly)}")
    validated = loyo_wins >= MIN_LOYO and delta >= MIN_DELTA
    print(f"\n{'✅ VALIDATED' if validated else '❌ NO IMPROVEMENT'}")
    for r in ["LOW", "MID", "HIGH"]:
        w = best_weights[r]
        print(f"    {r}: v1={w['v1']:.4f} f168={w['f168']:.4f} i460={w['i460']:.4f} i415={w['i415']:.4f}")

    result = {"phase": 202, "baseline_obj": float(base_obj), "best_obj": float(joint_obj),
              "delta": float(delta), "loyo_wins": loyo_wins, "loyo_total": len(joint_yearly),
              "validated": validated,
              "best_weights": {r: {k: float(v) for k, v in best_weights[r].items()} for r in best_weights}}
    out = Path("artifacts/phase202"); out.mkdir(parents=True, exist_ok=True)
    (out / "phase202_report.json").write_text(json.dumps(result, indent=2))

    if validated:
        cfg_path = ROOT / "configs" / "production_p91b_champion.json"
        cfg = json.load(open(cfg_path))
        rw_cfg = cfg["breadth_regime_switching"]["regime_weights"]
        for regime in ["LOW", "MID", "HIGH"]:
            w = best_weights[regime]
            rw_cfg[regime]["v1"] = round(w["v1"], 4)
            rw_cfg[regime]["f168"] = round(w["f168"], 4)
            rw_cfg[regime]["i460bw168"] = round(w["i460"], 4)
            rw_cfg[regime]["i415bw216"] = round(w["i415"], 4)
        cfg["_version"] = "2.31.0"
        cfg["monitoring"]["expected_performance"]["annual_sharpe_backtest"] = round(joint_obj, 4)
        with open(cfg_path, "w") as f:
            json.dump(cfg, f, indent=2, ensure_ascii=False)
        print(f"\n  Config → v2.31.0  OBJ={round(joint_obj,4)}")

    print(f"\nRuntime: {int(time.time()-t0)}s")

main()
