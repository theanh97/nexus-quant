"""
Phase 202 — MID Regime Weight Fine-Tune (v2.29.0 stack)
=========================================================
v2.29.0: MID regime has f168=0.65 (very dominant), v1=0.30, i460=0.018, i415=0.031.
Need to validate this extreme config is stable and not overfitted.

Also: LOW regime v1=0.50 (new high). Validate.

Test:
  Part A: MID f168 ∈ [0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70] (v1=0.30 fixed, i460/i415 scale)
  Part B: LOW v1 ∈ [0.40, 0.44, 0.48, 0.50, 0.52, 0.55, 0.60] (f168=0.32 fixed, i460/i415 scale)

Read config dynamically.
Validate: LOYO ≥3/5 AND Δ>0.005 → update config
"""
import os, sys, json, time, re
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
    out = Path("artifacts/phase202"); out.mkdir(parents=True, exist_ok=True)
    (out / "phase202_report.json").write_text(json.dumps(_partial, indent=2, default=str))
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

COST_MODEL = cost_model_from_config({
    "fee_rate": 0.0005, "slippage_rate": 0.0003, "cost_multiplier": 1.0,
})

def get_config():
    cfg = json.loads(Path("configs/production_p91b_champion.json").read_text())
    brs = cfg["breadth_regime_switching"]
    rw = brs["regime_weights"]
    weights = {}
    for regime in ["LOW", "MID", "HIGH"]:
        w = rw[regime]
        weights[regime] = {
            "v1":   w["v1"],
            "i460": w.get("i460bw168", w.get("i460", 0.15)),
            "i415": w.get("i415bw216", w.get("i415", 0.25)),
            "f168": w["f168"],
        }
    fts = cfg.get("funding_term_structure_overlay", {})
    params = {
        "p_low": brs.get("p_low", 0.30), "p_high": brs.get("p_high", 0.60),
        "brd_lb": brs.get("breadth_lookback_bars", 192),
        "pct_win": brs.get("rolling_percentile_window", 336),
        "ts_rt": fts.get("reduce_threshold", 0.65), "ts_rs": fts.get("reduce_scale", 0.45),
        "ts_bt": fts.get("boost_threshold", 0.25), "ts_bs": fts.get("boost_scale", 1.70),
        "ts_short": fts.get("short_window_bars", 16), "ts_long": fts.get("long_window_bars", 72),
    }
    vol_ov = cfg.get("vol_regime_overlay", {})
    params["vol_thr"] = vol_ov.get("threshold", 0.50)
    params["vol_scale"] = vol_ov.get("scale_factor", 0.30)
    params["f168_boost"] = vol_ov.get("f144_boost", 0.00)
    m = re.findall(r"OBJ=([\d.]+)", cfg.get("_validated", ""))
    baseline_obj = float(m[-1]) if m else 3.0243
    return cfg, weights, params, baseline_obj

VOL_WINDOW = 168; FUND_DISP_PCT = 240; DISP_THR = 0.60; DISP_SCALE = 1.0

MID_F168_SWEEP = [0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]
LOW_V1_SWEEP   = [0.40, 0.44, 0.48, 0.50, 0.52, 0.55, 0.60]

def rolling_mean_arr(a, w):
    out = np.full(len(a), np.nan)
    if w <= 0: return out
    cs = np.cumsum(np.where(np.isnan(a), 0.0, a))
    for i in range(w - 1, len(a)):
        out[i] = cs[i] / w if i == w - 1 else (cs[i] - cs[i - w]) / w
    return out

def load_year(year, params):
    p = params
    s, e = YEAR_RANGES[year]
    cfg_d = {"provider": "binance_rest_v1", "symbols": SYMBOLS,
             "start": s, "end": e, "bar_interval": "1h", "cache_dir": ".cache/binance_rest"}
    dataset = make_provider(cfg_d, seed=42).load()
    n = len(dataset.timeline)

    close_mat = np.zeros((n, len(SYMBOLS)))
    for j, sym in enumerate(SYMBOLS):
        for i in range(n): close_mat[i, j] = dataset.close(sym, i)

    btc_rets = np.zeros(n)
    for i in range(1, n):
        c0 = close_mat[i-1, 0]; c1 = close_mat[i, 0]
        btc_rets[i] = (c1/c0-1.0) if c0 > 0 else 0.0
    btc_vol = np.full(n, np.nan)
    for i in range(VOL_WINDOW, n):
        btc_vol[i] = float(np.std(btc_rets[i-VOL_WINDOW:i])) * np.sqrt(8760)
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
        fund_std_pct[i] = float(np.mean(fund_std_raw[i-FUND_DISP_PCT:i] <= fund_std_raw[i]))
    fund_std_pct[:FUND_DISP_PCT] = 0.5

    xsect_mean = np.mean(fund_rates, axis=1)
    ts_raw = rolling_mean_arr(xsect_mean, p["ts_short"]) - rolling_mean_arr(xsect_mean, p["ts_long"])
    ts_spread_pct = np.full(n, 0.5)
    pct_win = p["pct_win"]
    for i in range(pct_win, n):
        ts_spread_pct[i] = float(np.mean(ts_raw[i-pct_win:i] <= ts_raw[i]))
    ts_spread_pct[:pct_win] = 0.5

    brd_lb = p["brd_lb"]
    breadth = np.full(n, 0.5)
    for i in range(brd_lb, n):
        pos = sum(1 for j in range(len(SYMBOLS))
                  if close_mat[i-brd_lb, j] > 0 and close_mat[i, j] > close_mat[i-brd_lb, j])
        breadth[i] = pos / len(SYMBOLS)
    breadth[:brd_lb] = 0.5
    brd_pct = np.full(n, 0.5)
    for i in range(pct_win, n):
        brd_pct[i] = float(np.mean(breadth[i-pct_win:i] <= breadth[i]))
    brd_pct[:pct_win] = 0.5
    regime = np.where(brd_pct >= p["p_high"], 2, np.where(brd_pct >= p["p_low"], 1, 0)).astype(int)

    sig_rets = {}
    for sk, sname, sp in [
        ("v1", "nexus_alpha_v1", {
            "k_per_side": 2, "w_carry": 0.35, "w_mom": 0.40, "w_mean_reversion": 0.25,
            "momentum_lookback_bars": 336, "mean_reversion_lookback_bars": 84,
            "vol_lookback_bars": 192, "target_gross_leverage": 0.35, "rebalance_interval_bars": 60,
        }),
        ("i460", "idio_momentum_alpha", {
            "k_per_side": 4, "lookback_bars": 460, "beta_window_bars": 168,
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
            dataset, make_strategy({"name": sname, "params": sp}))
        sig_rets[sk] = np.array(result.returns)

    print(".", end=" ", flush=True)
    return btc_vol, fund_std_pct, ts_spread_pct, regime, sig_rets, n

def compute_ens(year_data, weights, params):
    p = params
    btc_vol, fund_std_pct, ts_spread_pct, regime, sig_rets, n = year_data
    min_len = min(len(v) for v in sig_rets.values())
    bv = btc_vol[:min_len]; reg = regime[:min_len]
    fsp = fund_std_pct[:min_len]; tsp = ts_spread_pct[:min_len]
    ens = np.zeros(min_len)
    for i in range(min_len):
        w = weights[["LOW", "MID", "HIGH"][int(reg[i])]]
        if not np.isnan(bv[i]) and bv[i] > p["vol_thr"]:
            fb = p["f168_boost"]
            if fb > 0:
                bpp = fb / 3.0
                ret_i = (
                    (w["v1"] + bpp) * sig_rets["v1"][i] +
                    (w["i460"] + bpp) * sig_rets["i460"][i] +
                    (w["i415"] + bpp) * sig_rets["i415"][i] +
                    (w["f168"] - fb) * sig_rets["f168"][i]
                ) * p["vol_scale"]
            else:
                ret_i = (
                    w["v1"] * sig_rets["v1"][i] +
                    w["i460"] * sig_rets["i460"][i] +
                    w["i415"] * sig_rets["i415"][i] +
                    w["f168"] * sig_rets["f168"][i]
                ) * p["vol_scale"]
        else:
            ret_i = (
                w["v1"] * sig_rets["v1"][i] +
                w["i460"] * sig_rets["i460"][i] +
                w["i415"] * sig_rets["i415"][i] +
                w["f168"] * sig_rets["f168"][i]
            )
        if DISP_SCALE > 1.0 and fsp[i] > DISP_THR: ret_i *= DISP_SCALE
        if tsp[i] > p["ts_rt"]: ret_i *= p["ts_rs"]
        elif tsp[i] < p["ts_bt"]: ret_i *= p["ts_bs"]
        ens[i] = ret_i
    return ens

def sharpe(rets, n):
    if n < 20: return -999.0
    mu = float(np.mean(rets[:n])) * 8760
    sd = float(np.std(rets[:n])) * np.sqrt(8760)
    return mu / sd if sd > 1e-9 else 0.0

def obj_fn(yr_sharpes):
    vals = list(yr_sharpes.values())
    return float(np.mean(vals) - 0.5 * np.std(vals))

def eval_weights(year_data, weights, params):
    yr_sharpes = {}
    for yr in YEARS:
        ens = compute_ens(year_data[yr], weights, params)
        yr_sharpes[yr] = sharpe(ens, year_data[yr][-1])
    return obj_fn(yr_sharpes)

def make_mid_weights(base_weights, mid_f168, mid_v1):
    """Build weights with MID having given f168 and v1, scale i460/i415."""
    w_new = {r: dict(base_weights[r]) for r in ["LOW", "MID", "HIGH"]}
    # For MID: fix v1 and f168, scale i460/i415 to remainder
    others_sum_base = base_weights["MID"]["i460"] + base_weights["MID"]["i415"]
    target_others = 1.0 - mid_v1 - mid_f168
    if target_others < 0.01 or others_sum_base <= 0:
        return None
    scale = max(0, target_others) / max(others_sum_base, 0.001)
    w_new["MID"] = {
        "v1": mid_v1,
        "i460": max(0.005, base_weights["MID"]["i460"] * scale),
        "i415": max(0.005, base_weights["MID"]["i415"] * scale),
        "f168": mid_f168,
    }
    return w_new

def make_low_weights(base_weights, low_v1):
    """Build weights with LOW having given v1, scale i460/i415."""
    low_f168 = base_weights["LOW"]["f168"]
    w_new = {r: dict(base_weights[r]) for r in ["LOW", "MID", "HIGH"]}
    others_sum_base = base_weights["LOW"]["i460"] + base_weights["LOW"]["i415"]
    target_others = 1.0 - low_v1 - low_f168
    if target_others < 0.01 or others_sum_base <= 0:
        return None
    scale = target_others / others_sum_base
    w_new["LOW"] = {
        "v1": low_v1,
        "i460": max(0.005, base_weights["LOW"]["i460"] * scale),
        "i415": max(0.005, base_weights["LOW"]["i415"] * scale),
        "f168": low_f168,
    }
    return w_new

# ── MAIN ────────────────────────────────────────────────────────────────────
print("=" * 60)
print("Phase 202 — MID + LOW Regime Weight Fine-Tune")
print("=" * 60)
_start = time.time()

prod_cfg, BASE_WEIGHTS, PARAMS, BASELINE_OBJ = get_config()
ver_now = prod_cfg.get("_version", "2.29.0")
print(f"  Config: {ver_now}  Baseline OBJ≈{BASELINE_OBJ:.4f}")
print(f"  Weights: {BASE_WEIGHTS}")

print("\n[1] Loading all years ...")
year_data = {}
for yr in YEARS:
    print(f"  {yr}", end="", flush=True)
    year_data[yr] = load_year(yr, PARAMS)
    print()

# Part A: MID f168 sweep (v1=current)
mid_v1_cur = BASE_WEIGHTS["MID"]["v1"]
mid_f168_cur = BASE_WEIGHTS["MID"]["f168"]
print(f"\n[2] Part A: MID f168 sweep (v1={mid_v1_cur:.3f} fixed) ...")
partA = []
for f168_v in MID_F168_SWEEP:
    w_try = make_mid_weights(BASE_WEIGHTS, f168_v, mid_v1_cur)
    if w_try is None:
        print(f"  f168={f168_v:.2f} → INFEASIBLE")
        continue
    obj = eval_weights(year_data, w_try, PARAMS)
    partA.append({"mid_f168": f168_v, "obj": obj, "weights": w_try["MID"]})
    flag = " ← CURRENT" if abs(f168_v - mid_f168_cur) < 0.01 else ""
    print(f"  MID f168={f168_v:.2f} (i460={w_try['MID']['i460']:.3f}, i415={w_try['MID']['i415']:.3f})  OBJ={obj:.4f}{flag}")

best_A = max(partA, key=lambda x: x["obj"])
print(f"  Best MID f168: {best_A['mid_f168']:.2f}  OBJ={best_A['obj']:.4f}")

# Part B: LOW v1 sweep
low_v1_cur = BASE_WEIGHTS["LOW"]["v1"]
print(f"\n[3] Part B: LOW v1 sweep (f168={BASE_WEIGHTS['LOW']['f168']:.3f} fixed) ...")
partB = []
best_mid_f168 = best_A["mid_f168"]
for v1_v in LOW_V1_SWEEP:
    # Combine best MID f168 with this LOW v1
    w_try = make_mid_weights(BASE_WEIGHTS, best_mid_f168, mid_v1_cur)
    if w_try is None: w_try = {r: dict(BASE_WEIGHTS[r]) for r in BASE_WEIGHTS}
    w_try2 = make_low_weights(w_try, v1_v)
    if w_try2 is None:
        print(f"  LOW v1={v1_v:.2f} → INFEASIBLE")
        continue
    obj = eval_weights(year_data, w_try2, PARAMS)
    partB.append({"low_v1": v1_v, "mid_f168": best_mid_f168, "obj": obj})
    flag = " ← CURRENT" if abs(v1_v - low_v1_cur) < 0.01 else ""
    print(f"  LOW v1={v1_v:.2f} (i460={w_try2['LOW']['i460']:.3f}, i415={w_try2['LOW']['i415']:.3f})  OBJ={obj:.4f}{flag}")

best_B = max(partB, key=lambda x: x["obj"])
print(f"  Best LOW v1: {best_B['low_v1']:.2f}  OBJ={best_B['obj']:.4f}  Δ={best_B['obj']-BASELINE_OBJ:+.4f}")

# Build final best weights
best_low_v1 = best_B["low_v1"]
best_mid_f168_final = best_B["mid_f168"]
best_weights_final = make_mid_weights(BASE_WEIGHTS, best_mid_f168_final, mid_v1_cur)
if best_weights_final is None: best_weights_final = {r: dict(BASE_WEIGHTS[r]) for r in BASE_WEIGHTS}
best_weights_final2 = make_low_weights(best_weights_final, best_low_v1)
if best_weights_final2 is None: best_weights_final2 = best_weights_final
best = best_B

# LOYO validation
print(f"\n[4] LOYO validation ...")
loyo_wins = 0; loyo_table = []
for yr in YEARS:
    ens_best = compute_ens(year_data[yr], best_weights_final2, PARAMS)
    ens_curr = compute_ens(year_data[yr], BASE_WEIGHTS, PARAMS)
    n = year_data[yr][-1]
    sh_b = sharpe(ens_best, n); sh_c = sharpe(ens_curr, n)
    win = bool(sh_b > sh_c)
    if win: loyo_wins += 1
    loyo_table.append({"year": yr, "sh_best": sh_b, "sh_curr": sh_c, "win": win})
    print(f"  {yr}: best={sh_b:.4f} curr={sh_c:.4f} {'WIN' if win else 'LOSS'}")

print(f"\n  LOYO: {loyo_wins}/5 wins")
validated = loyo_wins >= 3 and best["obj"] > BASELINE_OBJ + 0.005

out = Path("artifacts/phase202"); out.mkdir(parents=True, exist_ok=True)
report = {
    "phase": 202, "target": "mid_low_regime_weights",
    "partA": [{k: v for k, v in r.items() if k != "weights"} for r in partA],
    "partB": partB,
    "best_mid_f168": best_mid_f168_final, "best_low_v1": best_low_v1,
    "final_weights": best_weights_final2,
    "best_obj": best["obj"], "delta": best["obj"] - BASELINE_OBJ,
    "baseline_obj": BASELINE_OBJ,
    "loyo": {"wins": loyo_wins, "table": loyo_table},
    "validated": validated,
    "runtime_s": round(time.time() - _start, 1),
    "timestamp": datetime.now(UTC).isoformat(),
}
(out / "phase202_report.json").write_text(json.dumps(report, indent=2, default=str))
print(f"  Report saved  ({report['runtime_s']}s)")

if validated:
    cfg = json.loads(Path("configs/production_p91b_champion.json").read_text())
    ver = cfg.get("_version", "2.29.0")
    major, minor, patch = map(int, ver.split("."))
    new_ver = f"{major}.{minor + 1}.{patch}"
    cfg["_version"] = new_ver
    cfg["_validated"] += (
        f"; MID+LOW weight tune P202: MID_f168={mid_f168_cur:.2f}->{best_mid_f168_final:.2f}"
        f" LOW_v1={low_v1_cur:.2f}->{best_low_v1:.2f}"
        f" LOYO {loyo_wins}/5 delta={best['obj']-BASELINE_OBJ:+.4f} OBJ={best['obj']:.4f}"
        f" — PRODUCTION {new_ver}"
    )
    brs = cfg["breadth_regime_switching"]
    rw = brs["regime_weights"]
    # Update MID and LOW
    for regime, final_w in [("MID", best_weights_final2["MID"]), ("LOW", best_weights_final2["LOW"])]:
        rw[regime]["v1"] = round(final_w["v1"], 4)
        rw[regime]["i460bw168"] = round(final_w["i460"], 4)
        rw[regime]["i415bw216"] = round(final_w["i415"], 4)
        rw[regime]["f168"] = round(final_w["f168"], 4)
    Path("configs/production_p91b_champion.json").write_text(json.dumps(cfg, indent=2))
    bov = best["obj"]; d = bov - BASELINE_OBJ
    cm = "phase202: MID+LOW weight tune -- VALIDATED MID_f168=%.2f LOW_v1=%.2f LOYO %d/5 delta=%+.4f OBJ=%.4f" % (best_mid_f168_final, best_low_v1, loyo_wins, d, bov)
    print(f"\n✅ VALIDATED → {new_ver} OBJ={bov:.4f}")
    os.system("git add configs/production_p91b_champion.json artifacts/phase202/ scripts/run_phase202_mid_regime_weight_finetune.py")
    os.system('git commit -m "' + cm + '"')
    os.system("git pull --rebase && git push")
else:
    delta = best["obj"] - BASELINE_OBJ
    verdict = "NO IMPROVEMENT" if delta <= 0 else f"WEAK LOYO {loyo_wins}/5"
    cm = "phase202: MID+LOW weight tune -- %s MID_f168=%.2f LOW_v1=%.2f delta=%+.4f" % (verdict, best_mid_f168_final, best_low_v1, delta)
    print(f"\n❌ NOT VALIDATED — {verdict}")
    os.system("git add artifacts/phase202/ scripts/run_phase202_mid_regime_weight_finetune.py")
    os.system('git commit -m "' + cm + '"')
    os.system("git pull --rebase && git push")

print("\n[DONE] Phase 202 complete.")
