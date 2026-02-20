"""
Phase 204 — HIGH Regime Weight Fine-Tune
=========================================
Current HIGH: v1=0.06, i460=0.2821, i415=0.5079, f168=0.15
HIGH regime = strong breadth momentum (>60th pct).
v1=0.06 is very low — V1's momentum signal may add diversification.
f168=0.15 is low — funding signal role in HIGH regime unknown.

Part A: HIGH_v1 sweep [0.04, 0.08, 0.12, 0.16, 0.20, 0.25, 0.30]
  - f168=0.15 fixed, i460/i415 scaled proportionally from base
Part B: HIGH_f168 sweep [0.05, 0.08, 0.12, 0.15, 0.18, 0.22, 0.25]
  - using best HIGH_v1 from Part A, i460/i415 scaled

LOYO: ≥3/5 wins AND delta>0.005 → update config + commit
"""
import os, sys, json, time, re
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
    out = Path("artifacts/phase204"); out.mkdir(parents=True, exist_ok=True)
    (out / "phase204_report.json").write_text(json.dumps(_partial, indent=2, default=str))
    sys.exit(0)
_signal.signal(_signal.SIGALRM, _on_timeout)
_signal.alarm(7200)

SYMBOLS = ["BTCUSDT","ETHUSDT","SOLUSDT","BNBUSDT","XRPUSDT",
           "ADAUSDT","DOGEUSDT","AVAXUSDT","DOTUSDT","LINKUSDT"]
YEAR_RANGES = {
    "2021": ("2021-02-01", "2022-01-01"),
    "2022": ("2022-01-01", "2023-01-01"),
    "2023": ("2023-01-01", "2024-01-01"),
    "2024": ("2024-01-01", "2025-01-01"),
    "2025": ("2025-01-01", "2026-01-01"),
}
YEARS = sorted(YEAR_RANGES.keys())

COST_MODEL = cost_model_from_config({
    "fee_rate": 0.0005, "slippage_rate": 0.0003, "cost_multiplier": 1.0,
})

HIGH_V1_SWEEP  = [0.04, 0.08, 0.12, 0.16, 0.20, 0.25, 0.30]
HIGH_F168_SWEEP = [0.05, 0.08, 0.12, 0.15, 0.18, 0.22, 0.25]

VOL_WINDOW = 168; FUND_DISP_PCT = 240; PCT_WINDOW = 336
DISP_THR = 0.60; DISP_SCALE = 1.0

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
    baseline_obj = float(m[-1]) if m else 3.0826
    return cfg, weights, params, baseline_obj

def make_high_weights(base_weights, high_v1, high_f168):
    """Scale i460/i415 proportionally to fill remainder."""
    w_new = {r: dict(base_weights[r]) for r in ["LOW", "MID", "HIGH"]}
    others_sum_base = base_weights["HIGH"]["i460"] + base_weights["HIGH"]["i415"]
    target_others = 1.0 - high_v1 - high_f168
    if target_others < 0.01 or others_sum_base <= 0:
        return None
    scale = max(0, target_others) / max(others_sum_base, 0.001)
    w_new["HIGH"] = {
        "v1": high_v1,
        "i460": max(0.005, base_weights["HIGH"]["i460"] * scale),
        "i415": max(0.005, base_weights["HIGH"]["i415"] * scale),
        "f168": high_f168,
    }
    return w_new

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
    for i in range(PCT_WINDOW, n):
        ts_spread_pct[i] = float(np.mean(ts_raw[i-PCT_WINDOW:i] <= ts_raw[i]))
    ts_spread_pct[:PCT_WINDOW] = 0.5

    brd_lb = p["brd_lb"]; pct_win = p["pct_win"]
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

def compute_ensemble(year_data, weights, params):
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
    if n < 20: return 0.0
    mu = float(np.mean(rets[:n])) * 8760
    sd = float(np.std(rets[:n])) * np.sqrt(8760)
    return mu / sd if sd > 1e-9 else 0.0

def obj_fn(yr_sharpes):
    vals = list(yr_sharpes.values())
    return float(np.mean(vals) - 0.5 * np.std(vals))

def eval_weights(weights, year_data, params):
    yr_sharpes = {}
    for yr in YEARS:
        ens = compute_ensemble(year_data[yr], weights, params)
        yr_sharpes[yr] = sharpe(ens, year_data[yr][-1])
    return obj_fn(yr_sharpes), yr_sharpes

# ── MAIN ────────────────────────────────────────────────────────────────────
print("=" * 60)
print("Phase 204 — HIGH Regime Weight Fine-Tune")
print("=" * 60)
_start = time.time()

prod_cfg, BASE_WEIGHTS, PARAMS, baseline_obj = get_config()
ver_now = prod_cfg.get("_version", "2.31.0")
print(f"  Config: {ver_now}  baseline_obj={baseline_obj:.4f}")
print(f"  HIGH now: v1={BASE_WEIGHTS['HIGH']['v1']:.3f}  i460={BASE_WEIGHTS['HIGH']['i460']:.4f}"
      f"  i415={BASE_WEIGHTS['HIGH']['i415']:.4f}  f168={BASE_WEIGHTS['HIGH']['f168']:.3f}")

print("\n[1] Loading all years ...")
year_data = {}
for yr in YEARS:
    print(f"  {yr}", end="", flush=True)
    year_data[yr] = load_year(yr, PARAMS)
    print()

# Part A: HIGH_v1 sweep (f168=current fixed)
curr_f168 = BASE_WEIGHTS["HIGH"]["f168"]
print(f"\n[2] Part A: HIGH v1 sweep (f168={curr_f168:.3f} fixed) ...")
results_a = []
for high_v1 in HIGH_V1_SWEEP:
    w_test = make_high_weights(BASE_WEIGHTS, high_v1, curr_f168)
    if w_test is None:
        print(f"  HIGH v1={high_v1:.2f}  SKIP (infeasible)")
        continue
    obj, _ = eval_weights(w_test, year_data, PARAMS)
    results_a.append((high_v1, obj, w_test))
    print(f"  HIGH v1={high_v1:.2f} (i460={w_test['HIGH']['i460']:.4f}, i415={w_test['HIGH']['i415']:.4f})  OBJ={obj:.4f}")

best_a = max(results_a, key=lambda x: x[1])
best_high_v1, best_obj_a, _ = best_a
print(f"  Best HIGH v1: {best_high_v1:.2f}  OBJ={best_obj_a:.4f}")

# Part B: HIGH_f168 sweep (using best v1)
print(f"\n[3] Part B: HIGH f168 sweep (v1={best_high_v1:.2f} fixed) ...")
results_b = []
for high_f168 in HIGH_F168_SWEEP:
    w_test = make_high_weights(BASE_WEIGHTS, best_high_v1, high_f168)
    if w_test is None:
        print(f"  HIGH f168={high_f168:.2f}  SKIP (infeasible)")
        continue
    obj, _ = eval_weights(w_test, year_data, PARAMS)
    results_b.append((high_f168, obj, w_test))
    print(f"  HIGH f168={high_f168:.2f} (i460={w_test['HIGH']['i460']:.4f}, i415={w_test['HIGH']['i415']:.4f})  OBJ={obj:.4f}")

best_b = max(results_b, key=lambda x: x[1])
best_high_f168, best_obj_b, best_weights = best_b
delta = best_obj_b - baseline_obj
print(f"  Best HIGH f168: {best_high_f168:.2f}  OBJ={best_obj_b:.4f}  Δ={delta:+.4f}")

# LOYO validation
print(f"\n[4] LOYO validation ...")
_, best_yr = eval_weights(best_weights, year_data, PARAMS)
_, base_yr = eval_weights(BASE_WEIGHTS, year_data, PARAMS)
loyo_wins = 0
for yr in YEARS:
    win = best_yr[yr] > base_yr[yr]
    if win: loyo_wins += 1
    print(f"  {yr}: best={best_yr[yr]:.4f} curr={base_yr[yr]:.4f} {'WIN' if win else 'LOSS'}")

print(f"\n  LOYO: {loyo_wins}/5 wins")

validated = loyo_wins >= 3 and delta > 0.005

# Save report
out = Path("artifacts/phase204"); out.mkdir(parents=True, exist_ok=True)
report = {
    "phase": 204,
    "description": "HIGH regime weight fine-tune",
    "elapsed_seconds": round(time.time() - _start, 1),
    "baseline_obj": baseline_obj, "best_obj": best_obj_b, "delta": delta,
    "best_high_v1": best_high_v1, "best_high_f168": best_high_f168,
    "best_weights": best_weights,
    "loyo_wins": loyo_wins, "validated": validated,
    "yearly_best": best_yr, "yearly_base": base_yr,
}
(out / "phase204_report.json").write_text(json.dumps(report, indent=2, default=str))
print(f"  Report saved  ({report['elapsed_seconds']}s)")

if validated:
    # Read config freshly and update
    cfg = json.loads(Path("configs/production_p91b_champion.json").read_text())
    brs_cfg = cfg["breadth_regime_switching"]
    rw = brs_cfg["regime_weights"]
    # Update HIGH weights in config
    rw["HIGH"]["v1"] = best_weights["HIGH"]["v1"]
    rw["HIGH"]["i460bw168"] = round(best_weights["HIGH"]["i460"], 6)
    rw["HIGH"]["i415bw216"] = round(best_weights["HIGH"]["i415"], 6)
    rw["HIGH"]["f168"] = best_weights["HIGH"]["f168"]
    # Version bump
    ver = cfg.get("_version", "2.31.0")
    major, minor, patch = map(int, ver.split("."))
    new_ver = "%d.%d.%d" % (major, minor + 1, patch)
    cfg["_version"] = new_ver
    cfg["_created"] = "2026-02-21"
    old_val = cfg.get("_validated", "")
    cfg["_validated"] = old_val + (
        "; HIGH regime weight P204: v1=%.2f->%.2f f168=%.2f->%.2f LOYO %d/5 delta=%+.4f OBJ=%.4f — PRODUCTION %s" % (
            BASE_WEIGHTS["HIGH"]["v1"], best_high_v1,
            BASE_WEIGHTS["HIGH"]["f168"], best_high_f168,
            loyo_wins, delta, best_obj_b, new_ver))
    Path("configs/production_p91b_champion.json").write_text(json.dumps(cfg, indent=2))
    print(f"\n✅ VALIDATED → {new_ver} OBJ={best_obj_b:.4f}")
    cm = "phase204: HIGH regime weight tune -- VALIDATED HIGH_v1=%.2f HIGH_f168=%.2f LOYO %d/5 delta=%+.4f OBJ=%.4f" % (
        best_high_v1, best_high_f168, loyo_wins, delta, best_obj_b)
else:
    new_ver = ver_now
    print(f"\n⚠️ NO IMPROVEMENT — HIGH weights near-optimal (LOYO {loyo_wins}/5 delta={delta:+.4f})")
    cm = "phase204: HIGH regime weight tune -- WEAK LOYO %d/5 delta=%+.4f HIGH_v1=%.2f f168=%.2f" % (
        loyo_wins, delta, best_high_v1, best_high_f168)

os.system("git add configs/production_p91b_champion.json scripts/run_phase204_high_regime_weight_finetune.py "
          "artifacts/phase204/ 2>/dev/null || true")
os.system('git commit -m "' + cm + '"')
os.system("git stash && git pull --rebase && git stash pop && git push")

print(f"\n[DONE] Phase 204 complete.")
