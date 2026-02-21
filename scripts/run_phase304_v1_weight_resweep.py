"""
Phase 304 — Per-Regime V1 Internal Weight Re-Sweep [VECTORIZED]
================================================================
Baseline: v2.49.25, OBJ=5.8110 (Phase 300 vol+disp retune)

V1 internal weights (carry/mom/mean_reversion) last optimized in P298 (parallel).
Since then: P300 vol/disp retune.
The optimal V1 signal composition may have shifted. Re-sweep per regime.

Current (P298 parallel):
  LOW:  wc=0.10, wm=0.40, wmr=0.50
  MID:  wc=0.40, wm=0.40, wmr=0.20
  HIGH: wc=0.50, wm=0.40, wmr=0.10

Approach:
  Pass 1: coarse grid step=0.10 (66 candidates per regime)
  Pass 2: refine grid step=0.05 around winner (~25 candidates)
  Sweep regime-by-regime (coordinate descent)
"""

import os, sys, json, time, subprocess, gc
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

# Timeout handler — save partial results on SIGALRM
_partial: dict = {}
def _on_timeout(signum, frame):
    _partial["timeout"] = True
    out = Path("artifacts/phase304"); out.mkdir(parents=True, exist_ok=True)
    (out / "phase304_report.json").write_text(json.dumps(_partial, indent=2))
    print("\n⏰ TIMEOUT — partial saved.", flush=True)
    sys.exit(0)
_signal.signal(_signal.SIGALRM, _on_timeout)
_signal.alarm(10800)  # 3 hours

SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT",
           "ADAUSDT", "DOGEUSDT", "AVAXUSDT", "DOTUSDT", "LINKUSDT"]
YEAR_RANGES = {
    2021: ("2021-02-01", "2022-01-01"), 2022: ("2022-01-01", "2023-01-01"),
    2023: ("2023-01-01", "2024-01-01"), 2024: ("2024-01-01", "2025-01-01"),
    2025: ("2025-01-01", "2026-01-01"),
}
RNAMES = ["LOW", "MID", "HIGH"]
BRD_LB = 192; PCT_WINDOW = 336; FUND_DISP_PCT = 240
TS_SHORT = 16; TS_LONG = 72; VOL_WINDOW = 168
MIN_DELTA = 0.005; MIN_LOYO = 3

COST_MODEL = cost_model_from_config({
    "fee_rate": 0.0005, "slippage_rate": 0.0003, "cost_multiplier": 1.0,
})
CFG_PATH = ROOT / "configs" / "production_p91b_champion.json"

# V1 backtest cache: (year, wc, wm, wmr) -> returns array
_v1_cache: dict = {}


def make_v1_grid(step=0.10):
    """Generate (w_carry, w_mom, w_mean_reversion) triples summing to 1.0."""
    grid = []
    wc = 0.0
    while wc <= 1.0 + step / 2:
        wm = 0.0
        while wm <= 1.0 - wc + step / 2:
            wmr = round(1.0 - wc - wm, 4)
            if wmr >= -0.001:
                grid.append((round(wc, 2), round(wm, 2), max(0.0, round(wmr, 2))))
            wm = round(wm + step, 4)
        wc = round(wc + step, 4)
    return grid


def make_refine_grid(center, radius=0.15, step=0.05):
    """Generate fine grid around a center point on the simplex."""
    wc0, wm0, _ = center
    grid = []
    wc = max(0, round(wc0 - radius, 4))
    while wc <= min(1.0, wc0 + radius) + step / 2:
        wm = max(0, round(wm0 - radius, 4))
        while wm <= min(1.0 - wc, wm0 + radius) + step / 2:
            wmr = round(1.0 - wc - wm, 4)
            if wmr >= -0.001:
                grid.append((round(wc, 2), round(wm, 2), max(0.0, round(wmr, 2))))
            wm = round(wm + step, 4)
        wc = round(wc + step, 4)
    return grid


def load_config_params():
    cfg = json.load(open(CFG_PATH))
    brs = cfg["breadth_regime_switching"]
    fts = cfg.get("fts_overlay_params", {})
    vol = cfg.get("vol_overlay_params", {})
    disp = cfg.get("disp_overlay_params", {})
    rw = brs["regime_weights"]

    def sdget(d, r, fb):
        return float(d[r]) if isinstance(d, dict) and r in d else float(fb)

    regime_weights = {}
    for rn in RNAMES:
        w = rw[rn]
        regime_weights[rn] = {
            "v1":   float(w.get("v1", 0.0)),
            "i460": float(w.get("i460bw168", w.get("i460", 0.0))),
            "i415": float(w.get("i415bw216", w.get("i415", 0.0))),
            "f168": float(w.get("f168", 0.0)),
        }

    v1prw = cfg.get("v1_per_regime_weights", {})
    def v1p(rn):
        d = v1prw.get(rn, {})
        return {
            "k_per_side": 2,
            "w_carry": float(d.get("w_carry", 0.25)),
            "w_mom": float(d.get("w_mom", 0.50)),
            "w_mean_reversion": float(d.get("w_mean_reversion", 0.25)),
            "momentum_lookback_bars": 312, "mean_reversion_lookback_bars": 84,
            "vol_lookback_bars": 192, "target_gross_leverage": 0.35,
            "rebalance_interval_bars": 60,
        }

    return {
        "p_low": float(brs.get("p_low", 0.30)),
        "p_high": float(brs.get("p_high", 0.62)),
        "regime_weights": regime_weights,
        "fts_rs": {r: sdget(fts.get("per_regime_rs"), r, 0.05) for r in RNAMES},
        "fts_bs": {r: sdget(fts.get("per_regime_bs"), r, 2.0)  for r in RNAMES},
        "fts_rt": {r: sdget(fts.get("per_regime_rt"), r, 0.65) for r in RNAMES},
        "fts_bt": {r: sdget(fts.get("per_regime_bt"), r, 0.25) for r in RNAMES},
        "ts_pct_win": {r: int(fts.get("per_regime_ts_pct_win", {}).get(r, 288)) for r in RNAMES},
        "vol_thr":    {r: sdget(vol.get("per_regime_threshold"), r, 0.50) for r in RNAMES},
        "vol_scale":  {r: sdget(vol.get("per_regime_scale"),     r, 0.40) for r in RNAMES},
        "disp_thr":   {r: sdget(disp.get("per_regime_threshold"), r, 0.70) for r in RNAMES},
        "disp_scale": {r: sdget(disp.get("per_regime_scale"),     r, 0.50) for r in RNAMES},
        "baseline_obj": float(cfg["monitoring"]["expected_performance"]["annual_sharpe_backtest"]),
        "v1_params": {r: v1p(r) for r in RNAMES},
        "config_version": cfg.get("_version", "?"),
    }


def rolling_mean_arr(a, w):
    out = np.full(len(a), np.nan)
    if w <= 0 or len(a) == 0: return out
    cs = np.cumsum(np.where(np.isnan(a), 0.0, a))
    for i in range(w - 1, len(a)):
        out[i] = cs[i] / w if i == w - 1 else (cs[i] - cs[i - w]) / w
    return out


def load_year_data(year):
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

    breadth = np.full(n, 0.5)
    for i in range(BRD_LB, n):
        pos = sum(1 for j in range(len(SYMBOLS))
                  if close_mat[i-BRD_LB, j] > 0 and close_mat[i, j] > close_mat[i-BRD_LB, j])
        breadth[i] = pos / len(SYMBOLS)

    brd_pct = np.full(n, 0.5)
    for i in range(PCT_WINDOW, n):
        brd_pct[i] = float(np.mean(breadth[i-PCT_WINDOW:i] <= breadth[i]))

    # Fixed strategies (non-V1)
    I460_P = {"k_per_side": 4, "lookback_bars": 480, "beta_window_bars": 168,
              "target_gross_leverage": 0.3, "rebalance_interval_bars": 48}
    I415_P = {"k_per_side": 4, "lookback_bars": 415, "beta_window_bars": 144,
              "target_gross_leverage": 0.3, "rebalance_interval_bars": 48}
    F168_P = {"k_per_side": 2, "funding_lookback_bars": 168, "direction": "contrarian",
              "target_gross_leverage": 0.25, "rebalance_interval_bars": 36}
    fixed = {}
    for sk, sn, sp in [("i460", "idio_momentum_alpha", I460_P),
                        ("i415", "idio_momentum_alpha", I415_P),
                        ("f168", "funding_momentum_alpha", F168_P)]:
        res = BacktestEngine(BacktestConfig(costs=COST_MODEL)).run(
            dataset, make_strategy({"name": sn, "params": sp}))
        fixed[sk] = np.array(res.returns)

    return btc_vol, fund_std_pct, ts_raw, brd_pct, fixed, n, dataset


def compute_v1_cached(year, dataset, wc, wm, wmr):
    """Run V1 backtest with caching."""
    key = (year, round(wc, 2), round(wm, 2), round(wmr, 2))
    if key in _v1_cache:
        return _v1_cache[key]
    params = {
        "k_per_side": 2,
        "w_carry": wc, "w_mom": wm, "w_mean_reversion": wmr,
        "momentum_lookback_bars": 312, "mean_reversion_lookback_bars": 84,
        "vol_lookback_bars": 192, "target_gross_leverage": 0.35,
        "rebalance_interval_bars": 60,
    }
    res = BacktestEngine(BacktestConfig(costs=COST_MODEL)).run(
        dataset, make_strategy({"name": "nexus_alpha_v1", "params": params}))
    ret = np.array(res.returns)
    _v1_cache[key] = ret
    return ret


def build_fast_data(all_base, baseline_v1, p):
    """Pre-compute overlays and fixed signal contributions.
    V1 returns will be slotted in per-candidate during sweep."""
    fast = {}
    for yr, base in sorted(all_base.items()):
        btc_vol, fund_std_pct, ts_raw, brd_pct, fixed, n = base
        bl_v1 = baseline_v1[yr]
        ml = min(min(len(bl_v1[r]) for r in RNAMES),
                 min(len(fixed[k]) for k in ["i460", "i415", "f168"]))
        bpct = brd_pct[:ml]; bv = btc_vol[:ml]; fsp = fund_std_pct[:ml]; tr = ts_raw[:ml]

        regime_idx = np.where(bpct < p["p_low"], 0, np.where(bpct > p["p_high"], 2, 1))
        masks = [regime_idx == i for i in range(3)]

        # FTS overlay
        unique_wins = sorted(set(p["ts_pct_win"].values()))
        ts_pct_cache = {}
        for w in unique_wins:
            arr = np.full(ml, 0.5)
            for i in range(w, ml): arr[i] = float(np.mean(tr[i-w:i] <= tr[i]))
            ts_pct_cache[w] = arr

        fts_mult = np.ones(ml)
        for ridx, rn in enumerate(RNAMES):
            m = masks[ridx]
            tsp = ts_pct_cache[p["ts_pct_win"][rn]]
            fts_mult[m & (tsp > p["fts_rt"][rn])] *= p["fts_rs"][rn]
            fts_mult[m & (tsp < p["fts_bt"][rn])] *= p["fts_bs"][rn]

        # Vol + Disp overlay
        vd_mult = np.ones(ml)
        for ridx, rn in enumerate(RNAMES):
            m = masks[ridx]
            vd_mult[m & ~np.isnan(bv) & (bv > p["vol_thr"][rn])] *= p["vol_scale"][rn]
            vd_mult[m & (fsp > p["disp_thr"][rn])] *= p["disp_scale"][rn]

        overlay = fts_mult * vd_mult

        # Pre-scale fixed signals with overlay
        i460_sc = overlay * fixed["i460"][:ml]
        i415_sc = overlay * fixed["i415"][:ml]
        f168_sc = overlay * fixed["f168"][:ml]

        # Baseline V1 returns scaled with overlay (per regime)
        v1_bl_sc = {r: overlay * bl_v1[r][:ml] for r in RNAMES}

        fast[yr] = {
            "masks": masks, "ml": ml, "overlay": overlay,
            "i460_sc": i460_sc, "i415_sc": i415_sc, "f168_sc": f168_sc,
            "v1_bl_sc": v1_bl_sc,
        }
    return fast


def eval_with_v1(p, fast, sweep_regime, v1_new_by_year):
    """Evaluate OBJ with new V1 returns for one regime, baseline for others."""
    yearly = {}
    for yr, fd in sorted(fast.items()):
        masks = fd["masks"]; ml = fd["ml"]
        ens = np.zeros(ml)

        for ridx, rn in enumerate(RNAMES):
            m = masks[ridx]; w = p["regime_weights"][rn]
            if rn == sweep_regime:
                v1_sc = fd["overlay"][:ml] * v1_new_by_year[yr][:ml]
            else:
                v1_sc = fd["v1_bl_sc"][rn]
            ens[m] = (w["v1"] * v1_sc[m]
                      + w["i460"] * fd["i460_sc"][m]
                      + w["i415"] * fd["i415_sc"][m]
                      + w["f168"] * fd["f168_sc"][m])

        r = ens[~np.isnan(ens)]
        yearly[yr] = float(np.mean(r) / np.std(r, ddof=1) * np.sqrt(8760)) if len(r) > 1 else 0.0

    vals = list(yearly.values())
    return float(np.mean(vals) - 0.5 * np.std(vals, ddof=1)), yearly


def eval_baseline(p, fast):
    """Evaluate OBJ with all baseline V1 (for reference)."""
    yearly = {}
    for yr, fd in sorted(fast.items()):
        masks = fd["masks"]; ml = fd["ml"]
        ens = np.zeros(ml)
        for ridx, rn in enumerate(RNAMES):
            m = masks[ridx]; w = p["regime_weights"][rn]
            ens[m] = (w["v1"] * fd["v1_bl_sc"][rn][m]
                      + w["i460"] * fd["i460_sc"][m]
                      + w["i415"] * fd["i415_sc"][m]
                      + w["f168"] * fd["f168_sc"][m])
        r = ens[~np.isnan(ens)]
        yearly[yr] = float(np.mean(r) / np.std(r, ddof=1) * np.sqrt(8760)) if len(r) > 1 else 0.0
    vals = list(yearly.values())
    return float(np.mean(vals) - 0.5 * np.std(vals, ddof=1)), yearly


def main():
    t0 = time.time()
    print("=" * 70)
    print("Phase 304 — Per-Regime V1 Internal Weight Re-Sweep")
    print("Baseline: v2.49.25, OBJ=5.8110")
    print("=" * 70)

    p = load_config_params()
    STORED_OBJ = p["baseline_obj"]
    print(f"Config: v{p['config_version']}  stored OBJ={STORED_OBJ:.4f}")
    print(f"V1 per-regime weights:")
    for rn in RNAMES:
        vp = p["v1_params"][rn]
        print(f"  {rn}: wc={vp['w_carry']:.2f}  wm={vp['w_mom']:.2f}  wmr={vp['w_mean_reversion']:.2f}")

    # [1] Load data
    print("\n[1] Loading data + fixed backtests...", flush=True)
    all_base = {}
    datasets = {}
    for yr in sorted(YEAR_RANGES):
        tw = time.time()
        print(f"  {yr}: ", end="", flush=True)
        result = load_year_data(yr)
        datasets[yr] = result[6]  # keep dataset ref for V1 runs
        all_base[yr] = result[:6]  # (btc_vol, fund_std_pct, ts_raw, brd_pct, fixed, n)
        print(f"done ({int(time.time()-tw)}s)", flush=True)

    # [2] Compute baseline V1 per regime
    print("\n[2] Computing baseline V1 per regime...", flush=True)
    baseline_v1 = {}
    for yr in sorted(YEAR_RANGES):
        ds = datasets[yr]
        v1_per_regime = {}
        for rn in RNAMES:
            vp = p["v1_params"][rn]
            v1_per_regime[rn] = compute_v1_cached(
                yr, ds, vp["w_carry"], vp["w_mom"], vp["w_mean_reversion"])
        baseline_v1[yr] = v1_per_regime
        print(f"  {yr}: done.", flush=True)

    # [3] Build fast data
    print("\n[3] Building fast eval data (overlays pre-applied)...", flush=True)
    fast = build_fast_data(all_base, baseline_v1, p)
    gc.collect()

    base_obj, base_yr = eval_baseline(p, fast)
    print(f"\n  Computed baseline OBJ = {base_obj:.4f}  (stored = {STORED_OBJ:.4f})")
    print(f"  Per year: {' | '.join(f'{yr}={v:.4f}' for yr, v in sorted(base_yr.items()))}")
    _partial["baseline_obj"] = float(base_obj)
    _partial["stored_baseline"] = STORED_OBJ

    # [4] Sweep V1 weights per regime
    print("\n[4] V1 weight sweep per regime...", flush=True)
    coarse_grid = make_v1_grid(step=0.10)
    print(f"  Coarse grid: {len(coarse_grid)} candidates (step=0.10)")

    best_v1 = {}
    for rn in RNAMES:
        vp = p["v1_params"][rn]
        best_v1[rn] = (vp["w_carry"], vp["w_mom"], vp["w_mean_reversion"])

    best_global = base_obj
    regime_results = {}

    for rn in RNAMES:
        tw = time.time()
        print(f"\n  === {rn} regime ===", flush=True)

        # Pass 1: coarse grid
        best_for_regime = best_v1[rn]
        n_eval = 0
        for wc, wm, wmr in coarse_grid:
            # Compute V1 returns for this candidate across all years
            v1_new = {}
            for yr in sorted(YEAR_RANGES):
                v1_new[yr] = compute_v1_cached(yr, datasets[yr], wc, wm, wmr)
            n_eval += 1

            obj, _ = eval_with_v1(p, fast, rn, v1_new)
            if obj > best_global:
                best_global = obj
                best_for_regime = (wc, wm, wmr)

        print(f"    Pass 1 (coarse): best=({best_for_regime[0]:.2f}, {best_for_regime[1]:.2f}, "
              f"{best_for_regime[2]:.2f})  OBJ={best_global:.4f}  "
              f"Δ={best_global-base_obj:+.4f}  [{n_eval} evals, {int(time.time()-tw)}s]", flush=True)

        # Pass 2: refine around winner
        tw2 = time.time()
        refine_grid = make_refine_grid(best_for_regime, radius=0.15, step=0.05)
        n_eval2 = 0
        for wc, wm, wmr in refine_grid:
            v1_new = {}
            for yr in sorted(YEAR_RANGES):
                v1_new[yr] = compute_v1_cached(yr, datasets[yr], wc, wm, wmr)
            n_eval2 += 1

            obj, _ = eval_with_v1(p, fast, rn, v1_new)
            if obj > best_global:
                best_global = obj
                best_for_regime = (wc, wm, wmr)

        print(f"    Pass 2 (refine): best=({best_for_regime[0]:.2f}, {best_for_regime[1]:.2f}, "
              f"{best_for_regime[2]:.2f})  OBJ={best_global:.4f}  "
              f"Δ={best_global-base_obj:+.4f}  [{n_eval2} evals, {int(time.time()-tw2)}s]", flush=True)

        best_v1[rn] = best_for_regime

        # Update fast data with new baseline V1 for this regime
        # (so subsequent regimes see the updated V1)
        for yr in sorted(YEAR_RANGES):
            wc, wm, wmr = best_for_regime
            new_ret = compute_v1_cached(yr, datasets[yr], wc, wm, wmr)
            fast[yr]["v1_bl_sc"][rn] = fast[yr]["overlay"] * new_ret[:fast[yr]["ml"]]

        regime_results[rn] = {
            "w_carry": best_for_regime[0], "w_mom": best_for_regime[1],
            "w_mean_reversion": best_for_regime[2],
            "coarse_evals": n_eval, "refine_evals": n_eval2,
        }
        _partial[f"best_{rn}"] = regime_results[rn]

    # [5] Final evaluation + LOYO
    print("\n[5] Final evaluation + LOYO...", flush=True)

    # Compute final V1 per regime
    final_v1 = {}
    for yr in sorted(YEAR_RANGES):
        final_v1[yr] = {}
        for rn in RNAMES:
            wc, wm, wmr = best_v1[rn]
            final_v1[yr][rn] = compute_v1_cached(yr, datasets[yr], wc, wm, wmr)

    # Rebuild fast data with final V1
    fast_final = build_fast_data(all_base, final_v1, p)
    final_obj, final_yr = eval_baseline(p, fast_final)
    delta = final_obj - base_obj

    print(f"\n  Final OBJ = {final_obj:.4f}  Δ = {delta:+.4f}")
    print(f"  Per year: {' | '.join(f'{yr}={v:.4f}' for yr, v in sorted(final_yr.items()))}")

    wins = 0
    for yr in sorted(final_yr):
        d = final_yr[yr] - base_yr[yr]
        win = d > 0
        if win: wins += 1
        print(f"  {yr}: base={base_yr[yr]:.4f}  cand={final_yr[yr]:.4f}  "
              f"Δ={d:+.4f}  {'✓ WIN' if win else '✗ LOSE'}")

    validated = wins >= MIN_LOYO and delta >= MIN_DELTA
    print(f"\n  OBJ={final_obj:.4f}  Δ={delta:+.4f}  LOYO {wins}/5")
    print(f"\n{'✅ VALIDATED' if validated else '❌ NOT VALIDATED'}")

    # [6] Summary
    print(f"\n  V1 weight changes:")
    for rn in RNAMES:
        old = p["v1_params"][rn]
        nw = best_v1[rn]
        changed = (round(old["w_carry"], 2) != nw[0] or
                   round(old["w_mom"], 2) != nw[1] or
                   round(old["w_mean_reversion"], 2) != nw[2])
        tag = " ← CHANGED" if changed else " (unchanged)"
        print(f"    {rn}: ({old['w_carry']:.2f},{old['w_mom']:.2f},{old['w_mean_reversion']:.2f}) "
              f"→ ({nw[0]:.2f},{nw[1]:.2f},{nw[2]:.2f}){tag}")

    # [7] Save report
    result = {
        "phase": "304", "title": "Per-Regime V1 Weight Re-Sweep",
        "baseline_obj": float(base_obj), "stored_baseline": STORED_OBJ,
        "best_obj": float(final_obj), "delta": float(delta),
        "loyo_wins": wins, "loyo_total": 5, "validated": validated,
        "best_v1_weights": {rn: {"w_carry": best_v1[rn][0], "w_mom": best_v1[rn][1],
                                  "w_mean_reversion": best_v1[rn][2]} for rn in RNAMES},
        "original_v1_weights": {rn: {"w_carry": p["v1_params"][rn]["w_carry"],
                                      "w_mom": p["v1_params"][rn]["w_mom"],
                                      "w_mean_reversion": p["v1_params"][rn]["w_mean_reversion"]}
                                for rn in RNAMES},
        "per_year_baseline": {str(k): v for k, v in base_yr.items()},
        "per_year_final": {str(k): v for k, v in final_yr.items()},
        "regime_results": regime_results,
        "v1_cache_size": len(_v1_cache),
    }
    _partial.update(result)
    out = Path("artifacts/phase304"); out.mkdir(parents=True, exist_ok=True)
    (out / "phase304_report.json").write_text(json.dumps(result, indent=2))
    print(f"\n  Report → artifacts/phase304/phase304_report.json")

    # [8] Update config if validated
    if validated:
        cfg = json.load(open(CFG_PATH))
        v1prw = cfg.setdefault("v1_per_regime_weights", {})
        for rn in RNAMES:
            wc, wm, wmr = best_v1[rn]
            v1prw[rn] = {"w_carry": wc, "w_mom": wm, "w_mean_reversion": wmr}

        old_ver = cfg.get("_version", "2.49.25")
        parts = old_ver.split(".")
        parts[-1] = str(int(parts[-1]) + 1)
        new_ver = ".".join(parts)
        cfg["_version"] = new_ver

        old_vver = cfg.get("version", "v2.49.25").lstrip("v")
        vparts = old_vver.split(".")
        vparts[-1] = str(int(vparts[-1]) + 1)
        cfg["version"] = "v" + ".".join(vparts)

        stored_obj = cfg["monitoring"]["expected_performance"]["annual_sharpe_backtest"]
        reported_obj = round(stored_obj + delta, 4)
        cfg["monitoring"]["expected_performance"]["annual_sharpe_backtest"] = reported_obj
        cfg["obj"] = reported_obj

        with open(CFG_PATH, "w") as f: json.dump(cfg, f, indent=2, ensure_ascii=False)
        print(f"\n  Config → v{new_ver}  OBJ={reported_obj:.4f}")

        subprocess.run(["git", "add", str(CFG_PATH)], cwd=ROOT)
        msg = (f"feat: P304 V1 weight resweep OBJ={reported_obj:.4f} "
               f"LOYO={wins}/5 D={delta:+.4f} [v{new_ver}]")
        subprocess.run(["git", "commit", "-m", msg], cwd=ROOT)
        print(f"  Committed: {msg}")
    else:
        print("\n  No config change (not validated).")

    print(f"\nV1 cache entries: {len(_v1_cache)}")
    print(f"Total runtime: {int(time.time()-t0)}s ({int((time.time()-t0)/60)}m)", flush=True)


main()
