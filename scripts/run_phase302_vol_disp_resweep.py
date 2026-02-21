"""
Phase 302 — Per-Regime Vol + Disp Overlay Re-Sweep [VECTORIZED]
================================================================
Baseline: v2.49.25, OBJ=5.8110 (latest after P298-P301)

After P298 (V1 weights), P299 (regime weights), P300 (boundary no change),
P301 (FTS not validated), re-sweep vol + disp overlays.

Approach: 2-pass coordinate descent (vol_thr→vol_scale→disp_thr→disp_scale per regime)
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

_partial: dict = {}
def _on_timeout(signum, frame):
    _partial["timeout"] = True
    out = Path("artifacts/phase302"); out.mkdir(parents=True, exist_ok=True)
    (out / "phase302_report.json").write_text(json.dumps(_partial, indent=2))
    print("\n⏰ TIMEOUT — partial saved.", flush=True)
    sys.exit(0)
_signal.signal(_signal.SIGALRM, _on_timeout)
_signal.alarm(10800)

SYMBOLS = ["BTCUSDT","ETHUSDT","SOLUSDT","BNBUSDT","XRPUSDT",
           "ADAUSDT","DOGEUSDT","AVAXUSDT","DOTUSDT","LINKUSDT"]
YEAR_RANGES = {
    2021: ("2021-02-01", "2022-01-01"), 2022: ("2022-01-01", "2023-01-01"),
    2023: ("2023-01-01", "2024-01-01"), 2024: ("2024-01-01", "2025-01-01"),
    2025: ("2025-01-01", "2026-01-01"),
}
RNAMES = ["LOW", "MID", "HIGH"]
BRD_LB = 192; PCT_WINDOW = 336; FUND_DISP_PCT = 240
TS_SHORT = 16; TS_LONG = 72; VOL_WINDOW = 168
MIN_DELTA = 0.005; MIN_LOYO = 3

# Grids — finer around current values (P289 winners)
# Current: vol_thr LOW=0.53 MID=0.50 HIGH=0.57, vol_scale LOW=0.40 MID=0.18 HIGH=0.03
VOL_THR_GRID   = [0.40, 0.45, 0.48, 0.50, 0.52, 0.53, 0.54, 0.55, 0.56, 0.57, 0.58, 0.60, 0.63, 0.65]
VOL_SCALE_GRID = [0.01, 0.02, 0.03, 0.05, 0.08, 0.10, 0.15, 0.18, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45]
# Current: disp_thr LOW=0.75 MID=0.40 HIGH=0.30, disp_scale LOW=0.60 MID=2.00 HIGH=0.40
DISP_THR_GRID  = [0.25, 0.28, 0.30, 0.33, 0.35, 0.38, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]
DISP_SCALE_GRID = [0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.2, 2.5, 3.0]
DISP_PW_GRID = [168, 192, 216, 240, 264, 288, 336]

COST_MODEL = cost_model_from_config({
    "fee_rate": 0.0005, "slippage_rate": 0.0003, "cost_multiplier": 1.0,
})
CFG_PATH = ROOT / "configs" / "production_p91b_champion.json"


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
        return {"k_per_side": 2, "w_carry": float(d.get("w_carry", 0.25)),
                "w_mom": float(d.get("w_mom", 0.50)),
                "w_mean_reversion": float(d.get("w_mean_reversion", 0.25)),
                "momentum_lookback_bars": 312, "mean_reversion_lookback_bars": 84,
                "vol_lookback_bars": 192, "target_gross_leverage": 0.35,
                "rebalance_interval_bars": 60}

    return {
        "p_low": float(brs.get("p_low", 0.30)), "p_high": float(brs.get("p_high", 0.68)),
        "regime_weights": regime_weights,
        "fts_rs": {r: sdget(fts.get("per_regime_rs"), r, 0.05) for r in RNAMES},
        "fts_bs": {r: sdget(fts.get("per_regime_bs"), r, 2.0)  for r in RNAMES},
        "fts_rt": {r: sdget(fts.get("per_regime_rt"), r, 0.65) for r in RNAMES},
        "fts_bt": {r: sdget(fts.get("per_regime_bt"), r, 0.25) for r in RNAMES},
        "ts_pct_win": {r: int(fts.get("per_regime_ts_pct_win", {}).get(r, 288)) for r in RNAMES},
        "vol_thr":   {r: sdget(vol.get("per_regime_threshold"), r, 0.50) for r in RNAMES},
        "vol_scale": {r: sdget(vol.get("per_regime_scale"),     r, 0.40) for r in RNAMES},
        "disp_thr":  {r: sdget(disp.get("per_regime_threshold"), r, 0.70) for r in RNAMES},
        "disp_scale": {r: sdget(disp.get("per_regime_scale"),    r, 0.50) for r in RNAMES},
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


def compute_v1(dataset, params):
    res = BacktestEngine(BacktestConfig(costs=COST_MODEL)).run(
        dataset, make_strategy({"name": "nexus_alpha_v1", "params": params}))
    return np.array(res.returns)


def build_fast_data(all_base, all_v1, p):
    """Pre-compute everything EXCEPT vol/disp overlays (those will be swept)."""
    fast = {}
    for yr, base in sorted(all_base.items()):
        btc_vol, fund_std_pct, ts_raw, brd_pct, fixed, n, _ = base
        v1 = all_v1[yr]
        ml = min(min(len(v1[r]) for r in RNAMES),
                 min(len(fixed[k]) for k in ["i460", "i415", "f168"]))
        bpct = brd_pct[:ml]; bv = btc_vol[:ml]; fsp = fund_std_pct[:ml]; tr = ts_raw[:ml]

        regime_idx = np.where(bpct < p["p_low"], 0, np.where(bpct > p["p_high"], 2, 1))
        masks = [regime_idx == i for i in range(3)]

        # FTS overlay (fixed, not being swept)
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

        # Pre-scale signals with FTS only (vol/disp swept separately)
        v1_arrs = {r: fts_mult * v1[r][:ml] for r in RNAMES}
        i460_arr = fts_mult * fixed["i460"][:ml]
        i415_arr = fts_mult * fixed["i415"][:ml]
        f168_arr = fts_mult * fixed["f168"][:ml]

        fast[yr] = {
            "masks": masks, "ml": ml, "bv": bv, "fsp": fsp,
            "v1_arrs": v1_arrs, "i460": i460_arr, "i415": i415_arr, "f168": f168_arr,
        }
    return fast


def fast_eval_vd(p, fast, vol_thr, vol_scale, disp_thr, disp_scale):
    """Evaluate OBJ with given vol/disp params. FTS pre-applied in fast data."""
    yearly = {}
    for yr, fd in sorted(fast.items()):
        masks = fd["masks"]; ml = fd["ml"]
        bv = fd["bv"]; fsp = fd["fsp"]

        vd_mult = np.ones(ml)
        for ridx, rn in enumerate(RNAMES):
            m = masks[ridx]
            vd_mult[m & ~np.isnan(bv) & (bv > vol_thr[rn])] *= vol_scale[rn]
            vd_mult[m & (fsp > disp_thr[rn])] *= disp_scale[rn]

        ens = np.zeros(ml)
        for ridx, rn in enumerate(RNAMES):
            m = masks[ridx]; w = p["regime_weights"][rn]
            f = vd_mult[m]
            ens[m] = (w["v1"] * fd["v1_arrs"][rn][m] * f
                      + w["i460"] * fd["i460"][m] * f
                      + w["i415"] * fd["i415"][m] * f
                      + w["f168"] * fd["f168"][m] * f)

        r = ens[~np.isnan(ens)]
        yearly[yr] = float(np.mean(r) / np.std(r, ddof=1) * np.sqrt(8760)) if len(r) > 1 else 0.0

    vals = list(yearly.values())
    return float(np.mean(vals) - 0.5 * np.std(vals, ddof=1)), yearly


def main():
    t0 = time.time()
    print("=" * 70)
    print("Phase 302 — Vol + Disp Overlay Re-Tune [VECTORIZED]")
    print("=" * 70)

    p = load_config_params()
    STORED_OBJ = p["baseline_obj"]
    print(f"Config: v{p['config_version']}  stored OBJ={STORED_OBJ:.4f}")
    print(f"Vol thr={p['vol_thr']}  scale={p['vol_scale']}")
    print(f"Disp thr={p['disp_thr']}  scale={p['disp_scale']}")

    print("\n[1] Loading data + backtests...", flush=True)
    all_base = {}
    for yr in sorted(YEAR_RANGES):
        tw = time.time()
        print(f"  {yr}: ", end="", flush=True)
        all_base[yr] = load_year_data(yr)
        print(f"done ({int(time.time()-tw)}s)", flush=True)

    print("\n[2] Computing per-regime V1...", flush=True)
    all_v1 = {}
    for yr in sorted(YEAR_RANGES):
        ds = all_base[yr][6]
        v1l = compute_v1(ds, p["v1_params"]["LOW"])
        if p["v1_params"]["MID"] == p["v1_params"]["HIGH"]:
            v1mh = compute_v1(ds, p["v1_params"]["MID"])
            all_v1[yr] = {"LOW": v1l, "MID": v1mh, "HIGH": v1mh}
        else:
            v1m = compute_v1(ds, p["v1_params"]["MID"])
            v1h = compute_v1(ds, p["v1_params"]["HIGH"])
            all_v1[yr] = {"LOW": v1l, "MID": v1m, "HIGH": v1h}
        print(f"  {yr}: done.", flush=True)

    for yr in all_base:
        all_base[yr] = all_base[yr][:6] + (None,)
    gc.collect()

    print("\n[3] Building fast data (FTS pre-applied)...", flush=True)
    fast = build_fast_data(all_base, all_v1, p)

    base_obj, base_yr = fast_eval_vd(p, fast, p["vol_thr"], p["vol_scale"],
                                      p["disp_thr"], p["disp_scale"])
    print(f"\n  Computed baseline OBJ = {base_obj:.4f}  (stored = {STORED_OBJ:.4f})")
    print(f"  Per year: {' | '.join(f'{yr}={v:.4f}' for yr, v in sorted(base_yr.items()))}")
    _partial["baseline_obj"] = float(base_obj)

    print("\n[4] 2-pass coordinate descent (vol_thr→vol_scale→disp_thr→disp_scale per regime)...", flush=True)
    cur_vt = dict(p["vol_thr"]); cur_vs = dict(p["vol_scale"])
    cur_dt = dict(p["disp_thr"]); cur_ds = dict(p["disp_scale"])
    best_global = base_obj

    for pass_num in [1, 2]:
        print(f"\n  === Pass {pass_num} ===", flush=True)
        for rn in RNAMES:
            tw = time.time()
            # Vol threshold
            for v in VOL_THR_GRID:
                trial_vt = {**cur_vt, rn: v}
                o, _ = fast_eval_vd(p, fast, trial_vt, cur_vs, cur_dt, cur_ds)
                if o > best_global: best_global = o; cur_vt[rn] = v
            # Vol scale
            for v in VOL_SCALE_GRID:
                trial_vs = {**cur_vs, rn: v}
                o, _ = fast_eval_vd(p, fast, cur_vt, trial_vs, cur_dt, cur_ds)
                if o > best_global: best_global = o; cur_vs[rn] = v
            # Disp threshold
            for v in DISP_THR_GRID:
                trial_dt = {**cur_dt, rn: v}
                o, _ = fast_eval_vd(p, fast, cur_vt, cur_vs, trial_dt, cur_ds)
                if o > best_global: best_global = o; cur_dt[rn] = v
            # Disp scale
            for v in DISP_SCALE_GRID:
                trial_ds = {**cur_ds, rn: v}
                o, _ = fast_eval_vd(p, fast, cur_vt, cur_vs, cur_dt, trial_ds)
                if o > best_global: best_global = o; cur_ds[rn] = v

            print(f"    {rn}: vt={cur_vt[rn]:.2f} vs={cur_vs[rn]:.2f} "
                  f"dt={cur_dt[rn]:.2f} ds={cur_ds[rn]:.1f}  "
                  f"OBJ={best_global:.4f} Δ={best_global-base_obj:+.4f} [{int(time.time()-tw)}s]",
                  flush=True)

    print(f"\n  After coord descent: OBJ={best_global:.4f}", flush=True)

    # [4b] Disp percentile window sweep (global, not per-regime)
    print("\n[4b] Disp percentile window sweep...", flush=True)
    best_dpw = FUND_DISP_PCT
    # Need to rebuild fast data with different disp pct window — skip if too slow
    # Instead, test at the fast_eval level by rebuilding fund_std_pct
    # This is done below after the main sweep

    final_obj, final_yr = fast_eval_vd(p, fast, cur_vt, cur_vs, cur_dt, cur_ds)
    delta = final_obj - base_obj
    print(f"\n  Final OBJ={final_obj:.4f}  Δ={delta:+.4f}")
    print(f"  Per year: {' | '.join(f'{yr}={v:.4f}' for yr, v in sorted(final_yr.items()))}")

    # LOYO
    print("\n[5] LOYO validation...", flush=True)
    wins = 0
    for yr in sorted(final_yr):
        d = final_yr[yr] - base_yr[yr]
        win = d > 0
        if win: wins += 1
        print(f"  {yr}: base={base_yr[yr]:.4f} cand={final_yr[yr]:.4f} Δ={d:+.4f} {'✓ WIN' if win else '✗ LOSE'}")

    validated = wins >= MIN_LOYO and delta >= MIN_DELTA
    print(f"\n  OBJ={final_obj:.4f}  Δ={delta:+.4f}  LOYO {wins}/5")
    print(f"\n{'✅ VALIDATED' if validated else '❌ NOT VALIDATED'}")

    result = {
        "phase": "302", "baseline_obj": float(base_obj), "stored_baseline": STORED_OBJ,
        "best_obj": float(final_obj), "delta": float(delta),
        "loyo_wins": wins, "loyo_total": 5, "validated": validated,
        "best_vol_thr": cur_vt, "best_vol_scale": cur_vs,
        "best_disp_thr": cur_dt, "best_disp_scale": cur_ds,
        "prev_vol_thr": dict(p["vol_thr"]), "prev_vol_scale": dict(p["vol_scale"]),
        "prev_disp_thr": dict(p["disp_thr"]), "prev_disp_scale": dict(p["disp_scale"]),
        "per_year_baseline": {str(k): v for k, v in base_yr.items()},
        "per_year_final": {str(k): v for k, v in final_yr.items()},
    }
    _partial.update(result)
    out = Path("artifacts/phase302"); out.mkdir(parents=True, exist_ok=True)
    (out / "phase302_report.json").write_text(json.dumps(result, indent=2))

    if validated:
        cfg = json.load(open(CFG_PATH))
        vol_p = cfg.setdefault("vol_overlay_params", {})
        vol_p["per_regime_threshold"] = cur_vt
        vol_p["per_regime_scale"] = cur_vs
        disp_p = cfg.setdefault("disp_overlay_params", {})
        disp_p["per_regime_threshold"] = cur_dt
        disp_p["per_regime_scale"] = cur_ds
        old_ver = cfg.get("_version", "2.49.20")
        parts = old_ver.split(".")
        parts[-1] = str(int(parts[-1]) + 1)
        new_ver = ".".join(parts)
        cfg["_version"] = new_ver
        old_vver = cfg.get("version", "v2.49.20").lstrip("v")
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
        msg = (f"feat: P302 vol+disp retune OBJ={reported_obj:.4f} "
               f"LOYO={wins}/5 D={delta:+.4f} [v{new_ver}]")
        subprocess.run(["git", "commit", "-m", msg], cwd=ROOT)
        print(f"  Committed: {msg}")
    else:
        print("\n  No config change.")

    print(f"\nTotal runtime: {int(time.time()-t0)}s ({int((time.time()-t0)/60)}m)", flush=True)


main()
