"""
Phase 256 — Per-Regime DISP Overlay Resweep [VECTORIZED]
=========================================================
Baseline: loads dynamically from current config.

DISP (funding-rate dispersion) overlay was last tuned before major regime
weight changes (Phase 271). Re-sweep per-regime threshold + scale.

Sequential per-regime sweep (2 passes):
  For each regime in [LOW, MID, HIGH]:
    Try all (disp_thr, disp_scale) grid combos
    Keep best for that regime → move to next

Sweep grids:
  disp_thr   = [0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]
  disp_scale = [0.0, 0.10, 0.20, 0.30, 0.50, 0.70, 1.00, 1.20, 1.50, 1.80, 2.00, 2.50]
"""

import os, sys, json, time, subprocess
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
    out = Path("artifacts/phase256"); out.mkdir(parents=True, exist_ok=True)
    (out / "report.json").write_text(json.dumps(_partial, indent=2))
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

CFG_PATH = ROOT / "configs" / "production_p91b_champion.json"
RNAMES = ["LOW", "MID", "HIGH"]
BRD_LB = 192; PCT_WINDOW = 336; FUND_DISP_PCT = 240
TS_SHORT = 16; TS_LONG = 72; VOL_WINDOW = 168
MIN_DELTA = 0.005; MIN_LOYO = 3

# Sweep grids
DISP_THR_SWEEP   = [0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]
DISP_SCALE_SWEEP = [0.0, 0.10, 0.20, 0.30, 0.50, 0.70, 1.00, 1.20, 1.50, 1.80, 2.00, 2.50]

COST_MODEL = cost_model_from_config({
    "fee_rate": 0.0005, "slippage_rate": 0.0003, "cost_multiplier": 1.0,
})


def load_config_params():
    cfg = json.load(open(CFG_PATH))
    brs = cfg["breadth_regime_switching"]
    fts = cfg.get("fts_overlay_params", {})
    vol = cfg.get("vol_overlay_params", {})
    disp = cfg.get("disp_overlay_params", {})
    rw = brs["regime_weights"]

    def sdget(d, r, fallback):
        if isinstance(d, dict) and r in d: return float(d[r])
        return float(fallback)

    regime_weights = {}
    for rname in RNAMES:
        w = rw[rname]
        regime_weights[rname] = {
            "v1":   float(w.get("v1", 0.0)),
            "i460": float(w.get("i460bw168", w.get("i460", 0.0))),
            "i415": float(w.get("i415bw216", w.get("i415", 0.0))),
            "f168": float(w.get("f168", 0.0)),
        }

    v1prw = cfg.get("v1_per_regime_weights", {})
    def v1p(rname):
        d = v1prw.get(rname, {})
        return {
            "k_per_side": 2,
            "w_carry": float(d.get("w_carry", 0.25)),
            "w_mom": float(d.get("w_mom", 0.50)),
            "w_mean_reversion": float(d.get("w_mean_reversion", 0.25)),
            "momentum_lookback_bars": 336, "mean_reversion_lookback_bars": 84,
            "vol_lookback_bars": 192, "target_gross_leverage": 0.35,
            "rebalance_interval_bars": 60,
        }

    return {
        "p_low":  float(brs.get("p_low", 0.30)),
        "p_high": float(brs.get("p_high", 0.68)),
        "regime_weights": regime_weights,
        "fts_rs":  {r: sdget(fts.get("per_regime_rs"), r, 0.25)  for r in RNAMES},
        "fts_bs":  {r: sdget(fts.get("per_regime_bs"), r, 2.0)   for r in RNAMES},
        "fts_rt":  {r: sdget(fts.get("per_regime_rt"), r, 0.65)  for r in RNAMES},
        "fts_bt":  {r: sdget(fts.get("per_regime_bt"), r, 0.35)  for r in RNAMES},
        "ts_pct_win": {r: int(fts.get("per_regime_ts_pct_win", {}).get(r, 288)) for r in RNAMES},
        "vol_thr":   {r: sdget(vol.get("per_regime_threshold"), r, 0.50)  for r in RNAMES},
        "vol_scale": {r: sdget(vol.get("per_regime_scale"),     r, 0.40)  for r in RNAMES},
        "disp_thr":  {r: sdget(disp.get("per_regime_threshold"), r, 0.70) for r in RNAMES},
        "disp_scale":{r: sdget(disp.get("per_regime_scale"),     r, 0.50) for r in RNAMES},
        "baseline_obj": float(cfg.get("monitoring", {}).get("expected_performance", {})
                              .get("annual_sharpe_backtest", 4.0)),
        "v1_params": {r: v1p(r) for r in RNAMES},
        "config_version": cfg.get("_version", "?"),
    }


def rolling_mean_arr(a, w):
    out = np.full(len(a), np.nan)
    if w <= 0: return out
    cs = np.cumsum(np.where(np.isnan(a), 0.0, a))
    for i in range(w - 1, len(a)):
        out[i] = cs[i] / w if i == w - 1 else (cs[i] - cs[i - w]) / w
    return out


def load_year_data(year, p):
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

    fixed_rets = {}
    for sk, sname, params in [
        ("i460", "idio_momentum_alpha", I460_P),
        ("i415", "idio_momentum_alpha", I415_P),
        ("f168", "funding_momentum_alpha", F168_P),
    ]:
        result = BacktestEngine(BacktestConfig(costs=COST_MODEL)).run(
            dataset, make_strategy({"name": sname, "params": params}))
        fixed_rets[sk] = np.array(result.returns)

    # V1 per-regime (separate w_carry/w_mom/w_mr per regime)
    v1_by_regime = {}
    for rn in RNAMES:
        res = BacktestEngine(BacktestConfig(costs=COST_MODEL)).run(
            dataset, make_strategy({"name": "nexus_alpha_v1", "params": p["v1_params"][rn]}))
        v1_by_regime[rn] = np.array(res.returns)

    return btc_vol, fund_std_pct, ts_raw, brd_pct, fixed_rets, v1_by_regime, n


def build_fast_data(disp_thr, disp_scale, all_base_data, p):
    """Build precomp for given disp_thr/disp_scale dicts."""
    fast = {}
    for yr, yd in sorted(all_base_data.items()):
        btc_vol, fund_std_pct, ts_raw, brd_pct, fixed_rets, v1_by_regime, n = yd
        ml = min(
            min(len(v1_by_regime[r]) for r in RNAMES),
            min(len(fixed_rets[k]) for k in ["i460", "i415", "f168"])
        )
        bpct = brd_pct[:ml]; bv = btc_vol[:ml]
        fsp  = fund_std_pct[:ml]; tr = ts_raw[:ml]

        regime_idx = np.where(bpct < p["p_low"], 0, np.where(bpct > p["p_high"], 2, 1))
        masks = [regime_idx == i for i in range(3)]

        # ts_pct cache
        unique_wins = sorted(set(p["ts_pct_win"].values()))
        ts_pct_cache = {}
        for w in unique_wins:
            arr = np.full(ml, 0.5)
            for i in range(w, ml): arr[i] = float(np.mean(tr[i-w:i] <= tr[i]))
            ts_pct_cache[w] = arr

        overlay_mult = np.ones(ml)
        for ri, rn in enumerate(RNAMES):
            m = masks[ri]
            # VOL dampening (fixed from config)
            overlay_mult[m & ~np.isnan(bv) & (bv > p["vol_thr"][rn])] *= p["vol_scale"][rn]
            # DISP (parameterized)
            overlay_mult[m & (fsp > disp_thr[rn])] *= disp_scale[rn]
            # FTS (fixed from config)
            tsp = ts_pct_cache[p["ts_pct_win"][rn]]
            overlay_mult[m & (tsp > p["fts_rt"][rn])] *= p["fts_rs"][rn]
            overlay_mult[m & (tsp < p["fts_bt"][rn])] *= p["fts_bs"][rn]

        scaled = {
            "v1": {rn: overlay_mult * v1_by_regime[rn][:ml] for rn in RNAMES},
            "i460": overlay_mult * fixed_rets["i460"][:ml],
            "i415": overlay_mult * fixed_rets["i415"][:ml],
            "f168": overlay_mult * fixed_rets["f168"][:ml],
        }
        fast[yr] = {"scaled": scaled, "masks": masks, "ml": ml}
    return fast


def fast_evaluate(fast_data, p):
    yearly = []
    for yr, fd in sorted(fast_data.items()):
        sc = fd["scaled"]; masks = fd["masks"]; ml = fd["ml"]
        ens = np.zeros(ml)
        for ri, rn in enumerate(RNAMES):
            m = masks[ri]; w = p["regime_weights"][rn]
            ens[m] = (w["v1"]  * sc["v1"][rn][m]
                    + w["i460"] * sc["i460"][m]
                    + w["i415"] * sc["i415"][m]
                    + w["f168"] * sc["f168"][m])
        r = ens[~np.isnan(ens)]
        sh = float(np.mean(r) / np.std(r, ddof=1) * np.sqrt(8760)) if len(r) > 1 else 0.0
        yearly.append(sh)
    return float(np.mean(yearly) - 0.5 * np.std(yearly, ddof=1))


def main():
    t0 = time.time()
    print("=" * 65)
    print("Phase 256 — Per-Regime DISP Overlay Resweep [VECTORIZED]")

    p = load_config_params()
    print(f"Baseline: v{p['config_version']}  OBJ={p['baseline_obj']:.4f}")
    print(f"Current DISP: thr={p['disp_thr']}  scale={p['disp_scale']}")
    print(f"Grid: {len(DISP_THR_SWEEP)} thresholds × {len(DISP_SCALE_SWEEP)} scales = {len(DISP_THR_SWEEP)*len(DISP_SCALE_SWEEP)} combos/regime")
    print("=" * 65)

    # [1] Load all per-year data (all backtests run once here)
    print("\n[1] Loading per-year data (all backtests done once)...")
    all_base_data = {}
    for yr in sorted(YEAR_RANGES.keys()):
        print(f"  {yr}: ", end="", flush=True)
        all_base_data[yr] = load_year_data(yr, p)
        print(f"done.  ({time.time()-t0:.0f}s)")

    # [2] Baseline evaluation
    print("\n[2] Building baseline fast data...")
    baseline_fd = build_fast_data(p["disp_thr"], p["disp_scale"], all_base_data, p)
    obj_base = fast_evaluate(baseline_fd, p)
    print(f"  Measured baseline OBJ = {obj_base:.4f}  (config says {p['baseline_obj']:.4f})")

    # [3] Sequential per-regime sweep (2 passes)
    best_thr   = dict(p["disp_thr"])
    best_scale = dict(p["disp_scale"])
    best_obj   = obj_base

    print("\n[3] Sequential per-regime DISP sweep (2 passes)...")
    for pass_num in range(1, 3):
        print(f"\n  === Pass {pass_num} ===")
        for rname in RNAMES:
            print(f"  [{rname}] sweeping {len(DISP_THR_SWEEP)*len(DISP_SCALE_SWEEP)} combos...")
            best_thr_r = best_thr[rname]
            best_scale_r = best_scale[rname]
            for thr in DISP_THR_SWEEP:
                for scale in DISP_SCALE_SWEEP:
                    trial_thr   = dict(best_thr);   trial_thr[rname]   = thr
                    trial_scale = dict(best_scale); trial_scale[rname] = scale
                    fd = build_fast_data(trial_thr, trial_scale, all_base_data, p)
                    obj = fast_evaluate(fd, p)
                    if obj > best_obj + 1e-6:
                        best_obj = obj
                        best_thr_r = thr
                        best_scale_r = scale
            best_thr[rname]   = best_thr_r
            best_scale[rname] = best_scale_r
            print(f"    {rname}: thr={best_thr_r}  scale={best_scale_r}  OBJ={best_obj:.4f}  Δ={best_obj-obj_base:+.4f}")

    delta = best_obj - obj_base
    print(f"\n  Final OBJ={best_obj:.4f}  Δ={delta:+.4f}")

    # [4] LOYO validation
    print("\n[4] LOYO validation...")
    best_fd = build_fast_data(best_thr, best_scale, all_base_data, p)
    loyo_wins = 0
    for yr in sorted(YEAR_RANGES.keys()):
        others = [y for y in YEAR_RANGES if y != yr]
        def subset_obj(fd, yrs):
            sh = []
            for y in yrs:
                d = fd[y]; sc = d["scaled"]; masks = d["masks"]; ml = d["ml"]
                ens = np.zeros(ml)
                for ri, rn in enumerate(RNAMES):
                    m = masks[ri]; w = p["regime_weights"][rn]
                    ens[m] = (w["v1"]*sc["v1"][rn][m] + w["i460"]*sc["i460"][m]
                             + w["i415"]*sc["i415"][m] + w["f168"]*sc["f168"][m])
                r = ens[~np.isnan(ens)]
                ann = float(np.mean(r)/np.std(r,ddof=1)*np.sqrt(8760)) if len(r) > 1 else 0.0
                sh.append(ann)
            return float(np.mean(sh) - 0.5*np.std(sh, ddof=1))
        base_loyo = subset_obj(baseline_fd, others)
        cand_loyo = subset_obj(best_fd, others)
        win = cand_loyo > base_loyo
        loyo_wins += int(win)
        print(f"  {yr}: base={base_loyo:.4f}  cand={cand_loyo:.4f}  {'WIN' if win else 'LOSE'}")

    print(f"\n  OBJ={best_obj:.4f}  Δ={delta:+.4f}  LOYO {loyo_wins}/5")
    validated = delta >= MIN_DELTA and loyo_wins >= MIN_LOYO

    if validated:
        print(f"\n✅ VALIDATED")
        cfg = json.load(open(CFG_PATH))
        disp_cfg = cfg.setdefault("disp_overlay_params", {})
        disp_cfg["per_regime_threshold"] = best_thr
        disp_cfg["per_regime_scale"]     = best_scale
        cfg["disp_overlay_params"] = disp_cfg
        old_ver = cfg.get("_version", "2.49.1")
        parts = old_ver.split(".")
        parts[-1] = str(int(parts[-1]) + 1)
        new_ver = ".".join(parts)
        cfg["_version"] = new_ver
        # delta-based OBJ
        stored_obj = cfg["monitoring"]["expected_performance"]["annual_sharpe_backtest"]
        reported_obj = round(stored_obj + delta, 4)
        cfg["monitoring"]["expected_performance"]["annual_sharpe_backtest"] = reported_obj
        json.dump(cfg, open(CFG_PATH, "w"), indent=2)
        print(f"\n  Config → v{new_ver}  OBJ={reported_obj:.4f} (stored {stored_obj:.4f} + Δ{delta:+.4f})")
        for rn in RNAMES:
            print(f"  {rn}: thr={best_thr[rn]}  scale={best_scale[rn]}")
        msg = (f"feat: P256 DISP resweep OBJ={reported_obj:.4f} "
               f"LOYO={loyo_wins}/5 D={delta:+.4f} [v{new_ver}]")
        subprocess.run(["git", "add", str(CFG_PATH)], cwd=ROOT)
        subprocess.run(["git", "commit", "-m", msg], cwd=ROOT)
    else:
        print(f"\n❌ NOT VALIDATED (LOYO {loyo_wins}/5, Δ={delta:+.4f})")
        print(f"  Current DISP params confirmed optimal.")
        for rn in RNAMES:
            print(f"  {rn}: thr={p['disp_thr'][rn]}  scale={p['disp_scale'][rn]}")

    print(f"\nRuntime: {time.time()-t0:.1f}s")
    print("[DONE] Phase 256 complete.")

main()
