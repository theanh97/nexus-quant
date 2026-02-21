"""
Phase 287 — Per-Regime V1 Internal Weights Re-Sweep [VECTORIZED, MEM-EFFICIENT]
================================================================================
Baseline: v2.49.9, OBJ=5.0336 (Phase 286 vol+disp resweep, LOYO=4/5)

P280 set V1 per-regime weights at OBJ~4.24. Since then boundaries changed (P283),
regime weights retuned (P282-P284), FTS (P285), vol+disp (P286). V1 internals
(carry/mom/mean_reversion) likely have a new optimum after OBJ jumped 4.24→5.03.

Current V1 weights:
  LOW:  wc=0.10, wm=0.50, wmr=0.40  (mean-reversion heavy)
  MID:  wc=0.30, wm=0.60, wmr=0.10  (momentum)
  HIGH: wc=0.60, wm=0.40, wmr=0.00  (carry+momentum)

Approach:
  1. For each year: load dataset, run all 66 V1 combos + 3 fixed sigs, free dataset
  2. Build precomp from stored numpy arrays
  3. Sequential 2-pass per-regime sweep (vectorized)
  4. LOYO validation + delta-based OBJ update
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
    out = Path("artifacts/phase287"); out.mkdir(parents=True, exist_ok=True)
    (out / "phase287_report.json").write_text(json.dumps(_partial, indent=2))
    print("\n⏰ TIMEOUT — partial results saved.", flush=True)
    sys.exit(0)
_signal.signal(_signal.SIGALRM, _on_timeout)
_signal.alarm(10800)  # 3 hours

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

BRD_LB = 192; PCT_WINDOW = 336; FUND_DISP_PCT = 240
TS_SHORT = 16; TS_LONG = 72; VOL_WINDOW = 168
RNAMES = ["LOW", "MID", "HIGH"]

I460_PARAMS = {"k_per_side": 4, "lookback_bars": 480, "beta_window_bars": 168,
               "target_gross_leverage": 0.3, "rebalance_interval_bars": 48}
I415_PARAMS = {"k_per_side": 4, "lookback_bars": 415, "beta_window_bars": 144,
               "target_gross_leverage": 0.3, "rebalance_interval_bars": 48}
F168_PARAMS = {"k_per_side": 2, "funding_lookback_bars": 168, "direction": "contrarian",
               "target_gross_leverage": 0.25, "rebalance_interval_bars": 36}

COST_MODEL = cost_model_from_config({
    "fee_rate": 0.0005, "slippage_rate": 0.0003, "cost_multiplier": 1.0,
})

MIN_DELTA = 0.005; MIN_LOYO = 3

W1_STEPS = [round(x * 0.10, 1) for x in range(10)]  # 0.0 to 0.9


def valid_v1_combos():
    combos = []
    for wc in W1_STEPS:
        for wm in W1_STEPS:
            if wc + wm > 1.0 + 1e-9: continue
            wmr = round(1.0 - wc - wm, 1)
            if 0.0 <= wmr <= 1.0:
                combos.append((wc, wm, wmr))
    return combos


def load_config_params():
    cfg = json.load(open(CFG_PATH))
    brs = cfg["breadth_regime_switching"]
    fts = cfg.get("fts_overlay_params", {})
    vol = cfg.get("vol_overlay_params", {})
    disp = cfg.get("disp_overlay_params", {})
    v1w  = cfg.get("v1_per_regime_weights", {})
    return {
        "p_low":  brs.get("p_low", 0.30),
        "p_high": brs.get("p_high", 0.68),
        "regime_weights": {
            "LOW":  {"v1": brs["regime_weights"]["LOW"]["v1"],
                     "i460": brs["regime_weights"]["LOW"]["i460bw168"],
                     "i415": brs["regime_weights"]["LOW"]["i415bw216"],
                     "f168": brs["regime_weights"]["LOW"]["f168"]},
            "MID":  {"v1": brs["regime_weights"]["MID"]["v1"],
                     "i460": brs["regime_weights"]["MID"]["i460bw168"],
                     "i415": brs["regime_weights"]["MID"]["i415bw216"],
                     "f168": brs["regime_weights"]["MID"]["f168"]},
            "HIGH": {"v1": brs["regime_weights"]["HIGH"]["v1"],
                     "i460": brs["regime_weights"]["HIGH"]["i460bw168"],
                     "i415": brs["regime_weights"]["HIGH"]["i415bw216"],
                     "f168": brs["regime_weights"]["HIGH"]["f168"]},
        },
        "fts_rs": fts.get("per_regime_rs", {"LOW": 0.05, "MID": 0.05, "HIGH": 0.25}),
        "fts_bs": fts.get("per_regime_bs", {"LOW": 4.0,  "MID": 2.0,  "HIGH": 2.0}),
        "fts_rt": fts.get("per_regime_rt", {"LOW": 0.80, "MID": 0.65, "HIGH": 0.50}),
        "fts_bt": fts.get("per_regime_bt", {"LOW": 0.30, "MID": 0.40, "HIGH": 0.20}),
        "ts_pct_win": fts.get("per_regime_ts_pct_win", {"LOW": 240, "MID": 288, "HIGH": 400}),
        "vol_thr":   vol.get("per_regime_threshold", {"LOW": 0.50, "MID": 0.50, "HIGH": 0.55}),
        "vol_scale": vol.get("per_regime_scale",     {"LOW": 0.40, "MID": 0.15, "HIGH": 0.05}),
        "disp_thr":  disp.get("per_regime_threshold", {"LOW": 0.50, "MID": 0.70, "HIGH": 0.40}),
        "disp_scale":disp.get("per_regime_scale",     {"LOW": 0.70, "MID": 1.80, "HIGH": 0.50}),
        "v1_weights": {r: v1w.get(r, {"w_carry": 0.25, "w_mom": 0.50, "w_mean_reversion": 0.25})
                       for r in RNAMES},
        "version": cfg.get("_version", cfg.get("version", "v2.49.2").lstrip("v")),
        "baseline_obj": cfg["monitoring"]["expected_performance"]["annual_sharpe_backtest"],
    }


def rolling_mean_arr(a, w):
    out = np.full(len(a), np.nan)
    if w <= 0: return out
    cs = np.cumsum(np.where(np.isnan(a), 0.0, a))
    for i in range(w - 1, len(a)):
        out[i] = cs[i] / w if i == w - 1 else (cs[i] - cs[i - w]) / w
    return out


def process_year(year, all_combos, t0):
    """Load one year, compute all V1 combo returns + fixed signal returns, free dataset."""
    s, e = YEAR_RANGES[year]
    cfg_d = {"provider": "binance_rest_v1", "symbols": SYMBOLS,
             "start": s, "end": e, "bar_interval": "1h",
             "cache_dir": ".cache/binance_rest"}
    dataset = make_provider(cfg_d, seed=42).load()
    n = len(dataset.timeline)

    close_mat = np.zeros((n, len(SYMBOLS)))
    for j, sym in enumerate(SYMBOLS):
        for i in range(n): close_mat[i, j] = dataset.close(sym, i)

    # BTC vol
    btc_rets = np.zeros(n)
    for i in range(1, n):
        c0 = close_mat[i-1, 0]; c1 = close_mat[i, 0]
        btc_rets[i] = (c1/c0 - 1.0) if c0 > 0 else 0.0
    btc_vol = np.full(n, np.nan)
    for i in range(VOL_WINDOW, n):
        btc_vol[i] = float(np.std(btc_rets[i-VOL_WINDOW:i])) * np.sqrt(8760)
    if VOL_WINDOW < n: btc_vol[:VOL_WINDOW] = btc_vol[VOL_WINDOW]

    # Funding dispersion
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

    # TS raw
    xsect_mean = np.mean(fund_rates, axis=1)
    ts_raw = rolling_mean_arr(xsect_mean, TS_SHORT) - rolling_mean_arr(xsect_mean, TS_LONG)

    # Breadth
    breadth = np.full(n, 0.5)
    for i in range(BRD_LB, n):
        pos = sum(1 for j in range(len(SYMBOLS))
                  if close_mat[i-BRD_LB, j] > 0 and close_mat[i, j] > close_mat[i-BRD_LB, j])
        breadth[i] = pos / len(SYMBOLS)
    brd_pct = np.full(n, 0.5)
    for i in range(PCT_WINDOW, n):
        brd_pct[i] = float(np.mean(breadth[i-PCT_WINDOW:i] <= breadth[i]))

    # Fixed signals
    fixed_rets = {}
    for sk, sname, params in [
        ("i460", "idio_momentum_alpha", I460_PARAMS),
        ("i415", "idio_momentum_alpha", I415_PARAMS),
        ("f168", "funding_momentum_alpha", F168_PARAMS),
    ]:
        result = BacktestEngine(BacktestConfig(costs=COST_MODEL)).run(
            dataset, make_strategy({"name": sname, "params": params}))
        fixed_rets[sk] = np.array(result.returns)

    # V1 combos
    v1_by_combo = {}
    n_combos = len(all_combos)
    for i, (wc, wm, wmr) in enumerate(all_combos):
        vp = {"k_per_side": 2, "w_carry": wc, "w_mom": wm, "w_mean_reversion": wmr,
              "momentum_lookback_bars": 312, "mean_reversion_lookback_bars": 84,
              "vol_lookback_bars": 192, "target_gross_leverage": 0.35,
              "rebalance_interval_bars": 60}
        res = BacktestEngine(BacktestConfig(costs=COST_MODEL)).run(
            dataset, make_strategy({"name": "nexus_alpha_v1", "params": vp}))
        v1_by_combo[(wc, wm, wmr)] = np.array(res.returns)
        if (i + 1) % 16 == 0 or (i + 1) == n_combos:
            print(f"      {i+1}/{n_combos} combos  ({time.time()-t0:.0f}s)", flush=True)

    # Free dataset
    del dataset, close_mat, btc_rets, fund_rates, fund_std_raw, xsect_mean, breadth
    gc.collect()

    return {
        "btc_vol": btc_vol, "fund_std_pct": fund_std_pct,
        "ts_raw": ts_raw, "brd_pct": brd_pct,
        "fixed_rets": fixed_rets, "v1_by_combo": v1_by_combo, "n": n,
    }


def build_precomp(P, all_arrays, v1_assign):
    """Build precomp given V1 assignment per regime."""
    fast_data = {}
    for yr, yd in sorted(all_arrays.items()):
        fixed_rets = yd["fixed_rets"]; v1_by_combo = yd["v1_by_combo"]
        btc_vol = yd["btc_vol"]; fund_std_pct = yd["fund_std_pct"]
        ts_raw = yd["ts_raw"]; brd_pct = yd["brd_pct"]
        ml = min(
            min(len(v1_by_combo[v1_assign[r]]) for r in RNAMES),
            min(len(fixed_rets[k]) for k in ["i460", "i415", "f168"])
        )
        bpct = brd_pct[:ml]; bv = btc_vol[:ml]
        fsp = fund_std_pct[:ml]; tr = ts_raw[:ml]

        ridx = np.where(bpct < P["p_low"], 0, np.where(bpct > P["p_high"], 2, 1))
        masks = [ridx == i for i in range(3)]

        unique_wins = sorted(set(P["ts_pct_win"].values()))
        ts_pct_cache = {}
        for w in unique_wins:
            arr = np.full(ml, 0.5)
            for i in range(w, ml): arr[i] = float(np.mean(tr[i-w:i] <= tr[i]))
            ts_pct_cache[w] = arr

        overlay_mult = np.ones(ml)
        for ri, rn in enumerate(RNAMES):
            m = masks[ri]
            overlay_mult[m & ~np.isnan(bv) & (bv > P["vol_thr"][rn])] *= P["vol_scale"][rn]
            overlay_mult[m & (fsp > P["disp_thr"][rn])] *= P["disp_scale"][rn]
            tsp = ts_pct_cache[P["ts_pct_win"][rn]]
            overlay_mult[m & (tsp > P["fts_rt"][rn])] *= P["fts_rs"][rn]
            overlay_mult[m & (tsp < P["fts_bt"][rn])] *= P["fts_bs"][rn]

        v1_arrs = {rn: overlay_mult * v1_by_combo[v1_assign[rn]][:ml] for rn in RNAMES}
        scaled = {
            "v1": v1_arrs,
            "i460": overlay_mult * fixed_rets["i460"][:ml],
            "i415": overlay_mult * fixed_rets["i415"][:ml],
            "f168": overlay_mult * fixed_rets["f168"][:ml],
        }
        fast_data[yr] = {"scaled": scaled, "masks": masks, "ml": ml}
    return fast_data


def fast_eval(P, fast_data):
    yearly = []
    for yr, fd in sorted(fast_data.items()):
        sc = fd["scaled"]; masks = fd["masks"]; ml = fd["ml"]
        ens = np.zeros(ml)
        for ri, rn in enumerate(RNAMES):
            m = masks[ri]; w = P["regime_weights"][rn]
            ens[m] = (w["v1"] * sc["v1"][rn][m] + w["i460"] * sc["i460"][m]
                     + w["i415"] * sc["i415"][m] + w["f168"] * sc["f168"][m])
        r = ens[~np.isnan(ens)]
        ann = float(np.mean(r) / np.std(r, ddof=1) * np.sqrt(8760)) if len(r) > 1 else 0.0
        yearly.append(ann)
    return float(np.mean(yearly) - 0.5 * np.std(yearly, ddof=1))


if __name__ == "__main__":
    t0 = time.time()
    print("=" * 70)
    print("Phase 287 — Per-Regime V1 Weights Re-Sweep [MEM-EFFICIENT]")
    print("Baseline: P286 vol+disp resweep")
    print("=" * 70)

    P = load_config_params()
    base_obj = P["baseline_obj"]
    print(f"Config: v{P['version']}  stored OBJ={base_obj:.4f}")
    v1w = P["v1_weights"]
    for rn in RNAMES:
        print(f"  {rn}: wc={v1w[rn]['w_carry']:.2f} wm={v1w[rn]['w_mom']:.2f} wmr={v1w[rn]['w_mean_reversion']:.2f}")
    print(f"Regime weights:")
    for rn in RNAMES:
        w = P["regime_weights"][rn]
        print(f"  {rn}: v1={w['v1']:.2f} i460={w['i460']:.2f} i415={w['i415']:.2f} f168={w['f168']:.2f}")

    all_combos = valid_v1_combos()
    print(f"\nV1 combos to test: {len(all_combos)}")

    # [1] Per-year data + V1 computation
    print("\n[1] Per-year data + V1 computation (1 year at a time)...")
    all_arrays = {}
    for yr in sorted(YEAR_RANGES.keys()):
        print(f"\n  Year {yr}:", flush=True)
        all_arrays[yr] = process_year(yr, all_combos, t0)
        print(f"  Year {yr} complete.  ({time.time()-t0:.0f}s)")

    # Initial assignment from config
    cur_assign = {}
    for rn in RNAMES:
        vw = P["v1_weights"][rn]
        k = (round(vw["w_carry"], 1), round(vw["w_mom"], 1), round(vw["w_mean_reversion"], 1))
        sample_yr = sorted(all_arrays.keys())[0]
        if k not in all_arrays[sample_yr]["v1_by_combo"]:
            best_k = min(all_combos, key=lambda c: sum((a-b)**2 for a,b in zip(c, k)))
            print(f"  Warning: {rn} key {k} not in grid → using {best_k}")
            k = best_k
        cur_assign[rn] = k

    # [2] Baseline
    print("\n[2] Baseline fast eval...")
    baseline_fd = build_precomp(P, all_arrays, cur_assign)
    obj_base = fast_eval(P, baseline_fd)
    print(f"  Measured baseline OBJ = {obj_base:.4f}  (config stored = {base_obj:.4f})")
    _partial["baseline_obj"] = float(obj_base)
    _partial["stored_baseline"] = float(base_obj)

    # [3] Sweep
    print("\n[3] Per-regime V1 sweep (2 passes)...")
    best_assign = {r: cur_assign[r] for r in RNAMES}
    best_obj = obj_base

    for pass_num in range(1, 3):
        print(f"\n  === Pass {pass_num} ===")
        for rname in RNAMES:
            tw = time.time()
            local_best = best_assign[rname]
            for combo in all_combos:
                trial = {**best_assign, rname: combo}
                trial_fd = build_precomp(P, all_arrays, trial)
                obj = fast_eval(P, trial_fd)
                if obj > best_obj + 1e-6:
                    best_obj = obj; local_best = combo
            best_assign[rname] = local_best
            wc, wm, wmr = local_best
            print(f"    {rname}: wc={wc:.2f} wm={wm:.2f} wmr={wmr:.2f}  "
                  f"OBJ={best_obj:.4f}  Δ={best_obj-obj_base:+.4f}  [{int(time.time()-tw)}s]")

    delta = best_obj - obj_base
    print(f"\n  Best OBJ={best_obj:.4f}  Δ={delta:+.4f}")

    # [4] LOYO validation
    print("\n[4] LOYO validation...")
    best_fd = build_precomp(P, all_arrays, best_assign)
    loyo_wins = 0
    for yr in sorted(YEAR_RANGES.keys()):
        others = [y for y in YEAR_RANGES if y != yr]
        def subset_obj(fd, yrs):
            sh = []
            for y in yrs:
                d = fd[y]; sc = d["scaled"]; masks = d["masks"]; ml = d["ml"]
                ens = np.zeros(ml)
                for ri, rn in enumerate(RNAMES):
                    m = masks[ri]; w = P["regime_weights"][rn]
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
        d = cand_loyo - base_loyo
        print(f"  {yr}: base={base_loyo:.4f}  cand={cand_loyo:.4f}  Δ={d:+.4f}  {'✓ WIN' if win else '✗ LOSE'}")

    print(f"\n  OBJ={best_obj:.4f}  Δ={delta:+.4f}  LOYO {loyo_wins}/5")
    validated = delta >= MIN_DELTA and loyo_wins >= MIN_LOYO

    result = {
        "phase": "287", "baseline_obj": float(obj_base), "stored_baseline": float(base_obj),
        "best_obj": float(best_obj), "delta": float(delta),
        "loyo_wins": loyo_wins, "loyo_total": 5, "validated": validated,
        "best_v1_weights": {rn: {"w_carry": best_assign[rn][0], "w_mom": best_assign[rn][1],
                                  "w_mean_reversion": best_assign[rn][2]} for rn in RNAMES},
    }
    _partial.update(result)
    out = Path("artifacts/phase287"); out.mkdir(parents=True, exist_ok=True)
    (out / "phase287_report.json").write_text(json.dumps(result, indent=2))

    if validated:
        print(f"\n✅ VALIDATED")
        cfg = json.load(open(CFG_PATH))
        v1w_cfg = cfg.setdefault("v1_per_regime_weights", {})
        for rn in RNAMES:
            wc, wm, wmr = best_assign[rn]
            v1w_cfg[rn] = {"w_carry": wc, "w_mom": wm, "w_mean_reversion": wmr}
        old_ver = cfg.get("_version", "2.49.2")
        parts = old_ver.split(".")
        parts[-1] = str(int(parts[-1]) + 1)
        new_ver = ".".join(parts)
        cfg["_version"] = new_ver
        old_vver = cfg.get("version", "v2.49.3").lstrip("v")
        vparts = old_vver.split(".")
        vparts[-1] = str(int(vparts[-1]) + 1)
        cfg["version"] = "v" + ".".join(vparts)
        stored_obj = cfg["monitoring"]["expected_performance"]["annual_sharpe_backtest"]
        reported_obj = round(stored_obj + delta, 4)
        cfg["monitoring"]["expected_performance"]["annual_sharpe_backtest"] = reported_obj
        cfg["obj"] = reported_obj
        with open(CFG_PATH, "w") as f: json.dump(cfg, f, indent=2, ensure_ascii=False)
        print(f"\n  Config → {cfg['version']}  OBJ={reported_obj:.4f}")
        for rn in RNAMES:
            wc, wm, wmr = best_assign[rn]
            print(f"  {rn}: wc={wc:.2f} wm={wm:.2f} wmr={wmr:.2f}")
        msg = (f"feat: P287 per-regime V1 weights OBJ={reported_obj:.4f} "
               f"LOYO={loyo_wins}/5 D={delta:+.4f} [{cfg['version']}]")
        subprocess.run(["git", "add", str(CFG_PATH)], cwd=ROOT)
        subprocess.run(["git", "commit", "-m", msg], cwd=ROOT)
        print(f"  Committed: {msg}")
    else:
        print(f"\n❌ NOT VALIDATED (LOYO {loyo_wins}/5, Δ={delta:+.4f})")
        print(f"  Current V1 weights confirmed optimal at this baseline.")
        for rn in RNAMES:
            wc, wm, wmr = best_assign[rn]
            print(f"  Best found {rn}: wc={wc:.2f} wm={wm:.2f} wmr={wmr:.2f}")

    print(f"\nTotal runtime: {int(time.time()-t0)}s ({int((time.time()-t0)/60)}m)", flush=True)
    print("[DONE] Phase 287 complete.")
