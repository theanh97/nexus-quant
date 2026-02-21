"""
Phase 269 — Per-Regime FTS RS/RT/BS/BT Re-Sweep [VECTORIZED]
=============================================================
Baseline: v2.48.0, OBJ≈4.2978

Rationale:
  P262n (LOYO=5/5) found better FTS params stored in a DIFFERENT key format
  (cfg["fts_overlay_params"]["HIGH"] = {rs:0.40, bs:2.25, bt:0.22}) vs what
  scripts actually read (cfg["fts_overlay_params"]["per_regime_rs"] = {HIGH:0.25}).
  Those improvements are NOT being applied.
  P262n findings: LOW bt=0.33, MID bs=2.5/bt=0.22, HIGH rs=0.40/bs=2.25/bt=0.22

  Re-sweep RS/RT/BS/BT per regime from current high baseline to reconcile.
  All evaluations vectorized (no new strategy runs).
"""

import os, sys, json, time
import signal as _signal
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

import numpy as np
import subprocess

from nexus_quant.backtest.costs import cost_model_from_config
from nexus_quant.backtest.engine import BacktestConfig, BacktestEngine
from nexus_quant.data.providers.registry import make_provider
from nexus_quant.strategies.registry import make_strategy

_partial: dict = {}
def _on_timeout(signum, frame):
    _partial["timeout"] = True
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

def safe_dict(d, key, default):
    if isinstance(d, dict): return d.get(key, default)
    return default

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
        "disp_thr":  disp.get("per_regime_threshold", {"LOW": 0.70, "MID": 0.70, "HIGH": 0.40}),
        "disp_scale":disp.get("per_regime_scale",     {"LOW": 0.50, "MID": 1.50, "HIGH": 0.50}),
        "v1_weights": {r: v1w.get(r, {"w_carry": 0.25, "w_mom": 0.45, "w_mean_reversion": 0.30})
                       for r in ["LOW", "MID", "HIGH"]},
        "version": cfg["_version"],
        "baseline_obj": cfg["monitoring"]["expected_performance"]["annual_sharpe_backtest"],
    }

BRD_LB = 192; PCT_WINDOW = 336; FUND_DISP_PCT = 240
TS_SHORT = 16; TS_LONG = 72; VOL_WINDOW = 168
RNAMES = ["LOW", "MID", "HIGH"]

I460_PARAMS = {"k_per_side": 4, "lookback_bars": 480, "beta_window_bars": 168,
               "target_gross_leverage": 0.3, "rebalance_interval_bars": 48}
I415_PARAMS = {"k_per_side": 4, "lookback_bars": 415, "beta_window_bars": 144,
               "target_gross_leverage": 0.3, "rebalance_interval_bars": 48}
F168_PARAMS = {"k_per_side": 2, "funding_lookback_bars": 168, "direction": "contrarian",
               "target_gross_leverage": 0.25, "rebalance_interval_bars": 36}

MIN_DELTA = 0.005; MIN_LOYO = 3

COST_MODEL = cost_model_from_config({
    "fee_rate": 0.0005, "slippage_rate": 0.0003, "cost_multiplier": 1.0,
})

def rolling_mean_arr(a, w):
    out = np.full(len(a), np.nan)
    if w <= 0: return out
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
    breadth = np.full(n, 0.5)
    for i in range(BRD_LB, n):
        pos = sum(1 for j in range(len(SYMBOLS))
                  if close_mat[i-BRD_LB, j] > 0 and close_mat[i, j] > close_mat[i-BRD_LB, j])
        breadth[i] = pos / len(SYMBOLS)
    brd_pct = np.full(n, 0.5)
    for i in range(PCT_WINDOW, n):
        brd_pct[i] = float(np.mean(breadth[i-PCT_WINDOW:i] <= breadth[i]))
    fixed_rets = {}
    for sk, sname, params in [
        ("i460", "idio_momentum_alpha", I460_PARAMS),
        ("i415", "idio_momentum_alpha", I415_PARAMS),
        ("f168", "funding_momentum_alpha", F168_PARAMS),
    ]:
        result = BacktestEngine(BacktestConfig(costs=COST_MODEL)).run(
            dataset, make_strategy({"name": sname, "params": params}))
        fixed_rets[sk] = np.array(result.returns)
    return btc_vol, fund_std_pct, ts_raw, brd_pct, fixed_rets, n, dataset

def compute_v1(dataset, v1w_params):
    vp = {"k_per_side": 2,
          "w_carry": v1w_params["w_carry"],
          "w_mom": v1w_params["w_mom"],
          "w_mean_reversion": v1w_params["w_mean_reversion"],
          "momentum_lookback_bars": 336, "mean_reversion_lookback_bars": 84,
          "vol_lookback_bars": 192, "target_gross_leverage": 0.35, "rebalance_interval_bars": 60}
    result = BacktestEngine(BacktestConfig(costs=COST_MODEL)).run(
        dataset, make_strategy({"name": "nexus_alpha_v1", "params": vp}))
    return np.array(result.returns)

# ─────────────────────────────────────────────────────────────────────────────
# VECTORIZED FAST EVAL
# ─────────────────────────────────────────────────────────────────────────────

def build_fast_data(P, all_base_data, all_v1_raw):
    """Build per-year arrays needed for fast FTS evaluation.

    Returns for each year:
      masks[r]        — bool array (T,) for regime r
      ts_pct_arrs[r]  — percentile of ts_raw in window ts_pct_win[r]
      non_fts_mult    — overlay mult from VOL+DISP only (no FTS)
      scaled          — dict of pre-scaled signal arrays (before FTS overlay)
    """
    fast_data = {}
    for yr, base_data in sorted(all_base_data.items()):
        btc_vol, fund_std_pct, ts_raw, brd_pct, fixed_rets, n, _ = base_data
        v1 = all_v1_raw[yr]
        min_len = min(
            min(len(v1[r]) for r in RNAMES),
            min(len(fixed_rets[k]) for k in ["i460","i415","f168"])
        )
        bpct = brd_pct[:min_len]
        bv   = btc_vol[:min_len]
        fsp  = fund_std_pct[:min_len]
        tr   = ts_raw[:min_len]

        # Regime index
        regime_idx = np.where(bpct < P["p_low"], 0, np.where(bpct > P["p_high"], 2, 1))
        masks = [regime_idx == i for i in range(3)]

        # ts percentile per regime window
        unique_wins = sorted(set(P["ts_pct_win"].values()))
        ts_pct_cache = {}
        for w in unique_wins:
            arr = np.full(min_len, 0.5)
            for i in range(w, min_len):
                arr[i] = float(np.mean(tr[i-w:i] <= tr[i]))
            ts_pct_cache[w] = arr
        ts_pct_arrs = {r: ts_pct_cache[P["ts_pct_win"][r]] for r in RNAMES}

        # Non-FTS overlay (VOL + DISP only)
        non_fts_mult = np.ones(min_len)
        for ridx, rname in enumerate(RNAMES):
            m = masks[ridx]
            # VOL dampening
            vol_c = bv < P["vol_thr"][rname]
            non_fts_mult[m & vol_c] *= P["vol_scale"][rname]
            # DISP scaling
            disp_c = fsp > P["disp_thr"][rname]
            non_fts_mult[m & disp_c] *= P["disp_scale"][rname]

        # Pre-scale signals by non_fts_mult (FTS will be added per-regime during sweep)
        v1_arr_LOW  = non_fts_mult * v1["LOW"][:min_len]
        v1_arr_MID  = non_fts_mult * v1["MID"][:min_len]
        v1_arr_HIGH = non_fts_mult * v1["HIGH"][:min_len]
        i460_arr = non_fts_mult * fixed_rets["i460"][:min_len]
        i415_arr = non_fts_mult * fixed_rets["i415"][:min_len]
        f168_arr = non_fts_mult * fixed_rets["f168"][:min_len]

        fast_data[yr] = {
            "masks": masks,
            "ts_pct_arrs": ts_pct_arrs,
            "non_fts_mult": non_fts_mult,
            "v1_arrs": {"LOW": v1_arr_LOW, "MID": v1_arr_MID, "HIGH": v1_arr_HIGH},
            "i460_arr": i460_arr,
            "i415_arr": i415_arr,
            "f168_arr": f168_arr,
            "min_len": min_len,
        }
    return fast_data

def evaluate_with_fts(P, fast_data, fts_rs, fts_rt, fts_bs, fts_bt):
    """Evaluate OBJ with given per-regime FTS params (all others fixed).

    Returns (obj_full, yearly_sharpes).
    """
    yearly_sharpes = []
    for yr, fd in sorted(fast_data.items()):
        masks = fd["masks"]
        ts_pct_arrs = fd["ts_pct_arrs"]
        non_fts_mult = fd["non_fts_mult"]
        v1_arrs = fd["v1_arrs"]
        i460_arr = fd["i460_arr"]
        i415_arr = fd["i415_arr"]
        f168_arr = fd["f168_arr"]
        min_len = fd["min_len"]

        # Build FTS overlay factor per bar
        fts_mult = np.ones(min_len)
        for ridx, rname in enumerate(RNAMES):
            m = masks[ridx]
            tsp = ts_pct_arrs[rname]
            rs = fts_rs[rname]; rt = fts_rt[rname]
            bs = fts_bs[rname]; bt = fts_bt[rname]
            fts_mult[m & (tsp > rt)] *= rs
            fts_mult[m & (tsp < bt)] *= bs

        # Full overlay = non_fts * fts
        full_mult = non_fts_mult * fts_mult

        # Ensemble returns per bar
        ens = np.zeros(min_len)
        for ridx, rname in enumerate(RNAMES):
            m = masks[ridx]
            w = P["regime_weights"][rname]
            v1k = "LOW" if rname == "LOW" else ("MID" if rname == "MID" else "HIGH")
            # Apply FTS mult on top of non-fts pre-scaled signals
            # Actually we need: signal * non_fts * fts = signal * full_mult
            # But pre-scaled already has non_fts applied, so we need signal * fts_extra
            # Let's compute from scratch for affected regime bars
            fts_extra = fts_mult[m]  # the FTS-only factor for regime bars
            ens[m] = (w["v1"] * v1_arrs[v1k][m] * fts_extra
                    + w["i460"] * i460_arr[m] * fts_extra
                    + w["i415"] * i415_arr[m] * fts_extra
                    + w["f168"] * f168_arr[m] * fts_extra)

        # Annual Sharpe
        ann = float(np.mean(ens)) / (float(np.std(ens)) + 1e-9) * np.sqrt(8760)
        yearly_sharpes.append(ann)

    obj = float(np.mean(yearly_sharpes)) - 0.5 * float(np.std(yearly_sharpes))
    return obj, yearly_sharpes

# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    t0 = time.time()
    print("=" * 65)
    print("Phase 269 — Per-Regime FTS RS/RT/BS/BT Re-Sweep [VECTORIZED]")

    P = load_config_params()
    base_obj = P["baseline_obj"]
    print(f"Baseline: v{P['version']}  OBJ={base_obj:.4f}")
    print(f"Current FTS RS={P['fts_rs']}  RT={P['fts_rt']}")
    print(f"         BS={P['fts_bs']}  BT={P['fts_bt']}")
    print("=" * 65)

    # [1] Load per-year data
    print("\n[1] Loading per-year data...")
    all_base_data = {}
    for yr in sorted(YEAR_RANGES.keys()):
        print(f"  {yr}: ", end="", flush=True)
        all_base_data[yr] = load_year_data(yr)
        print("done.")

    # [2] Pre-compute V1 per regime per year
    print("\n[2] Pre-computing V1 per regime × year...")
    all_v1_raw = {}
    for yr, base_data in sorted(all_base_data.items()):
        _, _, _, _, _, _, dataset = base_data
        all_v1_raw[yr] = {}
        for rname in RNAMES:
            all_v1_raw[yr][rname] = compute_v1(dataset, P["v1_weights"][rname])
        print(f"  {yr}: done.")

    # [3] Build fast eval data
    print("\n[3] Building fast eval data...")
    fast_data = build_fast_data(P, all_base_data, all_v1_raw)

    # Verify baseline
    obj_base, _ = evaluate_with_fts(P, fast_data, P["fts_rs"], P["fts_rt"], P["fts_bs"], P["fts_bt"])
    print(f"  Baseline OBJ = {obj_base:.4f}  (expected ≈ {base_obj:.4f})")

    # Sweep grids
    RS_GRID = [0.00, 0.02, 0.05, 0.08, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]
    RT_GRID = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]
    BS_GRID = [1.0, 1.5, 2.0, 2.25, 2.5, 3.0, 3.5, 4.0, 5.0]
    BT_GRID = [0.10, 0.15, 0.20, 0.22, 0.25, 0.28, 0.30, 0.33, 0.35, 0.40]

    # Current best params
    best_rs = {r: P["fts_rs"][r] for r in RNAMES}
    best_rt = {r: P["fts_rt"][r] for r in RNAMES}
    best_bs = {r: P["fts_bs"][r] for r in RNAMES}
    best_bt = {r: P["fts_bt"][r] for r in RNAMES}
    best_obj = obj_base

    print("\n[4] Per-regime FTS sweep (2 passes)...")

    for pass_num in range(1, 3):
        print(f"\n  === Pass {pass_num} ===")
        for rname in RNAMES:
            # Sweep RS
            best_rs_r = best_rs[rname]
            for rs in RS_GRID:
                trial_rs = {**best_rs, rname: rs}
                obj, _ = evaluate_with_fts(P, fast_data, trial_rs, best_rt, best_bs, best_bt)
                if obj > best_obj + 1e-6:
                    best_obj = obj; best_rs_r = rs
            best_rs[rname] = best_rs_r

            # Sweep RT
            best_rt_r = best_rt[rname]
            for rt in RT_GRID:
                trial_rt = {**best_rt, rname: rt}
                obj, _ = evaluate_with_fts(P, fast_data, best_rs, trial_rt, best_bs, best_bt)
                if obj > best_obj + 1e-6:
                    best_obj = obj; best_rt_r = rt
            best_rt[rname] = best_rt_r

            # Sweep BS
            best_bs_r = best_bs[rname]
            for bs in BS_GRID:
                trial_bs = {**best_bs, rname: bs}
                obj, _ = evaluate_with_fts(P, fast_data, best_rs, best_rt, trial_bs, best_bt)
                if obj > best_obj + 1e-6:
                    best_obj = obj; best_bs_r = bs
            best_bs[rname] = best_bs_r

            # Sweep BT
            best_bt_r = best_bt[rname]
            for bt in BT_GRID:
                trial_bt = {**best_bt, rname: bt}
                obj, _ = evaluate_with_fts(P, fast_data, best_rs, best_rt, best_bs, trial_bt)
                if obj > best_obj + 1e-6:
                    best_obj = obj; best_bt_r = bt
            best_bt[rname] = best_bt_r

            print(f"    {rname}: rs={best_rs[rname]:.3f} rt={best_rt[rname]:.2f}"
                  f" bs={best_bs[rname]:.2f} bt={best_bt[rname]:.2f}  OBJ={best_obj:.4f}")

    delta = best_obj - obj_base
    print(f"\n  Final OBJ={best_obj:.4f}  Δ={delta:+.4f}")

    # [5] LOYO validation
    print("\n[5] LOYO validation...")
    loyo_wins = 0
    for yr in sorted(YEAR_RANGES.keys()):
        # Base: hold out yr, compute OBJ on other 4
        others = [y for y in YEAR_RANGES if y != yr]
        def obj_subset(yrs, rs, rt, bs, bt):
            sharpes = []
            for y in yrs:
                fd = fast_data[y]
                masks = fd["masks"]; ts_pct_arrs = fd["ts_pct_arrs"]
                v1_arrs = fd["v1_arrs"]; i460_arr = fd["i460_arr"]
                i415_arr = fd["i415_arr"]; f168_arr = fd["f168_arr"]
                non_fts_mult = fd["non_fts_mult"]; min_len = fd["min_len"]
                fts_mult = np.ones(min_len)
                for ridx, rn in enumerate(RNAMES):
                    m = masks[ridx]; tsp = ts_pct_arrs[rn]
                    fts_mult[m & (tsp > rt[rn])] *= rs[rn]
                    fts_mult[m & (tsp < bt[rn])] *= bs[rn]
                ens = np.zeros(min_len)
                for ridx, rn in enumerate(RNAMES):
                    m = masks[ridx]; w = P["regime_weights"][rn]
                    fts_extra = fts_mult[m]
                    v1k = rn
                    ens[m] = (w["v1"] * v1_arrs[v1k][m] * fts_extra
                            + w["i460"] * i460_arr[m] * fts_extra
                            + w["i415"] * i415_arr[m] * fts_extra
                            + w["f168"] * f168_arr[m] * fts_extra)
                ann = float(np.mean(ens)) / (float(np.std(ens)) + 1e-9) * np.sqrt(8760)
                sharpes.append(ann)
            return float(np.mean(sharpes)) - 0.5 * float(np.std(sharpes))

        obj_base_loyo = obj_subset(others, P["fts_rs"], P["fts_rt"], P["fts_bs"], P["fts_bt"])
        obj_cand_loyo = obj_subset(others, best_rs, best_rt, best_bs, best_bt)
        win = obj_cand_loyo > obj_base_loyo
        loyo_wins += int(win)
        print(f"  {yr}: base={obj_base_loyo:.4f}  cand={obj_cand_loyo:.4f}  {'WIN' if win else 'LOSE'}")

    print(f"\n  OBJ={best_obj:.4f}  Δ={delta:+.4f}  LOYO {loyo_wins}/5")

    validated = delta >= MIN_DELTA and loyo_wins >= MIN_LOYO

    if validated:
        print(f"\n✅ VALIDATED")
        cfg = json.load(open(CFG_PATH))
        fts = cfg.setdefault("fts_overlay_params", {})
        fts["per_regime_rs"] = best_rs
        fts["per_regime_rt"] = best_rt
        fts["per_regime_bs"] = best_bs
        fts["per_regime_bt"] = best_bt
        old_ver = cfg["_version"]
        parts = old_ver.split(".")
        parts[-1] = str(int(parts[-1]) + 1)
        new_ver = ".".join(parts)
        cfg["_version"] = new_ver
        cfg["monitoring"]["expected_performance"]["annual_sharpe_backtest"] = round(best_obj, 4)
        json.dump(cfg, open(CFG_PATH, "w"), indent=2)
        print(f"\n  Config → v{new_ver}  OBJ={best_obj:.4f}")
        # Commit
        msg = (f"feat: P269 per-regime FTS resweep OBJ={best_obj:.4f} "
               f"LOYO={loyo_wins}/5 D={delta:+.4f} [v{new_ver}]")
        subprocess.run(["git", "add", str(CFG_PATH)], cwd=ROOT)
        subprocess.run(["git", "commit", "-m", msg], cwd=ROOT)
    else:
        print(f"\n❌ NOT VALIDATED (LOYO {loyo_wins}/5, Δ={delta:+.4f})")
        print(f"  Current params confirmed as baseline:")
        print(f"  RS={P['fts_rs']}, RT={P['fts_rt']}")
        print(f"  BS={P['fts_bs']}, BT={P['fts_bt']}")

    print(f"\nRuntime: {time.time()-t0:.1f}s")
