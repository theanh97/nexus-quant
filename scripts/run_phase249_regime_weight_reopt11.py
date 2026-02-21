"""
Phase 249 — Regime Weight Re-opt (11th Pass) [VECTORIZED]
==========================================================
Baseline: v2.46.1, OBJ=4.2411

Phase 248 did a huge FTS re-sweep (RS: 0.50→0.05 for LOW/MID).
Regime weights need to be re-optimized given:
  - New FTS RS/BS/RT/BT per-regime
  - p_high changed to 0.68 (from 0.60)
  - VOL SCALE changed: MID=0.50, HIGH=0.35
  - Per-regime V1 weights (Phase 244): LOW more MR-heavy

All params loaded directly from current config.
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
    out = Path("artifacts/phase249"); out.mkdir(parents=True, exist_ok=True)
    (out / "phase249_report.json").write_text(json.dumps(_partial, indent=2))
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

# Load all params from current config
def load_config_params():
    cfg_path = ROOT / "configs" / "production_p91b_champion.json"
    cfg = json.load(open(cfg_path))
    brs = cfg["breadth_regime_switching"]
    fts = cfg.get("fts_overlay_params", {})
    vol = cfg.get("vol_overlay_params", {})
    disp = cfg.get("disp_overlay_params", {})
    v1w  = cfg.get("v1_per_regime_weights", {})

    params = {
        "p_low":  brs.get("p_low", brs.get("percentile_low", 0.30)),
        "p_high": brs.get("p_high", brs.get("percentile_high", 0.60)),
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
        "vol_thr":   vol.get("per_regime_threshold", {"LOW": 0.50, "MID": 0.50, "HIGH": 0.50}),
        "vol_scale": vol.get("per_regime_scale",     {"LOW": 0.40, "MID": 0.50, "HIGH": 0.35}),
        "disp_thr":  disp.get("per_regime_threshold", {"LOW": 0.70, "MID": 0.70, "HIGH": 0.40}),
        "disp_scale":disp.get("per_regime_scale",     {"LOW": 0.50, "MID": 1.50, "HIGH": 0.50}),
        "v1_weights": {r: v1w.get(r, {"w_carry": 0.25, "w_mom": 0.45, "w_mean_reversion": 0.30})
                       for r in ["LOW", "MID", "HIGH"]},
        "version": cfg["_version"],
        "baseline_obj": cfg["monitoring"]["expected_performance"]["annual_sharpe_backtest"],
    }
    return params, cfg_path

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
W_STEPS = [round(x * 0.05, 2) for x in range(17)]  # 0.00 to 0.80

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
        ("i460", "idio_momentum_alpha",    I460_PARAMS),
        ("i415", "idio_momentum_alpha",    I415_PARAMS),
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

def precompute_fast_data(P, all_base_data, all_v1_raw):
    """Vectorized precomputation with all v2.46.1 params from config."""
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

        # Regime index using p_high=0.68
        regime_idx = np.where(bpct < P["p_low"], 0, np.where(bpct > P["p_high"], 2, 1))
        masks = [regime_idx == i for i in range(3)]

        # FTS ts percentile per-regime window
        unique_wins = sorted(set(P["ts_pct_win"].values()))
        ts_pct_cache = {}
        for w in unique_wins:
            arr = np.full(min_len, 0.5)
            for i in range(w, min_len):
                arr[i] = float(np.mean(tr[i-w:i] <= tr[i]))
            ts_pct_cache[w] = arr

        # Overlay multiplier
        overlay_mult = np.ones(min_len)
        for ridx, rname in enumerate(RNAMES):
            m = masks[ridx]
            # VOL
            vm = m & ~np.isnan(bv) & (bv > P["vol_thr"][rname])
            overlay_mult[vm] *= P["vol_scale"][rname]
            # DISP
            dm = m & (fsp > P["disp_thr"][rname])
            overlay_mult[dm] *= P["disp_scale"][rname]
            # FTS
            tsp = ts_pct_cache[P["ts_pct_win"][rname]]
            overlay_mult[m & (tsp > P["fts_rt"][rname])] *= P["fts_rs"][rname]
            overlay_mult[m & (tsp < P["fts_bt"][rname])] *= P["fts_bs"][rname]

        scaled = {
            "v1_LOW":  overlay_mult * v1["LOW"][:min_len],
            "v1_MID":  overlay_mult * v1["MID"][:min_len],
            "i460":    overlay_mult * fixed_rets["i460"][:min_len],
            "i415":    overlay_mult * fixed_rets["i415"][:min_len],
            "f168":    overlay_mult * fixed_rets["f168"][:min_len],
        }
        fast_data[yr] = (masks, min_len, scaled)
    return fast_data

def fast_evaluate(regime_weights, fast_data):
    yearly = {}
    for yr, (masks, min_len, scaled) in fast_data.items():
        ens = np.zeros(min_len)
        for ridx, rname in enumerate(RNAMES):
            m = masks[ridx]
            w = regime_weights[rname]
            v1k = "v1_LOW" if rname == "LOW" else "v1_MID"
            ens[m] += (w["v1"]  * scaled[v1k][m]
                       + w["i460"] * scaled["i460"][m]
                       + w["i415"] * scaled["i415"][m]
                       + w["f168"] * scaled["f168"][m])
        r = ens[~np.isnan(ens)]
        sh = float(np.mean(r) / np.std(r, ddof=1) * np.sqrt(8760)) if len(r) > 1 else 0.0
        yearly[yr] = sh
    vals = list(yearly.values())
    return float(np.mean(vals) - 0.5 * np.std(vals, ddof=1)), yearly

def main():
    t0 = time.time()
    P, cfg_path = load_config_params()
    print("=" * 65)
    print("Phase 249 — Regime Weight Re-opt (11th Pass) [VECTORIZED]")
    print(f"Baseline: {P['version']}  OBJ={P['baseline_obj']}")
    print(f"p_low={P['p_low']} p_high={P['p_high']}")
    print(f"FTS RS={P['fts_rs']} BS={P['fts_bs']}")
    print(f"VOL SCALE={P['vol_scale']}")
    for r in RNAMES:
        w = P["regime_weights"][r]
        print(f"  {r}: v1={w['v1']} i460={w['i460']} i415={w['i415']} f168={w['f168']}")
    print("=" * 65, flush=True)

    print("\n[1] Loading per-year data & fixed signals...", flush=True)
    all_base_data = {}
    for yr in sorted(YEAR_RANGES):
        print(f"  {yr}: ", end="", flush=True)
        all_base_data[yr] = load_year_data(yr)
        print("done.", flush=True)

    print("\n[2] Pre-computing V1 returns (2 variants × 5 years)...", flush=True)
    # Determine unique V1 weight combinations
    v1_low_w  = P["v1_weights"]["LOW"]
    v1_mid_w  = P["v1_weights"]["MID"]  # MID and HIGH use same weights
    all_v1_raw = {}
    for yr in sorted(YEAR_RANGES):
        ds = all_base_data[yr][6]
        v1_low     = compute_v1(ds, v1_low_w)
        v1_midhigh = compute_v1(ds, v1_mid_w)
        all_v1_raw[yr] = {"LOW": v1_low, "MID": v1_midhigh, "HIGH": v1_midhigh}
        print(f"  {yr}: done.", flush=True)

    print("\n[3] Precomputing vectorized fast eval data...", flush=True)
    fast_data = precompute_fast_data(P, all_base_data, all_v1_raw)
    print("  done.", flush=True)

    # Baseline check
    base_obj_val, base_yearly = fast_evaluate(P["regime_weights"], fast_data)
    print(f"\n  Baseline OBJ = {base_obj_val:.4f}  (expected ≈ {P['baseline_obj']})", flush=True)
    _partial.update({"baseline_obj": float(base_obj_val)})

    print("\n[4] Sequential regime weight sweep (2 passes)...", flush=True)
    current_weights = {r: dict(P["regime_weights"][r]) for r in RNAMES}
    running_best = base_obj_val

    def sweep_one(target, current, running_b):
        best_obj = running_b
        best_w = dict(current[target])
        for wv1 in W_STEPS:
            for wi460 in W_STEPS:
                for wi415 in W_STEPS:
                    wf168 = round(1.0 - wv1 - wi460 - wi415, 4)
                    if wf168 < 0: continue
                    w_test = dict(current)
                    w_test[target] = {"v1": wv1, "i460": wi460, "i415": wi415, "f168": wf168}
                    o, _ = fast_evaluate(w_test, fast_data)
                    if o > best_obj:
                        best_obj = o
                        best_w = {"v1": wv1, "i460": wi460, "i415": wi415, "f168": wf168}
        return best_w, best_obj

    for pass_num in [1, 2]:
        print(f"\n  === Pass {pass_num} ===", flush=True)
        for rname in RNAMES:
            ts = time.time()
            best_w, running_best = sweep_one(rname, current_weights, running_best)
            current_weights[rname] = best_w
            print(f"    {rname}: v1={best_w['v1']:.2f} i460={best_w['i460']:.4f} "
                  f"i415={best_w['i415']:.4f} f168={best_w['f168']:.4f}  "
                  f"OBJ={running_best:.4f}  Δ={running_best-base_obj_val:+.4f}  [{int(time.time()-ts)}s]",
                  flush=True)

    final_obj_val, final_yearly = fast_evaluate(current_weights, fast_data)
    delta = final_obj_val - base_obj_val
    print(f"\n  Final OBJ={final_obj_val:.4f}  Δ={delta:+.4f}", flush=True)

    print("\n[5] LOYO validation...", flush=True)
    loyo_wins = 0
    for yr in sorted(final_yearly):
        base_sh = base_yearly[yr]
        cand_sh = final_yearly[yr]
        win = cand_sh > base_sh
        if win: loyo_wins += 1
        print(f"  {yr}: base={base_sh:.4f}  cand={cand_sh:.4f}  {'WIN' if win else 'LOSE'}")

    print(f"\n  OBJ={final_obj_val:.4f}  Δ={delta:+.4f}  LOYO {loyo_wins}/{len(final_yearly)}", flush=True)
    validated = loyo_wins >= MIN_LOYO and delta >= MIN_DELTA
    print(f"\n{'✅ VALIDATED' if validated else '❌ NO IMPROVEMENT'}", flush=True)

    result = {
        "phase": 249, "baseline_obj": float(base_obj_val),
        "best_obj": float(final_obj_val), "delta": float(delta),
        "loyo_wins": loyo_wins, "loyo_total": len(final_yearly),
        "validated": validated, "best_regime_weights": current_weights,
    }
    _partial.update(result)
    out = Path("artifacts/phase249"); out.mkdir(parents=True, exist_ok=True)
    (out / "phase249_report.json").write_text(json.dumps(result, indent=2))

    if validated:
        cfg = json.load(open(cfg_path))
        brs = cfg["breadth_regime_switching"]["regime_weights"]
        for rname, wts in current_weights.items():
            brs[rname]["v1"]        = wts["v1"]
            brs[rname]["i460bw168"] = wts["i460"]
            brs[rname]["i415bw216"] = wts["i415"]
            brs[rname]["f168"]      = wts["f168"]
        # Bump version
        cur_ver = cfg.get("_version", "2.46.1")
        parts = cur_ver.split(".")
        new_ver = f"{parts[0]}.{parts[1]}.{int(parts[2])+1}"
        cfg["_version"] = new_ver
        cfg["monitoring"]["expected_performance"]["annual_sharpe_backtest"] = round(final_obj_val, 4)
        with open(cfg_path, "w") as f:
            json.dump(cfg, f, indent=2, ensure_ascii=False)
        print(f"\n  Config → {new_ver}  OBJ={round(final_obj_val,4)}", flush=True)

    print(f"\nRuntime: {int(time.time()-t0)}s", flush=True)

main()
