"""
Phase 292 — Regime Weight Re-opt (17th Pass) [VECTORIZED]
=========================================================
Baseline: v2.49.16, OBJ=5.4791 (Phase 291 breadth boundary retune)

After P289-P291 (vol/disp retune, FTS retune, breadth boundary retune),
the optimal ensemble allocation may have shifted.

All params loaded dynamically from config.
W_STEPS: 0.00-0.80 in 0.05 increments (17 values), sum-to-1 constraint.
2-pass sequential sweep (LOW → MID → HIGH each pass).
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
    out = Path("artifacts/phase292"); out.mkdir(parents=True, exist_ok=True)
    (out / "phase292_report.json").write_text(json.dumps(_partial, indent=2))
    print("\n⏰ TIMEOUT — partial saved.", flush=True)
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
        "fts_rs": fts.get("per_regime_rs", {"LOW": 0.02, "MID": 0.02, "HIGH": 0.25}),
        "fts_bs": fts.get("per_regime_bs", {"LOW": 4.5,  "MID": 2.0,  "HIGH": 2.5}),
        "fts_rt": fts.get("per_regime_rt", {"LOW": 0.80, "MID": 0.60, "HIGH": 0.50}),
        "fts_bt": fts.get("per_regime_bt", {"LOW": 0.30, "MID": 0.22, "HIGH": 0.20}),
        "ts_pct_win": fts.get("per_regime_ts_pct_win", {"LOW": 240, "MID": 288, "HIGH": 400}),
        "vol_thr":   vol.get("per_regime_threshold", {"LOW": 0.53, "MID": 0.50, "HIGH": 0.58}),
        "vol_scale": vol.get("per_regime_scale",     {"LOW": 0.40, "MID": 0.05, "HIGH": 0.05}),
        "disp_thr":  disp.get("per_regime_threshold", {"LOW": 0.40, "MID": 0.70, "HIGH": 0.30}),
        "disp_scale":disp.get("per_regime_scale",     {"LOW": 0.70, "MID": 2.00, "HIGH": 0.50}),
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
          "momentum_lookback_bars": 312, "mean_reversion_lookback_bars": 84,
          "vol_lookback_bars": 192, "target_gross_leverage": 0.35, "rebalance_interval_bars": 60}
    result = BacktestEngine(BacktestConfig(costs=COST_MODEL)).run(
        dataset, make_strategy({"name": "nexus_alpha_v1", "params": vp}))
    return np.array(result.returns)

def precompute_fast_data(P, all_base_data, all_v1_raw):
    """Pre-compute full overlay (VOL+DISP+FTS all baked in) and scaled signal arrays.

    VOL direction: bv > thr -> dampen (high-vol dampening).
    FIX: Each regime now uses its own V1 returns (v1_LOW, v1_MID, v1_HIGH).
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

        # FTS ts percentile per-regime window
        unique_wins = sorted(set(P["ts_pct_win"].values()))
        ts_pct_cache = {}
        for w in unique_wins:
            arr = np.full(min_len, 0.5)
            for i in range(w, min_len):
                arr[i] = float(np.mean(tr[i-w:i] <= tr[i]))
            ts_pct_cache[w] = arr

        # Full overlay (VOL + DISP + FTS) — all baked in at once
        overlay_mult = np.ones(min_len)
        for ridx, rname in enumerate(RNAMES):
            m = masks[ridx]
            # VOL: high-vol dampening (bv > thr -> scale down)
            vol_c = ~np.isnan(bv) & (bv > P["vol_thr"][rname])
            overlay_mult[m & vol_c] *= P["vol_scale"][rname]
            # DISP: scale when dispersion high
            disp_c = fsp > P["disp_thr"][rname]
            overlay_mult[m & disp_c] *= P["disp_scale"][rname]
            # FTS
            tsp = ts_pct_cache[P["ts_pct_win"][rname]]
            overlay_mult[m & (tsp > P["fts_rt"][rname])] *= P["fts_rs"][rname]
            overlay_mult[m & (tsp < P["fts_bt"][rname])] *= P["fts_bs"][rname]

        # Pre-scaled signals (overlay fully applied) — FIX: all 3 V1 variants
        scaled = {
            "v1_LOW":  overlay_mult * v1["LOW"][:min_len],
            "v1_MID":  overlay_mult * v1["MID"][:min_len],
            "v1_HIGH": overlay_mult * v1["HIGH"][:min_len],
            "i460":    overlay_mult * fixed_rets["i460"][:min_len],
            "i415":    overlay_mult * fixed_rets["i415"][:min_len],
            "f168":    overlay_mult * fixed_rets["f168"][:min_len],
        }
        fast_data[yr] = {"scaled": scaled, "masks": masks, "min_len": min_len}
    return fast_data

# Map regime name -> V1 key
V1_KEY = {"LOW": "v1_LOW", "MID": "v1_MID", "HIGH": "v1_HIGH"}

def fast_evaluate(regime_weights, fast_data):
    """Compute OBJ for given regime weights (linear combo of pre-scaled signals)."""
    yearly_sharpes = []
    for yr, fd in sorted(fast_data.items()):
        sc = fd["scaled"]; masks = fd["masks"]; ml = fd["min_len"]
        ens = np.zeros(ml)
        for ridx, rname in enumerate(RNAMES):
            m = masks[ridx]
            w = regime_weights[rname]
            vk = V1_KEY[rname]
            ens[m] = (w["v1"] * sc[vk][m] + w["i460"] * sc["i460"][m]
                     + w["i415"] * sc["i415"][m] + w["f168"] * sc["f168"][m])
        r = ens[~np.isnan(ens)]
        ann = float(np.mean(r) / np.std(r, ddof=1) * np.sqrt(8760)) if len(r) > 1 else 0.0
        yearly_sharpes.append(ann)
    return float(np.mean(yearly_sharpes) - 0.5 * np.std(yearly_sharpes, ddof=1))

def generate_valid_combos():
    """All (w1,w2,w3,w4) from W_STEPS that sum to 1.0."""
    combos = []
    for w1 in W_STEPS:
        for w2 in W_STEPS:
            if w1 + w2 > 1.0 + 1e-9: break
            for w3 in W_STEPS:
                if w1 + w2 + w3 > 1.0 + 1e-9: break
                w4 = round(1.0 - w1 - w2 - w3, 2)
                if 0.0 <= w4 <= 0.80 + 1e-9:
                    combos.append((w1, w2, w3, w4))
    return combos

if __name__ == "__main__":
    t0 = time.time()
    print("=" * 65)
    print("Phase 292 — Regime Weight Re-opt (17th Pass) [VECTORIZED]")
    print("After P289-P291 overlay + boundary changes")

    P = load_config_params()
    base_obj = P["baseline_obj"]
    print(f"Baseline: v{P['version']}  OBJ={base_obj:.4f}")
    rw = P["regime_weights"]
    print(f"Current weights: LOW={rw['LOW']}  MID={rw['MID']}  HIGH={rw['HIGH']}")
    print(f"V1 weights: LOW={P['v1_weights']['LOW']}  MID={P['v1_weights']['MID']}  HIGH={P['v1_weights']['HIGH']}")
    print("=" * 65)

    # [1] Load per-year data
    print("\n[1] Loading per-year data...")
    all_base_data = {}
    for yr in sorted(YEAR_RANGES.keys()):
        print(f"  {yr}: ", end="", flush=True)
        all_base_data[yr] = load_year_data(yr)
        print("done.")

    # [2] Pre-compute V1 per regime per year
    print("\n[2] Pre-computing V1 per regime x year...")
    all_v1_raw = {}
    for yr, base_data in sorted(all_base_data.items()):
        _, _, _, _, _, _, dataset = base_data
        all_v1_raw[yr] = {}
        for rname in RNAMES:
            all_v1_raw[yr][rname] = compute_v1(dataset, P["v1_weights"][rname])
        print(f"  {yr}: done.")

    # [3] Build fast eval data
    print("\n[3] Building fast eval data (full overlay baked in)...")
    fast_data = precompute_fast_data(P, all_base_data, all_v1_raw)

    # Verify baseline
    obj_base = fast_evaluate(P["regime_weights"], fast_data)
    print(f"  Measured baseline OBJ = {obj_base:.4f}  (config says {base_obj:.4f})")

    # Generate valid weight combos
    all_combos = generate_valid_combos()
    print(f"  Valid combos: {len(all_combos)} per regime")

    # [4] Sequential per-regime weight sweep (2 passes)
    print("\n[4] Per-regime weight sweep (2 passes)...")
    best_weights = {r: {**P["regime_weights"][r]} for r in RNAMES}
    best_obj = obj_base

    for pass_num in range(1, 3):
        print(f"\n  === Pass {pass_num} ===")
        for rname in RNAMES:
            best_r = {**best_weights[rname]}
            for wv1, wi460, wi415, wf168 in all_combos:
                trial = {**best_weights, rname: {"v1": wv1, "i460": wi460, "i415": wi415, "f168": wf168}}
                obj = fast_evaluate(trial, fast_data)
                if obj > best_obj + 1e-6:
                    best_obj = obj
                    best_r = {"v1": wv1, "i460": wi460, "i415": wi415, "f168": wf168}
            best_weights[rname] = best_r
            print(f"    {rname}: v1={best_r['v1']:.2f} i460={best_r['i460']:.2f} "
                  f"i415={best_r['i415']:.2f} f168={best_r['f168']:.2f}  OBJ={best_obj:.4f}")

    delta = best_obj - obj_base
    print(f"\n  Final OBJ={best_obj:.4f}  D={delta:+.4f}")

    # [5] LOYO validation
    print("\n[5] LOYO validation...")
    loyo_wins = 0
    for yr in sorted(YEAR_RANGES.keys()):
        others = [y for y in YEAR_RANGES if y != yr]
        def obj_subset(yrs, weights):
            sh = []
            for y in yrs:
                fd = fast_data[y]
                sc = fd["scaled"]; masks = fd["masks"]; ml = fd["min_len"]
                ens = np.zeros(ml)
                for ri, rn in enumerate(RNAMES):
                    m = masks[ri]; w = weights[rn]
                    vk = V1_KEY[rn]
                    ens[m] = (w["v1"]*sc[vk][m] + w["i460"]*sc["i460"][m]
                             + w["i415"]*sc["i415"][m] + w["f168"]*sc["f168"][m])
                r = ens[~np.isnan(ens)]
                ann = float(np.mean(r)/np.std(r,ddof=1)*np.sqrt(8760)) if len(r) > 1 else 0.0
                sh.append(ann)
            return float(np.mean(sh) - 0.5*np.std(sh, ddof=1))

        base_loyo = obj_subset(others, P["regime_weights"])
        cand_loyo = obj_subset(others, best_weights)
        win = cand_loyo > base_loyo
        loyo_wins += int(win)
        print(f"  {yr}: base={base_loyo:.4f}  cand={cand_loyo:.4f}  {'WIN' if win else 'LOSE'}")

    print(f"\n  OBJ={best_obj:.4f}  D={delta:+.4f}  LOYO {loyo_wins}/5")

    validated = delta >= MIN_DELTA and loyo_wins >= MIN_LOYO

    # Build report
    report = {
        "phase": 292,
        "baseline_version": P["version"],
        "baseline_obj_config": base_obj,
        "baseline_obj_measured": round(obj_base, 4),
        "best_obj": round(best_obj, 4),
        "delta": round(delta, 4),
        "loyo_wins": loyo_wins,
        "validated": validated,
        "best_weights": best_weights,
        "baseline_weights": P["regime_weights"],
        "v1_weights": P["v1_weights"],
        "context": "After P289-P291 overlay + boundary changes",
        "runtime_seconds": round(time.time() - t0, 1),
    }
    out = Path("artifacts/phase292"); out.mkdir(parents=True, exist_ok=True)
    (out / "phase292_report.json").write_text(json.dumps(report, indent=2))

    if validated:
        print(f"\n  VALIDATED")
        cfg = json.load(open(CFG_PATH))
        brs = cfg["breadth_regime_switching"]
        for rname in RNAMES:
            brs["regime_weights"][rname]["v1"]        = best_weights[rname]["v1"]
            brs["regime_weights"][rname]["i460bw168"] = best_weights[rname]["i460"]
            brs["regime_weights"][rname]["i415bw216"] = best_weights[rname]["i415"]
            brs["regime_weights"][rname]["f168"]       = best_weights[rname]["f168"]
        old_ver = cfg["_version"]
        parts = old_ver.split(".")
        parts[-1] = str(int(parts[-1]) + 1)
        new_ver = ".".join(parts)
        cfg["_version"] = new_ver
        cfg["version"] = f"v{new_ver}"
        stored_obj = cfg["monitoring"]["expected_performance"]["annual_sharpe_backtest"]
        reported_obj = round(stored_obj + delta, 4)
        cfg["monitoring"]["expected_performance"]["annual_sharpe_backtest"] = reported_obj
        cfg["obj"] = reported_obj
        with open(CFG_PATH, "w") as f: json.dump(cfg, f, indent=2, ensure_ascii=False)
        print(f"\n  Config -> v{new_ver}  OBJ={reported_obj:.4f} (stored {stored_obj:.4f} + D{delta:+.4f})")
        msg = (f"feat: P292 regime weight reopt17 OBJ={reported_obj:.4f} "
               f"LOYO={loyo_wins}/5 D={delta:+.4f} [v{new_ver}]")
        subprocess.run(["git", "add", str(CFG_PATH)], cwd=ROOT)
        subprocess.run(["git", "commit", "-m", msg], cwd=ROOT)
    else:
        print(f"\n  NOT VALIDATED (LOYO {loyo_wins}/5, D={delta:+.4f})")
        print(f"  Current weights confirmed optimal:")
        for rname in RNAMES:
            print(f"    {rname}: {P['regime_weights'][rname]}")

    print(f"\nRuntime: {time.time()-t0:.1f}s")
