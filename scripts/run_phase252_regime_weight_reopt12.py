"""
Phase 252 — Regime Weight Re-opt (12th Pass, Vectorized)
=========================================================
Loads ALL params dynamically from current config.
After VOL overlay changes in v2.48.x, regime weights may have shifted.

Uses numpy vectorized approach: precompute overlay_mult once per year,
then evaluate all weight combos in microseconds each.
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
    out = Path("artifacts/phase252"); out.mkdir(parents=True, exist_ok=True)
    (out / "phase252_report.json").write_text(json.dumps(_partial, indent=2))
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

BRD_LB = 192; PCT_WINDOW = 336; FUND_DISP_PCT = 240
TS_SHORT = 16; TS_LONG = 72; VOL_WINDOW = 168
RNAMES = ["LOW", "MID", "HIGH"]

MIN_DELTA = 0.005; MIN_LOYO = 3

# Weight grid: steps of 0.05 for 4 signals (v1, i460, i415, f168), sum=1.0
W_VALS = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45,
          0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.0]

COST_MODEL = cost_model_from_config({
    "fee_rate": 0.0005, "slippage_rate": 0.0003, "cost_multiplier": 1.0,
})

def load_config_params():
    cfg_path = ROOT / "configs" / "production_p91b_champion.json"
    cfg = json.load(open(cfg_path))
    brs = cfg["breadth_regime_switching"]
    fts = cfg.get("fts_overlay_params", {})
    vol = cfg.get("vol_overlay_params", {})
    disp = cfg.get("disp_overlay_params", {})
    rw = brs["regime_weights"]

    def safe_dict(d, regime_key, fallback):
        if isinstance(d, dict):
            if regime_key in d: return float(d[regime_key])
            return float(fallback)
        return float(fallback)

    p_low  = float(brs.get("p_low", 0.30))
    p_high = float(brs.get("p_high", 0.68))

    # regime weights (for baseline)
    regime_weights = {}
    for rname in RNAMES:
        w = rw[rname]
        regime_weights[rname] = {
            "v1":   float(w.get("v1", 0.0)),
            "i460": float(w.get("i460bw168", w.get("i460", 0.0))),
            "i415": float(w.get("i415bw216", w.get("i415", 0.0))),
            "f168": float(w.get("f168", 0.0)),
        }

    # FTS per-regime
    fts_rs = {r: safe_dict(fts.get("per_regime_rs"), r, 0.25) for r in RNAMES}
    fts_bs = {r: safe_dict(fts.get("per_regime_bs"), r, 1.0)  for r in RNAMES}
    fts_rt = {r: safe_dict(fts.get("per_regime_rt"), r, 0.65) for r in RNAMES}
    fts_bt = {r: safe_dict(fts.get("per_regime_bt"), r, 0.35) for r in RNAMES}
    ts_pct_win = {r: int(fts.get("per_regime_ts_pct_win", {}).get(r, 288)) for r in RNAMES}

    # VOL per-regime
    vol_thr   = {r: safe_dict(vol.get("per_regime_threshold"), r, 0.50) for r in RNAMES}
    vol_scale = {r: safe_dict(vol.get("per_regime_scale"),     r, 0.40) for r in RNAMES}

    # DISP per-regime
    disp_thr   = {r: safe_dict(disp.get("per_regime_threshold"), r, 0.70) for r in RNAMES}
    disp_scale = {r: safe_dict(disp.get("per_regime_scale"),     r, 0.50) for r in RNAMES}

    baseline_obj = float(cfg.get("monitoring", {}).get("expected_performance", {}).get("annual_sharpe_backtest", 4.2))

    # V1 per-regime params
    v1prw = cfg.get("v1_per_regime_weights", {})
    def v1p(rname):
        d = v1prw.get(rname, {})
        return {
            "k_per_side": 2,
            "w_carry": float(d.get("w_carry", 0.25)),
            "w_mom": float(d.get("w_mom", 0.50)),
            "w_mean_reversion": float(d.get("w_mean_reversion", 0.25)),
            "momentum_lookback_bars": 336,
            "mean_reversion_lookback_bars": 84,
            "vol_lookback_bars": 192,
            "target_gross_leverage": 0.35,
            "rebalance_interval_bars": 60,
        }

    return {
        "p_low": p_low, "p_high": p_high,
        "regime_weights": regime_weights,
        "fts_rs": fts_rs, "fts_bs": fts_bs, "fts_rt": fts_rt, "fts_bt": fts_bt,
        "ts_pct_win": ts_pct_win,
        "vol_thr": vol_thr, "vol_scale": vol_scale,
        "disp_thr": disp_thr, "disp_scale": disp_scale,
        "baseline_obj": baseline_obj,
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
    fixed_rets = {}
    I460_P = {"k_per_side": 4, "lookback_bars": 480, "beta_window_bars": 168,
              "target_gross_leverage": 0.3, "rebalance_interval_bars": 48}
    I415_P = {"k_per_side": 4, "lookback_bars": 415, "beta_window_bars": 144,
              "target_gross_leverage": 0.3, "rebalance_interval_bars": 48}
    F168_P = {"k_per_side": 2, "funding_lookback_bars": 168, "direction": "contrarian",
              "target_gross_leverage": 0.25, "rebalance_interval_bars": 36}
    for sk, sname, params in [
        ("i460", "idio_momentum_alpha",    I460_P),
        ("i415", "idio_momentum_alpha",    I415_P),
        ("f168", "funding_momentum_alpha", F168_P),
    ]:
        result = BacktestEngine(BacktestConfig(costs=COST_MODEL)).run(
            dataset, make_strategy({"name": sname, "params": params}))
        fixed_rets[sk] = np.array(result.returns)
    return btc_vol, fund_std_pct, ts_raw, brd_pct, fixed_rets, n, dataset

def compute_v1(dataset, params):
    result = BacktestEngine(BacktestConfig(costs=COST_MODEL)).run(
        dataset, make_strategy({"name": "nexus_alpha_v1", "params": params}))
    return np.array(result.returns)

def precompute_fast_eval_data(all_base_data, all_v1_raw, p):
    fast = {}
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

        regime_idx = np.where(bpct < p["p_low"], 0, np.where(bpct > p["p_high"], 2, 1))
        masks = [regime_idx == i for i in range(3)]

        # ts_pct per regime
        ts_pct_cache = {}
        for rname in RNAMES:
            w = p["ts_pct_win"][rname]
            if w not in ts_pct_cache:
                arr = np.full(min_len, 0.5)
                for i in range(w, min_len):
                    arr[i] = float(np.mean(tr[i-w:i] <= tr[i]))
                ts_pct_cache[w] = arr

        overlay_mult = np.ones(min_len)
        for ridx, rname in enumerate(RNAMES):
            m = masks[ridx]
            vm = m & ~np.isnan(bv) & (bv > p["vol_thr"][rname])
            overlay_mult[vm] *= p["vol_scale"][rname]
            dm = m & (fsp > p["disp_thr"][rname])
            overlay_mult[dm] *= p["disp_scale"][rname]
            tsp = ts_pct_cache[p["ts_pct_win"][rname]]
            overlay_mult[m & (tsp > p["fts_rt"][rname])] *= p["fts_rs"][rname]
            overlay_mult[m & (tsp < p["fts_bt"][rname])] *= p["fts_bs"][rname]

        scaled = {
            "v1_LOW": overlay_mult * v1["LOW"][:min_len],
            "v1_MID": overlay_mult * v1["MID"][:min_len],
            "i460":   overlay_mult * fixed_rets["i460"][:min_len],
            "i415":   overlay_mult * fixed_rets["i415"][:min_len],
            "f168":   overlay_mult * fixed_rets["f168"][:min_len],
        }
        fast[yr] = (masks, min_len, scaled)
    return fast

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

def sweep_regime_weights(target_regime, current_weights, fast_data, base_obj):
    """Sweep all valid 4-signal weight combos for target_regime."""
    best_obj = base_obj
    best_w = dict(current_weights[target_regime])
    all_weights = dict(current_weights)

    for v1 in W_VALS:
        for i460 in W_VALS:
            for i415 in W_VALS:
                f168 = round(1.0 - v1 - i460 - i415, 10)
                if f168 < -1e-9 or f168 > 1.0 + 1e-9: continue
                f168 = max(0.0, min(1.0, f168))
                if abs(v1 + i460 + i415 + f168 - 1.0) > 1e-6: continue
                w = {"v1": v1, "i460": i460, "i415": i415, "f168": f168}
                all_weights[target_regime] = w
                o, _ = fast_evaluate(all_weights, fast_data)
                if o > best_obj:
                    best_obj = o
                    best_w = dict(w)
    return best_w, best_obj

def main():
    t0 = time.time()
    p = load_config_params()

    print("=" * 65)
    print("Phase 252 — Regime Weight Re-opt (12th Pass, Vectorized)")
    print(f"Baseline: v{p['config_version']}  OBJ={p['baseline_obj']:.4f}")
    print(f"p_low={p['p_low']}, p_high={p['p_high']}")
    print(f"VOL thr={p['vol_thr']}, scale={p['vol_scale']}")
    print(f"ts_pct_win={p['ts_pct_win']}")
    print("=" * 65, flush=True)

    BASELINE_OBJ = p["baseline_obj"]

    print("\n[1] Loading per-year data & signals...", flush=True)
    all_base_data = {}
    for yr in sorted(YEAR_RANGES):
        print(f"  {yr}: ", end="", flush=True)
        all_base_data[yr] = load_year_data(yr, p)
        print("done.", flush=True)

    print("\n[2] Pre-computing V1 returns (per-regime)...", flush=True)
    all_v1_raw = {}
    for yr in sorted(YEAR_RANGES):
        ds = all_base_data[yr][6]
        v1_low = compute_v1(ds, p["v1_params"]["LOW"])
        # MID and HIGH share same V1 params if identical
        if p["v1_params"]["MID"] == p["v1_params"]["HIGH"]:
            v1_mh = compute_v1(ds, p["v1_params"]["MID"])
            all_v1_raw[yr] = {"LOW": v1_low, "MID": v1_mh, "HIGH": v1_mh}
        else:
            v1_mid  = compute_v1(ds, p["v1_params"]["MID"])
            v1_high = compute_v1(ds, p["v1_params"]["HIGH"])
            all_v1_raw[yr] = {"LOW": v1_low, "MID": v1_mid, "HIGH": v1_high}
        print(f"  {yr}: done.", flush=True)

    print("\n[3] Precomputing fast eval data...", flush=True)
    fast_data = precompute_fast_eval_data(all_base_data, all_v1_raw, p)

    base_obj_val, base_yearly = fast_evaluate(p["regime_weights"], fast_data)
    print(f"  Baseline OBJ = {base_obj_val:.4f}  (expected ≈ {BASELINE_OBJ:.4f})", flush=True)
    _partial["baseline_obj"] = float(base_obj_val)

    print("\n[4] Sequential per-regime weight sweep (2 passes)...", flush=True)
    best_weights = {r: dict(p["regime_weights"][r]) for r in RNAMES}
    running_best = base_obj_val

    for pass_num in [1, 2]:
        print(f"\n  === Pass {pass_num} ===", flush=True)
        for rname in RNAMES:
            best_w, running_best = sweep_regime_weights(rname, best_weights, fast_data, running_best)
            best_weights[rname] = best_w
            print(f"    {rname}: {best_w}  OBJ={running_best:.4f}  Δ={running_best-base_obj_val:+.4f}", flush=True)

    final_obj_val, final_yearly = fast_evaluate(best_weights, fast_data)
    delta = final_obj_val - base_obj_val
    print(f"\n  → Best regime weights:")
    for rname in RNAMES:
        print(f"    {rname}: {best_weights[rname]}")
    print(f"  Final OBJ={final_obj_val:.4f}  Δ={delta:+.4f}", flush=True)

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
        "phase": 252, "baseline_obj": float(base_obj_val),
        "best_obj": float(final_obj_val), "delta": float(delta),
        "loyo_wins": loyo_wins, "loyo_total": len(final_yearly),
        "validated": validated, "best_weights": {r: best_weights[r] for r in RNAMES},
    }
    _partial.update(result)
    out = Path("artifacts/phase252"); out.mkdir(parents=True, exist_ok=True)
    (out / "phase252_report.json").write_text(json.dumps(result, indent=2))

    if validated:
        cfg_path = ROOT / "configs" / "production_p91b_champion.json"
        cfg = json.load(open(cfg_path))
        brs = cfg["breadth_regime_switching"]
        for rname in RNAMES:
            w = best_weights[rname]
            brs["regime_weights"][rname] = {
                "v1": w["v1"],
                "i460bw168": w["i460"],
                "i415bw216": w["i415"],
                "f168": w["f168"],
            }
        cur_ver = cfg.get("_version", "2.48.0")
        parts = cur_ver.split(".")
        new_ver = f"{parts[0]}.{int(parts[1])+1}.0"
        cfg["_version"] = new_ver
        cfg["monitoring"]["expected_performance"]["annual_sharpe_backtest"] = round(final_obj_val, 4)
        with open(cfg_path, "w") as f:
            json.dump(cfg, f, indent=2, ensure_ascii=False)
        print(f"\n  Config → {new_ver}  OBJ={round(final_obj_val,4)}", flush=True)

    print(f"\nRuntime: {int(time.time()-t0)}s", flush=True)

main()
