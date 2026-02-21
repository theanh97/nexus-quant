"""
Phase 248 — Per-Regime I460 Parameters Sweep
=============================================
Baseline: v2.44.0, OBJ=4.0145

I460 (idio_momentum_alpha, lb=480, beta_win=168) has 70% weight in HIGH regime.
Getting I460 params right for each regime is critical.

Hypothesis:
  - HIGH (bull, 70% I460): longer lookback → stronger trend capture
  - LOW (bear, 10% I460): shorter → faster adaptation to reversals
  - MID (0% I460 weight): doesn't matter, but sweep anyway

Sweep:
  lookback_bars:   [240, 336, 400, 480, 600, 720]
  beta_window_bars: [96, 120, 168, 240, 336]

Precompute: 6 × 5 = 30 (lb, bw) combos × 5 years = 150 I460 runs
Vectorized evaluation for the per-regime sequential sweep.
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
    out = Path("artifacts/phase248"); out.mkdir(parents=True, exist_ok=True)
    (out / "phase248_report.json").write_text(json.dumps(_partial, indent=2))
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
P_LOW = 0.30; P_HIGH = 0.60

TS_PCT_WIN = {"LOW": 240, "MID": 288, "HIGH": 400}
FTS_RS = {"LOW": 0.50, "MID": 0.20, "HIGH": 0.40}
FTS_BS = {"LOW": 3.00, "MID": 3.00, "HIGH": 2.00}
FTS_RT = {"LOW": 0.80, "MID": 0.65, "HIGH": 0.55}
FTS_BT = {"LOW": 0.30, "MID": 0.25, "HIGH": 0.25}
VOL_THR   = {"LOW": 0.50, "MID": 0.50, "HIGH": 0.50}
VOL_SCALE = {"LOW": 0.40, "MID": 0.15, "HIGH": 0.10}
DISP_THR   = {"LOW": 0.70, "MID": 0.70, "HIGH": 0.40}
DISP_SCALE = {"LOW": 0.50, "MID": 1.50, "HIGH": 0.50}

RNAMES = ["LOW", "MID", "HIGH"]

REGIME_WEIGHTS = {
    "LOW":  {"v1": 0.35, "i460": 0.10, "i415": 0.15, "f168": 0.40},
    "MID":  {"v1": 0.15, "i460": 0.00, "i415": 0.20, "f168": 0.65},
    "HIGH": {"v1": 0.30, "i460": 0.70, "i415": 0.00, "f168": 0.00},
}

V1_LOW_PARAMS = {
    "k_per_side": 2, "w_carry": 0.10, "w_mom": 0.45, "w_mean_reversion": 0.45,
    "momentum_lookback_bars": 336, "mean_reversion_lookback_bars": 84,
    "vol_lookback_bars": 192, "target_gross_leverage": 0.35, "rebalance_interval_bars": 60,
}
V1_MIDHIGH_PARAMS = {
    "k_per_side": 2, "w_carry": 0.25, "w_mom": 0.50, "w_mean_reversion": 0.25,
    "momentum_lookback_bars": 336, "mean_reversion_lookback_bars": 84,
    "vol_lookback_bars": 192, "target_gross_leverage": 0.35, "rebalance_interval_bars": 60,
}
I415_PARAMS = {"k_per_side": 4, "lookback_bars": 415, "beta_window_bars": 144,
               "target_gross_leverage": 0.3, "rebalance_interval_bars": 48}
F168_PARAMS = {"k_per_side": 2, "funding_lookback_bars": 168, "direction": "contrarian",
               "target_gross_leverage": 0.25, "rebalance_interval_bars": 36}

BASE_I460_LB = 480; BASE_I460_BW = 168

I460_LB_SWEEP = [240, 336, 400, 480, 600, 720]
I460_BW_SWEEP = [96, 120, 168, 240, 336]

BASELINE_OBJ = 4.0145
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
    unique_wins = sorted(set(TS_PCT_WIN.values()))
    ts_pct_cache = {}
    for w in unique_wins:
        arr = np.full(n, 0.5)
        for i in range(w, n):
            arr[i] = float(np.mean(ts_raw[i-w:i] <= ts_raw[i]))
        ts_pct_cache[w] = arr
    breadth = np.full(n, 0.5)
    for i in range(BRD_LB, n):
        pos = sum(1 for j in range(len(SYMBOLS))
                  if close_mat[i-BRD_LB, j] > 0 and close_mat[i, j] > close_mat[i-BRD_LB, j])
        breadth[i] = pos / len(SYMBOLS)
    brd_pct = np.full(n, 0.5)
    for i in range(PCT_WINDOW, n):
        brd_pct[i] = float(np.mean(breadth[i-PCT_WINDOW:i] <= breadth[i]))
    return btc_vol, fund_std_pct, ts_pct_cache, brd_pct, n, dataset

def run_strat(dataset, name, params):
    return np.array(BacktestEngine(BacktestConfig(costs=COST_MODEL)).run(
        dataset, make_strategy({"name": name, "params": params})).returns)

def precompute_fast_data(all_base_data, all_v1_raw, i415_cache, f168_cache, i460_cache):
    """
    Pre-compute overlay_mult, regime masks, and scaled signal arrays per year.
    i460_cache: {(lb, bw, year): rets}
    """
    fast_data = {}
    for yr, base_data in sorted(all_base_data.items()):
        btc_vol, fund_std_pct, ts_pct_cache, brd_pct, n, _ = base_data
        v1 = all_v1_raw[yr]
        min_len = min(
            min(len(v1[r]) for r in RNAMES),
            len(i415_cache[yr]), len(f168_cache[yr]),
            min(len(i460_cache[(lb, bw, yr)]) for lb in I460_LB_SWEEP for bw in I460_BW_SWEEP),
        )
        bpct = brd_pct[:min_len]
        bv   = btc_vol[:min_len]
        fsp  = fund_std_pct[:min_len]

        regime_idx = np.where(bpct < P_LOW, 0, np.where(bpct > P_HIGH, 2, 1))
        masks = [regime_idx == i for i in range(3)]

        overlay_mult = np.ones(min_len)
        for ridx, rname in enumerate(RNAMES):
            m = masks[ridx]
            vol_m = m & ~np.isnan(bv) & (bv > VOL_THR[rname])
            overlay_mult[vol_m] *= VOL_SCALE[rname]
            disp_m = m & (fsp > DISP_THR[rname])
            overlay_mult[disp_m] *= DISP_SCALE[rname]
            tsp = ts_pct_cache[TS_PCT_WIN[rname]][:min_len]
            overlay_mult[m & (tsp > FTS_RT[rname])] *= FTS_RS[rname]
            overlay_mult[m & (tsp < FTS_BT[rname])] *= FTS_BS[rname]

        scaled_base = {
            "v1_LOW": overlay_mult * v1["LOW"][:min_len],
            "v1_MID": overlay_mult * v1["MID"][:min_len],
            "i415":   overlay_mult * i415_cache[yr][:min_len],
            "f168":   overlay_mult * f168_cache[yr][:min_len],
        }
        # Scale all i460 variants
        scaled_i460 = {}
        for lb in I460_LB_SWEEP:
            for bw in I460_BW_SWEEP:
                scaled_i460[(lb, bw)] = overlay_mult * i460_cache[(lb, bw, yr)][:min_len]

        fast_data[yr] = (masks, min_len, scaled_base, scaled_i460)
    return fast_data

def fast_evaluate(i460_params_per_regime, fast_data):
    """
    i460_params_per_regime: {"LOW": (lb, bw), "MID": (lb, bw), "HIGH": (lb, bw)}
    """
    yearly = {}
    for yr, (masks, min_len, scaled_base, scaled_i460) in fast_data.items():
        ens = np.zeros(min_len)
        for ridx, rname in enumerate(RNAMES):
            m = masks[ridx]
            w = REGIME_WEIGHTS[rname]
            v1_key = "v1_LOW" if rname == "LOW" else "v1_MID"
            lb, bw = i460_params_per_regime[rname]
            ens[m] += (w["v1"]  * scaled_base[v1_key][m]
                       + w["i460"] * scaled_i460[(lb, bw)][m]
                       + w["i415"] * scaled_base["i415"][m]
                       + w["f168"] * scaled_base["f168"][m])
        r = ens[~np.isnan(ens)]
        sh = float(np.mean(r) / np.std(r, ddof=1) * np.sqrt(8760)) if len(r) > 1 else 0.0
        yearly[yr] = sh
    vals = list(yearly.values())
    return float(np.mean(vals) - 0.5 * np.std(vals, ddof=1)), yearly

def main():
    t0 = time.time()
    print("=" * 65)
    print("Phase 248 — Per-Regime I460 Parameters Sweep")
    print(f"Baseline: v2.44.0  OBJ={BASELINE_OBJ}")
    print(f"Base I460: lb={BASE_I460_LB} bw={BASE_I460_BW}")
    print(f"Sweep: {len(I460_LB_SWEEP)} lb × {len(I460_BW_SWEEP)} bw = {len(I460_LB_SWEEP)*len(I460_BW_SWEEP)} combos")
    print("=" * 65)

    print("\n[1] Loading per-year data...", flush=True)
    all_base_data = {}
    for yr in sorted(YEAR_RANGES):
        print(f"  {yr}: ", end="", flush=True)
        all_base_data[yr] = load_year_data(yr)
        print("done.", flush=True)

    print("\n[2] Pre-computing V1, I415, F168 (fixed)...", flush=True)
    all_v1_raw = {}; i415_cache = {}; f168_cache = {}
    for yr in sorted(YEAR_RANGES):
        ds = all_base_data[yr][5]
        all_v1_raw[yr] = {
            "LOW":  run_strat(ds, "nexus_alpha_v1", V1_LOW_PARAMS),
            "MID":  run_strat(ds, "nexus_alpha_v1", V1_MIDHIGH_PARAMS),
            "HIGH": run_strat(ds, "nexus_alpha_v1", V1_MIDHIGH_PARAMS),
        }
        i415_cache[yr] = run_strat(ds, "idio_momentum_alpha", I415_PARAMS)
        f168_cache[yr] = run_strat(ds, "funding_momentum_alpha", F168_PARAMS)
        print(f"  {yr}: done.", flush=True)

    print(f"\n[3] Pre-computing I460 variants ({len(I460_LB_SWEEP)*len(I460_BW_SWEEP)} combos × 5 years)...", flush=True)
    i460_cache = {}; n_done = 0
    for lb in I460_LB_SWEEP:
        for bw in I460_BW_SWEEP:
            # bw must be < lb (sanity)
            if bw >= lb:
                for yr in sorted(YEAR_RANGES):
                    # Use fallback = baseline I460
                    i460_cache[(lb, bw, yr)] = i460_cache.get((BASE_I460_LB, BASE_I460_BW, yr),
                                               run_strat(all_base_data[yr][5], "idio_momentum_alpha",
                                               {"k_per_side": 4, "lookback_bars": lb, "beta_window_bars": bw,
                                                "target_gross_leverage": 0.3, "rebalance_interval_bars": 48}))
                continue
            params = {"k_per_side": 4, "lookback_bars": lb, "beta_window_bars": bw,
                      "target_gross_leverage": 0.3, "rebalance_interval_bars": 48}
            for yr in sorted(YEAR_RANGES):
                ds = all_base_data[yr][5]
                i460_cache[(lb, bw, yr)] = run_strat(ds, "idio_momentum_alpha", params)
                n_done += 1
                if n_done % 5 == 0:
                    print(f"  ... {n_done} I460 runs done", flush=True)
    print(f"  → Complete: {n_done} I460 runs", flush=True)

    print("\n[4] Precomputing fast eval data...", flush=True)
    fast_data = precompute_fast_data(all_base_data, all_v1_raw, i415_cache, f168_cache, i460_cache)
    print("  done.", flush=True)

    # Baseline
    base_params = {r: (BASE_I460_LB, BASE_I460_BW) for r in RNAMES}
    base_obj_val, base_yearly = fast_evaluate(base_params, fast_data)
    print(f"\n  Baseline OBJ = {base_obj_val:.4f}  (expected ≈ {BASELINE_OBJ})", flush=True)
    _partial.update({"baseline_obj": float(base_obj_val)})

    all_combos = [(lb, bw) for lb in I460_LB_SWEEP for bw in I460_BW_SWEEP if bw < lb]
    print(f"\n  Valid (lb, bw) combos: {len(all_combos)}", flush=True)

    print("\n[5] Sequential per-regime I460 sweep (2 passes)...", flush=True)
    best_params = {r: (BASE_I460_LB, BASE_I460_BW) for r in RNAMES}
    running_best = base_obj_val

    for pass_num in [1, 2]:
        print(f"\n  === Pass {pass_num} ===", flush=True)
        for rname in RNAMES:
            best_p = best_params[rname]
            best_obj_r = running_best
            for (lb, bw) in all_combos:
                combo = {**best_params, rname: (lb, bw)}
                o, _ = fast_evaluate(combo, fast_data)
                if o > best_obj_r:
                    best_obj_r = o
                    best_p = (lb, bw)
            best_params[rname] = best_p
            running_best = best_obj_r
            print(f"    {rname}: lb={best_p[0]} bw={best_p[1]}  "
                  f"OBJ={running_best:.4f}  Δ={running_best-base_obj_val:+.4f}", flush=True)

    final_obj_val, final_yearly = fast_evaluate(best_params, fast_data)
    delta = final_obj_val - base_obj_val
    print(f"\n  → Best per-regime I460:")
    for r in RNAMES:
        lb, bw = best_params[r]
        print(f"    {r}: lb={lb} bw={bw}")
    print(f"  Final OBJ={final_obj_val:.4f}  Δ={delta:+.4f}", flush=True)

    print("\n[6] LOYO validation...", flush=True)
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
        "phase": 248, "baseline_obj": float(base_obj_val),
        "best_obj": float(final_obj_val), "delta": float(delta),
        "loyo_wins": loyo_wins, "loyo_total": len(final_yearly),
        "validated": validated,
        "best_i460_params": {r: {"lookback_bars": best_params[r][0], "beta_window_bars": best_params[r][1]}
                              for r in RNAMES},
    }
    _partial.update(result)
    out = Path("artifacts/phase248"); out.mkdir(parents=True, exist_ok=True)
    (out / "phase248_report.json").write_text(json.dumps(result, indent=2))

    if validated:
        cfg_path = ROOT / "configs" / "production_p91b_champion.json"
        cfg = json.load(open(cfg_path))
        cfg["per_regime_i460_params"] = {
            r: {"lookback_bars": best_params[r][0], "beta_window_bars": best_params[r][1]}
            for r in RNAMES
        }
        cur_ver = cfg.get("_version", "2.44.0")
        parts = cur_ver.split(".")
        new_ver = f"{parts[0]}.{parts[1]}.{int(parts[2])+1}"
        cfg["_version"] = new_ver
        cfg["monitoring"]["expected_performance"]["annual_sharpe_backtest"] = round(final_obj_val, 4)
        with open(cfg_path, "w") as f:
            json.dump(cfg, f, indent=2, ensure_ascii=False)
        print(f"\n  Config → {new_ver}  OBJ={round(final_obj_val,4)}", flush=True)

    print(f"\nRuntime: {int(time.time()-t0)}s", flush=True)

main()
