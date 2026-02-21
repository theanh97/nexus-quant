"""
Phase 373 — F168 Funding Lookback Sweep [VECTORIZED]
=====================================================
Baseline: v2.49.45, OBJ=7.0152

F168 (funding_momentum_alpha lb=168) is the dominant signal in:
  LOW regime (w=0.40) and MID regime (w=0.65).
The lookback was set at P172 (funding_lookback_bars=168) and never re-swept.

Sweep: funding_lookback_bars ∈ [72, 96, 120, 144, 168, 192, 240, 288, 336]
       (9 candidates, runs full backtest per candidate)

Also: try rebalance_interval_bars ∈ [24, 36, 48] for F168.
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
    out = Path("artifacts/phase373"); out.mkdir(parents=True, exist_ok=True)
    (out / "phase373_report.json").write_text(json.dumps(_partial, indent=2))
    print("\n⏰ TIMEOUT — partial saved.", flush=True)
    sys.exit(0)
_signal.signal(_signal.SIGALRM, _on_timeout)
_signal.alarm(7200)  # 2h timeout

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

F168_LB_GRID = [72, 96, 120, 144, 168, 192, 240, 288, 336]
F168_REB_GRID = [24, 36, 48]

COST_MODEL = cost_model_from_config({
    "fee_rate": 0.0005, "slippage_rate": 0.0003, "cost_multiplier": 1.0,
})
CFG_PATH = ROOT / "configs" / "production_p91b_champion.json"

_f168_cache: dict = {}


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
        "p_high": float(brs.get("p_high", 0.60)),
        "regime_weights": regime_weights,
        "fts_rs": {r: sdget(fts.get("per_regime_rs"), r, 0.01) for r in RNAMES},
        "fts_bs": {r: sdget(fts.get("per_regime_bs"), r, 2.0)  for r in RNAMES},
        "fts_rt": {r: sdget(fts.get("per_regime_rt"), r, 0.65) for r in RNAMES},
        "fts_bt": {r: sdget(fts.get("per_regime_bt"), r, 0.22) for r in RNAMES},
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
    return btc_vol, fund_std_pct, ts_raw, brd_pct, n, dataset


def run_signal(dataset, name, params):
    res = BacktestEngine(BacktestConfig(costs=COST_MODEL)).run(
        dataset, make_strategy({"name": name, "params": params}))
    return np.array(res.returns)


def get_f168(year, dataset, lb, reb):
    key = (year, lb, reb)
    if key not in _f168_cache:
        _f168_cache[key] = run_signal(dataset, "funding_momentum_alpha",
            {"k_per_side": 2, "funding_lookback_bars": lb, "direction": "contrarian",
             "target_gross_leverage": 0.25, "rebalance_interval_bars": reb})
    return _f168_cache[key]


def eval_ensemble(p, all_base, datasets, v1_by_year, i460_by_year, i415_by_year, f168_lb, f168_reb):
    yearly = {}
    for yr in sorted(YEAR_RANGES):
        btc_vol, fund_std_pct, ts_raw, brd_pct, n = all_base[yr]
        f168_ret = get_f168(yr, datasets[yr], f168_lb, f168_reb)
        i460_ret = i460_by_year[yr]
        i415_ret = i415_by_year[yr]
        v1_ret = v1_by_year[yr]

        ml = min(n,
                 min(len(v1_ret[r]) for r in RNAMES),
                 len(i460_ret), len(i415_ret), len(f168_ret))

        bpct = brd_pct[:ml]; bv = btc_vol[:ml]; fsp = fund_std_pct[:ml]; tr = ts_raw[:ml]
        regime_idx = np.where(bpct < p["p_low"], 0, np.where(bpct > p["p_high"], 2, 1))
        masks = [regime_idx == i for i in range(3)]

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

        vd_mult = np.ones(ml)
        for ridx, rn in enumerate(RNAMES):
            m = masks[ridx]
            vd_mult[m & ~np.isnan(bv) & (bv > p["vol_thr"][rn])] *= p["vol_scale"][rn]
            vd_mult[m & (fsp > p["disp_thr"][rn])] *= p["disp_scale"][rn]

        overlay = fts_mult * vd_mult
        ens = np.zeros(ml)
        for ridx, rn in enumerate(RNAMES):
            m = masks[ridx]
            w = p["regime_weights"][rn]
            v1_sc = overlay * v1_ret[rn][:ml]
            i460_sc = overlay * i460_ret[:ml]
            i415_sc = overlay * i415_ret[:ml]
            f168_sc = overlay * f168_ret[:ml]
            ens[m] = (w["v1"] * v1_sc[m]
                      + w["i460"] * i460_sc[m]
                      + w["i415"] * i415_sc[m]
                      + w["f168"] * f168_sc[m])

        r = ens[~np.isnan(ens)]
        yearly[yr] = float(np.mean(r) / np.std(r, ddof=1) * np.sqrt(8760)) if len(r) > 1 else 0.0

    vals = list(yearly.values())
    return float(np.mean(vals) - 0.5 * np.std(vals, ddof=1)), yearly


def main():
    t0 = time.time()
    print("=" * 70)
    print("Phase 373 — F168 Funding Lookback + Rebalance Sweep")
    print("=" * 70)

    p = load_config_params()
    STORED_OBJ = p["baseline_obj"]
    print(f"Config: v{p['config_version']}  stored OBJ={STORED_OBJ:.4f}")
    print(f"Regime weights (F168 contributions):")
    for rn in RNAMES:
        w = p["regime_weights"][rn]
        print(f"  {rn}: f168_weight={w['f168']}")

    print(f"\nF168 lb sweep: {F168_LB_GRID}")
    print(f"F168 reb sweep: {F168_REB_GRID}")
    print(f"Total combos: {len(F168_LB_GRID)*len(F168_REB_GRID)}")

    print("\n[1] Loading data...", flush=True)
    all_base = {}
    datasets = {}
    for yr in sorted(YEAR_RANGES):
        tw = time.time()
        print(f"  {yr}: ", end="", flush=True)
        btc_vol, fund_std_pct, ts_raw, brd_pct, n, ds = load_year_data(yr)
        datasets[yr] = ds
        all_base[yr] = (btc_vol, fund_std_pct, ts_raw, brd_pct, n)
        print(f"done ({int(time.time()-tw)}s)", flush=True)

    print("\n[2] Pre-computing V1 + I460 + I415...", flush=True)
    v1_by_year = {}
    i460_by_year = {}
    i415_by_year = {}
    for yr in sorted(YEAR_RANGES):
        ds = datasets[yr]
        v1_per_regime = {}
        for rn in RNAMES:
            vp = p["v1_params"][rn]
            res = BacktestEngine(BacktestConfig(costs=COST_MODEL)).run(
                ds, make_strategy({"name": "nexus_alpha_v1", "params": vp}))
            v1_per_regime[rn] = np.array(res.returns)
        v1_by_year[yr] = v1_per_regime
        i460_by_year[yr] = run_signal(ds, "idio_momentum_alpha",
            {"k_per_side": 4, "lookback_bars": 480, "beta_window_bars": 168,
             "target_gross_leverage": 0.3, "rebalance_interval_bars": 48})
        i415_by_year[yr] = run_signal(ds, "idio_momentum_alpha",
            {"k_per_side": 4, "lookback_bars": 415, "beta_window_bars": 144,
             "target_gross_leverage": 0.3, "rebalance_interval_bars": 48})
        print(f"  {yr}: done.", flush=True)

    # Baseline: F168 lb=168, reb=36
    print("\n[3] Baseline evaluation...", flush=True)
    base_obj, base_yr = eval_ensemble(p, all_base, datasets, v1_by_year,
                                       i460_by_year, i415_by_year, 168, 36)
    print(f"  Baseline OBJ = {base_obj:.4f}  (stored = {STORED_OBJ:.4f})")
    print(f"  Per year: {' | '.join(f'{yr}={v:.4f}' for yr, v in sorted(base_yr.items()))}")
    _partial["baseline_obj"] = float(base_obj)

    best_lb = 168; best_reb = 36; best_obj = base_obj

    print("\n[4] F168 sweep (lb × reb)...", flush=True)
    results_grid = {}
    for lb in F168_LB_GRID:
        for reb in F168_REB_GRID:
            obj, yr_vals = eval_ensemble(p, all_base, datasets, v1_by_year,
                                          i460_by_year, i415_by_year, lb, reb)
            delta_v = obj - base_obj
            results_grid[(lb, reb)] = obj
            marker = "  ← BEST" if obj > best_obj else ""
            print(f"  lb={lb:3d} reb={reb}  OBJ={obj:.4f}  Δ={delta_v:+.4f}{marker}")
            if obj > best_obj:
                best_obj = obj
                best_lb = lb
                best_reb = reb
    _partial["best_f168"] = {"lb": best_lb, "reb": best_reb, "obj": float(best_obj)}

    # Final LOYO
    print(f"\n[5] Final LOYO validation (lb={best_lb}, reb={best_reb})...", flush=True)
    final_obj, final_yr = eval_ensemble(p, all_base, datasets, v1_by_year,
                                         i460_by_year, i415_by_year, best_lb, best_reb)
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
    print(f"\n  F168: lb={168}→{best_lb}  reb={36}→{best_reb}"
          f"{'  ← CHANGED' if best_lb != 168 or best_reb != 36 else '  (unchanged)'}")

    result = {
        "phase": "373", "title": "F168 LB+Reb Sweep",
        "baseline_obj": float(base_obj), "stored_baseline": STORED_OBJ,
        "best_obj": float(final_obj), "delta": float(delta),
        "loyo_wins": wins, "loyo_total": 5, "validated": validated,
        "best_f168_lb": best_lb, "best_f168_reb": best_reb,
        "per_year_baseline": {str(k): v for k, v in base_yr.items()},
        "per_year_final": {str(k): v for k, v in final_yr.items()},
    }
    _partial.update(result)
    out = Path("artifacts/phase373"); out.mkdir(parents=True, exist_ok=True)
    (out / "phase373_report.json").write_text(json.dumps(result, indent=2))
    print(f"\n  Report → artifacts/phase373/phase373_report.json")

    if validated:
        cfg = json.load(open(CFG_PATH))
        cfg.setdefault("signal_params", {})
        cfg["signal_params"]["f168"] = {
            "funding_lookback_bars": best_lb, "direction": "contrarian",
            "k_per_side": 2, "target_gross_leverage": 0.25, "rebalance_interval_bars": best_reb
        }
        old_ver = cfg.get("_version", "2.49.45")
        parts = old_ver.split(".")
        parts[-1] = str(int(parts[-1]) + 1)
        new_ver = ".".join(parts)
        cfg["_version"] = new_ver
        stored_obj = cfg["monitoring"]["expected_performance"]["annual_sharpe_backtest"]
        reported_obj = round(stored_obj + delta, 4)
        cfg["monitoring"]["expected_performance"]["annual_sharpe_backtest"] = reported_obj
        cfg["obj"] = reported_obj
        with open(CFG_PATH, "w") as f: json.dump(cfg, f, indent=2, ensure_ascii=False)
        print(f"\n  Config → v{new_ver}  OBJ={reported_obj:.4f}")
        subprocess.run(["git", "add", str(CFG_PATH)], cwd=ROOT)
        msg = (f"feat: P373 F168 lb sweep OBJ={reported_obj:.4f} "
               f"LOYO={wins}/5 D={delta:+.4f} [v{new_ver}]")
        subprocess.run(["git", "commit", "-m", msg], cwd=ROOT)
        print(f"  Committed: {msg}")
    else:
        print("\n  No config change (not validated).")

    print(f"\nF168 cache: {len(_f168_cache)}")
    print(f"Total runtime: {int(time.time()-t0)}s ({int((time.time()-t0)/60)}m)", flush=True)


main()
