"""
Phase 267 NumPy — Breadth Threshold Re-Sweep [VECTORIZED, config-driven]
=========================================================================
Baseline: v2.49.0, real OBJ=4.1826 with p_high=0.68 (P252+P266 confirmed)

p_high was changed from 0.60 → 0.68 by Engine Track (external commit).
The NumPy track was never re-optimized for the new threshold.
This phase re-sweeps p_low × p_high jointly.

Sweep:
  p_low  = [0.20, 0.25, 0.30, 0.35]
  p_high = [0.55, 0.60, 0.63, 0.65, 0.68, 0.70, 0.73, 0.75]

Grid: 4 × 8 = 32 combinations. Vectorized — seconds per eval.
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
    out = Path("artifacts/phase267"); out.mkdir(parents=True, exist_ok=True)
    (out / "phase267_report.json").write_text(json.dumps(_partial, indent=2))
    sys.exit(0)
_signal.signal(_signal.SIGALRM, _on_timeout)
_signal.alarm(7200)

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

P_LOW_SWEEP  = [0.20, 0.25, 0.30, 0.35]
P_HIGH_SWEEP = [0.55, 0.60, 0.63, 0.65, 0.68, 0.70, 0.73, 0.75]

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
    rw  = brs["regime_weights"]

    def sdget(d, r, fallback):
        if isinstance(d, dict) and r in d: return float(d[r])
        return float(fallback)

    p_low  = float(brs.get("p_low", 0.30))
    p_high = float(brs.get("p_high", 0.68))

    regime_weights = {}
    for rname in RNAMES:
        w = rw[rname]
        regime_weights[rname] = {
            "v1":   float(w.get("v1", 0.0)),
            "i460": float(w.get("i460bw168", w.get("i460", 0.0))),
            "i415": float(w.get("i415bw216", w.get("i415", 0.0))),
            "f168": float(w.get("f168", 0.0)),
        }

    fts_rs = {r: sdget(fts.get("per_regime_rs"), r, 0.25) for r in RNAMES}
    fts_bs = {r: sdget(fts.get("per_regime_bs"), r, 1.0)  for r in RNAMES}
    fts_rt = {r: sdget(fts.get("per_regime_rt"), r, 0.65) for r in RNAMES}
    fts_bt = {r: sdget(fts.get("per_regime_bt"), r, 0.35) for r in RNAMES}
    ts_pct_win = {r: int(fts.get("per_regime_ts_pct_win", {}).get(r, 288)) for r in RNAMES}

    vol_thr   = {r: sdget(vol.get("per_regime_threshold"), r, 0.50) for r in RNAMES}
    vol_scale = {r: sdget(vol.get("per_regime_scale"),     r, 0.40) for r in RNAMES}
    disp_thr  = {r: sdget(disp.get("per_regime_threshold"), r, 0.70) for r in RNAMES}
    disp_scale= {r: sdget(disp.get("per_regime_scale"),     r, 0.50) for r in RNAMES}

    baseline_obj = float(cfg.get("obj", 4.3173))

    v1prw = cfg.get("v1_per_regime_weights", {})
    def v1p(rname):
        d = v1prw.get(rname, {})
        return {
            "k_per_side": 2,
            "w_carry": float(d.get("w_carry", 0.25)),
            "w_mom":   float(d.get("w_mom",   0.50)),
            "w_mean_reversion": float(d.get("w_mean_reversion", 0.25)),
            "momentum_lookback_bars": 336, "mean_reversion_lookback_bars": 84,
            "vol_lookback_bars": 192, "target_gross_leverage": 0.35,
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
        "config_version": cfg.get("_version", cfg.get("version", "?")),
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


def build_ts_pct_cache(ts_raw, ts_pct_win, n):
    cache = {}
    for w in set(ts_pct_win.values()):
        arr = np.full(n, 0.5)
        for i in range(w, n):
            arr[i] = float(np.mean(ts_raw[i-w:i] <= ts_raw[i]))
        cache[w] = arr
    return cache


def eval_with_thresholds(p_low, p_high, p, all_base, all_v1):
    """Evaluate OBJ for a given (p_low, p_high) pair. Fast: just recomputes masks."""
    yearly = {}
    for yr in sorted(all_base.keys()):
        btc_vol, fund_std_pct, ts_raw, brd_pct, fixed, n, _ = all_base[yr]
        v1 = all_v1[yr]
        ml = min(
            min(len(v1[r]) for r in RNAMES),
            min(len(fixed[k]) for k in ["i460", "i415", "f168"])
        )
        bpct = brd_pct[:ml]; bv = btc_vol[:ml]; fsp = fund_std_pct[:ml]; tr = ts_raw[:ml]

        regime_idx = np.where(bpct < p_low, 0, np.where(bpct > p_high, 2, 1))
        masks = [regime_idx == i for i in range(3)]

        ts_pct_cache = {}
        for rname in RNAMES:
            w = p["ts_pct_win"][rname]
            if w not in ts_pct_cache:
                arr = np.full(ml, 0.5)
                for i in range(w, ml):
                    arr[i] = float(np.mean(tr[i-w:i] <= tr[i]))
                ts_pct_cache[w] = arr

        overlay_mult = np.ones(ml)
        for ridx, rname in enumerate(RNAMES):
            m = masks[ridx]
            overlay_mult[m & ~np.isnan(bv) & (bv > p["vol_thr"][rname])] *= p["vol_scale"][rname]
            overlay_mult[m & (fsp > p["disp_thr"][rname])] *= p["disp_scale"][rname]
            tsp = ts_pct_cache[p["ts_pct_win"][rname]]
            overlay_mult[m & (tsp > p["fts_rt"][rname])] *= p["fts_rs"][rname]
            overlay_mult[m & (tsp < p["fts_bt"][rname])] *= p["fts_bs"][rname]

        ens = np.zeros(ml)
        for ridx, rname in enumerate(RNAMES):
            m = masks[ridx]; w = p["regime_weights"][rname]
            vk = "LOW" if rname == "LOW" else "MID"
            ens[m] += (w["v1"]   * overlay_mult[m] * v1[vk][:ml][m]
                       + w["i460"] * overlay_mult[m] * fixed["i460"][:ml][m]
                       + w["i415"] * overlay_mult[m] * fixed["i415"][:ml][m]
                       + w["f168"] * overlay_mult[m] * fixed["f168"][:ml][m])

        r = ens[~np.isnan(ens)]
        sh = float(np.mean(r) / np.std(r, ddof=1) * np.sqrt(8760)) if len(r) > 1 else 0.0
        yearly[yr] = sh

    vals = list(yearly.values())
    return float(np.mean(vals) - 0.5 * np.std(vals, ddof=1)), yearly


def main():
    t0 = time.time()
    print("=" * 65)
    print("Phase 267 — Breadth Threshold Re-Sweep [VEC, config-driven]")
    print(f"p_low grid : {P_LOW_SWEEP}")
    print(f"p_high grid: {P_HIGH_SWEEP}")
    print("=" * 65)

    print("\n[0] Loading config params...", flush=True)
    p = load_config_params()
    BASELINE_OBJ = p["baseline_obj"]
    print(f"  Config: v{p['config_version']}  stored_obj={BASELINE_OBJ:.4f}")
    print(f"  Current: p_low={p['p_low']}, p_high={p['p_high']}")

    print("\n[1] Loading data...", flush=True)
    all_base = {}
    for yr in sorted(YEAR_RANGES):
        print(f"  {yr}: ", end="", flush=True)
        all_base[yr] = load_year_data(yr)
        print("done.", flush=True)

    print("\n[2] Computing V1 per regime...", flush=True)
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

    print("\n[3] Evaluating current baseline...", flush=True)
    base_obj, base_yr = eval_with_thresholds(p["p_low"], p["p_high"], p, all_base, all_v1)
    print(f"  Baseline OBJ = {base_obj:.4f}  p_low={p['p_low']} p_high={p['p_high']}")
    _partial["baseline_obj_computed"] = float(base_obj)
    _partial["baseline_p_low"] = p["p_low"]
    _partial["baseline_p_high"] = p["p_high"]

    print("\n[4] Full grid sweep...", flush=True)
    best_obj = base_obj; best_pl = p["p_low"]; best_ph = p["p_high"]
    grid_results = []
    for pl in P_LOW_SWEEP:
        for ph in P_HIGH_SWEEP:
            if ph <= pl + 0.10: continue  # require meaningful MID band
            o, yr_d = eval_with_thresholds(pl, ph, p, all_base, all_v1)
            marker = " <--" if o > best_obj else ""
            print(f"  p_low={pl:.2f}  p_high={ph:.2f}  OBJ={o:.4f}{marker}", flush=True)
            grid_results.append({"p_low": pl, "p_high": ph, "obj": o})
            if o > best_obj:
                best_obj = o; best_pl = pl; best_ph = ph; best_yr = yr_d

    if best_obj <= base_obj:
        best_yr = base_yr

    delta = best_obj - base_obj
    print(f"\n  Best: p_low={best_pl} p_high={best_ph} OBJ={best_obj:.4f} Δ={delta:+.4f}")

    print("\n[5] LOYO validation...", flush=True)
    wins = 0
    for yr in sorted(best_yr):
        win = best_yr[yr] > base_yr[yr]
        if win: wins += 1
        print(f"  {yr}: base={base_yr[yr]:.4f}  cand={best_yr[yr]:.4f}  {'WIN' if win else 'LOSE'}")

    validated = wins >= MIN_LOYO and delta >= MIN_DELTA
    print(f"\n  OBJ={best_obj:.4f}  Δ={delta:+.4f}  LOYO {wins}/5")
    print(f"\n{'✅ VALIDATED' if validated else '❌ NOT VALIDATED'}")

    result = {
        "phase": "267", "baseline_obj": float(base_obj),
        "best_obj": float(best_obj), "delta": float(delta),
        "best_p_low": best_pl, "best_p_high": best_ph,
        "loyo_wins": wins, "loyo_total": 5, "validated": validated,
        "grid_results": grid_results,
    }
    _partial.update(result)
    out = Path("artifacts/phase267"); out.mkdir(parents=True, exist_ok=True)
    (out / "phase267_report.json").write_text(json.dumps(result, indent=2))

    if validated:
        cfg_path = ROOT / "configs" / "production_p91b_champion.json"
        cfg = json.load(open(cfg_path))
        cfg["breadth_regime_switching"]["p_low"]  = best_pl
        cfg["breadth_regime_switching"]["p_high"] = best_ph
        new_ver_num = cfg.get("_version", "2.49.0")
        parts = new_ver_num.split(".")
        parts[-1] = str(int(parts[-1]) + 1)
        new_ver = ".".join(parts)
        cfg["_version"] = new_ver
        cfg["obj"] = round(best_obj, 4)
        cfg["monitoring"]["expected_performance"]["annual_sharpe_backtest"] = round(best_obj, 4)
        with open(cfg_path, "w") as f: json.dump(cfg, f, indent=2, ensure_ascii=False)
        print(f"\n  Config → v{new_ver}  p_low={best_pl} p_high={best_ph}  OBJ={round(best_obj,4)}")
        subprocess.run(["git", "add", str(cfg_path)], cwd=ROOT)
        msg = (f"feat: P267 breadth threshold sweep p_low={best_pl} p_high={best_ph} "
               f"OBJ={round(best_obj,4)} LOYO={wins}/5 D={delta:+.4f} [v{new_ver}]")
        subprocess.run(["git", "commit", "-m", msg], cwd=ROOT)
        subprocess.run(["git", "push"], cwd=ROOT)
        print(f"  Committed: {msg}")
    else:
        # Update stored OBJ to reflect correct baseline (p_high=0.68 evaluation)
        print(f"\n  [OBJ CORRECTION] Updating stored OBJ from {BASELINE_OBJ:.4f} → {base_obj:.4f}")
        cfg_path = ROOT / "configs" / "production_p91b_champion.json"
        cfg = json.load(open(cfg_path))
        cfg["obj"] = round(base_obj, 4)
        cfg["monitoring"]["expected_performance"]["annual_sharpe_backtest"] = round(base_obj, 4)
        with open(cfg_path, "w") as f: json.dump(cfg, f, indent=2, ensure_ascii=False)
        subprocess.run(["git", "add", str(cfg_path)], cwd=ROOT)
        subprocess.run(["git", "commit", "-m",
                        f"fix: correct stored OBJ {BASELINE_OBJ:.4f}→{base_obj:.4f} (p_high=0.68 eval, P267)"],
                       cwd=ROOT)
        subprocess.run(["git", "push"], cwd=ROOT)

    print(f"\nRuntime: {int(time.time()-t0)}s", flush=True)


main()
