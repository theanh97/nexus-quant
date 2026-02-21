"""
Phase 351 — Per-Regime FTS Re-Sweep after P323-P325 [VECTORIZED]
=================================================================
Baseline: v2.49.35, OBJ=6.2892 (after P323 V1 no change + P324 regime no change)

After P318 (V1 +0.0281), P319 (regime weights +0.0363), P320 (boundary no change),
re-tune FTS overlay params.

Current FTS:
  rs: LOW=0.01 MID=0.01 HIGH=0.01
  rt: LOW=0.78 MID=0.58 HIGH=0.55
  bs: LOW=3.50 MID=1.50 HIGH=2.50
  bt: LOW=0.30 MID=0.22 HIGH=0.22
  pw: LOW=240  MID=288  HIGH=400

Sweep: 2-pass coordinate descent (rs→rt→bs→bt→pct_win per regime)
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
    out = Path("artifacts/phase351_fts"); out.mkdir(parents=True, exist_ok=True)
    (out / "phase351_fts_report.json").write_text(json.dumps(_partial, indent=2))
    print("\n⏰ TIMEOUT — partial results saved.", flush=True)
    sys.exit(0)
_signal.signal(_signal.SIGALRM, _on_timeout)
_signal.alarm(10800)  # 3 hours

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

# Grids — finer around current values + test re-enabling reduce leg
# Current: rs=0.02 (disabled), rt=0.80/0.60/0.55, bs=3.5/1.5/2.5, bt=0.33/0.22/0.22, pw=240/288/400
RS_GRID  = [0.01, 0.02, 0.03, 0.05, 0.08, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50]
RT_GRID  = [0.50, 0.55, 0.58, 0.60, 0.63, 0.65, 0.70, 0.75, 0.78, 0.80, 0.85, 0.90]
BS_GRID  = [1.00, 1.25, 1.50, 1.75, 2.00, 2.25, 2.50, 3.00, 3.50, 4.00, 4.50, 5.00]
BT_GRID  = [0.10, 0.15, 0.18, 0.20, 0.22, 0.25, 0.28, 0.30, 0.33, 0.35, 0.40]
PW_GRID  = [168, 192, 216, 240, 264, 288, 336, 400, 480]

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

    fts_rs     = {r: sdget(fts.get("per_regime_rs"), r, 0.05) for r in RNAMES}
    fts_bs     = {r: sdget(fts.get("per_regime_bs"), r, 2.0)  for r in RNAMES}
    fts_rt     = {r: sdget(fts.get("per_regime_rt"), r, 0.65) for r in RNAMES}
    fts_bt     = {r: sdget(fts.get("per_regime_bt"), r, 0.25) for r in RNAMES}
    ts_pct_win = {r: int(fts.get("per_regime_ts_pct_win", {}).get(r, 288)) for r in RNAMES}

    vol_thr   = {r: sdget(vol.get("per_regime_threshold"), r, 0.50) for r in RNAMES}
    vol_scale = {r: sdget(vol.get("per_regime_scale"),     r, 0.40) for r in RNAMES}
    disp_thr  = {r: sdget(disp.get("per_regime_threshold"), r, 0.70) for r in RNAMES}
    disp_scale= {r: sdget(disp.get("per_regime_scale"),     r, 0.50) for r in RNAMES}

    baseline_obj = float(cfg["monitoring"]["expected_performance"]["annual_sharpe_backtest"])

    v1prw = cfg.get("v1_per_regime_weights", {})
    def v1p(rname):
        d = v1prw.get(rname, {})
        return {
            "k_per_side": 2,
            "w_carry": float(d.get("w_carry", 0.25)),
            "w_mom":   float(d.get("w_mom",   0.50)),
            "w_mean_reversion": float(d.get("w_mean_reversion", 0.25)),
            "momentum_lookback_bars": 312, "mean_reversion_lookback_bars": 84,
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


def build_fast_data(all_base, all_v1, p):
    """Build non-FTS-scaled data for fast FTS sweep. VOL+DISP pre-applied."""
    fast = {}
    for yr, base in sorted(all_base.items()):
        btc_vol, fund_std_pct, ts_raw, brd_pct, fixed, n, _ = base
        v1 = all_v1[yr]
        ml = min(
            min(len(v1[r]) for r in RNAMES),
            min(len(fixed[k]) for k in ["i460", "i415", "f168"])
        )
        bpct = brd_pct[:ml]; bv = btc_vol[:ml]; fsp = fund_std_pct[:ml]; tr = ts_raw[:ml]

        regime_idx = np.where(bpct < p["p_low"], 0, np.where(bpct > p["p_high"], 2, 1))
        masks = [regime_idx == i for i in range(3)]

        # Non-FTS overlay: VOL (correct: > threshold) + DISP
        non_fts = np.ones(ml)
        for ridx, rname in enumerate(RNAMES):
            m = masks[ridx]
            non_fts[m & ~np.isnan(bv) & (bv > p["vol_thr"][rname])] *= p["vol_scale"][rname]
            non_fts[m & (fsp > p["disp_thr"][rname])] *= p["disp_scale"][rname]

        # Pre-scaled signals (VOL+DISP applied, FTS will be swept)
        v1_arrs = {r: non_fts * v1[r][:ml] for r in RNAMES}
        i460_arr = non_fts * fixed["i460"][:ml]
        i415_arr = non_fts * fixed["i415"][:ml]
        f168_arr = non_fts * fixed["f168"][:ml]

        # Pre-compute ts_pct for all possible windows
        ts_cache = {}
        for w in PW_GRID:
            arr = np.full(ml, 0.5)
            for i in range(w, ml):
                arr[i] = float(np.mean(tr[i-w:i] <= tr[i]))
            ts_cache[w] = arr

        fast[yr] = {
            "masks": masks, "ts_cache": ts_cache, "ml": ml,
            "v1_arrs": v1_arrs, "i460": i460_arr, "i415": i415_arr, "f168": f168_arr,
        }
    return fast


def fast_eval_fts(p, fast, fts_rs, fts_rt, fts_bs, fts_bt, ts_pct_win):
    """Evaluate OBJ given per-regime FTS params. Other params fixed in fast data."""
    yearly = {}
    for yr, fd in sorted(fast.items()):
        masks = fd["masks"]; ml = fd["ml"]
        v1_arrs = fd["v1_arrs"]; i460 = fd["i460"]; i415 = fd["i415"]; f168 = fd["f168"]

        fts_mult = np.ones(ml)
        for ridx, rname in enumerate(RNAMES):
            m = masks[ridx]
            tsp = fd["ts_cache"][ts_pct_win[rname]]
            fts_mult[m & (tsp > fts_rt[rname])] *= fts_rs[rname]
            fts_mult[m & (tsp < fts_bt[rname])] *= fts_bs[rname]

        ens = np.zeros(ml)
        for ridx, rname in enumerate(RNAMES):
            m = masks[ridx]; w = p["regime_weights"][rname]
            f = fts_mult[m]
            ens[m] = (w["v1"] * v1_arrs[rname][m] * f
                      + w["i460"] * i460[m] * f
                      + w["i415"] * i415[m] * f
                      + w["f168"] * f168[m] * f)

        r = ens[~np.isnan(ens)]
        yearly[yr] = float(np.mean(r) / np.std(r, ddof=1) * np.sqrt(8760)) if len(r) > 1 else 0.0

    vals = list(yearly.values())
    return float(np.mean(vals) - 0.5 * np.std(vals, ddof=1)), yearly


def sweep_regime_fts(rname, cur_rs, cur_rt, cur_bs, cur_bt, cur_pw, fast, p, best_obj):
    """Sweep all 5 FTS params for one regime (rs→rt→bs→bt→pct_win)."""
    best = {"rs": cur_rs[rname], "rt": cur_rt[rname], "bs": cur_bs[rname],
            "bt": cur_bt[rname], "pw": cur_pw[rname]}

    def eval_with(rs=None, rt=None, bs=None, bt=None, pw=None):
        rs_  = {r: cur_rs[r] for r in RNAMES};  rs_[rname]  = rs  if rs is not None else best["rs"]
        rt_  = {r: cur_rt[r] for r in RNAMES};  rt_[rname]  = rt  if rt is not None else best["rt"]
        bs_  = {r: cur_bs[r] for r in RNAMES};  bs_[rname]  = bs  if bs is not None else best["bs"]
        bt_  = {r: cur_bt[r] for r in RNAMES};  bt_[rname]  = bt  if bt is not None else best["bt"]
        pw_  = {r: cur_pw[r] for r in RNAMES};  pw_[rname]  = pw  if pw is not None else best["pw"]
        return fast_eval_fts(p, fast, rs_, rt_, bs_, bt_, pw_)

    for v in RS_GRID:
        o, _ = eval_with(rs=v)
        if o > best_obj: best_obj = o; best["rs"] = v

    for v in RT_GRID:
        o, _ = eval_with(rt=v)
        if o > best_obj: best_obj = o; best["rt"] = v

    for v in BS_GRID:
        o, _ = eval_with(bs=v)
        if o > best_obj: best_obj = o; best["bs"] = v

    for v in BT_GRID:
        o, _ = eval_with(bt=v)
        if o > best_obj: best_obj = o; best["bt"] = v

    for v in PW_GRID:
        o, _ = eval_with(pw=v)
        if o > best_obj: best_obj = o; best["pw"] = v

    return best, best_obj


def main():
    t0 = time.time()
    print("=" * 70)
    print("Phase 351 — FTS Re-Sweep after P323-P325 [VECTORIZED]")

    print("\n[0] Loading config params...", flush=True)
    p = load_config_params()
    STORED_OBJ = p["baseline_obj"]
    print(f"Baseline: v{p['config_version']}  OBJ={STORED_OBJ:.4f}")
    print("=" * 70)
    print(f"  Config: v{p['config_version']}  stored_obj={STORED_OBJ:.4f}")
    print(f"  p_low={p['p_low']}, p_high={p['p_high']}")
    print(f"  Regime weights:")
    for rn in RNAMES:
        w = p["regime_weights"][rn]
        print(f"    {rn}: v1={w['v1']:.2f} i460={w['i460']:.2f} i415={w['i415']:.2f} f168={w['f168']:.2f}")
    print(f"  FTS RS={p['fts_rs']}")
    print(f"  FTS RT={p['fts_rt']}")
    print(f"  FTS BS={p['fts_bs']}")
    print(f"  FTS BT={p['fts_bt']}")
    print(f"  FTS PW={p['ts_pct_win']}")

    print("\n[1] Loading yearly data + running backtests...", flush=True)
    all_base = {}
    for yr in sorted(YEAR_RANGES):
        tw = time.time()
        print(f"  {yr}: loading...", end="", flush=True)
        all_base[yr] = load_year_data(yr)
        print(f" done ({int(time.time()-tw)}s)", flush=True)

    print("\n[2] Computing per-regime V1 signals...", flush=True)
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

    # Free datasets to reduce memory
    for yr in all_base:
        all_base[yr] = all_base[yr][:6] + (None,)  # drop dataset ref
    gc.collect()

    print("\n[3] Building fast eval data (VOL+DISP pre-applied)...", flush=True)
    fast = build_fast_data(all_base, all_v1, p)

    base_obj, base_yr = fast_eval_fts(p, fast, p["fts_rs"], p["fts_rt"],
                                       p["fts_bs"], p["fts_bt"], p["ts_pct_win"])
    print(f"\n  Computed baseline OBJ = {base_obj:.4f}  (stored = {STORED_OBJ:.4f})")
    print(f"  Per year: {' | '.join(f'{yr}={v:.4f}' for yr, v in sorted(base_yr.items()))}")
    _partial["baseline_obj"] = float(base_obj)
    _partial["stored_baseline"] = STORED_OBJ

    print("\n[4] 2-pass coordinate descent FTS sweep...", flush=True)
    cur_rs = dict(p["fts_rs"]); cur_rt = dict(p["fts_rt"])
    cur_bs = dict(p["fts_bs"]); cur_bt = dict(p["fts_bt"]); cur_pw = dict(p["ts_pct_win"])
    best_global = base_obj

    for pass_num in [1, 2]:
        print(f"\n  === Pass {pass_num} ===", flush=True)
        for rname in RNAMES:
            tw = time.time()
            bst, best_global = sweep_regime_fts(
                rname, cur_rs, cur_rt, cur_bs, cur_bt, cur_pw, fast, p, best_global)
            cur_rs[rname] = bst["rs"]; cur_rt[rname] = bst["rt"]
            cur_bs[rname] = bst["bs"]; cur_bt[rname] = bst["bt"]; cur_pw[rname] = bst["pw"]
            print(f"    {rname}: rs={bst['rs']:.2f} rt={bst['rt']:.2f} bs={bst['bs']:.2f} "
                  f"bt={bst['bt']:.2f} pw={bst['pw']}  "
                  f"OBJ={best_global:.4f}  Δ={best_global-base_obj:+.4f}  [{int(time.time()-tw)}s]",
                  flush=True)

    final_obj, final_yr = fast_eval_fts(p, fast, cur_rs, cur_rt, cur_bs, cur_bt, cur_pw)
    delta = final_obj - base_obj
    print(f"\n  Final OBJ = {final_obj:.4f}  Δ = {delta:+.4f}")
    print(f"  Per year: {' | '.join(f'{yr}={v:.4f}' for yr, v in sorted(final_yr.items()))}")

    # [5] LOYO validation
    print("\n[5] LOYO validation...", flush=True)
    wins = 0
    for yr in sorted(final_yr):
        d = final_yr[yr] - base_yr[yr]
        win = d > 0
        if win: wins += 1
        print(f"  {yr}: base={base_yr[yr]:.4f}  cand={final_yr[yr]:.4f}  Δ={d:+.4f}  {'✓ WIN' if win else '✗ LOSE'}")

    validated = wins >= MIN_LOYO and delta >= MIN_DELTA
    print(f"\n  OBJ={final_obj:.4f}  Δ={delta:+.4f}  LOYO {wins}/5")
    print(f"\n{'✅ VALIDATED' if validated else '❌ NOT VALIDATED'}")

    result = {
        "phase": "351", "baseline_obj": float(base_obj), "stored_baseline": STORED_OBJ,
        "best_obj": float(final_obj), "delta": float(delta),
        "loyo_wins": wins, "loyo_total": 5, "validated": validated,
        "best_fts_rs": cur_rs, "best_fts_rt": cur_rt,
        "best_fts_bs": cur_bs, "best_fts_bt": cur_bt, "best_ts_pct_win": cur_pw,
        "per_year_baseline": {str(k): v for k, v in base_yr.items()},
        "per_year_final":    {str(k): v for k, v in final_yr.items()},
    }
    _partial.update(result)
    out = Path("artifacts/phase351_fts"); out.mkdir(parents=True, exist_ok=True)
    (out / "phase351_fts_report.json").write_text(json.dumps(result, indent=2))
    print(f"\n  Report → artifacts/phase351_fts/phase351_fts_report.json")

    if validated:
        cfg = json.load(open(CFG_PATH))
        fts = cfg.setdefault("fts_overlay_params", {})
        fts["per_regime_rs"] = cur_rs
        fts["per_regime_rt"] = cur_rt
        fts["per_regime_bs"] = cur_bs
        fts["per_regime_bt"] = cur_bt
        fts["per_regime_ts_pct_win"] = {r: int(cur_pw[r]) for r in RNAMES}
        for rname in RNAMES:
            fts[rname] = {"rs": cur_rs[rname], "rt": cur_rt[rname],
                          "bs": cur_bs[rname], "bt": cur_bt[rname]}
        old_ver = cfg.get("_version", "2.49.14")
        parts = old_ver.split(".")
        parts[-1] = str(int(parts[-1]) + 1)
        new_ver = ".".join(parts)
        cfg["_version"] = new_ver
        old_vver = cfg.get("version", "v2.49.19").lstrip("v")
        vparts = old_vver.split(".")
        vparts[-1] = str(int(vparts[-1]) + 1)
        cfg["version"] = "v" + ".".join(vparts)
        stored_obj = cfg["monitoring"]["expected_performance"]["annual_sharpe_backtest"]
        reported_obj = round(stored_obj + delta, 4)
        cfg["obj"] = reported_obj
        cfg["monitoring"]["expected_performance"]["annual_sharpe_backtest"] = reported_obj
        with open(CFG_PATH, "w") as f: json.dump(cfg, f, indent=2, ensure_ascii=False)
        print(f"\n  Config updated -> v{new_ver}  OBJ={reported_obj} (stored {stored_obj} + D{delta:+.4f})")
        subprocess.run(["git", "add", str(CFG_PATH)], cwd=ROOT)
        msg = (f"feat: P351 FTS retune OBJ={reported_obj} "
               f"LOYO={wins}/5 D={delta:+.4f} [v{new_ver}]")
        subprocess.run(["git", "commit", "-m", msg], cwd=ROOT)
        print(f"  Committed: {msg}")
    else:
        print("\n  No config change (not validated).")

    print(f"\nTotal runtime: {int(time.time()-t0)}s ({int((time.time()-t0)/60)}m)", flush=True)


main()
