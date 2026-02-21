"""
Phase 327 — Breadth Boundary Re-Tune (p_low / p_high)
=======================================================
Baseline: v2.49.36, OBJ=6.4480 (P326 FTS retune — no change)

Continue R&D cycle: re-sweep breadth regime boundaries.
Current: p_low=0.30, p_high=0.60 (confirmed 7x in P275, P302, P307, P312, P317, P322).
Finer grid around current values + wider exploration.
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
    out = Path("artifacts/phase327"); out.mkdir(parents=True, exist_ok=True)
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

# Finer grid around current p_low=0.30 p_high=0.60
P_LOW_SWEEP  = [0.22, 0.25, 0.27, 0.28, 0.29, 0.30, 0.31, 0.32, 0.33, 0.35, 0.38]
P_HIGH_SWEEP = [0.52, 0.55, 0.57, 0.58, 0.59, 0.60, 0.61, 0.62, 0.63, 0.65, 0.68]

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
            "momentum_lookback_bars": 312, "mean_reversion_lookback_bars": 84,
            "vol_lookback_bars": 192, "target_gross_leverage": 0.35,
            "rebalance_interval_bars": 60,
        }

    return {
        "p_low":  float(brs.get("p_low", 0.30)),
        "p_high": float(brs.get("p_high", 0.60)),
        "regime_weights": regime_weights,
        "fts_rs":  {r: sdget(fts.get("per_regime_rs"), r, 0.01)  for r in RNAMES},
        "fts_bs":  {r: sdget(fts.get("per_regime_bs"), r, 2.0)   for r in RNAMES},
        "fts_rt":  {r: sdget(fts.get("per_regime_rt"), r, 0.65)  for r in RNAMES},
        "fts_bt":  {r: sdget(fts.get("per_regime_bt"), r, 0.22)  for r in RNAMES},
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

    v1prw = p.get("v1_params", {})
    v1_by_regime = {}
    for rn in RNAMES:
        res = BacktestEngine(BacktestConfig(costs=COST_MODEL)).run(
            dataset, make_strategy({"name": "nexus_alpha_v1", "params": v1prw[rn]}))
        v1_by_regime[rn] = np.array(res.returns)

    return btc_vol, fund_std_pct, ts_raw, brd_pct, fixed_rets, v1_by_regime, n


def build_precomp_for_boundary(p, all_base_data, p_low, p_high):
    fast_data = {}
    for yr, yd in sorted(all_base_data.items()):
        btc_vol, fund_std_pct, ts_raw, brd_pct, fixed_rets, v1_by_regime, n = yd
        ml = min(
            min(len(v1_by_regime[r]) for r in RNAMES),
            min(len(fixed_rets[k]) for k in ["i460", "i415", "f168"])
        )
        bpct = brd_pct[:ml]; bv = btc_vol[:ml]
        fsp  = fund_std_pct[:ml]; tr = ts_raw[:ml]

        regime_idx = np.where(bpct < p_low, 0, np.where(bpct > p_high, 2, 1))
        masks = [regime_idx == i for i in range(3)]

        unique_wins = sorted(set(p["ts_pct_win"].values()))
        ts_pct_cache = {}
        for w in unique_wins:
            arr = np.full(ml, 0.5)
            for i in range(w, ml): arr[i] = float(np.mean(tr[i-w:i] <= tr[i]))
            ts_pct_cache[w] = arr

        overlay_mult = np.ones(ml)
        for ri, rn in enumerate(RNAMES):
            m = masks[ri]
            overlay_mult[m & ~np.isnan(bv) & (bv > p["vol_thr"][rn])] *= p["vol_scale"][rn]
            overlay_mult[m & (fsp > p["disp_thr"][rn])] *= p["disp_scale"][rn]
            tsp = ts_pct_cache[p["ts_pct_win"][rn]]
            overlay_mult[m & (tsp > p["fts_rt"][rn])] *= p["fts_rs"][rn]
            overlay_mult[m & (tsp < p["fts_bt"][rn])] *= p["fts_bs"][rn]

        scaled = {
            "v1": {rn: overlay_mult * v1_by_regime[rn][:ml] for rn in RNAMES},
            "i460": overlay_mult * fixed_rets["i460"][:ml],
            "i415": overlay_mult * fixed_rets["i415"][:ml],
            "f168": overlay_mult * fixed_rets["f168"][:ml],
        }
        fast_data[yr] = {"scaled": scaled, "masks": masks, "ml": ml}
    return fast_data


def fast_eval(rw, fast_data):
    yearly = []
    for yr, fd in sorted(fast_data.items()):
        sc = fd["scaled"]; masks = fd["masks"]; ml = fd["ml"]
        ens = np.zeros(ml)
        for ri, rn in enumerate(RNAMES):
            m = masks[ri]; w = rw[rn]
            ens[m] = (w["v1"]*sc["v1"][rn][m] + w["i460"]*sc["i460"][m]
                     + w["i415"]*sc["i415"][m] + w["f168"]*sc["f168"][m])
        r = ens[~np.isnan(ens)]
        sh = float(np.mean(r)/np.std(r,ddof=1)*np.sqrt(8760)) if len(r) > 1 else 0.0
        yearly.append(sh)
    return float(np.mean(yearly) - 0.5*np.std(yearly, ddof=1))


if __name__ == "__main__":
    t0 = time.time()
    print("=" * 65)
    print("Phase 327 — Regime Boundary Re-Sweep (p_low / p_high)")

    P = load_config_params()
    base_obj = P["baseline_obj"]
    print(f"Baseline: v{P['config_version']}  OBJ={base_obj:.4f}")
    print(f"Current p_low={P['p_low']:.2f}  p_high={P['p_high']:.2f}")
    print(f"Sweep p_low={P_LOW_SWEEP}  p_high={P_HIGH_SWEEP}")
    print("=" * 65)

    print("\n[1] Loading per-year data...", flush=True)
    all_base_data = {}
    for yr in sorted(YEAR_RANGES.keys()):
        print(f"  {yr}: ", end="", flush=True)
        all_base_data[yr] = load_year_data(yr, P)
        print(f"done.  ({time.time()-t0:.0f}s)")

    print("\n[2] Baseline eval...", flush=True)
    base_fd = build_precomp_for_boundary(P, all_base_data, P["p_low"], P["p_high"])
    obj_base = fast_eval(P["regime_weights"], base_fd)
    print(f"  Measured baseline OBJ = {obj_base:.4f}  (config says {base_obj:.4f})")

    print("\n[3] Boundary grid sweep...", flush=True)
    best_obj = obj_base
    best_plow = P["p_low"]; best_phigh = P["p_high"]
    results = []
    for pl in P_LOW_SWEEP:
        for ph in P_HIGH_SWEEP:
            if ph <= pl + 0.10: continue
            fd = build_precomp_for_boundary(P, all_base_data, pl, ph)
            obj = fast_eval(P["regime_weights"], fd)
            delta_local = obj - obj_base
            results.append((obj, pl, ph, delta_local))
            marker = " ← BEST" if obj > best_obj else ""
            if obj > best_obj:
                best_obj = obj; best_plow = pl; best_phigh = ph
            print(f"  p_low={pl:.2f} p_high={ph:.2f}  OBJ={obj:.4f}  Δ={delta_local:+.4f}{marker}")

    delta = best_obj - obj_base
    print(f"\n  Best: p_low={best_plow:.2f}  p_high={best_phigh:.2f}  OBJ={best_obj:.4f}  Δ={delta:+.4f}")

    if delta < MIN_DELTA:
        print(f"\n❌ NO IMPROVEMENT (Δ={delta:+.4f} < {MIN_DELTA})")
        print(f"  p_low={P['p_low']:.2f}  p_high={P['p_high']:.2f} confirmed optimal.")
        result = {
            "phase": "327", "baseline_obj": float(obj_base), "stored_baseline": base_obj,
            "best_obj": float(best_obj), "delta": float(delta), "validated": False,
            "best_p_low": best_plow, "best_p_high": best_phigh,
        }
        out = Path("artifacts/phase327"); out.mkdir(parents=True, exist_ok=True)
        (out / "report.json").write_text(json.dumps(result, indent=2))
        print(f"\nRuntime: {time.time()-t0:.1f}s")
        print("[DONE] Phase 327 complete.")
        sys.exit(0)

    print("\n[4] LOYO validation...", flush=True)
    best_fd = build_precomp_for_boundary(P, all_base_data, best_plow, best_phigh)
    loyo_wins = 0
    for yr in sorted(YEAR_RANGES.keys()):
        others = [y for y in YEAR_RANGES if y != yr]
        def _obj_subset(fd, yrs):
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
        base_loyo = _obj_subset(base_fd, others)
        cand_loyo = _obj_subset(best_fd, others)
        win = cand_loyo > base_loyo
        loyo_wins += int(win)
        print(f"  {yr}: base={base_loyo:.4f}  cand={cand_loyo:.4f}  {'WIN' if win else 'LOSE'}")

    print(f"\n  OBJ={best_obj:.4f}  Δ={delta:+.4f}  LOYO {loyo_wins}/5")
    validated = delta >= MIN_DELTA and loyo_wins >= MIN_LOYO

    result = {
        "phase": "327", "baseline_obj": float(obj_base), "stored_baseline": base_obj,
        "best_obj": float(best_obj), "delta": float(delta),
        "loyo_wins": loyo_wins, "loyo_total": 5, "validated": validated,
        "best_p_low": best_plow, "best_p_high": best_phigh,
    }
    out = Path("artifacts/phase327"); out.mkdir(parents=True, exist_ok=True)
    (out / "report.json").write_text(json.dumps(result, indent=2))

    if validated:
        print(f"\n✅ VALIDATED")
        cfg = json.load(open(CFG_PATH))
        brs = cfg["breadth_regime_switching"]
        brs["p_low"] = best_plow
        brs["p_high"] = best_phigh
        old_ver = cfg.get("_version", "2.49.35")
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
        print(f"  Config → v{new_ver}  p_low={best_plow:.2f}  p_high={best_phigh:.2f}  OBJ={reported_obj:.4f}")
        msg = (f"feat: P327 boundary resweep p_low={best_plow:.2f} p_high={best_phigh:.2f} "
               f"OBJ={reported_obj:.4f} LOYO={loyo_wins}/5 D={delta:+.4f} [v{new_ver}]")
        subprocess.run(["git", "add", str(CFG_PATH)], cwd=ROOT)
        subprocess.run(["git", "commit", "-m", msg], cwd=ROOT)
    else:
        print(f"\n❌ NOT VALIDATED (LOYO {loyo_wins}/5, Δ={delta:+.4f})")
        print(f"  Current boundary confirmed.")

    print(f"\nRuntime: {time.time()-t0:.1f}s")
    print("[DONE] Phase 327 complete.")
