"""
Phase 253 — Per-Regime DISP Overlay Re-Sweep (v2.48.x baseline)
================================================================
Loads params dynamically from current config.
DISP (funding rate cross-sectional dispersion) overlay may have shifted
after VOL scale changes. Current: LOW(0.70/0.50), MID(0.70/1.50), HIGH(0.40/0.50).

Sweep: per-regime THR and SCALE independently.
THR_SWEEP  = [0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]
SCALE_SWEEP = [0.0, 0.10, 0.20, 0.30, 0.50, 0.70, 1.00, 1.20, 1.50, 1.80, 2.00, 2.50]

All numpy — no new strategy runs.
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
    out = Path("artifacts/phase253"); out.mkdir(parents=True, exist_ok=True)
    (out / "phase253_report.json").write_text(json.dumps(_partial, indent=2))
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

DISP_THR_SWEEP   = [0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]
DISP_SCALE_SWEEP = [0.0, 0.10, 0.20, 0.30, 0.50, 0.70, 1.00, 1.20, 1.50, 1.80, 2.00, 2.50]

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

    disp_thr   = {r: sdget(disp.get("per_regime_threshold"), r, 0.70) for r in RNAMES}
    disp_scale = {r: sdget(disp.get("per_regime_scale"),     r, 0.50) for r in RNAMES}

    baseline_obj = float(cfg.get("monitoring", {}).get("expected_performance", {})
                         .get("annual_sharpe_backtest", 4.28))

    v1prw = cfg.get("v1_per_regime_weights", {})
    def v1p(rname):
        d = v1prw.get(rname, {})
        return {
            "k_per_side": 2,
            "w_carry": float(d.get("w_carry", 0.25)),
            "w_mom": float(d.get("w_mom", 0.50)),
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
        "config_version": cfg.get("_version", "?"),
    }

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

def build_fast_data(disp_thr, disp_scale, all_base_data, all_v1_raw, p):
    fast = {}
    for yr, base_data in sorted(all_base_data.items()):
        btc_vol, fund_std_pct, ts_raw, brd_pct, fixed_rets, n, _ = base_data
        v1 = all_v1_raw[yr]
        min_len = min(
            min(len(v1[r]) for r in RNAMES),
            min(len(fixed_rets[k]) for k in ["i460","i415","f168"])
        )
        bpct = brd_pct[:min_len]; bv = btc_vol[:min_len]
        fsp  = fund_std_pct[:min_len]; tr = ts_raw[:min_len]

        regime_idx = np.where(bpct < p["p_low"], 0, np.where(bpct > p["p_high"], 2, 1))
        masks = [regime_idx == i for i in range(3)]

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
            dm = m & (fsp > disp_thr[rname])
            overlay_mult[dm] *= disp_scale[rname]
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

def fast_evaluate(fast_data, p):
    yearly = {}
    for yr, (masks, min_len, scaled) in fast_data.items():
        ens = np.zeros(min_len)
        for ridx, rname in enumerate(RNAMES):
            m = masks[ridx]; w = p["regime_weights"][rname]
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
    p = load_config_params()

    print("=" * 65)
    print("Phase 253 — Per-Regime DISP Overlay Re-Sweep")
    print(f"Baseline: v{p['config_version']}  OBJ={p['baseline_obj']:.4f}")
    print(f"Current DISP thr={p['disp_thr']}, scale={p['disp_scale']}")
    print(f"THR grid: {DISP_THR_SWEEP}")
    print(f"SCALE grid: {DISP_SCALE_SWEEP}")
    print("=" * 65, flush=True)

    BASELINE_OBJ = p["baseline_obj"]

    print("\n[1] Loading per-year data & signals...", flush=True)
    all_base_data = {}
    for yr in sorted(YEAR_RANGES):
        print(f"  {yr}: ", end="", flush=True)
        all_base_data[yr] = load_year_data(yr)
        print("done.", flush=True)

    print("\n[2] Pre-computing V1 returns...", flush=True)
    all_v1_raw = {}
    for yr in sorted(YEAR_RANGES):
        ds = all_base_data[yr][6]
        v1_low = compute_v1(ds, p["v1_params"]["LOW"])
        if p["v1_params"]["MID"] == p["v1_params"]["HIGH"]:
            v1_mh = compute_v1(ds, p["v1_params"]["MID"])
            all_v1_raw[yr] = {"LOW": v1_low, "MID": v1_mh, "HIGH": v1_mh}
        else:
            v1_mid = compute_v1(ds, p["v1_params"]["MID"])
            v1_hi  = compute_v1(ds, p["v1_params"]["HIGH"])
            all_v1_raw[yr] = {"LOW": v1_low, "MID": v1_mid, "HIGH": v1_hi}
        print(f"  {yr}: done.", flush=True)

    # Baseline
    base_fast = build_fast_data(p["disp_thr"], p["disp_scale"], all_base_data, all_v1_raw, p)
    base_obj_val, base_yearly = fast_evaluate(base_fast, p)
    print(f"\n  Baseline OBJ = {base_obj_val:.4f}  (expected ≈ {BASELINE_OBJ:.4f})", flush=True)
    _partial["baseline_obj"] = float(base_obj_val)

    print("\n[3] Sequential per-regime DISP sweep (2 passes)...", flush=True)
    best_thr   = dict(p["disp_thr"])
    best_scale = dict(p["disp_scale"])
    running_best = base_obj_val

    for pass_num in [1, 2]:
        print(f"\n  === Pass {pass_num} ===", flush=True)
        for rname in RNAMES:
            best_t = best_thr[rname]; best_s = best_scale[rname]
            best_obj_this = running_best
            for thr in DISP_THR_SWEEP:
                for sc in DISP_SCALE_SWEEP:
                    dt = dict(best_thr);  dt[rname] = thr
                    ds_ = dict(best_scale); ds_[rname] = sc
                    fd = build_fast_data(dt, ds_, all_base_data, all_v1_raw, p)
                    o, _ = fast_evaluate(fd, p)
                    if o > best_obj_this:
                        best_obj_this = o; best_t = thr; best_s = sc
            best_thr[rname] = best_t; best_scale[rname] = best_s
            running_best = best_obj_this
            print(f"    {rname}: THR={best_t} SCALE={best_s}  OBJ={running_best:.4f}  Δ={running_best-base_obj_val:+.4f}", flush=True)

    final_fd = build_fast_data(best_thr, best_scale, all_base_data, all_v1_raw, p)
    final_obj_val, final_yearly = fast_evaluate(final_fd, p)
    delta = final_obj_val - base_obj_val
    print(f"\n  → Best DISP: thr={best_thr}, scale={best_scale}")
    print(f"  Final OBJ={final_obj_val:.4f}  Δ={delta:+.4f}", flush=True)

    print("\n[4] LOYO validation...", flush=True)
    loyo_wins = 0
    for yr in sorted(final_yearly):
        base_sh = base_yearly[yr]; cand_sh = final_yearly[yr]
        win = cand_sh > base_sh
        if win: loyo_wins += 1
        print(f"  {yr}: base={base_sh:.4f}  cand={cand_sh:.4f}  {'WIN' if win else 'LOSE'}")

    print(f"\n  OBJ={final_obj_val:.4f}  Δ={delta:+.4f}  LOYO {loyo_wins}/{len(final_yearly)}", flush=True)
    validated = loyo_wins >= MIN_LOYO and delta >= MIN_DELTA
    print(f"\n{'✅ VALIDATED' if validated else '❌ NO IMPROVEMENT'}", flush=True)

    result = {
        "phase": 253, "baseline_obj": float(base_obj_val),
        "best_obj": float(final_obj_val), "delta": float(delta),
        "loyo_wins": loyo_wins, "loyo_total": len(final_yearly),
        "validated": validated,
        "best_disp_thr": best_thr, "best_disp_scale": best_scale,
    }
    _partial.update(result)
    out = Path("artifacts/phase253"); out.mkdir(parents=True, exist_ok=True)
    (out / "phase253_report.json").write_text(json.dumps(result, indent=2))

    if validated:
        cfg_path = ROOT / "configs" / "production_p91b_champion.json"
        cfg = json.load(open(cfg_path))
        disp = cfg.get("disp_overlay_params", {})
        disp["per_regime_threshold"] = best_thr
        disp["per_regime_scale"]     = best_scale
        cfg["disp_overlay_params"] = disp
        cur_ver = cfg.get("_version", "2.48.0")
        parts = cur_ver.split(".")
        new_ver = f"{parts[0]}.{parts[1]}.{int(parts[2])+1}"
        cfg["_version"] = new_ver
        cfg["monitoring"]["expected_performance"]["annual_sharpe_backtest"] = round(final_obj_val, 4)
        with open(cfg_path, "w") as f:
            json.dump(cfg, f, indent=2, ensure_ascii=False)
        print(f"\n  Config → {new_ver}  OBJ={round(final_obj_val,4)}", flush=True)

    print(f"\nRuntime: {int(time.time()-t0)}s", flush=True)

main()
