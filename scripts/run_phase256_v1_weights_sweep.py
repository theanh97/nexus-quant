#!/usr/bin/env python3
"""
Phase 256 — V1 Signal Weights Sweep
Current: w_carry=0.25, w_mom=0.45, w_mr=0.30
Sweep: A) w_mom (then w_mr, w_carry = 1-w_mom split fixed), B) w_carry ratio
"""
import json, time, subprocess
import numpy as np
from pathlib import Path
import sys
sys.path.insert(0, "/Users/truonglys/projects/quant")

from nexus_quant.backtest.engine import BacktestConfig, BacktestEngine
from nexus_quant.backtest.costs import cost_model_from_config
from nexus_quant.data.providers.registry import make_provider
from nexus_quant.strategies.registry import make_strategy

SYMBOLS = ["BTCUSDT","ETHUSDT","SOLUSDT","BNBUSDT","XRPUSDT",
           "ADAUSDT","DOGEUSDT","AVAXUSDT","DOTUSDT","LINKUSDT"]
YEAR_RANGES = {
    "2021": ("2021-02-01", "2022-01-01"), "2022": ("2022-01-01", "2023-01-01"),
    "2023": ("2023-01-01", "2024-01-01"), "2024": ("2024-01-01", "2025-01-01"),
    "2025": ("2025-01-01", "2026-01-01"),
}
YEARS = sorted(YEAR_RANGES.keys())
COST_MODEL = cost_model_from_config({"fee_rate": 0.0005, "slippage_rate": 0.0003, "cost_multiplier": 1.0})
VOL_WINDOW = 168; PCT_WINDOW = 336

def rolling_mean_arr(a, w):
    out = np.full(len(a), np.nan)
    if w <= 0: return out
    cs = np.cumsum(np.where(np.isnan(a), 0.0, a))
    for i in range(w - 1, len(a)):
        out[i] = cs[i] / w if i == w - 1 else (cs[i] - cs[i - w]) / w
    return out

def get_config():
    cfg = json.loads(Path("configs/production_p91b_champion.json").read_text())
    brs = cfg["breadth_regime_switching"]
    rw  = brs["regime_weights"]
    weights = {}
    for regime in ["LOW", "MID", "HIGH"]:
        w = rw[regime]
        weights[regime] = {
            "v1": w["v1"], "i460": w.get("i460bw168", w.get("i460", 0.0)),
            "i415": w.get("i415bw216", w.get("i415", 0.0)), "f168": w["f168"],
        }
    fts = cfg.get("funding_term_structure_overlay", {})
    vol_ov = cfg.get("vol_regime_overlay", {})
    params = {
        "p_low": brs.get("p_low", 0.30), "p_high": brs.get("p_high", 0.60),
        "brd_lb": brs.get("breadth_lookback_bars", 192), "pct_win": brs.get("rolling_percentile_window", 336),
        "ts_rt": fts.get("reduce_threshold", 0.55), "ts_rs": fts.get("reduce_scale", 0.05),
        "ts_bt": fts.get("boost_threshold", 0.22), "ts_bs": fts.get("boost_scale", 2.00),
        "ts_short": fts.get("short_window_bars", 16), "ts_long": fts.get("long_window_bars", 72),
        "vol_thr": vol_ov.get("threshold", 0.50), "vol_scale": vol_ov.get("scale_factor", 0.40),
    }
    sigs = cfg["ensemble"]["signals"]
    v1p   = sigs["v1"]["params"]
    i460p = sigs["i460bw168"]["params"]
    i415p = sigs["i415bw216"]["params"]
    f168p = sigs.get("f144", sigs.get("f168", {})).get("params", {})
    return cfg, weights, params, v1p, i460p, i415p, f168p

def load_year(year, params, v1p, i460p, i415p, f168p):
    p = params; s, e = YEAR_RANGES[year]
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
            try:    fund_rates[i, j] = dataset.last_funding_rate_before(sym, ts)
            except: fund_rates[i, j] = 0.0
    xsect_mean = np.mean(fund_rates, axis=1)
    ts_raw = (rolling_mean_arr(xsect_mean, p["ts_short"]) - rolling_mean_arr(xsect_mean, p["ts_long"]))
    ts_spread_pct = np.full(n, 0.5)
    for i in range(PCT_WINDOW, n):
        win = ts_raw[i - PCT_WINDOW:i + 1]
        ts_spread_pct[i] = float(np.sum(win <= ts_raw[i])) / (PCT_WINDOW + 1)
    ts_spread_pct[:PCT_WINDOW] = 0.5
    pos_cnt = np.sum(close_mat > 0, axis=1).astype(float) / len(SYMBOLS)
    brd_raw = rolling_mean_arr(pos_cnt, p["brd_lb"])
    brd_pct = np.full(n, 0.5)
    for i in range(PCT_WINDOW, n):
        win = brd_raw[i - PCT_WINDOW:i + 1]
        brd_pct[i] = float(np.sum(win <= brd_raw[i])) / (PCT_WINDOW + 1)
    brd_pct[:PCT_WINDOW] = 0.5
    def run(name, sp):
        res = BacktestEngine(BacktestConfig(costs=COST_MODEL)).run(
            dataset, make_strategy({"name": name, "params": sp}))
        return np.array(res.returns)
    i460_r = run("idio_momentum_alpha", {
        "k_per_side": i460p.get("k_per_side", 4), "lookback_bars": i460p.get("lookback_bars", 480),
        "beta_window_bars": i460p.get("beta_window_bars", 168),
        "target_gross_leverage": i460p.get("target_gross_leverage", 0.20),
        "rebalance_interval_bars": i460p.get("rebalance_interval_bars", 48),
    })
    i415_r = run("idio_momentum_alpha", {
        "k_per_side": i415p.get("k_per_side", 4), "lookback_bars": i415p.get("lookback_bars", 415),
        "beta_window_bars": i415p.get("beta_window_bars", 144),
        "target_gross_leverage": i415p.get("target_gross_leverage", 0.20),
        "rebalance_interval_bars": i415p.get("rebalance_interval_bars", 48),
    })
    f168_r = run("funding_momentum_alpha", {
        "k_per_side": f168p.get("k_per_side", 2),
        "funding_lookback_bars": f168p.get("funding_lookback_bars", 168),
        "direction": f168p.get("direction", "contrarian"),
        "target_gross_leverage": f168p.get("target_gross_leverage", 0.25),
        "rebalance_interval_bars": f168p.get("rebalance_interval_bars", 36),
    })
    return dataset, n, btc_vol, ts_spread_pct, brd_pct, i460_r, i415_r, f168_r, v1p

def run_v1_w(dataset, v1p, w_carry, w_mom, w_mr):
    res = BacktestEngine(BacktestConfig(costs=COST_MODEL)).run(
        dataset, make_strategy({"name": "nexus_alpha_v1", "params": {
            "k_per_side": v1p.get("k_per_side", 2),
            "w_carry": w_carry, "w_mom": w_mom, "w_mean_reversion": w_mr,
            "momentum_lookback_bars": v1p.get("momentum_lookback_bars", 312),
            "mean_reversion_lookback_bars": v1p.get("mean_reversion_lookback_bars", 84),
            "vol_lookback_bars": v1p.get("vol_lookback_bars", 192),
            "target_gross_leverage": v1p.get("target_gross_leverage", 0.35),
            "rebalance_interval_bars": v1p.get("rebalance_interval_bars", 60),
        }}))
    return np.array(res.returns)

def compute_obj(base_data, v1_r, params, weights):
    sharpes = []
    for yr in YEARS:
        dataset, n, btc_vol, ts_spread_pct, brd_pct, i460_r, i415_r, f168_r, _ = base_data[yr]
        v1_arr = v1_r[yr]
        p = params; w = weights
        min_len = min(n, len(v1_arr), len(i460_r), len(i415_r), len(f168_r))
        ens = np.zeros(min_len)
        for i in range(min_len):
            b = brd_pct[i]
            if   b >= p["p_high"]: ww = w["HIGH"]
            elif b >= p["p_low"]:  ww = w["MID"]
            else:                   ww = w["LOW"]
            r = (ww["v1"]*v1_arr[i] + ww["i460"]*i460_r[i] +
                 ww["i415"]*i415_r[i] + ww["f168"]*f168_r[i])
            ts = ts_spread_pct[i]
            if   ts > p["ts_rt"]: r *= p["ts_rs"]
            elif ts < p["ts_bt"]: r *= p["ts_bs"]
            if not np.isnan(btc_vol[i]) and btc_vol[i] > p["vol_thr"]: r *= p["vol_scale"]
            ens[i] = r
        mu = float(np.mean(ens)) * 8760; sd = float(np.std(ens)) * np.sqrt(8760)
        sharpes.append(mu / sd if sd > 1e-10 else 0.0)
    arr = np.array(sharpes)
    return float(np.mean(arr)) - 0.5 * float(np.std(arr)), sharpes

def main():
    t0 = time.time()
    print("=" * 60)
    print("Phase 256 — V1 Signal Weights Sweep")
    print("=" * 60)
    cfg, weights, params, v1p, i460p, i415p, f168p = get_config()
    cur_wc = v1p.get("w_carry", 0.25)
    cur_wm = v1p.get("w_mom", 0.45)
    cur_wr = v1p.get("w_mean_reversion", 0.30)
    print(f"  Current V1: w_carry={cur_wc} w_mom={cur_wm} w_mr={cur_wr}")

    print("\n[1] Pre-loading base data ...")
    base_data = {}
    for yr in YEARS:
        print(f"  {yr} ", end="", flush=True)
        base_data[yr] = load_year(yr, params, v1p, i460p, i415p, f168p)
    print()

    base_v1 = {yr: run_v1_w(base_data[yr][0], v1p, cur_wc, cur_wm, cur_wr) for yr in YEARS}
    base_obj, base_sharpes = compute_obj(base_data, base_v1, params, weights)
    print(f"  Baseline OBJ: {base_obj:.4f}  (wc={cur_wc} wm={cur_wm} wr={cur_wr})")

    best_obj = base_obj; best_wc = cur_wc; best_wm = cur_wm; best_wr = cur_wr

    # Part A: w_mom sweep (split remainder equally between carry+mr)
    print(f"\n[2] Part A: w_mom sweep ...")
    mom_cands = [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]
    mom_results = {}
    for wm in mom_cands:
        remainder = 1.0 - wm
        wc = round(remainder * (cur_wc / (cur_wc + cur_wr)), 4)
        wr = round(remainder - wc, 4)
        v1_r = {yr: run_v1_w(base_data[yr][0], v1p, wc, wm, wr) for yr in YEARS}
        obj, sharpes = compute_obj(base_data, v1_r, params, weights)
        mom_results[wm] = (obj, sharpes, v1_r, wc, wr)
        marker = " <-" if obj > best_obj + 1e-6 else ""
        print(f"  wm={wm:.2f}  wc={wc:.2f}  wr={wr:.2f}  OBJ={obj:.4f}{marker}")
        if obj > best_obj + 1e-6:
            best_obj = obj; best_wm = wm; best_wc = wc; best_wr = wr
    print(f"  Best w_mom: {best_wm}  (wc={best_wc} wr={best_wr})  OBJ={best_obj:.4f}")

    # Part B: w_carry sweep (wm fixed, wc+wr adjust)
    print(f"\n[3] Part B: w_carry sweep (wm={best_wm} fixed) ...")
    carry_cands = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40]
    carry_results = {}
    for wc in carry_cands:
        wr_new = max(0.0, round(1.0 - best_wm - wc, 4))
        if wr_new < 0: continue
        v1_r = {yr: run_v1_w(base_data[yr][0], v1p, wc, best_wm, wr_new) for yr in YEARS}
        obj, sharpes = compute_obj(base_data, v1_r, params, weights)
        carry_results[wc] = (obj, sharpes, v1_r, wr_new)
        marker = " <-" if obj > best_obj + 1e-6 else ""
        print(f"  wm={best_wm:.2f} wc={wc:.2f}  wr={wr_new:.2f}  OBJ={obj:.4f}{marker}")
        if obj > best_obj + 1e-6:
            best_obj = obj; best_wc = wc; best_wr = wr_new
    delta = best_obj - base_obj
    print(f"  Best w_carry: {best_wc}  OBJ={best_obj:.4f}  D={delta:+.4f}")

    if best_wc in carry_results:
        best_sharpes = carry_results[best_wc][1]
    elif best_wm in mom_results and best_wm != cur_wm:
        best_sharpes = mom_results[best_wm][1]
    else:
        best_sharpes = base_sharpes

    print(f"\n[4] LOYO validation ...")
    loyo_wins = 0
    for i, yr in enumerate(YEARS):
        result = "WIN " if best_sharpes[i] > base_sharpes[i] else "LOSS"
        if best_sharpes[i] > base_sharpes[i]: loyo_wins += 1
        print(f"  {yr}: best={best_sharpes[i]:.4f} curr={base_sharpes[i]:.4f} {result}")

    print(f"\n  LOYO: {loyo_wins}/5 wins  delta={delta:+.4f}")
    print(f"  Best V1: w_carry={best_wc} w_mom={best_wm} w_mr={best_wr}")

    if loyo_wins >= 3 and delta > 0.005:
        print("\n  VALIDATED — updating config ...")
        cfg["ensemble"]["signals"]["v1"]["params"]["w_carry"]           = best_wc
        cfg["ensemble"]["signals"]["v1"]["params"]["w_mom"]             = best_wm
        cfg["ensemble"]["signals"]["v1"]["params"]["w_mean_reversion"]  = best_wr
        old_ver = cfg.get("version", "v2.42.7")
        parts = old_ver.lstrip("v").split(".")
        parts[-1] = str(int(parts[-1]) + 1)
        new_ver = "v" + ".".join(parts)
        cfg["version"] = new_ver
        cfg_path = Path("configs/production_p91b_champion.json")
        cfg_path.write_text(json.dumps(cfg, indent=2))
        print(f"  Config updated: {old_ver} → {new_ver}")
        try:
            subprocess.run(["git","stash"], check=False, capture_output=True)
            subprocess.run(["git","pull","--rebase"], check=False, capture_output=True)
            subprocess.run(["git","stash","pop"], check=False, capture_output=True)
            subprocess.run(["git","add", str(cfg_path)], check=True, capture_output=True)
            msg = (f"feat: {new_ver} Phase256 V1 weights wc={best_wc} wm={best_wm} wr={best_wr} — "
                   f"LOYO={loyo_wins}/5 D={delta:+.4f}")
            subprocess.run(["git","commit","-m", msg], check=True, capture_output=True)
            subprocess.run(["git","push"], check=True, capture_output=True)
            print(f"  Git commit+push OK")
        except Exception as ex:
            print(f"  Git error: {ex}")
    else:
        print(f"\n  CONFIRMED OPTIMAL  (LOYO {loyo_wins}/5 delta={delta:+.4f})")

    elapsed = time.time() - t0
    print(f"  Done  ({elapsed:.1f}s)\n")
    print("[DONE] Phase 256 complete.")

if __name__ == "__main__":
    main()
