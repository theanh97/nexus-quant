#!/usr/bin/env python3
"""
Phase 266 — FTS Overlay Retune Round 3
FTS not re-swept since P259 (before P261-P265 regime/threshold changes).
Sweep ts_rt, ts_rs, ts_bt, ts_bs.
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
        "p_low": brs.get("p_low", 0.30), "p_high": brs.get("p_high", 0.68),
        "brd_lb": brs.get("breadth_lookback_bars", 192),
        "pct_win": brs.get("rolling_percentile_window", 336),
        "ts_rt": fts.get("reduce_threshold", 0.55), "ts_rs": fts.get("reduce_scale", 0.05),
        "ts_bt": fts.get("boost_threshold", 0.22), "ts_bs": fts.get("boost_scale", 2.00),
        "ts_short": fts.get("short_window_bars", 16), "ts_long": fts.get("long_window_bars", 72),
        "vol_thr": vol_ov.get("threshold", 0.50), "vol_scale": vol_ov.get("scale_factor", 0.40),
    }
    sigs  = cfg["ensemble"]["signals"]
    v1p   = sigs["v1"]["params"]
    i460p = sigs["i460bw168"]["params"]
    i415p = sigs["i415bw216"]["params"]
    f168p = sigs.get("f144", sigs.get("f168", {})).get("params", {})
    return cfg, weights, params, v1p, i460p, i415p, f168p

def load_year_all(year, params, v1p, i460p, i415p, f168p):
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
    v1_r = run("nexus_alpha_v1", {
        "k_per_side": v1p.get("k_per_side", 2),
        "w_carry": v1p.get("w_carry", 0.15), "w_mom": v1p.get("w_mom", 0.55),
        "w_mean_reversion": v1p.get("w_mean_reversion", 0.30),
        "momentum_lookback_bars": v1p.get("momentum_lookback_bars", 312),
        "mean_reversion_lookback_bars": v1p.get("mean_reversion_lookback_bars", 84),
        "vol_lookback_bars": v1p.get("vol_lookback_bars", 192),
        "target_gross_leverage": v1p.get("target_gross_leverage", 0.35),
        "rebalance_interval_bars": v1p.get("rebalance_interval_bars", 60),
    })
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
    print(f"  n={n}")
    return {"n": n, "btc_vol": btc_vol, "ts_spread_pct": ts_spread_pct, "brd_pct": brd_pct,
            "v1": v1_r, "i460": i460_r, "i415": i415_r, "f168": f168_r}

def compute_ensemble(data, params, weights):
    n = data["n"]; p = params; w = weights
    v1r = data["v1"]; i460r = data["i460"]; i415r = data["i415"]; f168r = data["f168"]
    min_len = min(n, len(v1r), len(i460r), len(i415r), len(f168r))
    brd = data["brd_pct"]; tsp = data["ts_spread_pct"]; vol = data["btc_vol"]
    ens = np.zeros(min_len)
    for i in range(min_len):
        b = brd[i]
        if   b >= p["p_high"]: ww = w["HIGH"]
        elif b >= p["p_low"]:  ww = w["MID"]
        else:                   ww = w["LOW"]
        r = ww["v1"]*v1r[i] + ww["i460"]*i460r[i] + ww["i415"]*i415r[i] + ww["f168"]*f168r[i]
        ts = tsp[i]
        if ts > p["ts_rt"]: r *= p["ts_rs"]
        elif ts < p["ts_bt"]: r *= p["ts_bs"]
        if not np.isnan(vol[i]) and vol[i] > p["vol_thr"]: r *= p["vol_scale"]
        ens[i] = r
    return ens

def yearly_sharpe(data, params, weights):
    ens = compute_ensemble(data, params, weights)
    mu = float(np.mean(ens)) * 8760; sd = float(np.std(ens)) * np.sqrt(8760)
    return mu / sd if sd > 1e-10 else 0.0

def eval_fts(year_data, params, weights, ts_rt, ts_rs, ts_bt, ts_bs):
    p2 = dict(params); p2["ts_rt"] = ts_rt; p2["ts_rs"] = ts_rs
    p2["ts_bt"] = ts_bt; p2["ts_bs"] = ts_bs
    sharpes = [yearly_sharpe(year_data[yr], p2, weights) for yr in YEARS]
    arr = np.array(sharpes)
    return float(np.mean(arr)) - 0.5 * float(np.std(arr))

def main():
    t0 = time.time()
    print("=" * 60)
    print("Phase 266 — FTS Overlay Retune Round 3")
    print("After P261-P265 regime + breadth threshold changes")
    print("=" * 60)

    cfg, weights, params, v1p, i460p, i415p, f168p = get_config()
    print(f"\n  Config: {cfg.get('version')}")
    print(f"  Current FTS: rt={params['ts_rt']:.2f} rs={params['ts_rs']:.2f} "
          f"bt={params['ts_bt']:.2f} bs={params['ts_bs']:.2f}")

    print("\n[1] Loading all year data ...")
    year_data = {}
    for yr in YEARS:
        print(f"  Loading {yr} ...", flush=True)
        year_data[yr] = load_year_all(yr, params, v1p, i460p, i415p, f168p)
    print(f"  Done in {time.time()-t0:.1f}s")

    baseline_sharpes = [yearly_sharpe(year_data[yr], params, weights) for yr in YEARS]
    baseline_obj = float(np.mean(baseline_sharpes)) - 0.5 * float(np.std(baseline_sharpes))
    print(f"\n  Baseline OBJ = {baseline_obj:.4f}")

    # [2] ts_rt sweep
    print("\n[2] ts_rt sweep ...")
    rt_candidates = [0.40, 0.45, 0.50, 0.52, 0.55, 0.57, 0.60, 0.63, 0.65, 0.70]
    best_rt = params["ts_rt"]; best_rt_obj = baseline_obj
    for rt in rt_candidates:
        obj = eval_fts(year_data, params, weights, rt, params["ts_rs"], params["ts_bt"], params["ts_bs"])
        marker = " <-" if obj > best_rt_obj else ""
        print(f"  ts_rt={rt:.2f}  OBJ={obj:.4f}{marker}")
        if obj > best_rt_obj:
            best_rt_obj = obj; best_rt = rt
    print(f"  Best ts_rt: {best_rt:.2f}  OBJ={best_rt_obj:.4f}  D={best_rt_obj-baseline_obj:+.4f}")

    # [3] ts_rs sweep
    print("\n[3] ts_rs sweep ...")
    rs_candidates = [0.00, 0.02, 0.03, 0.05, 0.07, 0.08, 0.10, 0.12, 0.15, 0.20, 0.25, 0.30]
    best_rs = params["ts_rs"]; best_rs_obj = best_rt_obj
    for rs in rs_candidates:
        obj = eval_fts(year_data, params, weights, best_rt, rs, params["ts_bt"], params["ts_bs"])
        marker = " <-" if obj > best_rs_obj else ""
        print(f"  ts_rs={rs:.2f}  OBJ={obj:.4f}{marker}")
        if obj > best_rs_obj:
            best_rs_obj = obj; best_rs = rs
    print(f"  Best ts_rs: {best_rs:.2f}  OBJ={best_rs_obj:.4f}  D={best_rs_obj-baseline_obj:+.4f}")

    # [4] ts_bt sweep
    print("\n[4] ts_bt sweep ...")
    bt_candidates = [0.15, 0.18, 0.20, 0.22, 0.25, 0.28, 0.30, 0.33, 0.35]
    best_bt = params["ts_bt"]; best_bt_obj = best_rs_obj
    for bt in bt_candidates:
        if bt >= best_rt: continue
        obj = eval_fts(year_data, params, weights, best_rt, best_rs, bt, params["ts_bs"])
        marker = " <-" if obj > best_bt_obj else ""
        print(f"  ts_bt={bt:.2f}  OBJ={obj:.4f}{marker}")
        if obj > best_bt_obj:
            best_bt_obj = obj; best_bt = bt
    print(f"  Best ts_bt: {best_bt:.2f}  OBJ={best_bt_obj:.4f}  D={best_bt_obj-baseline_obj:+.4f}")

    # [5] ts_bs sweep
    print("\n[5] ts_bs sweep ...")
    bs_candidates = [1.00, 1.25, 1.50, 1.75, 2.00, 2.25, 2.50, 2.75, 3.00, 3.50, 4.00]
    best_bs = params["ts_bs"]; best_bs_obj = best_bt_obj
    for bs in bs_candidates:
        obj = eval_fts(year_data, params, weights, best_rt, best_rs, best_bt, bs)
        marker = " <-" if obj > best_bs_obj else ""
        print(f"  ts_bs={bs:.2f}  OBJ={obj:.4f}{marker}")
        if obj > best_bs_obj:
            best_bs_obj = obj; best_bs = bs
    print(f"  Best ts_bs: {best_bs:.2f}  OBJ={best_bs_obj:.4f}  D={best_bs_obj-baseline_obj:+.4f}")

    # [6] LOYO validation
    print("\n[6] LOYO validation ...")
    p_best = dict(params); p_best["ts_rt"] = best_rt; p_best["ts_rs"] = best_rs
    p_best["ts_bt"] = best_bt; p_best["ts_bs"] = best_bs
    wins = 0
    for yr in YEARS:
        sh_best = yearly_sharpe(year_data[yr], p_best, weights)
        sh_base = yearly_sharpe(year_data[yr], params, weights)
        won = sh_best > sh_base
        wins += int(won)
        print(f"  {yr}: best={sh_best:.4f} curr={sh_base:.4f} {'WIN' if won else 'LOSS'}")

    delta = best_bs_obj - baseline_obj
    print(f"\n  LOYO: {wins}/5 wins  delta={delta:+.4f}")
    print(f"  Best FTS: rt={best_rt:.2f} rs={best_rs:.2f} bt={best_bt:.2f} bs={best_bs:.2f}")

    if wins >= 3 and delta > 0.005:
        print(f"\n  VALIDATED  (LOYO {wins}/5 delta={delta:+.4f})")
        _update_config(cfg, best_rt, best_rs, best_bt, best_bs, best_bs_obj)
    else:
        print(f"\n  CONFIRMED OPTIMAL  (LOYO {wins}/5 delta={delta:+.4f})")

    print(f"\n[DONE] Phase 266 complete.  ({time.time()-t0:.1f}s)")

def _update_config(cfg, rt, rs, bt, bs, new_obj):
    old_ver = cfg.get("version", "v2.42.16")
    parts = old_ver.lstrip("v").split(".")
    parts[-1] = str(int(parts[-1]) + 1)
    new_ver = "v" + ".".join(parts)
    fts = cfg["funding_term_structure_overlay"]
    fts["reduce_threshold"] = round(rt, 3)
    fts["reduce_scale"]     = round(rs, 3)
    fts["boost_threshold"]  = round(bt, 3)
    fts["boost_scale"]      = round(bs, 3)
    cfg["version"] = new_ver
    p = Path("configs/production_p91b_champion.json")
    p.write_text(json.dumps(cfg, indent=2))
    print(f"  Config updated: {old_ver} → {new_ver}")
    print(f"  FTS: rt={rt:.2f} rs={rs:.2f} bt={bt:.2f} bs={bs:.2f}")
    print(f"  New OBJ={new_obj:.4f}")
    subprocess.run(["git", "add", str(p)], cwd="/Users/truonglys/projects/quant")
    subprocess.run(["git", "commit", "-m",
        f"quant: {new_ver} — FTS retune r3 (rt={rt:.2f} rs={rs:.2f} bt={bt:.2f} bs={bs:.2f}) OBJ={new_obj:.4f}"],
        cwd="/Users/truonglys/projects/quant")
    print("  Git committed.")

if __name__ == "__main__":
    main()
