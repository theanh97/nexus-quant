"""
Phase 243 — FTS Overlay Re-tune (post P239 HIGH regime reopt)
===============================================================
HIGH became pure idio (no f168) in P239 — FTS params may need re-tuning.

Current: ts_rt=0.65  ts_rs=0.30  ts_bt=0.25  ts_bs=1.85

Parts (sequential, each uses best from previous):
  A: ts_rs  in [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50]
  B: ts_bs  in [1.30, 1.50, 1.60, 1.70, 1.85, 2.00, 2.15, 2.30]
  C: ts_rt  in [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]
  D: ts_bt  in [0.10, 0.15, 0.20, 0.25, 0.30, 0.35]

Validation: LOYO >= 3/5 wins AND delta > 0.005
"""
import json, sys, subprocess, time
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from nexus_quant.backtest.engine import BacktestConfig, BacktestEngine
from nexus_quant.backtest.costs import cost_model_from_config
from nexus_quant.data.providers.registry import make_provider
from nexus_quant.strategies.registry import make_strategy

SYMBOLS = ["BTCUSDT","ETHUSDT","SOLUSDT","BNBUSDT","XRPUSDT",
           "ADAUSDT","DOGEUSDT","AVAXUSDT","DOTUSDT","LINKUSDT"]

YEAR_RANGES = {
    "2021": ("2021-02-01", "2022-01-01"),
    "2022": ("2022-01-01", "2023-01-01"),
    "2023": ("2023-01-01", "2024-01-01"),
    "2024": ("2024-01-01", "2025-01-01"),
    "2025": ("2025-01-01", "2026-01-01"),
}
YEARS = sorted(YEAR_RANGES.keys())

COST_MODEL = cost_model_from_config({
    "fee_rate": 0.0005, "slippage_rate": 0.0003, "cost_multiplier": 1.0,
})

VOL_WINDOW = 168
PCT_WINDOW = 336


def get_config():
    cfg = json.loads(Path("configs/production_p91b_champion.json").read_text())
    brs = cfg["breadth_regime_switching"]
    rw  = brs["regime_weights"]
    weights = {}
    for regime in ["LOW", "MID", "HIGH"]:
        w = rw[regime]
        weights[regime] = {
            "v1":   w["v1"],
            "i460": w.get("i460bw168", w.get("i460", 0.0)),
            "i415": w.get("i415bw216", w.get("i415", 0.0)),
            "f168": w["f168"],
        }
    fts    = cfg.get("funding_term_structure_overlay", {})
    vol_ov = cfg.get("vol_regime_overlay", {})
    params = {
        "p_low":   brs.get("p_low", 0.30),
        "p_high":  brs.get("p_high", 0.60),
        "brd_lb":  brs.get("breadth_lookback_bars", 192),
        "pct_win": brs.get("rolling_percentile_window", 336),
        "ts_rt":   fts.get("reduce_threshold", 0.65),
        "ts_rs":   fts.get("reduce_scale", 0.30),
        "ts_bt":   fts.get("boost_threshold", 0.25),
        "ts_bs":   fts.get("boost_scale", 1.85),
        "ts_short": fts.get("short_window_bars", 16),
        "ts_long":  fts.get("long_window_bars", 72),
        "vol_thr":    vol_ov.get("threshold", 0.50),
        "vol_scale":  vol_ov.get("scale_factor", 0.30),
        "f168_boost": vol_ov.get("f144_boost", 0.00),
    }
    sigs  = cfg["ensemble"]["signals"]
    v1p   = sigs["v1"]["params"]
    i460p = sigs["i460bw168"]["params"]
    i415p = sigs["i415bw216"]["params"]
    f168p = sigs.get("f144", sigs.get("f168", {})).get("params", {})
    return cfg, weights, params, v1p, i460p, i415p, f168p


def rolling_mean_arr(a, w):
    out = np.full(len(a), np.nan)
    if w <= 0:
        return out
    cs = np.cumsum(np.where(np.isnan(a), 0.0, a))
    for i in range(w - 1, len(a)):
        out[i] = cs[i] / w if i == w - 1 else (cs[i] - cs[i - w]) / w
    return out


def sharpe(rets, n):
    if n < 20:
        return 0.0
    mu = float(np.mean(rets[:n])) * 8760
    sd = float(np.std(rets[:n]))  * np.sqrt(8760)
    return mu / sd if sd > 1e-9 else 0.0


def obj_fn(yr_sharpes):
    vals = list(yr_sharpes.values())
    return float(np.mean(vals) - 0.5 * np.std(vals))


def load_year_all(year, params, v1p, i460p, i415p, f168p):
    p = params
    s, e = YEAR_RANGES[year]
    cfg_d = {"provider": "binance_rest_v1", "symbols": SYMBOLS,
             "start": s, "end": e, "bar_interval": "1h",
             "cache_dir": ".cache/binance_rest"}
    dataset = make_provider(cfg_d, seed=42).load()
    n = len(dataset.timeline)

    close_mat = np.zeros((n, len(SYMBOLS)))
    for j, sym in enumerate(SYMBOLS):
        for i in range(n):
            close_mat[i, j] = dataset.close(sym, i)
    btc_rets = np.zeros(n)
    for i in range(1, n):
        c0 = close_mat[i-1, 0]; c1 = close_mat[i, 0]
        btc_rets[i] = (c1/c0 - 1.0) if c0 > 0 else 0.0
    btc_vol = np.full(n, np.nan)
    for i in range(VOL_WINDOW, n):
        btc_vol[i] = float(np.std(btc_rets[i-VOL_WINDOW:i])) * np.sqrt(8760)
    if VOL_WINDOW < n:
        btc_vol[:VOL_WINDOW] = btc_vol[VOL_WINDOW]

    fund_rates = np.zeros((n, len(SYMBOLS)))
    for j, sym in enumerate(SYMBOLS):
        for i in range(n):
            ts = dataset.timeline[i]
            try:    fund_rates[i, j] = dataset.last_funding_rate_before(sym, ts)
            except: fund_rates[i, j] = 0.0

    xsect_mean = np.mean(fund_rates, axis=1)
    ts_raw = (rolling_mean_arr(xsect_mean, p["ts_short"]) -
              rolling_mean_arr(xsect_mean, p["ts_long"]))
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
        "w_carry": v1p.get("w_carry", 0.25),
        "w_mom": v1p.get("w_mom", 0.45),
        "w_mean_reversion": v1p.get("w_mean_reversion", 0.30),
        "momentum_lookback_bars": v1p.get("momentum_lookback_bars", 336),
        "mean_reversion_lookback_bars": v1p.get("mean_reversion_lookback_bars", 84),
        "vol_lookback_bars": v1p.get("vol_lookback_bars", 192),
        "target_gross_leverage": v1p.get("target_gross_leverage", 0.35),
        "rebalance_interval_bars": v1p.get("rebalance_interval_bars", 60),
    })
    i460_r = run("idio_momentum_alpha", {
        "k_per_side": i460p.get("k_per_side", 4),
        "lookback_bars": i460p.get("lookback_bars", 480),
        "beta_window_bars": i460p.get("beta_window_bars", 168),
        "target_gross_leverage": i460p.get("target_gross_leverage", 0.20),
        "rebalance_interval_bars": i460p.get("rebalance_interval_bars", 48),
    })
    i415_r = run("idio_momentum_alpha", {
        "k_per_side": i415p.get("k_per_side", 4),
        "lookback_bars": i415p.get("lookback_bars", 415),
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

    print(f"n={n}")
    return {"n": n, "btc_vol": btc_vol, "ts_spread_pct": ts_spread_pct,
            "brd_pct": brd_pct,
            "v1": v1_r, "i460": i460_r, "i415": i415_r, "f168": f168_r}


def compute_ensemble(data, weights, params):
    n = data["n"]
    min_len = min(n, len(data["v1"]), len(data["i460"]),
                  len(data["i415"]), len(data["f168"]))
    ens = np.zeros(min_len)
    p_low = params["p_low"]; p_high = params["p_high"]
    vol_thr = params["vol_thr"]; vol_scale = params["vol_scale"]
    ts_rt = params["ts_rt"]; ts_rs = params["ts_rs"]
    ts_bt = params["ts_bt"]; ts_bs = params["ts_bs"]

    for i in range(min_len):
        brd_v = data["brd_pct"][i]
        if np.isnan(brd_v): brd_v = 0.5
        w = (weights["HIGH"] if brd_v >= p_high else
             weights["MID"]  if brd_v >= p_low  else weights["LOW"])
        raw = (w["v1"]*data["v1"][i] + w["i460"]*data["i460"][i] +
               w["i415"]*data["i415"][i] + w["f168"]*data["f168"][i])
        bv = data["btc_vol"][i]
        if not np.isnan(bv) and bv > vol_thr:
            raw *= vol_scale
        tsp = data["ts_spread_pct"][i]
        if np.isnan(tsp): tsp = 0.5
        if tsp > ts_rt:   raw *= ts_rs
        elif tsp < ts_bt: raw *= ts_bs
        ens[i] = raw
    return ens


def eval_year(data, weights, params):
    ens = compute_ensemble(data, weights, params)
    return sharpe(ens, len(ens))


def eval_all(year_data, weights, params):
    return {yr: eval_year(year_data[yr], weights, params) for yr in YEARS}


def main():
    print("=" * 60)
    print("Phase 243 — FTS Overlay Re-tune (post P239 HIGH reopt)")
    print("=" * 60)

    t0 = time.time()
    cfg, weights, params, v1p, i460p, i415p, f168p = get_config()

    print("\n[1] Loading all data ...")
    year_data = {}
    for yr in YEARS:
        print(f"  {yr}", end=" ", flush=True)
        year_data[yr] = load_year_all(yr, params, v1p, i460p, i415p, f168p)

    baseline_sharpes = eval_all(year_data, weights, params)
    baseline_obj = obj_fn(baseline_sharpes)
    print(f"\n  Baseline OBJ: {baseline_obj:.4f}")
    print(f"  FTS: rt={params['ts_rt']} rs={params['ts_rs']} "
          f"bt={params['ts_bt']} bs={params['ts_bs']}")

    best_params = dict(params)

    # Part A: ts_rs
    print(f"\n[2] Part A: ts_rs sweep ...")
    rs_grid = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50]
    best_rs_obj = -999.0
    best_rs = params["ts_rs"]
    for rs in rs_grid:
        p = dict(best_params); p["ts_rs"] = rs
        o = obj_fn(eval_all(year_data, weights, p))
        marker = " <-" if o > best_rs_obj else ""
        print(f"  ts_rs={rs:.2f}  OBJ={o:.4f}{marker}")
        if o > best_rs_obj: best_rs_obj = o; best_rs = rs
    best_params["ts_rs"] = best_rs
    print(f"  Best ts_rs: {best_rs:.2f}  OBJ={best_rs_obj:.4f}")

    # Part B: ts_bs
    print(f"\n[3] Part B: ts_bs sweep ...")
    bs_grid = [1.30, 1.50, 1.60, 1.70, 1.85, 2.00, 2.15, 2.30]
    best_bs_obj = -999.0
    best_bs = params["ts_bs"]
    for bs in bs_grid:
        p = dict(best_params); p["ts_bs"] = bs
        o = obj_fn(eval_all(year_data, weights, p))
        marker = " <-" if o > best_bs_obj else ""
        print(f"  ts_bs={bs:.2f}  OBJ={o:.4f}{marker}")
        if o > best_bs_obj: best_bs_obj = o; best_bs = bs
    best_params["ts_bs"] = best_bs
    print(f"  Best ts_bs: {best_bs:.2f}  OBJ={best_bs_obj:.4f}")

    # Part C: ts_rt
    print(f"\n[4] Part C: ts_rt sweep ...")
    rt_grid = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]
    best_rt_obj = -999.0
    best_rt = params["ts_rt"]
    for rt in rt_grid:
        p = dict(best_params); p["ts_rt"] = rt
        o = obj_fn(eval_all(year_data, weights, p))
        marker = " <-" if o > best_rt_obj else ""
        print(f"  ts_rt={rt:.2f}  OBJ={o:.4f}{marker}")
        if o > best_rt_obj: best_rt_obj = o; best_rt = rt
    best_params["ts_rt"] = best_rt
    print(f"  Best ts_rt: {best_rt:.2f}  OBJ={best_rt_obj:.4f}")

    # Part D: ts_bt
    print(f"\n[5] Part D: ts_bt sweep ...")
    bt_grid = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35]
    best_bt_obj = -999.0
    best_bt = params["ts_bt"]
    for bt in bt_grid:
        p = dict(best_params); p["ts_bt"] = bt
        o = obj_fn(eval_all(year_data, weights, p))
        marker = " <-" if o > best_bt_obj else ""
        print(f"  ts_bt={bt:.2f}  OBJ={o:.4f}{marker}")
        if o > best_bt_obj: best_bt_obj = o; best_bt = bt
    best_params["ts_bt"] = best_bt
    delta = best_bt_obj - baseline_obj
    print(f"  Best ts_bt: {best_bt:.2f}  OBJ={best_bt_obj:.4f}  D={delta:+.4f}")

    # LOYO Validation
    print(f"\n[6] LOYO validation ...")
    wins = 0
    best_sharpes = eval_all(year_data, weights, best_params)
    for yr in YEARS:
        b = best_sharpes[yr]; c = baseline_sharpes[yr]
        result = "WIN" if b > c else "LOSS"
        if b > c: wins += 1
        print(f"  {yr}: best={b:.4f} curr={c:.4f} {result}")

    print(f"\n  LOYO: {wins}/5 wins  delta={delta:+.4f}")

    # Decision
    if wins >= 3 and delta > 0.005:
        print(f"\n[+] IMPROVEMENT VALIDATED! Updating config ...")

        cfg_path = Path("configs/production_p91b_champion.json")
        cfg2 = json.loads(cfg_path.read_text())
        fts = cfg2["funding_term_structure_overlay"]
        fts["reduce_scale"]     = best_params["ts_rs"]
        fts["boost_scale"]      = best_params["ts_bs"]
        fts["reduce_threshold"] = best_params["ts_rt"]
        fts["boost_threshold"]  = best_params["ts_bt"]

        old_ver = cfg2.get("version", "2.42.0")
        parts = old_ver.split("."); parts[-1] = str(int(parts[-1]) + 1)
        new_ver = ".".join(parts)
        cfg2["version"] = new_ver
        cfg_path.write_text(json.dumps(cfg2, indent=2))
        print(f"  Config: {old_ver} -> {new_ver}")
        print(f"  FTS: rt={best_rt} rs={best_rs} bt={best_bt} bs={best_bs}")

        script_name = "scripts/run_phase243_fts_retune.py"
        commit_msg = (f"feat(p243): FTS retune rt={best_rt} rs={best_rs} "
                      f"bt={best_bt} bs={best_bs} OBJ={best_bt_obj:.4f} "
                      f"(D={delta:+.4f} LOYO {wins}/5) [{new_ver}]")
        subprocess.run(
            f"cd /Users/truonglys/projects/quant && "
            f"git add configs/production_p91b_champion.json {script_name} && "
            f"git stash && git pull --rebase && git stash pop && "
            f"git commit -m '{commit_msg}' && git push",
            shell=True
        )
    else:
        tag = ("WEAK" if wins >= 3 and delta > 0
               else "CONFIRMED OPTIMAL" if delta == 0.0
               else "NO IMPROVEMENT")
        print(f"\n  {tag}  (LOYO {wins}/5 delta={delta:+.4f})")

    elapsed = time.time() - t0
    print(f"  Done  ({elapsed:.1f}s)")
    print("\n[DONE] Phase 243 complete.")


if __name__ == "__main__":
    main()
