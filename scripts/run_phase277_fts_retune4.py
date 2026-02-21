#!/usr/bin/env python3
"""Phase 277 — FTS Overlay Retune Round 4. Sweep ts_rt, ts_rs, ts_bt, ts_bs."""
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
    brs = cfg["breadth_regime_switching"]; rw = brs["regime_weights"]
    weights = {}
    for regime in ["LOW", "MID", "HIGH"]:
        w = rw[regime]
        weights[regime] = {"v1": w["v1"], "i460": w.get("i460bw168", w.get("i460", 0.0)),
                           "i415": w.get("i415bw216", w.get("i415", 0.0)), "f168": w["f168"]}
    fts = cfg.get("funding_term_structure_overlay", {}); vol_ov = cfg.get("vol_regime_overlay", {})
    params = {
        "p_low": brs.get("p_low", 0.30), "p_high": brs.get("p_high", 0.68),
        "brd_lb": brs.get("breadth_lookback_bars", 192), "pct_win": brs.get("rolling_percentile_window", 336),
        "ts_rt": fts.get("reduce_threshold", 0.55), "ts_rs": fts.get("reduce_scale", 0.05),
        "ts_bt": fts.get("boost_threshold", 0.22), "ts_bs": fts.get("boost_scale", 2.00),
        "ts_short": fts.get("short_window_bars", 16), "ts_long": fts.get("long_window_bars", 72),
        "vol_thr": vol_ov.get("threshold", 0.50), "vol_scale": vol_ov.get("scale_factor", 0.40),
    }
    sigs = cfg["ensemble"]["signals"]
    return cfg, weights, params, sigs["v1"]["params"], sigs["i460bw168"]["params"], sigs["i415bw216"]["params"], sigs.get("f144", sigs.get("f168", {})).get("params", {})

def load_year_all(year, params, v1p, i460p, i415p, f168p):
    p = params; s, e = YEAR_RANGES[year]
    dataset = make_provider({"provider": "binance_rest_v1", "symbols": SYMBOLS, "start": s, "end": e, "bar_interval": "1h", "cache_dir": ".cache/binance_rest"}, seed=42).load()
    n = len(dataset.timeline)
    close_mat = np.zeros((n, len(SYMBOLS)))
    for j, sym in enumerate(SYMBOLS):
        for i in range(n): close_mat[i, j] = dataset.close(sym, i)
    btc_rets = np.zeros(n)
    for i in range(1, n):
        c0 = close_mat[i-1, 0]; c1 = close_mat[i, 0]
        btc_rets[i] = (c1/c0 - 1.0) if c0 > 0 else 0.0
    btc_vol = np.full(n, np.nan)
    for i in range(VOL_WINDOW, n): btc_vol[i] = float(np.std(btc_rets[i-VOL_WINDOW:i])) * np.sqrt(8760)
    if VOL_WINDOW < n: btc_vol[:VOL_WINDOW] = btc_vol[VOL_WINDOW]
    fund_rates = np.zeros((n, len(SYMBOLS)))
    for j, sym in enumerate(SYMBOLS):
        for i in range(n):
            ts = dataset.timeline[i]
            try: fund_rates[i, j] = dataset.last_funding_rate_before(sym, ts)
            except: fund_rates[i, j] = 0.0
    xsect_mean = np.mean(fund_rates, axis=1)
    ts_raw = rolling_mean_arr(xsect_mean, p["ts_short"]) - rolling_mean_arr(xsect_mean, p["ts_long"])
    ts_spread_pct = np.full(n, 0.5)
    for i in range(PCT_WINDOW, n):
        win = ts_raw[i - PCT_WINDOW:i + 1]; ts_spread_pct[i] = float(np.sum(win <= ts_raw[i])) / (PCT_WINDOW + 1)
    ts_spread_pct[:PCT_WINDOW] = 0.5
    pos_cnt = np.sum(close_mat > 0, axis=1).astype(float) / len(SYMBOLS)
    brd_raw = rolling_mean_arr(pos_cnt, p["brd_lb"])
    brd_pct = np.full(n, 0.5)
    for i in range(PCT_WINDOW, n):
        win = brd_raw[i - PCT_WINDOW:i + 1]; brd_pct[i] = float(np.sum(win <= brd_raw[i])) / (PCT_WINDOW + 1)
    brd_pct[:PCT_WINDOW] = 0.5
    def run(name, sp):
        res = BacktestEngine(BacktestConfig(costs=COST_MODEL)).run(dataset, make_strategy({"name": name, "params": sp}))
        return np.array(res.returns)
    v1_r = run("nexus_alpha_v1", {"k_per_side": v1p.get("k_per_side", 2), "w_carry": v1p.get("w_carry", 0.15), "w_mom": v1p.get("w_mom", 0.55), "w_mean_reversion": v1p.get("w_mean_reversion", 0.30), "momentum_lookback_bars": v1p.get("momentum_lookback_bars", 312), "mean_reversion_lookback_bars": v1p.get("mean_reversion_lookback_bars", 84), "vol_lookback_bars": v1p.get("vol_lookback_bars", 192), "target_gross_leverage": v1p.get("target_gross_leverage", 0.35), "rebalance_interval_bars": v1p.get("rebalance_interval_bars", 60)})
    i460_r = run("idio_momentum_alpha", {"k_per_side": i460p.get("k_per_side", 4), "lookback_bars": i460p.get("lookback_bars", 480), "beta_window_bars": i460p.get("beta_window_bars", 168), "target_gross_leverage": i460p.get("target_gross_leverage", 0.20), "rebalance_interval_bars": i460p.get("rebalance_interval_bars", 48)})
    i415_r = run("idio_momentum_alpha", {"k_per_side": i415p.get("k_per_side", 4), "lookback_bars": i415p.get("lookback_bars", 415), "beta_window_bars": i415p.get("beta_window_bars", 144), "target_gross_leverage": i415p.get("target_gross_leverage", 0.20), "rebalance_interval_bars": i415p.get("rebalance_interval_bars", 48)})
    f168_r = run("funding_momentum_alpha", {"k_per_side": f168p.get("k_per_side", 2), "funding_lookback_bars": f168p.get("funding_lookback_bars", 168), "direction": f168p.get("direction", "contrarian"), "target_gross_leverage": f168p.get("target_gross_leverage", 0.25), "rebalance_interval_bars": f168p.get("rebalance_interval_bars", 36)})
    print(f"  n={n}")
    return {"n": n, "btc_vol": btc_vol, "ts_raw": ts_raw, "brd_pct": brd_pct, "v1": v1_r, "i460": i460_r, "i415": i415_r, "f168": f168_r, "ts_spread_pct_cache": ts_spread_pct}

def compute_ensemble(data, params, weights, ts_rt=None, ts_rs=None, ts_bt=None, ts_bs=None):
    n = data["n"]; p = params; w = weights
    v1r = data["v1"]; i460r = data["i460"]; i415r = data["i415"]; f168r = data["f168"]
    min_len = min(n, len(v1r), len(i460r), len(i415r), len(f168r))
    brd = data["brd_pct"]; vol = data["btc_vol"]
    ts_raw = data["ts_raw"]
    # Recompute ts_spread_pct if custom ts params
    ts_spread_pct = data["ts_spread_pct_cache"]
    _rt = ts_rt if ts_rt is not None else p["ts_rt"]
    _rs = ts_rs if ts_rs is not None else p["ts_rs"]
    _bt = ts_bt if ts_bt is not None else p["ts_bt"]
    _bs = ts_bs if ts_bs is not None else p["ts_bs"]
    ens = np.zeros(min_len)
    for i in range(min_len):
        b = brd[i]
        if   b >= p["p_high"]: ww = w["HIGH"]
        elif b >= p["p_low"]:  ww = w["MID"]
        else:                   ww = w["LOW"]
        r = ww["v1"]*v1r[i] + ww["i460"]*i460r[i] + ww["i415"]*i415r[i] + ww["f168"]*f168r[i]
        ts = ts_spread_pct[i]
        if ts > _rt:   r *= _rs
        elif ts < _bt: r *= _bs
        if not np.isnan(vol[i]) and vol[i] > p["vol_thr"]: r *= p["vol_scale"]
        ens[i] = r
    return ens

def yearly_sharpe_fts(data, params, weights, ts_rt, ts_rs, ts_bt, ts_bs):
    ens = compute_ensemble(data, params, weights, ts_rt, ts_rs, ts_bt, ts_bs)
    mu = float(np.mean(ens)) * 8760; sd = float(np.std(ens)) * np.sqrt(8760)
    return mu / sd if sd > 1e-10 else 0.0

def eval_fts(yearly_data, params, weights, ts_rt, ts_rs, ts_bt, ts_bs):
    sharpes = [yearly_sharpe_fts(yearly_data[yr], params, weights, ts_rt, ts_rs, ts_bt, ts_bs) for yr in YEARS]
    return float(np.mean(sharpes)) - 0.5 * float(np.std(sharpes))

def main():
    t0 = time.time()
    print("[1] Loading config ...")
    cfg, weights, params, v1p, i460p, i415p, f168p = get_config()
    ts_rt0 = params["ts_rt"]; ts_rs0 = params["ts_rs"]
    ts_bt0 = params["ts_bt"]; ts_bs0 = params["ts_bs"]
    print(f"    Current FTS: rt={ts_rt0} rs={ts_rs0} bt={ts_bt0} bs={ts_bs0}")

    print("[2] Loading yearly data ...")
    yearly_data = {}
    for yr in YEARS:
        print(f"    Loading {yr} ...", flush=True)
        yearly_data[yr] = load_year_all(yr, params, v1p, i460p, i415p, f168p)

    curr_obj = eval_fts(yearly_data, params, weights, ts_rt0, ts_rs0, ts_bt0, ts_bs0)
    print(f"    Baseline OBJ={curr_obj:.4f}")

    best_rt = ts_rt0; best_rs = ts_rs0; best_bt = ts_bt0; best_bs = ts_bs0; best_obj = curr_obj

    # Sweep ts_rt (reduce threshold)
    print("\n[3] ts_rt sweep ...")
    for rt in [0.45, 0.50, 0.55, 0.60, 0.65, 0.70]:
        obj = eval_fts(yearly_data, params, weights, rt, best_rs, best_bt, best_bs)
        marker = " <-" if obj > best_obj else ""
        print(f"  rt={rt:.2f}  OBJ={obj:.4f}{marker}", flush=True)
        if obj > best_obj: best_obj = obj; best_rt = rt
    print(f"  Best rt: {best_rt:.2f}  OBJ={best_obj:.4f}  D={best_obj-curr_obj:+.4f}")

    # Sweep ts_rs (reduce scale)
    print("\n[4] ts_rs sweep ...")
    for rs in [0.00, 0.02, 0.05, 0.08, 0.10, 0.15, 0.20]:
        obj = eval_fts(yearly_data, params, weights, best_rt, rs, best_bt, best_bs)
        marker = " <-" if obj > best_obj else ""
        print(f"  rs={rs:.2f}  OBJ={obj:.4f}{marker}", flush=True)
        if obj > best_obj: best_obj = obj; best_rs = rs
    print(f"  Best rs: {best_rs:.2f}  OBJ={best_obj:.4f}  D={best_obj-curr_obj:+.4f}")

    # Sweep ts_bt (boost threshold)
    print("\n[5] ts_bt sweep ...")
    for bt in [0.15, 0.18, 0.20, 0.22, 0.25, 0.28, 0.30, 0.35]:
        obj = eval_fts(yearly_data, params, weights, best_rt, best_rs, bt, best_bs)
        marker = " <-" if obj > best_obj else ""
        print(f"  bt={bt:.2f}  OBJ={obj:.4f}{marker}", flush=True)
        if obj > best_obj: best_obj = obj; best_bt = bt
    print(f"  Best bt: {best_bt:.2f}  OBJ={best_obj:.4f}  D={best_obj-curr_obj:+.4f}")

    # Sweep ts_bs (boost scale)
    print("\n[6] ts_bs sweep ...")
    for bs in [1.50, 1.75, 2.00, 2.25, 2.50, 3.00]:
        obj = eval_fts(yearly_data, params, weights, best_rt, best_rs, best_bt, bs)
        marker = " <-" if obj > best_obj else ""
        print(f"  bs={bs:.2f}  OBJ={obj:.4f}{marker}", flush=True)
        if obj > best_obj: best_obj = obj; best_bs = bs
    print(f"  Best bs: {best_bs:.2f}  OBJ={best_obj:.4f}  D={best_obj-curr_obj:+.4f}")

    # LOYO validation
    print("\n[7] LOYO validation ...")
    wins = 0
    for yr in YEARS:
        s_best = yearly_sharpe_fts(yearly_data[yr], params, weights, best_rt, best_rs, best_bt, best_bs)
        s_curr = yearly_sharpe_fts(yearly_data[yr], params, weights, ts_rt0, ts_rs0, ts_bt0, ts_bs0)
        result = "WIN" if s_best > s_curr else "LOSS"
        if s_best > s_curr: wins += 1
        print(f"  {yr}: best={s_best:.4f} curr={s_curr:.4f} {result}")

    delta = best_obj - curr_obj
    print(f"\n  LOYO: {wins}/5 wins  delta={delta:+.4f}")
    print(f"  Best FTS: rt={best_rt} rs={best_rs} bt={best_bt} bs={best_bs}")

    if wins >= 3 and delta > 0.005:
        print(f"\n  VALIDATED  (LOYO {wins}/5 delta={delta:+.4f})")
        cfg2 = json.loads(Path("configs/production_p91b_champion.json").read_text())
        old_ver = cfg2.get("version", "v0.0.0"); parts = old_ver.lstrip("v").split(".")
        parts[-1] = str(int(parts[-1]) + 1); new_ver = "v" + ".".join(parts)
        fts2 = cfg2.get("funding_term_structure_overlay", {})
        fts2["reduce_threshold"] = best_rt; fts2["reduce_scale"] = best_rs
        fts2["boost_threshold"] = best_bt; fts2["boost_scale"] = best_bs
        cfg2["version"] = new_ver
        p = Path("configs/production_p91b_champion.json")
        p.write_text(json.dumps(cfg2, indent=2))
        print(f"  Config updated: {old_ver} → {new_ver}")
        print(f"  FTS: rt={best_rt} rs={best_rs} bt={best_bt} bs={best_bs}  New OBJ={best_obj:.4f}")
        msg = f"quant: {new_ver} — FTS retune r4 (rt={best_rt} bs={best_bs}) OBJ={best_obj:.4f}"
        subprocess.run(["git", "add", str(p)], cwd="/Users/truonglys/projects/quant")
        subprocess.run(["git", "commit", "-m", msg], cwd="/Users/truonglys/projects/quant")
        print("  Git committed.")
    else:
        if wins >= 3: print(f"\n  CONFIRMED OPTIMAL  (LOYO {wins}/5 delta={delta:+.4f} below threshold)")
        else: print(f"\n  NOT VALIDATED  (LOYO {wins}/5 delta={delta:+.4f})")
        print(f"  FTS stays: rt={ts_rt0} rs={ts_rs0} bt={ts_bt0} bs={ts_bs0}")

    elapsed = time.time() - t0
    print(f"\n[DONE] Phase 277 complete.  ({elapsed:.1f}s)")

if __name__ == "__main__":
    main()
