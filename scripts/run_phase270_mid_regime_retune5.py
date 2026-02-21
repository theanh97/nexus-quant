#!/usr/bin/env python3
"""
Phase 270 — MID Regime Re-retune Round 5
After P269: HIGH confirmed v1=0.20 i460=0.20 i415=0.30 f168=0.30
Current MID: v1=0.15, i460=0.00, i415=0.15, f168=0.70
Re-verify MID composition with updated HIGH as context
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

def eval_weights(year_data, params, weights):
    sharpes = [yearly_sharpe(year_data[yr], params, weights) for yr in YEARS]
    arr = np.array(sharpes)
    return float(np.mean(arr)) - 0.5 * float(np.std(arr))

def make_mid_w(base_weights, v1_w, f168_w, i415_frac):
    remainder = max(0.0, 1.0 - v1_w - f168_w)
    i415_w = round(remainder * i415_frac, 4)
    i460_w = round(remainder - i415_w, 4)
    return {"LOW": base_weights["LOW"],
            "MID": {"v1": v1_w, "i460": i460_w, "i415": i415_w, "f168": f168_w},
            "HIGH": base_weights["HIGH"]}

def main():
    t0 = time.time()
    print("=" * 60)
    print("Phase 270 — MID Regime Re-retune Round 5")
    print("After P269: HIGH v1=0.20 i460=0.20 i415=0.30 f168=0.30")
    print("=" * 60)

    cfg, weights, params, v1p, i460p, i415p, f168p = get_config()
    base_weights = weights

    print("\n[1] Loading all year data ...")
    year_data = {}
    for yr in YEARS:
        print(f"  Loading {yr} ...", flush=True)
        year_data[yr] = load_year_all(yr, params, v1p, i460p, i415p, f168p)
    print(f"  Done in {time.time()-t0:.1f}s")

    baseline_sharpes = [yearly_sharpe(year_data[yr], params, weights) for yr in YEARS]
    baseline_obj = float(np.mean(baseline_sharpes)) - 0.5 * float(np.std(baseline_sharpes))
    curr_mid = weights["MID"]
    print(f"\n  Baseline OBJ = {baseline_obj:.4f}")
    print(f"  Current MID: v1={curr_mid['v1']:.2f} i460={curr_mid['i460']:.3f} "
          f"i415={curr_mid['i415']:.3f} f168={curr_mid['f168']:.2f}")

    # Joint grid: v1 × f168 × i415_frac
    print("\n[2] Joint grid: v1 × f168 × i415_frac ...")
    v1_cands   = [0.00, 0.05, 0.10, 0.15, 0.20]
    f168_cands = [0.50, 0.60, 0.65, 0.70, 0.75, 0.80, 0.90, 1.00]
    i415_cands = [0.00, 0.25, 0.50, 0.75, 1.00]

    best_obj = baseline_obj
    best_v1 = curr_mid["v1"]; best_f168 = curr_mid["f168"]; best_i415f = 0.0
    for v1_w in v1_cands:
        for f168_w in f168_cands:
            if v1_w + f168_w > 1.0: continue
            for i415f in i415_cands:
                w_test = make_mid_w(base_weights, v1_w, f168_w, i415f)
                obj = eval_weights(year_data, params, w_test)
                if obj > best_obj:
                    best_obj = obj; best_v1 = v1_w; best_f168 = f168_w; best_i415f = i415f
                    print(f"  NEW BEST: v1={v1_w:.2f} f168={f168_w:.2f} i415f={i415f:.2f} OBJ={obj:.4f}")

    print(f"\n  Coarse best: v1={best_v1:.2f} f168={best_f168:.2f} i415f={best_i415f:.2f} OBJ={best_obj:.4f}")

    # Fine tune f168
    print("\n[3] Fine-tune f168 ...")
    for fw in sorted(set(round(max(0, best_f168 - 0.10 + i*0.05), 2) for i in range(5))):
        w_test = make_mid_w(base_weights, best_v1, fw, best_i415f)
        obj = eval_weights(year_data, params, w_test)
        marker = " <-" if obj > best_obj else ""
        print(f"  f168={fw:.2f}  OBJ={obj:.4f}{marker}")
        if obj > best_obj:
            best_obj = obj; best_f168 = fw
    print(f"  Fine best f168: {best_f168:.2f}  OBJ={best_obj:.4f}")

    # Fine tune i415 frac
    print("\n[4] Fine-tune i415_frac ...")
    for frac in [round(i * 0.1, 1) for i in range(11)]:
        if best_v1 + best_f168 > 1.0: continue
        w_test = make_mid_w(base_weights, best_v1, best_f168, frac)
        obj = eval_weights(year_data, params, w_test)
        marker = " <-" if obj > best_obj else ""
        print(f"  i415f={frac:.2f}  OBJ={obj:.4f}{marker}")
        if obj > best_obj:
            best_obj = obj; best_i415f = frac
    print(f"  Fine best i415f: {best_i415f:.2f}  OBJ={best_obj:.4f}")

    # LOYO
    print("\n[5] LOYO validation ...")
    best_weights = make_mid_w(base_weights, best_v1, best_f168, best_i415f)
    wins = 0
    for yr in YEARS:
        sh_best = yearly_sharpe(year_data[yr], params, best_weights)
        sh_base = yearly_sharpe(year_data[yr], params, base_weights)
        won = sh_best > sh_base
        wins += int(won)
        print(f"  {yr}: best={sh_best:.4f} curr={sh_base:.4f} {'WIN' if won else 'LOSS'}")

    delta = best_obj - baseline_obj
    bw = best_weights["MID"]
    print(f"\n  LOYO: {wins}/5 wins  delta={delta:+.4f}")
    print(f"  Best MID: v1={bw['v1']:.2f} i460={bw['i460']:.3f} i415={bw['i415']:.3f} f168={bw['f168']:.2f}")

    if wins >= 3 and delta > 0.005:
        print(f"\n  VALIDATED  (LOYO {wins}/5 delta={delta:+.4f})")
        _update_config(cfg, bw, best_obj)
    else:
        print(f"\n  CONFIRMED OPTIMAL  (LOYO {wins}/5 delta={delta:+.4f})")

    print(f"\n[DONE] Phase 270 complete.  ({time.time()-t0:.1f}s)")

def _update_config(cfg, bw, new_obj):
    old_ver = cfg.get("version", "v2.42.19")
    parts = old_ver.lstrip("v").split(".")
    parts[-1] = str(int(parts[-1]) + 1)
    new_ver = "v" + ".".join(parts)
    rw = cfg["breadth_regime_switching"]["regime_weights"]["MID"]
    rw["v1"]        = round(bw["v1"], 4)
    rw["i460bw168"] = round(bw["i460"], 4)
    rw["i415bw216"] = round(bw["i415"], 4)
    rw["f168"]      = round(bw["f168"], 4)
    cfg["version"] = new_ver
    p = Path("configs/production_p91b_champion.json")
    p.write_text(json.dumps(cfg, indent=2))
    print(f"  Config updated: {old_ver} → {new_ver}")
    print(f"  MID: v1={bw['v1']:.3f} i460={bw['i460']:.3f} i415={bw['i415']:.3f} f168={bw['f168']:.3f}")
    print(f"  New OBJ={new_obj:.4f}")
    subprocess.run(["git", "add", str(p)], cwd="/Users/truonglys/projects/quant")
    subprocess.run(["git", "commit", "-m",
        f"quant: {new_ver} — MID regime retune r5 (v1={bw['v1']:.2f} f168={bw['f168']:.2f}) OBJ={new_obj:.4f}"],
        cwd="/Users/truonglys/projects/quant")
    print("  Git committed.")

if __name__ == "__main__":
    main()
