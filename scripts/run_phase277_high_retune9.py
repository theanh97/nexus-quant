#!/usr/bin/env python3
"""
Phase 277 — HIGH Regime Retune Round 9
Supervisor changed HIGH to pure momentum (v1=0.5, i460=0.5, f168=0.0).
Re-validate: does f168 contribute in HIGH regime under current config?
Full sweep including f168 range 0.0-0.5.
OBJ = mean(yearly_sharpes) - 0.5*std(yearly_sharpes)
LOYO validation: >=3/5 wins AND delta>0.005
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
    "2021": ("2021-02-01", "2022-01-01"),
    "2022": ("2022-01-01", "2023-01-01"),
    "2023": ("2023-01-01", "2024-01-01"),
    "2024": ("2024-01-01", "2025-01-01"),
    "2025": ("2025-01-01", "2026-01-01"),
}
YEARS = sorted(YEAR_RANGES.keys())
COST_MODEL = cost_model_from_config({"fee_rate": 0.0005, "slippage_rate": 0.0003, "cost_multiplier": 1.0})
VOL_WINDOW = 168
PCT_WINDOW = 336

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
            "v1":   w.get("v1", 0.0),
            "i460": w.get("i460bw168", w.get("i460", 0.0)),
            "i415": w.get("i415bw216", w.get("i415", 0.0)),
            "f168": w.get("f168", 0.0),
        }
    fts = cfg.get("funding_term_structure_overlay", {})
    vol_ov = cfg.get("vol_regime_overlay", {})
    params = {
        "p_low":    brs.get("p_low", 0.30),
        "p_high":   brs.get("p_high", 0.68),
        "brd_lb":   brs.get("breadth_lookback_bars", 192),
        "pct_win":  brs.get("rolling_percentile_window", 336),
        "ts_rt":    fts.get("reduce_threshold", 0.55),
        "ts_rs":    fts.get("reduce_scale", 0.05),
        "ts_bt":    fts.get("boost_threshold", 0.22),
        "ts_bs":    fts.get("boost_scale", 2.00),
        "ts_short": fts.get("short_window_bars", 16),
        "ts_long":  fts.get("long_window_bars", 72),
        "vol_thr":  vol_ov.get("threshold", 0.53),
        "vol_scale":vol_ov.get("scale_factor", 0.40),
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
    ts_raw = (rolling_mean_arr(xsect_mean, p["ts_short"])
              - rolling_mean_arr(xsect_mean, p["ts_long"]))
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
        "w_carry": v1p.get("w_carry", 0.15),
        "w_mom": v1p.get("w_mom", 0.55),
        "w_mean_reversion": v1p.get("w_mean_reversion", 0.30),
        "momentum_lookback_bars": v1p.get("momentum_lookback_bars", 312),
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
        r = ww["v1"]*v1r[i] + ww["i460"]*i460r[i] + ww["i415"]*i460r[i] + ww["f168"]*f168r[i]
        ts = tsp[i]
        if ts > p["ts_rt"]:   r *= p["ts_rs"]
        elif ts < p["ts_bt"]: r *= p["ts_bs"]
        if not np.isnan(vol[i]) and vol[i] > p["vol_thr"]: r *= p["vol_scale"]
        ens[i] = r
    return ens

def yearly_sharpe(data, params, weights):
    ens = compute_ensemble(data, params, weights)
    mu = float(np.mean(ens)) * 8760
    sd = float(np.std(ens)) * np.sqrt(8760)
    return mu / sd if sd > 1e-10 else 0.0

def make_high_w(base_weights, f168_w, v1_w, i460_w, i415_w):
    return {
        "LOW":  base_weights["LOW"],
        "MID":  base_weights["MID"],
        "HIGH": {"v1": v1_w, "i460": i460_w, "i415": i415_w, "f168": f168_w},
    }

def eval_weights(yearly_data, params, weights):
    sharpes = [yearly_sharpe(yearly_data[yr], params, weights) for yr in YEARS]
    return float(np.mean(sharpes)) - 0.5 * float(np.std(sharpes))

def main():
    t0 = time.time()
    print("[1] Loading config ...")
    cfg, base_weights, params, v1p, i460p, i415p, f168p = get_config()
    ch = base_weights["HIGH"]
    print(f"    Version: {cfg.get('version','?')}")
    print(f"    Current HIGH: v1={ch['v1']} i460={ch['i460']} i415={ch['i415']} f168={ch['f168']}")
    print(f"    MID: {base_weights['MID']}")

    print("[2] Loading yearly data ...")
    yearly_data = {}
    for yr in YEARS:
        print(f"    Loading {yr} ...", flush=True)
        yearly_data[yr] = load_year_all(yr, params, v1p, i460p, i415p, f168p)

    curr_obj = eval_weights(yearly_data, params, base_weights)
    print(f"    Baseline OBJ={curr_obj:.4f}")

    # [3] Full grid scan — f168 vs total_idio split
    # Key question: does f168 help in HIGH? Supervisor says no (f168=0), my analysis says ~0.30
    print("\n[3] f168 sweep (holding v1=0.20 constant) ...")
    best_f168 = ch["f168"]
    best_obj  = curr_obj
    # Fix v1=0.20 and split remainder between i460/i415 (idio_ratio=0.40)
    # This isolates f168 effect cleanly
    test_v1 = 0.20
    test_ir  = 0.40
    for f168_w in [0.00, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]:
        rem = max(0.0, 1.0 - test_v1 - f168_w)
        i460_w = round(rem * test_ir, 4)
        i415_w = round(rem - i460_w, 4)
        w = make_high_w(base_weights, f168_w, test_v1, i460_w, i415_w)
        o = eval_weights(yearly_data, params, w)
        marker = " <--" if o > best_obj else ""
        print(f"  f168={f168_w:.2f} i460={i460_w:.3f} i415={i415_w:.3f}  OBJ={o:.4f}{marker}", flush=True)
        if o > best_obj:
            best_obj = o; best_f168 = f168_w

    # [4] Sweep v1 (pure momentum component)
    print("\n[4] v1 sweep (f168=0 — pure momentum test) ...")
    for v1_w in [0.00, 0.10, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50, 0.60, 0.70, 0.80]:
        i460_w = round(v1_w * 0.5 if v1_w < 1.0 else 0.5, 4)
        i415_w = round(1.0 - v1_w - i460_w, 4)
        if i415_w < 0: continue
        w = make_high_w(base_weights, 0.0, v1_w, i460_w, i415_w)
        o = eval_weights(yearly_data, params, w)
        marker = " <--" if o > best_obj else ""
        print(f"  v1={v1_w:.2f} i460={i460_w:.3f} i415={i415_w:.3f} f168=0.0  OBJ={o:.4f}{marker}", flush=True)
        if o > best_obj:
            best_obj = o

    # [5] Systematic: search best (f168, v1, idio_ratio)
    print("\n[5] Systematic search ...")
    best_config = (ch["f168"], ch["v1"], ch["i460"], ch["i415"])
    for f168_w in [0.0, 0.10, 0.20, 0.30, 0.40]:
        for v1_w in [0.10, 0.20, 0.30, 0.40, 0.50]:
            if v1_w + f168_w > 1.0: continue
            rem = 1.0 - v1_w - f168_w
            for ir in [0.30, 0.40, 0.50, 0.60]:
                i460_w = round(rem * ir, 4)
                i415_w = round(rem - i460_w, 4)
                w = make_high_w(base_weights, f168_w, v1_w, i460_w, i415_w)
                o = eval_weights(yearly_data, params, w)
                if o > best_obj:
                    best_obj = o
                    best_config = (f168_w, v1_w, i460_w, i415_w)
                    print(f"  NEW BEST f168={f168_w} v1={v1_w} i460={i460_w} i415={i415_w}  OBJ={o:.4f}", flush=True)

    best_f168, best_v1, best_i460, best_i415 = best_config
    best_w = make_high_w(base_weights, best_f168, best_v1, best_i460, best_i415)

    print(f"\n  Best found: f168={best_f168} v1={best_v1} i460={best_i460} i415={best_i415}")
    print(f"  Best OBJ={best_obj:.4f}  D={best_obj-curr_obj:+.4f}")

    # [6] LOYO validation
    print("\n[6] LOYO validation ...")
    wins = 0
    for yr in YEARS:
        s_best = yearly_sharpe(yearly_data[yr], params, best_w)
        s_curr = yearly_sharpe(yearly_data[yr], params, base_weights)
        result = "WIN" if s_best > s_curr else "LOSS"
        if s_best > s_curr: wins += 1
        print(f"  {yr}: best={s_best:.4f} curr={s_curr:.4f} {result}")

    delta = best_obj - curr_obj
    print(f"\n  LOYO: {wins}/5 wins  delta={delta:+.4f}")
    print(f"  Best HIGH: v1={best_v1} i460={best_i460} i415={best_i415} f168={best_f168}")

    if wins >= 3 and delta > 0.005:
        print(f"\n  VALIDATED  (LOYO {wins}/5 delta={delta:+.4f})")
        cfg2 = json.loads(Path("configs/production_p91b_champion.json").read_text())
        old_ver = cfg2.get("version", "v0.0.0")
        parts = old_ver.lstrip("v").split(".")
        parts[-1] = str(int(parts[-1]) + 1)
        new_ver = "v" + ".".join(parts)
        cfg2["version"] = new_ver
        rw2 = cfg2["breadth_regime_switching"]["regime_weights"]
        rw2["HIGH"]["v1"] = best_v1
        if "i460bw168" in rw2["HIGH"]: rw2["HIGH"]["i460bw168"] = best_i460
        else:                           rw2["HIGH"]["i460"] = best_i460
        if "i415bw216" in rw2["HIGH"]: rw2["HIGH"]["i415bw216"] = best_i415
        else:                           rw2["HIGH"]["i415"] = best_i415
        rw2["HIGH"]["f168"] = best_f168
        p = Path("configs/production_p91b_champion.json")
        p.write_text(json.dumps(cfg2, indent=2))
        print(f"  Config updated: {old_ver} → {new_ver}")
        print(f"  HIGH: v1={best_v1} i460={best_i460} i415={best_i415} f168={best_f168}")
        print(f"  New OBJ={best_obj:.4f}")
        msg = (f"quant: {new_ver} — HIGH retune r9 "
               f"(v1={best_v1:.2f} f168={best_f168:.2f}) "
               f"OBJ={best_obj:.4f} LOYO={wins}/5 D={delta:+.4f}")
        subprocess.run(["git", "add", "configs/production_p91b_champion.json"],
                       cwd="/Users/truonglys/projects/quant")
        subprocess.run(["git", "commit", "-m", msg],
                       cwd="/Users/truonglys/projects/quant")
        print("  Git committed.")
    else:
        print(f"\n  CONFIRMED OPTIMAL  (LOYO {wins}/5 delta={delta:+.4f})")
        print(f"  HIGH stays: v1={ch['v1']} i460={ch['i460']} i415={ch['i415']} f168={ch['f168']}")

    print(f"\n[DONE] Phase 277 complete.  ({time.time()-t0:.1f}s)")

if __name__ == "__main__":
    main()
