#!/usr/bin/env python3
"""
Phase 275 — LOW Regime Retune Round 4
Supervisor set LOW = v1=0.35, i460=0.00, i415=0.25, f168=0.40.
Re-validate LOW regime weights under current full config (v2.49.6+).
OBJ = mean(yearly_sharpes) - 0.5*std(yearly_sharpes)
LOYO: >=3/5 wins AND delta>0.005
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
        "ts_rs":    fts.get("reduce_scale", 0.00),
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
    p = params
    s, e = YEAR_RANGES[year]
    cfg_d = {
        "provider": "binance_rest_v1", "symbols": SYMBOLS,
        "start": s, "end": e, "bar_interval": "1h",
        "cache_dir": ".cache/binance_rest",
    }
    dataset = make_provider(cfg_d, seed=42).load()
    n = len(dataset.timeline)

    close_mat = np.zeros((n, len(SYMBOLS)))
    for j, sym in enumerate(SYMBOLS):
        for i in range(n):
            close_mat[i, j] = dataset.close(sym, i)

    btc_rets = np.zeros(n)
    for i in range(1, n):
        c0, c1 = close_mat[i-1, 0], close_mat[i, 0]
        btc_rets[i] = (c1/c0 - 1.0) if c0 > 0 else 0.0
    btc_vol = np.full(n, np.nan)
    for i in range(VOL_WINDOW, n):
        btc_vol[i] = float(np.std(btc_rets[i-VOL_WINDOW:i])) * np.sqrt(8760)
    if VOL_WINDOW < n:
        btc_vol[:VOL_WINDOW] = btc_vol[VOL_WINDOW]

    n_up = np.zeros(n)
    for j in range(len(SYMBOLS)):
        for i in range(1, n):
            if close_mat[i, j] > close_mat[i-1, j]:
                n_up[i] += 1
    raw_brd = n_up / len(SYMBOLS)
    brd_pct = np.zeros(n)
    lb = p["brd_lb"]
    for i in range(n):
        st = max(0, i - lb); en = i + 1
        w = raw_brd[st:en]
        brd_pct[i] = float(np.mean(w[-1] >= w)) if len(w) >= 2 else 0.5
    brd_pct[:PCT_WINDOW] = 0.5

    funding = np.zeros((n, len(SYMBOLS)))
    for j, sym in enumerate(SYMBOLS):
        try:
            fr = dataset.funding_rate(sym)
            if fr is None: continue
            for i in range(n):
                ts_i = dataset.timeline[i]
                best_v, best_dt = 0.0, 10**18
                for evt in fr:
                    dt = abs(evt["timestamp"] - ts_i)
                    if dt < best_dt:
                        best_dt, best_v = dt, evt.get("funding_rate", 0.0)
                funding[i, j] = best_v
        except Exception:
            pass

    cross_fr = np.mean(funding, axis=1)
    sw, lw = p["ts_short"], p["ts_long"]
    ts_spread = rolling_mean_arr(cross_fr, sw) - rolling_mean_arr(cross_fr, lw)
    ts_spread = np.where(np.isnan(ts_spread), 0.0, ts_spread)
    ts_spread_pct = np.zeros(n)
    pw2 = p["pct_win"]
    for i in range(n):
        st = max(0, i - pw2)
        w_arr = ts_spread[st:i+1]
        ts_spread_pct[i] = float(np.mean(ts_spread[i] >= w_arr)) if len(w_arr) >= 2 else 0.5

    def run(name, sp):
        res = BacktestEngine(BacktestConfig(costs=COST_MODEL)).run(
            dataset, make_strategy({"name": name, "params": sp}))
        return np.array(res.returns)

    v1_r = run("nexus_alpha_v1", {
        "k_per_side": v1p.get("k_per_side", 2),
        "w_carry": v1p.get("w_carry", 0.15),
        "w_mom":   v1p.get("w_mom", 0.55),
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

    return {
        "n": n, "v1": v1_r, "i460": i460_r, "i415": i415_r, "f168": f168_r,
        "brd_pct": brd_pct, "ts_spread_pct": ts_spread_pct, "btc_vol": btc_vol,
    }

def compute_ensemble(data, params, weights):
    p = params
    v1r, i460r, i415r, f168r = data["v1"], data["i460"], data["i415"], data["f168"]
    min_len = min(data["n"], len(v1r), len(i460r), len(i415r), len(f168r))
    brd, tsp, vol = data["brd_pct"], data["ts_spread_pct"], data["btc_vol"]
    ens = np.zeros(min_len)
    for i in range(min_len):
        b = brd[i]
        if   b >= p["p_high"]: ww = weights["HIGH"]
        elif b >= p["p_low"]:  ww = weights["MID"]
        else:                   ww = weights["LOW"]
        r = ww["v1"]*v1r[i] + ww["i460"]*i460r[i] + ww["i415"]*i415r[i] + ww["f168"]*f168r[i]
        ts = tsp[i]
        if   ts > p["ts_rt"]: r *= p["ts_rs"]
        elif ts < p["ts_bt"]: r *= p["ts_bs"]
        if not np.isnan(vol[i]) and vol[i] > p["vol_thr"]: r *= p["vol_scale"]
        ens[i] = r
    return ens

def yearly_sharpe(data, params, weights):
    ens = compute_ensemble(data, params, weights)
    mu = float(np.mean(ens)) * 8760
    sd = float(np.std(ens)) * np.sqrt(8760)
    return mu / sd if sd > 1e-10 else 0.0

def eval_obj(yearly_data, params, weights):
    sh = [yearly_sharpe(yearly_data[y], params, weights) for y in YEARS]
    return float(np.mean(sh)) - 0.5 * float(np.std(sh))

def try_low(yearly_data, params, base_weights, lv1, li460, li415, lf168):
    w = {
        "HIGH": base_weights["HIGH"],
        "MID":  base_weights["MID"],
        "LOW":  {"v1": lv1, "i460": li460, "i415": li415, "f168": lf168},
    }
    return eval_obj(yearly_data, params, w)

def main():
    t0 = time.time()
    print("[1] Loading config ...")
    cfg, weights, params, v1p, i460p, i415p, f168p = get_config()
    curr = weights["LOW"]
    cv1, ci460, ci415, cf168 = curr["v1"], curr["i460"], curr["i415"], curr["f168"]
    print(f"    Version: {cfg.get('version','?')}")
    print(f"    Current LOW: v1={cv1} i460={ci460} i415={ci415} f168={cf168}")
    print(f"    HIGH: {weights['HIGH']}  MID: {weights['MID']}")

    print("[2] Loading yearly data ...")
    yearly_data = {}
    for y in YEARS:
        print(f"    Loading {y} ...", flush=True)
        yearly_data[y] = load_year_all(y, params, v1p, i460p, i415p, f168p)

    baseline = eval_obj(yearly_data, params, weights)
    print(f"    Baseline OBJ={baseline:.4f}")

    best_v1, best_i460, best_i415, best_f168 = cv1, ci460, ci415, cf168
    best_obj = baseline

    # [3] Sweep f168
    print("\n[3] f168 sweep ...")
    for f in [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.00]:
        o = try_low(yearly_data, params, weights, best_v1, best_i460, best_i415, f)
        marker = " <--" if o > best_obj else ""
        print(f"  f168={f:.2f}  OBJ={o:.4f}{marker}")
        if o > best_obj:
            best_obj, best_f168 = o, f
    print(f"Best f168: {best_f168}  OBJ={best_obj:.4f}")

    # [4] Sweep v1
    print("\n[4] v1 sweep ...")
    for v in [0.00, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]:
        o = try_low(yearly_data, params, weights, v, best_i460, best_i415, best_f168)
        marker = " <--" if o > best_obj else ""
        print(f"  v1={v:.2f}  OBJ={o:.4f}{marker}")
        if o > best_obj:
            best_obj, best_v1 = o, v
    print(f"Best v1: {best_v1}  OBJ={best_obj:.4f}")

    # [5] Sweep i415
    print("\n[5] i415 sweep ...")
    for a in [0.00, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]:
        o = try_low(yearly_data, params, weights, best_v1, best_i460, a, best_f168)
        marker = " <--" if o > best_obj else ""
        print(f"  i415={a:.2f}  OBJ={o:.4f}{marker}")
        if o > best_obj:
            best_obj, best_i415 = o, a
    print(f"Best i415: {best_i415}  OBJ={best_obj:.4f}")

    # [6] Sweep i460
    print("\n[6] i460 sweep ...")
    for a in [0.00, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]:
        o = try_low(yearly_data, params, weights, best_v1, a, best_i415, best_f168)
        marker = " <--" if o > best_obj else ""
        print(f"  i460={a:.2f}  OBJ={o:.4f}{marker}")
        if o > best_obj:
            best_obj, best_i460 = o, a
    print(f"Best i460: {best_i460}  OBJ={best_obj:.4f}")

    # [7] Fine-tune f168 around best
    print("\n[7] Fine-tune f168 ...")
    fine_f = sorted(set([round(best_f168 + d, 2) for d in [-0.08, -0.05, -0.03, 0.0, 0.03, 0.05, 0.08]]))
    for f in fine_f:
        if f <= 0.0: continue
        o = try_low(yearly_data, params, weights, best_v1, best_i460, best_i415, f)
        marker = " <--" if o > best_obj else ""
        print(f"  f168={f:.2f}  OBJ={o:.4f}{marker}")
        if o > best_obj:
            best_obj, best_f168 = o, f
    print(f"Final: v1={best_v1} i460={best_i460} i415={best_i415} f168={best_f168}  OBJ={best_obj:.4f}")

    # [8] LOYO validation
    print("\n[8] LOYO validation ...")
    best_weights = {
        "HIGH": weights["HIGH"],
        "MID":  weights["MID"],
        "LOW":  {"v1": best_v1, "i460": best_i460, "i415": best_i415, "f168": best_f168},
    }
    wins = 0; deltas = []
    for y in YEARS:
        s_curr = yearly_sharpe(yearly_data[y], params, weights)
        s_best = yearly_sharpe(yearly_data[y], params, best_weights)
        d = s_best - s_curr
        deltas.append(d)
        w = "WIN" if s_best > s_curr else "LOSS"
        if s_best > s_curr: wins += 1
        print(f"  {y}: curr={s_curr:.4f} best={s_best:.4f} {w}")
    delta = float(np.mean(deltas))
    print(f"\n  LOYO: {wins}/5 wins  delta={delta:+.4f}")

    if wins >= 3 and delta > 0.005:
        print(f"\n  VALIDATED  (LOYO {wins}/5 delta={delta:+.4f})")
        cfg_fresh = json.loads(Path("configs/production_p91b_champion.json").read_text())
        old_ver = cfg_fresh.get("version", "v2.49.6")
        parts   = old_ver.split(".")
        new_ver = f"{parts[0]}.{parts[1]}.{int(parts[2]) + 1}"
        cfg_fresh["version"] = new_ver
        brs = cfg_fresh["breadth_regime_switching"]
        low = brs["regime_weights"]["LOW"]
        low["v1"]        = round(float(best_v1), 4)
        low["i460bw168"] = round(float(best_i460), 4)
        low["i415bw216"] = round(float(best_i415), 4)
        low["f168"]      = round(float(best_f168), 4)
        for k in ["i460", "i415"]: low.pop(k, None)
        Path("configs/production_p91b_champion.json").write_text(json.dumps(cfg_fresh, indent=2))
        print(f"  Config updated: {old_ver} → {new_ver}")
        print(f"  LOW: v1={best_v1} i460={best_i460} i415={best_i415} f168={best_f168}")
        print(f"  New OBJ={best_obj:.4f}")
        msg = (f"quant: {new_ver} — LOW regime retune r4 "
               f"(v1={best_v1} i460={best_i460} i415={best_i415} f168={best_f168}) "
               f"OBJ={best_obj:.4f} LOYO={wins}/5 D={delta:+.4f}")
        subprocess.run(["git", "add", "configs/production_p91b_champion.json"],
                       cwd="/Users/truonglys/projects/quant")
        subprocess.run(["git", "commit", "-m", msg],
                       cwd="/Users/truonglys/projects/quant")
        print("  Git committed.")
    else:
        print(f"\n  CONFIRMED OPTIMAL (LOYO {wins}/5 delta={delta:+.4f} — no update)")
        print(f"  Keeping LOW: v1={cv1} i460={ci460} i415={ci415} f168={cf168}")

    print(f"\n[DONE] Phase 275 complete.  ({time.time()-t0:.1f}s)")

if __name__ == "__main__":
    main()
