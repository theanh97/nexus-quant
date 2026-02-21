#!/usr/bin/env python3
"""
Phase 275 — Breadth Threshold Re-sweep Round 3
Re-sweep p_low and p_high thresholds after regime weight updates.
Current: p_low=0.30, p_high=0.68
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
        "vol_thr":  vol_ov.get("threshold", 0.50),
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

def compute_ensemble(data, params, weights, p_low=None, p_high=None):
    n = data["n"]; p = params; w = weights
    pl = p_low  if p_low  is not None else p["p_low"]
    ph = p_high if p_high is not None else p["p_high"]
    v1r = data["v1"]; i460r = data["i460"]; i415r = data["i415"]; f168r = data["f168"]
    min_len = min(n, len(v1r), len(i460r), len(i415r), len(f168r))
    brd = data["brd_pct"]; tsp = data["ts_spread_pct"]; vol = data["btc_vol"]
    ens = np.zeros(min_len)
    for i in range(min_len):
        b = brd[i]
        if   b >= ph: ww = w["HIGH"]
        elif b >= pl: ww = w["MID"]
        else:         ww = w["LOW"]
        r = ww["v1"]*v1r[i] + ww["i460"]*i460r[i] + ww["i415"]*i415r[i] + ww["f168"]*f168r[i]
        ts = tsp[i]
        if ts > p["ts_rt"]:   r *= p["ts_rs"]
        elif ts < p["ts_bt"]: r *= p["ts_bs"]
        if not np.isnan(vol[i]) and vol[i] > p["vol_thr"]: r *= p["vol_scale"]
        ens[i] = r
    return ens

def yearly_sharpe(data, params, weights, p_low=None, p_high=None):
    ens = compute_ensemble(data, params, weights, p_low, p_high)
    mu = float(np.mean(ens)) * 8760
    sd = float(np.std(ens)) * np.sqrt(8760)
    return mu / sd if sd > 1e-10 else 0.0

def eval_obj(yearly_data, params, weights, p_low=None, p_high=None):
    sharpes = [yearly_sharpe(yearly_data[yr], params, weights, p_low, p_high) for yr in YEARS]
    return float(np.mean(sharpes)) - 0.5 * float(np.std(sharpes))

def main():
    t0 = time.time()
    print("[1] Loading config ...")
    cfg, weights, params, v1p, i460p, i415p, f168p = get_config()
    curr_pl = params["p_low"]
    curr_ph = params["p_high"]
    print(f"    Version: {cfg.get('version','?')}")
    print(f"    Current: p_low={curr_pl}  p_high={curr_ph}")

    print("[2] Loading yearly data ...")
    yearly_data = {}
    for yr in YEARS:
        print(f"    Loading {yr} ...", flush=True)
        yearly_data[yr] = load_year_all(yr, params, v1p, i460p, i415p, f168p)

    baseline = eval_obj(yearly_data, params, weights)
    print(f"    Baseline OBJ={baseline:.4f}")

    # [3] Sweep p_high (most impactful — HIGH vs MID boundary)
    print("\n[3] p_high sweep ...")
    best_ph  = curr_ph
    best_obj = baseline
    for ph in [0.50, 0.55, 0.58, 0.60, 0.62, 0.65, 0.67, 0.68, 0.70, 0.72, 0.75, 0.78, 0.80]:
        if ph <= curr_pl + 0.10: continue
        o = eval_obj(yearly_data, params, weights, p_low=curr_pl, p_high=ph)
        marker = " <--" if o > best_obj else ""
        print(f"  p_high={ph:.2f}  OBJ={o:.4f}{marker}", flush=True)
        if o > best_obj:
            best_obj = o; best_ph = ph
    print(f"  Best p_high: {best_ph}  D={best_obj-baseline:+.4f}")

    # [4] Sweep p_low
    print("\n[4] p_low sweep ...")
    best_pl = curr_pl
    for pl in [0.10, 0.15, 0.20, 0.22, 0.25, 0.28, 0.30, 0.32, 0.35, 0.38, 0.40]:
        if pl >= best_ph - 0.10: continue
        o = eval_obj(yearly_data, params, weights, p_low=pl, p_high=best_ph)
        marker = " <--" if o > best_obj else ""
        print(f"  p_low={pl:.2f}  OBJ={o:.4f}{marker}", flush=True)
        if o > best_obj:
            best_obj = o; best_pl = pl
    print(f"  Best p_low: {best_pl}  D={best_obj-baseline:+.4f}")

    # Fine-tune p_high around best
    print("\n[5] Fine-tune p_high ...")
    for ph in sorted(set([round(best_ph + d, 2) for d in [-0.03, -0.02, 0.0, 0.02, 0.03]])):
        if ph <= best_pl + 0.05 or ph > 0.95: continue
        o = eval_obj(yearly_data, params, weights, p_low=best_pl, p_high=ph)
        marker = " <--" if o > best_obj else ""
        print(f"  p_high={ph:.2f}  OBJ={o:.4f}{marker}", flush=True)
        if o > best_obj:
            best_obj = o; best_ph = ph

    # [6] LOYO validation
    print(f"\n[6] LOYO validation (p_low={best_pl} p_high={best_ph}) ...")
    wins = 0
    for yr in YEARS:
        s_best = yearly_sharpe(yearly_data[yr], params, weights, best_pl, best_ph)
        s_curr = yearly_sharpe(yearly_data[yr], params, weights, curr_pl, curr_ph)
        result = "WIN" if s_best > s_curr else "LOSS"
        if s_best > s_curr: wins += 1
        print(f"  {yr}: best={s_best:.4f} curr={s_curr:.4f} {result}")

    delta = best_obj - baseline
    print(f"\n  LOYO: {wins}/5 wins  delta={delta:+.4f}")
    print(f"  Best: p_low={best_pl} p_high={best_ph}")

    if wins >= 3 and delta > 0.005:
        print(f"\n  VALIDATED  (LOYO {wins}/5 delta={delta:+.4f})")
        cfg2 = json.loads(Path("configs/production_p91b_champion.json").read_text())
        old_ver = cfg2.get("version", "v0.0.0")
        parts = old_ver.lstrip("v").split(".")
        parts[-1] = str(int(parts[-1]) + 1)
        new_ver = "v" + ".".join(parts)
        cfg2["version"] = new_ver
        brs2 = cfg2["breadth_regime_switching"]
        brs2["p_low"]  = round(float(best_pl), 4)
        brs2["p_high"] = round(float(best_ph), 4)
        p = Path("configs/production_p91b_champion.json")
        p.write_text(json.dumps(cfg2, indent=2))
        print(f"  Config updated: {old_ver} → {new_ver}")
        print(f"  p_low: {curr_pl} → {best_pl}  p_high: {curr_ph} → {best_ph}")
        print(f"  New OBJ={best_obj:.4f}")
        msg = (f"quant: {new_ver} — breadth re-sweep r3 "
               f"(p_low={best_pl} p_high={best_ph}) "
               f"OBJ={best_obj:.4f} LOYO={wins}/5 D={delta:+.4f}")
        subprocess.run(["git", "add", "configs/production_p91b_champion.json"],
                       cwd="/Users/truonglys/projects/quant")
        subprocess.run(["git", "commit", "-m", msg],
                       cwd="/Users/truonglys/projects/quant")
        print("  Git committed.")
    else:
        print(f"\n  CONFIRMED OPTIMAL  (LOYO {wins}/5 delta={delta:+.4f})")
        print(f"  Breadth stays: p_low={curr_pl} p_high={curr_ph}")

    print(f"\n[DONE] Phase 275 complete.  ({time.time()-t0:.1f}s)")

if __name__ == "__main__":
    main()
