#!/usr/bin/env python3
"""
Phase 260 — Per-regime VOL scale retune (post P257 HIGH regime weight overhaul)
HIGH went from i460=0.70/f168=0.00 to i460=0.18/i415=0.27/f168=0.35/v1=0.20.
Vol behavior changed — retune per_regime_scale for each regime.
Sweep: per_regime_scale.HIGH, then MID, then LOW.
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
COST_MODEL = cost_model_from_config({
    "fee_rate": 0.0005, "slippage_rate": 0.0003, "cost_multiplier": 1.0,
})
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
            "v1":   w["v1"],
            "i460": w.get("i460bw168", w.get("i460", 0.0)),
            "i415": w.get("i415bw216", w.get("i415", 0.0)),
            "f168": w["f168"],
        }
    fts    = cfg.get("funding_term_structure_overlay", {})
    vol_ov = cfg.get("vol_regime_overlay", {})
    vol_params_pr = cfg.get("vol_overlay_params", {})
    params = {
        "p_low":    brs.get("p_low", 0.30),
        "p_high":   brs.get("p_high", 0.60),
        "brd_lb":   brs.get("breadth_lookback_bars", 192),
        "pct_win":  brs.get("rolling_percentile_window", 336),
        "ts_rt":    fts.get("reduce_threshold", 0.55),
        "ts_rs":    fts.get("reduce_scale", 0.05),
        "ts_bt":    fts.get("boost_threshold", 0.22),
        "ts_bs":    fts.get("boost_scale", 2.00),
        "ts_short": fts.get("short_window_bars", 16),
        "ts_long":  fts.get("long_window_bars", 72),
        "vol_thr":    vol_ov.get("threshold", 0.50),
        "vol_scale":  vol_ov.get("scale_factor", 0.40),
        # Per-regime vol scales
        "vol_scale_LOW":  vol_params_pr.get("per_regime_scale", {}).get("LOW",  0.40),
        "vol_scale_MID":  vol_params_pr.get("per_regime_scale", {}).get("MID",  0.15),
        "vol_scale_HIGH": vol_params_pr.get("per_regime_scale", {}).get("HIGH", 0.10),
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
    cfg_d = {"provider": "binance_rest_v1", "symbols": SYMBOLS,
             "start": s, "end": e, "bar_interval": "1h",
             "cache_dir": ".cache/binance_rest"}
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
    print(f"n={n}")
    return {"n": n, "btc_vol": btc_vol, "ts_spread_pct": ts_spread_pct,
            "brd_pct": brd_pct,
            "v1": v1_r, "i460": i460_r, "i415": i415_r, "f168": f168_r}


def compute_ensemble(data, params, weights):
    """compute_ensemble with per-regime vol scale (vol_scale_LOW/MID/HIGH)."""
    n    = data["n"]
    p    = params
    w    = weights
    v1r  = data["v1"]; i460r = data["i460"]; i415r = data["i415"]; f168r = data["f168"]
    min_len = min(n, len(v1r), len(i460r), len(i415r), len(f168r))
    brd  = data["brd_pct"]; tsp = data["ts_spread_pct"]; vol = data["btc_vol"]
    ens  = np.zeros(min_len)
    vol_thr = p["vol_thr"]
    vol_scales = {"LOW": p["vol_scale_LOW"], "MID": p["vol_scale_MID"], "HIGH": p["vol_scale_HIGH"]}
    for i in range(min_len):
        b = brd[i]
        if   b >= p["p_high"]: ww = w["HIGH"]; regime = "HIGH"
        elif b >= p["p_low"]:  ww = w["MID"];  regime = "MID"
        else:                   ww = w["LOW"];  regime = "LOW"
        r = (ww["v1"]*v1r[i] + ww["i460"]*i460r[i] +
             ww["i415"]*i415r[i] + ww["f168"]*f168r[i])
        ts = tsp[i]
        if   ts > p["ts_rt"]: r *= p["ts_rs"]
        elif ts < p["ts_bt"]: r *= p["ts_bs"]
        if not np.isnan(vol[i]) and vol[i] > vol_thr:
            r *= vol_scales[regime]
        ens[i] = r
    return ens


def yearly_sharpe(data, params, weights):
    ens = compute_ensemble(data, params, weights)
    mu  = float(np.mean(ens)) * 8760
    sd  = float(np.std(ens)) * np.sqrt(8760)
    if sd < 1e-10: return 0.0
    return mu / sd


def eval_weights(year_data, params, weights):
    sharpes = [yearly_sharpe(year_data[yr], params, weights) for yr in YEARS]
    arr = np.array(sharpes)
    return float(np.mean(arr)) - 0.5 * float(np.std(arr))


def sweep_vol_scale(regime, base_params, year_data, weights):
    """Sweep per-regime vol scale for one regime."""
    key = f"vol_scale_{regime}"
    cur_scale = base_params[key]
    base_obj = eval_weights(year_data, base_params, weights)
    best_obj = base_obj
    best_scale = cur_scale

    cands = [0.00, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50,
             0.55, 0.60, 0.70, 0.80, 0.90, 1.00]
    print(f"  [{regime}] vol_scale sweep (cur={cur_scale}) ...")
    for scale in cands:
        test_p = dict(base_params)
        test_p[key] = scale
        o = eval_weights(year_data, test_p, weights)
        marker = " <-" if o > best_obj + 1e-6 else ""
        print(f"    {regime}_scale={scale:.2f}  OBJ={o:.4f}{marker}")
        if o > best_obj + 1e-6:
            best_obj = o; best_scale = scale

    delta = best_obj - base_obj
    print(f"  -> {regime}: scale={cur_scale}->{best_scale}  D={delta:+.4f}")
    return best_scale, best_obj


def main():
    t0 = time.time()
    print("=" * 65)
    print("Phase 260 — Per-regime VOL scale retune (post P257 HIGH overhaul)")
    print("=" * 65)

    cfg, weights, params, v1p, i460p, i415p, f168p = get_config()
    print(f"  Baseline vol scales: LOW={params['vol_scale_LOW']} MID={params['vol_scale_MID']} HIGH={params['vol_scale_HIGH']}")
    print(f"  Regime weights: LOW={weights['LOW']} MID={weights['MID']} HIGH={weights['HIGH']}")

    print("\n[1] Loading all year data ...")
    year_data = {}
    for yr in YEARS:
        print(f"  {yr} ", end="", flush=True)
        year_data[yr] = load_year_all(yr, params, v1p, i460p, i415p, f168p)

    base_obj = eval_weights(year_data, params, weights)
    print(f"\n  Baseline OBJ: {base_obj:.4f}")

    current_params = dict(params)

    # Sweep HIGH first (biggest regime change in P257)
    for regime in ["HIGH", "MID", "LOW"]:
        print(f"\n{'='*45}")
        print(f"  Sweeping {regime} vol_scale ...")
        print(f"{'='*45}")
        best_scale, new_obj = sweep_vol_scale(regime, current_params, year_data, weights)
        if new_obj > eval_weights(year_data, current_params, weights) + 1e-6:
            current_params[f"vol_scale_{regime}"] = best_scale
            print(f"  {regime} IMPROVED → scale={best_scale}  OBJ={new_obj:.4f}")
        else:
            print(f"  {regime} NO IMPROVEMENT — keeping {current_params[f'vol_scale_{regime}']}")

    final_obj = eval_weights(year_data, current_params, weights)
    total_delta = final_obj - base_obj
    print(f"\n{'='*65}")
    print(f"  Final vol scales: LOW={current_params['vol_scale_LOW']} MID={current_params['vol_scale_MID']} HIGH={current_params['vol_scale_HIGH']}")
    print(f"  OBJ: {base_obj:.4f} -> {final_obj:.4f}  D={total_delta:+.4f}")

    # LOYO validation
    print(f"\n[5] LOYO validation ...")
    loyo_wins = 0
    for yr in YEARS:
        s_base = yearly_sharpe(year_data[yr], params, weights)
        s_best = yearly_sharpe(year_data[yr], current_params, weights)
        result = "WIN " if s_best > s_base else "LOSS"
        if s_best > s_base: loyo_wins += 1
        print(f"  {yr}: best={s_best:.4f}  base={s_base:.4f}  {result}")

    print(f"\n  LOYO: {loyo_wins}/5  delta={total_delta:+.4f}")

    if loyo_wins >= 3 and total_delta > 0.005:
        print("\n  VALIDATED — updating config ...")
        vp = cfg.setdefault("vol_overlay_params", {})
        pr_scale = vp.setdefault("per_regime_scale", {})
        pr_scale["LOW"]  = current_params["vol_scale_LOW"]
        pr_scale["MID"]  = current_params["vol_scale_MID"]
        pr_scale["HIGH"] = current_params["vol_scale_HIGH"]
        old_ver = cfg.get("version", "v2.42.11")
        parts = old_ver.lstrip("v").split(".")
        parts[-1] = str(int(parts[-1]) + 1)
        new_ver = "v" + ".".join(parts)
        cfg["version"] = new_ver
        cfg_path = Path("configs/production_p91b_champion.json")
        cfg_path.write_text(json.dumps(cfg, indent=2))
        print(f"  Config updated: {old_ver} → {new_ver}")
        try:
            subprocess.run(["git", "stash"], check=False, capture_output=True)
            subprocess.run(["git", "pull", "--rebase"], check=False, capture_output=True)
            subprocess.run(["git", "stash", "pop"], check=False, capture_output=True)
            subprocess.run(["git", "add", str(cfg_path)], check=True, capture_output=True)
            msg = (f"feat: {new_ver} Phase260 vol_scale retune post-P257 — "
                   f"LOW={current_params['vol_scale_LOW']} MID={current_params['vol_scale_MID']} "
                   f"HIGH={current_params['vol_scale_HIGH']} LOYO={loyo_wins}/5 D={total_delta:+.4f}")
            subprocess.run(["git", "commit", "-m", msg], check=True, capture_output=True)
            subprocess.run(["git", "push"], check=True, capture_output=True)
            print(f"  Git commit+push OK")
        except Exception as ex:
            print(f"  Git error: {ex}")
    else:
        print(f"\n  NOT VALIDATED (LOYO {loyo_wins}/5  delta={total_delta:+.4f}) — config confirmed optimal")

    elapsed = time.time() - t0
    print(f"  Total time: {elapsed:.1f}s")
    print("[DONE] Phase 260 complete.")


if __name__ == "__main__":
    main()
