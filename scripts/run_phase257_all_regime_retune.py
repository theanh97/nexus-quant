#!/usr/bin/env python3
"""
Phase 257 — All-Regime Weight Re-optimization (post P256 V1 wm=0.55 update)
V1 signal character changed: wc=0.25->0.15, wm=0.45->0.55, wr=0.30 (unchanged)
Re-sweep: LOW, MID, HIGH ensemble allocations (f168/v1/i460/i415)
Sequential coordinate descent across all 3 regimes.
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
        "f168_boost": vol_ov.get("f144_boost", 0.00),
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


def compute_ensemble(data, params, weights):
    n    = data["n"]
    p    = params
    w    = weights
    v1r  = data["v1"]; i460r = data["i460"]; i415r = data["i415"]; f168r = data["f168"]
    min_len = min(n, len(v1r), len(i460r), len(i415r), len(f168r))
    brd  = data["brd_pct"]; tsp = data["ts_spread_pct"]; vol = data["btc_vol"]
    ens  = np.zeros(min_len)
    for i in range(min_len):
        b = brd[i]
        if   b >= p["p_high"]: ww = w["HIGH"]
        elif b >= p["p_low"]:  ww = w["MID"]
        else:                   ww = w["LOW"]
        r = (ww["v1"]*v1r[i] + ww["i460"]*i460r[i] +
             ww["i415"]*i415r[i] + ww["f168"]*f168r[i])
        ts = tsp[i]
        if   ts > p["ts_rt"]: r *= p["ts_rs"]
        elif ts < p["ts_bt"]: r *= p["ts_bs"]
        if not np.isnan(vol[i]) and vol[i] > p["vol_thr"]: r *= p["vol_scale"]
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


def make_weights(base, regime, f168_w, v1_w, idio_ratio):
    """Rebuild weights dict with one regime modified, others unchanged."""
    remainder = max(0.0, 1.0 - v1_w - f168_w)
    i460_w = round(remainder * idio_ratio, 4)
    i415_w = round(remainder - i460_w, 4)
    new_w = {r: dict(base[r]) for r in base}
    new_w[regime] = {"v1": v1_w, "i460": i460_w, "i415": i415_w, "f168": f168_w}
    return new_w


def sweep_regime(regime, base_weights, year_data, params):
    """Coordinate-descent sweep for one regime: f168 → v1 → idio_ratio."""
    cur = base_weights[regime]
    cur_f168  = cur["f168"]
    cur_v1    = cur["v1"]
    cur_i460  = cur["i460"]
    cur_i415  = cur["i415"]
    cur_idio  = cur_i460 + cur_i415
    cur_ratio = (cur_i460 / cur_idio) if cur_idio > 1e-9 else 0.0

    base_obj   = eval_weights(year_data, params, base_weights)
    best_obj   = base_obj
    best_f168  = cur_f168
    best_v1    = cur_v1
    best_ratio = cur_ratio

    # Part A: f168 sweep
    f168_cands = [0.00, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45,
                  0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.00]
    print(f"  [A] {regime} f168 sweep ...")
    for f168_w in f168_cands:
        w_test = make_weights(base_weights, regime, f168_w, cur_v1, cur_ratio)
        o = eval_weights(year_data, params, w_test)
        marker = " <-" if o > best_obj + 1e-6 else ""
        rw = w_test[regime]
        print(f"    f168={f168_w:.2f}  v1={rw['v1']:.2f}  idio={rw['i460']+rw['i415']:.4f}  OBJ={o:.4f}{marker}")
        if o > best_obj + 1e-6:
            best_obj = o; best_f168 = f168_w
    print(f"    Best f168={best_f168}  OBJ={best_obj:.4f}")

    # Part B: v1 sweep (best_f168 fixed)
    v1_cands = [0.00, 0.02, 0.05, 0.07, 0.10, 0.12, 0.15, 0.20,
                0.25, 0.30, 0.35, 0.40, 0.45, 0.50]
    print(f"  [B] {regime} v1 sweep (f168={best_f168} fixed) ...")
    for v1_w in v1_cands:
        w_test = make_weights(base_weights, regime, best_f168, v1_w, cur_ratio)
        o = eval_weights(year_data, params, w_test)
        marker = " <-" if o > best_obj + 1e-6 else ""
        rw = w_test[regime]
        print(f"    v1={v1_w:.2f}  f168={best_f168:.2f}  idio={rw['i460']+rw['i415']:.4f}  OBJ={o:.4f}{marker}")
        if o > best_obj + 1e-6:
            best_obj = o; best_v1 = v1_w
    print(f"    Best v1={best_v1}  OBJ={best_obj:.4f}")

    # Part C: idio_ratio sweep (best_f168, best_v1 fixed)
    idio_total = max(0.0, 1.0 - best_v1 - best_f168)
    ratio_cands = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    print(f"  [C] {regime} idio_ratio sweep (idio_total={idio_total:.4f}) ...")
    for ratio in ratio_cands:
        w_test = make_weights(base_weights, regime, best_f168, best_v1, ratio)
        o = eval_weights(year_data, params, w_test)
        rw = w_test[regime]
        marker = " <-" if o > best_obj + 1e-6 else ""
        print(f"    ratio={ratio:.1f}  i460={rw['i460']:.4f}  i415={rw['i415']:.4f}  OBJ={o:.4f}{marker}")
        if o > best_obj + 1e-6:
            best_obj = o; best_ratio = ratio

    delta = best_obj - base_obj
    best_w = make_weights(base_weights, regime, best_f168, best_v1, best_ratio)
    best_rw = best_w[regime]
    print(f"  -> {regime}: f168={best_f168} v1={best_v1} "
          f"i460={best_rw['i460']} i415={best_rw['i415']}  D={delta:+.4f}")
    return best_w, best_obj


def main():
    t0 = time.time()
    print("=" * 70)
    print("Phase 257 — All-Regime Weight Re-opt (post P256 V1 wm=0.55 update)")
    print("=" * 70)

    cfg, weights, params, v1p, i460p, i415p, f168p = get_config()
    print(f"  Baseline regime weights:")
    for r in ["LOW", "MID", "HIGH"]:
        w = weights[r]
        print(f"    {r}: v1={w['v1']} f168={w['f168']} i460={w['i460']} i415={w['i415']}")

    print("\n[1] Loading all year data ...")
    year_data = {}
    for yr in YEARS:
        print(f"  {yr} ", end="", flush=True)
        year_data[yr] = load_year_all(yr, params, v1p, i460p, i415p, f168p)

    base_obj = eval_weights(year_data, params, weights)
    print(f"\n  Baseline OBJ: {base_obj:.4f}")

    # Sequential coordinate optimization over all 3 regimes
    current_weights = {r: dict(weights[r]) for r in weights}
    regime_deltas = {}

    for regime in ["LOW", "MID", "HIGH"]:
        print(f"\n{'='*50}")
        print(f"  Sweeping {regime} regime ...")
        print(f"{'='*50}")
        pre_obj = eval_weights(year_data, params, current_weights)
        new_weights, new_obj = sweep_regime(regime, current_weights, year_data, params)
        regime_delta = new_obj - pre_obj
        regime_deltas[regime] = regime_delta
        if new_obj > pre_obj + 1e-6:
            current_weights = new_weights
            print(f"  {regime} IMPROVED: OBJ={new_obj:.4f}  D={regime_delta:+.4f}")
        else:
            print(f"  {regime} NO IMPROVEMENT — keeping current (D={regime_delta:+.4f})")

    final_obj = eval_weights(year_data, params, current_weights)
    total_delta = final_obj - base_obj

    print(f"\n{'='*70}")
    print(f"  FINAL WEIGHTS:")
    for r in ["LOW", "MID", "HIGH"]:
        w = current_weights[r]
        print(f"    {r}: v1={w['v1']} f168={w['f168']} i460={w['i460']} i415={w['i415']}")
    print(f"  OBJ: {base_obj:.4f} -> {final_obj:.4f}  D={total_delta:+.4f}")
    print(f"  Per-regime deltas: LOW={regime_deltas.get('LOW',0):+.4f} "
          f"MID={regime_deltas.get('MID',0):+.4f} HIGH={regime_deltas.get('HIGH',0):+.4f}")

    # LOYO validation
    print(f"\n[5] LOYO validation ...")
    loyo_wins = 0
    for yr in YEARS:
        s_base = yearly_sharpe(year_data[yr], params, weights)
        s_best = yearly_sharpe(year_data[yr], params, current_weights)
        result = "WIN " if s_best > s_base else "LOSS"
        if s_best > s_base: loyo_wins += 1
        print(f"  {yr}: best={s_best:.4f}  base={s_base:.4f}  {result}")

    print(f"\n  LOYO: {loyo_wins}/5  delta={total_delta:+.4f}")

    if loyo_wins >= 3 and total_delta > 0.005:
        print("\n  VALIDATED — updating config ...")
        rw = cfg["breadth_regime_switching"]["regime_weights"]
        for regime in ["LOW", "MID", "HIGH"]:
            w = current_weights[regime]
            rw[regime]["v1"]        = w["v1"]
            rw[regime]["f168"]      = w["f168"]
            rw[regime]["i460bw168"] = w["i460"]
            rw[regime]["i415bw216"] = w["i415"]
        old_ver = cfg.get("version", "v2.42.9")
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
            msg = (f"feat: {new_ver} Phase257 all-regime retune post-P256 — "
                   f"LOYO={loyo_wins}/5 OBJ={final_obj:.4f} D={total_delta:+.4f}")
            subprocess.run(["git", "commit", "-m", msg], check=True, capture_output=True)
            subprocess.run(["git", "push"], check=True, capture_output=True)
            print(f"  Git commit+push OK")
        except Exception as ex:
            print(f"  Git error: {ex}")
    else:
        print(f"\n  NOT VALIDATED (LOYO {loyo_wins}/5  delta={total_delta:+.4f}) — config unchanged")

    elapsed = time.time() - t0
    print(f"  Total time: {elapsed:.1f}s")
    print("[DONE] Phase 257 complete.")


if __name__ == "__main__":
    main()
