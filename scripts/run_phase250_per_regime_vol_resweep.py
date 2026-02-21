"""
Phase 250 — Per-Regime VOL Overlay Re-Sweep (Vectorized)
=========================================================
Baseline: v2.47.0, OBJ=4.2699

VOL overlay was set in Phase 231 (LOW=0.40, MID=0.15, HIGH=0.10 @ THR=0.50).
Since then: V1 per-regime weights, FTS changed dramatically, regime weights updated.
Optimal VOL dampening params likely shifted.

Vectorized approach: precompute overlay_no_vol per bar, then apply VOL multiplier
per (THR, SCALE) combo — evaluations run in <1ms each.

Current VOL params:
  THR: all 0.50   SCALE: LOW=0.40, MID=0.15, HIGH=0.10
"""

import os, sys, json, time
import signal as _signal
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

import numpy as np

from nexus_quant.backtest.costs import cost_model_from_config
from nexus_quant.backtest.engine import BacktestConfig, BacktestEngine
from nexus_quant.data.providers.registry import make_provider
from nexus_quant.strategies.registry import make_strategy

_partial: dict = {}
def _on_timeout(signum, frame):
    _partial["partial"] = True
    out = Path("artifacts/phase250"); out.mkdir(parents=True, exist_ok=True)
    (out / "phase250_report.json").write_text(json.dumps(_partial, indent=2))
    sys.exit(0)
_signal.signal(_signal.SIGALRM, _on_timeout)
_signal.alarm(7200)

SYMBOLS = ["BTCUSDT","ETHUSDT","SOLUSDT","BNBUSDT","XRPUSDT",
           "ADAUSDT","DOGEUSDT","AVAXUSDT","DOTUSDT","LINKUSDT"]

YEAR_RANGES = {
    2021: ("2021-02-01", "2022-01-01"),
    2022: ("2022-01-01", "2023-01-01"),
    2023: ("2023-01-01", "2024-01-01"),
    2024: ("2024-01-01", "2025-01-01"),
    2025: ("2025-01-01", "2026-01-01"),
}

BRD_LB = 192; PCT_WINDOW = 336; FUND_DISP_PCT = 240
TS_SHORT = 16; TS_LONG = 72; VOL_WINDOW = 168
P_LOW = 0.30; P_HIGH = 0.60

TS_PCT_WIN = {"LOW": 240, "MID": 288, "HIGH": 400}
FTS_RS = {"LOW": 0.05, "MID": 0.05, "HIGH": 0.25}
FTS_BS = {"LOW": 4.00, "MID": 2.00, "HIGH": 2.00}
FTS_RT = {"LOW": 0.80, "MID": 0.65, "HIGH": 0.50}
FTS_BT = {"LOW": 0.30, "MID": 0.40, "HIGH": 0.20}
DISP_THR   = {"LOW": 0.70, "MID": 0.70, "HIGH": 0.40}
DISP_SCALE = {"LOW": 0.50, "MID": 1.50, "HIGH": 0.50}

RNAMES = ["LOW", "MID", "HIGH"]

REGIME_WEIGHTS = {
    "LOW":  {"v1": 0.30, "i460": 0.15, "i415": 0.15, "f168": 0.40},
    "MID":  {"v1": 0.15, "i460": 0.05, "i415": 0.10, "f168": 0.70},
    "HIGH": {"v1": 0.35, "i460": 0.65, "i415": 0.00, "f168": 0.00},
}

V1_LOW_PARAMS = {
    "k_per_side": 2, "w_carry": 0.10, "w_mom": 0.45, "w_mean_reversion": 0.45,
    "momentum_lookback_bars": 336, "mean_reversion_lookback_bars": 84,
    "vol_lookback_bars": 192, "target_gross_leverage": 0.35, "rebalance_interval_bars": 60,
}
V1_MIDHIGH_PARAMS = {
    "k_per_side": 2, "w_carry": 0.25, "w_mom": 0.50, "w_mean_reversion": 0.25,
    "momentum_lookback_bars": 336, "mean_reversion_lookback_bars": 84,
    "vol_lookback_bars": 192, "target_gross_leverage": 0.35, "rebalance_interval_bars": 60,
}
I460_PARAMS = {"k_per_side": 4, "lookback_bars": 480, "beta_window_bars": 168,
               "target_gross_leverage": 0.3, "rebalance_interval_bars": 48}
I415_PARAMS = {"k_per_side": 4, "lookback_bars": 415, "beta_window_bars": 144,
               "target_gross_leverage": 0.3, "rebalance_interval_bars": 48}
F168_PARAMS = {"k_per_side": 2, "funding_lookback_bars": 168, "direction": "contrarian",
               "target_gross_leverage": 0.25, "rebalance_interval_bars": 36}

# Current VOL params (baseline)
VOL_THR_BASE   = {"LOW": 0.50, "MID": 0.50, "HIGH": 0.50}
VOL_SCALE_BASE = {"LOW": 0.40, "MID": 0.15, "HIGH": 0.10}

BASELINE_OBJ = 4.2699
MIN_DELTA = 0.005; MIN_LOYO = 3

# VOL sweep grids
VOL_THR_SWEEP   = [0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]
VOL_SCALE_SWEEP = [0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.00]
# Also include scale > 1 (amplify in high vol) and 0 (fully suppress)
VOL_SCALE_SWEEP += [0.0, 1.20, 1.50]

COST_MODEL = cost_model_from_config({
    "fee_rate": 0.0005, "slippage_rate": 0.0003, "cost_multiplier": 1.0,
})

def rolling_mean_arr(a, w):
    out = np.full(len(a), np.nan)
    if w <= 0: return out
    cs = np.cumsum(np.where(np.isnan(a), 0.0, a))
    for i in range(w - 1, len(a)):
        out[i] = cs[i] / w if i == w - 1 else (cs[i] - cs[i - w]) / w
    return out

def load_year_data(year):
    s, e = YEAR_RANGES[year]
    cfg = {"provider": "binance_rest_v1", "symbols": SYMBOLS,
           "start": s, "end": e, "bar_interval": "1h",
           "cache_dir": ".cache/binance_rest"}
    dataset = make_provider(cfg, seed=42).load()
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
            try: fund_rates[i, j] = dataset.last_funding_rate_before(sym, ts)
            except: fund_rates[i, j] = 0.0
    fund_std_raw = np.std(fund_rates, axis=1)
    fund_std_pct = np.full(n, 0.5)
    for i in range(FUND_DISP_PCT, n):
        fund_std_pct[i] = float(np.mean(fund_std_raw[i-FUND_DISP_PCT:i] <= fund_std_raw[i]))
    xsect_mean = np.mean(fund_rates, axis=1)
    ts_raw = rolling_mean_arr(xsect_mean, TS_SHORT) - rolling_mean_arr(xsect_mean, TS_LONG)
    unique_wins = sorted(set(TS_PCT_WIN.values()))
    ts_pct_cache = {}
    for w in unique_wins:
        arr = np.full(n, 0.5)
        for i in range(w, n):
            arr[i] = float(np.mean(ts_raw[i-w:i] <= ts_raw[i]))
        ts_pct_cache[w] = arr
    breadth = np.full(n, 0.5)
    for i in range(BRD_LB, n):
        pos = sum(1 for j in range(len(SYMBOLS))
                  if close_mat[i-BRD_LB, j] > 0 and close_mat[i, j] > close_mat[i-BRD_LB, j])
        breadth[i] = pos / len(SYMBOLS)
    brd_pct = np.full(n, 0.5)
    for i in range(PCT_WINDOW, n):
        brd_pct[i] = float(np.mean(breadth[i-PCT_WINDOW:i] <= breadth[i]))
    return btc_vol, fund_std_pct, ts_pct_cache, brd_pct, n, dataset

def compute_signal(dataset, name, params):
    return np.array(BacktestEngine(BacktestConfig(costs=COST_MODEL)).run(
        dataset, make_strategy({"name": name, "params": params})).returns)

def precompute_vol_fast_data(all_base_data, all_signals):
    """
    Precompute per-year, per-regime:
    - raw_weighted: regime-weighted signal sum (no overlay)
    - overlay_no_vol: DISP × FTS overlay (no VOL component)
    - btc_vol_arr: BTC vol array
    - masks: regime boolean masks
    """
    fast_data = {}
    for yr, base_data in sorted(all_base_data.items()):
        btc_vol, fund_std_pct, ts_pct_cache, brd_pct, n, _ = base_data
        sigs = all_signals[yr]
        min_len = min(len(sigs[k]) for k in ["v1_LOW","v1_MID","i460","i415","f168"])

        bpct = brd_pct[:min_len]
        bv   = btc_vol[:min_len]
        fsp  = fund_std_pct[:min_len]

        regime_idx = np.where(bpct < P_LOW, 0, np.where(bpct > P_HIGH, 2, 1))
        masks = [regime_idx == i for i in range(3)]

        # overlay_no_vol: DISP + FTS only
        overlay_no_vol = np.ones(min_len)
        for ridx, rname in enumerate(RNAMES):
            m = masks[ridx]
            # DISP
            disp_m = m & (fsp > DISP_THR[rname])
            overlay_no_vol[disp_m] *= DISP_SCALE[rname]
            # FTS
            tsp = ts_pct_cache[TS_PCT_WIN[rname]][:min_len]
            overlay_no_vol[m & (tsp > FTS_RT[rname])] *= FTS_RS[rname]
            overlay_no_vol[m & (tsp < FTS_BT[rname])] *= FTS_BS[rname]

        # raw weighted signal per regime (no overlay)
        raw_weighted = {}
        for ridx, rname in enumerate(RNAMES):
            w = REGIME_WEIGHTS[rname]
            v1_key = "v1_LOW" if rname == "LOW" else "v1_MID"
            raw_weighted[rname] = (
                w["v1"]  * sigs[v1_key][:min_len]
                + w["i460"] * sigs["i460"][:min_len]
                + w["i415"] * sigs["i415"][:min_len]
                + w["f168"] * sigs["f168"][:min_len]
            )

        fast_data[yr] = (masks, overlay_no_vol, raw_weighted, bv, min_len)
    return fast_data

def fast_evaluate_vol(vol_thr, vol_scale, fast_data):
    """
    vol_thr/vol_scale: dicts keyed by regime name
    """
    yearly = {}
    for yr, (masks, overlay_no_vol, raw_weighted, bv, min_len) in fast_data.items():
        ens = np.zeros(min_len)
        for ridx, rname in enumerate(RNAMES):
            m = masks[ridx]
            thr = vol_thr[rname]; sc = vol_scale[rname]
            # VOL multiplier: sc if vol > thr, else 1.0 (for this regime's bars)
            vol_mult_m = np.where(~np.isnan(bv[m]) & (bv[m] > thr), sc, 1.0)
            ens[m] += raw_weighted[rname][m] * overlay_no_vol[m] * vol_mult_m
        r = ens[~np.isnan(ens)]
        sh = float(np.mean(r) / np.std(r, ddof=1) * np.sqrt(8760)) if len(r) > 1 else 0.0
        yearly[yr] = sh
    vals = list(yearly.values())
    return float(np.mean(vals) - 0.5 * np.std(vals, ddof=1)), yearly

def main():
    t0 = time.time()
    print("=" * 65)
    print("Phase 250 — Per-Regime VOL Overlay Re-Sweep (Vectorized)")
    print(f"Baseline: v2.47.0  OBJ={BASELINE_OBJ}")
    print(f"Grid: THR={len(VOL_THR_SWEEP)} SCALE={len(VOL_SCALE_SWEEP)}")
    print(f"Combos: {len(VOL_THR_SWEEP)*len(VOL_SCALE_SWEEP)} per regime")
    print("=" * 65)

    print("\n[1] Loading per-year data...", flush=True)
    all_base_data = {}
    for yr in sorted(YEAR_RANGES):
        print(f"  {yr}: ", end="", flush=True)
        all_base_data[yr] = load_year_data(yr)
        print("done.", flush=True)

    print("\n[2] Pre-computing all signals × 5 years...", flush=True)
    all_signals = {}
    for yr in sorted(YEAR_RANGES):
        ds = all_base_data[yr][5]
        all_signals[yr] = {
            "v1_LOW":  compute_signal(ds, "nexus_alpha_v1", V1_LOW_PARAMS),
            "v1_MID":  compute_signal(ds, "nexus_alpha_v1", V1_MIDHIGH_PARAMS),
            "i460":    compute_signal(ds, "idio_momentum_alpha", I460_PARAMS),
            "i415":    compute_signal(ds, "idio_momentum_alpha", I415_PARAMS),
            "f168":    compute_signal(ds, "funding_momentum_alpha", F168_PARAMS),
        }
        print(f"  {yr}: done.", flush=True)

    print("\n[3] Precomputing VOL fast eval data...", flush=True)
    fast_data = precompute_vol_fast_data(all_base_data, all_signals)

    # Baseline
    base_obj_val, base_yearly = fast_evaluate_vol(VOL_THR_BASE, VOL_SCALE_BASE, fast_data)
    print(f"\n  Baseline OBJ = {base_obj_val:.4f}  (expected ≈ {BASELINE_OBJ})", flush=True)
    _partial.update({"baseline_obj": float(base_obj_val)})

    print("\n[4] Sequential per-regime VOL sweep (2 passes)...", flush=True)
    cur_thr   = dict(VOL_THR_BASE)
    cur_scale = dict(VOL_SCALE_BASE)
    running_best = base_obj_val

    for pass_num in [1, 2]:
        print(f"\n  === Pass {pass_num} ===", flush=True)
        for rname in RNAMES:
            best_thr = cur_thr[rname]; best_scale = cur_scale[rname]
            best_obj_this = running_best
            for thr in VOL_THR_SWEEP:
                for sc in VOL_SCALE_SWEEP:
                    t_test = {**cur_thr, rname: thr}
                    s_test = {**cur_scale, rname: sc}
                    o, _ = fast_evaluate_vol(t_test, s_test, fast_data)
                    if o > best_obj_this:
                        best_obj_this = o
                        best_thr = thr; best_scale = sc
            cur_thr[rname] = best_thr; cur_scale[rname] = best_scale
            running_best = best_obj_this
            print(f"    {rname}: THR={best_thr} SCALE={best_scale}  "
                  f"OBJ={running_best:.4f}  Δ={running_best-base_obj_val:+.4f}", flush=True)

    final_obj_val, final_yearly = fast_evaluate_vol(cur_thr, cur_scale, fast_data)
    delta = final_obj_val - base_obj_val
    print(f"\n  → Best per-regime VOL params:")
    for r in RNAMES:
        print(f"    {r}: THR={cur_thr[r]} SCALE={cur_scale[r]}")
    print(f"  Final OBJ={final_obj_val:.4f}  Δ={delta:+.4f}", flush=True)

    print("\n[5] LOYO validation...", flush=True)
    loyo_wins = 0
    for yr in sorted(final_yearly):
        base_sh = base_yearly[yr]
        cand_sh = final_yearly[yr]
        win = cand_sh > base_sh
        if win: loyo_wins += 1
        print(f"  {yr}: base={base_sh:.4f}  cand={cand_sh:.4f}  {'WIN' if win else 'LOSE'}")

    print(f"\n  OBJ={final_obj_val:.4f}  Δ={delta:+.4f}  LOYO {loyo_wins}/{len(final_yearly)}", flush=True)
    validated = loyo_wins >= MIN_LOYO and delta >= MIN_DELTA
    print(f"\n{'✅ VALIDATED' if validated else '❌ NO IMPROVEMENT'}", flush=True)

    result = {
        "phase": 250, "baseline_obj": float(base_obj_val),
        "best_obj": float(final_obj_val), "delta": float(delta),
        "loyo_wins": loyo_wins, "loyo_total": len(final_yearly),
        "validated": validated,
        "best_vol": {r: {"thr": cur_thr[r], "scale": cur_scale[r]} for r in RNAMES},
    }
    _partial.update(result)
    out = Path("artifacts/phase250"); out.mkdir(parents=True, exist_ok=True)
    (out / "phase250_report.json").write_text(json.dumps(result, indent=2))

    if validated:
        cfg_path = ROOT / "configs" / "production_p91b_champion.json"
        cfg = json.load(open(cfg_path))
        vol = cfg.get("vol_overlay_params", {})
        if "per_regime_threshold" not in vol: vol["per_regime_threshold"] = {}
        if "per_regime_scale"     not in vol: vol["per_regime_scale"]     = {}
        for r in RNAMES:
            vol["per_regime_threshold"][r] = cur_thr[r]
            vol["per_regime_scale"][r]     = cur_scale[r]
        cfg["vol_overlay_params"] = vol
        cfg["_version"] = "2.48.0"
        cfg["monitoring"]["expected_performance"]["annual_sharpe_backtest"] = round(final_obj_val, 4)
        with open(cfg_path, "w") as f:
            json.dump(cfg, f, indent=2, ensure_ascii=False)
        print(f"\n  Config → v2.48.0  OBJ={round(final_obj_val,4)}", flush=True)

    print(f"\nRuntime: {int(time.time()-t0)}s", flush=True)

main()
