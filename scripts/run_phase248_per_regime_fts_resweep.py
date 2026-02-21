"""
Phase 248 — Per-Regime FTS RS/BS/RT/BT Re-Sweep (Vectorized)
=============================================================
Baseline: v2.44.0, OBJ=4.0145

FTS overlay params were set in Phases 229-230, before:
  - Per-regime V1 internal weights (P244)
  - Regime weight re-opt (P245)
  - Per-regime ts_pct_win (P238)
Many structural changes → optimal FTS params may have shifted.

Vectorized approach: precompute signal+overlay_no_fts arrays, then
apply FTS factor per combo in <1ms → 1260 combos × 6 sweeps ≈ 8s.

Current FTS params (v2.38.0):
  LOW:  RS=0.50, BS=3.00, RT=0.80, BT=0.30
  MID:  RS=0.20, BS=3.00, RT=0.65, BT=0.25
  HIGH: RS=0.40, BS=2.00, RT=0.55, BT=0.25
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
    out = Path("artifacts/phase248"); out.mkdir(parents=True, exist_ok=True)
    (out / "phase248_report.json").write_text(json.dumps(_partial, indent=2))
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
VOL_THR   = {"LOW": 0.50, "MID": 0.50, "HIGH": 0.50}
VOL_SCALE = {"LOW": 0.40, "MID": 0.15, "HIGH": 0.10}
DISP_THR   = {"LOW": 0.70, "MID": 0.70, "HIGH": 0.40}
DISP_SCALE = {"LOW": 0.50, "MID": 1.50, "HIGH": 0.50}

RNAMES = ["LOW", "MID", "HIGH"]

REGIME_WEIGHTS = {
    "LOW":  {"v1": 0.35, "i460": 0.10, "i415": 0.15, "f168": 0.40},
    "MID":  {"v1": 0.15, "i460": 0.00, "i415": 0.20, "f168": 0.65},
    "HIGH": {"v1": 0.30, "i460": 0.70, "i415": 0.00, "f168": 0.00},
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

# Current FTS params (baseline)
FTS_RS_BASE = {"LOW": 0.50, "MID": 0.20, "HIGH": 0.40}
FTS_BS_BASE = {"LOW": 3.00, "MID": 3.00, "HIGH": 2.00}
FTS_RT_BASE = {"LOW": 0.80, "MID": 0.65, "HIGH": 0.55}
FTS_BT_BASE = {"LOW": 0.30, "MID": 0.25, "HIGH": 0.25}

BASELINE_OBJ = 4.0145
MIN_DELTA = 0.005; MIN_LOYO = 3

# FTS sweep grids
FTS_RS_SWEEP = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.60, 0.70]
FTS_BS_SWEEP = [1.00, 1.50, 2.00, 2.50, 3.00, 3.50, 4.00, 5.00]
FTS_RT_SWEEP = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]
FTS_BT_SWEEP = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40]

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

def precompute_fts_fast_data(all_base_data, all_signals):
    """
    Precompute per-year, per-regime:
    - raw_weighted[yr][r]: weighted signal array (no overlay applied), shape (min_len,)
      (full array, non-masked — we'll apply mask during evaluation)
    - overlay_no_fts[yr]: VOL * DISP overlay without FTS, shape (min_len,)
    - ts_by_regime[yr][r]: ts_spread_pct for regime r's window
    - masks[yr][r]: boolean mask for regime r
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

        # overlay_no_fts: VOL and DISP only
        overlay_no_fts = np.ones(min_len)
        for ridx, rname in enumerate(RNAMES):
            m = masks[ridx]
            vol_m = m & ~np.isnan(bv) & (bv > VOL_THR[rname])
            overlay_no_fts[vol_m] *= VOL_SCALE[rname]
            disp_m = m & (fsp > DISP_THR[rname])
            overlay_no_fts[disp_m] *= DISP_SCALE[rname]

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

        # ts percentile per regime
        ts_by_regime = {rname: ts_pct_cache[TS_PCT_WIN[rname]][:min_len] for rname in RNAMES}

        fast_data[yr] = (masks, overlay_no_fts, raw_weighted, ts_by_regime, min_len)
    return fast_data

def fast_evaluate_fts(fts_rs, fts_bs, fts_rt, fts_bt, fast_data):
    """
    fts_rs/bs/rt/bt: dicts keyed by regime name
    Returns (obj, yearly_sharpes)
    """
    yearly = {}
    for yr, (masks, overlay_no_fts, raw_weighted, ts_by_regime, min_len) in fast_data.items():
        ens = np.zeros(min_len)
        for ridx, rname in enumerate(RNAMES):
            m = masks[ridx]
            ts = ts_by_regime[rname]
            rs = fts_rs[rname]; bs = fts_bs[rname]
            rt = fts_rt[rname]; bt = fts_bt[rname]
            fts_mult = np.where(ts > rt, rs, np.where(ts < bt, bs, 1.0))
            ens[m] += raw_weighted[rname][m] * overlay_no_fts[m] * fts_mult[m]
        r = ens[~np.isnan(ens)]
        sh = float(np.mean(r) / np.std(r, ddof=1) * np.sqrt(8760)) if len(r) > 1 else 0.0
        yearly[yr] = sh
    vals = list(yearly.values())
    return float(np.mean(vals) - 0.5 * np.std(vals, ddof=1)), yearly

def sweep_regime_fts(target_regime, cur_rs, cur_bs, cur_rt, cur_bt,
                     fast_data, starting_obj):
    """4D sweep of (RS, BS, RT, BT) for one regime. Returns best params and obj."""
    best_obj = starting_obj
    best = (cur_rs[target_regime], cur_bs[target_regime],
            cur_rt[target_regime], cur_bt[target_regime])
    for rs in FTS_RS_SWEEP:
        for bs in FTS_BS_SWEEP:
            for rt in FTS_RT_SWEEP:
                for bt in FTS_BT_SWEEP:
                    if rt <= bt: continue  # need RT > BT
                    rs_t = {**cur_rs, target_regime: rs}
                    bs_t = {**cur_bs, target_regime: bs}
                    rt_t = {**cur_rt, target_regime: rt}
                    bt_t = {**cur_bt, target_regime: bt}
                    o, _ = fast_evaluate_fts(rs_t, bs_t, rt_t, bt_t, fast_data)
                    if o > best_obj:
                        best_obj = o
                        best = (rs, bs, rt, bt)
    return best, best_obj

def main():
    t0 = time.time()
    print("=" * 65)
    print("Phase 248 — Per-Regime FTS RS/BS/RT/BT Re-Sweep (Vectorized)")
    print(f"Baseline: v2.44.0  OBJ={BASELINE_OBJ}")
    print(f"Grid: RS={len(FTS_RS_SWEEP)} BS={len(FTS_BS_SWEEP)} RT={len(FTS_RT_SWEEP)} BT={len(FTS_BT_SWEEP)}")
    n_combos = sum(1 for rs in FTS_RS_SWEEP for bs in FTS_BS_SWEEP
                   for rt in FTS_RT_SWEEP for bt in FTS_BT_SWEEP if rt > bt)
    print(f"Valid combos: {n_combos} per regime (RT>BT constraint)")
    print("=" * 65)

    print("\n[1] Loading per-year data...", flush=True)
    all_base_data = {}
    for yr in sorted(YEAR_RANGES):
        print(f"  {yr}: ", end="", flush=True)
        all_base_data[yr] = load_year_data(yr)
        print("done.", flush=True)

    print("\n[2] Pre-computing all signals (V1×2, I460, I415, F168) × 5 years...", flush=True)
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

    print("\n[3] Precomputing FTS fast eval data...", flush=True)
    fast_data = precompute_fts_fast_data(all_base_data, all_signals)
    print("  done.", flush=True)

    # Baseline check
    base_obj_val, base_yearly = fast_evaluate_fts(
        FTS_RS_BASE, FTS_BS_BASE, FTS_RT_BASE, FTS_BT_BASE, fast_data)
    print(f"\n  Baseline OBJ = {base_obj_val:.4f}  (expected ≈ {BASELINE_OBJ})", flush=True)
    _partial.update({"baseline_obj": float(base_obj_val)})

    print("\n[4] Sequential per-regime FTS sweep (2 passes)...", flush=True)
    cur_rs = dict(FTS_RS_BASE); cur_bs = dict(FTS_BS_BASE)
    cur_rt = dict(FTS_RT_BASE); cur_bt = dict(FTS_BT_BASE)
    running_best = base_obj_val

    for pass_num in [1, 2]:
        print(f"\n  === Pass {pass_num} ===", flush=True)
        for rname in RNAMES:
            ts = time.time()
            (rs, bs, rt, bt), running_best = sweep_regime_fts(
                rname, cur_rs, cur_bs, cur_rt, cur_bt, fast_data, running_best)
            cur_rs[rname] = rs; cur_bs[rname] = bs
            cur_rt[rname] = rt; cur_bt[rname] = bt
            elapsed = int(time.time() - ts)
            print(f"    {rname}: RS={rs} BS={bs} RT={rt} BT={bt}  "
                  f"OBJ={running_best:.4f}  Δ={running_best-base_obj_val:+.4f}  [{elapsed}s]",
                  flush=True)

    final_obj_val, final_yearly = fast_evaluate_fts(cur_rs, cur_bs, cur_rt, cur_bt, fast_data)
    delta = final_obj_val - base_obj_val
    print(f"\n  → Best per-regime FTS params:")
    for r in RNAMES:
        print(f"    {r}: RS={cur_rs[r]} BS={cur_bs[r]} RT={cur_rt[r]} BT={cur_bt[r]}")
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
        "phase": 248, "baseline_obj": float(base_obj_val),
        "best_obj": float(final_obj_val), "delta": float(delta),
        "loyo_wins": loyo_wins, "loyo_total": len(final_yearly),
        "validated": validated,
        "best_fts": {r: {"RS": cur_rs[r], "BS": cur_bs[r], "RT": cur_rt[r], "BT": cur_bt[r]}
                     for r in RNAMES},
    }
    _partial.update(result)
    out = Path("artifacts/phase248"); out.mkdir(parents=True, exist_ok=True)
    (out / "phase248_report.json").write_text(json.dumps(result, indent=2))

    if validated:
        cfg_path = ROOT / "configs" / "production_p91b_champion.json"
        cfg = json.load(open(cfg_path))
        fts = cfg.get("fts_overlay_params", {})
        for rname in RNAMES:
            if "per_regime_rs" not in fts: fts["per_regime_rs"] = {}
            if "per_regime_bs" not in fts: fts["per_regime_bs"] = {}
            if "per_regime_rt" not in fts: fts["per_regime_rt"] = {}
            if "per_regime_bt" not in fts: fts["per_regime_bt"] = {}
            fts["per_regime_rs"][rname] = cur_rs[rname]
            fts["per_regime_bs"][rname] = cur_bs[rname]
            fts["per_regime_rt"][rname] = cur_rt[rname]
            fts["per_regime_bt"][rname] = cur_bt[rname]
        cfg["fts_overlay_params"] = fts
        cur_ver = cfg.get("_version", "2.44.0")
        parts = cur_ver.split(".")
        new_ver = f"{parts[0]}.{parts[1]}.{int(parts[2])+1}"
        cfg["_version"] = new_ver
        cfg["monitoring"]["expected_performance"]["annual_sharpe_backtest"] = round(final_obj_val, 4)
        with open(cfg_path, "w") as f:
            json.dump(cfg, f, indent=2, ensure_ascii=False)
        print(f"\n  Config → {new_ver}  OBJ={round(final_obj_val,4)}", flush=True)

    print(f"\nRuntime: {int(time.time()-t0)}s", flush=True)

main()
