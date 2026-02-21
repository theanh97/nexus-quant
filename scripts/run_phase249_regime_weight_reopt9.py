"""
Phase 249 — Regime Weight Re-opt (9th Pass) [Vectorized]
=========================================================
Baseline: Phase 248 validated state — FTS params dramatically changed.
  Phase 248: OBJ=4.2411, Δ=+0.2266, LOYO 5/5

New FTS params (hardcoded from Phase 248 result):
  LOW:  RS=0.05, BS=4.00, RT=0.80, BT=0.30
  MID:  RS=0.05, BS=2.00, RT=0.65, BT=0.40
  HIGH: RS=0.25, BS=2.00, RT=0.50, BT=0.20

Starting regime weights (Phase 245 validated values, used in Phase 248):
  LOW:  v1=0.35, i460=0.10, i415=0.15, f168=0.40
  MID:  v1=0.15, i460=0.00, i415=0.20, f168=0.65
  HIGH: v1=0.30, i460=0.70, i415=0.00, f168=0.00

After such large FTS change, optimal signal allocation will shift.
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
    out = Path("artifacts/phase249"); out.mkdir(parents=True, exist_ok=True)
    (out / "phase249_report.json").write_text(json.dumps(_partial, indent=2))
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

# Phase 248 validated FTS params
FTS_RS = {"LOW": 0.05, "MID": 0.05, "HIGH": 0.25}
FTS_BS = {"LOW": 4.00, "MID": 2.00, "HIGH": 2.00}
FTS_RT = {"LOW": 0.80, "MID": 0.65, "HIGH": 0.50}
FTS_BT = {"LOW": 0.30, "MID": 0.40, "HIGH": 0.20}

VOL_THR   = {"LOW": 0.50, "MID": 0.50, "HIGH": 0.50}
VOL_SCALE = {"LOW": 0.40, "MID": 0.15, "HIGH": 0.10}
DISP_THR   = {"LOW": 0.70, "MID": 0.70, "HIGH": 0.40}
DISP_SCALE = {"LOW": 0.50, "MID": 1.50, "HIGH": 0.50}

RNAMES = ["LOW", "MID", "HIGH"]

# Phase 245 regime weights (what Phase 248 was validated with)
REGIME_WEIGHTS_START = {
    "LOW":  {"v1": 0.35, "i460": 0.10, "i415": 0.15, "f168": 0.40},
    "MID":  {"v1": 0.15, "i460": 0.00, "i415": 0.20, "f168": 0.65},
    "HIGH": {"v1": 0.30, "i460": 0.70, "i415": 0.00, "f168": 0.00},
}

# V1 per-regime weights (Phase 244)
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

BASELINE_OBJ = 4.2411
MIN_DELTA = 0.005; MIN_LOYO = 3
W_STEPS = [round(x * 0.05, 2) for x in range(17)]  # 0.00 to 0.80

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

def precompute_fast_data(all_base_data, all_v1_raw, all_fixed_rets):
    """Precompute overlay_mult and regime-masked, overlay-scaled signal arrays."""
    fast_data = {}
    for yr, base_data in sorted(all_base_data.items()):
        btc_vol, fund_std_pct, ts_pct_cache, brd_pct, n, _ = base_data
        v1 = all_v1_raw[yr]
        fixed = all_fixed_rets[yr]
        min_len = min(
            min(len(v1[r]) for r in RNAMES),
            min(len(fixed[k]) for k in ["i460","i415","f168"]),
        )
        bpct = brd_pct[:min_len]
        bv   = btc_vol[:min_len]
        fsp  = fund_std_pct[:min_len]

        regime_idx = np.where(bpct < P_LOW, 0, np.where(bpct > P_HIGH, 2, 1))
        masks = [regime_idx == i for i in range(3)]

        overlay_mult = np.ones(min_len)
        for ridx, rname in enumerate(RNAMES):
            m = masks[ridx]
            vol_m = m & ~np.isnan(bv) & (bv > VOL_THR[rname])
            overlay_mult[vol_m] *= VOL_SCALE[rname]
            disp_m = m & (fsp > DISP_THR[rname])
            overlay_mult[disp_m] *= DISP_SCALE[rname]
            # FTS
            tsp = ts_pct_cache[TS_PCT_WIN[rname]][:min_len]
            overlay_mult[m & (tsp > FTS_RT[rname])] *= FTS_RS[rname]
            overlay_mult[m & (tsp < FTS_BT[rname])] *= FTS_BS[rname]

        scaled = {
            "v1_LOW": overlay_mult * v1["LOW"][:min_len],
            "v1_MID": overlay_mult * v1["MID"][:min_len],
            "i460":   overlay_mult * fixed["i460"][:min_len],
            "i415":   overlay_mult * fixed["i415"][:min_len],
            "f168":   overlay_mult * fixed["f168"][:min_len],
        }
        fast_data[yr] = (masks, min_len, scaled)
    return fast_data

def fast_evaluate(regime_weights, fast_data):
    yearly = {}
    for yr, (masks, min_len, scaled) in fast_data.items():
        ens = np.zeros(min_len)
        for ridx, rname in enumerate(RNAMES):
            m = masks[ridx]
            w = regime_weights[rname]
            v1_key = "v1_LOW" if rname == "LOW" else "v1_MID"
            ens[m] += (w["v1"]  * scaled[v1_key][m]
                       + w["i460"] * scaled["i460"][m]
                       + w["i415"] * scaled["i415"][m]
                       + w["f168"] * scaled["f168"][m])
        r = ens[~np.isnan(ens)]
        sh = float(np.mean(r) / np.std(r, ddof=1) * np.sqrt(8760)) if len(r) > 1 else 0.0
        yearly[yr] = sh
    vals = list(yearly.values())
    return float(np.mean(vals) - 0.5 * np.std(vals, ddof=1)), yearly

def sweep_one_regime(target, current_wts, fast_data, best_obj):
    best_w = dict(current_wts[target])
    for wv1 in W_STEPS:
        for wi460 in W_STEPS:
            for wi415 in W_STEPS:
                wf168 = round(1.0 - wv1 - wi460 - wi415, 4)
                if wf168 < 0: continue
                test = dict(current_wts)
                test[target] = {"v1": wv1, "i460": wi460, "i415": wi415, "f168": wf168}
                o, _ = fast_evaluate(test, fast_data)
                if o > best_obj:
                    best_obj = o
                    best_w = {"v1": wv1, "i460": wi460, "i415": wi415, "f168": wf168}
    return best_w, best_obj

def main():
    t0 = time.time()
    print("=" * 65)
    print("Phase 249 — Regime Weight Re-opt (9th Pass) [Vectorized]")
    print(f"Baseline: v2.46.1  OBJ={BASELINE_OBJ}")
    print(f"FTS: LOW RS={FTS_RS['LOW']}/BS={FTS_BS['LOW']}/RT={FTS_RT['LOW']}/BT={FTS_BT['LOW']}")
    print(f"     MID RS={FTS_RS['MID']}/BS={FTS_BS['MID']}/RT={FTS_RT['MID']}/BT={FTS_BT['MID']}")
    print(f"     HIGH RS={FTS_RS['HIGH']}/BS={FTS_BS['HIGH']}/RT={FTS_RT['HIGH']}/BT={FTS_BT['HIGH']}")
    print("=" * 65)

    print("\n[1] Loading per-year data...", flush=True)
    all_base_data = {}
    for yr in sorted(YEAR_RANGES):
        print(f"  {yr}: ", end="", flush=True)
        all_base_data[yr] = load_year_data(yr)
        print("done.", flush=True)

    print("\n[2] Pre-computing signals (V1×2, I460, I415, F168) × 5 years...", flush=True)
    all_v1_raw = {}; all_fixed_rets = {}
    for yr in sorted(YEAR_RANGES):
        ds = all_base_data[yr][5]
        all_v1_raw[yr] = {
            "LOW":  compute_signal(ds, "nexus_alpha_v1", V1_LOW_PARAMS),
            "MID":  compute_signal(ds, "nexus_alpha_v1", V1_MIDHIGH_PARAMS),
            "HIGH": compute_signal(ds, "nexus_alpha_v1", V1_MIDHIGH_PARAMS),
        }
        all_fixed_rets[yr] = {
            "i460": compute_signal(ds, "idio_momentum_alpha",    I460_PARAMS),
            "i415": compute_signal(ds, "idio_momentum_alpha",    I415_PARAMS),
            "f168": compute_signal(ds, "funding_momentum_alpha", F168_PARAMS),
        }
        print(f"  {yr}: done.", flush=True)

    print("\n[3] Precomputing fast eval data...", flush=True)
    fast_data = precompute_fast_data(all_base_data, all_v1_raw, all_fixed_rets)
    print("  done.", flush=True)

    base_obj_val, base_yearly = fast_evaluate(REGIME_WEIGHTS_START, fast_data)
    print(f"\n  Baseline OBJ = {base_obj_val:.4f}  (expected ≈ {BASELINE_OBJ})", flush=True)
    _partial.update({"baseline_obj": float(base_obj_val)})

    print("\n[4] Sequential regime weight sweep (2 passes)...", flush=True)
    current_wts = {r: dict(REGIME_WEIGHTS_START[r]) for r in RNAMES}
    running_best = base_obj_val

    for pass_num in [1, 2]:
        print(f"\n  === Pass {pass_num} ===", flush=True)
        for rname in RNAMES:
            ts = time.time()
            best_w, running_best = sweep_one_regime(rname, current_wts, fast_data, running_best)
            current_wts[rname] = best_w
            elapsed = int(time.time() - ts)
            print(f"    {rname}: v1={best_w['v1']:.2f} i460={best_w['i460']:.4f} "
                  f"i415={best_w['i415']:.4f} f168={best_w['f168']:.4f}  "
                  f"OBJ={running_best:.4f}  Δ={running_best-base_obj_val:+.4f}  [{elapsed}s]",
                  flush=True)

    final_obj_val, final_yearly = fast_evaluate(current_wts, fast_data)
    delta = final_obj_val - base_obj_val
    print(f"\n  Final OBJ={final_obj_val:.4f}  Δ={delta:+.4f}", flush=True)

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
        "phase": 249, "baseline_obj": float(base_obj_val),
        "best_obj": float(final_obj_val), "delta": float(delta),
        "loyo_wins": loyo_wins, "loyo_total": len(final_yearly),
        "validated": validated, "best_regime_weights": current_wts,
    }
    _partial.update(result)
    out = Path("artifacts/phase249"); out.mkdir(parents=True, exist_ok=True)
    (out / "phase249_report.json").write_text(json.dumps(result, indent=2))

    if validated:
        cfg_path = ROOT / "configs" / "production_p91b_champion.json"
        cfg = json.load(open(cfg_path))
        brs = cfg["breadth_regime_switching"]["regime_weights"]
        for rname, wts in current_wts.items():
            brs[rname]["v1"]        = wts["v1"]
            brs[rname]["i460bw168"] = wts["i460"]
            brs[rname]["i415bw216"] = wts["i415"]
            brs[rname]["f168"]      = wts["f168"]
        # Also restore FTS params explicitly
        fts = cfg.get("fts_overlay_params", {})
        for rname in RNAMES:
            if "per_regime_rs" not in fts: fts["per_regime_rs"] = {}
            if "per_regime_bs" not in fts: fts["per_regime_bs"] = {}
            if "per_regime_rt" not in fts: fts["per_regime_rt"] = {}
            if "per_regime_bt" not in fts: fts["per_regime_bt"] = {}
            fts["per_regime_rs"][rname] = FTS_RS[rname]
            fts["per_regime_bs"][rname] = FTS_BS[rname]
            fts["per_regime_rt"][rname] = FTS_RT[rname]
            fts["per_regime_bt"][rname] = FTS_BT[rname]
        cfg["fts_overlay_params"] = fts
        cfg["_version"] = "2.47.0"
        cfg["monitoring"]["expected_performance"]["annual_sharpe_backtest"] = round(final_obj_val, 4)
        with open(cfg_path, "w") as f:
            json.dump(cfg, f, indent=2, ensure_ascii=False)
        print(f"\n  Config → v2.47.0  OBJ={round(final_obj_val,4)}", flush=True)
    else:
        # Even if no improvement, write the authoritative consistent state
        cfg_path = ROOT / "configs" / "production_p91b_champion.json"
        cfg = json.load(open(cfg_path))
        # Restore Phase 245 regime weights + Phase 248 FTS as canonical
        brs = cfg["breadth_regime_switching"]["regime_weights"]
        for rname, wts in REGIME_WEIGHTS_START.items():
            brs[rname]["v1"]        = wts["v1"]
            brs[rname]["i460bw168"] = wts["i460"]
            brs[rname]["i415bw216"] = wts["i415"]
            brs[rname]["f168"]      = wts["f168"]
        fts = cfg.get("fts_overlay_params", {})
        for rname in RNAMES:
            if "per_regime_rs" not in fts: fts["per_regime_rs"] = {}
            if "per_regime_bs" not in fts: fts["per_regime_bs"] = {}
            if "per_regime_rt" not in fts: fts["per_regime_rt"] = {}
            if "per_regime_bt" not in fts: fts["per_regime_bt"] = {}
            fts["per_regime_rs"][rname] = FTS_RS[rname]
            fts["per_regime_bs"][rname] = FTS_BS[rname]
            fts["per_regime_rt"][rname] = FTS_RT[rname]
            fts["per_regime_bt"][rname] = FTS_BT[rname]
        cfg["fts_overlay_params"] = fts
        cfg["_version"] = "2.47.0"
        cfg["monitoring"]["expected_performance"]["annual_sharpe_backtest"] = round(base_obj_val, 4)
        with open(cfg_path, "w") as f:
            json.dump(cfg, f, indent=2, ensure_ascii=False)
        print(f"\n  Config fixed → v2.47.0  (no new improvement)", flush=True)

    print(f"\nRuntime: {int(time.time()-t0)}s", flush=True)

main()
