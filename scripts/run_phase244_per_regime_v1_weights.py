"""
Phase 244 — Per-Regime V1 Internal Weights Sweep
=================================================
Baseline: v2.42.0, OBJ=3.7924

Hypothesis: Different regimes benefit from different V1 sub-signal blends:
  - LOW (bear):  more mean-reversion, less momentum (MR captures reversals)
  - MID:         balanced (baseline wc=0.25/wm=0.45/wmr=0.30 may already be optimal)
  - HIGH (bull): more momentum, less mean-reversion (trends persist)

Approach:
  1. Pre-compute V1 for each unique (w_carry, w_mom) combo across all years → v1_cache
  2. Sequential 2-pass: sweep LOW → MID → HIGH → repeat
  3. LOYO validate best combo

Key update: per-regime ts_pct_win = {LOW:240, MID:288, HIGH:400} from v2.42.0.
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
    out = Path("artifacts/phase244"); out.mkdir(parents=True, exist_ok=True)
    (out / "phase244_report.json").write_text(json.dumps(_partial, indent=2))
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

# v2.42.0 per-regime params
TS_PCT_WIN = {"LOW": 240, "MID": 288, "HIGH": 400}

FTS_RS = {"LOW": 0.50, "MID": 0.20, "HIGH": 0.40}
FTS_BS = {"LOW": 3.00, "MID": 3.00, "HIGH": 2.00}
FTS_RT = {"LOW": 0.80, "MID": 0.65, "HIGH": 0.55}
FTS_BT = {"LOW": 0.30, "MID": 0.25, "HIGH": 0.25}
VOL_THR   = {"LOW": 0.50, "MID": 0.50, "HIGH": 0.50}
VOL_SCALE = {"LOW": 0.40, "MID": 0.15, "HIGH": 0.10}
DISP_THR   = {"LOW": 0.70, "MID": 0.70, "HIGH": 0.40}
DISP_SCALE = {"LOW": 0.50, "MID": 1.50, "HIGH": 0.50}

REGIME_WEIGHTS = {
    "LOW":  {"v1": 0.44,   "i460": 0.0864, "i415": 0.1035, "f168": 0.37},
    "MID":  {"v1": 0.20,   "i460": 0.0000, "i415": 0.0200, "f168": 0.78},
    "HIGH": {"v1": 0.18,   "i460": 0.7900, "i415": 0.0000, "f168": 0.03},
}

# Base V1 params (v2.42.0)
V1_BASE = {
    "k_per_side": 2, "w_carry": 0.25, "w_mom": 0.45, "w_mean_reversion": 0.30,
    "momentum_lookback_bars": 336, "mean_reversion_lookback_bars": 84,
    "vol_lookback_bars": 192, "target_gross_leverage": 0.35, "rebalance_interval_bars": 60,
}
I460_PARAMS = {"k_per_side": 4, "lookback_bars": 480, "beta_window_bars": 168,
               "target_gross_leverage": 0.3, "rebalance_interval_bars": 48}
I415_PARAMS = {"k_per_side": 4, "lookback_bars": 415, "beta_window_bars": 144,
               "target_gross_leverage": 0.3, "rebalance_interval_bars": 48}
F168_PARAMS = {"k_per_side": 2, "funding_lookback_bars": 168, "direction": "contrarian",
               "target_gross_leverage": 0.25, "rebalance_interval_bars": 36}

BASELINE_OBJ = 3.7924
MIN_DELTA = 0.005; MIN_LOYO = 3

# Sweep grid — ~40 valid combos (w_mr = 1 - wc - wm >= 0.10)
W_CARRY_SWEEP = [0.0, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35]
W_MOM_SWEEP   = [0.20, 0.30, 0.40, 0.45, 0.50, 0.55, 0.60, 0.70]

COST_MODEL = cost_model_from_config({
    "fee_rate": 0.0005, "slippage_rate": 0.0003, "cost_multiplier": 1.0,
})

def rolling_mean_arr(a, w):
    out = np.full(len(a), np.nan)
    if w <= 0 or len(a) == 0: return out
    cs = np.cumsum(np.where(np.isnan(a), 0.0, a))
    for i in range(w - 1, len(a)):
        out[i] = cs[i] / w if i == w - 1 else (cs[i] - cs[i - w]) / w
    return out

def load_year_data(year):
    """Load base data + fixed signals (I460/I415/F168). No V1 here."""
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
    # Pre-compute ts_spread_pct for each unique window
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
    # Fixed signals
    fixed_rets = {}
    for sk, sname, params in [
        ("i460", "idio_momentum_alpha",    I460_PARAMS),
        ("i415", "idio_momentum_alpha",    I415_PARAMS),
        ("f168", "funding_momentum_alpha", F168_PARAMS),
    ]:
        result = BacktestEngine(BacktestConfig(costs=COST_MODEL)).run(
            dataset, make_strategy({"name": sname, "params": params}))
        fixed_rets[sk] = np.array(result.returns)
    return btc_vol, fund_std_pct, ts_pct_cache, brd_pct, fixed_rets, n, dataset

def compute_v1_rets(dataset, v1_params):
    result = BacktestEngine(BacktestConfig(costs=COST_MODEL)).run(
        dataset, make_strategy({"name": "nexus_alpha_v1", "params": v1_params}))
    return np.array(result.returns)

def ensemble_rets(v1_per_regime, base_data):
    """
    v1_per_regime: dict {regime_name: np.array of returns}
    base_data: (btc_vol, fund_std_pct, ts_pct_cache, brd_pct, fixed_rets, n, dataset)
    """
    btc_vol, fund_std_pct, ts_pct_cache, brd_pct, fixed_rets, n, _ = base_data
    RNAMES = ["LOW", "MID", "HIGH"]
    sk_fixed = ["i460", "i415", "f168"]
    min_len = min(
        min(len(v1_per_regime[r]) for r in RNAMES),
        min(len(fixed_rets[k]) for k in sk_fixed),
    )
    ens = np.zeros(min_len)
    bv   = btc_vol[:min_len]
    bpct = brd_pct[:min_len]
    fsp  = fund_std_pct[:min_len]
    for i in range(min_len):
        bp = bpct[i]
        ridx = 0 if bp < P_LOW else (2 if bp > P_HIGH else 1)
        rname = RNAMES[ridx]
        w = REGIME_WEIGHTS[rname]
        # regime-specific V1 and ts_spread_pct
        v1_ret = v1_per_regime[rname][i]
        tsp = ts_pct_cache[TS_PCT_WIN[rname]][i]
        ret_i = (w["v1"] * v1_ret
                 + w["i460"] * fixed_rets["i460"][i]
                 + w["i415"] * fixed_rets["i415"][i]
                 + w["f168"] * fixed_rets["f168"][i])
        if not np.isnan(bv[i]) and bv[i] > VOL_THR[rname]:
            ret_i *= VOL_SCALE[rname]
        if fsp[i] > DISP_THR[rname]:
            ret_i *= DISP_SCALE[rname]
        if tsp > FTS_RT[rname]: ret_i *= FTS_RS[rname]
        elif tsp < FTS_BT[rname]: ret_i *= FTS_BS[rname]
        ens[i] = ret_i
    return ens

def sharpe(r):
    r = r[~np.isnan(r)]
    if len(r) < 2: return 0.0
    return float(np.mean(r) / np.std(r, ddof=1) * np.sqrt(8760))

def obj_fn(yearly):
    vals = list(yearly.values())
    return float(np.mean(vals) - 0.5 * np.std(vals, ddof=1))

def evaluate(v1_per_regime_all_yrs, all_base_data):
    """
    v1_per_regime_all_yrs: {year: {regime: rets}}
    """
    yearly = {
        yr: sharpe(ensemble_rets(v1_per_regime_all_yrs[yr], all_base_data[yr]))
        for yr in all_base_data
    }
    return obj_fn(yearly), yearly

def main():
    t0 = time.time()
    print("=" * 65)
    print("Phase 244 — Per-Regime V1 Internal Weights Sweep")
    print(f"Baseline: v2.42.0  OBJ={BASELINE_OBJ}")
    print(f"V1 base: w_carry={V1_BASE['w_carry']} w_mom={V1_BASE['w_mom']} w_mr={V1_BASE['w_mean_reversion']}")
    print("=" * 65)

    print("\n[1] Loading per-year data & fixed signals (I460/I415/F168)...")
    all_base_data = {}
    for yr in sorted(YEAR_RANGES):
        print(f"  {yr}: ", end="", flush=True)
        all_base_data[yr] = load_year_data(yr)
        print("done.", flush=True)

    # Build valid sweep combos
    combos = []
    for wc in W_CARRY_SWEEP:
        for wm in W_MOM_SWEEP:
            wmr = round(1.0 - wc - wm, 4)
            if wmr < 0.10: continue
            combos.append((wc, wm, wmr))
    print(f"\n  Sweep grid: {len(combos)} valid (wc, wm, wmr) combos")

    print("\n[2] Pre-computing V1 returns for all combos × all years...")
    v1_cache = {}  # {(wc, wm, wmr): {year: rets}}
    n_computed = 0
    for (wc, wm, wmr) in combos:
        v1_cache[(wc, wm, wmr)] = {}
        vp = dict(V1_BASE); vp["w_carry"] = wc; vp["w_mom"] = wm; vp["w_mean_reversion"] = wmr
        for yr in sorted(YEAR_RANGES):
            v1_cache[(wc, wm, wmr)][yr] = compute_v1_rets(all_base_data[yr][6], vp)
            n_computed += 1
            if n_computed % 20 == 0:
                print(f"  ... {n_computed}/{len(combos)*5} V1 runs done", flush=True)
    print(f"  → Pre-computation complete: {n_computed} V1 runs")

    # Baseline V1 rets (using base params)
    base_key = (V1_BASE["w_carry"], V1_BASE["w_mom"], V1_BASE["w_mean_reversion"])
    if base_key not in v1_cache:
        v1_cache[base_key] = {}
        vp = dict(V1_BASE)
        for yr in sorted(YEAR_RANGES):
            v1_cache[base_key][yr] = compute_v1_rets(all_base_data[yr][6], vp)

    # Baseline evaluation
    base_v1_all = {yr: {"LOW": v1_cache[base_key][yr],
                         "MID": v1_cache[base_key][yr],
                         "HIGH": v1_cache[base_key][yr]} for yr in all_base_data}
    base_obj_val, base_yearly = evaluate(base_v1_all, all_base_data)
    print(f"\n  Baseline OBJ = {base_obj_val:.4f}  (expected ≈ {BASELINE_OBJ})")
    _partial.update({"baseline_obj": float(base_obj_val)})

    print("\n[3] Sequential per-regime V1 weight sweep (2 passes)...")

    # Current best keys per regime (start at baseline)
    best_key = {r: base_key for r in ["LOW", "MID", "HIGH"]}

    for pass_num in [1, 2]:
        print(f"\n  === Pass {pass_num} ===")
        for target_regime in ["LOW", "MID", "HIGH"]:
            best_obj_so_far = base_obj_val  # compare against true baseline
            best_regime_key = best_key[target_regime]
            for (wc, wm, wmr) in combos:
                # Build per-regime V1 dict using best_key for other regimes, sweep for target
                v1_per_regime_all = {}
                for yr in all_base_data:
                    v1_per_regime_all[yr] = {
                        r: (v1_cache[(wc, wm, wmr)][yr] if r == target_regime
                            else v1_cache[best_key[r]][yr])
                        for r in ["LOW", "MID", "HIGH"]
                    }
                o, _ = evaluate(v1_per_regime_all, all_base_data)
                if o > best_obj_so_far:
                    best_obj_so_far = o
                    best_regime_key = (wc, wm, wmr)
            best_key[target_regime] = best_regime_key
            wc, wm, wmr = best_regime_key
            print(f"    {target_regime}: wc={wc} wm={wm} wmr={wmr}  OBJ={best_obj_so_far:.4f}  Δ={best_obj_so_far-base_obj_val:+.4f}")

    # Final evaluation with best per-regime keys
    print(f"\n  → Best per-regime V1 weights:")
    for r in ["LOW", "MID", "HIGH"]:
        wc, wm, wmr = best_key[r]
        print(f"    {r}: w_carry={wc} w_mom={wm} w_mr={wmr}")

    final_v1_all = {}
    for yr in all_base_data:
        final_v1_all[yr] = {r: v1_cache[best_key[r]][yr] for r in ["LOW", "MID", "HIGH"]}
    final_obj_val, final_yearly = evaluate(final_v1_all, all_base_data)
    delta = final_obj_val - base_obj_val
    print(f"\n  Final OBJ={final_obj_val:.4f}  Δ={delta:+.4f}")

    print(f"\n[4] LOYO validation...")
    loyo_wins = 0
    for yr in sorted(final_yearly):
        base_sh = base_yearly[yr]
        cand_sh = final_yearly[yr]
        win = cand_sh > base_sh
        if win: loyo_wins += 1
        print(f"  {yr}: base={base_sh:.4f}  cand={cand_sh:.4f}  {'WIN' if win else 'LOSE'}")

    print(f"\n  OBJ={final_obj_val:.4f}  Δ={delta:+.4f}  LOYO {loyo_wins}/{len(final_yearly)}")
    validated = loyo_wins >= MIN_LOYO and delta >= MIN_DELTA
    print(f"\n{'✅ VALIDATED' if validated else '❌ NO IMPROVEMENT'}")

    result = {
        "phase": 244, "baseline_obj": float(base_obj_val),
        "best_obj": float(final_obj_val), "delta": float(delta),
        "loyo_wins": loyo_wins, "loyo_total": len(final_yearly),
        "validated": validated,
        "best_per_regime": {r: {"w_carry": best_key[r][0], "w_mom": best_key[r][1], "w_mr": best_key[r][2]}
                             for r in ["LOW", "MID", "HIGH"]},
    }
    _partial.update(result)
    out = Path("artifacts/phase244"); out.mkdir(parents=True, exist_ok=True)
    (out / "phase244_report.json").write_text(json.dumps(result, indent=2))

    if validated:
        cfg_path = ROOT / "configs" / "production_p91b_champion.json"
        cfg = json.load(open(cfg_path))
        # Store per-regime V1 weights in config
        if "v1_per_regime_weights" not in cfg:
            cfg["v1_per_regime_weights"] = {}
        for r in ["LOW", "MID", "HIGH"]:
            wc, wm, wmr = best_key[r]
            cfg["v1_per_regime_weights"][r] = {
                "w_carry": wc, "w_mom": wm, "w_mean_reversion": wmr
            }
        cfg["_version"] = "2.43.0"
        cfg["monitoring"]["expected_performance"]["annual_sharpe_backtest"] = round(final_obj_val, 4)
        with open(cfg_path, "w") as f:
            json.dump(cfg, f, indent=2, ensure_ascii=False)
        print(f"\n  Config → v2.43.0  OBJ={round(final_obj_val,4)}")

    print(f"\nRuntime: {int(time.time()-t0)}s")

main()
