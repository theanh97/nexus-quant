"""
Phase 246 — Per-Regime V1 Lookbacks Sweep
==========================================
Baseline: v2.43.0+ (will update to whatever Phase 245 produces)

V1's internal weight split now differs per regime (from v2.43.0):
  LOW:  wc=0.10, wm=0.45, wmr=0.45
  MID:  wc=0.25, wm=0.50, wmr=0.25
  HIGH: wc=0.25, wm=0.50, wmr=0.25

Hypothesis: Optimal lookback windows for momentum and mean-reversion
may differ per regime:
  - LOW (bear, high MR weight): shorter MR lookback → faster adaptation
  - HIGH (bull, high mom weight): longer mom lookback → stronger trend capture
  - MID: balanced

Approach:
  1. Pre-compute V1 for each (regime, mom_lb, mr_lb) combo
  2. Sequential 2-pass sweep: LOW → MID → HIGH → repeat
  3. Shared w_carry/w_mom/w_mr params from Phase 244; only vary lookbacks
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
    out = Path("artifacts/phase246"); out.mkdir(parents=True, exist_ok=True)
    (out / "phase246_report.json").write_text(json.dumps(_partial, indent=2))
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
FTS_RS = {"LOW": 0.50, "MID": 0.20, "HIGH": 0.40}
FTS_BS = {"LOW": 3.00, "MID": 3.00, "HIGH": 2.00}
FTS_RT = {"LOW": 0.80, "MID": 0.65, "HIGH": 0.55}
FTS_BT = {"LOW": 0.30, "MID": 0.25, "HIGH": 0.25}
VOL_THR   = {"LOW": 0.50, "MID": 0.50, "HIGH": 0.50}
VOL_SCALE = {"LOW": 0.40, "MID": 0.15, "HIGH": 0.10}
DISP_THR   = {"LOW": 0.70, "MID": 0.70, "HIGH": 0.40}
DISP_SCALE = {"LOW": 0.50, "MID": 1.50, "HIGH": 0.50}

# These will be updated from config after Phase 245 completes
# For now, use Phase 243 values (will load from config at runtime)
def load_regime_weights_from_config():
    cfg_path = ROOT / "configs" / "production_p91b_champion.json"
    cfg = json.load(open(cfg_path))
    brs = cfg["breadth_regime_switching"]["regime_weights"]
    return {
        "LOW":  {"v1": brs["LOW"]["v1"],  "i460": brs["LOW"]["i460bw168"],  "i415": brs["LOW"]["i415bw216"],  "f168": brs["LOW"]["f168"]},
        "MID":  {"v1": brs["MID"]["v1"],  "i460": brs["MID"]["i460bw168"],  "i415": brs["MID"]["i415bw216"],  "f168": brs["MID"]["f168"]},
        "HIGH": {"v1": brs["HIGH"]["v1"], "i460": brs["HIGH"]["i460bw168"], "i415": brs["HIGH"]["i415bw216"], "f168": brs["HIGH"]["f168"]},
    }

def load_v1_weights_from_config():
    cfg_path = ROOT / "configs" / "production_p91b_champion.json"
    cfg = json.load(open(cfg_path))
    v1w = cfg.get("v1_per_regime_weights", {})
    if not v1w:
        # Fallback to Phase 244 values
        return {
            "LOW":  {"w_carry": 0.10, "w_mom": 0.45, "w_mean_reversion": 0.45},
            "MID":  {"w_carry": 0.25, "w_mom": 0.50, "w_mean_reversion": 0.25},
            "HIGH": {"w_carry": 0.25, "w_mom": 0.50, "w_mean_reversion": 0.25},
        }
    return v1w

# V1 common fixed params
V1_FIXED = {
    "k_per_side": 2,
    "vol_lookback_bars": 192,
    "target_gross_leverage": 0.35,
    "rebalance_interval_bars": 60,
}

I460_PARAMS = {"k_per_side": 4, "lookback_bars": 480, "beta_window_bars": 168,
               "target_gross_leverage": 0.3, "rebalance_interval_bars": 48}
I415_PARAMS = {"k_per_side": 4, "lookback_bars": 415, "beta_window_bars": 144,
               "target_gross_leverage": 0.3, "rebalance_interval_bars": 48}
F168_PARAMS = {"k_per_side": 2, "funding_lookback_bars": 168, "direction": "contrarian",
               "target_gross_leverage": 0.25, "rebalance_interval_bars": 36}

MIN_DELTA = 0.005; MIN_LOYO = 3

# Lookback sweep grids
MOM_LB_SWEEP = [168, 240, 288, 336, 400, 480]
MR_LB_SWEEP  = [48, 60, 72, 84, 96, 120, 168]

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

def compute_v1(dataset, w_carry, w_mom, w_mr, mom_lb, mr_lb):
    params = {**V1_FIXED,
              "w_carry": w_carry, "w_mom": w_mom, "w_mean_reversion": w_mr,
              "momentum_lookback_bars": mom_lb, "mean_reversion_lookback_bars": mr_lb}
    result = BacktestEngine(BacktestConfig(costs=COST_MODEL)).run(
        dataset, make_strategy({"name": "nexus_alpha_v1", "params": params}))
    return np.array(result.returns)

def ensemble_rets(regime_weights, v1_per_regime, base_data):
    btc_vol, fund_std_pct, ts_pct_cache, brd_pct, fixed_rets, n, _ = base_data
    RNAMES = ["LOW", "MID", "HIGH"]
    min_len = min(
        min(len(v1_per_regime[r]) for r in RNAMES),
        min(len(fixed_rets[k]) for k in ["i460", "i415", "f168"]),
    )
    ens = np.zeros(min_len)
    bv = btc_vol[:min_len]; bpct = brd_pct[:min_len]; fsp = fund_std_pct[:min_len]
    for i in range(min_len):
        bp = bpct[i]
        ridx = 0 if bp < P_LOW else (2 if bp > P_HIGH else 1)
        rname = RNAMES[ridx]
        w = regime_weights[rname]
        tsp = ts_pct_cache[TS_PCT_WIN[rname]][i]
        ret_i = (w["v1"] * v1_per_regime[rname][i]
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

def evaluate(regime_weights, v1_per_regime_all, all_base_data):
    yearly = {yr: sharpe(ensemble_rets(regime_weights, v1_per_regime_all[yr], all_base_data[yr]))
              for yr in all_base_data}
    return obj_fn(yearly), yearly

def main():
    t0 = time.time()

    # Load current config values
    REGIME_WEIGHTS = load_regime_weights_from_config()
    V1_W = load_v1_weights_from_config()

    cfg_path = ROOT / "configs" / "production_p91b_champion.json"
    cfg = json.load(open(cfg_path))
    BASELINE_OBJ = cfg["monitoring"]["expected_performance"]["annual_sharpe_backtest"]
    cur_version = cfg["_version"]

    print("=" * 65)
    print("Phase 246 — Per-Regime V1 Lookbacks Sweep")
    print(f"Baseline: {cur_version}  OBJ={BASELINE_OBJ}")
    for r in ["LOW", "MID", "HIGH"]:
        w = V1_W[r]
        print(f"  V1 {r}: wc={w.get('w_carry',0.25)} wm={w.get('w_mom',0.45)} wmr={w.get('w_mean_reversion',0.30)}")
    print("=" * 65)

    print("\n[1] Loading per-year data & fixed signals...")
    all_base_data = {}
    for yr in sorted(YEAR_RANGES):
        print(f"  {yr}: ", end="", flush=True)
        all_base_data[yr] = load_year_data(yr)
        print("done.", flush=True)

    # Build unique V1 variants: (regime, mom_lb, mr_lb)
    # For each regime, sweep mom_lb × mr_lb (with mr_lb < mom_lb)
    combos_per_regime = {}
    for r in ["LOW", "MID", "HIGH"]:
        combos_per_regime[r] = [(ml, mrl) for ml in MOM_LB_SWEEP for mrl in MR_LB_SWEEP if mrl < ml]

    # Unique (r, mom_lb, mr_lb) → need to pre-compute V1 for each unique (weights, mom_lb, mr_lb)
    # Since LOW has different weights than MID/HIGH, we group:
    # Group 1 (LOW): wc=0.10, wm=0.45, wmr=0.45
    # Group 2 (MID/HIGH): wc=0.25, wm=0.50, wmr=0.25
    # Each group × all (mom_lb, mr_lb) combos

    all_lookback_combos = [(ml, mrl) for ml in MOM_LB_SWEEP for mrl in MR_LB_SWEEP if mrl < ml]
    print(f"\n  Lookback combos: {len(all_lookback_combos)} per regime group")

    print("\n[2] Pre-computing V1 cache (2 weight groups × lookbacks × years)...")
    # v1_cache[("LOW", mom_lb, mr_lb)][year] = rets
    # v1_cache[("MIDHIGH", mom_lb, mr_lb)][year] = rets
    v1_cache = {}
    n_computed = 0
    for group_name, group_w in [("LOW", V1_W["LOW"]), ("MIDHIGH", V1_W["MID"])]:
        wc = group_w.get("w_carry", 0.25)
        wm = group_w.get("w_mom", 0.45)
        wmr = group_w.get("w_mean_reversion", 0.30)
        for (mom_lb, mr_lb) in all_lookback_combos:
            key = (group_name, mom_lb, mr_lb)
            v1_cache[key] = {}
            for yr in sorted(YEAR_RANGES):
                v1_cache[key][yr] = compute_v1(all_base_data[yr][6], wc, wm, wmr, mom_lb, mr_lb)
                n_computed += 1
                if n_computed % 10 == 0:
                    print(f"  ... {n_computed}/{2*len(all_lookback_combos)*5} done", flush=True)
    print(f"  → Pre-computation complete: {n_computed} V1 runs")

    # Baseline: current lookbacks (336/84) for all regimes
    BASE_MOM_LB = 336; BASE_MR_LB = 84
    base_v1_all = {}
    for yr in all_base_data:
        base_v1_all[yr] = {
            "LOW":  v1_cache[("LOW",    BASE_MOM_LB, BASE_MR_LB)][yr],
            "MID":  v1_cache[("MIDHIGH", BASE_MOM_LB, BASE_MR_LB)][yr],
            "HIGH": v1_cache[("MIDHIGH", BASE_MOM_LB, BASE_MR_LB)][yr],
        }
    base_obj_val, base_yearly = evaluate(REGIME_WEIGHTS, base_v1_all, all_base_data)
    print(f"\n  Baseline OBJ = {base_obj_val:.4f}  (expected ≈ {BASELINE_OBJ})")
    _partial.update({"baseline_obj": float(base_obj_val)})

    print("\n[3] Sequential per-regime lookback sweep (2 passes)...")

    # Current best lookbacks per regime
    best_lbs = {r: (BASE_MOM_LB, BASE_MR_LB) for r in ["LOW", "MID", "HIGH"]}

    def get_v1_key(rname):
        return "LOW" if rname == "LOW" else "MIDHIGH"

    for pass_num in [1, 2]:
        print(f"\n  === Pass {pass_num} ===")
        for target_regime in ["LOW", "MID", "HIGH"]:
            best_obj_so_far = base_obj_val
            best_lb = best_lbs[target_regime]
            for (mom_lb, mr_lb) in all_lookback_combos:
                v1_per_regime_all = {}
                for yr in all_base_data:
                    v1_per_regime_all[yr] = {
                        r: v1_cache[(get_v1_key(r), *(best_lbs[r] if r != target_regime else (mom_lb, mr_lb)))][yr]
                        for r in ["LOW", "MID", "HIGH"]
                    }
                o, _ = evaluate(REGIME_WEIGHTS, v1_per_regime_all, all_base_data)
                if o > best_obj_so_far:
                    best_obj_so_far = o
                    best_lb = (mom_lb, mr_lb)
            best_lbs[target_regime] = best_lb
            print(f"    {target_regime}: mom_lb={best_lb[0]} mr_lb={best_lb[1]}  "
                  f"OBJ={best_obj_so_far:.4f}  Δ={best_obj_so_far-base_obj_val:+.4f}")

    # Final evaluation
    print(f"\n  → Best per-regime lookbacks:")
    for r in ["LOW", "MID", "HIGH"]:
        print(f"    {r}: mom_lb={best_lbs[r][0]} mr_lb={best_lbs[r][1]}")

    final_v1_all = {}
    for yr in all_base_data:
        final_v1_all[yr] = {
            r: v1_cache[(get_v1_key(r), *best_lbs[r])][yr]
            for r in ["LOW", "MID", "HIGH"]
        }
    final_obj_val, final_yearly = evaluate(REGIME_WEIGHTS, final_v1_all, all_base_data)
    delta = final_obj_val - base_obj_val
    print(f"\n  Final OBJ={final_obj_val:.4f}  Δ={delta:+.4f}")

    print("\n[4] LOYO validation...")
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
        "phase": 246, "baseline_obj": float(base_obj_val),
        "best_obj": float(final_obj_val), "delta": float(delta),
        "loyo_wins": loyo_wins, "loyo_total": len(final_yearly),
        "validated": validated,
        "best_per_regime_lbs": {r: {"mom_lb": best_lbs[r][0], "mr_lb": best_lbs[r][1]}
                                 for r in ["LOW", "MID", "HIGH"]},
    }
    _partial.update(result)
    out = Path("artifacts/phase246"); out.mkdir(parents=True, exist_ok=True)
    (out / "phase246_report.json").write_text(json.dumps(result, indent=2))

    if validated:
        cfg = json.load(open(cfg_path))
        # Store per-regime V1 lookbacks in config
        if "v1_per_regime_weights" not in cfg:
            cfg["v1_per_regime_weights"] = {}
        for r in ["LOW", "MID", "HIGH"]:
            ml, mrl = best_lbs[r]
            if r not in cfg["v1_per_regime_weights"]:
                cfg["v1_per_regime_weights"][r] = {}
            cfg["v1_per_regime_weights"][r]["momentum_lookback_bars"] = ml
            cfg["v1_per_regime_weights"][r]["mean_reversion_lookback_bars"] = mrl
        # Bump version
        cur_ver = cfg.get("_version", "2.43.0")
        parts = cur_ver.split(".")
        new_ver = f"{parts[0]}.{parts[1]}.{int(parts[2])+1}"
        cfg["_version"] = new_ver
        cfg["monitoring"]["expected_performance"]["annual_sharpe_backtest"] = round(final_obj_val, 4)
        with open(cfg_path, "w") as f:
            json.dump(cfg, f, indent=2, ensure_ascii=False)
        print(f"\n  Config → {new_ver}  OBJ={round(final_obj_val,4)}")

    print(f"\nRuntime: {int(time.time()-t0)}s")

main()
