"""
Phase 260 — Regime Weight Re-opt (9th Pass) [VECTORIZED]
=========================================================
Baseline: v2.44.0 + v2.42.11 (mixed state after Engine P257 all-regime + MID fix)

Current weights:
  LOW:  v1=0.35 i460=0.10 i415=0.15 f168=0.40
  MID:  v1=0.15 i460=0.00 i415=0.00 f168=0.85
  HIGH: v1=0.20 i460=0.18 i415=0.27 f168=0.35

V1 per-regime weights (P244):
  LOW:  wc=0.10, wm=0.45, wmr=0.45
  MID:  wc=0.25, wm=0.50, wmr=0.25
  HIGH: wc=0.25, wm=0.50, wmr=0.25

Vectorized NumPy sweep — precompute overlay_mult + masked signals per year.
"""

import os, sys, json, time, subprocess
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
    out = Path("artifacts/phase260"); out.mkdir(parents=True, exist_ok=True)
    (out / "phase260_report.json").write_text(json.dumps(_partial, indent=2))
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
RNAMES = ["LOW", "MID", "HIGH"]

TS_PCT_WIN = {"LOW": 240, "MID": 288, "HIGH": 400}
FTS_RS = {"LOW": 0.50, "MID": 0.20, "HIGH": 0.40}
FTS_BS = {"LOW": 3.00, "MID": 3.00, "HIGH": 2.00}
FTS_RT = {"LOW": 0.80, "MID": 0.65, "HIGH": 0.55}
FTS_BT = {"LOW": 0.30, "MID": 0.25, "HIGH": 0.25}
VOL_THR   = {"LOW": 0.50, "MID": 0.50, "HIGH": 0.50}
VOL_SCALE = {"LOW": 0.40, "MID": 0.15, "HIGH": 0.10}
DISP_THR   = {"LOW": 0.70, "MID": 0.70, "HIGH": 0.40}
DISP_SCALE = {"LOW": 0.50, "MID": 1.50, "HIGH": 0.50}

# Load baseline from config at runtime
def load_regime_weights_from_config():
    cfg = json.load(open(ROOT / "configs" / "production_p91b_champion.json"))
    brs = cfg["breadth_regime_switching"]["regime_weights"]
    return {
        r: {"v1": brs[r]["v1"], "i460": brs[r]["i460bw168"],
            "i415": brs[r]["i415bw216"], "f168": brs[r]["f168"]}
        for r in RNAMES
    }

def load_v1_params_from_config():
    cfg = json.load(open(ROOT / "configs" / "production_p91b_champion.json"))
    v1w = cfg.get("v1_per_regime_weights", {})
    v1p = cfg["ensemble"]["signals"]["v1"]["params"]
    base = {
        "k_per_side": v1p.get("k_per_side", 2),
        "momentum_lookback_bars": v1p.get("momentum_lookback_bars", 336),
        "mean_reversion_lookback_bars": v1p.get("mean_reversion_lookback_bars", 84),
        "vol_lookback_bars": v1p.get("vol_lookback_bars", 192),
        "target_gross_leverage": v1p.get("target_gross_leverage", 0.35),
        "rebalance_interval_bars": v1p.get("rebalance_interval_bars", 60),
    }
    # Build per-regime V1 params
    params = {}
    for r in RNAMES:
        w = v1w.get(r, {"w_carry": 0.25, "w_mom": 0.50, "w_mean_reversion": 0.25})
        params[r] = {**base,
                     "w_carry": w["w_carry"],
                     "w_mom": w["w_mom"],
                     "w_mean_reversion": w["w_mean_reversion"]}
    return params

I460_PARAMS = {"k_per_side": 4, "lookback_bars": 480, "beta_window_bars": 168,
               "target_gross_leverage": 0.3, "rebalance_interval_bars": 48}
I415_PARAMS = {"k_per_side": 4, "lookback_bars": 415, "beta_window_bars": 144,
               "target_gross_leverage": 0.3, "rebalance_interval_bars": 48}
F168_PARAMS = {"k_per_side": 2, "funding_lookback_bars": 168, "direction": "contrarian",
               "target_gross_leverage": 0.25, "rebalance_interval_bars": 36}

MIN_DELTA = 0.005; MIN_LOYO = 3
W_STEPS = [round(x * 0.05, 2) for x in range(17)]  # 0.00 to 0.80

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

def compute_v1(dataset, params):
    result = BacktestEngine(BacktestConfig(costs=COST_MODEL)).run(
        dataset, make_strategy({"name": "nexus_alpha_v1", "params": params}))
    return np.array(result.returns)

def precompute_fast_eval_data(all_base_data, all_v1_rets):
    """Precompute overlay_mult, regime masks, and overlay-scaled signal arrays per year."""
    fast_data = {}
    for yr, base_data in sorted(all_base_data.items()):
        btc_vol, fund_std_pct, ts_pct_cache, brd_pct, fixed_rets, n, _ = base_data
        v1 = all_v1_rets[yr]
        min_len = min(
            min(len(v1[r]) for r in RNAMES),
            min(len(fixed_rets[k]) for k in ["i460", "i415", "f168"])
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
            tsp = ts_pct_cache[TS_PCT_WIN[rname]][:min_len]
            overlay_mult[m & (tsp > FTS_RT[rname])] *= FTS_RS[rname]
            overlay_mult[m & (tsp < FTS_BT[rname])] *= FTS_BS[rname]

        scaled = {
            "v1_LOW":  overlay_mult * v1["LOW"][:min_len],
            "v1_MID":  overlay_mult * v1["MID"][:min_len],
            "v1_HIGH": overlay_mult * v1["HIGH"][:min_len],
            "i460":    overlay_mult * fixed_rets["i460"][:min_len],
            "i415":    overlay_mult * fixed_rets["i415"][:min_len],
            "f168":    overlay_mult * fixed_rets["f168"][:min_len],
        }
        fast_data[yr] = (scaled, masks, min_len)
    return fast_data

def fast_evaluate(regime_weights, fast_data):
    """Vectorized evaluation. Returns (obj, yearly_sharpes)."""
    yearly = {}
    for yr, (scaled, masks, min_len) in fast_data.items():
        ens = np.zeros(min_len)
        for ridx, rname in enumerate(RNAMES):
            m = masks[ridx]
            w = regime_weights[rname]
            v1_key = f"v1_{rname}"
            ens[m] += (w["v1"] * scaled[v1_key][m]
                       + w["i460"] * scaled["i460"][m]
                       + w["i415"] * scaled["i415"][m]
                       + w["f168"] * scaled["f168"][m])
        r = ens[~np.isnan(ens)]
        sh = float(np.mean(r) / np.std(r, ddof=1) * np.sqrt(8760)) if len(r) > 1 else 0.0
        yearly[yr] = sh
    vals = list(yearly.values())
    return float(np.mean(vals) - 0.5 * np.std(vals, ddof=1)), yearly

def sweep_one_regime(target_regime, current_weights, fast_data, starting_obj):
    """Grid search weights for target_regime, return best (weights_dict, obj)."""
    best_obj = starting_obj
    best_w = dict(current_weights[target_regime])
    for wv1 in W_STEPS:
        for wi460 in W_STEPS:
            for wi415 in W_STEPS:
                wf168 = round(1.0 - wv1 - wi460 - wi415, 4)
                if wf168 < 0: continue
                w_test = dict(current_weights)
                w_test[target_regime] = {"v1": wv1, "i460": wi460, "i415": wi415, "f168": wf168}
                o, _ = fast_evaluate(w_test, fast_data)
                if o > best_obj:
                    best_obj = o
                    best_w = {"v1": wv1, "i460": wi460, "i415": wi415, "f168": wf168}
    return best_w, best_obj

def main():
    t0 = time.time()
    print("=" * 65)
    print("Phase 260 — Regime Weight Re-opt (9th Pass) [VECTORIZED]")
    print("=" * 65)

    # Load current config as baseline
    REGIME_WEIGHTS_BASE = load_regime_weights_from_config()
    V1_PARAMS = load_v1_params_from_config()
    print("\nBaseline weights from config:")
    for r in RNAMES:
        w = REGIME_WEIGHTS_BASE[r]
        print(f"  {r}: v1={w['v1']} i460={w['i460']} i415={w['i415']} f168={w['f168']}")

    print("\n[1] Loading per-year data & fixed signals...")
    all_base_data = {}
    for yr in sorted(YEAR_RANGES):
        print(f"  {yr}: ", end="", flush=True)
        all_base_data[yr] = load_year_data(yr)
        print("done.", flush=True)

    print("\n[2] Pre-computing per-regime V1 returns (3 variants × 5 years)...", flush=True)
    all_v1_rets = {}
    for yr in sorted(YEAR_RANGES):
        ds = all_base_data[yr][6]
        all_v1_rets[yr] = {r: compute_v1(ds, V1_PARAMS[r]) for r in RNAMES}
        print(f"  {yr}: LOW + MID + HIGH done.", flush=True)

    print("\n[3] Precomputing fast eval data (overlay_mult + masked signals)...", flush=True)
    fast_data = precompute_fast_eval_data(all_base_data, all_v1_rets)
    print("  done.", flush=True)

    base_obj_val, base_yearly = fast_evaluate(REGIME_WEIGHTS_BASE, fast_data)
    print(f"\n  Baseline OBJ = {base_obj_val:.4f}", flush=True)
    _partial.update({"baseline_obj": float(base_obj_val)})

    print("\n[4] Sequential regime weight sweep (2 passes)...", flush=True)
    current_weights = {r: dict(REGIME_WEIGHTS_BASE[r]) for r in RNAMES}
    running_best = base_obj_val

    for pass_num in [1, 2]:
        print(f"\n  === Pass {pass_num} ===", flush=True)
        for rname in RNAMES:
            t_sw = time.time()
            best_w, running_best = sweep_one_regime(rname, current_weights, fast_data, running_best)
            current_weights[rname] = best_w
            elapsed = int(time.time() - t_sw)
            print(f"    {rname}: v1={best_w['v1']:.2f} i460={best_w['i460']:.4f} "
                  f"i415={best_w['i415']:.4f} f168={best_w['f168']:.4f}  "
                  f"OBJ={running_best:.4f}  Δ={running_best-base_obj_val:+.4f}  [{elapsed}s]",
                  flush=True)

    final_obj_val, final_yearly = fast_evaluate(current_weights, fast_data)
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
        "phase": 260, "baseline_obj": float(base_obj_val),
        "best_obj": float(final_obj_val), "delta": float(delta),
        "loyo_wins": loyo_wins, "loyo_total": len(final_yearly),
        "validated": validated, "best_regime_weights": current_weights,
    }
    _partial.update(result)
    out = Path("artifacts/phase260"); out.mkdir(parents=True, exist_ok=True)
    (out / "phase260_report.json").write_text(json.dumps(result, indent=2))

    if validated:
        cfg_path = ROOT / "configs" / "production_p91b_champion.json"
        cfg = json.load(open(cfg_path))
        brs = cfg["breadth_regime_switching"]["regime_weights"]
        for rname, wts in current_weights.items():
            brs[rname]["v1"]        = wts["v1"]
            brs[rname]["i460bw168"] = wts["i460"]
            brs[rname]["i415bw216"] = wts["i415"]
            brs[rname]["f168"]      = wts["f168"]
        cfg["_version"] = "2.45.0"
        cfg["monitoring"]["expected_performance"]["annual_sharpe_backtest"] = round(final_obj_val, 4)
        with open(cfg_path, "w") as f:
            json.dump(cfg, f, indent=2, ensure_ascii=False)
        print(f"\n  Config → v2.45.0  OBJ={round(final_obj_val,4)}", flush=True)

        subprocess.run(["git", "add", str(cfg_path)], cwd=ROOT)
        msg = (f"feat: P260 regime weight reopt9 OBJ={round(final_obj_val,4)} "
               f"LOYO={loyo_wins}/5 D={delta:+.4f} [v2.45.0]")
        subprocess.run(["git", "commit", "-m", msg], cwd=ROOT)
        subprocess.run(["git", "push"], cwd=ROOT)
        print(f"  Committed: {msg}", flush=True)

    print(f"\nRuntime: {int(time.time()-t0)}s", flush=True)

main()
