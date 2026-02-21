"""
Phase 256 — Breadth Threshold Sweep (P_LOW × P_HIGH)
=====================================================
Baseline: v2.49.0, OBJ=4.3173

Sweep regime boundary thresholds. Currently hardcoded:
  P_LOW=0.30  → below 30th pct = LOW regime
  P_HIGH=0.60 → above 60th pct = HIGH regime

All per-regime overlay params (FTS, VOL, DISP, ts_pct_win) and regime
weights held constant. If different thresholds give better OBJ+LOYO,
validate and commit as v2.50.0.

P_LOW sweep:  [0.15, 0.20, 0.25, 0.30, 0.35, 0.40]
P_HIGH sweep: [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]
Total combos: 6 × 7 = 42 (all valid since max P_LOW < min P_HIGH)

Vectorized: rebuild overlay per (P_LOW,P_HIGH) pair, ~1ms eval/combo.
"""

import os, sys, json, time
from pathlib import Path
import signal as _signal

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
    out = Path("artifacts/phase256"); out.mkdir(parents=True, exist_ok=True)
    (out / "phase256_report.json").write_text(json.dumps(_partial, indent=2))
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

# Baseline thresholds
P_LOW_BASE = 0.30; P_HIGH_BASE = 0.60

RNAMES = ["LOW", "MID", "HIGH"]
TS_PCT_WIN = {"LOW": 240, "MID": 288, "HIGH": 400}

# Validated params from Phases 248-252
FTS_RS = {"LOW": 0.05, "MID": 0.05, "HIGH": 0.25}
FTS_BS = {"LOW": 4.00, "MID": 2.00, "HIGH": 2.00}
FTS_RT = {"LOW": 0.80, "MID": 0.65, "HIGH": 0.50}
FTS_BT = {"LOW": 0.30, "MID": 0.40, "HIGH": 0.20}
VOL_THR   = {"LOW": 0.50, "MID": 0.50, "HIGH": 0.55}
VOL_SCALE = {"LOW": 0.40, "MID": 0.15, "HIGH": 0.05}
DISP_THR   = {"LOW": 0.50, "MID": 0.70, "HIGH": 0.40}
DISP_SCALE = {"LOW": 0.70, "MID": 1.80, "HIGH": 0.50}

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

BASELINE_OBJ = 4.3173
MIN_DELTA = 0.005; MIN_LOYO = 3

# Sweep grid
P_LOW_SWEEP  = [0.15, 0.20, 0.25, 0.30, 0.35, 0.40]
P_HIGH_SWEEP = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]

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

def build_eval_data(all_base_data, all_v1_raw, all_signals, p_low, p_high):
    """Build overlay-scaled signals for a given (p_low, p_high) threshold pair."""
    fast_data = {}
    for yr, base_data in sorted(all_base_data.items()):
        btc_vol, fund_std_pct, ts_pct_cache, brd_pct, n, _ = base_data
        v1 = all_v1_raw[yr]
        sigs = all_signals[yr]
        min_len = min(len(v1["LOW"]), len(sigs["i460"]), len(sigs["i415"]), len(sigs["f168"]))

        bpct = brd_pct[:min_len]
        bv   = btc_vol[:min_len]
        fsp  = fund_std_pct[:min_len]

        regime_idx = np.where(bpct < p_low, 0, np.where(bpct > p_high, 2, 1))
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
            "i460":    overlay_mult * sigs["i460"][:min_len],
            "i415":    overlay_mult * sigs["i415"][:min_len],
            "f168":    overlay_mult * sigs["f168"][:min_len],
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

def main():
    t0 = time.time()
    print("=" * 65)
    print("Phase 256 — Breadth Threshold Sweep (P_LOW × P_HIGH)")
    print(f"Baseline: v2.49.0  OBJ={BASELINE_OBJ}")
    print(f"P_LOW sweep: {P_LOW_SWEEP}")
    print(f"P_HIGH sweep: {P_HIGH_SWEEP}")
    print(f"Total combos: {len(P_LOW_SWEEP) * len(P_HIGH_SWEEP)}")
    print("=" * 65)

    print("\n[1] Loading per-year data...", flush=True)
    all_base_data = {}
    for yr in sorted(YEAR_RANGES):
        print(f"  {yr}: ", end="", flush=True)
        all_base_data[yr] = load_year_data(yr)
        print("done.", flush=True)

    print("\n[2] Pre-computing signals × 5 years...", flush=True)
    all_signals = {}
    all_v1_raw = {}
    for yr in sorted(YEAR_RANGES):
        ds = all_base_data[yr][5]
        all_v1_raw[yr] = {
            "LOW": compute_signal(ds, "nexus_alpha_v1", V1_LOW_PARAMS),
            "MID": compute_signal(ds, "nexus_alpha_v1", V1_MIDHIGH_PARAMS),
        }
        all_signals[yr] = {
            "i460": compute_signal(ds, "idio_momentum_alpha", I460_PARAMS),
            "i415": compute_signal(ds, "idio_momentum_alpha", I415_PARAMS),
            "f168": compute_signal(ds, "funding_momentum_alpha", F168_PARAMS),
        }
        print(f"  {yr}: done.", flush=True)

    print("\n[3] Computing baseline (P_LOW=0.30, P_HIGH=0.60)...", flush=True)
    base_fd = build_eval_data(all_base_data, all_v1_raw, all_signals, P_LOW_BASE, P_HIGH_BASE)
    base_obj, base_yearly = fast_evaluate(REGIME_WEIGHTS, base_fd)
    print(f"  Baseline OBJ = {base_obj:.4f}  (expected ≈ {BASELINE_OBJ})", flush=True)

    print("\n[4] Joint (P_LOW, P_HIGH) sweep...", flush=True)
    best_obj = base_obj
    best_plow = P_LOW_BASE
    best_phigh = P_HIGH_BASE
    best_yearly = dict(base_yearly)
    results = []

    for p_low in P_LOW_SWEEP:
        for p_high in P_HIGH_SWEEP:
            fd = build_eval_data(all_base_data, all_v1_raw, all_signals, p_low, p_high)
            obj, yearly = fast_evaluate(REGIME_WEIGHTS, fd)
            marker = " ★" if obj > best_obj else ""
            print(f"  P_LOW={p_low:.2f}  P_HIGH={p_high:.2f}  OBJ={obj:.4f}{marker}", flush=True)
            results.append({"p_low": p_low, "p_high": p_high, "obj": obj})
            if obj > best_obj:
                best_obj = obj
                best_plow = p_low
                best_phigh = p_high
                best_yearly = dict(yearly)

    print(f"\n  Tested {len(results)} combos.", flush=True)
    print(f"  Best: P_LOW={best_plow}  P_HIGH={best_phigh}  OBJ={best_obj:.4f}  Δ={best_obj-base_obj:+.4f}", flush=True)

    print("\n[5] LOYO validation...", flush=True)
    loyo_wins = 0
    for yr in sorted(base_yearly):
        base_sh = base_yearly[yr]
        cand_sh = best_yearly[yr]
        win = cand_sh > base_sh
        if win: loyo_wins += 1
        print(f"  {yr}: base={base_sh:.4f}  cand={cand_sh:.4f}  {'WIN' if win else 'LOSE'}")

    delta = best_obj - base_obj
    print(f"\n  OBJ={best_obj:.4f}  Δ={delta:+.4f}  LOYO {loyo_wins}/{len(base_yearly)}", flush=True)
    validated = loyo_wins >= MIN_LOYO and delta >= MIN_DELTA
    print(f"\n{'✅ VALIDATED' if validated else '❌ NO IMPROVEMENT'}", flush=True)

    result = {
        "phase": 256, "baseline_obj": float(base_obj),
        "best_obj": float(best_obj), "delta": float(delta),
        "loyo_wins": loyo_wins, "loyo_total": len(base_yearly),
        "validated": validated,
        "best_p_low": best_plow, "best_p_high": best_phigh,
        "all_results": results,
    }
    _partial.update(result)
    out = Path("artifacts/phase256"); out.mkdir(parents=True, exist_ok=True)
    (out / "phase256_report.json").write_text(json.dumps(result, indent=2))

    if validated:
        cfg_path = ROOT / "configs" / "production_p91b_champion.json"
        cfg = json.load(open(cfg_path))
        cfg["breadth_regime_switching"]["p_low"]  = best_plow
        cfg["breadth_regime_switching"]["p_high"] = best_phigh
        cfg["version"] = "v2.50.0"
        cfg["_version"] = "2.50.0"
        cfg["obj"] = round(best_obj, 4)
        with open(cfg_path, "w") as f:
            json.dump(cfg, f, indent=2, ensure_ascii=False)
        import subprocess
        subprocess.run(["git", "add", str(cfg_path)], cwd=str(ROOT))
        subprocess.run(["git", "commit", "-m",
            f"feat: P256 breadth threshold sweep OBJ={round(best_obj,4)} "
            f"LOYO={loyo_wins}/5 D={delta:+.4f} P_LOW={best_plow} P_HIGH={best_phigh} [v2.50.0]"],
            cwd=str(ROOT))
        print(f"\n  Config → v2.50.0  P_LOW={best_plow}  P_HIGH={best_phigh}  OBJ={round(best_obj,4)}", flush=True)
    else:
        print(f"\n  Config unchanged (v2.49.0)", flush=True)

    print(f"\nRuntime: {int(time.time()-t0)}s", flush=True)

main()
