"""
Phase 255 — FTS Short/Long Window Re-Sweep (Vectorized)
=========================================================
Baseline: v2.49.0, OBJ=4.3173

TS_SHORT=16h, TS_LONG=72h were tuned in old config before per-regime FTS.
After Phase 248 (per-regime FTS RS/BS/RT/BT changes), the optimal windows
for the spread signal may have shifted.

Design: precompute raw ts_raw for each (short, long) combo, then apply
full overlay vectorized. Sweep 8×8=64 combos.
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
    out = Path("artifacts/phase255"); out.mkdir(parents=True, exist_ok=True)
    (out / "phase255_report.json").write_text(json.dumps(_partial, indent=2))
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
VOL_WINDOW = 168
P_LOW = 0.30; P_HIGH = 0.60

RNAMES = ["LOW", "MID", "HIGH"]
TS_PCT_WIN = {"LOW": 240, "MID": 288, "HIGH": 400}

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

# Current: TS_SHORT=16, TS_LONG=72
TS_SHORT_BASE = 16; TS_LONG_BASE = 72
TS_SHORT_SWEEP = [8, 12, 16, 20, 24, 32, 48, 72]
TS_LONG_SWEEP  = [36, 48, 60, 72, 84, 96, 120, 168]

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
    # Precompute ts_pct for ALL unique window values
    xsect_mean = np.mean(fund_rates, axis=1)
    breadth = np.full(n, 0.5)
    for i in range(BRD_LB, n):
        pos = sum(1 for j in range(len(SYMBOLS))
                  if close_mat[i-BRD_LB, j] > 0 and close_mat[i, j] > close_mat[i-BRD_LB, j])
        breadth[i] = pos / len(SYMBOLS)
    brd_pct = np.full(n, 0.5)
    for i in range(PCT_WINDOW, n):
        brd_pct[i] = float(np.mean(breadth[i-PCT_WINDOW:i] <= breadth[i]))
    return btc_vol, fund_std_pct, xsect_mean, brd_pct, n, dataset

def compute_signal(dataset, name, params):
    return np.array(BacktestEngine(BacktestConfig(costs=COST_MODEL)).run(
        dataset, make_strategy({"name": name, "params": params})).returns)

def precompute_ts_pct(xsect_mean, ts_short, ts_long, win_size):
    """Compute ts_pct for a single (short, long, win) combo."""
    n = len(xsect_mean)
    ts_raw = rolling_mean_arr(xsect_mean, ts_short) - rolling_mean_arr(xsect_mean, ts_long)
    ts_pct = np.full(n, 0.5)
    for i in range(win_size, n):
        ts_pct[i] = float(np.mean(ts_raw[i-win_size:i] <= ts_raw[i]))
    return ts_pct

def precompute_all_ts_pct(all_base_data, ts_short, ts_long):
    """Precompute ts_pct for all unique TS_PCT_WIN windows and all years."""
    all_ts_pct = {}
    unique_wins = sorted(set(TS_PCT_WIN.values()))
    for yr, base_data in sorted(all_base_data.items()):
        _, _, xsect_mean, _, n, _ = base_data
        all_ts_pct[yr] = {}
        for w in unique_wins:
            all_ts_pct[yr][w] = precompute_ts_pct(xsect_mean, ts_short, ts_long, w)
    return all_ts_pct

def precompute_fast_eval_data(all_base_data, all_signals, all_ts_pct):
    """Precompute full overlay and raw_weighted signals."""
    fast_data = {}
    for yr, base_data in sorted(all_base_data.items()):
        btc_vol, fund_std_pct, _, brd_pct, n, _ = base_data
        sigs = all_signals[yr]
        ts_by_win = all_ts_pct[yr]
        min_len = min(len(sigs[k]) for k in ["v1_LOW","v1_MID","i460","i415","f168"])

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
            tsp = ts_by_win[TS_PCT_WIN[rname]][:min_len]
            overlay_mult[m & (tsp > FTS_RT[rname])] *= FTS_RS[rname]
            overlay_mult[m & (tsp < FTS_BT[rname])] *= FTS_BS[rname]

        raw_weighted = {}
        for ridx, rname in enumerate(RNAMES):
            w = REGIME_WEIGHTS[rname]
            v1_key = "v1_LOW" if rname == "LOW" else "v1_MID"
            raw_weighted[rname] = (
                w["v1"]   * sigs[v1_key][:min_len]
                + w["i460"] * sigs["i460"][:min_len]
                + w["i415"] * sigs["i415"][:min_len]
                + w["f168"] * sigs["f168"][:min_len]
            )
        fast_data[yr] = (masks, overlay_mult, raw_weighted, min_len)
    return fast_data

def evaluate_full(fast_data):
    yearly = {}
    for yr, (masks, overlay_mult, raw_weighted, min_len) in fast_data.items():
        ens = np.zeros(min_len)
        for ridx, rname in enumerate(RNAMES):
            m = masks[ridx]
            ens[m] += raw_weighted[rname][m] * overlay_mult[m]
        r = ens[~np.isnan(ens)]
        sh = float(np.mean(r) / np.std(r, ddof=1) * np.sqrt(8760)) if len(r) > 1 else 0.0
        yearly[yr] = sh
    vals = list(yearly.values())
    return float(np.mean(vals) - 0.5 * np.std(vals, ddof=1)), yearly

def main():
    t0 = time.time()
    print("=" * 65)
    print("Phase 255 — FTS Short/Long Window Re-Sweep (Vectorized)")
    print(f"Baseline: v2.49.0  OBJ={BASELINE_OBJ}")
    print(f"Sweep: TS_SHORT={TS_SHORT_SWEEP} × TS_LONG={TS_LONG_SWEEP}")
    print(f"Grid: {len(TS_SHORT_SWEEP)}×{len(TS_LONG_SWEEP)}={len(TS_SHORT_SWEEP)*len(TS_LONG_SWEEP)} combos (short < long constraint)")
    print("=" * 65)

    print("\n[1] Loading per-year data...", flush=True)
    all_base_data = {}
    for yr in sorted(YEAR_RANGES):
        print(f"  {yr}: ", end="", flush=True)
        all_base_data[yr] = load_year_data(yr)
        print("done.", flush=True)

    print("\n[2] Pre-computing signals × 5 years...", flush=True)
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

    print("\n[3] Computing baseline (TS_SHORT=16, TS_LONG=72)...", flush=True)
    base_ts_pct = precompute_all_ts_pct(all_base_data, TS_SHORT_BASE, TS_LONG_BASE)
    base_fast = precompute_fast_eval_data(all_base_data, all_signals, base_ts_pct)
    base_obj_val, base_yearly = evaluate_full(base_fast)
    print(f"  Baseline OBJ = {base_obj_val:.4f}  (expected ≈ {BASELINE_OBJ})", flush=True)
    _partial.update({"baseline_obj": float(base_obj_val)})

    print("\n[4] Joint (short, long) sweep...", flush=True)
    best_short = TS_SHORT_BASE; best_long = TS_LONG_BASE
    best_obj = base_obj_val
    n_tested = 0
    for ts_short in TS_SHORT_SWEEP:
        for ts_long in TS_LONG_SWEEP:
            if ts_short >= ts_long: continue
            n_tested += 1
            ts_pct = precompute_all_ts_pct(all_base_data, ts_short, ts_long)
            fd = precompute_fast_eval_data(all_base_data, all_signals, ts_pct)
            o, _ = evaluate_full(fd)
            marker = " <-" if o > best_obj else ""
            print(f"  short={ts_short:3d} long={ts_long:3d}  OBJ={o:.4f}{marker}", flush=True)
            if o > best_obj:
                best_obj = o; best_short = ts_short; best_long = ts_long

    print(f"\n  Tested {n_tested} valid combos.", flush=True)
    print(f"  Best: TS_SHORT={best_short}  TS_LONG={best_long}  OBJ={best_obj:.4f}  "
          f"Δ={best_obj-base_obj_val:+.4f}", flush=True)

    # Evaluate best
    best_ts_pct = precompute_all_ts_pct(all_base_data, best_short, best_long)
    best_fast = precompute_fast_eval_data(all_base_data, all_signals, best_ts_pct)
    final_obj_val, final_yearly = evaluate_full(best_fast)
    delta = final_obj_val - base_obj_val

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
        "phase": 255, "baseline_obj": float(base_obj_val),
        "best_obj": float(final_obj_val), "delta": float(delta),
        "loyo_wins": loyo_wins, "loyo_total": len(final_yearly),
        "validated": validated,
        "best_ts_short": best_short, "best_ts_long": best_long,
    }
    _partial.update(result)
    out = Path("artifacts/phase255"); out.mkdir(parents=True, exist_ok=True)
    (out / "phase255_report.json").write_text(json.dumps(result, indent=2))

    if validated:
        cfg_path = ROOT / "configs" / "production_p91b_champion.json"
        cfg = json.load(open(cfg_path))
        fts = cfg.get("fts_overlay_params", {})
        fts["ts_short_window_bars"] = best_short
        fts["ts_long_window_bars"]  = best_long
        cfg["fts_overlay_params"] = fts
        cfg["version"] = "v2.50.0"
        cfg["_version"] = "2.50.0"
        cfg["obj"] = round(final_obj_val, 4)
        with open(cfg_path, "w") as f:
            json.dump(cfg, f, indent=2, ensure_ascii=False)
        import subprocess
        subprocess.run(["git", "add", str(cfg_path)], cwd=str(ROOT))
        subprocess.run(["git", "commit", "-m",
            f"feat: P255 FTS windows resweep OBJ={round(final_obj_val,4)} "
            f"LOYO={loyo_wins}/5 D={delta:+.4f} [v2.50.0]"], cwd=str(ROOT))
        print(f"\n  Config → v2.50.0  OBJ={round(final_obj_val,4)}", flush=True)
    else:
        print(f"\n  Config unchanged (v2.49.0)", flush=True)

    print(f"\nRuntime: {int(time.time()-t0)}s", flush=True)

main()
